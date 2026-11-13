"""М2 — ошибка замещения спроса и доля захвата (пропозал v2, п. 3.10).

    Z_N(S)    = Σ_i N_i · t_i(S)                       человеко-минуты нуждающихся
    потолок   = Z_N(S_B) − Z_N(S_BN) ≥ 0               на сколько хуже решение,
                                                       считающее всех жителей
    захват_X  = (Z_N(S_B) − Z_N(S_X)) / потолок, %     какую часть отыграл X
    порог     = 95-й процентиль |Z_N(S_B) − Z_N(S_B′)|, B′ — P-median при
                P_i × U[0,9; 1,1], 200 повторов

Захват показывается, только если потолок выше порога шума: иначе «отыгранная
доля» мерила бы случайное колебание решения, а не знание о нуждаемости.

Решения B′ зависят только от зашумлённого P, а не от того, как определена
нуждаемость, — поэтому они считаются один раз на k и оцениваются по всем
определениям N (основное и проверки устойчивости).

Порог шума отвечает на вопрос «не дрожит ли само решение B». Он не ловит другого:
скор меняет вес клетки до двух раз, и даже скор без знания о городе сдвигает решение, а
иногда попадает в площадку BN. Поэтому рядом с захватом считается сравнение с
перестановкой (решение 10.11, PROPOSAL_CHANGES п. 16): те же ответы модели розданы
клеткам с жителями наугад, решатель и вес спроса те же. Захват называется «выше
случайного», только если не больше `permutation_level` перестановок дают захват не ниже.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..scoring import weights as weights_mod
from ..sites import feasible as sites_mod
from ..variants import solve as solver
from .coverage import times_with


def need_time(site_times: np.ndarray, base_times: np.ndarray, sites,
              need: np.ndarray, t_max: float) -> float:
    """Z_N(S): суммарное время пути нуждающихся до ближайшего учреждения."""
    times = times_with(base_times, site_times[list(sites)], t_max)
    return float(np.dot(np.asarray(need, dtype=float), times))


def capture(z_b: float, z_x: float, ceiling: float, noise: float) -> float | None:
    """Доля захвата варианта X в процентах; None, если потолок не выше шума."""
    if ceiling <= noise or ceiling <= 0:
        return None
    return 100.0 * (z_b - z_x) / ceiling


@dataclass
class NoiseDraws:
    """Решения B′ при зашумлённом населении — для одного k."""
    k: int
    sites: list[tuple[int, ...]] = field(default_factory=list)

    def threshold(self, site_times, base_times, need, z_b: float, t_max: float,
                  percentile: float) -> tuple[float, np.ndarray]:
        """Порог шума и все отклонения |Z_N(S_B) − Z_N(S_B′)|."""
        gaps = np.array([abs(z_b - need_time(site_times, base_times, s, need, t_max))
                         for s in self.sites])
        return float(np.percentile(gaps, percentile)), gaps


def noise_draws(data, cfg: dict, k: int, draws: int, amplitude: float,
                seed: int) -> NoiseDraws:
    """B′: тот же P-median, тот же решатель, вес P_i × U[1 − a; 1 + a]."""
    rng = np.random.default_rng(seed)
    result = NoiseDraws(k=k)
    for _ in range(draws):
        noisy = data.population * rng.uniform(1 - amplitude, 1 + amplitude,
                                              len(data.population))
        solution = solver.solve_pmedian_any(data.site_times, data.base_times, noisy,
                                            data.allowed_pairs, k, "B′", cfg)
        result.sites.append(tuple(solution.sites))
    return result


def permutation_draws(data, cfg: dict, k: int, scores: np.ndarray, draws: int,
                      seed) -> list[tuple[int, ...]]:
    """Решения при перемешанных ответах модели: тот же решатель, тот же вес спроса.

    Перемешиваются только клетки с жителями. У пустой клетки вес спроса ноль при любом
    скоре, а её ответ почти всегда «не определить»: отдай его клетке с жителями — и
    набор ответов станет ровнее настоящего, а проверка мягче. Набор значений по всем
    клеткам при этом тот же, поэтому нормализация скора не меняется."""
    population = np.asarray(data.population, dtype=float)
    live = population > 0
    values = np.asarray(scores, dtype=float)
    rng = np.random.default_rng(seed)
    found = []
    for _ in range(draws):
        shuffled = values.copy()
        shuffled[live] = rng.permutation(values[live])
        weights = weights_mod.demand(population, shuffled, cfg["alpha"])
        solution = solver.solve_pmedian_any(data.site_times, data.base_times, weights,
                                            data.allowed_pairs, k, "перестановка", cfg)
        found.append(tuple(solution.sites))
    return found


def share_not_below(gain: float, gains: np.ndarray, scale: float) -> float:
    """Доля перестановок, отыгравших не меньше варианта. Сравнение с допуском: суммы
    человеко-минут одного и того же набора площадок не должны разойтись на шуме
    округления."""
    tolerance = 1e-9 * max(1.0, abs(scale))
    return float((np.asarray(gains, dtype=float) >= gain - tolerance).mean())


def with_permutations(table: pd.DataFrame, data, needs: dict[str, np.ndarray], cfg: dict,
                      variants: dict[int, dict], scores: dict[str, np.ndarray],
                      logger=None) -> tuple[pd.DataFrame, dict]:
    """Дописать в таблицу М2 сравнение с перестановкой ответов модели.

    scores — скоры по именам вариантов (C, D, D10). Для каждого варианта и k решения
    перестановок считаются один раз и оцениваются по всем определениям N, как B′.
    Колонки: `perm_share_X` — доля перестановок с захватом не ниже, чем у X;
    `above_chance_X` — правило чтения, заполняется там, где захват показывается.
    Возвращает таблицу и решения перестановок {k: {вариант: наборы}}."""
    say = logger.info if logger else (lambda *a, **kw: None)
    params = cfg["m2"]
    draws = params.get("permutation_draws", 0)
    if not draws or not scores or table.empty:
        return table, {}
    level, t_max = params["permutation_level"], cfg["t_max_minutes"]
    table = table.copy()
    for name in scores:
        table[f"perm_share_{name}"] = np.nan
        table[f"above_chance_{name}"] = None
    found: dict = {}
    for requested in sorted(table["k"].unique()):
        rows = table.index[table["k"] == requested]
        k = int(table.at[rows[0], "k_used"])
        found[int(requested)] = {}
        for position, (name, values) in enumerate(scores.items()):
            chosen = (variants.get(requested) or {}).get(name)
            if not chosen:
                continue
            shuffled = permutation_draws(data, cfg, k, values, draws,
                                         [cfg["seed"], 2, int(requested), position])
            found[int(requested)][name] = [list(sites) for sites in shuffled]
            for index in rows:
                need = needs[table.at[index, "need"]]
                z_b = float(table.at[index, "Z_N_B"])
                gain = z_b - need_time(data.site_times, data.base_times, chosen, need, t_max)
                gains = [z_b - need_time(data.site_times, data.base_times, sites, need, t_max)
                         for sites in shuffled]
                share = share_not_below(gain, gains, z_b)
                table.at[index, f"perm_share_{name}"] = share
                shown = (f"capture_{name}" in table.columns
                         and pd.notna(table.at[index, f"capture_{name}"]))
                table.at[index, f"above_chance_{name}"] = bool(share <= level) if shown else None
            say("М2, перестановки", variant=name, k=int(requested),
                distinct=len(set(shuffled)))
    return table, found


def study(data, needs: dict[str, np.ndarray], cfg: dict, k_values: list[int],
          logger=None, variants: dict[int, dict] | None = None) -> tuple[pd.DataFrame, dict]:
    """М2 при каждом k и каждом определении нуждаемости.

    Возвращает таблицу (строка — определение × k) и решения B′ по k, чтобы
    порог шума можно было пересчитать из сохранённых файлов без решателя.
    variants — решения проверяемых вариантов {k: {имя: площадки}}: по ним считается
    их доля захвата.

    Если k площадок разнести нельзя, задача решается с наибольшим допустимым
    k (п. 3.2, ограничение 5); в таблице видны оба числа."""
    say = logger.info if logger else (lambda *a, **kw: None)
    params = cfg["m2"]
    t_max = cfg["t_max_minutes"]
    rows, noise_sites = [], {}
    for requested in k_values:
        k = min(requested, sites_mod.max_feasible_k(data.allowed_pairs, requested))
        b = solver.solve_pmedian_any(data.site_times, data.base_times,
                                     data.population, data.allowed_pairs, k, "B", cfg)
        draws = noise_draws(data, cfg, k, params["noise_draws"],
                            params["noise_amplitude"], cfg["seed"])
        noise_sites[requested] = [list(s) for s in draws.sites]
        for name, need in needs.items():
            bn = solver.solve_pmedian_any(data.site_times, data.base_times, need,
                                          data.allowed_pairs, k, "BN", cfg)
            z_b = need_time(data.site_times, data.base_times, b.sites, need, t_max)
            z_bn = need_time(data.site_times, data.base_times, bn.sites, need, t_max)
            ceiling = z_b - z_bn
            noise, _ = draws.threshold(data.site_times, data.base_times, need,
                                       z_b, t_max, params["noise_percentile"])
            captures = {
                f"capture_{variant}": capture(
                    z_b, need_time(data.site_times, data.base_times, sites, need, t_max),
                    ceiling, noise)
                for variant, sites in (variants or {}).get(requested, {}).items()}
            rows.append({
                "need": name, "k": requested, "k_used": k,
                "B_sites": list(b.sites), "BN_sites": list(bn.sites),
                "same_sites": set(b.sites) == set(bn.sites),
                "Z_N_B": z_b, "Z_N_BN": z_bn,
                "ceiling": ceiling,
                "ceiling_per_1000": 1000 * ceiling / float(need.sum()),
                "noise_threshold": noise,
                "noise_per_1000": 1000 * noise / float(need.sum()),
                "B_prime_distinct": len(set(draws.sites)),
                "above_noise": bool(ceiling > noise),
                # потолок < 0 возможен только у эвристики (k > 2): BN хуже B
                # по собственной цели — это повод проверить решатель
                "negative_ceiling": bool(ceiling < -1e-6 * max(1.0, abs(z_b))),
                "capture_B": capture(z_b, z_b, ceiling, noise),
                "capture_BN": capture(z_b, z_bn, ceiling, noise),
                **captures,
            })
            say("М2", need=name, k=requested, ceiling=round(ceiling, 1),
                noise=round(noise, 1), above=ceiling > noise)
    return pd.DataFrame(rows), noise_sites
