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
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

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


def study(data, needs: dict[str, np.ndarray], cfg: dict, k_values: list[int],
          logger=None) -> tuple[pd.DataFrame, dict]:
    """М2 при каждом k и каждом определении нуждаемости.

    Возвращает таблицу (строка — определение × k) и решения B′ по k, чтобы
    порог шума можно было пересчитать из сохранённых файлов без решателя."""
    say = logger.info if logger else (lambda *a, **kw: None)
    params = cfg["m2"]
    t_max = cfg["t_max_minutes"]
    rows, noise_sites = [], {}
    for k in k_values:
        b = solver.solve_pmedian_any(data.site_times, data.base_times,
                                     data.population, data.allowed_pairs, k, "B", cfg)
        draws = noise_draws(data, cfg, k, params["noise_draws"],
                            params["noise_amplitude"], cfg["seed"])
        noise_sites[k] = [list(s) for s in draws.sites]
        for name, need in needs.items():
            bn = solver.solve_pmedian_any(data.site_times, data.base_times, need,
                                          data.allowed_pairs, k, "BN", cfg)
            z_b = need_time(data.site_times, data.base_times, b.sites, need, t_max)
            z_bn = need_time(data.site_times, data.base_times, bn.sites, need, t_max)
            ceiling = z_b - z_bn
            noise, gaps = draws.threshold(data.site_times, data.base_times, need,
                                          z_b, t_max, params["noise_percentile"])
            rows.append({
                "need": name, "k": k,
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
            })
            say("М2", need=name, k=k, ceiling=round(ceiling, 1),
                noise=round(noise, 1), above=ceiling > noise)
    return pd.DataFrame(rows), noise_sites
