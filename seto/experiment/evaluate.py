"""Решение вариантов и расчёт метрик (п. 10-12, `solve` и `evaluate`).

Метрики считаются ОДНИМ кодом для всех вариантов — отдельных расчётов
«под вариант» не существует (п. 11). Поэтому здесь нет ветвлений по имени
варианта: на вход приходит набор площадок, дальше всё одинаково.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..costs.timer import CostLog
from ..data import demography as demography_mod
from ..metrics import coverage as cov
from ..metrics import equity
from ..metrics import sfca
from ..scoring import weights as weights_mod
from ..sites import feasible as sites_mod
from ..variants import solve as solver


@dataclass
class VariantResult:
    """Решение варианта и его статус."""
    name: str
    sites: tuple[int, ...]
    status: str = "ok"            # ok | skipped_duplicate | failed | missing_external
    objective: float | None = None
    tie_count: int = 0
    note: str = ""
    distribution: np.ndarray | None = None   # только для A

    def as_dict(self) -> dict:
        return {"variant": self.name, "sites": list(self.sites),
                "status": self.status, "objective": self.objective,
                "tie_count": self.tie_count, "note": self.note}


@dataclass
class Evaluation:
    """Метрики всех вариантов города."""
    table: pd.DataFrame
    differences: pd.DataFrame
    variants: list[VariantResult] = field(default_factory=list)


@dataclass
class Targeting:
    """Что нужно блоку «адресность»; одно и то же для всех вариантов города."""
    need: np.ndarray
    seniors: np.ndarray
    districts: np.ndarray | None
    kappa: dict[float, float]     # κ для каждого ε, по исходной сети
    floor: float                  # A_0 для 2SFCA с весами N
    slow_speed: float


def targeting_inputs(data, cfg: dict) -> Targeting | None:
    """Подготовить блок «адресность»; None, если у города нет нуждаемости."""
    need = need_weights(data, cfg)
    if need is None:
        return None
    params = cfg["targeting"]
    group = cfg["city"]["need"]["seniors_group"]
    districts = getattr(data, "districts", None)
    return Targeting(
        need=need,
        seniors=data.groups[group].to_numpy(dtype=float),
        districts=None if districts is None else np.asarray(districts, dtype=object),
        kappa={epsilon: equity.kolm_pollak_kappa(data.base_times, need, epsilon)
               for epsilon in (params["ede_epsilon"],
                               *params["ede_epsilon_sensitivity"])},
        floor=sfca.floor_value(len(data.clinics), need),
        slow_speed=params["walk_speed_seniors_m_per_min"])


def _targeting_row(times: np.ndarray, base: np.ndarray, target: Targeting, cfg: dict,
                   facility_times: np.ndarray) -> dict:
    """Метрики адресности одного варианта (пропозал v2, п. 3.10)."""
    row: dict = {}
    main = cfg["T_minutes"]
    need, seniors = target.need, target.seniors
    for threshold in sorted({main, *cfg["T_sensitivity"]}):
        # М1: та же главная метрика, но вместо жителей — нуждающиеся
        row[f"Cov_N_{threshold}"] = cov.coverage(times, need, threshold)
        row[f"dCov_N_{threshold}"] = cov.delta_coverage(times, base, need, threshold)
        row[f"dN_N_{threshold}"] = cov.delta_people(times, base, need, threshold)
        row[f"dN_65_{threshold}"] = cov.delta_people(times, base, seniors, threshold)

    speed, t_max = cfg["walk_speed_m_per_min"], cfg["t_max_minutes"]
    slow = cov.at_speed(times, speed, target.slow_speed, t_max)
    slow_base = cov.at_speed(base, speed, target.slow_speed, t_max)
    row[f"Cov_65_slow_{main}"] = cov.coverage(slow, seniors, main)
    row[f"dCov_65_slow_{main}"] = cov.delta_coverage(slow, slow_base, seniors, main)
    row[f"dN_65_slow_{main}"] = cov.delta_people(slow, slow_base, seniors, main)

    # М3: κ один на все варианты, иначе EDE окажутся в разных шкалах
    for epsilon, kappa in target.kappa.items():
        row[f"EDE_{abs(epsilon):g}"] = equity.equally_distributed(times, need, kappa)
    row["mean_time_N"] = cov.mean_time(times, need)
    row[f"deficit_N_{main}"] = equity.deficit(times, need, main)
    row["gini_N"] = equity.gini(times, need)
    row["theil_N"] = equity.theil(times, need)
    if target.districts is not None:
        parts = equity.theil_decomposition(times, need, target.districts)
        row["theil_between"], row["theil_within"] = parts["between"], parts["within"]

    row.update({f"N_{key}": value for key, value in sfca.summary(
        facility_times, need, main, target.floor).items()})
    return row


def _site_distance(data, sites, cfg: dict) -> dict:
    """От новых точек до ближайшего существующего учреждения, м по сети.

    Описательно: сравнение с сетью, а не мера качества. Берётся из матрицы
    времён, поэтому недостижимая точка покажет потолок t_max."""
    minutes = data.clinic_times[:, data.site_index[list(sites)]].min(axis=0)
    metres = minutes * cfg["walk_speed_m_per_min"]
    return {"site_to_network_mean_m": float(metres.mean()),
            "site_to_network_min_m": float(metres.min())}


def _metrics_row(name: str, times: np.ndarray, base: np.ndarray,
                 population: np.ndarray, cfg: dict,
                 facility_times: np.ndarray, floor: float,
                 target: Targeting | None = None) -> dict:
    """Все метрики одного варианта. Общий код — в этом весь смысл (п. 11)."""
    row: dict = {"variant": name}
    for threshold in sorted({cfg["T_minutes"], *cfg["T_sensitivity"]}):
        row[f"Cov_{threshold}"] = cov.coverage(times, population, threshold)
        row[f"dCov_{threshold}"] = cov.delta_coverage(times, base, population,
                                                      threshold)
        row[f"dN_{threshold}"] = cov.delta_people(times, base, population,
                                                  threshold)
    row["worst_decile_min"] = cov.worst_decile_time(times, population)
    row["mean_time_min"] = cov.mean_time(times, population)
    row["unreachable_people"] = cov.unreachable_people(times, population,
                                                       cfg["t_max_minutes"])
    row.update(sfca.summary(facility_times, population, cfg["T_minutes"], floor))
    if target is not None:
        row.update(_targeting_row(times, base, target, cfg, facility_times))
    return row


def need_weights(data, cfg: dict) -> np.ndarray | None:
    """Нуждаемость N_i по основному определению (пропозал v2, п. 3.3).

    None, если у города нет возрастных групп или раздела need: тогда вариант
    BN помечается «нет данных», а не считается на всём населении — иначе BN
    молча совпал бы с B, и потолок М2 оказался бы нулём по построению."""
    city = cfg.get("city") or {}
    groups = getattr(data, "groups", None)
    if groups is None or "need" not in city:
        return None
    rules = city["need"]
    need = demography_mod.need(groups, rules["definitions"][rules["main"]],
                               rules["r"], city["demography"]["groups"])
    return need.to_numpy(dtype=float)


def run_variants(data, scores: dict[str, np.ndarray], cfg: dict,
                 costs: CostLog | None = None, logger=None) -> Evaluation:
    """Решить A, B, M, C, D и посчитать метрики. scores — скоры по вариантам."""
    costs = costs or CostLog()
    say = logger.info if logger else (lambda *a, **k: None)

    base = data.base_times
    population = data.population
    site_times = data.site_times
    allowed = data.allowed_pairs
    k = min(cfg["k"], sites_mod.max_feasible_k(allowed, cfg["k"]))
    t_max = cfg["t_max_minutes"]

    results: list[VariantResult] = []
    base_weights = weights_mod.demand(population)

    # --- B: p-median по населению ---
    with costs.stage("solve", "variant:B"):
        solution = solver.solve_pmedian_any(site_times, base, base_weights, allowed,
                                            k, "B", cfg)
    results.append(VariantResult("B", solution.sites, objective=solution.objective,
                                 tie_count=solution.tie_count))
    say("вариант B", sites=list(solution.sites), ties=solution.tie_count)

    # --- BN: p-median по нуждаемости — эталон для М2 (пропозал v2, п. 3.4) ---
    need = need_weights(data, cfg)
    if need is None:
        results.append(VariantResult(
            "BN", (), status="missing_external",
            note="нет нуждаемости: у города нет возрастных групп или раздела need"))
    else:
        with costs.stage("solve", "variant:BN"):
            solution = solver.solve_pmedian_any(site_times, base, need, allowed,
                                                k, "BN", cfg)
        comparison = weights_mod.compare(base_weights, need)
        # совпадение с B здесь — результат (потолок М2 = 0), а не дубликат
        same_as_b = solution.sites == results[0].sites
        results.append(VariantResult(
            "BN", solution.sites, objective=solution.objective,
            tie_count=solution.tie_count,
            note=(("решение совпало с B; " if same_as_b else "")
                  + f"корреляция весов с B {comparison['correlation']:.4f}")))
        say("вариант BN", sites=list(solution.sites), same_as_b=same_as_b)

    # --- M: максимальное покрытие ---
    with costs.stage("solve", "variant:M"):
        solution = solver.solve_max_coverage_any(site_times, base, population, allowed,
                                                 k, cfg["T_minutes"], cfg)
    results.append(VariantResult("M", solution.sites, objective=solution.objective,
                                 tie_count=solution.tie_count))
    say("вариант M", sites=list(solution.sites), ties=solution.tie_count)

    # --- C и D: те же веса плюс скор ---
    for name, key in (("C", "text"), ("D", "image")):
        if key not in scores:
            results.append(VariantResult(name, (), status="missing_external",
                                         note=f"нет скоров режима {key}"))
            continue
        with costs.stage("solve", f"variant:{name}"):
            weights = weights_mod.demand(population, scores[key], cfg["alpha"])
            solution = solver.solve_pmedian_any(site_times, base, weights, allowed,
                                                k, name, cfg)
        comparison = weights_mod.compare(base_weights, weights)
        # п. 9: если решение совпало с B, дубликат не считается
        same_as_b = solution.sites == results[0].sites
        results.append(VariantResult(
            name, solution.sites,
            status="skipped_duplicate" if same_as_b else "ok",
            objective=solution.objective, tie_count=solution.tie_count,
            note=(f"решение совпало с B; корреляция весов "
                  f"{comparison['correlation']:.4f}" if same_as_b else
                  f"корреляция весов с B {comparison['correlation']:.4f}")))
        say(f"вариант {name}", sites=list(solution.sites), same_as_b=same_as_b)

    # --- A: случайный выбор ---
    with costs.stage("solve", "variant:A"):
        sets = solver.solve_random(allowed, k, cfg["random_draws"], cfg["seed"])
        deltas = np.array([
            cov.delta_coverage(cov.times_with(base, site_times[list(s)], t_max),
                               base, population, cfg["T_minutes"]) for s in sets])
    results.append(VariantResult("A", (), note=f"{len(sets)} наборов",
                                 distribution=deltas))
    say("вариант A", sets=len(sets), median=float(np.median(deltas)),
        p95=float(np.percentile(deltas, 95)))

    # --- метрики ---
    floor = sfca.floor_value(len(data.clinics), population)
    clinic_times = _facility_times(data, None)
    target = targeting_inputs(data, cfg)
    rows = [_metrics_row("исходная сеть", base, base, population, cfg,
                         clinic_times, floor, target)]
    for result in results:
        if not result.sites:
            continue
        times = cov.times_with(base, site_times[list(result.sites)], t_max)
        row = _metrics_row(result.name, times, base, population, cfg,
                           _facility_times(data, result.sites), floor, target)
        row.update(_site_distance(data, result.sites, cfg))
        rows.append(row)

    table = pd.DataFrame(rows)
    return Evaluation(table=table,
                      differences=_differences(table, results, cfg),
                      variants=results)


def _facility_times(data, sites) -> np.ndarray:
    """Времена [учреждения x клетки] для 2SFCA: существующие плюс новые."""
    existing = data.clinic_times if hasattr(data, "clinic_times") else None
    if existing is None:
        existing = data.base_times[None, :]
    if not sites:
        return existing
    return np.vstack([existing, data.site_times[list(sites)]])


# BN стоит перед B: пропозал v2 (п. 3.4) читает разницу как BN − B
_PAIR_ORDER = ("D", "C", "BN", "B", "E", "M")


def _differences(table: pd.DataFrame, results: list[VariantResult],
                 cfg: dict) -> pd.DataFrame:
    """Разницы между вариантами в п.п. и жителях (п. 11.4)."""
    threshold = cfg["T_minutes"]
    indexed = table.set_index("variant")
    available = [r.name for r in results if r.sites and r.name in indexed.index]

    # Порядок пар — как в METRICS.md: проверяемый вариант слева (D − C, D − B,
    # C − B). Алфавитный порядок дал бы «C − D», то есть главную разницу
    # эксперимента с обратным знаком.
    order = [name for name in _PAIR_ORDER if name in available]
    rows = []
    for position, left in enumerate(order):
        for right in order[position + 1:]:
            rows.append({
                "pair": f"{left} − {right}",
                f"dCov_{threshold}_pp": (indexed.loc[left, f"dCov_{threshold}"]
                                         - indexed.loc[right, f"dCov_{threshold}"]),
                f"dN_{threshold}_people": (indexed.loc[left, f"dN_{threshold}"]
                                           - indexed.loc[right, f"dN_{threshold}"]),
            })

    random_result = next((r for r in results if r.name == "A"), None)
    if random_result is not None and random_result.distribution is not None:
        p95 = float(np.percentile(random_result.distribution, 95))
        for name in available:
            value = indexed.loc[name, f"dCov_{threshold}"]
            rows.append({
                "pair": f"{name} − A (медиана)",
                f"dCov_{threshold}_pp": value - float(np.median(random_result.distribution)),
                f"dN_{threshold}_people": np.nan,
                "above_A_p95": bool(value > p95),
                "percentile_in_A": float(100 * (random_result.distribution < value).mean()),
            })
    # строка-напоминание: без D главная разница эксперимента не посчитана,
    # даже если C и B есть
    if "D" not in available:
        rows.append({"pair": "D − …", f"dCov_{threshold}_pp": np.nan,
                     f"dN_{threshold}_people": np.nan,
                     "note": "вариант D не считался: нет снимков"})
    return pd.DataFrame(rows)
