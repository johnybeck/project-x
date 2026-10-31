"""Устойчивость варианта E: Монте-Карло по спорным выборам (пропозал v2, п. 3.6).

В каждом повторе случайно меняются сразу все выборы, которых нет в статье: схема
классов, способ получить веса, правило ничьих и сами веса в пределах ±разброс. Так
видно, что сильнее определяет решение E. Доля источника здесь — эффект первого
порядка: часть разброса ΔCov_T, которую объясняет он один; остаток — совместное
действие нескольких источников.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..metrics import coverage as cov
from ..sites import feasible as sites_mod
from ..variants import ahp

_MAIN_RULE = "population_T"
_MAIN_METHOD = "published"
# столько групп для непрерывного множителя веса при подсчёте его доли
_FACTOR_BINS = 5


def _tie_values(data, cfg: dict) -> dict[str, np.ndarray]:
    """Значения для правил ничьих; cell_order — просто порядок в списке площадок."""
    sites = data.site_index
    return {_MAIN_RULE: ahp.population_within(data.site_times, data.population,
                                              cfg["T_minutes"]),
            "population_cell": np.asarray(data.population, dtype=float)[sites],
            "cell_order": np.zeros(len(sites))}


def _delta(data, sites: tuple[int, ...], cfg: dict) -> float:
    """ΔCov_T: тот же расчёт, что у всех вариантов."""
    times = cov.times_with(data.base_times, data.site_times[list(sites)],
                           cfg["t_max_minutes"])
    return cov.delta_coverage(times, data.base_times, data.population, cfg["T_minutes"])


def _pick(values: list, rng: np.random.Generator):
    return values[int(rng.integers(len(values)))]


def run_draws(data, cfg: dict, criteria: dict[str, np.ndarray],
              k_values: list[int]) -> pd.DataFrame:
    """Повторы E: одна строка на повтор и k."""
    rules = cfg["ahp"]
    params = rules["stability"]
    names = list(ahp.kept_weights(rules))
    criteria = {name: criteria[name] for name in names}
    ties = _tie_values(data, cfg)
    grades = {scheme: ahp.graded(criteria, data.site_index, scheme)
              for scheme in params["schemes"]}
    base_weights = {method: ahp.kept_weights(rules, method)
                    for method in params["weight_methods"]}
    spread = params["weight_spread"]

    rng = np.random.default_rng(cfg["seed"])
    rows = []
    for draw in range(params["draws"]):
        scheme = _pick(params["schemes"], rng)
        method = _pick(params["weight_methods"], rng)
        rule = _pick(params["tie_rules"], rng)
        factors = rng.uniform(1 - spread, 1 + spread, size=len(names))
        raw = np.array([base_weights[method][name] for name in names]) * factors
        weights = dict(zip(names, raw / raw.sum()))
        score = ahp.combine(grades[scheme], weights)[data.site_index]
        picked = ahp.choose(score, ties[rule], data.allowed_pairs, max(k_values))
        for k in k_values:
            sites = tuple(sorted(picked[:k]))
            rows.append({"draw": draw, "k": k, "scheme": str(scheme), "method": method,
                         "ties": rule, "top_cells": ahp.top_cells(score),
                         "sites": " ".join(map(str, sites)),
                         "dCov": _delta(data, sites, cfg),
                         **{f"factor_{name}": float(value)
                            for name, value in zip(names, factors)}})
    return pd.DataFrame(rows)


def _explained(values: pd.Series, groups: pd.Series) -> float:
    """Доля разброса, которую объясняет один источник."""
    total = float(values.var(ddof=0))
    if total == 0:
        return 0.0
    return float(values.groupby(groups).transform("mean").var(ddof=0) / total)


def summarize(draws: pd.DataFrame) -> pd.DataFrame:
    """По каждому k: разброс ΔCov_T и его доли по источникам."""
    rows = []
    factors = [column for column in draws.columns if column.startswith("factor_")]
    for k, part in draws.groupby("k"):
        values = part["dCov"]
        shares = {f"share_{source}": _explained(values, part[source])
                  for source in ("scheme", "method", "ties")}
        shares["share_weights"] = sum(
            _explained(values, pd.qcut(part[factor], _FACTOR_BINS, labels=False))
            for factor in factors)
        rows.append({"k": k, "draws": len(part), "solutions": part["sites"].nunique(),
                     "dCov_min": values.min(), "dCov_p5": values.quantile(0.05),
                     "dCov_median": values.median(), "dCov_p95": values.quantile(0.95),
                     "dCov_max": values.max(), **shares,
                     "share_joint": max(0.0, 1.0 - sum(shares.values()))})
    return pd.DataFrame(rows)


def tie_effect(data, cfg: dict, criteria: dict[str, np.ndarray],
               k_values: list[int]) -> pd.DataFrame:
    """Решило ли дело правило ничьих у основного E: сколько клеток с верхним баллом
    и меняется ли решение, если ничьи разбирать иначе."""
    rules = cfg["ahp"]
    weights = ahp.kept_weights(rules, _MAIN_METHOD)
    names = list(weights)
    grades = ahp.graded({name: criteria[name] for name in names}, data.site_index,
                        rules["classes"])
    score = ahp.combine(grades, weights)[data.site_index]
    ties = _tie_values(data, cfg)
    rows = []
    for k in k_values:
        main = set(ahp.choose(score, ties[_MAIN_RULE], data.allowed_pairs, k))
        others = [set(ahp.choose(score, values, data.allowed_pairs, k))
                  for rule, values in ties.items() if rule != _MAIN_RULE]
        rows.append({"k": k, "top_cells": ahp.top_cells(score),
                     "rule_decided": any(other != main for other in others)})
    return pd.DataFrame(rows)


def study(data, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Вся проверка устойчивости: повторы, сводка по k, влияние правила ничьих."""
    criteria = ahp.layers(data, cfg)
    if criteria is None:
        raise ValueError("нет критериев AHP: подготовь город заново (`prepare-city`)")
    limit = sites_mod.max_feasible_k(data.allowed_pairs, max(cfg["m2"]["k_values"]))
    k_values = sorted({min(k, limit) for k in cfg["m2"]["k_values"]})
    draws = run_draws(data, cfg, criteria, k_values)
    return draws, summarize(draws), tie_effect(data, cfg, criteria, k_values)
