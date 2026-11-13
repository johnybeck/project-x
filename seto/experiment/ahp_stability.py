"""Устойчивость варианта E: Монте-Карло по спорным выборам (пропозал v2, п. 3.6).

В каждом повторе случайно меняются сразу все выборы, которых нет в статье: схема
классов, способ получить веса, правило ничьих, сами веса в пределах ±разброс и два
наших порога — какая доля клеток считается плотным районом и какие дороги главными
(решение 08.11, PROPOSAL_CHANGES п. 10). Так видно, что сильнее определяет решение E.
Доля источника здесь — эффект первого порядка: часть разброса ΔCov_T, которую
объясняет он один; остаток — совместное действие нескольких источников.

Основной E пороги не перебирает: он считается в `evaluate` по `ahp.dense_share` и
`osm.main_roads`, как и раньше.
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
# источники разброса с конечным числом вариантов: колонка таблицы повторов
_SOURCES = ("scheme", "method", "ties", "dense", "roads")
# имя единственного варианта порога, когда конфиг пороги не перебирает
_MAIN_OPTION = "main"


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


def threshold_options(data, cfg: dict,
                      criteria: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    """Критерии «плотные районы» и «главные дороги» при каждом пороге из конфига.

    Слои считает подготовка города: здесь нет ни графа, ни дорог. Если конфиг пороги
    не перебирает, у критерия один вариант — основной слой. Если перебирает, а в
    подготовке слоёв нет, это ошибка: молча взять основной слой значило бы выдать
    узкий разброс E за полный."""
    params = cfg["ahp"]["stability"]
    wanted = {"dense": [ahp.share_name(share) for share in params.get("dense_shares") or []],
              "roads": list(params.get("main_roads") or {})}
    stored = getattr(data, "ahp_options", None) or {}
    options = {}
    for criterion, names in wanted.items():
        if criterion not in criteria:
            continue
        if not names:
            options[criterion] = {_MAIN_OPTION: criteria[criterion]}
            continue
        missing = [name for name in names if name not in stored.get(criterion, {})]
        if missing:
            raise ValueError(
                f"в подготовке города нет слоёв «{criterion}» для порогов {missing}: "
                "подготовь город заново (`prepare-city`)")
        # тот же знак, что в ahp.layers: ближе к дорогам и плотным районам — лучше
        options[criterion] = {name: -np.asarray(stored[criterion][name], dtype=float)
                              for name in names}
    return options


def run_draws(data, cfg: dict, criteria: dict[str, np.ndarray],
              k_values: list[int]) -> pd.DataFrame:
    """Повторы E: одна строка на повтор и k."""
    rules = cfg["ahp"]
    params = rules["stability"]
    names = list(ahp.kept_weights(rules))
    criteria = {name: criteria[name] for name in names}
    ties = _tie_values(data, cfg)
    options = threshold_options(data, cfg, criteria)

    def grade(values: np.ndarray, scheme) -> np.ndarray:
        return ahp.graded({"criterion": values}, data.site_index, scheme)["criterion"]

    # баллы критерия зависят от схемы классов и, у двух критериев, от порога
    fixed = {scheme: {name: grade(criteria[name], scheme)
                      for name in names if name not in options}
             for scheme in params["schemes"]}
    varied = {scheme: {name: {option: grade(values, scheme)
                              for option, values in options[name].items()}
                       for name in options}
              for scheme in params["schemes"]}
    base_weights = {method: ahp.kept_weights(rules, method)
                    for method in params["weight_methods"]}
    spread = params["weight_spread"]

    rng = np.random.default_rng(cfg["seed"])
    # У порогов свой генератор: выборы пропозала (схема, веса, ничьи) в каждом повторе
    # остаются теми же, что были до включения порогов, и разницу даёт только порог.
    rng_options = np.random.default_rng([cfg["seed"], 1])
    rows = []
    for draw in range(params["draws"]):
        scheme = _pick(params["schemes"], rng)
        method = _pick(params["weight_methods"], rng)
        rule = _pick(params["tie_rules"], rng)
        factors = rng.uniform(1 - spread, 1 + spread, size=len(names))
        chosen = {name: _pick(list(options[name]), rng_options) for name in options}
        raw = np.array([base_weights[method][name] for name in names]) * factors
        weights = dict(zip(names, raw / raw.sum()))
        grades = {name: varied[scheme][name][chosen[name]] if name in chosen
                  else fixed[scheme][name] for name in names}
        score = ahp.combine(grades, weights)[data.site_index]
        picked = ahp.choose(score, ties[rule], data.allowed_pairs, max(k_values))
        for k in k_values:
            sites = tuple(sorted(picked[:k]))
            rows.append({"draw": draw, "k": k, "scheme": str(scheme), "method": method,
                         "ties": rule,
                         **{name: chosen.get(name, _MAIN_OPTION)
                            for name in ("dense", "roads")},
                         "top_cells": ahp.top_cells(score),
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
                  for source in _SOURCES}
        shares["share_weights"] = sum(
            _explained(values, pd.qcut(part[factor], _FACTOR_BINS, labels=False))
            for factor in factors)
        rows.append({"k": k, "draws": len(part), "solutions": part["sites"].nunique(),
                     "dCov_min": values.min(), "dCov_p5": values.quantile(0.05),
                     "dCov_median": values.median(), "dCov_p95": values.quantile(0.95),
                     "dCov_max": values.max(), **shares,
                     "share_joint": max(0.0, 1.0 - sum(shares.values()))})
    return pd.DataFrame(rows)


def by_option(draws: pd.DataFrame) -> pd.DataFrame:
    """Разброс ΔCov_T по вариантам каждого порога: строка на k, порог и вариант.

    Доля разброса говорит, насколько порог важен, но не в какую сторону он двигает E.
    Эта таблица показывает сторону."""
    rows = []
    for source in ("dense", "roads"):
        for (k, option), part in draws.groupby(["k", source]):
            values = part["dCov"]
            rows.append({"k": k, "source": source, "option": option, "draws": len(part),
                         "dCov_p5": values.quantile(0.05), "dCov_median": values.median(),
                         "dCov_p95": values.quantile(0.95)})
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
