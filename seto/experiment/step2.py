"""Шаг 2 пропозала v2 (п. 3.9): A, B, M, BN и М2 — без снимков и без модели.
Заодно считаются E, EN и EP: им тоже не нужны ни снимки, ни модель.

Самый дешёвый ответ на вопрос «есть ли снимку что закрывать». Если потолок
М2 везде ниже порога шума, это результат о постановке, и прогоны VLM можно
сократить; если выше хотя бы при каком-то k — изображению есть что отыграть.

Все варианты считаются при каждом k из `m2.k_values`: сравнивать их между
собой можно только при одном и том же числе новых учреждений.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..costs.timer import CostLog
from ..data import demography as demography_mod
from ..metrics import substitution as m2_mod
from .evaluate import SCORED_VARIANTS, run_variants


def needs_for(data, cfg: dict) -> dict[str, np.ndarray]:
    """Все определения нуждаемости из конфига города — по клеткам."""
    city = cfg["city"]
    if getattr(data, "groups", None) is None or "need" not in city:
        raise ValueError("у города нет возрастных групп или раздела need — "
                         "М2 без нуждаемости не считается")
    variants = demography_mod.need_all(data.groups, city["need"],
                                       city["demography"]["groups"])
    return {name: series.to_numpy(dtype=float) for name, series in variants.items()}


def coverage_by_k(metrics: pd.DataFrame, random_a: pd.DataFrame,
                  threshold: int) -> pd.DataFrame:
    """Главная метрика по k: ΔCov_T у A (медиана и p95) и остальных вариантов."""
    column = f"dCov_{threshold}"
    table = metrics.pivot(index="k", columns="variant", values=column)
    spread = random_a.groupby("k")["dCov"]
    result = pd.DataFrame({"A_median": spread.median(),
                           "A_p95": spread.quantile(0.95)})
    for name in ("B", "BN", "M", "C", "D", "D10", "CN", "DN", "E", "EN", "EP"):
        if name in table.columns:
            result[name] = table[name]
    return result.reset_index()


def run_step2(data, cfg: dict, out_dir: Path, logger=None,
              costs: CostLog | None = None,
              scores: dict[str, np.ndarray] | None = None) -> dict[str, Path]:
    """Посчитать шаг 2 и сохранить всё, из чего выводы пересчитываются.

    scores — скоры модели, если они уже есть: тогда при каждом k считаются C и D и
    их доли захвата. Без них шаг 2 по-прежнему не требует ни снимков, ни модели."""
    costs = costs or CostLog()
    out_dir.mkdir(parents=True, exist_ok=True)
    k_values = cfg["m2"]["k_values"]

    metrics, differences, solutions, random_a, captured = [], [], {}, [], {}
    for k in k_values:
        evaluation = run_variants(data, scores or {}, {**cfg, "k": k}, costs, logger)
        metrics.append(evaluation.table.assign(k=k))
        differences.append(evaluation.differences.assign(k=k))
        solutions[str(k)] = [variant.as_dict() for variant in evaluation.variants]
        spread = next(v for v in evaluation.variants if v.name == "A").distribution
        random_a.append(pd.DataFrame({"k": k, "dCov": spread}))
        captured[k] = {v.name: v.sites for v in evaluation.variants
                       if v.name in SCORED_VARIANTS.values() and v.sites}
    metrics, differences = pd.concat(metrics), pd.concat(differences)
    random_a = pd.concat(random_a, ignore_index=True)

    needs = needs_for(data, cfg)
    with costs.stage("m2", "shared_prep"):
        table, noise_sites = m2_mod.study(data, needs, cfg, k_values, logger, captured)

    paths = {"solutions": out_dir / "solutions.json",
             "metrics": out_dir / "metrics.csv",
             "differences": out_dir / "differences.csv",
             "random_A": out_dir / "random_A.parquet",
             "coverage_by_k": out_dir / "coverage_by_k.csv",
             "m2": out_dir / "m2.csv",
             "m2_noise_sites": out_dir / "m2_noise_sites.json"}
    paths["solutions"].write_text(json.dumps(solutions, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
    metrics.to_csv(paths["metrics"], index=False, encoding="utf-8")
    differences.to_csv(paths["differences"], index=False, encoding="utf-8")
    random_a.to_parquet(paths["random_A"])
    coverage_by_k(metrics, random_a, cfg["T_minutes"]).to_csv(
        paths["coverage_by_k"], index=False, encoding="utf-8")
    table.to_csv(paths["m2"], index=False, encoding="utf-8")
    paths["m2_noise_sites"].write_text(
        json.dumps({"parameters": cfg["m2"], "seed": cfg["seed"],
                    "B_prime_sites": {str(k): v for k, v in noise_sites.items()}},
                   ensure_ascii=False), encoding="utf-8")
    return paths
