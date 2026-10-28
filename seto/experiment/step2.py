"""Шаг 2 пропозала v2 (п. 3.9): A, B, M, BN и М2 — без снимков и без модели.

Самый дешёвый ответ на вопрос «есть ли снимку что закрывать». Если потолок
М2 везде ниже порога шума, это результат о постановке, и прогоны VLM можно
сократить; если выше хотя бы при каком-то k — изображению есть что отыграть.

A и M считаются при основном k (и k = 1 — устойчивость), B, BN и М2 — при
всех k из `m2.k_values`: п. 3.4 пропозала оставляет k = 5 и 10 «только М2».
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..costs.timer import CostLog
from ..data import demography as demography_mod
from ..metrics import substitution as m2_mod
from .evaluate import run_variants
from .runner import _save_solutions


def needs_for(data, cfg: dict) -> dict[str, np.ndarray]:
    """Все определения нуждаемости из конфига города — по клеткам."""
    city = cfg["city"]
    if getattr(data, "groups", None) is None or "need" not in city:
        raise ValueError("у города нет возрастных групп или раздела need — "
                         "М2 без нуждаемости не считается")
    variants = demography_mod.need_all(data.groups, city["need"],
                                       city["demography"]["groups"])
    return {name: series.to_numpy(dtype=float) for name, series in variants.items()}


def run_step2(data, cfg: dict, out_dir: Path, logger=None,
              costs: CostLog | None = None) -> dict[str, Path]:
    """Посчитать шаг 2 и сохранить всё, из чего выводы пересчитываются."""
    costs = costs or CostLog()
    out_dir.mkdir(parents=True, exist_ok=True)

    evaluation = run_variants(data, {}, cfg, costs, logger)
    _save_solutions(evaluation, out_dir)
    paths = {"solutions": out_dir / "solutions.json",
             "metrics": out_dir / "metrics.csv",
             "differences": out_dir / "differences.csv",
             "m2": out_dir / "m2.csv",
             "m2_noise_sites": out_dir / "m2_noise_sites.json"}
    evaluation.table.to_csv(paths["metrics"], index=False, encoding="utf-8")
    evaluation.differences.to_csv(paths["differences"], index=False, encoding="utf-8")

    needs = needs_for(data, cfg)
    with costs.stage("m2", "shared_prep"):
        table, noise_sites = m2_mod.study(data, needs, cfg, cfg["m2"]["k_values"],
                                          logger)
    table.to_csv(paths["m2"], index=False, encoding="utf-8")
    paths["m2_noise_sites"].write_text(
        json.dumps({"parameters": cfg["m2"], "seed": cfg["seed"],
                    "B_prime_sites": {str(k): v for k, v in noise_sites.items()}},
                   ensure_ascii=False), encoding="utf-8")
    if (solutions := out_dir / "random_A.parquet").exists():
        paths["random_A"] = solutions
    return paths
