"""Сравнение версий населения (задача 3.5; решение 03.11, PROPOSAL_CHANGES п. 2).

Одни и те же варианты считаются на трёх картах жителей: основной (реестр,
внутри участка по квартирам) и двух проверках. Сравниваются:

1. порядок вариантов по главной метрике;
2. главные разницы: B − A, M − B, а при скорах модели — D − C и D − B;
3. М2: выше ли потолок порога шума и при каких k;
4. перекрёстная оценка: решение, найденное по карте проверки, оценивается
   по основной карте.

Пункт 4 — та же ошибка замещения, что М2 (Hodgson, 2002), но для источника
данных о населении. Её можно поставить рядом с потолком М2: что весит больше —
точность карты жителей или знание о том, где живут пожилые.

Карта жителей не меняет ни граф, ни допустимые площадки, поэтому времена у
версий общие. Это проверяется: иначе перекрёстная оценка сравнивала бы разные
задачи, а не разные карты.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..data import demography as demography_mod
from ..metrics.coverage import coverage, times_with
from ..metrics.substitution import need_time

VARIANTS = ("B", "BN", "M", "C", "D", "D10", "CN", "DN", "E", "EN", "EP")
# что переносится на основную карту: решения, которые опираются на карту жителей
CROSS_VARIANTS = ("B", "M", "C", "D", "D10")
_SHARED_TIMES = ("base", "sites", "site_index", "allowed")


@dataclass
class Version:
    """Карта жителей: папка подготовки (prepare-city или run-city) и step2 на ней."""
    name: str
    prepared: Path
    step2: Path

    @property
    def method(self) -> str:
        cfg = yaml.safe_load((self.prepared / "effective_config.yaml")
                             .read_text(encoding="utf-8"))
        return cfg["city"]["population"]["method"]

    def times(self):
        return np.load(self.prepared / "times.npz")

    def features(self) -> pd.DataFrame:
        return pd.read_parquet(self.prepared / "cell_features.parquet")

    def solutions(self) -> dict[int, dict[str, list[int]]]:
        raw = json.loads((self.step2 / "solutions.json").read_text(encoding="utf-8"))
        return {int(k): {item["variant"]: item["sites"] for item in items
                         if item.get("sites")}
                for k, items in raw.items()}


def check_same_task(versions: list[Version]) -> None:
    """Времена и площадки у всех версий обязаны совпадать (см. докстринг модуля)."""
    main = versions[0].times()
    for version in versions[1:]:
        other = version.times()
        for key in _SHARED_TIMES:
            if not np.array_equal(main[key], other[key]):
                raise ValueError(
                    f"у версии {version.name} другие {key!r}, чем у основной: "
                    "версии должны различаться только картой жителей")
    cells = versions[0].features()["cell_id"]
    for version in versions[1:]:
        if not version.features()["cell_id"].equals(cells):
            raise ValueError(f"у версии {version.name} другой порядок клеток")


def variant_order(values: pd.Series, digits: int = 2) -> str:
    """Порядок вариантов по ΔCov_T: «M > B = BN > E». Равные до сотой п.п. — через «=»."""
    rounded = values.dropna().round(digits).sort_values(ascending=False, kind="stable")
    groups, previous = [], None
    for name, value in rounded.items():
        if groups and value == previous:
            groups[-1] += f" = {name}"
        else:
            groups.append(str(name))
        previous = value
    return " > ".join(groups)


def _pair(differences: pd.DataFrame, k: int, pair: str) -> float | None:
    row = differences[(differences["k"] == k) & (differences["pair"] == pair)]
    return float(row["dCov_15_pp"].iloc[0]) if len(row) else None


def summary(version: Version, need_name: str) -> pd.DataFrame:
    """Строка на k: главная метрика, порядок вариантов, разницы и М2 версии."""
    by_k = pd.read_csv(version.step2 / "coverage_by_k.csv").set_index("k")
    differences = pd.read_csv(version.step2 / "differences.csv")
    metrics = pd.read_csv(version.step2 / "metrics.csv")
    m2 = pd.read_csv(version.step2 / "m2.csv")
    m2 = m2[m2["need"] == need_name].set_index("k")
    base = metrics.loc[metrics["variant"] == "исходная сеть", "Cov_15"].iloc[0]
    rows = []
    for k, row in by_k.iterrows():
        present = [name for name in VARIANTS if name in row.index]
        rows.append({
            "version": version.name, "method": version.method, "k": k,
            "base_Cov_15": base, "A_median": row["A_median"], "A_p95": row["A_p95"],
            **{name: row[name] for name in present},
            "order": variant_order(row[present]),
            "B_minus_A": _pair(differences, k, "B − A"),
            "M_minus_B": _pair(differences, k, "M − B"),
            "D_minus_C": _pair(differences, k, "D − C"),
            "D_minus_B": (row["D"] - row["B"]) if "D" in present else None,
            "m2_ceiling_per_1000": m2.loc[k, "ceiling_per_1000"],
            "m2_noise_per_1000": m2.loc[k, "noise_per_1000"],
            "m2_above_noise": bool(m2.loc[k, "above_noise"]),
        })
    return pd.DataFrame(rows)


def main_need(features: pd.DataFrame, cfg: dict) -> np.ndarray:
    """Основное определение нуждаемости по клеткам основной версии."""
    rules = cfg["city"]["need"]
    needs = demography_mod.need_all(demography_mod.cell_groups(features), rules,
                                    cfg["city"]["demography"]["groups"])
    return needs[rules["main"]].to_numpy(dtype=float)


def cross_evaluate(versions: list[Version], need: np.ndarray, threshold: float,
                   t_max: float) -> pd.DataFrame:
    """Решения каждой проверки — на основной карте, против решения того же варианта,
    найденного по основной карте: ΔCov_T по основному P и Z_N по основному N,
    чел.-мин на 1000 нуждающихся.

    У B потеря Z_N — та же ошибка замещения, что потолок М2, только от карты
    жителей, а не от незнания нуждаемости: их можно ставить рядом. Потеря ΔCov_T
    бывает отрицательной: B и варианты с моделью ищут минимум суммарного времени,
    а не покрытия, и чужое решение может покрыть больше."""
    main = versions[0]
    times = main.times()
    population = main.features()["population"].to_numpy(dtype=float)
    base_cov = coverage(np.minimum(times["base"], t_max), population, threshold)

    def on_main(sites) -> tuple[float, float]:
        after = times_with(times["base"], times["sites"][list(sites)], t_max)
        return (coverage(after, population, threshold) - base_cov,
                need_time(times["sites"], times["base"], sites, need, t_max))

    own = main.solutions()
    rows = []
    for version in versions[1:]:
        for k, solutions in version.solutions().items():
            for name in CROSS_VARIANTS:
                if name not in solutions or name not in own[k]:
                    continue
                d_cov, z_n = on_main(solutions[name])
                d_cov_own, z_n_own = on_main(own[k][name])
                rows.append({
                    "version": version.name, "k": k, "variant": name,
                    "sites": solutions[name], "main_sites": own[k][name],
                    "same_sites": set(solutions[name]) == set(own[k][name]),
                    "dCov_on_main": d_cov, "dCov_main_own": d_cov_own,
                    "dCov_loss_pp": d_cov_own - d_cov,
                    "Z_N_loss_per_1000": 1000 * (z_n - z_n_own) / float(need.sum()),
                })
    return pd.DataFrame(rows)
