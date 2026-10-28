"""Жители по статистическим участкам и возрастным группам (пропозал v2, п. 3.3).

Нуждаемость N_i = P_18–64 + r · P_65+ строится из открытой статистики, а не
из оценки модели. Здесь — первый шаг: итоги групп по каждому участку. Перенос
на клетки (через пиксели GHS-POP) — отдельный шаг, задача 3.2.

Формат таблицы (колонки, код участка, возрастные границы) задаёт конфиг
города, раздел `demography`. Числа обязаны сходиться: группы разбиения дают
всё население участка, а люди без участка не исчезают, а пишутся в отчёт.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class SectionTotals:
    """Итоги по участкам и то, что в них не попало."""
    table: pd.DataFrame          # индекс — код участка; колонки total и группы
    unassigned_people: float     # строки без кода участка
    unassigned_rows: int

    def as_dict(self) -> dict:
        return {"sections": len(self.table),
                "people": float(self.table["total"].sum()),
                "groups": {name: float(self.table[name].sum())
                           for name in self.table.columns if name != "total"},
                "unassigned_people": self.unassigned_people,
                "unassigned_rows": self.unassigned_rows}


def read(path: Path, rules: dict) -> pd.DataFrame:
    """Прочитать таблицу как есть, всё строками."""
    return pd.read_csv(path, sep=rules.get("sep", ";"),
                       encoding=rules.get("encoding", "utf-8"), dtype=str)


def section_codes(rows: pd.DataFrame, rules: dict) -> pd.Series:
    """Код участка INE из кодов района и участка; пусто, если кода нет."""
    district = pd.to_numeric(rows[rules["district_column"]], errors="coerce")
    section = pd.to_numeric(rows[rules["section_column"]], errors="coerce")
    known = district.notna() & section.notna()
    codes = pd.Series(pd.NA, index=rows.index, dtype=object)
    codes[known] = [
        f"{rules['code_prefix']}{int(d):0{rules['district_digits']}d}"
        f"{int(s):0{rules['section_digits']}d}"
        for d, s in zip(district[known], section[known])]
    return codes


def ages(rows: pd.DataFrame, rules: dict) -> pd.Series:
    """Возраст в годах. Последний класс вида «100 o +» считается за его нижнюю
    границу — для групп «65+» и «80+» этого достаточно. Нераспознанный
    возраст — ошибка: тихо выкинуть строку значит потерять людей."""
    text = rows[rules["age_column"]].fillna("").astype(str)
    parsed = pd.to_numeric(text.str.extract(r"(\d+)", expand=False), errors="coerce")
    bad = parsed.isna()
    if bad.any():
        raise ValueError(f"не распознан возраст в {int(bad.sum())} строках: "
                         f"{text[bad].unique()[:5].tolist()}")
    return parsed.astype(int)


def people(rows: pd.DataFrame, rules: dict) -> pd.Series:
    """Жители строки: сумма колонок-счётчиков, пропуск — ноль человек."""
    counts = rows[rules["count_columns"]].apply(pd.to_numeric, errors="coerce")
    return counts.fillna(0).sum(axis=1)


def totals(rows: pd.DataFrame, rules: dict) -> SectionTotals:
    """Итоги по участкам: всё население и каждая группа из конфига."""
    code = section_codes(rows, rules)
    age = ages(rows, rules)
    count = people(rows, rules)
    assigned = code.notna()

    frame = pd.DataFrame({"code": code[assigned], "age": age[assigned],
                          "n": count[assigned]})
    table = pd.DataFrame({"total": frame.groupby("code")["n"].sum()})
    for name, (low, high) in rules["groups"].items():
        inside = frame["age"] >= low
        if high is not None:
            inside &= frame["age"] <= high
        table[name] = frame[inside].groupby("code")["n"].sum()
    table = table.fillna(0.0)
    table.index.name = "code"

    _check_partition(table, rules.get("partition", []))
    return SectionTotals(table=table,
                         unassigned_people=float(count[~assigned].sum()),
                         unassigned_rows=int((~assigned).sum()))


def _check_partition(table: pd.DataFrame, partition: list[str]) -> None:
    """Группы разбиения обязаны дать всё население: иначе границы групп в
    конфиге перекрываются или оставляют дыру, и N_i посчитается не на тех."""
    if not partition:
        return
    gap = (table[partition].sum(axis=1) - table["total"]).abs()
    if (gap > 1e-9).any():
        raise ValueError(
            f"группы {partition} не дают всё население в {int((gap > 1e-9).sum())} "
            "участках — проверь границы групп в конфиге")


def boundary_codes(path: Path, rules: dict) -> set[str]:
    """Коды участков из файла границ (GeoJSON), без контуров районов и т. п."""
    with path.open("r", encoding="utf-8") as f:
        features = json.load(f)["features"]
    exclude = rules.get("exclude", {})
    return {str(feature["properties"][rules["code_column"]])
            for feature in features
            if not any(feature["properties"].get(key) == value
                       for key, value in exclude.items())}


def boundaries(path: Path, rules: dict, crs) -> "gpd.GeoDataFrame":
    """Полигоны участков (колонка code) в проекции сетки, без контуров районов."""
    import geopandas as gpd

    frame = gpd.read_file(path)
    for key, value in rules.get("exclude", {}).items():
        frame = frame[frame[key] != value]
    frame = frame.rename(columns={rules["code_column"]: "code"})
    return frame[["code", "geometry"]].to_crs(crs).reset_index(drop=True)


def match(totals_: SectionTotals, codes: set[str]) -> dict:
    """Сверка участков населения с участками на карте.

    Участок с жителями, но без границ, не перенесётся на клетки — его люди
    пропадут; участок на карте без жителей — нормально (парк, промзона), но
    его тоже видно в отчёте."""
    population = set(totals_.table.index)
    only_population = sorted(population - codes)
    return {"matched": len(population & codes),
            "only_population": {code: float(totals_.table.loc[code, "total"])
                                for code in only_population},
            "only_boundaries": sorted(codes - population)}
