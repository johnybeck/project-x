"""Базовая сеть из официального реестра медучреждений (пропозал v2, п. 3.2).

В Москве реестра с координатами нет, и сеть приходится выделять по названию
в OSM (`data/clinics.py`) — это прокси. Там, где реестр есть (Мадрид), сеть
берётся из него: тип учреждения и подчинённость в реестре записаны полями, а
не угадываются по словам.

Правила отбора лежат в конфиге города (раздел `registry`), здесь только их
применение. Как и в `clinics.py`, каждое исключение записывается с правилом,
а числа обязаны сходиться: отобрано + исключено + без координат = кандидаты.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import pandas as pd


@dataclass
class RegistrySelection:
    """Результат отбора из реестра: что взяли, что выбросили и почему."""
    frame: gpd.GeoDataFrame
    no_coordinates: pd.DataFrame
    candidates: int = 0
    excluded_by: dict[str, int] = field(default_factory=dict)

    @property
    def selected(self) -> int:
        return len(self.frame)

    @property
    def uncertain(self) -> pd.DataFrame:
        """Что нужно человеку: учреждения сети без координат.

        Они прошли все правила, но поставить их на карту реестр не даёт;
        prepare-city сохраняет этот список рядом с прогоном."""
        return self.no_coordinates

    def check(self) -> None:
        """Отобранные, исключённые и без координат обязаны дать всех кандидатов."""
        total = self.selected + len(self.no_coordinates) + sum(self.excluded_by.values())
        if total != self.candidates:
            raise ValueError(
                f"отбор из реестра не сходится: {self.selected} отобрано + "
                f"{len(self.no_coordinates)} без координат + "
                f"{sum(self.excluded_by.values())} исключено != "
                f"{self.candidates} кандидатов")

    def as_dict(self) -> dict:
        manual = self.frame.index[self.frame["coord_source"] == "manual"]
        return {"source": "registry", "candidates": self.candidates,
                "selected": self.selected, "excluded_by": self.excluded_by,
                "manual_points": manual.tolist(),
                "no_coordinates": len(self.no_coordinates),
                "no_coordinates_ids": self.no_coordinates.index.tolist()}


def read(path: Path, rules: dict) -> pd.DataFrame:
    """Прочитать реестр как есть: все поля строками, без догадок о типах."""
    return pd.read_csv(path, sep=rules.get("sep", ";"),
                       encoding=rules.get("encoding", "utf-8"), dtype=str)


def collapse(rows: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """Одна строка на учреждение.

    В реестре строка — это учреждение × услуга, поэтому одно учреждение
    повторяется десятки раз. Поля, по которым идёт отбор, обязаны быть у всех
    его строк одинаковыми: иначе непонятно, по какому значению судить, и
    молча взять первое — значит угадывать."""
    id_column = rules["id_column"]
    columns = [rules["scope"]["column"],
               *(rule["column"] for rule in rules.get("require", {}).values())]
    for column in columns:
        variants = rows.groupby(id_column)[column].nunique(dropna=False)
        conflicting = variants[variants > 1]
        if len(conflicting):
            raise ValueError(
                f"в реестре у {len(conflicting)} учреждений разные значения "
                f"поля {column}: {conflicting.index[:5].tolist()}")
    return rows.drop_duplicates(id_column).set_index(id_column)


def _matches(values: pd.Series, allowed: list[str]) -> pd.Series:
    """Сравнение без пробелов по краям: в выгрузке значения бывают с хвостами."""
    return values.fillna("").str.strip().isin([v.strip() for v in allowed])


def select(centres: pd.DataFrame, rules: dict) -> RegistrySelection:
    """Отобрать сеть по правилам конфига и поставить точки на карту."""
    scope = rules["scope"]
    candidates = centres[_matches(centres[scope["column"]], scope["values"])]

    remaining = candidates
    excluded_by: dict[str, int] = {}
    for rule_name, rule in rules.get("require", {}).items():
        passed = _matches(remaining[rule["column"]], rule["values"])
        excluded_by[rule_name] = int((~passed).sum())
        remaining = remaining[passed]

    x = pd.to_numeric(remaining[rules["x_column"]], errors="coerce")
    y = pd.to_numeric(remaining[rules["y_column"]], errors="coerce")
    source = pd.Series("registry", index=remaining.index, dtype=object)
    note = pd.Series(None, index=remaining.index, dtype=object)

    manual = rules.get("manual_points") or {}
    _check_manual(manual, remaining.index, x.notna() & y.notna())
    for centre, point in manual.items():
        x[centre], y[centre] = float(point["x"]), float(point["y"])
        source[centre], note[centre] = "manual", point["source"]

    located = x.notna() & y.notna()
    kept = remaining[located]
    frame = gpd.GeoDataFrame(
        kept.assign(coord_source=source[located], coord_note=note[located]),
        geometry=gpd.points_from_xy(x[located], y[located]),
        crs=rules["crs"])

    selection = RegistrySelection(frame=frame, no_coordinates=remaining[~located],
                                  candidates=len(candidates), excluded_by=excluded_by)
    selection.check()
    return selection


def _check_manual(manual: dict, network: pd.Index, located: pd.Series) -> None:
    """Ручная точка — только дыра в реестре, а не его правка.

    Точка для учреждения вне сети значит, что конфиг устарел или в нём
    опечатка; точка для учреждения с координатами молча перебила бы реестр.
    И то и другое — ошибка, а не повод выбрать одно из двух."""
    outside = [centre for centre in manual if centre not in network]
    if outside:
        raise ValueError(f"ручные точки для учреждений вне сети: {outside}")
    overriding = [centre for centre in manual if located[centre]]
    if overriding:
        raise ValueError(
            f"ручные точки для учреждений, у которых координаты в реестре есть: "
            f"{overriding} — реестр ручной точкой не перебивается")
    unexplained = [centre for centre, point in manual.items() if not point.get("source")]
    if unexplained:
        raise ValueError(f"у ручных точек не указан источник: {unexplained}")


def load(path: Path, rules: dict) -> RegistrySelection:
    """Прочитать реестр, схлопнуть по учреждениям и отобрать сеть."""
    return select(collapse(read(path, rules), rules), rules)
