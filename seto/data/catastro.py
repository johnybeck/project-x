"""Здания кадастра Испании (INSPIRE, тема Buildings): год постройки и квартиры.

Зачем кадастр, а не OSM: в OSM Мадрида год есть у 13 жилых зданий из 69 180, и
текст варианта C остался бы без эпохи (MADRID_DATA п. 8). В кадастре год есть
у 99,99 % жилых зданий. Число квартир нужно населению: кадастр говорит, где
люди живут, а растр GHS кладёт их и на склады (MADRID_DATA п. 9).

Формат посмотрен на файле A.ES.SDGC.BU.28900.zip: внутри GML на 600 МБ, поэтому
читаем потоком, а не целиком. Одно «здание» кадастра — все постройки участка:
контур состоит из нескольких лоскутов, а дат две, самой старой постройки
(beginning) и самой новой (end). Берём beginning: дом стоит с первой постройки,
а поздняя пристройка не делает квартал новым. Эпоха по двум датам совпадает
у 95 % зданий.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import MultiPolygon, Polygon

_CORE = "{http://inspire.jrc.ec.europa.eu/schemas/bu-core2d/2.0}"
_EXT = "{http://inspire.jrc.ec.europa.eu/schemas/bu-ext2d/2.0}"
_GML = "{http://www.opengis.net/gml/3.2}"
_BUILDINGS_FILE = ".building.gml"

# Пустой год приходит как «--01-01T00:00:00», испорченный — как «975-01-01»
_YEAR = re.compile(r"^(\d{4})-")


def _year(text: str | None) -> float:
    """Год из даты «1963-01-01T00:00:00»; нет года — NaN."""
    match = _YEAR.match(text or "")
    return float(match.group(1)) if match else np.nan


def _footprint(building: ET.Element) -> MultiPolygon:
    """Контур участка застройки. Внутренние кольца не нужны: контур здесь
    только для того, чтобы попасть в клетку и в участок падрона."""
    patches = []
    for ring in building.iter(f"{_GML}exterior"):
        values = ring.find(f"{_GML}LinearRing/{_GML}posList").text.split()
        patches.append(Polygon(zip(map(float, values[::2]), map(float, values[1::2]))))
    return MultiPolygon(patches)


def read(archive: Path, rules: dict) -> gpd.GeoDataFrame:
    """Здания с жильём: год постройки (NaN, если неизвестен), число квартир,
    признак жилого назначения и контур.

    Берём жилые по назначению и любые другие, где есть квартиры: в Мадриде
    таких квартир 5,7 тыс. из 1,53 млн, и жить в них это не мешает.
    rules — раздел `buildings` конфига города: проекция файла и код жилого
    назначения."""
    rows, shapes = [], []
    with zipfile.ZipFile(archive) as bundle:
        inside = [n for n in bundle.namelist() if n.endswith(_BUILDINGS_FILE)]
        if len(inside) != 1:
            raise ValueError(f"в архиве {archive} не один файл зданий, а {len(inside)}")
        with bundle.open(inside[0]) as stream:
            for _, element in ET.iterparse(stream, events=("end",)):
                if element.tag != f"{_EXT}Building":
                    continue
                residential = element.findtext(f"{_EXT}currentUse") == rules["residential_use"]
                dwellings = int(element.findtext(f"{_EXT}numberOfDwellings") or 0)
                if residential or dwellings > 0:
                    begin = element.findtext(
                        f"{_CORE}dateOfConstruction/{_CORE}DateOfEvent/{_CORE}beginning")
                    rows.append({"year": _year(begin), "dwellings": dwellings,
                                 "residential": residential})
                    shapes.append(_footprint(element))
                element.clear()      # иначе дерево всего файла останется в памяти

    if not any(row["residential"] for row in rows):
        raise ValueError(f"в {archive} нет зданий с назначением "
                         f"{rules['residential_use']!r}: проверь раздел buildings")
    return gpd.GeoDataFrame(rows, geometry=shapes, crs=rules["crs"])


def homes(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Жилые здания точками: чтобы отнести здание к эпохе в клетке, контур не
    нужен, хватает точки внутри него."""
    residential = buildings[buildings["residential"]]
    return residential.assign(geometry=residential.geometry.representative_point())
