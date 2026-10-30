"""Жилые здания кадастра Испании (INSPIRE, тема Buildings) с годом постройки.

Зачем кадастр, а не OSM: в OSM Мадрида год есть у 13 жилых зданий из 69 180, и
текст варианта C остался бы без эпохи (MADRID_DATA п. 8). В кадастре год есть
у 99,99 % жилых зданий.

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
    только для того, чтобы попасть точкой в клетку."""
    patches = []
    for ring in building.iter(f"{_GML}exterior"):
        values = ring.find(f"{_GML}LinearRing/{_GML}posList").text.split()
        patches.append(Polygon(zip(map(float, values[::2]), map(float, values[1::2]))))
    return MultiPolygon(patches)


def read(archive: Path, rules: dict) -> gpd.GeoDataFrame:
    """Жилые здания: год постройки (NaN, если неизвестен) и точка внутри контура.

    rules — раздел `building_years` конфига города: проекция файла и код
    жилого назначения."""
    rows, shapes = [], []
    with zipfile.ZipFile(archive) as bundle:
        inside = [n for n in bundle.namelist() if n.endswith(_BUILDINGS_FILE)]
        if len(inside) != 1:
            raise ValueError(f"в архиве {archive} не один файл зданий, а {len(inside)}")
        with bundle.open(inside[0]) as stream:
            for _, element in ET.iterparse(stream, events=("end",)):
                if element.tag != f"{_EXT}Building":
                    continue
                if element.findtext(f"{_EXT}currentUse") == rules["residential_use"]:
                    begin = element.findtext(
                        f"{_CORE}dateOfConstruction/{_CORE}DateOfEvent/{_CORE}beginning")
                    rows.append(_year(begin))
                    shapes.append(_footprint(element))
                element.clear()      # иначе дерево всего файла останется в памяти

    if not rows:
        raise ValueError(f"в {archive} нет зданий с назначением "
                         f"{rules['residential_use']!r}: проверь раздел building_years")
    homes = gpd.GeoDataFrame({"year": rows}, geometry=shapes, crs=rules["crs"])
    return homes.assign(geometry=homes.geometry.representative_point())
