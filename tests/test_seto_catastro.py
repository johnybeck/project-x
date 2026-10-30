"""Тесты чтения зданий кадастра: GML собран вручную по образцу настоящего файла."""
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seto.data import catastro as C  # noqa: E402

RULES = {"crs": 25830, "residential_use": "1_residential"}


def _building(use: str, begin: str, patches=((440_000, 4_470_000),)) -> str:
    """Запись здания: каждый лоскут — квадрат 10 м с углом в заданной точке."""
    rings = "".join(
        f"""<gml:PolygonPatch><gml:exterior><gml:LinearRing>
        <gml:posList srsDimension="2" count="5">{x} {y} {x + 10} {y} {x + 10} {y + 10}
        {x} {y + 10} {x} {y}</gml:posList></gml:LinearRing></gml:exterior></gml:PolygonPatch>"""
        for x, y in patches)
    return f"""<gml:featureMember><bu-ext2d:Building gml:id="ES.SDGC.BU.1">
      <bu-core2d:dateOfConstruction><bu-core2d:DateOfEvent>
        <bu-core2d:beginning>{begin}</bu-core2d:beginning>
        <bu-core2d:end>2020-01-01T00:00:00</bu-core2d:end>
      </bu-core2d:DateOfEvent></bu-core2d:dateOfConstruction>
      <bu-ext2d:geometry><bu-core2d:BuildingGeometry><bu-core2d:geometry>
        <gml:Surface gml:id="s1"><gml:patches>{rings}</gml:patches></gml:Surface>
      </bu-core2d:geometry></bu-core2d:BuildingGeometry></bu-ext2d:geometry>
      <bu-ext2d:currentUse>{use}</bu-ext2d:currentUse>
    </bu-ext2d:Building></gml:featureMember>"""


def _archive(path: Path, *buildings: str, name="A.ES.SDGC.BU.28900.building.gml") -> Path:
    body = f"""<?xml version="1.0" encoding="ISO-8859-1"?>
    <!--Edificios de la D.G. del Catastro.-->
    <gml:FeatureCollection xmlns:gml="http://www.opengis.net/gml/3.2"
      xmlns:bu-core2d="http://inspire.jrc.ec.europa.eu/schemas/bu-core2d/2.0"
      xmlns:bu-ext2d="http://inspire.jrc.ec.europa.eu/schemas/bu-ext2d/2.0">
      {"".join(buildings)}</gml:FeatureCollection>"""
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr(name, body.encode("iso-8859-1"))
        bundle.writestr("A.ES.SDGC.BU.MD.28900.xml", "<metadata/>")
    return path


def test_reads_only_residential_buildings(tmp_path):
    archive = _archive(tmp_path / "bu.zip",
                       _building("1_residential", "1963-01-01T00:00:00"),
                       _building("4_1_office", "1990-01-01T00:00:00"))
    homes = C.read(archive, RULES)
    assert homes["year"].tolist() == [1963.0]
    assert homes.crs.to_epsg() == 25830


def test_year_is_oldest_construction():
    """Берётся beginning, а не end: дом стоит с первой постройки."""
    assert C._year("1963-01-01T00:00:00") == 1963.0


@pytest.mark.parametrize("text", ["--01-01T00:00:00", "975-01-01T00:00:00", "", None])
def test_broken_year_is_missing(text):
    """Пустой год кадастр пишет как «--01-01», и нулём он стать не должен."""
    assert np.isnan(C._year(text))


def test_building_without_year_is_kept(tmp_path):
    """Здание есть, года нет — оно идёт в класс «год неизвестен», а не пропадает."""
    archive = _archive(tmp_path / "bu.zip", _building("1_residential", "--01-01T00:00:00"))
    assert np.isnan(C.read(archive, RULES)["year"].iloc[0])


def test_point_lies_inside_footprint(tmp_path):
    archive = _archive(tmp_path / "bu.zip", _building(
        "1_residential", "1963-01-01T00:00:00",
        patches=((440_000, 4_470_000), (440_100, 4_470_100))))
    point = C.read(archive, RULES).geometry.iloc[0]
    assert (box(440_000, 4_470_000, 440_010, 4_470_010).contains(point)
            or box(440_100, 4_470_100, 440_110, 4_470_110).contains(point))


def test_no_residential_buildings_is_an_error(tmp_path):
    """Неверный код назначения иначе дал бы «нет данных» по всему городу."""
    archive = _archive(tmp_path / "bu.zip", _building("4_1_office", "1990-01-01T00:00:00"))
    with pytest.raises(ValueError, match="1_residential"):
        C.read(archive, RULES)


def test_archive_without_buildings_file_is_an_error(tmp_path):
    archive = _archive(tmp_path / "bu.zip", name="другой.gml")
    with pytest.raises(ValueError, match="не один файл"):
        C.read(archive, RULES)
