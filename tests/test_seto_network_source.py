"""Выбор источника базовой сети в prepare-city (задача 2.3 TODO_V2)."""
import sys
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seto.experiment.prepare import _base_network  # noqa: E402
from seto.settings import ConfigError  # noqa: E402

UTM = "EPSG:25830"

REGISTRY_RULES = {
    "table": "reg",
    "sep": ";",
    "id_column": "id",
    "x_column": "x",
    "y_column": "y",
    "crs": 25830,
    "scope": {"column": "muni", "values": ["Madrid"]},
    "require": {"не первичная помощь": {"column": "tipo",
                                        "values": ["Centro de salud"]}},
}

OSM_RULES = {"include": "поликлин", "typical": r"поликлиник\w*\s*№?\s*\d",
             "exclude": {"детская": "детск"}}


def cfg(city_network: dict | None) -> dict:
    """Общий конфиг с правилами OSM и город с выбранным источником (или без)."""
    city = {"name": "x", "registry": REGISTRY_RULES,
            "sources": {"tables": {"reg": {"file": "reg.csv"}}}}
    if city_network is not None:
        city["base_network"] = city_network
    return {"base_network": OSM_RULES, "city": city,
            "osm": {"clinic_tags": {"amenity": ["clinic"], "healthcare": []}}}


def osm_layers():
    """Две OSM-поликлиники точками, одна из них детская; полигонов нет."""
    points = gpd.GeoDataFrame(
        {"osm_id": ["1", "2"], "amenity": ["clinic", "clinic"],
         "healthcare": [None, None],
         "name": ["Городская поликлиника №1", "Детская поликлиника №2"]},
        geometry=[Point(0, 0), Point(100, 0)], crs=UTM)
    return points, points.iloc[0:0]


def test_registry_source_reads_the_registry_not_osm(tmp_path):
    (tmp_path / "reg.csv").write_text(
        "id;tipo;muni;x;y\n"
        "CS1;Centro de salud;Madrid;440000;4474000\n"
        "H1;Hospital general;Madrid;441000;4474000\n", encoding="utf-8")
    points, polygons = osm_layers()
    clinics, selection, candidates = _base_network(
        cfg({"source": "registry"}), tmp_path, points, polygons, UTM)
    assert clinics["id"].tolist() == ["CS1"]
    assert candidates == 2 and selection.as_dict()["source"] == "registry"
    assert clinics.crs.to_epsg() == 25830


def test_without_source_osm_names_are_used_as_before(tmp_path):
    """Москва и Люцерн источника не задают — у них ничего не меняется."""
    points, polygons = osm_layers()
    clinics, selection, candidates = _base_network(
        cfg(None), tmp_path, points, polygons, UTM)
    assert clinics["name"].tolist() == ["Городская поликлиника №1"]
    assert candidates == 2 and selection.excluded_by == {"детская": 1}


def test_unknown_source_is_a_config_error(tmp_path):
    points, polygons = osm_layers()
    with pytest.raises(ConfigError, match="base_network.source"):
        _base_network(cfg({"source": "registr"}), tmp_path, points, polygons, UTM)


def test_registry_not_downloaded_says_what_to_run(tmp_path):
    points, polygons = osm_layers()
    with pytest.raises(FileNotFoundError, match="acquire x"):
        _base_network(cfg({"source": "registry"}), tmp_path, points, polygons, UTM)
