"""Тесты п. 4.6 и 4.7: перенос населения и признаки клеток.

Растр создаётся синтетический, чтобы ответ был известен заранее.
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, Point, box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seto.data import features as F, population as POP  # noqa: E402

UTM = "EPSG:32637"


def make_raster(path: Path, values: np.ndarray, origin=(400_000.0, 6_200_000.0),
                pixel=100.0, crs=UTM, nodata=-200.0) -> Path:
    """Растр населения в той же проекции, что и сетка (для простоты счёта)."""
    import rasterio
    from rasterio.transform import from_origin

    with rasterio.open(
            path, "w", driver="GTiff", height=values.shape[0],
            width=values.shape[1], count=1, dtype="float32", crs=crs,
            transform=from_origin(origin[0], origin[1], pixel, pixel),
            nodata=nodata) as dst:
        dst.write(values.astype("float32"), 1)
    return path


def make_grid(x0=400_000.0, y0=6_199_500.0, size=500.0, cols=2, rows=1):
    cells, ids = [], []
    for row in range(rows):
        for col in range(cols):
            cells.append(box(x0 + col * size, y0 + row * size,
                             x0 + (col + 1) * size, y0 + (row + 1) * size))
            ids.append(f"cell_{row * cols + col:06d}")
    return gpd.GeoDataFrame({"cell_id": ids}, geometry=cells, crs=UTM)


# --- перенос населения ---

def test_population_is_conserved(tmp_path):
    """Главная проверка п. 4.6: ни один человек не потерялся и не удвоился."""
    values = np.full((5, 10), 7.0)          # 50 пикселей по 100 м
    raster = make_raster(tmp_path / "pop.tif", values)
    grid = make_grid()
    transfer = POP.to_cells(raster, grid)

    assert transfer.transferred_total == pytest.approx(values.sum())
    assert transfer.loss_share < 1e-9


def test_population_splits_between_cells_by_area(tmp_path):
    """Ровно половина пикселей в каждой клетке — поровну и население."""
    values = np.full((5, 10), 2.0)
    raster = make_raster(tmp_path / "pop.tif", values)
    transfer = POP.to_cells(raster, make_grid())
    per_cell = transfer.per_cell.to_numpy()
    assert per_cell[0] == pytest.approx(per_cell[1])
    assert per_cell.sum() == pytest.approx(100.0)


def test_pixel_on_border_is_split_proportionally(tmp_path):
    """Пиксель поперёк границы клеток делится по площади, а не целиком."""
    # сетка сдвинута на 50 м — каждый пиксель колонки 5 разрезан пополам
    values = np.zeros((1, 10)); values[0, 4] = 10.0; values[0, 5] = 10.0
    raster = make_raster(tmp_path / "pop.tif", values,
                         origin=(400_000.0, 6_200_000.0))
    grid = make_grid(x0=399_950.0, y0=6_199_900.0, size=500.0, cols=2)
    transfer = POP.to_cells(raster, grid)
    assert transfer.transferred_total == pytest.approx(20.0)
    # пиксель 4 (400400-400500) целиком в первой клетке (399950-400450)? нет:
    # он разрезан, значит обе клетки получают ненулевую часть
    assert (transfer.per_cell.to_numpy() > 0).all()


def test_nodata_is_not_counted_as_population(tmp_path):
    values = np.full((5, 10), -200.0)
    values[0, 0] = 5.0
    raster = make_raster(tmp_path / "pop.tif", values)
    transfer = POP.to_cells(raster, make_grid())
    assert transfer.transferred_total == pytest.approx(5.0)
    assert transfer.nodata_pixels == 49


def test_negative_values_are_dropped(tmp_path):
    values = np.full((5, 10), 1.0)
    values[0, 0] = -3.0          # служебное значение, не nodata
    raster = make_raster(tmp_path / "pop.tif", values)
    transfer = POP.to_cells(raster, make_grid())
    assert transfer.transferred_total == pytest.approx(49.0)


def test_geographic_grid_is_rejected(tmp_path):
    raster = make_raster(tmp_path / "pop.tif", np.ones((2, 2)))
    grid = make_grid().to_crs(4326)
    with pytest.raises(ValueError, match="метрической"):
        POP.to_cells(raster, grid)


def test_empty_raster_raises(tmp_path):
    raster = make_raster(tmp_path / "pop.tif", np.zeros((5, 10)))
    with pytest.raises(ValueError, match="ни одного пикселя"):
        POP.to_cells(raster, make_grid())


# --- итоги регистра, разложенные по растру (задача 3.2 TODO_V2) ---
# Сетка — две клетки 500 м: x 400 000–400 500 и 400 500–401 000. Участок A
# занимает x 400 000–400 700 (пересекает обе клетки), участок B — 400 700–401 000.

def _sections(with_b: bool = True):
    rows = [("A", box(400_000, 6_199_500, 400_700, 6_200_000))]
    if with_b:
        rows.append(("B", box(400_700, 6_199_500, 401_000, 6_200_000)))
    return gpd.GeoDataFrame({"code": [r[0] for r in rows]},
                            geometry=[r[1] for r in rows], crs=UTM)


def _totals(**sections):
    """Итоги по участкам: code=(всего, до 18, 18–64, 65+)."""
    import pandas as pd
    return pd.DataFrame(
        [(code, *values) for code, values in sections.items()],
        columns=["code", "total", "under_18", "adults_18_64", "seniors_65"]
    ).set_index("code").astype(float)


def test_registry_totals_spread_by_area_when_raster_is_uniform(tmp_path):
    """5/7 участка A лежит в первой клетке — туда и 5/7 его жителей."""
    raster = make_raster(tmp_path / "pop.tif", np.ones((5, 10)))
    transfer = POP.registry_to_cells(
        raster, make_grid(), _sections(),
        _totals(A=(70, 10, 40, 20), B=(300, 0, 200, 100)))
    cells = transfer.groups
    assert cells["total"].tolist() == pytest.approx([50, 320])
    assert cells["seniors_65"].tolist() == pytest.approx([20 * 5 / 7, 20 * 2 / 7 + 100])
    assert transfer.loss_share < 1e-9


def test_raster_decides_where_people_live_inside_a_section(tmp_path):
    """Внутри A растр плотнее слева (3 на пиксель против 1) — туда и люди."""
    values = np.ones((5, 10))
    values[:, :5] = 3.0
    raster = make_raster(tmp_path / "pop.tif", values)
    cells = POP.registry_to_cells(raster, make_grid(), _sections(),
                                  _totals(A=(85, 0, 85, 0), B=(0, 0, 0, 0))).groups
    # в A: 25 пикселей по 3 слева и 10 по 1 справа — веса 75 и 10
    assert cells["total"].tolist() == pytest.approx([75, 10])


def test_registry_overrides_raster_totals(tmp_path):
    """Аэропорт: растр видит в участке тысячи «жителей», регистр — троих."""
    values = np.ones((5, 10))
    values[:, 7:] = 1000.0
    raster = make_raster(tmp_path / "pop.tif", values)
    transfer = POP.registry_to_cells(raster, make_grid(), _sections(),
                                     _totals(A=(70, 0, 70, 0), B=(3, 0, 3, 0)))
    assert transfer.groups.loc["cell_000001", "total"] == pytest.approx(20 + 3)
    assert transfer.report["raster_total_in_sections"] == pytest.approx(35 + 15_000)


def test_section_without_raster_is_spread_by_area_and_reported(tmp_path):
    values = np.ones((5, 10))
    values[:, 7:] = 0.0
    raster = make_raster(tmp_path / "pop.tif", values)
    transfer = POP.registry_to_cells(raster, make_grid(), _sections(),
                                     _totals(A=(70, 0, 70, 0), B=(30, 0, 30, 0)))
    assert transfer.groups["total"].tolist() == pytest.approx([50, 50])
    assert transfer.report["sections_without_raster"] == {"B": 30}


def test_cell_remembers_the_section_that_gave_it_most_people(tmp_path):
    """Клетка 1 лежит в A и B, но люди в ней в основном из B."""
    raster = make_raster(tmp_path / "pop.tif", np.ones((5, 10)))
    transfer = POP.registry_to_cells(
        raster, make_grid(), _sections(),
        _totals(A=(70, 10, 40, 20), B=(300, 0, 200, 100)))
    assert transfer.main_section.tolist() == ["A", "B"]


def test_empty_cell_has_no_main_section(tmp_path):
    raster = make_raster(tmp_path / "pop.tif", np.ones((5, 10)))
    sections = gpd.GeoDataFrame(
        {"code": ["A"]}, geometry=[box(400_000, 6_199_500, 400_400, 6_200_000)], crs=UTM)
    transfer = POP.registry_to_cells(raster, make_grid(), sections,
                                     _totals(A=(70, 0, 70, 0)))
    assert transfer.main_section.iloc[0] == "A"
    assert transfer.main_section.isna().iloc[1]


def test_section_without_boundary_is_reported_not_spread(tmp_path):
    """Участок есть в регистре, но не на карте: класть его людей некуда."""
    raster = make_raster(tmp_path / "pop.tif", np.ones((5, 10)))
    transfer = POP.registry_to_cells(raster, make_grid(), _sections(with_b=False),
                                     _totals(A=(70, 0, 70, 0), B=(3, 0, 3, 0)))
    assert transfer.groups["total"].sum() == pytest.approx(70)
    assert transfer.report["sections_without_geometry"] == {"B": 3}
    assert transfer.report["source_total"] == 70


# --- признаки клеток ---

def _buildings(*specs):
    """specs: (building, geometry, levels)."""
    return gpd.GeoDataFrame(
        {"building": [s[0] for s in specs], "levels": [s[2] for s in specs]},
        geometry=[s[1] for s in specs], crs=UTM)


def test_built_share_is_area_fraction():
    grid = make_grid(cols=1)                      # одна клетка 500x500 = 250000 м²
    houses = _buildings(("apartments", box(400_000, 6_199_500, 400_100, 6_199_600), 5.0))
    result = F.build(grid, {"buildings": houses}, None, None)
    assert result["built_share"].iloc[0] == pytest.approx(10_000 / 250_000)
    assert result["residential_share"].iloc[0] == pytest.approx(0.04)
    assert result["public_share"].iloc[0] == 0.0


def test_overlapping_polygons_do_not_double_count():
    """Иначе доля площади могла бы превысить единицу."""
    grid = make_grid(cols=1)
    overlapping = _buildings(
        ("apartments", box(400_000, 6_199_500, 400_200, 6_199_700), 5.0),
        ("apartments", box(400_100, 6_199_600, 400_300, 6_199_800), 9.0))
    result = F.build(grid, {"buildings": overlapping}, None, None)
    assert result["built_share"].iloc[0] <= 1.0
    # площадь объединения = 2*40000 - 10000 = 70000
    assert result["built_share"].iloc[0] == pytest.approx(70_000 / 250_000)


def test_mean_levels_ignores_missing():
    grid = make_grid(cols=1)
    houses = _buildings(
        ("apartments", box(400_000, 6_199_500, 400_050, 6_199_550), 4.0),
        ("apartments", box(400_100, 6_199_600, 400_150, 6_199_650), 8.0),
        ("apartments", box(400_200, 6_199_700, 400_250, 6_199_750), None))
    result = F.build(grid, {"buildings": houses}, None, None)
    assert result["mean_levels"].iloc[0] == pytest.approx(6.0)
    assert result["levels_known_share"].iloc[0] == pytest.approx(2 / 3)


def test_barrier_shares_are_separate():
    grid = make_grid(cols=1)
    water = gpd.GeoDataFrame(
        geometry=[box(400_000, 6_199_500, 400_250, 6_199_750)], crs=UTM)
    landuse = gpd.GeoDataFrame(
        {"landuse": ["industrial"]},
        geometry=[box(400_250, 6_199_750, 400_500, 6_200_000)], crs=UTM)
    result = F.build(grid, {"water": water, "landuse": landuse}, None, None)
    assert result["water_share"].iloc[0] == pytest.approx(0.25)
    assert result["landuse_industrial_share"].iloc[0] == pytest.approx(0.25)
    assert result["landuse_railway_share"].iloc[0] == 0.0


def test_points_are_counted_per_cell():
    grid = make_grid(cols=2)
    points = gpd.GeoDataFrame(
        {"amenity": ["school", "school", "pharmacy"],
         "shop": [None, None, None], "public_transport": [None, None, None],
         "highway": [None, None, None]},
        geometry=[Point(400_100, 6_199_600), Point(400_200, 6_199_700),
                  Point(400_700, 6_199_600)], crs=UTM)
    result = F.build(grid, {}, None, points)
    assert result["schools"].tolist() == [2, 0]
    assert result["pharmacies"].tolist() == [0, 1]


def test_street_density_is_metres_per_km2():
    grid = make_grid(cols=1)        # 0.25 км²
    roads = gpd.GeoDataFrame(
        geometry=[LineString([(400_000, 6_199_600), (400_500, 6_199_600)])], crs=UTM)
    result = F.build(grid, {}, roads, None)
    assert result["street_density_m_per_km2"].iloc[0] == pytest.approx(500 / 0.25)


def test_features_reject_geographic_crs():
    with pytest.raises(ValueError, match="метрической"):
        F.build(make_grid().to_crs(4326), {}, None, None)


def test_polygon_poi_are_counted_too():
    """Школы и супермаркеты в OSM чаще полигоны, чем точки.

    На районе MVP счёт только по точкам давал 9 школ вместо нескольких
    десятков — поэтому полигональные POI тоже учитываются."""
    grid = make_grid(cols=1)
    points = gpd.GeoDataFrame(
        {"amenity": ["school"], "shop": [None],
         "public_transport": [None], "highway": [None]},
        geometry=[Point(400_100, 6_199_600)], crs=UTM)
    poi_polygons = gpd.GeoDataFrame(
        {"amenity": ["school", "kindergarten"], "shop": [None, None],
         "public_transport": [None, None], "highway": [None, None]},
        geometry=[box(400_200, 6_199_600, 400_300, 6_199_700),
                  box(400_300, 6_199_700, 400_400, 6_199_800)], crs=UTM)
    result = F.build(grid, {"poi_polygons": poi_polygons}, None, points)
    assert result["schools"].iloc[0] == 2          # точка плюс полигон
    assert result["kindergartens"].iloc[0] == 1


def test_merging_poi_keeps_columns_absent_in_polygons():
    """Регрессия: пересечение колонок обнуляло остановки.

    У полигонов нет highway и public_transport, и если объединять точки
    с полигонами по общим колонкам, счёт остановок молча становится нулём."""
    grid = make_grid(cols=1)
    points = gpd.GeoDataFrame(
        {"amenity": [None], "shop": [None], "public_transport": [None],
         "highway": ["bus_stop"]},
        geometry=[Point(400_100, 6_199_600)], crs=UTM)
    poi_polygons = gpd.GeoDataFrame(
        {"amenity": ["school"], "shop": [None]},
        geometry=[box(400_200, 6_199_600, 400_300, 6_199_700)], crs=UTM)
    result = F.build(grid, {"poi_polygons": poi_polygons}, None, points)
    assert result["transport_stops"].iloc[0] == 1
    assert result["schools"].iloc[0] == 1
