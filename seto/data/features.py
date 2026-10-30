"""Признаки клетки по её полигону (п. 4.7).

Считаются по ПОЛИГОНУ клетки, а не по bbox и не по центроиду: именно доли
площади нужны правилу допустимых площадок (п. 8), и именно их нельзя было
получить в MVP, где Overpass отдавал полигоны центроидами.

Один файл cell_features.parquet используется трижды: для допустимости (п. 8),
для текста варианта C (п. 7.3) и для передачи команде AHP (п. 10.5).
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd

from ..acquire.osm import parse_other_tags

# какие building=* считаем жилыми и общественными (п. 3.1: «жилая или
# общественная застройка»)
RESIDENTIAL = ("apartments", "house", "detached", "residential", "dormitory",
               "terrace", "semidetached_house")
PUBLIC = ("commercial", "office", "retail", "public", "civic", "school",
          "hospital", "university", "kindergarten")

# Магазины для текста варианта C: продуктовые, по их сетям виден район
SHOPS = ("supermarket", "convenience")

# Эпохи жилых зданий: те же классы, что в вопросе epoch промпта, границы одни
# для текста C и эталона М4. (колонка, подпись, первый год класса); 1955 уже не
# «до 1955», 2000 ещё не «после 2000».
EPOCHS = (("epoch_before_1955", "до 1955", None),
          ("epoch_1955_1975", "1955-1975", 1955),
          ("epoch_1975_2000", "1975-2000", 1975),
          ("epoch_after_2000", "после 2000", 2001))
EPOCH_UNKNOWN = "epoch_unknown"


def _area_share(cells: gpd.GeoDataFrame, polygons: gpd.GeoDataFrame,
                name: str) -> pd.Series:
    """Доля площади клетки, накрытая полигонами слоя.

    Полигоны слоя объединяются перед пересечением: иначе наложения
    (здание внутри landuse) посчитались бы дважды и доля превысила бы 1."""
    empty = pd.Series(0.0, index=cells.index, name=name)
    if polygons is None or polygons.empty:
        return empty

    # Сначала режем по клеткам, и только потом объединяем — внутри клетки
    # полигонов десятки, а на весь город их сотни тысяч, и один общий
    # union_all на городском масштабе считается минутами.
    pieces = gpd.overlay(cells[["cell_id", "geometry"]],
                         polygons[["geometry"]], how="intersection",
                         keep_geom_type=True)
    if pieces.empty:
        return empty

    merged = pieces.dissolve(by="cell_id")      # unary_union внутри каждой клетки
    share = (cells["cell_id"].map(merged.geometry.area).fillna(0.0)
             / cells.geometry.area)
    return share.clip(upper=1.0).rename(name)


def _count_within(cells: gpd.GeoDataFrame, points: gpd.GeoDataFrame,
                  name: str) -> pd.Series:
    """Сколько точек слоя попало в клетку."""
    empty = pd.Series(0, index=cells.index, name=name, dtype="int64")
    if points is None or points.empty:
        return empty
    joined = gpd.sjoin(points[["geometry"]], cells[["cell_id", "geometry"]],
                       how="inner", predicate="within")
    counts = joined.groupby("cell_id").size()
    return cells["cell_id"].map(counts).fillna(0).astype("int64").rename(name)


def _all_poi(points: gpd.GeoDataFrame,
             poi_polygons: gpd.GeoDataFrame | None) -> gpd.GeoDataFrame | None:
    """Точечные и полигональные объекты в одном наборе.

    Школы, детсады и супермаркеты в OSM чаще нарисованы полигоном, а не точкой:
    если считать только точки, школ в районе находится 9 вместо нескольких
    десятков. Полигон представляем внутренней точкой."""
    if poi_polygons is None or poi_polygons.empty:
        return points
    as_points = poi_polygons.assign(
        geometry=poi_polygons.geometry.representative_point())
    if points is None or points.empty:
        return as_points
    # объединяем по union колонок, а не по пересечению: у полигонов нет
    # highway и public_transport, и пересечение молча обнулило бы остановки
    columns = list(dict.fromkeys([*points.columns, *as_points.columns]))
    joined = pd.concat([points.reindex(columns=columns),
                        as_points.reindex(columns=columns)], ignore_index=True)
    return gpd.GeoDataFrame(joined, geometry="geometry", crs=points.crs)


def _epochs(cells: gpd.GeoDataFrame, homes: gpd.GeoDataFrame | None) -> pd.DataFrame:
    """Жилые здания клетки по эпохам: колонка на класс и на неизвестный год.

    Нет источника года — пропуск, а не ноль зданий: это разные вещи."""
    columns = [column for column, _, _ in EPOCHS] + [EPOCH_UNKNOWN]
    if homes is None:
        return pd.DataFrame(np.nan, index=cells.index, columns=columns)

    starts = [first for _, _, first in EPOCHS[1:]]
    known = homes["year"].notna()
    epoch = pd.Series(EPOCH_UNKNOWN, index=homes.index)
    epoch[known] = np.array(columns)[
        np.searchsorted(starts, homes.loc[known, "year"], side="right")]
    joined = gpd.sjoin(homes.assign(epoch=epoch)[["epoch", "geometry"]],
                       cells[["cell_id", "geometry"]], how="inner", predicate="within")
    counts = joined.groupby(["cell_id", "epoch"]).size().unstack(fill_value=0)
    table = counts.reindex(index=cells["cell_id"], columns=columns, fill_value=0)
    return table.astype("int64").set_axis(cells.index)


def _chain_shops(cells: gpd.GeoDataFrame, shops: gpd.GeoDataFrame) -> pd.DataFrame:
    """Магазины клетки с тегом brand: сколько и какие, частые бренды первыми.

    Тег есть у 41 % продуктовых магазинов Мадрида, так что это нижняя оценка."""
    result = pd.DataFrame({"chain_shops": 0, "chain_brands": ""}, index=cells.index)
    brands = shops["other_tags"].map(lambda tags: parse_other_tags(tags).get("brand"))
    branded = shops.assign(brand=brands).dropna(subset=["brand"])
    if branded.empty:
        return result
    joined = gpd.sjoin(branded[["brand", "geometry"]], cells[["cell_id", "geometry"]],
                       how="inner", predicate="within")
    counts = joined.groupby(["cell_id", "brand"]).size().rename("count").reset_index()
    ordered = counts.sort_values(["count", "brand"], ascending=[False, True])
    labels = ordered.assign(label=ordered["brand"] + " — " + ordered["count"].astype(str))
    result["chain_shops"] = (cells["cell_id"].map(counts.groupby("cell_id")["count"].sum())
                             .fillna(0).astype("int64"))
    result["chain_brands"] = (cells["cell_id"].map(labels.groupby("cell_id")["label"]
                                                   .agg(", ".join)).fillna(""))
    return result


def build(cells: gpd.GeoDataFrame, layers: dict, roads: gpd.GeoDataFrame,
          points: gpd.GeoDataFrame,
          homes: gpd.GeoDataFrame | None = None) -> pd.DataFrame:
    """Таблица признаков: одна строка на клетку.

    cells и все слои должны быть в одной метрической проекции. homes — жилые
    здания с годом постройки (точки), если у города есть такой источник."""
    if cells.crs is None or cells.crs.is_geographic:
        raise ValueError("клетки должны быть в метрической проекции (UTM)")
    points = _all_poi(points, layers.get("poi_polygons"))

    buildings = layers.get("buildings")
    features = pd.DataFrame({"cell_id": cells["cell_id"].to_numpy()})

    # --- застройка ---
    if buildings is not None and not buildings.empty:
        residential = buildings[buildings["building"].isin(RESIDENTIAL)]
        public = buildings[buildings["building"].isin(PUBLIC)]
        features["built_share"] = _area_share(cells, buildings, "built_share").to_numpy()
        features["residential_share"] = _area_share(
            cells, residential, "residential_share").to_numpy()
        features["public_share"] = _area_share(cells, public, "public_share").to_numpy()
        features["buildings_count"] = _count_within(
            cells, buildings.assign(geometry=buildings.geometry.representative_point()),
            "buildings_count").to_numpy()
        features["mean_levels"] = _mean_levels(cells, buildings).to_numpy()
        features["levels_known_share"] = _levels_known(cells, buildings).to_numpy()
    else:
        for column in ("built_share", "residential_share", "public_share"):
            features[column] = 0.0
        features["buildings_count"] = 0
        features["mean_levels"] = np.nan
        features["levels_known_share"] = 0.0
    features = features.join(_epochs(cells, homes).reset_index(drop=True))

    # --- препятствия: именно они отсекают клетку (п. 8) ---
    features["water_share"] = _area_share(cells, layers.get("water"),
                                          "water_share").to_numpy()
    features["green_share"] = _area_share(cells, layers.get("green"),
                                          "green_share").to_numpy()
    landuse = layers.get("landuse")
    for value in ("industrial", "railway", "cemetery", "military"):
        subset = None if landuse is None else landuse[landuse["landuse"] == value]
        features[f"landuse_{value}_share"] = _area_share(
            cells, subset, f"landuse_{value}_share").to_numpy()

    # --- точечные объекты для текста варианта C (п. 7.3) ---
    if points is not None and not points.empty:
        for name, column, values in (
                ("schools", "amenity", ("school",)),
                ("kindergartens", "amenity", ("kindergarten",)),
                ("pharmacies", "amenity", ("pharmacy",)),
                ("shops", "shop", SHOPS),
                ("transport_stops", "public_transport", ("platform",)),
        ):
            subset = points[points[column].isin(values)] if column in points else None
            features[name] = _count_within(cells, subset, name).to_numpy()
        bus = points[points["highway"] == "bus_stop"] if "highway" in points else None
        features["transport_stops"] += _count_within(cells, bus, "bus").to_numpy()
        if {"shop", "other_tags"} <= set(points.columns):
            chains = _chain_shops(cells, points[points["shop"].isin(SHOPS)])
            features = features.join(chains.reset_index(drop=True))

    # --- плотность пешеходной сети ---
    features["street_density_m_per_km2"] = _street_density(cells, roads).to_numpy()
    return features


def _mean_levels(cells: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame) -> pd.Series:
    """Средняя этажность среди зданий С ТЕГОМ. Пропуск остаётся пропуском."""
    known = buildings[buildings["levels"].notna()]
    if known.empty:
        return pd.Series(np.nan, index=cells.index)
    points = known.assign(geometry=known.geometry.representative_point())
    joined = gpd.sjoin(points[["levels", "geometry"]], cells[["cell_id", "geometry"]],
                       how="inner", predicate="within")
    mean = joined.groupby("cell_id")["levels"].mean()
    return cells["cell_id"].map(mean)


def _levels_known(cells: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame) -> pd.Series:
    """Доля зданий клетки с известной этажностью — честность текста для C."""
    points = buildings.assign(geometry=buildings.geometry.representative_point())
    joined = gpd.sjoin(points[["levels", "geometry"]], cells[["cell_id", "geometry"]],
                       how="inner", predicate="within")
    if joined.empty:
        return pd.Series(0.0, index=cells.index)
    share = joined.groupby("cell_id")["levels"].apply(lambda s: s.notna().mean())
    return cells["cell_id"].map(share).fillna(0.0)


def _street_density(cells: gpd.GeoDataFrame, roads: gpd.GeoDataFrame) -> pd.Series:
    """Метры улиц на квадратный километр клетки."""
    if roads is None or roads.empty:
        return pd.Series(0.0, index=cells.index)
    pieces = gpd.overlay(cells[["cell_id", "geometry"]],
                         roads[["geometry"]], how="intersection",
                         keep_geom_type=False)
    if pieces.empty:
        return pd.Series(0.0, index=cells.index)
    length = pieces.groupby("cell_id").geometry.apply(lambda g: g.length.sum())
    area_km2 = cells.geometry.area / 1e6
    return (cells["cell_id"].map(length).fillna(0.0) / area_km2).rename(
        "street_density_m_per_km2")
