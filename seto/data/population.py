"""Перенос населения из растра в клетки сетки (п. 4.6).

Растр и сетка не выровнены: GHS-POP лежит в Mollweide, сетка — в UTM города.
Поэтому пиксели полигонизуются в СВОЕЙ проекции, переводятся в UTM, и площади
пересечения считаются уже там — как требует п. 4.6. Присваивать пиксель
целиком ближайшей клетке нельзя: на границах клеток это сдвигает население.

Полигонизуются только пиксели с ненулевым населением — их на порядок меньше,
чем всех, и это единственное, что делает честный overlay подъёмным.

Проверка сохранения массы обязательна: расхождение больше 0,1 % означает,
что часть людей потерялась или удвоилась, и дальше считать нельзя.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box


@dataclass
class PopulationTransfer:
    """Население по клеткам и отчёт о переносе."""
    per_cell: pd.Series          # индекс — cell_id
    source_total: float          # сколько ДОЛЖНО было перенестись (по накрытой доле)
    transferred_total: float
    pixels_used: int
    nodata_pixels: int
    partial_pixels: int = 0      # пиксели на краю сетки, накрытые не целиком
    outside_grid: float = 0.0    # их население за пределами сетки

    @property
    def loss_share(self) -> float:
        if self.source_total <= 0:
            return 0.0
        return abs(self.transferred_total - self.source_total) / self.source_total

    def as_dict(self) -> dict:
        return {
            "source_total": round(self.source_total, 1),
            "transferred_total": round(self.transferred_total, 1),
            "loss_share": self.loss_share,
            "pixels_used": self.pixels_used,
            "nodata_pixels": self.nodata_pixels,
            "partial_pixels": self.partial_pixels,
            "outside_grid": round(self.outside_grid, 1),
        }


def read_pixels(raster_path: Path,
                bounds_in_raster_crs: tuple) -> tuple[gpd.GeoDataFrame, int]:
    """Пиксели с населением > 0 внутри рамки, как полигоны в проекции растра.

    Возвращает (пиксели, число пикселей nodata)."""
    import rasterio
    from rasterio.windows import from_bounds

    with rasterio.open(raster_path) as source:
        window = from_bounds(*bounds_in_raster_crs, transform=source.transform)
        window = window.round_offsets().round_lengths()
        data = source.read(1, window=window)
        transform = source.window_transform(window)
        nodata = source.nodata
        crs = source.crs

    values = data.astype("float64")
    missing = np.zeros_like(values, dtype=bool) if nodata is None else (values == nodata)
    values[missing] = 0.0
    # отрицательные значения в растрах населения встречаются как служебные
    values[values < 0] = 0.0

    rows, cols = np.nonzero(values)
    if rows.size == 0:
        empty = gpd.GeoDataFrame({"pop": []}, geometry=[], crs=crs)
        return empty, int(missing.sum())

    size_x, size_y = transform.a, -transform.e
    x0 = transform.c + cols * size_x
    y0 = transform.f - rows * size_y
    cells = [box(x, y - size_y, x + size_x, y) for x, y in zip(x0, y0)]
    return gpd.GeoDataFrame({"pop": values[rows, cols]}, geometry=cells, crs=crs), \
        int(missing.sum())


def to_cells(raster_path: Path, grid: gpd.GeoDataFrame,
             tolerance: float = 0.001) -> PopulationTransfer:
    """Разнести население растра по клеткам пропорционально площади пересечения."""
    import rasterio

    if grid.crs is None or grid.crs.is_geographic:
        raise ValueError("сетка должна быть в метрической проекции (UTM)")

    with rasterio.open(raster_path) as source:
        raster_crs = source.crs
    # рамку сетки переводим в проекцию растра, иначе прочитаем не то окно
    frame = gpd.GeoDataFrame(geometry=[box(*grid.total_bounds)], crs=grid.crs)
    bounds = tuple(frame.to_crs(raster_crs).total_bounds)

    pixels, nodata_count = read_pixels(raster_path, bounds)
    if pixels.empty:
        raise ValueError("в рамке сетки нет ни одного пикселя с населением")

    pixels = pixels.to_crs(grid.crs)
    pixels["pixel_area"] = pixels.geometry.area
    pixels["pixel_id"] = np.arange(len(pixels))

    pieces = gpd.overlay(pixels[["pixel_id", "pop", "pixel_area", "geometry"]],
                         grid[["cell_id", "geometry"]], how="intersection")
    pieces["share"] = pieces.geometry.area / pieces["pixel_area"]
    pieces["people"] = pieces["pop"] * pieces["share"]

    per_cell = pieces.groupby("cell_id")["people"].sum()
    per_cell = per_cell.reindex(grid["cell_id"], fill_value=0.0)

    # Какая доля каждого пикселя накрыта сеткой. Пиксель на краю сетки накрыт
    # частично, и переносить его целиком неверно — поэтому эталон считается
    # по накрытой доле, а не по полному населению пикселя.
    coverage = pieces.groupby("pixel_id")["share"].sum()
    if float(coverage.max()) > 1.0 + 1e-6:
        # клетки не должны перекрываться; если перекрылись — людей удвоили
        raise ValueError(
            f"пиксель накрыт клетками более чем на 100 % "
            f"({100 * coverage.max():.2f} %): клетки сетки перекрываются")

    used = pixels[pixels["pixel_id"].isin(coverage.index)].set_index("pixel_id")
    expected = float((used["pop"] * coverage).sum())
    partial = coverage[coverage < 1.0 - 1e-6]

    transfer = PopulationTransfer(
        per_cell=per_cell,
        source_total=expected,
        transferred_total=float(per_cell.sum()),
        pixels_used=int(len(used)),
        nodata_pixels=nodata_count,
        partial_pixels=int(len(partial)),
        outside_grid=float((used.loc[partial.index, "pop"] * (1 - partial)).sum()),
    )
    if transfer.loss_share > tolerance:
        raise ValueError(
            f"население не сохранилось при переносе: ожидалось {transfer.source_total:,.0f}, "
            f"перенесено {transfer.transferred_total:,.0f} "
            f"(расхождение {100 * transfer.loss_share:.3f} % > {100 * tolerance:.1f} %)")
    return transfer


# --- итоги регистра, разложенные по растру (MADRID_DATA п. 6) ---

@dataclass
class RegistryTransfer:
    """Население и возрастные группы по клеткам из итогов по участкам.

    Поля per_cell, as_dict() и loss_share — те же, что у PopulationTransfer:
    prepare-city работает с обоими одинаково."""
    groups: pd.DataFrame         # индекс — cell_id; total и группы из конфига
    report: dict
    main_section: pd.Series | None = None    # участок, давший клетке больше людей
    districts: pd.Series | None = None       # его район; заполняет prepare-city
    by_section: pd.DataFrame | None = None   # cell_id, code, total: откуда жители клетки

    @property
    def per_cell(self) -> pd.Series:
        return self.groups["total"]

    @property
    def loss_share(self) -> float:
        expected = self.report["source_total"]
        return abs(self.report["transferred_total"] - expected) / expected if expected else 0.0

    def as_dict(self) -> dict:
        return {**self.report, "loss_share": self.loss_share}


def _section_pixels(raster_path: Path, sections: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Пиксели с населением в рамке участков, в проекции участков; вес — население."""
    import rasterio

    with rasterio.open(raster_path) as source:
        raster_crs = source.crs
    frame = gpd.GeoDataFrame(geometry=[box(*sections.total_bounds)], crs=sections.crs)
    pixels, _ = read_pixels(raster_path, tuple(frame.to_crs(raster_crs).total_bounds))
    return pixels.to_crs(sections.crs).rename(columns={"pop": "weight"})


def _cut_by_sections(sources: gpd.GeoDataFrame,
                     sections: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Разрезать пиксели или здания по участкам: вес куска — вес источника ×
    доля его площади. Участок мельче пикселя так тоже получает свою долю — при
    отнесении к участку по центру такие участки оставались бы пустыми (в
    центре Мадрида их 16)."""
    sources = sources[["weight", "geometry"]].assign(source_area=sources.geometry.area)
    pieces = gpd.overlay(sources, sections[["code", "geometry"]], how="intersection")
    pieces["weight"] = pieces["weight"] * pieces.geometry.area / pieces["source_area"]
    return pieces[pieces["weight"] > 0]


def _sections_to_cells(pieces: gpd.GeoDataFrame, grid: gpd.GeoDataFrame,
                       sections: gpd.GeoDataFrame, totals: pd.DataFrame,
                       tolerance: float, details: dict) -> RegistryTransfer:
    """Жители участка (по каждой группе) делятся между его кусками
    пропорционально весу, а куски режутся по клеткам по площади."""
    with_geometry = totals.index.isin(sections["code"])
    populated = totals[with_geometry & (totals["total"] > 0)]
    # участок с жителями, где веса нулевые: делим жителей по его площади
    no_raster = populated.index.difference(pieces["code"].unique())
    if len(no_raster):
        uniform = sections[sections["code"].isin(no_raster)].copy()
        uniform["weight"] = uniform.geometry.area
        pieces = pd.concat([pieces[["code", "weight", "geometry"]],
                            uniform[["code", "weight", "geometry"]]], ignore_index=True)

    pieces = pieces[pieces["code"].isin(populated.index)].copy()
    share = pieces["weight"] / pieces.groupby("code")["weight"].transform("sum")
    columns = list(totals.columns)
    for column in columns:
        pieces[column] = share * pieces["code"].map(populated[column])

    pieces["piece_area"] = pieces.geometry.area
    parts = gpd.overlay(pieces[["code", *columns, "piece_area", "geometry"]],
                        grid[["cell_id", "geometry"]], how="intersection")
    inside = parts.geometry.area / parts["piece_area"]
    for column in columns:
        parts[column] = parts[column] * inside
    groups = parts.groupby("cell_id")[columns].sum().reindex(
        grid["cell_id"], fill_value=0.0)

    # Клетка может лежать в нескольких участках, а разложению неравенства по
    # районам нужен один. Берём участок, давший клетке больше всего жителей;
    # при равенстве — меньший код, чтобы ответ не зависел от порядка строк.
    by_section = parts.groupby(["cell_id", "code"], as_index=False)["total"].sum()
    main_section = (by_section.sort_values(["total", "code"], ascending=[False, True])
                    .drop_duplicates("cell_id").set_index("cell_id")["code"]
                    .reindex(grid["cell_id"]))

    report = {
        **details,
        "source_total": float(populated["total"].sum()),
        "transferred_total": float(groups["total"].sum()),
        "sections_used": int(len(populated)),
        "sections_without_raster": {code: float(populated.loc[code, "total"])
                                    for code in no_raster},
        "sections_without_geometry": {code: float(totals.loc[code, "total"])
                                      for code in totals.index[~with_geometry]},
    }
    transfer = RegistryTransfer(groups=groups, report=report, main_section=main_section,
                                by_section=by_section)
    if transfer.loss_share > tolerance:
        raise ValueError(
            f"население не сохранилось при переносе: в участках "
            f"{report['source_total']:,.0f}, в клетках {report['transferred_total']:,.0f} "
            f"(расхождение {100 * transfer.loss_share:.3f} % > {100 * tolerance:.1f} %)")
    return transfer


def registry_to_cells(raster_path: Path, grid: gpd.GeoDataFrame,
                      sections: gpd.GeoDataFrame, totals: pd.DataFrame,
                      tolerance: float = 0.001) -> RegistryTransfer:
    """Жители каждого участка — по регистру, распределение внутри — по растру.

    Растр (GHS-POP) на Мадриде ошибается на уровне районов: юг занижен, север
    завышен, в аэропорту тысячи «жителей». Поэтому сколько людей в участке,
    берётся из регистра, а растр решает только, как они лежат внутри участка."""
    if grid.crs is None or grid.crs.is_geographic:
        raise ValueError("сетка должна быть в метрической проекции (UTM)")

    sections = sections.to_crs(grid.crs)
    pixels = _section_pixels(raster_path, sections)
    pieces = _cut_by_sections(pixels, sections)
    populated = totals.index[totals["total"] > 0]
    details = {
        "method": "registry_by_raster",
        "raster_total_in_sections": round(
            float(pieces.loc[pieces["code"].isin(populated), "weight"].sum()), 1),
        "raster_total_in_window": round(float(pixels["weight"].sum()), 1),
        "pixels_used": int(len(pixels)),
    }
    return _sections_to_cells(pieces, grid, sections, totals, tolerance, details)


def raster_by_sections(raster_path: Path, grid: gpd.GeoDataFrame,
                       sections: gpd.GeoDataFrame, totals: pd.DataFrame,
                       tolerance: float = 0.001) -> RegistryTransfer:
    """Проверка 1, как в пропозале (п. 3.2, 3.3): и сколько людей, и где они —
    по растру GHS-POP, а возрастные группы — по долям участка в реестре.

    Растр берётся только внутри участков города: сеть — центры Мадрида, и жители
    соседних городов к ней не относятся. Иначе версии населения различались бы не
    только источником, но и территорией. Участок, где по реестру никого нет, а
    по растру люди есть, получает доли групп всего города — это в отчёт."""
    if grid.crs is None or grid.crs.is_geographic:
        raise ValueError("сетка должна быть в метрической проекции (UTM)")

    sections = sections.to_crs(grid.crs)
    pixels = _section_pixels(raster_path, sections)
    pieces = _cut_by_sections(pixels, sections)
    raster_total = pieces.groupby("code")["weight"].sum()

    known = totals[totals["total"] > 0]
    city_shares = totals.sum() / totals["total"].sum()
    shares = known.div(known["total"], axis=0).reindex(raster_total.index)
    by_city = shares.index[shares["total"].isna()]
    raster_totals = shares.fillna(city_shares).mul(raster_total, axis=0)

    details = {
        "method": "raster_by_sections",
        "raster_total_in_window": round(float(pixels["weight"].sum()), 1),
        "registry_total": round(float(totals["total"].sum()), 1),
        "sections_with_city_shares": {code: round(float(raster_total[code]), 1)
                                      for code in by_city},
        "pixels_used": int(len(pixels)),
    }
    return _sections_to_cells(pieces, grid, sections, raster_totals, tolerance, details)


def registry_by_dwellings(buildings: gpd.GeoDataFrame, raster_path: Path,
                          grid: gpd.GeoDataFrame, sections: gpd.GeoDataFrame,
                          totals: pd.DataFrame, max_people_per_dwelling: float,
                          tolerance: float = 0.001) -> RegistryTransfer:
    """Жители каждого участка — по регистру, распределение внутри — по квартирам.

    Растр не отличает жилой дом от склада: в Мадриде он кладёт жителей в 1057
    клеток без единого жилого здания (MADRID_DATA п. 9). Кадастр знает, сколько
    квартир в каждом здании.

    Здание режется по контуру, а не ставится точкой: один комплекс на 1473
    квартиры лежит сразу в четырёх участках, и точка дала бы 820 жителей на
    1473 квартиры в одном участке и ни одной квартиры в соседнем.

    Если жителей на квартиру больше max_people_per_dwelling, кадастр не видит
    жильё участка: на Каньяда-Реаль это 3213 жителей на 29 квартир, при 603
    жилых зданиях в OSM. Такие участки и участки без квартир лежат по растру,
    как в registry_to_cells: посадить всех жителей в 29 квартир хуже, чем
    немного ошибиться с пикселями."""
    if grid.crs is None or grid.crs.is_geographic:
        raise ValueError("сетка должна быть в метрической проекции (UTM)")

    sections = sections.to_crs(grid.crs)
    homes = buildings.loc[buildings["dwellings"] > 0, ["dwellings", "geometry"]]
    homes = homes.assign(geometry=homes.geometry.make_valid()).rename(
        columns={"dwellings": "weight"})
    pieces = _cut_by_sections(homes, sections)
    dwellings = pieces.groupby("code")["weight"].sum()

    people = totals["total"]
    people = people[people.index.isin(sections["code"]) & (people > 0)]
    per_dwelling = people / dwellings.reindex(people.index, fill_value=0.0)
    unseen = people.index[per_dwelling > max_people_per_dwelling]
    if len(unseen):
        chosen = sections[sections["code"].isin(unseen)]
        pieces = pd.concat([pieces[~pieces["code"].isin(unseen)],
                            _cut_by_sections(_section_pixels(raster_path, chosen), chosen)],
                           ignore_index=True)

    details = {
        "method": "registry_by_dwellings",
        "dwellings_total": float(homes["weight"].sum()),
        "dwellings_in_sections": round(float(dwellings.sum()), 1),
        "max_people_per_dwelling": max_people_per_dwelling,
        "sections_by_raster": {code: float(people[code]) for code in unseen},
    }
    return _sections_to_cells(pieces, grid, sections, totals, tolerance, details)
