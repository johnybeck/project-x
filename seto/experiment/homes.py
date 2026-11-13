"""Оценка решений от жилых домов — заранее объявленная проверка
(решение 10.11, PROPOSAL_CHANGES п. 17, `evaluate-homes`).

Основной расчёт считает время от центра клетки 500 м: все жители клетки получают одно
время, и клетка целиком либо «в 15 минутах», либо нет (пропозал v2, п. 3.2 и 3.10). Здесь
те же решения оцениваются точнее: время считается от каждого жилого дома кадастра.

Что НЕ меняется: площадки, решения вариантов и число жителей клетки. Жители дома — это
жители клетки, разложенные по квартирам домов этой клетки, поэтому сумма по клетке и по
городу та же, меняется только место внутри клетки. Клетка с жителями, где домов кадастра
нет, считается по своему центру, как в основном расчёте.

Проверка возможна только там, где есть данные о квартирах по домам. М2 и блок
«адресность» от домов не считаются: возраст жителей по домам неизвестен.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from .. import settings
from ..data import boundary as boundary_mod
from ..data import catastro as catastro_mod
from ..data import osm_layers as layers_mod
from ..routing import graph as graph_mod
from ..routing import matrix as matrix_mod
from .evaluate import CHAIN

# насколько времена, пересчитанные здесь, могут отличаться от времён подготовки города
_SAME_MINUTES = 1e-6


@dataclass
class Homes:
    """Жилые дома города: в какой клетке стоит, сколько жителей, время до исходной сети."""
    cell: np.ndarray            # позиция клетки дома в сетке
    residents: np.ndarray       # жители дома
    base_time: np.ndarray       # минуты до ближайшего существующего учреждения
    snapped: matrix_mod.Snapped | None = None   # привязка к графу; в файл не пишется

    def table(self) -> pd.DataFrame:
        return pd.DataFrame({"cell": self.cell, "residents": self.residents,
                             "base_time": self.base_time})


def split_residents(cell: np.ndarray, dwellings: np.ndarray,
                    population: np.ndarray) -> np.ndarray:
    """Жители дома: жители клетки × доля квартир дома среди квартир клетки."""
    in_cell = np.bincount(cell, weights=dwellings, minlength=len(population))
    return np.asarray(population, dtype=float)[cell] * dwellings / in_cell[cell]


def cells_without_homes(cell: np.ndarray, residents: np.ndarray,
                        population: np.ndarray) -> np.ndarray:
    """Клетки с жителями, которым не досталось ни одного дома: считаются по центру."""
    housed = np.bincount(cell, weights=residents, minlength=len(population))
    return (np.asarray(population, dtype=float) > 0) & (housed == 0)


def coverage(home_times: np.ndarray, homes: Homes, cell_times: np.ndarray,
             population: np.ndarray, threshold: float) -> float:
    """Cov_T в процентах: жители домов — по времени своего дома, жители клеток без
    домов — по времени клетки. Знаменатель — все жители города, как в главной метрике."""
    population = np.asarray(population, dtype=float)
    rest = cells_without_homes(homes.cell, homes.residents, population)
    covered = (homes.residents[home_times <= threshold].sum()
               + population[rest & (cell_times <= threshold)].sum())
    return float(100 * covered / population.sum())


def base_summary(homes: Homes, base_cell: np.ndarray, population: np.ndarray,
                 threshold: float) -> dict:
    """Исходная сеть: покрытие двумя способами и сколько жителей клетка относит не на
    ту сторону порога."""
    population = np.asarray(population, dtype=float)
    total = float(population.sum())
    in_cell = base_cell[homes.cell] <= threshold
    in_home = homes.base_time <= threshold
    wrongly_covered = float(homes.residents[in_cell & ~in_home].sum())
    wrongly_uncovered = float(homes.residents[~in_cell & in_home].sum())
    rest = cells_without_homes(homes.cell, homes.residents, population)
    return {
        "threshold_min": threshold,
        "homes": int(len(homes.cell)),
        "residents_in_homes": float(homes.residents.sum()),
        "cells_without_homes": int(rest.sum()),
        "residents_without_homes": float(population[rest].sum()),
        "Cov_cells": float(100 * population[base_cell <= threshold].sum() / total),
        "Cov_homes": coverage(homes.base_time, homes, base_cell, population, threshold),
        "wrongly_covered_people": wrongly_covered,
        "wrongly_uncovered_people": wrongly_uncovered,
        "misplaced_share_pct": 100 * (wrongly_covered + wrongly_uncovered) / total,
    }


def variant_table(solutions: dict[int, dict[str, tuple]], homes: Homes,
                  base_cell: np.ndarray, site_times_cell: np.ndarray,
                  site_times_home: dict[int, np.ndarray], population: np.ndarray,
                  threshold: float, t_max: float) -> pd.DataFrame:
    """ΔCov_T каждого варианта при каждом k: по клеткам и по домам, п.п. и жители.

    site_times_home — времена от площадки до каждого дома, по позиции площадки."""
    population = np.asarray(population, dtype=float)
    total = float(population.sum())
    cov_cells = 100 * population[base_cell <= threshold].sum() / total
    cov_homes = coverage(homes.base_time, homes, base_cell, population, threshold)
    rows = []
    for k, variants in sorted(solutions.items()):
        for name, sites in variants.items():
            if not sites:
                continue
            to_cells = np.minimum(base_cell, np.minimum(
                site_times_cell[list(sites)].min(axis=0), t_max))
            to_homes = np.minimum(homes.base_time, np.min(
                [site_times_home[site] for site in sites], axis=0))
            d_cells = 100 * population[to_cells <= threshold].sum() / total - cov_cells
            d_homes = coverage(to_homes, homes, to_cells, population, threshold) - cov_homes
            rows.append({"k": k, "variant": name, "sites": list(sites),
                         "dCov_cells": float(d_cells), "dCov_homes": float(d_homes),
                         "dN_cells": float(d_cells * total / 100),
                         "dN_homes": float(d_homes * total / 100)})
    return pd.DataFrame(rows)


def differences(table: pd.DataFrame) -> pd.DataFrame:
    """Цепочка разниц при каждом k: по клеткам и по домам, п.п. Пары со случайным
    выбором A пропускаются: его тысяча наборов от домов не оценивается."""
    rows = []
    for k, part in table.groupby("k"):
        by_name = part.set_index("variant")
        for left, right, meaning in CHAIN:
            if left not in by_name.index or right not in by_name.index:
                continue
            rows.append({
                "k": k, "pair": f"{left} − {right}", "meaning": meaning,
                "cells_pp": float(by_name.at[left, "dCov_cells"] - by_name.at[right, "dCov_cells"]),
                "homes_pp": float(by_name.at[left, "dCov_homes"] - by_name.at[right, "dCov_homes"])})
    return pd.DataFrame(rows)


def read_solutions(path: Path, k_default: int) -> dict[int, dict[str, tuple]]:
    """Решения из `solutions.json`: шаг 2 хранит их по k, `run-city` — одним списком."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    by_k = {k_default: raw["variants"]} if "variants" in raw else raw
    return {int(k): {item["variant"]: tuple(item["sites"]) for item in items
                     if item.get("sites")}
            for k, items in by_k.items()}


def home_points(cfg: dict, raw: Path, crs) -> gpd.GeoDataFrame:
    """Жилые дома точками и число квартир — те же дома, по которым разложены жители:
    с той же отсечкой по году постройки (`buildings.residents_from_year`)."""
    rules = (cfg.get("city") or {}).get("buildings")
    if not rules:
        raise ValueError("нет данных: у города нет зданий с квартирами (раздел buildings)")
    path = raw / cfg["city"]["sources"]["tables"][rules["table"]]["file"]
    if not path.is_file():
        raise FileNotFoundError(
            f"нет {path}. Сначала `python -m seto acquire {cfg['city']['name']}`")
    buildings = catastro_mod.built_before(catastro_mod.read(path, rules).to_crs(crs),
                                          rules.get("residents_from_year"))
    buildings = buildings[buildings["dwellings"] > 0]
    return gpd.GeoDataFrame({"dwellings": buildings["dwellings"].to_numpy(dtype=float)},
                            geometry=buildings.geometry.representative_point().to_numpy(),
                            crs=crs)


def locate(points: gpd.GeoDataFrame, grid: gpd.GeoDataFrame) -> pd.DataFrame:
    """В какой клетке сетки стоит дом; дома вне сетки отбрасываются. Точка на общей
    границе клеток идёт в первую из них: дом не должен посчитаться дважды."""
    cells = grid[["geometry"]].reset_index(drop=True)
    joined = gpd.sjoin(points, cells, predicate="intersects", how="inner")
    joined = joined[~joined.index.duplicated()]
    return pd.DataFrame({"cell": joined["index_right"].to_numpy(dtype=int),
                         "dwellings": joined["dwellings"].to_numpy(dtype=float),
                         "x": joined.geometry.x.to_numpy(),
                         "y": joined.geometry.y.to_numpy()})


def walk_graph(city: str, cfg: dict, raw: Path):
    """Пешеходный граф — тем же способом, что в подготовке города. Граф в папку прогона
    не пишется, поэтому строится заново; совпадение с подготовкой проверяет `run`."""
    pbf = raw / "osm.pbf"
    if not pbf.is_file():
        raise FileNotFoundError(f"нет снимка OSM: {pbf}. Сначала `python -m seto acquire {city}`")
    bounds = cfg["city"]["boundary"]
    area = boundary_mod.load(pbf, bounds.get("relation_id"), bounds.get("admin_level"),
                             epsg=cfg["city"].get("utm_epsg"))
    crs = area.frame.crs
    roads = layers_mod.extract_roads(
        pbf, tuple(area.frame.to_crs(4326).total_bounds)).to_crs(crs)
    return graph_mod.build(roads, cfg["walk_speed_m_per_min"], cfg["osm"]["walk_network"]), crs


def run(city: str, cfg: dict, data, solutions: dict[int, dict[str, tuple]],
        out_dir: Path, logger=None) -> dict[str, Path]:
    """Посчитать проверку и сохранить таблицы. data — подготовка города (CityData)."""
    say = logger.info if logger else (lambda *a, **kw: None)
    threshold, t_max = cfg["T_minutes"], cfg["t_max_minutes"]
    speed, snap_max = cfg["walk_speed_m_per_min"], cfg["snap_max_m"]
    raw = settings.raw_dir(cfg, city, cfg["snapshot_date"])
    population = np.asarray(data.population, dtype=float)

    graph, crs = walk_graph(city, cfg, raw)
    cell_xy = data.grid[["x", "y"]].to_numpy()
    cells = matrix_mod.snap(graph, cell_xy, snap_max, speed)
    clinics = matrix_mod.snap(graph, np.c_[data.clinics.geometry.x, data.clinics.geometry.y],
                              snap_max, speed)
    base_cell = matrix_mod.nearest_time(
        matrix_mod.travel_times(graph, clinics, cells, t_max), t_max)
    gap = float(np.abs(base_cell - data.base_times).max())
    if gap > _SAME_MINUTES:
        raise ValueError(f"граф не совпал с подготовкой города: времена до сети расходятся "
                         f"на {gap:.4f} мин — другая выгрузка OSM или другой код")
    say("граф построен заново и совпал с подготовкой", nodes=graph.node_count)

    located = locate(home_points(cfg, raw, crs), data.grid)
    residents = split_residents(located["cell"].to_numpy(), located["dwellings"].to_numpy(),
                                population)
    keep = residents > 0                      # дома в клетках без жителей не нужны
    located, residents = located[keep], residents[keep]
    snapped = matrix_mod.snap(graph, located[["x", "y"]].to_numpy(), snap_max, speed)
    base_home = matrix_mod.nearest_time(
        matrix_mod.travel_times(graph, clinics, snapped, t_max), t_max)
    homes = Homes(cell=located["cell"].to_numpy(), residents=residents,
                  base_time=base_home, snapped=snapped)
    summary = base_summary(homes, base_cell, population, threshold)
    summary["unattached_homes"] = snapped.unattached_count
    say("дома", homes=summary["homes"], cov_cells=round(summary["Cov_cells"], 2),
        cov_homes=round(summary["Cov_homes"], 2))

    used = sorted({site for variants in solutions.values()
                   for sites in variants.values() for site in sites})
    site_snapped = matrix_mod.snap(graph, cell_xy[data.site_index[used]], snap_max, speed)
    to_cells = matrix_mod.travel_times(graph, site_snapped, cells, t_max)
    gap = float(np.abs(to_cells - data.site_times[used]).max()) if used else 0.0
    if gap > _SAME_MINUTES:
        raise ValueError(f"времена от площадок расходятся с подготовкой на {gap:.4f} мин")
    to_homes = matrix_mod.travel_times(graph, site_snapped, snapped, t_max)
    site_times_home = {site: to_homes[position] for position, site in enumerate(used)}

    table = variant_table(solutions, homes, base_cell, data.site_times, site_times_home,
                          population, threshold, t_max)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"homes_base": out_dir / "homes_base.json",
             "homes_variants": out_dir / "homes_variants.csv",
             "homes_differences": out_dir / "homes_differences.csv",
             "homes": out_dir / "homes.parquet"}
    paths["homes_base"].write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    table.to_csv(paths["homes_variants"], index=False, encoding="utf-8")
    differences(table).to_csv(paths["homes_differences"], index=False, encoding="utf-8")
    homes.table().to_parquet(paths["homes"])
    return paths
