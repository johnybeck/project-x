"""Подготовка города: от снимка до допустимых площадок (п. 15, `prepare-city`).

Всё, что считается ОДИН раз и общее для всех вариантов: граница, сетка,
население, признаки, поликлиники, пешеходный граф, матрица времён, площадки.
Затраты на это идут тегом shared_prep и в отчёте показываются отдельно —
приписывать общую подготовку одному варианту было бы нечестно (п. 14).

Результаты складываются в папку прогона и больше не перезаписываются.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from .. import settings
from ..acquire.osm import describe_layers
from ..costs.timer import SHARED, CostLog
from ..data import boundary as boundary_mod
from ..data import catastro as catastro_mod
from ..data import clinics as clinics_mod
from ..data import demography as demography_mod
from ..data import features as features_mod
from ..data import grid as grid_mod
from ..data import osm_layers as layers_mod
from ..data import population as population_mod
from ..data import registry as registry_mod
from ..routing import graph as graph_mod
from ..routing import matrix as matrix_mod
from ..sites import feasible as sites_mod


@dataclass
class CityData:
    """Готовые данные города, на которых дальше считаются варианты."""
    grid: gpd.GeoDataFrame
    features: pd.DataFrame
    population: np.ndarray
    clinics: gpd.GeoDataFrame
    base_times: np.ndarray            # до ближайшей существующей поликлиники
    clinic_times: np.ndarray          # [поликлиники x клетки], нужна для 2SFCA
    site_times: np.ndarray            # [площадки x клетки]
    site_index: np.ndarray            # позиции площадок в сетке
    allowed_pairs: np.ndarray
    report: dict
    # жители клетки по возрастным группам (колонки из конфига) — есть только
    # там, где население берётся из регистра; иначе None
    groups: pd.DataFrame | None = None
    # район клетки — для разложения неравенства; там же, где группы
    districts: pd.Series | None = None

    def save(self, out_dir: Path) -> dict[str, Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "grid": out_dir / "grid.parquet",
            "cell_features": out_dir / "cell_features.parquet",
            "clinics": out_dir / "clinics.csv",
            "times": out_dir / "times.npz",
            "sites": out_dir / "sites.parquet",
        }
        self.grid.to_parquet(paths["grid"])
        features = self.features.assign(population=self.population)
        if self.groups is not None:
            groups = self.groups.drop(columns="total", errors="ignore")
            features = features.assign(**{f"pop_{name}": groups[name].to_numpy()
                                          for name in groups.columns})
        if self.districts is not None:
            features = features.assign(district=np.asarray(self.districts, dtype=object))
        features.to_parquet(paths["cell_features"])
        self.clinics.drop(columns="geometry").assign(
            lon=self.clinics.to_crs(4326).geometry.x,
            lat=self.clinics.to_crs(4326).geometry.y).to_csv(
                paths["clinics"], index=False, encoding="utf-8")
        np.savez_compressed(paths["times"], base=self.base_times,
                            clinics=self.clinic_times,
                            sites=self.site_times, site_index=self.site_index,
                            allowed=self.allowed_pairs)
        self.grid.iloc[self.site_index][["cell_id", "geometry"]].to_parquet(
            paths["sites"])
        return paths


def _state_clinics(medical: gpd.GeoDataFrame, cfg: dict):
    """Сеть государственных ВЗРОСЛЫХ поликлиник ПМСП (пропозал п. 3.1).

    Отбор по названию: тег operator заполнен у 6 % объектов, а operator:type
    у 0,2 % — по тегам OSM гос и частные не разделить. Правила живут в
    конфиге (`base_network`), здесь только применение. Настоящее выделение
    требует реестра (п. 4.5, Л7), и до него ограничение идёт в протокол.

    Возвращает (сеть, отчёт об отборе): отчёт показывает, что и каким
    правилом выброшено, — без него число поликлиник нечем проверить."""
    selection = clinics_mod.select(medical, settings.setting(cfg, "base_network"))
    selection.check()
    return selection.frame.reset_index(drop=True), selection


# что записать в отчёт о том, откуда взята сеть
_NETWORK_SOURCES = {
    "osm_names": "государственные взрослые ПМСП по названию — реестра нет (Л7)",
    "registry": "государственная сеть из официального реестра",
}


def _base_network(cfg: dict, raw: Path, points: gpd.GeoDataFrame,
                  polygons: gpd.GeoDataFrame, crs):
    """Базовая сеть города: из реестра или из OSM по названию.

    Источник задаёт конфиг (`base_network.source`, по умолчанию osm_names —
    так у Москвы и Люцерна). Возвращает (сеть, отбор, число кандидатов):
    у реестра кандидаты — учреждения города, у OSM — его медобъекты."""
    source = settings.setting(cfg, "base_network").get("source", "osm_names")
    if source not in _NETWORK_SOURCES:
        raise settings.ConfigError(
            f"base_network.source должен быть одним из {list(_NETWORK_SOURCES)}, "
            f"а не {source!r}")

    if source == "registry":
        rules = cfg["city"]["registry"]
        path = raw / cfg["city"]["sources"]["tables"][rules["table"]]["file"]
        if not path.is_file():
            raise FileNotFoundError(
                f"нет реестра {path}. Сначала `python -m seto acquire {cfg['city']['name']}`")
        selection = registry_mod.load(path, rules)
        return selection.frame.reset_index().to_crs(crs), selection, selection.candidates

    medical = layers_mod.medical_facilities(
        points, polygons, cfg["osm"]["clinic_tags"]).to_crs(crs)
    clinics, selection = _state_clinics(medical, cfg)
    return clinics, selection, len(medical)


def _population(cfg: dict, raw: Path, grid: gpd.GeoDataFrame):
    """Население клеток: из растра (как в Москве) или итоги регистра по участкам,
    разложенные по растру (Мадрид, MADRID_DATA п. 6). Способ — в конфиге
    города, `population.method`; по умолчанию raster."""
    method = settings.setting(cfg, "population").get("method", "raster")
    if method not in ("raster", "registry_by_raster"):
        raise settings.ConfigError(
            f"population.method должен быть raster или registry_by_raster, "
            f"а не {method!r}")
    tolerance = cfg["population"]["mass_tolerance"]
    raster = _population_raster(raw)
    if method == "raster":
        return population_mod.to_cells(raster, grid, tolerance)

    rules = cfg["city"]["demography"]
    tables = cfg["city"]["sources"]["tables"]
    paths = {key: raw / tables[table]["file"]
             for key, table in (("people", rules["table"]),
                                ("sections", rules["sections"]["table"]))}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(
                f"нет {path}. Сначала `python -m seto acquire {cfg['city']['name']}`")
    totals = demography_mod.totals(demography_mod.read(paths["people"], rules), rules)
    sections = demography_mod.boundaries(paths["sections"], rules["sections"], grid.crs)
    transfer = population_mod.registry_to_cells(raster, grid, sections, totals.table,
                                                tolerance)
    # люди без участка в регистре — тоже в отчёт, а не в никуда
    transfer.report.update({f"registry_{k}": v for k, v in totals.as_dict().items()
                            if k.startswith("unassigned")})
    transfer.districts = demography_mod.district_codes(transfer.main_section, rules)
    return transfer


def _homes(cfg: dict, raw: Path, crs) -> gpd.GeoDataFrame | None:
    """Жилые здания с годом постройки; None, если источник у города не задан:
    тогда в тексте варианта C про год будет «нет данных»."""
    rules = cfg["city"].get("building_years")
    if not rules:
        return None
    path = raw / cfg["city"]["sources"]["tables"][rules["table"]]["file"]
    if not path.is_file():
        raise FileNotFoundError(
            f"нет {path}. Сначала `python -m seto acquire {cfg['city']['name']}`")
    return catastro_mod.read(path, rules).to_crs(crs)


def prepare_city(city: str, cfg: dict, out_dir: Path, logger=None,
                 costs: CostLog | None = None) -> CityData:
    """Собрать все общие данные города."""
    costs = costs or CostLog()
    say = logger.info if logger else (lambda *a, **k: None)

    raw = settings.raw_dir(cfg, city, cfg["snapshot_date"])
    pbf = raw / "osm.pbf"
    if not pbf.is_file():
        raise FileNotFoundError(
            f"нет снимка OSM: {pbf}. Сначала `python -m seto acquire {city}`")

    with costs.stage("boundary", SHARED):
        bounds_cfg = cfg["city"]["boundary"]
        area = boundary_mod.load(pbf, bounds_cfg.get("relation_id"),
                                 bounds_cfg.get("admin_level"),
                                 epsg=cfg["city"].get("utm_epsg"))
        say("граница", **area.as_dict())

    with costs.stage("grid", SHARED) as extra:
        grid = grid_mod.build_grid(area.frame, cfg["grid"]["size_m"],
                                   cfg["grid"]["id_format"])
        extra.update(grid_mod.grid_summary(grid))
        say("сетка", cells=len(grid))

    bbox = tuple(area.frame.to_crs(4326).total_bounds)
    crs = area.frame.crs

    with costs.stage("osm_layers", SHARED) as extra:
        layers = {k: v.to_crs(crs)
                  for k, v in layers_mod.extract_polygons(pbf, bbox).items()}
        roads = layers_mod.extract_roads(pbf, bbox).to_crs(crs)
        points = layers_mod.extract_points(pbf, bbox).to_crs(crs)
        extra.update(layers_mod.completeness_report(
            layers["buildings"], roads, points))
        say("слои OSM", buildings=len(layers["buildings"]), roads=len(roads))

    with costs.stage("population", SHARED) as extra:
        transfer = _population(cfg, raw, grid)
        extra.update(transfer.as_dict())
        population = transfer.per_cell.to_numpy()
        groups = getattr(transfer, "groups", None)
        districts = getattr(transfer, "districts", None)
        say("население", total=int(population.sum()),
            loss=transfer.loss_share)

    with costs.stage("features", SHARED) as extra:
        homes = _homes(cfg, raw, crs)
        features = features_mod.build(grid, layers, roads, points, homes)
        if homes is not None:
            extra.update({"homes": len(homes),
                          "homes_with_year": int(homes["year"].notna().sum())})
        say("признаки", columns=len(features.columns))

    with costs.stage("clinics", SHARED) as extra:
        clinics, selection, candidates = _base_network(
            cfg, raw, points, layers["poi_polygons"], crs)
        extra.update({"candidates": candidates, "state_proxy": len(clinics),
                      **selection.as_dict()})
        say("поликлиники", candidates=candidates, state=len(clinics),
            uncertain=len(selection.uncertain))
        # то, что нужно человеку: у OSM — нетиповые названия (Л9), у реестра —
        # центры без координат; одного счётчика в отчёте для проверки мало
        _save_uncertain(selection.uncertain, out_dir / "clinics_uncertain.csv")

    with costs.stage("graph", SHARED) as extra:
        graph = graph_mod.build(roads, cfg["walk_speed_m_per_min"],
                                cfg["osm"]["walk_network"])
        extra.update({"nodes": graph.node_count, "edges": graph.edge_count,
                      "dropped_nodes": graph.dropped_nodes})
        say("граф", nodes=graph.node_count, edges=graph.edge_count)

    speed = cfg["walk_speed_m_per_min"]
    snap_max = cfg["snap_max_m"]
    t_max = cfg["t_max_minutes"]
    cell_xy = grid[["x", "y"]].to_numpy()

    with costs.stage("times_existing", SHARED) as extra:
        cells_snapped = matrix_mod.snap(graph, cell_xy, snap_max, speed)
        clinic_snapped = matrix_mod.snap(
            graph, np.c_[clinics.geometry.x, clinics.geometry.y], snap_max, speed)
        clinic_times = matrix_mod.travel_times(graph, clinic_snapped,
                                               cells_snapped, t_max)
        base_times = matrix_mod.nearest_time(clinic_times, t_max)
        unreachable = int((base_times >= t_max).sum())
        extra.update({"unattached_cells": cells_snapped.unattached_count,
                      "unreachable_cells": unreachable,
                      "unreachable_people": float(population[base_times >= t_max].sum())})
        say("времена до существующей сети",
            median=float(np.median(base_times)), unreachable=unreachable)

    with costs.stage("sites", SHARED) as extra:
        to_existing = matrix_mod.network_distance(
            graph, cells_snapped, clinic_snapped, speed,
            limit_m=cfg["min_dist_to_existing_m"]).min(axis=1)
        to_existing = np.where(np.isfinite(to_existing), to_existing, np.inf)
        chosen = sites_mod.select(features, settings.setting(cfg, "feasible_site"),
                                  cells_snapped.attached, to_existing,
                                  cfg["min_dist_to_existing_m"])
        extra.update(chosen.as_dict())
        say("допустимые площадки", **chosen.as_dict())

    with costs.stage("times_sites", SHARED) as extra:
        site_snapped = matrix_mod.snap(graph, cell_xy[chosen.index], snap_max, speed)
        site_times = matrix_mod.travel_times(graph, site_snapped,
                                             cells_snapped, t_max)
        between = matrix_mod.network_distance(graph, site_snapped, site_snapped,
                                              speed, cfg["min_dist_between_new_m"])
        allowed = sites_mod.pair_allowed(
            np.where(np.isfinite(between), between, np.inf),
            cfg["min_dist_between_new_m"])
        feasible_k = sites_mod.max_feasible_k(allowed, cfg["k"])
        extra.update({"matrix": list(site_times.shape),
                      "max_feasible_k": feasible_k})
        say("матрица площадок", shape=list(site_times.shape),
            max_feasible_k=feasible_k)

    report = {
        "name": city,
        "snapshot_date": cfg.get("snapshot_date"),
        "boundary": area.as_dict(),
        "grid": grid_mod.grid_summary(grid),
        "population": transfer.as_dict(),
        "osm": layers_mod.completeness_report(layers["buildings"], roads, points),
        "clinics": {"candidates": candidates, "state_proxy": len(clinics),
                    "selection": _NETWORK_SOURCES[
                        settings.setting(cfg, "base_network").get("source", "osm_names")],
                    **selection.as_dict()},
        "graph": {"nodes": graph.node_count, "edges": graph.edge_count,
                  "dropped_nodes": graph.dropped_nodes},
        "accessibility": {"unattached_cells": cells_snapped.unattached_count,
                          "unreachable_cells": unreachable},
        "sites": chosen.as_dict(),
        "max_feasible_k": feasible_k,
        "k_reduced": feasible_k < cfg["k"],
        "layers": describe_layers(pbf),
    }
    data = CityData(grid=grid, features=features, population=population,
                    clinics=clinics, base_times=base_times,
                    clinic_times=clinic_times,
                    site_times=site_times, site_index=chosen.index,
                    allowed_pairs=allowed, report=report, groups=groups,
                    districts=districts)
    data.save(out_dir)
    return data


def _save_uncertain(uncertain: pd.DataFrame, path: Path) -> None:
    """Сохранить очередь сомнительных поликлиник с координатами."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(uncertain.drop(columns="geometry", errors="ignore"))
    if isinstance(uncertain, gpd.GeoDataFrame) and len(uncertain):
        points = uncertain.to_crs(4326).geometry
        table = table.assign(lon=points.x.to_numpy(), lat=points.y.to_numpy())
    table.to_csv(path, index=False, encoding="utf-8")


def _population_raster(raw: Path) -> Path:
    """Найти растр населения в папке снимка.

    `acquire` растр НЕ скачивает: тайл GHS-POP кладётся в папку снимка
    вручную (см. docs/SETO_README.md). Берётся первый *.tif по алфавиту,
    поэтому второго растра в папке быть не должно — иначе выбор молчаливый."""
    candidates = sorted(raw.glob("*.tif"))
    if not candidates:
        raise FileNotFoundError(
            f"в {raw} нет растра населения (*.tif). acquire его не скачивает: "
            "положи тайл GHS-POP вручную, см. docs/SETO_README.md")
    if len(candidates) > 1:
        raise FileNotFoundError(
            f"в {raw} несколько растров ({', '.join(p.name for p in candidates)}): "
            "оставь один, иначе непонятно, на каком населении считать")
    return candidates[0]
