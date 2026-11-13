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
from ..variants import ahp as ahp_mod


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
    # откуда жители клетки: cell_id, code, total; нужно, чтобы перенести на клетки
    # доход участков (М4). В признаки не входит: модель эталона не видит
    cell_sections: pd.DataFrame | None = None
    # критерии GIS-AHP: минуты пешком до главных дорог и до плотных районов
    road_times: np.ndarray | None = None
    dense_times: np.ndarray | None = None
    # те же критерии при других порогах — для проверки устойчивости E:
    # {"roads": {имя набора дорог: минуты}, "dense": {доля клеток: минуты}}
    ahp_options: dict[str, dict[str, np.ndarray]] | None = None

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
        layers = {} if self.road_times is None else {"to_roads": self.road_times,
                                                     "to_dense": self.dense_times}
        for criterion, options in (self.ahp_options or {}).items():
            layers.update({ahp_mod.option_key(criterion, name): values
                           for name, values in options.items()})
        np.savez_compressed(paths["times"], base=self.base_times,
                            clinics=self.clinic_times,
                            sites=self.site_times, site_index=self.site_index,
                            allowed=self.allowed_pairs, **layers)
        self.grid.iloc[self.site_index][["cell_id", "geometry"]].to_parquet(
            paths["sites"])
        if self.cell_sections is not None:
            paths["cell_sections"] = out_dir / "cell_sections.parquet"
            self.cell_sections.to_parquet(paths["cell_sections"])
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


_POPULATION_METHODS = ("raster", "raster_by_sections", "registry_by_raster",
                       "registry_by_dwellings")


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


def _population(cfg: dict, raw: Path, grid: gpd.GeoDataFrame,
                buildings: gpd.GeoDataFrame | None):
    """Население клеток: из растра (как в Москве) или итоги регистра по участкам,
    разложенные по растру либо по квартирам кадастра (Мадрид, MADRID_DATA п. 6
    и 9). Способ — в конфиге города, `population.method`; по умолчанию raster."""
    rules_population = settings.setting(cfg, "population")
    method = rules_population.get("method", "raster")
    if method not in _POPULATION_METHODS:
        raise settings.ConfigError(
            f"population.method должен быть одним из {list(_POPULATION_METHODS)}, "
            f"а не {method!r}")
    if method == "registry_by_dwellings" and buildings is None:
        raise settings.ConfigError(
            "population.method = registry_by_dwellings требует раздел buildings")
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
    if method == "registry_by_raster":
        transfer = population_mod.registry_to_cells(raster, grid, sections, totals.table,
                                                    tolerance)
    elif method == "raster_by_sections":
        transfer = population_mod.raster_by_sections(raster, grid, sections, totals.table,
                                                     tolerance)
    else:
        transfer = population_mod.registry_by_dwellings(
            buildings, raster, grid, sections, totals.table,
            rules_population["max_people_per_dwelling"], tolerance)
    # люди без участка в регистре — тоже в отчёт, а не в никуда
    transfer.report.update({f"registry_{k}": v for k, v in totals.as_dict().items()
                            if k.startswith("unassigned")})
    transfer.districts = demography_mod.district_codes(transfer.main_section, rules)
    return transfer


def _buildings(cfg: dict, raw: Path, crs) -> gpd.GeoDataFrame | None:
    """Здания с годом постройки и квартирами; None, если источник у города не
    задан: тогда в тексте варианта C про год будет «нет данных»."""
    rules = cfg["city"].get("buildings")
    if not rules:
        return None
    path = raw / cfg["city"]["sources"]["tables"][rules["table"]]["file"]
    if not path.is_file():
        raise FileNotFoundError(
            f"нет {path}. Сначала `python -m seto acquire {cfg['city']['name']}`")
    return catastro_mod.read(path, rules).to_crs(crs)


def use_population_check(cfg: dict, name: str) -> None:
    """Заменить основную раскладку жителей проверкой из `population.checks` (задача 3.5).

    Только из списка в конфиге: он входит в протокол, и карту населения нельзя
    выбрать после результатов. Основной способ остаётся в `check_of` — по папке
    прогона видно, что это проверка, а не основная версия."""
    rules = cfg["city"].get("population") or {}
    checks = rules.get("checks") or []
    if name not in checks:
        raise settings.ConfigError(
            f"проверки населения {name!r} нет в population.checks города: {checks}")
    cfg["city"]["population"] = {**rules, "method": name, "check_of": rules["method"]}


def use_site_check(cfg: dict, name: str) -> None:
    """Заменить правило допустимой площадки проверкой из `feasible_site.checks`
    (решение 10.11, PROPOSAL_CHANGES п. 15).

    Только из списка в конфиге города: он входит в протокол, и правило нельзя
    выбрать после результатов. Имя проверки остаётся в `check` — по папке прогона
    видно, что это проверка, а не основная версия."""
    rules = settings.setting(cfg, "feasible_site") or {}
    checks = rules.get("checks") or {}
    if name not in checks:
        raise settings.ConfigError(
            f"проверки правила площадки {name!r} нет в feasible_site.checks города: "
            f"{sorted(checks)}")
    cfg["city"]["feasible_site"] = {**rules, **checks[name], "check": name}


def _cutoff(cfg: dict, key: str) -> int | None:
    """Год, с которого дома кадастра не сравниваются с более старым источником
    (`buildings.<key>` в конфиге города, решение 05.11). Нет ключа — без отсечки."""
    return (cfg["city"].get("buildings") or {}).get(key)


def _dropped(buildings: gpd.GeoDataFrame, year: int | None, name: str) -> dict:
    """Сколько домов и квартир убрала отсечка по году — в отчёт подготовки."""
    if year is None:
        return {}
    new = buildings["year"] >= year
    return {f"{name}_from_year": year,
            f"{name}_dropped_buildings": int(new.sum()),
            f"{name}_dropped_residential": int((new & buildings["residential"]).sum()),
            f"{name}_dropped_dwellings": int(buildings.loc[new, "dwellings"].sum())}


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

    with costs.stage("buildings", SHARED):
        buildings = _buildings(cfg, raw, crs)

    with costs.stage("population", SHARED) as extra:
        residents_year = _cutoff(cfg, "residents_from_year")
        transfer = _population(cfg, raw, grid,
                               catastro_mod.built_before(buildings, residents_year)
                               if buildings is not None else None)
        extra.update(transfer.as_dict())
        # отсечка касается только раскладки по квартирам: растру кадастр не нужен
        if buildings is not None and extra.get("method") == "registry_by_dwellings":
            extra.update(_dropped(buildings, residents_year, "residents"))
        population = transfer.per_cell.to_numpy()
        groups = getattr(transfer, "groups", None)
        districts = getattr(transfer, "districts", None)
        cell_sections = getattr(transfer, "by_section", None)
        say("население", total=int(population.sum()),
            loss=transfer.loss_share)

    with costs.stage("features", SHARED) as extra:
        epochs_year = _cutoff(cfg, "epochs_from_year")
        homes = None if buildings is None else catastro_mod.homes(
            catastro_mod.built_before(buildings, epochs_year))
        features = features_mod.build(grid, layers, roads, points, homes)
        if homes is not None:
            extra.update({"homes": len(homes),
                          "homes_with_year": int(homes["year"].notna().sum()),
                          **_dropped(buildings, epochs_year, "epochs")})
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

    with costs.stage("ahp_layers", SHARED) as extra:
        # те же сеть и привязка, что у всех времён: пропозал требует расстояний по сети
        main_roads = matrix_mod.road_nodes(graph, roads, cfg["osm"]["main_roads"], snap_max)
        road_times = matrix_mod.nearest_source(graph, main_roads, cells_snapped, t_max)
        dense = ahp_mod.dense_cells(population, cfg["ahp"]["dense_share"])
        dense_nodes = cells_snapped.node[dense & cells_snapped.attached]
        dense_times = matrix_mod.nearest_source(graph, np.unique(dense_nodes),
                                                cells_snapped, t_max)
        dense_times[dense] = 0.0        # плотная клетка сама и есть плотный район
        extra.update({"road_nodes": len(main_roads), "dense_cells": int(dense.sum())})
        say("слои AHP", road_nodes=len(main_roads), dense_cells=int(dense.sum()))

        ahp_options = ahp_option_layers(graph, roads, cells_snapped, population, cfg,
                                        road_times, dense_times)
        extra.update({"road_options": list(ahp_options["roads"]),
                      "dense_options": list(ahp_options["dense"])})

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
                    districts=districts, cell_sections=cell_sections,
                    road_times=road_times, dense_times=dense_times,
                    ahp_options=ahp_options)
    data.save(out_dir)
    return data


def ahp_option_layers(graph, roads: gpd.GeoDataFrame, cells_snapped,
                      population: np.ndarray, cfg: dict, road_times: np.ndarray,
                      dense_times: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    """Критерии GIS-AHP при других порогах — для Монте-Карло E (PROPOSAL_CHANGES п. 10).

    Считаются так же, как основные слои: та же сеть, та же привязка клеток. Порог,
    совпавший с основным, берёт основной слой, а не считается заново: основной E и
    его повтор в Монте-Карло не могут разойтись."""
    stability = cfg["ahp"].get("stability") or {}
    snap_max, t_max = cfg["snap_max_m"], cfg["t_max_minutes"]
    options: dict[str, dict[str, np.ndarray]] = {"roads": {}, "dense": {}}
    for name, classes in (stability.get("main_roads") or {}).items():
        if sorted(classes) == sorted(cfg["osm"]["main_roads"]):
            options["roads"][name] = road_times
            continue
        nodes = matrix_mod.road_nodes(graph, roads, classes, snap_max)
        options["roads"][name] = matrix_mod.nearest_source(graph, nodes, cells_snapped,
                                                           t_max)
    for share in stability.get("dense_shares") or []:
        if share == cfg["ahp"]["dense_share"]:
            options["dense"][ahp_mod.share_name(share)] = dense_times
            continue
        dense = ahp_mod.dense_cells(population, share)
        nodes = cells_snapped.node[dense & cells_snapped.attached]
        times = matrix_mod.nearest_source(graph, np.unique(nodes), cells_snapped, t_max)
        times[dense] = 0.0              # плотная клетка сама и есть плотный район
        options["dense"][ahp_mod.share_name(share)] = times
    return options


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
