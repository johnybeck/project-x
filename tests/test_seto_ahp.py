"""Тесты GIS-AHP: числа статьи и пропозала, выбор на игрушке с ручным расчётом."""
import sys
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seto import settings  # noqa: E402
from seto.experiment import ahp_stability as stability  # noqa: E402
from seto.experiment.evaluate import run_variants  # noqa: E402
from seto.routing import graph as graph_mod  # noqa: E402
from seto.routing import matrix as matrix_mod  # noqa: E402
from seto.scoring import weights as weights_mod  # noqa: E402
from seto.variants import ahp  # noqa: E402
from seto.variants import solve as solver  # noqa: E402

UTM = "EPSG:32637"
RULES = settings.load_global()["ahp"]


# --- веса: числа пропозала, п. 3.6 ---

def test_eigenvector_of_the_published_matrix():
    weights, consistency = ahp.eigen_weights(ahp.pairwise_matrix(RULES["matrix"]))
    assert weights.round(3).tolist() == [0.393, 0.211, 0.179, 0.114, 0.103]
    assert consistency == pytest.approx(0.084, abs=0.0005)


def test_matrix_is_reciprocal():
    matrix = ahp.pairwise_matrix(RULES["matrix"])
    assert matrix.shape == (5, 5)
    assert (matrix * matrix.T).round(12).tolist() == np.ones((5, 5)).tolist()


def test_published_weights_sum_to_one():
    assert sum(RULES["published_weights"]) == pytest.approx(1.0, abs=1e-6)


def test_weights_of_e():
    weights = ahp.kept_weights(RULES)
    assert list(weights) == ["network", "roads", "dense"]
    assert [round(value, 3) for value in weights.values()] == [0.567, 0.257, 0.176]


def test_submatrix_ambiguity_is_the_one_in_the_proposal():
    weights = ahp.kept_weights(RULES, "submatrix")
    assert [round(value, 3) for value in weights.values()] == [0.584, 0.281, 0.135]
    matrix = ahp.pairwise_matrix(RULES["matrix"])[np.ix_([0, 2, 3], [0, 2, 3])]
    assert ahp.eigen_weights(matrix)[1] == pytest.approx(0.117, abs=0.0005)


def test_weights_of_en_keep_published_values():
    weights = ahp.need_weights(RULES)
    assert {name: round(value, 3) for name, value in weights.items()} == {
        "network": 0.388, "roads": 0.176, "dense": 0.121, "need": 0.315}
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)


def test_unknown_weight_method_is_an_error():
    with pytest.raises(ValueError, match="способ"):
        ahp.kept_weights(RULES, "наугад")


# --- классы ---

def test_quartiles_by_hand():
    values = np.arange(1.0, 9.0)
    assert ahp.classes(values, np.arange(8), 4).tolist() == [1, 1, 2, 2, 3, 3, 4, 4]


def test_equal_best_values_go_to_the_top_class():
    """Расстояние 0 у трети клеток: все они верхний класс, а не случайная часть."""
    values = np.array([0.0, 0.0, 0.0, 0.0, -1.0, -2.0, -3.0, -4.0])
    assert ahp.classes(values, np.arange(8), 4).tolist() == [4, 4, 4, 4, 2, 2, 1, 1]


def test_edges_come_from_reference_cells_only():
    values = np.array([1.0, 2.0, 3.0, 4.0, 100.0, -100.0])
    assert ahp.classes(values, np.arange(4), 2).tolist() == [1, 1, 2, 2, 2, 1]


def test_continuous_scale_is_clipped_outside_reference():
    values = np.array([0.0, 5.0, 10.0, 20.0, -5.0])
    assert ahp.scaled(values, np.arange(3)).tolist() == [0.0, 0.5, 1.0, 1.0, 0.0]


def test_flat_criterion_gives_zero_scale():
    assert ahp.scaled(np.ones(3), np.arange(3)).tolist() == [0.0, 0.0, 0.0]


# --- выбор ---

def _all_allowed(size):
    allowed = np.ones((size, size), dtype=bool)
    np.fill_diagonal(allowed, False)
    return allowed


def test_suitability_by_hand():
    """Классы при медиане: сеть 1 1 2 2, дороги 2 2 1 1, плотность 1 1 2 2."""
    criteria = {"network": np.array([1.0, 2.0, 3.0, 4.0]),
                "roads": np.array([4.0, 3.0, 2.0, 1.0]),
                "dense": np.array([1.0, 2.0, 3.0, 4.0])}
    grades = ahp.graded(criteria, np.arange(4), 2)
    score = ahp.combine(grades, {"network": 0.5, "roads": 0.3, "dense": 0.2})
    assert score.round(6).tolist() == [1.3, 1.3, 1.7, 1.7]


def test_highest_score_wins_and_ties_go_by_people():
    score = np.array([2.0, 3.0, 3.0, 1.0])
    ties = np.array([0.0, 10.0, 20.0, 0.0])
    assert ahp.choose(score, ties, _all_allowed(4), 2) == (2, 1)


def test_tie_survives_rounding_noise():
    score = np.array([0.1 + 0.2, 0.3])
    assert ahp.choose(score, np.array([0.0, 5.0]), _all_allowed(2), 1) == (1,)


def test_common_constraint_skips_a_site_too_close():
    allowed = _all_allowed(4)
    allowed[1, 2] = allowed[2, 1] = False
    score = np.array([2.0, 3.0, 3.0, 1.0])
    assert ahp.choose(score, np.array([0.0, 10.0, 20.0, 0.0]), allowed, 2) == (2, 0)


def test_too_few_separated_sites_is_an_error():
    with pytest.raises(ValueError, match="разнесённых"):
        ahp.choose(np.ones(2), np.zeros(2), np.zeros((2, 2), dtype=bool), 2)


def test_population_within_radius():
    times = np.array([[5.0, 20.0, 15.0], [30.0, 30.0, 30.0]])
    result = ahp.population_within(times, np.array([100.0, 10.0, 1.0]), 15)
    assert result.tolist() == [101.0, 0.0]


def test_top_cells_counts_the_shared_maximum():
    assert ahp.top_cells(np.array([1.0, 0.1 + 0.2, 0.3])) == 1
    assert ahp.top_cells(np.array([0.1 + 0.2, 0.3, 0.1])) == 2


# --- критерии ---

def test_dense_cells_are_the_top_of_inhabited():
    people = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    assert ahp.dense_cells(people, 0.25).tolist() == [False, False, False, False, True]


def test_share_of_seniors_in_empty_cell_is_zero():
    data = SimpleNamespace(population=np.array([100.0, 0.0]),
                           groups=pd.DataFrame({"seniors_65": [20.0, 0.0]}))
    cfg = {"city": {"need": {"seniors_group": "seniors_65"}}}
    assert ahp.need_share(data, cfg).tolist() == [0.2, 0.0]


def test_closer_means_better_for_roads_and_dense():
    data = SimpleNamespace(base_times=np.array([1.0]), road_times=np.array([2.0]),
                           dense_times=np.array([3.0]), population=np.array([1.0]))
    layers = ahp.layers(data, {})
    assert (layers["network"][0], layers["roads"][0], layers["dense"][0]) == (1.0, -2.0, -3.0)
    assert "need" not in layers


def _road_graph():
    lines = gpd.GeoDataFrame(
        {"highway": ["residential", "primary", "motorway"]},
        geometry=[LineString([(0, 0), (750, 0)]), LineString([(750, 0), (1500, 0)]),
                  LineString([(0, 100), (1500, 100)])], crs=UTM)
    config = {"include": ["residential", "primary"], "exclude": ["motorway"]}
    return lines, graph_mod.build(lines, 75, config)


def test_road_nodes_of_a_walkable_road():
    lines, graph = _road_graph()
    nodes = matrix_mod.road_nodes(graph, lines, ["primary"], 300)
    assert sorted(graph.nodes[nodes][:, 0].tolist()) == [750.0, 1500.0]


def test_motorway_outside_the_graph_snaps_to_nearest_nodes():
    lines, graph = _road_graph()
    nodes = matrix_mod.road_nodes(graph, lines, ["motorway"], 300)
    assert sorted(graph.nodes[nodes][:, 0].tolist()) == [0.0, 1500.0]


def test_time_to_the_nearest_source_along_the_network():
    lines, graph = _road_graph()
    nodes = matrix_mod.road_nodes(graph, lines, ["primary"], 300)
    cells = matrix_mod.snap(graph, np.array([[0.0, 75.0], [750.0, 0.0], [0.0, 5000.0]]),
                            300, 75)
    times = matrix_mod.nearest_source(graph, nodes, cells, 60)
    # 10 минут по улице и минута подхода; вторая клетка на дороге; третья не привязана
    assert times.round(6).tolist() == [11.0, 0.0, 60.0]


def test_time_is_capped_by_the_limit():
    lines, graph = _road_graph()
    nodes = matrix_mod.road_nodes(graph, lines, ["primary"], 300)
    cells = matrix_mod.snap(graph, np.array([[0.0, 0.0]]), 300, 75)
    assert matrix_mod.nearest_source(graph, nodes, cells, 5).tolist() == [5.0]


# --- варианты в общем расчёте ---

def _city(**extra):
    """Четыре клетки, три площадки: площадки 0 и 1 далеко от сети, 2 — у дорог."""
    base = np.array([50.0, 40.0, 30.0, 20.0])
    return SimpleNamespace(
        population=np.array([1000.0, 800.0, 50.0, 10.0]),
        clinics=gpd.GeoDataFrame(geometry=[Point(0, 0)], crs=UTM),
        base_times=base, clinic_times=base[None, :],
        site_times=np.array([[5.0, 40.0, 40.0, 40.0], [40.0, 5.0, 40.0, 40.0],
                             [40.0, 40.0, 5.0, 5.0]]),
        site_index=np.array([0, 1, 2]), allowed_pairs=_all_allowed(3),
        road_times=np.array([9.0, 8.0, 1.0, 1.0]), dense_times=np.array([0.0, 0.0, 9.0, 9.0]),
        **{"ahp_options": _options(), **extra})


def _options():
    """Слои при других порогах: основной порог повторяет основной слой, остальные
    меняют порядок площадок по критерию."""
    return {"roads": {"primary": np.array([1.0, 9.0, 8.0, 8.0]),
                      "secondary": np.array([9.0, 8.0, 1.0, 1.0]),
                      "tertiary": np.array([2.0, 2.0, 2.0, 2.0])},
            "dense": {"0.10": np.array([9.0, 0.0, 9.0, 9.0]),
                      "0.25": np.array([0.0, 0.0, 9.0, 9.0]),
                      "0.50": np.array([0.0, 0.0, 0.0, 9.0])}}


NEED_CITY = {"role": "dev", "name": "test",
             "demography": {"groups": {"adults_18_64": [18, 64],
                                       "seniors_65": [65, None]}},
             "need": {"r": 2.0, "main": "age", "seniors_group": "seniors_65",
                      "definitions": {"age": {"adults_18_64": 1, "seniors_65": "r"}}}}


@pytest.fixture
def cfg():
    return {"k": 1, "T_minutes": 15, "T_sensitivity": [10, 20], "t_max_minutes": 60,
            "alpha": 1.0, "random_draws": 3, "seed": 0, "walk_speed_m_per_min": 75,
            "targeting": {"ede_epsilon": -1.0, "ede_epsilon_sensitivity": [-0.5, -2.0],
                          "walk_speed_seniors_m_per_min": 60},
            "ahp": RULES, "m2": {"k_values": [1, 2]}, "city": {"role": "dev", "name": "test"}}


@pytest.fixture
def need_cfg(cfg):
    return {**cfg, "city": NEED_CITY}


def _groups():
    # доля пожилых выше всего у площадки 2
    return pd.DataFrame({"adults_18_64": [900.0, 700.0, 10.0, 5.0],
                         "seniors_65": [100.0, 100.0, 40.0, 5.0]})


def _by_name(city, cfg):
    return {variant.name: variant for variant in run_variants(city, {}, cfg).variants}


def test_without_layers_the_ahp_variants_are_no_data(cfg):
    city = _city()
    city.road_times = None
    found = _by_name(city, cfg)
    assert {found[name].status for name in ("E", "EN", "EP")} == {"missing_external"}


def test_without_need_only_e_is_computed(cfg):
    city = _city()
    city.groups = None
    found = _by_name(city, cfg)
    assert found["E"].status == "ok" and found["E"].sites
    assert found["EN"].status == "missing_external"
    assert found["EP"].status == "missing_external"


def test_e_by_hand(need_cfg):
    """Классы площадок: сеть 4 3 1, дороги 1 3 4, плотность 4 4 1. Пригодность при весах
    0,567 / 0,257 / 0,176: 3,23, 3,18 и 1,77, поэтому при k = 1 выбирается площадка 0."""
    city = _city(groups=_groups())
    grades = ahp.graded(ahp.layers(city, need_cfg), city.site_index, RULES["classes"])
    assert grades["network"][:3].tolist() == [4, 3, 1]
    assert grades["roads"][:3].tolist() == [1, 3, 4]
    assert grades["dense"][:3].tolist() == [4, 4, 1]
    found = _by_name(city, need_cfg)["E"]
    assert found.sites == (0,)
    assert found.objective == pytest.approx(3.23, abs=0.005)


def test_en_follows_the_share_of_seniors():
    """Вес нуждаемости 0,315 перевешивает разницу классов остальных критериев."""
    grades = {"network": np.array([3.0, 2.0, 2.0]), "roads": np.array([2.0, 2.0, 2.0]),
              "dense": np.array([2.0, 2.0, 2.0]), "need": np.array([1.0, 1.0, 4.0])}
    assert ahp.combine(grades, ahp.kept_weights(RULES)).argmax() == 0
    assert ahp.combine(grades, ahp.need_weights(RULES)).argmax() == 2


def test_ep_is_the_same_solver_with_suitability_in_demand(need_cfg):
    city = _city(groups=_groups())
    criteria = ahp.layers(city, need_cfg)
    grades = ahp.graded(criteria, city.site_index, RULES["classes"])
    demand = weights_mod.demand(city.population,
                                ahp.combine(grades, ahp.need_weights(RULES)),
                                need_cfg["alpha"])
    expected = solver.solve_pmedian(city.site_times, city.base_times, demand,
                                    city.allowed_pairs, 1, "EP")
    assert _by_name(city, need_cfg)["EP"].sites == expected.sites
    assert ((demand >= city.population) & (demand <= 2 * city.population)).all()


def test_metrics_come_from_the_common_table(need_cfg):
    table = run_variants(_city(groups=_groups()), {}, need_cfg).table
    assert {"E", "EN", "EP"} <= set(table["variant"])


# --- устойчивость E ---

def _stability_cfg(cfg, draws=30):
    params = {**RULES["stability"], "draws": draws}
    return {**cfg, "ahp": {**RULES, "stability": params}}


def test_stability_repeats_with_the_same_seed(cfg):
    city = _city(groups=_groups())
    first = stability.study(city, _stability_cfg(cfg))[0]
    second = stability.study(city, _stability_cfg(cfg))[0]
    pd.testing.assert_frame_equal(first, second)


def test_stability_has_a_row_per_draw_and_k(cfg):
    draws, summary, ties = stability.study(_city(groups=_groups()), _stability_cfg(cfg))
    assert len(draws) == 30 * 2
    assert summary["k"].tolist() == [1, 2]
    assert ties["k"].tolist() == [1, 2]


def test_shares_are_parts_of_one_whole(cfg):
    summary = stability.study(_city(groups=_groups()), _stability_cfg(cfg, 200))[1]
    shares = summary.filter(like="share_")
    assert (shares >= 0).all(axis=None)
    assert (shares.drop(columns="share_joint").sum(axis=1) + summary["share_joint"]
            >= 1 - 1e-9).all()


def test_thresholds_are_drawn_in_every_repeat(cfg):
    """Наши пороги — такой же спорный выбор, как схема классов (PROPOSAL_CHANGES п. 10)."""
    draws = stability.study(_city(groups=_groups()), _stability_cfg(cfg, 200))[0]
    assert set(draws["dense"]) == {"0.10", "0.25", "0.50"}
    assert set(draws["roads"]) == {"primary", "secondary", "tertiary"}


def test_thresholds_get_their_own_share_of_the_spread(cfg):
    summary = stability.study(_city(groups=_groups()), _stability_cfg(cfg, 200))[1]
    assert {"share_dense", "share_roads"} <= set(summary.columns)
    # при k = 1 площадку решает порог дорог: у «primary» лучшая площадка другая
    assert summary.loc[summary["k"] == 1, "share_roads"].iloc[0] > 0


def _main_thresholds_only(cfg, draws=60):
    params = {**RULES["stability"], "draws": draws, "dense_shares": [RULES["dense_share"]],
              "main_roads": {"secondary": RULES["stability"]["main_roads"]["secondary"]}}
    return {**cfg, "ahp": {**RULES, "stability": params}}


def _no_thresholds(cfg, draws=60):
    params = {key: value for key, value in RULES["stability"].items()
              if key not in ("dense_shares", "main_roads")}
    return {**cfg, "ahp": {**RULES, "stability": {**params, "draws": draws}}}


def test_main_thresholds_alone_repeat_the_draws_without_thresholds(cfg):
    """Порог не трогает выборы пропозала: с одним основным порогом повторы те же, что
    были до включения порогов, — та же схема, веса, ничьи и то же решение."""
    city = _city(groups=_groups())
    with_main = stability.study(city, _main_thresholds_only(cfg))[0]
    city.ahp_options = None
    without = stability.study(city, _no_thresholds(cfg))[0]
    same = ["draw", "k", "scheme", "method", "ties", "sites", "dCov"]
    pd.testing.assert_frame_equal(with_main[same], without[same])
    assert set(without["dense"]) == set(without["roads"]) == {"main"}


def test_thresholds_without_layers_are_an_error(cfg):
    """Молча взять основной слой значило бы выдать узкий разброс E за полный."""
    city = _city(groups=_groups())
    city.ahp_options = None
    with pytest.raises(ValueError, match="prepare-city"):
        stability.study(city, _stability_cfg(cfg))


def test_spread_by_option_has_a_row_per_k_and_option(cfg):
    draws = stability.study(_city(groups=_groups()), _stability_cfg(cfg, 200))[0]
    table = stability.by_option(draws)
    assert len(table) == 2 * (3 + 3)
    assert table.groupby(["k", "source"])["draws"].sum().tolist() == [200] * 4


def test_option_layers_survive_saving(tmp_path):
    options = _options()
    np.savez(tmp_path / "times.npz", base=np.zeros(4),
             **{ahp.option_key(criterion, name): values
                for criterion, layers in options.items() for name, values in layers.items()})
    found = ahp.read_options(np.load(tmp_path / "times.npz"))
    assert sorted(found["dense"]) == ["0.10", "0.25", "0.50"]
    assert found["roads"]["primary"].tolist() == options["roads"]["primary"].tolist()


def test_old_preparation_has_no_option_layers(tmp_path):
    np.savez(tmp_path / "times.npz", base=np.zeros(4), to_roads=np.zeros(4))
    assert ahp.read_options(np.load(tmp_path / "times.npz")) is None


def test_share_name_keeps_two_digits():
    assert [ahp.share_name(value) for value in (0.1, 0.25, 0.5)] == ["0.10", "0.25", "0.50"]


def test_option_equal_to_the_main_threshold_is_the_main_layer():
    """Основной E и его повтор в Монте-Карло считаются по одному и тому же слою."""
    from seto.experiment.prepare import ahp_option_layers

    lines, graph = _road_graph()
    cells = matrix_mod.snap(graph, np.array([[0.0, 75.0], [750.0, 0.0], [1500.0, 0.0]]),
                            300, 75)
    population = np.array([10.0, 500.0, 20.0])
    roads_main = np.array([11.0, 0.0, 0.0])
    dense_main = np.array([7.0, 0.0, 7.0])
    cfg = {"snap_max_m": 300, "t_max_minutes": 60,
           "osm": {"main_roads": ["primary"]},
           "ahp": {"dense_share": 0.25,
                   "stability": {"dense_shares": [0.25, 1.0],
                                 "main_roads": {"same": ["primary"],
                                                "wider": ["primary", "residential"]}}}}
    found = ahp_option_layers(graph, lines, cells, population, cfg, roads_main, dense_main)
    assert found["roads"]["same"] is roads_main
    assert found["dense"]["0.25"] is dense_main
    # шире набор дорог — первая клетка стоит у «главной» улицы, остаётся минута подхода
    assert found["roads"]["wider"].round(6).tolist() == [1.0, 0.0, 0.0]
    # плотными считаются все клетки с жителями — до плотного района ноль отовсюду
    assert found["dense"]["1.00"].tolist() == [0.0, 0.0, 0.0]


def test_one_source_explains_a_split_by_itself():
    values = pd.Series([1.0, 1.0, 3.0, 3.0])
    assert stability._explained(values, pd.Series(list("aabb"))) == pytest.approx(1.0)
    assert stability._explained(values, pd.Series(list("abab"))) == pytest.approx(0.0)


def test_flat_result_has_no_shares():
    assert stability._explained(pd.Series([2.0, 2.0]), pd.Series(list("ab"))) == 0.0


def test_stability_needs_layers(cfg):
    city = _city()
    city.road_times = None
    with pytest.raises(ValueError, match="prepare-city"):
        stability.study(city, cfg)
