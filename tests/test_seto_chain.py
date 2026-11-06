"""Цепочка разниц, новые варианты, критерии пилота и протокол (блок 10 TODO_V2)."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seto import __main__ as cli  # noqa: E402
from seto import settings  # noqa: E402
from seto.costs import human  # noqa: E402
from seto.experiment import protocol as proto  # noqa: E402
from seto.experiment import runner  # noqa: E402
from seto.experiment.evaluate import CHAIN, run_variants  # noqa: E402
from seto.experiment.step2 import run_step2  # noqa: E402
from seto.report import city as report_mod  # noqa: E402
from seto.scoring import weights as weights_mod  # noqa: E402
from seto.variants import solve as solver  # noqa: E402

UTM = "EPSG:32637"
NEED_CITY = {"role": "dev", "name": "test",
             "demography": {"groups": {"adults_18_64": [18, 64],
                                       "seniors_65": [65, None]}},
             "need": {"r": 2.0, "main": "age", "seniors_group": "seniors_65",
                      "definitions": {"age": {"adults_18_64": 1, "seniors_65": "r"}}}}


def _city(groups=True):
    """Четыре клетки, три площадки: каждая закрывает свою часть города."""
    base = np.array([40.0, 40.0, 40.0, 40.0])
    allowed = np.ones((3, 3), dtype=bool)
    np.fill_diagonal(allowed, False)
    return SimpleNamespace(
        population=np.array([1000.0, 800.0, 50.0, 10.0]),
        clinics=gpd.GeoDataFrame(geometry=[Point(0, 0)], crs=UTM),
        base_times=base, clinic_times=base[None, :],
        site_times=np.array([[5.0, 40.0, 40.0, 40.0], [40.0, 5.0, 40.0, 40.0],
                             [40.0, 40.0, 5.0, 5.0]]),
        site_index=np.array([0, 1, 2]), allowed_pairs=allowed,
        groups=pd.DataFrame({"adults_18_64": [100.0, 0.0, 0.0, 0.0],
                             "seniors_65": [0.0, 0.0, 500.0, 500.0]}) if groups else None)


@pytest.fixture
def cfg():
    return {"k": 1, "T_minutes": 15, "T_sensitivity": [10, 20], "t_max_minutes": 60,
            "alpha": 1.0, "random_draws": 3, "seed": 0, "walk_speed_m_per_min": 75,
            "targeting": {"ede_epsilon": -1.0, "ede_epsilon_sensitivity": [-0.5, -2.0],
                          "walk_speed_seniors_m_per_min": 60},
            "pilot": {"e_cost_measure": "algorithm"},
            "m2": {"k_values": [1, 2], "noise_draws": 4, "noise_amplitude": 0.1,
                   "noise_percentile": 95},
            "grid": {"size_m": 500}, "city": NEED_CITY}


SCORES = {"text": np.array([0.0, 10.0, 0.0, 0.0]),
          "image": np.array([0.0, 0.0, 10.0, 10.0]),
          "image_10m": np.array([10.0, 0.0, 0.0, 0.0])}


def _by_name(result):
    return {variant.name: variant for variant in result.variants}


def _row(result, pair):
    return result.differences.set_index("pair").loc[pair]


# --- цепочка разниц ---

def test_chain_is_the_one_in_the_proposal():
    assert [f"{left} − {right}" for left, right, _ in CHAIN] == [
        "B − A", "M − B", "C − B", "D − C", "D − D10", "BN − B", "CN − BN", "DN − BN",
        "D − E", "D − EP"]


def test_every_pair_of_the_chain_is_in_the_table(cfg):
    pairs = run_variants(_city(), SCORES, cfg).differences["pair"].tolist()
    assert all(f"{left} − {right}" in pairs for left, right, _ in CHAIN)


def test_missing_variant_leaves_a_marked_row(cfg):
    """Без снимков разница «нет данных», а не пропавшая строка."""
    row = _row(run_variants(_city(), {}, cfg), "D − C")
    assert np.isnan(row["dCov_15_pp"])
    assert "вариант D" in row["note"]


def test_difference_names_the_right_missing_side(cfg):
    row = _row(run_variants(_city(), {"image": SCORES["image"]}, cfg), "D − C")
    assert "вариант C" in row["note"]


def test_difference_is_left_minus_right(cfg):
    result = run_variants(_city(), SCORES, cfg)
    table = result.table.set_index("variant")
    row = _row(result, "D − C")
    assert row["dCov_15_pp"] == pytest.approx(table.loc["D", "dCov_15"] - table.loc["C", "dCov_15"])
    assert row["dN_15_people"] == pytest.approx(table.loc["D", "dN_15"] - table.loc["C", "dN_15"])


def test_b_minus_a_is_given_in_people_too(cfg):
    city = _city()
    result = run_variants(city, {}, cfg)
    spread = next(v for v in result.variants if v.name == "A").distribution
    row = _row(result, "B − A")
    b = result.table.set_index("variant").loc["B"]
    assert row["dCov_15_pp"] == pytest.approx(b["dCov_15"] - np.median(spread))
    people = city.population.sum()
    assert row["dN_15_people"] == pytest.approx(b["dN_15"] - np.median(spread) * people / 100)


def test_meaning_is_stated(cfg):
    assert "изображение" in _row(run_variants(_city(), SCORES, cfg), "D − C")["meaning"]


# --- D на 10 м, CN и DN ---

def test_d10_is_d_with_its_own_scores(cfg):
    city = _city()
    found = _by_name(run_variants(city, SCORES, cfg))
    weights = weights_mod.demand(city.population, SCORES["image_10m"], cfg["alpha"])
    expected = solver.solve_pmedian(city.site_times, city.base_times, weights,
                                    city.allowed_pairs, 1, "D10")
    assert found["D10"].sites == expected.sites


def test_d10_without_second_image_set_is_no_data(cfg):
    scores = {key: SCORES[key] for key in ("text", "image")}
    found = _by_name(run_variants(_city(), scores, cfg))
    assert found["D10"].status == "missing_external"
    assert found["D"].sites


def test_cn_and_dn_multiply_need_not_population(cfg):
    city = _city()
    found = _by_name(run_variants(city, SCORES, cfg))
    need = city.groups["adults_18_64"].to_numpy() + 2 * city.groups["seniors_65"].to_numpy()
    for name, key in (("CN", "text"), ("DN", "image")):
        weights = weights_mod.demand(need, SCORES[key], cfg["alpha"])
        expected = solver.solve_pmedian(city.site_times, city.base_times, weights,
                                        city.allowed_pairs, 1, name)
        assert found[name].sites == expected.sites


def test_cn_equal_to_bn_is_a_result_not_a_duplicate(cfg):
    scores = {"text": np.zeros(4)}
    found = _by_name(run_variants(_city(), scores, cfg))
    assert found["CN"].status == "ok"
    assert "совпало с BN" in found["CN"].note


def test_without_need_cn_and_dn_are_no_data(cfg):
    city = _city(groups=False)
    found = _by_name(run_variants(city, SCORES, {**cfg, "city": {"name": "test"}}))
    assert found["CN"].status == "missing_external"
    assert found["DN"].status == "missing_external"


# --- доля захвата ---

def test_step2_gives_capture_of_scored_variants(cfg, tmp_path):
    paths = run_step2(_city(), cfg, tmp_path, scores=SCORES)
    table = pd.read_csv(paths["m2"])
    assert {"capture_C", "capture_D", "capture_D10"} <= set(table.columns)
    coverage = pd.read_csv(paths["coverage_by_k"])
    assert {"C", "D", "D10"} <= set(coverage.columns)


def test_capture_is_100_when_a_variant_matches_bn(cfg):
    """Решение, совпавшее с BN, отыгрывает весь потолок."""
    from seto.metrics import substitution as m2
    city = _city()
    need = city.groups["adults_18_64"].to_numpy() + 2 * city.groups["seniors_65"].to_numpy()
    bn = solver.solve_pmedian(city.site_times, city.base_times, need,
                              city.allowed_pairs, 1, "BN")
    table, _ = m2.study(city, {"age": need}, cfg, [1], variants={1: {"D": bn.sites}})
    assert table["capture_D"].iloc[0] == pytest.approx(100.0)


def test_step2_without_scores_needs_no_model(cfg, tmp_path):
    table = pd.read_csv(run_step2(_city(), cfg, tmp_path)["m2"])
    assert "capture_D" not in table.columns


# --- скоры из прогона ---

def test_scores_are_loaded_for_every_image_set(tmp_path):
    for key in ("text", "image", "image_10m"):
        pd.DataFrame({"cell_id": ["a", "b"], "score": [1.0, 2.0]}).to_parquet(
            tmp_path / f"scores_{key}.parquet")
    assert set(cli._load_scores(tmp_path, ["a", "b"])) == {"text", "image", "image_10m"}


def test_scores_of_other_cells_are_refused(tmp_path):
    pd.DataFrame({"cell_id": ["b", "a"], "score": [1.0, 2.0]}).to_parquet(
        tmp_path / "scores_text.parquet")
    with pytest.raises(ValueError, match="порядок"):
        cli._load_scores(tmp_path, ["a", "b"])


def test_two_image_sets_go_to_two_scores():
    cfg = {"city": {"imagery": {"main_set": "pnoa_25cm", "low_set": "s2_10m"}}}
    assert runner._image_sets(cfg) == {"image": "pnoa_25cm", "image_10m": "s2_10m"}


def test_city_without_second_set_has_one_d():
    assert runner._image_sets({"city": {"imagery": {"main_set": "s2"}}}) == {"image": "s2"}
    assert runner._image_sets({"city": {}}) == {"image": None}


# --- критерии пилота ---

def _journal(path, *rows):
    for variant, minutes, activity in rows:
        human.append(human.HumanEntry("test", variant, "onboard_city", minutes, activity), path)
    return path


def _with_d(cfg):
    return run_variants(_city(), SCORES, cfg)


def test_cost_criterion_reads_the_journal(cfg, tmp_path):
    path = _journal(tmp_path / "h.csv", ("E", 100, "настройка"), ("D", 40, "настройка"))
    criteria = report_mod.pilot_criteria(_with_d(cfg), cfg, human_log=path)
    assert criteria["costs"]["status"] == "выполнен"
    assert criteria["costs"]["E_measured_as"] == "algorithm"


def test_cost_criterion_fails_above_half(cfg, tmp_path):
    path = _journal(tmp_path / "h.csv", ("E", 100, "настройка"), ("D", 60, "настройка"))
    assert report_mod.pilot_criteria(_with_d(cfg), cfg, human_log=path)["costs"]["status"] == "не выполнен"


def test_cost_criterion_without_journal_is_no_data(cfg):
    assert "нет данных" in report_mod.pilot_criteria(_with_d(cfg), cfg)["costs"]["status"]


def test_reliability_needs_a_journal(cfg, tmp_path):
    """Нет записей — неизвестно, были ли правки; это не «выполнен»."""
    criteria = report_mod.pilot_criteria(_with_d(cfg), cfg, human_log=tmp_path / "нет.csv")
    assert "нет данных" in criteria["reliability"]["status"]


def test_reliability_allows_one_restart(cfg, tmp_path):
    path = _journal(tmp_path / "h.csv", ("D", 40, "настройка"), ("D", 5, human.RESTART))
    criteria = report_mod.pilot_criteria(_with_d(cfg), cfg, human_log=path)
    assert criteria["reliability"]["status"] == "выполнен"


def test_reliability_fails_after_second_restart(cfg, tmp_path):
    path = _journal(tmp_path / "h.csv", ("D", 5, human.RESTART), ("D", 5, human.RESTART))
    assert report_mod.pilot_criteria(_with_d(cfg), cfg, human_log=path)["reliability"]["status"] == "не выполнен"


def test_reliability_fails_after_manual_fix(cfg, tmp_path):
    path = _journal(tmp_path / "h.csv", ("D", 5, human.MANUAL_FIX))
    criteria = report_mod.pilot_criteria(_with_d(cfg), cfg, human_log=path)
    assert criteria["reliability"]["status"] == "не выполнен"
    assert criteria["reliability"]["events"]["manual_fixes"] == 1


def test_reliability_counts_only_d(cfg, tmp_path):
    path = _journal(tmp_path / "h.csv", ("D", 5, "настройка"), ("C", 5, human.MANUAL_FIX))
    assert report_mod.pilot_criteria(_with_d(cfg), cfg, human_log=path)["reliability"]["status"] == "выполнен"


def test_failed_d_fails_reliability(cfg, tmp_path):
    result = _with_d(cfg)
    next(v for v in result.variants if v.name == "D").status = "failed"
    path = _journal(tmp_path / "h.csv", ("D", 5, "настройка"))
    assert report_mod.pilot_criteria(result, cfg, human_log=path)["reliability"]["status"] == "не выполнен"


# --- протокол ---

@pytest.fixture
def clean_tree(monkeypatch):
    monkeypatch.setattr(proto, "git_state",
                        lambda: {"commit": "c1", "branch": "main", "dirty": False})


def _freeze(path, city="madrid"):
    cfg = settings.load(city)
    return proto.freeze(cfg, {city: "dev"}, {city: "h"}, path,
                        {city: proto.city_rules(cfg)},
                        {city: [{"name": "osm_pbf", "url": "u", "license": "ODbL",
                                 "downloaded_at": "2026-10-28", "sha256": "x"}]})


def test_protocol_fixes_what_the_proposal_lists(clean_tree, tmp_path):
    json_path, _ = _freeze(tmp_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    for key in ("m2", "targeting", "ahp", "score_rules", "alpha", "pilot"):
        assert key in payload["parameters"]
    rules = payload["city_rules"]["madrid"]
    assert rules["need"]["r"] == 1.93
    assert rules["snapshot_date"] == "2026-10-28"
    assert len(payload["chain"]) == len(CHAIN)
    assert "algorithm" in payload["pilot_criteria"]["costs"]
    assert payload["data_sources"]["madrid"][0]["license"] == "ODbL"


def test_protocol_page_lists_deviations_and_licenses(clean_tree, tmp_path):
    _, md_path = _freeze(tmp_path)
    text = md_path.read_text(encoding="utf-8")
    assert "0,56 м на пиксель" in text
    assert "**все города**: ответ модели, где признак вне списка" in text
    assert "**madrid**, реестр жителей (падрон): 01.01.2024 — P и N" in text
    assert "ODbL" in text
    assert "D − C" in text


def test_verify_detects_changed_date_cutoff(clean_tree, tmp_path):
    """Порог по году дома меняет P и эпохи — после заморозки его не сдвинуть."""
    json_path, _ = _freeze(tmp_path)
    cfg = settings.load("madrid")
    assert proto.city_rules(cfg)["buildings"]["residents_from_year"] == 2024
    cfg["city"]["buildings"]["epochs_from_year"] = 2024
    result = proto.verify(json_path, cfg, {"madrid": "h"})
    assert any("правила города madrid" in problem for problem in result["problems"])


def test_verify_detects_changed_need_of_a_city(clean_tree, tmp_path):
    json_path, _ = _freeze(tmp_path)
    cfg = settings.load("madrid")
    assert proto.verify(json_path, cfg, {"madrid": "h"})["ok"]
    cfg["city"]["need"]["r"] = 3.0
    result = proto.verify(json_path, cfg, {"madrid": "h"})
    assert any("правила города madrid" in problem for problem in result["problems"])


def test_data_sources_come_from_the_manifest(tmp_path):
    manifest = tmp_path / "data_manifest.json"
    manifest.write_text(json.dumps({"sources": [{
        "name": "a", "url": "u", "license": "l", "downloaded_at": "d", "sha256": "s",
        "path": "C:\\локальный\\путь"}]}), encoding="utf-8")
    result = proto.data_sources({"x": manifest, "y": tmp_path / "нет.json"})
    assert result == {"x": [{"name": "a", "url": "u", "license": "l",
                             "downloaded_at": "d", "sha256": "s"}]}


def test_city_keeps_its_own_snapshot():
    assert settings.load("madrid")["snapshot_date"] == "2026-10-28"
    assert settings.load("moscow")["snapshot_date"] == settings.load_global()["snapshot_date"]
