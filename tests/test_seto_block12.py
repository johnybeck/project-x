"""Тесты блока 12, вторая часть: задачи 12.9–12.13 TODO_V2.

Перестановка ответов модели рядом с захватом (12.9), оценка решений от домов (12.10),
правила вывода в протоколе (12.11), рабочие документы вне git (12.12) и пробелы кода,
которые пропозал обещает закрытыми (12.13).
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seto import __main__ as cli  # noqa: E402
from seto import settings  # noqa: E402
from seto.costs import human  # noqa: E402
from seto.experiment import homes as H  # noqa: E402
from seto.experiment import protocol as proto  # noqa: E402
from seto.experiment import sensitivity as sens  # noqa: E402
from seto.experiment.evaluate import VariantResult, run_variants  # noqa: E402
from seto.experiment.step2 import run_step2, starts_table  # noqa: E402
from seto.metrics import substitution as M2  # noqa: E402
from seto.report import city as report_city  # noqa: E402

GLOBAL = settings.load_global()
UTM = "EPSG:32637"
ROOT = Path(__file__).resolve().parents[1]


# ====================================================================== игрушечный город

def _need_city(**definitions):
    found = {"age": {"adults_18_64": 1, "seniors_65": "r"}, **definitions}
    return {"role": "dev", "name": "test",
            "demography": {"groups": {"adults_18_64": [18, 64], "seniors_65": [65, None]}},
            "need": {"r": 2.0, "main": "age", "seniors_group": "seniors_65",
                     "definitions": found}}


def _city(population=(1000.0, 800.0, 50.0, 10.0)):
    """Четыре клетки, три площадки: площадка 0 закрывает клетку 0, 1 — клетку 1,
    2 — клетки 2 и 3. Нуждающиеся живут в клетках 2 и 3."""
    base = np.full(4, 40.0)
    allowed = np.ones((3, 3), dtype=bool)
    np.fill_diagonal(allowed, False)
    return SimpleNamespace(
        population=np.array(population),
        clinics=gpd.GeoDataFrame(geometry=[Point(0, 0)], crs=UTM),
        base_times=base, clinic_times=base[None, :],
        site_times=np.array([[5.0, 40.0, 40.0, 40.0], [40.0, 5.0, 40.0, 40.0],
                             [40.0, 40.0, 5.0, 5.0]]),
        site_index=np.array([0, 1, 2]), allowed_pairs=allowed,
        groups=pd.DataFrame({"adults_18_64": [100.0, 0.0, 0.0, 0.0],
                             "seniors_65": [0.0, 0.0, 500.0, 500.0]}))


def _cfg(k=1, k_values=(1, 2), permutations=0, random_draws=3, **city):
    m2 = {"k_values": list(k_values), "noise_draws": 4, "noise_amplitude": 0.1,
          "noise_percentile": 95}
    if permutations:
        m2.update({"permutation_draws": permutations, "permutation_level": 0.05})
    return {"k": k, "T_minutes": 15, "T_sensitivity": [10, 20], "t_max_minutes": 60,
            "alpha": 1.0, "alpha_sensitivity": [0.5, 2.0], "random_draws": random_draws,
            "seed": 0,
            "walk_speed_m_per_min": 75,
            "swap_heuristic": {"starts": 4, "max_iterations": 50},
            "targeting": {"ede_epsilon": -1.0, "ede_epsilon_sensitivity": [-0.5, -2.0],
                          "walk_speed_seniors_m_per_min": 60},
            "pilot": {"e_cost_measure": "algorithm"}, "m2": m2,
            "grid": {"size_m": 500}, "city": _need_city(**city)}


# ====================================================================== 12.9 перестановка

def _m2_city():
    """B при k = 1 ставит точку в клетку 0 (100 жителей), BN — на площадку 2: там
    нуждающиеся. Скор «1 в клетках 2 и 3» переводит решение на площадку 2."""
    allowed = np.ones((3, 3), dtype=bool)
    np.fill_diagonal(allowed, False)
    return SimpleNamespace(
        population=np.array([100.0, 10.0, 45.0, 45.0]),
        base_times=np.full(4, 40.0),
        site_times=np.array([[5.0, 40, 40, 40], [40.0, 5, 40, 40], [40.0, 40, 5, 5]]),
        allowed_pairs=allowed)


NEED = np.array([10.0, 0.0, 500.0, 500.0])
GOOD_SCORE = np.array([0.0, 0.0, 1.0, 1.0])


def _m2_cfg(draws=200, noise=0.0):
    return {"t_max_minutes": 60, "seed": 0, "alpha": 1.0,
            "swap_heuristic": {"starts": 3, "max_iterations": 50},
            "m2": {"k_values": [1], "noise_draws": 5, "noise_amplitude": noise,
                   "noise_percentile": 95, "permutation_draws": draws,
                   "permutation_level": 0.05}}


def _m2_table(cfg, sites=(2,)):
    data = _m2_city()
    table, _ = M2.study(data, {"age": NEED}, cfg, [1], variants={1: {"D": sites}})
    return data, table


def test_share_is_counted_among_permutations():
    assert M2.share_not_below(5.0, np.array([0.0, 5.0, 7.0, 4.9]), 100.0) == 0.5
    assert M2.share_not_below(0.0, np.array([-1.0, -2.0]), 100.0) == 0.0


def test_equal_gains_do_not_split_on_rounding_noise():
    assert M2.share_not_below(0.1 + 0.2, np.array([0.3]), 1.0) == 1.0


def test_permutations_repeat_with_the_same_seed():
    data, cfg = _m2_city(), _m2_cfg()
    first = M2.permutation_draws(data, cfg, 1, GOOD_SCORE, 30, [0, 2, 1, 0])
    again = M2.permutation_draws(data, cfg, 1, GOOD_SCORE, 30, [0, 2, 1, 0])
    other = M2.permutation_draws(data, cfg, 1, GOOD_SCORE, 30, [0, 2, 1, 1])
    assert first == again and first != other


def test_only_inhabited_cells_are_shuffled(monkeypatch):
    """Ответ пустой клетки не должен достаться клетке с жителями: набор ответов у
    жилых клеток стал бы другим, а проверка — мягче."""
    data = _m2_city()
    data.population = np.array([100.0, 0.0, 45.0, 45.0])
    scores = np.array([0.0, 9.0, 1.0, 2.0])          # 9 — ответ пустой клетки
    seen = []
    real = M2.weights_mod.demand

    def spy(population, shuffled, alpha):
        seen.append(np.asarray(shuffled).copy())
        return real(population, shuffled, alpha)

    monkeypatch.setattr(M2.weights_mod, "demand", spy)
    M2.permutation_draws(data, _m2_cfg(), 1, scores, 25, 0)
    assert all(row[1] == 9.0 for row in seen)
    assert all(sorted(row[[0, 2, 3]]) == [0.0, 1.0, 2.0] for row in seen)
    assert len({tuple(row) for row in seen}) > 1


def test_capture_gets_the_share_of_permutations_next_to_it():
    """Перемешанный скор попадает на площадку 2 в половине раскладок: захват 100 %
    здесь не отличим от случайного."""
    cfg = _m2_cfg()
    data, table = _m2_table(cfg)
    table, found = M2.with_permutations(table, data, {"age": NEED}, cfg, {1: {"D": (2,)}},
                                        {"D": GOOD_SCORE})
    row = table.iloc[0]
    assert row["capture_D"] == pytest.approx(100)
    hits = sum(sites == [2] for sites in found[1]["D"]) / 200
    assert row["perm_share_D"] == pytest.approx(hits)
    assert 0.3 < row["perm_share_D"] < 0.7
    assert row["above_chance_D"] is False


def test_capture_is_above_chance_when_permutations_rarely_reach_it(monkeypatch):
    cfg = _m2_cfg(draws=40)
    data, table = _m2_table(cfg)
    draws = [(0,)] * 39 + [(2,)]                     # 1 из 40 — 2,5 %
    monkeypatch.setattr(M2, "permutation_draws", lambda *args: draws)
    table, _ = M2.with_permutations(table, data, {"age": NEED}, cfg, {1: {"D": (2,)}},
                                    {"D": GOOD_SCORE})
    assert table.iloc[0]["perm_share_D"] == pytest.approx(0.025)
    assert table.iloc[0]["above_chance_D"] is True


def test_variant_equal_to_b_is_never_above_chance():
    cfg = _m2_cfg()
    data, table = _m2_table(cfg, sites=(0,))
    table, _ = M2.with_permutations(table, data, {"age": NEED}, cfg, {1: {"D": (0,)}},
                                    {"D": GOOD_SCORE})
    assert table.iloc[0]["capture_D"] == 0
    assert table.iloc[0]["perm_share_D"] == 1.0
    assert table.iloc[0]["above_chance_D"] is False


def test_share_is_given_even_where_capture_is_hidden():
    """Потолок ноль — захват не показывается, но доля перестановок посчитана: правило
    чтения применяется только там, где есть захват."""
    cfg = _m2_cfg()
    data = _m2_city()
    need = data.population.copy()                    # N = P: B и BN совпадают
    table, _ = M2.study(data, {"age": need}, cfg, [1], variants={1: {"D": (2,)}})
    table, _ = M2.with_permutations(table, data, {"age": need}, cfg, {1: {"D": (2,)}},
                                    {"D": GOOD_SCORE})
    row = table.iloc[0]
    assert pd.isna(row["capture_D"]) and row["above_chance_D"] is None
    assert 0 <= row["perm_share_D"] <= 1


def test_without_scores_or_draws_the_table_is_untouched():
    cfg = _m2_cfg()
    data, table = _m2_table(cfg)
    same, found = M2.with_permutations(table, data, {"age": NEED}, cfg, {1: {}}, {})
    assert found == {} and list(same.columns) == list(table.columns)
    none, found = M2.with_permutations(table, data, {"age": NEED}, _m2_cfg(draws=0),
                                       {1: {"D": (2,)}}, {"D": GOOD_SCORE})
    assert found == {} and "perm_share_D" not in none.columns


def test_step2_saves_permutations_and_their_share(tmp_path):
    scores = {"image": np.array([0.0, 0.0, 10.0, 10.0])}
    paths = run_step2(_city(), _cfg(permutations=6), tmp_path, scores=scores)
    table = pd.read_csv(paths["m2"])
    assert "perm_share_D" in table.columns and table["perm_share_D"].between(0, 1).all()
    saved = json.loads(paths["m2_permutation_sites"].read_text(encoding="utf-8"))
    assert len(saved["sites"]["1"]["D"]) == 6
    assert saved["parameters"]["permutation_level"] == 0.05


def test_config_declares_the_permutation_test():
    assert GLOBAL["m2"]["permutation_draws"] == GLOBAL["m2"]["noise_draws"] == 200
    assert GLOBAL["m2"]["permutation_level"] == 0.05


# ====================================================================== 12.10 оценка от домов

def _homes(cell, residents, base_time):
    return H.Homes(cell=np.array(cell), residents=np.array(residents, dtype=float),
                   base_time=np.array(base_time, dtype=float))


def test_residents_of_a_cell_are_split_by_dwellings():
    residents = H.split_residents(np.array([0, 0, 2]), np.array([1.0, 3.0, 5.0]),
                                  np.array([100.0, 50.0, 7.0]))
    assert residents.tolist() == [25.0, 75.0, 7.0]


def test_split_keeps_the_population_of_every_cell():
    cell = np.array([0, 0, 1, 1, 1])
    population = np.array([120.0, 30.0])
    residents = H.split_residents(cell, np.array([2.0, 4.0, 1.0, 1.0, 1.0]), population)
    assert np.bincount(cell, weights=residents).tolist() == population.tolist()


def test_coverage_from_homes_looks_at_each_home():
    """Клетка 0 по центру в 14 минутах, но один её дом — в 17: его жители не покрыты."""
    homes = _homes([0, 0, 1], [60, 40, 50], [12.0, 17.0, 16.0])
    population = np.array([100.0, 50.0])
    assert H.coverage(homes.base_time, homes, np.array([14.0, 16.0]), population, 15) \
        == pytest.approx(100 * 60 / 150)


def test_cell_without_homes_is_counted_by_its_centre():
    homes = _homes([0], [100], [20.0])
    population = np.array([100.0, 50.0, 0.0])           # в клетке 1 домов кадастра нет
    assert H.cells_without_homes(homes.cell, homes.residents, population).tolist() \
        == [False, True, False]
    assert H.coverage(homes.base_time, homes, np.array([20.0, 10.0, 5.0]), population, 15) \
        == pytest.approx(100 * 50 / 150)


def test_base_summary_counts_people_on_the_wrong_side():
    homes = _homes([0, 0, 1, 1], [60, 40, 30, 20], [12.0, 17.0, 14.0, 18.0])
    summary = H.base_summary(homes, np.array([14.0, 16.0]), np.array([100.0, 50.0]), 15)
    assert summary["Cov_cells"] == pytest.approx(100 * 100 / 150)
    assert summary["Cov_homes"] == pytest.approx(100 * 90 / 150)
    assert summary["wrongly_covered_people"] == 40      # клетка покрыта, дом нет
    assert summary["wrongly_uncovered_people"] == 30    # клетка не покрыта, дом да
    assert summary["misplaced_share_pct"] == pytest.approx(100 * 70 / 150)


def _toy_evaluation():
    """Две клетки, две площадки. По клеткам площадка 0 «покрывает» всю клетку 0, по
    домам — только ближний дом."""
    homes = _homes([0, 0, 1], [60, 40, 50], [30.0, 30.0, 30.0])
    population = np.array([100.0, 50.0])
    base_cell = np.array([30.0, 30.0])
    site_times_cell = np.array([[14.0, 30.0], [30.0, 10.0]])
    site_times_home = {0: np.array([12.0, 17.0, 30.0]), 1: np.array([30.0, 30.0, 9.0])}
    solutions = {1: {"B": (0,), "M": (1,)}, 2: {"B": (0, 1)}}
    return H.variant_table(solutions, homes, base_cell, site_times_cell, site_times_home,
                           population, 15, 60)


def test_variant_gain_is_given_by_cells_and_by_homes():
    table = _toy_evaluation().set_index(["k", "variant"])
    assert table.at[(1, "B"), "dCov_cells"] == pytest.approx(100 * 100 / 150)
    assert table.at[(1, "B"), "dCov_homes"] == pytest.approx(100 * 60 / 150)
    assert table.at[(1, "M"), "dCov_cells"] == table.at[(1, "M"), "dCov_homes"]
    assert table.at[(2, "B"), "dCov_homes"] == pytest.approx(100 * 110 / 150)
    assert table.at[(1, "B"), "dN_homes"] == pytest.approx(60)


def test_chain_differences_are_given_both_ways():
    diff = H.differences(_toy_evaluation())
    row = diff[(diff["k"] == 1) & (diff["pair"] == "M − B")].iloc[0]
    assert row["cells_pp"] == pytest.approx(100 * (50 - 100) / 150)
    assert row["homes_pp"] == pytest.approx(100 * (50 - 60) / 150)
    assert not (diff["k"] == 2).any()                # у k = 2 второго варианта нет


def test_solutions_are_read_from_step2_and_from_run_city(tmp_path):
    step2 = {"1": [{"variant": "B", "sites": [4]}, {"variant": "A", "sites": []}],
             "2": [{"variant": "B", "sites": [4, 7]}]}
    (tmp_path / "s.json").write_text(json.dumps(step2), encoding="utf-8")
    assert H.read_solutions(tmp_path / "s.json", 2) == {1: {"B": (4,)}, 2: {"B": (4, 7)}}
    (tmp_path / "r.json").write_text(json.dumps({"variants": step2["2"]}), encoding="utf-8")
    assert H.read_solutions(tmp_path / "r.json", 2) == {2: {"B": (4, 7)}}


def test_home_goes_to_one_cell_only():
    grid = gpd.GeoDataFrame(geometry=[box(0, 0, 500, 500), box(500, 0, 1000, 500)], crs=UTM)
    points = gpd.GeoDataFrame({"dwellings": [3.0, 5.0, 1.0]},
                              geometry=[Point(100, 100), Point(500, 250), Point(5000, 5000)],
                              crs=UTM)
    located = H.locate(points, grid)
    assert located["cell"].tolist() == [0, 0]        # дом на границе — в первую клетку
    assert located["dwellings"].tolist() == [3.0, 5.0]


def test_city_without_dwellings_has_no_data(capsys):
    """В городе без данных о квартирах проверки нет, и это сказано прямо."""
    assert cli.main(["evaluate-homes", "moscow", "нет_папки", "нет_папки"]) == 1
    assert "нет данных" in capsys.readouterr().err


def test_madrid_declares_the_check():
    cfg = settings.load("madrid")
    assert cfg["city"]["buildings"]["evaluate_from_homes"] is True
    lines = proto.declared_checks(cfg, {"madrid": proto.city_rules(cfg)})
    assert any("оценка от домов" in line for line in lines)


# ====================================================================== протокол: 12.11, 12.12

@pytest.fixture
def frozen(tmp_path, monkeypatch):
    monkeypatch.setattr(proto, "git_state",
                        lambda: {"commit": "c1", "branch": "main", "dirty": False})
    manifest = tmp_path / "data_manifest.json"
    manifest.write_text('{"sources": []}', encoding="utf-8")
    monkeypatch.setattr(cli, "_data_manifests",
                        lambda cities: {name: manifest for name in cities})
    cfg = settings.load("madrid")
    json_path, md_path = proto.freeze(cfg, {"madrid": "dev"},
                                      proto.city_data_hashes({"madrid": manifest}),
                                      tmp_path / "protocol",
                                      {"madrid": proto.city_rules(cfg)})
    return SimpleNamespace(payload=json.loads(json_path.read_text(encoding="utf-8")),
                           page=md_path.read_text(encoding="utf-8"), path=json_path,
                           cfg=cfg, manifest=manifest)


def test_protocol_holds_the_inference_rules_of_the_config(frozen):
    assert frozen.payload["inference_rules"] == GLOBAL["inference_rules"]
    assert frozen.payload["parameters"]["inference_rules"] == GLOBAL["inference_rules"]


def test_five_rules_come_from_the_proposal_and_two_are_ours():
    sources = [rule["source"] for rule in GLOBAL["inference_rules"]]
    assert sources.count("proposal") == 5 and sources.count("added") == 2
    assert all(rule.get("note") for rule in GLOBAL["inference_rules"]
               if rule["source"] == "added")


def test_step2_outcome_has_its_own_rule():
    """Д21: потолок выше шума только при k = 1 — строки пропозала для этого нет."""
    rule = next(rule for rule in GLOBAL["inference_rules"] if "только при k = 1" in rule["if"])
    assert "результат о постановке" in rule["then"]
    assert rule["source"] == "added"


def test_protocol_page_marks_the_rules_we_added(frozen):
    assert "## Правила вывода" in frozen.page
    for rule in GLOBAL["inference_rules"]:
        assert rule["if"] in frozen.page and rule["then"] in frozen.page
    assert frozen.page.count("Дописано нами") == 2


def test_verify_notices_a_changed_inference_rule(frozen):
    changed = settings.load("madrid")
    changed["inference_rules"] = changed["inference_rules"][:-1]
    problems = proto.verify(frozen.path, changed, {})["problems"]
    assert any("inference_rules" in problem for problem in problems)


def test_protocol_pins_the_working_documents(frozen, reference_docs_root):
    docs = frozen.payload["reference_docs"]
    assert list(docs) == GLOBAL["protocol"]["reference_docs"]
    for name, digest in docs.items():
        assert digest == proto.sha256_of(reference_docs_root / name)
        assert f"`{name}`: `{digest}`" in frozen.page


def test_another_version_of_a_document_gives_another_sum(frozen, reference_docs_root):
    before = dict(frozen.payload["reference_docs"])
    (reference_docs_root / "docs/TODO_V2.md").write_text("другая версия", encoding="utf-8")
    after = proto.reference_docs(settings.load("madrid"))
    assert after["docs/TODO_V2.md"] != before["docs/TODO_V2.md"]
    assert after["docs/MADRID_DATA.md"] == before["docs/MADRID_DATA.md"]


def test_freeze_refuses_without_a_working_document(tmp_path, monkeypatch, reference_docs_root):
    monkeypatch.setattr(proto, "git_state",
                        lambda: {"commit": "c1", "branch": "main", "dirty": False})
    (reference_docs_root / "docs/MADRID_DATA.md").unlink()
    with pytest.raises(proto.ProtocolMismatch, match="MADRID_DATA"):
        proto.freeze(settings.load("madrid"), {"madrid": "dev"}, {}, tmp_path)


def test_documents_may_be_edited_while_the_runs_go(frozen, reference_docs_root):
    """Ради этого они и вынесены из git: список задач правится во время прогонов."""
    (reference_docs_root / "docs/TODO_V2.md").write_text("задача закрыта", encoding="utf-8")
    hashes = proto.city_data_hashes({"madrid": frozen.manifest})
    assert proto.verify(frozen.path, settings.load("madrid"), hashes)["ok"]


def test_git_ignores_the_working_documents():
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    for name in GLOBAL["protocol"]["reference_docs"]:
        assert f"/{name}" in ignored


def test_protocol_names_what_is_not_implemented(frozen):
    text = " ".join(frozen.payload["not_computed"])
    assert "частный сценарий" in text and "δ" in text
    assert "частный сценарий" in frozen.page


# ====================================================================== 12.13 сверка с протоколом

def _folder(path: Path, **overrides) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    manifest = {"city": "madrid", "git": {"commit": "c1", "dirty": False},
                "snapshot_date": "2026-10-28", **overrides}
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _commands(folder: Path, protocol: Path) -> dict[str, list[str]]:
    scores = str(folder / "scores_image.parquet")
    return {
        "ahp-stability": ["ahp-stability", "madrid", str(folder)],
        "sensitivity": ["sensitivity", "madrid", str(folder)],
        "m4": ["m4", "madrid", str(folder), scores],
        "compare-population": ["compare-population", "madrid",
                               "--version", "основная", str(folder), str(folder),
                               "--version", "проверка", str(folder), str(folder)],
        "evaluate-homes": ["evaluate-homes", "madrid", str(folder), str(folder)],
    }


COMMANDS = ("ahp-stability", "sensitivity", "m4", "compare-population", "evaluate-homes")


@pytest.mark.parametrize("command", COMMANDS)
def test_every_step_refuses_other_code(command, frozen, tmp_path, monkeypatch, capsys):
    """Число, которое идёт в выводы, не должно зависеть от того, какой командой оно
    посчитано: сверку с протоколом делают все шаги, а не только run-city и step2."""
    folder = _folder(tmp_path / "run")
    monkeypatch.setattr(proto, "git_state",
                        lambda: {"commit": "c2", "branch": "main", "dirty": False})
    argv = _commands(folder, frozen.path)[command] + ["--protocol", str(frozen.path)]
    assert cli.main(argv) == 3
    assert "коммит c2" in capsys.readouterr().err


@pytest.mark.parametrize("command", COMMANDS)
def test_every_step_refuses_a_folder_made_before_the_freeze(command, frozen, tmp_path,
                                                            capsys):
    folder = _folder(tmp_path / "run", git={"commit": "c0", "dirty": False})
    argv = _commands(folder, frozen.path)[command] + ["--protocol", str(frozen.path)]
    assert cli.main(argv) == 3
    assert "коммит c0" in capsys.readouterr().err


def test_gate_lets_a_matching_folder_through(frozen, tmp_path):
    args = argparse.Namespace(protocol=str(frozen.path))
    folder = _folder(tmp_path / "run")
    assert cli._protocol_gate(args, settings.load("madrid"), (folder,)) == (frozen.path, None)


def test_without_protocol_a_dev_city_goes_as_before(tmp_path):
    args = argparse.Namespace(protocol=None)
    assert cli._protocol_gate(args, settings.load("madrid"), (tmp_path,)) == (None, None)


def test_test_city_needs_a_protocol_at_every_step(capsys):
    cfg = settings.load("madrid")
    cfg["city"]["role"] = "test"
    assert cli._protocol_gate(argparse.Namespace(protocol=None), cfg, ()) == (None, 3)
    assert "протокол" in capsys.readouterr().err


# ====================================================================== 12.13 разброс между стартами

def test_heuristic_solution_carries_the_spread_of_its_starts():
    """Пропозал v2, п. 3.4: «разброс между стартами приводится»."""
    # три площадки и k = 3: набор один, поэтому случайному выбору A хватает одного
    result = run_variants(_city(), {}, _cfg(k=3, random_draws=1))
    by_name = {variant.name: variant for variant in result.variants}
    spread = by_name["B"].spread
    assert spread["starts"] == 4 and spread["reached_best"] >= 1
    assert by_name["B"].as_dict()["spread"] == spread
    assert by_name["M"].spread is not None and by_name["BN"].spread is not None


def test_exact_solution_has_no_spread():
    result = run_variants(_city(), {}, _cfg(k=2))
    by_name = {variant.name: variant for variant in result.variants}
    assert by_name["B"].spread is None
    assert "spread" not in by_name["B"].as_dict()


def test_starts_table_has_a_row_per_k_and_heuristic_variant():
    spread = {"starts": 20, "seed": 0, "best": 9.0, "worst": 7.0, "median": 9.0,
              "reached_best": 18, "distinct_solutions": 2}
    solutions = {"2": [{"variant": "B", "sites": [1, 2]}],
                 "5": [{"variant": "B", "sites": [1], "spread": spread},
                       {"variant": "E", "sites": [3]}]}
    table = starts_table(solutions)
    assert table[["k", "variant", "reached_best", "distinct_solutions"]].values.tolist() \
        == [[5, "B", 18, 2]]


def test_step2_writes_the_starts_table(tmp_path):
    paths = run_step2(_city(), _cfg(k_values=(1, 3), random_draws=1), tmp_path)
    table = pd.read_csv(paths["starts"])
    assert set(table["k"]) == {3} and {"B", "BN", "M"} <= set(table["variant"])
    saved = json.loads(paths["solutions"].read_text(encoding="utf-8"))
    assert "spread" in next(item for item in saved["3"] if item["variant"] == "B")
    assert "spread" not in next(item for item in saved["1"] if item["variant"] == "B")


def test_no_spread_table_rows_without_heuristics():
    assert starts_table({"1": [VariantResult("B", (0,)).as_dict()]}).empty


# ====================================================================== 12.13 М1 и М3 по определениям N

def test_need_checks_get_their_own_m1_and_m3():
    """Пропозал v2, п. 3.3: «Устойчивость: r = 1, r удвоенное, подгруппа 80+»."""
    cfg = _cfg(age_r1={"adults_18_64": 1, "seniors_65": 1})
    by_need = run_variants(_city(), {}, cfg).by_need
    assert set(by_need["need"]) == {"age_r1"}
    assert {"исходная сеть", "B", "BN", "M"} <= set(by_need["variant"])
    assert {"Cov_N_15", "dCov_N_15", "dN_N_15", "EDE_1", "gini_N", "theil_N"} \
        <= set(by_need.columns)
    assert not [column for column in by_need.columns if "_65" in column]


def test_m1_of_a_check_uses_the_need_of_that_check():
    """Решение то же, меняется только то, кого считать нуждающимся."""
    cfg = _cfg(age_r1={"adults_18_64": 1, "seniors_65": 1})
    result = run_variants(_city(), {}, cfg)
    main = result.table.set_index("variant")
    check = result.by_need.set_index("variant")
    # BN ставит точку к клеткам 2 и 3: там 1000 пожилых. При r = 2 это 2000 единиц
    # нуждаемости, при r = 1 — 1000.
    assert main.at["BN", "dN_N_15"] == pytest.approx(2000)
    assert check.at["BN", "dN_N_15"] == pytest.approx(1000)


def test_city_with_one_definition_has_no_extra_table():
    assert run_variants(_city(), {}, _cfg()).by_need.empty


def test_step2_writes_metrics_by_need(tmp_path):
    cfg = _cfg(age_r1={"adults_18_64": 1, "seniors_65": 1})
    table = pd.read_csv(run_step2(_city(), cfg, tmp_path)["metrics_by_need"])
    assert set(table["k"]) == {1, 2} and set(table["need"]) == {"age_r1"}


# ====================================================================== 12.13 α, источники, цена прироста

def test_alpha_is_checked_in_a_test_city_too():
    cfg = _cfg()
    cfg["city"]["role"] = "test"
    table = sens.over_alpha(_city(), cfg, {"image": np.array([0.0, 0.0, 10.0, 10.0])})
    assert set(table["alpha"]) == {0.5, 1.0, 2.0} and set(table["variant"]) == {"D"}


def test_report_credits_the_images_that_were_scored():
    cfg = settings.load("madrid")
    credits = report_city.data_credits(cfg, {"image": {"cells": 10},
                                             "image_10m": {"status": "missing"}})
    text = "\n".join(credits)
    assert "Obra derivada de PNOA 2023 CC-BY 4.0 scne.es" in text
    assert "Copernicus" not in text
    assert "D.G. del Catastro" in text and "OpenStreetMap" in text


def test_report_without_images_credits_only_the_data():
    credits = report_city.data_credits(settings.load("madrid"), {})
    assert not any(item.startswith("снимки") for item in credits)
    assert any("Catastro" in item for item in credits)


def _evaluation(d_people, c_people):
    table = pd.DataFrame({"variant": ["C", "D"], "dN_15": [c_people, d_people]})
    return SimpleNamespace(table=table, variants=[])


def _journal(path, **minutes):
    for variant, value in minutes.items():
        human.append(human.HumanEntry("test", variant, "onboard_city", value, "настройка"),
                     path)
    return path


def test_gain_price_is_minutes_per_resident_gained_by_the_image(tmp_path):
    """Пропозал v2, п. 3.10: «если D лучше, но дороже, — цена прироста на одного жителя»."""
    path = _journal(tmp_path / "h.csv", D=30, C=10)
    price = report_city.gain_price(_evaluation(100.0, 40.0), _cfg(), path)
    assert price["status"] == "посчитана"
    assert price["extra_minutes"] == 20 and price["delta_people"] == 60
    assert price["minutes_per_1000_people"] == pytest.approx(1000 * 20 / 60)


def test_gain_price_is_not_computed_when_d_is_not_better(tmp_path):
    path = _journal(tmp_path / "h.csv", D=30, C=10)
    price = report_city.gain_price(_evaluation(40.0, 40.0), _cfg(), path)
    assert price["status"] == "не считается"


def test_gain_price_without_journal_is_no_data(tmp_path):
    assert "нет данных" in report_city.gain_price(_evaluation(100.0, 40.0), _cfg(),
                                                  None)["status"]
    path = _journal(tmp_path / "h.csv", D=30)
    assert "нет данных" in report_city.gain_price(_evaluation(100.0, 40.0), _cfg(),
                                                  path)["status"]


def test_gain_price_without_d_is_no_data():
    evaluation = SimpleNamespace(table=pd.DataFrame({"variant": ["C"], "dN_15": [1.0]}),
                                 variants=[])
    assert "нет данных" in report_city.gain_price(evaluation, _cfg(), None)["status"]
