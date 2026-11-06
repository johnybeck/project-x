"""Тесты задачи 3.5: проверки населения и сравнение версий."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seto import settings  # noqa: E402
from seto.experiment import population_checks as PC  # noqa: E402
from seto.experiment.prepare import use_population_check  # noqa: E402


# --- выбор проверки ---

def test_check_replaces_method_and_keeps_the_main_one():
    cfg = settings.load("madrid")
    use_population_check(cfg, "raster_by_sections")
    rules = cfg["city"]["population"]
    assert rules["method"] == "raster_by_sections"
    assert rules["check_of"] == "registry_by_dwellings"


def test_check_must_be_listed_in_config():
    """Список проверок входит в протокол: другую карту подставить нельзя."""
    cfg = settings.load("madrid")
    with pytest.raises(settings.ConfigError, match="population.checks"):
        use_population_check(cfg, "raster")


# --- порядок вариантов ---

def test_order_joins_equal_variants():
    values = pd.Series({"B": 0.9013, "BN": 0.9011, "M": 1.27, "E": 0.68})
    assert PC.variant_order(values) == "M > B = BN > E"


def test_order_skips_missing_variants():
    assert PC.variant_order(pd.Series({"B": 1.0, "D": np.nan})) == "B"


# --- одна задача и перекрёстная оценка ---
# Три клетки, две площадки. Площадка 0 закрывает клетку 0, площадка 1 — клетку 2.

def _version(root: Path, name: str, population, solutions, sites=None) -> PC.Version:
    prepared, step2 = root / f"{name}_prep", root / f"{name}_step2"
    prepared.mkdir()
    step2.mkdir()
    np.savez(prepared / "times.npz",
             base=np.array([30.0, 30.0, 30.0]),
             sites=np.array([[5.0, 30.0, 30.0], [30.0, 30.0, 5.0]]) if sites is None
             else sites,
             site_index=np.array([0, 2]), allowed=np.ones((2, 2), dtype=bool))
    pd.DataFrame({"cell_id": ["a", "b", "c"], "population": population}).to_parquet(
        prepared / "cell_features.parquet")
    (prepared / "effective_config.yaml").write_text(
        yaml.safe_dump({"city": {"population": {"method": name}}}), encoding="utf-8")
    (step2 / "solutions.json").write_text(json.dumps(
        {str(k): [{"variant": v, "sites": s} for v, s in items.items()]
         for k, items in solutions.items()}), encoding="utf-8")
    return PC.Version(name, prepared, step2)


def test_versions_must_share_times(tmp_path):
    main = _version(tmp_path, "main", [10, 0, 1], {1: {"B": [0]}})
    other = _version(tmp_path, "check", [1, 0, 10], {1: {"B": [1]}},
                     sites=np.array([[5.0, 5.0, 30.0], [30.0, 30.0, 5.0]]))
    with pytest.raises(ValueError, match="только картой жителей"):
        PC.check_same_task([main, other])


def test_cross_evaluation_scores_check_solution_on_main_map(tmp_path):
    """Карта проверки селит людей в клетку c, основная — в a. B проверки выбирает
    площадку 1 и на основной карте закрывает лишь 1 жителя из 11."""
    main = _version(tmp_path, "main", [10, 0, 1], {1: {"B": [0], "M": [0]}})
    check = _version(tmp_path, "check", [1, 0, 10], {1: {"B": [1], "M": [1]}})
    PC.check_same_task([main, check])
    need = np.array([10.0, 0.0, 1.0])
    cross = PC.cross_evaluate([main, check], need, threshold=15, t_max=60)
    b = cross[cross["variant"] == "B"].iloc[0]
    assert not b["same_sites"]
    assert b["dCov_main_own"] == pytest.approx(100 * 10 / 11)
    assert b["dCov_on_main"] == pytest.approx(100 * 1 / 11)
    assert b["dCov_loss_pp"] == pytest.approx(100 * 9 / 11)
    # Z_N: у основного B 10·5 + 1·30 = 80, у B проверки 10·30 + 1·5 = 305
    assert b["Z_N_loss_per_1000"] == pytest.approx(1000 * 225 / 11)


def test_same_solution_loses_nothing(tmp_path):
    main = _version(tmp_path, "main", [10, 0, 1], {1: {"B": [0]}})
    check = _version(tmp_path, "check", [9, 0, 1], {1: {"B": [0]}})
    cross = PC.cross_evaluate([main, check], np.array([10.0, 0.0, 1.0]), 15, 60)
    assert cross["same_sites"].all()
    assert cross["dCov_loss_pp"].iloc[0] == 0
    assert cross["Z_N_loss_per_1000"].iloc[0] == 0
