"""Тесты блока «адресность»: М3, дефицит, Джини, Тейль (задачи 5.2, 5.3, 5.5)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seto.metrics import equity as E  # noqa: E402


# --- М3: равноценное время ---

def test_ede_of_equal_times_is_that_time():
    """Главное свойство: если всем одинаково, штрафа за неравенство нет."""
    times, need = np.full(4, 12.0), np.array([1.0, 2.0, 3.0, 4.0])
    kappa = E.kolm_pollak_kappa(times, need, -1.0)
    assert E.equally_distributed(times, need, kappa) == pytest.approx(12.0)


def test_ede_of_unequal_times_exceeds_the_mean():
    """Пример пропозала: у половины 5 минут, у половины 25 — заметно больше 15."""
    times, need = np.array([5.0, 25.0]), np.array([1.0, 1.0])
    kappa = E.kolm_pollak_kappa(times, need, -1.0)
    assert kappa == pytest.approx(-1.0 * 15 / 325)
    assert E.equally_distributed(times, need, kappa) > 15.0


def test_stronger_penalty_gives_larger_ede():
    times, need = np.array([5.0, 25.0, 40.0]), np.array([3.0, 2.0, 1.0])
    values = [E.equally_distributed(times, need, E.kolm_pollak_kappa(times, need, e))
              for e in (-0.5, -1.0, -2.0)]
    assert values[0] < values[1] < values[2]


def test_ede_sees_improvement_beyond_the_threshold():
    """40 минут стали 20: покрытие при T = 15 не изменилось, EDE упал."""
    need = np.array([1.0, 1.0])
    before, after = np.array([5.0, 40.0]), np.array([5.0, 20.0])
    kappa = E.kolm_pollak_kappa(before, need, -1.0)
    assert E.equally_distributed(after, need, kappa) \
        < E.equally_distributed(before, need, kappa)


def test_ede_rises_when_gap_grows_at_same_mean():
    """В отличие от Джини, EDE чувствует и уровень, и разброс сразу."""
    need = np.ones(2)
    kappa = E.kolm_pollak_kappa(np.array([10.0, 20.0]), need, -1.0)
    assert E.equally_distributed(np.array([5.0, 25.0]), need, kappa) \
        > E.equally_distributed(np.array([10.0, 20.0]), need, kappa)


def test_kappa_must_be_negative_for_a_cost():
    with pytest.raises(ValueError, match="отрицательным"):
        E.kolm_pollak_kappa(np.array([1.0, 2.0]), np.ones(2), 1.0)


# --- дефицит ---

def test_deficit_sums_minutes_over_threshold():
    times = np.array([10.0, 15.0, 40.0])
    need = np.array([100.0, 100.0, 2.0])
    assert E.deficit(times, need, 15) == pytest.approx(2 * 25)


def test_deficit_is_zero_when_everyone_is_inside():
    assert E.deficit(np.array([1.0, 14.0]), np.ones(2), 15) == 0


# --- Джини и Тейль ---

def test_gini_is_zero_for_equal_times():
    assert E.gini(np.full(5, 7.0), np.arange(1.0, 6.0)) == pytest.approx(0)


def test_gini_of_half_with_nothing_is_one_half():
    assert E.gini(np.array([0.0, 1.0]), np.ones(2)) == pytest.approx(0.5)


def test_gini_respects_weights():
    """Вес 3 у клетки — то же, что три одинаковых клетки."""
    weighted = E.gini(np.array([1.0, 5.0]), np.array([3.0, 1.0]))
    expanded = E.gini(np.array([1.0, 1.0, 1.0, 5.0]), np.ones(4))
    assert weighted == pytest.approx(expanded)


def test_theil_is_zero_for_equal_times():
    assert E.theil(np.full(3, 9.0), np.array([1.0, 2.0, 3.0])) == pytest.approx(0)


def test_theil_matches_hand_calculation():
    # среднее 2: (0,5·0,5·ln 0,5 + 0,5·1,5·ln 1,5)
    expected = 0.5 * 0.5 * np.log(0.5) + 0.5 * 1.5 * np.log(1.5)
    assert E.theil(np.array([1.0, 3.0]), np.ones(2)) == pytest.approx(expected)


def test_theil_handles_zero_time():
    assert np.isfinite(E.theil(np.array([0.0, 4.0]), np.ones(2)))


def test_theil_decomposes_exactly():
    """Между + внутри = всего: тождество, и оно же проверка расчёта."""
    rng = np.random.default_rng(0)
    times = rng.uniform(1, 60, 200)
    need = rng.uniform(0, 1000, 200)
    groups = rng.integers(0, 7, 200).astype(str).astype(object)
    parts = E.theil_decomposition(times, need, groups)
    assert parts["between"] + parts["within"] == pytest.approx(parts["total"], abs=1e-12)
    assert parts["between"] >= 0 and parts["within"] >= 0


def test_theil_between_is_zero_when_groups_have_equal_means():
    times = np.array([1.0, 3.0, 1.0, 3.0])
    parts = E.theil_decomposition(times, np.ones(4), np.array(["a", "a", "b", "b"],
                                                              dtype=object))
    assert parts["between"] == pytest.approx(0, abs=1e-12)
    assert parts["within"] == pytest.approx(parts["total"])


def test_theil_within_is_zero_when_groups_are_uniform():
    times = np.array([2.0, 2.0, 8.0, 8.0])
    parts = E.theil_decomposition(times, np.ones(4), np.array(["a", "a", "b", "b"],
                                                              dtype=object))
    assert parts["within"] == pytest.approx(0, abs=1e-12)
    assert parts["between"] == pytest.approx(parts["total"])


def test_cell_with_people_must_have_a_group():
    groups = np.array(["a", None], dtype=object)
    with pytest.raises(ValueError, match="нет группы"):
        E.theil_decomposition(np.array([1.0, 2.0]), np.ones(2), groups)


def test_empty_cell_may_have_no_group():
    groups = np.array(["a", None], dtype=object)
    parts = E.theil_decomposition(np.array([1.0, 9.0]), np.array([1.0, 0.0]), groups)
    assert parts["total"] == pytest.approx(0)


def test_mismatched_lengths_and_empty_weights_are_errors():
    with pytest.raises(ValueError, match="разной длины"):
        E.deficit(np.ones(3), np.ones(2), 15)
    with pytest.raises(ValueError, match="не все нулевые"):
        E.gini(np.ones(2), np.zeros(2))
