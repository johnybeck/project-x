"""Тесты М4: ответы модели здесь выдуманы, чтобы результат был известен заранее."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seto.data import income as I  # noqa: E402
from seto.metrics import perception as M  # noqa: E402
from seto.scoring.prompt import UNKNOWN  # noqa: E402

OLD, MID, LATE, NEW = "до 1955", "1955-1975", "1975-2000", "после 2000"


def _features(*rows):
    """rows: (cell_id, до 1955, 1955–75, 1975–2000, после 2000)."""
    return pd.DataFrame(rows, columns=["cell_id", "epoch_before_1955", "epoch_1955_1975",
                                       "epoch_1975_2000", "epoch_after_2000"])


# --- эталон эпохи ---

def test_main_epoch_is_the_most_common_one():
    truth = M.main_epoch(_features(("a", 1, 7, 2, 0), ("b", 0, 0, 1, 9)), min_homes=5)
    assert truth.to_dict() == {"a": MID, "b": NEW}


def test_too_few_homes_give_no_truth():
    truth = M.main_epoch(_features(("a", 0, 4, 0, 0)), min_homes=5)
    assert truth.isna().all()


def test_tie_gives_no_truth():
    truth = M.main_epoch(_features(("a", 5, 5, 1, 0)), min_homes=5)
    assert truth.isna().all()


def test_cell_without_homes_gives_no_truth():
    truth = M.main_epoch(_features(("a", 0, 0, 0, 0)), min_homes=0)
    assert truth.isna().all()


# --- каппа ---

def test_full_agreement_is_one():
    labels = pd.Series([OLD, MID, NEW, NEW])
    assert M.cohen_kappa(labels, labels) == pytest.approx(1.0)


def test_kappa_by_hand():
    """Совпало 3 из 4; случайно ожидается 0,5·0,25 + 0,5·0,75 = 0,5; (0,75−0,5)/0,5."""
    truth = pd.Series([OLD, OLD, NEW, NEW])
    answers = pd.Series([OLD, NEW, NEW, NEW])
    assert M.cohen_kappa(truth, answers) == pytest.approx(0.5)


def test_always_the_same_answer_gives_zero():
    """Модель всегда отвечает одним классом: совпадения есть, знания нет."""
    truth = pd.Series([OLD, OLD, OLD, NEW])
    assert M.cohen_kappa(truth, pd.Series([OLD] * 4)) == pytest.approx(0.0)


def test_refusal_counts_as_a_miss():
    """Отказ на трудных клетках не должен поднимать каппу."""
    truth = pd.Series([OLD, OLD, NEW, NEW], index=list("abcd"))
    answers = pd.Series([OLD, UNKNOWN, NEW, UNKNOWN], index=list("abcd"))
    result = M.epoch_agreement(truth, answers)
    assert result["kappa"] < 1.0
    assert result["kappa_answered"] == pytest.approx(1.0)
    assert result["answered_share"] == pytest.approx(0.5)


def test_cells_without_truth_are_left_out():
    truth = pd.Series([OLD, np.nan], index=["a", "b"])
    result = M.epoch_agreement(truth, pd.Series([OLD, NEW], index=["a", "b"]))
    assert result["cells"] == 1


# --- доход ---

def test_cell_income_is_weighted_by_people():
    parts = pd.DataFrame({"cell_id": ["a", "a"], "code": ["s1", "s2"], "total": [300.0, 100.0]})
    income = pd.Series({"s1": 10_000.0, "s2": 30_000.0})
    assert M.cell_income(parts, income)["a"] == pytest.approx(15_000)


def test_section_without_income_is_not_zero():
    parts = pd.DataFrame({"cell_id": ["a", "a"], "code": ["s1", "s2"], "total": [300.0, 100.0]})
    assert M.cell_income(parts, pd.Series({"s1": 10_000.0}))["a"] == pytest.approx(10_000)


def test_rank_correlation_with_income():
    income = pd.Series([10.0, 20.0, 30.0, 40.0], index=list("abcd"))
    answers = pd.Series(["низкий", "ниже среднего", "средний", "высокий"], index=list("abcd"))
    assert M.affluence_agreement(income, answers)["spearman"] == pytest.approx(1.0)


def test_refusals_have_no_rank():
    income = pd.Series([10.0, 20.0, 30.0, 40.0], index=list("abcd"))
    answers = pd.Series(["низкий", UNKNOWN, "средний", "высокий"], index=list("abcd"))
    result = M.affluence_agreement(income, answers)
    assert result["answered_share"] == pytest.approx(0.75)
    assert result["spearman"] == pytest.approx(1.0)


def test_one_class_gives_no_correlation():
    income = pd.Series([10.0, 20.0, 30.0], index=list("abc"))
    assert "spearman" not in M.affluence_agreement(income, pd.Series(["средний"] * 3, index=list("abc")))


# --- весь расчёт ---

def _scores():
    return pd.DataFrame({"cell_id": ["a", "b", "c"], "epoch": [MID, NEW, OLD],
                         "affluence": ["низкий", "высокий", "средний"],
                         "imputed": [False, False, True]})


def _parts():
    return pd.DataFrame({"cell_id": ["a", "b"], "code": ["s1", "s2"], "total": [10.0, 10.0]})


def test_failed_cells_are_left_out():
    features = _features(("a", 0, 9, 0, 0), ("b", 0, 0, 0, 9), ("c", 9, 0, 0, 0))
    result = M.evaluate(features, _parts(), pd.Series({"s1": 1.0, "s2": 2.0}), _scores(), 5, True)
    assert result["cells_failed"] == 1
    assert result["epoch"]["cells"] == 2
    assert result["epoch"]["kappa"] == pytest.approx(1.0)


def test_text_is_not_checked_on_epoch():
    """Текст C сам содержит год: сверять его с годом бессмысленно."""
    features = _features(("a", 0, 9, 0, 0), ("b", 0, 0, 0, 9), ("c", 9, 0, 0, 0))
    result = M.evaluate(features, _parts(), pd.Series({"s1": 1.0, "s2": 2.0}), _scores(), 5, False)
    assert "epoch" not in result
    assert "affluence" in result


# --- таблица дохода ---

RULES = {"sep": ";", "encoding": "utf-8", "section_column": "Secciones",
         "indicator_column": "Indicador", "indicator": "Renta neta media por persona",
         "year_column": "Periodo", "year": 2023, "value_column": "Total"}


def _table(path):
    path.write_text(
        "Municipios;Distritos;Secciones;Indicador;Periodo;Total\n"
        "28079 Madrid;2807901 Madrid distrito 01;2807901001 Madrid sección 01001;"
        "Renta neta media por persona;2023;27.408\n"
        "28079 Madrid;2807901 Madrid distrito 01;2807901001 Madrid sección 01001;"
        "Renta neta media por persona;2022;26.032\n"
        "28079 Madrid;2807901 Madrid distrito 01;2807901002 Madrid sección 01002;"
        "Renta neta media por persona;2023;.\n"
        "28079 Madrid;2807901 Madrid distrito 01;2807901001 Madrid sección 01001;"
        "Renta neta media por hogar;2023;55.000\n"
        "28079 Madrid;2807901 Madrid distrito 01;;Renta neta media por persona;2023;22.073\n",
        encoding="utf-8")
    return path


def test_income_is_read_for_the_year_and_indicator(tmp_path):
    income = I.read(_table(tmp_path / "t.csv"), RULES)
    assert income.to_dict() == {"2807901001": 27408.0}


def test_wrong_indicator_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="проверь раздел income"):
        I.read(_table(tmp_path / "t.csv"), {**RULES, "indicator": "нет такого"})
