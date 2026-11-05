"""Тесты сети из реестра медучреждений (задача 2.1 TODO_V2)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seto.data import registry as RG  # noqa: E402

RULES = {
    "sep": ";",
    "encoding": "utf-8",
    "id_column": "id",
    "x_column": "x",
    "y_column": "y",
    "crs": 25830,
    "scope": {"column": "muni", "values": ["Madrid"]},
    "require": {
        "не первичная помощь": {"column": "tipo",
                                "values": ["Centro de salud", "Consultorio"]},
        "не государственная": {"column": "dep", "values": ["SERMAS"]},
    },
}

HEADER = "id;tipo;dep;muni;servicio;x;y"


def csv(tmp_path, *rows) -> Path:
    path = tmp_path / "registry.csv"
    path.write_text("\n".join([HEADER, *rows]) + "\n", encoding="utf-8")
    return path


def test_one_row_per_centre_although_file_has_row_per_service(tmp_path):
    """Учреждение с тремя услугами — одна точка сети, а не три."""
    path = csv(tmp_path,
               "CS1;Centro de salud;SERMAS;Madrid;Medicina de familia;440000;4474000",
               "CS1;Centro de salud;SERMAS;Madrid;Enfermería;440000;4474000",
               "CS1;Centro de salud;SERMAS;Madrid;Pediatría;440000;4474000")
    sel = RG.load(path, RULES)
    assert sel.selected == 1 and sel.candidates == 1


def test_other_municipalities_are_not_candidates(tmp_path):
    path = csv(tmp_path,
               "CS1;Centro de salud;SERMAS;Madrid;x;440000;4474000",
               "CS2;Centro de salud;SERMAS;Getafe;x;440000;4460000")
    sel = RG.load(path, RULES)
    assert sel.candidates == 1 and sel.selected == 1


def test_exclusion_goes_to_first_failed_rule(tmp_path):
    """Частная больница не прошла ни одного правила, но записывается одним —
    первым по порядку; иначе исключённые посчитались бы дважды."""
    path = csv(tmp_path,
               "H1;Hospital general;Privado;Madrid;x;440000;4474000",
               "CS2;Centro de salud;Privado;Madrid;x;440100;4474000",
               "CS3;Consultorio;SERMAS;Madrid;x;440200;4474000")
    sel = RG.load(path, RULES)
    assert sel.excluded_by == {"не первичная помощь": 1, "не государственная": 1}
    assert sel.frame.index.tolist() == ["CS3"]


def test_centre_without_coordinates_is_kept_aside_not_dropped(tmp_path):
    """Центр сети без координат не исчезает молча: он нужен человеку."""
    path = csv(tmp_path,
               "CS1;Centro de salud;SERMAS;Madrid;x;440000;4474000",
               "CS2;Centro de salud;SERMAS;Madrid;x;;")
    sel = RG.load(path, RULES)
    assert sel.selected == 1
    assert sel.no_coordinates.index.tolist() == ["CS2"]
    assert sel.uncertain is sel.no_coordinates
    assert sel.as_dict()["no_coordinates_ids"] == ["CS2"]


def test_points_carry_crs_and_coordinate_source(tmp_path):
    path = csv(tmp_path, "CS1;Centro de salud;SERMAS;Madrid;x;440000;4474000")
    frame = RG.load(path, RULES).frame
    assert frame.crs.to_epsg() == 25830
    assert (frame.geometry.x.iloc[0], frame.geometry.y.iloc[0]) == (440000, 4474000)
    assert frame["coord_source"].tolist() == ["registry"]


def test_trailing_spaces_in_values_do_not_break_rules(tmp_path):
    path = csv(tmp_path, "CS1; Centro de salud ;SERMAS ;Madrid ;x;440000;4474000")
    assert RG.load(path, RULES).selected == 1


def test_conflicting_type_within_one_centre_is_an_error(tmp_path):
    """Разные типы у строк одного учреждения — не угадываем, а падаем."""
    path = csv(tmp_path,
               "CS1;Centro de salud;SERMAS;Madrid;a;440000;4474000",
               "CS1;Hospital general;SERMAS;Madrid;b;440000;4474000")
    with pytest.raises(ValueError, match="tipo"):
        RG.load(path, RULES)


MANUAL = {"CS2": {"x": 440500, "y": 4474500, "source": "OSM w1 «Centro de Salud X»"}}


def test_manual_point_fills_the_gap_and_is_marked(tmp_path):
    """Точка из конфига ставит центр на карту, и видно, что она ручная."""
    path = csv(tmp_path,
               "CS1;Centro de salud;SERMAS;Madrid;x;440000;4474000",
               "CS2;Centro de salud;SERMAS;Madrid;x;;")
    sel = RG.load(path, {**RULES, "manual_points": MANUAL})
    assert sel.selected == 2 and sel.no_coordinates.empty
    point = sel.frame.loc["CS2"]
    assert (point.geometry.x, point.geometry.y) == (440500, 4474500)
    assert point.coord_source == "manual" and point.coord_note.startswith("OSM w1")
    assert sel.frame.loc["CS1", "coord_source"] == "registry"
    assert sel.as_dict()["manual_points"] == ["CS2"]


def test_manual_point_never_overrides_registry(tmp_path):
    path = csv(tmp_path, "CS2;Centro de salud;SERMAS;Madrid;x;440000;4474000")
    with pytest.raises(ValueError, match="перебивается"):
        RG.load(path, {**RULES, "manual_points": MANUAL})


def test_manual_point_for_centre_outside_network_is_an_error(tmp_path):
    """Устаревший конфиг: центр исключён правилом, а точка для него осталась."""
    path = csv(tmp_path, "CS2;Hospital general;SERMAS;Madrid;x;;")
    with pytest.raises(ValueError, match="вне сети"):
        RG.load(path, {**RULES, "manual_points": MANUAL})


def test_manual_point_must_say_where_it_came_from(tmp_path):
    path = csv(tmp_path, "CS2;Centro de salud;SERMAS;Madrid;x;;")
    with pytest.raises(ValueError, match="источник"):
        RG.load(path, {**RULES, "manual_points": {"CS2": {"x": 1, "y": 2}}})


ANY_OF = {
    **RULES,
    "require": {
        "не первичная помощь": RULES["require"]["не первичная помощь"],
        "не государственная": {"any_of": [{"column": "dep", "values": ["SERMAS"]},
                                          {"column": "own", "values": ["SERMAS"]}]},
    },
}


def any_of_csv(tmp_path, *rows) -> Path:
    path = tmp_path / "registry.csv"
    path.write_text("\n".join(["id;tipo;dep;own;muni;x;y", *rows]) + "\n",
                    encoding="utf-8")
    return path


def test_any_of_rule_passes_when_one_condition_holds(tmp_path):
    """Как Las Tablas в Мадриде: подчинение странное, имущество — SERMAS.
    Центр остаётся в сети; не прошедший ни одного условия — исключён."""
    path = any_of_csv(tmp_path,
                      "CS1;Centro de salud;SERMAS;Estado;Madrid;440000;4474000",
                      "CS2;Centro de salud;ISCIII;SERMAS;Madrid;440100;4474000",
                      "CS3;Centro de salud;Estado;Estado;Madrid;440200;4474000")
    sel = RG.load(path, ANY_OF)
    assert sel.frame.index.tolist() == ["CS1", "CS2"]
    assert sel.excluded_by == {"не первичная помощь": 0, "не государственная": 1}


def test_any_of_columns_must_agree_within_one_centre(tmp_path):
    """Поле из any_of тоже обязано совпадать у всех строк учреждения."""
    path = any_of_csv(tmp_path,
                      "CS1;Centro de salud;ISCIII;SERMAS;Madrid;440000;4474000",
                      "CS1;Centro de salud;ISCIII;Estado;Madrid;440000;4474000")
    with pytest.raises(ValueError, match="own"):
        RG.load(path, ANY_OF)


def test_counts_always_add_up(tmp_path):
    path = csv(tmp_path,
               "CS1;Centro de salud;SERMAS;Madrid;x;440000;4474000",
               "CS2;Centro de salud;SERMAS;Madrid;x;;",
               "H1;Hospital general;SERMAS;Madrid;x;440000;4474000",
               "CS4;Centro de salud;Privado;Madrid;x;440000;4474000")
    sel = RG.load(path, RULES)
    assert (sel.selected + len(sel.no_coordinates)
            + sum(sel.excluded_by.values())) == sel.candidates == 4
