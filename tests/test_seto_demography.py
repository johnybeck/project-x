"""Тесты итогов по участкам и возрастным группам (задача 3.1 TODO_V2)."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seto.data import demography as DM  # noqa: E402

RULES = {
    "code_prefix": "28079",
    "district_column": "dist",
    "district_digits": 2,
    "section_column": "sec",
    "section_digits": 3,
    "age_column": "edad",
    "count_columns": ["h", "m"],
    "groups": {"under_18": [0, 17], "adults_18_64": [18, 64],
               "seniors_65": [65, None], "seniors_80": [80, None]},
    "partition": ["under_18", "adults_18_64", "seniors_65"],
}


def rows(*records) -> pd.DataFrame:
    """Строки падрона: (район, участок, возраст, мужчины, женщины), всё строками."""
    return pd.DataFrame([[str(v) if v is not None else None for v in r] for r in records],
                        columns=["dist", "sec", "edad", "h", "m"])


def test_section_code_is_padded_like_ine():
    code = DM.section_codes(rows(("3", "97", "40", "1", "1")), RULES)
    assert code.tolist() == ["2807903097"]


def test_groups_per_section_and_80_plus_inside_65_plus():
    table = DM.totals(rows(("1", "1", "  10", "1", "2"),
                           ("1", "1", "  40", "5", "5"),
                           ("1", "1", "  70", "2", "0"),
                           ("1", "1", "  85", "1", "1")), RULES).table
    section = table.loc["2807901001"]
    assert section["total"] == 17
    assert (section["under_18"], section["adults_18_64"], section["seniors_65"]) == (3, 10, 4)
    assert section["seniors_80"] == 2


def test_top_age_class_counts_as_its_lower_bound():
    """«100 o +» — это 100+, то есть и 65+, и 80+."""
    table = DM.totals(rows(("1", "1", "100 o +", "1", "0")), RULES).table
    assert table.loc["2807901001", "seniors_80"] == 1


def test_people_without_section_are_reported_not_lost():
    result = DM.totals(rows(("1", "1", "30", "1", "1"),
                            ("2", None, "30", "3", "0")), RULES)
    assert result.table["total"].sum() == 2
    assert (result.unassigned_people, result.unassigned_rows) == (3, 1)


def test_missing_count_is_zero_people():
    table = DM.totals(rows(("1", "1", "30", "4", None)), RULES).table
    assert table.loc["2807901001", "total"] == 4


def test_overlapping_partition_is_an_error():
    rules = {**RULES, "groups": {**RULES["groups"], "adults_18_64": [17, 64]}}
    with pytest.raises(ValueError, match="не дают всё население"):
        DM.totals(rows(("1", "1", "17", "1", "0")), rules)


def test_unreadable_age_is_an_error():
    with pytest.raises(ValueError, match="возраст"):
        DM.totals(rows(("1", "1", "n/a", "1", "0")), RULES)


# --- нуждаемость (задача 3.3) ---

GROUPS = {"adults_18_64": [18, 64], "seniors_65": [65, None],
          "seniors_80": [80, None], "adults_18_79": [18, 79]}
PEOPLE = pd.DataFrame({"adults_18_64": [100.0, 0.0], "seniors_65": [10.0, 50.0],
                       "seniors_80": [4.0, 30.0], "adults_18_79": [106.0, 20.0]})


def test_need_is_weighted_sum_of_groups():
    n = DM.need(PEOPLE, {"adults_18_64": 1, "seniors_65": "r"}, 2.0, GROUPS)
    assert n.tolist() == [120.0, 100.0]


def test_multiple_of_r_follows_r():
    """«2r» пересчитывается из r, а не вписан числом — правка r его не забудет."""
    assert DM.weight("2r", 1.93) == pytest.approx(3.86)
    assert DM.weight("r", 1.5) == 1.5
    assert DM.weight(1, 9.0) == 1.0


def test_seniors_80_variant_shifts_the_threshold():
    n = DM.need(PEOPLE, {"adults_18_79": 1, "seniors_80": "r"}, 2.0, GROUPS)
    assert n.tolist() == [114.0, 80.0]


def test_overlapping_groups_would_count_people_twice():
    with pytest.raises(ValueError, match="пересекаются"):
        DM.need(PEOPLE, {"adults_18_64": 1, "seniors_65": "r", "seniors_80": "r"},
                2.0, GROUPS)


def test_unknown_group_and_bad_weight_are_errors():
    with pytest.raises(ValueError, match="неизвестные группы"):
        DM.need(PEOPLE, {"adults": 1}, 2.0, GROUPS)
    with pytest.raises(ValueError, match="кратное r"):
        DM.weight("r2", 2.0)
    with pytest.raises(ValueError, match="отрицательный"):
        DM.weight(-1, 2.0)


def test_madrid_config_definitions_are_valid():
    """Все определения из madrid.yaml собираются на его группах."""
    from seto import settings
    city = settings.load("madrid")["city"]
    people = pd.DataFrame({name: [1.0] for name in city["demography"]["groups"]})
    variants = DM.need_all(people, city["need"], city["demography"]["groups"])
    assert set(variants) == {"age", "age_r1", "age_r2", "age_80"}
    assert variants["age_r2"].iloc[0] == pytest.approx(1 + 2 * city["need"]["r"])


def test_cell_groups_strips_prefix():
    features = pd.DataFrame({"cell_id": ["a"], "pop_seniors_65": [3.0], "population": [9.0]})
    assert DM.cell_groups(features).columns.tolist() == ["seniors_65"]


def test_match_reports_both_sides(tmp_path):
    path = tmp_path / "sections.geojson"
    path.write_text(json.dumps({"features": [
        {"properties": {"CUSEC": "2807901001", "TIPO": "SECCIONADO"}},
        {"properties": {"CUSEC": "2807901002", "TIPO": "SECCIONADO"}},
        {"properties": {"CUSEC": "2807901000", "TIPO": "DISTRITO"}},
    ]}), encoding="utf-8")
    codes = DM.boundary_codes(path, {"code_column": "CUSEC",
                                     "exclude": {"TIPO": "DISTRITO"}})
    assert codes == {"2807901001", "2807901002"}

    result = DM.totals(rows(("1", "1", "30", "1", "0"),
                            ("1", "9", "30", "3", "0")), RULES)
    report = DM.match(result, codes)
    assert report["matched"] == 1
    assert report["only_population"] == {"2807901009": 3}
    assert report["only_boundaries"] == ["2807901002"]
