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
