"""Доход по участкам: эталон достатка для М4 и частного сценария (пропозал v2, п. 3.8).

Формат посмотрен на файле ADRH, табл. 31097: одна строка — территория ×
показатель × год. Участок записан как «2807901001 Madrid sección 01001», строки
районов и муниципалитетов идут в том же файле с пустым участком. Тысячи
отделяются точкой («27.408»), а точка без цифр значит «нет данных».
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

_CODE = r"^(\d{10})\s"


def read(path: Path, rules: dict) -> pd.Series:
    """Доход по коду участка за год и показатель из конфига, без пропусков."""
    table = pd.read_csv(path, sep=rules["sep"], encoding=rules["encoding"], dtype=str)
    chosen = table[(table[rules["indicator_column"]] == rules["indicator"])
                   & (table[rules["year_column"]] == str(rules["year"]))]
    codes = chosen[rules["section_column"]].str.extract(_CODE, expand=False)
    values = pd.to_numeric(chosen[rules["value_column"]].str.replace(".", "", regex=False),
                           errors="coerce")
    income = pd.Series(values.to_numpy(), index=codes.to_numpy(), name="income", dtype=float)
    income = income[income.index.notna()].dropna()
    if income.empty:
        raise ValueError(f"в {path} нет дохода {rules['indicator']!r} за {rules['year']}: "
                         "проверь раздел income")
    income.index.name = "code"
    return income
