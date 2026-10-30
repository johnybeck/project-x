"""Подоценки модели → скор s_i (п. 3.5 пропозала).

Правило задаётся заранее, лежит в конфиге и уходит в протокол: ничего не
обучается и на результатах не подбирается. Модель здесь уже не участвует —
код только подставляет баллы вместо меток. Верно ли правило, проверяет сам
эксперимент: если оно не отражает нуждаемость, захват М2 у C и D будет около нуля.
"""
from __future__ import annotations

import pandas as pd

from .prompt import RATINGS

# Доля жилья говорит, сколько людей живёт в клетке, а это уже учтено в P_i.
# Пусти её в скор — и D − C смешает «видит ли модель нуждаемость» с «поправляет
# ли модель растр населения» (п. 3.5).
_FORBIDDEN = ("housing_share",)


def check(rule: dict) -> None:
    """Правило должно давать балл каждому ответу модели, и только им.

    Пропущенная метка молча превратила бы ответ в пустой скор, а опечатка в
    конфиге — в метку, которой модель никогда не вернёт."""
    rating = rule["rating"]
    if rating in _FORBIDDEN:
        raise ValueError(f"подоценка {rating} не может входить в правило")
    if rating not in RATINGS:
        raise ValueError(f"нет такой подоценки: {rating}")
    expected, actual = set(RATINGS[rating]), set(rule["points"])
    if expected != actual:
        raise ValueError(f"правило для {rating}: нет баллов для "
                         f"{sorted(expected - actual)}, лишние метки "
                         f"{sorted(actual - expected)}")


def active(cfg: dict) -> dict:
    """Действующее правило из конфига, проверенное."""
    rule = cfg["score_rules"][cfg["score_rule"]]
    check(rule)
    return rule


def score(ratings: pd.DataFrame, rule: dict) -> pd.Series:
    """Скор клеток по правилу. Клетка без ответа модели остаётся пустой."""
    return ratings[rule["rating"]].map(rule["points"]).astype(float)
