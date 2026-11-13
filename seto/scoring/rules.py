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
    """Действующее правило из конфига, проверенное. Заодно проверяются правила из
    списка проверок: опечатка в них не должна всплыть уже после прогона модели."""
    rule = cfg["score_rules"][cfg["score_rule"]]
    check(rule)
    for name in cfg.get("score_rule_checks") or {}:
        for_check(cfg, name)
    return rule


def for_check(cfg: dict, name: str) -> dict:
    """Правило-проверка из закрытого списка `score_rule_checks`.

    Список входит в протокол: правило, которого в нём нет, посчитать нельзя, даже
    если оно описано в `score_rules`. Иначе перевод подоценок можно было бы
    подобрать после результатов."""
    listed = cfg.get("score_rule_checks") or {}
    if name not in listed:
        raise ValueError(f"правила {name!r} нет в списке проверок score_rule_checks: "
                         f"{sorted(listed)}")
    if name == cfg["score_rule"]:
        raise ValueError(f"{name!r} — основное правило, а не проверка")
    if name not in cfg["score_rules"]:
        raise ValueError(f"правило-проверка {name!r} не описано в score_rules")
    rule = cfg["score_rules"][name]
    check(rule)
    return rule


def score(ratings: pd.DataFrame, rule: dict) -> pd.Series:
    """Скор клеток по правилу. Клетка без ответа модели остаётся пустой."""
    return ratings[rule["rating"]].map(rule["points"]).astype(float)


def score_filled(ratings: pd.DataFrame, rule: dict) -> pd.Series:
    """Скор клеток; клетке без ответа модели (`imputed`) — медиана скоров города
    (п. 7.1). Один расчёт для основного правила и для правил-проверок: проверка
    пересчитывает скор из сохранённых ответов тем же способом."""
    values = score(ratings, rule)
    missing = ratings["imputed"].to_numpy(dtype=bool)
    if missing.any():
        median = values[~missing].median()
        if pd.isna(median):
            raise RuntimeError("ни одна клетка не получила скор — считать нечего")
        values = values.mask(missing, median)
    return values.astype(float)
