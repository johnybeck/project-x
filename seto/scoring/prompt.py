"""Промпт и текстовое описание клетки (п. 3.5 пропозала v2).

Один шаблон на оба варианта: C и D отличаются РОВНО одной фразой — тем, что
подаётся на вход. Иначе разница D − C перестала бы означать вклад изображения
и начала бы мерить разницу формулировок.

Модель не спрашивают про нуждаемость: она отмечает, что видит, а скор из этих
подоценок получается по заранее заданному правилу (scoring/rules.py). Прямой
числовой ответ без обучения стягивается к среднему по городу, а признаки,
извлечённые моделью, информативнее.
"""
from __future__ import annotations

import hashlib

import pandas as pd

from ..data.features import EPOCH_UNKNOWN, EPOCHS

UNKNOWN = "не определить"

# Подоценки и допустимые ответы. Текст тех же меток — в TEMPLATE: разбор ответа
# и правила перевода опираются на этот словарь, промпт читает человек.
RATINGS = {
    "housing_type": ("многоквартирные кварталы", "многоквартирные отдельные дома",
                     "малоэтажные дома рядами", "дома с участками", "смешанное",
                     "жилья нет", UNKNOWN),
    "epoch": ("до 1955", "1955-1975", "1975-2000", "после 2000", UNKNOWN),
    "housing_share": ("нет", "меньше половины", "около половины",
                      "больше половины", UNKNOWN),
    "construction": ("да", "нет", UNKNOWN),
    "purpose": ("жилое", "общественное и деловое", "промышленное и складское",
                "транспортное", "зелень и вода", "незастроенное", UNKNOWN),
    "affluence": ("низкий", "ниже среднего", "средний", "выше среднего",
                  "высокий", UNKNOWN),
}

# Общая часть. {input_clause} — единственное, чем отличаются C и D.
TEMPLATE = """Ты описываешь городскую клетку {size} м по {input_clause}

Отмечай только то, что действительно можно определить. Не вычисляй ничего и не \
оценивай численность людей — суди качественно, как если бы просто осмотрелся \
вокруг. Если признак определить нельзя, отвечай "не определить", не угадывай.

Признаки и допустимые ответы:
- housing_type — какое жильё преобладает: "многоквартирные кварталы" (дома \
сомкнуты по периметру кварталов), "многоквартирные отдельные дома" (башни и \
пластины, стоящие свободно), "малоэтажные дома рядами", "дома с участками", \
"смешанное", "жилья нет", "не определить".
- epoch — когда построена большая часть жилых зданий: "до 1955", "1955-1975", \
"1975-2000", "после 2000", "не определить".
- housing_share — какую часть клетки занимают жилые здания с их дворами: "нет", \
"меньше половины", "около половины", "больше половины", "не определить".
- construction — идёт ли заметная стройка: "да", "нет", "не определить".
- purpose — преобладающее назначение клетки: "жилое", "общественное и деловое", \
"промышленное и складское", "транспортное", "зелень и вода", "незастроенное", \
"не определить".
- affluence — достаток района по плотности машин, качеству застройки и кровель: \
"низкий", "ниже среднего", "средний", "выше среднего", "высокий", "не определить".

Верни СТРОГО один JSON-объект без markdown:
{{"housing_type": "...", "epoch": "...", "housing_share": "...", \
"construction": "...", "purpose": "...", "affluence": "...", \
"rationale": "одно короткое предложение"}}"""

IMAGE_CLAUSE = "спутниковому снимку."
TEXT_CLAUSE = "описанию ниже.\n\nОписание клетки:\n{text}"


def build(mode: str, size_m: int, text: str | None = None) -> str:
    """Промпт для варианта D (mode='image') или C (mode='text')."""
    size = f"{size_m}x{size_m}"
    if mode == "image":
        return TEMPLATE.format(size=size, input_clause=IMAGE_CLAUSE)
    if mode == "text":
        if not text:
            raise ValueError("для режима 'text' нужно описание клетки")
        return TEMPLATE.format(size=size,
                               input_clause=TEXT_CLAUSE.format(text=text))
    raise ValueError(f"режим должен быть 'image' или 'text', а не {mode!r}")


def prompt_hash(prompt: str) -> str:
    """Хеш промпта для манифеста и протокола (п. 7.1)."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def diff() -> dict:
    """Чем именно отличаются промпты C и D — идёт в протокол (п. 7.1)."""
    return {"common_template": TEMPLATE,
            "image_clause": IMAGE_CLAUSE,
            "text_clause": TEXT_CLAUSE,
            "differs_only_in": "input_clause"}


# --- текст варианта C ---

TEXT_TEMPLATE_VERSION = "1.1"
_MISSING = "нет данных"
_EPOCH_COLUMNS = (*((column, label) for column, label, _ in EPOCHS),
                  (EPOCH_UNKNOWN, "год неизвестен"))


def _number(value, digits: int = 0) -> str:
    """Пропуск — явным словом, а не нулём (п. 7.3): ноль школ и неизвестное
    число школ это разные вещи, и модель не должна их путать."""
    if value is None or pd.isna(value):
        return _MISSING
    return f"{float(value):.{digits}f}" if digits else f"{int(value)}"


def _percent(value) -> str:
    if value is None or pd.isna(value):
        return _MISSING
    return f"{100 * float(value):.0f} %"


def _epochs(row) -> str:
    """Жилые здания по эпохам в процентах: без числа зданий, потому что здание
    кадастра — это участок и с числом зданий OSM в том же тексте оно не сходится."""
    counts = [row.get(column) for column, _ in _EPOCH_COLUMNS]
    if any(value is None or pd.isna(value) for value in counts) or not sum(counts):
        return _MISSING
    return ", ".join(f"{label} — {100 * count / sum(counts):.0f} %"
                     for (_, label), count in zip(_EPOCH_COLUMNS, counts))


def _chains(row) -> str:
    """Сколько магазинов с брендом сети и каких. Ноль здесь значит «бренд не
    указан», а не «сетей нет»: тег стоит не у всех магазинов."""
    count = row.get("chain_shops")
    if count is None or pd.isna(count):
        return _MISSING
    return f"{int(count)} ({row['chain_brands']})" if count else "0"


def describe(row) -> str:
    """Детерминированное описание клетки из её признаков (п. 7.3).

    Населения в тексте НЕТ: оно уже входит в вес w_i = P_i·(1 + α·ŝ_i),
    а модель не должна считать числа."""
    levels = (_MISSING if pd.isna(row.get("mean_levels"))
              else f"{float(row['mean_levels']):.1f}")
    return (
        f"Застройка занимает {_percent(row.get('built_share'))} площади клетки: "
        f"жилая {_percent(row.get('residential_share'))}, "
        f"общественная и деловая {_percent(row.get('public_share'))}. "
        f"Зданий: {_number(row.get('buildings_count'))}, "
        f"средняя этажность {levels} "
        f"(этажность известна у {_percent(row.get('levels_known_share'))} зданий). "
        f"Жилые здания по году постройки: {_epochs(row)}. "
        f"Школы: {_number(row.get('schools'))}, "
        f"детские сады: {_number(row.get('kindergartens'))}, "
        f"аптеки: {_number(row.get('pharmacies'))}, "
        f"магазины: {_number(row.get('shops'))}, "
        f"из них с брендом сети: {_chains(row)}, "
        f"остановки транспорта: {_number(row.get('transport_stops'))}. "
        f"Парки и зелень: {_percent(row.get('green_share'))}, "
        f"вода: {_percent(row.get('water_share'))}, "
        f"промзона: {_percent(row.get('landuse_industrial_share'))}. "
        f"Плотность улиц: "
        f"{_number(row.get('street_density_m_per_km2'))} м на кв. км."
    )


def describe_all(features: pd.DataFrame) -> pd.Series:
    """Описания по всем клеткам, в порядке таблицы признаков."""
    return features.apply(describe, axis=1)
