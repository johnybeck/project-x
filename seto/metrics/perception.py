"""М4: видит ли модель то, о чём её спрашивают (пропозал v2, п. 3.10).

Проверка оценщика до проверки размещения: если модель не видит эпоху, нечего ждать
и от весов. Эпоха сверяется по каппе Коэна с годом постройки, достаток — по ранговой
корреляции с доходом. Эпоха считается только для снимка: текст C сам содержит год.

Три решения приняты до первых ответов модели, чтобы не подбирать их под результат:

1. Эталон эпохи — самая частая эпоха жилых зданий клетки. Клетка, где зданий меньше
   `min_homes` или где две эпохи делят первое место, в сверку не идёт: эталона там нет.
2. «Не определить» считается промахом. Иначе модель, отказавшаяся от всех трудных
   клеток, получила бы завышенную каппу. Отдельно показана каппа без отказов.
3. Доход клетки — среднее по участкам, взвешенное жителями: клетка лежит в нескольких
   участках, а отвечает модель про клетку целиком.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.features import EPOCHS
from ..scoring.prompt import RATINGS, UNKNOWN

_EPOCH_LABELS = [label for _, label, _ in EPOCHS]
_EPOCH_COLUMNS = [column for column, _, _ in EPOCHS]


def main_epoch(features: pd.DataFrame, min_homes: int) -> pd.Series:
    """Эталон: самая частая эпоха жилых зданий клетки, индекс — cell_id. Нет эталона — NaN."""
    counts = features.set_index("cell_id")[_EPOCH_COLUMNS].astype(float)
    top = counts.max(axis=1)
    single = (counts.eq(top, axis=0).sum(axis=1) == 1) & (top > 0)
    enough = counts.sum(axis=1) >= min_homes
    labels = counts.to_numpy().argmax(axis=1)
    result = pd.Series(np.array(_EPOCH_LABELS, dtype=object)[labels], index=counts.index,
                       name="epoch_truth")
    return result.where(single & enough)


def cohen_kappa(truth: pd.Series, answers: pd.Series) -> float:
    """(доля совпадений − ожидаемая случайно) / (1 − ожидаемая случайно)."""
    agreed = float((truth.to_numpy() == answers.to_numpy()).mean())
    p_truth = truth.value_counts(normalize=True)
    p_answer = answers.value_counts(normalize=True)
    chance = float((p_truth * p_answer.reindex(p_truth.index, fill_value=0.0)).sum())
    return 1.0 if chance == 1.0 else (agreed - chance) / (1.0 - chance)


def epoch_agreement(truth: pd.Series, answers: pd.Series) -> dict:
    """Согласие с эталоном эпохи на клетках, где есть и эталон, и ответ модели."""
    both = pd.DataFrame({"truth": truth, "answer": answers}).dropna()
    if both.empty:
        return {"cells": 0}
    answered = both[both["answer"] != UNKNOWN]
    return {"cells": len(both),
            "kappa": cohen_kappa(both["truth"], both["answer"]),
            "agreement": float((both["truth"] == both["answer"]).mean()),
            "answered_share": len(answered) / len(both),
            "kappa_answered": (cohen_kappa(answered["truth"], answered["answer"])
                               if len(answered) else None),
            "confusion": pd.crosstab(both["truth"], both["answer"]).to_dict()}


def cell_income(cell_sections: pd.DataFrame, income: pd.Series) -> pd.Series:
    """Доход клетки: среднее по участкам, взвешенное жителями; индекс — cell_id.

    Участок без дохода выпадает из среднего, а не считается нулём."""
    parts = cell_sections.assign(income=cell_sections["code"].map(income)).dropna(subset=["income"])
    parts = parts[parts["total"] > 0]
    weighted = (parts["income"] * parts["total"]).groupby(parts["cell_id"]).sum()
    return (weighted / parts.groupby("cell_id")["total"].sum()).rename("income")


def affluence_agreement(income: pd.Series, answers: pd.Series) -> dict:
    """Ранговая корреляция дохода с ответом модели; «не определить» не имеет ранга."""
    order = {label: rank for rank, label in enumerate(RATINGS["affluence"])
             if label != UNKNOWN}
    both = pd.DataFrame({"income": income, "answer": answers}).dropna()
    ranks = both["answer"].map(order)
    known = ranks.notna()
    result = {"cells": len(both),
              "answered_share": float(known.mean()) if len(both) else None}
    if known.sum() >= 3 and ranks[known].nunique() > 1:
        result["spearman"] = float(both.loc[known, "income"].corr(ranks[known], method="spearman"))
    return result


def evaluate(features: pd.DataFrame, cell_sections: pd.DataFrame, income: pd.Series,
             scores: pd.DataFrame, min_homes: int, with_epoch: bool) -> dict:
    """М4 для одного набора ответов. Сбойные клетки в сверку не идут."""
    answers = scores[~scores["imputed"]].set_index("cell_id")
    result = {"cells_scored": len(scores), "cells_failed": int(scores["imputed"].sum()),
              "affluence": affluence_agreement(cell_income(cell_sections, income),
                                               answers["affluence"])}
    if with_epoch:
        result["epoch"] = epoch_agreement(main_epoch(features, min_homes), answers["epoch"])
    return result
