"""Блок «адресность»: кому досталось улучшение (пропозал v2, п. 3.10).

Главная метрика считает всех жителей одинаково и видит только порог T. Здесь
меры, которые различают, кого затронуло решение и насколько тяжёлым осталось
положение тех, кто далеко:

    EDE(S)   = −(1/κ) · ln[Σ_i N_i · e^(−κ·t_i(S)) / Σ_i N_i]      М3, минуты
    κ        = ε · Σ_i N_i t_i^E / Σ_i N_i (t_i^E)²                ε < 0
    дефицит  = Σ_i N_i · max(0, t_i(S) − T)                        чел.-мин
    Джини, Тейль — неравенство времени доступа, взвешенное по N

Все они считаются по времени, поэтому БОЛЬШЕ — ХУЖЕ (кроме покрытия).
Покрытие нуждающихся (М1) и 2SFCA с весами N отдельного кода не требуют:
это те же функции `coverage` и `sfca`, в которые вместо P подан N.
"""
from __future__ import annotations

import numpy as np


def _clean(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Проверить вход и вернуть только клетки с ненулевым весом."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.shape != weights.shape:
        raise ValueError(f"значения {values.shape} и веса {weights.shape} разной длины")
    if (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("веса должны быть неотрицательны и не все нулевые")
    keep = weights > 0
    return values[keep], weights[keep]


def kolm_pollak_kappa(times: np.ndarray, need: np.ndarray, epsilon: float) -> float:
    """κ = ε · Σ N t / Σ N t², по ИСХОДНОЙ сети, один на город.

    Если считать κ заново для каждого решения, EDE разных вариантов окажутся в
    разных шкалах, и разница между ними будет мерить ещё и разницу шкал."""
    if epsilon >= 0:
        raise ValueError("ε должен быть отрицательным: время — это издержка")
    t, n = _clean(times, need)
    second = float((n * t * t).sum())
    if second <= 0:
        raise ValueError("все времена нулевые: κ не определён")
    return epsilon * float((n * t).sum()) / second


def equally_distributed(times: np.ndarray, need: np.ndarray, kappa: float) -> float:
    """М3: одинаковое для всех время, «такое же плохое», как нынешнее неравное.

    При равных временах равно этому времени, при разбросе — больше среднего."""
    if kappa == 0:
        raise ValueError("κ не может быть нулём")
    t, n = _clean(times, need)
    return float(-np.log((n * np.exp(-kappa * t)).sum() / n.sum()) / kappa)


def deficit(times: np.ndarray, need: np.ndarray, threshold: float) -> float:
    """Сумма минут сверх порога у тех, кто не уложился: видит сокращение пути,
    которое бинарная главная метрика не замечает (40 минут стали 20)."""
    t, n = _clean(times, need)
    return float((n * np.maximum(0.0, t - threshold)).sum())


def gini(values: np.ndarray, weights: np.ndarray) -> float:
    """Взвешенный коэффициент Джини: 0 — у всех одинаково."""
    v, w = _clean(values, weights)
    order = np.argsort(v, kind="stable")
    v, w = v[order], w[order]
    held = np.cumsum(v * w)
    if held[-1] <= 0:
        return 0.0
    below = np.concatenate([[0.0], held[:-1]])
    # площадь под кривой Лоренца по трапециям
    area = float((w * (held + below)).sum()) / (2.0 * w.sum() * held[-1])
    return 1.0 - 2.0 * area


def theil(values: np.ndarray, weights: np.ndarray) -> float:
    """Индекс Тейля T: 0 — у всех одинаково. Клетка с нулём даёт ноль (предел)."""
    v, w = _clean(values, weights)
    mean = float((v * w).sum() / w.sum())
    if mean <= 0:
        return 0.0
    ratio = v / mean
    terms = np.zeros_like(ratio)
    positive = ratio > 0
    terms[positive] = ratio[positive] * np.log(ratio[positive])
    return float((w * terms).sum() / w.sum())


def theil_decomposition(values: np.ndarray, weights: np.ndarray, groups) -> dict:
    """Тейль = между группами + внутри групп, с долями времени как весами.

    Это точное тождество, а не приближение, и ради него Тейль и берётся:
    Джини так не раскладывается. Клетка с весом обязана иметь группу."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    labels = np.asarray(groups, dtype=object)
    if labels.shape != values.shape:
        raise ValueError(f"группы {labels.shape} и значения {values.shape} разной длины")
    missing = (weights > 0) & np.array([label is None or label != label
                                        for label in labels])
    if missing.any():
        raise ValueError(f"у {int(missing.sum())} клеток с весом нет группы")

    total_time = float((values * weights).sum())
    mean = total_time / float(weights.sum())
    between, within = 0.0, 0.0
    for label in set(labels[weights > 0]):
        inside = labels == label
        group_time = float((values[inside] * weights[inside]).sum())
        if group_time <= 0:
            continue                   # в группе все времена нулевые: доля ноль
        share = group_time / total_time
        group_mean = group_time / float(weights[inside].sum())
        between += share * np.log(group_mean / mean)
        within += share * theil(values[inside], weights[inside])
    return {"total": theil(values, weights), "between": float(between),
            "within": float(within)}
