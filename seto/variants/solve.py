"""Варианты размещения A, B, M и решатель (п. 10).

При k ≤ 2 перебираются ВСЕ допустимые пары, поэтому B, C, D и M получают
точный оптимум и результат не зависит от решателя (п. 10.3). Это важно для
вывода: разница между вариантами тогда объясняется весами, а не качеством
эвристики.

Ничьи разрешаются детерминированным правилом (п. 10.2): среди пар с равным
значением цели берётся лексикографически меньшая пара индексов площадок.
У B, C, D цель — выигрыш во взвешенном времени, у M — население в зоне T;
времена при ничьей M НЕ сравниваются. Без правила два прогона могли бы
дать разные точки при одинаковой метрике — на сухом прогоне ничьи встретились
сразу же.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..sites.feasible import validate


@dataclass
class Solution:
    """Выбранный набор и то, чем он получен."""
    sites: tuple[int, ...]        # позиции в списке допустимых площадок
    objective: float
    variant: str
    evaluated: int                # сколько наборов рассмотрено
    tie_count: int = 0            # сколько наборов дали тот же оптимум
    spread: dict | None = None    # только у эвристики: разброс между стартами

    def as_dict(self) -> dict:
        result = {"variant": self.variant, "sites": list(self.sites),
                  "objective": self.objective, "evaluated": self.evaluated,
                  "tie_count": self.tie_count}
        if self.spread is not None:
            result["spread"] = self.spread
        return result


def _improvement(site_times: np.ndarray, base_times: np.ndarray) -> np.ndarray:
    """Насколько площадка сокращает время в каждой клетке, но не меньше нуля.

    min(base, t) = base − max(0, base − t). В таком виде задача раскладывается
    на «вклад площадки», и перебор пар сводится к поэлементному максимуму."""
    return np.maximum(0.0, base_times[None, :] - site_times)


# Перебор пар держит в памяти массив (чанк x площадки x клетки). На городе это
# 1497 площадок и 3958 клеток, то есть 47 МБ на каждую строку чанка: без лимита
# размер улетает в гигабайты.
_CHUNK_BYTES = 256 << 20


def chunk_for(count: int, cells: int) -> int:
    """Сколько строк брать за раз, чтобы уложиться в бюджет памяти."""
    per_row = max(1, count * cells * 8)
    return max(1, int(_CHUNK_BYTES / per_row))


def _best_pair(score_pair, count: int, allowed: np.ndarray,
               chunk: int = 64) -> tuple[tuple[int, int], float, int, int]:
    """Перебор всех допустимых пар с детерминированным разрешением ничьих.

    score_pair(a_rows, b_all) возвращает матрицу значений цели: чем больше,
    тем лучше. Пары, нарушающие разнесение, отбрасываются ДО расчёта."""
    best_value = -np.inf
    best_pair = None
    evaluated = 0
    ties = 0

    for start in range(0, count, chunk):
        rows = np.arange(start, min(start + chunk, count))
        values = score_pair(rows)                       # (len(rows), count)
        mask = allowed[rows]
        # верхний треугольник: пара (a, b) рассматривается один раз
        mask &= np.arange(count)[None, :] > rows[:, None]
        if not mask.any():
            continue

        evaluated += int(mask.sum())
        candidates = np.where(mask, values, -np.inf)
        local = candidates.max()
        if local < best_value:
            continue
        hits = np.argwhere(candidates == local)
        ties += len(hits) if local == best_value else 0
        if local > best_value:
            best_value = local
            ties = len(hits) - 1
            first = hits[0]
            best_pair = (int(rows[first[0]]), int(first[1]))
        else:
            first = hits[0]
            candidate = (int(rows[first[0]]), int(first[1]))
            best_pair = min(best_pair, candidate)       # меньший индекс
    if best_pair is None:
        raise ValueError("нет ни одной допустимой пары площадок")
    return best_pair, float(best_value), evaluated, ties


def _exhaustive_only(k: int) -> None:
    """Полный перебор умеет только k ≤ 2 — иначе ошибка, а не тихая пара."""
    if k > 2:
        raise ValueError(f"полный перебор считается при k ≤ 2, а не {k}: "
                         "для больших k есть solve_*_any")


def solve_pmedian(site_times: np.ndarray, base_times: np.ndarray,
                  weights: np.ndarray, allowed: np.ndarray, k: int,
                  variant: str = "B") -> Solution:
    """S* = argmin Σ w_i · t_i(S). Веса задают вариант: B, C или D (п. 10.1).

    Полный перебор — только при k ≤ 2: дальше он перебирал бы пары и молча
    вернул две площадки вместо k. При больших k — solve_pmedian_any."""
    _exhaustive_only(k)
    gain = _improvement(site_times, base_times) * weights[None, :]

    if k == 1:
        totals = gain.sum(axis=1)
        best = int(np.argmax(totals))
        ties = int((totals == totals[best]).sum()) - 1
        return Solution((best,), float(totals[best]), variant, len(totals), ties)

    def score(rows: np.ndarray) -> np.ndarray:
        # выигрыш пары — поэлементный максимум вкладов, а не их сумма
        return np.maximum(gain[rows][:, None, :], gain[None, :, :]).sum(axis=2)

    pair, value, evaluated, ties = _best_pair(
        score, len(site_times), allowed,
        chunk=chunk_for(len(site_times), site_times.shape[1]))
    return Solution(tuple(sorted(pair)), float(value), variant, evaluated, ties)


# --- k = 5 и 10: эвристика обмена (пропозал v2, п. 3.4) ---
# Полный перебор при k > 2 невозможен (434 площадки — 1,3·10^10 пятёрок), а
# ILP с ограничением на каждую неразнесённую пару на городе не укладывается.
# Пропозал закладывает эвристику обмена с нескольких случайных стартов — она
# не гарантирует оптимум, поэтому разброс между стартами всегда выводится, а
# при k = 2 её ответ сверяется с полным перебором (тест).

def _random_start(allowed: np.ndarray, k: int, rng: np.random.Generator,
                  attempts: int = 1000) -> list[int]:
    """Случайный допустимый набор: площадки в случайном порядке, берём каждую,
    что разнесена со всеми уже взятыми."""
    for _ in range(attempts):
        chosen: list[int] = []
        for candidate in rng.permutation(len(allowed)):
            if all(allowed[candidate, other] for other in chosen):
                chosen.append(int(candidate))
                if len(chosen) == k:
                    return chosen
    raise ValueError(f"не удалось набрать {k} разнесённых площадок "
                     f"за {attempts} попыток")


def _swap_search(gain: np.ndarray, allowed: np.ndarray, start: list[int],
                 max_iterations: int) -> tuple[list[int], float, int]:
    """Обмен Тейтца — Барта до локального оптимума.

    На каждом шаге перебираются ВСЕ замены «площадку из набора на площадку вне
    его», допустимые по разнесению, и делается лучшая. Ценность набора — та
    же, что у полного перебора: Σ_i max_{j∈S} gain[j, i]."""
    chosen = list(start)
    value = float(gain[chosen].max(axis=0).sum())
    iterations = 0
    while iterations < max_iterations:
        iterations += 1
        best_value, best_move = value, None
        for position, leaving in enumerate(chosen):
            rest = chosen[:position] + chosen[position + 1:]
            # лучший выигрыш клетки без уходящей площадки
            kept = gain[rest].max(axis=0) if rest else np.zeros(gain.shape[1])
            values = np.maximum(gain, kept[None, :]).sum(axis=1)
            feasible = (allowed[:, rest].all(axis=1) if rest
                        else np.ones(len(gain), bool))
            feasible[chosen] = False
            if not feasible.any():
                continue
            values = np.where(feasible, values, -np.inf)
            entering = int(np.argmax(values))          # при ничьей — меньший индекс
            if values[entering] > best_value * (1 + 1e-12) + 1e-12:
                best_value, best_move = float(values[entering]), (position, entering)
        if best_move is None:
            break
        chosen[best_move[0]] = best_move[1]
        value = best_value
    return sorted(chosen), value, iterations


def solve_pmedian_swap(site_times: np.ndarray, base_times: np.ndarray,
                       weights: np.ndarray, allowed: np.ndarray, k: int,
                       variant: str = "B", starts: int = 20, seed: int = 0,
                       max_iterations: int = 1000) -> Solution:
    """P-median при любом k эвристикой обмена с `starts` случайных стартов.

    Цель та же, что у solve_pmedian: argmin Σ w_i·t_i(S). Из стартов берётся
    лучший набор; при равной цели — лексикографически меньший (как у
    перебора). В spread — цели всех стартов: если они сильно расходятся,
    стартов мало или задача рельефная, и это видно в отчёте."""
    gain = _improvement(site_times, base_times) * weights[None, :]
    return _swap_multistart(gain, allowed, k, variant, starts, seed, max_iterations)


def solve_max_coverage_swap(site_times: np.ndarray, base_times: np.ndarray,
                            population: np.ndarray, allowed: np.ndarray, k: int,
                            threshold: float, starts: int = 20, seed: int = 0,
                            max_iterations: int = 1000) -> Solution:
    """Максимальное покрытие при любом k той же эвристикой обмена.

    Покрытие устроено так же, как выигрыш p-median: клетка считается один раз,
    за лучшую площадку набора, Σ_i max_{j∈S} P_i·new_ij. Поэтому поиск общий."""
    gain = _newly_covered(site_times, base_times, threshold) * population[None, :]
    return _swap_multistart(gain, allowed, k, "M", starts, seed, max_iterations)


def _swap_multistart(gain: np.ndarray, allowed: np.ndarray, k: int, variant: str,
                     starts: int, seed: int, max_iterations: int) -> Solution:
    """Обмен с нескольких случайных стартов: лучший набор и разброс целей."""
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(starts):
        sites, value, iterations = _swap_search(
            gain, allowed, _random_start(allowed, k, rng), max_iterations)
        if iterations >= max_iterations:
            raise ValueError(f"обмен не сошёлся за {max_iterations} шагов")
        results.append((value, tuple(sites)))

    best_value = max(value for value, _ in results)
    # ничья между стартами — с той же относительной точностью, что и шаг обмена
    winners = sorted(sites for value, sites in results
                     if value >= best_value * (1 - 1e-12) - 1e-12)
    best = winners[0]
    validate(best, allowed, expected_k=k)
    objectives = np.array([value for value, _ in results])
    spread = {"starts": starts, "seed": seed,
              "best": float(best_value), "worst": float(objectives.min()),
              "median": float(np.median(objectives)),
              "reached_best": len(winners),
              "distinct_solutions": len({sites for _, sites in results})}
    return Solution(best, float(best_value), variant, evaluated=-1,
                    tie_count=len(set(winners)) - 1, spread=spread)


def solve_pmedian_any(site_times: np.ndarray, base_times: np.ndarray,
                      weights: np.ndarray, allowed: np.ndarray, k: int,
                      variant: str, cfg: dict) -> Solution:
    """P-median при любом k: при k ≤ 2 — полный перебор (точный оптимум),
    иначе — эвристика обмена с параметрами из `swap_heuristic` конфига.

    Одна точка входа для B, BN, C, D: варианты обязаны решаться одним и тем
    же решателем, иначе разница между ними мерила бы и разницу решателей."""
    if k <= 2:
        return solve_pmedian(site_times, base_times, weights, allowed, k, variant)
    return solve_pmedian_swap(site_times, base_times, weights, allowed, k, variant,
                              **_swap_parameters(cfg))


def solve_max_coverage_any(site_times: np.ndarray, base_times: np.ndarray,
                           population: np.ndarray, allowed: np.ndarray, k: int,
                           threshold: float, cfg: dict) -> Solution:
    """Максимальное покрытие при любом k: k ≤ 2 — перебор, иначе обмен."""
    if k <= 2:
        return solve_max_coverage(site_times, base_times, population, allowed, k,
                                  threshold)
    return solve_max_coverage_swap(site_times, base_times, population, allowed, k,
                                   threshold, **_swap_parameters(cfg))


def _swap_parameters(cfg: dict) -> dict:
    """Параметры эвристики из конфига — одни на все варианты."""
    params = cfg.get("swap_heuristic") or {}
    return {"starts": params.get("starts", 20), "seed": cfg.get("seed", 0),
            "max_iterations": params.get("max_iterations", 1000)}


def _newly_covered(site_times: np.ndarray, base_times: np.ndarray,
                   threshold: float) -> np.ndarray:
    """Клетки, которые площадка покрывает ДОПОЛНИТЕЛЬНО к существующей сети."""
    return (site_times <= threshold) & ~(base_times <= threshold)[None, :]


def solve_max_coverage(site_times: np.ndarray, base_times: np.ndarray,
                       population: np.ndarray, allowed: np.ndarray, k: int,
                       threshold: float) -> Solution:
    """S* = argmax Σ P_i · 1[t_i(S) ≤ T] (п. 10.2).

    M прямо оптимизирует оцениваемую метрику, поэтому служит верхней границей
    прироста покрытия, а не рядовым конкурентом. Перебор — только при k ≤ 2."""
    _exhaustive_only(k)
    new = _newly_covered(site_times, base_times, threshold)
    weighted = new * population[None, :]

    if k == 1:
        totals = weighted.sum(axis=1)
        best = int(np.argmax(totals))
        ties = int((totals == totals[best]).sum()) - 1
        return Solution((best,), float(totals[best]), "M", len(totals), ties)

    def score(rows: np.ndarray) -> np.ndarray:
        # объединение покрытий: сумма минус пересечение
        union = (new[rows][:, None, :] | new[None, :, :])
        return (union * population[None, None, :]).sum(axis=2)

    pair, value, evaluated, ties = _best_pair(
        score, len(site_times), allowed,
        chunk=chunk_for(len(site_times), site_times.shape[1]))
    return Solution(tuple(sorted(pair)), float(value), "M", evaluated, ties)


def random_sets(allowed: np.ndarray, k: int, draws: int, seed: int,
                max_attempts_factor: int = 200) -> list[tuple[int, ...]]:
    """Случайные ДОПУСТИМЫЕ наборы для варианта A (п. 10.4).

    Выборка с отказом; если за отведённые попытки не набралось нужного числа,
    перечисляются все допустимые наборы и берутся из них — молча вернуть
    меньше наборов нельзя, это исказило бы распределение."""
    rng = np.random.default_rng(seed)
    count = len(allowed)
    if k == 1:
        picks = rng.choice(count, size=min(draws, count), replace=False)
        return [(int(p),) for p in picks]
    if k > 2:
        return _random_large_sets(allowed, k, draws, rng, max_attempts_factor)

    found: set[tuple[int, ...]] = set()
    for _ in range(draws * max_attempts_factor):
        if len(found) >= draws:
            break
        a, b = rng.choice(count, size=2, replace=False)
        if allowed[a, b]:
            found.add(tuple(sorted((int(a), int(b)))))

    if len(found) < draws:
        every = [(int(a), int(b)) for a, b in zip(*np.triu_indices(count, k=1))
                 if allowed[a, b]]
        if len(every) <= draws:
            return sorted(every)
        picks = rng.choice(len(every), size=draws, replace=False)
        return sorted(every[int(i)] for i in picks)
    return sorted(found)


def _random_large_sets(allowed: np.ndarray, k: int, draws: int,
                       rng: np.random.Generator,
                       max_attempts_factor: int) -> list[tuple[int, ...]]:
    """Случайные допустимые наборы при k > 2: выборка с отказом.

    Запасного перечисления, как у пар, здесь нет: наборов слишком много.
    Поэтому недобор — ошибка, а не укороченное распределение."""
    found: set[tuple[int, ...]] = set()
    for _ in range(draws * max_attempts_factor):
        if len(found) >= draws:
            break
        picks = rng.choice(len(allowed), size=k, replace=False)
        # диагональ матрицы ложна, поэтому допустимый набор даёт ровно k(k−1)
        if allowed[np.ix_(picks, picks)].sum() == k * (k - 1):
            found.add(tuple(sorted(int(p) for p in picks)))
    if len(found) < draws:
        raise ValueError(f"набрано {len(found)} случайных допустимых наборов "
                         f"размера {k} из {draws}")
    return sorted(found)


def solve_random(allowed: np.ndarray, k: int, draws: int, seed: int) -> list:
    """Набор решений варианта A: каждое проходит общий валидатор."""
    sets = random_sets(allowed, k, draws, seed)
    for selection in sets:
        validate(selection, allowed, expected_k=k)
    return sets


def estimate_pairs(count: int, allowed: np.ndarray) -> dict:
    """Оценка объёма перебора ДО запуска (п. 10.3)."""
    total = count * (count - 1) // 2
    return {"sites": int(count), "pairs_total": int(total),
            "pairs_allowed": int(np.triu(allowed, k=1).sum())}


# --- k > 2: целочисленная оптимизация ---

def solve_ilp(site_times: np.ndarray, base_times: np.ndarray,
              weights: np.ndarray, allowed: np.ndarray, k: int,
              variant: str = "B", threshold: float | None = None,
              time_limit_s: int = 300) -> Solution:
    """Решение при k > 2 через ILP (п. 10.3).

    Полный перебор растёт как C(|J|, k) и при k > 2 становится неподъёмным:
    на 1497 площадках троек уже полмиллиарда. Формулировка классическая —
    переменные открытия и назначения.

    threshold задан → максимальное покрытие, иначе p-median с весами."""
    import pulp

    sites, cells = site_times.shape
    problem = pulp.LpProblem("seto", pulp.LpMinimize if threshold is None
                             else pulp.LpMaximize)
    opened = [pulp.LpVariable(f"y_{j}", cat="Binary") for j in range(sites)]
    problem += pulp.lpSum(opened) == k

    # разнесение площадок: пара, нарушающая ограничение, не открывается вместе
    for a in range(sites):
        for b in range(a + 1, sites):
            if not allowed[a, b]:
                problem += opened[a] + opened[b] <= 1

    if threshold is None:
        gain = _improvement(site_times, base_times) * weights[None, :]
        # Классические переменные назначения (п. 10.3): клетка приписывается
        # не более чем одной ОТКРЫТОЙ площадке. Через «большое M» эту задачу
        # записать нельзя: закрытая площадка тогда не связывает выигрыш.
        # Переменные заводятся только там, где выигрыш положителен — иначе
        # на реальном городе их были бы миллионы.
        assign = {}
        for j in range(sites):
            for i in np.flatnonzero(gain[j] > 0):
                assign[(j, int(i))] = pulp.LpVariable(f"x_{j}_{i}", lowBound=0,
                                                      upBound=1)
        for (j, i), variable in assign.items():
            problem += variable <= opened[j]
        for i in range(cells):
            linked = [assign[(j, i)] for j in range(sites) if (j, i) in assign]
            if linked:
                problem += pulp.lpSum(linked) <= 1
        problem += -pulp.lpSum(gain[j, i] * variable
                               for (j, i), variable in assign.items())
    else:
        new = (site_times <= threshold) & ~(base_times <= threshold)[None, :]
        covered = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(cells)]
        for i in range(cells):
            reachable = np.flatnonzero(new[:, i])
            if reachable.size == 0:
                problem += covered[i] == 0
            else:
                problem += covered[i] <= pulp.lpSum(opened[j] for j in reachable)
        problem += pulp.lpSum(weights[i] * covered[i] for i in range(cells))

    problem.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_s))
    chosen = tuple(j for j in range(sites) if opened[j].value() and opened[j].value() > 0.5)
    if len(chosen) != k:
        raise ValueError(f"решатель вернул {len(chosen)} площадок вместо {k}: "
                         f"статус {pulp.LpStatus[problem.status]}")
    return Solution(chosen, float(pulp.value(problem.objective) or 0.0),
                    variant, evaluated=-1, tie_count=0)
