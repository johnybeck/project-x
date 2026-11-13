"""GIS-AHP: реконструкция Dutta et al. (2022), варианты E, EN, EP (пропозал v2, п. 3.6).

Из статьи взяты только матрица парных сравнений, её веса и критерии. Порогов классов
и правила выбора точек в статье нет: это наши решения, они записаны в конфиге (`ahp`).
Трущоб и бедности в открытых данных нет: у E их вес перенормируется, у EN переходит
к критерию нуждаемости.

Критерии считаются для ВСЕХ клеток, а границы классов — по допустимым площадкам. E и
EN смотрят только на площадки, но EP кладёт пригодность в вес спроса, а спрос есть и
там, где площадку открыть нельзя.
"""
from __future__ import annotations

import numpy as np

# индекс случайной согласованности Саати по размеру матрицы
_RANDOM_INDEX = {3: 0.58, 4: 0.90, 5: 1.12}
# баллы — суммы дробных весов: без округления ничья рассыпалась бы на шуме в 16-м знаке
_SCORE_DIGITS = 9


def pairwise_matrix(upper: list[list[float]]) -> np.ndarray:
    """Полная матрица из верхнего треугольника: a_ji = 1 / a_ij."""
    size = len(upper) + 1
    matrix = np.eye(size)
    for row, values in enumerate(upper):
        for column, value in enumerate(values, start=row + 1):
            matrix[row, column], matrix[column, row] = value, 1 / value
    return matrix


def eigen_weights(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """Веса — главный собственный вектор матрицы, и индекс согласованности CR."""
    values, vectors = np.linalg.eig(matrix)
    main = int(np.argmax(values.real))
    weights = np.abs(vectors[:, main].real)
    size = len(matrix)
    consistency = (values[main].real - size) / (size - 1)
    return weights / weights.sum(), float(consistency / _RANDOM_INDEX[size])


def kept_weights(rules: dict, method: str = "published") -> dict[str, float]:
    """Веса E: критерии без исключённых, перенормированные к единице.

    method — откуда веса: таблица 6 статьи (`published`), собственный вектор всей
    матрицы (`eigenvector`) или подматрицы оставшихся критериев (`submatrix`).
    Подматрица несогласованна, поэтому основной способ — опубликованные веса."""
    names = rules["criteria"]
    kept = [position for position, name in enumerate(names)
            if name not in rules["excluded"]]
    matrix = pairwise_matrix(rules["matrix"])
    if method == "published":
        values = np.asarray(rules["published_weights"], dtype=float)[kept]
    elif method == "eigenvector":
        values = eigen_weights(matrix)[0][kept]
    elif method == "submatrix":
        values = eigen_weights(matrix[np.ix_(kept, kept)])[0]
    else:
        raise ValueError(f"неизвестный способ получить веса: {method!r}")
    return {names[position]: float(value)
            for position, value in zip(kept, values / values.sum())}


def need_weights(rules: dict) -> dict[str, float]:
    """Веса EN: опубликованные без перенормировки, вес исключённых — нуждаемости."""
    published = dict(zip(rules["criteria"], rules["published_weights"]))
    weights = {name: float(value) for name, value in published.items()
               if name not in rules["excluded"]}
    weights["need"] = float(sum(published[name] for name in rules["excluded"]))
    return weights


def dense_cells(population: np.ndarray, share: float) -> np.ndarray:
    """«Районы высокой плотности»: верхняя доля клеток среди клеток с жителями."""
    people = np.asarray(population, dtype=float)
    inhabited = people[people > 0]
    if inhabited.size == 0:
        return np.zeros(len(people), dtype=bool)
    return people >= np.quantile(inhabited, 1 - share)


def share_name(share: float) -> str:
    """Имя порога плотности в файлах и таблицах: 0.1 → «0.10»."""
    return f"{float(share):.2f}"


_OPTION_PREFIX = {"roads": "to_roads__", "dense": "to_dense__"}


def option_key(criterion: str, name: str) -> str:
    """Под каким ключом слой при другом пороге лежит в times.npz."""
    return _OPTION_PREFIX[criterion] + name


def read_options(times) -> dict[str, dict[str, np.ndarray]] | None:
    """Слои при других порогах из times.npz; None, если подготовка города их не считала."""
    found = {criterion: {key[len(prefix):]: times[key] for key in times.files
                         if key.startswith(prefix)}
             for criterion, prefix in _OPTION_PREFIX.items()}
    return found if any(found.values()) else None


def need_share(data, cfg: dict) -> np.ndarray | None:
    """Доля пожилых в клетке, без жителей — 0; None, если возрастных групп нет."""
    groups = getattr(data, "groups", None)
    city = cfg.get("city") or {}
    if groups is None or "need" not in city:
        return None
    people = np.asarray(data.population, dtype=float)
    seniors = groups[city["need"]["seniors_group"]].to_numpy(dtype=float)
    return np.divide(seniors, people, out=np.zeros_like(people), where=people > 0)


def layers(data, cfg: dict) -> dict[str, np.ndarray] | None:
    """Критерии всех клеток: чем значение больше, тем клетка лучше.

    Дальше от сети — лучше, ближе к дорогам и плотным районам — лучше. None, если в
    подготовке города нет времён до дорог и плотных районов."""
    roads = getattr(data, "road_times", None)
    dense = getattr(data, "dense_times", None)
    if roads is None or dense is None:
        return None
    result = {"network": np.asarray(data.base_times, dtype=float),
              "roads": -np.asarray(roads, dtype=float),
              "dense": -np.asarray(dense, dtype=float)}
    share = need_share(data, cfg)
    if share is not None:
        result["need"] = share
    return result


def classes(values: np.ndarray, reference: np.ndarray, count: int) -> np.ndarray:
    """Класс 1…count, границы — квантили по клеткам `reference`.

    Равные значения идут в лучший класс: иначе клетки внутри плотного района
    (расстояние 0, их больше четверти) не попали бы в верхний."""
    edges = np.quantile(values[reference], np.arange(1, count) / count)
    return 1 + (values[:, None] >= edges[None, :]).sum(axis=1)


def scaled(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Непрерывная шкала: 0…1 по клеткам `reference`, вне диапазона — обрезка."""
    low, high = float(values[reference].min()), float(values[reference].max())
    if high <= low:
        return np.zeros(len(values), dtype=float)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def graded(criteria: dict[str, np.ndarray], reference: np.ndarray,
           scheme: int | str) -> dict[str, np.ndarray]:
    """Критерии в баллах. scheme — число классов или "continuous"."""
    if scheme == "continuous":
        return {name: scaled(values, reference) for name, values in criteria.items()}
    return {name: classes(values, reference, int(scheme))
            for name, values in criteria.items()}


def combine(grades: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """Пригодность клетки: взвешенная сумма баллов."""
    return sum(weight * grades[name] for name, weight in weights.items())


def population_within(site_times: np.ndarray, population: np.ndarray,
                      threshold: float) -> np.ndarray:
    """Сколько жителей в радиусе T от каждой площадки: правило ничьих E."""
    return (site_times <= threshold).astype(float) @ np.asarray(population, dtype=float)


def top_cells(score: np.ndarray) -> int:
    """Сколько площадок делят верхний балл."""
    rounded = np.round(score, _SCORE_DIGITS)
    return int((rounded == rounded.max()).sum())


def choose(score: np.ndarray, ties: np.ndarray, allowed: np.ndarray,
           k: int) -> tuple[int, ...]:
    """Площадки сверху вниз по баллу, ничьи — по `ties`, затем по порядку в списке.

    Площадка, нарушающая разнесение с уже взятыми, пропускается. Порядок в ответе —
    порядок выбора. Перекрытие зон E не учитывает: так работает ранжирование AHP, и
    разница с оптимизатором входит в D − E (пропозал v2, п. 3.6)."""
    order = np.lexsort((-np.asarray(ties, dtype=float), -np.round(score, _SCORE_DIGITS)))
    chosen: list[int] = []
    for candidate in order:
        if all(allowed[candidate, other] for other in chosen):
            chosen.append(int(candidate))
            if len(chosen) == k:
                return tuple(chosen)
    raise ValueError(f"разнесённых площадок нашлось {len(chosen)}, а нужно {k}")
