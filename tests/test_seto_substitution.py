"""Тесты М2: ошибка замещения, доля захвата, порог шума (задачи 4.3–4.4 TODO_V2)."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seto.metrics import substitution as M2  # noqa: E402


def city():
    """Четыре клетки: люди живут в 0 и 1, пожилые — в мелких 2 и 3."""
    allowed = np.ones((3, 3), dtype=bool)
    np.fill_diagonal(allowed, False)
    return SimpleNamespace(
        population=np.array([1000.0, 800.0, 50.0, 10.0]),
        base_times=np.full(4, 40.0),
        site_times=np.array([[5.0, 40, 40, 40],      # площадка 0 — клетке 0
                             [40.0, 5, 40, 40],      # площадка 1 — клетке 1
                             [40.0, 40, 5, 5]]),     # площадка 2 — клеткам 2 и 3
        allowed_pairs=allowed)


def cfg(draws=5, amplitude=0.1):
    return {"t_max_minutes": 60, "seed": 0,
            "swap_heuristic": {"starts": 3, "max_iterations": 50},
            "m2": {"k_values": [1], "noise_draws": draws,
                   "noise_amplitude": amplitude, "noise_percentile": 95}}


def test_need_time_sums_needy_minutes():
    data = city()
    need = np.array([1.0, 0, 2.0, 0])
    # площадка 0: клетка 0 → 5 мин, клетка 2 остаётся 40
    assert M2.need_time(data.site_times, data.base_times, (0,), need, 60) == 5 + 80


def test_capture_is_share_of_ceiling_and_hidden_under_noise():
    assert M2.capture(z_b=100, z_x=40, ceiling=80, noise=10) == pytest.approx(75)
    assert M2.capture(z_b=100, z_x=40, ceiling=80, noise=80) is None   # не выше шума
    assert M2.capture(z_b=100, z_x=100, ceiling=0, noise=0) is None


def test_ceiling_appears_when_need_lives_elsewhere():
    """Нуждающиеся в клетках 2–3: B ставит точку в 0, BN — в 2."""
    data = city()
    need = np.array([10.0, 0, 500, 500])
    table, noise_sites = M2.study(data, {"age": need}, cfg(amplitude=0.0), [1])
    row = table.iloc[0]
    assert row.B_sites == [0] and row.BN_sites == [2]
    # B: клетки 2, 3 по 40 мин; BN: клетки 2, 3 по 5 мин, клетка 0 — 40
    assert row.ceiling == pytest.approx((10 * 5 + 1000 * 40) - (10 * 40 + 1000 * 5))
    assert row.noise_threshold == 0 and row.above_noise
    assert row.capture_B == 0 and row.capture_BN == pytest.approx(100)
    assert noise_sites[1] == [[0]] * 5          # без шума B′ = B


def test_need_like_population_gives_zero_ceiling_and_no_capture():
    data = city()
    table, _ = M2.study(data, {"age": data.population.copy()}, cfg(), [1, 2])
    assert table["same_sites"].all()
    assert (table["ceiling"] == 0).all()
    assert table["capture_BN"].isna().all() and not table["above_noise"].any()


def test_noise_moves_b_prime_and_sets_threshold():
    """Клетки 0 и 1 почти равны по весу — шум ±10 % меняет выбор B′."""
    data = city()
    data.population = np.array([1000.0, 990.0, 50.0, 10.0])
    need = np.array([1.0, 1.0, 0, 0])
    table, noise_sites = M2.study(data, {"age": need}, cfg(draws=40), [1])
    assert {tuple(s) for s in noise_sites[1]} == {(0,), (1,)}
    assert table.iloc[0].B_prime_distinct == 2
    assert table.iloc[0].noise_threshold == 0   # для этих N клетки 0 и 1 равноценны


def test_every_need_definition_gets_its_own_row():
    data = city()
    needs = {"age": np.array([10.0, 0, 500, 500]), "age_r1": data.population.copy()}
    table, _ = M2.study(data, needs, cfg(amplitude=0.0), [1, 2])
    assert sorted(zip(table["need"], table["k"])) == [
        ("age", 1), ("age", 2), ("age_r1", 1), ("age_r1", 2)]
