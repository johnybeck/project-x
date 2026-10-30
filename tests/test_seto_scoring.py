"""Тесты п. 7 и 9: промпты, разбор ответа, кэш, веса спроса."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seto import settings  # noqa: E402
from seto.data import features as F  # noqa: E402
from seto.experiment.runner import sample_cells  # noqa: E402
from seto.scoring import client as CL, prompt as P, rules as R, weights as W  # noqa: E402

GLOBAL = settings.load_global()
RULE = GLOBAL["score_rules"][GLOBAL["score_rule"]]


def _answer(**overrides) -> dict:
    """Правильный ответ модели; тесты меняют в нём только нужное поле."""
    base = {"housing_type": "многоквартирные кварталы", "epoch": "1955-1975",
            "housing_share": "больше половины", "construction": "нет",
            "purpose": "жилое", "affluence": "средний",
            "rationale": "плотная застройка"}
    base.update(overrides)
    return base


def _scoring_cfg(retries: int = 1) -> dict:
    return {"vlm": {"model": "m", "temperature": 0, "max_tokens": 10,
                    "max_parallel": 1, "retries": retries,
                    "base_url_env": "NO_URL", "api_key_env": "NO_KEY"},
            "grid": {"size_m": 500},
            "score_rule": GLOBAL["score_rule"],
            "score_rules": GLOBAL["score_rules"]}


# --- промпты ---

def test_prompts_differ_only_in_input_clause():
    """Иначе разница D − C мерила бы формулировку, а не изображение (п. 7.1)."""
    image = P.build("image", 500)
    text = P.build("text", 500, "описание")
    common = P.TEMPLATE.split("{input_clause}")[1].format(size="500x500")
    assert common in image and common in text
    assert P.IMAGE_CLAUSE in image and P.IMAGE_CLAUSE not in text
    assert "описание" in text


def test_prompt_does_not_ask_about_need():
    """Модель отмечает видимое; нуждаемость получается по правилу (п. 3.5)."""
    text = P.build("text", 500, "x").lower()
    for word in ("поликлиник", "нужд", "медицин", "score"):
        assert word not in text


def test_prompt_lists_every_allowed_answer():
    """Разбор ответа строгий: метка, которой нет в промпте, — это сбой клетки."""
    text = P.build("image", 500)
    for name, labels in P.RATINGS.items():
        assert f'"{name}"' in text
        for label in labels:
            assert f'"{label}"' in text


def test_prompt_lets_the_model_decline():
    """Эпоху по снимку можно не различить: угадывание хуже, чем отказ."""
    assert all(P.UNKNOWN in labels for labels in P.RATINGS.values())


def test_prompt_forbids_counting():
    assert "не вычисляй" in P.build("image", 500).lower()


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        P.build("audio", 500)


def test_text_mode_requires_text():
    with pytest.raises(ValueError):
        P.build("text", 500)


def test_prompt_hash_is_stable():
    assert P.prompt_hash(P.build("image", 500)) == P.prompt_hash(P.build("image", 500))
    assert P.prompt_hash(P.build("image", 500)) != P.prompt_hash(P.build("text", 500, "a"))


# --- описание клетки ---

def _row(**overrides):
    base = {"built_share": 0.3, "residential_share": 0.2, "public_share": 0.1,
            "buildings_count": 42, "mean_levels": 9.0, "levels_known_share": 0.8,
            "schools": 2, "kindergartens": 3, "pharmacies": 4, "shops": 5,
            "transport_stops": 6, "green_share": 0.05, "water_share": 0.0,
            "landuse_industrial_share": 0.0, "street_density_m_per_km2": 12000}
    base.update(overrides)
    return pd.Series(base)


def test_description_has_no_population():
    """Население уже в весе w_i, и модель не должна считать числа."""
    text = P.describe(_row()).lower()
    assert "населен" not in text and "жител" not in text


def test_missing_values_say_so_explicitly():
    """Пропуск — словом, не нулём: ноль школ и неизвестно сколько — разное."""
    text = P.describe(_row(mean_levels=np.nan, schools=None))
    assert text.count("нет данных") >= 2
    assert "Школы: нет данных" in text


def test_description_is_deterministic():
    assert P.describe(_row()) == P.describe(_row())


_EPOCH_ROW = {"epoch_before_1955": 1, "epoch_1955_1975": 2, "epoch_1975_2000": 0,
              "epoch_after_2000": 0, "epoch_unknown": 1}


def test_description_has_epochs_in_prompt_classes():
    """Те же классы, что модель выбирает в ответе: иначе эпоху пришлось бы вычислять."""
    text = P.describe(_row(**_EPOCH_ROW))
    assert ("Жилые здания по году постройки: до 1955 — 25 %, 1955-1975 — 50 %, "
            "1975-2000 — 0 %, после 2000 — 0 %, год неизвестен — 25 %.") in text


def test_epochs_match_prompt_classes():
    labels = [label for _, label, _ in F.EPOCHS]
    assert labels == [r for r in P.RATINGS["epoch"] if r != P.UNKNOWN]


@pytest.mark.parametrize("overrides", [
    {},                                   # у города нет источника года
    {column: np.nan for column in _EPOCH_ROW},
    {column: 0 for column in _EPOCH_ROW},  # жилых зданий в кадастре нет
])
def test_epochs_missing_say_so(overrides):
    text = P.describe(_row(**overrides))
    assert "Жилые здания по году постройки: нет данных." in text


def test_description_names_chains():
    text = P.describe(_row(chain_shops=3, chain_brands="Dia — 2, Lidl — 1"))
    assert "магазины: 5, из них с брендом сети: 3 (Dia — 2, Lidl — 1)," in text


def test_zero_chains_have_no_empty_brackets():
    text = P.describe(_row(chain_shops=0, chain_brands=""))
    assert "из них с брендом сети: 0," in text


def test_unknown_chains_say_so():
    assert "из них с брендом сети: нет данных," in P.describe(_row())


# --- разбор ответа ---

def test_parses_plain_json():
    assert CL.parse_response(json.dumps(_answer(), ensure_ascii=False)) == _answer()


def test_parses_markdown_wrapped():
    raw = f"```json\n{json.dumps(_answer(), ensure_ascii=False)}\n```"
    assert CL.parse_response(raw)["epoch"] == "1955-1975"


def test_dash_and_case_do_not_fail_a_cell():
    """Модели пишут «1955–1975» с длинным тире: это та же метка, а не сбой."""
    raw = json.dumps(_answer(epoch="1955–1975", affluence=" Средний "),
                     ensure_ascii=False)
    parsed = CL.parse_response(raw)
    assert parsed["epoch"] == "1955-1975" and parsed["affluence"] == "средний"


def test_empty_content_is_an_error_not_zero():
    """У reasoning-моделей content бывает пустым, когда рассуждение съело лимит."""
    for value in (None, "", "   "):
        with pytest.raises(CL.ScoreSchemaError, match="пустой"):
            CL.parse_response(value)


def test_label_outside_the_list_is_rejected():
    raw = json.dumps(_answer(epoch="1960-е"), ensure_ascii=False)
    with pytest.raises(CL.ScoreSchemaError, match="нет в списке"):
        CL.parse_response(raw)


def test_number_instead_of_label_is_rejected():
    raw = json.dumps(_answer(epoch=1960), ensure_ascii=False)
    with pytest.raises(CL.ScoreSchemaError, match="нет в списке"):
        CL.parse_response(raw)


def test_every_rating_is_required():
    for name in P.RATINGS:
        answer = _answer()
        del answer[name]
        with pytest.raises(CL.ScoreSchemaError, match=f"нет поля {name}"):
            CL.parse_response(json.dumps(answer, ensure_ascii=False))


def test_old_score_format_is_rejected():
    """Ответ по старой схеме не должен тихо пройти как подоценки."""
    with pytest.raises(CL.ScoreSchemaError, match="нет поля"):
        CL.parse_response('{"score": 7, "rationale": "что-то"}')


def test_broken_json_is_rejected():
    with pytest.raises(CL.ScoreSchemaError):
        CL.parse_response("вообще не json")


# --- кэш ---

def test_cache_key_depends_on_input():
    """Мина MVP: ключ не учитывал вход, и снимок подхватывал текстовый ответ."""
    prompt = P.build("image", 500)
    first = CL.cache_key("cell_000001", "m", prompt, "image", b"\x01\x02")
    second = CL.cache_key("cell_000001", "m", prompt, "image", b"\x03\x04")
    assert first != second


def test_cache_key_depends_on_cell_model_and_prompt():
    prompt = P.build("text", 500, "a")
    other = P.build("text", 500, "b")
    base = CL.cache_key("cell_1", "m1", prompt, "text", "a")
    assert base != CL.cache_key("cell_2", "m1", prompt, "text", "a")
    assert base != CL.cache_key("cell_1", "m2", prompt, "text", "a")
    assert base != CL.cache_key("cell_1", "m1", other, "text", "a")


def test_cached_cells_do_not_call_the_api(tmp_path):
    """Повторный запуск не ходит в API — он же даёт возобновление после обрыва."""
    cell_ids, texts = ["cell_000001"], ["описание"]
    prompt = P.build("text", 500, texts[0])
    key = CL.cache_key(cell_ids[0], "m", prompt, "text", texts[0])
    (tmp_path / f"{key}.json").write_text(
        json.dumps(_answer(epoch="до 1955"), ensure_ascii=False), encoding="utf-8")

    scores, report = CL.score_cells(cell_ids, texts, "text", _scoring_cfg(), tmp_path)
    assert scores["epoch"].iloc[0] == "до 1955"
    assert scores["score"].iloc[0] == RULE["points"]["до 1955"]
    assert report.from_cache == 1 and report.api_calls == 0


def test_every_rating_is_kept_for_m4(tmp_path):
    """М4 сверяет эпоху и достаток с эталоном, поэтому они нужны по клеткам."""
    prompt = P.build("text", 500, "x")
    key = CL.cache_key("a", "m", prompt, "text", "x")
    (tmp_path / f"{key}.json").write_text(
        json.dumps(_answer(), ensure_ascii=False), encoding="utf-8")

    scores, _ = CL.score_cells(["a"], ["x"], "text", _scoring_cfg(), tmp_path)
    assert set(P.RATINGS) <= set(scores.columns)


def test_old_prompt_cache_is_not_picked_up(tmp_path, monkeypatch):
    """Ответ старого промпта лежит под старым хешем и не должен найтись."""
    monkeypatch.setattr(CL, "_make_client", lambda _cfg: _BrokenClient())
    old_prompt = "Вопрос один: насколько нужна поликлиника?"
    key = CL.cache_key("a", "m", old_prompt, "text", "x")
    (tmp_path / f"{key}.json").write_text(
        json.dumps({"score": 7, "rationale": ""}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="ни одна клетка"):
        CL.score_cells(["a"], ["x"], "text", _scoring_cfg(retries=0), tmp_path)


def test_bad_rule_stops_before_any_model_call(tmp_path, monkeypatch):
    """Ошибка в правиле не должна стоить прогона модели."""
    def fail(_cfg):
        raise AssertionError("дошли до вызова модели")

    monkeypatch.setattr(CL, "_make_client", fail)
    cfg = _scoring_cfg()
    cfg["score_rules"] = {"age": {"rating": "housing_share", "points": {}}}
    with pytest.raises(ValueError, match="не может входить"):
        CL.score_cells(["a"], ["x"], "text", cfg, tmp_path)


class _BrokenClient:
    """Клиент, у которого любой вызов падает — так проверяется путь imputed."""

    class chat:                                     # noqa: N801
        class completions:                          # noqa: N801
            @staticmethod
            def create(**_kwargs):
                raise RuntimeError("модель недоступна")


def test_failed_cells_get_median_and_are_flagged(tmp_path, monkeypatch):
    """Сбойная клетка получает медиану города и флаг imputed (п. 7.1)."""
    monkeypatch.setattr(CL, "_make_client", lambda _cfg: _BrokenClient())
    ids = ["a", "b", "c"]
    for cell_id, epoch in zip(ids[:2], ("до 1955", "после 2000")):
        prompt = P.build("text", 500, cell_id)
        key = CL.cache_key(cell_id, "m", prompt, "text", cell_id)
        (tmp_path / f"{key}.json").write_text(
            json.dumps(_answer(epoch=epoch), ensure_ascii=False), encoding="utf-8")

    scores, report = CL.score_cells(ids, ids, "text", _scoring_cfg(retries=0), tmp_path)
    assert report.imputed == 1                      # третья клетка не в кэше
    assert scores.loc[scores.cell_id == "c", "score"].iloc[0] == 1.5
    assert bool(scores.loc[scores.cell_id == "c", "imputed"].iloc[0])
    assert scores.loc[scores.cell_id == "c", "epoch"].isna().all()
    assert report.failures and "модель недоступна" in report.failures[0]


def test_missing_credentials_fail_loudly(tmp_path, monkeypatch):
    """Отсутствие ключа — ошибка настройки, а не повод пометить всё imputed."""
    monkeypatch.delenv("NO_URL", raising=False)
    monkeypatch.delenv("NO_KEY", raising=False)
    with pytest.raises(RuntimeError, match="не заданы"):
        CL.score_cells(["a"], ["текст"], "text", _scoring_cfg(retries=0), tmp_path)


def test_too_many_imputed_fails_the_variant():
    report = CL.ScoringReport(cells=100, imputed=6)
    with pytest.raises(RuntimeError, match="неуспешен"):
        CL.check_imputed(report, {"imputation_max_share": 0.05})
    CL.check_imputed(CL.ScoringReport(cells=100, imputed=5),
                     {"imputation_max_share": 0.05})


# --- правило перевода подоценок в скор ---

def test_config_rule_is_valid():
    assert R.active(GLOBAL) is RULE


def test_older_buildings_score_higher():
    """Для N_возр скор растёт с возрастом застройки (п. 3.5)."""
    points = RULE["points"]
    assert (points["до 1955"] > points["1955-1975"] > points["1975-2000"]
            > points["после 2000"])


def test_unknown_epoch_sits_in_the_middle():
    """Неизвестная эпоха не поднимает клетку и не опускает её."""
    points = RULE["points"]
    assert points[P.UNKNOWN] == (points["до 1955"] + points["после 2000"]) / 2


def test_housing_share_cannot_enter_the_rule():
    """Доля жилья — это население, а оно уже в P_i (п. 3.5)."""
    rule = {"rating": "housing_share",
            "points": dict.fromkeys(P.RATINGS["housing_share"], 1)}
    with pytest.raises(ValueError, match="не может входить"):
        R.check(rule)


def test_rule_must_score_every_label():
    points = {label: 1 for label in P.RATINGS["epoch"] if label != "до 1955"}
    with pytest.raises(ValueError, match="до 1955"):
        R.check({"rating": "epoch", "points": points})


def test_rule_rejects_labels_the_model_never_returns():
    points = dict.fromkeys(P.RATINGS["epoch"], 1) | {"1960-е": 1}
    with pytest.raises(ValueError, match="1960-е"):
        R.check({"rating": "epoch", "points": points})


def test_unknown_rating_is_rejected():
    with pytest.raises(ValueError, match="нет такой подоценки"):
        R.check({"rating": "colour", "points": {}})


def test_score_maps_labels_and_leaves_failures_empty():
    ratings = pd.DataFrame({"epoch": ["до 1955", "после 2000", None]})
    result = R.score(ratings, RULE)
    assert result.iloc[:2].tolist() == [3.0, 0.0] and np.isnan(result.iloc[2])


def test_sample_takes_inhabited_cells_and_grows_from_the_first():
    """10 клеток включают ту одну, с которой начинали проверку."""
    features = pd.DataFrame({"cell_id": list("abcdef"),
                             "population": [0, 5, 0, 7, 9, 3]})
    one, three = sample_cells(features, 1, 0), sample_cells(features, 3, 0)
    assert one == three[:1]
    assert set(three) <= {"b", "d", "e", "f"} and len(set(three)) == 3


# --- веса спроса ---

def test_normalize_maps_to_unit_range():
    normalized, degenerate = W.normalize(np.array([2.0, 6.0, 10.0]))
    assert normalized.tolist() == [0.0, 0.5, 1.0]
    assert not degenerate


def test_identical_scores_are_flagged_and_give_zeros():
    """Модель не различила клетки — вариант сводится к B, и это помечается."""
    normalized, degenerate = W.normalize(np.array([7.0, 7.0, 7.0]))
    assert normalized.tolist() == [0.0, 0.0, 0.0]
    assert degenerate


def test_variant_b_weight_is_population():
    population = np.array([100.0, 200.0])
    assert W.demand(population).tolist() == [100.0, 200.0]


def test_score_multiplies_population():
    population = np.array([100.0, 100.0])
    weights = W.demand(population, np.array([0.0, 10.0]), alpha=1.0)
    assert weights.tolist() == [100.0, 200.0]       # ŝ = 0 и 1


def test_alpha_scales_the_effect():
    population = np.array([100.0, 100.0])
    half = W.demand(population, np.array([0.0, 10.0]), alpha=0.5)
    double = W.demand(population, np.array([0.0, 10.0]), alpha=2.0)
    assert half.tolist() == [100.0, 150.0]
    assert double.tolist() == [100.0, 300.0]


def test_empty_cell_has_zero_weight_whatever_the_score():
    """Скор считается для всех клеток (нужен для нормализации), но спроса нет."""
    weights = W.demand(np.array([0.0, 100.0]), np.array([10.0, 0.0]), alpha=1.0)
    assert weights[0] == 0.0


def test_negative_alpha_is_rejected():
    with pytest.raises(ValueError):
        W.demand(np.array([1.0]), np.array([1.0]), alpha=-1.0)


def test_compare_detects_identical_weights():
    population = np.array([10.0, 20.0, 30.0])
    result = W.compare(population, population)
    assert result["max_relative_diff"] == 0.0
    assert result["correlation"] == pytest.approx(1.0)


def test_compare_reports_relative_difference():
    result = W.compare(np.array([100.0, 100.0]), np.array([100.0, 150.0]))
    assert result["max_relative_diff"] == pytest.approx(0.5)


# --- привязка исходящего адреса (обход VPN) ---

def test_stale_source_ip_is_rejected_with_a_readable_message():
    """Адрес Wi-Fi меняется при переподключении; устаревший давал WinError
    10049 изнутри httpx и выглядел как недоступный сервер.

    Берём адрес из TEST-NET-3: он зарезервирован под документацию и своим
    не бывает, поэтому тест не мигает при смене сети."""
    with pytest.raises(CL.SourceIPUnavailable, match="нет на машине"):
        CL.check_source_ip("203.0.113.1")


def test_error_lists_current_addresses_so_the_fix_is_obvious():
    with pytest.raises(CL.SourceIPUnavailable) as caught:
        CL.check_source_ip("203.0.113.7")
    message = str(caught.value)
    assert "Адреса сейчас" in message and ".env" in message


def test_loopback_binds_fine():
    CL.check_source_ip("127.0.0.1")              # не бросает


def test_local_ipv4_returns_addresses():
    addresses = CL.local_ipv4()
    assert all(a.count(".") == 3 for a in addresses)
