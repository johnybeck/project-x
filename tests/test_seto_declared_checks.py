"""Тесты блока 12: проверки, объявленные до заморозки протокола.

Правило допустимой площадки (12.7), правило перевода подоценок (12.4), текст критерия
затрат (12.3) и то, как всё это попадает в протокол. Общая мысль у всех одна: проверка
берётся только из закрытого списка в конфиге, а список входит в протокол.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seto import __main__ as cli  # noqa: E402
from seto import settings  # noqa: E402
from seto.experiment import protocol as proto  # noqa: E402
from seto.experiment import step2 as step2_mod  # noqa: E402
from seto.experiment.prepare import use_site_check  # noqa: E402
from seto.report import city as report_city  # noqa: E402
from seto.scoring import prompt as P  # noqa: E402
from seto.scoring import rules as R  # noqa: E402
from seto.sites import feasible as FS  # noqa: E402

GLOBAL = settings.load_global()
TYPED = ["residential_share", "public_share"]


# --- 12.7: правило допустимой площадки ---

def _features(**columns) -> pd.DataFrame:
    size = len(next(iter(columns.values())))
    base = {name: np.zeros(size) for name in (
        "residential_share", "public_share", "built_share", "water_share", "green_share",
        "landuse_industrial_share", "landuse_railway_share", "landuse_cemetery_share",
        "landuse_military_share")}
    return pd.DataFrame({**base, **columns})


def test_madrid_counts_all_buildings():
    """Тип здания в OSM — дело усердия картографа: у 39 % зданий Мадрида его нет."""
    rule = settings.setting(settings.load("madrid"), "feasible_site")
    assert rule["built_from"] == ["built_share"]
    assert rule["min_built_area_share"] == GLOBAL["feasible_site"]["min_built_area_share"]


def test_madrid_keeps_every_other_condition_of_the_common_rule():
    """Город переопределяет раздел целиком, поэтому пороги препятствий легко потерять."""
    rule = settings.setting(settings.load("madrid"), "feasible_site")
    common = GLOBAL["feasible_site"]
    for key in common:
        if key != "built_from":
            assert rule[key] == common[key], key


def test_untyped_buildings_make_a_cell_feasible_under_the_main_rule():
    features = _features(built_share=np.array([0.14, 0.14]),
                         residential_share=np.array([0.02, 0.10]))
    main = settings.setting(settings.load("madrid"), "feasible_site")
    sites = FS.select(features, main, np.ones(2, bool), np.full(2, 900.0), 500)
    assert sites.index.tolist() == [0, 1]


def test_site_check_brings_back_typed_buildings():
    cfg = settings.load("madrid")
    use_site_check(cfg, "typed_buildings")
    rule = cfg["city"]["feasible_site"]
    assert rule["built_from"] == TYPED
    assert rule["check"] == "typed_buildings"
    # клетка, где тип указан у зданий на 2 % площади, под проверкой снова выпадает
    features = _features(built_share=np.array([0.14, 0.14]),
                         residential_share=np.array([0.02, 0.10]))
    sites = FS.select(features, rule, np.ones(2, bool), np.full(2, 900.0), 500)
    assert sites.index.tolist() == [1]


def test_site_check_changes_nothing_but_the_buildings():
    main = settings.setting(settings.load("madrid"), "feasible_site")
    cfg = settings.load("madrid")
    use_site_check(cfg, "typed_buildings")
    check = cfg["city"]["feasible_site"]
    assert {key for key in main if main[key] != check[key]} == {"built_from"}


def test_site_check_must_be_listed_in_config():
    """Список проверок входит в протокол: другое правило подставить нельзя."""
    cfg = settings.load("madrid")
    with pytest.raises(settings.ConfigError, match="feasible_site.checks"):
        use_site_check(cfg, "all_cells")


def test_city_without_checks_has_nothing_to_choose_from():
    cfg = settings.load("moscow")
    with pytest.raises(settings.ConfigError, match="feasible_site.checks"):
        use_site_check(cfg, "typed_buildings")


def test_report_names_which_buildings_were_counted():
    features = _features(built_share=np.array([0.2, 0.0]))
    all_buildings = FS.select(features, {"min_built_area_share": 0.05,
                                         "built_from": ["built_share"]},
                              np.ones(2, bool), np.full(2, 900.0), 500)
    assert all_buildings.reasons["мало застройки (все здания)"] == 1
    typed = FS.select(features, {"min_built_area_share": 0.05}, np.ones(2, bool),
                      np.full(2, 900.0), 500)
    assert typed.reasons["мало жилой и общественной застройки"] == 2


def test_prepare_city_takes_one_check_at_a_time(capsys):
    """Сочетание проверок в протокол не входит."""
    code = cli.main(["prepare-city", "madrid", "--population-check", "raster_by_sections",
                     "--site-check", "typed_buildings"])
    assert code == 1
    assert "одна проверка" in capsys.readouterr().err


def test_prepare_city_refuses_unlisted_site_check(capsys):
    assert cli.main(["prepare-city", "madrid", "--site-check", "all_cells"]) == 1
    assert "feasible_site.checks" in capsys.readouterr().err


# --- 12.4: правило-проверка скора ---

def test_step_rule_is_listed_and_valid():
    rule = R.for_check(GLOBAL, "age_step")
    assert set(rule["points"]) == set(P.RATINGS["epoch"])


def test_step_rule_is_a_step_at_2000():
    """По кадастру Мадрида доля 65+ почти одна до 2000 года и резко ниже после."""
    points = R.for_check(GLOBAL, "age_step")["points"]
    assert points["до 1955"] == points["1955-1975"] == points["1975-2000"] == 1
    assert points["после 2000"] == 0
    assert points[P.UNKNOWN] == 0.5


def test_main_rule_did_not_change():
    assert GLOBAL["score_rule"] == "age"
    assert GLOBAL["score_rules"]["age"]["points"] == {
        "до 1955": 3, "1955-1975": 2, "1975-2000": 1, "после 2000": 0, P.UNKNOWN: 1.5}


def test_rule_outside_the_list_cannot_be_used():
    """Правило описано в конфиге, но в списке проверок его нет — считать нельзя."""
    cfg = {**GLOBAL, "score_rules": {**GLOBAL["score_rules"],
                                     "handmade": GLOBAL["score_rules"]["age_step"]}}
    with pytest.raises(ValueError, match="score_rule_checks"):
        R.for_check(cfg, "handmade")


def test_main_rule_is_not_a_check():
    cfg = {**GLOBAL, "score_rule_checks": {"age": "ошибка в конфиге"}}
    with pytest.raises(ValueError, match="основное правило"):
        R.for_check(cfg, "age")


def test_broken_check_rule_stops_before_any_model_call():
    broken = {"rating": "epoch", "points": {"до 1955": 1}}
    cfg = {**GLOBAL, "score_rules": {**GLOBAL["score_rules"], "age_step": broken}}
    with pytest.raises(ValueError, match="нет баллов"):
        R.active(cfg)


def _answers() -> pd.DataFrame:
    """Четыре клетки с ответом модели и одна без него."""
    return pd.DataFrame({
        "cell_id": list("abcde"),
        "epoch": ["до 1955", "1975-2000", "после 2000", P.UNKNOWN, None],
        "imputed": [False, False, False, False, True]})


def test_cell_without_answer_gets_the_median_under_any_rule():
    answers = _answers()
    main = R.score_filled(answers, GLOBAL["score_rules"]["age"])
    assert main.tolist() == [3.0, 1.0, 0.0, 1.5, 1.25]
    step = R.score_filled(answers, R.for_check(GLOBAL, "age_step"))
    assert step.tolist() == [1.0, 1.0, 0.0, 0.5, 0.75]


def test_filled_score_equals_plain_score_when_every_cell_answered():
    answers = _answers().iloc[:4]
    rule = GLOBAL["score_rules"]["age"]
    assert R.score_filled(answers, rule).tolist() == R.score(answers, rule).tolist()


def test_no_answers_at_all_is_an_error():
    answers = pd.DataFrame({"epoch": [None, None], "imputed": [True, True]})
    with pytest.raises(RuntimeError, match="считать нечего"):
        R.score_filled(answers, GLOBAL["score_rules"]["age"])


def _scored_folder(path: Path) -> Path:
    """Папка прогона со скорами: колонка score посчитана по основному правилу."""
    path.mkdir(parents=True)
    answers = _answers()
    answers["score"] = R.score_filled(answers, GLOBAL["score_rules"]["age"])
    answers.to_parquet(path / "scores_image.parquet")
    return path


def test_saved_scores_are_read_as_is_without_a_rule(tmp_path):
    scores = cli._load_scores(_scored_folder(tmp_path / "run"), list("abcde"))
    assert scores["image"].tolist() == [3.0, 1.0, 0.0, 1.5, 1.25]


def test_check_rule_rescores_the_saved_answers(tmp_path):
    """Те же ответы модели, другой перевод: новых вызовов нет."""
    scores = cli._load_scores(_scored_folder(tmp_path / "run"), list("abcde"),
                              R.for_check(GLOBAL, "age_step"))
    assert scores["image"].tolist() == [1.0, 1.0, 0.0, 0.5, 0.75]


@pytest.fixture
def stubbed_step2(monkeypatch, tmp_path):
    """Шаг 2 без вычислений: запоминает, какие скоры и какой конфиг до него дошли."""
    calls = []
    monkeypatch.setenv("SETO_DATA_ROOT", str(tmp_path / "root"))
    monkeypatch.setattr(cli, "_load_city_data", lambda run_dir: SimpleNamespace(
        features=pd.DataFrame({"cell_id": list("abcde")})))

    def run(data, cfg, out_dir, logger, costs, scores):
        calls.append(SimpleNamespace(out_dir=out_dir, cfg=cfg, scores=scores))
        paths = {"coverage_by_k": out_dir / "coverage_by_k.csv", "m2": out_dir / "m2.csv"}
        pd.DataFrame({"k": [2]}).to_csv(paths["coverage_by_k"], index=False)
        pd.DataFrame(columns=["need", "k", "same_sites", "ceiling_per_1000",
                              "noise_per_1000", "above_noise"]).to_csv(paths["m2"],
                                                                       index=False)
        return paths

    monkeypatch.setattr(step2_mod, "run_step2", run)
    return calls


def _prepared(path: Path) -> Path:
    path.mkdir(parents=True)
    for name in ("manifest.json", "times.npz", "cell_features.parquet"):
        (path / name).write_text("{}", encoding="utf-8")
    return path


def test_step2_with_a_check_rule_gets_its_own_folder(stubbed_step2, tmp_path, capsys):
    prepared, scored = _prepared(tmp_path / "prepared"), _scored_folder(tmp_path / "run")
    code = cli.main(["step2", "madrid", str(prepared), "--scores", str(scored),
                     "--score-rule", "age_step"])
    assert code == 0
    call = stubbed_step2[0]
    assert call.scores["image"].tolist() == [1.0, 1.0, 0.0, 0.5, 0.75]
    assert call.out_dir.name.endswith("step2-age_step")
    saved = yaml.safe_load((call.out_dir / "effective_config.yaml")
                           .read_text(encoding="utf-8"))
    assert (saved["score_rule"], saved["score_rule_check_of"]) == ("age_step", "age")
    assert "выбрана после просмотра эталонных данных Мадрида" in capsys.readouterr().out


def test_step2_without_a_check_rule_goes_as_before(stubbed_step2, tmp_path):
    prepared, scored = _prepared(tmp_path / "prepared"), _scored_folder(tmp_path / "run")
    assert cli.main(["step2", "madrid", str(prepared), "--scores", str(scored)]) == 0
    call = stubbed_step2[0]
    assert call.scores["image"].tolist() == [3.0, 1.0, 0.0, 1.5, 1.25]
    assert call.out_dir.name.endswith("-step2")
    assert "score_rule_check_of" not in call.cfg


def test_step2_refuses_a_rule_outside_the_list(stubbed_step2, tmp_path, capsys):
    prepared, scored = _prepared(tmp_path / "prepared"), _scored_folder(tmp_path / "run")
    code = cli.main(["step2", "madrid", str(prepared), "--scores", str(scored),
                     "--score-rule", "handmade"])
    assert code == 1 and not stubbed_step2
    assert "score_rule_checks" in capsys.readouterr().err


def test_check_rule_needs_saved_answers(stubbed_step2, tmp_path, capsys):
    prepared = _prepared(tmp_path / "prepared")
    assert cli.main(["step2", "madrid", str(prepared), "--score-rule", "age_step"]) == 1
    assert not stubbed_step2
    assert "--scores" in capsys.readouterr().err


# --- протокол: проверки, критерий затрат, правило площадки ---

@pytest.fixture
def frozen(tmp_path, monkeypatch):
    monkeypatch.setattr(proto, "git_state",
                        lambda: {"commit": "c1", "branch": "main", "dirty": False})
    cfg = settings.load("madrid")
    json_path, md_path = proto.freeze(cfg, {"madrid": "dev"}, {}, tmp_path,
                                      {"madrid": proto.city_rules(cfg)})
    return SimpleNamespace(payload=json.loads(json_path.read_text(encoding="utf-8")),
                           page=md_path.read_text(encoding="utf-8"), path=json_path,
                           cfg=cfg)


def test_protocol_lists_every_declared_check(frozen):
    text = "\n".join(frozen.payload["declared_checks"])
    assert "правило перевода `age_step`" in text
    assert "выбрана после просмотра эталонных данных Мадрида" in text
    assert "age_r1, age_r2, age_80" in text
    assert "raster_by_sections, registry_by_raster" in text
    assert "правило допустимой площадки: typed_buildings" in text
    assert "доля плотных клеток [0.1, 0.25, 0.5]" in text
    assert "главные дороги до ['primary', 'secondary', 'tertiary']" in text


def test_protocol_page_has_a_section_for_declared_checks(frozen):
    assert "## Проверки устойчивости, объявленные заранее" in frozen.page
    for line in frozen.payload["declared_checks"]:
        assert f"- {line}" in frozen.page


def test_protocol_says_in_advance_what_the_cost_criterion_will_show(frozen):
    """Что мерится у E и чего ждать, пропозал велит назвать до прогонов (Д18)."""
    text = frozen.payload["pilot_criteria"]["costs"]
    assert "у E мерится: algorithm" in text
    assert "критерий ожидаемо не выполнится" in text
    assert "этим критерием не проверяется" in text
    assert text in frozen.page


def test_cost_text_comes_from_config_not_from_code():
    cfg = {**GLOBAL, "pilot": {"e_cost_measure": "manual_analyst",
                               "e_cost_expectation": "замер одного аналитика"}}
    text = proto._cost_text(cfg)
    assert "у E мерится: manual_analyst" in text and "замер одного аналитика" in text
    assert "Заранее" not in proto._cost_text({**GLOBAL, "pilot": {"e_cost_measure": "x"}})


def test_city_report_repeats_what_was_said_in_advance():
    verdict = report_city._cost_criterion(settings.load("madrid"), None, None)
    assert verdict["E_measured_as"] == "algorithm"
    assert "ожидаемо не выполнится" in verdict["stated_in_advance"]


def test_protocol_holds_the_site_rule_of_the_city(frozen):
    """Общий раздел параметров показывает правило по умолчанию, а Мадрид считает иначе."""
    rule = frozen.payload["city_rules"]["madrid"]["feasible_site"]
    assert rule["built_from"] == ["built_share"]
    assert list(rule["checks"]) == ["typed_buildings"]
    assert frozen.payload["parameters"]["feasible_site"]["built_from"] == TYPED


def test_protocol_holds_the_list_of_check_rules(frozen):
    assert frozen.payload["parameters"]["score_rule_checks"] == GLOBAL["score_rule_checks"]
    assert "age_step" in frozen.payload["parameters"]["score_rules"]


def test_verify_notices_a_changed_site_rule(frozen, monkeypatch):
    changed = settings.load("madrid")
    changed["city"]["feasible_site"]["built_from"] = TYPED
    problems = proto.verify(frozen.path, changed, {})["problems"]
    assert any("правила города madrid" in problem for problem in problems)


def test_verify_notices_a_new_check_rule(frozen):
    changed = settings.load("madrid")
    changed["score_rule_checks"] = {**changed["score_rule_checks"], "handmade": "после"}
    problems = proto.verify(frozen.path, changed, {})["problems"]
    assert any("score_rule_checks" in problem for problem in problems)


def test_deviation_about_sites_is_in_the_protocol(frozen):
    assert any("все здания OSM" in item
               for item in frozen.payload["city_rules"]["madrid"]["deviations"])
    assert "все здания OSM" in frozen.page
