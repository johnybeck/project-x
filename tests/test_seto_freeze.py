"""Заморозка защищает город с любой ролью (задача 10.4 TODO_V2)."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seto import __main__ as cli  # noqa: E402
from seto import settings  # noqa: E402
from seto.experiment import protocol as proto  # noqa: E402
from seto.experiment import runner  # noqa: E402
from seto.experiment import step2 as step2_mod  # noqa: E402


def _git(monkeypatch, commit: str = "c1", dirty: bool = False) -> None:
    monkeypatch.setattr(proto, "git_state",
                        lambda: {"commit": commit, "branch": "main", "dirty": dirty})


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    """Протокол Мадрида на чистом дереве; манифест данных лежит во временной папке."""
    _git(monkeypatch)
    manifest = tmp_path / "data_manifest.json"
    manifest.write_text('{"sources": []}', encoding="utf-8")
    monkeypatch.setattr(cli, "_data_manifests",
                        lambda cities: {name: manifest for name in cities})
    cfg = settings.load("madrid")
    json_path, md_path = proto.freeze(cfg, {"madrid": "dev"},
                                      proto.city_data_hashes({"madrid": manifest}),
                                      tmp_path / "protocol",
                                      {"madrid": proto.city_rules(cfg)})
    return SimpleNamespace(path=json_path, page=md_path, manifest=manifest, cfg=cfg)


@pytest.fixture
def started(monkeypatch, tmp_path):
    """Вместо настоящего прогона города — запись вызова."""
    calls = []
    monkeypatch.setattr(runner, "run_city",
                        lambda *args: calls.append(args) or tmp_path)
    return calls


@pytest.fixture
def stubbed_step2(monkeypatch, tmp_path):
    """Шаг 2 без вычислений: важно только, дошло ли до него дело."""
    calls = []
    monkeypatch.setenv("SETO_DATA_ROOT", str(tmp_path / "root"))
    monkeypatch.setattr(cli, "_load_city_data", lambda run_dir: SimpleNamespace(
        features=pd.DataFrame({"cell_id": ["a"]})))

    def run(data, cfg, out_dir, logger, costs, scores):
        calls.append(out_dir)
        paths = {"coverage_by_k": out_dir / "coverage_by_k.csv", "m2": out_dir / "m2.csv"}
        pd.DataFrame({"k": [2]}).to_csv(paths["coverage_by_k"], index=False)
        pd.DataFrame(columns=["need", "k", "same_sites", "ceiling_per_1000",
                              "noise_per_1000", "above_noise"]).to_csv(paths["m2"],
                                                                       index=False)
        return paths

    monkeypatch.setattr(step2_mod, "run_step2", run)
    return calls


def _folder(path: Path, **overrides) -> Path:
    """Папка прогона, сделанная по протоколу, если не сказано иначе."""
    path.mkdir()
    manifest = {"city": "madrid", "snapshot_date": "2026-10-28",
                "git": {"commit": "c1", "dirty": False}}
    (path / "manifest.json").write_text(json.dumps({**manifest, **overrides}),
                                        encoding="utf-8")
    for name in ("times.npz", "cell_features.parquet"):
        (path / name).write_bytes(b"x")
    return path


def _test_role(monkeypatch) -> None:
    """Мадрид с ролью test, остальное как в конфигах."""
    real = settings.load

    def load(name=None):
        cfg = real(name)
        if name == "madrid":
            cfg["city"]["role"] = "test"
        return cfg

    monkeypatch.setattr(settings, "load", load)


# --- run-city ---

def test_run_city_passes_with_matching_protocol(frozen, started):
    assert cli.main(["run-city", "madrid", "--protocol", str(frozen.path)]) == 0
    assert started[0][2] == frozen.path


@pytest.mark.parametrize("commit, dirty, reason", [
    ("c2", False, "коммит c2"),
    ("c1", True, "дерево грязное"),
])
def test_run_city_refuses_other_code(frozen, started, monkeypatch, capsys,
                                     commit, dirty, reason):
    _git(monkeypatch, commit, dirty)
    assert cli.main(["run-city", "madrid", "--protocol", str(frozen.path)]) == 3
    assert not started
    assert reason in capsys.readouterr().err


def test_run_city_refuses_changed_data(frozen, started, capsys):
    """Перекачка данных после заморозки у города dev теперь тоже видна."""
    frozen.manifest.write_text('{"sources": ["другие"]}', encoding="utf-8")
    assert cli.main(["run-city", "madrid", "--protocol", str(frozen.path)]) == 3
    assert not started
    assert "данные города madrid изменились" in capsys.readouterr().err


def test_run_city_refuses_changed_rules(frozen, started, monkeypatch, capsys):
    changed = settings.load("madrid")
    changed["city"]["need"]["r"] = 3.0
    monkeypatch.setattr(settings, "load", lambda name=None: changed)
    assert cli.main(["run-city", "madrid", "--protocol", str(frozen.path)]) == 3
    assert not started
    assert "правила города madrid" in capsys.readouterr().err


def test_run_city_refuses_other_slice(frozen, started, capsys):
    assert cli.main(["run-city", "madrid", "--protocol", str(frozen.path),
                     "--snapshot-date", "2026-10-08"]) == 3
    assert not started
    assert "правила города madrid" in capsys.readouterr().err


def test_run_city_refuses_missing_protocol_file(frozen, started, tmp_path):
    assert cli.main(["run-city", "madrid", "--protocol", str(tmp_path / "нет.json")]) == 3
    assert not started


def test_run_city_without_protocol_goes_as_before(started, monkeypatch):
    """Разработка до заморозки не меняется: сверки нет, город не отказывает."""
    def forbidden(*_args):
        raise AssertionError("без --protocol сверки быть не должно")

    monkeypatch.setattr(cli, "_refused", forbidden)
    assert cli.main(["run-city", "madrid"]) == 0
    assert started[0][2] is None


def test_run_city_records_protocol_in_manifest(frozen, monkeypatch, tmp_path):
    """Результатом считается прогон, в манифесте которого есть протокол."""
    monkeypatch.setenv("SETO_DATA_ROOT", str(tmp_path / "root"))

    class Stop(Exception):
        pass

    def prepare(*_args):
        raise Stop

    monkeypatch.setattr(runner, "prepare_city", prepare)
    with pytest.raises(Stop):
        runner.run_city("madrid", frozen.cfg, frozen.path)
    run_dir = next((tmp_path / "root" / "runs" / "madrid").iterdir())
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert str(frozen.path) in manifest["inputs"]


# --- step2 ---

def test_step2_passes_with_matching_protocol(frozen, stubbed_step2, tmp_path):
    prepared = _folder(tmp_path / "prepared")
    assert cli.main(["step2", "madrid", str(prepared), "--protocol", str(frozen.path)]) == 0
    assert stubbed_step2
    manifest = json.loads((stubbed_step2[0] / "manifest.json").read_text(encoding="utf-8"))
    assert str(frozen.path) in manifest["inputs"]


@pytest.mark.parametrize("overrides, reason", [
    ({"git": {"commit": "c0", "dirty": False}}, "коммит c0"),
    ({"git": {"commit": "c1", "dirty": True}}, "грязном дереве"),
    ({"city": "moscow"}, "город moscow"),
    ({"snapshot_date": "2026-10-08"}, "срез 2026-10-08"),
])
def test_step2_refuses_foreign_prepared_folder(frozen, stubbed_step2, tmp_path, capsys,
                                               overrides, reason):
    """Шаг читает готовую папку: подготовка до заморозки обошла бы всю сверку."""
    prepared = _folder(tmp_path / "prepared", **overrides)
    assert cli.main(["step2", "madrid", str(prepared), "--protocol", str(frozen.path)]) == 3
    assert not stubbed_step2
    assert reason in capsys.readouterr().err


def test_step2_refuses_scores_from_before_the_freeze(frozen, stubbed_step2, tmp_path):
    prepared = _folder(tmp_path / "prepared")
    scored = _folder(tmp_path / "scored", git={"commit": "c0", "dirty": False})
    assert cli.main(["step2", "madrid", str(prepared), "--scores", str(scored),
                     "--protocol", str(frozen.path)]) == 3
    assert not stubbed_step2


def test_step2_refuses_folder_without_manifest(frozen, stubbed_step2, tmp_path, capsys):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    assert cli.main(["step2", "madrid", str(prepared), "--protocol", str(frozen.path)]) == 3
    assert "нет манифеста" in capsys.readouterr().err


def test_step2_refuses_changed_data(frozen, stubbed_step2, tmp_path):
    frozen.manifest.write_text('{"sources": ["другие"]}', encoding="utf-8")
    prepared = _folder(tmp_path / "prepared")
    assert cli.main(["step2", "madrid", str(prepared), "--protocol", str(frozen.path)]) == 3
    assert not stubbed_step2


def test_step2_without_protocol_goes_as_before(stubbed_step2, tmp_path):
    prepared = _folder(tmp_path / "prepared", git={"commit": "c0", "dirty": True})
    assert cli.main(["step2", "madrid", str(prepared)]) == 0
    assert stubbed_step2


def test_step2_of_test_city_works_with_protocol(frozen, stubbed_step2, tmp_path,
                                                monkeypatch):
    """Раньше step2 у города test отказывал всегда: протокол ему не передавался."""
    _test_role(monkeypatch)
    prepared = _folder(tmp_path / "prepared")
    assert cli.main(["step2", "madrid", str(prepared), "--protocol", str(frozen.path)]) == 0
    assert stubbed_step2


def test_step2_of_test_city_needs_protocol(stubbed_step2, tmp_path, monkeypatch, capsys):
    _test_role(monkeypatch)
    prepared = _folder(tmp_path / "prepared")
    assert cli.main(["step2", "madrid", str(prepared)]) == 3
    assert not stubbed_step2
    assert "протокол не заморожен" in capsys.readouterr().err


# --- run-test ---

def test_run_test_without_test_cities_is_not_silent(frozen, started, capsys):
    """Молчаливый выход с кодом 0 выглядел бы как успешный прогон."""
    assert cli.main(["run-test", str(frozen.path)]) == 1
    assert not started
    assert "с ролью test нет" in capsys.readouterr().err


def test_run_test_runs_city_with_test_role(frozen, started, monkeypatch):
    _test_role(monkeypatch)
    assert cli.main(["run-test", str(frozen.path)]) == 0
    assert [call[0] for call in started] == ["madrid"]


def test_run_test_refuses_changed_data(frozen, started, monkeypatch):
    _test_role(monkeypatch)
    frozen.manifest.write_text('{"sources": ["другие"]}', encoding="utf-8")
    assert cli.main(["run-test", str(frozen.path)]) == 3
    assert not started


# --- заморозка и сверка ---

def test_freeze_binds_data_of_dev_city(tmp_path, monkeypatch):
    _git(monkeypatch)
    manifest = tmp_path / "data_manifest.json"
    manifest.write_text('{"sources": []}', encoding="utf-8")
    monkeypatch.setattr(cli, "_data_manifests",
                        lambda cities: {name: manifest for name in cities})
    assert cli.main(["freeze-protocol", "--out", str(tmp_path / "p")]) == 0
    payload = json.loads((tmp_path / "p" / "protocol.json").read_text(encoding="utf-8"))
    assert payload["cities"]["madrid"] == "dev"
    assert payload["data_hashes"]["madrid"] == proto.hash_city_data(manifest)


def test_freeze_needs_data_of_test_city_only(tmp_path, monkeypatch, capsys):
    """Города dev без манифеста не мешают, у города проверки манифест обязателен."""
    _git(monkeypatch)
    monkeypatch.setattr(cli, "_data_manifests",
                        lambda cities: {name: tmp_path / "нет.json" for name in cities})
    assert cli.main(["freeze-protocol", "--out", str(tmp_path / "dev")]) == 0
    _test_role(monkeypatch)
    assert cli.main(["freeze-protocol", "--out", str(tmp_path / "test")]) == 1
    assert "madrid" in capsys.readouterr().err


def test_verify_looks_only_at_own_city(tmp_path, monkeypatch):
    """Иначе при двух городах каждый отказывал бы из-за манифеста соседа."""
    _git(monkeypatch)
    cfg = settings.load("madrid")
    json_path, _ = proto.freeze(cfg, {"madrid": "dev", "kazan": "test"},
                                {"madrid": "h1", "kazan": "h2"}, tmp_path,
                                {"madrid": proto.city_rules(cfg)})
    assert proto.verify(json_path, cfg, {"madrid": "h1"})["ok"]


def test_verify_refuses_city_missing_from_protocol(tmp_path, monkeypatch):
    _git(monkeypatch)
    cfg = settings.load("madrid")
    json_path, _ = proto.freeze(cfg, {"moscow": "dev"}, {"moscow": "h"}, tmp_path,
                                {"moscow": {}})
    problems = proto.verify(json_path, cfg, {"madrid": "h"})["problems"]
    assert any("города madrid нет в протоколе" in problem for problem in problems)


def test_verify_refuses_city_without_frozen_data(tmp_path, monkeypatch):
    """Город без хеша данных в протоколе нельзя считать сверенным."""
    _git(monkeypatch)
    cfg = settings.load("madrid")
    json_path, _ = proto.freeze(cfg, {"madrid": "dev"}, {}, tmp_path,
                                {"madrid": proto.city_rules(cfg)})
    problems = proto.verify(json_path, cfg, {"madrid": "h"})["problems"]
    assert any("нет данных города madrid" in problem for problem in problems)


def test_protocol_page_gives_slice_of_each_city(frozen):
    """Общая дата из global.yaml выдавала бы чужой срез за срез Мадрида."""
    text = frozen.page.read_text(encoding="utf-8")
    assert "**madrid** — dev, срез данных 2026-10-28" in text
    assert "snapshot_date" not in json.loads(frozen.path.read_text(encoding="utf-8"))
