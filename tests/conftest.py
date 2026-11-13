"""Общая подготовка тестов SETO.

Протокол при заморозке запоминает контрольные суммы рабочих документов, которые лежат вне
git (`protocol.reference_docs`, Д22). В чистой копии репозитория этих файлов нет, поэтому
тесты подставляют свои: проверяется механизм, а не содержимое чужих документов.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seto import settings  # noqa: E402
from seto.experiment import protocol as proto  # noqa: E402


@pytest.fixture(autouse=True)
def reference_docs_root(tmp_path_factory, monkeypatch):
    """Корень с рабочими документами-заглушками вместо корня репозитория."""
    root = tmp_path_factory.mktemp("repo_root")
    for name in settings.load_global()["protocol"]["reference_docs"]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"заглушка {name}\n", encoding="utf-8")
    monkeypatch.setattr(proto, "REPO_ROOT", root)
    return root
