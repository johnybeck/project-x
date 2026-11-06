"""Протокол и предрегистрация (п. 13).

Смысл простой: после заморозки ничего из того, что влияет на вывод, меняться
не должно. Поэтому протокол фиксирует не только параметры, но и хеши — кода,
данных, промптов. Расхождение любого из них при запуске на городе проверки
останавливает прогон.

Без этого «предрегистрация» была бы декларацией: параметры можно было бы
подправить после того, как увидели результаты.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from ..run import git_state, sha256_of
from ..scoring import client as scoring_client
from ..scoring import prompt as prompt_mod
from .evaluate import CHAIN

# что фиксируется по городу, кроме среза; остальное считается одинаково везде
_CITY_KEYS = ("need", "population", "buildings", "imagery", "income", "m4",
              "source_dates", "deviations")


class ProtocolMismatch(RuntimeError):
    """Текущее состояние расходится с замороженным протоколом."""


def city_rules(cfg: dict) -> dict:
    """Правила города: срез, определения нуждаемости, снимки, отступления от пропозала."""
    city = cfg["city"]
    return {"snapshot_date": cfg["snapshot_date"],
            **{key: city[key] for key in _CITY_KEYS if key in city}}


def data_sources(manifests: dict[str, Path]) -> dict[str, list[dict]]:
    """Срезы и лицензии: что, откуда и когда взято у каждого города.

    Берётся из манифестов загрузки. Города без манифеста пропускаются, как и в
    city_data_hashes."""
    result = {}
    for city, path in manifests.items():
        if Path(path).is_file():
            sources = json.loads(Path(path).read_text(encoding="utf-8"))["sources"]
            result[city] = [{key: source.get(key) for key in
                             ("name", "url", "license", "downloaded_at", "sha256")}
                            for source in sources]
    return result


def _hash_payload(payload: dict) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def freeze(cfg: dict, cities: dict[str, str], data_hashes: dict[str, str],
           out_dir: Path, rules: dict[str, dict] | None = None,
           sources: dict[str, list[dict]] | None = None) -> tuple[Path, Path]:
    """Собрать protocol.json и текст для страницы предрегистрации.

    cities — роль каждого города, data_hashes — хеши манифестов данных городов,
    rules и sources — правила городов и их источники данных."""
    state = git_state()
    if state.get("dirty"):
        raise ProtocolMismatch(
            "рабочее дерево грязное: сначала закоммить изменения, иначе "
            "протокол будет ссылаться на коммит, которого нет в истории")

    size = cfg["grid"]["size_m"]
    protocol = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "commit": state["commit"],
        "branch": state["branch"],
        "model": cfg["vlm"]["model"],
        "prompt_hashes": {
            "image": prompt_mod.prompt_hash(prompt_mod.build("image", size)),
            "text_template_version": prompt_mod.TEXT_TEMPLATE_VERSION,
        },
        "prompt_diff": prompt_mod.diff(),
        "answer_parsing": scoring_client.PARSE_RULE,
        "parameters": {key: cfg[key] for key in (
            "scenario", "k", "k_sensitivity", "T_minutes", "T_sensitivity",
            "walk_speed_m_per_min", "t_max_minutes", "snap_max_m",
            "min_dist_to_existing_m", "min_dist_between_new_m",
            "alpha", "alpha_sensitivity", "score_rule", "score_rules",
            "random_draws", "seed", "swap_heuristic", "m2", "targeting", "ahp",
            "pilot", "imputation_max_share", "feasible_site", "grid") if key in cfg},
        "cities": cities,
        "city_rules": rules or {},
        "data_sources": sources or {},
        "data_hashes": data_hashes,
        "chain": [f"{left} − {right}: {meaning}" for left, right, meaning in CHAIN],
        "metrics": ["Cov_T", "dCov_T", "dN_T", "worst_decile", "mean_time",
                    "2SFCA: share_below_floor, p10, p25",
                    "М1: Cov_N_T, dCov_N_T, dN_N_T, новые 65+, 65+ при медленной ходьбе",
                    "М2: потолок, порог шума, доля захвата C, D, D10 при всех k",
                    "М3: EDE, дефицит, Джини, Тейль с разложением, 2SFCA по N",
                    "М4: каппа эпохи у D, ранговая корреляция достатка с доходом"],
        "pilot_criteria": {
            "quality": "dCov_T варианта D выше 95-го процентиля варианта A",
            "costs": ("активное время человека на подключение нового города у D не "
                      "более половины, чем у E; у E мерится: "
                      f"{(cfg.get('pilot') or {}).get('e_cost_measure')}"),
            "reliability": ("D дал допустимое решение без ручной правки; "
                            "при сбое допускается один перезапуск"),
        },
        "not_computed": ["порог допустимой потери δ", "порог ε",
                         "среднее A_i по 2SFCA"],
    }
    protocol["sha256"] = _hash_payload(protocol)

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "protocol.json"
    json_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    md_path = out_dir / "protocol.md"
    md_path.write_text(_markdown(protocol), encoding="utf-8")
    return json_path, md_path


def verify(protocol_path: Path, cfg: dict,
           data_hashes: dict[str, str] | None = None) -> dict:
    """Проверить, что состояние совпадает с протоколом (п. 13).

    Если в cfg есть город, сверяется он один: чужие данные тут ни при чём, а город
    без записи в протоколе считать нельзя. Возвращает список расхождений; пустой
    означает, что запускать можно."""
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    problems: list[str] = []

    state = git_state()
    if state.get("dirty"):
        problems.append("рабочее дерево грязное")
    if state.get("commit") != protocol.get("commit"):
        problems.append(f"коммит {state.get('commit')} вместо "
                        f"{protocol.get('commit')}")
    if cfg["vlm"]["model"] != protocol.get("model"):
        problems.append(f"модель {cfg['vlm']['model']} вместо {protocol.get('model')}")

    size = cfg["grid"]["size_m"]
    current = prompt_mod.prompt_hash(prompt_mod.build("image", size))
    if current != protocol.get("prompt_hashes", {}).get("image"):
        problems.append("промпт изменился")

    for key, value in protocol.get("parameters", {}).items():
        if cfg.get(key) != value:
            problems.append(f"параметр {key}: {cfg.get(key)} вместо {value}")

    name = (cfg.get("city") or {}).get("name")
    frozen = (protocol.get("city_rules") or {}).get(name)
    if name is not None and frozen is None:
        problems.append(f"города {name} нет в протоколе")
    elif frozen is not None and frozen != city_rules(cfg):
        problems.append(f"правила города {name} изменились")

    frozen_data = protocol.get("data_hashes") or {}
    if name is not None and name not in frozen_data:
        problems.append(f"в протоколе нет данных города {name}")
    for city, expected in frozen_data.items():
        if name is not None and city != name:
            continue
        if data_hashes is None:
            # вызывающий не проверял данные — молча считать их совпавшими нельзя
            problems.append(f"данные города {city} не проверены")
            continue
        actual = data_hashes.get(city)
        if actual is None:
            problems.append(f"нет манифеста данных города {city}")
        elif actual != expected:
            problems.append(f"данные города {city} изменились")

    return {"ok": not problems, "problems": problems,
            "protocol_sha256": protocol.get("sha256")}


def verify_run(protocol_path: Path, run_dir: Path, city: str) -> list[str]:
    """Чем готовая папка прогона расходится с протоколом; пусто — ей можно верить.

    Шаг, который читает готовую папку, сам ничего не пересчитывает: свежие код и
    данные не спасут, если папка сделана до заморозки или на другом срезе."""
    run_dir = Path(run_dir)
    path = run_dir / "manifest.json"
    if not path.is_file():
        return [f"в папке {run_dir.name} нет манифеста"]
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    manifest = json.loads(path.read_text(encoding="utf-8"))
    git = manifest.get("git") or {}
    snapshot = ((protocol.get("city_rules") or {}).get(city) or {}).get("snapshot_date")

    problems = []
    if manifest.get("city") != city:
        problems.append(f"папка {run_dir.name}: город {manifest.get('city')} вместо {city}")
    if git.get("commit") != protocol.get("commit"):
        problems.append(f"папка {run_dir.name}: коммит {git.get('commit')} вместо "
                        f"{protocol.get('commit')}")
    if git.get("dirty"):
        problems.append(f"папка {run_dir.name}: сделана на грязном дереве")
    if manifest.get("snapshot_date") != snapshot:
        problems.append(f"папка {run_dir.name}: срез {manifest.get('snapshot_date')} "
                        f"вместо {snapshot}")
    return problems


def hash_city_data(manifest_path: Path) -> str:
    """Хеш манифеста данных города — им протокол привязывается к данным."""
    return sha256_of(manifest_path)


def city_data_hashes(manifests: dict[str, Path]) -> dict[str, str]:
    """Хеши data_manifest.json по городам; города без манифеста пропускаются.

    Манифест пишет `acquire` в raw/<город>/<срез>/. В нём хеши сырых файлов,
    поэтому хеш манифеста меняется при любой перекачке данных."""
    return {city: hash_city_data(path) for city, path in manifests.items()
            if Path(path).is_file()}


def _markdown(protocol: dict) -> str:
    lines = [
        "# Протокол эксперимента (предрегистрация)",
        "",
        f"Заморожен: {protocol['frozen_at']}",
        f"Коммит: `{protocol['commit']}`",
        f"Модель: `{protocol['model']}`",
        f"Хеш протокола: `{protocol['sha256']}`",
        "",
        "## Города",
        "",
    ]
    for city, role in protocol["cities"].items():
        snapshot = (protocol["city_rules"].get(city) or {}).get("snapshot_date")
        lines.append(f"- **{city}** — {role}, срез данных {snapshot}")
    lines += ["", "## Параметры", "", "```json",
              json.dumps(protocol["parameters"], ensure_ascii=False, indent=2),
              "```", "", "## Правила городов", "", "```json",
              json.dumps(protocol["city_rules"], ensure_ascii=False, indent=2),
              "```", "", "## Цепочка разниц", ""]
    lines += [f"- {item}" for item in protocol["chain"]]
    lines += ["", "## Метрики", ""]
    lines += [f"- {m}" for m in protocol["metrics"]]
    lines += ["", "## Критерии пилота", ""]
    for name, text in protocol["pilot_criteria"].items():
        lines.append(f"- **{name}**: {text}")
    lines += ["", "## Отступления от пропозала", ""]
    if protocol.get("answer_parsing"):
        lines.append(f"- **все города**: {protocol['answer_parsing']}")
    for city, rules in protocol["city_rules"].items():
        lines += [f"- **{city}**: {item}" for item in rules.get("deviations", [])]
    lines += ["", "## На какую дату источники описывают город", ""]
    for city, rules in protocol["city_rules"].items():
        lines += [f"- **{city}**, {item['source']}: {item['date']} — {item['used_for']}"
                  for item in rules.get("source_dates", [])]
    lines += ["", "## Данные и лицензии", ""]
    for city, sources in protocol["data_sources"].items():
        lines += [f"- **{city}**, {item['name']}: {item['license']}, "
                  f"загружено {item['downloaded_at']}" for item in sources]
    lines += ["", "## Чего эксперимент НЕ делает", ""]
    lines += [f"- {m}" for m in protocol["not_computed"]]
    lines += ["", "После просмотра результатов проверки ничего из этого "
              "не меняется.", ""]
    return "\n".join(lines)
