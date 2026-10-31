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
from ..scoring import prompt as prompt_mod
from .evaluate import CHAIN

# что фиксируется по городу, кроме среза; остальное считается одинаково везде
_CITY_KEYS = ("need", "population", "imagery", "income", "m4", "deviations")


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

    cities — роль каждого города, data_hashes — хеши манифестов данных
    городов проверки, rules и sources — правила городов и их источники данных."""
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
        "parameters": {key: cfg[key] for key in (
            "scenario", "k", "k_sensitivity", "T_minutes", "T_sensitivity",
            "walk_speed_m_per_min", "t_max_minutes", "snap_max_m",
            "min_dist_to_existing_m", "min_dist_between_new_m",
            "alpha", "alpha_sensitivity", "score_rule", "score_rules",
            "random_draws", "seed", "swap_heuristic", "m2", "targeting", "ahp",
            "pilot", "imputation_max_share", "feasible_site", "grid") if key in cfg},
        "snapshot_date": cfg.get("snapshot_date"),
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

    Возвращает список расхождений; пустой означает, что запускать можно."""
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
    if frozen is not None and frozen != city_rules(cfg):
        problems.append(f"правила города {name} изменились")

    for city, expected in (protocol.get("data_hashes") or {}).items():
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
        f"Срез данных: {protocol['snapshot_date']}",
        f"Хеш протокола: `{protocol['sha256']}`",
        "",
        "## Города",
        "",
    ]
    for city, role in protocol["cities"].items():
        lines.append(f"- **{city}** — {role}")
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
    for city, rules in protocol["city_rules"].items():
        lines += [f"- **{city}**: {item}" for item in rules.get("deviations", [])]
    lines += ["", "## Данные и лицензии", ""]
    for city, sources in protocol["data_sources"].items():
        lines += [f"- **{city}**, {item['name']}: {item['license']}, "
                  f"загружено {item['downloaded_at']}" for item in sources]
    lines += ["", "## Чего эксперимент НЕ делает", ""]
    lines += [f"- {m}" for m in protocol["not_computed"]]
    lines += ["", "После просмотра результатов проверки ничего из этого "
              "не меняется.", ""]
    return "\n".join(lines)
