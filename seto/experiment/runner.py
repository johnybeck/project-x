"""Сквозной прогон города: подготовка → скоринг → варианты → метрики → отчёт.

Это `run-city` из п. 15. Правила сбоев по п. 13: вариант, который не удалось
посчитать, получает статус, а не роняет весь город; статусы идут в отчёт.

Города проверки отдельно защищены: без замороженного протокола запуск на них
запрещён (п. 13), это проверяется до начала работы, а не после.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..costs.timer import CostLog
from ..log import RunLogger
from ..report import city as report_mod
from ..run import Run
from ..scoring import client as scoring_client
from ..scoring import prompt as prompt_mod
from .evaluate import SCORED_VARIANTS, run_variants
from .prepare import prepare_city


class ProtocolRequired(RuntimeError):
    """Город проверки без замороженного протокола запускать нельзя."""


def guard_test_city(cfg: dict, protocol_path: Path | None) -> None:
    """Запрет запуска на городах проверки без протокола (п. 13).

    Проверяется ДО работы: узнать об этом после прогона уже поздно —
    результат придётся выбросить."""
    if cfg["city"].get("role") != "test":
        return
    if protocol_path is None or not Path(protocol_path).is_file():
        raise ProtocolRequired(
            f"город {cfg['city']['name']} помечен как test, но протокол не "
            "заморожен: сначала `python -m seto freeze-protocol`")


def sample_cells(features: pd.DataFrame, count: int, seed: int) -> list[str]:
    """Случайные заселённые клетки: проверка промпта на малом числе вызовов.

    Сид один и тот же, поэтому 10 клеток включают ту единственную, с которой
    начинали, а C и D смотрят на одни и те же клетки."""
    inhabited = features.loc[features["population"] > 0, "cell_id"].to_numpy()
    order = np.random.default_rng(seed).permutation(inhabited)
    return order[:count].tolist()


def _score_variant(key: str, cell_ids, inputs, cfg: dict, run_dir: Path,
                   costs: CostLog, logger) -> tuple[np.ndarray | None, dict]:
    """Скоры одного варианта. Сбой не роняет город — вариант станет failed."""
    variant = SCORED_VARIANTS[key]
    mode = "text" if key == "text" else "image"
    # кэш должен лежать рядом с raw/ и runs/, а не в текущем каталоге:
    # иначе запуск из другой папки заново тратит тысячи вызовов модели.
    # Наборы снимков делят один кэш: в его ключе байты снимка, они не смешаются
    from .. import settings as settings_mod
    cache = (settings_mod.data_root(cfg)
             / cfg["paths"].get("vlm_cache", "cache/vlm") / mode)
    try:
        with costs.stage("scoring", f"variant:{variant}") as extra:
            scores, report = scoring_client.score_cells(
                cell_ids, inputs, mode, cfg, cache, run_dir / f"vlm_raw_{key}")
            if report.failures:
                # в отчёт идут первые 20 причин, а разбирать сбои нужно по всем
                (run_dir / f"vlm_failures_{key}.txt").write_text(
                    "\n".join(report.failures) + "\n", encoding="utf-8")
            scoring_client.check_imputed(report, cfg)
            extra.update(report.as_dict())
        scores.to_parquet(run_dir / f"scores_{key}.parquet")
        logger.info(f"скоринг {variant}", **report.as_dict())
        return scores["score"].to_numpy(), report.as_dict()
    except Exception as error:
        logger.error(f"скоринг {variant} не удался",
                     reason=f"{type(error).__name__}: {error}")
        return None, {"status": "failed", "reason": str(error)}


def run_city(city: str, cfg: dict, protocol_path: Path | None = None,
             with_image: bool = False, mode: str = "onboard_city") -> Path:
    """Полный прогон города. Возвращает папку прогона."""
    guard_test_city(cfg, protocol_path)
    costs = CostLog(mode=mode)

    with Run.start(city, "run-city", cfg) as run:
        logger = RunLogger(run.run_id, run.dir / "log.jsonl")
        logger.info("старт", city=city, role=cfg["city"].get("role"))
        if protocol_path is not None:
            run.record_input(protocol_path)

        data = prepare_city(city, cfg, run.dir, logger, costs)
        for path in data.save(run.dir).values():
            run.record_output(path)

        scores: dict[str, np.ndarray] = {}
        scoring_reports: dict[str, dict] = {}

        cell_ids = data.features["cell_id"].tolist()
        texts = prompt_mod.describe_all(data.features).tolist()
        values, report = _score_variant("text", cell_ids, texts, cfg, run.dir, costs,
                                        logger)
        scoring_reports["text"] = report
        if values is not None:
            scores["text"] = values

        if with_image:
            for key, image_set in _image_sets(cfg).items():
                images = _load_images(_imagery_dir(cfg, city, image_set), cell_ids)
                if images is None:
                    scoring_reports[key] = {
                        "status": "missing", "reason": "снимки не загружены (п. 5, Л6)"}
                    logger.warn(f"вариант {SCORED_VARIANTS[key]} пропущен: нет снимков")
                    continue
                values, report = _score_variant(key, cell_ids, images, cfg, run.dir,
                                                costs, logger)
                scoring_reports[key] = report
                if values is not None:
                    scores[key] = values
        else:
            scoring_reports["image"] = {
                "status": "skipped", "reason": "запуск без --with-image"}

        evaluation = run_variants(data, scores, cfg, costs, logger)
        _save_solutions(evaluation, run.dir)

        costs_path = costs.save(run.dir / "costs.json")
        run.record_output(costs_path)
        from .. import settings as settings_mod
        paths = report_mod.build(evaluation, data.report, cfg, run.dir,
                                 costs=json.loads(costs_path.read_text("utf-8")),
                                 scoring=scoring_reports,
                                 human_log=settings_mod.data_root(cfg) / "human_log.csv")
        for path in paths.values():
            run.record_output(path)
        logger.info("готово", seconds=round(costs.total(), 1))
        logger.close()
    return run.dir


def _save_solutions(evaluation, run_dir: Path) -> None:
    payload = {"variants": [v.as_dict() for v in evaluation.variants]}
    (run_dir / "solutions.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    random_result = next((v for v in evaluation.variants if v.name == "A"), None)
    if random_result is not None and random_result.distribution is not None:
        pd.DataFrame({"dCov": random_result.distribution}).to_parquet(
            run_dir / "random_A.parquet")


def _image_sets(cfg: dict) -> dict[str, str | None]:
    """Какие снимки в какой скор: основной набор — D, `low_set` — D на 10 м.
    Без раздела imagery снимки лежат одной папкой, и D считается один раз."""
    rules = cfg["city"].get("imagery")
    if not rules:
        return {"image": None}
    sets = {"image": rules["main_set"]}
    if rules.get("low_set"):
        sets["image_10m"] = rules["low_set"]
    return sets


def _imagery_dir(cfg: dict, city: str, image_set: str | None = None) -> Path:
    """Где лежат снимки города: raw/<город>/<срез>/imagery/<cell_id>.png.

    Снимки — сырые данные того же среза, что OSM и население, поэтому живут
    рядом с ними. Папка своя у каждого города: cell_id нумеруются внутри
    города, и общая папка смешала бы клетки разных городов. Если у города
    несколько наборов (D на двух разрешениях), у каждого своя папка внутри;
    без явного выбора берётся основной набор из конфига."""
    from .. import settings as settings_mod
    folder = settings_mod.raw_dir(cfg, city, cfg["snapshot_date"]) / "imagery"
    rules = cfg["city"].get("imagery")
    if not rules:
        return folder
    name = image_set or rules["main_set"]
    if name not in rules["sets"]:
        raise settings_mod.ConfigError(
            f"нет набора снимков {name!r}; есть {list(rules['sets'])}")
    return folder / name


def _load_images(folder: Path, cell_ids) -> list[Path] | None:
    """Пути к снимкам клеток, если все они загружены (п. 5). Иначе None.

    Файлы лежат в folder по одному PNG на клетку, обрезанные ровно по её
    квадрату. Нет хотя бы одного — вариант D не считается целиком, частичный D
    сравнивать нельзя. Возвращаются пути, а не байты: снимки Мадрида занимают
    4 ГБ, и клиент модели читает каждый в момент вызова."""
    if not folder.is_dir():
        return None
    images = []
    for cell_id in cell_ids:
        path = folder / f"{cell_id}.png"
        if not path.is_file():
            return None
        images.append(path)
    return images
