"""Снимки клеток на диск: исходник, вход модели и манифест набора (пропозал v2, п. 3.8).

D считается дважды, на высоком разрешении и на 10 м, поэтому набор снимков
называется в конфиге (`imagery.sets`). Раскладка внутри raw/<город>/<срез>/imagery:

    <набор>/<cell_id>.png          вход модели, ровно квадрат клетки
    <набор>/source/                исходники: PNG по клеткам или целая сцена
    <набор>/manifest.csv           по строке на клетку: дата, разрешение, хеши

Исходники хранятся, потому что WMS показывает последнюю съёмку: повторный запрос
через год может вернуть другой снимок той же клетки.

Частичный набор не выдаётся за полный: клетка, которую не удалось получить,
пишется в отчёт, а вариант D без неё не запустится (`runner._load_images`).
"""
from __future__ import annotations

import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from PIL import Image

from ..acquire.download import Source, fetch
from ..run import sha256_of
from . import source as source_mod
from . import wms
from .tiles import quality_flags

_MANIFEST = "manifest.csv"


@dataclass
class ImageSet:
    """Что получилось по набору снимков."""
    name: str
    rows: pd.DataFrame
    failed: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"set": self.name, "cells": len(self.rows), "failed": len(self.failed),
                "input_mb": round(float(self.rows["input_bytes"].sum()) / 1e6, 1),
                "source_mb": round(float(self.rows["source_bytes"].sum()) / 1e6, 1),
                "capture_dates": self.rows["capture_date"].value_counts().to_dict(),
                "gsd_m": self.rows["gsd_m"].value_counts().to_dict(),
                "max_blank_share": float(self.rows["blank_share"].max())}


def side_px(bounds: tuple, gsd_m: float) -> int:
    """Сколько пикселей снимка приходится на сторону клетки."""
    return int(round((bounds[2] - bounds[0]) / gsd_m))


def model_input(pixels: np.ndarray, size_px: int) -> bytes:
    """Вход модели: один размер у всех наборов, PNG без потерь (RESOLUTION_STUDY).

    Один фильтр на оба направления. BOX при уменьшении усредняет по площади, как
    сенсор с крупным пикселем, а при увеличении оставляет те же квадраты: снимок
    10 м не получает сглаженных деталей, которых в нём нет."""
    image = Image.fromarray(pixels).resize((size_px, size_px), Image.BOX)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _decode(data: bytes, side: int) -> np.ndarray:
    """PNG в RGB-массив. Снимок не того размера — не снимок этой клетки."""
    image = Image.open(io.BytesIO(data))
    if image.size != (side, side):
        raise ValueError(f"снимок {image.size} вместо {side}×{side}")
    return np.asarray(image.convert("RGB"))


def _blank_share(pixels: np.ndarray) -> float:
    """Доля чисто чёрных и чисто белых пикселей: так выглядит отсутствие данных."""
    return float(((pixels == 0).all(axis=-1) | (pixels == 255).all(axis=-1)).mean())


def _save(path: Path, payload: bytes) -> None:
    """Запись через временный файл: обрыв не оставит битый файл под нужным именем."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    partial.write_bytes(payload)
    partial.replace(path)


def _input_row(cell_id: str, pixels: np.ndarray, folder: Path, input_px: int,
               details: dict) -> dict:
    """Проверить снимок, записать вход модели и вернуть строку манифеста."""
    flags = quality_flags(pixels)
    if not flags["ok"]:
        raise ValueError(f"снимок не годится: {flags['image_qc']}")
    payload = model_input(pixels, input_px)
    _save(folder / f"{cell_id}.png", payload)
    return {"cell_id": cell_id, **details, "input_sha256": hashlib.sha256(payload).hexdigest(),
            "input_bytes": len(payload), "blank_share": _blank_share(pixels)}


def _wms_cell(cell_id: str, bounds: tuple, rules: dict, folder: Path, input_px: int) -> dict:
    """Одна клетка. Готовую не трогаем: обрыв на 2000-й не должен стоить первых 1999."""
    done = folder / "source" / f"{cell_id}.json"
    if done.is_file() and (folder / f"{cell_id}.png").is_file():
        return json.loads(done.read_text(encoding="utf-8"))

    side = side_px(bounds, rules["gsd_m"])
    info = wms.fetch_info(rules, bounds)
    data = wms.fetch_map(rules, bounds, side)
    pixels = _decode(data, side)              # до записи: битый ответ не ляжет как готовый
    _save(folder / "source" / f"{cell_id}.png", data)
    row = _input_row(cell_id, pixels, folder, input_px,
                     {**info, "source_sha256": hashlib.sha256(data).hexdigest(),
                      "source_bytes": len(data),
                      "acquired_at": datetime.now(timezone.utc).isoformat()})
    _save(done, json.dumps(row, ensure_ascii=False).encode("utf-8"))
    return row


def acquire_wms(name: str, cells: gpd.GeoDataFrame, rules: dict, folder: Path,
                input_px: int, say=lambda *a, **k: None) -> ImageSet:
    """Набор снимков из WMS, с ограничением параллельности: сервер не наш.

    Подряд идущие сбои означают, что сервер лежит: тогда останавливаемся, а не
    перебираем весь город с паузами. Повторный запуск продолжит с места обрыва."""
    in_a_row = {"count": 0}
    limit = rules.get("stop_after_failures", 12)

    def one(cell) -> tuple[str, dict | None, str | None]:
        cell_id, bounds = cell
        if in_a_row["count"] >= limit:
            return cell_id, None, "остановлено после сбоев подряд"
        try:
            row = _wms_cell(cell_id, bounds, rules, folder, input_px)
            in_a_row["count"] = 0
            return cell_id, row, None
        except (wms.WmsError, ValueError, OSError) as error:
            in_a_row["count"] += 1
            return cell_id, None, str(error)

    rows, failed = [], {}
    todo = list(zip(cells["cell_id"], cells.geometry.bounds.itertuples(index=False, name=None)))
    with ThreadPoolExecutor(max_workers=rules.get("max_parallel", 2)) as pool:
        for n, (cell_id, row, error) in enumerate(pool.map(one, todo), 1):
            if row is None:
                failed[cell_id] = error
            else:
                rows.append(row)
            if n % 100 == 0:
                say("снимки", set=name, done=n, total=len(todo), failed=len(failed))
    return ImageSet(name, pd.DataFrame(rows), failed)


def acquire_scene(name: str, cells: gpd.GeoDataFrame, rules: dict, folder: Path,
                  input_px: int, say=lambda *a, **k: None) -> ImageSet:
    """Набор из одной сцены: сцена качается один раз, клетки режутся с диска."""
    scene = folder / "source" / rules["file"]
    source = fetch(rules["url"], scene, name=f"imagery_{name}", license_note=rules["license"])
    crs = f"EPSG:{cells.crs.to_epsg()}"
    rows, failed = [], {}
    bounds_all = cells.geometry.bounds.itertuples(index=False, name=None)
    for n, (cell_id, bounds) in enumerate(zip(cells["cell_id"], bounds_all), 1):
        try:
            pixels = source_mod.read_cell_warped(str(scene), bounds, side_px(bounds, rules["gsd_m"]),
                                                 crs, rules["gsd_m"])
            rows.append(_input_row(cell_id, pixels, folder, input_px, {
                "capture_date": rules["capture_date"], "gsd_m": rules["gsd_m"],
                "source_sha256": source.sha256, "source_bytes": 0,
                "acquired_at": source.downloaded_at}))
        except (source_mod.CoverageError, ValueError) as error:
            failed[cell_id] = str(error)
        if n % 500 == 0:
            say("снимки", set=name, done=n, total=len(cells), failed=len(failed))
    return ImageSet(name, pd.DataFrame(rows), failed)


def save_manifest(image_set: ImageSet, folder: Path) -> Path:
    """Манифест набора: по строке на клетку."""
    path = folder / _MANIFEST
    image_set.rows.sort_values("cell_id").to_csv(path, index=False, encoding="utf-8")
    return path


def record_source(data_manifest: Path, image_set: ImageSet, rules: dict,
                  manifest: Path, url: str) -> None:
    """Записать готовый набор в манифест данных среза, не трогая остальные
    источники: `acquire` переписал бы им даты загрузки."""
    payload = json.loads(data_manifest.read_text(encoding="utf-8"))
    entry = Source(name=f"imagery_{image_set.name}", url=url, path=str(manifest),
                   sha256=sha256_of(manifest), size_bytes=manifest.stat().st_size,
                   downloaded_at=datetime.now(timezone.utc).isoformat(),
                   license=rules["license"],
                   note=f"{len(image_set.rows)} клеток; {rules['attribution']}")
    payload["sources"] = [s for s in payload["sources"] if s["name"] != entry.name]
    payload["sources"].append(entry.as_dict())
    data_manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                             encoding="utf-8")
