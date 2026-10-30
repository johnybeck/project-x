"""Снимок клетки по WMS: одна клетка — один запрос GetMap (PNOA, Испания).

Сетка Мадрида лежит в EPSG:25830, как и снимки IGN, поэтому клетка — осевой
квадрат в системе снимка: сервер отдаёт ровно её, без склейки и
перепроецирования (MADRID_DATA п. 4). Формат — PNG: артефакты JPEG подмешались
бы к разрешению, а оно здесь переменная исследования.

Формат ответов посмотрен на настоящем сервере: GetFeatureInfo по слою мозаики
отдаёт дату съёмки и разрешение в полях Fecha и Resolucion, а ошибку WMS может
прислать с кодом 200, поэтому тип ответа проверяется всегда.
"""
from __future__ import annotations

import http.client
import json
import time
import urllib.parse
import urllib.request

_HEADERS = {"User-Agent": "seto-research/1.0 (university project)"}
_ATTEMPTS = 4
_PAUSE_S = 5
_INFO_PIXELS = 101        # окно GetFeatureInfo; середина — центр клетки


class WmsError(RuntimeError):
    """Сервер не отдал снимок или ответил не тем, что просили."""


def _query(rules: dict, request: str, bounds: tuple, side_px: int) -> dict:
    return {"SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": request,
            "CRS": f"EPSG:{rules['crs']}",
            "BBOX": ",".join(f"{value:.2f}" for value in bounds),
            "WIDTH": side_px, "HEIGHT": side_px, "STYLES": ""}


def map_url(rules: dict, bounds: tuple, side_px: int) -> str:
    """GetMap ровно на квадрат клетки."""
    query = {**_query(rules, "GetMap", bounds, side_px),
             "LAYERS": rules["layer"], "FORMAT": "image/png"}
    return f"{rules['url']}?{urllib.parse.urlencode(query)}"


def info_url(rules: dict, bounds: tuple) -> str:
    """GetFeatureInfo в центре клетки: дата съёмки и разрешение."""
    middle = _INFO_PIXELS // 2
    query = {**_query(rules, "GetFeatureInfo", bounds, _INFO_PIXELS),
             "LAYERS": rules["info_layer"], "QUERY_LAYERS": rules["info_layer"],
             "FORMAT": "image/png", "INFO_FORMAT": "application/json",
             "I": middle, "J": middle}
    return f"{rules['url']}?{urllib.parse.urlencode(query)}"


def _get(url: str) -> tuple[bytes, str]:
    """Тело и тип ответа. Сервер общий, поэтому после сбоя — пауза, а не новый
    запрос сразу; пауза растёт, чтобы не добивать его при перегрузке."""
    request = urllib.request.Request(url, headers=_HEADERS)
    for attempt in range(_ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read(), response.headers.get("Content-Type", "")
        except (OSError, http.client.HTTPException) as error:
            if attempt == _ATTEMPTS - 1:
                raise WmsError(f"сервер не ответил: {error}") from error
            time.sleep(_PAUSE_S * 2 ** attempt)
    raise AssertionError("цикл попыток вышел без результата")


def fetch_map(rules: dict, bounds: tuple, side_px: int) -> bytes:
    """PNG клетки."""
    body, kind = _get(map_url(rules, bounds, side_px))
    if not kind.startswith("image/png"):
        raise WmsError(f"вместо снимка пришло {kind}: {body[:200]!r}")
    return body


def fetch_info(rules: dict, bounds: tuple) -> dict:
    """Дата съёмки и разрешение в центре клетки."""
    body, kind = _get(info_url(rules, bounds))
    if "json" not in kind:
        raise WmsError(f"вместо сведений о съёмке пришло {kind}: {body[:200]!r}")
    features = json.loads(body).get("features") or []
    if not features:
        raise WmsError(f"нет сведений о съёмке в клетке {bounds}")
    properties = features[0]["properties"]
    return {"capture_date": properties["Fecha"],
            "gsd_m": float(properties["Resolucion"])}
