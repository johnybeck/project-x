"""Тесты загрузки снимков: сервер подменён, ответы взяты с настоящего WMS IGN."""
import io
import json
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from PIL import Image
from rasterio.transform import from_origin
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seto.experiment import runner  # noqa: E402
from seto.imagery import acquire as A, source as S, wms as W  # noqa: E402
from seto.settings import ConfigError  # noqa: E402

RULES = {"url": "https://example.test/wms", "layer": "OI.OrthoimageCoverage",
         "info_layer": "OI.MosaicElement", "crs": 25830, "gsd_m": 0.25, "max_parallel": 2,
         "license": "CC BY 4.0", "attribution": "Obra derivada de PNOA 2023"}
BOUNDS = (440000.0, 4474000.0, 440500.0, 4474500.0)
INFO = b"""{"type": "FeatureCollection", "features": [{"type": "Feature", "id": "",
  "geometry": null, "properties": {"Fecha": "2023-07", "Resolucion": "0.25"}}]}"""


def _png(side=2000, seed=0) -> bytes:
    pixels = np.random.default_rng(seed).integers(1, 250, (side, side, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue()


def _query(url: str) -> dict:
    return {key.upper(): value[0] for key, value in parse_qs(
        urlparse(url).query, keep_blank_values=True).items()}


def _cells(*bounds):
    return gpd.GeoDataFrame({"cell_id": [f"cell_{i:06d}" for i in range(len(bounds))]},
                            geometry=[box(*b) for b in bounds], crs=25830)


@pytest.fixture
def server(monkeypatch):
    """Подменить сервер; calls считает обращения за снимком."""
    state = {"calls": 0, "png": _png()}

    def answer(url):
        if _query(url)["REQUEST"] == "GetFeatureInfo":
            return INFO, "application/json"
        state["calls"] += 1
        return state["png"], "image/png"

    monkeypatch.setattr(W, "_get", answer)
    return state


# --- запросы к WMS ---

def test_map_request_asks_for_exactly_the_cell():
    query = _query(W.map_url(RULES, BOUNDS, 2000))
    assert query["BBOX"] == "440000.00,4474000.00,440500.00,4474500.00"
    assert (query["WIDTH"], query["HEIGHT"]) == ("2000", "2000")
    assert query["CRS"] == "EPSG:25830"
    assert query["FORMAT"] == "image/png"
    assert query["LAYERS"] == "OI.OrthoimageCoverage"


def test_info_request_asks_about_the_centre_of_the_cell():
    query = _query(W.info_url(RULES, BOUNDS))
    assert query["QUERY_LAYERS"] == "OI.MosaicElement"
    assert query["INFO_FORMAT"] == "application/json"
    assert query["I"] == query["J"] == str(int(query["WIDTH"]) // 2)


def test_capture_date_is_read_from_the_real_answer(server):
    assert W.fetch_info(RULES, BOUNDS) == {"capture_date": "2023-07", "gsd_m": 0.25}


def test_no_capture_info_is_an_error(monkeypatch):
    monkeypatch.setattr(W, "_get", lambda url: (b'{"features": []}', "application/json"))
    with pytest.raises(W.WmsError, match="нет сведений"):
        W.fetch_info(RULES, BOUNDS)


def test_service_error_with_code_200_is_not_an_image(monkeypatch):
    """WMS может прислать ошибку XML-документом с кодом 200."""
    monkeypatch.setattr(W, "_get", lambda url: (b"<ServiceException/>", "text/xml"))
    with pytest.raises(W.WmsError, match="вместо снимка"):
        W.fetch_map(RULES, BOUNDS, 2000)


def test_failed_request_is_repeated_with_a_pause(monkeypatch):
    attempts, pauses = [], []

    def refuse(request, timeout):
        attempts.append(request)
        raise OSError("обрыв")

    monkeypatch.setattr(W.urllib.request, "urlopen", refuse)
    monkeypatch.setattr(W.time, "sleep", pauses.append)
    with pytest.raises(W.WmsError, match="не ответил"):
        W._get("https://example.test/wms")
    assert len(attempts) == 4
    assert pauses == sorted(pauses) and len(pauses) == 3


# --- вход модели ---

def test_cell_side_in_pixels():
    assert A.side_px(BOUNDS, 0.25) == 2000
    assert A.side_px(BOUNDS, 10) == 50


def test_input_has_the_size_of_the_encoder():
    pixels = np.zeros((2000, 2000, 3), dtype=np.uint8)
    assert Image.open(io.BytesIO(A.model_input(pixels, 896))).size == (896, 896)


def test_shrinking_averages_over_the_area():
    """Шахматка 1×1 после уменьшения — ровный серый, а не случайный пиксель."""
    pixels = np.zeros((1024, 1024, 3), dtype=np.uint8)
    pixels[::2, ::2] = pixels[1::2, 1::2] = 200
    result = np.asarray(Image.open(io.BytesIO(A.model_input(pixels, 512))))
    assert result.min() == result.max() == 100


def test_enlarging_keeps_the_same_coarse_pixels():
    """10 м остаются квадратами: деталей, которых нет в снимке, не появляется."""
    pixels = np.zeros((2, 2, 3), dtype=np.uint8)
    pixels[0, 0] = 200
    result = np.asarray(Image.open(io.BytesIO(A.model_input(pixels, 8))))
    assert set(np.unique(result)) == {0, 200}
    assert (result[:4, :4] == 200).all() and (result[4:, :] == 0).all()


def test_input_is_lossless():
    pixels = np.random.default_rng(1).integers(0, 255, (32, 32, 3), dtype=np.uint8)
    assert (np.asarray(Image.open(io.BytesIO(A.model_input(pixels, 32)))) == pixels).all()


# --- набор из WMS ---

def test_cell_gets_source_input_and_manifest_row(server, tmp_path):
    result = A.acquire_wms("pnoa", _cells(BOUNDS), RULES, tmp_path, 896)
    assert result.failed == {}
    row = result.rows.iloc[0]
    assert row["capture_date"] == "2023-07" and row["gsd_m"] == 0.25
    assert (tmp_path / "source" / "cell_000000.png").read_bytes() == server["png"]
    assert Image.open(tmp_path / "cell_000000.png").size == (896, 896)
    assert row["source_bytes"] == len(server["png"])


def test_second_run_does_not_download_again(server, tmp_path):
    """Обрыв на 2000-й клетке не должен стоить первых 1999."""
    first = A.acquire_wms("pnoa", _cells(BOUNDS), RULES, tmp_path, 896)
    second = A.acquire_wms("pnoa", _cells(BOUNDS), RULES, tmp_path, 896)
    assert server["calls"] == 1
    assert second.rows.to_dict() == first.rows.to_dict()


def test_wrong_size_is_refused_and_not_saved(server, tmp_path):
    server["png"] = _png(side=100)
    result = A.acquire_wms("pnoa", _cells(BOUNDS), RULES, tmp_path, 896)
    assert "вместо 2000" in result.failed["cell_000000"]
    assert not (tmp_path / "source" / "cell_000000.png").exists()


def test_blank_image_is_refused(server, tmp_path):
    buffer = io.BytesIO()
    Image.new("RGB", (2000, 2000), "white").save(buffer, format="PNG")
    server["png"] = buffer.getvalue()
    result = A.acquire_wms("pnoa", _cells(BOUNDS), RULES, tmp_path, 896)
    assert "не годится" in result.failed["cell_000000"]
    assert not (tmp_path / "cell_000000.png").exists()


def test_one_bad_cell_does_not_stop_the_others(monkeypatch, tmp_path):
    good = _png()

    def answer(url):
        query = _query(url)
        if query["REQUEST"] == "GetFeatureInfo":
            return INFO, "application/json"
        if query["BBOX"].startswith("440500"):
            return b"<ServiceException/>", "text/xml"
        return good, "image/png"

    monkeypatch.setattr(W, "_get", answer)
    cells = _cells(BOUNDS, (440500.0, 4474000.0, 441000.0, 4474500.0))
    result = A.acquire_wms("pnoa", cells, RULES, tmp_path, 896)
    assert len(result.rows) == 1 and list(result.failed) == ["cell_000001"]


def test_download_stops_when_the_server_is_down(monkeypatch, tmp_path):
    calls = []

    def down(url):
        calls.append(url)
        raise W.WmsError("сервер не ответил")

    monkeypatch.setattr(W, "_get", down)
    cells = _cells(*[(440000.0 + 500 * i, 4474000.0, 440500.0 + 500 * i, 4474500.0)
                     for i in range(50)])
    rules = {**RULES, "max_parallel": 1, "stop_after_failures": 3}
    result = A.acquire_wms("pnoa", cells, rules, tmp_path, 896)
    assert len(calls) == 3
    assert len(result.failed) == 50


# --- набор из сцены ---

def _scene(path: Path, value=120) -> Path:
    """Сцена в UTM WGS84, как Sentinel-2: пиксель 10 м, начало как у тайла 30TVK."""
    data = np.random.default_rng(3).integers(1, 250, (3, 600, 600), dtype=np.uint8)
    with rasterio.open(path, "w", driver="GTiff", height=600, width=600, count=3, dtype="uint8",
                       crs="EPSG:32630", transform=from_origin(439960, 4478000, 10, 10)) as dst:
        dst.write(data)
    return path


def test_scene_cell_covers_exactly_the_square_of_the_cell(tmp_path):
    """Сетка в ETRS89, сцена в WGS84: на картинке обязан быть ровно этот квадрат."""
    scene = _scene(tmp_path / "scene.tif")
    pixels = S.read_cell_warped(str(scene), BOUNDS, 50, "EPSG:25830", 10)
    with rasterio.open(scene) as src:
        col, row = int((440000 - 439960) / 10), int((4478000 - 4474500) / 10)
        expected = src.read(window=((row, row + 50), (col, col + 50)))
    assert (pixels == np.moveaxis(expected, 0, -1)).all()


def test_scene_set_is_cut_from_the_file_on_disk(monkeypatch, tmp_path):
    folder = tmp_path / "s2"
    scene = _scene(tmp_path / "scene.tif")

    def fake_fetch(url, target, name, license_note):
        target.parent.mkdir(parents=True)
        target.write_bytes(scene.read_bytes())
        return A.Source(name, url, str(target), "abc", 1, "2026-10-30T00:00:00+00:00")

    monkeypatch.setattr(A, "fetch", fake_fetch)
    rules = {"url": "https://example.test/TCI.tif", "file": "TCI.tif", "gsd_m": 10,
             "capture_date": "2023-08-01", "license": "открытая"}
    far = (500000.0, 4400000.0, 500500.0, 4400500.0)
    result = A.acquire_scene("s2", _cells(BOUNDS, far), rules, folder, 896)
    assert result.rows["capture_date"].tolist() == ["2023-08-01"]
    assert Image.open(folder / "cell_000000.png").size == (896, 896)
    assert "cell_000001" in result.failed


# --- манифесты и выбор набора ---

def test_data_manifest_keeps_other_sources(server, tmp_path):
    data_manifest = tmp_path / "data_manifest.json"
    data_manifest.write_text(json.dumps({"sources": [{"name": "osm_pbf"}]}), encoding="utf-8")
    result = A.acquire_wms("pnoa", _cells(BOUNDS), RULES, tmp_path, 896)
    manifest = A.save_manifest(result, tmp_path)
    for _ in range(2):        # повторная запись заменяет запись набора, а не дублирует
        A.record_source(data_manifest, result, RULES, manifest, RULES["url"])
    sources = json.loads(data_manifest.read_text(encoding="utf-8"))["sources"]
    assert [s["name"] for s in sources] == ["osm_pbf", "imagery_pnoa"]
    assert "Obra derivada de PNOA" in sources[1]["note"]


def _cfg(tmp_path, imagery=None):
    return {"paths": {"raw": "raw"}, "data_root": str(tmp_path), "snapshot_date": "2026-10-28",
            "city": {"imagery": imagery} if imagery else {}}


def test_main_set_is_used_by_default(tmp_path):
    cfg = _cfg(tmp_path, {"main_set": "pnoa", "sets": {"pnoa": {}, "s2": {}}})
    assert runner._imagery_dir(cfg, "madrid").name == "pnoa"
    assert runner._imagery_dir(cfg, "madrid", "s2").name == "s2"


def test_unknown_set_is_an_error(tmp_path):
    cfg = _cfg(tmp_path, {"main_set": "pnoa", "sets": {"pnoa": {}}})
    with pytest.raises(ConfigError, match="нет набора"):
        runner._imagery_dir(cfg, "madrid", "s3")


def test_city_without_sets_keeps_the_old_folder(tmp_path):
    assert runner._imagery_dir(_cfg(tmp_path), "luzern").name == "imagery"


def test_set_with_a_missing_cell_is_not_loaded(tmp_path):
    """Частичный D сравнивать нельзя."""
    (tmp_path / "cell_000000.png").write_bytes(b"x")
    assert runner._load_images(tmp_path, ["cell_000000", "cell_000001"]) is None


def test_images_are_paths_not_bytes(tmp_path):
    """Снимки Мадрида — 4 ГБ: в память их не грузим, клиент читает по одному."""
    for name in ("cell_000000", "cell_000001"):
        (tmp_path / f"{name}.png").write_bytes(b"x")
    paths = runner._load_images(tmp_path, ["cell_000000", "cell_000001"])
    assert paths == [tmp_path / "cell_000000.png", tmp_path / "cell_000001.png"]
