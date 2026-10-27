"""main.py katalog ve renderer testleri: SpeciesCatalog, ThemeRenderer."""

from __future__ import annotations

import numpy as np
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from config import CLASS_NAMES
from main import (
    CLASS_COLORS_RGB,
    NEGATIVE_CLASS,
    ModelService,
    SpeciesCatalog,
    ThemeRenderer,
    VideoResult,
)


# ---------------------------------------------------------- SpeciesCatalog
def test_catalog_covers_all_classes():
    for name in CLASS_NAMES:
        info = SpeciesCatalog.get_info(name)
        assert info["tr_name"] and info["scientific"] and info["family"]
        assert len(info["desc"]) > 10


def test_catalog_unknown_class_fallback():
    info = SpeciesCatalog.get_info("Tanimsiz Tur")
    assert info["tr_name"] == "Tanimsiz Tur"


def test_short_name_negative():
    assert SpeciesCatalog.short_name(NEGATIVE_CLASS) == "Temiz Arazi"


def test_index_of_matches_config_order():
    for i, name in enumerate(CLASS_NAMES):
        assert SpeciesCatalog.index_of(name) == i


def test_class_colors_match_class_count():
    assert len(CLASS_COLORS_RGB) == len(CLASS_NAMES)


# ------------------------------------------------------------ ThemeRenderer
def test_get_leaf_b64_reads_local_asset():
    b64 = ThemeRenderer.get_leaf_b64()
    assert len(b64) > 1000  # gerçek görsel yüklendi
    import base64

    raw = base64.b64decode(b64)
    assert raw[:3] == b"\xff\xd8\xff"  # JPEG imzası


def test_fallback_leaf_is_local_data_uri():
    uri = ThemeRenderer.fallback_leaf_data_uri()
    assert uri.startswith("data:image/png;base64,")
    assert "http" not in uri


def test_compact_html_collapses_whitespace():
    html = "<div>\n   <p> a </p>\n</div>"
    out = ThemeRenderer._compact_html(html)
    assert "\n" not in out and "  " not in out
    assert out == "<div><p> a </p></div>"


def test_image_to_b64_roundtrip():
    from PIL import Image

    img = Image.new("RGB", (10, 10), color=(10, 200, 30))
    b64 = ThemeRenderer.image_to_b64(img)
    assert len(b64) > 100


def test_build_probs_html_top4():
    res = VideoResult(
        top_class="Lantana",
        confidence=50.0,
        probs=np.array([0.5, 0.2, 0.1, 0.05, 0.05, 0.04, 0.03, 0.02, 0.01], dtype=np.float32),
        latency_ms=1.0,
        frames_analyzed=1,
        duration_sec=1.0,
        tr_name="a",
        scientific="b",
        family="c",
        description="d",
    )
    html = ThemeRenderer.build_probs_html(res, CLASS_NAMES, "test")
    assert "test" in html
    assert "%50.0" in html
    assert html.count("prob-row") == 4


def test_build_counts_html_empty():
    res = VideoResult(
        top_class=NEGATIVE_CLASS,
        confidence=0.0,
        probs=np.full(9, 1 / 9, dtype=np.float32),
        latency_ms=1.0,
        frames_analyzed=1,
        duration_sec=1.0,
        tr_name="a",
        scientific="b",
        family="c",
        description="d",
        species_counts={},
    )
    html = ThemeRenderer.build_counts_html(res)
    assert "bulunamadı" in html


# -------------------------------------------------------------- ModelService
def test_model_service_arch_label(tiny_service):
    label = tiny_service.arch_label
    assert isinstance(label, str) and label


def test_model_service_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ModelService(tmp_path / "olmayan" / "best.pt")


def test_model_service_img_size(tiny_service):
    assert tiny_service.img_size == 96


def test_model_service_device_name(tiny_service):
    assert isinstance(tiny_service.device_name, str)


# --------------------------------------------------------- st bağımlılıksız
def test_streamlit_import_version():
    assert st.__version__


@pytest.mark.slow
def test_full_app_renders_without_exception():
    """Uygulamayı gerçek checkpoint ile uçtan uca çalıştırır."""
    at = AppTest.from_file("main.py", default_timeout=300)
    at.run()
    assert len(at.exception) == 0, [e.value for e in at.exception]
    assert len(at.error) == 0, [e.value for e in at.error]
    assert len(at.button) >= 9
