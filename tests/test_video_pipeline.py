"""main.py PlantDetector testleri: pencere ızgarası, IoU, NMS, video analizi."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from main import DEFAULT_WINDOW_SCALES, InferenceEngine, PlantDetector, _Box, _Track


@pytest.fixture
def detector(tiny_service) -> PlantDetector:
    return PlantDetector(tiny_service)


@pytest.fixture
def engine(tiny_service) -> InferenceEngine:
    return InferenceEngine(tiny_service)


def _mp4_bytes(path, frames: int = 20, w: int = 320, h: int = 240, fps: int = 10) -> bytes:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert writer.isOpened()
    for _ in range(frames):
        frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path.read_bytes()


# -------------------------------------------------------------- pencere ızgarası
def test_grid_windows_within_frame_bounds(detector):
    wins = detector._grid_windows(640, 360)
    assert wins, "pencere üretilmeli"
    assert len(wins) <= 26
    for x, y, w in wins:
        assert 0 <= x and 0 <= y
        assert x + w <= 640 and y + w <= 360


def test_grid_windows_minimum_size_filter(detector):
    # 90 piksel kare: tüm ölçekler 96 piksel altına düşer → pencere yok
    assert detector._grid_windows(90, 90) == []


# ----------------------------------------------------------------------- IoU
def test_iou_identical_and_disjoint():
    a = _Box(0, 0, 100, 100, "Lantana", 0.9)
    same = _Box(0, 0, 100, 100, "Lantana", 0.8)
    disjoint = _Box(200, 200, 300, 300, "Lantana", 0.7)
    d = PlantDetector.__new__(PlantDetector)
    assert d._iou(a, same) == 1.0
    assert d._iou(a, disjoint) == 0.0


# ----------------------------------------------------------------------- NMS
def test_nms_suppresses_same_class_overlap(detector):
    a = _Box(0, 0, 100, 100, "Lantana", 0.9)
    b = _Box(10, 10, 110, 110, "Lantana", 0.8)  # IoU ~0.68 → baskılanır
    kept = detector._nms([a, b])
    assert len(kept) == 1 and kept[0] is a


def test_nms_keeps_different_class_low_overlap(detector):
    a = _Box(0, 0, 100, 100, "Lantana", 0.9)
    c = _Box(80, 0, 180, 100, "Siam weed", 0.8)  # IoU ~0.11 → kalır
    kept = detector._nms([a, c])
    assert {b.class_name for b in kept} == {"Lantana", "Siam weed"}


# ------------------------------------------------------------ video analizi
def test_analyze_video_end_to_end(detector, tmp_path):
    vid = _mp4_bytes(tmp_path / "v.mp4")
    res = detector.analyze_video(vid, max_frames=4)
    assert res.frames_analyzed == 4
    assert res.duration_sec > 0
    assert res.probs.shape == (9,)
    assert abs(res.probs.sum() - 1.0) < 1e-4
    assert res.total_plants == sum(res.species_counts.values())
    assert res.top_class in ["Chinee apple", "Lantana", "Parkinsonia", "Parthenium",
                             "Prickly acacia", "Rubber vine", "Siam weed", "Snake weed",
                             "Negative"]
    for fr in res.frames:
        assert fr.jpeg[:2] == b"\xff\xd8"  # JPEG SOI imzası


def test_analyze_video_invalid_bytes_raises(detector):
    with pytest.raises(ValueError):
        detector.analyze_video(b"video-degil", max_frames=2)


def test_engine_predict_video_wiring(engine, tmp_path):
    res = engine.predict_video(_mp4_bytes(tmp_path / "v2.mp4"), max_frames=3)
    assert res.frames_analyzed == 3
    assert hasattr(res, "species_counts")


def test_track_pick_class_votes_majority():
    t = _Track(box=_Box(0, 0, 10, 10, "Lantana", 0.9))
    t.vote["Lantana"] += 2
    t.vote["Siam weed"] += 1
    assert t.pick_class() == "Lantana"


def test_default_window_scales_valid():
    assert all(0 < s < 1 for s in DEFAULT_WINDOW_SCALES)
