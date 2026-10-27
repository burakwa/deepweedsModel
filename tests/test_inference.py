"""inference.py testleri: Predictor çıkarım yolları ve CSV yazımı.

Gerçek checkpoint yerine conftest'teki sentetik (1M param) checkpoint
kullanılır; ayrıca gerçek checkpoint varsa yüklenebilirliği de denenir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from config import CLASS_NAMES
from inference import Predictor, write_predictions_csv

REAL_CHECKPOINT = (
    "runs/exp-effb3-sqrt/weights/best.pt"
)


def _rand_image(size: tuple[int, int] = (60, 90)) -> Image.Image:
    arr = np.random.randint(0, 255, (size[0], size[1], 3), dtype=np.uint8)
    return Image.fromarray(arr)


# ------------------------------------------------------------------ yükleme
def test_from_checkpoint_metadata(tiny_checkpoint_path, tiny_predictor):
    assert tiny_predictor.img_size == 96
    assert list(tiny_predictor.class_names) == list(CLASS_NAMES)
    assert tiny_predictor.device.type == "cpu"
    assert tiny_predictor.size_bytes == tiny_checkpoint_path.stat().st_size


@pytest.mark.slow
def test_real_checkpoint_loads():
    real = Predictor.from_checkpoint(REAL_CHECKPOINT, device="cpu")
    assert real.img_size == 256
    assert len(real.class_names) == 9


# ------------------------------------------------------------------ tahmin
def test_predict_pil_probs(tiny_predictor):
    probs = tiny_predictor.predict_pil(_rand_image())
    assert probs.shape == (9,)
    assert abs(probs.sum() - 1.0) < 1e-4
    assert int(probs.argmax()) in range(9)


def test_predict_batch_probs(tiny_predictor):
    out = tiny_predictor.predict_batch([_rand_image() for _ in range(5)], batch_size=2)
    assert out.shape == (5, 9)
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-4)


def test_predict_batch_empty(tiny_predictor):
    out = tiny_predictor.predict_batch([])
    assert out.shape == (0, 9)


def test_predict_tta_probs(tiny_predictor):
    probs = tiny_predictor.predict_tta(_rand_image())
    assert probs.shape == (9,)
    assert abs(probs.sum() - 1.0) < 1e-4


def test_predict_df_labels_and_probs(tiny_predictor, tmp_path):
    paths = []
    for i in range(3):
        p = tmp_path / f"i{i}.jpg"
        _rand_image().save(p)
        paths.append(str(p))
    df = pd.DataFrame({"_path": paths, "label": [0, 3, 8]})
    res = tiny_predictor.predict_df(df, batch_size=2)
    assert res.y_true.tolist() == [0, 3, 8]
    assert res.y_prob.shape == (3, 9)
    assert np.allclose(res.y_prob.sum(axis=1), 1.0, atol=1e-4)


# --------------------------------------------------------------- CSV yazımı
def test_write_predictions_csv_format(tiny_predictor, tmp_path):
    probs = np.tile(np.linspace(0.5, 1.5, 9), (2, 1))
    probs = probs / probs.sum(axis=1, keepdims=True)
    ids = pd.Series(["a.jpg", "b.jpg"])
    out = write_predictions_csv(tmp_path / "sub.csv", ids, probs, 11.5, 123)
    assert len(out) == 2 and out["id"].is_unique
    prob_cols = [f"prob_{j}" for j in range(9)]
    assert np.allclose(out[prob_cols].sum(axis=1), 1.0, atol=1e-4)
    assert (out["bytes"] == 123).all()
    assert out["latency_ms"].iloc[0] == 11.5


def test_write_predictions_csv_rejects_duplicate_ids(tiny_predictor, tmp_path):
    probs = torch.softmax(torch.randn(2, 9), dim=1).numpy()
    with pytest.raises(AssertionError):
        write_predictions_csv(
            tmp_path / "bad.csv", pd.Series(["a.jpg", "a.jpg"]), probs, 1.0, 1
        )
