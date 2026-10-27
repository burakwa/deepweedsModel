"""data.py testleri: transformlar, join_path, dataset, resolve_paths."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from config import NUM_CLASSES
from data import (
    DeepWeedsDataset,
    build_eval_transform,
    build_tta_transforms,
    build_train_transform,
    class_counts,
    join_path,
    resolve_paths,
)

IMG_SIZE = 96


def _rand_image(size: tuple[int, int] = (64, 80)) -> Image.Image:
    arr = np.random.randint(0, 255, (size[0], size[1], 3), dtype=np.uint8)
    return Image.fromarray(arr)


# ---------------------------------------------------------------- transformlar
def test_eval_transform_shape_and_normalization():
    tf = build_eval_transform(IMG_SIZE)
    t = tf(_rand_image())
    assert t.shape == (3, IMG_SIZE, IMG_SIZE)
    assert -3.0 <= float(t.min()) and float(t.max()) <= 3.0


def test_train_transform_shape():
    t = build_train_transform(IMG_SIZE)(_rand_image())
    assert t.shape == (3, IMG_SIZE, IMG_SIZE)


def test_tta_transforms_count_and_shape():
    tta = build_tta_transforms(IMG_SIZE)
    assert len(tta) == 5
    for tf in tta:
        t = tf(_rand_image())
        assert t.shape == (3, IMG_SIZE, IMG_SIZE)


def test_tta_transforms_are_deterministic():
    tta = build_tta_transforms(IMG_SIZE)
    img = _rand_image()
    for tf in tta:
        assert torch.equal(tf(img), tf(img))


# ------------------------------------------------------------------ join_path
def test_join_path_relative_and_absolute(tmp_path):
    assert join_path("data", "img/a.jpg") == str(Path("data") / "img" / "a.jpg")
    abs_p = tmp_path / "a.jpg"
    assert join_path("data", str(abs_p)) == str(abs_p)


# ------------------------------------------------------------------ dataset
def _make_df_with_images(tmp_path, n: int = 3) -> pd.DataFrame:
    paths = []
    for i in range(n):
        p = tmp_path / f"img_{i}.jpg"
        _rand_image().save(p)
        paths.append(str(p))
    return pd.DataFrame({"_path": paths, "label": list(range(n))})


def test_dataset_returns_tensor_and_label(tmp_path):
    df = _make_df_with_images(tmp_path)
    ds = DeepWeedsDataset(df, build_eval_transform(IMG_SIZE))
    assert len(ds) == 3
    x, y = ds[1]
    assert x.shape == (3, IMG_SIZE, IMG_SIZE)
    assert y == 1


def test_dataset_without_label_column_defaults_to_minus_one(tmp_path):
    df = _make_df_with_images(tmp_path).drop(columns=["label"])
    ds = DeepWeedsDataset(df, None)
    _, y = ds[0]
    assert y == -1


def test_dataset_missing_path_column_raises():
    with pytest.raises(ValueError, match="kolon"):
        DeepWeedsDataset(pd.DataFrame({"label": [0]}))


# --------------------------------------------------------------- resolve_paths
def test_resolve_paths_builds_path_column(tmp_path):
    df = pd.DataFrame({"image": ["a.jpg", "b.jpg"], "label": [0, 1]})
    out = resolve_paths(df, data_dir=str(tmp_path), image_dir=None, image_col="image")
    assert "_path" in out.columns
    assert len(out) == 2
    assert all(p.endswith(".jpg") for p in out["_path"])


def test_resolve_paths_existing_path_column_is_kept(tmp_path):
    df = pd.DataFrame({"_path": ["zaten/var.jpg"]})
    out = resolve_paths(df, data_dir=str(tmp_path), image_dir=None, image_col=None)
    assert out["_path"].iloc[0] == "zaten/var.jpg"


# --------------------------------------------------------------- class_counts
def test_class_counts_and_negative_ignored():
    counts = class_counts([0, 0, 1, -1, 8], NUM_CLASSES)
    assert counts[0] == 2 and counts[1] == 1 and counts[8] == 1
    assert counts.sum() == 4
