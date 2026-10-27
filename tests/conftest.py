"""Paylaşılan fixture'lar: sentetik checkpoint ve hızlı Predictor/ModelService.

Gerçek checkpoint (43 MB) yüklenmeden tüm çıkarım hattı, 1M parametrik
mobilnetv3_small_075'ten üretilen sentetik checkpoint ile test edilir.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

import timm

from config import CLASS_NAMES

ROOT = Path(__file__).resolve().parents[1]

TINY_MODEL = "mobilenetv3_small_075"
TINY_IMG_SIZE = 96


@pytest.fixture(autouse=True)
def _project_cwd():
    """Tüm testler proje kökünden çalışır (göreli yollar deterministik olsun)."""
    old = Path.cwd()
    os.chdir(ROOT)
    yield
    os.chdir(old)


def _make_checkpoint_file(path: Path, img_size: int) -> Path:
    model = timm.create_model(
        TINY_MODEL, pretrained=False, num_classes=len(CLASS_NAMES)
    )
    # weights_only=True yüklemesine uyum için yalnızca düz tipler kullanılır
    # (Path/picklable olmayan değerler serileştirilemez).
    ckpt = {
        "state_dict": model.state_dict(),
        "class_names": list(CLASS_NAMES),
        "label2id": {str(i): i for i in range(len(CLASS_NAMES))},
        "config": {
            "model": {
                "name": TINY_MODEL,
                "num_classes": len(CLASS_NAMES),
                "pretrained": False,
            },
            "train": {
                "epochs": 2,
                "lr": 1e-4,
                "weight_decay": 1e-4,
                "label_smoothing": 0.1,
                "balance": "weighted_loss",
                "patience": 2,
                "seed": 42,
                "amp": False,
                "deterministic": False,
                "run_root": "runs",
                "run_name": "tests-tiny",
            },
            "data": {
                "data_dir": "data",
                "image_dir": None,
                "image_col": None,
                "label_col": None,
                "img_size": img_size,
                "batch_size": 2,
                "num_workers": 0,
                "use_aug": False,
            },
        },
        "epoch": 1,
        "val_macro_f1": 0.0,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)
    return path


@pytest.fixture(scope="session")
def tiny_checkpoint_path(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("tiny_run")
    return _make_checkpoint_file(root / "weights" / "best.pt", TINY_IMG_SIZE)


@pytest.fixture(scope="session")
def tiny_model_config() -> ModelConfig:
    return ModelConfig(name=TINY_MODEL, num_classes=len(CLASS_NAMES), pretrained=False)


@pytest.fixture(scope="session")
def tiny_predictor(tiny_checkpoint_path):
    from inference import Predictor

    return Predictor.from_checkpoint(tiny_checkpoint_path, device="cpu")


@pytest.fixture(scope="session")
def tiny_service(tiny_checkpoint_path):
    from main import ModelService

    return ModelService(tiny_checkpoint_path)
