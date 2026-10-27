"""models.py testleri: build_model çıktı şekli ve checkpoint anahtar uyumu."""

from __future__ import annotations

import torch

from config import NUM_CLASSES, ModelConfig
from models import build_model


def test_build_tiny_timm_model_forward():
    cfg = ModelConfig(name="mobilenetv3_small_075", num_classes=NUM_CLASSES, pretrained=False)
    model = build_model(cfg).eval()
    with torch.no_grad():
        y = model(torch.randn(2, 3, 96, 96))
    assert y.shape == (2, NUM_CLASSES)


def test_efficientnet_b3_keys_match_checkpoint_format():
    """timm modelleri checkpoint'teki (conv_stem/blocks/classifier) adlarla kurulmalı."""
    cfg = ModelConfig(name="efficientnet_b3", num_classes=NUM_CLASSES, pretrained=False)
    model = build_model(cfg)
    keys = set(model.state_dict().keys())
    assert "conv_stem.weight" in keys
    assert "classifier.weight" in keys


def test_build_model_torchvision_resnet50():
    cfg = ModelConfig(name="resnet50", num_classes=NUM_CLASSES, pretrained=False)
    model = build_model(cfg).eval()
    with torch.no_grad():
        y = model(torch.randn(1, 3, 224, 224))
    assert y.shape == (1, NUM_CLASSES)
    assert model.fc.out_features == NUM_CLASSES


def test_build_model_unknown_name_raises():
    cfg = ModelConfig(name="yok_olecek_bir_model", num_classes=NUM_CLASSES, pretrained=False)
    try:
        build_model(cfg)
    except ValueError as e:
        assert "Bilinmeyen model adı" in str(e) or "yok_olecek_bir_model" in str(e)
    else:
        raise AssertionError("Bilinmeyen model adı ValueError vermeli")
