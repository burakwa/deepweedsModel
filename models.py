"""Model kurucu: timm öncelikli, torchvision yedekli.

DeepWeeds checkpoint'leri timm anahtar adlarıyla (conv_stem, blocks,
classifier) kaydedildiğinden efficientnet ailesi timm ile kurulur; timm
yoksa/bilinmiyorsa torchvision eşdeğerine düşer.
"""

from __future__ import annotations

import torch

from config import ModelConfig

_TORCHVISION_KNOWN: frozenset[str] = frozenset(
    {
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
        "efficientnet_b0",
        "efficientnet_b1",
        "efficientnet_b2",
        "efficientnet_b3",
        "efficientnet_b4",
        "efficientnet_b5",
        "efficientnet_b6",
        "efficientnet_b7",
        "efficientnet_v2_s",
        "efficientnet_v2_m",
        "efficientnet_v2_l",
    }
)


def _replace_head(model: torch.nn.Module, num_classes: int) -> torch.nn.Module:
    """torchvision modellerinin sınıflandırıcı katmanını yeniden boyutlandırır."""
    import torch.nn as nn

    fc = getattr(model, "fc", None)
    if isinstance(fc, nn.Linear):
        model.fc = nn.Linear(fc.in_features, num_classes)
        return model
    clf = getattr(model, "classifier", None)
    if isinstance(clf, nn.Sequential):
        last = clf[-1]
        if isinstance(last, nn.Linear):
            clf[-1] = nn.Linear(last.in_features, num_classes)
            return model
    if isinstance(clf, nn.Linear):
        model.classifier = nn.Linear(clf.in_features, num_classes)
        return model
    raise ValueError(
        f"{type(model).__name__} sınıflandırıcı katmanı bulunamadı; model başı değiştirilemedi."
    )


def _build_torchvision(name: str, num_classes: int, pretrained: bool) -> torch.nn.Module:
    from torchvision import models as tv

    if name not in _TORCHVISION_KNOWN or not hasattr(tv, name):
        raise ValueError(
            f"Bilinmeyen model adı: {name!r}. timm kurulu değilse şu adlar "
            f"torchvision ile kullanılabilir: {sorted(_TORCHVISION_KNOWN)}"
        )
    builder = getattr(tv, name)
    weights = "DEFAULT" if pretrained else None
    try:
        model = builder(weights=weights)
    except TypeError:
        model = builder(pretrained=pretrained)
    return _replace_head(model, num_classes)


def build_model(cfg: ModelConfig) -> torch.nn.Module:
    """Config'e göre sınıflandırma modeli kurar.

    timm kuruluysa istenen ad önce timm'de denenir (checkpoint uyumu için
    zorunlu); bilinmeyen ad/timm yoksa torchvision'a düşülür.
    """
    try:
        import timm

        return timm.create_model(
            cfg.name, pretrained=cfg.pretrained, num_classes=cfg.num_classes
        )
    except (ImportError, RuntimeError, ValueError):
        # timm kurulu değil ya da ad timm'de tanımlı değil → torchvision yedeği
        return _build_torchvision(cfg.name, cfg.num_classes, cfg.pretrained)
