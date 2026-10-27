"""DeepWeeds veri kümesi ve transform'lar.

CSV/DataFrame tabanlı görüntü listesi (`_path` kolonu) ile çalışır;
değerlendirme ve TTA için ImageNet normalizasyonlu transform üreticileri
sunar.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

PATH_CANDIDATES: tuple[str, ...] = ("_path", "path", "image", "filepath", "filename")
LABEL_CANDIDATES: tuple[str, ...] = ("label", "y", "target", "class_id", "label_int")


def join_path(image_dir: str | Path, path: str) -> str:
    p = Path(path)
    return str(Path(image_dir) / p) if not p.is_absolute() else str(p)


def build_eval_transform(img_size: int) -> transforms.Compose:
    """Değerlendirme/çıkarım transform'u: yeniden boyutla + normalize et."""
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_train_transform(img_size: int) -> transforms.Compose:
    """Eğitim zamanı hafif rastgele artırma zinciri."""
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=12),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
        ]
    )


def build_tta_transforms(img_size: int) -> list[Callable[[Image.Image], torch.Tensor]]:
    """Test zamanı artırma (TTA): 5 deterministik görünüm döndürür.

    Sıra: orijinal, yatay çevirme, hafif ±döndürme, hafif yakınlaştırma.
    `Predictor.predict_tta` bu görünümlerin olasılıklarını ortalar.
    """
    norm = [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return [
        transforms.Compose([transforms.Resize((img_size, img_size)), *norm]),
        transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=1.0),
                *norm,
            ]
        ),
        transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomRotation(degrees=(5, 5), fill=0),
                *norm,
            ]
        ),
        transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomRotation(degrees=(-5, -5), fill=0),
                *norm,
            ]
        ),
        transforms.Compose(
            [
                transforms.Resize((int(img_size * 1.12), int(img_size * 1.12))),
                transforms.CenterCrop(img_size),
                *norm,
            ]
        ),
    ]


class DeepWeedsDataset(Dataset):
    """DataFrame'den görüntü+etiket okuyan veri kümesi.

    Parametreler
    ------------
    df : pd.DataFrame
        Görüntü yolu kolonu (`_path`/`path`/...) içeren tablo. Etiket kolonu
        (`label`/`y`/...) yoksa etiketler -1 döner (test seti senaryosu).
    transform : çağrılabilir, opsiyonel
        PIL görüntüsünü tensöre çeviren zincir.
    image_dir : str | Path, opsiyonel
        Yol kolonu göreliyse birleştirilecek kök klasör.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
        image_dir: str | None = None,
    ) -> None:
        path_col = next((c for c in PATH_CANDIDATES if c in df.columns), None)
        if path_col is None:
            raise ValueError(
                f"Görüntü yolu kolonu bulunamadı. Beklenen: {PATH_CANDIDATES}"
            )
        self.transform = transform

        paths = df[path_col].astype(str).tolist()
        if image_dir is not None:
            paths = [join_path(image_dir, p) for p in paths]
        self.paths: list[str] = paths

        label_col = next((c for c in LABEL_CANDIDATES if c in df.columns), None)
        if label_col is not None:
            self.labels: torch.Tensor = torch.as_tensor(
                df[label_col].to_numpy(), dtype=torch.long
            )
        else:
            self.labels = torch.full((len(paths),), -1, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = Image.open(self.paths[idx]).convert("RGB")
        x = self.transform(img) if self.transform is not None else img
        return x, int(self.labels[idx])


def resolve_paths(
    df: pd.DataFrame,
    data_dir: str | None,
    image_dir: str | None,
    image_col: str | None,
) -> pd.DataFrame:
    """Config'e göre `_path` kolonunu üretir (eğitim/tanı amaçlı yardımcı)."""
    out = df.copy()
    if "_path" in out.columns:
        return out
    if image_col is None or image_col not in out.columns:
        raise ValueError("Görüntü kolonu bulunamadı; image_col belirtin.")
    base = image_dir or data_dir or "."
    out["_path"] = out[image_col].astype(str).map(lambda p: join_path(base, p))
    return out


def class_counts(labels: Sequence[int], num_classes: int) -> torch.Tensor:
    """Sınıf frekans tensörü (ağırlıklı kayıp hesaplarına girdi)."""
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for y in labels:
        if 0 <= int(y) < num_classes:
            counts[int(y)] += 1
    return counts
