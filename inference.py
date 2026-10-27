from __future__ import annotations

import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

from config import ModelConfig
from data import DeepWeedsDataset, build_eval_transform, build_tta_transforms
from models import build_model


class PredictResult(NamedTuple):
    """predict_df dönüş tipi: gerçek etiketler ve olasılık matrisi."""

    y_true: np.ndarray  # int, test setinde -1
    y_prob: np.ndarray  # shape (N, num_classes), satırlar toplamı 1


class Predictor:
    def __init__(
        self,
        model,
        class_names: list[str],
        img_size: int,
        device,
        meta: dict,
        checkpoint_path: Path | None = None,
    ):
        self.model, self.class_names = model, class_names
        self.img_size, self.device, self.meta = img_size, device, meta
        self.checkpoint_path = checkpoint_path
        self.transform = build_eval_transform(img_size)

    @classmethod
    def from_checkpoint(cls, path: str | Path, device=None) -> Predictor:
        path = Path(path)
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(device, str):
            device = torch.device(device)
        ckpt = torch.load(
            path, map_location="cpu", weights_only=True
        )  # torch>=2.6 uyumlu
        cfg = ckpt["config"]
        model_cfg = ModelConfig(**cfg["model"])
        model_cfg.pretrained = False  # offline: ağırlık zaten checkpoint içinde
        model = build_model(model_cfg)
        model.load_state_dict(ckpt["state_dict"])
        model.to(device).eval()
        return cls(
            model, ckpt["class_names"], int(cfg["data"]["img_size"]), device, ckpt, path
        )

    @property
    def size_bytes(self) -> int:
        return self.checkpoint_path.stat().st_size if self.checkpoint_path else 0

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image.convert("RGB")).unsqueeze(0)

    @torch.no_grad()
    def predict_pil(self, image: Image.Image) -> np.ndarray:
        """Tek görüntü için olasılık vektörü döner. shape=(num_classes,)"""
        with torch.autocast("cuda", enabled=self.device.type == "cuda"):
            logits = self.model(self._preprocess(image).to(self.device))
        return torch.softmax(logits.float(), dim=1).cpu().numpy()[0]

    @torch.no_grad()
    def predict_batch(self, images: list[Image.Image], batch_size: int = 16) -> np.ndarray:
        """Birden çok görüntüyü tek başı ileri geçişlerle sınıflandırır.

        Video kare taraması gibi çoklu pencere senaryolarında tek tek
        predict_pil çağrısına göre belirgin biçimde hızlıdır.

        Returns
        -------
        np.ndarray: shape (len(images), num_classes) softmax olasılık matrisi.
        """
        if not images:
            return np.zeros((0, len(self.class_names)), dtype=np.float32)
        probs: list[np.ndarray] = []
        for i in range(0, len(images), batch_size):
            chunk = [img.convert("RGB") for img in images[i : i + batch_size]]
            x = torch.stack([self.transform(img) for img in chunk]).to(self.device)
            with torch.autocast("cuda", enabled=self.device.type == "cuda"):
                logits = self.model(x)
            probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        return np.vstack(probs)

    @torch.no_grad()
    def predict_df(self, df: pd.DataFrame, batch_size: int = 32) -> PredictResult:
        """DataFrame üzerinde toplu çıkarım.

        Returns
        -------
        PredictResult(y_true, y_prob)
            y_true : shape (N,)  — gerçek etiketler (test için -1)
            y_prob : shape (N, C) — softmax olasılıkları
        """
        loader = DataLoader(
            DeepWeedsDataset(df, self.transform),
            batch_size=batch_size,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
        )
        y_trues, probs = [], []
        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            with torch.autocast("cuda", enabled=self.device.type == "cuda"):
                logits = self.model(x)
            probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            y_trues.append(y.numpy())
        return PredictResult(
            y_true=np.concatenate(y_trues),
            y_prob=np.vstack(probs),
        )

    def benchmark_latency(self, df: pd.DataFrame, n: int = 64) -> dict:
        """Tek görüntü gecikmesi (API senaryosu). p50/p95 rapor için."""
        n = min(n, len(df))
        times = []
        for i in range(n):
            img = Image.open(df["_path"].iloc[i])
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            self.predict_pil(img)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
        t = np.asarray(times)
        return {
            "mean_ms": float(t.mean()),
            "p50_ms": float(np.percentile(t, 50)),
            "p95_ms": float(np.percentile(t, 95)),
        }

    @torch.no_grad()
    def predict_tta(self, image: Image.Image) -> np.ndarray:
        """Test-Time Augmentation: 5 farklı transform ile tahmin alıp ortalama alır.

        Returns
        -------
        np.ndarray: shape (num_classes,) — ortalama softmax olasılıkları
        """
        tta_transforms = build_tta_transforms(self.img_size)
        probs = []
        for transform in tta_transforms:
            tensor = transform(image.convert("RGB")).unsqueeze(0)
            with torch.autocast("cuda", enabled=self.device.type == "cuda"):
                logits = self.model(tensor.to(self.device))
            probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy()[0])
        return np.mean(probs, axis=0)


def write_predictions_csv(
    path: str | Path,
    ids: pd.Series,
    probs: np.ndarray,
    latency_ms: float,
    size_bytes: int,
    multipliers: np.ndarray | list[float] | None = None,
) -> pd.DataFrame:
    """Yarışma formatı: id,prediction,prob_0..prob_8,latency_ms,bytes"""
    k = probs.shape[1]
    if multipliers is not None:
        scaled = probs * np.asarray(multipliers)
        pred = scaled.argmax(1)
    else:
        pred = probs.argmax(1)
    out = pd.DataFrame({"id": ids.values, "prediction": pred})
    for j in range(k):
        out[f"prob_{j}"] = probs[:, j]
    out["latency_ms"] = round(float(latency_ms), 3)
    out["bytes"] = int(size_bytes)
    out.to_csv(path, index=False)
    assert np.allclose(out[[f"prob_{j}" for j in range(k)]].sum(axis=1), 1.0, atol=1e-4)
    assert len(out) == len(ids) and out["id"].is_unique
    return out
