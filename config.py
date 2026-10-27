from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Literal

CLASS_NAMES: tuple[str, ...] = (
    "Chinee apple",
    "Lantana",
    "Parkinsonia",
    "Parthenium",
    "Prickly acacia",
    "Rubber vine",
    "Siam weed",
    "Snake weed",
    "Negative",
)
NUM_CLASSES = len(CLASS_NAMES)
BalanceStrategy = Literal["none", "weighted_loss", "balanced_sampler"]


@dataclass(slots=True)
class DataConfig:
    data_dir: Path = Path("data")
    image_dir: Path | None = None  # görüntülerin ayrı bir klasörde olduğu durumlar için
    image_col: str | None = None  # CSV'deki görüntü yolu kolonu
    label_col: str | None = None  # CSV'deki etiket kolonu
    img_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    use_aug: bool = True


@dataclass(slots=True)
class ModelConfig:
    name: str = "resnet50"
    num_classes: int = NUM_CLASSES
    pretrained: bool = True


@dataclass(slots=True)
class TrainConfig:
    epochs: int = 25
    lr: float = 1e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    balance: BalanceStrategy = "weighted_loss"
    patience: int = 5
    seed: int = 42
    amp: bool = True
    deterministic: bool = False
    run_root: Path = Path("runs")
    run_name: str | None = None


def jsonable(obj: Any) -> Any:
    """Path/np/dataclass tiplerini JSON-uyumlu hale getirir (checkpoint'a gömmeden önce şart)."""
    if is_dataclass(obj) and not isinstance(obj, type):
        obj = asdict(obj)  # dataclass → dict; sonra dict dalına düşer
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, str):  # str alt sınıfları (ör. TorchVersion) düz metne iner
        return str(obj)
    if obj is None or isinstance(obj, (int, float, bool)):
        return obj
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)


def save_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(obj), indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _norm(s: str) -> str:
    return " ".join(str(s).strip().lower().replace("_", " ").replace("-", " ").split())


def build_label_map(series) -> dict[Any, int]:
    """Etiketleri yarışma sırasına (0..8) eşler. Sayısal etiketlerde kimlik eşleme kullanılır."""
    uniq = list(series.unique())
    try:
        as_int = {v: int(v) for v in uniq}
        if sorted(as_int.values()) == list(range(NUM_CLASSES)):
            return as_int
    except (TypeError, ValueError):
        pass

    mapping: dict[Any, int | None] = {}
    for v in uniq:
        s = _norm(v)
        hit = next((i for i, c in enumerate(CLASS_NAMES) if s == c.lower()), None)
        if hit is None:
            hit = next(
                (
                    i
                    for i, c in enumerate(CLASS_NAMES)
                    if s.startswith(c.lower()[:8]) or c.lower().startswith(s[:8])
                ),
                None,
            )
        mapping[v] = hit
    if (
        any(v is None for v in mapping.values())
        or len(set(mapping.values())) != NUM_CLASSES
    ):
        bad = [k for k, v in mapping.items() if v is None]
        raise ValueError(
            f"Etiketler sınıf adlarıyla eşleşmedi: {bad}. Kolon adını kontrol edin."
        )
    return {k: int(v) for k, v in mapping.items()}
