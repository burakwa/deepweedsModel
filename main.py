"""DeepWeeds Streamlit uygulaması.

Template (template/index.html) referans alınarak tasarlanmış arayüz:
- Lacivert üst bar + açılır yan panel (saf CSS checkbox hack)
- Ortada büyük DEEPWEEDS başlığı ve rüzgârda sallanan yaprak SVG'si
- Görsel/Video yükleme kutuları
- Video karelerinde sliding-window sınıflandırma ile bitki kutulama,
  tür etiketleme ve benzersiz bitki sayımı
"""

from __future__ import annotations

import base64
import io
import json
import re
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

from config import CLASS_NAMES
from inference import Predictor

NEGATIVE_CLASS = "Negative"

# ---------------------------------------------------------------------------
# Video bitki taraması sabitleri
# ---------------------------------------------------------------------------
DEFAULT_MAX_FRAMES = 10  # örneklenen kare sayısı
DEFAULT_WINDOW_SCALES = (0.42, 0.72)  # pencere boyutları (min kenara oran)
DEFAULT_STRIDE_FACTOR = 0.62  # pencere kaydırma adımı (pencere boyutuna oran)
DEFAULT_CONF_THRESHOLD = 0.60  # kutucuk için minimum güven
WORK_MAX_SIDE = 640  # tarama çözünürlüğü (en uzun kenar)
MAX_VIDEO_BYTES = 300 * 1024 * 1024  # 300 MB kullanıcı videosu üst sınırı

# st.cache_resource / st.cache_data mute olduktan sonra kod da değişebilir; bu
# anahtar değiştirildiğinde önbellek sıfırlanır ve taze ModelService kuruluşu
# garanti edilir (örn. yeni alan/property'li sınıflar).
CACHE_VERSION = "2026.10.27-video-detect-v3"


# ===========================================================================
# 1. Alan Modeli ve Tür Kataloğu (Domain / Catalog)
# ===========================================================================
class SpeciesCatalog:
    """DeepWeeds veri setindeki 9 sınıfın Türkçe ve botanik bilgilerini yönetir."""

    _DATA: ClassVar[dict[str, dict[str, str]]] = {
        "Chinee apple": {
            "tr_name": "Hint hünnabı",
            "scientific": "Ziziphus mauritiana",
            "family": "Rhamnaceae",
            "desc": "Kurak ve yarı kurak tropikal meralarda yayılıp otlatmayı engelleyen odunsu çalı/ağaç türü.",
        },
        "Lantana": {
            "tr_name": "Ağaç minesi",
            "scientific": "Lantana camara",
            "family": "Verbenaceae",
            "desc": "Zehirli meyvelere sahip, yerli florayı boğan agresif istilacı çiçekli çalı.",
        },
        "Parkinsonia": {
            "tr_name": "Kudüs dikeni",
            "scientific": "Parkinsonia aculeata",
            "family": "Fabaceae",
            "desc": "Su yollarını ve nehir yataklarını tıkayan, keskin dikenli ve tohumları suda yayılan çalı.",
        },
        "Parthenium": {
            "tr_name": "Partenyum otu",
            "scientific": "Parthenium hysterophorus",
            "family": "Asteraceae",
            "desc": "Alerjiye yol açan, tarla verimini ve hayvan yemini düşüren tehlikeli istilacı ot.",
        },
        "Prickly acacia": {
            "tr_name": "Dikenli akasya",
            "scientific": "Vachellia nilotica",
            "family": "Fabaceae",
            "desc": "Otlatma arazilerini işgal ederek toprağın su dengesini bozan dikenli küçük ağaç türü.",
        },
        "Rubber vine": {
            "tr_name": "Kauçuk sarmaşığı",
            "scientific": "Cryptostegia grandiflora",
            "family": "Apocynaceae",
            "desc": "Ağaçların üzerini sararak fotosentezi engelleyen ve kurutan zehirli sarmaşık.",
        },
        "Siam weed": {
            "tr_name": "Siyam otu",
            "scientific": "Chromolaena odorata",
            "family": "Asteraceae",
            "desc": "Dünyanın en zararlı 100 istilacı türünden biri olan, aşırı hızlı büyüyen çok yıllık çalı.",
        },
        "Snake weed": {
            "tr_name": "Yılan otu",
            "scientific": "Stachytarpheta spp.",
            "family": "Verbenaceae",
            "desc": "Mera arazilerinde ot kalitesini bozan, hayvanların beslenmesine zarar veren dik çalı türü.",
        },
        "Negative": {
            "tr_name": "Hedef otlardan hiçbiri yok (Temiz Arazi)",
            "scientific": "Negative / Background",
            "family": "N/A",
            "desc": "Hedeflenen 8 istilacı yabani ot türünden hiçbiri tespit edilmedi.",
        },
    }

    SAMPLES: ClassVar[list[dict[str, str]]] = [
        {"label": "Hint hünnabı", "path": "data/images/T03_00542.jpg"},
        {"label": "Ağaç minesi", "path": "data/images/T03_01735.jpg"},
        {"label": "Kudüs dikeni", "path": "data/images/T03_05202.jpg"},
        {"label": "Partenyum otu", "path": "data/images/T03_05928.jpg"},
        {"label": "Dikenli akasya", "path": "data/images/T03_03671.jpg"},
        {"label": "Kauçuk sarmaşığı", "path": "data/images/T03_06955.jpg"},
        {"label": "Siyam otu", "path": "data/images/T03_08437.jpg"},
        {"label": "Yılan otu", "path": "data/images/T03_00492.jpg"},
        {"label": "Temiz Arazi", "path": "data/images/T03_03835.jpg"},
    ]

    @classmethod
    def get_info(cls, class_name: str) -> dict[str, str]:
        return cls._DATA.get(
            class_name,
            {
                "tr_name": class_name,
                "scientific": class_name,
                "family": "N/A",
                "desc": "Sınıflandırma sonucu.",
            },
        )

    @classmethod
    def short_name(cls, class_name: str) -> str:
        """Kutu etiketi/badge için kısa Türkçe ad (Negative kısaltılır)."""
        if class_name == NEGATIVE_CLASS:
            return "Temiz Arazi"
        return cls.get_info(class_name)["tr_name"]

    @classmethod
    def index_of(cls, class_name: str) -> int:
        return CLASS_NAMES.index(class_name)


CLASS_COLORS_RGB: tuple[tuple[int, int, int], ...] = (
    (46, 204, 113),   # Chinee apple — yeşil
    (52, 152, 219),   # Lantana — mavi
    (155, 89, 182),   # Parkinsonia — mor
    (241, 196, 15),   # Parthenium — sarı
    (230, 126, 34),   # Prickly acacia — turuncu
    (231, 76, 60),    # Rubber vine — kırmızı
    (26, 188, 156),   # Siam weed — turkuaz
    (93, 109, 126),   # Snake weed — gri-mavi
    (149, 165, 165),  # Negative — gri (kutu olarak kullanılmaz)
)


# ===========================================================================
# 2. Veri Taşıma Nesneleri (DTOs)
# ===========================================================================
@dataclass(slots=True)
class PredictionResult:
    top_class: str
    confidence: float
    probs: np.ndarray
    latency_ms: float
    tr_name: str
    scientific: str
    family: str
    description: str


@dataclass(slots=True)
class VideoFrame:
    """Kutulanmış tek karenin önizlemesi."""

    index: int
    t_sec: float
    n_plants: int
    jpeg: bytes


@dataclass(slots=True)
class VideoResult:
    """Video analizinin birleşik sonucu: baskın tür + kutu sayımları."""

    top_class: str
    confidence: float
    probs: np.ndarray
    latency_ms: float
    frames_analyzed: int
    duration_sec: float
    tr_name: str
    scientific: str
    family: str
    description: str
    species_counts: dict[str, int] = field(default_factory=dict)
    total_plants: int = 0
    annotated_strip_b64: str = ""
    best_frame_b64: str = ""
    frames: list[VideoFrame] = field(default_factory=list)
    top_count_class: str = NEGATIVE_CLASS


# ===========================================================================
# 3. Model Servis Katmanı (ModelService - Singleton / Resource Manager)
# ===========================================================================
class ModelService:
    """Model checkpoint yükleme, kalibrasyon ve sistem bilgilerini yönetir."""

    _instance: ModelService | None = None

    def __init__(self, checkpoint_path: Path | None = None) -> None:
        self.checkpoint_path = checkpoint_path or Path("runs/exp-effb3-sqrt/weights/best.pt")
        self._predictor: Predictor | None = None
        self._multipliers: np.ndarray | None = None
        self._load()

    def _load(self) -> None:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint bulunamadı: {self.checkpoint_path}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._predictor = Predictor.from_checkpoint(self.checkpoint_path, device=device)
        self._load_calibration()

    def _load_calibration(self) -> None:
        calib_file = self.checkpoint_path.parent.parent / "calibration.json"
        if calib_file.exists():
            try:
                data = json.loads(calib_file.read_text(encoding="utf-8"))
                if "multipliers" in data:
                    self._multipliers = np.asarray(data["multipliers"], dtype=np.float32)
            except (OSError, ValueError, TypeError):
                self._multipliers = None

    @property
    def predictor(self) -> Predictor:
        if self._predictor is None:
            self._load()
        return self._predictor

    @property
    def multipliers(self) -> np.ndarray | None:
        return self._multipliers

    @property
    def class_names(self) -> tuple[str, ...]:
        return CLASS_NAMES

    @property
    def device_name(self) -> str:
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    @property
    def is_cuda(self) -> bool:
        return torch.cuda.is_available()

    @property
    def model_size_mb(self) -> float:
        return (
            self.checkpoint_path.stat().st_size / (1024 * 1024)
            if self.checkpoint_path.exists()
            else 0.0
        )

    @property
    def img_size(self) -> int:
        return getattr(self.predictor, "img_size", 256)

    @property
    def model_label(self) -> str:
        return f"{self.checkpoint_path.parent.parent.name} ({self.model_size_mb:.1f} MB)"

    @property
    def arch_label(self) -> str:
        """Template alt başlığı: 'EFFICIENTNET-B3 (sqrt weighted loss) - exp-effb3-sqrt'."""
        try:
            cfg = self.predictor.meta["config"]
            arch = str(cfg["model"]["name"]).replace("_", "-").upper()
            balance = str(cfg["train"]["balance"]).replace("_", " ")
            run_name = str(cfg["train"]["run_name"])
            return f"{arch} ({balance}) - {run_name}"
        except (KeyError, TypeError, AttributeError):
            return self.model_label


@st.cache_resource(show_spinner=False)
def get_model_service(checkpoint_path_str: str, version: str = CACHE_VERSION) -> ModelService:
    """Streamlit önbellekli ModelService sağlayıcısı.

    ``version`` parametresi önbellek anahtarına dahildir. Kod güncellendiğinde
    CACHE_VERSION yükseltilirse yeni bir ModelService oluşturulur; böylece
    eski oturumdan kalan, yeni property/alanları olmayan nesneler tazelenir.
    """
    return ModelService(Path(checkpoint_path_str))


# ===========================================================================
# 4. PlantDetector - Video Karelerinde Bitki Kutulama ve Sayım
# ===========================================================================
@dataclass(slots=True)
class _Box:
    x1: int
    y1: int
    x2: int
    y2: int
    class_name: str
    conf: float

    @property
    def area(self) -> float:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)


@dataclass(slots=True)
class _Track:
    """Kareler arası basit IoU takibi: benzersiz bitki sayımı sağlar."""

    box: _Box
    hits: int = 1
    miss: int = 0
    vote: Counter = field(default_factory=Counter)

    def pick_class(self) -> str:
        return self.vote.most_common(1)[0][0]


class PlantDetector:
    """Sliding-window sınıflandırma ile karelerdeki bitkileri kutular ve sayar.

    Model yalnızca sınıflandırıcı olduğundan kareler örtüşen pencerelere
    bölünüp parça-parça sınıflandırılır; "Negative" dışına çıkan eşik üstü
    güvenli pencereler bitki kutusu kabul edilir. Kareler arası IoU takibi,
    aynı bitkinin farklı karelerde tekrar sayılmasını engeller.
    """

    def __init__(
        self,
        service: ModelService,
        window_scales: tuple[float, ...] = DEFAULT_WINDOW_SCALES,
        stride_factor: float = DEFAULT_STRIDE_FACTOR,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        max_boxes_per_frame: int = 6,
        gap_tolerance: int = 2,
    ) -> None:
        self.service = service
        self.window_scales = window_scales
        self.stride_factor = stride_factor
        self.conf_threshold = conf_threshold
        self.max_boxes_per_frame = max_boxes_per_frame
        self.gap_tolerance = gap_tolerance

    def _calibrate(self, probs: np.ndarray) -> np.ndarray:
        mult = self.service.multipliers
        if mult is not None:
            probs = probs * mult
            total = probs.sum()
            if total > 0:
                probs = probs / total
        return probs

    def _grid_windows(self, fw: int, fh: int) -> list[tuple[int, int, int]]:
        """Ölçek ve kaydırma adımına göre (x, y, pencere_boyutu) listesi üretir."""
        windows: list[tuple[int, int, int]] = []
        min_side = min(fw, fh)
        for f in self.window_scales:
            w = round(f * min_side)
            if w < 96 or w > min(fw, fh):
                continue
            stride = max(24, int(w * self.stride_factor))
            xs = list(range(0, max(fw - w, 0) + 1, stride)) or [0]
            ys = list(range(0, max(fh - w, 0) + 1, stride)) or [0]
            for y in ys:
                for x in xs:
                    windows.append((x, y, w))
        if len(windows) > 26:
            step = len(windows) / 26
            windows = [windows[int(i * step)] for i in range(26)]
        return windows

    def _iou(self, a: _Box, b: _Box) -> float:
        ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
        ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = a.area + b.area - inter
        return inter / union if union > 0 else 0.0

    def _nms(self, boxes: list[_Box]) -> list[_Box]:
        """Güven sırasına NMS: aynı sınıf 0.35, farklı sınıf 0.50 IoU eşiği."""
        boxes = sorted(boxes, key=lambda b: b.conf, reverse=True)
        kept: list[_Box] = []
        for b in boxes:
            ok = True
            for k in kept:
                same = b.class_name == k.class_name
                if self._iou(b, k) >= (0.35 if same else 0.50):
                    ok = False
                    break
            if ok:
                kept.append(b)
            if len(kept) >= self.max_boxes_per_frame:
                break
        return kept

    def _predict_probs(self, crops: list) -> np.ndarray:
        """Yığın sınıflandırma; stale Predictor (predict_batch öncesi) için güvenli fallback."""
        predictor = self.service.predictor
        batched = getattr(predictor, "predict_batch", None)
        if batched is not None:
            return np.asarray(batched(crops), dtype=np.float32)
        return np.asarray([predictor.predict_pil(c) for c in crops], dtype=np.float32)

    def detect_boxes(self, rgb_work: np.ndarray) -> list[_Box]:
        """Çalışma koordinatında kare içindeki bitki kutularını tespit eder."""
        fh, fw = rgb_work.shape[:2]
        wins = self._grid_windows(fw, fh)
        if not wins:
            return []
        crops = [Image.fromarray(rgb_work[y : y + w, x : x + w]) for x, y, w in wins]
        probs = self._predict_probs(crops)
        cands: list[_Box] = []
        for (x, y, w), p in zip(wins, probs):
            p = self._calibrate(np.asarray(p, dtype=np.float32))
            idx = int(np.argmax(p))
            name = self.service.class_names[idx]
            conf = float(p[idx])
            if name != NEGATIVE_CLASS and conf >= self.conf_threshold:
                cands.append(_Box(x, y, x + w, y + w, name, conf))
        return self._nms(cands)

    def annotate(self, rgb_work: np.ndarray, boxes: list[_Box], t_sec: float) -> np.ndarray:
        """Kare üzerine kutu + tür etiketi + sayım çipi + zaman damgası çizer."""
        img = cv2.cvtColor(rgb_work, cv2.COLOR_RGB2BGR)
        _fh, fw = img.shape[:2]

        for b in boxes:
            ci = SpeciesCatalog.index_of(b.class_name)
            crgb = CLASS_COLORS_RGB[ci % len(CLASS_COLORS_RGB)]
            color_bgr = (crgb[2], crgb[1], crgb[0])
            cv2.rectangle(img, (b.x1, b.y1), (b.x2, b.y2), color_bgr, 2)
            label = f"{SpeciesCatalog.short_name(b.class_name)} %{b.conf * 100:.0f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ly = b.y1 - 6 if b.y1 - th - 10 > 0 else b.y2 + th + 6
            cv2.rectangle(img, (b.x1, ly - th - 6), (b.x1 + tw + 8, ly + 4), color_bgr, -1)
            cv2.putText(
                img,
                label,
                (b.x1 + 4, ly),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        n = len(boxes)
        chip = f"{n} bitki"
        (cw, chh), _ = cv2.getTextSize(chip, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (8, 8), (8 + cw + 16, 8 + chh + 14), (255, 255, 255), -1)
        cv2.rectangle(img, (8, 8), (8 + cw + 16, 8 + chh + 14), (80, 62, 44), 1)
        cv2.putText(
            img,
            chip,
            (16, 8 + chh + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (80, 62, 44),
            2,
            cv2.LINE_AA,
        )

        tstamp = f"{t_sec:.1f} sn"
        (tw2, th2), _ = cv2.getTextSize(tstamp, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(img, (fw - tw2 - 24, 8), (fw - 8, 8 + th2 + 12), (255, 255, 255), -1)
        cv2.rectangle(img, (fw - tw2 - 24, 8), (fw - 8, 8 + th2 + 12), (80, 62, 44), 1)
        cv2.putText(
            img,
            tstamp,
            (fw - tw2 - 20, 8 + th2 + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (80, 62, 44),
            1,
            cv2.LINE_AA,
        )
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def analyze_video(self, video_bytes: bytes, max_frames: int = DEFAULT_MAX_FRAMES) -> VideoResult:
        """Videoyu örnekler, bitki kutularını çizer ve sayım özetini döndürür."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = Path(tmp.name)

        cap = cv2.VideoCapture(str(tmp_path))
        if not cap.isOpened():
            tmp_path.unlink(missing_ok=True)
            raise ValueError("Video dosyası açılamadı (codec desteklenmiyor olabilir).")

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            duration_sec = total_frames / fps if fps > 0 else 0.0

            if total_frames <= max_frames:
                indices = list(range(max(1, total_frames)))
            else:
                indices = np.linspace(0, total_frames - 1, max_frames, dtype=int).tolist()

            # Sıralı okuma: POS_FRAMES seek bazı kodlayıcılarda güvenilmez olabilir
            want = sorted(set(indices))
            frames_rgb: dict[int, np.ndarray] = {}
            wi = 0
            fi = 0
            while wi < len(want):
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                if fi == want[wi]:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if max(rgb.shape) > WORK_MAX_SIDE:
                        s = WORK_MAX_SIDE / max(rgb.shape)
                        rgb = cv2.resize(
                            rgb,
                            (int(rgb.shape[1] * s), int(rgb.shape[0] * s)),
                            interpolation=cv2.INTER_AREA,
                        )
                    frames_rgb[fi] = rgb
                    wi += 1
                fi += 1
        finally:
            cap.release()
            tmp_path.unlink(missing_ok=True)

        if not frames_rgb:
            raise ValueError("Videodan geçerli kare okunamadı.")

        t0 = time.perf_counter()
        tracks: list[_Track] = []
        finished: list[_Track] = []
        frame_probs: list[np.ndarray] = []
        frames_meta: list[VideoFrame] = []
        annotated: list[tuple[int, float, np.ndarray, int]] = []

        for idx in sorted(frames_rgb):
            rgb = frames_rgb[idx]
            t_sec = idx / fps if fps > 0 else 0.0

            # Tam kare olasılıkları (sonuç çubukları için)
            fp = self._calibrate(
                np.asarray(
                    self.service.predictor.predict_pil(Image.fromarray(rgb)),
                    dtype=np.float32,
                )
            )
            frame_probs.append(fp)

            boxes = self.detect_boxes(rgb)
            prev_count = len(tracks)
            matched: set[int] = set()
            for b in boxes:
                best_i: int | None = None
                best_iou = 0.30
                for i, tr in enumerate(tracks):
                    if i in matched or tr.box.class_name != b.class_name:
                        continue
                    val = self._iou(tr.box, b)
                    if val > best_iou:
                        best_i, best_iou = i, val
                if best_i is not None:
                    tr = tracks[best_i]
                    tr.box, tr.hits, tr.miss = b, tr.hits + 1, 0
                    tr.vote[b.class_name] += 1
                    matched.add(best_i)
                else:
                    vote: Counter = Counter()
                    vote[b.class_name] = 1
                    tracks.append(_Track(box=b, hits=1, miss=0, vote=vote))
            for i in range(prev_count - 1, -1, -1):
                if i not in matched:
                    tracks[i].miss += 1
                    if tracks[i].miss > self.gap_tolerance:
                        finished.append(tracks.pop(i))

            ann = self.annotate(rgb, boxes, t_sec)
            annotated.append((idx, t_sec, ann, len(boxes)))
            encoded, jpeg = cv2.imencode(
                ".jpg", cv2.cvtColor(ann, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85]
            )
            if encoded:
                frames_meta.append(
                    VideoFrame(index=idx, t_sec=t_sec, n_plants=len(boxes), jpeg=jpeg.tobytes())
                )

        finished.extend(tracks)
        species_counts: dict[str, int] = {}
        for tr in finished:
            species_counts[tr.pick_class()] = species_counts.get(tr.pick_class(), 0) + 1
        total_plants = sum(species_counts.values())
        latency_ms = (time.perf_counter() - t0) * 1000

        if frame_probs:
            mean_probs = np.mean(frame_probs, axis=0)
        else:
            k = len(self.service.class_names)
            mean_probs = np.full(k, 1.0 / k)
        top_idx = int(mean_probs.argmax())
        top_class = self.service.class_names[top_idx]
        confidence = float(mean_probs[top_idx]) * 100.0

        strip_b64, best_frame_b64 = self._build_visuals(annotated)
        top_class_by_count = (
            max(species_counts.items(), key=lambda kv: kv[1])[0] if species_counts else NEGATIVE_CLASS
        )
        info = SpeciesCatalog.get_info(top_class)

        return VideoResult(
            top_class=top_class,
            confidence=confidence,
            probs=mean_probs,
            latency_ms=latency_ms,
            frames_analyzed=len(annotated),
            duration_sec=duration_sec,
            tr_name=info["tr_name"],
            scientific=info["scientific"],
            family=info["family"],
            description=info["desc"],
            species_counts=species_counts,
            total_plants=total_plants,
            annotated_strip_b64=strip_b64,
            best_frame_b64=best_frame_b64,
            frames=frames_meta,
            top_count_class=top_class_by_count,
        )

    def _build_visuals(
        self, annotated: list[tuple[int, float, np.ndarray, int]]
    ) -> tuple[str, str]:
        """Animasyonlu WEBP şerit + en yoğun karenin JPEG'ini (base64) üretir."""
        strip_b64 = ""
        if annotated:
            step = max(1, int(np.ceil(len(annotated) / 6)))
            selected = annotated[::step][:6]
            pil_frames: list[Image.Image] = []
            for _, _, ann, _ in selected:
                if max(ann.shape[:2]) > 520:
                    s = 520 / max(ann.shape[:2])
                    ann = cv2.resize(
                        ann,
                        (int(ann.shape[1] * s), int(ann.shape[0] * s)),
                        interpolation=cv2.INTER_AREA,
                    )
                pil_frames.append(Image.fromarray(ann))
            buf = io.BytesIO()
            try:
                pil_frames[0].save(
                    buf,
                    format="WEBP",
                    save_all=True,
                    append_images=pil_frames[1:],
                    duration=1100,
                    loop=0,
                    quality=78,
                    method=4,
                )
                if buf.tell() <= 12 * 1024 * 1024:
                    strip_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            except (OSError, ValueError, KeyError):
                strip_b64 = ""

        best_frame_b64 = ""
        if annotated:
            idx_best = max(range(len(annotated)), key=lambda i: annotated[i][3])
            ann = annotated[idx_best][2]
            pil = Image.fromarray(ann)
            b = io.BytesIO()
            pil.save(b, format="JPEG", quality=88)
            best_frame_b64 = base64.b64encode(b.getvalue()).decode("utf-8")
        return strip_b64, best_frame_b64


# ===========================================================================
# 5. Çıkarım Motoru (InferenceEngine - Single Responsibility)
# ===========================================================================
class InferenceEngine:
    """Görsel ve video çıkarım mantığını yürüten servis."""

    def __init__(self, service: ModelService) -> None:
        self.service = service

    def predict_image(self, image: Image.Image, use_calibration: bool = True) -> PredictionResult:
        """PIL görüntüsü için model çıkarımı yapar (tamamen RAM içinde)."""
        t0 = time.perf_counter()
        probs = self.service.predictor.predict_pil(image)
        if use_calibration and self.service.multipliers is not None:
            probs = probs * self.service.multipliers
            total = probs.sum()
            if total > 0:
                probs = probs / total

        latency_ms = (time.perf_counter() - t0) * 1000
        top_idx = int(probs.argmax())
        top_class = self.service.class_names[top_idx]
        confidence = float(probs[top_idx]) * 100.0

        info = SpeciesCatalog.get_info(top_class)
        return PredictionResult(
            top_class=top_class,
            confidence=confidence,
            probs=probs,
            latency_ms=latency_ms,
            tr_name=info["tr_name"],
            scientific=info["scientific"],
            family=info["family"],
            description=info["desc"],
        )

    def predict_video(
        self,
        video_bytes: bytes,
        max_frames: int = DEFAULT_MAX_FRAMES,
        window_scales: tuple[float, ...] = DEFAULT_WINDOW_SCALES,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    ) -> VideoResult:
        """Video karelerinde bitki kutulama + tür etiketi + benzersiz sayım."""
        detector = PlantDetector(
            self.service,
            window_scales=window_scales,
            conf_threshold=conf_threshold,
        )
        return detector.analyze_video(video_bytes, max_frames=max_frames)


@st.cache_data(show_spinner=False, max_entries=6)
def analyze_video_cached(
    video_bytes: bytes,
    checkpoint_path_str: str,
    max_frames: int,
    window_scales: tuple[float, ...],
    conf_threshold: float,
    version: str = CACHE_VERSION,
) -> VideoResult:
    """Video analizini Streamlit veri önbelleğinde tutar (rerun'da yeniden hesaplanmaz)."""
    service = get_model_service(checkpoint_path_str, version)
    engine = InferenceEngine(service)
    return engine.predict_video(
        video_bytes,
        max_frames=max_frames,
        window_scales=window_scales,
        conf_threshold=conf_threshold,
    )


# ===========================================================================
# 6. Görsel Arayüz Oluşturucu (ThemeRenderer - Presentation / CSS / HTML)
# ===========================================================================
class ThemeRenderer:
    """template/index.html'e sadık arayüzü Streamlit'te render eder."""

    @staticmethod
    def get_leaf_b64() -> str:
        """Logo görselini assets/leaf.jpeg'den base64'e çevirir (disk I/O tek seferlik)."""
        for path in [
            Path("assets/leaf.jpeg"),
            Path(__file__).resolve().parent / "assets" / "leaf.jpeg",
        ]:
            if path.exists():
                return base64.b64encode(path.read_bytes()).decode("utf-8")
        return ""

    @staticmethod
    def fallback_leaf_data_uri() -> str:
        """assets/leaf.jpeg bulunamazsa şeffaf 1px yedek (harici istek yok)."""
        px = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGMA"
            "AABAQAQFgAABkgBBBOsUAAAAAElFTkSuQmCC"
        )
        return f"data:image/png;base64,{px}"

    @staticmethod
    def arch_subtitle(service: ModelService) -> str:
        """Hero/sidebar alt başlığı: eski cache'lenmiş service nesneleriyle de çalışır."""
        label = getattr(service, "arch_label", None)
        if isinstance(label, str) and label:
            return label
        return service.model_label

    @staticmethod
    def image_to_b64(image: Image.Image) -> str:
        """PIL görüntüsünü base64'e çevirir (disk I/O yok)."""
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=92)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @staticmethod
    def _compact_html(html: str) -> str:
        """HTML'i tek satıra sıkıştırır.

        Streamlit st.markdown; çok satırlı/div'li HTML'i markdown blokları
        olarak ayrıştırıp bazı bölümleri (ör. olasılık çubukları) düz metin
        olarak gösterebilir. Sıkıştırma, CSS `<style>` bloğuna uygulanmadan
        tüm sonuç HTML'ine uygulanır: markdown-it güvenilir biçimde tek
        HTML bloğu görür.
        """
        html = re.sub(r">\s*\n\s*<", "><", html)
        html = re.sub(r"[ \t]*\n[ \t]*", " ", html)
        return re.sub(r"\s{2,}", " ", html).strip()

    @classmethod
    def inject_styles(cls) -> None:
        """index.html template'i ile birebir uyumlu CSS kuralları."""
        st.markdown(
            """
            <style>
            /* Streamlit varsayılan arayüzünü sıfırla */
            header[data-testid="stHeader"],
            div[data-testid="stToolbar"],
            div[data-testid="stDecoration"],
            div[data-testid="stStatusWidget"],
            footer {
                display: none !important;
                visibility: hidden !important;
                height: 0 !important;
                margin: 0 !important;
                padding: 0 !important;
            }

            html, body, [data-testid="stAppViewContainer"], .main {
                background-color: #f5f5f5 !important;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
            }

            html, body, [data-testid="stAppViewContainer"] {
                overflow-x: hidden !important;
                -webkit-tap-highlight-color: transparent;
            }

            .main .block-container {
                max-width: 100% !important;
                padding: 0 !important;
                margin: 0 !important;
            }

            /* ÜSTTEKİ AYDINLIK YATAY ŞERİT (beyaz yüzey + lacivert metin vurgusu) */
            :root {
                --nav-h: 64px;
                --ink: #2c3e50;
                --ink-soft: #5d6d7e;
                --surface: #ffffff;
                --surface-2: #f2f5f8;
                --card-light: #eaeef3;
                --card-light-hover: #e0e7ee;
            }

            .top-navbar {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: var(--nav-h);
                background: linear-gradient(180deg, rgba(255, 255, 255, 0.97) 0%, rgba(244, 247, 250, 0.94) 100%);
                backdrop-filter: blur(8px);
                -webkit-backdrop-filter: blur(8px);
                border-bottom: 1px solid rgba(44, 62, 80, 0.10);
                box-shadow: 0 4px 18px rgba(44, 62, 80, 0.10);
                display: flex;
                align-items: center;
                justify-content: space-between;
                z-index: 997;
                box-sizing: border-box;
                padding-left: env(safe-area-inset-left, 0px);
                padding-right: calc(env(safe-area-inset-right, 0px) + 8px);
            }

            .navbar-title {
                font-family: 'Franklin Gothic Medium', sans-serif;
                font-size: 1.35rem;
                font-weight: 700;
                color: var(--ink);
                letter-spacing: 1.5px;
                margin-left: 78px;
                white-space: nowrap;
                text-shadow: 0 1px 0 rgba(255, 255, 255, 0.9);
            }

            .navbar-chip {
                background: rgba(46, 204, 113, 0.10);
                border: 1px solid rgba(46, 204, 113, 0.35);
                color: #16a085;
                padding: 5px 13px;
                border-radius: 15px;
                font-size: 0.84rem;
                font-weight: 600;
                display: flex;
                align-items: center;
                gap: 7px;
                margin-right: 18px;
                white-space: nowrap;
            }

            .chip-dot {
                width: 7px;
                height: 7px;
                background-color: #2ecc71;
                border-radius: 50%;
                box-shadow: 0 0 6px #2ecc71;
            }

            /* SİDEBAR AÇILIR PANEL */
            .sidebar-toggle-input {
                display: none !important;
            }

            .sidebar-overlay {
                position: fixed;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
                height: 100dvh;
                background-color: rgba(44, 62, 80, 0.22);
                backdrop-filter: blur(2px);
                -webkit-backdrop-filter: blur(2px);
                opacity: 0;
                visibility: hidden;
                transition: opacity 0.3s ease, visibility 0.3s ease;
                z-index: 998;
            }

            .sidebar-panel {
                position: fixed;
                top: 0;
                left: 0;
                width: min(280px, 82vw);
                max-width: 82vw;
                min-height: 100vh;
                min-height: 100dvh;
                background: linear-gradient(180deg, #ffffff 0%, var(--surface-2) 100%);
                border-right: 1px solid rgba(44, 62, 80, 0.08);
                box-shadow: 6px 0 24px rgba(44, 62, 80, 0.16);
                transition: left 0.4s cubic-bezier(0.32, 1, 0.5, 1), opacity 0.35s ease, visibility 0.35s ease;
                opacity: 0;
                visibility: hidden;
                z-index: 999;
                display: flex;
                flex-direction: column;
                box-sizing: border-box;
                transform: translateX(-100%);
                padding: calc(76px + env(safe-area-inset-top, 0px)) 18px calc(28px + env(safe-area-inset-bottom, 0px)) 18px;
                overflow-y: auto;
                overscroll-behavior: contain;
            }

            .sidebar-toggle-input:checked ~ .sidebar-overlay {
                opacity: 1;
                visibility: visible;
            }

            .sidebar-toggle-input:checked ~ .sidebar-panel {
                opacity: 1;
                visibility: visible;
                transform: translateX(0);
            }

            .sidebar-toggle-input:checked ~ .menu-btn img {
                transform: rotate(360deg);
            }

            /* PANEL KAPAK BAŞLIĞI */
            .panel-head {
                display: flex;
                align-items: center;
                gap: 10px;
                padding: 4px 4px 16px 4px;
                border-bottom: 1px solid rgba(44, 62, 80, 0.10);
                margin-bottom: 18px;
            }

            .panel-logo {
                width: 34px;
                height: 34px;
                border-radius: 9px;
                object-fit: cover;
                box-shadow: 0 1px 3px rgba(44, 62, 80, 0.2);
            }

            .panel-title {
                flex: 1;
                font-family: 'Franklin Gothic Medium', sans-serif;
                font-size: 1rem;
                font-weight: 600;
                letter-spacing: 0.8px;
                color: var(--ink);
                white-space: nowrap;
            }

            .panel-close {
                width: 32px;
                height: 32px;
                display: flex;
                align-items: center;
                justify-content: center;
                border-radius: 9px;
                background: rgba(44, 62, 80, 0.06);
                color: #7f8c8d;
                font-size: 1.25rem;
                line-height: 1;
                cursor: pointer;
                user-select: none;
                -webkit-user-select: none;
                transition: background-color 0.2s ease, color 0.2s ease;
            }

            .panel-close:hover {
                background: rgba(192, 57, 43, 0.10);
                color: #c0392b;
            }

            .panel-close:active {
                transform: scale(0.92);
            }

            .panel-foot {
                margin-top: auto;
                padding-top: 14px;
                border-top: 1px solid rgba(44, 62, 80, 0.10);
                font-size: 0.78rem;
                letter-spacing: 0.6px;
                color: #95a5a6;
                text-align: center;
                text-transform: uppercase;
                font-weight: 600;
            }

            .sidebar-menu {
                list-style: none;
                padding: 0;
                margin: 0;
            }

            .sidebar-menu li {
                margin-bottom: 15px;
            }

            .sidebar-card {
                display: block;
                padding: 13px 15px;
                color: var(--ink-soft);
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                border-radius: 12px;
                background-color: var(--card-light);
                box-shadow:
                    inset 2px 2px 5px rgba(44, 62, 80, 0.10),
                    inset -2px -2px 5px rgba(255, 255, 255, 0.9),
                    0 1px 2px rgba(255, 255, 255, 0.7);
                transition: all 0.3s cubic-bezier(0.25, 1, 0.5, 1);
                cursor: pointer;
                overflow: hidden;
                user-select: none;
                -webkit-user-select: none;
                -webkit-tap-highlight-color: transparent;
            }

            .sidebar-card:hover {
                background-color: var(--card-light-hover);
                transform: translateX(2px);
            }

            .sidebar-card:active {
                background-color: #d9e0e8;
                transform: translateX(0) scale(0.99);
            }

            .card-title {
                font-weight: 600;
                font-size: 1.05rem;
                color: var(--ink);
                display: flex;
                justify-content: space-between;
                align-items: center;
            }

            .card-title::after {
                content: '▾';
                font-size: 0.9rem;
                color: #95a5a6;
                transition: transform 0.3s ease, color 0.3s ease;
            }

            .card-detail {
                max-height: 0;
                opacity: 0;
                overflow: hidden;
                font-size: 0.88rem;
                color: #7a8a99;
                font-weight: 400;
                margin-top: 0;
                line-height: 1.4;
                transition: max-height 0.4s ease, opacity 0.3s ease, margin-top 0.3s ease;
            }

            .sidebar-card:hover .card-title::after {
                transform: rotate(180deg);
                color: #2980b9;
            }

            .sidebar-card:hover .card-detail {
                max-height: 150px;
                opacity: 1;
                margin-top: 10px;
                padding-top: 8px;
                border-top: 1px solid rgba(44, 62, 80, 0.10);
            }

            /* SÍDEBAR BUTONU */
            .menu-btn {
                position: fixed;
                z-index: 1000;
                width: 46px;
                height: 46px;
                background-color: #ffffff;
                border: 1px solid rgba(44, 62, 80, 0.10);
                border-radius: 12px;
                cursor: pointer;
                padding: 0;
                display: flex;
                align-items: center;
                justify-content: center;
                box-shadow:
                    inset 2px 2px 5px rgba(44, 62, 80, 0.10),
                    inset -2px -2px 5px rgba(255, 255, 255, 0.95),
                    0 2px 8px rgba(44, 62, 80, 0.12);
                transition: background-color 0.2s ease, transform 0.15s ease, box-shadow 0.2s ease;
                top: calc((var(--nav-h) - 46px) / 2);
                left: 18px;
                overflow: hidden;
                user-select: none;
                -webkit-user-select: none;
                touch-action: manipulation;
                -webkit-tap-highlight-color: transparent;
            }

            .menu-btn:hover {
                background-color: var(--card-light-hover);
                transform: translateY(-1px);
            }

            .menu-btn:active {
                transform: scale(0.94);
                box-shadow:
                    inset 2px 2px 6px rgba(44, 62, 80, 0.18),
                    inset -2px -2px 6px rgba(255, 255, 255, 0.7);
            }

            .menu-btn:focus-visible {
                outline: 2px solid #16a085;
                outline-offset: 2px;
            }

            .menu-btn img {
                width: 28px;
                height: 28px;
                border-radius: 7px;
                object-fit: cover;
                box-shadow: 0 1px 3px rgba(44, 62, 80, 0.25);
                transition: transform 0.45s cubic-bezier(0.25, 1, 0.5, 1);
            }

            /* ORTADAKİ BÜYÜK BAŞLIK */
            .title-container {
                position: relative;
                text-align: center;
                margin-top: clamp(120px, 20vh, 200px);
                font-family: 'Franklin Gothic Medium', sans-serif;
                font-size: clamp(2.5rem, 8.5vw, 6rem);
                line-height: 1.05;
                color: #d0d0d0;
                letter-spacing: 2px;
                text-shadow:
                    0 -1px 0 rgba(0,0,0,0.8),
                    0 -2px 1px rgba(0,0,0,0.5),
                    0 -3px 2px rgba(0,0,0,0.3),
                    0 1px 0 rgba(255,255,255,1),
                    0 2px 1px rgba(255,255,255,0.6);
            }

            .letter-w {
                position: relative;
                display: inline-block;
            }

            .title-leaf-svg {
                position: absolute;
                top: -42px;
                left: 90%;
                transform: translateX(-50%);
                width: clamp(48px, 9vw, 75px);
                height: clamp(48px, 9vw, 75px);
                fill: #cccccc;
                filter:
                    drop-shadow(0 -1px 0 rgba(0,0,0,0.8))
                    drop-shadow(0 -2px 1px rgba(0,0,0,0.5))
                    drop-shadow(0 1px 0 rgba(255,255,255,1));
                pointer-events: none;
                z-index: 2;
                transform-origin: bottom center;
                animation: windBlow 12s ease-in-out infinite;
            }

            @keyframes windBlow {
                0%   { transform: translateX(-50%) rotate(0deg); }
                35%  { transform: translateX(-50%) rotate(0deg); }
                45%  { transform: translateX(-50%) rotate(15deg); }
                60%  { transform: translateX(-50%) rotate(-4deg); }
                68%  { transform: translateX(-50%) rotate(3deg); }
                78%  { transform: translateX(-50%) rotate(0deg); }
                100% { transform: translateX(-50%) rotate(0deg); }
            }

            .hero-subtitle {
                text-align: center;
                margin-top: 10px;
                margin-bottom: 45px;
                padding: 0 20px;
                font-family: 'Franklin Gothic Medium', sans-serif;
                color: #b3bcc4;
                font-size: clamp(0.92rem, 2.5vw, 1.15rem);
                letter-spacing: 0.5px;
                text-shadow: 0 1px 0 rgba(255, 255, 255, 0.85);
            }

            /* GÖRSEL / VİDEO YÜKLEME KUTULARI (template: 220x200) */
            div[data-testid="stFileUploader"] {
                width: min(220px, 100%) !important;
                margin: 0 auto !important;
            }

            div[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] {
                width: 100% !important;
                height: 200px !important;
                min-height: 200px !important;
                max-height: 200px !important;
                border: 2px dashed #b0b0b0 !important;
                border-radius: 15px !important;
                background-color: #ededed !important;
                display: flex !important;
                flex-direction: column !important;
                align-items: center !important;
                justify-content: center !important;
                cursor: pointer !important;
                box-shadow: inset 2px 2px 5px rgba(0,0,0,0.1) !important;
                transition: all 0.3s ease !important;
                padding: 10px !important;
                box-sizing: border-box !important;
                margin-left: auto !important;
                margin-right: auto !important;
            }

            div[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"]:hover {
                border-color: #3498db !important;
                background-color: #e5e9ec !important;
                transform: translateY(-2px) !important;
                box-shadow:
                    inset 2px 2px 6px rgba(0,0,0,0.15),
                    0 8px 18px rgba(0,0,0,0.08) !important;
            }

            div[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] svg,
            div[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] small,
            div[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] button,
            div[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] [data-testid="stMarkdownContainer"] {
                display: none !important;
            }

            div[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"]::before {
                content: "+" !important;
                font-size: 3.5rem !important;
                color: #7f8c8d !important;
                line-height: 1 !important;
                font-weight: 300 !important;
                transition: color 0.2s ease !important;
            }

            div[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"]:hover::before {
                color: #3498db !important;
            }

            /* Dropzone etiketleri: dosya input accept niteliğiyle hedeflenir.
               (Eski `div[data-testid="column"]` seçicileri Streamlit'te
               hiç eşleşmiyordu ve etiketler görünmüyordu.) */
            div[data-testid="stFileUploader"]:has(input[accept*="jpg"])
            section[data-testid="stFileUploadDropzone"]::after {
                content: "Görsel Yükle" !important;
                font-size: 1rem !important;
                color: #7f8c8d !important;
                font-weight: bold !important;
                margin-top: 10px !important;
            }

            div[data-testid="stFileUploader"]:has(input[accept*="mp4"])
            section[data-testid="stFileUploadDropzone"]::after {
                content: "Video Yükle" !important;
                font-size: 1rem !important;
                color: #7f8c8d !important;
                font-weight: bold !important;
                margin-top: 10px !important;
            }

            /* Örnek görseller paneli */
            .sample-tray {
                width: 82%;
                max-width: 820px;
                margin: 25px auto 40px auto;
                text-align: center;
            }

            .sample-tray-title {
                font-size: 0.88rem;
                color: #7f8c8d;
                font-weight: 700;
                margin-bottom: 12px;
                text-transform: uppercase;
                letter-spacing: 0.8px;
            }

            /* SONUÇ KARTI - template: #resultsSection */
            .results-card {
                width: 100%;
                max-width: min(900px, 96vw);
                margin: 140px auto 20px auto;
                background: #ededed;
                border-radius: 15px;
                padding: 30px;
                box-shadow:
                    inset 3px 3px 6px rgba(0,0,0,0.12),
                    inset -3px -3px 6px rgba(255,255,255,0.8),
                    0 10px 30px rgba(0,0,0,0.08);
                display: grid;
                grid-template-columns: 1fr 1fr;
                grid-template-areas: "media info";
                gap: 20px 40px;
                align-items: center;
                box-sizing: border-box;
                animation: smoothEnter 0.8s ease;
            }

            @keyframes smoothEnter {
                from { opacity: 0; transform: translateY(60px); }
                to { opacity: 1; transform: translateY(0); }
            }

            .result-media-box {
                grid-area: media;
                text-align: center;
                background: #e2e8f0;
                padding: 20px;
                border-radius: 15px;
                box-shadow:
                    inset 2px 2px 5px rgba(0,0,0,0.15),
                    inset -2px -2px 5px rgba(255,255,255,0.9);
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                gap: 10px;
            }

            .result-media-box img,
            .result-media-box video {
                max-width: 100%;
                max-height: 300px;
                border-radius: 10px;
                object-fit: contain;
                box-shadow: 0 4px 10px rgba(0,0,0,0.1);
            }

            .media-caption {
                font-size: 0.8rem;
                color: #7f8c8d;
                font-weight: 600;
                letter-spacing: 0.4px;
            }

            .result-info-box {
                grid-area: info;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }

            .status-tag {
                display: inline-flex;
                align-items: center;
                gap: 6px;
                font-size: 0.9rem;
                color: #27ae60;
                font-weight: bold;
                background: #e2e8f0;
                padding: 5px 12px;
                border-radius: 6px;
                box-shadow:
                    inset 1px 1px 3px rgba(0,0,0,0.1),
                    inset -1px -1px 3px rgba(255,255,255,0.8);
                margin-bottom: 10px;
            }

            .press-heading {
                color: #2c3e50;
                margin: 0 0 8px 0;
                font-size: 1.2rem;
                font-weight: 600;
                text-shadow:
                    0 -1px 0 rgba(0,0,0,0.4),
                    0 1px 0 rgba(255,255,255,0.9);
            }

            .press-result {
                font-size: 1.4rem;
                font-weight: bold;
                color: #c0392b;
                margin: 0;
                line-height: 1.25;
                text-shadow:
                    0 -1px 0 rgba(0,0,0,0.3),
                    0 1px 0 rgba(255,255,255,0.9);
            }

            .latin-title {
                font-size: 0.95rem;
                color: #7f8c8d;
                font-style: italic;
                margin-bottom: 12px;
            }

            .confidence-badge {
                display: inline-block;
                margin-top: 15px;
                padding: 10px 20px;
                background: #e2e8f0;
                color: #16a085;
                border-radius: 10px;
                font-size: 1.2rem;
                font-weight: bold;
                box-shadow:
                    inset 2px 2px 4px rgba(0,0,0,0.1),
                    inset -2px -2px 4px rgba(255,255,255,0.8);
                text-shadow: 0 1px 0 rgba(255,255,255,0.8);
            }

            .desc-box {
                font-size: 0.88rem;
                color: #555555;
                line-height: 1.4;
                background: rgba(255,255,255,0.45);
                padding: 8px 12px;
                border-radius: 6px;
                margin-bottom: 14px;
            }

            /* TESPİT SAYIM ÇİPLERİ */
            .counts-block {
                padding: 10px 12px;
                background: #e2e8f0;
                border-radius: 10px;
                box-shadow:
                    inset 2px 2px 5px rgba(0,0,0,0.12),
                    inset -2px -2px 5px rgba(255,255,255,0.85);
                margin-bottom: 14px;
            }

            .counts-title {
                font-size: 0.8rem;
                color: #7f8c8d;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.8px;
                margin-bottom: 8px;
            }

            .count-chip {
                display: inline-block;
                margin: 3px 6px 3px 0;
                padding: 6px 12px;
                background: #ffffff;
                color: #27ae60;
                font-size: 0.85rem;
                font-weight: 700;
                border-radius: 10px;
                box-shadow: 0 1px 4px rgba(0,0,0,0.12);
            }

            .count-chip.count-zero {
                color: #7f8c8d;
                background: transparent;
                box-shadow: none;
            }

            .prob-caption {
                font-size: 0.82rem;
                color: #7f8c8d;
                font-weight: 700;
                margin-bottom: 8px;
                text-transform: uppercase;
            }

            .prob-row {
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 5px;
                font-size: 0.82rem;
                color: #34495e;
                font-weight: 600;
            }

            .prob-track {
                flex: 1;
                height: 7px;
                background: #dcdde1;
                border-radius: 4px;
                margin: 0 10px;
                overflow: hidden;
            }

            .prob-fill {
                height: 100%;
                background: linear-gradient(90deg, #16a085, #2ecc71);
                border-radius: 4px;
            }

            .prob-fill.top-hit {
                background: linear-gradient(90deg, #e74c3c, #c0392b);
            }

            /* ISARETLI KARE GALERISI */
            .frame-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 14px;
            }

            .frame-item {
                background: #e2e8f0;
                border-radius: 12px;
                padding: 10px;
                box-shadow:
                    inset 2px 2px 5px rgba(0,0,0,0.12),
                    inset -2px -2px 5px rgba(255,255,255,0.85);
            }

            .frame-item img {
                width: 100%;
                max-height: 260px;
                object-fit: contain;
                border-radius: 8px;
                display: block;
            }

            .frame-item .frame-label {
                display: flex;
                justify-content: space-between;
                font-size: 0.8rem;
                color: #34495e;
                font-weight: 600;
                padding-top: 8px;
            }

            /* ANA EKRANA DÖN (template: .back-btn) */
            div.stButton > button {
                padding: 12px 20px !important;
                background: linear-gradient(180deg, #ffffff 0%, #e7ecf1 100%) !important;
                color: var(--ink) !important;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
                font-size: 1rem !important;
                font-weight: 600 !important;
                text-align: center !important;
                border: 1px solid rgba(44, 62, 80, 0.12) !important;
                border-radius: 12px !important;
                cursor: pointer !important;
                box-shadow:
                    inset 2px 2px 5px rgba(255, 255, 255, 0.95),
                    inset -2px -2px 5px rgba(44, 62, 80, 0.08),
                    0 6px 15px rgba(44, 62, 80, 0.14) !important;
                text-shadow: 0 1px 0 rgba(255, 255, 255, 0.85) !important;
                transition: all 0.2s ease !important;
            }

            div.stButton > button:hover {
                background: linear-gradient(180deg, #ffffff 0%, #dce4ec 100%) !important;
                transform: translateY(-2px) !important;
                box-shadow:
                    inset 2px 2px 5px rgba(255, 255, 255, 0.95),
                    inset -2px -2px 5px rgba(44, 62, 80, 0.08),
                    0 8px 20px rgba(44, 62, 80, 0.20) !important;
                color: #22303f !important;
            }

            div.stButton > button:active {
                transform: translateY(1px) !important;
                box-shadow: inset 2px 3px 6px rgba(44, 62, 80, 0.18) !important;
            }

            /* ------------------------------------------------------------------
               RESPONSIVE
               ------------------------------------------------------------------ */
            @media (max-width: 900px) {
                .results-card {
                    grid-template-columns: 1fr;
                    grid-template-areas: "media" "info";
                    margin-top: 100px;
                    gap: 16px 20px;
                }
                .title-leaf-svg {
                    top: -28px;
                }
                .frame-grid {
                    grid-template-columns: 1fr;
                }
                .navbar-title {
                    margin-left: 74px;
                }
            }

            @media (max-width: 640px) {
                :root {
                    --nav-h: 58px;
                }
                .menu-btn {
                    width: 42px;
                    height: 42px;
                    left: 12px;
                    border-radius: 11px;
                }
                .menu-btn img {
                    width: 26px;
                    height: 26px;
                }
                .navbar-title {
                    font-size: 1.05rem;
                    margin-left: 66px;
                    letter-spacing: 1px;
                }
                .navbar-chip {
                    padding: 4px 10px;
                    font-size: 0.72rem;
                    gap: 5px;
                    margin-right: 10px;
                }
                .chip-dot {
                    width: 6px;
                    height: 6px;
                }
                .sidebar-menu li {
                    margin-bottom: 11px;
                }
                .sidebar-card {
                    padding: 12px 13px;
                }
                .card-title {
                    font-size: 0.98rem;
                }
                .card-detail {
                    font-size: 0.84rem;
                }
                .panel-logo {
                    width: 30px;
                    height: 30px;
                }
                .panel-title {
                    font-size: 0.92rem;
                }
                .title-container {
                    margin-top: 128px;
                    letter-spacing: 1px;
                }
                .title-leaf-svg {
                    top: -24px;
                    width: 44px;
                    height: 44px;
                }
                .hero-subtitle {
                    margin-bottom: 32px;
                    padding: 0 16px;
                }
                div[data-testid="stFileUploader"] {
                    width: 100% !important;
                    max-width: 260px;
                }
                div[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] {
                    height: 170px !important;
                    min-height: 170px !important;
                    max-height: 170px !important;
                }
                .sample-tray {
                    width: 94%;
                    margin: 18px auto 28px auto;
                }
                .sample-tray-title {
                    font-size: 0.78rem;
                }
                .results-card {
                    margin-top: 92px;
                    padding: 16px;
                    border-radius: 13px;
                }
                .result-media-box {
                    padding: 12px;
                }
                .result-media-box img,
                .result-media-box video {
                    max-height: 52vw;
                }
                .status-tag {
                    font-size: 0.8rem;
                }
                .press-result {
                    font-size: 1.25rem;
                }
                .confidence-badge {
                    font-size: 1.05rem;
                    padding: 9px 16px;
                }
                .desc-box {
                    font-size: 0.82rem;
                }
                .count-chip {
                    font-size: 0.78rem;
                    padding: 5px 10px;
                }
                .prob-row {
                    font-size: 0.78rem;
                }
                .prob-row > span:first-child {
                    min-width: 0 !important;
                    flex: 1 1 auto;
                    overflow: hidden;
                    text-overflow: ellipsis;
                    white-space: nowrap;
                }
                .prob-row > .prob-track {
                    margin: 0 8px;
                }
                .prob-row > span:last-child {
                    min-width: 38px !important;
                }
                .frame-item img {
                    max-height: 190px;
                }
                .frame-item .frame-label {
                    font-size: 0.74rem;
                }
            }

            /* Yatay / kısa ekranlar: hero'yu sıkıştır */
            @media (max-height: 500px) and (orientation: landscape) {
                .title-container {
                    margin-top: 88px;
                }
                .results-card {
                    margin-top: 68px;
                }
            }

            /* Hareket azaltma tercihi */
            @media (prefers-reduced-motion: reduce) {
                .title-leaf-svg {
                    animation: none !important;
                }
                .sidebar-panel,
                .sidebar-overlay,
                .menu-btn,
                .menu-btn img,
                .sidebar-card {
                    transition: none !important;
                }
                .results-card {
                    animation: none !important;
                }
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    @classmethod
    def render_navbar_and_sidebar(cls, service: ModelService) -> None:
        """Üst lacivert barı ve açılır yan panel kartlarını çizer.

        ``menu-root`` sarmalayıcısı zorunludur: Streamlit HTML'i paragraf
        bloklarına bölebildiğinden ``input``/``label`` ve ``.sidebar-panel``
        aynı üst etiket altında kalmazsa CSS ``~`` seçicisi eşleşmez ve
        menü açılmaz.
        """
        leaf_b64 = cls.get_leaf_b64()
        leaf_src = (
            f"data:image/jpeg;base64,{leaf_b64}"
            if leaf_b64
            else cls.fallback_leaf_data_uri()
        )
        status_text = "CUDA Aktif" if service.is_cuda else "CPU Aktif"

        st.markdown(
            cls._compact_html(
                f"""
            <div class="menu-root">
                <input type="checkbox" id="sidebarToggle" class="sidebar-toggle-input">

                <label for="sidebarToggle" class="menu-btn" title="Model Bilgi Menüsü" aria-label="Menüyü aç/kapat">
                    <img src="{leaf_src}" alt="Menü">
                </label>

                <label for="sidebarToggle" class="sidebar-overlay"></label>

                <div class="sidebar-panel">
                    <div class="panel-head">
                        <img class="panel-logo" src="{leaf_src}" alt="">
                        <span class="panel-title">Model Paneli</span>
                        <label for="sidebarToggle" class="panel-close" title="Kapat" aria-label="Menüyü kapat">&times;</label>
                    </div>
                    <ul class="sidebar-menu">
                        <li>
                            <div class="sidebar-card">
                                <div class="card-title">Model Bilgisi</div>
                                <div class="card-detail">
                                    {cls.arch_subtitle(service)} mimarisiyle eğitilmiş özel derin öğrenme sınıflandırma modeli.
                                </div>
                            </div>
                        </li>
                        <li>
                            <div class="sidebar-card">
                                <div class="card-title">Cihaz</div>
                                <div class="card-detail">
                                    {status_text} ({service.device_name}).
                                </div>
                            </div>
                        </li>
                        <li>
                            <div class="sidebar-card">
                                <div class="card-title">Model Boyutu</div>
                                <div class="card-detail">
                                    Yaklaşık {service.model_size_mb:.1f} MB (.pt ağırlık dosyası).
                                </div>
                            </div>
                        </li>
                        <li>
                            <div class="sidebar-card">
                                <div class="card-title">Giriş Boyutu</div>
                                <div class="card-detail">
                                    {service.img_size}x{service.img_size} piksel RGB çözünürlük standart girdi boyutu.
                                </div>
                            </div>
                        </li>
                        <li>
                            <div class="sidebar-card">
                                <div class="card-title">Sınıf Sayısı</div>
                                <div class="card-detail">
                                    9 farklı yabani ot (weed) türü sınıflandırma kapasitesi.
                                </div>
                            </div>
                        </li>
                    </ul>
                    <div class="panel-foot">DEEPWEEDS AI &middot; {status_text}</div>
                </div>

                <div class="top-navbar">
                    <div class="navbar-title">DEEPWEEDS AI</div>
                    <div class="navbar-chip">
                        <span class="chip-dot"></span>
                        {status_text}
                    </div>
                </div>
            </div>
            """
            ),
            unsafe_allow_html=True,
        )

    @classmethod
    def render_hero(cls, subtitle: str) -> None:
        """DEEPWEEDS başlığı ve rüzgârda sallanan yaprak SVG'sini çizer."""
        st.markdown(
            cls._compact_html(
                f"""
            <div class="title-container">
                DEEP<span class="letter-w"><svg class="title-leaf-svg" viewBox="0 0 100 100">
                    <path d="M50,85 C50,85 58,60 78,45 C88,37.5 95,20 90,5 C70,10 55,25 50,40 C45,20 30,10 10,15 C5,30 15,50 35,55 C45,57.5 48,70 50,85 Z" />
                    <path d="M48,42 C38,30 25,28 15,32 C20,45 32,50 44,46 Z" opacity="0.15" fill="#000" />
                    <path d="M54,38 C68,26 80,18 86,10 C82,24 72,36 56,41 Z" opacity="0.15" fill="#000" />
                </svg>W</span>EEDS
            </div>
            <div class="hero-subtitle">{subtitle}</div>
            """
            ),
            unsafe_allow_html=True,
        )

    @classmethod
    def build_probs_html(
        cls,
        res: PredictionResult | VideoResult,
        class_names: tuple[str, ...],
        caption: str,
    ) -> str:
        """En yüksek 4 sınıfın olasılık çubuklarını döndürür."""
        top_indices = np.argsort(res.probs)[::-1][:4]
        rows = []
        for rank, idx in enumerate(top_indices):
            c_name = class_names[idx]
            info = SpeciesCatalog.get_info(c_name)
            p_val = res.probs[idx] * 100.0
            fill_class = "prob-fill top-hit" if rank == 0 else "prob-fill"
            rows.append(
                f"""
                <div class="prob-row">
                    <span style="min-width:130px; text-align:left;">{info['tr_name']}</span>
                    <div class="prob-track">
                        <div class="{fill_class}" style="width: {p_val:.1f}%;"></div>
                    </div>
                    <span style="min-width:45px; text-align:right;">%{p_val:.1f}</span>
                </div>
                """
            )
        return f"<div class='prob-caption'>{caption}</div>" + "".join(rows)

    @classmethod
    def build_counts_html(cls, v: VideoResult) -> str:
        """Video tespitinin sınıf bazlı sayım çiplerini döndürür."""
        if not v.species_counts:
            return (
                "<div class='counts-block'>"
                "<div class='counts-title'>Tespit Özeti</div>"
                "<span class='count-chip count-zero'>Kutulanmış bitki bulunamadı</span>"
                "</div>"
            )
        chips = [
            f"<span class='count-chip'>&#127807; {SpeciesCatalog.short_name(c_name)} &times; {count}</span>"
            for c_name, count in sorted(v.species_counts.items(), key=lambda kv: -kv[1])
        ]
        return (
            "<div class='counts-block'>"
            f"<div class='counts-title'>Kutu Özeti — Toplam {v.total_plants} Bitki</div>"
            + "".join(chips)
            + "</div>"
        )

    @classmethod
    def render_result_card(
        cls,
        media_html: str,
        res: PredictionResult | VideoResult,
        class_names: tuple[str, ...],
        kind: str = "image",
        media_caption: str = "",
        extra_note: str = "",
    ) -> None:
        """Template yerleşimiyle (media | info) aynı hizda sonuç kartını çizer."""
        media_area = media_html
        if media_caption:
            media_area += f"<div class='media-caption'>{media_caption}</div>"

        if kind == "video" and hasattr(res, "total_plants"):
            heading = "Tespit Edilen Bitkiler:"
            main_title = f"{res.total_plants} adet bitki" if res.total_plants > 0 else "Tespit yok"
            dominant = SpeciesCatalog.short_name(res.top_count_class)
            latin = (
                f"Yoğunluk: {dominant}" if res.total_plants > 0 else "Hedef tür kutulanamadı"
            )
            badge = (
                f"Baskın Tür: {dominant} (%{res.probs[SpeciesCatalog.index_of(res.top_count_class)] * 100:.1f})"
                if res.total_plants > 0
                else f"{SpeciesCatalog.short_name(res.top_class)} — %{res.confidence:.1f}"
            )
            probs_html = cls.build_probs_html(
                res, class_names, "Tam Kare Ortalaması — En Yüksek 4 Olasılık:"
            )
            counts_html = cls.build_counts_html(res)
        else:
            heading = "Tahmin Edilen Tür:"
            main_title = res.tr_name
            latin = f"{res.scientific} · Familya: {res.family}"
            badge = f"Doğruluk: %{res.confidence:.1f}"
            probs_html = cls.build_probs_html(res, class_names, "En Yüksek 4 Olasılık:")
            counts_html = ""

        outer = cls._compact_html(
            f"""
            <div class="results-card">
                <div class="result-media-box">
                    {media_area}
                </div>
                <div class="result-info-box">
                    <div class="status-tag">&#10003; Analiz Tamamlandı ({res.latency_ms:.1f} ms)</div>
                    <h3 class="press-heading">{heading}</h3>
                    <p class="press-result">{main_title}</p>
                    <div class="latin-title">{latin}</div>
                    <div class="confidence-badge">{badge}</div>
                    <div class="desc-box">{res.description} {extra_note}</div>
                    {counts_html}
                    {probs_html}
                </div>
            </div>
            """
        )
        st.markdown(outer, unsafe_allow_html=True)


# ===========================================================================
# 7. Uygulama Yöneticisi (StreamlitApp - Controller / Facade)
# ===========================================================================
class StreamlitApp:
    """Uygulama akışını ve kullanıcı etkileşimini yöneten ana kontrolcü.

    Akış, template'teki 'ana ekran -> sonuç ekranı -> geri dönüş' döngüsünü
    izler: yükleme sonrası streamlit session_state'te tutulan 'payload' ile
    sonuç ekranına geçilir; 'Ana Ekrana Dön' ile üstler temizlenip ana ekran
    tekrar görüntülenir.
    """

    PAYLOAD_KEY = "payload"

    def __init__(self, checkpoint_path: Path | None = None) -> None:
        ckpt_str = str(checkpoint_path or Path("runs/exp-effb3-sqrt/weights/best.pt"))
        self.service = get_model_service(ckpt_str)
        self.engine = InferenceEngine(self.service)

    def _init_session(self) -> None:
        if "uploader_key" not in st.session_state:
            st.session_state["uploader_key"] = 0
        if self.PAYLOAD_KEY not in st.session_state:
            st.session_state[self.PAYLOAD_KEY] = None

    def _advance_uploader_key(self) -> None:
        st.session_state["uploader_key"] = int(st.session_state["uploader_key"]) + 1

    # ------------------------------------------------------------------ #
    # ANA EKRAN
    # ------------------------------------------------------------------ #
    def _render_home(self) -> None:
        ThemeRenderer.render_hero(ThemeRenderer.arch_subtitle(self.service))

        _, col_img, _, col_vid, _ = st.columns([1.5, 2.2, 0.4, 2.2, 1.5])
        with col_img:
            uploaded_image = st.file_uploader(
                "Görsel Yükle",
                type=["jpg", "jpeg", "png", "webp"],
                key=f"img_{st.session_state['uploader_key']}",
                label_visibility="collapsed",
            )
        with col_vid:
            uploaded_video = st.file_uploader(
                "Video Yükle",
                type=["mp4", "avi", "mov", "mkv", "webm"],
                key=f"vid_{st.session_state['uploader_key']}",
                label_visibility="collapsed",
            )

        st.markdown(
            ThemeRenderer._compact_html(
                """
                <div class="sample-tray">
                    <div class="sample-tray-title">Örnek Veriseti Görselleriyle Hızlı Test:</div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )
        cols = st.columns(len(SpeciesCatalog.SAMPLES))
        for i, s in enumerate(SpeciesCatalog.SAMPLES):
            with cols[i]:
                if st.button(s["label"], key=f"btn_sample_{i}", width="stretch"):
                    self._handle_sample(s["path"])
                    return

        if uploaded_image is not None:
            self._handle_uploaded_image(uploaded_image)
        elif uploaded_video is not None:
            self._handle_uploaded_video(uploaded_video)

    def _store_and_rerun(self, payload: dict[str, Any]) -> None:
        st.session_state[self.PAYLOAD_KEY] = payload
        st.rerun()

    def _handle_uploaded_image(self, uploaded_image: Any) -> None:
        try:
            img = Image.open(uploaded_image).convert("RGB")
            with st.spinner("Görsel analiz ediliyor..."):
                result = self.engine.predict_image(img)
            media_b64 = ThemeRenderer.image_to_b64(img)
            self._store_and_rerun(
                {
                    "kind": "image",
                    "media_html": f'<img src="data:image/jpeg;base64,{media_b64}" alt="Görsel">',
                    "media_caption": "",
                    "res": result,
                    "extra_note": "",
                    "vid_bytes": None,
                }
            )
        except (OSError, ValueError, TypeError, KeyError, cv2.error) as e:
            self._advance_uploader_key()
            st.error(f"Görsel işlenemedi: {e}")

    def _handle_uploaded_video(self, uploaded_video: Any) -> None:
        try:
            vid_bytes = uploaded_video.read()
            if len(vid_bytes) > MAX_VIDEO_BYTES:
                st.error("Video 300 MB sınırını aşıyor. Lütfen daha küçük bir video yükleyin.")
                self._advance_uploader_key()
                return
            with st.spinner("Video kareleri taranıyor, bitkiler kutulanıyor..."):
                v_res = analyze_video_cached(
                    vid_bytes,
                    str(self.service.checkpoint_path),
                    DEFAULT_MAX_FRAMES,
                    DEFAULT_WINDOW_SCALES,
                    DEFAULT_CONF_THRESHOLD,
                )
            media_html = ""
            media_caption = ""
            if v_res.annotated_strip_b64:
                media_html = (
                    '<img src="data:image/webp;base64,'
                    f"{v_res.annotated_strip_b64}\" alt=\"Kutulanmış Kare Şeridi\">"
                )
                media_caption = f"İşaretlenmiş kare ilerlemesi · {v_res.frames_analyzed} kare"
            elif v_res.best_frame_b64:
                media_html = (
                    '<img src="data:image/jpeg;base64,'
                    f"{v_res.best_frame_b64}\" alt=\"Kutulanmış Kare\">"
                )
                media_caption = "En yoğun kare"
            self._store_and_rerun(
                {
                    "kind": "video",
                    "media_html": media_html,
                    "media_caption": media_caption,
                    "res": v_res,
                    "extra_note": (
                        f"<br><strong>Video Özeti:</strong> {v_res.duration_sec:.1f} sn · "
                        f"{v_res.frames_analyzed} kare tarandı · {v_res.latency_ms:.0f} ms analiz."
                    ),
                    "vid_bytes": vid_bytes,
                }
            )
        except (OSError, ValueError, TypeError, KeyError, cv2.error) as e:
            self._advance_uploader_key()
            st.error(f"Video işlenemedi: {e}")

    def _handle_sample(self, path_str: str) -> None:
        p = Path(path_str)
        if not p.exists():
            st.error(f"Örnek görsel bulunamadı: {p}")
            return
        try:
            img = Image.open(p).convert("RGB")
            with st.spinner("Görsel analiz ediliyor..."):
                result = self.engine.predict_image(img)
            media_b64 = ThemeRenderer.image_to_b64(img)
            self._store_and_rerun(
                {
                    "kind": "image",
                    "media_html": f'<img src="data:image/jpeg;base64,{media_b64}" alt="Örnek">',
                    "media_caption": "",
                    "res": result,
                    "extra_note": f"<br><small style='color:#7f8c8d;'>Veriseti Örneği: {p.name}</small>",
                    "vid_bytes": None,
                }
            )
        except (OSError, ValueError, TypeError, KeyError, cv2.error) as e:
            st.error(f"Örnek işlenemedi: {e}")

    # ------------------------------------------------------------------ #
    # SONUÇ EKRANI
    # ------------------------------------------------------------------ #
    def _render_results(self, payload: dict[str, Any]) -> None:
        _, col_res, _ = st.columns([1, 6, 1])
        with col_res:
            v = payload["res"]
            ThemeRenderer.render_result_card(
                media_html=payload["media_html"],
                res=v,
                class_names=self.service.class_names,
                kind=payload["kind"],
                media_caption=payload["media_caption"],
                extra_note=payload["extra_note"],
            )

            if payload["kind"] == "video" and hasattr(v, "frames"):
                with st.expander("İşaretlenmiş Kareler", icon="🎬"):
                    self._render_frame_gallery(v)
                if payload["vid_bytes"]:
                    with st.expander("Orijinal Video", icon="▶️"):
                        st.video(payload["vid_bytes"])

            _, col_btn = st.columns([1.9, 1])
            with col_btn:
                if st.button("Ana Ekrana Dön", key="back_to_home", width="stretch"):
                    st.session_state[self.PAYLOAD_KEY] = None
                    self._advance_uploader_key()
                    st.rerun()

    def _render_frame_gallery(self, v: VideoResult) -> None:
        items = []
        for frame in v.frames:
            img_b64 = base64.b64encode(frame.jpeg).decode("utf-8")
            items.append(
                f"""
                <div class="frame-item">
                    <img src="data:image/jpeg;base64,{img_b64}" alt="Kare {frame.index}">
                    <div class="frame-label">
                        <span>&#9200; {frame.t_sec:.1f} sn</span>
                        <span>&#127807; {frame.n_plants} bitki</span>
                    </div>
                </div>
                """
            )
        grid_html = ThemeRenderer._compact_html(f"<div class='frame-grid'>{''.join(items)}</div>")
        st.markdown(grid_html, unsafe_allow_html=True)

    # ------------------------------------------------------------------ #
    # ANA AKIŞ
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        st.set_page_config(
            page_title="DeepWeeds Ot Sınıflandırıcı",
            page_icon="🌿",
            layout="wide",
            initial_sidebar_state="collapsed",
        )

        ThemeRenderer.inject_styles()
        self._init_session()
        ThemeRenderer.render_navbar_and_sidebar(self.service)

        payload = st.session_state.get(self.PAYLOAD_KEY)
        if payload:
            self._render_results(payload)
        else:
            self._render_home()


# ===========================================================================
# 8. Giriş Noktası
# ===========================================================================
def main(checkpoint_path: Path | None = None) -> None:
    app = StreamlitApp(checkpoint_path)
    app.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DeepWeeds Streamlit Uygulaması")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model checkpoint yolu (varsayılan: runs/exp-effb3-sqrt/weights/best.pt)",
    )
    args, _ = parser.parse_known_args()
    main(Path(args.model) if args.model else None)
