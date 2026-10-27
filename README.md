# 🌿 DeepWeeds AI — Yabani Ot Sınıflandırma Uygulaması

DeepWeeds veri seti üzerinde eğitilmiş derin öğrenme modeliyle **8 istilacı yabani ot türünü** sınıflandıran ve video karelerinde **bitkileri kutulayıp sayan** Streamlit uygulaması.

Model, giriş görüntüsünü 9 sınıftan birine atar: 8 hedef yabani ot türü ya da **Negative** (temiz arazi — hedef tür yok).

## Özellikler

- **🖼️ Görsel sınıflandırma** — Yüklenen görüntüyü sınıflandırır; Türkçe tür adı, bilimsel ad, familya ve açıklama ile olasılık çubukları gösterir.
- **🎬 Video tespiti** — Videodan örneklenen karelerde *sliding-window* sınıflandırma ile bitkileri kutular, tür etiketi yazar ve IoU takibiyle **benzersiz bitki sayımı** yapar.
- **⚖️ Kalibrasyon** — Checkpoint yanındaki `calibration.json` içindeki sınıf çarpanları uygulanarak makro-F1 dengelenir (0.818 → 0.857).
- **🌡️ TTA** — Tek görsel için 5 farklı transform ortalamasıyla daha stabil tahmin (`predict_tta`).
- **📊 Örnek görseller** — Ana ekranda veri setinden 9 örnek görselle hızlı test.
- **💻 Cihaz algılama** — CUDA varsa GPU, yoksa CPU üzerinde çalışır (otomatik).

## Sınıflar

| # | Sınıf | Türkçe | Bilimsel Ad |
|---|-------|--------|-------------|
| 0 | Chinee apple | Hint hünnabı | *Ziziphus mauritiana* |
| 1 | Lantana | Ağaç minesi | *Lantana camara* |
| 2 | Parkinsonia | Kudüs dikeni | *Parkinsonia aculeata* |
| 3 | Parthenium | Partenyum otu | *Parthenium hysterophorus* |
| 4 | Prickly acacia | Dikenli akasya | *Vachellia nilotica* |
| 5 | Rubber vine | Kauçuk sarmaşığı | *Cryptostegia grandiflora* |
| 6 | Siam weed | Siyam otu | *Chromolaena odorata* |
| 7 | Snake weed | Yılan otu | *Stachytarpheta spp.* |
| 8 | Negative | Temiz Arazi | — |

## Proje Yapısı

```
model/
├── main.py              # Streamlit uygulaması (arayüz + video tespit hattı)
├── inference.py         # Predictor: checkpoint yükleme ve çıkarım API'si
├── models.py            # build_model: timm öncelikli, torchvision yedekli
├── data.py              # DeepWeedsDataset + eval/train/TTA transformları
├── config.py            # Sınıf adları, veri/model/eğitim config dataclass'ları
├── requirements.txt     # Bağımlılıklar
├── assets/
│   └── leaf.jpeg        # Logo görseli (menü düğmesi)
└── runs/
    └── exp-effb3-sqrt/  # Eğitim çıktıları
        ├── weights/best.pt        # Model ağırlıkları (43 MB)
        ├── calibration.json       # Sınıf çarpanları (kalibrasyon)
        ├── metrics.json           # Doğrulama metrikleri
        ├── config.json            # Eğitim yapılandırması
        ├── history.csv            # Epoch bazlı eğitim geçmişi
        └── predictions_val.csv    # Doğrulama seti tahminleri
```

## Kurulum

Python **3.10+** (3.12 ile test edildi) gerektirir.

```bash
pip install -r requirements.txt
```

## Kullanım

```bash
streamlit run main.py
```

Farklı bir checkpoint ile:

```bash
streamlit run main.py --model runs/exp-effb3-sqrt/weights/best.pt
```

1. Sol üstteki **yaprak düğmesine** tıklayarak model bilgisi panelini açabilirsiniz.
2. **Görsel Yükle** kutusuna bir JPG/PNG/WEBP bırakın ya da örnek görsel düğmelerinden birine basın.
3. **Video Yükle** kutusuna bir MP4/AVI/MOV/MKV/WEBM bırakın (maks. 300 MB); işaretlenmiş kare şeridi, kare galerisi, kutu özeti ve baskın tür raporlanır.

## Model Performansı

**EfficientNet-B3** (timm, `img_size=256`, `sqrt_weighted_loss` dengesi) — doğrulama seti:

| Metrik | Değer |
|--------|-------|
| Macro F1 (ham) | 0.818 |
| Macro F1 (kalibre) | **0.857** |
| Accuracy | 0.859 |

Sınıf bazlı F1: Parkinsonia 0.94 · Parthenium 0.90 · Siam weed 0.90 · Lantana 0.89 · Rubber vine 0.88 · Negative 0.86 · Prickly acacia 0.73 · Chinee apple 0.66 · Snake weed 0.60

## Python API

Uygulamayı açmadan modeli doğrudan da kullanabilirsiniz:

```python
from PIL import Image
from inference import Predictor

predictor = Predictor.from_checkpoint("runs/exp-effb3-sqrt/weights/best.pt")

img = Image.open("ornek.jpg").convert("RGB")

probs = predictor.predict_pil(img)          # shape (9,) — softmax olasılıkları
batch = predictor.predict_batch([img, img]) # shape (N, 9)
tta = predictor.predict_tta(img)            # 5 transform ortalaması
```

Sonucu sınıf adına çevirmek için:

```python
from config import CLASS_NAMES
tur = CLASS_NAMES[int(probs.argmax())]
```

## Testler

Hataları erken yakalamak için pytest tabanlı test paketi bulunur (`tests/`):

```bash
pytest                # tüm testler
pytest -m "not slow"  # uzun sürenler hariç
pytest tests/test_inference.py -v  # tek dosya
```

Testler, gerçek checkpoint yerine 1M parametrik sentetik bir checkpoint
kullanır; böylece ağırlık dosyası olmasa bile tüm çıkarım hattı doğrulanır.
Gerçek checkpoint ve tüm uygulama render'ı `slow` işaretli testlerde koşar.

## Video Tespit Hattı Nasıl Çalışır?

Model yalnızca **sınıflandırıcı** olduğundan, video kareleri şu adımlarla işlenir:

1. Videodan en fazla **10 kare** örnekleme (eşit aralıklı).
2. Her kare, iki ölçekte (%42, %72) örtüşen pencerelere bölünüp parça-parça sınıflandırılır (maks. 26 pencere/kare).
3. "Negative" dışına çıkan, güveni %60 üstü pencereler bitki kutusu olur; **NMS** ile çakışan kutular elenir.
4. Kareler arası **IoU takibi** aynı bitkinin tekrar sayılmasını engeller (2 kare tolerans).
5. Sonuç: işaretli kare şeridi (animasyonlu WEBP), kare galerisi, sınıf bazlı sayım ve baskın tür.

## Notlar

- Checkpoint'ler timm anahtar adlarıyla kaydedilir; `models.py` timm'i önceliklendirir, timm yoksa desteklenen torchvision modellerine düşer.
- Kalibrasyon çarpanları `calibration.json` bulunursa otomatik uygulanır; olasılıklar yeniden normalize edilir.
- Eğitim betikleri bu depoda yer almaz; yalnızca eğitilmiş model ve çıkarım/uygulama katmanı paylaşılmıştır.
- Veri seti: [DeepWeeds — University of Queensland](https://github.com/AlexOlsen/DeepWeeds).
