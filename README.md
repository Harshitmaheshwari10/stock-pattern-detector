# 📈 AI Stock Chart Pattern Detector

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Upload any NSE/BSE chart screenshot · Computer vision extracts the price curve · ML scores 9 chart patterns**

[Live Demo](#deployment) · [Features](#features) · [Setup](#quickstart) · [Architecture](#architecture)

</div>

---

## ✨ Features

| Feature | Details |
|---|---|
| 🖼 **Chart Image Upload** | Accepts PNG, JPG, WEBP from TradingView, Zerodha Kite, NSE site, etc. |
| 🔬 **OpenCV Extraction** | Multi-strategy price curve extraction from chart screenshots |
| 🧠 **9 Pattern Detectors** | Head & Shoulders, Double Top/Bottom, 3 Triangles, Cup & Handle, Flags, Wedges |
| 📊 **Probability Scores** | Confidence score (0–100%) for each detected pattern |
| 🎯 **Signal Classification** | BULLISH / BEARISH / NEUTRAL with key price levels |
| 📉 **Interactive Charts** | Plotly visualizations with dark theme, overlays, and gauge |
| 🎮 **Demo Mode** | 7 synthetic chart patterns for instant testing |
| ⬇️ **CSV Export** | Download extracted price series for further analysis |

---

## 🏗 Architecture

```
stock-pattern-detector/
├── app.py                    # Streamlit frontend (UI, routing, layout)
├── src/
│   ├── pattern_detector.py   # Core detection engine (OpenCV + scipy)
│   ├── visualizer.py         # Plotly chart rendering
│   └── sample_generator.py   # Synthetic chart generator (demo + tests)
├── tests/
│   └── test_patterns.py      # pytest unit tests (30+ tests)
├── .streamlit/
│   └── config.toml           # Dark theme + server config
├── .github/
│   └── workflows/ci.yml      # GitHub Actions CI pipeline
├── Dockerfile                # Container deployment
├── requirements.txt
└── README.md
```

### Detection Pipeline

```
Chart Image (PNG/JPG)
       │
       ▼
┌─────────────────────┐
│   ChartExtractor    │  ← OpenCV: color detection, edge tracking,
│   (OpenCV)          │    gradient analysis, gap interpolation
└─────────────────────┘
       │  price series (normalized 0–100)
       ▼
┌─────────────────────┐
│   PatternDetector   │  ← scipy signal processing: peaks, valleys,
│   (Signal Analysis) │    trendlines, shape correlation
└─────────────────────┘
       │  List[PatternResult] sorted by probability
       ▼
┌─────────────────────┐
│   Streamlit UI      │  ← Pattern cards, Plotly charts, gauge, export
└─────────────────────┘
```

---

## 🚀 Quickstart

### Local Setup

```bash
# 1. Clone
git clone https://github.com/Harshitmaheshwari10/ai-stock-pattern-detector.git
cd ai-stock-pattern-detector

# 2. Create virtualenv
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install
pip install -r requirements.txt

# 4. Run
streamlit run app.py
# → Opens at http://localhost:8501
```

### Run Tests

```bash
pip install pytest pytest-cov
pytest tests/ -v --cov=src
```

### Docker

```bash
docker build -t pattern-detector .
docker run -p 8501:8501 pattern-detector
# → http://localhost:8501
```

---

## ☁️ Deployment

### Streamlit Community Cloud (Recommended — Free)

1. Push repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → set `app.py` as main file
4. Deploy → get a public URL in ~2 minutes

### Railway / Render

```bash
# Procfile (already included)
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

---

## 🧠 Detected Patterns

| Pattern | Signal | Key Characteristics |
|---|---|---|
| **Head & Shoulders** | 🔴 BEARISH | 3 peaks, middle highest, flat neckline |
| **Double Top** | 🔴 BEARISH | 2 equal peaks, valley in between |
| **Double Bottom** | 🟢 BULLISH | 2 equal troughs, peak in between |
| **Ascending Triangle** | 🟢 BULLISH | Flat top resistance, rising support |
| **Descending Triangle** | 🔴 BEARISH | Flat support, falling resistance |
| **Symmetrical Triangle** | 🟡 NEUTRAL | Both trendlines converging |
| **Cup & Handle** | 🟢 BULLISH | U-shaped cup + small pullback handle |
| **Bull / Bear Flag** | 🟢/🔴 | Sharp pole + tight consolidation |
| **Rising / Falling Wedge** | 🔴/🟢 | Converging trendlines, same direction |

---

## 🔬 How It Works

### 1. Price Extraction (OpenCV)

Three strategies are attempted in order, falling back if confidence is low:

- **Color-based detection** — identifies dominant chart line color (blue, green, orange, dark) using HSV color space masking
- **Darkness-based detection** — Otsu thresholding to find dark line pixels per column
- **Edge tracking** — Canny edge detection with centroid tracking

Gaps are filled via linear interpolation; the result is smoothed with a Savitzky-Golay filter.

### 2. Pattern Detection (scipy + numpy)

Each detector uses:
- `scipy.signal.argrelextrema` for peak/valley detection
- Custom scoring functions combining symmetry, proportions, and shape metrics
- `scipy.stats.pearsonr` for Cup & Handle roundness correlation
- Linear regression for trendline slope analysis

### 3. Probability Scoring

Each pattern computes a `score ∈ [0, 1]` from weighted sub-criteria:
```python
score = (
    w1 * symmetry_score +
    w2 * depth_score +
    w3 * shape_score +
    w4 * proportion_score
)
probability = min(0.95, score * scale_factor)
```

---

## 📸 Screenshots

> **Upload Mode** — drop a TradingView or Kite screenshot, get instant analysis
> **Demo Mode** — explore all 7 synthetic patterns with one click
> **Dark chart theme** — Plotly visualizations with pattern overlays and gauge

---

## 🧰 Tech Stack

- **Python 3.10+**
- **Streamlit** — web UI framework
- **OpenCV (headless)** — image processing and price extraction
- **NumPy / SciPy** — signal processing and pattern scoring
- **Plotly** — interactive visualizations
- **Matplotlib** — synthetic chart generation
- **Pillow** — image handling
- **pytest** — unit testing

---

## 👤 Author

**Harshit Maheshwari**
B.Tech IT · Manipal University Jaipur · GPA 9.28 · Dean's List ×6

[![GitHub](https://img.shields.io/badge/GitHub-Harshitmaheshwari10-181717?style=flat&logo=github)](https://github.com/Harshitmaheshwari10)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
