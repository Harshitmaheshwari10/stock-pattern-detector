"""
AI Stock Chart Pattern Detector
Streamlit frontend — upload a chart screenshot and get pattern detection results.
"""

import streamlit as st
import numpy as np
from PIL import Image
import io
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pattern_detector import PatternDetector, PatternResult
from visualizer import (
    plot_extracted_price,
    plot_pattern_overlay,
    plot_probability_bars,
    plot_signal_gauge,
)
from sample_generator import SAMPLE_PATTERNS, get_sample_image, price_series_to_chart_image

# ── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Stock Pattern Detector",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
  /* Base */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  .stApp {
    background-color: #0D1117;
    color: #E6EDF3;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background-color: #161B22;
    border-right: 1px solid #21262D;
  }

  /* Cards */
  .pattern-card {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
    position: relative;
    transition: border-color 0.2s;
  }
  .pattern-card:hover { border-color: #58A6FF; }

  .pattern-card.bullish  { border-left: 4px solid #00C896; }
  .pattern-card.bearish  { border-left: 4px solid #FF4D6D; }
  .pattern-card.neutral  { border-left: 4px solid #FFA500; }

  .pattern-name {
    font-size: 1.05rem;
    font-weight: 700;
    color: #E6EDF3;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 0.2rem;
  }

  .pattern-signal-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    margin-left: 0.5rem;
    vertical-align: middle;
  }
  .badge-bullish { background: rgba(0,200,150,0.15); color: #00C896; }
  .badge-bearish { background: rgba(255,77,109,0.15); color: #FF4D6D; }
  .badge-neutral { background: rgba(255,165,0,0.15);  color: #FFA500; }

  .prob-bar-wrap {
    margin: 0.6rem 0 0.3rem;
    background: #0D1117;
    border-radius: 6px;
    height: 8px;
    overflow: hidden;
  }
  .prob-bar {
    height: 100%;
    border-radius: 6px;
    transition: width 0.6s ease;
  }

  .prob-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    font-weight: 600;
    color: #58A6FF;
  }

  .pattern-desc {
    font-size: 0.82rem;
    color: #8B949E;
    margin-top: 0.5rem;
    line-height: 1.5;
  }

  .key-level-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.78rem;
    color: #8B949E;
    font-family: 'JetBrains Mono', monospace;
    padding: 2px 0;
    border-bottom: 1px solid #21262D;
  }
  .key-level-val { color: #E6EDF3; font-weight: 500; }

  .factor-pill {
    display: inline-block;
    background: #1C2128;
    border: 1px solid #30363D;
    color: #8B949E;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.72rem;
    margin: 2px 2px 0 0;
  }

  /* Hero header */
  .hero {
    background: linear-gradient(135deg, #161B22 0%, #0D1117 60%, #1C2128 100%);
    border: 1px solid #21262D;
    border-radius: 12px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.5rem;
  }
  .hero-title {
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #58A6FF, #00C896);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-family: 'JetBrains Mono', monospace;
  }
  .hero-sub {
    color: #8B949E;
    font-size: 0.9rem;
    margin-top: 0.3rem;
  }

  /* Upload box */
  .stFileUploader > div {
    background: #161B22 !important;
    border: 2px dashed #30363D !important;
    border-radius: 10px !important;
  }
  .stFileUploader > div:hover {
    border-color: #58A6FF !important;
  }

  /* Metric boxes */
  .metric-box {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 8px;
    padding: 0.9rem 1rem;
    text-align: center;
  }
  .metric-label {
    font-size: 0.72rem;
    color: #8B949E;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  .metric-value {
    font-size: 1.4rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: #58A6FF;
    margin-top: 0.2rem;
  }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
    background: #161B22;
    border-radius: 8px;
    gap: 4px;
    padding: 4px;
    border: 1px solid #21262D;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #8B949E;
    border-radius: 6px;
    font-size: 0.85rem;
    font-weight: 500;
  }
  .stTabs [aria-selected="true"] {
    background: #1F6FEB !important;
    color: #ffffff !important;
  }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, #1F6FEB, #0D47A1);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.9rem;
    padding: 0.5rem 1.5rem;
    transition: opacity 0.2s;
    width: 100%;
  }
  .stButton > button:hover { opacity: 0.88; }

  /* Divider */
  hr { border-color: #21262D !important; }

  /* Selectbox, radio */
  .stSelectbox > div > div,
  .stRadio > div {
    background: #161B22 !important;
    border-color: #30363D !important;
    color: #E6EDF3 !important;
  }

  /* Spinner */
  .stSpinner > div { border-top-color: #58A6FF !important; }

  /* No patterns */
  .no-patterns {
    text-align: center;
    padding: 3rem 1rem;
    color: #8B949E;
    font-size: 0.9rem;
  }

  /* Info box */
  .info-box {
    background: rgba(31, 111, 235, 0.08);
    border: 1px solid rgba(31, 111, 235, 0.3);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-size: 0.82rem;
    color: #8B949E;
    margin-bottom: 1rem;
  }
</style>
""", unsafe_allow_html=True)


# ── Session State ────────────────────────────────────────────────────────────

if "detector" not in st.session_state:
    st.session_state.detector = PatternDetector()
if "results" not in st.session_state:
    st.session_state.results = None
if "price_series" not in st.session_state:
    st.session_state.price_series = None
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 0.5rem 0 1.2rem;">
      <div style="font-size:2.2rem;">📈</div>
      <div style="font-family:'JetBrains Mono',monospace; font-size:0.95rem;
                  font-weight:700; color:#58A6FF; letter-spacing:0.04em;">
        PatternAI
      </div>
      <div style="font-size:0.72rem; color:#8B949E; margin-top:2px;">
        NSE Chart Pattern Detector
      </div>
    </div>
    <hr>
    """, unsafe_allow_html=True)

    st.markdown("**📥 Input Mode**")
    input_mode = st.radio(
        "Choose input",
        ["Upload Chart Image", "Use Demo Pattern"],
        label_visibility="collapsed",
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    if input_mode == "Use Demo Pattern":
        st.markdown("**🎯 Pattern Demo**")
        selected_demo = st.selectbox(
            "Select pattern",
            list(SAMPLE_PATTERNS.keys()),
            label_visibility="collapsed",
        )
        run_demo = st.button("▶ Run Detection", use_container_width=True)
    else:
        run_demo = False
        selected_demo = None

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("**⚙️ Detection Settings**")
    min_prob = st.slider("Min. Probability Threshold", 0.0, 0.9, 0.25, 0.05,
                         format="%.0f%%",
                         help="Only show patterns above this confidence")
    max_patterns = st.slider("Max patterns to show", 1, 7, 5)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:0.75rem; color:#8B949E; line-height:1.7;">
      <b style="color:#E6EDF3;">Detected Patterns</b><br>
      • Head & Shoulders<br>
      • Double Top / Bottom<br>
      • Ascending Triangle<br>
      • Descending Triangle<br>
      • Symmetrical Triangle<br>
      • Cup & Handle<br>
      • Bull / Bear Flag<br>
      • Rising / Falling Wedge
    </div>
    <hr>
    <div style="font-size:0.72rem; color:#8B949E;">
      Built with Python · OpenCV · Streamlit<br>
      <span style="color:#58A6FF;">github.com/Harshitmaheshwari10</span>
    </div>
    """, unsafe_allow_html=True)


# ── Hero Header ──────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
  <div class="hero-title">📈 AI Stock Chart Pattern Detector</div>
  <div class="hero-sub">
    Upload any NSE/BSE chart screenshot · Computer vision extracts the price curve ·
    ML engine scores 9 chart patterns in real-time
  </div>
</div>
""", unsafe_allow_html=True)


# ── Main Layout ──────────────────────────────────────────────────────────────

col_left, col_right = st.columns([1, 1.6], gap="large")

with col_left:
    st.markdown("#### 🖼 Chart Input")

    # ── Upload Mode ──────────────────────────────────────────────────────────
    if input_mode == "Upload Chart Image":
        st.markdown("""
        <div class="info-box">
          Upload a screenshot from TradingView, Zerodha Kite, NSE website,
          or any charting platform. Works best with line/candlestick charts.
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Drop chart image here",
            type=["png", "jpg", "jpeg", "webp"],
            label_visibility="collapsed",
        )

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.session_state.uploaded_image = image
            st.image(image, caption="Uploaded chart", use_container_width=True)

            if st.button("🔍 Detect Patterns", use_container_width=True):
                with st.spinner("Extracting price curve & running pattern analysis…"):
                    img_array = np.array(image)
                    price_series, patterns = st.session_state.detector.detect_from_image(img_array)
                    st.session_state.price_series = price_series
                    st.session_state.results = patterns

    # ── Demo Mode ────────────────────────────────────────────────────────────
    else:
        st.markdown(f"""
        <div class="info-box">
          Generating synthetic <b style="color:#58A6FF;">{selected_demo}</b> chart
          with realistic noise — useful for testing the detector.
        </div>
        """, unsafe_allow_html=True)

        # Show the demo image
        demo_img = get_sample_image(selected_demo)
        st.image(demo_img, caption=f"Demo: {selected_demo}", use_container_width=True)

        if run_demo:
            with st.spinner("Running pattern analysis on demo chart…"):
                img_array = np.array(demo_img.convert("RGB"))
                price_series, patterns = st.session_state.detector.detect_from_image(img_array)

                # For demo mode, also run directly on known series for accuracy
                series_fn = SAMPLE_PATTERNS[selected_demo]
                known_series = series_fn()
                direct_patterns = st.session_state.detector.detect_all_patterns(known_series)

                # Merge: prefer direct detection, top up with image detection
                seen = {p.name for p in direct_patterns}
                for p in patterns:
                    if p.name not in seen:
                        direct_patterns.append(p)
                        seen.add(p.name)

                st.session_state.price_series = known_series
                st.session_state.results = direct_patterns


# ── Results Panel ────────────────────────────────────────────────────────────

with col_right:
    st.markdown("#### 🧠 Pattern Analysis")

    if st.session_state.results is None:
        st.markdown("""
        <div class="no-patterns">
          <div style="font-size:3rem; margin-bottom:0.8rem;">📊</div>
          <div style="font-size:1rem; color:#E6EDF3; font-weight:600; margin-bottom:0.4rem;">
            No analysis yet
          </div>
          Upload a chart image or choose a demo pattern to get started.
        </div>
        """, unsafe_allow_html=True)

    else:
        results = st.session_state.results
        price_series = st.session_state.price_series

        # Filter by threshold
        filtered = [r for r in results if r.probability >= min_prob][:max_patterns]

        # ── Summary Metrics ──────────────────────────────────────────────────
        top = filtered[0] if filtered else None
        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.markdown(f"""
            <div class="metric-box">
              <div class="metric-label">Patterns Found</div>
              <div class="metric-value">{len(filtered)}</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            top_prob = f"{top.probability*100:.0f}%" if top else "—"
            st.markdown(f"""
            <div class="metric-box">
              <div class="metric-label">Top Confidence</div>
              <div class="metric-value">{top_prob}</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            signal_color = {"BULLISH": "#00C896", "BEARISH": "#FF4D6D", "NEUTRAL": "#FFA500"}
            sig = top.signal if top else "—"
            sig_c = signal_color.get(sig, "#8B949E")
            st.markdown(f"""
            <div class="metric-box">
              <div class="metric-label">Top Signal</div>
              <div class="metric-value" style="color:{sig_c}; font-size:1rem;">{sig}</div>
            </div>""", unsafe_allow_html=True)
        with m4:
            bull_count = sum(1 for r in filtered if r.signal == "BULLISH")
            bear_count = sum(1 for r in filtered if r.signal == "BEARISH")
            bias = "BULL" if bull_count > bear_count else ("BEAR" if bear_count > bull_count else "MIX")
            bias_c = "#00C896" if bias == "BULL" else ("#FF4D6D" if bias == "BEAR" else "#FFA500")
            st.markdown(f"""
            <div class="metric-box">
              <div class="metric-label">Overall Bias</div>
              <div class="metric-value" style="color:{bias_c}; font-size:1rem;">{bias}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Tabs ─────────────────────────────────────────────────────────────
        tab_patterns, tab_charts, tab_raw = st.tabs(
            ["🎯 Patterns", "📊 Visualizations", "🔬 Price Data"]
        )

        # ── Tab 1: Pattern Cards ──────────────────────────────────────────────
        with tab_patterns:
            if not filtered:
                st.markdown("""
                <div class="no-patterns">
                  No patterns above the confidence threshold.<br>
                  Try lowering the threshold in the sidebar.
                </div>""", unsafe_allow_html=True)
            else:
                for idx, pattern in enumerate(filtered):
                    sig = pattern.signal.lower()
                    sig_icon = {"bullish": "▲", "bearish": "▼", "neutral": "◆"}.get(sig, "●")
                    prob_pct = int(pattern.probability * 100)
                    bar_color = {"bullish": "#00C896", "bearish": "#FF4D6D", "neutral": "#FFA500"}.get(sig, "#58A6FF")

                    # Build key levels HTML
                    levels_html = ""
                    for k, v in pattern.key_levels.items():
                        levels_html += f"""
                        <div class="key-level-row">
                          <span>{k}</span>
                          <span class="key-level-val">{v}</span>
                        </div>"""

                    # Build confidence factors
                    factors_html = " ".join([
                        f'<span class="factor-pill">✓ {f}</span>'
                        for f in pattern.confidence_factors if f
                    ])

                    rank_badge = ["🥇", "🥈", "🥉", "4th", "5th", "6th", "7th"][idx] if idx < 7 else f"#{idx+1}"

                    st.markdown(f"""
                    <div class="pattern-card {sig}">
                      <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                          <span style="font-size:0.75rem; color:#8B949E; margin-right:6px;">{rank_badge}</span>
                          <span class="pattern-name">{pattern.name}</span>
                          <span class="pattern-signal-badge badge-{sig}">{sig_icon} {pattern.signal}</span>
                        </div>
                        <span class="prob-label">{prob_pct}%</span>
                      </div>

                      <div class="prob-bar-wrap">
                        <div class="prob-bar" style="width:{prob_pct}%; background:{bar_color};"></div>
                      </div>

                      <div class="pattern-desc">{pattern.description}</div>

                      <details style="margin-top:0.7rem;">
                        <summary style="cursor:pointer; font-size:0.78rem; color:#58A6FF; font-weight:600;">
                          Key Levels & Confirmation
                        </summary>
                        <div style="margin-top:0.5rem;">
                          {levels_html}
                          <div style="margin-top:0.5rem;">{factors_html}</div>
                        </div>
                      </details>
                    </div>
                    """, unsafe_allow_html=True)

        # ── Tab 2: Charts ─────────────────────────────────────────────────────
        with tab_charts:
            if price_series is not None and len(price_series) > 10:
                # Gauge
                if top:
                    g_col1, g_col2 = st.columns([1, 2])
                    with g_col1:
                        gauge_fig = plot_signal_gauge(top.probability, top.signal)
                        st.plotly_chart(gauge_fig, use_container_width=True, config={"displayModeBar": False})
                    with g_col2:
                        prob_fig = plot_probability_bars(filtered)
                        st.plotly_chart(prob_fig, use_container_width=True, config={"displayModeBar": False})

                # Extracted price curve
                price_fig = plot_extracted_price(price_series, "Extracted Price Curve")
                st.plotly_chart(price_fig, use_container_width=True, config={"displayModeBar": False})

                # Pattern overlay
                overlay_fig = plot_pattern_overlay(price_series, filtered, show_top_n=1)
                st.plotly_chart(overlay_fig, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("Price series could not be extracted from the image. Try a cleaner chart screenshot.")

        # ── Tab 3: Raw Data ───────────────────────────────────────────────────
        with tab_raw:
            if price_series is not None:
                import pandas as pd

                st.markdown("**📈 Extracted Price Series**")
                n = len(price_series)
                df = pd.DataFrame({
                    "Index": np.arange(n),
                    "Normalized Price": np.round(price_series, 2),
                    "Rolling Avg (15)": pd.Series(price_series).rolling(15).mean().round(2),
                })
                st.dataframe(
                    df,
                    use_container_width=True,
                    height=220,
                    hide_index=True,
                )

                st.markdown("**📋 Detection Summary**")
                summary_data = []
                for p in results:
                    summary_data.append({
                        "Pattern": p.name,
                        "Probability": f"{p.probability*100:.1f}%",
                        "Signal": p.signal,
                        "Description": p.description[:60] + "…",
                    })
                if summary_data:
                    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

                # Download CSV
                csv_data = pd.DataFrame({
                    "index": np.arange(n),
                    "price": np.round(price_series, 4),
                }).to_csv(index=False)
                st.download_button(
                    "⬇️ Download Price CSV",
                    data=csv_data,
                    file_name="extracted_price_series.csv",
                    mime="text/csv",
                )


# ── Footer ───────────────────────────────────────────────────────────────────

st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#8B949E; font-size:0.78rem; padding:0.5rem 0;">
  Built by <b style="color:#58A6FF;">Harshit Maheshwari</b> ·
  Python · OpenCV · Streamlit · Plotly ·
  <a href="https://github.com/Harshitmaheshwari10" target="_blank"
     style="color:#58A6FF; text-decoration:none;">GitHub ↗</a>
</div>
""", unsafe_allow_html=True)
