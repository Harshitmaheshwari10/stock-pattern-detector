"""
Chart Visualization Module
Renders price series and overlays detected pattern annotations.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pattern_detector import PatternResult


SIGNAL_COLORS = {
    "BULLISH": "#00C896",
    "BEARISH": "#FF4D6D",
    "NEUTRAL": "#FFA500",
}

SIGNAL_ICONS = {
    "BULLISH": "▲",
    "BEARISH": "▼",
    "NEUTRAL": "◆",
}

# Dark chart theme
THEME = {
    "bg": "#0D1117",
    "paper": "#161B22",
    "grid": "#21262D",
    "text": "#E6EDF3",
    "subtext": "#8B949E",
    "line": "#58A6FF",
    "accent": "#1F6FEB",
}


def plot_extracted_price(price_series: np.ndarray, title: str = "Extracted Price Curve") -> go.Figure:
    """Plot the extracted price series from the chart image."""
    n = len(price_series)
    x = np.linspace(0, 100, n)

    fig = go.Figure()

    # Gradient-like area fill
    fig.add_trace(go.Scatter(
        x=x, y=price_series,
        mode='lines',
        name='Price',
        line=dict(color=THEME["line"], width=2.5),
        fill='tozeroy',
        fillcolor='rgba(88, 166, 255, 0.08)',
    ))

    # Overlay rolling average
    if n > 20:
        window = max(5, n // 15)
        rolling_avg = np.convolve(price_series, np.ones(window)/window, mode='valid')
        x_avg = np.linspace(0, 100, len(rolling_avg))
        fig.add_trace(go.Scatter(
            x=x_avg, y=rolling_avg,
            mode='lines',
            name='Smoothed',
            line=dict(color='rgba(255, 165, 0, 0.6)', width=1.5, dash='dot'),
        ))

    _apply_dark_theme(fig, title)
    fig.update_layout(height=320)
    return fig


def plot_pattern_overlay(
    price_series: np.ndarray,
    patterns: List[PatternResult],
    show_top_n: int = 1,
) -> go.Figure:
    """Plot price series with pattern annotations overlaid."""
    n = len(price_series)
    x = np.linspace(0, 100, n)

    fig = go.Figure()

    # Base price line
    fig.add_trace(go.Scatter(
        x=x, y=price_series,
        mode='lines',
        name='Price',
        line=dict(color=THEME["line"], width=2),
        fill='tozeroy',
        fillcolor='rgba(88, 166, 255, 0.06)',
    ))

    # Overlay top patterns
    top_patterns = patterns[:show_top_n]
    for pattern in top_patterns:
        color = SIGNAL_COLORS.get(pattern.signal, "#FFA500")
        _add_pattern_annotations(fig, price_series, pattern, x, color)

    _apply_dark_theme(fig, "Pattern Detection Overlay")
    fig.update_layout(height=380)
    return fig


def _add_pattern_annotations(fig, price_series, pattern: PatternResult, x, color):
    """Add visual annotations for a pattern."""
    n = len(price_series)

    # Add horizontal lines for key levels
    for level_name, level_val in pattern.key_levels.items():
        if "Target" in level_name:
            dash = "dash"
            opacity = 0.5
        else:
            dash = "solid"
            opacity = 0.7

        # Normalize level to price series range
        min_p, max_p = np.min(price_series), np.max(price_series)
        if max_p == min_p:
            continue

        # If level_val is in normalized space (0-100), use directly
        # Map level to actual y value
        actual_val = np.clip(level_val, min_p - 10, max_p + 10)

        fig.add_hline(
            y=actual_val,
            line=dict(color=color, width=1.2, dash=dash),
            opacity=opacity,
            annotation_text=f"{level_name}: {level_val:.1f}",
            annotation_position="right",
            annotation_font_size=9,
            annotation_font_color=color,
        )

    # Add pattern name annotation in the center
    mid_x = 50
    mid_y = (np.max(price_series) + np.min(price_series)) / 2

    icon = SIGNAL_ICONS.get(pattern.signal, "◆")
    fig.add_annotation(
        x=mid_x, y=np.max(price_series) * 0.95,
        text=f"{icon} {pattern.name} ({pattern.probability*100:.0f}%)",
        showarrow=False,
        font=dict(size=12, color=color, family="monospace"),
        bgcolor=f"rgba(13, 17, 23, 0.8)",
        bordercolor=color,
        borderwidth=1,
        borderpad=4,
    )


def plot_probability_bars(patterns: List[PatternResult]) -> go.Figure:
    """Horizontal bar chart of pattern probabilities."""
    if not patterns:
        return go.Figure()

    names = [p.name for p in patterns]
    probs = [p.probability * 100 for p in patterns]
    colors = [SIGNAL_COLORS.get(p.signal, "#FFA500") for p in patterns]
    signals = [f"{SIGNAL_ICONS.get(p.signal, '')} {p.signal}" for p in patterns]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=names,
        x=probs,
        orientation='h',
        marker=dict(
            color=colors,
            opacity=0.85,
            line=dict(color=colors, width=1),
        ),
        text=[f"{p:.1f}%" for p in probs],
        textposition='outside',
        textfont=dict(color=THEME["text"], size=12),
        hovertemplate='<b>%{y}</b><br>Probability: %{x:.1f}%<extra></extra>',
    ))

    # Add signal labels
    for i, (name, signal) in enumerate(zip(names, signals)):
        fig.add_annotation(
            x=2, y=i,
            text=signal,
            showarrow=False,
            font=dict(size=10, color=colors[i]),
            xanchor='left',
        )

    _apply_dark_theme(fig, "Pattern Probability Scores")
    fig.update_layout(
        height=max(200, len(patterns) * 55 + 80),
        xaxis=dict(
            range=[0, 115],
            ticksuffix="%",
            title="Probability",
        ),
        yaxis=dict(
            autorange="reversed",
            title="",
        ),
        showlegend=False,
    )
    return fig


def plot_signal_gauge(probability: float, signal: str) -> go.Figure:
    """Gauge chart showing top pattern confidence."""
    color = SIGNAL_COLORS.get(signal, "#FFA500")

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Pattern Confidence", 'font': {'size': 14, 'color': THEME["text"]}},
        number={'suffix': "%", 'font': {'size': 28, 'color': color}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': THEME["subtext"]},
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': THEME["paper"],
            'borderwidth': 0,
            'steps': [
                {'range': [0, 40], 'color': '#1C2128'},
                {'range': [40, 65], 'color': '#1C2128'},
                {'range': [65, 100], 'color': '#1C2128'},
            ],
            'threshold': {
                'line': {'color': color, 'width': 3},
                'thickness': 0.75,
                'value': probability * 100,
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor=THEME["paper"],
        plot_bgcolor=THEME["paper"],
        font=dict(color=THEME["text"]),
        height=220,
        margin=dict(l=20, r=20, t=40, b=10),
    )
    return fig


def _apply_dark_theme(fig: go.Figure, title: str):
    """Apply consistent dark theme to all charts."""
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=14, color=THEME["text"], family="monospace"),
            x=0.02,
        ),
        paper_bgcolor=THEME["paper"],
        plot_bgcolor=THEME["bg"],
        font=dict(color=THEME["text"], family="Inter, sans-serif"),
        xaxis=dict(
            gridcolor=THEME["grid"],
            zerolinecolor=THEME["grid"],
            tickfont=dict(color=THEME["subtext"], size=10),
        ),
        yaxis=dict(
            gridcolor=THEME["grid"],
            zerolinecolor=THEME["grid"],
            tickfont=dict(color=THEME["subtext"], size=10),
        ),
        legend=dict(
            bgcolor="rgba(22, 27, 34, 0.8)",
            bordercolor=THEME["grid"],
            borderwidth=1,
            font=dict(color=THEME["text"], size=10),
        ),
        margin=dict(l=40, r=40, t=50, b=40),
        hovermode='x unified',
    )
