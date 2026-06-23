"""
Sample Chart Generator
Creates synthetic chart images with known patterns for demo and testing.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from io import BytesIO
from PIL import Image


def _base_noise(n, seed=42):
    rng = np.random.RandomState(seed)
    return rng.normal(0, 0.3, n).cumsum() * 0.1


def generate_head_and_shoulders(n=200) -> np.ndarray:
    """Generate a Head & Shoulders price series."""
    x = np.linspace(0, 1, n)
    base = 50 + x * 5

    # Left shoulder
    ls_center = 0.15
    ls_height = 12
    left_shoulder = ls_height * np.exp(-((x - ls_center)**2) / (2 * 0.025**2))

    # Head
    head_center = 0.45
    head_height = 20
    head = head_height * np.exp(-((x - head_center)**2) / (2 * 0.03**2))

    # Right shoulder
    rs_center = 0.75
    rs_height = 11
    right_shoulder = rs_height * np.exp(-((x - rs_center)**2) / (2 * 0.025**2))

    noise = _base_noise(n, 1)
    price = base + left_shoulder + head + right_shoulder + noise
    return price


def generate_double_top(n=200) -> np.ndarray:
    """Generate a Double Top price series."""
    x = np.linspace(0, 1, n)
    base = 45 + x * 8

    peak1 = 18 * np.exp(-((x - 0.25)**2) / (2 * 0.04**2))
    valley = -8 * np.exp(-((x - 0.5)**2) / (2 * 0.04**2))
    peak2 = 17 * np.exp(-((x - 0.75)**2) / (2 * 0.04**2))

    noise = _base_noise(n, 2)
    price = base + peak1 + valley + peak2 + noise
    # Add downtrend after second peak
    decline = np.zeros(n)
    decline[int(n*0.80):] = -np.linspace(0, 12, n - int(n*0.80))
    return price + decline


def generate_double_bottom(n=200) -> np.ndarray:
    """Generate a Double Bottom price series."""
    x = np.linspace(0, 1, n)
    base = 65 - x * 10

    bottom1 = -18 * np.exp(-((x - 0.25)**2) / (2 * 0.04**2))
    bounce = 8 * np.exp(-((x - 0.5)**2) / (2 * 0.04**2))
    bottom2 = -17 * np.exp(-((x - 0.75)**2) / (2 * 0.04**2))

    noise = _base_noise(n, 3)
    price = base + bottom1 + bounce + bottom2 + noise
    recovery = np.zeros(n)
    recovery[int(n*0.80):] = np.linspace(0, 12, n - int(n*0.80))
    return price + recovery


def generate_ascending_triangle(n=200) -> np.ndarray:
    """Generate an Ascending Triangle price series."""
    rng = np.random.RandomState(7)
    price = np.zeros(n)
    price[0] = 50
    resistance = 65

    for i in range(1, n):
        # Rising support
        support = 50 + (i / n) * 12
        mid = (support + resistance) / 2
        # Oscillate within narrowing range
        amplitude = (resistance - support) / 2
        wave = amplitude * np.sin(i * 0.3) * (1 - i / (n * 1.5))
        noise = rng.normal(0, 0.4)
        price[i] = mid + wave + noise

    return np.clip(price, 44, 70)


def generate_cup_and_handle(n=200) -> np.ndarray:
    """Generate a Cup & Handle price series."""
    cup_n = int(n * 0.78)
    handle_n = n - cup_n

    # Cup: U-shaped
    x_cup = np.linspace(-np.pi/2, np.pi/2, cup_n)
    cup = 55 + 15 * np.cos(x_cup) - 15  # U shape
    cup_noise = _base_noise(cup_n, 5) * 0.8
    cup = cup + cup_noise

    # Handle: slight downward drift then recovery
    x_handle = np.linspace(0, 1, handle_n)
    handle_base = cup[-1]
    handle = handle_base - 3 * np.sin(x_handle * np.pi) + x_handle * 1
    handle_noise = _base_noise(handle_n, 6) * 0.5
    handle = handle + handle_noise

    price = np.concatenate([cup, handle])
    return price


def generate_bull_flag(n=200) -> np.ndarray:
    """Generate a Bull Flag price series."""
    pole_n = int(n * 0.25)
    flag_n = n - pole_n

    rng = np.random.RandomState(9)

    # Strong upward pole
    pole = np.linspace(40, 70, pole_n) + rng.normal(0, 0.5, pole_n)

    # Consolidating flag (slight downward drift)
    flag_x = np.linspace(0, 1, flag_n)
    flag = 70 - 4 * flag_x + rng.normal(0, 0.6, flag_n)
    # Add oscillation
    flag += 2 * np.sin(flag_x * 15)

    return np.concatenate([pole, flag])


def generate_symmetrical_triangle(n=200) -> np.ndarray:
    """Generate a Symmetrical Triangle price series."""
    rng = np.random.RandomState(11)
    x = np.linspace(0, 1, n)

    upper = 70 - 15 * x
    lower = 50 + 15 * x

    # Oscillate between converging trendlines
    t = np.linspace(0, 1, n)
    amplitude = (upper - lower) / 2 * (1 - t * 0.9)
    mid = (upper + lower) / 2
    price = mid + amplitude * np.sin(t * 20) + rng.normal(0, 0.5, n)

    return price


SAMPLE_PATTERNS = {
    "Head & Shoulders": generate_head_and_shoulders,
    "Double Top": generate_double_top,
    "Double Bottom": generate_double_bottom,
    "Ascending Triangle": generate_ascending_triangle,
    "Cup & Handle": generate_cup_and_handle,
    "Bull Flag": generate_bull_flag,
    "Symmetrical Triangle": generate_symmetrical_triangle,
}


def price_series_to_chart_image(
    price_series: np.ndarray,
    pattern_name: str = "",
    width: int = 900,
    height: int = 500,
    style: str = "dark",
) -> Image.Image:
    """Render a price series as a realistic-looking chart image."""
    dpi = 100
    fig_w, fig_h = width / dpi, height / dpi

    # Style presets
    if style == "dark":
        bg_color = '#0D1117'
        line_color = '#58A6FF'
        fill_color = 'rgba(88, 166, 255, 0.1)'
        grid_color = '#21262D'
        text_color = '#E6EDF3'
        plt.rcParams['text.color'] = text_color
        plt.rcParams['axes.labelcolor'] = text_color
        plt.rcParams['xtick.color'] = '#8B949E'
        plt.rcParams['ytick.color'] = '#8B949E'
    else:
        bg_color = '#FFFFFF'
        line_color = '#1A56DB'
        grid_color = '#E5E7EB'
        text_color = '#111827'

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    n = len(price_series)
    x = np.arange(n)

    # Plot price line with fill
    ax.plot(x, price_series, color=line_color, linewidth=2.0, zorder=3)
    ax.fill_between(x, price_series, price_series.min() - 2,
                    alpha=0.12, color=line_color)

    # Grid
    ax.grid(True, color=grid_color, linewidth=0.5, alpha=0.7, zorder=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ax.spines.values():
        spine.set_edgecolor(grid_color)

    # Labels
    title = f"Chart — {pattern_name}" if pattern_name else "Stock Price Chart"
    ax.set_title(title, color=text_color, fontsize=12, pad=10, fontweight='bold')
    ax.set_xlabel("Time", color=text_color, fontsize=9)
    ax.set_ylabel("Price", color=text_color, fontsize=9)

    # Add fake price ticks on y-axis (scale to realistic numbers)
    min_p, max_p = price_series.min(), price_series.max()
    range_p = max_p - min_p
    base_price = 1000
    scale = base_price / 50
    y_ticks = ax.get_yticks()
    ax.set_yticklabels([f"₹{(v * scale + base_price):.0f}" for v in y_ticks], fontsize=8)

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                facecolor=bg_color, edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def get_sample_image(pattern_name: str, style: str = "dark") -> Image.Image:
    """Get a sample chart image for a named pattern."""
    if pattern_name not in SAMPLE_PATTERNS:
        pattern_name = list(SAMPLE_PATTERNS.keys())[0]

    series = SAMPLE_PATTERNS[pattern_name]()
    return price_series_to_chart_image(series, pattern_name, style=style)
