"""
Pattern Detection Engine
Detects chart patterns using computer vision and signal analysis.
"""

import cv2
import numpy as np
from scipy.signal import argrelextrema, find_peaks, savgol_filter
from scipy.stats import pearsonr
from dataclasses import dataclass
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")


@dataclass
class PatternResult:
    name: str
    probability: float
    description: str
    signal: str  # "BULLISH", "BEARISH", "NEUTRAL"
    key_levels: dict
    confidence_factors: List[str]


class ChartExtractor:
    """Extracts price data from chart images using OpenCV."""

    def __init__(self):
        self.price_series = None
        self.image_rgb = None
        self.chart_region = None

    def extract_from_image(self, image_array: np.ndarray) -> Optional[np.ndarray]:
        """Extract price curve from chart image."""
        self.image_rgb = image_array.copy()
        img = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        # Detect chart area (crop borders, legends etc.)
        chart_region = self._detect_chart_area(img)
        self.chart_region = chart_region

        # Extract dominant price line
        price_series = self._extract_price_line(chart_region)

        if price_series is not None and len(price_series) > 20:
            self.price_series = price_series
            return price_series
        return None

    def _detect_chart_area(self, img: np.ndarray) -> np.ndarray:
        """Crop to the main chart plotting area."""
        h, w = img.shape[:2]

        # Try to detect axes by finding white/light background region
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Use edge detection to find chart boundaries
        edges = cv2.Canny(gray, 50, 150)

        # Find contours of large rectangular regions
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_region = img
        max_area = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > (h * w * 0.1):  # At least 10% of image
                x, y, rw, rh = cv2.boundingRect(cnt)
                if rw > w * 0.3 and rh > h * 0.3:
                    if area > max_area:
                        max_area = area
                        best_region = img[y:y+rh, x:x+rw]

        # Fallback: trim standard margins (10% each side)
        if max_area < h * w * 0.15:
            margin_x = int(w * 0.08)
            margin_y = int(h * 0.08)
            best_region = img[margin_y:h-margin_y, margin_x:w-margin_x]

        return best_region

    def _extract_price_line(self, chart_img: np.ndarray) -> Optional[np.ndarray]:
        """Extract the main price line from the chart region."""
        h, w = chart_img.shape[:2]

        # Convert to multiple color spaces for robust detection
        hsv = cv2.cvtColor(chart_img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(chart_img, cv2.COLOR_BGR2GRAY)

        # Try different strategies to find price line
        strategies = [
            self._detect_dominant_color_line,
            self._detect_darkest_curve,
            self._detect_by_edge_tracking,
        ]

        for strategy in strategies:
            result = strategy(chart_img, hsv, gray)
            if result is not None and self._validate_series(result):
                return self._normalize_and_smooth(result, h)

        # Fallback: synthetic extraction from brightness gradient
        return self._gradient_extraction(gray, h)

    def _detect_dominant_color_line(self, img, hsv, gray):
        """Find the most prominent colored line (usually the price line)."""
        h, w = img.shape[:2]
        price_series = np.zeros(w)
        confidence = np.zeros(w)

        # Common chart line colors: blue, green, red, black, white-on-dark
        color_masks = []

        # Blue lines
        blue_mask = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([140, 255, 255]))
        color_masks.append(blue_mask)

        # Green lines
        green_mask = cv2.inRange(hsv, np.array([40, 50, 50]), np.array([80, 255, 255]))
        color_masks.append(green_mask)

        # Orange/Red lines (common in TradingView)
        orange_mask = cv2.inRange(hsv, np.array([5, 100, 100]), np.array([20, 255, 255]))
        color_masks.append(orange_mask)

        # Dark lines on light background
        dark_mask = cv2.inRange(gray.reshape(h, w), np.array([0]), np.array([80]))
        color_masks.append(dark_mask)

        best_mask = None
        best_coverage = 0

        for mask in color_masks:
            # Check how many columns have pixels
            col_coverage = np.sum(mask > 0, axis=0)
            coverage = np.sum(col_coverage > 0)
            if coverage > best_coverage and coverage > w * 0.3:
                best_coverage = coverage
                best_mask = mask

        if best_mask is None:
            return None

        # For each column, find the topmost pixel of the dominant line
        for col in range(w):
            col_pixels = np.where(best_mask[:, col] > 0)[0]
            if len(col_pixels) > 0:
                # Use centroid of detected pixels
                price_series[col] = np.mean(col_pixels)
                confidence[col] = 1.0

        if np.sum(confidence) < w * 0.3:
            return None

        # Fill gaps using interpolation
        price_series = self._fill_gaps(price_series, confidence)
        return price_series

    def _detect_darkest_curve(self, img, hsv, gray):
        """Detect price line by finding the most prominent dark curve."""
        h, w = img.shape[:2]

        # Apply threshold to find dark pixels
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        price_series = np.zeros(w)
        confidence = np.zeros(w)

        for col in range(w):
            col_pixels = np.where(cleaned[:, col] > 0)[0]
            if len(col_pixels) > 0:
                # Find centroid, excluding top/bottom 5% (likely axes/text)
                filtered = col_pixels[(col_pixels > h * 0.05) & (col_pixels < h * 0.95)]
                if len(filtered) > 0:
                    price_series[col] = np.median(filtered)
                    confidence[col] = 1.0

        if np.sum(confidence) < w * 0.4:
            return None

        price_series = self._fill_gaps(price_series, confidence)
        return price_series

    def _detect_by_edge_tracking(self, img, hsv, gray):
        """Use Canny edge detection to find the main price curve."""
        h, w = img.shape[:2]

        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 30, 100)

        # Find horizontal extent of edges
        price_series = np.zeros(w)
        confidence = np.zeros(w)

        for col in range(w):
            edge_rows = np.where(edges[:, col] > 0)[0]
            if len(edge_rows) > 0:
                # Filter chart area (exclude extreme borders)
                valid = edge_rows[(edge_rows > h * 0.05) & (edge_rows < h * 0.95)]
                if len(valid) > 0:
                    price_series[col] = np.median(valid)
                    confidence[col] = min(1.0, len(valid) / 5)

        if np.sum(confidence) < w * 0.35:
            return None

        return self._fill_gaps(price_series, confidence)

    def _gradient_extraction(self, gray, h):
        """Fallback: extract price signal from vertical brightness gradients."""
        w = gray.shape[1]
        price_series = np.zeros(w)

        for col in range(w):
            col_data = gray[:, col].astype(float)
            # Find the "edge" in brightness (transition from chart bg to line)
            gradient = np.abs(np.diff(col_data))
            if len(gradient) > 0:
                peak_row = np.argmax(gradient)
                price_series[col] = float(peak_row)

        return price_series

    def _fill_gaps(self, series, confidence):
        """Fill gaps in series using linear interpolation."""
        filled = series.copy()
        n = len(series)

        # Find valid indices
        valid_idx = np.where(confidence > 0)[0]
        if len(valid_idx) < 2:
            return series

        # Interpolate
        filled = np.interp(np.arange(n), valid_idx, series[valid_idx])
        return filled

    def _validate_series(self, series):
        """Check if extracted series looks like a valid price curve."""
        if series is None or len(series) < 20:
            return False
        # Should have some variation
        std = np.std(series)
        mean = np.mean(series)
        if mean == 0 or std / mean < 0.005:
            return False
        # Should not be all zeros
        if np.sum(series != 0) < len(series) * 0.3:
            return False
        return True

    def _normalize_and_smooth(self, series, image_height):
        """Convert pixel Y coordinates to normalized price (invert Y axis)."""
        # Invert: higher pixel Y = lower price
        inverted = image_height - series

        # Normalize to 0-100 range
        min_val = np.min(inverted)
        max_val = np.max(inverted)
        if max_val == min_val:
            return None

        normalized = (inverted - min_val) / (max_val - min_val) * 100

        # Smooth slightly to reduce noise
        if len(normalized) > 15:
            window = min(15, len(normalized) // 5 * 2 + 1)
            if window >= 3:
                normalized = savgol_filter(normalized, window, 2)

        return normalized


class PatternDetector:
    """Detects chart patterns in price series."""

    def __init__(self):
        self.extractor = ChartExtractor()

    def detect_all_patterns(self, price_series: np.ndarray) -> List[PatternResult]:
        """Run all pattern detectors and return sorted results."""
        patterns = []

        detectors = [
            self._detect_head_and_shoulders,
            self._detect_double_top,
            self._detect_double_bottom,
            self._detect_triangle_patterns,
            self._detect_cup_and_handle,
            self._detect_flags_pennants,
            self._detect_wedge,
        ]

        for detector in detectors:
            try:
                result = detector(price_series)
                if result is not None:
                    if isinstance(result, list):
                        patterns.extend(result)
                    else:
                        patterns.append(result)
            except Exception:
                pass

        # Sort by probability descending
        patterns.sort(key=lambda x: x.probability, reverse=True)
        return patterns

    def detect_from_image(self, image_array: np.ndarray) -> Tuple[Optional[np.ndarray], List[PatternResult]]:
        """Full pipeline: extract price from image, then detect patterns."""
        price_series = self.extractor.extract_from_image(image_array)

        if price_series is None:
            return None, []

        patterns = self.detect_all_patterns(price_series)
        return price_series, patterns

    # ── Pivot Point Utilities ────────────────────────────────────────────────

    def _find_peaks_valleys(self, series: np.ndarray, order: int = None):
        """Find significant peaks and valleys."""
        n = len(series)
        if order is None:
            order = max(5, n // 20)

        peaks_idx = argrelextrema(series, np.greater, order=order)[0]
        valleys_idx = argrelextrema(series, np.less, order=order)[0]

        # Filter to significant ones (top/bottom 30%)
        if len(peaks_idx) > 0:
            threshold = np.percentile(series, 60)
            peaks_idx = peaks_idx[series[peaks_idx] >= threshold]

        if len(valleys_idx) > 0:
            threshold = np.percentile(series, 40)
            valleys_idx = valleys_idx[series[valleys_idx] <= threshold]

        return peaks_idx, valleys_idx

    def _slope(self, x1, y1, x2, y2):
        if x2 == x1:
            return 0
        return (y2 - y1) / (x2 - x1)

    # ── Pattern 1: Head and Shoulders ───────────────────────────────────────

    def _detect_head_and_shoulders(self, series: np.ndarray) -> Optional[PatternResult]:
        n = len(series)
        peaks, valleys = self._find_peaks_valleys(series, order=max(5, n // 25))

        if len(peaks) < 3 or len(valleys) < 2:
            return None

        best_score = 0
        best_config = None

        # Try all combinations of 3 consecutive peaks
        for i in range(len(peaks) - 2):
            p1, p2, p3 = peaks[i], peaks[i+1], peaks[i+2]
            h1, h2, h3 = series[p1], series[p2], series[p3]

            # Head must be highest
            if not (h2 > h1 and h2 > h3):
                continue

            # Shoulders should be roughly equal (within 20%)
            shoulder_diff = abs(h1 - h3) / max(h1, h3)
            if shoulder_diff > 0.25:
                continue

            # Head should be significantly taller
            head_prominence = (h2 - max(h1, h3)) / h2
            if head_prominence < 0.03:
                continue

            # Find neckline (valleys between peaks)
            v_between_p1_p2 = valleys[(valleys > p1) & (valleys < p2)]
            v_between_p2_p3 = valleys[(valleys > p2) & (valleys < p3)]

            if len(v_between_p1_p2) == 0 or len(v_between_p2_p3) == 0:
                continue

            v1 = v_between_p1_p2[np.argmin(series[v_between_p1_p2])]
            v2 = v_between_p2_p3[np.argmin(series[v_between_p2_p3])]

            # Neckline should be roughly horizontal
            neckline_slope = abs(self._slope(v1, series[v1], v2, series[v2]))
            neckline_level = (series[v1] + series[v2]) / 2
            neckline_flatness = 1 - min(1, neckline_slope * 10)

            score = (
                0.35 * (1 - shoulder_diff) +
                0.30 * head_prominence * 3 +
                0.20 * neckline_flatness +
                0.15 * (1 - abs(p3 - p1) / n)  # Prefer patterns not too spread out
            )

            if score > best_score:
                best_score = score
                best_config = (p1, p2, p3, v1, v2, h1, h2, h3, neckline_level, shoulder_diff)

        if best_config is None or best_score < 0.3:
            return None

        p1, p2, p3, v1, v2, h1, h2, h3, neckline_level, shoulder_diff = best_config
        probability = min(0.95, best_score * 1.1)

        factors = []
        if shoulder_diff < 0.1:
            factors.append("Symmetrical shoulders — strong signal")
        if best_score > 0.6:
            factors.append("Clear head prominence detected")
        factors.append(f"Neckline identified at ~{neckline_level:.1f}")
        if probability > 0.65:
            factors.append("Price structure highly consistent with H&S")

        return PatternResult(
            name="Head & Shoulders",
            probability=round(probability, 2),
            description="Classic reversal pattern with three peaks — left shoulder, head (highest), right shoulder. Signals bearish reversal when neckline breaks.",
            signal="BEARISH",
            key_levels={
                "Left Shoulder": round(float(h1), 1),
                "Head": round(float(h2), 1),
                "Right Shoulder": round(float(h3), 1),
                "Neckline": round(float(neckline_level), 1),
                "Target (est.)": round(float(neckline_level - (h2 - neckline_level)), 1),
            },
            confidence_factors=factors,
        )

    # ── Pattern 2: Double Top ────────────────────────────────────────────────

    def _detect_double_top(self, series: np.ndarray) -> Optional[PatternResult]:
        n = len(series)
        peaks, valleys = self._find_peaks_valleys(series, order=max(4, n // 25))

        if len(peaks) < 2:
            return None

        best_score = 0
        best_config = None

        for i in range(len(peaks) - 1):
            p1, p2 = peaks[i], peaks[i+1]
            h1, h2 = series[p1], series[p2]

            # Peaks must be close in height (within 5%)
            height_diff = abs(h1 - h2) / max(h1, h2)
            if height_diff > 0.08:
                continue

            # Must have a valley between them
            v_between = valleys[(valleys > p1) & (valleys < p2)]
            if len(v_between) == 0:
                continue

            valley_idx = v_between[np.argmin(series[v_between])]
            valley_val = series[valley_idx]

            # Valley must be meaningfully below peaks
            pullback = (min(h1, h2) - valley_val) / min(h1, h2)
            if pullback < 0.03:
                continue

            # Gap between peaks — not too close, not too far
            gap_score = 1 - abs((p2 - p1) / n - 0.25)

            score = (
                0.40 * (1 - height_diff * 10) +
                0.35 * min(1, pullback * 5) +
                0.25 * gap_score
            )

            if score > best_score:
                best_score = score
                best_config = (p1, p2, valley_idx, h1, h2, valley_val, pullback)

        if best_config is None or best_score < 0.3:
            return None

        p1, p2, vi, h1, h2, valley_val, pullback = best_config
        probability = min(0.93, best_score * 1.15)

        factors = []
        factors.append(f"Two peaks at similar heights (~{max(h1, h2):.1f})")
        if abs(h1 - h2) / max(h1, h2) < 0.03:
            factors.append("Near-identical peak heights — strong confirmation")
        factors.append(f"Valley pullback: {pullback*100:.1f}%")
        if pullback > 0.07:
            factors.append("Deep retracement between peaks")

        return PatternResult(
            name="Double Top",
            probability=round(probability, 2),
            description="Two peaks at approximately the same level after an uptrend. Bearish reversal pattern confirmed when price breaks below the valley (neckline).",
            signal="BEARISH",
            key_levels={
                "First Peak": round(float(h1), 1),
                "Second Peak": round(float(h2), 1),
                "Valley (Neckline)": round(float(valley_val), 1),
                "Target (est.)": round(float(valley_val - (max(h1,h2) - valley_val)), 1),
            },
            confidence_factors=factors,
        )

    # ── Pattern 3: Double Bottom ─────────────────────────────────────────────

    def _detect_double_bottom(self, series: np.ndarray) -> Optional[PatternResult]:
        n = len(series)
        # Use a broader search — double bottom valleys may be shallow
        order = max(3, n // 30)
        peaks, valleys = self._find_peaks_valleys(series, order=order)

        # Also try with tighter order if no valleys found
        if len(valleys) < 2:
            alt_peaks, alt_valleys = self._find_peaks_valleys(series, order=max(3, n // 40))
            valleys = alt_valleys
            peaks = alt_peaks

        if len(valleys) < 2:
            return None

        best_score = 0
        best_config = None

        for i in range(len(valleys) - 1):
            v1, v2 = valleys[i], valleys[i+1]
            l1, l2 = series[v1], series[v2]

            range_p = np.max(series) - np.min(series)
            if range_p < 0.01:
                continue

            height_diff = abs(l1 - l2) / (range_p + 0.01)
            if height_diff > 0.20:
                continue

            # Peak between valleys — also look at any local max
            p_between = peaks[(peaks > v1) & (peaks < v2)]
            if len(p_between) == 0:
                # Fall back: find max in between regardless of peak detection
                between_slice = series[v1:v2]
                if len(between_slice) < 3:
                    continue
                local_max_offset = np.argmax(between_slice)
                peak_idx = v1 + local_max_offset
                peak_val = series[peak_idx]
            else:
                peak_idx = p_between[np.argmax(series[p_between])]
                peak_val = series[peak_idx]

            bounce = (peak_val - max(l1, l2)) / (range_p + 0.01)
            if bounce < 0.02:
                continue

            gap_score = 1 - abs((v2 - v1) / n - 0.25)

            score = (
                0.40 * (1 - height_diff * 8) +
                0.35 * min(1, bounce * 5) +
                0.25 * max(0, gap_score)
            )

            if score > best_score:
                best_score = score
                best_config = (v1, v2, peak_idx, l1, l2, peak_val, bounce)

        if best_config is None or best_score < 0.25:
            return None

        v1, v2, pi, l1, l2, peak_val, bounce = best_config
        probability = min(0.93, best_score * 1.15)

        factors = []
        factors.append(f"Two troughs at similar lows (~{min(l1, l2):.1f})")
        if abs(l1 - l2) / max(abs(l1) + 0.01, 1) < 0.03:
            factors.append("Near-identical bottom levels — strong confirmation")
        factors.append(f"Bounce between bottoms: {bounce*100:.1f}%")

        return PatternResult(
            name="Double Bottom",
            probability=round(probability, 2),
            description="Two troughs at approximately the same level after a downtrend. Bullish reversal pattern confirmed when price breaks above the peak between bottoms.",
            signal="BULLISH",
            key_levels={
                "First Bottom": round(float(l1), 1),
                "Second Bottom": round(float(l2), 1),
                "Neckline (Peak)": round(float(peak_val), 1),
                "Target (est.)": round(float(peak_val + (peak_val - min(l1, l2))), 1),
            },
            confidence_factors=factors,
        )

    # ── Pattern 4: Triangle Patterns ─────────────────────────────────────────

    def _detect_triangle_patterns(self, series: np.ndarray) -> List[PatternResult]:
        n = len(series)
        if n < 30:
            return []

        results = []

        # Split series into thirds and compute stats
        third = n // 3
        s1 = series[:third]
        s2 = series[third:2*third]
        s3 = series[2*third:]

        # Compute high/low range for each third
        ranges = [np.max(s) - np.min(s) for s in [s1, s2, s3]]
        highs = [np.max(s) for s in [s1, s2, s3]]
        lows = [np.min(s) for s in [s1, s2, s3]]

        # Ascending Triangle: flat top, rising bottom
        top_flat = (highs[1] - highs[0]) / (highs[0] + 0.01) < 0.05 and \
                   (highs[2] - highs[1]) / (highs[1] + 0.01) < 0.05
        bottom_rising = lows[1] > lows[0] and lows[2] > lows[1]
        range_contracting = ranges[1] < ranges[0] and ranges[2] < ranges[1]

        if top_flat and bottom_rising and range_contracting:
            score = 0.5
            if top_flat:
                score += 0.2
            if range_contracting:
                score += 0.15
            results.append(PatternResult(
                name="Ascending Triangle",
                probability=round(min(0.88, score * 1.2), 2),
                description="Flat resistance top with rising support. Bullish continuation pattern — breakout typically occurs upward.",
                signal="BULLISH",
                key_levels={
                    "Resistance": round(float(np.mean(highs)), 1),
                    "Support (Start)": round(float(lows[0]), 1),
                    "Support (End)": round(float(lows[2]), 1),
                },
                confidence_factors=[
                    "Horizontal resistance line detected",
                    "Rising trough sequence confirmed",
                    "Range contracting (price coiling)" if range_contracting else "",
                ],
            ))

        # Descending Triangle: falling top, flat bottom
        top_falling = highs[0] > highs[1] > highs[2]
        bottom_flat = abs(lows[1] - lows[0]) / (lows[0] + 0.01) < 0.05 and \
                      abs(lows[2] - lows[1]) / (lows[1] + 0.01) < 0.05

        if top_falling and bottom_flat and range_contracting:
            score = 0.5
            if bottom_flat:
                score += 0.2
            if range_contracting:
                score += 0.15
            results.append(PatternResult(
                name="Descending Triangle",
                probability=round(min(0.88, score * 1.2), 2),
                description="Flat support bottom with falling resistance. Bearish continuation — breakout typically occurs downward.",
                signal="BEARISH",
                key_levels={
                    "Support": round(float(np.mean(lows)), 1),
                    "Resistance (Start)": round(float(highs[0]), 1),
                    "Resistance (End)": round(float(highs[2]), 1),
                },
                confidence_factors=[
                    "Horizontal support line detected",
                    "Falling peak sequence confirmed",
                    "Range contracting" if range_contracting else "",
                ],
            ))

        # Symmetrical Triangle: both converging
        tops_falling = highs[0] > highs[1] > highs[2]
        bottoms_rising = lows[0] < lows[1] < lows[2]

        if tops_falling and bottoms_rising and range_contracting:
            score = 0.45
            if range_contracting:
                score += 0.25
            results.append(PatternResult(
                name="Symmetrical Triangle",
                probability=round(min(0.82, score * 1.15), 2),
                description="Converging trendlines from both sides. Continuation pattern — direction of breakout determines trade bias.",
                signal="NEUTRAL",
                key_levels={
                    "Upper Trendline (Start)": round(float(highs[0]), 1),
                    "Upper Trendline (End)": round(float(highs[2]), 1),
                    "Lower Trendline (Start)": round(float(lows[0]), 1),
                    "Lower Trendline (End)": round(float(lows[2]), 1),
                },
                confidence_factors=[
                    "Both trendlines converging",
                    "Classic coiling price action",
                    "Watch for volume expansion on breakout",
                ],
            ))

        return results

    # ── Pattern 5: Cup and Handle ────────────────────────────────────────────

    def _detect_cup_and_handle(self, series: np.ndarray) -> Optional[PatternResult]:
        """
        Detect Cup & Handle using two complementary strategies:

        Strategy A — U-shaped cup (rims are peaks, bottom is valley):
          Two local peaks at similar heights with a valley between them.

        Strategy B — Dome-routed cup (single high peak, price descends to rims):
          Single prominent peak; left rim and right rim are where price
          returns to a base level (~70-80% of peak) on either side.
        """
        n = len(series)
        if n < 40:
            return None

        range_p = np.max(series) - np.min(series)
        if range_p < 0.01:
            return None

        best_score = 0
        best_config = None

        order = max(4, n // 22)
        peaks_idx = argrelextrema(series, np.greater, order=order)[0]
        valleys_idx = argrelextrema(series, np.less, order=order)[0]

        # ── Strategy A: two rims (peaks), one bottom (valley) ────────────────
        if len(peaks_idx) >= 2:
            for i in range(len(peaks_idx) - 1):
                for j in range(i + 1, min(i + 4, len(peaks_idx))):
                    lr_idx, rr_idx = peaks_idx[i], peaks_idx[j]
                    if rr_idx > int(n * 0.90):
                        continue
                    gap = rr_idx - lr_idx
                    if gap < n * 0.15 or gap > n * 0.80:
                        continue

                    lr_val, rr_val = series[lr_idx], series[rr_idx]
                    rim_diff = abs(lr_val - rr_val) / (range_p + 0.01)
                    if rim_diff > 0.22:
                        continue

                    cup_region = series[lr_idx:rr_idx + 1]
                    bot_off = int(np.argmin(cup_region))
                    bot_val = cup_region[bot_off]
                    rel_pos = bot_off / max(len(cup_region) - 1, 1)
                    if not (0.15 < rel_pos < 0.85):
                        continue

                    cup_depth = (min(lr_val, rr_val) - bot_val) / (range_p + 0.01)
                    if cup_depth < 0.05:
                        continue

                    cup_norm = (cup_region - bot_val) / (max(lr_val, rr_val) - bot_val + 0.01)
                    x_u = np.linspace(0, np.pi, len(cup_norm))
                    ideal_shape = 1 - (1 - np.cos(x_u)) / 2
                    try:
                        cup_roundness = max(0, pearsonr(cup_norm, ideal_shape)[0])
                    except Exception:
                        cup_roundness = 0.3

                    handle_s = series[rr_idx:]
                    if len(handle_s) < 3:
                        continue
                    handle_min = float(np.min(handle_s))
                    handle_pullback = (rr_val - handle_min) / (range_p + 0.01)
                    if handle_pullback > 0.22:
                        continue

                    score = (
                        0.30 * (1 - rim_diff * 4) +
                        0.28 * min(1.0, cup_depth * 4) +
                        0.22 * cup_roundness +
                        0.20 * max(0.0, 1 - handle_pullback * 5)
                    )
                    if score > best_score:
                        best_score = score
                        best_config = dict(
                            left_rim=lr_val, right_rim=rr_val,
                            cup_bottom=bot_val, handle_min=handle_min,
                            cup_depth=cup_depth, handle_pullback=handle_pullback,
                            handle_uptrend=handle_s[-1] > handle_min,
                        )

        # ── Strategy B: dome cup — single high peak, rims are base-level points ─
        for pk_idx in peaks_idx:
            if pk_idx < n * 0.10 or pk_idx > n * 0.85:
                continue
            pk_val = series[pk_idx]
            base = series[0]
            if (pk_val - base) / (range_p + 0.01) < 0.20:
                continue  # peak not prominent enough

            # Left rim: last point to the left of peak that is at or near base level
            base_ceiling = base * 1.15 + range_p * 0.05
            left_rim_idx = 0
            for i in range(pk_idx - 1, -1, -1):
                if series[i] <= base_ceiling:
                    left_rim_idx = i
                    break
            lr_val = series[left_rim_idx]

            # Right rim: first point after peak that returns to base level
            right_rim_idx = None
            for i in range(pk_idx + 1, int(n * 0.92)):
                if series[i] <= base_ceiling:
                    right_rim_idx = i
                    break
            if right_rim_idx is None:
                continue
            rr_val = series[right_rim_idx]

            rim_diff = abs(lr_val - rr_val) / (range_p + 0.01)
            if rim_diff > 0.30:
                continue

            gap = right_rim_idx - left_rim_idx
            if gap < n * 0.15 or gap > n * 0.85:
                continue

            rim_level = (lr_val + rr_val) / 2
            cup_depth = (pk_val - rim_level) / (range_p + 0.01)
            if cup_depth < 0.05:
                continue

            # Roundness: dome cup region → compare to sine arch
            cup_region = series[left_rim_idx:right_rim_idx + 1]
            cup_norm = (cup_region - rim_level) / (pk_val - rim_level + 0.01)
            x_arch = np.linspace(0, np.pi, len(cup_norm))
            ideal_arch = np.sin(x_arch)
            try:
                cup_roundness = max(0, pearsonr(cup_norm, ideal_arch)[0])
            except Exception:
                cup_roundness = 0.3

            handle_s = series[right_rim_idx:]
            if len(handle_s) < 3:
                continue
            handle_min = float(np.min(handle_s))
            # Dome cups can have deeper handles since rim is already near base
            handle_pullback = (rr_val - handle_min) / (range_p + 0.01)
            if handle_pullback > 0.60:
                continue

            score = (
                0.28 * (1 - rim_diff * 3) +
                0.30 * min(1.0, cup_depth * 3) +
                0.22 * cup_roundness +
                0.20 * max(0.0, 1 - handle_pullback * 2.0)
            )
            if score > best_score:
                best_score = score
                best_config = dict(
                    left_rim=lr_val, right_rim=rr_val,
                    cup_bottom=rim_level,
                    handle_min=handle_min, cup_depth=cup_depth,
                    handle_pullback=handle_pullback,
                    handle_uptrend=handle_s[-1] > handle_min,
                )

        if best_config is None or best_score < 0.20:
            return None

        cup_start       = best_config["left_rim"]
        cup_end_val     = best_config["right_rim"]
        cup_min         = best_config["cup_bottom"]
        handle_min      = best_config["handle_min"]
        cup_depth       = best_config["cup_depth"]
        handle_pullback = best_config["handle_pullback"]
        handle_uptrend  = best_config["handle_uptrend"]

        probability = min(0.91, best_score * 1.15)

        rim_diff_pct = abs(cup_start - cup_end_val) / (range_p + 0.01)
        factors = []
        factors.append(f"Cup depth: {cup_depth*100:.1f}%")
        if rim_diff_pct < 0.08:
            factors.append("Symmetric cup lips detected")
        if handle_uptrend:
            factors.append("Handle forming with upward bias")
        factors.append(f"Handle pullback: {handle_pullback*100:.1f}%")

        return PatternResult(
            name="Cup & Handle",
            probability=round(probability, 2),
            description="Rounded cup consolidation followed by a brief handle pullback. Strong bullish continuation pattern — breakout above cup rim is the entry signal.",
            signal="BULLISH",
            key_levels={
                "Cup Rim (Left)": round(float(cup_start), 1),
                "Cup Bottom": round(float(cup_min), 1),
                "Cup Rim (Right)": round(float(cup_end_val), 1),
                "Handle Low": round(float(handle_min), 1),
                "Breakout Target": round(float(cup_start + (cup_start - cup_min)), 1),
            },
            confidence_factors=factors,
        )

    # ── Pattern 6: Flags & Pennants ──────────────────────────────────────────

    def _detect_flags_pennants(self, series: np.ndarray) -> Optional[PatternResult]:
        n = len(series)
        if n < 30:
            return None

        # Flagpole: sharp move in first 25% of chart
        pole_end = n // 4
        pole_series = series[:pole_end]
        flag_series = series[pole_end:]

        pole_move = (pole_series[-1] - pole_series[0]) / (abs(pole_series[0]) + 0.01)
        pole_strength = abs(pole_move)

        if pole_strength < 0.08:
            return None

        # Flag: consolidation with slight counter-trend drift
        flag_range = np.max(flag_series) - np.min(flag_series)
        pole_range = np.max(pole_series) - np.min(pole_series)

        if pole_range < 0.01:
            return None

        # Flag should be narrower than pole
        range_ratio = flag_range / pole_range
        if range_ratio > 0.5:
            return None

        # Determine direction
        is_bullish = pole_move > 0

        score = 0.4 + (1 - range_ratio) * 0.3 + min(0.3, pole_strength)

        if score < 0.4:
            return None

        probability = min(0.87, score)

        return PatternResult(
            name="Bull Flag" if is_bullish else "Bear Flag",
            probability=round(probability, 2),
            description=f"Strong {'upward' if is_bullish else 'downward'} flagpole followed by tight consolidation. Continuation pattern — breakout expected in the direction of the original move.",
            signal="BULLISH" if is_bullish else "BEARISH",
            key_levels={
                "Pole Start": round(float(pole_series[0]), 1),
                "Pole End / Flag Start": round(float(pole_series[-1]), 1),
                "Flag Range": round(float(flag_range), 1),
            },
            confidence_factors=[
                f"Strong flagpole: {pole_move*100:.1f}% move",
                f"Flag consolidation: {range_ratio*100:.1f}% of pole range",
                "Tight consolidation confirms pattern",
            ],
        )

    # ── Pattern 7: Rising/Falling Wedge ─────────────────────────────────────

    def _detect_wedge(self, series: np.ndarray) -> Optional[PatternResult]:
        n = len(series)
        if n < 30:
            return None

        # Compute rolling highs and lows across 5 windows
        window = n // 5
        segments = [series[i*window:(i+1)*window] for i in range(5)]
        highs = np.array([np.max(s) for s in segments])
        lows = np.array([np.min(s) for s in segments])

        x = np.arange(5)
        high_slope = np.polyfit(x, highs, 1)[0]
        low_slope = np.polyfit(x, lows, 1)[0]

        # Rising wedge: both trendlines slope up but converge
        if high_slope > 0 and low_slope > 0 and low_slope > high_slope:
            converging = (highs[-1] - lows[-1]) < (highs[0] - lows[0])
            if converging:
                score = 0.5 + min(0.3, abs(low_slope - high_slope) * 10)
                return PatternResult(
                    name="Rising Wedge",
                    probability=round(min(0.85, score), 2),
                    description="Both support and resistance trendlines slope upward but converge. Despite appearing bullish, this is a bearish reversal signal.",
                    signal="BEARISH",
                    key_levels={
                        "Resistance Start": round(float(highs[0]), 1),
                        "Resistance End": round(float(highs[-1]), 1),
                        "Support Start": round(float(lows[0]), 1),
                        "Support End": round(float(lows[-1]), 1),
                    },
                    confidence_factors=[
                        "Both trendlines rising",
                        "Range contraction detected",
                        "Bearish divergence pattern",
                    ],
                )

        # Falling wedge: both trendlines slope down but converge
        if high_slope < 0 and low_slope < 0 and high_slope < low_slope:
            converging = (highs[-1] - lows[-1]) < (highs[0] - lows[0])
            if converging:
                score = 0.5 + min(0.3, abs(high_slope - low_slope) * 10)
                return PatternResult(
                    name="Falling Wedge",
                    probability=round(min(0.85, score), 2),
                    description="Both trendlines slope downward but converge. Despite appearing bearish, this is a bullish reversal signal.",
                    signal="BULLISH",
                    key_levels={
                        "Resistance Start": round(float(highs[0]), 1),
                        "Resistance End": round(float(highs[-1]), 1),
                        "Support Start": round(float(lows[0]), 1),
                        "Support End": round(float(lows[-1]), 1),
                    },
                    confidence_factors=[
                        "Both trendlines falling",
                        "Range contraction detected",
                        "Bullish reversal signal",
                    ],
                )

        return None
