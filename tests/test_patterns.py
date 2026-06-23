"""
Tests for the Pattern Detection Engine.
Run: pytest tests/test_patterns.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import numpy as np
import pytest

from src.pattern_detector import PatternDetector, PatternResult
from src.sample_generator import (
    generate_head_and_shoulders,
    generate_double_top,
    generate_double_bottom,
    generate_ascending_triangle,
    generate_cup_and_handle,
    generate_bull_flag,
    generate_symmetrical_triangle,
    SAMPLE_PATTERNS,
)


@pytest.fixture
def detector():
    return PatternDetector()


# ── Smoke Tests ──────────────────────────────────────────────────────────────

class TestPatternDetectorSmoke:
    """Smoke tests — detector should return results without crashing."""

    def test_detector_initializes(self, detector):
        assert detector is not None

    def test_detect_all_returns_list(self, detector):
        series = generate_head_and_shoulders()
        results = detector.detect_all_patterns(series)
        assert isinstance(results, list)

    def test_results_are_pattern_results(self, detector):
        series = generate_double_top()
        results = detector.detect_all_patterns(series)
        for r in results:
            assert isinstance(r, PatternResult)

    def test_probabilities_in_range(self, detector):
        series = generate_cup_and_handle()
        results = detector.detect_all_patterns(series)
        for r in results:
            assert 0.0 <= r.probability <= 1.0, f"{r.name} probability out of range: {r.probability}"

    def test_signals_valid(self, detector):
        series = generate_bull_flag()
        results = detector.detect_all_patterns(series)
        valid_signals = {"BULLISH", "BEARISH", "NEUTRAL"}
        for r in results:
            assert r.signal in valid_signals

    def test_results_sorted_by_probability(self, detector):
        series = generate_double_bottom()
        results = detector.detect_all_patterns(series)
        if len(results) >= 2:
            for i in range(len(results) - 1):
                assert results[i].probability >= results[i+1].probability

    def test_short_series_no_crash(self, detector):
        """Very short series should not crash the detector."""
        short_series = np.random.randn(10)
        results = detector.detect_all_patterns(short_series)
        assert isinstance(results, list)

    def test_flat_series_no_crash(self, detector):
        """Completely flat series (no variation) should not crash."""
        flat_series = np.ones(200) * 50.0
        results = detector.detect_all_patterns(flat_series)
        assert isinstance(results, list)

    def test_noisy_series_no_crash(self, detector):
        """Pure noise should not crash."""
        noisy = np.random.randn(200) * 100
        results = detector.detect_all_patterns(noisy)
        assert isinstance(results, list)


# ── Pattern-Specific Tests ───────────────────────────────────────────────────

class TestHeadAndShoulders:
    def test_detects_pattern(self, detector):
        series = generate_head_and_shoulders()
        results = detector.detect_all_patterns(series)
        names = [r.name for r in results]
        # Should detect H&S in synthetic data
        assert any("Head" in n for n in names), f"H&S not detected. Got: {names}"

    def test_is_bearish(self, detector):
        series = generate_head_and_shoulders()
        results = detector.detect_all_patterns(series)
        hs = next((r for r in results if "Head" in r.name), None)
        if hs:
            assert hs.signal == "BEARISH"

    def test_has_key_levels(self, detector):
        series = generate_head_and_shoulders()
        results = detector.detect_all_patterns(series)
        hs = next((r for r in results if "Head" in r.name), None)
        if hs:
            assert "Head" in hs.key_levels
            assert "Neckline" in hs.key_levels


class TestDoubleTop:
    def test_detects_pattern(self, detector):
        series = generate_double_top()
        results = detector.detect_all_patterns(series)
        names = [r.name for r in results]
        assert any("Double Top" in n for n in names), f"Double Top not detected. Got: {names}"

    def test_is_bearish(self, detector):
        series = generate_double_top()
        results = detector.detect_all_patterns(series)
        dt = next((r for r in results if r.name == "Double Top"), None)
        if dt:
            assert dt.signal == "BEARISH"


class TestDoubleBottom:
    def test_detects_pattern(self, detector):
        series = generate_double_bottom()
        results = detector.detect_all_patterns(series)
        names = [r.name for r in results]
        assert any("Double Bottom" in n for n in names), f"Double Bottom not detected. Got: {names}"

    def test_is_bullish(self, detector):
        series = generate_double_bottom()
        results = detector.detect_all_patterns(series)
        db = next((r for r in results if r.name == "Double Bottom"), None)
        if db:
            assert db.signal == "BULLISH"


class TestTriangles:
    def test_ascending_triangle_detected(self, detector):
        series = generate_ascending_triangle()
        results = detector.detect_all_patterns(series)
        names = [r.name for r in results]
        assert any("Triangle" in n for n in names), f"No triangle detected. Got: {names}"

    def test_symmetrical_triangle_detected(self, detector):
        series = generate_symmetrical_triangle()
        results = detector.detect_all_patterns(series)
        names = [r.name for r in results]
        assert any("Triangle" in n for n in names), f"No triangle detected. Got: {names}"

    def test_ascending_is_bullish(self, detector):
        series = generate_ascending_triangle()
        results = detector.detect_all_patterns(series)
        asc = next((r for r in results if r.name == "Ascending Triangle"), None)
        if asc:
            assert asc.signal == "BULLISH"


class TestCupAndHandle:
    def test_detects_pattern(self, detector):
        series = generate_cup_and_handle()
        results = detector.detect_all_patterns(series)
        names = [r.name for r in results]
        assert any("Cup" in n for n in names), f"Cup & Handle not detected. Got: {names}"

    def test_is_bullish(self, detector):
        series = generate_cup_and_handle()
        results = detector.detect_all_patterns(series)
        cup = next((r for r in results if "Cup" in r.name), None)
        if cup:
            assert cup.signal == "BULLISH"


class TestBullFlag:
    def test_detects_flag(self, detector):
        series = generate_bull_flag()
        results = detector.detect_all_patterns(series)
        names = [r.name for r in results]
        assert any("Flag" in n for n in names), f"Flag not detected. Got: {names}"


# ── Sample Generator Tests ───────────────────────────────────────────────────

class TestSampleGenerator:
    def test_all_generators_return_arrays(self):
        for name, fn in SAMPLE_PATTERNS.items():
            series = fn()
            assert isinstance(series, np.ndarray), f"{name} generator didn't return ndarray"
            assert len(series) >= 100, f"{name} series too short: {len(series)}"

    def test_series_have_variation(self):
        for name, fn in SAMPLE_PATTERNS.items():
            series = fn()
            std = np.std(series)
            assert std > 0.1, f"{name} series has no variation (std={std})"

    def test_no_nan_in_series(self):
        for name, fn in SAMPLE_PATTERNS.items():
            series = fn()
            assert not np.any(np.isnan(series)), f"{name} series contains NaN"


# ── PatternResult Dataclass Tests ────────────────────────────────────────────

class TestPatternResult:
    def test_pattern_result_fields(self):
        r = PatternResult(
            name="Test Pattern",
            probability=0.75,
            description="Test description",
            signal="BULLISH",
            key_levels={"Support": 100.0, "Resistance": 120.0},
            confidence_factors=["Factor 1", "Factor 2"],
        )
        assert r.name == "Test Pattern"
        assert r.probability == 0.75
        assert r.signal == "BULLISH"
        assert len(r.key_levels) == 2
        assert len(r.confidence_factors) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
