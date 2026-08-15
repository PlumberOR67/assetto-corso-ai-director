"""Tests for cutting-rhythm measurement and generation.

The property under test is the one that makes an edit feel human rather than
mechanical: shot lengths must be *correlated*, not independent. A generator that
gets the mean and variance right but produces white noise will pass every naive
check and still feel wrong on screen.
"""

from __future__ import annotations

import numpy as np
import pytest

from assetto_ai_director.application.director.rhythm import (
    pink_noise_durations,
    spectral_slope,
)
from assetto_ai_director.domain.units import Seconds


class TestSpectralSlope:
    def test_too_few_shots_returns_none(self) -> None:
        # A short sequence yields a confident-looking slope that is pure
        # sampling noise. Refusing to report it is the point.
        assert spectral_slope([1.0, 2.0, 1.5]) is None

    def test_constant_sequence_returns_none(self) -> None:
        assert spectral_slope([2.0] * 32) is None

    def test_white_noise_slope_is_near_zero(self) -> None:
        generator = np.random.default_rng(11)
        durations = np.exp(generator.standard_normal(4096) * 0.4 + 1.0)
        slope = spectral_slope(tuple(float(v) for v in durations))
        assert slope is not None
        assert abs(slope) < 0.15

    def test_pink_noise_slope_is_clearly_negative(self) -> None:
        durations = pink_noise_durations(4096, Seconds(2.4), 1.0, seed=3)
        slope = spectral_slope(tuple(float(d) for d in durations))
        assert slope is not None
        assert slope < -0.4

    def test_pink_is_distinguishable_from_white(self) -> None:
        # The whole justification for the Voss-McCartney generator: if these two
        # produced the same spectrum, an independent sampler would do.
        generator = np.random.default_rng(5)
        white = tuple(float(v) for v in np.exp(generator.standard_normal(2048) * 0.4 + 1.0))
        pink = tuple(float(d) for d in pink_noise_durations(2048, Seconds(2.4), 1.0, seed=5))
        white_slope = spectral_slope(white)
        pink_slope = spectral_slope(pink)
        assert white_slope is not None
        assert pink_slope is not None
        assert pink_slope < white_slope - 0.3

    def test_ignores_non_positive_durations(self) -> None:
        assert spectral_slope([0.0] * 40) is None


class TestPinkNoiseDurations:
    def test_count_is_honoured(self) -> None:
        assert len(pink_noise_durations(50, Seconds(2.0), 0.8)) == 50

    def test_zero_count_is_empty(self) -> None:
        assert pink_noise_durations(0, Seconds(2.0), 0.8) == ()

    def test_negative_count_is_refused(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            pink_noise_durations(-1, Seconds(2.0), 0.8)

    def test_non_positive_mean_is_refused(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            pink_noise_durations(10, Seconds(0.0), 0.8)

    def test_mean_is_approximately_the_target(self) -> None:
        durations = pink_noise_durations(8192, Seconds(2.4), 0.9, seed=7)
        assert float(np.mean(durations)) == pytest.approx(2.4, rel=0.15)

    def test_spread_is_approximately_the_target(self) -> None:
        durations = pink_noise_durations(8192, Seconds(2.4), 1.0, seed=7)
        values = np.asarray(durations)
        assert values.std() / values.mean() == pytest.approx(1.0, rel=0.3)

    def test_bounds_are_respected(self) -> None:
        durations = pink_noise_durations(
            2000, Seconds(2.0), 2.0, seed=1, minimum=Seconds(0.5), maximum=Seconds(6.0)
        )
        assert min(durations) >= 0.5
        assert max(durations) <= 6.0

    def test_is_reproducible(self) -> None:
        # Directing the same session twice must produce the same edit.
        first = pink_noise_durations(200, Seconds(2.4), 0.9, seed=42)
        second = pink_noise_durations(200, Seconds(2.4), 0.9, seed=42)
        assert first == second

    def test_different_seeds_differ(self) -> None:
        first = pink_noise_durations(200, Seconds(2.4), 0.9, seed=1)
        second = pink_noise_durations(200, Seconds(2.4), 0.9, seed=2)
        assert first != second

    def test_distribution_is_right_skewed(self) -> None:
        # Real shot-length distributions have a mass of short shots and a tail of
        # long ones, so the mean must sit above the median.
        durations = pink_noise_durations(4096, Seconds(2.4), 1.0, seed=9)
        values = np.asarray(durations)
        assert values.mean() > np.median(values)

    def test_all_durations_are_positive(self) -> None:
        assert all(d > 0.0 for d in pink_noise_durations(500, Seconds(1.0), 1.5, seed=4))
