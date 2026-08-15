"""Tests for unit conversions and angle wrapping."""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from assetto_ai_director.domain.units import (
    Degrees,
    KilometresPerHour,
    MetresPerSecond,
    Radians,
    degrees_to_radians,
    kmh_to_mps,
    mps_to_kmh,
    radians_to_degrees,
    wrap_pi,
)


class TestSpeedConversion:
    def test_known_value(self) -> None:
        assert kmh_to_mps(KilometresPerHour(180.0)) == pytest.approx(50.0)

    def test_mf_ghost_drone_cap(self) -> None:
        # The one verified style constant in the whole project: a drone that
        # cannot exceed 180 km/h is 50 m/s exactly.
        assert kmh_to_mps(KilometresPerHour(180.0)) == pytest.approx(50.0, abs=1e-9)

    @given(st.floats(min_value=-1e4, max_value=1e4, allow_nan=False))
    def test_round_trip(self, value: float) -> None:
        original = MetresPerSecond(value)
        assert kmh_to_mps(mps_to_kmh(original)) == pytest.approx(original)


class TestAngleConversion:
    def test_known_values(self) -> None:
        assert radians_to_degrees(Radians(math.pi)) == pytest.approx(180.0)
        assert degrees_to_radians(Degrees(90.0)) == pytest.approx(math.pi / 2)

    @given(st.floats(min_value=-1e4, max_value=1e4, allow_nan=False))
    def test_round_trip(self, value: float) -> None:
        assert degrees_to_radians(radians_to_degrees(Radians(value))) == pytest.approx(value)


class TestWrapPi:
    """Angle wrapping is load-bearing.

    Sideslip is a difference of two independently wrapping angles. Without
    wrapping, every branch-cut crossing produces a spurious 2*pi spike that looks
    exactly like a violent drift initiation to an event detector.
    """

    def test_wraps_above_pi(self) -> None:
        assert wrap_pi(Radians(3 * math.pi / 2)) == pytest.approx(-math.pi / 2)

    def test_positive_pi_is_preserved(self) -> None:
        # The convention is (-pi, pi], so +pi stays positive rather than
        # flipping sign and inverting the apparent direction of a slide.
        assert wrap_pi(Radians(math.pi)) == pytest.approx(math.pi)

    def test_negative_pi_maps_to_positive(self) -> None:
        assert wrap_pi(Radians(-math.pi)) == pytest.approx(math.pi)

    def test_small_angles_unchanged(self) -> None:
        assert wrap_pi(Radians(0.3)) == pytest.approx(0.3)

    @given(st.floats(min_value=-100.0, max_value=100.0, allow_nan=False))
    def test_always_in_range(self, angle: float) -> None:
        result = wrap_pi(Radians(angle))
        assert -math.pi < result <= math.pi + 1e-12

    @given(st.floats(min_value=-50.0, max_value=50.0, allow_nan=False))
    def test_congruent_modulo_tau(self, angle: float) -> None:
        result = wrap_pi(Radians(angle))
        difference = (angle - result) / math.tau
        assert difference == pytest.approx(round(difference), abs=1e-9)

    def test_no_spike_across_branch_cut(self) -> None:
        # A car whose heading crosses the branch cut must not appear to swing
        # through a full turn between consecutive samples.
        before = Radians(math.pi - 0.01)
        after = Radians(-math.pi + 0.01)
        delta = wrap_pi(Radians(after - before))
        assert abs(delta) == pytest.approx(0.02, abs=1e-9)
