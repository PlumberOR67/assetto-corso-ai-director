"""Tests for the measured-style domain model."""

from __future__ import annotations

import pytest

from assetto_ai_director.domain.measurement import (
    MIN_SHOTS_FOR_SPECTRUM,
    CameraMotion,
    ShotMeasurement,
    StyleMeasurement,
)
from assetto_ai_director.domain.units import Seconds


def _shot(index: int, start: float, end: float, motion: CameraMotion) -> ShotMeasurement:
    return ShotMeasurement(
        index=index,
        start=Seconds(start),
        end=Seconds(end),
        motion=motion,
        motion_magnitude=0.5,
        motion_coherence=0.8,
    )


def _style(bounds: list[tuple[float, float]], motions: list[CameraMotion]) -> StyleMeasurement:
    shots = tuple(
        _shot(i, a, b, m) for i, ((a, b), m) in enumerate(zip(bounds, motions, strict=True))
    )
    return StyleMeasurement(
        style_name="test",
        source_names=("clip.mp4",),
        shots=shots,
        total_duration=Seconds(bounds[-1][1] - bounds[0][0]) if bounds else Seconds(0.0),
        clip_count=1,
    )


class TestCameraMotion:
    def test_only_static_is_still(self) -> None:
        assert not CameraMotion.STATIC.is_moving
        assert CameraMotion.AXIAL.is_moving
        assert CameraMotion.TRACK.is_moving

    def test_axial_exists_and_zoom_does_not(self) -> None:
        # Zoom and axial dolly were merged: optical flow cannot separate them,
        # and the split reported 60% zoom shots in touge footage, which is false.
        assert CameraMotion.AXIAL
        assert not hasattr(CameraMotion, "ZOOM")


class TestStyleMeasurement:
    def _sample(self) -> StyleMeasurement:
        return _style(
            [(0.0, 1.0), (1.0, 3.0), (3.0, 4.0), (4.0, 10.0)],
            [
                CameraMotion.STATIC,
                CameraMotion.AXIAL,
                CameraMotion.AXIAL,
                CameraMotion.STATIC,
            ],
        )

    def test_average_shot_length(self) -> None:
        assert float(self._sample().average_shot_length) == pytest.approx(2.5)

    def test_median_differs_from_mean_on_skewed_data(self) -> None:
        # Shot-length distributions are right-skewed: a few long establishing
        # shots drag the mean above what most cuts feel like.
        sample = self._sample()
        assert float(sample.median_shot_length) < float(sample.average_shot_length)

    def test_median_of_even_count_averages_the_middle(self) -> None:
        assert float(self._sample().median_shot_length) == pytest.approx(1.5)

    def test_spread_is_scale_free(self) -> None:
        slow = _style([(0.0, 2.0), (2.0, 6.0)], [CameraMotion.STATIC] * 2)
        fast = _style([(0.0, 1.0), (1.0, 3.0)], [CameraMotion.STATIC] * 2)
        assert slow.duration_spread == pytest.approx(fast.duration_spread)

    def test_spread_needs_two_shots(self) -> None:
        assert _style([(0.0, 1.0)], [CameraMotion.STATIC]).duration_spread == 0.0

    def test_percentiles(self) -> None:
        sample = self._sample()
        assert float(sample.percentile(0.0)) == pytest.approx(1.0)
        assert float(sample.percentile(1.0)) == pytest.approx(6.0)

    def test_cuts_per_minute(self) -> None:
        assert self._sample().cuts_per_minute == pytest.approx(4 / (10.0 / 60.0))

    def test_durations_preserve_edit_order(self) -> None:
        # Order carries the 1/f signal; sorting would destroy what is measured.
        assert self._sample().durations == pytest.approx((1.0, 2.0, 1.0, 6.0))

    def test_spectrum_gate_refuses_small_samples(self) -> None:
        assert not self._sample().has_enough_shots_for_spectrum

    def test_spectrum_gate_allows_large_samples(self) -> None:
        bounds = [(float(i), float(i + 1)) for i in range(MIN_SHOTS_FOR_SPECTRUM)]
        big = _style(bounds, [CameraMotion.STATIC] * MIN_SHOTS_FOR_SPECTRUM)
        assert big.has_enough_shots_for_spectrum

    def test_empty_measurement_is_safe(self) -> None:
        empty = StyleMeasurement("x", (), (), Seconds(0.0), 0)
        assert float(empty.average_shot_length) == 0.0
        assert float(empty.median_shot_length) == 0.0
        assert empty.cuts_per_minute == 0.0
        assert empty.motion.total == 0


class TestMotionSummary:
    def test_static_fraction(self) -> None:
        sample = _style(
            [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0)],
            [
                CameraMotion.STATIC,
                CameraMotion.STATIC,
                CameraMotion.AXIAL,
                CameraMotion.TRACK,
            ],
        )
        assert sample.motion.static_fraction == pytest.approx(0.5)

    def test_fraction_of_absent_class_is_zero(self) -> None:
        sample = _style([(0.0, 1.0)], [CameraMotion.STATIC])
        assert sample.motion.fraction(CameraMotion.PAN) == 0.0

    def test_counts_sum_to_total(self) -> None:
        sample = _style(
            [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)],
            [CameraMotion.STATIC, CameraMotion.AXIAL, CameraMotion.AXIAL],
        )
        assert sample.motion.total == 3
        assert sample.motion.counts[CameraMotion.AXIAL] == 2
