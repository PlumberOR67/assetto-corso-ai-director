"""Tests for moment scoring.

The scores are the director's taste made numeric, so the tests assert *ordering*
rather than absolute values: a committed slide must outscore a twitch, a fast
moment must outscore a slow one in the same session, smooth driving must outscore
hesitant driving. Absolute thresholds would lock in the current tuning and break
on every future adjustment without saying anything about correctness.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from assetto_ai_director.application.events.detect import CarSignals
from assetto_ai_director.application.scoring.features import SessionFeatures, compute_features
from assetto_ai_director.application.scoring.moments import (
    DRIFT_REFERENCE_DEGREES,
    cinematic_score,
    drift_score,
    flow_score,
    risk_score,
    score_car,
    score_session,
    speed_score,
)
from assetto_ai_director.domain.scoring import ScoreKind, ScoreWeights

RATE = 33.3


def _signals(
    count: int,
    *,
    speed: float | np.ndarray = 30.0,
    sideslip_deg: float | np.ndarray = 0.0,
    car_id: int = 0,
    elevation: np.ndarray | None = None,
    offset: float = 0.0,
) -> CarSignals:
    """Build a car travelling straight, with the given speed and slip."""
    speeds = np.full(count, speed) if np.isscalar(speed) else np.asarray(speed)
    slips = (
        np.full(count, math.radians(float(np.asarray(sideslip_deg).item())))
        if np.isscalar(sideslip_deg)
        else np.radians(np.asarray(sideslip_deg))
    )
    z = np.cumsum(speeds) / RATE
    y = np.zeros(count) if elevation is None else elevation
    position = np.stack([np.full(count, offset), y, z], axis=1)
    return CarSignals(car_id, position, np.zeros(count), speeds, slips)


def _features(*cars: CarSignals) -> SessionFeatures:
    count = cars[0].speed.shape[0]
    return compute_features({car.car_id: car for car in cars}, np.arange(count) / RATE, RATE)


class TestDriftScore:
    def test_straight_running_scores_zero(self) -> None:
        car = _signals(400)
        assert float(drift_score(_features(car).cars[0], RATE).max()) == 0.0

    def test_committed_slide_outscores_a_marginal_one(self) -> None:
        big = _signals(400, sideslip_deg=DRIFT_REFERENCE_DEGREES)
        small = _signals(400, sideslip_deg=12.0)
        assert float(drift_score(_features(big).cars[0], RATE).max()) > float(
            drift_score(_features(small).cars[0], RATE).max()
        )

    def test_sustained_slide_outscores_a_twitch(self) -> None:
        # Same peak angle; only the duration differs.
        sustained = np.zeros(400)
        sustained[100:300] = 30.0
        twitch = np.zeros(400)
        twitch[100:104] = 30.0
        a = drift_score(_features(_signals(400, sideslip_deg=sustained)).cars[0], RATE)
        b = drift_score(_features(_signals(400, sideslip_deg=twitch)).cars[0], RATE)
        assert float(a.max()) > float(b.max())

    def test_slow_rotation_scores_zero(self) -> None:
        # A car pirouetting at walking pace is not a drift.
        car = _signals(400, speed=2.0, sideslip_deg=40.0)
        assert float(drift_score(_features(car).cars[0], RATE).max()) == 0.0

    def test_spin_does_not_top_the_drift_score(self) -> None:
        # A spin is dramatic but it is not a drift; letting it win would send
        # the best camera to a botched corner.
        spin = _signals(400, sideslip_deg=150.0)
        good = _signals(400, sideslip_deg=DRIFT_REFERENCE_DEGREES)
        assert float(drift_score(_features(spin).cars[0], RATE).max()) < float(
            drift_score(_features(good).cars[0], RATE).max()
        )


class TestSpeedScore:
    def test_is_session_relative_not_absolute(self) -> None:
        # The same 60 km/h is a fast moment in a slow session and a slow moment
        # in a fast one. An absolute threshold would lose that entirely.
        varying = np.concatenate([np.full(200, 10.0), np.full(200, 20.0)])
        slow = _features(_signals(400, speed=varying))
        fast = _features(_signals(400, speed=varying * 5.0))
        assert float(speed_score(slow.cars[0], slow).max()) == pytest.approx(
            float(speed_score(fast.cars[0], fast).max()), abs=0.05
        )

    def test_faster_frames_score_higher(self) -> None:
        ramp = np.linspace(10.0, 60.0, 400)
        features = _features(_signals(400, speed=ramp))
        scores = speed_score(features.cars[0], features)
        assert scores[-1] > scores[0]

    def test_bounded(self) -> None:
        features = _features(_signals(400, speed=np.linspace(5.0, 90.0, 400)))
        scores = speed_score(features.cars[0], features)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0


class TestFlowScore:
    def test_smooth_driving_outscores_hesitant_driving(self) -> None:
        smooth = _signals(600, speed=np.full(600, 30.0))
        jittery_speed = 30.0 + 6.0 * np.sin(np.arange(600) * 0.9)
        jittery = _signals(600, speed=jittery_speed)
        smooth_features = _features(smooth)
        jittery_features = _features(jittery)
        assert float(np.mean(flow_score(smooth_features.cars[0], smooth_features))) > float(
            np.mean(flow_score(jittery_features.cars[0], jittery_features))
        )

    def test_stationary_car_scores_zero(self) -> None:
        car = _signals(400, speed=0.5)
        features = _features(car)
        assert float(flow_score(features.cars[0], features).max()) == 0.0


class TestRiskScore:
    def test_lone_car_on_a_smooth_line_scores_low(self) -> None:
        car = _signals(400)
        features = _features(car)
        assert float(risk_score(features.cars[0], features).max()) < 0.2

    def test_close_converging_cars_outscore_distant_ones(self) -> None:
        count = 400
        near = _signals(count, car_id=1, offset=5.0, speed=32.0)
        far = _signals(count, car_id=1, offset=200.0, speed=32.0)
        subject = _signals(count, car_id=0)
        close_features = _features(subject, near)
        far_features = _features(_signals(count, car_id=0), far)
        assert float(risk_score(close_features.cars[0], close_features).max()) > float(
            risk_score(far_features.cars[0], far_features).max()
        )

    def test_bounded(self) -> None:
        features = _features(_signals(400), _signals(400, car_id=1, offset=4.0))
        scores = risk_score(features.cars[0], features)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0


class TestCinematicScore:
    def test_elevation_change_raises_the_score(self) -> None:
        flat = _signals(600)
        descending = _signals(600, elevation=np.linspace(0.0, -60.0, 600))
        flat_features = _features(flat)
        descending_features = _features(descending)
        assert float(
            np.mean(cinematic_score(descending_features.cars[0], descending_features))
        ) > float(np.mean(cinematic_score(flat_features.cars[0], flat_features)))

    def test_stationary_car_scores_zero(self) -> None:
        car = _signals(400, speed=0.5)
        features = _features(car)
        assert float(cinematic_score(features.cars[0], features).max()) == 0.0

    def test_bounded(self) -> None:
        features = _features(
            _signals(400, elevation=np.linspace(0.0, -40.0, 400)),
            _signals(400, car_id=1, offset=15.0),
        )
        scores = cinematic_score(features.cars[0], features)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0


class TestScoreVectors:
    def test_one_vector_per_frame(self) -> None:
        features = _features(_signals(300))
        assert len(score_car(features, 0)) == 300

    def test_every_component_is_normalised(self) -> None:
        features = _features(
            _signals(400, sideslip_deg=25.0, elevation=np.linspace(0.0, -30.0, 400)),
            _signals(400, car_id=1, offset=8.0),
        )
        for vector in score_car(features, 0):
            for kind in ScoreKind:
                assert 0.0 <= vector[kind] <= 1.0

    def test_score_session_covers_every_car(self) -> None:
        features = _features(_signals(200), _signals(200, car_id=1, offset=20.0))
        assert set(score_session(features)) == {0, 1}

    def test_style_weighting_changes_the_ranking(self) -> None:
        # The whole point of styles: the same session must yield different
        # favourites under different weightings.
        slip = np.zeros(600)
        slip[100:200] = 35.0
        speeds = np.full(600, 20.0)
        speeds[400:500] = 70.0
        features = _features(_signals(600, speed=speeds, sideslip_deg=slip))
        scores = score_car(features, 0)

        drift_led = ScoreWeights(drift=1.0, speed=0.0, risk=0.0, flow=0.0, cinematic=0.0)
        speed_led = ScoreWeights(drift=0.0, speed=1.0, risk=0.0, flow=0.0, cinematic=0.0)
        best_for_drift = int(np.argmax([drift_led.combine(v) for v in scores]))
        best_for_speed = int(np.argmax([speed_led.combine(v) for v in scores]))
        assert 100 <= best_for_drift < 200
        assert 400 <= best_for_speed < 500


class TestDegenerateSessions:
    def test_single_frame_session_does_not_crash(self) -> None:
        features = _features(_signals(1))
        assert len(score_car(features, 0)) == 1

    def test_session_shorter_than_a_smoothing_window(self) -> None:
        features = _features(_signals(3))
        assert len(score_car(features, 0)) == 3

    def test_all_cars_stationary(self) -> None:
        features = _features(_signals(200, speed=0.0), _signals(200, speed=0.0, car_id=1))
        for vector in score_car(features, 0):
            assert vector.peak == 0.0

    def test_cars_with_different_frame_counts(self) -> None:
        # Multi-car sessions can end up with unequal decoded lengths; the
        # pairwise scorers must not index past the shorter array.
        long_car = _signals(400)
        short_car = _signals(250, car_id=1, offset=6.0)
        features = compute_features({0: long_car, 1: short_car}, np.arange(400) / RATE, RATE)
        vectors = score_car(features, 0)
        assert len(vectors) == 400
