"""Tests for the remaining domain models: telemetry, events, scoring, shots, timeline."""

from __future__ import annotations

import numpy as np
import pytest

from assetto_ai_director.domain.camera import (
    CameraKeyframe,
    CameraPath,
    CameraSpec,
    RigKind,
)
from assetto_ai_director.domain.errors import ChannelUnavailableError
from assetto_ai_director.domain.events import Event, EventKind, EventSeverity
from assetto_ai_director.domain.frames import ReferenceFrame, WheelIndex
from assetto_ai_director.domain.geometry import CameraBasis, Vec3
from assetto_ai_director.domain.identity import CarIdentity
from assetto_ai_director.domain.scoring import ScoreVector, ScoreWeights
from assetto_ai_director.domain.session import RecordingQuality, SessionMeta
from assetto_ai_director.domain.shot import LineSide, Shot, ShotCandidate, ShotSize
from assetto_ai_director.domain.telemetry import (
    CarTelemetry,
    Channel,
    ChannelAvailability,
    SessionTelemetry,
)
from assetto_ai_director.domain.timeline import Cut, Timeline, TransitionKind
from assetto_ai_director.domain.units import FovDegrees, Hertz, Score, Seconds


class TestWheelIndex:
    def test_ordering_matches_the_game(self) -> None:
        # FL, FR, RL, RR = range(4). A rear-pair swap inverts every drift sign
        # while leaving magnitudes plausible, so this ordering is asserted.
        assert [w.slot for w in WheelIndex] == [0, 1, 2, 3]
        assert WheelIndex.REAR_LEFT.slot == 2

    def test_rear_detection(self) -> None:
        assert WheelIndex.REAR_LEFT.is_rear
        assert not WheelIndex.FRONT_LEFT.is_rear


class TestChannelAvailability:
    def test_declares_what_it_has(self) -> None:
        availability = ChannelAvailability.of("test", [Channel.POSITION, Channel.SPEED])
        assert availability.has(Channel.POSITION)
        assert not availability.has(Channel.WHEEL_SLIP_ANGLE)

    def test_require_raises_naming_the_source(self) -> None:
        availability = ChannelAvailability.of("sharedmem", [Channel.POSITION])
        with pytest.raises(ChannelUnavailableError) as excinfo:
            availability.require(Channel.WHEEL_SLIP_ANGLE)
        assert "sharedmem" in str(excinfo.value)

    def test_uncalibrated_rotation_blocks_true_sideslip(self) -> None:
        # Heading present but unvalidated must not be treated as trustworthy.
        availability = ChannelAvailability.of(
            "acreplay", [Channel.HEADING], rotation_calibrated=False
        )
        assert not availability.supports_true_sideslip

    def test_calibrated_rotation_enables_true_sideslip(self) -> None:
        availability = ChannelAvailability.of(
            "acreplay", [Channel.HEADING], rotation_calibrated=True
        )
        assert availability.supports_true_sideslip

    def test_drift_scoring_falls_back_to_wheel_slip(self) -> None:
        # No validated heading, but real per-wheel slip angle: drift is still
        # scoreable via the documented proxy.
        availability = ChannelAvailability.of("acreplay", [Channel.WHEEL_SLIP_ANGLE])
        assert not availability.supports_true_sideslip
        assert availability.supports_drift_scoring

    def test_velocity_frame_defaults_to_unknown(self) -> None:
        # No source documents the replay's velocity frame, so the default must
        # not claim to know it.
        availability = ChannelAvailability.of("acreplay", [Channel.VELOCITY])
        assert availability.velocity_frame is ReferenceFrame.UNKNOWN


class TestSessionTelemetry:
    def _session(self) -> SessionTelemetry:
        time = np.arange(0.0, 1.0, 0.1)
        car = CarTelemetry(0, {Channel.SPEED: np.full(time.shape, 30.0)})
        return SessionTelemetry(
            cars=(car,),
            sample_rate=Hertz(10.0),
            availability=ChannelAvailability.of("t", [Channel.SPEED]),
            time=time,
        )

    def test_frame_count_and_duration(self) -> None:
        session = self._session()
        assert session.frame_count == 10
        assert session.duration == pytest.approx(0.9)

    def test_index_at_finds_nearest_sample(self) -> None:
        session = self._session()
        assert session.index_at(Seconds(0.44)) == 4
        assert session.index_at(Seconds(0.46)) == 5

    def test_index_at_clamps_beyond_the_end(self) -> None:
        assert self._session().index_at(Seconds(99.0)) == 9

    def test_unknown_car_raises(self) -> None:
        with pytest.raises(IndexError):
            self._session().car(7)


class TestRecordingQuality:
    @pytest.mark.parametrize(
        ("rate", "expected"),
        [
            (8.0, RecordingQuality.UNUSABLE),
            (16.0, RecordingQuality.COARSE),
            (33.3, RecordingQuality.GOOD),
            (67.0, RecordingQuality.EXCELLENT),
        ],
    )
    def test_bands(self, rate: float, expected: RecordingQuality) -> None:
        assert RecordingQuality.from_rate(Hertz(rate)) is expected

    def test_only_good_and_above_support_event_timing(self) -> None:
        assert not RecordingQuality.COARSE.supports_event_timing
        assert RecordingQuality.GOOD.supports_event_timing

    def test_every_band_has_advice(self) -> None:
        assert all(band.advice for band in RecordingQuality)


class TestSessionMeta:
    def _meta(self) -> SessionMeta:
        return SessionMeta(
            track_id="pk_akina",
            layout="akina_downhill",
            weather_id="7_heavy_clouds",
            sample_rate=Hertz(33.333),
            frame_count=49012,
            cars=(
                CarIdentity(0, "ks_toyota_ae86_tuned", "ducheck", "RUS", None, "09_panda_b", True),
                CarIdentity(1, "ks_mazda_rx7_spirit_r"),
            ),
            source_name="acreplay",
        )

    def test_track_label_includes_layout(self) -> None:
        assert self._meta().track_label == "pk_akina/akina_downhill"

    def test_duration_from_rate(self) -> None:
        assert self._meta().duration == pytest.approx(49012 / 33.333, rel=1e-4)

    def test_player_is_found(self) -> None:
        player = self._meta().player()
        assert player is not None
        assert player.driver_name == "ducheck"

    def test_display_name_disambiguates_identical_models(self) -> None:
        # A one-make field is exactly where unnamed AI cars collide.
        assert CarIdentity(3, "ks_toyota_ae86").display_name == "ks_toyota_ae86 (#3)"

    def test_short_model_strips_author_prefix(self) -> None:
        assert CarIdentity(0, "ks_toyota_ae86_tuned").short_model == "toyota_ae86_tuned"

    def test_short_model_keeps_long_first_tokens(self) -> None:
        assert CarIdentity(0, "cherie_bomex_supra").short_model == "cherie_bomex_supra"


class TestEvents:
    def test_span_must_not_run_backwards(self) -> None:
        with pytest.raises(ValueError, match="ends"):
            Event(EventKind.APEX, Seconds(5.0), Seconds(4.0), (0,))

    def test_subject_is_required(self) -> None:
        with pytest.raises(ValueError, match="no subject"):
            Event(EventKind.APEX, Seconds(1.0), Seconds(2.0), ())

    def test_confidence_is_bounded(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            Event(EventKind.APEX, Seconds(1.0), Seconds(2.0), (0,), confidence=1.5)

    def test_focus_prefers_peak_over_midpoint(self) -> None:
        # A drift's peak angle is well after its start; the cut wants the peak.
        event = Event(EventKind.DRIFT_START, Seconds(0.0), Seconds(4.0), (0,), peak=Seconds(3.2))
        assert event.focus == pytest.approx(3.2)

    def test_focus_falls_back_to_midpoint(self) -> None:
        event = Event(EventKind.DRIFT_START, Seconds(0.0), Seconds(4.0), (0,))
        assert event.focus == pytest.approx(2.0)

    def test_overlap_detection(self) -> None:
        first = Event(EventKind.APEX, Seconds(0.0), Seconds(2.0), (0,))
        second = Event(EventKind.APEX, Seconds(1.0), Seconds(3.0), (0,))
        third = Event(EventKind.APEX, Seconds(5.0), Seconds(6.0), (0,))
        assert first.overlaps(second)
        assert not first.overlaps(third)

    def test_multi_car_kinds(self) -> None:
        assert EventKind.OVERTAKE_COMPLETE.is_multi_car
        assert not EventKind.APEX.is_multi_car

    def test_severity_from_ratio(self) -> None:
        assert EventSeverity.from_ratio(1.1) is EventSeverity.SUBTLE
        assert EventSeverity.from_ratio(2.0) is EventSeverity.MODERATE
        assert EventSeverity.from_ratio(5.0) is EventSeverity.STRONG


class TestScoring:
    def test_components_must_be_normalised(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            ScoreVector(drift=Score(1.5))

    def test_weights_must_be_non_negative(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            ScoreWeights(drift=-1.0)

    def test_combine_is_a_normalised_weighted_sum(self) -> None:
        weights = ScoreWeights(drift=1.0, speed=0.0, risk=0.0, flow=0.0, cinematic=0.0)
        scores = ScoreVector(drift=Score(0.8))
        assert weights.combine(scores) == pytest.approx(0.8)

    def test_legibility_penalty_reduces_the_score(self) -> None:
        # The sharpest style distinction: a static legible frame lets an
        # audience read a drift; frenetic coverage converts craft into noise.
        weights = ScoreWeights(drift=1.0, speed=0.0, risk=0.0, flow=0.0, cinematic=0.0)
        weights_strict = ScoreWeights(
            drift=1.0, speed=0.0, risk=0.0, flow=0.0, cinematic=0.0, legibility=1.0
        )
        scores = ScoreVector(drift=Score(0.8))
        assert weights_strict.combine(scores, illegibility=0.5) < weights.combine(scores, 0.5)

    def test_combine_is_clamped(self) -> None:
        weights = ScoreWeights(drift=1.0, legibility=10.0)
        assert weights.combine(ScoreVector(drift=Score(0.5)), illegibility=1.0) == 0.0

    def test_zero_weights_yield_zero(self) -> None:
        weights = ScoreWeights(drift=0.0, speed=0.0, risk=0.0, flow=0.0, cinematic=0.0)
        assert weights.combine(ScoreVector(drift=Score(1.0))) == 0.0

    def test_peak_reports_the_strongest_component(self) -> None:
        assert ScoreVector(drift=Score(0.2), risk=Score(0.9)).peak == pytest.approx(0.9)


class TestShots:
    def test_jump_cut_rule_needs_a_marked_size_change(self) -> None:
        # Full to medium reads as a glitch; full to close-up reads as an edit.
        assert not ShotSize.FULL.differs_markedly_from(ShotSize.MEDIUM)
        assert ShotSize.FULL.differs_markedly_from(ShotSize.CLOSE_UP)

    def test_crossing_the_line_is_detected(self) -> None:
        assert LineSide.LEFT.crosses_line_with(LineSide.RIGHT)
        assert not LineSide.LEFT.crosses_line_with(LineSide.LEFT)

    def test_neutral_shot_is_a_legal_bridge(self) -> None:
        # A camera may move to the far side through a neutral shot without a
        # visible reversal of screen direction.
        assert not LineSide.LEFT.crosses_line_with(LineSide.NEUTRAL)
        assert not LineSide.NEUTRAL.crosses_line_with(LineSide.RIGHT)

    def test_candidate_requires_positive_duration(self) -> None:
        with pytest.raises(ValueError, match="positive duration"):
            ShotCandidate(
                Seconds(4.0),
                Seconds(4.0),
                (0,),
                RigKind.ROADSIDE,
                ShotSize.FULL,
                LineSide.LEFT,
                Score(0.5),
            )

    def test_two_shot_detection(self) -> None:
        candidate = ShotCandidate(
            Seconds(0.0),
            Seconds(2.0),
            (0, 1),
            RigKind.ROADSIDE,
            ShotSize.FULL,
            LineSide.LEFT,
            Score(0.5),
        )
        assert candidate.is_two_shot

    def test_subject_sharing_distinguishes_continuity_from_subject_change(self) -> None:
        # Continuity rules apply between shots of the same car. Cutting to a
        # different car is a change of subject, not a line crossing.
        a = ShotCandidate(
            Seconds(0.0),
            Seconds(2.0),
            (0,),
            RigKind.ROADSIDE,
            ShotSize.FULL,
            LineSide.LEFT,
            Score(0.5),
        )
        b = ShotCandidate(
            Seconds(2.0),
            Seconds(4.0),
            (1,),
            RigKind.ROADSIDE,
            ShotSize.FULL,
            LineSide.RIGHT,
            Score(0.5),
        )
        assert not a.shares_subject_with(b)


class TestCamera:
    def test_implausible_fov_is_refused(self) -> None:
        with pytest.raises(ValueError, match="Field of view"):
            CameraSpec(RigKind.DRONE, fov=FovDegrees(200.0))

    def test_zoom_tracking_interpolates(self) -> None:
        spec = CameraSpec(RigKind.ROADSIDE, fov_near=FovDegrees(60.0), fov_far=FovDegrees(20.0))
        assert spec.zooms
        # Wider lens up close, narrower far away, holding apparent size constant.
        assert spec.fov_at(0.0, 0.0, 100.0) == pytest.approx(60.0)  # type: ignore[arg-type]
        assert spec.fov_at(100.0, 0.0, 100.0) == pytest.approx(20.0)  # type: ignore[arg-type]
        assert spec.fov_at(50.0, 0.0, 100.0) == pytest.approx(40.0)  # type: ignore[arg-type]

    def test_without_zoom_the_nominal_fov_is_used(self) -> None:
        spec = CameraSpec(RigKind.STATIC, fov=FovDegrees(35.0))
        assert spec.fov_at(50.0, 0.0, 100.0) == pytest.approx(35.0)  # type: ignore[arg-type]

    def test_rig_classification(self) -> None:
        assert RigKind.BONNET.is_car_mounted
        assert RigKind.DRONE.is_free_flying
        assert RigKind.ROADSIDE.is_fixed_position
        assert not RigKind.BONNET.can_frame_two_subjects

    def test_path_requires_keyframes(self) -> None:
        with pytest.raises(ValueError, match="at least one keyframe"):
            CameraPath((), CameraSpec(RigKind.STATIC))

    def test_path_rejects_unordered_keyframes(self) -> None:
        basis = CameraBasis.looking_at(Vec3(0.0, 1.0, 0.0), Vec3(0.0, 0.0, 10.0))
        frames = (
            CameraKeyframe(Seconds(1.0), basis, FovDegrees(40.0)),
            CameraKeyframe(Seconds(0.0), basis, FovDegrees(40.0)),
        )
        with pytest.raises(ValueError, match="increasing time"):
            CameraPath(frames, CameraSpec(RigKind.STATIC))


class TestTimeline:
    def _shot(self, index: int, start: float, end: float) -> Shot:
        basis = CameraBasis.looking_at(Vec3(0.0, 1.0, 0.0), Vec3(0.0, 0.0, 10.0))
        spec = CameraSpec(RigKind.ROADSIDE)
        path = CameraPath(
            (
                CameraKeyframe(Seconds(start), basis, FovDegrees(40.0)),
                CameraKeyframe(Seconds(end), basis, FovDegrees(40.0)),
            ),
            spec,
        )
        candidate = ShotCandidate(
            Seconds(start),
            Seconds(end),
            (0,),
            RigKind.ROADSIDE,
            ShotSize.FULL,
            LineSide.LEFT,
            Score(0.5),
            spec=spec,
        )
        return Shot(candidate, path, index)

    def _timeline(self, bounds: list[tuple[float, float]]) -> Timeline:
        shots = tuple(self._shot(i, a, b) for i, (a, b) in enumerate(bounds))
        cuts = tuple(
            Cut(Seconds(b), TransitionKind.CUT, from_index=i, to_index=i + 1)
            for i, (_, b) in enumerate(bounds[:-1])
        )
        meta = SessionMeta("pk_akina", "akina_downhill", "clear", Hertz(33.3), 100, (), "test")
        return Timeline(shots, cuts, meta, "initial_d")

    def test_contiguous_shots_tile(self) -> None:
        assert self._timeline([(0.0, 2.0), (2.0, 5.0), (5.0, 6.5)]).tiles_without_gaps()

    def test_gaps_are_detected(self) -> None:
        # A gap would mean a moment with no camera assigned, which an edit
        # cannot contain.
        assert not self._timeline([(0.0, 2.0), (2.5, 5.0)]).tiles_without_gaps()

    def test_average_shot_length(self) -> None:
        timeline = self._timeline([(0.0, 2.0), (2.0, 4.0), (4.0, 8.0)])
        assert timeline.average_shot_length == pytest.approx(8.0 / 3.0)

    def test_shot_durations_preserve_order(self) -> None:
        # Order carries the 1/f structure the rhythm check looks for.
        timeline = self._timeline([(0.0, 1.0), (1.0, 4.0), (4.0, 5.0)])
        assert list(timeline.shot_durations) == pytest.approx([1.0, 3.0, 1.0])

    def test_shot_at_finds_the_covering_shot(self) -> None:
        timeline = self._timeline([(0.0, 2.0), (2.0, 5.0)])
        shot = timeline.shot_at(Seconds(3.0))
        assert shot is not None
        assert shot.index == 1

    def test_shot_at_returns_none_outside_the_edit(self) -> None:
        assert self._timeline([(0.0, 2.0)]).shot_at(Seconds(99.0)) is None

    def test_notes_accumulate(self) -> None:
        timeline = self._timeline([(0.0, 2.0)]).with_note("degraded sideslip")
        assert "degraded sideslip" in timeline.notes

    def test_cut_cannot_have_duration(self) -> None:
        with pytest.raises(ValueError, match="cannot have duration"):
            Cut(Seconds(1.0), TransitionKind.CUT, duration=Seconds(0.5))

    def test_dissolve_may_have_duration(self) -> None:
        assert Cut(Seconds(1.0), TransitionKind.DISSOLVE, duration=Seconds(0.5)).duration == 0.5

    def test_slow_motion_detection(self) -> None:
        assert Cut(Seconds(1.0), speed_factor=0.4).is_slow_motion
        assert not Cut(Seconds(1.0)).is_slow_motion
