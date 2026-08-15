"""Tests for the camera checks.

Two kinds of test here, and the split is deliberate.

The **constructed** ones build a shot whose geometry is known — a camera facing
away from its subject, a camera underground — and assert the corresponding check
fails. They are the ones that prove a check can detect anything at all; a check
that is never shown failing is a check nobody has shown works.

The **end-to-end** ones run a real synthetic session through the director and
assert properties of the whole report: that every camera gets every check, that
the blind-spot list is honest about what was skipped, and that nothing crashes on
a session with cars that come and go.
"""

from __future__ import annotations

import numpy as np
import pytest

from assetto_ai_director.application.camera.paths import SubjectTracks
from assetto_ai_director.application.camera.rigs import place_camera
from assetto_ai_director.application.diagnostics import diagnose_cameras
from assetto_ai_director.application.director.styles import BUILTIN_STYLES
from assetto_ai_director.application.pipeline import direct
from assetto_ai_director.domain.camera import (
    CameraKeyframe,
    CameraPath,
    CameraSpec,
    RigKind,
)
from assetto_ai_director.domain.diagnostics import Check, CheckStatus, DiagnosticReport
from assetto_ai_director.domain.geometry import CameraBasis, Vec3
from assetto_ai_director.domain.identity import CarIdentity
from assetto_ai_director.domain.session import SessionMeta
from assetto_ai_director.domain.shot import LineSide, Shot, ShotCandidate, ShotSize
from assetto_ai_director.domain.timeline import Timeline
from assetto_ai_director.domain.track import SplineSource, TrackSpline
from assetto_ai_director.domain.units import FovDegrees, Hertz, Score, Seconds
from assetto_ai_director.infrastructure.export.cameras_ini import planned_spans
from assetto_ai_director.infrastructure.synthetic.source import SyntheticSource

_FRAMES = 60
_RATE = 20.0
_RUN_METRES = 120.0
_FRAME_STEP_METRES = _RUN_METRES / (_FRAMES - 1)
"""Distance between consecutive samples of the straight-line fixture.

The span check reads the frame nearest a moment, so no measurement it makes can
be finer than this. Tests compare against it rather than against zero.
"""


def _tracks(positions: np.ndarray) -> SubjectTracks:
    """Wrap one car's positions as subject tracks.

    Args:
        positions: World positions, shape ``(n, 3)``.

    Returns:
        Tracks carrying that car as id zero.
    """
    count = positions.shape[0]
    return SubjectTracks(
        positions={0: positions},
        headings={0: np.zeros(count)},
        time=np.arange(count) / _RATE,
        sample_rate=_RATE,
    )


def _straight_line() -> np.ndarray:
    """A car driving along ``+z`` at a constant rate.

    Returns:
        World positions, shape ``(_FRAMES, 3)``.
    """
    return np.stack(
        [np.zeros(_FRAMES), np.zeros(_FRAMES), np.linspace(0.0, _RUN_METRES, _FRAMES)], axis=1
    )


def _timeline(basis: CameraBasis, *, rig: RigKind = RigKind.ROADSIDE) -> Timeline:
    """Build a one-shot timeline whose camera sits exactly where told.

    Args:
        basis: The camera placement to hold for the whole shot.
        rig: Which rig the shot claims to be.

    Returns:
        The timeline.
    """
    duration = (_FRAMES - 1) / _RATE
    keyframes = tuple(
        CameraKeyframe(Seconds(index / _RATE), basis, FovDegrees(40.0)) for index in range(_FRAMES)
    )
    candidate = ShotCandidate(
        start=Seconds(0.0),
        end=Seconds(duration),
        subjects=(0,),
        rig=rig,
        size=ShotSize.FULL,
        side=LineSide.LEFT,
        score=Score(0.5),
    )
    shot = Shot(
        candidate=candidate,
        path=CameraPath(keyframes=keyframes, spec=CameraSpec(rig=rig)),
        index=0,
    )
    session = SessionMeta(
        track_id="pk_akina",
        layout="akina_downhill",
        weather_id="3_clear",
        sample_rate=Hertz(_RATE),
        frame_count=_FRAMES,
        cars=(CarIdentity(0, "ks_toyota_ae86_tuned", "ducheck", "RUS", None, "09_panda_b", True),),
        source_name="test",
    )
    return Timeline(shots=(shot,), cuts=(), session=session, style_name="initial_d")


def _check(report: DiagnosticReport, name: str) -> Check:
    """Pull one named check off the first camera of a report.

    Args:
        report: The diagnostic report.
        name: The check's identifier.

    Returns:
        The check.

    Raises:
        AssertionError: If no check by that name exists, naming the ones that do.
            A test that silently skipped a renamed check would pass forever.
    """
    camera = report.cameras[0]
    for check in camera.checks:
        if check.name == name:
            return check
    message = f"No check named {name!r}; got {[c.name for c in camera.checks]}."
    raise AssertionError(message)


class TestChecksCanFail:
    """A check nobody has seen fail is a check nobody has shown works."""

    def test_a_camera_facing_away_is_caught(self) -> None:
        positions = _straight_line()
        # Sits at the far end looking further away, so the car is always behind.
        basis = CameraBasis.looking_at(Vec3(0.0, 1.0, 200.0), Vec3(0.0, 1.0, 400.0))
        report = diagnose_cameras(_timeline(basis), _tracks(positions))

        behind = _check(report, "subject_behind_camera")
        assert behind.status is CheckStatus.FAIL
        assert behind.value == pytest.approx(1.0)
        assert _check(report, "subject_in_frame").status is CheckStatus.FAIL

    def test_a_well_aimed_camera_passes(self) -> None:
        positions = _straight_line()
        basis = CameraBasis.looking_at(Vec3(18.0, 1.0, 60.0), Vec3(0.0, 0.5, 60.0))
        report = diagnose_cameras(_timeline(basis), _tracks(positions))

        assert _check(report, "subject_behind_camera").status is CheckStatus.PASS
        assert _check(report, "camera_height_above_subject").status is CheckStatus.PASS

    def test_a_buried_camera_is_caught(self) -> None:
        positions = _straight_line()
        basis = CameraBasis.looking_at(Vec3(18.0, -4.0, 60.0), Vec3(0.0, 0.0, 60.0))
        report = diagnose_cameras(_timeline(basis), _tracks(positions))

        height = _check(report, "camera_height_above_subject")
        assert height.status is CheckStatus.FAIL
        assert height.value == pytest.approx(-4.0)

    def test_a_camera_far_from_its_shot_size_is_caught(self) -> None:
        positions = _straight_line()
        # A full shot intends about 18 m; this stands 300 m off.
        basis = CameraBasis.looking_at(Vec3(300.0, 1.0, 60.0), Vec3(0.0, 0.5, 60.0))
        report = diagnose_cameras(_timeline(basis), _tracks(positions))

        assert _check(report, "camera_distance").status is CheckStatus.FAIL
        assert _check(report, "subject_apparent_size").status is CheckStatus.FAIL

    def test_a_missing_subject_track_is_unknown_not_a_pass(self) -> None:
        basis = CameraBasis.looking_at(Vec3(18.0, 1.0, 60.0), Vec3(0.0, 0.5, 60.0))
        empty = SubjectTracks(
            positions={7: _straight_line()},
            headings={7: np.zeros(_FRAMES)},
            time=np.arange(_FRAMES) / _RATE,
            sample_rate=_RATE,
        )
        report = diagnose_cameras(_timeline(basis), empty)

        assert _check(report, "subject_in_frame").status is CheckStatus.UNKNOWN
        assert report.cameras[0].status is not CheckStatus.PASS


class TestUnmeasurableChecksAreVisible:
    def test_track_geometry_checks_report_unknown(self) -> None:
        basis = CameraBasis.looking_at(Vec3(18.0, 1.0, 60.0), Vec3(0.0, 0.5, 60.0))
        report = diagnose_cameras(_timeline(basis), _tracks(_straight_line()))

        for name in ("camera_on_surface", "sight_line_clear"):
            assert _check(report, name).status is CheckStatus.UNKNOWN
            assert _check(report, name).value is None

    def test_every_unknown_reaches_the_blind_spot_list(self) -> None:
        basis = CameraBasis.looking_at(Vec3(18.0, 1.0, 60.0), Vec3(0.0, 0.5, 60.0))
        report = diagnose_cameras(_timeline(basis), _tracks(_straight_line()))

        unmeasured = {
            check.name
            for camera in report.cameras
            for check in camera.checks
            if check.status is CheckStatus.UNKNOWN
        }
        assert unmeasured
        for name in unmeasured:
            assert any(gap.startswith(name) for gap in report.blind_spots)

    def test_a_report_full_of_unknowns_is_not_reported_as_passing(self) -> None:
        # The failure mode this exists against: a green report that is green
        # because nothing was measured.
        basis = CameraBasis.looking_at(Vec3(18.0, 1.0, 60.0), Vec3(0.0, 0.5, 60.0))
        report = diagnose_cameras(_timeline(basis), _tracks(_straight_line()))
        assert report.status is not CheckStatus.PASS


class TestSpanCheck:
    def test_it_is_unknown_without_exported_spans(self) -> None:
        basis = CameraBasis.looking_at(Vec3(18.0, 1.0, 60.0), Vec3(0.0, 0.5, 60.0))
        report = diagnose_cameras(_timeline(basis), _tracks(_straight_line()))
        assert _check(report, "span_vs_position").status is CheckStatus.UNKNOWN

    def test_a_span_matching_the_car_passes(self) -> None:
        basis = CameraBasis.looking_at(Vec3(18.0, 1.0, 60.0), Vec3(0.0, 0.5, 60.0))
        timeline = _timeline(basis)
        # One shot covering the whole run: its mid-span is 0.5 and a constant-rate
        # car is halfway along at mid-shot, so the two agree exactly.
        report = diagnose_cameras(
            timeline, _tracks(_straight_line()), exported_spans={0: (0.0, 1.0)}
        )
        span = _check(report, "span_vs_position")
        assert span.status is CheckStatus.PASS
        # Not exactly zero: the check samples the nearest frame, and 60 frames
        # over 120 m puts consecutive samples ~2 m apart. Half a frame step is
        # the floor on this measurement, not an error in it.
        assert span.value == pytest.approx(0.0, abs=_FRAME_STEP_METRES)

    def test_a_span_pointing_elsewhere_is_measured_in_metres(self) -> None:
        basis = CameraBasis.looking_at(Vec3(18.0, 1.0, 60.0), Vec3(0.0, 0.5, 60.0))
        timeline = _timeline(basis)
        # The exporter claims the far end of the track; the car is halfway.
        report = diagnose_cameras(
            timeline, _tracks(_straight_line()), exported_spans={0: (0.9, 1.0)}
        )
        span = _check(report, "span_vs_position")
        assert span.status is CheckStatus.FAIL
        assert span.unit == "m"
        # 0.95 written against 0.5 actual, over the run's length.
        assert span.value == pytest.approx(0.45 * _RUN_METRES, rel=0.05)

    def test_the_threshold_is_the_camera_s_own_span_not_a_flat_fraction(self) -> None:
        # The same 0.03 drift is fine for a camera covering a tenth of the lap
        # and wrong for one covering a hundredth: in the second case the game
        # shows a different camera than the shot intended. A flat "percent of a
        # lap" cannot express that.
        basis = CameraBasis.looking_at(Vec3(18.0, 1.0, 60.0), Vec3(0.0, 0.5, 60.0))
        timeline = _timeline(basis)
        tracks = _tracks(_straight_line())

        wide = diagnose_cameras(timeline, tracks, exported_spans={0: (0.03, 0.99)})
        narrow = diagnose_cameras(timeline, tracks, exported_spans={0: (0.52, 0.54)})

        assert _check(wide, "span_vs_position").status is CheckStatus.PASS
        assert _check(narrow, "span_vs_position").status is CheckStatus.FAIL

    def test_a_shot_the_file_drops_is_unknown_not_a_failure(self) -> None:
        basis = CameraBasis.looking_at(Vec3(18.0, 1.0, 60.0), Vec3(0.0, 0.5, 60.0))
        report = diagnose_cameras(
            _timeline(basis, rig=RigKind.DRONE), _tracks(_straight_line()), exported_spans={}
        )
        span = _check(report, "span_vs_position")
        assert span.status is CheckStatus.UNKNOWN
        assert "cameras.ini does not carry this shot" in span.detail


@pytest.fixture(scope="module")
def directed() -> tuple[Timeline, SubjectTracks]:
    """Direct a synthetic session once and rebuild the tracks it used.

    Module-scoped because the DP scheduler is the expensive part of this project
    — roughly ten seconds per three minutes of horizon — and running it once per
    test turned this file into the slowest in the suite.

    Returns:
        The timeline and the subject tracks the director worked from.
    """
    session = SyntheticSource(seconds=40.0, cars=3, seed=5).generate()
    timeline = direct(
        session.signals,
        session.meta,
        BUILTIN_STYLES["initial_d"],
        session.time,
        float(session.sample_rate),
    )
    tracks = SubjectTracks(
        positions={car_id: car.position for car_id, car in session.signals.items()},
        headings={car_id: car.heading for car_id, car in session.signals.items()},
        time=session.time,
        sample_rate=float(session.sample_rate),
    )
    return timeline, tracks


class TestOnARealSession:
    def test_every_shot_gets_a_diagnostic(self, directed: tuple[Timeline, SubjectTracks]) -> None:
        timeline, tracks = directed
        report = diagnose_cameras(timeline, tracks, exported_spans=planned_spans(timeline))
        assert len(report.cameras) == timeline.shot_count

    def test_every_camera_gets_the_same_checks_in_the_same_order(
        self, directed: tuple[Timeline, SubjectTracks]
    ) -> None:
        # Reports get diffed between runs; a check appearing only sometimes
        # makes a diff unreadable.
        timeline, tracks = directed
        report = diagnose_cameras(timeline, tracks, exported_spans=planned_spans(timeline))
        orders = {tuple(check.name for check in camera.checks) for camera in report.cameras}
        assert len(orders) == 1

    def test_the_session_is_named_in_the_report(
        self, directed: tuple[Timeline, SubjectTracks]
    ) -> None:
        timeline, tracks = directed
        report = diagnose_cameras(timeline, tracks)
        assert report.track_id == timeline.session.track_id
        assert report.style_name == "initial_d"

    def test_the_assumptions_are_stated(self, directed: tuple[Timeline, SubjectTracks]) -> None:
        timeline, tracks = directed
        report = diagnose_cameras(timeline, tracks)
        assert any("Car size is a guess" in line for line in report.assumptions)
        assert any("Field of view is treated as vertical" in line for line in report.assumptions)

    def test_the_tally_accounts_for_every_camera(
        self, directed: tuple[Timeline, SubjectTracks]
    ) -> None:
        timeline, tracks = directed
        report = diagnose_cameras(timeline, tracks, exported_spans=planned_spans(timeline))
        assert sum(report.tally().values()) == len(report.cameras)


def _road(half_width: float = 5.0, count: int = 25) -> TrackSpline:
    """A level road along the fixture's line of travel, with known edges.

    Args:
        half_width: Distance from the centreline to each edge, in metres.
        count: How many vertices.

    Returns:
        The spline.
    """
    points = np.stack(
        [np.zeros(count), np.zeros(count), np.linspace(0.0, _RUN_METRES, count)], axis=1
    )
    return TrackSpline(
        points=points,
        source=SplineSource.AI_LINE,
        left_width=np.full(count, half_width),
        right_width=np.full(count, half_width),
    )


class TestWithATrack:
    """The checks that were unmeasurable until a centreline existed."""

    @staticmethod
    def _at(x: float, y: float, z: float = 60.0) -> CameraBasis:
        """A camera at a chosen place, looking at the road beside it."""
        return CameraBasis.looking_at(Vec3(x, y, z), Vec3(0.0, 0.5, z))

    def test_without_a_track_both_stay_unknown(self) -> None:
        report = diagnose_cameras(_timeline(self._at(18.0, 1.0)), _tracks(_straight_line()))
        assert _check(report, "camera_on_surface").status is CheckStatus.UNKNOWN
        assert _check(report, "camera_lateral_offset").status is CheckStatus.UNKNOWN

    def test_a_camera_beside_the_road_passes_both(self) -> None:
        report = diagnose_cameras(
            _timeline(self._at(4.0, 1.0)), _tracks(_straight_line()), track=_road()
        )
        assert _check(report, "camera_on_surface").status is CheckStatus.PASS
        assert _check(report, "camera_lateral_offset").status is CheckStatus.PASS

    def test_a_camera_far_past_the_road_edge_is_caught(self) -> None:
        # The failure the placement code has always been able to produce and
        # nothing could see: a long shot offsets sideways with no idea where the
        # road ends.
        report = diagnose_cameras(
            _timeline(self._at(40.0, 1.0)), _tracks(_straight_line()), track=_road()
        )
        check = _check(report, "camera_lateral_offset")
        assert check.status is CheckStatus.FAIL
        assert check.value == pytest.approx(40.0, rel=0.05)

    def test_a_camera_below_the_road_is_caught(self) -> None:
        report = diagnose_cameras(
            _timeline(self._at(4.0, -3.0)), _tracks(_straight_line()), track=_road()
        )
        check = _check(report, "camera_on_surface")
        assert check.status is CheckStatus.FAIL
        assert check.value == pytest.approx(-3.0)

    def test_a_driven_line_answers_height_but_refuses_width(self) -> None:
        # It knows where the road was; it has no idea how wide it is, and a
        # plausible number would be worse than an admission.
        driven = TrackSpline(points=_road().points, source=SplineSource.DRIVEN)
        report = diagnose_cameras(
            _timeline(self._at(40.0, 1.0)), _tracks(_straight_line()), track=driven
        )
        assert _check(report, "camera_on_surface").status is CheckStatus.PASS
        lateral = _check(report, "camera_lateral_offset")
        assert lateral.status is CheckStatus.UNKNOWN
        assert "records no" in lateral.detail

    def test_the_span_check_switches_to_the_track_s_own_position(self) -> None:
        report = diagnose_cameras(
            _timeline(self._at(4.0, 1.0)),
            _tracks(_straight_line()),
            exported_spans={0: (0.0, 1.0)},
            track=_road(),
        )
        check = _check(report, "span_vs_position")
        assert "the track's ai_line" in check.detail

    def test_the_assumptions_stop_claiming_the_ai_line_is_unread(self) -> None:
        # A report produced from the AI line must not carry a note saying the AI
        # line was not parsed. That would be the report lying about its footing.
        without = diagnose_cameras(_timeline(self._at(4.0, 1.0)), _tracks(_straight_line()))
        with_track = diagnose_cameras(
            _timeline(self._at(4.0, 1.0)), _tracks(_straight_line()), track=_road()
        )
        assert any("no centreline was available" in line for line in without.assumptions)
        assert not any("no centreline was available" in line for line in with_track.assumptions)

    def test_a_driven_line_says_so_in_the_assumptions(self) -> None:
        driven = TrackSpline(points=_road().points, source=SplineSource.DRIVEN)
        report = diagnose_cameras(
            _timeline(self._at(4.0, 1.0)), _tracks(_straight_line()), track=driven
        )
        assert any("averaged from a car" in line for line in report.assumptions)

    def test_caller_notes_reach_the_blind_spots(self) -> None:
        report = diagnose_cameras(
            _timeline(self._at(4.0, 1.0)),
            _tracks(_straight_line()),
            notes=("The track is not installed on this machine.",),
        )
        assert "The track is not installed on this machine." in report.blind_spots


class TestPlacementIsClampedToTheRoad:
    """The fix for what the road-edge check measured."""

    @staticmethod
    def _session() -> tuple[SubjectTracks, TrackSpline]:
        """A straight run and the road it is on."""
        return _tracks(_straight_line()), _road(half_width=4.0)

    def test_a_trackside_camera_is_pulled_back_onto_the_verge(self) -> None:
        tracks, road = self._session()
        subject = tracks.positions[0]
        loose = place_camera(
            RigKind.ROADSIDE,
            ShotSize.LONG,
            LineSide.LEFT,
            subject,
            heading=tracks.headings[0],
            companion=None,
            dt=1.0 / _RATE,
        )
        clamped = place_camera(
            RigKind.ROADSIDE,
            ShotSize.LONG,
            LineSide.LEFT,
            subject,
            heading=tracks.headings[0],
            companion=None,
            dt=1.0 / _RATE,
            track=road,
        )
        before = abs(float(road.project(loose.positions[0]).lateral))
        after = abs(float(road.project(clamped.positions[0]).lateral))
        assert before > 30.0, "a long shot offsets far enough to leave the road"
        assert after < before
        assert after <= 4.0 + 2.0

    def test_it_keeps_the_side_the_shot_was_composed_from(self) -> None:
        # The 180-degree rule depends on it: a camera dragged across the line
        # would reverse the car's screen direction between two shots.
        tracks, road = self._session()
        sides = {}
        for side in (LineSide.LEFT, LineSide.RIGHT):
            placement = place_camera(
                RigKind.ROADSIDE,
                ShotSize.LONG,
                side,
                tracks.positions[0],
                heading=tracks.headings[0],
                companion=None,
                dt=1.0 / _RATE,
                track=road,
            )
            sides[side] = float(road.project(placement.positions[0]).lateral)
        assert sides[LineSide.LEFT] * sides[LineSide.RIGHT] < 0.0

    def test_a_chase_camera_is_left_alone(self) -> None:
        # A follow rig trails the car and is meant to be off the road; clamping
        # it would be correcting it for doing its job.
        tracks, road = self._session()
        without = place_camera(
            RigKind.FOLLOW,
            ShotSize.LONG,
            LineSide.LEFT,
            tracks.positions[0],
            heading=tracks.headings[0],
            companion=None,
            dt=1.0 / _RATE,
        )
        with_road = place_camera(
            RigKind.FOLLOW,
            ShotSize.LONG,
            LineSide.LEFT,
            tracks.positions[0],
            heading=tracks.headings[0],
            companion=None,
            dt=1.0 / _RATE,
            track=road,
        )
        assert np.allclose(without.positions, with_road.positions)

    def test_the_road_edge_check_does_not_fault_a_chase_camera(self) -> None:
        tracks, road = self._session()
        basis = CameraBasis.looking_at(Vec3(40.0, 1.0, 60.0), Vec3(0.0, 0.5, 60.0))
        report = diagnose_cameras(_timeline(basis, rig=RigKind.FOLLOW), tracks, track=road)
        check = _check(report, "camera_lateral_offset")
        assert check.status is CheckStatus.UNKNOWN
        assert "not meant to stand beside the road" in check.detail
