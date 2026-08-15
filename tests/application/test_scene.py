"""Tests for building the wireframe a camera would see.

The clipping tests carry the weight. Projecting a segment that runs from in front
of the camera to behind it, without cutting it at the camera plane, draws a
stroke straight across the frame in the opposite direction — and that artefact
looks exactly like a piece of track geometry. A diagnostic image that invents
geometry is worse than no image.
"""

from __future__ import annotations

import numpy as np
import pytest

from assetto_ai_director.application.camera.paths import SubjectTracks
from assetto_ai_director.application.diagnostics import diagnose_cameras
from assetto_ai_director.application.diagnostics.scene import (
    Polyline,
    StrokeKind,
    build_previews,
    build_scene,
    project_scene,
)
from assetto_ai_director.domain.camera import (
    CameraKeyframe,
    CameraPath,
    CameraSpec,
    RigKind,
)
from assetto_ai_director.domain.geometry import CameraBasis, Vec3
from assetto_ai_director.domain.identity import CarIdentity
from assetto_ai_director.domain.session import SessionMeta
from assetto_ai_director.domain.shot import LineSide, Shot, ShotCandidate, ShotSize
from assetto_ai_director.domain.timeline import Timeline
from assetto_ai_director.domain.units import FovDegrees, Hertz, Score, Seconds

_FOV = FovDegrees(40.0)
_FRAMES = 40
_RATE = 20.0
_CAR_BOX_LINES = 6
"""A car box is its footprint, its roof, and the four uprights joining them."""


def _camera() -> CameraBasis:
    """A camera at the origin looking along ``+z``."""
    return CameraBasis.looking_at(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 10.0))


def _tracks(cars: int = 2) -> SubjectTracks:
    """Tracks for a few cars driving along ``+z`` in parallel.

    Args:
        cars: How many cars.

    Returns:
        The tracks.
    """
    positions = {}
    headings = {}
    for car in range(cars):
        positions[car] = np.stack(
            [
                np.full(_FRAMES, float(car) * 4.0),
                np.zeros(_FRAMES),
                np.linspace(0.0, 200.0, _FRAMES),
            ],
            axis=1,
        )
        headings[car] = np.zeros(_FRAMES)
    return SubjectTracks(positions, headings, np.arange(_FRAMES) / _RATE, _RATE)


class TestClipping:
    def test_a_segment_crossing_behind_the_camera_is_cut(self) -> None:
        line = Polyline(
            points=(Vec3(0.0, 0.0, 20.0), Vec3(0.0, 0.0, -20.0)), kind=StrokeKind.SUBJECT_PATH
        )
        strokes = [
            s
            for s in project_scene(_camera(), _FOV, [line])
            if s.kind is not StrokeKind.FRAME_CENTRE
        ]
        assert len(strokes) == 1
        # Both ends must now be in front, so neither coordinate is a reflection.
        assert all(abs(v) < 1e6 for _, v in strokes[0].points)

    def test_a_line_entirely_behind_draws_nothing(self) -> None:
        line = Polyline(
            points=(Vec3(0.0, 0.0, -10.0), Vec3(0.0, 0.0, -30.0)), kind=StrokeKind.SUBJECT_PATH
        )
        strokes = [
            s
            for s in project_scene(_camera(), _FOV, [line])
            if s.kind is not StrokeKind.FRAME_CENTRE
        ]
        assert strokes == []

    def test_a_line_that_leaves_and_returns_becomes_two_strokes(self) -> None:
        line = Polyline(
            points=(
                Vec3(0.0, 0.0, 20.0),
                Vec3(0.0, 0.0, -20.0),
                Vec3(5.0, 0.0, -20.0),
                Vec3(5.0, 0.0, 20.0),
            ),
            kind=StrokeKind.OTHER_PATH,
        )
        strokes = [
            s
            for s in project_scene(_camera(), _FOV, [line])
            if s.kind is not StrokeKind.FRAME_CENTRE
        ]
        assert len(strokes) == 2

    def test_a_line_entirely_in_front_survives_whole(self) -> None:
        points = tuple(Vec3(0.0, 0.0, 10.0 + index) for index in range(5))
        strokes = [
            s
            for s in project_scene(_camera(), _FOV, [Polyline(points, StrokeKind.SUBJECT_PATH)])
            if s.kind is not StrokeKind.FRAME_CENTRE
        ]
        assert len(strokes) == 1
        assert len(strokes[0].points) == len(points)


class TestScene:
    def test_every_car_gets_a_path_and_a_body(self) -> None:
        scene = build_scene(_tracks(3), moment=1.0, subject=0)
        paths = [
            line for line in scene if line.kind in (StrokeKind.SUBJECT_PATH, StrokeKind.OTHER_PATH)
        ]
        boxes = [
            line for line in scene if line.kind in (StrokeKind.SUBJECT_CAR, StrokeKind.OTHER_CAR)
        ]
        assert len(paths) == 3
        assert len(boxes) == 3 * _CAR_BOX_LINES

    def test_the_subject_is_distinguished_from_the_rest(self) -> None:
        scene = build_scene(_tracks(3), moment=1.0, subject=1)
        subject_paths = [line for line in scene if line.kind is StrokeKind.SUBJECT_PATH]
        assert len(subject_paths) == 1

    def test_the_horizon_is_optional(self) -> None:
        without = build_scene(_tracks(1), moment=1.0, subject=0)
        with_ring = build_scene(_tracks(1), moment=1.0, subject=0, horizon_at=Vec3(0.0, 1.0, 0.0))
        assert not any(line.kind is StrokeKind.HORIZON for line in without)
        assert sum(line.kind is StrokeKind.HORIZON for line in with_ring) == 1

    def test_a_car_with_no_samples_is_skipped_rather_than_crashing(self) -> None:
        tracks = SubjectTracks(
            positions={0: np.zeros((0, 3)), 1: _tracks(1).positions[0]},
            headings={0: np.zeros(0), 1: np.zeros(_FRAMES)},
            time=np.arange(_FRAMES) / _RATE,
            sample_rate=_RATE,
        )
        scene = build_scene(tracks, moment=1.0, subject=1)
        assert scene


class TestBuildPreviews:
    @staticmethod
    def _timeline(tracks: SubjectTracks) -> Timeline:
        """A two-shot timeline whose cameras look at the car, not past it.

        Aimed from the tracks rather than at a fixed point: the preview draws the
        car where it is at the *middle of the shot*, so a camera pointed at some
        other stretch of the same path would fail a test that is meant to be
        about projection.

        Args:
            tracks: The cars' motion.

        Returns:
            The timeline.
        """
        shots = []
        for index in range(2):
            start, end = float(index), float(index) + 0.5
            frame = int(np.searchsorted(tracks.time, (start + end) * 0.5))
            at = tracks.positions[0][min(frame, tracks.positions[0].shape[0] - 1)]
            target = Vec3(float(at[0]), float(at[1]) + 0.5, float(at[2]))
            basis = CameraBasis.looking_at(Vec3(target.x + 18.0, target.y + 1.0, target.z), target)
            keyframes = tuple(
                CameraKeyframe(Seconds(index + frame * 0.1), basis, FovDegrees(40.0))
                for frame in range(5)
            )
            candidate = ShotCandidate(
                start=Seconds(start),
                end=Seconds(end),
                subjects=(0,),
                rig=RigKind.ROADSIDE,
                size=ShotSize.FULL,
                side=LineSide.LEFT,
                score=Score(0.5),
            )
            shots.append(
                Shot(
                    candidate=candidate,
                    path=CameraPath(keyframes=keyframes, spec=CameraSpec(rig=RigKind.ROADSIDE)),
                    index=index,
                )
            )
        session = SessionMeta(
            track_id="pk_akina",
            layout="akina_downhill",
            weather_id="3_clear",
            sample_rate=Hertz(_RATE),
            frame_count=_FRAMES,
            cars=(
                CarIdentity(0, "ks_toyota_ae86_tuned", "ducheck", "RUS", None, "09_panda_b", True),
            ),
            source_name="test",
        )
        return Timeline(shots=tuple(shots), cuts=(), session=session, style_name="initial_d")

    def test_one_preview_per_camera_in_the_same_order(self) -> None:
        tracks = _tracks(2)
        report = diagnose_cameras(self._timeline(tracks), tracks)
        previews = build_previews(report, tracks)
        assert [p.shot_index for p in previews] == [c.shot_index for c in report.cameras]

    def test_the_subject_is_drawn_near_the_frame_centre(self) -> None:
        # The camera looks straight at the car, so a projection that agreed with
        # the checks must put it in the middle. This is the test that would fail
        # if the preview and the report ever disagreed about where things are.
        tracks = _tracks(2)
        report = diagnose_cameras(self._timeline(tracks), tracks)
        for preview in build_previews(report, tracks):
            points = [
                point
                for stroke in preview.strokes
                if stroke.kind is StrokeKind.SUBJECT_CAR
                for point in stroke.points
            ]
            assert points, "the subject must be drawn"
            centre_u = sum(u for u, _ in points) / len(points)
            centre_v = sum(v for _, v in points) / len(points)
            assert abs(centre_u) < 0.5
            assert abs(centre_v) < 0.5

    def test_every_preview_carries_a_label_and_both_verdicts(self) -> None:
        tracks = _tracks(2)
        report = diagnose_cameras(self._timeline(tracks), tracks)
        for preview in build_previews(report, tracks):
            assert preview.label
            assert preview.status is not None
            assert preview.measured_status is not None

    def test_a_crosshair_marks_where_the_camera_actually_points(self) -> None:
        tracks = _tracks(1)
        report = diagnose_cameras(self._timeline(tracks), tracks)
        for preview in build_previews(report, tracks):
            centres = [s for s in preview.strokes if s.kind is StrokeKind.FRAME_CENTRE]
            assert len(centres) == 1
            assert centres[0].points == ((0.0, 0.0),)


class TestProjectionAgreesWithTheChecks:
    def test_a_camera_facing_away_draws_nothing_of_its_subject(self) -> None:
        # Standing well clear of the car and looking the other way. Placing the
        # camera on top of it instead would put half the box in front of the
        # lens and prove nothing.
        tracks = _tracks(1)
        scene = build_scene(tracks, moment=0.0, subject=0)
        away = CameraBasis.looking_at(Vec3(0.0, 1.0, 40.0), Vec3(0.0, 1.0, 140.0))
        strokes = project_scene(away, _FOV, scene)
        subject = [s for s in strokes if s.kind is StrokeKind.SUBJECT_CAR]
        assert not subject, "a car behind the camera must not be drawn at all"


def test_a_wider_frame_pulls_horizontal_coordinates_in() -> None:
    # Aspect widens the frame without moving the camera, so a fixed point off to
    # one side sits proportionally nearer the middle of a wider one.
    line = Polyline(points=(Vec3(5.0, 0.0, 10.0), Vec3(5.0, 1.0, 10.0)), kind=StrokeKind.OTHER_PATH)

    def horizontal(aspect: float) -> float:
        strokes = [
            stroke
            for stroke in project_scene(_camera(), _FOV, [line], aspect=aspect)
            if stroke.kind is not StrokeKind.FRAME_CENTRE
        ]
        assert strokes
        return abs(strokes[0].points[0][0])

    square = horizontal(1.0)
    wide = horizontal(16.0 / 9.0)
    ultrawide = horizontal(21.0 / 9.0)
    assert square > wide > ultrawide
    assert wide == pytest.approx(square * 9.0 / 16.0)
