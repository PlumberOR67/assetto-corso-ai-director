"""Tests for timeline export and the JSON round trip.

The round trip is the load-bearing property: a timeline is meant to be a
document a person can edit and re-export, not a transient pipeline artefact. If
anything is lost on the way through the file, that promise is broken.
"""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from assetto_ai_director.application.camera.spline import distance_profile
from assetto_ai_director.domain.camera import (
    CameraKeyframe,
    CameraPath,
    CameraSpec,
    RigKind,
)
from assetto_ai_director.domain.events import Event, EventKind
from assetto_ai_director.domain.geometry import CameraBasis, Vec3
from assetto_ai_director.domain.identity import CarIdentity
from assetto_ai_director.domain.session import SessionMeta
from assetto_ai_director.domain.shot import LineSide, Shot, ShotCandidate, ShotSize
from assetto_ai_director.domain.timeline import Cut, Timeline, TransitionKind
from assetto_ai_director.domain.units import FovDegrees, Hertz, Score, Seconds
from assetto_ai_director.infrastructure.export.cameras_ini import (
    CamerasIniSink,
    planned_spans,
    suggested_path,
    write_cameras_ini,
)
from assetto_ai_director.infrastructure.export.json_timeline import (
    JsonTimelineSink,
    write_timeline,
)
from assetto_ai_director.infrastructure.timeline_io import read_timeline


def _path(start: float, end: float, *, zooms: bool = True) -> CameraPath:
    spec = CameraSpec(
        rig=RigKind.ROADSIDE,
        fov=FovDegrees(38.0),
        fov_near=FovDegrees(52.0) if zooms else None,
        fov_far=FovDegrees(22.0) if zooms else None,
    )
    basis = CameraBasis.looking_at(Vec3(3.0, 0.6, 1.0), Vec3(0.0, 0.0, 10.0))
    return CameraPath(
        keyframes=(
            CameraKeyframe(Seconds(start), basis, FovDegrees(40.0), own_share=0.5),
            CameraKeyframe(Seconds((start + end) / 2), basis, FovDegrees(35.0)),
            CameraKeyframe(Seconds(end), basis, FovDegrees(30.0), own_share=0.5),
        ),
        spec=spec,
    )


def _shot(index: int, start: float, end: float, rig: RigKind = RigKind.ROADSIDE) -> Shot:
    anchor = Event(
        EventKind.DRIFT_START,
        Seconds(start),
        Seconds(end),
        (0,),
        peak=Seconds((start + end) / 2),
        detail={"peak_angle_deg": 28.5},
    )
    candidate = ShotCandidate(
        start=Seconds(start),
        end=Seconds(end),
        subjects=(0,),
        rig=rig,
        size=ShotSize.FULL,
        side=LineSide.LEFT,
        score=Score(0.65),
        anchor=anchor,
    )
    return Shot(candidate=candidate, path=_path(start, end), index=index)


def _timeline(rigs: list[RigKind] | None = None) -> Timeline:
    rigs = rigs or [RigKind.ROADSIDE, RigKind.FOLLOW, RigKind.ROADSIDE, RigKind.STATIC]
    shots = tuple(
        _shot(index, float(index) * 3.0, float(index + 1) * 3.0, rig)
        for index, rig in enumerate(rigs)
    )
    cuts = tuple(
        Cut(
            time=later.start,
            kind=TransitionKind.CUT,
            from_index=earlier.index,
            to_index=later.index,
        )
        for earlier, later in pairwise(shots)
    )
    session = SessionMeta(
        track_id="pk_akina",
        layout="akina_downhill",
        weather_id="7_heavy_clouds",
        sample_rate=Hertz(33.3),
        frame_count=1000,
        cars=(CarIdentity(0, "ks_toyota_ae86_tuned", "ducheck", "RUS", None, "09_panda_b", True),),
        source_name="test",
    )
    return Timeline(
        shots=shots,
        cuts=cuts,
        session=session,
        style_name="initial_d",
        notes=("a note that must survive",),
    )


class TestJsonRoundTrip:
    def test_written_file_is_valid_json(self, tmp_path: Path) -> None:
        destination = tmp_path / "t.json"
        write_timeline(_timeline(), destination)
        assert json.loads(destination.read_text(encoding="utf-8"))["style"] == "initial_d"

    def test_shots_survive(self, tmp_path: Path) -> None:
        original = _timeline()
        destination = tmp_path / "t.json"
        write_timeline(original, destination)
        restored = read_timeline(destination)
        assert restored.shot_count == original.shot_count
        for before, after in zip(original.shots, restored.shots, strict=True):
            assert float(after.start) == pytest.approx(float(before.start))
            assert float(after.duration) == pytest.approx(float(before.duration))
            assert after.candidate.rig is before.candidate.rig
            assert after.candidate.size is before.candidate.size
            assert after.candidate.side is before.candidate.side

    def test_zoom_tracking_survives(self, tmp_path: Path) -> None:
        # Dropped by an earlier schema, which silently turned every trackside
        # camera into a fixed lens on export to cameras.ini.
        destination = tmp_path / "t.json"
        write_timeline(_timeline(), destination)
        restored = read_timeline(destination)
        spec = restored.shots[0].path.spec
        assert spec.zooms
        assert float(spec.fov_near or 0) != float(spec.fov_far or 0)

    def test_camera_geometry_survives(self, tmp_path: Path) -> None:
        original = _timeline()
        destination = tmp_path / "t.json"
        write_timeline(original, destination)
        restored = read_timeline(destination)
        before = original.shots[0].path.keyframes[0].basis
        after = restored.shots[0].path.keyframes[0].basis
        assert after.position.x == pytest.approx(before.position.x, abs=1e-3)
        assert after.look.z == pytest.approx(before.look.z, abs=1e-3)

    def test_notes_survive(self, tmp_path: Path) -> None:
        # The blind spots must travel with the file, not be inferred from its
        # confident appearance.
        destination = tmp_path / "t.json"
        write_timeline(_timeline(), destination)
        assert "a note that must survive" in read_timeline(destination).notes

    def test_shots_record_why_they_were_chosen(self, tmp_path: Path) -> None:
        destination = tmp_path / "t.json"
        write_timeline(_timeline(), destination)
        document = json.loads(destination.read_text(encoding="utf-8"))
        assert document["shots"][0]["because"]["event"] == "drift_start"

    def test_unknown_schema_is_refused(self, tmp_path: Path) -> None:
        destination = tmp_path / "t.json"
        write_timeline(_timeline(), destination)
        document = json.loads(destination.read_text(encoding="utf-8"))
        document["schema_version"] = 99
        destination.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported timeline schema"):
            read_timeline(destination)

    def test_truncated_vector_is_refused(self, tmp_path: Path) -> None:
        destination = tmp_path / "t.json"
        write_timeline(_timeline(), destination)
        document = json.loads(destination.read_text(encoding="utf-8"))
        document["shots"][0]["camera"]["keyframes"][0]["position"] = [1.0, 2.0]
        destination.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(ValueError, match="three vector components"):
            read_timeline(destination)


class TestJsonSupport:
    def test_json_is_lossless(self) -> None:
        report = JsonTimelineSink().supports(_timeline())
        assert report.supported
        assert not report.lossy


class TestCamerasIni:
    def test_reports_what_it_cannot_carry(self) -> None:
        report = CamerasIniSink().supports(_timeline())
        assert report.supported
        assert report.lossy
        assert report.dropped_shots == 1
        assert any("position, not on time" in note for note in report.limitations)

    def test_refuses_a_timeline_with_no_fixed_camera(self) -> None:
        timeline = _timeline([RigKind.DRONE, RigKind.FOLLOW, RigKind.BONNET])
        report = CamerasIniSink().supports(timeline)
        assert not report.supported

    def test_writes_a_parsable_header(self, tmp_path: Path) -> None:
        destination = tmp_path / "cameras_9.ini"
        write_cameras_ini(_timeline(), destination)
        text = destination.read_text(encoding="utf-8")
        assert "[HEADER]" in text
        assert "VERSION=3" in text
        # Three of the four shots use fixed rigs.
        assert "CAMERA_COUNT=3" in text

    def test_spans_are_contiguous_and_cover_the_track(self, tmp_path: Path) -> None:
        # The game selects by position; a gap would leave a stretch of track
        # with no camera assigned.
        destination = tmp_path / "c.ini"
        write_cameras_ini(_timeline(), destination)
        text = destination.read_text(encoding="utf-8")
        ins = [float(line.split("=")[1]) for line in text.splitlines() if line.startswith("IN_")]
        outs = [float(line.split("=")[1]) for line in text.splitlines() if line.startswith("OUT_")]
        assert ins[0] == pytest.approx(0.0)
        assert outs[-1] == pytest.approx(1.0)
        for out, next_in in zip(outs, ins[1:], strict=False):
            assert out == pytest.approx(next_in)

    def test_min_fov_is_the_far_value_not_the_smaller_one(self, tmp_path: Path) -> None:
        # MIN_FOV applies when the car is furthest, so it is routinely the
        # numerically larger. Writing them the intuitive way round makes the
        # camera zoom in as the car approaches, which is backwards.
        destination = tmp_path / "c.ini"
        write_cameras_ini(_timeline(), destination)
        text = destination.read_text(encoding="utf-8")
        lines = text.splitlines()
        min_fov = float(next(x for x in lines if x.startswith("MIN_FOV")).split("=")[1])
        max_fov = float(next(x for x in lines if x.startswith("MAX_FOV")).split("=")[1])
        assert min_fov == pytest.approx(22.0)
        assert max_fov == pytest.approx(52.0)

    def test_refuses_to_write_without_a_fixed_camera(self, tmp_path: Path) -> None:
        timeline = _timeline([RigKind.DRONE, RigKind.FOLLOW])
        with pytest.raises(ValueError, match="no fixed trackside camera"):
            write_cameras_ini(timeline, tmp_path / "c.ini")

    def test_path_is_layout_scoped(self) -> None:
        # A multi-layout track keeps cameras under the layout, not the track
        # root; the wrong one produces a file the game never loads.
        path = suggested_path(Path("content/tracks/pk_akina"), _timeline(), set_index=9)
        assert path.parts[-3:] == ("akina_downhill", "data", "cameras_9.ini")

    def test_index_zero_writes_the_default_set_name(self) -> None:
        path = suggested_path(Path("t"), _timeline(), set_index=0)
        assert path.name == "cameras.ini"


class TestPlannedSpans:
    """The spans the diagnostic checks are the spans the exporter writes.

    Asking the exporter rather than recomputing is the whole point: a
    reimplementation would keep agreeing right up until the export policy
    changed, and then the check would be measuring the copy.
    """

    def test_reports_a_span_for_every_shot_the_file_keeps(self) -> None:
        spans = planned_spans(_timeline())
        # FOLLOW at index 1 is a moving rig and never reaches the file.
        assert sorted(spans) == [0, 2, 3]

    def test_spans_are_contiguous_and_cover_the_track(self) -> None:
        spans = planned_spans(_timeline())
        ordered = [spans[index] for index in sorted(spans)]
        assert ordered[0][0] == pytest.approx(0.0)
        assert ordered[-1][1] == pytest.approx(1.0)
        for earlier, later in pairwise(ordered):
            assert earlier[1] == pytest.approx(later[0])

    def test_matches_what_the_written_file_contains(self, tmp_path: Path) -> None:
        timeline = _timeline()
        destination = tmp_path / "cameras_9.ini"
        write_cameras_ini(timeline, destination)

        written = [
            float(line.split("=", 1)[1])
            for line in destination.read_text(encoding="utf-8").splitlines()
            if line.startswith("IN_POINT=")
        ]
        spans = planned_spans(timeline)
        expected = [spans[index][0] for index in sorted(spans)]
        assert written == pytest.approx(expected, abs=1e-5)

    def test_an_unexportable_timeline_plans_nothing(self) -> None:
        assert planned_spans(_timeline([RigKind.DRONE, RigKind.FOLLOW])) == {}


class TestDistanceProfile:
    # The estimator moved to application/camera/spline.py so the camera
    # diagnostic and this exporter share one implementation rather than
    # drifting apart. The behaviour it pins is unchanged.
    def test_point_to_point_run_spans_zero_to_one(self) -> None:
        position = np.stack([np.zeros(500), np.zeros(500), np.linspace(0.0, 2000.0, 500)], axis=1)
        fraction = distance_profile(position).fraction
        assert fraction[0] == pytest.approx(0.0)
        assert fraction[-1] < 1.0 + 1e-9
        assert np.all(np.diff(fraction) >= -1e-9)

    def test_a_closed_lap_wraps(self) -> None:
        angle = np.linspace(0.0, 2 * np.pi, 600)
        radius = 200.0
        position = np.stack([radius * np.cos(angle), np.zeros(600), radius * np.sin(angle)], axis=1)
        fraction = distance_profile(position).fraction
        assert float(fraction.max()) < 1.0 + 1e-9
        assert float(fraction[len(fraction) // 2]) > 0.3

    def test_stationary_car_is_safe(self) -> None:
        assert float(distance_profile(np.zeros((10, 3))).fraction.max()) == 0.0
