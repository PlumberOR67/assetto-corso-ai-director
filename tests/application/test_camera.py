"""Tests for camera placement, smoothing and path generation."""

from __future__ import annotations

import numpy as np
import pytest

from assetto_ai_director.application.camera.damping import (
    clamp_speed,
    smooth_angles,
    smooth_track,
)
from assetto_ai_director.application.camera.paths import SubjectTracks, build_camera_path
from assetto_ai_director.application.camera.rigs import (
    SIZE_DISTANCE_METRES,
    place_camera,
    toric_position,
)
from assetto_ai_director.application.director.styles import BUILTIN_STYLES
from assetto_ai_director.domain.camera import RigKind
from assetto_ai_director.domain.shot import LineSide, ShotCandidate, ShotSize
from assetto_ai_director.domain.units import Score, Seconds

RATE = 33.3
DT = 1.0 / RATE


def _straight(count: int = 200, speed: float = 30.0) -> tuple[np.ndarray, np.ndarray]:
    z = np.arange(count) * speed / RATE
    position = np.stack([np.zeros(count), np.zeros(count), z], axis=1)
    return position, np.zeros(count)


class TestDamping:
    def test_smoothing_is_frame_rate_independent(self) -> None:
        """The whole reason for the exponential form.

        The naive ``x += (target - x) * k`` smooths twice as hard at 66 Hz as at
        33 Hz, so the same edit would move differently depending on the replay's
        recording quality — for reasons unconnected to the driving.
        """
        seconds = 4.0
        results = []
        for rate in (33.0, 66.0, 132.0):
            count = int(seconds * rate)
            step = np.zeros((count, 1))
            step[count // 2 :] = 1.0
            smoothed = smooth_track(step, 1.0 / rate, tau=0.5)
            # Sample at the same wall-clock instant, not the same index.
            results.append(float(smoothed[int(0.75 * count), 0]))
        assert max(results) - min(results) < 0.05

    def test_smoothing_reduces_variation(self) -> None:
        generator = np.random.default_rng(3)
        noisy = generator.standard_normal((400, 1))
        smoothed = smooth_track(noisy, DT, tau=0.5)
        assert smoothed.std() < noisy.std()

    def test_smoothing_has_no_phase_lag(self) -> None:
        # Applied forward and backward. A one-directional follow would shift the
        # ramp's midpoint later, so every camera would reach every corner late.
        ramp = np.linspace(0.0, 1.0, 400).reshape(-1, 1)
        smoothed = smooth_track(ramp, DT, tau=0.4)
        midpoint = int(np.argmin(np.abs(smoothed - 0.5)))
        assert abs(midpoint - 200) < 12

    def test_zero_tau_is_a_passthrough(self) -> None:
        values = np.linspace(0.0, 1.0, 50).reshape(-1, 1)
        assert np.allclose(smooth_track(values, DT, tau=0.0), values)

    def test_angles_take_the_short_way_round(self) -> None:
        # Interpolating 179 to -179 as scalars sweeps the whole circle.
        angles = np.concatenate([np.full(50, 3.10), np.full(50, -3.10)])
        smoothed = smooth_angles(angles, DT, tau=0.2)
        assert float(np.abs(smoothed).min()) > 2.5

    def test_speed_clamp_limits_travel(self) -> None:
        path = np.stack([np.arange(100) * 10.0, np.zeros(100), np.zeros(100)], axis=1)
        clamped = clamp_speed(path, DT, max_speed=50.0)
        steps = np.linalg.norm(np.diff(clamped, axis=0), axis=1)
        assert float(steps.max()) <= 50.0 * DT + 1e-6

    def test_speed_clamp_leaves_slow_paths_alone(self) -> None:
        path = np.stack([np.arange(100) * 0.1, np.zeros(100), np.zeros(100)], axis=1)
        assert np.allclose(clamp_speed(path, DT, max_speed=50.0), path)


class TestRigPlacement:
    @pytest.mark.parametrize("rig", list(RigKind))
    def test_every_rig_produces_a_usable_placement(self, rig: RigKind) -> None:
        position, heading = _straight()
        placement = place_camera(
            rig, ShotSize.FULL, LineSide.LEFT, position, heading=heading, companion=None, dt=DT
        )
        assert placement.positions.shape == position.shape
        assert placement.targets.shape == position.shape
        assert np.isfinite(placement.positions).all()

    def test_trackside_camera_does_not_move(self) -> None:
        position, heading = _straight()
        placement = place_camera(
            RigKind.ROADSIDE,
            ShotSize.FULL,
            LineSide.LEFT,
            position,
            heading=heading,
            companion=None,
            dt=DT,
        )
        travel = np.linalg.norm(np.diff(placement.positions, axis=0), axis=1)
        assert float(travel.max()) < 1e-9

    def test_sides_place_the_camera_opposite_each_other(self) -> None:
        position, heading = _straight()
        left = place_camera(
            RigKind.ROADSIDE,
            ShotSize.FULL,
            LineSide.LEFT,
            position,
            heading=heading,
            companion=None,
            dt=DT,
        )
        right = place_camera(
            RigKind.ROADSIDE,
            ShotSize.FULL,
            LineSide.RIGHT,
            position,
            heading=heading,
            companion=None,
            dt=DT,
        )
        assert np.sign(left.positions[0][0]) != np.sign(right.positions[0][0])

    def test_mounted_camera_travels_with_the_car(self) -> None:
        position, heading = _straight()
        placement = place_camera(
            RigKind.BONNET,
            ShotSize.MEDIUM,
            LineSide.NEUTRAL,
            position,
            heading=heading,
            companion=None,
            dt=DT,
        )
        offset = placement.positions - position
        assert float(offset.std(axis=0).max()) < 1e-9

    def test_tighter_sizes_place_the_camera_closer(self) -> None:
        assert SIZE_DISTANCE_METRES[ShotSize.CLOSE_UP] < SIZE_DISTANCE_METRES[ShotSize.LONG]

    def test_aerial_rigs_sit_above_the_car(self) -> None:
        position, heading = _straight()
        placement = place_camera(
            RigKind.HELICOPTER,
            ShotSize.LONG,
            LineSide.LEFT,
            position,
            heading=heading,
            companion=None,
            dt=DT,
        )
        assert float((placement.positions[:, 1] - position[:, 1]).min()) > 10.0


class TestToricFraming:
    def test_camera_sees_both_cars_from_outside_the_pair(self) -> None:
        count = 100
        first = np.stack([np.zeros(count), np.zeros(count), np.arange(count) * 0.5], axis=1)
        second = first + np.array([0.0, 0.0, 10.0])
        camera = toric_position(first, second, theta=0.5, phi=0.25, distance=30.0)
        midpoint = (first + second) * 0.5
        to_first = np.linalg.norm(camera - first, axis=1)
        to_second = np.linalg.norm(camera - second, axis=1)
        to_mid = np.linalg.norm(camera - midpoint, axis=1)
        # Standing off the pair rather than between them.
        assert float(to_mid.min()) > 5.0
        assert float(np.abs(to_first - to_second).max()) < 40.0

    def test_coincident_cars_do_not_make_the_camera_spin(self) -> None:
        count = 50
        first = np.zeros((count, 3))
        camera = toric_position(first, first.copy(), theta=0.0, phi=0.0, distance=20.0)
        assert np.isfinite(camera).all()
        assert float(np.linalg.norm(np.diff(camera, axis=0), axis=1).max()) < 1e-6

    def test_elevation_raises_the_camera(self) -> None:
        count = 20
        first = np.zeros((count, 3))
        second = first + np.array([0.0, 0.0, 8.0])
        high = toric_position(first, second, theta=0.0, phi=0.9, distance=20.0)
        level = toric_position(first, second, theta=0.0, phi=0.0, distance=20.0)
        assert float(high[:, 1].mean()) > float(level[:, 1].mean())


def _tracks(count: int = 200) -> SubjectTracks:
    position, heading = _straight(count)
    return SubjectTracks(
        positions={0: position, 1: position + np.array([4.0, 0.0, -8.0])},
        headings={0: heading, 1: heading},
        time=np.arange(count) / RATE,
        sample_rate=RATE,
    )


def _shot(
    rig: RigKind = RigKind.ROADSIDE,
    size: ShotSize = ShotSize.FULL,
    subjects: tuple[int, ...] = (0,),
    start: float = 0.5,
    end: float = 3.5,
) -> ShotCandidate:
    return ShotCandidate(
        start=Seconds(start),
        end=Seconds(end),
        subjects=subjects,
        rig=rig,
        size=size,
        side=LineSide.LEFT,
        score=Score(0.6),
    )


class TestCameraPaths:
    @pytest.mark.parametrize("rig", list(RigKind))
    def test_every_rig_yields_a_valid_path(self, rig: RigKind) -> None:
        path = build_camera_path(_shot(rig=rig), _tracks(), BUILTIN_STYLES["mf_ghost"])
        assert path.keyframes
        assert all(np.isfinite([k.basis.position.x for k in path.keyframes]))

    def test_keyframes_are_ordered_and_cover_the_shot(self) -> None:
        shot = _shot(start=1.0, end=4.0)
        path = build_camera_path(shot, _tracks(), BUILTIN_STYLES["initial_d"])
        times = [float(k.time) for k in path.keyframes]
        assert times == sorted(times)
        assert times[0] >= 1.0 - 0.1
        assert times[-1] <= 4.0 + 0.1

    def test_look_is_a_unit_direction_not_a_target(self) -> None:
        # CSP's GrabbedCamera consumes a direction; a target cannot be recovered
        # from one without a distance nothing records.
        path = build_camera_path(_shot(), _tracks(), BUILTIN_STYLES["initial_d"])
        for keyframe in path.keyframes:
            assert keyframe.basis.look.length == pytest.approx(1.0, abs=1e-6)

    def test_basis_is_orthonormal(self) -> None:
        path = build_camera_path(_shot(), _tracks(), BUILTIN_STYLES["initial_d"])
        for keyframe in path.keyframes:
            assert keyframe.basis.look.dot(keyframe.basis.up) == pytest.approx(0.0, abs=1e-6)

    def test_ownership_ramps_at_both_ends(self) -> None:
        # A hard takeover snaps; the ramp reads as a cut.
        path = build_camera_path(
            _shot(start=0.5, end=5.0), _tracks(300), BUILTIN_STYLES["initial_d"]
        )
        shares = [k.own_share for k in path.keyframes]
        assert shares[0] < 1.0
        assert shares[-1] < 1.0
        assert max(shares) == pytest.approx(1.0)

    def test_long_shots_are_subsampled(self) -> None:
        from assetto_ai_director.application.camera.paths import MAX_KEYFRAMES  # noqa: PLC0415

        path = build_camera_path(
            _shot(start=0.0, end=5.9), _tracks(400), BUILTIN_STYLES["initial_d"]
        )
        assert len(path.keyframes) <= MAX_KEYFRAMES + 1

    def test_drone_speed_is_clamped(self) -> None:
        # The one verified style constant: an MF Ghost drone caps at 180 km/h.
        tracks = _tracks(300)
        path = build_camera_path(
            _shot(rig=RigKind.DRONE, start=0.5, end=6.0), tracks, BUILTIN_STYLES["mf_ghost"]
        )
        positions = np.array(
            [[k.basis.position.x, k.basis.position.y, k.basis.position.z] for k in path.keyframes]
        )
        times = np.array([float(k.time) for k in path.keyframes])
        steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        gaps = np.diff(times)
        speeds = steps / np.maximum(gaps, 1e-6)
        assert float(speeds.max()) < 60.0

    def test_two_subject_shot_frames_between_the_cars(self) -> None:
        shot = _shot(subjects=(0, 1), size=ShotSize.LONG)
        path = build_camera_path(shot, _tracks(), BUILTIN_STYLES["broadcast"])
        assert path.keyframes
        assert all(np.isfinite([k.fov for k in path.keyframes]))

    def test_fixed_rigs_zoom_with_distance(self) -> None:
        # A car approaching from 60 m and passing at 8 m would otherwise be a
        # speck that becomes a blur.
        path = build_camera_path(
            _shot(rig=RigKind.ROADSIDE, start=0.5, end=5.0),
            _tracks(300),
            BUILTIN_STYLES["broadcast"],
        )
        fovs = [float(k.fov) for k in path.keyframes]
        assert max(fovs) - min(fovs) > 1.0
