"""Turning a selected shot into a sampled camera path.

Places the rig, smooths its motion, clamps it to what its notional vehicle could
do, and hands back keyframes.

**Field of view tracks distance for the rigs that hold position.** A locked-off
trackside camera watching a car approach from 60 m and pass at 8 m would
otherwise show a speck that becomes a blur. Widening the lens as the car closes
holds its apparent size roughly constant, which is how the game's own trackside
cameras behave and how television covers motorsport.

**Shake comes from the road, not from noise.** Amplitude is driven by the
subject's vertical acceleration, so the camera trembles where the surface is
actually rough and settles where it is smooth. Procedural noise trembles
uniformly and reads as a filter; load-driven shake reads as a place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from assetto_ai_director.application.camera.damping import (
    clamp_speed,
    smooth_track,
)
from assetto_ai_director.application.camera.rigs import (
    RIG_HEIGHT_METRES,
    SIZE_DISTANCE_METRES,
    place_camera,
)
from assetto_ai_director.domain.camera import (
    CameraKeyframe,
    CameraPath,
    CameraSpec,
    RigKind,
    ShakeProfile,
)
from assetto_ai_director.domain.geometry import CameraBasis, Vec3
from assetto_ai_director.domain.units import FovDegrees, Metres, MetresPerSecond, Seconds

if TYPE_CHECKING:
    from collections.abc import Mapping

    from assetto_ai_director.application.events.detect import CarSignals
    from assetto_ai_director.domain.shot import ShotCandidate
    from assetto_ai_director.domain.style import StyleProfile
    from assetto_ai_director.domain.track import TrackSpline

__all__ = ["SubjectTracks", "build_camera_path"]

MAX_KEYFRAMES = 240
"""Most keyframes emitted for one shot.

A shot is a few seconds; sampling every frame of a 67 Hz replay would emit
hundreds of near-identical keyframes for no visible gain and bloat the exported
timeline. Long shots are subsampled to this ceiling.
"""

BLEND_SECONDS = 0.25
"""How long a shot spends handing control to and from the game's own camera.

Expressed through the ownership share the renderer blends on. A hard takeover
snaps; a quarter second reads as a cut rather than a glitch.
"""

NEAR_FOV = FovDegrees(52.0)
FAR_FOV = FovDegrees(22.0)
"""Field of view at the closest and furthest tracked distance, vertical degrees.

Wider when the car is close, narrower when far, so its apparent size stays put.
"""

SHAKE_REFERENCE_G = 6.0
"""Vertical acceleration, m/s^2, that produces full shake amplitude."""

_MIN_SAMPLES_FOR_SHAKE = 3
"""Frames needed before a second derivative of elevation exists."""


class SubjectTracks:
    """The car motion a shot needs in order to place a camera.

    Attributes:
        positions: World positions per car id, each shape ``(n, 3)``.
        headings: Headings per car id, each shape ``(n,)``.
        time: Time per frame, seconds from session start.
        sample_rate: Frames per second.
        track: The road the cars are on, when it could be read. Carried here
            rather than passed separately because every rig that needs it is
            already being handed this object, and a placement that knows where
            the road ends is the difference between a camera beside it and a
            camera in the trees.
    """

    __slots__ = ("headings", "positions", "sample_rate", "time", "track")

    def __init__(
        self,
        positions: dict[int, np.ndarray],
        headings: dict[int, np.ndarray],
        time: np.ndarray,
        sample_rate: float,
        track: TrackSpline | None = None,
    ) -> None:
        """Store the tracks.

        Args:
            positions: World positions per car id.
            headings: Headings per car id.
            time: Time per frame.
            sample_rate: Frames per second.
            track: The track's centreline, or ``None``.
        """
        self.positions = positions
        self.headings = headings
        self.time = time
        self.sample_rate = sample_rate
        self.track = track

    @classmethod
    def from_signals(
        cls,
        signals: Mapping[int, CarSignals],
        time: np.ndarray,
        sample_rate: float,
        track: TrackSpline | None = None,
    ) -> SubjectTracks:
        """Build tracks from decoded per-car signals.

        The director already does this inline. It is a classmethod because the
        camera diagnostic needs the same object from the same inputs, and three
        callers each writing the same dictionary comprehension is three places
        for the two to drift apart.

        Args:
            signals: Decoded per-car signals.
            time: Time per frame, seconds from session start.
            sample_rate: Frames per second.
            track: The track's centreline, or ``None``.

        Returns:
            The tracks.
        """
        return cls(
            positions={car_id: car.position for car_id, car in signals.items()},
            headings={car_id: car.heading for car_id, car in signals.items()},
            time=time,
            sample_rate=sample_rate,
            track=track,
        )

    def window(self, start: float, end: float) -> tuple[int, int]:
        """Find the frame range covering a span.

        Args:
            start: Span start, seconds.
            end: Span end, seconds.

        Returns:
            ``(first, last)`` indices, end-exclusive, always at least one frame.
        """
        first = int(np.searchsorted(self.time, start))
        last = int(np.searchsorted(self.time, end))
        first = max(0, min(first, self.time.shape[0] - 1))
        last = max(first + 1, min(last, self.time.shape[0]))
        return first, last


def build_camera_path(
    shot: ShotCandidate, tracks: SubjectTracks, style: StyleProfile
) -> CameraPath:
    """Generate the camera motion for one selected shot.

    Args:
        shot: The chosen shot.
        tracks: The cars' motion.
        style: The style profile, supplying smoothing and speed limits.

    Returns:
        The sampled camera path.
    """
    first, last = tracks.window(float(shot.start), float(shot.end))
    primary = shot.primary_subject
    subject = tracks.positions[primary][first:last]
    heading = tracks.headings[primary][first:last]
    times = tracks.time[first:last]
    dt = 1.0 / tracks.sample_rate

    companion = None
    if shot.is_two_shot:
        other = shot.subjects[1]
        if other in tracks.positions:
            companion = tracks.positions[other][first:last]

    placement = place_camera(
        shot.rig,
        shot.size,
        shot.side,
        subject,
        heading=heading,
        companion=companion,
        dt=dt,
        track=tracks.track,
    )
    spec = _spec_for(shot, style)

    positions = placement.positions
    if not shot.rig.is_car_mounted:
        # A mounted camera *should* reproduce the car's motion exactly; anything
        # else should look operated.
        positions = smooth_track(positions, dt, float(spec.smoothing))
    if spec.max_speed is not None:
        positions = clamp_speed(positions, dt, float(spec.max_speed))

    positions = _apply_shake(positions, subject, spec.shake, tracks.sample_rate)
    targets = smooth_track(placement.targets, dt, float(spec.smoothing) * 0.5)

    keyframes = _sample(positions, targets, times, spec, shot)
    return CameraPath(keyframes=keyframes, spec=spec)


def _spec_for(shot: ShotCandidate, style: StyleProfile) -> CameraSpec:
    """Resolve the camera parameters for a shot.

    Args:
        shot: The chosen shot.
        style: The style profile. Not yet consulted — rig parameters are
            currently derived from the rig alone. The parameter is kept because
            per-style camera packs are the next thing to land here, and changing
            the signature later would ripple through every call site.

    Returns:
        The resolved specification.
    """
    del style
    distance = SIZE_DISTANCE_METRES[shot.size]
    zooms = shot.rig.is_fixed_position
    shake = _shake_for(shot.rig)
    smoothing = _smoothing_for(shot.rig)

    max_speed: MetresPerSecond | None = None
    if shot.rig is RigKind.DRONE:
        # The one style constant with a real source: an MF Ghost drone is
        # canonically capped at 180 km/h, which is 50.0 m/s exactly. A camera
        # that outruns its own diegetic aircraft breaks the reference.
        max_speed = MetresPerSecond(50.0)
    elif shot.rig is RigKind.HELICOPTER:
        max_speed = MetresPerSecond(80.0)

    return CameraSpec(
        rig=shot.rig,
        fov=FovDegrees(_fov_for(distance)),
        fov_near=NEAR_FOV if zooms else None,
        fov_far=FAR_FOV if zooms else None,
        height=Metres(RIG_HEIGHT_METRES.get(shot.rig, 1.2)),
        smoothing=Seconds(smoothing),
        shake=shake,
        max_speed=max_speed,
        dof_factor=0.35 if shot.size.is_tight else 0.0,
    )


def _smoothing_for(rig: RigKind) -> float:
    """Choose a smoothing time constant for a rig.

    Args:
        rig: The rig.

    Returns:
        The time constant in seconds.
    """
    if rig.is_car_mounted:
        return 0.0
    if rig.is_fixed_position:
        return 0.15
    if rig.is_free_flying:
        return 0.9
    return 0.4


def _shake_for(rig: RigKind) -> ShakeProfile:
    """Choose a shake profile for a rig.

    Args:
        rig: The rig.

    Returns:
        The profile. Trackside cameras are still; mounted ones inherit the
        suspension.
    """
    if rig.is_fixed_position:
        return ShakeProfile(amplitude=Metres(0.0), rotational=0.0)
    if rig.is_car_mounted:
        return ShakeProfile(amplitude=Metres(0.045), frequency=14.0, rotational=0.9)
    if rig.is_free_flying:
        return ShakeProfile(amplitude=Metres(0.02), frequency=4.0, rotational=0.2)
    return ShakeProfile(amplitude=Metres(0.03), frequency=9.0, rotational=0.4)


def _fov_for(distance: float) -> float:
    """Pick a nominal field of view for a standoff distance.

    Args:
        distance: Camera-to-subject distance, metres.

    Returns:
        Vertical field of view in degrees.
    """
    span = SIZE_DISTANCE_METRES[list(SIZE_DISTANCE_METRES)[-1]]
    fraction = min(1.0, max(0.0, distance / span))
    return float(NEAR_FOV + (FAR_FOV - NEAR_FOV) * fraction)


def _apply_shake(
    positions: np.ndarray, subject: np.ndarray, shake: ShakeProfile, sample_rate: float
) -> np.ndarray:
    """Add camera shake driven by how rough the road actually is.

    Args:
        positions: Camera positions, shape ``(n, 3)``.
        subject: Subject positions, used to measure vertical acceleration.
        shake: The shake profile.
        sample_rate: Frames per second.

    Returns:
        The shaken positions.
    """
    if shake.is_still or positions.shape[0] < _MIN_SAMPLES_FOR_SHAKE:
        return positions

    elevation = subject[:, 1]
    vertical_g = np.gradient(np.gradient(elevation)) * sample_rate**2
    intensity = np.clip(np.abs(vertical_g) / SHAKE_REFERENCE_G, 0.0, 1.0)
    if not shake.load_driven:
        intensity = np.ones_like(intensity)

    phase = np.arange(positions.shape[0]) / sample_rate * shake.frequency * 2.0 * np.pi
    displacement = np.stack([np.sin(phase * 1.13), np.sin(phase), np.sin(phase * 0.87)], axis=1) * (
        float(shake.amplitude) * intensity[:, None]
    )
    return np.asarray(positions + displacement, dtype=np.float64)


def _sample(
    positions: np.ndarray,
    targets: np.ndarray,
    times: np.ndarray,
    spec: CameraSpec,
    shot: ShotCandidate,
) -> tuple[CameraKeyframe, ...]:
    """Convert arrays into keyframes, subsampling long shots.

    Args:
        positions: Camera positions.
        targets: Aim points.
        times: Time per frame.
        spec: The camera specification.
        shot: The shot being generated, for the blend envelope.

    Returns:
        The keyframes, ordered by time.
    """
    count = positions.shape[0]
    step = max(1, count // MAX_KEYFRAMES)
    indices = list(range(0, count, step))
    if indices[-1] != count - 1:
        indices.append(count - 1)

    duration = max(float(shot.duration), 1e-6)
    near = SIZE_DISTANCE_METRES[shot.size] * 0.4
    far = SIZE_DISTANCE_METRES[shot.size] * 2.5

    keyframes: list[CameraKeyframe] = []
    for index in indices:
        position = Vec3(
            float(positions[index][0]), float(positions[index][1]), float(positions[index][2])
        )
        target = Vec3(float(targets[index][0]), float(targets[index][1]), float(targets[index][2]))
        basis = CameraBasis.looking_at(position, target)
        distance = basis.distance_to(target)

        elapsed = float(times[index]) - float(shot.start)
        keyframes.append(
            CameraKeyframe(
                time=Seconds(float(times[index])),
                basis=basis,
                fov=spec.fov_at(distance, Metres(near), Metres(far)),
                dof_distance=distance if spec.dof_factor > 0.0 else None,
                dof_factor=spec.dof_factor,
                own_share=_blend(elapsed, duration),
            )
        )
    return tuple(keyframes)


def _blend(elapsed: float, duration: float) -> float:
    """Compute how much of the frame this camera owns at a moment.

    Ramped at both ends so control is handed to and from the renderer's own
    camera smoothly instead of snapping.

    Args:
        elapsed: Seconds since the shot began.
        duration: The shot's length in seconds.

    Returns:
        Ownership share in ``[0, 1]``.
    """
    if duration <= 2.0 * BLEND_SECONDS:
        return 1.0
    if elapsed < BLEND_SECONDS:
        return max(0.0, elapsed / BLEND_SECONDS)
    remaining = duration - elapsed
    if remaining < BLEND_SECONDS:
        return max(0.0, remaining / BLEND_SECONDS)
    return 1.0
