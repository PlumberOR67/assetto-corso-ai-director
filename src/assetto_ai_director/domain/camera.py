"""Camera rigs, their tunable parameters, and the keyframed paths they produce.

A **rig** is a *kind* of camera behaviour — where it sits and how it moves in
relation to the car. A :class:`CameraSpec` is a rig with its parameters filled
in. A :class:`CameraPath` is the concrete, sampled result: the actual positions
and orientations, frame by frame.

Field of view is stored as **vertical degrees** throughout, because that is what
Custom Shaders Patch's ``ac.GrabbedCamera.fov`` is documented to be. Assetto
Corsa's ``cameras.ini`` uses ``MIN_FOV``/``MAX_FOV`` whose axis is undocumented,
so the exporter marks that conversion as an explicit assumption rather than
quietly treating the two as identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from itertools import pairwise

from assetto_ai_director.domain.geometry import CameraBasis, Vec3
from assetto_ai_director.domain.units import (
    FovDegrees,
    Metres,
    MetresPerSecond,
    Seconds,
)

__all__ = [
    "CameraKeyframe",
    "CameraPath",
    "CameraSpec",
    "RigKind",
    "ShakeProfile",
]

_MIN_FOV_DEGREES = 1.0
_MAX_FOV_DEGREES = 179.0
"""Bounds of a physically meaningful lens. Outside these a projection degenerates."""


class RigKind(StrEnum):
    """The camera behaviours this system can direct.

    Attributes:
        STATIC: Locked off. Does not move at all; the car passes through frame.
        ROADSIDE: Locked position at the trackside, but rotating to follow the
            car and zooming to hold its size. The trackside television grammar,
            and the one rig that maps cleanly onto ``cameras.ini``.
        FOLLOW: Chase camera trailing the car at a fixed offset.
        DRONE: Free-flying camera on a smooth path, unconstrained by the track.
        HELICOPTER: High, slow, wide. Establishing coverage.
        ORBITAL: Circles the car while tracking it.
        SPLINE: Travels along an authored spline, independent of the car.
        INTERIOR: Cockpit view from the driver's position.
        BONNET: Mounted on the bonnet, looking forward.
        ROOF: Mounted on the roof.
        WHEEL: Mounted at a wheel, looking at the tyre and the road surface.
            The shot that makes slip angle legible, because the contact patch is
            visible.
        SUSPENSION: Mounted low on the suspension, watching the geometry work.
    """

    STATIC = "static"
    ROADSIDE = "roadside"
    FOLLOW = "follow"
    DRONE = "drone"
    HELICOPTER = "helicopter"
    ORBITAL = "orbital"
    SPLINE = "spline"
    INTERIOR = "interior"
    BONNET = "bonnet"
    ROOF = "roof"
    WHEEL = "wheel"
    SUSPENSION = "suspension"

    @property
    def is_car_mounted(self) -> bool:
        """Whether the rig is rigidly attached to the car.

        Car-mounted rigs inherit the car's motion exactly, so their shake comes
        from the suspension rather than from an operator, and they can never lose
        the subject.

        Returns:
            ``True`` for rigs bolted to the vehicle.
        """
        return self in {
            RigKind.INTERIOR,
            RigKind.BONNET,
            RigKind.ROOF,
            RigKind.WHEEL,
            RigKind.SUSPENSION,
        }

    @property
    def is_free_flying(self) -> bool:
        """Whether the rig moves through space independently of the track.

        Free-flying rigs need velocity and acceleration clamps, or they produce
        moves no physical camera could make.

        Returns:
            ``True`` for rigs that fly.
        """
        return self in {RigKind.DRONE, RigKind.HELICOPTER, RigKind.ORBITAL}

    @property
    def is_fixed_position(self) -> bool:
        """Whether the rig's position is constant for the duration of a shot.

        Returns:
            ``True`` for rigs that do not translate.
        """
        return self in {RigKind.STATIC, RigKind.ROADSIDE}

    @property
    def can_frame_two_subjects(self) -> bool:
        """Whether the rig can hold two cars in frame at once.

        Only rigs with free placement can satisfy a two-target framing
        constraint; a bonnet camera sees whatever is in front of the bonnet.

        Returns:
            ``True`` for rigs eligible for Toric-space two-subject framing.
        """
        return self in {
            RigKind.ROADSIDE,
            RigKind.STATIC,
            RigKind.DRONE,
            RigKind.HELICOPTER,
            RigKind.ORBITAL,
            RigKind.FOLLOW,
        }


@dataclass(frozen=True, slots=True)
class ShakeProfile:
    """How much the camera trembles, and where the trembling comes from.

    Shake is derived from the car's own measured wheel loads rather than from
    procedural noise. The difference is audible, so to speak: noise-driven shake
    trembles uniformly, whereas load-driven shake lands exactly where the road is
    actually rough and goes quiet on smooth tarmac. The road becomes visible
    through the camera.

    Attributes:
        amplitude: Peak positional displacement in metres.
        frequency: Dominant frequency in hertz, when procedural.
        rotational: Peak angular displacement in degrees.
        load_driven: Whether amplitude is modulated by measured wheel load.
            ``False`` falls back to smooth procedural noise, used when the source
            does not provide load.
    """

    amplitude: Metres = Metres(0.0)
    frequency: float = 12.0
    rotational: float = 0.0
    load_driven: bool = True

    @property
    def is_still(self) -> bool:
        """Whether this profile produces no visible shake.

        Returns:
            ``True`` if both positional and rotational amplitudes are zero.
        """
        return self.amplitude <= 0.0 and self.rotational <= 0.0


@dataclass(frozen=True, slots=True)
class CameraSpec:
    """A rig with all its parameters resolved, ready to generate a path.

    Attributes:
        rig: Which behaviour.
        fov: Vertical field of view in degrees at the shot's nominal distance.
        fov_near: Field of view when the subject is closest. Roadside cameras
            zoom out as the car approaches to hold its apparent size; leaving
            this ``None`` disables zoom tracking.
        fov_far: Field of view when the subject is furthest.
        offset: Position offset from the rig's anchor, in metres. Interpretation
            depends on the rig: lateral/longitudinal from the car for mounted and
            follow rigs, from the racing line for trackside rigs.
        height: Height above the anchor, in metres. Separated from ``offset``
            because it is the single most stylistically loaded camera parameter —
            a knee-height camera and a chest-height camera of the same subject are
            different films.
        smoothing: Time constant of the critically-damped follow, in seconds.
            Zero snaps instantly; larger values lag and glide.
        shake: Shake profile.
        max_speed: Speed clamp for free-flying rigs, in m/s. A camera that
            outruns what its notional vehicle could do breaks the illusion the
            style depends on.
        max_acceleration: Acceleration clamp for free-flying rigs, in m/s^2.
        roll: Fixed roll angle in degrees. Dutch tilt.
        dof_factor: Depth-of-field strength in ``[0, 1]``. Requires the
            renderer's post-processing; ignored by exporters that cannot express it.
    """

    rig: RigKind
    fov: FovDegrees = FovDegrees(40.0)
    fov_near: FovDegrees | None = None
    fov_far: FovDegrees | None = None
    offset: Vec3 | None = None
    height: Metres = Metres(1.2)
    smoothing: Seconds = Seconds(0.25)
    shake: ShakeProfile = field(default_factory=ShakeProfile)
    max_speed: MetresPerSecond | None = None
    max_acceleration: float | None = None
    roll: float = 0.0
    dof_factor: float = 0.0

    def __post_init__(self) -> None:
        """Validate the parameters.

        Raises:
            ValueError: If the field of view is not a plausible lens, or the
                smoothing time constant is negative.
        """
        if not _MIN_FOV_DEGREES < self.fov < _MAX_FOV_DEGREES:
            message = (
                f"Field of view must be in "
                f"({_MIN_FOV_DEGREES}, {_MAX_FOV_DEGREES}) degrees, got {self.fov}."
            )
            raise ValueError(message)
        if self.smoothing < 0.0:
            message = f"Smoothing must be non-negative, got {self.smoothing}."
            raise ValueError(message)

    @property
    def zooms(self) -> bool:
        """Whether this spec varies its field of view with subject distance.

        Returns:
            ``True`` if both near and far fields of view are set.
        """
        return self.fov_near is not None and self.fov_far is not None

    def fov_at(self, distance: Metres, near: Metres, far: Metres) -> FovDegrees:
        """Interpolate field of view for a given subject distance.

        Mirrors the behaviour of Assetto Corsa's trackside cameras, which use a
        wider lens when the car is close and a narrower one when it is distant,
        so the car holds a roughly constant size in frame.

        Args:
            distance: Current camera-to-subject distance.
            near: Distance at which ``fov_near`` applies.
            far: Distance at which ``fov_far`` applies.

        Returns:
            The interpolated vertical field of view, or the nominal ``fov`` if
            zoom tracking is not configured.
        """
        if self.fov_near is None or self.fov_far is None:
            return self.fov
        span = far - near
        if span <= 0.0:
            return self.fov_near
        t = min(1.0, max(0.0, (distance - near) / span))
        return FovDegrees(self.fov_near + (self.fov_far - self.fov_near) * t)


@dataclass(frozen=True, slots=True)
class CameraKeyframe:
    """The camera's complete state at one instant.

    Attributes:
        time: When, in seconds from session start.
        basis: Position, look direction and up vector.
        fov: Vertical field of view in degrees.
        dof_distance: Focus distance in metres, or ``None`` to leave it alone.
        dof_factor: Depth-of-field strength in ``[0, 1]``.
        exposure: Exposure multiplier, or ``None`` to leave it alone.
        own_share: How much of the frame this camera controls, in ``[0, 1]``.
            Zero hands control back to the game's own camera; one takes it fully.
            Ramping it at a shot's edges gives an engine-blessed blend in and
            out, which is how a generated cut hands over gracefully rather than
            snapping.
    """

    time: Seconds
    basis: CameraBasis
    fov: FovDegrees
    dof_distance: Metres | None = None
    dof_factor: float = 0.0
    exposure: float | None = None
    own_share: float = 1.0


@dataclass(frozen=True, slots=True)
class CameraPath:
    """A camera's motion over the span of one shot.

    Attributes:
        keyframes: Samples ordered by time. Sampled densely rather than stored as
            control points, because the consumer is a per-frame renderer and the
            generator has already done the interpolation with full knowledge of
            the constraints.
        spec: The specification this path was generated from.
    """

    keyframes: tuple[CameraKeyframe, ...]
    spec: CameraSpec

    def __post_init__(self) -> None:
        """Validate the path.

        Raises:
            ValueError: If the path is empty or its keyframes are not ordered
                by increasing time.
        """
        if not self.keyframes:
            message = "A camera path must contain at least one keyframe."
            raise ValueError(message)
        times = [keyframe.time for keyframe in self.keyframes]
        if any(later < earlier for earlier, later in pairwise(times)):
            message = "Camera path keyframes must be ordered by increasing time."
            raise ValueError(message)

    @property
    def start(self) -> Seconds:
        """When the path begins.

        Returns:
            The first keyframe's time.
        """
        return self.keyframes[0].time

    @property
    def end(self) -> Seconds:
        """When the path ends.

        Returns:
            The last keyframe's time.
        """
        return self.keyframes[-1].time

    @property
    def duration(self) -> Seconds:
        """How long the path lasts.

        Returns:
            The span in seconds.
        """
        return Seconds(self.end - self.start)

    def at(self, moment: Seconds) -> CameraKeyframe:
        """Find the keyframe nearest a moment.

        Args:
            moment: The time of interest, in seconds from session start.

        Returns:
            The closest keyframe. Clamped to the path's ends rather than
            extrapolated, since extrapolating a camera path produces positions
            the constraints were never checked against.
        """
        best = self.keyframes[0]
        best_distance = abs(best.time - moment)
        for keyframe in self.keyframes[1:]:
            distance = abs(keyframe.time - moment)
            if distance < best_distance:
                best, best_distance = keyframe, distance
        return best
