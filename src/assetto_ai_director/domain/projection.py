"""Projecting a world point into a camera's frame.

This is what makes "the camera is pointing the wrong way" into a number. Until
now the project could place a camera but had no way to ask what it would see, so
a misaimed shot was indistinguishable from a good one until somebody loaded the
game and looked.

**The basis convention is inherited, not chosen.**
:meth:`~assetto_ai_director.domain.geometry.CameraBasis.orthonormalised` builds
its right vector as ``up x look``. With ``look`` along ``+z`` and ``up`` along
``+y`` that yields ``+x``, so screen-right is world ``+x`` — the left-handed
arrangement Assetto Corsa uses. Projection follows the same rule rather than
deriving its own, because two places computing "right" differently is precisely
how a diagnostic ends up confidently mirroring the thing it is diagnosing.

**Field of view is vertical.** That is the project's canonical convention, set
out in :mod:`assetto_ai_director.domain.units`, and it is what Custom Shaders
Patch's ``GrabbedCamera.fov`` is documented to mean. Assetto Corsa's
``cameras.ini`` uses ``MIN_FOV``/``MAX_FOV`` whose axis is **not** documented
anywhere this project could find, so nothing here silently equates the two: a
caller diagnosing an exported camera set passes the field of view it believes
applies and the report records what was assumed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from assetto_ai_director.domain.geometry import CameraBasis, Vec3
from assetto_ai_director.domain.units import FovDegrees, Metres, Radians

__all__ = [
    "DEFAULT_ASPECT",
    "ScreenPoint",
    "angle_off_axis",
    "frame_fraction",
    "project_point",
]

DEFAULT_ASPECT = 16.0 / 9.0
"""Frame width divided by height, when a caller does not say.

Sixteen by nine because that is what the game is played at. The value only
affects the horizontal extent of the frame, so a wrong aspect makes a camera
look narrower or wider than it is but never changes whether the subject is in
front of it.
"""

_EPSILON = 1e-9

_MAX_FOV_DEGREES = 180.0
"""Above this a lens has no meaning: the frame would wrap around the camera."""


@dataclass(frozen=True, slots=True)
class ScreenPoint:
    """Where a world point lands in a camera's frame.

    Attributes:
        u: Horizontal position, ``-1`` at the left edge and ``+1`` at the right.
            Values outside that range are off frame, not clamped, because how
            far off is the useful part of the answer.
        v: Vertical position, ``-1`` at the bottom edge and ``+1`` at the top.
        depth: Distance along the camera's look direction, in metres. **Negative
            means the point is behind the camera**, in which case ``u`` and ``v``
            are meaningless and :attr:`in_frame` is false regardless of them —
            a point behind the lens projects to plausible screen coordinates if
            the sign is not checked, which is the classic way to render a shot
            of nothing and report it as well framed.
    """

    u: float
    v: float
    depth: Metres

    @property
    def in_front(self) -> bool:
        """Whether the point is on the visible side of the camera.

        Returns:
            ``True`` when depth is positive.
        """
        return self.depth > _EPSILON

    @property
    def in_frame(self) -> bool:
        """Whether the point would be visible.

        Returns:
            ``True`` when the point is in front of the camera and inside both
            frame edges.
        """
        return self.in_front and abs(self.u) <= 1.0 and abs(self.v) <= 1.0

    @property
    def off_frame_by(self) -> float:
        """How far outside the frame the point is, in frame half-widths.

        Distinguishes "just outside the edge" from "nowhere near", which is the
        difference between a shot that needs nudging and a camera aimed at the
        wrong corner of the track.

        Returns:
            Zero when in frame, otherwise the largest excess over an edge. A
            point behind the camera returns ``inf``, since no amount of nudging
            brings it back.
        """
        if not self.in_front:
            return math.inf
        return max(0.0, abs(self.u) - 1.0, abs(self.v) - 1.0)


def project_point(
    basis: CameraBasis,
    point: Vec3,
    fov: FovDegrees,
    aspect: float = DEFAULT_ASPECT,
) -> ScreenPoint:
    """Project a world point into a camera's frame.

    A pinhole projection: no lens distortion, no near or far clipping. Both are
    deliberate — this answers "is the car in shot and where", and a distortion
    model would add error to that answer rather than removing any.

    Args:
        basis: The camera's placement. Orthonormalised first, so a caller passing
            a basis assembled by hand gets the same answer as one built by
            :meth:`~assetto_ai_director.domain.geometry.CameraBasis.looking_at`.
        point: The world point to project.
        fov: Vertical field of view in degrees.
        aspect: Frame width divided by height.

    Returns:
        Where the point lands.

    Raises:
        ValueError: If the two vectors are in different reference frames, or the
            field of view or aspect is not a usable lens.
    """
    if not 0.0 < fov < _MAX_FOV_DEGREES:
        message = f"Field of view must be in (0, {_MAX_FOV_DEGREES}) degrees, got {fov}."
        raise ValueError(message)
    if aspect <= 0.0:
        message = f"Aspect ratio must be positive, got {aspect}."
        raise ValueError(message)

    square = basis.orthonormalised()
    relative = point - square.position
    depth = relative.dot(square.look)

    # Same handedness as CameraBasis.orthonormalised(), on purpose.
    right = square.up.cross(square.look)

    if depth <= _EPSILON:
        # Behind the lens. Screen coordinates would be a reflected fiction, so
        # the depth sign is the answer and the rest is reported as centred.
        return ScreenPoint(u=0.0, v=0.0, depth=Metres(depth))

    half_height = depth * math.tan(math.radians(fov) * 0.5)
    half_width = half_height * aspect
    return ScreenPoint(
        u=relative.dot(right) / half_width,
        v=relative.dot(square.up) / half_height,
        depth=Metres(depth),
    )


def frame_fraction(size: Metres, depth: Metres, fov: FovDegrees) -> float:
    """Work out how much of the frame's height an object occupies.

    Turns "the car is a speck" and "the car overflows the frame" into one
    comparable number. A car roughly filling the frame vertically scores ``1.0``.

    Args:
        size: The object's extent perpendicular to the view, in metres.
        depth: Distance along the look direction, in metres.
        fov: Vertical field of view in degrees.

    Returns:
        The fraction of frame height covered, or ``0.0`` when the object is at
        or behind the camera, where the question has no answer.
    """
    if depth <= _EPSILON:
        return 0.0
    return float(size) / (2.0 * float(depth) * math.tan(math.radians(fov) * 0.5))


def angle_off_axis(basis: CameraBasis, point: Vec3) -> Radians:
    """Measure the angle between where the camera looks and where a point is.

    Independent of field of view, which makes it the right measure for "is this
    camera aimed at the car at all" — a shot can be badly aimed and still contain
    the subject simply because the lens is wide.

    Args:
        basis: The camera's placement.
        point: The world point.

    Returns:
        The angle in radians, in ``[0, pi]``. A point coincident with the camera
        returns zero, since no direction is defined.

    Raises:
        ValueError: If the two vectors are in different reference frames.
    """
    square = basis.orthonormalised()
    relative = point - square.position
    length = relative.length
    if length < _EPSILON:
        return Radians(0.0)
    cosine = min(1.0, max(-1.0, relative.dot(square.look) / length))
    return Radians(math.acos(cosine))
