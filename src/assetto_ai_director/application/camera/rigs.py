"""Where each kind of camera goes, and what it looks at.

One function per rig, all with the same signature, so the path generator does not
need to know what a drone is.

**Two-subject framing uses Toric space** (Lino & Christie, ACM TOG 2015). Given
two targets and where each should sit on screen, the set of camera positions that
satisfies both is a spindle-torus surface, parameterised by two angles: theta
around the pair, and phi in elevation. That reduces a seven-degree-of-freedom
search to two, and — more usefully — the two parameters carry cinematographic
meaning rather than geometric meaning. theta = 0 looks from behind the second
car, theta = ±1 from behind the first, phi = 0 is at their height, +1 above and
-1 below.

Hand-rolling a lerp between two look-at targets instead, which is the obvious
thing to try, does not hold both cars in frame: it aims at a point neither of
them occupies and both drift out of shot as the gap between them changes.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from assetto_ai_director.domain.camera import RigKind
from assetto_ai_director.domain.shot import LineSide, ShotSize

if TYPE_CHECKING:
    from collections.abc import Callable

    from assetto_ai_director.domain.terrain import TerrainGrid
    from assetto_ai_director.domain.track import TrackSpline

__all__ = ["RigPlacement", "place_camera", "toric_position"]

# --- Framing geometry -------------------------------------------------------

SIZE_DISTANCE_METRES: dict[ShotSize, float] = {
    ShotSize.EXTREME_CLOSE_UP: 1.5,
    ShotSize.CLOSE_UP: 4.0,
    ShotSize.MEDIUM: 9.0,
    ShotSize.FULL: 18.0,
    ShotSize.LONG: 40.0,
    ShotSize.EXTREME_LONG: 90.0,
}
"""How far the camera sits from the subject for each shot size.

Derived from the framing rather than measured: at a 40 degree vertical field of
view a 4.2 m car fills the frame at roughly 6 m and occupies a third of it at
about 18 m. The values are engineering judgement and carry GUESS provenance in
any profile built from them.
"""

RIG_HEIGHT_METRES: dict[RigKind, float] = {
    RigKind.STATIC: 0.7,
    RigKind.ROADSIDE: 0.6,
    RigKind.FOLLOW: 1.6,
    RigKind.DRONE: 8.0,
    RigKind.HELICOPTER: 45.0,
    RigKind.ORBITAL: 3.0,
    RigKind.SPLINE: 2.0,
    RigKind.INTERIOR: 1.1,
    RigKind.BONNET: 1.0,
    RigKind.ROOF: 1.45,
    RigKind.WHEEL: 0.35,
    RigKind.SUSPENSION: 0.25,
}
"""Camera height above the car's reference point, per rig.

The trackside rigs sit low deliberately. A camera at knee height against a car
that fills the frame is the single most recognisable choice in touge
cinematography, and it is what makes a slide read as fast.
"""

_ORBIT_RATE = 0.35
"""Radians per second an orbital camera circles its subject."""

_MOUNT_FORWARD: dict[RigKind, float] = {
    RigKind.INTERIOR: -0.2,
    RigKind.BONNET: 1.3,
    RigKind.ROOF: 0.1,
    RigKind.WHEEL: 1.2,
    RigKind.SUSPENSION: 1.2,
}
"""Longitudinal offset of a car-mounted camera, metres from the car origin."""

_MOUNT_LATERAL: dict[RigKind, float] = {
    RigKind.WHEEL: 0.8,
    RigKind.SUSPENSION: 0.75,
}
"""Lateral offset of a car-mounted camera, metres."""

VERGE_METRES = 2.0
"""How far past the road's edge a trackside camera may stand, in metres.

Not zero: a camera on the tarmac is in the way of the cars and reads as a mistake
of a different kind. Two metres is a verge — where a marshal would stand, and
where the game's own trackside cameras generally are.
"""

_VERGE_TARGET = 0.5
"""Where in the verge a clamped camera is put, as a fraction of its width.

The middle rather than the far edge. Landing exactly on the outer limit puts the
camera where the barrier is, and puts every clamped camera precisely on the
threshold the diagnostic warns at — so a fix aimed at the boundary reports as a
report full of warnings, which is a confusing way to succeed.
"""

_VERGE_PASSES = 4
"""How many times to re-measure and slide a camera back toward the road."""

_EPSILON = 1e-9


class RigPlacement:
    """The camera positions and aim points a rig produces over a shot.

    Attributes:
        positions: Camera positions per frame, shape ``(n, 3)``.
        targets: Points the camera aims at per frame, shape ``(n, 3)``.
    """

    __slots__ = ("positions", "targets")

    def __init__(self, positions: np.ndarray, targets: np.ndarray) -> None:
        """Store a placement.

        Args:
            positions: Camera positions.
            targets: Aim points.
        """
        self.positions = positions
        self.targets = targets


def place_camera(
    rig: RigKind,
    size: ShotSize,
    side: LineSide,
    subject: np.ndarray,
    *,
    heading: np.ndarray,
    companion: np.ndarray | None,
    dt: float,
    track: TrackSpline | None = None,
    terrain: TerrainGrid | None = None,
) -> RigPlacement:
    """Compute where a camera sits and what it aims at, over a shot.

    Args:
        rig: Which camera behaviour.
        size: How large the subject should sit in frame.
        side: Which side of the line of action the camera occupies.
        subject: Subject positions, shape ``(n, 3)``.
        heading: Subject headings per frame, radians.
        companion: A second car's positions when the shot frames two, else
            ``None``.
        dt: Time step in seconds.
        track: The road, when it could be read. Only the rigs that stand beside
            it use it, and only to be pulled back onto the verge.
        terrain: The terrain grid from a CSP scan, when available. Used to reject
            candidates that are underground or have no sight line to the track.

    Returns:
        The camera's positions and aim points.
    """
    if companion is not None and rig.can_frame_two_subjects:
        placement = _keep_beside_road(_two_subject(subject, companion, size, side), rig, track)
        return _filter_by_terrain(placement, terrain) if terrain else placement

    handler: Callable[..., RigPlacement] = _HANDLERS.get(rig, _trackside)
    placement = _keep_beside_road(handler(subject, heading, size, side, dt), rig, track)
    return _filter_by_terrain(placement, terrain) if terrain else placement


def _keep_beside_road(
    placement: RigPlacement, rig: RigKind, track: TrackSpline | None
) -> RigPlacement:
    """Pull a camera that holds position back to the edge of the road.

    Placement offsets sideways from the car by the shot size alone, which on a
    narrow road puts a long shot tens of metres into whatever is beside it. The
    camera diagnostic measured that: on one synthetic session, sixteen of
    thirty-seven trackside cameras stood past the road edge, the furthest at
    29 m from a racing line with 5.5 m of road either side.

    **Only rigs that hold position are moved.** A drone is supposed to be off the
    road and above it, and a car-mounted camera is on the car. Clamping those
    would be correcting them for doing their job.

    **The camera keeps its side and its aim.** Only the distance out is reduced,
    so a shot framed from the left stays framed from the left and the continuity
    rules that depend on it still hold. Height follows the road rather than the
    car, which is what stops a camera on a descent from being left in the air.

    Args:
        placement: What the rig produced.
        rig: Which rig produced it.
        track: The road, or ``None`` to leave the placement alone.

    Returns:
        The placement, adjusted where it could be.
    """
    if track is None or not track.knows_width or not rig.is_fixed_position:
        return placement

    positions = placement.positions.copy()
    height = RIG_HEIGHT_METRES.get(rig, 0.6)
    for index in range(positions.shape[0]):
        positions[index] = _onto_verge(track, positions[index], height)
    return RigPlacement(positions, placement.targets)


def _onto_verge(track: TrackSpline, position: np.ndarray, height: float) -> np.ndarray:
    """Move one camera to within a verge of the road, repeating until it settles.

    **One pass is not enough, and the reason is geometric.** Sliding a camera
    from 29 m out to 7.5 m out can put it beside a different stretch of a curving
    road, where the nearest point — and so the measured offset — is different
    again. Measuring after each move and repeating converges in two or three
    passes; a single pass left cameras at nearly 19 m on a road 5.5 m wide.

    The loop is bounded rather than run to convergence: on a hairpin the nearest
    segment can alternate between two, and an unbounded loop would swap between
    them. Stopping early leaves a camera closer than it started, which is the
    point.

    Args:
        track: The road.
        position: The camera's world position.
        height: How high above the road the rig sits.

    Returns:
        The adjusted position, or the original when it was already beside the road.
    """
    current = position
    for _ in range(_VERGE_PASSES):
        point = track.project(current)
        half_width = float(point.half_width or 0.0)
        limit = half_width + VERGE_METRES
        lateral = abs(float(point.lateral))
        if lateral <= limit:
            break
        limit = half_width + VERGE_METRES * _VERGE_TARGET
        # Slide back along the same bearing rather than snapping to the line, so
        # the camera stays on the side the shot was composed from.
        centre = _centre_of(track, current)
        current = centre + (current - centre) * (limit / lateral)
        current[1] = float(point.elevation) + height
    return current


def _centre_of(track: TrackSpline, position: np.ndarray) -> np.ndarray:
    """Find the point on the centreline a camera stands beside.

    Args:
        track: The road.
        position: The camera's world position.

    Returns:
        The nearest point on the centreline, at the camera's own height so the
        slide back is horizontal.
    """
    point = track.project(position)
    along = float(point.distance)
    steps = np.linalg.norm(np.diff(track.points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(steps)))
    centre = np.array([np.interp(along, cumulative, track.points[:, axis]) for axis in range(3)])
    centre[1] = position[1]
    return centre


def toric_position(
    first: np.ndarray, second: np.ndarray, theta: float, phi: float, distance: float
) -> np.ndarray:
    """Place a camera on the Toric surface that frames two targets.

    Args:
        first: First target's positions, shape ``(n, 3)``.
        second: Second target's positions, shape ``(n, 3)``.
        theta: Normalised angle around the pair in ``[-1, 1]``. Zero looks from
            behind the second target; ``±1`` from behind the first.
        phi: Normalised elevation in ``[-1, 1]``. Zero is at the targets' height,
            ``+1`` above, ``-1`` below.
        distance: How far the camera sits from the pair's midpoint, metres.

    Returns:
        Camera positions, shape ``(n, 3)``.
    """
    midpoint = (first + second) * 0.5
    axis = second - first
    ground = axis[:, [0, 2]]
    length = np.linalg.norm(ground, axis=1, keepdims=True)
    # When the two cars occupy the same spot the axis is undefined; fall back to
    # a fixed bearing so the camera stays put rather than spinning.
    unit = np.where(length > _EPSILON, ground / np.maximum(length, _EPSILON), np.array([0.0, 1.0]))

    bearing = np.arctan2(unit[:, 0], unit[:, 1]) + theta * math.pi
    elevation = phi * (math.pi * 0.4)

    horizontal = distance * math.cos(elevation)
    offset = np.stack(
        [
            horizontal * np.sin(bearing),
            np.full(bearing.shape, distance * math.sin(elevation)),
            horizontal * np.cos(bearing),
        ],
        axis=1,
    )
    return np.asarray(midpoint + offset, dtype=np.float64)


def _two_subject(
    subject: np.ndarray, companion: np.ndarray, size: ShotSize, side: LineSide
) -> RigPlacement:
    """Frame two cars using the Toric parameterisation.

    Args:
        subject: Primary car positions.
        companion: Second car positions.
        size: Shot size, which sets how far back the camera sits.
        side: Which side of the line the camera occupies.

    Returns:
        The placement, aiming at the midpoint between the two cars.
    """
    limit = min(subject.shape[0], companion.shape[0])
    subject = subject[:limit]
    companion = companion[:limit]

    separation = np.linalg.norm(subject - companion, axis=1)
    # Far enough back that the pair fits, never closer than the size implies.
    distance = float(max(SIZE_DISTANCE_METRES[size], float(np.median(separation)) * 1.6))
    theta = 0.5 if side is LineSide.LEFT else -0.5
    positions = toric_position(subject, companion, theta=theta, phi=0.25, distance=distance)
    targets = (subject + companion) * 0.5
    return RigPlacement(positions, targets)


def _trackside(
    subject: np.ndarray, heading: np.ndarray, size: ShotSize, side: LineSide, dt: float
) -> RigPlacement:
    """Place a locked-off camera beside the road, aimed at the car throughout.

    Positioned once, from the car's state at the middle of the shot, so the car
    passes closest to the camera around the shot's midpoint rather than arriving
    already gone.

    Args:
        subject: Subject positions.
        heading: Subject headings.
        size: Shot size, setting the standoff distance.
        side: Which side of the road.
        dt: Unused; kept so every rig shares one signature.

    Returns:
        The placement.
    """
    del dt
    count = subject.shape[0]
    middle = count // 2
    distance = SIZE_DISTANCE_METRES[size]
    lateral = _lateral_offset(heading[middle], side, distance)
    anchor = subject[middle] + lateral
    anchor[1] = subject[middle][1] + RIG_HEIGHT_METRES[RigKind.ROADSIDE]
    positions = np.repeat(anchor[None, :], count, axis=0)
    return RigPlacement(positions, subject.copy())


def _static(
    subject: np.ndarray, heading: np.ndarray, size: ShotSize, side: LineSide, dt: float
) -> RigPlacement:
    """Place a fully locked-off camera that does not even rotate.

    Args:
        subject: Subject positions.
        heading: Subject headings.
        size: Shot size.
        side: Which side of the road.
        dt: Unused.

    Returns:
        The placement, aiming at a fixed point rather than tracking.
    """
    del dt
    count = subject.shape[0]
    middle = count // 2
    distance = SIZE_DISTANCE_METRES[size]
    lateral = _lateral_offset(heading[middle], side, distance)
    anchor = subject[middle] + lateral
    anchor[1] = subject[middle][1] + RIG_HEIGHT_METRES[RigKind.STATIC]
    positions = np.repeat(anchor[None, :], count, axis=0)
    # Aims where the car will be, not where it is: a true locked-off shot lets
    # the subject move through a fixed frame.
    targets = np.repeat(subject[middle][None, :], count, axis=0)
    return RigPlacement(positions, targets)


def _follow(
    subject: np.ndarray, heading: np.ndarray, size: ShotSize, side: LineSide, dt: float
) -> RigPlacement:
    """Trail the car at a fixed offset behind and above.

    Args:
        subject: Subject positions.
        heading: Subject headings.
        size: Shot size, setting how far back.
        side: Which side to bias toward.
        dt: Unused.

    Returns:
        The placement.
    """
    del dt
    distance = SIZE_DISTANCE_METRES[size]
    behind = np.stack(
        [-np.sin(heading) * distance, np.zeros(heading.shape), -np.cos(heading) * distance],
        axis=1,
    )
    bias = _lateral_offset_array(heading, side, distance * 0.25)
    positions = subject + behind + bias
    positions[:, 1] = subject[:, 1] + RIG_HEIGHT_METRES[RigKind.FOLLOW]
    return RigPlacement(positions, subject.copy())


def _aerial(height: float) -> Callable[..., RigPlacement]:
    """Build a handler for a camera that flies above the car.

    Args:
        height: How high above the car it sits, metres.

    Returns:
        A placement function.
    """

    def handler(
        subject: np.ndarray, heading: np.ndarray, size: ShotSize, side: LineSide, dt: float
    ) -> RigPlacement:
        del dt
        lead = SIZE_DISTANCE_METRES[size] * 0.4
        ahead = np.stack(
            [np.sin(heading) * lead, np.zeros(heading.shape), np.cos(heading) * lead], axis=1
        )
        bias = _lateral_offset_array(heading, side, SIZE_DISTANCE_METRES[size] * 0.3)
        positions = subject + ahead + bias
        positions[:, 1] = subject[:, 1] + height
        return RigPlacement(positions, subject.copy())

    return handler


def _orbital(
    subject: np.ndarray, heading: np.ndarray, size: ShotSize, side: LineSide, dt: float
) -> RigPlacement:
    """Circle the car while keeping it centred.

    Args:
        subject: Subject positions.
        heading: Subject headings.
        size: Shot size, setting the orbit radius.
        side: Which way round to orbit.
        dt: Time step, which sets the angular rate.

    Returns:
        The placement.
    """
    count = subject.shape[0]
    radius = SIZE_DISTANCE_METRES[size]
    direction = 1.0 if side is LineSide.LEFT else -1.0
    angle = heading[0] + direction * _ORBIT_RATE * dt * np.arange(count)
    offset = np.stack([np.sin(angle) * radius, np.zeros(count), np.cos(angle) * radius], axis=1)
    positions = subject + offset
    positions[:, 1] = subject[:, 1] + RIG_HEIGHT_METRES[RigKind.ORBITAL]
    return RigPlacement(positions, subject.copy())


def _mounted(rig: RigKind) -> Callable[..., RigPlacement]:
    """Build a handler for a camera bolted to the car.

    Args:
        rig: Which mount.

    Returns:
        A placement function.
    """

    def handler(
        subject: np.ndarray, heading: np.ndarray, size: ShotSize, side: LineSide, dt: float
    ) -> RigPlacement:
        del size, dt
        forward = _MOUNT_FORWARD.get(rig, 0.0)
        lateral = _MOUNT_LATERAL.get(rig, 0.0)
        sign = 1.0 if side is not LineSide.RIGHT else -1.0
        along = np.stack(
            [np.sin(heading) * forward, np.zeros(heading.shape), np.cos(heading) * forward],
            axis=1,
        )
        across = _lateral_offset_array(heading, LineSide.LEFT, lateral * sign)
        positions = subject + along + across
        positions[:, 1] = subject[:, 1] + RIG_HEIGHT_METRES[rig]

        if rig in (RigKind.WHEEL, RigKind.SUSPENSION):
            # Aims at its own contact patch, which is what makes slip visible.
            targets = positions.copy()
            targets[:, 1] = subject[:, 1]
        else:
            look_ahead = 25.0
            targets = subject + np.stack(
                [
                    np.sin(heading) * look_ahead,
                    np.zeros(heading.shape),
                    np.cos(heading) * look_ahead,
                ],
                axis=1,
            )
        return RigPlacement(positions, targets)

    return handler


def _lateral_offset(heading: float, side: LineSide, distance: float) -> np.ndarray:
    """Compute a single offset perpendicular to the direction of travel.

    Args:
        heading: The car's heading, radians.
        side: Which side of the line.
        distance: How far out, metres.

    Returns:
        An offset vector, shape ``(3,)``.
    """
    sign = 0.0 if side is LineSide.NEUTRAL else (1.0 if side is LineSide.LEFT else -1.0)
    return np.array(
        [math.cos(heading) * distance * sign, 0.0, -math.sin(heading) * distance * sign]
    )


def _lateral_offset_array(heading: np.ndarray, side: LineSide, distance: float) -> np.ndarray:
    """Compute per-frame offsets perpendicular to the direction of travel.

    Args:
        heading: Headings per frame, radians.
        side: Which side of the line.
        distance: How far out, metres.

    Returns:
        Offset vectors, shape ``(n, 3)``.
    """
    sign = 0.0 if side is LineSide.NEUTRAL else (1.0 if side is LineSide.LEFT else -1.0)
    return np.stack(
        [
            np.cos(heading) * distance * sign,
            np.zeros(heading.shape),
            -np.sin(heading) * distance * sign,
        ],
        axis=1,
    )


def _filter_by_terrain(placement: RigPlacement, terrain: TerrainGrid | None) -> RigPlacement:
    """Reject camera positions that are underground or have no sight line.

    **Only fixed-position rigs are filtered.** A chase camera is meant to follow
    the car and a drone is meant to fly; faulting them for being above ground at
    an arbitrary point would be judging them against a job they do not have.

    **The filter is destructive when terrain says no.** If every position is
    underground or has no sight line, the placement returned has zero frames. The
    caller must detect this and fall back to the formula-based behaviour. This is
    deliberate: a camera underground is worse than no camera at all, because it
    produces frames that look like a mistake rather than a choice.

    Args:
        placement: What the rig produced.
        terrain: The terrain grid, or ``None`` to pass everything through.

    Returns:
        The placement with underground or blind positions removed, or the original
        when no terrain data is available.
    """
    if terrain is None:
        return placement

    positions = placement.positions.copy()
    targets = placement.targets.copy()
    keep = np.ones(positions.shape[0], dtype=bool)

    for i in range(positions.shape[0]):
        x, y, z = float(positions[i][0]), float(positions[i][1]), float(positions[i][2])

        # Check if underground
        if terrain.is_underground(x, y, z, tolerance=0.5):
            keep[i] = False
            continue

        # Check sight line. The scan measured it to the centreline, not to this
        # rig's target, so this rejects positions that cannot see the road at all.
        sight = terrain.sight_clear(x, z)
        if sight is False:  # None means unknown, let it pass
            keep[i] = False

    if not np.any(keep):
        # All positions rejected — return empty so caller can fall back
        return RigPlacement(
            positions=np.empty((0, 3), dtype=np.float64),
            targets=np.empty((0, 3), dtype=np.float64),
        )

    return RigPlacement(positions[keep], targets[keep])


_HANDLERS: dict[RigKind, Callable[..., RigPlacement]] = {
    RigKind.STATIC: _static,
    RigKind.ROADSIDE: _trackside,
    RigKind.FOLLOW: _follow,
    RigKind.DRONE: _aerial(RIG_HEIGHT_METRES[RigKind.DRONE]),
    RigKind.HELICOPTER: _aerial(RIG_HEIGHT_METRES[RigKind.HELICOPTER]),
    RigKind.ORBITAL: _orbital,
    RigKind.SPLINE: _trackside,
    RigKind.INTERIOR: _mounted(RigKind.INTERIOR),
    RigKind.BONNET: _mounted(RigKind.BONNET),
    RigKind.ROOF: _mounted(RigKind.ROOF),
    RigKind.WHEEL: _mounted(RigKind.WHEEL),
    RigKind.SUSPENSION: _mounted(RigKind.SUSPENSION),
}
