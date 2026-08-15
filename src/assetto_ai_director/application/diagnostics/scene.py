"""Building the wireframe a camera would see, so a person can look at it.

The checks answer "is the car in frame" in numbers. This answers "and what else
is going on" in a picture, which is the part a number cannot carry: a camera
pointing across the wrong side of a hairpin passes every check that exists today
and is obviously wrong the moment you see it.

**Wireframe, not a render.** The project reads no track mesh and no car model, so
there is nothing to shade. What it does have is every car's path through the
session and every car's position at any instant, and those are enough to draw the
road as the cars traced it, the cars as boxes, and the horizon. A preview that
looked photographic would be inviting the reader to trust detail that was
invented.

**Clipping is done properly rather than skipped.** A line from a point in front
of the camera to a point behind it must be cut at the camera plane; projecting
both ends and joining them draws a stroke straight across the frame in the
opposite direction. That artefact looks exactly like a piece of track geometry,
which is the worst possible failure for a diagnostic image.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np

from assetto_ai_director.application.diagnostics.checks import (
    NOMINAL_CAR_HEIGHT_METRES,
    NOMINAL_CAR_LENGTH_METRES,
)
from assetto_ai_director.domain.geometry import CameraBasis, Vec3
from assetto_ai_director.domain.projection import DEFAULT_ASPECT, project_point
from assetto_ai_director.domain.units import FovDegrees

if TYPE_CHECKING:
    from collections.abc import Sequence

    from assetto_ai_director.application.camera.paths import SubjectTracks
    from assetto_ai_director.domain.diagnostics import CheckStatus, DiagnosticReport

__all__ = [
    "Preview",
    "Stroke",
    "StrokeKind",
    "build_previews",
    "build_scene",
    "project_scene",
]

CAR_WIDTH_METRES = 1.80
"""Width of the box drawn for a car. A guess, like its length and height."""

_PATH_SAMPLES = 400
"""How many points of a car's path to draw.

A session is tens of thousands of samples and a preview is a few hundred pixels
wide, so drawing every one costs time and shows nothing extra.
"""

_HORIZON_METRES = 400.0
"""How far out the horizon ring is drawn."""

_HORIZON_SEGMENTS = 72

_DEGENERATE_SPAN = 1e-9
"""Depth difference below which a segment has no crossing to interpolate."""

_NEAR_PLANE = 0.05
"""Where a segment crossing behind the camera is cut, in metres.

Not zero: a point exactly on the camera plane projects to infinity, and the
resulting stroke runs off to the edge of the image looking like real geometry.
"""


class StrokeKind(StrEnum):
    """What a stroke represents, so a renderer can style it.

    Attributes:
        SUBJECT_PATH: The path of the car this shot is about.
        OTHER_PATH: Another car's path.
        SUBJECT_CAR: The subject's body at the moment being previewed.
        OTHER_CAR: Another car's body.
        HORIZON: The horizon ring, for orientation.
        FRAME_CENTRE: The crosshair marking where the camera actually points.
    """

    SUBJECT_PATH = "subject_path"
    OTHER_PATH = "other_path"
    SUBJECT_CAR = "subject_car"
    OTHER_CAR = "other_car"
    HORIZON = "horizon"
    FRAME_CENTRE = "frame_centre"


@dataclass(frozen=True, slots=True)
class Preview:
    """One camera's view, ready to be drawn.

    Attributes:
        shot_index: Which shot this is, matching the diagnostic.
        label: A caption identifying the shot at a glance.
        status: The camera's overall verdict, which sets the border colour.
        measured_status: The verdict counting only checks that ran. Carried
            separately because until the track's geometry can be read no camera
            can be fully green, and a sheet of uniformly blue borders tells a
            reader nothing.
        strokes: The projected geometry.
    """

    shot_index: int
    label: str
    status: CheckStatus
    measured_status: CheckStatus
    strokes: tuple[Stroke, ...]


@dataclass(frozen=True, slots=True)
class Polyline:
    """A connected run of world-space points.

    Attributes:
        points: The vertices, in order.
        kind: What it represents.
    """

    points: tuple[Vec3, ...]
    kind: StrokeKind


@dataclass(frozen=True, slots=True)
class Stroke:
    """A polyline projected into a camera's frame.

    Attributes:
        points: ``(u, v)`` pairs, with ``u`` and ``v`` in ``[-1, 1]`` at the
            frame edges. Values outside that range are kept rather than clipped
            to the border, so a renderer can decide whether to show the
            surrounding context or crop to the frame.
        kind: What it represents.
    """

    points: tuple[tuple[float, float], ...]
    kind: StrokeKind


def build_scene(
    tracks: SubjectTracks,
    moment: float,
    subject: int,
    *,
    horizon_at: Vec3 | None = None,
) -> list[Polyline]:
    """Assemble the world geometry worth drawing for one instant.

    Args:
        tracks: The cars' motion.
        moment: The instant to draw the cars at, seconds from session start.
        subject: The car this shot is about, drawn differently from the rest.
        horizon_at: Where to centre the horizon ring, normally the camera. Omit
            it to leave the horizon out.

    Returns:
        World-space polylines, paths first so cars draw over them.
    """
    scene: list[Polyline] = []
    if horizon_at is not None:
        scene.append(_horizon(horizon_at))

    for car_id, positions in sorted(tracks.positions.items()):
        if positions.shape[0] == 0:
            continue
        is_subject = car_id == subject
        scene.append(
            Polyline(
                points=_sample_path(positions),
                kind=StrokeKind.SUBJECT_PATH if is_subject else StrokeKind.OTHER_PATH,
            )
        )

    index = _frame_index(tracks.time, moment)
    for car_id, positions in sorted(tracks.positions.items()):
        if positions.shape[0] == 0:
            continue
        at = min(index, positions.shape[0] - 1)
        heading = float(tracks.headings.get(car_id, np.zeros(1))[at])
        scene.extend(
            _car_box(
                _vec(positions[at]),
                heading,
                StrokeKind.SUBJECT_CAR if car_id == subject else StrokeKind.OTHER_CAR,
            )
        )
    return scene


def project_scene(
    basis: CameraBasis,
    fov: FovDegrees,
    scene: Sequence[Polyline],
    *,
    aspect: float = DEFAULT_ASPECT,
) -> list[Stroke]:
    """Project world polylines into a camera's frame, clipping at the near plane.

    Args:
        basis: The camera's placement.
        fov: Vertical field of view in degrees.
        scene: The world geometry.
        aspect: Frame width divided by height.

    Returns:
        Projected strokes. One polyline can become several strokes, or none, when
        it passes behind the camera.
    """
    strokes: list[Stroke] = [
        Stroke(points=((0.0, 0.0),), kind=StrokeKind.FRAME_CENTRE),
    ]
    for line in scene:
        for run in _visible_runs(basis, line.points):
            if len(run) < 2:  # noqa: PLR2004 - a single point is not a stroke
                continue
            strokes.append(
                Stroke(
                    points=tuple(
                        (point.u, point.v)
                        for point in (project_point(basis, vertex, fov, aspect) for vertex in run)
                    ),
                    kind=line.kind,
                )
            )
    return strokes


def build_previews(
    report: DiagnosticReport,
    tracks: SubjectTracks,
    *,
    aspect: float = DEFAULT_ASPECT,
) -> list[Preview]:
    """Render every camera in a report to projected strokes.

    The one place that turns a diagnosis into pictures, so the desktop window,
    the command line and the MCP server all show the same view of the same shot
    rather than three approximations of it.

    Args:
        report: The diagnosis.
        tracks: The cars' motion.
        aspect: Frame width divided by height.

    Returns:
        One preview per camera, in cut order.
    """
    previews: list[Preview] = []
    for camera in report.cameras:
        moment = (float(camera.start) + float(camera.end)) * 0.5
        basis = CameraBasis(position=camera.position, look=camera.look).orthonormalised()
        subject = camera.subjects[0] if camera.subjects else -1
        scene = build_scene(tracks, moment, subject, horizon_at=camera.position)
        previews.append(
            Preview(
                shot_index=camera.shot_index,
                label=(
                    f"#{camera.shot_index:02d} {camera.rig.value} "
                    f"{camera.size.name.lower()} {moment:.1f}s"
                ),
                status=camera.status,
                measured_status=camera.measured_status,
                strokes=tuple(project_scene(basis, FovDegrees(camera.fov), scene, aspect=aspect)),
            )
        )
    return previews


def _visible_runs(basis: CameraBasis, points: Sequence[Vec3]) -> list[list[Vec3]]:
    """Split a polyline into the pieces in front of the camera.

    A segment straddling the camera plane is cut at it, so the visible part is
    drawn to the edge of what the camera can see and the invisible part is
    dropped. Joining the two ends instead draws a stroke across the frame that
    reads as track geometry and is not there.

    Args:
        basis: The camera's placement.
        points: The polyline's world vertices.

    Returns:
        Runs of vertices, each entirely in front of the camera.
    """
    square = basis.orthonormalised()
    depths = [(vertex - square.position).dot(square.look) for vertex in points]

    runs: list[list[Vec3]] = []
    current: list[Vec3] = []
    for index, (vertex, depth) in enumerate(zip(points, depths, strict=True)):
        if depth > _NEAR_PLANE:
            if not current and index > 0:
                current.append(_cut(points[index - 1], depths[index - 1], vertex, depth))
            current.append(vertex)
            continue
        if current:
            current.append(_cut(points[index - 1], depths[index - 1], vertex, depth))
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def _cut(inside: Vec3, inside_depth: float, outside: Vec3, outside_depth: float) -> Vec3:
    """Find where a segment crosses the near plane.

    Args:
        inside: The endpoint whose depth is on one side.
        inside_depth: Its depth along the look direction.
        outside: The other endpoint.
        outside_depth: Its depth.

    Returns:
        The crossing point, or the endpoint nearest the plane when the two
        depths are equal and there is nothing to interpolate.
    """
    span = inside_depth - outside_depth
    if abs(span) < _DEGENERATE_SPAN:
        return inside
    t = (inside_depth - _NEAR_PLANE) / span
    t = min(1.0, max(0.0, t))
    return Vec3(
        inside.x + (outside.x - inside.x) * t,
        inside.y + (outside.y - inside.y) * t,
        inside.z + (outside.z - inside.z) * t,
    )


def _sample_path(positions: np.ndarray) -> tuple[Vec3, ...]:
    """Reduce a car's path to a drawable number of points.

    Args:
        positions: World positions, shape ``(n, 3)``.

    Returns:
        The sampled path, always including the last point so the line does not
        stop short of where the car ended.
    """
    count = positions.shape[0]
    step = max(1, count // _PATH_SAMPLES)
    indices = list(range(0, count, step))
    if indices[-1] != count - 1:
        indices.append(count - 1)
    return tuple(_vec(positions[index]) for index in indices)


def _car_box(centre: Vec3, heading: float, kind: StrokeKind) -> list[Polyline]:
    """Draw a car as a box on the ground.

    Dimensions are the same guess the apparent-size check uses, so a car that
    looks right in the preview and measures right in the report are the same
    claim rather than two independent ones.

    Args:
        centre: The car's reference point.
        heading: Its heading in radians, ``atan2(x, z)``.
        kind: Which stroke kind to tag the edges with.

    Returns:
        The box's outline: its footprint, its roof, and the uprights joining them.
    """
    forward = Vec3(float(np.sin(heading)), 0.0, float(np.cos(heading)))
    right = Vec3(forward.z, 0.0, -forward.x)
    half_length = NOMINAL_CAR_LENGTH_METRES * 0.5
    half_width = CAR_WIDTH_METRES * 0.5

    def corner(along: float, across: float, up: float) -> Vec3:
        return Vec3(
            centre.x + forward.x * along + right.x * across,
            centre.y + up,
            centre.z + forward.z * along + right.z * across,
        )

    offsets = (
        (half_length, half_width),
        (half_length, -half_width),
        (-half_length, -half_width),
        (-half_length, half_width),
    )
    floor = [corner(along, across, 0.0) for along, across in offsets]
    roof = [corner(along, across, NOMINAL_CAR_HEIGHT_METRES) for along, across in offsets]

    lines = [
        Polyline(points=(*floor, floor[0]), kind=kind),
        Polyline(points=(*roof, roof[0]), kind=kind),
    ]
    lines.extend(
        Polyline(points=(low, high), kind=kind) for low, high in zip(floor, roof, strict=True)
    )
    return lines


def _horizon(centre: Vec3) -> Polyline:
    """Draw a ring at the camera's own height, far enough away to read as level.

    Without it a wireframe of nothing but car paths gives no sense of which way
    is up, and a camera rolled or pitched wrongly looks the same as one that is
    not.

    Args:
        centre: Where to centre the ring, normally the camera.

    Returns:
        The ring.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, _HORIZON_SEGMENTS + 1)
    return Polyline(
        points=tuple(
            Vec3(
                centre.x + _HORIZON_METRES * float(np.sin(angle)),
                centre.y,
                centre.z + _HORIZON_METRES * float(np.cos(angle)),
            )
            for angle in angles
        ),
        kind=StrokeKind.HORIZON,
    )


def _frame_index(time: np.ndarray, moment: float) -> int:
    """Find the frame nearest a moment.

    Args:
        time: The time axis.
        moment: Seconds from session start.

    Returns:
        A valid index.
    """
    last = max(0, time.shape[0] - 1)
    return int(np.clip(np.searchsorted(time, moment), 0, last))


def _vec(row: np.ndarray) -> Vec3:
    """Convert one row of a position array into a world vector.

    Args:
        row: A ``(3,)`` slice of world position.

    Returns:
        The point as a :class:`Vec3`.
    """
    return Vec3(float(row[0]), float(row[1]), float(row[2]))
