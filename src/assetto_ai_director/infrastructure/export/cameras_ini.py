"""Export a timeline as an Assetto Corsa trackside camera set.

**This is the one output that works in unmodified Assetto Corsa today**, which is
why it exists despite being the lossiest exporter in the project.

It is lossy for a structural reason, not an implementation one.
``cameras.ini`` describes static trackside cameras that the game switches between
according to **where the car is on the track**, expressed as a normalised spline
position in ``[0, 1]``. A directed timeline switches according to **when
something happened**. Those are different coordinate systems, and no amount of
care converts one into the other:

* Only cameras that hold position can be expressed. A drone, a chase camera or a
  bonnet mount has no representation, so those shots are dropped.
* Cutting on an event is impossible. The game cuts when the car reaches a spline
  position, so an overtake that happens on lap three at the same place a quiet
  moment happened on lap one gets the same camera.
* Shot order becomes track order. The timeline's rhythm does not survive.

The exporter reports all of this through its support report rather than writing a
file that silently means something different from the timeline it came from.

Two format details that are easy to get wrong, both confirmed against real files
and modding references:

* ``MIN_FOV`` is the field of view when the car is **furthest** and ``MAX_FOV``
  when it is **closest**. ``MIN_FOV`` is therefore routinely the numerically
  larger of the two, and a validator asserting ``MIN <= MAX`` would reject
  Kunos's own camera sets.
* The data directory is **layout-scoped**. A multi-layout track keeps its
  cameras under ``content/tracks/<track>/<layout>/data/``, not under the track
  root, and writing to the wrong one produces a file the game never loads.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from assetto_ai_director.application.camera.spline import position_spans, time_fraction_spans
from assetto_ai_director.application.ports.timeline_sink import SupportReport
from assetto_ai_director.domain.camera import RigKind

if TYPE_CHECKING:
    from pathlib import Path

    from assetto_ai_director.domain.shot import Shot
    from assetto_ai_director.domain.timeline import Timeline
    from assetto_ai_director.domain.track import TrackSpline

__all__ = [
    "EXPORTABLE_RIGS",
    "CamerasIniSink",
    "planned_spans",
    "suggested_path",
    "write_cameras_ini",
]

FORMAT_VERSION = 3
"""The ``[HEADER] VERSION`` value current AC track camera sets use."""

MAX_CAMERAS = 40
"""Most cameras written to one set.

Track camera sets are hand-authored and typically hold a dozen or two. A
directed edit can contain hundreds of shots; emitting one camera each would
produce a set that switches every few metres and reads as a strobe.
"""

EXPORTABLE_RIGS = frozenset({RigKind.ROADSIDE, RigKind.STATIC, RigKind.SPLINE})
"""Rigs that hold position and can therefore be a trackside camera.

Public because the camera diagnostic has to know which shots this format would
actually keep. A check that reported a span problem for a drone shot — which
never reaches the file — would be reporting a defect that cannot exist.
"""


class CamerasIniSink:
    """Writes a timeline as an Assetto Corsa camera set.

    Satisfies the ``TimelineSink`` protocol structurally.

    Attributes:
        track: The track's centreline, when the caller has one. It cannot be a
            parameter of :meth:`write` because that signature belongs to the
            ``TimelineSink`` protocol, and it matters enough to be worth
            carrying: without it the file switches cameras on elapsed time
            wearing a track position's clothes.
    """

    def __init__(self, track: TrackSpline | None = None) -> None:
        """Build a sink.

        Args:
            track: The track's centreline, or ``None``.
        """
        self.track = track

    @property
    def name(self) -> str:
        """A short identifier for this sink.

        Returns:
            The sink name.
        """
        return "cameras-ini"

    @property
    def extension(self) -> str:
        """The file extension this sink writes.

        Returns:
            ``".ini"``.
        """
        return ".ini"

    def supports(self, timeline: Timeline) -> SupportReport:
        """Report what would survive exporting this timeline.

        Args:
            timeline: The timeline to assess.

        Returns:
            An honest account of what the format cannot carry.
        """
        exportable = [shot for shot in timeline.shots if shot.candidate.rig in EXPORTABLE_RIGS]
        dropped = timeline.shot_count - len(exportable)

        if not exportable:
            return SupportReport.refused(
                "No shot in this timeline uses a camera that holds position, and "
                "cameras.ini can only describe fixed trackside cameras."
            )

        limitations = [
            (
                "Cameras switch on track position, not on time, so cuts land "
                "where the car reaches a place rather than when something happens."
            ),
            "Shot order becomes track order; the edit's rhythm is not preserved.",
        ]
        if dropped:
            limitations.insert(
                0,
                f"{dropped} of {timeline.shot_count} shots use moving cameras "
                f"(drone, chase, mounted) which this format cannot express.",
            )
        if len(exportable) > MAX_CAMERAS:
            limitations.append(
                f"Only the {MAX_CAMERAS} highest-scoring fixed cameras are kept; "
                f"{len(exportable) - MAX_CAMERAS} more were discarded."
            )
        return SupportReport(
            supported=True, lossy=True, limitations=tuple(limitations), dropped_shots=dropped
        )

    def write(self, timeline: Timeline, destination: Path) -> None:
        """Write the camera set.

        Args:
            timeline: The timeline to export.
            destination: Where to write it.

        Raises:
            ValueError: If no shot can be expressed as a trackside camera.
            OSError: If the destination cannot be written.
        """
        write_cameras_ini(timeline, destination, track=self.track)


def suggested_path(track_root: Path, timeline: Timeline, set_index: int = 9) -> Path:
    """Work out where a camera set belongs inside an Assetto Corsa install.

    Args:
        track_root: The ``content/tracks/<track>`` directory.
        timeline: The timeline, which knows the track layout.
        set_index: Which camera set this is. Sets are ``cameras.ini``,
            ``cameras_1.ini`` and so on; a high index avoids colliding with the
            ones the track already ships.

    Returns:
        The full path to write to, layout-scoped when the track has layouts.
    """
    layout = timeline.session.layout
    base = track_root / layout / "data" if layout else track_root / "data"
    name = "cameras.ini" if set_index == 0 else f"cameras_{set_index}.ini"
    return base / name


def write_cameras_ini(
    timeline: Timeline, destination: Path, *, track: TrackSpline | None = None
) -> None:
    """Write a timeline's fixed cameras as an Assetto Corsa camera set.

    Args:
        timeline: The timeline to export.
        destination: Where to write it. Parent directories are created.
        track: The track's centreline. **Supplying it is what makes the file
            switch cameras where they actually stand.** Without one the spans
            fall back to dividing the timeline's duration, which is the same
            axis confusion the camera diagnostic exists to measure.

    Raises:
        ValueError: If no shot uses a camera that holds position.
        OSError: If the destination cannot be written.
    """
    cameras = _select(timeline)
    if not cameras:
        message = (
            "This timeline contains no fixed trackside camera, and cameras.ini "
            "cannot express moving ones."
        )
        raise ValueError(message)

    cameras, spans = _lay_out(timeline, cameras, track)
    lines = [
        "; Generated by Assetto AI Director.",
        "; Cameras switch on track position, not on time: this file cannot",
        "; reproduce the timeline's cuts, only where its fixed cameras stood.",
        "",
        "[HEADER]",
        f"VERSION={FORMAT_VERSION}",
        f"CAMERA_COUNT={len(cameras)}",
        f"SET_NAME=AI Director {timeline.style_name}",
        "",
    ]

    for index, (shot, (in_point, out_point)) in enumerate(zip(cameras, spans, strict=True)):
        lines.extend(_camera_block(index, shot, in_point, out_point))
        lines.append("")

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")


def planned_spans(
    timeline: Timeline, *, track: TrackSpline | None = None
) -> dict[int, tuple[float, float]]:
    """Report which stretch of track each shot would claim, without writing anything.

    Exists so the camera diagnostic can check the numbers this exporter *would*
    produce rather than a reimplementation of them. Asking the exporter is the
    only way for the check to stay true after the export policy changes — and
    changing that policy is the point of the work the diagnostic exists to
    justify.

    Args:
        timeline: The timeline that would be exported.
        track: The track's centreline, matching whatever the write would be
            given. Passing a different one here would have the diagnostic check
            a file nobody is going to write.

    Returns:
        ``shot.index`` to ``(in_point, out_point)``, for the shots this format
        would keep. Shots it drops — every moving rig, and anything past the
        camera limit — are absent, because a span it never writes is not a span
        the game can get wrong.
    """
    cameras = _select(timeline)
    if not cameras:
        return {}
    ordered, spans = _lay_out(timeline, cameras, track)
    return {shot.index: span for shot, span in zip(ordered, spans, strict=True)}


def _lay_out(
    timeline: Timeline, cameras: list[Shot], track: TrackSpline | None
) -> tuple[list[Shot], list[tuple[float, float]]]:
    """Decide the order the cameras are written in and what each one claims.

    **With a centreline, order follows the track and each camera claims the
    stretch nearest to it.** Without one, order follows time and the spans divide
    the timeline's duration — the original behaviour, kept as the fallback rather
    than removed, because a session whose track is not installed still deserves a
    file and the report will say what it settled for.

    Args:
        timeline: The timeline, for the total time span.
        cameras: The selected shots.
        track: The track's centreline, or ``None``.

    Returns:
        The shots in the order they should be written, and their spans.
    """
    placed = _positions(cameras, track)
    if placed is None:
        ordered = sorted(cameras, key=lambda shot: float(shot.start))
        return ordered, time_fraction_spans(
            float(timeline.start),
            float(timeline.end),
            [(float(shot.start), float(shot.end)) for shot in ordered],
        )

    together = sorted(zip(cameras, placed, strict=True), key=lambda pair: pair[1])
    ordered = [shot for shot, _ in together]
    return ordered, position_spans([position for _, position in together])


def _positions(cameras: list[Shot], track: TrackSpline | None) -> list[float] | None:
    """Find where each camera stands along the track.

    Args:
        cameras: The selected shots.
        track: The track's centreline, or ``None``.

    Returns:
        Normalised position per camera, or ``None`` when there is no centreline
        to measure against.
    """
    if track is None:
        return None
    return [float(track.project(_middle_position(shot)).position) for shot in cameras]


def _middle_position(shot: Shot) -> np.ndarray:
    """Where a shot's camera sits, as an array the spline can project.

    The camera rather than the car, because a trackside camera's span is about
    where the camera is: a set ordered by where the cars were would put two
    cameras watching the same corner from opposite ends of the file.

    Args:
        shot: The shot.

    Returns:
        A ``(3,)`` world position.
    """
    keyframe = shot.path.keyframes[len(shot.path.keyframes) // 2]
    point = keyframe.basis.position
    return np.array([point.x, point.y, point.z], dtype=np.float64)


def _select(timeline: Timeline) -> list[Shot]:
    """Choose which shots become cameras, ordered by track position.

    Args:
        timeline: The timeline.

    Returns:
        The selected shots, in the order the car encounters them.
    """
    exportable = [shot for shot in timeline.shots if shot.candidate.rig in EXPORTABLE_RIGS]
    if len(exportable) > MAX_CAMERAS:
        exportable = sorted(exportable, key=lambda s: -float(s.candidate.score))[:MAX_CAMERAS]
    # Track order, not edit order: the game plays them by position.
    return sorted(exportable, key=lambda s: float(s.start))


def _camera_block(index: int, shot: Shot, in_point: float, out_point: float) -> list[str]:
    """Render one ``[CAMERA_N]`` section.

    Args:
        index: The camera's zero-based index.
        shot: The shot it came from.
        in_point: Normalised track position where it takes over.
        out_point: Normalised track position where it hands over.

    Returns:
        The section's lines.
    """
    keyframe = shot.path.keyframes[len(shot.path.keyframes) // 2]
    position = keyframe.basis.position
    forward = keyframe.basis.look
    up = keyframe.basis.up

    # MIN_FOV applies when the car is furthest, MAX_FOV when closest, so the
    # wider value belongs to MAX_FOV. Emitting them the intuitive way round
    # makes the camera zoom in as the car approaches, which is backwards.
    spec = shot.path.spec
    far_fov = float(spec.fov_far) if spec.fov_far is not None else float(spec.fov)
    near_fov = float(spec.fov_near) if spec.fov_near is not None else float(spec.fov)

    label = shot.candidate.anchor.kind.value if shot.candidate.anchor else "cover"
    return [
        f"[CAMERA_{index}]",
        f"NAME=AID {index:02d} {label}",
        f"POSITION={position.x:.3f},{position.y:.3f},{position.z:.3f}",
        f"FORWARD={forward.x:.5f},{forward.y:.5f},{forward.z:.5f}",
        f"UP={up.x:.5f},{up.y:.5f},{up.z:.5f}",
        f"MIN_FOV={far_fov:.2f}",
        f"MAX_FOV={near_fov:.2f}",
        f"IN_POINT={in_point:.5f}",
        f"OUT_POINT={out_point:.5f}",
        "SPLINE=",
        "SPLINE_ANIMATION_LENGTH=0",
        "IS_FIXED=1",
        "NEAR_PLANE=0.5",
        "FAR_PLANE=8000",
    ]
