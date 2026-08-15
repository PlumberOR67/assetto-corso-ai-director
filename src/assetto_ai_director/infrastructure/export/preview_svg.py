"""Rendering camera previews as SVG, using nothing but the standard library.

**No new dependency, on purpose.** The project's runtime is numpy, scipy,
pydantic, loguru, rich and typer; Qt and OpenCV are optional extras. A contact
sheet that needed a raster library would either add a dependency for one feature
or make the feature unavailable over SSH, and the thing being drawn is a
wireframe — lines with no shading — which is what vector graphics are for.

SVG also has two properties a PNG does not, both of which matter here: it opens
in any browser without a viewer, and it is text, so two previews of the same
camera before and after a change produce a readable diff.

**The frame border is drawn, and geometry outside it is kept.** Cropping to the
frame would hide the most useful thing a preview can show — a camera aimed just
past the car, with the car sitting visibly outside the border. Everything within
a margin of the frame is drawn dimmed, so what the camera misses is as legible as
what it catches.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from xml.sax.saxutils import escape

from assetto_ai_director.application.diagnostics.scene import StrokeKind
from assetto_ai_director.domain.diagnostics import CheckStatus

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from assetto_ai_director.application.diagnostics.scene import Preview, Stroke

__all__ = [
    "CELL_HEIGHT",
    "CELL_WIDTH",
    "contact_sheet",
    "render_preview",
    "write_contact_sheet",
]

CELL_WIDTH = 320
CELL_HEIGHT = 180
"""Size of one preview in the sheet, in SVG units. Sixteen by nine."""

_OUTSIDE = 1.6
"""How far past the frame edge geometry is still drawn, in frame half-widths.

Enough to show a car the camera has just missed, which is the failure a preview
is best at making obvious, without drawing the whole session as a tangle.
"""

_COLUMNS = 4

_STROKE_STYLE: dict[StrokeKind, tuple[str, float, float]] = {
    StrokeKind.HORIZON: ("#3b4252", 1.0, 0.7),
    StrokeKind.OTHER_PATH: ("#4c566a", 1.0, 0.8),
    StrokeKind.SUBJECT_PATH: ("#88c0d0", 1.4, 1.0),
    StrokeKind.OTHER_CAR: ("#d08770", 1.2, 0.9),
    StrokeKind.SUBJECT_CAR: ("#ebcb8b", 2.0, 1.0),
    StrokeKind.FRAME_CENTRE: ("#bf616a", 1.0, 0.9),
}
"""Colour, width and opacity per stroke kind.

The subject is the brightest thing in the image and the horizon the dimmest,
because the question a reader is asking is "where is the car", not "where is the
scenery".
"""

_STATUS_COLOUR: dict[CheckStatus, str] = {
    CheckStatus.PASS: "#a3be8c",
    CheckStatus.WARN: "#ebcb8b",
    CheckStatus.FAIL: "#bf616a",
    CheckStatus.UNKNOWN: "#5e81ac",
}

_BACKGROUND = "#2e3440"
_FRAME_LINE = "#616e88"
_LABEL = "#d8dee9"


def render_preview(
    strokes: Sequence[Stroke],
    *,
    width: int = CELL_WIDTH,
    height: int = CELL_HEIGHT,
    label: str = "",
    status: CheckStatus = CheckStatus.UNKNOWN,
) -> str:
    """Render one camera's view as a standalone SVG document.

    Args:
        strokes: The projected geometry.
        width: Image width in SVG units.
        height: Image height.
        label: Caption drawn in the corner.
        status: Verdict, which sets the border colour.

    Returns:
        A complete SVG document.
    """
    body = _cell_body(strokes, width=width, height=height, label=label, status=status)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">{body}</svg>\n'
    )


def contact_sheet(
    previews: Sequence[Preview],
    *,
    title: str = "",
    columns: int = _COLUMNS,
) -> str:
    """Render every camera as a grid a person can scan in one look.

    Borders are coloured by the *measured* verdict. Until the track's geometry
    can be read, two checks never run and every camera's overall verdict is at
    best unknown — a sheet of forty identical blue borders would carry no
    information at all, while the honest per-camera verdict stays in the report
    beside it.

    Args:
        previews: One per camera, in cut order.
        title: Caption for the sheet.
        columns: How many previews per row.

    Returns:
        A complete SVG document.
    """
    columns = max(1, columns)
    rows = max(1, -(-len(previews) // columns))
    header = 28 if title else 0
    width = columns * CELL_WIDTH
    height = rows * CELL_HEIGHT + header

    opening = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
    )
    parts = [opening, f'<rect width="{width}" height="{height}" fill="{_BACKGROUND}"/>']
    if title:
        parts.append(
            f'<text x="8" y="19" fill="{_LABEL}" font-family="monospace" '
            f'font-size="13">{escape(title)}</text>'
        )

    for index, preview in enumerate(previews):
        x = (index % columns) * CELL_WIDTH
        y = header + (index // columns) * CELL_HEIGHT
        body = _cell_body(
            preview.strokes,
            width=CELL_WIDTH,
            height=CELL_HEIGHT,
            label=preview.label,
            status=preview.measured_status,
        )
        parts.append(
            f'<svg x="{x}" y="{y}" width="{CELL_WIDTH}" height="{CELL_HEIGHT}">{body}</svg>'
        )

    parts.append("</svg>")
    return "".join(parts) + "\n"


def write_contact_sheet(
    previews: Sequence[Preview],
    destination: Path,
    *,
    title: str = "",
    columns: int = _COLUMNS,
) -> None:
    """Write a contact sheet to a file.

    Args:
        previews: One per camera.
        destination: Where to write it. Parent directories are created.
        title: Caption for the sheet.
        columns: How many previews per row.

    Raises:
        OSError: If the destination cannot be written.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(contact_sheet(previews, title=title, columns=columns), encoding="utf-8")


def _cell_body(
    strokes: Sequence[Stroke],
    *,
    width: int,
    height: int,
    label: str,
    status: CheckStatus,
) -> str:
    """Render the contents of one preview cell.

    Args:
        strokes: The projected geometry.
        width: Cell width.
        height: Cell height.
        label: Caption.
        status: Verdict, which sets the border colour.

    Returns:
        SVG fragment, without the enclosing element.
    """
    parts = [f'<rect width="{width}" height="{height}" fill="{_BACKGROUND}"/>']

    # The frame border sits inside the cell so geometry the camera misses has
    # somewhere to be drawn. A preview cropped to the frame cannot show a car
    # the camera just missed, which is the thing a preview is best at.
    inset_x = width / (2.0 * _OUTSIDE)
    inset_y = height / (2.0 * _OUTSIDE)
    parts.append(
        f'<rect x="{(width - 2 * inset_x) / 2:.1f}" y="{(height - 2 * inset_y) / 2:.1f}" '
        f'width="{2 * inset_x:.1f}" height="{2 * inset_y:.1f}" fill="none" '
        f'stroke="{_FRAME_LINE}" stroke-width="1" stroke-dasharray="4 3"/>'
    )

    parts.extend(_stroke(stroke, width=width, height=height) for stroke in strokes)

    parts.append(
        f'<rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" fill="none" '
        f'stroke="{_STATUS_COLOUR[status]}" stroke-width="2"/>'
    )
    if label:
        parts.append(
            f'<text x="6" y="{height - 6}" fill="{_LABEL}" font-family="monospace" '
            f'font-size="10">{escape(label)}</text>'
        )
    return "".join(parts)


def _stroke(stroke: Stroke, *, width: int, height: int) -> str:
    """Render one projected polyline.

    Args:
        stroke: The polyline.
        width: Cell width.
        height: Cell height.

    Returns:
        An SVG fragment, or an empty string when nothing of it is near the frame.
    """
    colour, thickness, opacity = _STROKE_STYLE[stroke.kind]

    if stroke.kind is StrokeKind.FRAME_CENTRE:
        x, y = _to_pixels(0.0, 0.0, width, height)
        return (
            f'<path d="M{x - 5:.1f},{y:.1f}h10M{x:.1f},{y - 5:.1f}v10" stroke="{colour}" '
            f'stroke-width="{thickness}" opacity="{opacity}" fill="none"/>'
        )

    runs = _near_frame(stroke.points)
    if not runs:
        return ""

    commands: list[str] = []
    for run in runs:
        coordinates = [_to_pixels(u, v, width, height) for u, v in run]
        commands.append(f"M{coordinates[0][0]:.1f},{coordinates[0][1]:.1f}")
        commands.extend(f"L{x:.1f},{y:.1f}" for x, y in coordinates[1:])
    return (
        f'<path d="{"".join(commands)}" stroke="{colour}" stroke-width="{thickness}" '
        f'opacity="{opacity}" fill="none" stroke-linejoin="round" stroke-linecap="round"/>'
    )


def _near_frame(points: Sequence[tuple[float, float]]) -> list[list[tuple[float, float]]]:
    """Keep only the parts of a polyline anywhere near the drawn area.

    A car's path spans the whole track and a preview shows a few dozen metres of
    it, so without this every cell carries several hundred coordinates that land
    far outside the image. That cost a 377 kB contact sheet for thirty-seven
    cameras, most of it invisible, and a browser has to parse all of it.

    One point either side of each visible run is kept, so a line entering the
    frame still arrives from off-screen rather than starting at the edge.

    Args:
        points: The polyline's frame coordinates.

    Returns:
        Runs of consecutive points worth drawing.
    """
    visible = [abs(u) <= _OUTSIDE and abs(v) <= _OUTSIDE for u, v in points]
    runs: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []

    for index, point in enumerate(points):
        neighbourly = visible[index] or (index > 0 and visible[index - 1])
        if not neighbourly and index + 1 < len(points) and visible[index + 1]:
            neighbourly = True
        if neighbourly:
            current.append(point)
            continue
        if current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return [run for run in runs if len(run) > 1]


def _to_pixels(u: float, v: float, width: int, height: int) -> tuple[float, float]:
    """Map normalised frame coordinates onto the cell.

    Args:
        u: Horizontal position, ``-1`` at the frame's left edge.
        v: Vertical position, ``-1`` at the frame's bottom edge.
        width: Cell width.
        height: Cell height.

    Returns:
        ``(x, y)`` in SVG units. The vertical axis is flipped, because SVG's y
        grows downward and the frame's grows upward.
    """
    x = (u / _OUTSIDE + 1.0) * 0.5 * width
    y = (1.0 - v / _OUTSIDE) * 0.5 * height
    # Keep wildly distant geometry from producing megabyte-long path data.
    limit = max(width, height) * 8.0
    return max(-limit, min(limit, x)), max(-limit, min(limit, y))
