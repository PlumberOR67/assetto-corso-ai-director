r"""Reading Assetto Corsa's AI line, and refusing to guess when it does not fit.

``fast_lane.ai`` is the racing line the game drives its AI along, and it is the
only file that carries both a track centreline and how wide the road is at each
point. Every camera check that this project reports as unmeasurable is
unmeasurable because nothing here read it.

**The layout below is community reverse-engineering, and this project has not
verified it against a real file.** That is stated rather than buried, because it
is the difference between what follows and a specification. What makes it safe to
ship anyway is that the parse checks its own arithmetic: the header declares a
point count, each record has a fixed size, and the whole thing must land exactly
on the end of the file. A wrong layout does not produce plausible garbage — it
produces :class:`~assetto_ai_director.domain.errors.StrideMismatchError`, the
same self-verification the ``.acreplay`` reader applies to its own stride.

The layout, with what is claimed about each field:

======================  =========  ==========================================
Field                   Size       Confidence
======================  =========  ==========================================
``version``             int32      VERIFIED by refusal: only 7 is accepted
``detail_count``        int32      VERIFIED by the size check landing on EOF
``lap_time``            int32      GUESS — read and ignored, never interpreted
``sample_count``        int32      GUESS — read and ignored
per point, x N:
  ``position``          3x float32 DERIVED: values must plot as a closed track
  ``length``            float32    GUESS — normalised progress, not relied on
  ``id``                int32      GUESS — read and ignored
``extra_count``         int32      VERIFIED by the size check
per point, x N:
  18 float32            72 bytes   Only ``side_left`` and ``side_right`` are
                                   named; the rest stay an opaque slice
======================  =========  ==========================================

Two fields out of that block are used: the distances to the left and right edges
of the road. Every other float in it keeps its offset and no name, because naming
a field this project cannot check is how a wrong struct layout becomes
documentation somewhere else.

**Positions are not re-derived from the extra block.** Both halves carry
geometry, and a reader that mixed them would hide a layout error behind
agreement. The point block is the source; the extra block contributes width
alone.
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING, Final

import numpy as np

from assetto_ai_director.domain.errors import (
    StrideMismatchError,
    TruncatedReplayError,
    UnsupportedReplayVersionError,
)
from assetto_ai_director.domain.track import SplineSource, TrackSpline

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "EXTRA_STRIDE",
    "HEADER_SIZE",
    "POINT_STRIDE",
    "SUPPORTED_VERSIONS",
    "read_fast_lane",
]

SUPPORTED_VERSIONS: Final = (7,)
"""Versions this reader accepts.

Version 7 is what current Assetto Corsa tracks ship. Older versions exist and
have a different layout; decoding one best-effort would produce a centreline
that plots convincingly and sits in the wrong place, which is the failure this
project is built against.
"""

HEADER_SIZE: Final = 16
"""``version``, ``detail_count``, ``lap_time``, ``sample_count``."""

POINT_STRIDE: Final = 20
"""Three float32 of position, one float32 of length, one int32 of id."""

EXTRA_STRIDE: Final = 72
"""Eighteen float32 per point. Only two of them are named here."""

_SIDE_LEFT_INDEX: Final = 5
_SIDE_RIGHT_INDEX: Final = 6
"""Where the road's half widths sit in the extra block, counting float32s.

The only two fields in that block this reader names. Both are checked for
plausibility before use — a negative or absurd width means the layout is wrong
and the widths are dropped rather than reported.
"""

_MAX_PLAUSIBLE_WIDTH: Final = 60.0
"""Metres from the centreline to a road edge, above which the value is not a width.

Wider than any real circuit — the Nordschleife's widest point is around 12 m of
half width, an airfield layout perhaps double that. Sixty is chosen to reject a
misread float rather than to police track design.
"""

_MIN_POINTS: Final = 2

_COUNT_SIZE: Final = 4
"""One int32, the length prefix on the extra block."""

_HEADER_WHAT: Final = "the fast_lane header"
_POINTS_WHAT: Final = "the fast_lane point block"
_EXTRA_WHAT: Final = "the fast_lane extra block"


def read_fast_lane(path: Path) -> TrackSpline:
    """Read a track's AI line.

    Args:
        path: Path to ``fast_lane.ai``.

    Returns:
        The centreline, with the road's half widths when the extra block is
        present and plausible.

    Raises:
        UnsupportedReplayVersionError: If the file's version is not one this
            reader decodes. Named for replays because it is the project's shared
            "this is a format I will not guess at" error.
        TruncatedReplayError: If the file ends before a structure it declares
            would fit. Checked before any allocation, so a corrupt count is an
            error rather than an out-of-memory kill.
        StrideMismatchError: If the parse does not finish exactly at the end of
            the file. **This is the check that makes the layout safe to use**
            without having verified it: an assumption that is wrong almost
            always lands somewhere other than the last byte.
        OSError: If the file cannot be read.
    """
    data = path.read_bytes()
    if len(data) < HEADER_SIZE:
        raise TruncatedReplayError(_HEADER_WHAT, HEADER_SIZE, len(data))

    version, detail_count, _lap_time, _sample_count = struct.unpack_from("<4i", data, 0)
    if version not in SUPPORTED_VERSIONS:
        raise UnsupportedReplayVersionError(version, SUPPORTED_VERSIONS)
    if detail_count < _MIN_POINTS:
        message = (
            f"An AI line needs at least {_MIN_POINTS} points, the file declares {detail_count}."
        )
        raise ValueError(message)

    points_end = HEADER_SIZE + detail_count * POINT_STRIDE
    if len(data) < points_end + _COUNT_SIZE:
        raise TruncatedReplayError(_POINTS_WHAT, points_end + _COUNT_SIZE, len(data))

    positions = _positions(data, detail_count)
    (extra_count,) = struct.unpack_from("<i", data, points_end)

    expected_end = points_end + _COUNT_SIZE + extra_count * EXTRA_STRIDE
    if len(data) < expected_end:
        raise TruncatedReplayError(_EXTRA_WHAT, expected_end, len(data))
    if expected_end != len(data):
        raise StrideMismatchError(
            expected_end,
            len(data),
            f"{detail_count} points and {extra_count} extra records do not account "
            f"for the whole file; the record layout assumption is wrong",
        )

    left, right = _widths(data, points_end + _COUNT_SIZE, extra_count, detail_count)
    return TrackSpline(
        points=positions,
        source=SplineSource.AI_LINE,
        left_width=left,
        right_width=right,
        closed=_is_closed(positions),
    )


def _positions(data: bytes, count: int) -> np.ndarray:
    """Decode the point block's world positions.

    Args:
        data: The whole file.
        count: How many points the header declared.

    Returns:
        World positions, shape ``(count, 3)``, y up.
    """
    records = np.frombuffer(
        data, dtype=np.uint8, count=count * POINT_STRIDE, offset=HEADER_SIZE
    ).reshape(count, POINT_STRIDE)
    return np.asarray(records[:, :12].copy().view(np.float32).reshape(count, 3), dtype=np.float64)


def _widths(
    data: bytes, offset: int, extra_count: int, point_count: int
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Decode the road's half widths, or decline to.

    Args:
        data: The whole file.
        offset: Where the extra block starts.
        extra_count: How many extra records the file declared.
        point_count: How many points the header declared.

    Returns:
        Left and right half widths, or ``(None, None)`` when the block is absent,
        does not correspond to the points one for one, or holds values that are
        not plausibly widths. Declining is the honest outcome: a check that
        cannot get the road's width must report that it could not, and a zero
        would read as a road with no width at all.
    """
    if extra_count != point_count:
        return None, None

    floats = np.frombuffer(
        data, dtype=np.float32, count=extra_count * (EXTRA_STRIDE // 4), offset=offset
    ).reshape(extra_count, EXTRA_STRIDE // 4)
    left = np.asarray(floats[:, _SIDE_LEFT_INDEX], dtype=np.float64)
    right = np.asarray(floats[:, _SIDE_RIGHT_INDEX], dtype=np.float64)

    if not _plausible(left) or not _plausible(right):
        return None, None
    return left, right


def _plausible(width: np.ndarray) -> bool:
    """Test whether an array reads as road half widths.

    Args:
        width: Candidate values in metres.

    Returns:
        ``True`` when every value is finite, non-negative and below
        :data:`_MAX_PLAUSIBLE_WIDTH`.
    """
    return bool(
        np.all(np.isfinite(width))
        and np.all(width >= 0.0)
        and np.all(width <= _MAX_PLAUSIBLE_WIDTH)
    )


def _is_closed(positions: np.ndarray) -> bool:
    """Test whether the line returns to where it started.

    Args:
        positions: The centreline.

    Returns:
        ``True`` for a circuit. The threshold is one spacing of the line's own
        samples rather than a fixed distance, because a point-to-point course
        that happens to finish near its start is a real thing and a lap that
        closes does so to within a sample.
    """
    steps = np.linalg.norm(np.diff(positions[:, [0, 2]], axis=0), axis=1)
    if steps.size == 0:
        return False
    spacing = float(np.median(steps))
    gap = float(np.linalg.norm(positions[-1, [0, 2]] - positions[0, [0, 2]]))
    return gap <= max(spacing * 2.0, 1.0)
