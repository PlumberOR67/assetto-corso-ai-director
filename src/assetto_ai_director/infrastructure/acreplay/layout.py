"""The verified structural layout of an ``.acreplay`` v16 file.

Everything here was confirmed byte-exactly against four real replays. The
verification is not a spot check: a complete structural walk of each file lands
on precisely the offset that file records in its own Custom Shaders Patch footer,
with **zero drift across all 17 cars** in every fixture.

Whole-file structure::

    u32   version                     == 16
    f64   recording_interval_ms       30.0 in the fixtures, i.e. 33.3 Hz
    str   weather, track, track_config
    u32   car_count
    u32   current_recording_index     != frame_count when the ring buffer wrapped
    u32   frame_count
    u32   track_object_count

    frame_count x (4 + 12 * track_object_count) bytes    # opaque global block

    for each car:
        str  model, driver, nationality, team, skin
        u32  car_frame_count          == frame_count - 1
        u32  wing_count
        car_frame_count x (276 + 4 * wing_count) bytes   # opaque frame records
        u32  extra_count
        extra_count x 8 bytes                            # opaque; player car only

    u32   lap_record_count
    car_count x lap_record_count x 20 bytes              # opaque

    <-- must land exactly here, at the CSP block offset -->

    last 31 bytes: b"__AC_SHADERS_PATCH_v1__" + u32 block_offset + u32 block_version

**What is deliberately not claimed.** The interiors of the opaque regions are
unknown and stay that way. Their *sizes* are verified; their *contents* are not
decoded, and no field in them has been given a name. Naming them would convert an
unverified guess into something that reads like documentation, which is the
failure this project is organised against.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "CAR_FRAME_BASE_BYTES",
    "CAR_FRAME_BYTES_PER_WING",
    "CSP_FOOTER_LENGTH",
    "CSP_MARKER",
    "GLOBAL_FRAME_BASE_BYTES",
    "LAP_RECORD_BYTES",
    "PER_CAR_EXTRA_RECORD_BYTES",
    "SUPPORTED_VERSIONS",
    "TRACK_OBJECT_BYTES_PER_FRAME",
    "car_frame_stride",
]

SUPPORTED_VERSIONS: Final[tuple[int, ...]] = (16,)
"""Replay versions this reader will decode.

A hard gate, not a preference. Versions 14 and 15 exist and have a different
header — the eight bytes this reader interprets as a recording interval are not
that in earlier versions. Decoding them "best effort" without a fixture to verify
against would produce a confident, wrong session.
"""

CAR_FRAME_BASE_BYTES: Final = 276
CAR_FRAME_BYTES_PER_WING: Final = 4
"""The per-frame car record grows with the car's wing count.

Verified across both values present in the fixtures: 288 bytes at three wings
(fifteen cars) and 296 at five (two cars). Published parsers assume a fixed
record and therefore silently corrupt every channel of any car whose wing count
differs from the one they were written against.
"""

GLOBAL_FRAME_BASE_BYTES: Final = 4
TRACK_OBJECT_BYTES_PER_FRAME: Final = 12
"""Size of the per-frame global block, which scales with animated track objects.

Both fixtures' tracks have zero track objects, so the 12-byte term is confirmed
only by arithmetic consistency with the two implementations that agree on it,
never exercised against a non-zero case. Flagged here so a future failure on a
track with animated objects is diagnosed in seconds rather than hours.
"""

PER_CAR_EXTRA_RECORD_BYTES: Final = 8
"""Size of each entry in the per-car trailing list.

Present only on the player's car in every fixture (counts of 3, 4 and 5); all AI
cars declare zero.
"""

LAP_RECORD_BYTES: Final = 20
"""Size of each entry in the session-level trailing table.

The table is ``car_count x lap_record_count`` entries. The count tracked session
length across the fixtures (4, 5 and 6 for progressively longer sessions), and
the first entry observed decodes as four zero words followed by the float
``-1.0`` — a pattern consistent with an unset time. That is an observation, not
a decoding: nothing here depends on what these bytes mean.
"""

CSP_MARKER: Final = b"__AC_SHADERS_PATCH_v1__"
"""Sentinel introducing the optional Custom Shaders Patch extension block."""

CSP_FOOTER_LENGTH: Final = len(CSP_MARKER) + 8
"""Bytes occupied by the trailing footer: marker, block offset, block version."""


def car_frame_stride(wing_count: int) -> int:
    """Compute the size of one car's per-frame record.

    Args:
        wing_count: The car's declared wing count, read from its header.

    Returns:
        The record size in bytes.

    Examples:
        >>> car_frame_stride(3)
        288
        >>> car_frame_stride(5)
        296
    """
    return CAR_FRAME_BASE_BYTES + CAR_FRAME_BYTES_PER_WING * wing_count
