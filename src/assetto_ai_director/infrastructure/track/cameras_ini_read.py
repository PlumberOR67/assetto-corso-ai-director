"""Reading an Assetto Corsa camera set, which this project has only ever written.

The exporter has been emitting ``cameras.ini`` since early on and nothing has
ever checked its output against a real file. That is a gap of exactly the kind
this project's rules exist to close: the format details in the writer are
correct as far as anyone knows, and "as far as anyone knows" is not the standard
applied to the replay reader.

Two things a reader buys.

**Checking the writer.** A Kunos or modder camera set parsed by this and written
back out by the exporter should agree field for field. Where it does not, one of
them is wrong, and until now there was no way to find out which.

**Turning guesses into measurements.** ``SIZE_DISTANCE_METRES`` and
``RIG_HEIGHT_METRES`` are engineering judgement carrying ``GUESS`` provenance.
A real camera set, projected onto the track's own AI line, says where authored
cameras actually stand: how far back, how high, how far off the racing line.
That is a measurement, and this is the file it comes out of.

**``MIN_FOV`` is the field of view when the car is furthest** and ``MAX_FOV``
when it is closest, so ``MIN_FOV`` is routinely the larger number. The reader
keeps them under those names rather than sorting them into near and far, because
renaming a field on the way in is how a reader stops being able to check a
writer.
"""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["CameraSet", "TrackCamera", "read_cameras_ini"]

_ENCODING = "utf-8-sig"
_HEADER = "HEADER"
_VECTOR_COMPONENTS = 3

_SPAN_TOLERANCE = 1e-3
"""How far two spans may miss each other and still count as meeting."""


@dataclass(frozen=True, slots=True)
class TrackCamera:
    """One ``[CAMERA_N]`` section.

    Attributes:
        index: Which camera it is.
        name: Its ``NAME``, empty when it has none.
        position: World position in metres, y up.
        forward: Direction the camera faces. A direction, not a target point.
        up: Its up vector.
        min_fov: Field of view when the car is furthest, in degrees. Whether
            that is measured horizontally or vertically is **not documented
            anywhere this project could find**, and this reader does not decide.
        max_fov: Field of view when the car is closest.
        in_point: Normalised track position where the camera takes over.
        out_point: Where it hands over.
        spline: The ``SPLINE`` value, empty for a fixed camera.
        is_fixed: Whether the camera holds position.
        extra: Every other key in the section, verbatim. Kept rather than
            dropped so a round trip through this reader can be compared against
            the original without a diff full of fields it silently discarded.
    """

    index: int
    name: str
    position: tuple[float, float, float]
    forward: tuple[float, float, float]
    up: tuple[float, float, float]
    min_fov: float
    max_fov: float
    in_point: float
    out_point: float
    spline: str = ""
    is_fixed: bool = True
    extra: tuple[tuple[str, str], ...] = ()

    @property
    def span(self) -> float:
        """How much of the track this camera covers.

        Returns:
            ``out_point - in_point``, which is negative for a camera that wraps
            the start line — a real arrangement, and one a naive reader treats
            as an error.
        """
        return self.out_point - self.in_point


@dataclass(frozen=True, slots=True)
class CameraSet:
    """A whole ``cameras.ini`` file.

    Attributes:
        path: Where it was read from.
        version: The ``[HEADER] VERSION`` value.
        set_name: The set's name, empty when it has none.
        cameras: The cameras, in index order.
        declared_count: What ``CAMERA_COUNT`` claimed. Kept separately from the
            number actually parsed, because a file whose header disagrees with
            its body is a fact about the file rather than something to correct
            silently.
    """

    path: Path
    version: int
    set_name: str
    cameras: tuple[TrackCamera, ...]
    declared_count: int

    @property
    def count_agrees(self) -> bool:
        """Whether the header's count matches the sections present.

        Returns:
            ``True`` when they agree.
        """
        return self.declared_count == len(self.cameras)

    @property
    def spans_are_contiguous(self) -> bool:
        """Whether each camera hands over exactly where the next takes up.

        The property this project's own exporter works to maintain, so being
        able to test a real file for it is how that effort gets checked against
        practice rather than against belief.

        Returns:
            ``True`` when consecutive spans meet, within a tolerance of a
            thousandth of the track.
        """
        ordered = sorted(self.cameras, key=lambda camera: camera.in_point)
        return all(
            abs(later.in_point - earlier.out_point) < _SPAN_TOLERANCE
            for earlier, later in pairwise(ordered)
        )


def read_cameras_ini(path: Path) -> CameraSet:
    """Read an Assetto Corsa camera set.

    Args:
        path: Path to a ``cameras*.ini``.

    Returns:
        The parsed set.

    Raises:
        ValueError: If the file is not readable as ini, or a camera section is
            missing a field with no sensible reading. Absent optional fields are
            defaulted; absent geometry is not.
        OSError: If the file cannot be read.
    """
    parser = configparser.ConfigParser(strict=False)
    try:
        parser.read_string(path.read_text(encoding=_ENCODING, errors="replace"))
    except configparser.Error as error:
        message = f"{path} is not a readable ini file: {error}"
        raise ValueError(message) from error

    header = parser[_HEADER] if parser.has_section(_HEADER) else {}
    cameras = [
        _camera(parser[section], section, path)
        for section in parser.sections()
        if section.upper().startswith("CAMERA_")
    ]
    cameras.sort(key=lambda camera: camera.index)

    return CameraSet(
        path=path,
        version=int(_first_number(header.get("VERSION", "0"))),
        set_name=header.get("SET_NAME", ""),
        cameras=tuple(cameras),
        declared_count=int(_first_number(header.get("CAMERA_COUNT", str(len(cameras))))),
    )


def _camera(section: configparser.SectionProxy, name: str, path: Path) -> TrackCamera:
    """Parse one camera section.

    Args:
        section: The section.
        name: Its name, for the index and for error messages.
        path: Which file, for error messages.

    Returns:
        The camera.

    Raises:
        ValueError: If a geometry field is missing or malformed.
    """
    known = {
        "NAME",
        "POSITION",
        "FORWARD",
        "UP",
        "MIN_FOV",
        "MAX_FOV",
        "IN_POINT",
        "OUT_POINT",
        "SPLINE",
        "IS_FIXED",
    }
    return TrackCamera(
        index=_index(name),
        name=section.get("NAME", ""),
        position=_vector(section, "POSITION", name, path),
        forward=_vector(section, "FORWARD", name, path),
        up=_vector(section, "UP", name, path, default=(0.0, 1.0, 0.0)),
        min_fov=_number(section, "MIN_FOV", name, path),
        max_fov=_number(section, "MAX_FOV", name, path),
        in_point=_number(section, "IN_POINT", name, path),
        out_point=_number(section, "OUT_POINT", name, path),
        spline=section.get("SPLINE", ""),
        is_fixed=_first_number(section.get("IS_FIXED", "1")) != 0.0,
        extra=tuple(
            (key.upper(), value) for key, value in section.items() if key.upper() not in known
        ),
    )


def _index(section: str) -> int:
    """Pull the camera number out of a section name.

    Args:
        section: A name like ``CAMERA_3``.

    Returns:
        The number, or ``0`` when the name does not carry one.
    """
    tail = section.split("_", 1)[-1]
    return int(tail) if tail.isdigit() else 0


def _vector(
    section: configparser.SectionProxy,
    key: str,
    name: str,
    path: Path,
    default: tuple[float, float, float] | None = None,
) -> tuple[float, float, float]:
    """Parse a comma-separated three-component value.

    Args:
        section: The section.
        key: Which key.
        name: The section's name, for the message.
        path: Which file, for the message.
        default: What to return when the key is absent. Omit it to make the key
            required.

    Returns:
        The three components.

    Raises:
        ValueError: If the key is absent with no default, or does not hold three
            numbers.
    """
    raw = section.get(key)
    if raw is None:
        if default is not None:
            return default
        message = f"{path} section [{name}] has no {key}."
        raise ValueError(message)

    parts = [part.strip() for part in raw.split(";")[0].split(",")]
    if len(parts) != _VECTOR_COMPONENTS:
        message = f"{path} section [{name}] has {key}={raw!r}, which is not three numbers."
        raise ValueError(message)
    try:
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError as error:
        message = f"{path} section [{name}] has {key}={raw!r}, which is not three numbers."
        raise ValueError(message) from error


def _number(section: configparser.SectionProxy, key: str, name: str, path: Path) -> float:
    """Parse one required numeric value.

    Args:
        section: The section.
        key: Which key.
        name: The section's name, for the message.
        path: Which file, for the message.

    Returns:
        The value.

    Raises:
        ValueError: If the key is absent or not a number.
    """
    raw = section.get(key)
    if raw is None:
        message = f"{path} section [{name}] has no {key}."
        raise ValueError(message)
    try:
        return _first_number(raw)
    except ValueError as error:
        message = f"{path} section [{name}] has {key}={raw!r}, which is not a number."
        raise ValueError(message) from error


def _first_number(raw: str) -> float:
    """Read a number, ignoring a trailing inline comment.

    Args:
        raw: The text.

    Returns:
        The value.

    Raises:
        ValueError: If it is not a number.
    """
    return float(raw.split(";", 1)[0].strip())
