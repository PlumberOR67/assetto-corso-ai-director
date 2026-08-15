"""The world-to-map transform, which is the answer to "the map is wrong".

Every Assetto Corsa track ships a ``map.png`` and a ``map.ini`` describing how
world coordinates land on it. That pair is the only reference this project can
check its own top-down drawing against — until now the track map drew a driven
line auto-fitted to its own bounds, which looks right in every case including the
ones where it is not, because there was nothing beside it to disagree with.

The transform, from the file's own ``[PARAMETERS]``::

    pixel_x = (world_x + X_OFFSET) / SCALE_FACTOR
    pixel_y = (world_z + Z_OFFSET) / SCALE_FACTOR

``y`` is elevation and takes no part in it: the map is a plan view. The inverse
is the same arithmetic run backwards, and both directions are provided because a
reader wanting to place a camera marker on the image needs one and a reader
wanting to know what a pixel means needs the other.

**Every field is required and none is defaulted.** A missing ``SCALE_FACTOR``
silently treated as ``1.0`` produces a map that is wrong by a factor of a hundred
and looks like a track drawn very small — which is indistinguishable from a
correct map of a short track. The reader raises instead.
"""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["MapTransform", "read_map_ini"]

_SECTION = "PARAMETERS"
_ENCODING = "utf-8-sig"
"""Windows tooling writes a byte-order mark; utf-8 alone leaves it in the first key."""

_REQUIRED = ("WIDTH", "HEIGHT", "X_OFFSET", "Z_OFFSET", "SCALE_FACTOR")


@dataclass(frozen=True, slots=True)
class MapTransform:
    """How a track's world coordinates land on its map image.

    Attributes:
        width: Image width in pixels.
        height: Image height in pixels.
        x_offset: Added to world ``x`` before scaling, in metres.
        z_offset: Added to world ``z`` before scaling, in metres.
        scale_factor: Metres per pixel.
        margin: Border the image was drawn with, in pixels. Recorded because it
            is in the file, not used — the offsets already account for it.
    """

    width: float
    height: float
    x_offset: float
    z_offset: float
    scale_factor: float
    margin: float = 0.0

    def to_pixels(self, position: np.ndarray) -> np.ndarray:
        """Convert world positions to map pixels.

        Args:
            position: World positions, shape ``(n, 3)`` or ``(3,)``, y up.

        Returns:
            Pixel coordinates, shape ``(n, 2)``, with the image's origin at its
            top left.
        """
        flat = np.atleast_2d(position)
        return np.stack(
            [
                (flat[:, 0] + self.x_offset) / self.scale_factor,
                (flat[:, 2] + self.z_offset) / self.scale_factor,
            ],
            axis=1,
        )

    def to_world(self, pixels: np.ndarray) -> np.ndarray:
        """Convert map pixels back to world coordinates.

        Args:
            pixels: Pixel coordinates, shape ``(n, 2)``.

        Returns:
            World positions on the ground plane, shape ``(n, 3)``, with ``y``
            zero — the map carries no elevation and inventing one would be the
            kind of plausible fiction this project refuses.
        """
        flat = np.atleast_2d(pixels)
        return np.stack(
            [
                flat[:, 0] * self.scale_factor - self.x_offset,
                np.zeros(flat.shape[0]),
                flat[:, 1] * self.scale_factor - self.z_offset,
            ],
            axis=1,
        )

    def covers(self, position: np.ndarray) -> bool:
        """Test whether world positions land inside the image.

        The check that answers "is the map positioned incorrectly": a track's own
        driven line must plot inside its own map, and if it does not, either the
        transform was read wrong or the coordinates are not what they are assumed
        to be.

        Args:
            position: World positions, shape ``(n, 3)``.

        Returns:
            ``True`` when every point falls within the image bounds.
        """
        pixels = self.to_pixels(position)
        inside_x = np.all((pixels[:, 0] >= 0.0) & (pixels[:, 0] <= self.width))
        inside_y = np.all((pixels[:, 1] >= 0.0) & (pixels[:, 1] <= self.height))
        return bool(inside_x and inside_y)


def read_map_ini(path: Path) -> MapTransform:
    """Read a track's map transform.

    Args:
        path: Path to ``map.ini``.

    Returns:
        The transform.

    Raises:
        ValueError: If the file has no ``[PARAMETERS]`` section, is missing a
            field the transform needs, or holds a value that is not a number.
            Nothing is defaulted: a missing ``SCALE_FACTOR`` read as ``1.0``
            produces a map wrong by a factor of a hundred that looks exactly
            like a correct map of a very short track.
        OSError: If the file cannot be read.
    """
    parser = configparser.ConfigParser(strict=False)
    try:
        parser.read_string(path.read_text(encoding=_ENCODING, errors="replace"))
    except configparser.Error as error:
        message = f"{path} is not a readable ini file: {error}"
        raise ValueError(message) from error

    if not parser.has_section(_SECTION):
        message = f"{path} has no [{_SECTION}] section, so it carries no map transform."
        raise ValueError(message)

    section = parser[_SECTION]
    missing = [key for key in _REQUIRED if key not in section]
    if missing:
        message = (
            f"{path} is missing {', '.join(missing)}. Refusing to default them: "
            f"a guessed scale produces a map that is wrong and looks right."
        )
        raise ValueError(message)

    values = {key: _number(section[key], key, path) for key in _REQUIRED}
    if values["SCALE_FACTOR"] <= 0.0:
        message = f"{path} declares a scale factor of {values['SCALE_FACTOR']}, which cannot be."
        raise ValueError(message)

    margin = _number(section["MARGIN"], "MARGIN", path) if "MARGIN" in section else 0.0
    return MapTransform(
        width=values["WIDTH"],
        height=values["HEIGHT"],
        x_offset=values["X_OFFSET"],
        z_offset=values["Z_OFFSET"],
        scale_factor=values["SCALE_FACTOR"],
        margin=margin,
    )


def _number(raw: str, key: str, path: Path) -> float:
    """Parse one value, saying which one failed.

    Args:
        raw: The text from the file.
        key: Which key it came from.
        path: Which file, for the message.

    Returns:
        The value.

    Raises:
        ValueError: If it is not a number.
    """
    try:
        return float(raw.split(";", 1)[0].strip())
    except ValueError as error:
        message = f"{path} has {key}={raw!r}, which is not a number."
        raise ValueError(message) from error
