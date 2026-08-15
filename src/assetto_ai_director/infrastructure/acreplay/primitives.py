"""Bounds-checked primitive reads over a replay file.

Every read is checked against the remaining length **before** it happens, and
every length-prefixed string is checked against a sanity limit before any
allocation.

That ordering is the point. A corrupt or truncated file will declare a string of
two billion characters or a frame count that cannot fit; a reader that trusts the
declaration and allocates first turns a bad file into an out-of-memory kill
instead of an error message naming the problem.
"""

from __future__ import annotations

import struct
from typing import Final

from assetto_ai_director.domain.errors import TruncatedReplayError

__all__ = ["MAX_STRING_LENGTH", "BinaryCursor"]

MAX_STRING_LENGTH: Final = 256
"""Longest plausible length-prefixed string in a replay.

Car models, driver names, track ids and skin folder names are all well under
this. A declared length above it means the cursor is not where it thinks it is,
which is worth failing on immediately rather than reading 2 GB of noise.
"""

_U32 = struct.Struct("<I")
_I32 = struct.Struct("<i")
_F64 = struct.Struct("<d")


class BinaryCursor:
    """A position within a byte buffer, with checked little-endian reads.

    Assetto Corsa is a Windows x86 title and its replay data is little-endian
    throughout; that is assumed rather than detected, and it is the one
    assumption in this module that no fixture could falsify.

    Attributes:
        data: The buffer being read.
        offset: The current read position.
    """

    __slots__ = ("data", "offset")

    def __init__(self, data: bytes | memoryview, offset: int = 0) -> None:
        """Create a cursor over a buffer.

        Args:
            data: The bytes to read.
            offset: Starting position.
        """
        self.data = data
        self.offset = offset

    @property
    def remaining(self) -> int:
        """How many bytes lie ahead of the cursor.

        Returns:
            The count of unread bytes.
        """
        return len(self.data) - self.offset

    def require(self, count: int, what: str) -> None:
        """Assert that at least ``count`` bytes remain.

        Args:
            count: Bytes needed.
            what: Description of the structure being read, for the error.

        Raises:
            TruncatedReplayError: If fewer bytes remain.
        """
        if count > self.remaining:
            raise TruncatedReplayError(what, count, self.remaining)

    def skip(self, count: int, what: str = "padding") -> None:
        """Advance past a region without interpreting it.

        Used for the parts of the format whose meaning is unknown. Skipping is
        deliberate: naming those bytes would imply an understanding this project
        does not have.

        Args:
            count: Bytes to skip.
            what: Description, for the error if the file is too short.

        Raises:
            TruncatedReplayError: If the region does not fit.
        """
        self.require(count, what)
        self.offset += count

    def u32(self, what: str = "uint32") -> int:
        """Read an unsigned 32-bit integer.

        Args:
            what: Description, for the error if the file is too short.

        Returns:
            The value read.

        Raises:
            TruncatedReplayError: If four bytes do not remain.
        """
        self.require(4, what)
        (value,) = _U32.unpack_from(self.data, self.offset)
        self.offset += 4
        return int(value)

    def i32(self, what: str = "int32") -> int:
        """Read a signed 32-bit integer.

        Args:
            what: Description, for the error if the file is too short.

        Returns:
            The value read.

        Raises:
            TruncatedReplayError: If four bytes do not remain.
        """
        self.require(4, what)
        (value,) = _I32.unpack_from(self.data, self.offset)
        self.offset += 4
        return int(value)

    def f64(self, what: str = "float64") -> float:
        """Read a double-precision float.

        Args:
            what: Description, for the error if the file is too short.

        Returns:
            The value read.

        Raises:
            TruncatedReplayError: If eight bytes do not remain.
        """
        self.require(8, what)
        (value,) = _F64.unpack_from(self.data, self.offset)
        self.offset += 8
        return float(value)

    def string(self, what: str = "string") -> str:
        """Read a length-prefixed string.

        The encoding is a signed 32-bit length followed by that many raw bytes,
        with no terminator. Decoded as ``latin-1`` because it never fails and
        never silently substitutes characters — car and track folder names are
        ASCII in practice, and a mojibake driver name is far better than an
        exception that aborts an otherwise valid 300 MB parse.

        Args:
            what: Description, for errors.

        Returns:
            The decoded string.

        Raises:
            TruncatedReplayError: If the declared length does not fit, or is
                negative or implausibly large — both of which mean the cursor is
                misaligned rather than that the file contains a huge string.
        """
        length = self.i32(f"{what} length")
        if length < 0 or length > MAX_STRING_LENGTH:
            # Not a truncation in the literal sense, but the same diagnosis:
            # the structure is not where we believed it was.
            detail = f"{what} (implausible declared length {length})"
            raise TruncatedReplayError(detail, max(length, 0), self.remaining)
        self.require(length, what)
        raw = bytes(self.data[self.offset : self.offset + length])
        self.offset += length
        return raw.decode("latin-1")

    def optional_string(self, what: str = "string") -> str | None:
        """Read a length-prefixed string, mapping the empty string to ``None``.

        Empty strings are ubiquitous in replays — AI cars have no driver name and
        most entries have no team. Returning ``None`` keeps "absent" distinct
        from "present but blank" in the domain model.

        Args:
            what: Description, for errors.

        Returns:
            The string, or ``None`` if it was empty.

        Raises:
            TruncatedReplayError: As for :meth:`string`.
        """
        value = self.string(what)
        return value or None
