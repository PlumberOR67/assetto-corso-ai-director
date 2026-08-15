"""Structural parse of an ``.acreplay`` file, and the self-verification it enables.

This is tier one of the reader: it walks the whole file's *structure* — header,
per-car headers, region sizes and offsets — without decoding a single frame. On a
327 MB replay it touches a few kilobytes.

That is useful in itself (it is what populates a session browser), but its more
important job is **proving the layout assumption before any telemetry is
believed**. The walk ends by checking that it has arrived at exactly the offset
the file records in its own Custom Shaders Patch footer. If the record stride
were wrong by a single byte, the accumulated error over tens of thousands of
frames would be enormous and the check would fail loudly — instead of the reader
happily emitting a smooth, plausible, completely fictitious race.
"""

from __future__ import annotations

import mmap
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from assetto_ai_director.domain.errors import (
    StrideMismatchError,
    TruncatedReplayError,
    UnsupportedReplayVersionError,
)
from assetto_ai_director.domain.identity import CarIdentity
from assetto_ai_director.domain.session import SessionMeta
from assetto_ai_director.domain.units import Hertz
from assetto_ai_director.infrastructure.acreplay.layout import (
    CSP_FOOTER_LENGTH,
    CSP_MARKER,
    GLOBAL_FRAME_BASE_BYTES,
    LAP_RECORD_BYTES,
    PER_CAR_EXTRA_RECORD_BYTES,
    SUPPORTED_VERSIONS,
    TRACK_OBJECT_BYTES_PER_FRAME,
    car_frame_stride,
)
from assetto_ai_director.infrastructure.acreplay.primitives import BinaryCursor

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["CarBlock", "ReplayStructure", "map_replay", "read_structure"]

_MILLISECONDS_PER_SECOND = 1000.0


@dataclass(frozen=True, slots=True)
class CarBlock:
    """Where one car's data lives in the file, and how to stride through it.

    Attributes:
        identity: Who the car is.
        frame_count: Number of per-frame records. One fewer than the session's
            declared frame count, in every fixture examined.
        wing_count: Declared wing count, which determines the record stride.
        frames_offset: Absolute file offset of the first frame record.
        extra_count: Length of this car's trailing list. Non-zero only for the
            player's car in every fixture, which is how the reader identifies the
            player structurally instead of assuming car zero.
    """

    identity: CarIdentity
    frame_count: int
    wing_count: int
    frames_offset: int
    extra_count: int

    @property
    def stride(self) -> int:
        """Bytes per frame record for this car.

        Returns:
            ``276 + 4 * wing_count``.
        """
        return car_frame_stride(self.wing_count)

    @property
    def frames_length(self) -> int:
        """Total bytes occupied by this car's frame records.

        Returns:
            The region length in bytes.
        """
        return self.frame_count * self.stride


@dataclass(frozen=True, slots=True)
class ReplayStructure:
    """The verified structural map of a replay file.

    Attributes:
        path: The file this was read from.
        version: Replay format version. Always in
            :data:`~assetto_ai_director.infrastructure.acreplay.layout.SUPPORTED_VERSIONS`
            once construction succeeds.
        recording_interval_ms: Milliseconds between recorded frames.
        weather_id: Weather preset name.
        track_id: Track folder name.
        track_config: Track layout name. Empty for single-layout tracks.
        frame_count: Declared session frame count.
        current_recording_index: The game's ring-buffer write position.
        track_object_count: Number of animated track objects.
        cars: One entry per car, in file order.
        csp_block_offset: Offset of the Custom Shaders Patch block, or ``None``
            if the file has no CSP footer.
        structural_end: Offset the structural walk finished at.
        file_size: Total file length.
    """

    path: Path
    version: int
    recording_interval_ms: float
    weather_id: str
    track_id: str
    track_config: str
    frame_count: int
    current_recording_index: int
    track_object_count: int
    cars: tuple[CarBlock, ...]
    csp_block_offset: int | None
    structural_end: int
    file_size: int

    @property
    def sample_rate(self) -> Hertz:
        """The session's sampling rate.

        Returns:
            Frames per second, derived from the recording interval.
        """
        if self.recording_interval_ms <= 0.0:
            return Hertz(0.0)
        return Hertz(_MILLISECONDS_PER_SECOND / self.recording_interval_ms)

    @property
    def wrapped(self) -> bool:
        """Whether the game's ring buffer overwrote the start of the session.

        When the recording index does not sit at the end of the buffer, the
        oldest frames have been overwritten by newer ones, so file order is not
        chronological order. Reading such a file as a simple sequence splices the
        session's end onto its front.

        Returns:
            ``True`` if the buffer wrapped.
        """
        return self.current_recording_index != self.frame_count

    @property
    def verified(self) -> bool:
        """Whether the structural walk landed exactly where the file says it should.

        Returns:
            ``True`` if the walk ended at the CSP block offset, or at end of file
            for a replay with no CSP block.
        """
        expected = self.csp_block_offset if self.csp_block_offset is not None else self.file_size
        return self.structural_end == expected

    def to_session_meta(self, source_name: str = "acreplay") -> SessionMeta:
        """Project this structure onto the domain's session metadata.

        Args:
            source_name: Name to record as the providing source.

        Returns:
            The session metadata.
        """
        return SessionMeta(
            track_id=self.track_id,
            layout=self.track_config,
            weather_id=self.weather_id,
            sample_rate=self.sample_rate,
            frame_count=self.frame_count,
            cars=tuple(block.identity for block in self.cars),
            source_name=source_name,
            wrapped=self.wrapped,
        )


@contextmanager
def map_replay(path: Path) -> Iterator[memoryview]:
    """Memory-map a replay file for reading.

    Replays run to hundreds of megabytes. Mapping rather than reading keeps
    resident memory proportional to what is actually touched, which matters
    because the structural walk needs only a few kilobytes of a 327 MB file.

    Args:
        path: The replay to map.

    Yields:
        A read-only view of the file's bytes.

    Raises:
        OSError: If the file cannot be opened or mapped.
    """
    with (
        path.open("rb") as handle,
        mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped,
    ):
        view = memoryview(mapped)
        try:
            yield view
        finally:
            view.release()


def _find_csp_block_offset(data: memoryview, file_size: int) -> int | None:
    """Locate the Custom Shaders Patch block via the file's trailing footer.

    Args:
        data: The mapped file.
        file_size: Total file length.

    Returns:
        The block offset, or ``None`` if there is no CSP footer or it does not
        point somewhere sane.
    """
    if file_size < CSP_FOOTER_LENGTH:
        return None
    footer_start = file_size - CSP_FOOTER_LENGTH
    if bytes(data[footer_start : footer_start + len(CSP_MARKER)]) != CSP_MARKER:
        return None
    cursor = BinaryCursor(data, footer_start + len(CSP_MARKER))
    offset = cursor.u32("CSP block offset")
    cursor.u32("CSP block version")
    # A footer that points outside the file is a footer we do not understand;
    # treat it as absent rather than trusting it as a verification target.
    return offset if 0 < offset <= footer_start else None


def read_structure(path: Path) -> ReplayStructure:
    """Walk a replay's structure and verify the layout against the file itself.

    Args:
        path: The replay to read.

    Returns:
        The verified structural map.

    Raises:
        UnsupportedReplayVersionError: If the version is not supported.
        TruncatedReplayError: If any declared structure does not fit.
        StrideMismatchError: If the walk does not finish exactly where the file's
            own footer says the data ends — meaning the record layout assumption
            is wrong and no telemetry from this file can be trusted.
    """
    with map_replay(path) as data:
        file_size = len(data)
        cursor = BinaryCursor(data)

        version = cursor.u32("version")
        if version not in SUPPORTED_VERSIONS:
            raise UnsupportedReplayVersionError(version, SUPPORTED_VERSIONS)

        recording_interval_ms = cursor.f64("recording interval")
        weather_id = cursor.string("weather")
        track_id = cursor.string("track")
        track_config = cursor.string("track config")
        car_count = cursor.u32("car count")
        current_recording_index = cursor.u32("current recording index")
        frame_count = cursor.u32("frame count")
        track_object_count = cursor.u32("track object count")

        _validate_counts(car_count, frame_count, track_object_count, file_size)

        global_block = frame_count * (
            GLOBAL_FRAME_BASE_BYTES + TRACK_OBJECT_BYTES_PER_FRAME * track_object_count
        )
        cursor.skip(global_block, "global per-frame block")

        cars = tuple(_read_car_block(cursor, index) for index in range(car_count))

        lap_record_count = cursor.u32("lap record count")
        cursor.skip(car_count * lap_record_count * LAP_RECORD_BYTES, "session lap record table")

        structural_end = cursor.offset
        csp_block_offset = _find_csp_block_offset(data, file_size)

        expected_end = csp_block_offset if csp_block_offset is not None else file_size
        if structural_end != expected_end:
            raise StrideMismatchError(
                expected_end,
                structural_end,
                "car frame stride is 276 + 4 * wing_count; a mismatch here means "
                "that assumption does not hold for this file",
            )

        return ReplayStructure(
            path=path,
            version=version,
            recording_interval_ms=recording_interval_ms,
            weather_id=weather_id,
            track_id=track_id,
            track_config=track_config,
            frame_count=frame_count,
            current_recording_index=current_recording_index,
            track_object_count=track_object_count,
            cars=cars,
            csp_block_offset=csp_block_offset,
            structural_end=structural_end,
            file_size=file_size,
        )


def _validate_counts(
    car_count: int, frame_count: int, track_object_count: int, file_size: int
) -> None:
    """Sanity-check declared counts before anything is sized from them.

    Args:
        car_count: Declared car count.
        frame_count: Declared frame count.
        track_object_count: Declared track object count.
        file_size: Total file length.

    Raises:
        TruncatedReplayError: If a count could not possibly fit in the file. The
            smallest per-car record is 276 bytes, so the product of counts gives
            a hard lower bound that a corrupt header will violate immediately.
    """
    if car_count < 0 or frame_count < 0 or track_object_count < 0:
        negative = "header counts (negative)"
        raise TruncatedReplayError(negative, 0, file_size)
    # One byte per car-frame is a deliberately conservative floor: the true
    # record is at least 276 bytes, so anything failing this test is nonsense by
    # a wide margin rather than merely optimistic.
    minimum_bytes = car_count * frame_count
    if minimum_bytes > file_size:
        detail = f"{car_count} cars x {frame_count} frames"
        raise TruncatedReplayError(detail, minimum_bytes, file_size)


def _read_car_block(cursor: BinaryCursor, index: int) -> CarBlock:
    """Read one car's header and skip its frame region.

    Args:
        cursor: Positioned at the start of the car's header.
        index: The car's index in the session.

    Returns:
        The car block, with the cursor advanced past it.

    Raises:
        TruncatedReplayError: If the car's declared regions do not fit.
    """
    model = cursor.string(f"car {index} model")
    driver = cursor.optional_string(f"car {index} driver")
    nationality = cursor.optional_string(f"car {index} nationality")
    team = cursor.optional_string(f"car {index} team")
    skin = cursor.optional_string(f"car {index} skin")

    frame_count = cursor.u32(f"car {index} frame count")
    wing_count = cursor.u32(f"car {index} wing count")

    frames_offset = cursor.offset
    cursor.skip(frame_count * car_frame_stride(wing_count), f"car {index} frame records")

    extra_count = cursor.u32(f"car {index} extra count")
    cursor.skip(extra_count * PER_CAR_EXTRA_RECORD_BYTES, f"car {index} extra records")

    return CarBlock(
        identity=CarIdentity(
            car_id=index,
            model=model,
            driver_name=driver,
            nationality=nationality,
            team=team,
            skin_id=skin,
            # Only the player's car carries trailing extra records in every
            # fixture examined. Structural, rather than assuming car zero.
            is_player=extra_count > 0,
        ),
        frame_count=frame_count,
        wing_count=wing_count,
        frames_offset=frames_offset,
        extra_count=extra_count,
    )
