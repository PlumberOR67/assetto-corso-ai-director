"""Read and write telemetry as CSV.

Two jobs. It is the **debug format** — a decoded session written out is something
you can open in a spreadsheet and check by eye, which is worth a great deal when
the alternative is trusting a binary reader on faith. And it is the **escape
hatch**: anything that can produce a table of position and time can feed this
director, including simulators this project knows nothing about.

**The schema is defined here, deliberately, rather than sniffed.** There is no
common convention for sim-racing CSV — every logging tool emits a different shape,
and some split one session across several files. Accepting "sim racing CSV"
generically means guessing which column is which, and a wrong guess produces
smooth, plausible, entirely wrong telemetry that no downstream check would catch.
So this defines one explicit schema, documents it, and rejects anything else with
a message naming the columns it wanted.

Heading is optional on import, and that option has a consequence worth stating
plainly. Derived heading is the path tangent, which is exact for a car that is not
sliding and increasingly wrong for one that is. Sideslip computed from it is
identically zero, so a file without a heading column yields a session in which
drift analysis will find nothing — not because nothing happened, but because the
file could not say. The reader records that in :attr:`CsvSession.notes` and
declares the session's rotation uncalibrated, so the difference reaches the
consumer instead of being smoothed over.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from assetto_ai_director.application.events.detect import CarSignals
from assetto_ai_director.domain.frames import ReferenceFrame
from assetto_ai_director.domain.identity import CarIdentity
from assetto_ai_director.domain.session import SessionMeta
from assetto_ai_director.domain.telemetry import Channel, ChannelAvailability
from assetto_ai_director.domain.units import Hertz, Seconds

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

__all__ = [
    "CSV_COLUMNS",
    "DERIVABLE_COLUMNS",
    "REQUIRED_COLUMNS",
    "CsvSession",
    "read_csv",
    "write_csv",
]

CSV_COLUMNS: tuple[str, ...] = (
    "time_s",
    "car_id",
    "x",
    "y",
    "z",
    "speed_mps",
    "heading_rad",
    "sideslip_rad",
)
"""The full schema this module writes and understands.

Positions are Assetto Corsa's world axes in metres, with ``y`` up. Angles are
radians. ``time_s`` counts from the start of the session, and every car shares
one time base — index *i* is the same instant for every car, which is what makes
multi-car analysis a vectorised comparison rather than a resampling problem.
"""

REQUIRED_COLUMNS: tuple[str, ...] = ("time_s", "car_id", "x", "y", "z")
"""Columns without which there is no session. Everything else can be derived."""

DERIVABLE_COLUMNS: tuple[str, ...] = ("speed_mps", "heading_rad", "sideslip_rad")
"""Columns computed from position when absent, and named in the notes when they are."""

_MIN_SAMPLES = 2
"""Two samples are the minimum from which a sampling interval can be measured."""

_DEFAULT_RATE = 1.0
"""Only reached for a session too short to measure, which :func:`read_csv` rejects."""


@dataclass(frozen=True, slots=True)
class CsvSession:
    """A session read from CSV, carrying what had to be derived.

    Deliberately the same shape as the synthetic provider's session, so a CSV
    file is a drop-in input to the director with no adapter in between.

    Attributes:
        signals: Decoded per-car signals, keyed by car id.
        meta: Session metadata.
        time: Time per frame, seconds.
        sample_rate: Frames per second, measured from the time column.
        derived: Names of columns that were computed rather than read.
    """

    signals: dict[int, CarSignals]
    meta: SessionMeta
    time: np.ndarray
    sample_rate: Hertz
    derived: tuple[str, ...] = field(default_factory=tuple)

    @property
    def duration(self) -> Seconds:
        """How long the session runs.

        Returns:
            Duration in seconds.
        """
        return Seconds(float(self.time[-1] - self.time[0]) if self.time.size else 0.0)

    @property
    def availability(self) -> ChannelAvailability:
        """Declare what this session actually provides.

        Returns:
            The capability declaration. Rotation counts as calibrated only when
            heading was read from the file; a path tangent computed here is not
            a measured chassis angle and must not be presented as one.
        """
        return ChannelAvailability.of(
            "csv",
            (Channel.POSITION, Channel.SPEED, Channel.HEADING),
            velocity_frame=ReferenceFrame.WORLD,
            rotation_calibrated="heading_rad" not in self.derived,
        )

    @property
    def notes(self) -> tuple[str, ...]:
        """Caveats a consumer needs in order to read the results correctly.

        Returns:
            One note per derived column, plus the consequence when the derived
            column is heading — an analysis told nothing would otherwise conclude
            the driving was clean.
        """
        if not self.derived:
            return ()
        notes = [f"Derived rather than read from the CSV: {', '.join(self.derived)}."]
        if "heading_rad" in self.derived:
            notes.append(
                "Heading was derived from the path, so sideslip is identically zero "
                "and drift detection will find nothing. Supply a heading_rad column "
                "to analyse drift."
            )
        return tuple(notes)


def write_csv(
    destination: Path,
    signals: Mapping[int, CarSignals],
    time: np.ndarray,
) -> None:
    """Write decoded telemetry as CSV.

    Args:
        destination: Where to write. Parent directories are created.
        signals: Decoded signals per car.
        time: Time per frame, seconds.

    Raises:
        OSError: If the destination cannot be written.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for car_id in sorted(signals):
            car = signals[car_id]
            # Cars can legitimately hold fewer samples than the time base — a car
            # that joined late, or a truncated decode. Writing past the shorter of
            # the two would pair a position with someone else's timestamp.
            limit = min(int(time.shape[0]), int(car.position.shape[0]))
            for index in range(limit):
                writer.writerow(
                    (
                        # Six decimals, not four. The sampling rate is recovered
                        # from the *interval* between timestamps, and at 33 Hz
                        # that interval is 30 ms — so rounding to 0.1 ms is a
                        # 0.3% error in the rate, which then rescales every
                        # derived speed and curvature in the session.
                        f"{float(time[index]):.6f}",
                        car_id,
                        f"{float(car.position[index, 0]):.4f}",
                        f"{float(car.position[index, 1]):.4f}",
                        f"{float(car.position[index, 2]):.4f}",
                        f"{float(car.speed[index]):.4f}",
                        f"{float(car.heading[index]):.6f}",
                        f"{float(car.sideslip[index]):.6f}",
                    )
                )


def read_csv(source: Path, *, track: str = "csv", layout: str = "") -> CsvSession:
    """Read telemetry from CSV.

    Args:
        source: The file to read.
        track: Track name to record in the metadata. There is no column for it —
            a CSV carries telemetry, not track identity — so the caller supplies
            it or the session is labelled ``csv``.
        layout: Track layout to record in the metadata.

    Returns:
        The parsed session, with any derived columns named in
        :attr:`CsvSession.derived`.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If the file is empty, a required column is missing, a value
            is not numeric, or there are too few samples to measure a rate.
    """
    with source.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        message = f"{source.name} contains no data rows."
        raise ValueError(message)

    missing = [name for name in REQUIRED_COLUMNS if name not in rows[0]]
    if missing:
        message = (
            f"{source.name} is missing required column(s): {', '.join(missing)}. "
            f"The expected schema is: {', '.join(CSV_COLUMNS)} — of which "
            f"{', '.join(DERIVABLE_COLUMNS)} may be omitted and will be derived."
        )
        raise ValueError(message)

    by_car = _group_by_car(rows, source.name)
    time = _time_base(by_car)
    if time.shape[0] < _MIN_SAMPLES:
        message = (
            f"{source.name} holds {time.shape[0]} sample(s); at least "
            f"{_MIN_SAMPLES} are needed to measure a sampling rate."
        )
        raise ValueError(message)

    rate = _rate_from(time, source.name)
    derived: set[str] = set()
    signals = {
        car_id: _car_from(car_id, car_rows, rate, derived, source.name)
        for car_id, car_rows in sorted(by_car.items())
    }

    return CsvSession(
        signals=signals,
        meta=_meta_for(signals, track, layout, rate, int(time.shape[0])),
        time=time,
        sample_rate=Hertz(rate),
        derived=tuple(sorted(derived)),
    )


def _group_by_car(rows: list[dict[str, str]], name: str) -> dict[int, list[dict[str, str]]]:
    """Split rows by car, preserving file order within each car.

    Args:
        rows: Every row in the file.
        name: File name, for error messages.

    Returns:
        Rows grouped by car id.

    Raises:
        ValueError: If a car id is not a number.
    """
    grouped: dict[int, list[dict[str, str]]] = {}
    for number, row in enumerate(rows, start=2):  # Row 1 is the header.
        try:
            car_id = int(float(row["car_id"]))
        except (TypeError, ValueError):
            message = f"{name} line {number}: car_id {row['car_id']!r} is not a number."
            raise ValueError(message) from None
        grouped.setdefault(car_id, []).append(row)
    return grouped


def _time_base(by_car: dict[int, list[dict[str, str]]]) -> np.ndarray:
    """Establish the shared time base.

    Taken from the car with the most samples rather than the first car, so a car
    that joined late does not truncate the session for everyone else.

    Args:
        by_car: Rows grouped by car.

    Returns:
        Time per frame, seconds.
    """
    longest = max(by_car.values(), key=len)
    return _column(longest, "time_s")


def _rate_from(time: np.ndarray, name: str) -> float:
    """Measure the sampling rate from the time column.

    Uses the median interval, not the mean: a single dropped or duplicated
    timestamp shifts a mean and leaves a median alone, and a wrong rate silently
    rescales every speed and every event time downstream.

    Args:
        time: Time per frame.
        name: File name, for error messages.

    Returns:
        Frames per second.

    Raises:
        ValueError: If the timestamps do not advance.
    """
    interval = float(np.median(np.diff(time)))
    if interval <= 0.0:
        message = (
            f"{name} has a non-increasing time column (median interval "
            f"{interval:.6f} s). Rows must be in time order."
        )
        raise ValueError(message)
    return 1.0 / interval


def _car_from(
    car_id: int,
    rows: list[dict[str, str]],
    rate: float,
    derived: set[str],
    name: str,
) -> CarSignals:
    """Build one car's signals from its rows.

    Args:
        car_id: The car's index.
        rows: Its rows, in file order.
        rate: Sampling rate.
        derived: Set to record derived column names into.
        name: File name, for error messages.

    Returns:
        The car's signals.

    Raises:
        ValueError: If a value is not numeric.
    """
    position = np.stack(
        [_column(rows, "x", name), _column(rows, "y", name), _column(rows, "z", name)],
        axis=1,
    )

    if _has_column(rows, "speed_mps"):
        speed = _column(rows, "speed_mps", name)
    else:
        derived.add("speed_mps")
        speed = _speed_from(position, rate)

    bearing = _bearing_from(position)
    if _has_column(rows, "heading_rad"):
        heading = _column(rows, "heading_rad", name)
    else:
        derived.add("heading_rad")
        heading = bearing

    if _has_column(rows, "sideslip_rad"):
        sideslip = _column(rows, "sideslip_rad", name)
    else:
        derived.add("sideslip_rad")
        sideslip = _wrap(bearing - heading)

    return CarSignals(car_id, position, heading, speed, sideslip)


def _meta_for(
    signals: dict[int, CarSignals],
    track: str,
    layout: str,
    rate: float,
    frames: int,
) -> SessionMeta:
    """Assemble metadata for a parsed file.

    A CSV carries no car models and no player flag. Rather than leaving the
    session with no player at all — which would send the scoring layer looking for
    a default — the lowest car id is marked, and the model is a placeholder that
    reads as one.

    Args:
        signals: The parsed cars.
        track: Track name supplied by the caller.
        layout: Track layout supplied by the caller.
        rate: Sampling rate.
        frames: Frame count.

    Returns:
        The session metadata.
    """
    player_id = min(signals) if signals else -1
    return SessionMeta(
        track_id=track,
        layout=layout,
        weather_id="unknown",
        sample_rate=Hertz(rate),
        frame_count=frames,
        cars=tuple(
            CarIdentity(car_id=car_id, model="csv_car", is_player=car_id == player_id)
            for car_id in sorted(signals)
        ),
        source_name="csv",
    )


def _has_column(rows: list[dict[str, str]], name: str) -> bool:
    """Report whether a column is present and populated.

    A column present in the header but blank in the rows is treated as absent:
    reading a blank as ``0.0`` would put a stationary car at a due-north heading
    and call it measured data.

    Args:
        rows: The car's rows.
        name: Column name.

    Returns:
        ``True`` when the column exists and the first row carries a value.
    """
    return bool(rows) and bool((rows[0].get(name) or "").strip())


def _column(rows: list[dict[str, str]], name: str, file_name: str = "") -> np.ndarray:
    """Extract one numeric column.

    Args:
        rows: The rows to read.
        name: Column name.
        file_name: File name, for error messages.

    Returns:
        The column as a float array.

    Raises:
        ValueError: If a value is not numeric, naming the offending line.
    """
    values = np.empty(len(rows), dtype=np.float64)
    for index, row in enumerate(rows):
        raw = row.get(name)
        try:
            values[index] = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            where = f"{file_name} " if file_name else ""
            message = f"{where}column {name!r}: {raw!r} is not a number."
            raise ValueError(message) from None
    return values


def _speed_from(position: np.ndarray, rate: float) -> np.ndarray:
    """Derive ground speed by differencing position.

    Ground speed, not three-dimensional speed: this is the quantity the rest of
    the pipeline means by "speed", and on a downhill road the vertical component
    is a real difference rather than a rounding one.

    Args:
        position: World positions.
        rate: Sampling rate.

    Returns:
        Ground speed per frame, metres per second.
    """
    count = int(position.shape[0])
    speed = np.zeros(count, dtype=np.float64)
    if count < _MIN_SAMPLES:
        return speed
    ground = position[:, [0, 2]]
    speed[1:] = np.linalg.norm(np.diff(ground, axis=0), axis=1) * rate
    speed[0] = speed[1]
    return speed


def _bearing_from(position: np.ndarray) -> np.ndarray:
    """Derive the direction of travel from positions.

    Uses a centred difference. A trailing one lags by half a sample interval,
    which at corner entry is a systematic bias towards the previous heading —
    the same mistake that had to be corrected in the kinematics module.

    Args:
        position: World positions.

    Returns:
        Bearings per frame, radians.
    """
    count = int(position.shape[0])
    if count < _MIN_SAMPLES:
        return np.zeros(count, dtype=np.float64)
    delta = np.gradient(position[:, [0, 2]], axis=0)
    return np.asarray(np.arctan2(delta[:, 0], delta[:, 1]), dtype=np.float64)


def _wrap(angle: np.ndarray) -> np.ndarray:
    """Wrap angles into ``(-pi, pi]``.

    Args:
        angle: Angles in radians.

    Returns:
        The wrapped angles.
    """
    return np.asarray((angle + np.pi) % (2.0 * np.pi) - np.pi, dtype=np.float64)
