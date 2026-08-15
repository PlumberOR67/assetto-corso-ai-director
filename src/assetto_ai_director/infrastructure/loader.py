"""One way to turn a file into a session.

This existed four times before it existed once: in the GUI worker, in the MCP
session store, and twice in the CLI. Three of those declared their own
``MIN_TRAVELLED_METRES = 50.0``. They had already drifted — only the CLI knew how
to read a CSV, only the GUI reported progress, and none of them could be
cancelled — which is what four copies of a thing always become.

**Parked cars are dropped here rather than downstream.** A full grid is mostly
empty slots: the fixture replays carry seventeen cars of which fifteen never move.
Feeding those to the analysis is not merely wasteful — a stationary car has a
heading that is pure numerical noise and a proximity to the player that never
changes, and the risk score would happily send a camera to an empty grid box. The
ids that were dropped travel with the session so the user can be told, rather than
quietly seeing a two-car analysis of a seventeen-car race.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from assetto_ai_director.application.events.detect import CarSignals
from assetto_ai_director.application.ports.progress import NullProgress, Stage
from assetto_ai_director.domain.errors import OperationCancelledError
from assetto_ai_director.infrastructure.acreplay.frames import read_car_frames
from assetto_ai_director.infrastructure.acreplay.structure import CarBlock, read_structure
from assetto_ai_director.infrastructure.csvio.source import read_csv

if TYPE_CHECKING:
    from pathlib import Path

    from assetto_ai_director.application.ports.cancellation import CancelToken
    from assetto_ai_director.application.ports.progress import ProgressReporter
    from assetto_ai_director.domain.session import SessionMeta

__all__ = ["MIN_TRAVELLED_METRES", "LoadedSession", "load_session", "skipped_note"]

MIN_TRAVELLED_METRES = 50.0
"""Distance a car must cover to count as a participant.

Generous on purpose. A car that pulled off after one corner is a participant with
a short story; a car that never left its grid box is furniture. Fifty metres
separates those without needing to guess at intent.
"""


def skipped_note(skipped: tuple[int, ...]) -> str:
    """Describe the cars that were dropped.

    The filtering used to happen inside ``detect_all``, which reported it. Now
    that it happens here, the note has to move with it — otherwise a user is
    silently shown a two-car analysis of a seventeen-car race, which is exactly
    the kind of quiet narrowing this project reports rather than performs.

    Args:
        skipped: Car ids that never moved.

    Returns:
        A note naming them.
    """
    ids = ", ".join(str(car_id) for car_id in sorted(skipped))
    return (
        f"Skipped {len(skipped)} car(s) that never moved (ids {ids}) — "
        f"empty grid slots, not participants."
    )


@dataclass(frozen=True, slots=True)
class LoadedSession:
    """A decoded session, ready to analyse.

    Attributes:
        path: Where it came from.
        meta: Session metadata.
        signals: Decoded per-car signals, excluding parked entries.
        time: Time per frame, seconds from session start.
        sample_rate: Frames per second.
        skipped: Car ids that never moved, so the user can be told why the
            analysis covers fewer cars than the session contains.
        notes: Caveats from the source itself, distinct from the analysis's own.
    """

    path: Path
    meta: SessionMeta
    signals: dict[int, CarSignals]
    time: np.ndarray
    sample_rate: float
    skipped: tuple[int, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def duration(self) -> float:
        """How long the loaded span runs.

        Returns:
            Duration in seconds.
        """
        return float(self.time[-1] - self.time[0]) if self.time.size else 0.0


def load_session(
    source: Path,
    *,
    seconds: float = 0.0,
    progress: ProgressReporter | None = None,
    cancel: CancelToken | None = None,
) -> LoadedSession:
    """Load a session from a replay or a CSV.

    Args:
        source: The file to read. A ``.csv`` suffix selects the CSV reader;
            anything else is treated as an Assetto Corsa replay.
        seconds: Load only the first N seconds. Zero means all.
        progress: Optional reporter. Owns the :attr:`Stage.DECODE` phase.
        cancel: Optional cancellation token, polled once per car.

    Returns:
        The decoded session.

    Raises:
        OperationCancelledError: If cancellation is requested part-way through.
        ReplayError: If the replay cannot be read.
        ValueError: If the file holds no frames, or no car in it moved.
    """
    if source.suffix.lower() == ".csv":
        return _from_csv(source, seconds)
    return _from_replay(source, seconds, progress, cancel)


def _from_replay(
    source: Path,
    seconds: float,
    progress: ProgressReporter | None,
    cancel: CancelToken | None,
) -> LoadedSession:
    """Decode an Assetto Corsa replay.

    Args:
        source: The replay file.
        seconds: Load only the first N seconds. Zero means all.
        progress: Optional reporter.
        cancel: Optional cancellation token.

    Returns:
        The decoded session.

    Raises:
        OperationCancelledError: If cancellation is requested part-way through.
        ValueError: If the replay holds no frames, or no car moved.
    """
    structure = read_structure(source)
    rate = float(structure.sample_rate)

    limit = min((block.frame_count for block in structure.cars), default=0)
    if seconds > 0.0:
        limit = min(limit, int(seconds * rate))
    if limit <= 0:
        message = f"{source.name} contains no frames."
        raise ValueError(message)

    reporter: ProgressReporter = progress if progress is not None else NullProgress()
    reporter.start(str(Stage.DECODE), len(structure.cars))

    signals: dict[int, CarSignals] = {}
    skipped: list[int] = []
    for block in structure.cars:
        # Per car rather than finer. One `read_car_frames` is atomic, so on a
        # 233 MB replay cancelling can take a second or two to take effect;
        # splitting it would mean chunking the reader for a latency nobody has
        # complained about.
        if cancel is not None and cancel.cancelled():
            raise OperationCancelledError(str(Stage.DECODE))
        decoded = read_car_frames(
            source,
            CarBlock(
                identity=block.identity,
                frame_count=limit,
                wing_count=block.wing_count,
                frames_offset=block.frames_offset,
                extra_count=block.extra_count,
            ),
            rate,
        )
        travelled = float(
            np.linalg.norm(np.diff(decoded.position[:, [0, 2]], axis=0), axis=1).sum()
        )
        if travelled >= MIN_TRAVELLED_METRES:
            signals[block.identity.car_id] = CarSignals(
                car_id=block.identity.car_id,
                position=decoded.position,
                heading=decoded.heading,
                speed=decoded.speed,
                sideslip=decoded.sideslip,
            )
        else:
            skipped.append(block.identity.car_id)
        reporter.advance()

    reporter.finish()

    if not signals:
        message = f"No car in {source.name} moved; there is nothing to analyse."
        raise ValueError(message)

    return LoadedSession(
        path=source,
        meta=structure.to_session_meta(),
        signals=signals,
        time=np.arange(limit, dtype=np.float64) / rate,
        sample_rate=rate,
        skipped=tuple(skipped),
        notes=(skipped_note(tuple(skipped)),) if skipped else (),
    )


def _from_csv(source: Path, seconds: float) -> LoadedSession:
    """Read a session from a CSV in this project's schema.

    No progress or cancellation: the reader is a single pass over a text file
    and finishes before a progress bar would have painted.

    Args:
        source: The CSV file.
        seconds: Load only the first N seconds. Zero means all.

    Returns:
        The parsed session.

    Raises:
        ValueError: If the file cannot be parsed, or no car in it moved.
    """
    parsed = read_csv(source)
    time = parsed.time
    signals = parsed.signals

    if seconds > 0.0:
        limit = int(seconds * float(parsed.sample_rate))
        time = time[:limit]
        signals = {
            car_id: CarSignals(
                car_id=car_id,
                position=car.position[:limit],
                heading=car.heading[:limit],
                speed=car.speed[:limit],
                sideslip=car.sideslip[:limit],
            )
            for car_id, car in signals.items()
        }

    moving = {
        car_id: car
        for car_id, car in signals.items()
        if float(np.linalg.norm(np.diff(car.position[:, [0, 2]], axis=0), axis=1).sum())
        >= MIN_TRAVELLED_METRES
    }
    if not moving:
        message = f"No car in {source.name} moved; there is nothing to analyse."
        raise ValueError(message)

    skipped = tuple(sorted(set(signals) - set(moving)))
    return LoadedSession(
        path=source,
        meta=parsed.meta,
        signals=moving,
        time=time,
        sample_rate=float(parsed.sample_rate),
        skipped=skipped,
        notes=(*parsed.notes, *((skipped_note(skipped),) if skipped else ())),
    )
