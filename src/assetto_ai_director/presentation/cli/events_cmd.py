"""The ``aidirector events`` command: what happened in this session?

Prints a tally and a timeline of detected events, and states plainly which event
kinds the source cannot support. A user who sees no contacts reported should know
whether that means there were none or that the detector cannot see them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from assetto_ai_director.application.events.detect import (
    CarSignals,
    DetectionResult,
    detect_all,
)
from assetto_ai_director.domain.events import EventKind
from assetto_ai_director.infrastructure.acreplay.frames import read_car_frames
from assetto_ai_director.infrastructure.acreplay.structure import CarBlock, read_structure
from assetto_ai_director.presentation.cli.progress import RichProgress

__all__ = ["events"]

console = Console()

_DEFAULT_LIST_LIMIT = 40


def events(
    replay: Annotated[
        Path, typer.Argument(help="Replay file to analyse.", exists=True, dir_okay=False)
    ],
    *,
    cars: Annotated[
        int, typer.Option("--cars", "-c", help="Analyse this many cars. 0 means all.")
    ] = 0,
    seconds: Annotated[
        float, typer.Option("--seconds", help="Analyse only the first N seconds. 0 means all.")
    ] = 0.0,
    show: Annotated[
        int, typer.Option("--show", help="List this many events. 0 lists none.")
    ] = _DEFAULT_LIST_LIMIT,
    kind: Annotated[str | None, typer.Option("--kind", "-k", help="Only this event kind.")] = None,
) -> None:
    """Detect driving events in a replay."""
    structure = read_structure(replay)
    blocks = structure.cars if cars <= 0 else structure.cars[:cars]
    rate = float(structure.sample_rate)

    limit = min(
        (block.frame_count for block in blocks),
        default=0,
    )
    if seconds > 0.0:
        limit = min(limit, int(seconds * rate))
    if limit <= 0:
        console.print("[red]Replay contains no frames.[/]")
        raise typer.Exit(code=1)

    signals: dict[int, CarSignals] = {}
    with RichProgress(console) as progress:
        progress.start("Decoding cars", total=len(blocks))
        for block in blocks:
            trimmed = CarBlock(
                identity=block.identity,
                frame_count=limit,
                wing_count=block.wing_count,
                frames_offset=block.frames_offset,
                extra_count=block.extra_count,
            )
            decoded = read_car_frames(replay, trimmed, rate)
            signals[block.identity.car_id] = CarSignals(
                car_id=block.identity.car_id,
                position=decoded.position,
                heading=decoded.heading,
                speed=decoded.speed,
                sideslip=decoded.sideslip,
            )
            progress.advance()

    time = np.arange(limit, dtype=np.float64) / rate
    result = detect_all(signals, time, rate)

    _render_summary(structure.to_session_meta().track_label, limit / rate, len(blocks), result)
    if show > 0:
        _render_events(result, structure, show, kind)
    for note in result.notes:
        console.print(f"\n  [dim]{note}[/]")


def _render_summary(track: str, duration: float, car_count: int, result: DetectionResult) -> None:
    """Print the event tally.

    Args:
        track: Track label.
        duration: Analysed duration in seconds.
        car_count: Cars analysed.
        result: The detection result.
    """
    console.print(
        f"[bold cyan]{track}[/]  {duration:.0f}s  {car_count} car(s)  "
        f"[bold]{len(result.events)}[/] events\n"
    )
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim")
    table.add_column(justify="right")
    table.add_column()
    counts = result.counts
    for event_kind in sorted(counts, key=lambda k: -counts[k]):
        count = counts[event_kind]
        bar = "#" * min(40, count)
        table.add_row(event_kind.value, str(count), f"[cyan]{bar}[/]")
    console.print(table)


def _render_events(
    result: DetectionResult, structure: object, limit: int, kind: str | None
) -> None:
    """List individual events.

    Args:
        result: The detection result.
        structure: The replay structure, for car names.
        limit: How many events to list.
        kind: Optional event-kind filter.
    """
    selected = result.events
    if kind:
        try:
            wanted = EventKind(kind)
        except ValueError:
            console.print(f"[red]Unknown event kind {kind!r}[/]")
            raise typer.Exit(code=1) from None
        selected = result.of_kind(wanted)

    names = {
        block.identity.car_id: block.identity.short_model
        for block in getattr(structure, "cars", ())
    }

    console.print(f"\n  [dim]showing {min(limit, len(selected))} of {len(selected)}[/]")
    table = Table(header_style="dim", box=None, padding=(0, 2))
    table.add_column("t", justify="right")
    table.add_column("dur", justify="right")
    table.add_column("kind")
    table.add_column("car")
    table.add_column("detail", style="dim")

    for event in selected[:limit]:
        subjects = " -> ".join(names.get(car, str(car)) for car in event.subjects)
        detail = "  ".join(f"{key}={value:.4g}" for key, value in list(event.detail.items())[:3])
        table.add_row(
            f"{float(event.focus):.1f}s",
            f"{float(event.duration):.1f}s",
            event.kind.value,
            subjects,
            detail,
        )
    console.print(table)
