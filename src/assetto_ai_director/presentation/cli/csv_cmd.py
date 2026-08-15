"""The ``aidirector csv`` command: get the decoded telemetry out.

Exists for two reasons that pull in the same direction.

The first is **verification**. A binary reader that nobody can check is a binary
reader you have to trust, and this project's whole position on the ``.acreplay``
format is that trust is not a method. Writing the decode out as a table means
anyone can open it, sort it, plot it, and see for themselves whether the car goes
where the track goes.

The second is **interchange**. The same table read back is a session, so a
simulator this project has never heard of can drive the director as long as it can
emit position and time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from rich.console import Console

from assetto_ai_director.application.events.detect import CarSignals
from assetto_ai_director.infrastructure.acreplay.frames import read_car_frames
from assetto_ai_director.infrastructure.acreplay.structure import CarBlock, read_structure
from assetto_ai_director.infrastructure.csvio.source import CSV_COLUMNS, write_csv
from assetto_ai_director.presentation.cli.progress import RichProgress

__all__ = ["csv"]

console = Console()


def csv(
    replay: Annotated[
        Path, typer.Argument(help="Replay file to decode.", exists=True, dir_okay=False)
    ],
    out: Annotated[Path, typer.Argument(help="Where to write the CSV.")],
    *,
    cars: Annotated[
        int, typer.Option("--cars", "-c", help="Export this many cars. 0 means all.")
    ] = 0,
    seconds: Annotated[
        float, typer.Option("--seconds", help="Export only the first N seconds. 0 means all.")
    ] = 0.0,
) -> None:
    """Export a replay's decoded telemetry as CSV."""
    structure = read_structure(replay)
    blocks = structure.cars if cars <= 0 else structure.cars[:cars]
    rate = float(structure.sample_rate)

    limit = min((block.frame_count for block in blocks), default=0)
    if seconds > 0.0:
        limit = min(limit, int(seconds * rate))
    if limit <= 0:
        console.print("[red]Replay contains no frames.[/]")
        raise typer.Exit(code=1)

    signals: dict[int, CarSignals] = {}
    with RichProgress(console) as progress:
        progress.start("Decoding cars", total=len(blocks))
        for block in blocks:
            decoded = read_car_frames(
                replay,
                CarBlock(
                    identity=block.identity,
                    frame_count=limit,
                    wing_count=block.wing_count,
                    frames_offset=block.frames_offset,
                    extra_count=block.extra_count,
                ),
                rate,
            )
            signals[block.identity.car_id] = CarSignals(
                car_id=block.identity.car_id,
                position=decoded.position,
                heading=decoded.heading,
                speed=decoded.speed,
                sideslip=decoded.sideslip,
            )
            progress.advance()

    time = np.arange(limit, dtype=np.float64) / rate
    write_csv(out, signals, time)

    rows = len(signals) * limit
    console.print(
        f"\n[green]Wrote[/] {out}  "
        f"[dim]{rows} rows, {len(signals)} car(s), {limit / rate:.0f}s at {rate:.1f} Hz[/]"
    )
    console.print(f"  [dim]columns: {', '.join(CSV_COLUMNS)}[/]")
    console.print("  [dim]Read it back with any command that accepts a .csv path.[/]")
