"""The ``aidirector export`` command: project a timeline into another format.

Reports what each target loses **before** writing, because "your Initial D cut
has been exported" is a lie if the format dropped every event-triggered cut in it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from assetto_ai_director.infrastructure.export.cameras_ini import CamerasIniSink
from assetto_ai_director.infrastructure.export.json_timeline import JsonTimelineSink
from assetto_ai_director.infrastructure.timeline_io import read_timeline

__all__ = ["export"]

console = Console()

SINKS: dict[str, type[JsonTimelineSink] | type[CamerasIniSink]] = {
    "json": JsonTimelineSink,
    "cameras-ini": CamerasIniSink,
}


def export(
    fmt: Annotated[str, typer.Argument(help="Target format: json or cameras-ini.")],
    timeline_path: Annotated[
        Path, typer.Argument(help="Timeline JSON to convert.", exists=True, dir_okay=False)
    ],
    *,
    out: Annotated[Path | None, typer.Option("--out", "-o", help="Where to write.")] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Export even when the format loses a great deal.")
    ] = False,
) -> None:
    """Convert a directed timeline into another format."""
    if fmt not in SINKS:
        console.print(f"[red]Unknown format {fmt!r}.[/] Available: {', '.join(sorted(SINKS))}")
        raise typer.Exit(code=1)

    try:
        timeline = read_timeline(timeline_path)
    except (ValueError, json.JSONDecodeError) as error:
        console.print(f"[red]{error}[/]")
        raise typer.Exit(code=1) from error

    sink = SINKS[fmt]()
    report = sink.supports(timeline)

    console.print(f"[bold]{sink.name}[/] — {report.summary}")
    for limitation in report.limitations:
        console.print(f"  [yellow]•[/] {limitation}")

    if not report.supported:
        raise typer.Exit(code=1)
    if report.lossy and not force and out is not None:
        console.print("\n  [dim]Pass --force to write anyway.[/]")
        raise typer.Exit(code=1)

    if out is None:
        console.print("\n  [dim]No --out given; nothing written.[/]")
        return

    sink.write(timeline, out)
    console.print(f"\n[green]Wrote[/] {out}")
