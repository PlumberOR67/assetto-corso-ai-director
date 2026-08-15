"""The ``aidirector inspect`` command: what is in this replay?

Reads only the file's structure, so it returns in well under a second even on a
327 MB replay, and it reports whether the layout **verified** — that is, whether
the structural walk landed exactly where the file's own footer says the data
ends. A replay that fails that check is reported as unusable rather than
half-parsed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from assetto_ai_director.domain.errors import ReplayError
from assetto_ai_director.infrastructure.acreplay.structure import ReplayStructure, read_structure

__all__ = ["inspect"]

console = Console()

_MAX_LISTED_CARS = 24


def inspect(
    paths: Annotated[
        list[Path],
        typer.Argument(help="Replay files to inspect.", exists=True, dir_okay=False),
    ],
    *,
    cars: Annotated[bool, typer.Option("--cars/--no-cars", help="List the entry list.")] = True,
) -> None:
    """Print a replay's metadata without decoding telemetry."""
    failures = 0
    for path in paths:
        try:
            structure = read_structure(path)
        except ReplayError as error:
            failures += 1
            console.print(f"[bold red]{path.name}[/]\n  [red]{error}[/]\n")
            continue
        _render(structure, show_cars=cars)

    if failures:
        raise typer.Exit(code=1)


def _render(structure: ReplayStructure, *, show_cars: bool) -> None:
    """Print one replay's structure.

    Args:
        structure: The parsed structure.
        show_cars: Whether to list the entry list.
    """
    meta = structure.to_session_meta()
    quality = meta.quality

    console.print(f"[bold cyan]{structure.path.name}[/]")

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim")
    table.add_column()
    table.add_row("Track", meta.track_label)
    table.add_row("Weather", meta.weather_id)
    table.add_row("Cars", str(meta.car_count))
    table.add_row("Frames", f"{structure.frame_count:,}")
    table.add_row(
        "Rate",
        f"{meta.sample_rate:.1f} Hz  ({structure.recording_interval_ms:g} ms)  "
        f"[{_quality_style(quality.name)}]{quality.name.lower()}[/]",
    )
    table.add_row("Duration", _format_duration(float(meta.duration)))
    table.add_row("Version", str(structure.version))
    table.add_row("Size", f"{structure.file_size / 1_048_576:.1f} MiB")
    table.add_row(
        "Layout check",
        "[bold green]verified[/] — parse ends exactly at the file's own footer offset"
        if structure.verified
        else "[bold red]FAILED[/]",
    )
    if structure.csp_block_offset is not None:
        extension = structure.file_size - structure.csp_block_offset
        table.add_row("CSP block", f"present, {extension / 1_048_576:.1f} MiB")
    if structure.wrapped:
        table.add_row(
            "[yellow]Ring buffer[/]",
            "[yellow]wrapped — the session start was overwritten during recording[/]",
        )
    console.print(table)

    if not quality.supports_event_timing:
        console.print(f"  [yellow]! {quality.advice}[/]")

    if show_cars:
        _render_cars(structure)
    console.print()


def _render_cars(structure: ReplayStructure) -> None:
    """Print the entry list.

    Args:
        structure: The parsed structure.
    """
    table = Table(title=None, header_style="dim", box=None, padding=(0, 2))
    table.add_column("#", justify="right", style="dim")
    table.add_column("Car")
    table.add_column("Driver")
    table.add_column("Skin", style="dim")
    table.add_column("Wings", justify="right", style="dim")
    table.add_column("Stride", justify="right", style="dim")

    for block in structure.cars[:_MAX_LISTED_CARS]:
        identity = block.identity
        marker = " [green]*[/]" if identity.is_player else ""
        table.add_row(
            str(identity.car_id),
            identity.short_model,
            (identity.driver_name or "[dim]AI[/]") + marker,
            identity.skin_id or "",
            str(block.wing_count),
            str(block.stride),
        )
    console.print(table)
    if len(structure.cars) > _MAX_LISTED_CARS:
        console.print(f"  [dim]... and {len(structure.cars) - _MAX_LISTED_CARS} more[/]")


def _quality_style(name: str) -> str:
    """Pick a colour for a recording-quality band.

    Args:
        name: The band's name.

    Returns:
        A Rich style name.
    """
    return {
        "UNUSABLE": "bold red",
        "COARSE": "yellow",
        "GOOD": "green",
        "EXCELLENT": "bold green",
    }.get(name, "white")


def _format_duration(seconds: float) -> str:
    """Render a duration as ``h:mm:ss`` or ``m:ss``.

    Args:
        seconds: The duration.

    Returns:
        A human-readable string.
    """
    total = round(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"
