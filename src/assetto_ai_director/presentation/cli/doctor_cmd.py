"""The ``aidirector doctor`` command: what does this machine have?

Takes no required arguments, on purpose. It exists for the moment when a
diagnostic report comes back saying every track-dependent check was skipped and
nobody can tell whether that means "the cameras are fine" or "the game was never
found". Running it answers that in one line.

It is **not** a prerequisite. The same snapshot it prints is embedded
automatically in every camera report, because a step a user has to remember is a
step that gets skipped exactly when it matters.

Never fails on a machine without Assetto Corsa: absence is a finding, and this
command's job is to report findings. Continuous integration runs it for that
reason.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from assetto_ai_director.infrastructure.ac_install import (
    ENV_VAR,
    EnvironmentSnapshot,
    environment_snapshot,
)

__all__ = ["doctor"]

console = Console()


def doctor(
    track: Annotated[
        str, typer.Option("--track", help="Track folder name to check the data files of.")
    ] = "",
    layout: Annotated[
        str, typer.Option("--layout", help="Layout name, for a multi-layout track.")
    ] = "",
    ac_root: Annotated[
        Path | None,
        typer.Option("--ac-root", help="Assetto Corsa directory, when discovery cannot find it."),
    ] = None,
) -> None:
    """Report what this machine has: Assetto Corsa, Custom Shaders Patch, track data."""
    snapshot = environment_snapshot(track, layout, explicit_root=ac_root)
    _render(snapshot)


def _render(snapshot: EnvironmentSnapshot) -> None:
    """Print a snapshot.

    Args:
        snapshot: What was found.
    """
    _render_install(snapshot)
    _render_csp(snapshot)
    _render_track(snapshot)
    _render_blind_spots(snapshot)


def _render_install(snapshot: EnvironmentSnapshot) -> None:
    """Print the installation section.

    Args:
        snapshot: What was found.
    """
    console.print("[bold]Assetto Corsa[/]")
    if snapshot.install is None:
        console.print("  [yellow]not found[/]")
        console.print(f"  [dim]Set {ENV_VAR} to point at it, or pass --ac-root.[/]")
        console.print()
        return

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim")
    # Folded, not truncated: a path with an ellipsis in the middle is useless in
    # a report that gets pasted into a conversation.
    table.add_column(overflow="fold")
    table.add_row("path", str(snapshot.install.root))
    table.add_row("found by", snapshot.install.found_by)
    table.add_row(
        "executable",
        "present" if snapshot.install.has_executable else "[yellow]absent (content only)[/]",
    )
    console.print(table)

    others = snapshot.candidates[1:]
    if others:
        console.print(f"  [dim]{len(others)} further install(s) found:[/]")
        for candidate in others:
            console.print(f"  [dim]  {candidate.root} — {candidate.found_by}[/]")
    console.print()


def _render_csp(snapshot: EnvironmentSnapshot) -> None:
    """Print the Custom Shaders Patch section.

    Args:
        snapshot: What was found.
    """
    console.print("[bold]Custom Shaders Patch[/]")
    if snapshot.install is None:
        console.print("  [dim]not checked — no installation found[/]")
        console.print()
        return

    csp = snapshot.csp
    if csp is None:
        console.print("  [yellow]not installed[/]  [dim](no extension/config/data_manifest.ini)[/]")
        console.print()
        return
    if csp.values is None:
        # Distinguished from absence deliberately: reporting an unreadable
        # manifest as "not installed" is a confident false negative.
        console.print("  installed, [yellow]but its manifest could not be read[/]")
        console.print(f"  [dim]{csp.path}[/]")
        console.print()
        return
    if not csp.values:
        console.print(r"  installed  [dim](manifest carries no \[VERSION] section)[/]")
        console.print()
        return

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim")
    table.add_column()
    for key, value in csp.values.items():
        table.add_row(key, value)
    console.print(table)
    console.print("  [dim]Keys reported as the manifest writes them, not interpreted.[/]")
    console.print()


def _render_track(snapshot: EnvironmentSnapshot) -> None:
    """Print the track data section.

    Args:
        snapshot: What was found.
    """
    if snapshot.track is None:
        return

    console.print(f"[bold]Track[/] {snapshot.track.track_id}/{snapshot.track.layout or '(root)'}")
    table = Table(header_style="dim", box=None, padding=(0, 2))
    table.add_column("file")
    table.add_column("bytes", justify="right")
    for name, size in snapshot.track.sizes().items():
        style = "yellow" if size == 0 else ""
        table.add_row(f"[{style}]{name}[/]" if style else name, f"{size:,}")
    for name in snapshot.track.missing:
        table.add_row(f"[red]{name}[/]", "[red]absent[/]")
    console.print(table)
    console.print()


def _render_blind_spots(snapshot: EnvironmentSnapshot) -> None:
    """Print what could not be established.

    Args:
        snapshot: What was found.
    """
    gaps = snapshot.blind_spots
    if not gaps:
        console.print("[green]Nothing missing.[/]")
        return
    console.print("[bold]What is missing[/]")
    for gap in gaps:
        console.print(f"  [yellow]•[/] {gap}")
