"""The ``aidirector styles`` command: what the director's taste is made of.

Lists the available styles and, with ``--provenance``, audits where every number
in them came from. That audit is the project's core rule made visible: a reader
can see at a glance which figures rest on evidence and which are judgement, and
CI prints it on every run so a silent drift toward more guesses is impossible to
miss.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from assetto_ai_director.application.director.styles import BUILTIN_STYLES
from assetto_ai_director.domain.provenance import Provenance, ProvenanceReport
from assetto_ai_director.domain.style import StyleProfile
from assetto_ai_director.infrastructure.style.toml_io import read_profile

__all__ = ["styles"]

console = Console()

_PROVENANCE_STYLE = {
    Provenance.VERIFIED: "green",
    Provenance.MEASURED: "bold green",
    Provenance.DERIVED: "cyan",
    Provenance.GUESS: "yellow",
}


def styles(
    *,
    provenance: Annotated[
        bool, typer.Option("--provenance", help="Audit where every number came from.")
    ] = False,
    profile: Annotated[
        Path | None,
        typer.Option("--profile", help="Audit a TOML profile instead of the built-ins."),
    ] = None,
) -> None:
    """List the available directorial styles."""
    profiles = (
        {profile.stem: read_profile(profile)} if profile is not None else dict(BUILTIN_STYLES)
    )

    if not provenance:
        _render_summary(profiles)
        return

    total_guesses = 0
    for name, entry in sorted(profiles.items()):
        total_guesses += _render_provenance(name, entry)

    console.print(f"\n[bold]{total_guesses}[/] unsourced value(s) across {len(profiles)} style(s).")
    console.print(
        "  [dim]Guesses are legitimate — a director has to pick something — but they\n"
        "  are the measurement backlog. Run `aidirector measure` on reference\n"
        "  footage to replace them.[/]"
    )
    console.print(
        "  [yellow]Not counted above:[/] the six score weights in each style are\n"
        "  editorial judgement and are deliberately not provenance-tracked."
    )


def _render_summary(profiles: dict[str, StyleProfile]) -> None:
    """Print one line per style.

    Args:
        profiles: The styles to list.
    """
    table = Table(header_style="dim", box=None, padding=(0, 2))
    table.add_column("style")
    table.add_column("median shot", justify="right")
    table.add_column("spread", justify="right")
    table.add_column("static", justify="right")
    table.add_column("measured", justify="center")
    table.add_column("description", style="dim")

    for name, entry in sorted(profiles.items()):
        table.add_row(
            name,
            f"{entry.rhythm.median_shot_length.value:.2f}s",
            f"{entry.rhythm.spread.value:.2f}",
            f"{entry.camera_mix.static_fraction.value * 100:.0f}%",
            "[green]yes[/]" if entry.is_measured else "[yellow]no[/]",
            entry.description.split(".")[0],
        )
    console.print(table)


def _format(value: object) -> str:
    """Render a tunable's value for display.

    ``Tunable`` is generic, so a profile may legitimately hold an integer count
    or a boolean switch alongside its floats. Formatting everything as a float
    would raise on those.

    Args:
        value: The tunable's value.

    Returns:
        A short display string.
    """
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int | float):
        return f"{float(value):.4g}"
    return str(value)


def _render_provenance(name: str, profile: StyleProfile) -> int:
    """Print a style's provenance audit.

    Args:
        name: The style's name.
        profile: The style.

    Returns:
        How many of its tunables are unsourced guesses.
    """
    report = ProvenanceReport.of(profile)
    console.print(f"\n[bold cyan]{name}[/]  [dim]{report.trust_ratio * 100:.0f}% sourced[/]")

    table = Table(header_style="dim", box=None, padding=(0, 2))
    table.add_column("field")
    table.add_column("value", justify="right")
    table.add_column("provenance")
    table.add_column("source", style="dim")

    for field, tunable in report.entries:
        colour = _PROVENANCE_STYLE.get(tunable.provenance, "white")
        table.add_row(
            field,
            _format(tunable.value),
            f"[{colour}]{tunable.provenance.value}[/]",
            (tunable.source or tunable.note or "")[:60],
        )
    console.print(table)
    return report.count(Provenance.GUESS)
