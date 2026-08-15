"""The ``aidirector diagnose`` command: are these cameras any good?

One required argument, the session. Everything else — the style, where Assetto
Corsa is, which track the replay was recorded on, where to put the output — is
either defaulted or discovered, because this command exists to be run by someone
who is verifying somebody else's work and should not have to learn the tool
first.

It takes a session rather than a ``timeline.json`` for a reason that is not an
oversight: the timeline records subject *identifiers*, not subject positions, so
a timeline on its own cannot answer whether a car is in frame. Pass ``--timeline``
to reuse a cut that has already been directed and skip the expensive part.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console
from rich.table import Table

from assetto_ai_director.application.camera.paths import SubjectTracks
from assetto_ai_director.application.diagnostics import diagnose_cameras
from assetto_ai_director.application.diagnostics.scene import build_previews
from assetto_ai_director.application.director.styles import BUILTIN_STYLES
from assetto_ai_director.application.pipeline import direct as run_director
from assetto_ai_director.domain.diagnostics import CheckStatus
from assetto_ai_director.domain.errors import AssettoDirectorError
from assetto_ai_director.infrastructure.export.cameras_ini import planned_spans
from assetto_ai_director.infrastructure.export.diagnostic_json import (
    diagnostic_to_text,
    write_diagnostic,
)
from assetto_ai_director.infrastructure.export.preview_svg import write_contact_sheet
from assetto_ai_director.infrastructure.loader import LoadedSession, load_session
from assetto_ai_director.infrastructure.style.toml_io import read_profile
from assetto_ai_director.infrastructure.timeline_io import read_timeline
from assetto_ai_director.infrastructure.track.resolve import resolve_spline
from assetto_ai_director.presentation.cli.progress import RichProgress

if TYPE_CHECKING:
    from assetto_ai_director.domain.diagnostics import DiagnosticReport
    from assetto_ai_director.domain.style import StyleProfile

__all__ = ["diagnose"]

console = Console()

_STATUS_STYLE = {
    CheckStatus.PASS: "green",
    CheckStatus.WARN: "yellow",
    CheckStatus.FAIL: "red",
    CheckStatus.UNKNOWN: "blue",
}


def diagnose(
    session: Annotated[
        Path,
        typer.Argument(
            help="Replay file to check, or a .csv in this project's schema.",
            exists=True,
            dir_okay=False,
        ),
    ],
    *,
    style: Annotated[
        str, typer.Option("--style", "-s", help="Built-in style name, or a path to a profile.")
    ] = "initial_d",
    timeline_path: Annotated[
        Path | None,
        typer.Option("--timeline", help="Reuse an already-directed timeline.json."),
    ] = None,
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Folder for the report and contact sheet.")
    ] = None,
    seconds: Annotated[
        float, typer.Option("--seconds", help="Check only the first N seconds. 0 means all.")
    ] = 0.0,
    ac_root: Annotated[
        Path | None,
        typer.Option("--ac-root", help="Assetto Corsa directory, when discovery cannot find it."),
    ] = None,
) -> None:
    """Check what a directed cut's cameras would actually see."""
    loaded = _load(session, seconds)
    tracks = SubjectTracks.from_signals(loaded.signals, loaded.time, loaded.sample_rate)

    if timeline_path is not None:
        try:
            timeline = read_timeline(timeline_path)
        except (ValueError, json.JSONDecodeError) as error:
            console.print(f"[red]{error}[/]")
            raise typer.Exit(code=1) from error
    else:
        try:
            with RichProgress(console) as progress:
                timeline = run_director(
                    loaded.signals,
                    loaded.meta,
                    _load_style(style),
                    loaded.time,
                    loaded.sample_rate,
                    progress=progress,
                )
        except AssettoDirectorError as error:
            console.print(f"[red]{error}[/]")
            raise typer.Exit(code=1) from error

    resolved = resolve_spline(
        timeline.session.track_id,
        timeline.session.layout,
        next(iter(tracks.positions.values()), None),
        ac_root=ac_root,
    )
    report = diagnose_cameras(
        timeline,
        tracks,
        exported_spans=planned_spans(timeline, track=resolved.spline),
        track=resolved.spline,
        notes=(resolved.note,) if resolved.note else (),
    )
    _render(report)

    if out is None:
        console.print("\n  [dim]No --out given; nothing written.[/]")
        return

    _write(report, tracks, out, ac_root)


def _write(
    report: DiagnosticReport, tracks: SubjectTracks, out: Path, ac_root: Path | None
) -> None:
    """Write the report, its compact form and the contact sheet.

    Three files rather than one, because they answer different questions: the
    full report is the record, the compact one is what fits in a conversation,
    and the sheet is what a person looks at.

    Args:
        report: The diagnosis.
        tracks: The cars' motion, for rendering the previews.
        out: The output folder.
        ac_root: An Assetto Corsa directory configured by hand.
    """
    out.mkdir(parents=True, exist_ok=True)
    full = out / "diagnostic.json"
    compact = out / "diagnostic_compact.json"
    sheet = out / "cameras.svg"

    write_diagnostic(report, full, ac_root=ac_root)
    compact.write_text(
        diagnostic_to_text(report, ac_root=ac_root, notable_only=True), encoding="utf-8"
    )
    write_contact_sheet(
        build_previews(report, tracks),
        sheet,
        title=f"{report.track_id} / {report.layout or 'default'} — {report.style_name}",
    )

    console.print()
    for path, what in (
        (full, "every camera, every check"),
        (compact, "the cameras worth looking at — paste this one"),
        (sheet, "what each camera sees, open in a browser"),
    ):
        console.print(f"[green]Wrote[/] {path}  [dim]{what}[/]")


def _render(report: DiagnosticReport) -> None:
    """Print the report to the terminal.

    Args:
        report: The diagnosis.
    """
    tally = report.tally()
    console.print(
        f"[bold]{len(report.cameras)} cameras[/] — "
        f"[{_STATUS_STYLE[report.measured_status]}]{report.measured_status.value}[/] "
        f"on what could be measured, "
        f"[{_STATUS_STYLE[report.status]}]{report.status.value}[/] overall  "
        f"[dim]({tally[CheckStatus.FAIL]} failing, {tally[CheckStatus.WARN]} warning)[/]"
    )
    console.print()

    table = Table(header_style="dim", box=None, padding=(0, 2))
    table.add_column("check")
    for name in ("pass", "warn", "fail", "unknown"):
        table.add_column(name, justify="right")
    for name, counts in report.check_tally().items():
        table.add_row(
            name,
            *(
                _count(counts[status], status)
                for status in (
                    CheckStatus.PASS,
                    CheckStatus.WARN,
                    CheckStatus.FAIL,
                    CheckStatus.UNKNOWN,
                )
            ),
        )
    console.print(table)

    failures = report.failing()
    if failures:
        console.print("\n[bold]What failed[/]")
        for camera in failures:
            for check in camera.failed():
                console.print(
                    f"  [red]shot {camera.shot_index:>3}[/] "
                    f"[dim]{float(camera.start):>7.1f}s[/]  {check.name}"
                )
                console.print(f"      [dim]{check.detail}[/]")

    if report.blind_spots:
        console.print("\n[bold]What could not be checked[/]")
        for gap in report.blind_spots:
            console.print(f"  [yellow]•[/] {gap}")


def _count(value: int, status: CheckStatus) -> str:
    """Render one tally cell, dimming the zeros.

    Args:
        value: The count.
        status: Which status it counts.

    Returns:
        Rich markup.
    """
    if value == 0:
        return "[dim]·[/]"
    return f"[{_STATUS_STYLE[status]}]{value}[/]"


def _load(source: Path, seconds: float) -> LoadedSession:
    """Load a session from a replay or a CSV.

    Args:
        source: The input file.
        seconds: Limit to the first N seconds; 0 means all.

    Returns:
        The loaded session.

    Raises:
        typer.Exit: If the file holds no usable frames, or no car moved.
    """
    try:
        with RichProgress(console) as progress:
            return load_session(source, seconds=seconds, progress=progress)
    except ValueError as error:
        console.print(f"[red]{error}[/]")
        raise typer.Exit(code=1) from error


def _load_style(style: str) -> StyleProfile:
    """Resolve a style name or profile path.

    Args:
        style: A built-in style name, or a path to a TOML profile.

    Returns:
        The style profile.

    Raises:
        typer.Exit: If the name is not a built-in style and not a readable file.
    """
    candidate = Path(style)
    if candidate.suffix == ".toml" and candidate.is_file():
        return read_profile(candidate)
    if style not in BUILTIN_STYLES:
        console.print(
            f"[red]Unknown style {style!r}.[/] Available: {', '.join(sorted(BUILTIN_STYLES))}"
        )
        raise typer.Exit(code=1)
    return BUILTIN_STYLES[style]
