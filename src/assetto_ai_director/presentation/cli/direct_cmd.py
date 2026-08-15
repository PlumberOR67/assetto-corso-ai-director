"""The ``aidirector direct`` command: replay in, cut out.

The end-to-end path this project exists to provide.

The input may also be a CSV in this project's schema, which is how a simulator
other than Assetto Corsa reaches the director. The two paths converge immediately:
past :func:`_load`, nothing downstream knows or cares which one it came from.
"""

from __future__ import annotations

import signal
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from assetto_ai_director.application.director.styles import BUILTIN_STYLES
from assetto_ai_director.application.events.detect import CarSignals
from assetto_ai_director.application.pipeline import direct as run_director
from assetto_ai_director.application.ports.cancellation import CancelFlag
from assetto_ai_director.domain.errors import AssettoDirectorError, OperationCancelledError
from assetto_ai_director.domain.session import SessionMeta
from assetto_ai_director.domain.style import StyleProfile
from assetto_ai_director.domain.timeline import Timeline
from assetto_ai_director.infrastructure.export.json_timeline import write_timeline
from assetto_ai_director.infrastructure.loader import load_session
from assetto_ai_director.infrastructure.style.toml_io import read_profile
from assetto_ai_director.presentation.cli.progress import RichProgress

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["direct"]

console = Console()

_PREVIEW_SHOTS = 16


@dataclass(frozen=True, slots=True)
class _Loaded:
    """A session from either input path, in the one shape the director takes.

    Attributes:
        signals: Decoded per-car signals.
        meta: Session metadata.
        time: Time per frame, seconds.
        rate: Frames per second.
        notes: Caveats the loader wants carried to the user.
    """

    signals: dict[int, CarSignals]
    meta: SessionMeta
    time: np.ndarray
    rate: float
    notes: tuple[str, ...] = ()


def direct(
    replay: Annotated[
        Path,
        typer.Argument(
            help="Replay file to direct, or a .csv in this project's schema.",
            exists=True,
            dir_okay=False,
        ),
    ],
    *,
    style: Annotated[
        str, typer.Option("--style", "-s", help="Built-in style name, or a path to a profile.")
    ] = "initial_d",
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Write the timeline JSON here.")
    ] = None,
    seconds: Annotated[
        float, typer.Option("--seconds", help="Direct only the first N seconds. 0 means all.")
    ] = 0.0,
    show: Annotated[int, typer.Option("--show", help="Preview this many shots.")] = _PREVIEW_SHOTS,
) -> None:
    """Analyse a session and direct a cinematic cut."""
    profile = _load_style(style)
    loaded = _load(replay, seconds)

    try:
        with RichProgress(console) as progress, _interruptible() as cancel:
            timeline = run_director(
                loaded.signals,
                loaded.meta,
                profile,
                loaded.time,
                loaded.rate,
                progress=progress,
                cancel=cancel,
            )
    except OperationCancelledError as cancelled:
        # Before the general case, and in a different colour: the user asked for
        # this. Exit code 130 is the shell convention for death by SIGINT.
        console.print(f"[yellow]{cancelled}[/]")
        raise typer.Exit(code=130) from cancelled
    except AssettoDirectorError as error:
        console.print(f"[red]{error}[/]")
        raise typer.Exit(code=1) from error

    # Caveats from loading belong on the timeline, not only on this terminal.
    # An exported file read by someone else must carry the same warnings the
    # person who ran the command saw.
    for note in loaded.notes:
        timeline = timeline.with_note(note)

    _render(timeline, show)

    if out is not None:
        write_timeline(timeline, out)
        console.print(f"\n[green]Wrote[/] {out}")


@contextmanager
def _interruptible() -> Iterator[CancelFlag]:
    """Route Ctrl+C into a cancellation token for the duration of a run.

    Without this, an interrupt raises ``KeyboardInterrupt`` straight through a
    live Rich display, which leaves the terminal mid-repaint and prints a
    traceback for what was a deliberate act. With it, the analysis stops at its
    next checkpoint and reports the stage it got to.

    The previous handler is always restored, including when the run raises, so
    a second Ctrl+C after the run has finished behaves normally. Signal handlers
    can only be installed on the main thread; when this is called from anywhere
    else — a GUI worker, a test — installation is skipped and the token simply
    never trips, which is the correct behaviour there anyway.

    Yields:
        The token to pass to the pipeline.
    """
    flag = CancelFlag()
    try:
        previous = signal.signal(signal.SIGINT, lambda *_: flag.cancel())
    except ValueError:
        yield flag
        return
    try:
        yield flag
    finally:
        signal.signal(signal.SIGINT, previous)


def _load(source: Path, seconds: float) -> _Loaded:
    """Load a session from a replay or a CSV.

    Thin wrapper over the shared loader: the CLI's only addition is turning a
    loader error into a clean exit rather than a traceback.

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
            loaded = load_session(source, seconds=seconds, progress=progress)
    except ValueError as error:
        console.print(f"[red]{error}[/]")
        raise typer.Exit(code=1) from error
    return _Loaded(loaded.signals, loaded.meta, loaded.time, loaded.sample_rate, loaded.notes)


def _load_style(style: str) -> StyleProfile:
    """Resolve a style name or profile path.

    Args:
        style: A built-in style name, or a path to a TOML profile.

    Returns:
        The style profile.
    """
    candidate = Path(style)
    if candidate.suffix == ".toml" and candidate.is_file():
        return read_profile(candidate)
    if style not in BUILTIN_STYLES:
        available = ", ".join(sorted(BUILTIN_STYLES))
        console.print(f"[red]Unknown style {style!r}.[/] Available: {available}")
        raise typer.Exit(code=1)
    return BUILTIN_STYLES[style]


def _render(timeline: Timeline, show: int) -> None:
    """Print the directed cut.

    Args:
        timeline: The finished timeline.
        show: How many shots to preview.
    """
    console.print(
        f"\n[bold cyan]{timeline.session.track_label}[/]  "
        f"style [bold]{timeline.style_name}[/]  "
        f"[bold]{timeline.shot_count}[/] shots over "
        f"{float(timeline.duration):.0f}s  "
        f"ASL {float(timeline.average_shot_length):.2f}s"
    )
    if not timeline.tiles_without_gaps():
        console.print("  [red]WARNING: the edit has gaps — this is a bug, please report it.[/]")

    rigs: dict[str, int] = {}
    for shot in timeline.shots:
        rigs[shot.candidate.rig.value] = rigs.get(shot.candidate.rig.value, 0) + 1
    mix = "  ".join(f"{name} {count}" for name, count in sorted(rigs.items(), key=lambda p: -p[1]))
    console.print(f"  [dim]{mix}[/]\n")

    if show > 0:
        table = Table(header_style="dim", box=None, padding=(0, 2))
        table.add_column("#", justify="right", style="dim")
        table.add_column("t", justify="right")
        table.add_column("dur", justify="right")
        table.add_column("rig")
        table.add_column("size", style="dim")
        table.add_column("side", style="dim")
        table.add_column("score", justify="right")
        table.add_column("because", style="dim")

        for shot in timeline.shots[:show]:
            candidate = shot.candidate
            because = candidate.anchor.kind.value if candidate.anchor else ""
            table.add_row(
                str(shot.index),
                f"{float(shot.start):.1f}s",
                f"{float(shot.duration):.1f}s",
                candidate.rig.value,
                candidate.size.name.lower(),
                candidate.side.value,
                f"{float(candidate.score):.3f}",
                because,
            )
        console.print(table)
        if timeline.shot_count > show:
            console.print(f"  [dim]... and {timeline.shot_count - show} more[/]")

    for note in timeline.notes:
        console.print(f"\n  [dim]{note}[/]")
