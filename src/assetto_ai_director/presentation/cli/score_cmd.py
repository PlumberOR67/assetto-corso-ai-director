"""The ``aidirector score`` command: how good is each moment, and to whom?

Prints the five moment scores under a chosen style's weighting, plus the peaks
each score found. Its real job is to make the director's taste inspectable before
any camera is placed: if the top-scoring moments look wrong to a human watching
the same replay, the weights are wrong, and that is much cheaper to discover here
than after a full edit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from assetto_ai_director.application.director.styles import BUILTIN_STYLES
from assetto_ai_director.application.events.detect import CarSignals
from assetto_ai_director.application.scoring.features import compute_features
from assetto_ai_director.application.scoring.moments import (
    RISK_LIMITATION_NOTE,
    score_car,
)
from assetto_ai_director.domain.scoring import ScoreKind, ScoreVector, ScoreWeights
from assetto_ai_director.infrastructure.acreplay.frames import read_car_frames
from assetto_ai_director.infrastructure.acreplay.structure import CarBlock, read_structure
from assetto_ai_director.presentation.cli.console import SUPPORTS_BLOCK_GLYPHS
from assetto_ai_director.presentation.cli.progress import RichProgress

__all__ = ["score"]

console = Console()

_TOP_MOMENTS = 10
_HISTOGRAM_BINS = 10
_MIN_TRAVELLED_METRES = 50.0
"""Distance a car must cover to be worth scoring. Empty grid slots never move."""


def score(
    replay: Annotated[
        Path, typer.Argument(help="Replay file to score.", exists=True, dir_okay=False)
    ],
    *,
    style: Annotated[
        str, typer.Option("--style", "-s", help="Style profile to weight the scores by.")
    ] = "initial_d",
    seconds: Annotated[
        float, typer.Option("--seconds", help="Score only the first N seconds. 0 means all.")
    ] = 0.0,
    car: Annotated[
        int | None, typer.Option("--car", help="Score one car. Defaults to the busiest.")
    ] = None,
) -> None:
    """Score every moment of a replay under a style's weighting."""
    if style not in BUILTIN_STYLES:
        available = ", ".join(sorted(BUILTIN_STYLES))
        console.print(f"[red]Unknown style {style!r}.[/] Available: {available}")
        raise typer.Exit(code=1)
    profile = BUILTIN_STYLES[style]

    structure = read_structure(replay)
    rate = float(structure.sample_rate)
    limit = min((block.frame_count for block in structure.cars), default=0)
    if seconds > 0.0:
        limit = min(limit, int(seconds * rate))
    if limit <= 0:
        console.print("[red]Replay contains no frames.[/]")
        raise typer.Exit(code=1)

    signals = _decode(replay, structure, limit, rate)
    if not signals:
        console.print("[red]No car in this replay moved.[/]")
        raise typer.Exit(code=1)

    subject = car if car is not None else _default_subject(structure, signals)
    if subject not in signals:
        console.print(f"[red]Car {subject} did not move, or does not exist.[/]")
        raise typer.Exit(code=1)

    time = np.arange(limit, dtype=np.float64) / rate
    features = compute_features(signals, time, rate)
    scores = score_car(features, subject)

    names = {b.identity.car_id: b.identity.short_model for b in structure.cars}
    console.print(
        f"[bold cyan]{structure.to_session_meta().track_label}[/]  "
        f"car {subject} ({names.get(subject, '?')})  "
        f"style [bold]{style}[/]  {limit / rate:.0f}s\n"
    )
    _render_distributions(scores)
    _render_top(scores, time, profile.weights, style)
    console.print(f"\n  [dim]{RISK_LIMITATION_NOTE}[/]")
    if not profile.is_measured:
        console.print(
            f"  [yellow]Style {style!r} carries unmeasured weights[/] — "
            f"[dim]run `aidirector measure` on reference footage to replace them.[/]"
        )


def _default_subject(structure: object, signals: dict[int, CarSignals]) -> int:
    """Choose which car to score when the user did not say.

    Prefers the player's car, identified structurally from the replay rather than
    by assuming car zero. Picking the fastest-moving car instead looks reasonable
    and is wrong in practice: on one real fixture it selects a grippy all-wheel-
    drive car that never slides, over the rear-drive car the session is about.

    Args:
        structure: The parsed replay structure.
        signals: Cars that actually moved.

    Returns:
        The car id to score.
    """
    for block in getattr(structure, "cars", ()):
        if block.identity.is_player and block.identity.car_id in signals:
            return int(block.identity.car_id)
    return max(signals, key=lambda car_id: float(signals[car_id].speed.sum()))


def _decode(replay: Path, structure: object, limit: int, rate: float) -> dict[int, CarSignals]:
    """Decode every car that actually moved.

    Args:
        replay: The replay file.
        structure: The parsed replay structure.
        limit: Frames to decode.
        rate: Sample rate.

    Returns:
        Signals per car id, excluding parked entries.
    """
    blocks = getattr(structure, "cars", ())
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
            travelled = float(
                np.linalg.norm(np.diff(decoded.position[:, [0, 2]], axis=0), axis=1).sum()
            )
            if travelled >= _MIN_TRAVELLED_METRES:
                signals[block.identity.car_id] = CarSignals(
                    car_id=block.identity.car_id,
                    position=decoded.position,
                    heading=decoded.heading,
                    speed=decoded.speed,
                    sideslip=decoded.sideslip,
                )
            progress.advance()
    return signals


def _render_distributions(scores: list[ScoreVector]) -> None:
    """Print a histogram per score.

    Args:
        scores: Per-frame score vectors.
    """
    table = Table(header_style="dim", box=None, padding=(0, 2))
    table.add_column("score")
    table.add_column("mean", justify="right")
    table.add_column("p95", justify="right")
    table.add_column("distribution  0 -> 1")

    for kind in ScoreKind:
        values = np.array([vector[kind] for vector in scores])
        counts, _ = np.histogram(values, bins=_HISTOGRAM_BINS, range=(0.0, 1.0))
        peak = counts.max() or 1
        blocks = "".join(_block(count / peak) for count in counts)
        table.add_row(
            kind.value,
            f"{values.mean():.3f}",
            f"{np.percentile(values, 95):.3f}",
            f"[cyan]{blocks}[/]",
        )
    console.print(table)


def _render_top(
    scores: list[ScoreVector], time: np.ndarray, weights: ScoreWeights, style: str
) -> None:
    """Print the highest-scoring moments under the style's weighting.

    Args:
        scores: Per-frame score vectors.
        time: Time per frame.
        weights: The style's score weights.
        style: The style name, for the heading.
    """
    combined = np.array([weights.combine(vector) for vector in scores])
    # Keep peaks apart so the list is ten different moments rather than ten
    # frames of one.
    spacing = max(int(len(combined) / 200), 1)
    order = np.argsort(combined)[::-1]
    chosen: list[int] = []
    for index in order:
        if all(abs(int(index) - kept) > spacing * 10 for kept in chosen):
            chosen.append(int(index))
        if len(chosen) >= _TOP_MOMENTS:
            break

    console.print(f"\n  [dim]best moments for {style}[/]")
    table = Table(header_style="dim", box=None, padding=(0, 2))
    table.add_column("t", justify="right")
    table.add_column("total", justify="right")
    for kind in ScoreKind:
        table.add_column(kind.value[:4], justify="right")

    for index in sorted(chosen, key=lambda i: -combined[i]):
        vector = scores[index]
        table.add_row(
            f"{time[index]:.1f}s",
            f"[bold]{combined[index]:.3f}[/]",
            *[f"{vector[kind]:.2f}" for kind in ScoreKind],
        )
    console.print(table)


_BLOCK_LADDER = " ▁▂▃▄▅▆▇█"
_ASCII_LADDER = " .:-=+*#@"
"""Used where the console cannot encode block glyphs.

Nine steps, like the block ladder, and ordered by visual weight so the sparkline
still reads as a height. A row of substitution marks would not — which is why
this one place asks rather than relying on the stdio safety net.
"""


def _block(fraction: float) -> str:
    """Pick a bar character for a normalised height.

    Args:
        fraction: Height in ``[0, 1]``.

    Returns:
        A single character from whichever ladder this console can print.
    """
    ladder = _BLOCK_LADDER if SUPPORTS_BLOCK_GLYPHS else _ASCII_LADDER
    return ladder[min(len(ladder) - 1, int(fraction * (len(ladder) - 1) + 0.5))]
