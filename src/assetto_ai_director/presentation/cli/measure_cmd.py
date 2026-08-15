"""The ``aidirector measure`` command: turn reference footage into style numbers.

Reports what was measured and, just as importantly, what could not be. A short
clip yields a usable average shot length but not a usable spectral slope, and the
command says so rather than printing a number that is really sampling noise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from assetto_ai_director.application.director.rhythm import spectral_slope
from assetto_ai_director.domain.measurement import (
    MIN_SHOTS_FOR_SPECTRUM,
    CameraMotion,
    ShotMeasurement,
    StyleMeasurement,
)
from assetto_ai_director.domain.provenance import Provenance, ProvenanceReport
from assetto_ai_director.domain.units import Seconds
from assetto_ai_director.presentation.cli.progress import RichProgress

__all__ = ["measure"]

console = Console()

VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v"}

_WHITE_NOISE_SLOPE = -0.35
"""Above this slope, shot lengths are statistically indistinguishable from independent."""

_STRONG_CORRELATION_SLOPE = -1.5
"""Below this slope, pacing drifts in long runs rather than varying shot to shot."""


def measure(
    source: Annotated[
        Path,
        typer.Argument(help="A video file, or a directory of them.", exists=True),
    ],
    *,
    style: Annotated[
        str, typer.Option("--style", "-s", help="Name for the measured style.")
    ] = "unnamed",
    per_clip: Annotated[
        bool, typer.Option("--per-clip/--combined", help="Also report each clip separately.")
    ] = True,
    out: Annotated[
        Path | None,
        typer.Option("--out", "-o", help="Write a measured style profile to this TOML file."),
    ] = None,
) -> None:
    """Measure shot lengths and camera behaviour in reference footage."""
    # Imported here so the command only requires the optional `measure` extra
    # when it is actually run — the rest of the CLI works without OpenCV.
    from assetto_ai_director.infrastructure.measure.video import (  # noqa: PLC0415
        measure_style,
    )

    paths = _collect(source)
    if not paths:
        console.print(f"[red]No video files found in {source}[/]")
        raise typer.Exit(code=1)

    console.print(f"Measuring [bold]{len(paths)}[/] clip(s) for style [bold cyan]{style}[/]\n")

    # Measured once. An earlier version measured each clip and then measured the
    # whole set again for the combined figures, doubling the runtime of the most
    # expensive command in the tool for no new information.
    with RichProgress(console) as progress:
        per_clip_results = [measure_style(style, [path], progress) for path in paths]

    if per_clip:
        for path, single in zip(paths, per_clip_results, strict=True):
            console.print(f"[dim]{path.name}[/]")
            _render_compact(single)
        console.print()

    combined = _combine(style, paths, per_clip_results)
    console.rule(f"[bold]{style}[/] — combined")
    _render(combined)

    if out is not None:
        _write_profile(combined, out)


def _write_profile(measurement: StyleMeasurement, destination: Path) -> None:
    """Derive a style profile from a measurement and write it.

    Args:
        measurement: The aggregate measurement.
        destination: Where to write the TOML profile.
    """
    from assetto_ai_director.application.director.styles import (  # noqa: PLC0415
        profile_from_measurement,
    )
    from assetto_ai_director.infrastructure.style.toml_io import write_profile  # noqa: PLC0415

    profile = profile_from_measurement(measurement)
    write_profile(profile, destination)

    report = ProvenanceReport.of(profile)
    console.print(f"\n[green]Wrote[/] {destination}")
    console.print(
        f"  [dim]tracked tunables: {report.count(Provenance.MEASURED)} measured, "
        f"{report.count(Provenance.VERIFIED)} verified, "
        f"{report.count(Provenance.GUESS)} guessed[/]"
    )
    if report.guesses:
        names = ", ".join(name for name, _ in report.guesses)
        console.print(f"  [yellow]still guessed:[/] {names}")

    # Stated explicitly because the tunable count above would otherwise read as
    # "everything in this file is evidence-backed". It is not: the six score
    # weights decide which moments get filmed at all, and nothing recoverable
    # from video says what they should be. They are judgement, they are not
    # provenance-tracked, and a report that quietly omitted them would be
    # exactly the false confidence this project exists to prevent.
    console.print(
        "  [yellow]not provenance-tracked:[/] the six [weights] values are "
        "editorial judgement, not measurements"
    )


def _combine(style: str, paths: list[Path], results: list[StyleMeasurement]) -> StyleMeasurement:
    """Merge per-clip measurements into one aggregate.

    Args:
        style: Style name.
        paths: The clips measured, in order.
        results: Their individual measurements.

    Returns:
        The combined measurement, with shot indices renumbered across clips.
    """
    shots: list[ShotMeasurement] = []
    for result in results:
        offset = len(shots)
        shots.extend(
            ShotMeasurement(
                index=offset + shot.index,
                start=shot.start,
                end=shot.end,
                motion=shot.motion,
                motion_magnitude=shot.motion_magnitude,
                motion_coherence=shot.motion_coherence,
            )
            for shot in result.shots
        )
    return StyleMeasurement(
        style_name=style,
        source_names=tuple(path.name for path in paths),
        shots=tuple(shots),
        total_duration=Seconds(sum(float(r.total_duration) for r in results)),
        clip_count=len(paths),
    )


def _collect(source: Path) -> list[Path]:
    """Gather video files from a path.

    Args:
        source: A file or directory.

    Returns:
        The video files found, sorted by name.
    """
    if source.is_file():
        return [source]
    return sorted(p for p in source.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES)


def _render_compact(measurement: StyleMeasurement) -> None:
    """Print a one-line summary for a single clip.

    Args:
        measurement: The clip's measurement.
    """
    motion = measurement.motion
    # Per-clip slope, not just the pooled one. Concatenating clips inserts
    # artificial boundaries into the sequence, and an artificial boundary is
    # uncorrelated with its neighbours — so pooling systematically *whitens* the
    # spectrum and understates the 1/f structure of each individual edit.
    slope = spectral_slope(measurement.durations)
    slope_text = (
        f"slope {slope:+.2f}"
        if slope is not None and measurement.has_enough_shots_for_spectrum
        else "slope    - "
    )
    console.print(
        f"  {measurement.shot_count:>4} shots  "
        f"ASL {float(measurement.average_shot_length):>5.2f}s  "
        f"median {float(measurement.median_shot_length):>5.2f}s  "
        f"sigma/mu {measurement.duration_spread:>4.2f}  "
        f"static {motion.static_fraction * 100:>3.0f}%  "
        f"{measurement.cuts_per_minute:>5.1f} cuts/min  "
        f"{slope_text}"
    )


def _render(measurement: StyleMeasurement) -> None:
    """Print the full report for a style.

    Args:
        measurement: The aggregate measurement.
    """
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim")
    table.add_column()

    table.add_row("Clips", str(measurement.clip_count))
    table.add_row("Footage", f"{float(measurement.total_duration):.1f} s")
    table.add_row("Shots", str(measurement.shot_count))
    table.add_row("Average shot length", f"{float(measurement.average_shot_length):.2f} s")
    table.add_row("Median shot length", f"{float(measurement.median_shot_length):.2f} s")
    table.add_row("Spread (sigma/mu)", f"{measurement.duration_spread:.2f}")
    table.add_row("Cutting rate", f"{measurement.cuts_per_minute:.1f} cuts/min")
    table.add_row(
        "Shot length range",
        f"{float(measurement.percentile(0.1)):.2f} s (p10) "
        f"- {float(measurement.percentile(0.9)):.2f} s (p90)",
    )

    slope = spectral_slope(measurement.durations)
    if not measurement.has_enough_shots_for_spectrum or slope is None:
        table.add_row(
            "Rhythm structure",
            f"[yellow]not reported — {measurement.shot_count} shots, "
            f"need {MIN_SHOTS_FOR_SPECTRUM}[/]",
        )
    else:
        table.add_row("Rhythm structure", f"{slope:+.2f} log-log slope  {_slope_reading(slope)}")

    console.print(table)

    motion = measurement.motion
    console.print("\n  [dim]Camera behaviour[/]")
    motion_table = Table(show_header=False, box=None, padding=(0, 2))
    motion_table.add_column(style="dim")
    motion_table.add_column(justify="right")
    motion_table.add_column()
    for kind in CameraMotion:
        count = motion.counts.get(kind, 0)
        if not count:
            continue
        share = motion.fraction(kind)
        bar = "#" * round(share * 30)
        motion_table.add_row(kind.value, f"{count} ({share * 100:.0f}%)", f"[cyan]{bar}[/]")
    console.print(motion_table)
    console.print(f"  [dim]mean camera speed {motion.mean_magnitude:.2f} frame heights/second[/]")


def _slope_reading(slope: float) -> str:
    """Translate a spectral slope into plain language.

    Args:
        slope: The fitted log-log slope.

    Returns:
        A short interpretation.
    """
    if slope > _WHITE_NOISE_SLOPE:
        return "[yellow]near-white — shot lengths look independent[/]"
    if slope > _STRONG_CORRELATION_SLOPE:
        return "[green]pink-ish — real 1/f editing structure[/]"
    return "[yellow]strongly correlated — long drifting runs[/]"
