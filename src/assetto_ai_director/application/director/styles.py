"""Building style profiles, from measurement or from judgement.

Two routes in, and the difference between them is recorded rather than blurred.
:func:`profile_from_measurement` derives a profile from real footage and stamps
every value it took from that footage as
:attr:`~assetto_ai_director.domain.provenance.Provenance.MEASURED`, naming the
clips. :data:`BUILTIN_STYLES` holds hand-authored starting points whose numbers
are honestly labelled as guesses.

A profile mixes both. Nothing recoverable from video tells you how much a style
should weight risk against flow, so those weights stay judgement; the pace and
camera mix come from measurement once footage exists.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from assetto_ai_director.domain.camera import RigKind
from assetto_ai_director.domain.measurement import CameraMotion
from assetto_ai_director.domain.provenance import Provenance, Tunable
from assetto_ai_director.domain.scoring import ScoreWeights
from assetto_ai_director.domain.style import CameraMixProfile, RhythmProfile, StyleProfile

if TYPE_CHECKING:
    from assetto_ai_director.domain.measurement import StyleMeasurement

__all__ = ["BUILTIN_STYLES", "profile_from_measurement"]

_FALLBACK_SLOPE = -1.0
"""Pink noise, used when a corpus is too small for its own slope to be estimated.

Cited rather than invented: real film shot-length sequences are 1/f-correlated
(Cutting, DeLong & Nothelfer, *Psychological Science* 2010). It is a far better
default than white noise, which is what an independent sampler would produce.
"""

_MIN_SHOT_PERCENTILE = 0.05
_MAX_SHOT_PERCENTILE = 0.95
"""Percentiles used for the shot-length floor and ceiling.

Not the observed minimum and maximum: those are single samples and often
artefacts — a flash frame at one end, a title card at the other.
"""


def profile_from_measurement(
    measurement: StyleMeasurement,
    *,
    description: str = "",
    weights: ScoreWeights | None = None,
    seed: int = 0,
) -> StyleProfile:
    """Derive a style profile from measured reference footage.

    Pace and camera mix come from the measurement and are marked measured. Score
    weights cannot be recovered from video — nothing in a frame says how much a
    style values risk over flow — so they stay judgement and are labelled as such.

    Args:
        measurement: The measured footage.
        description: Human-readable summary of the style.
        weights: Score weights to attach. Defaults to a neutral set.
        seed: Random seed for the rhythm generator.

    Returns:
        The profile, with measured values sourced to the clips they came from.
    """
    # Imported here to keep the module import-time free of numpy work.
    from assetto_ai_director.application.director.rhythm import spectral_slope  # noqa: PLC0415

    source = f"measured: {measurement.clip_count} clips, {measurement.shot_count} shots"
    slope = spectral_slope(measurement.durations)
    use_slope = slope is not None and measurement.has_enough_shots_for_spectrum

    rhythm = RhythmProfile(
        median_shot_length=Tunable(
            float(measurement.median_shot_length), Provenance.MEASURED, source=source
        ),
        spread=Tunable(measurement.duration_spread, Provenance.MEASURED, source=source),
        spectral_slope=(
            Tunable(float(slope), Provenance.MEASURED, source=source)
            if use_slope and slope is not None
            else Tunable(
                _FALLBACK_SLOPE,
                Provenance.VERIFIED,
                source="Cutting, DeLong & Nothelfer 2010: film shot lengths are 1/f",
                note=(
                    f"corpus of {measurement.shot_count} shots is too small to "
                    f"estimate its own slope"
                ),
            )
        ),
        min_shot_length=Tunable(
            float(measurement.percentile(_MIN_SHOT_PERCENTILE)), Provenance.MEASURED, source=source
        ),
        max_shot_length=Tunable(
            float(measurement.percentile(_MAX_SHOT_PERCENTILE)), Provenance.MEASURED, source=source
        ),
    )

    motion = measurement.motion
    camera_mix = CameraMixProfile(
        static_fraction=Tunable(motion.static_fraction, Provenance.MEASURED, source=source),
        axial_fraction=Tunable(
            motion.fraction(CameraMotion.AXIAL), Provenance.MEASURED, source=source
        ),
        track_fraction=Tunable(
            motion.fraction(CameraMotion.TRACK), Provenance.MEASURED, source=source
        ),
        pan_fraction=Tunable(motion.fraction(CameraMotion.PAN), Provenance.MEASURED, source=source),
    )

    return StyleProfile(
        name=measurement.style_name,
        description=description or f"Measured from {measurement.clip_count} reference clips.",
        rhythm=rhythm,
        camera_mix=camera_mix,
        weights=weights or ScoreWeights(),
        seed=seed,
        measured_from=measurement.source_names,
    )


def _guess_rhythm(median: float, spread: float, floor: float, ceiling: float) -> RhythmProfile:
    """Build a hand-authored rhythm profile with honest provenance.

    Args:
        median: Target median shot length in seconds.
        spread: Target coefficient of variation.
        floor: Minimum shot length.
        ceiling: Maximum shot length.

    Returns:
        The profile, every value marked as a guess.
    """
    return RhythmProfile(
        median_shot_length=Tunable(median, Provenance.GUESS, note="replace by measurement"),
        spread=Tunable(spread, Provenance.GUESS, note="replace by measurement"),
        spectral_slope=Tunable(
            _FALLBACK_SLOPE,
            Provenance.VERIFIED,
            source="Cutting, DeLong & Nothelfer 2010: film shot lengths are 1/f",
        ),
        min_shot_length=Tunable(floor, Provenance.GUESS),
        max_shot_length=Tunable(ceiling, Provenance.GUESS),
    )


def _guess_mix(
    static: float, axial: float, track: float, pan: float, rigs: frozenset[RigKind]
) -> CameraMixProfile:
    """Build a hand-authored camera mix with honest provenance.

    Args:
        static: Target share of locked-off shots.
        axial: Target share travelling along the optical axis.
        track: Target share translating laterally.
        pan: Target share rotating in place.
        rigs: Rigs the style permits.

    Returns:
        The mix, every fraction marked as a guess.
    """
    return CameraMixProfile(
        static_fraction=Tunable(static, Provenance.GUESS),
        axial_fraction=Tunable(axial, Provenance.GUESS),
        track_fraction=Tunable(track, Provenance.GUESS),
        pan_fraction=Tunable(pan, Provenance.GUESS),
        allowed_rigs=rigs,
    )


_NO_DRONES = frozenset(RigKind) - {RigKind.DRONE, RigKind.HELICOPTER, RigKind.ORBITAL}

BUILTIN_STYLES: dict[str, StyleProfile] = {
    "initial_d": StyleProfile(
        name="initial_d",
        description=(
            "Held, legible frames. Low roadside cameras, long lenses, minimal "
            "shake. The technique must stay readable, so coverage never competes "
            "with the driving."
        ),
        rhythm=_guess_rhythm(2.6, 1.1, 0.8, 9.0),
        camera_mix=_guess_mix(0.55, 0.35, 0.05, 0.05, _NO_DRONES),
        weights=ScoreWeights(
            drift=0.35, speed=0.15, risk=0.15, flow=0.10, cinematic=0.25, legibility=0.9
        ),
    ),
    "mf_ghost": StyleProfile(
        name="mf_ghost",
        description="Modern aerial coverage: drones, fast flythroughs, FPV.",
        rhythm=_guess_rhythm(2.2, 1.0, 0.6, 7.0),
        camera_mix=_guess_mix(0.20, 0.55, 0.20, 0.05, frozenset(RigKind)),
        weights=ScoreWeights(
            drift=0.15, speed=0.25, risk=0.15, flow=0.20, cinematic=0.25, legibility=0.5
        ),
    ),
    "best_motoring": StyleProfile(
        name="best_motoring",
        description=(
            "Television coverage. Chase-car tracking, in-car with the driver, "
            "technical inserts, longer takes that let a sequence connect."
        ),
        rhythm=_guess_rhythm(3.5, 0.9, 1.0, 12.0),
        camera_mix=_guess_mix(0.40, 0.45, 0.10, 0.05, _NO_DRONES),
        weights=ScoreWeights(
            drift=0.20, speed=0.10, risk=0.10, flow=0.40, cinematic=0.20, legibility=0.8
        ),
    ),
    "fast_furious": StyleProfile(
        name="fast_furious",
        description="Aggressive movement, whip pans, undercar, quick cuts, speed ramps.",
        rhythm=_guess_rhythm(1.4, 1.2, 0.4, 5.0),
        camera_mix=_guess_mix(0.10, 0.55, 0.25, 0.10, frozenset(RigKind)),
        weights=ScoreWeights(
            drift=0.20, speed=0.30, risk=0.30, flow=0.05, cinematic=0.15, legibility=0.1
        ),
    ),
    "broadcast": StyleProfile(
        name="broadcast",
        description="Gran Turismo replay theatre: trackside cameras with FOV zoom tracking.",
        rhythm=_guess_rhythm(3.0, 0.8, 1.2, 10.0),
        camera_mix=_guess_mix(0.65, 0.25, 0.05, 0.05, _NO_DRONES),
        weights=ScoreWeights(
            drift=0.10, speed=0.20, risk=0.20, flow=0.25, cinematic=0.25, legibility=0.6
        ),
    ),
}
"""Hand-authored starting points.

Their pace and camera-mix numbers are **guesses** and say so. No shot log,
average-shot-length study or lens table exists for any of these styles in any
language — that was a confirmed negative finding, not a gap in searching. The
measurement harness is how they stop being guesses.
"""
