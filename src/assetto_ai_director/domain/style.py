"""Style profiles: what makes an Initial D cut different from a Best Motoring cut.

A style is not a score and not a filter. It is the set of parameters that decide
how the director values moments, how fast it cuts, and which camera rigs it is
allowed to use.

**Every number here carries its provenance.** That is the point of the type: a
reader can tell at a glance whether a threshold came from measured footage, from
a cited source, from arithmetic, or from taste. ``aidirector provenance-report``
turns that into an audit, and the measurement harness exists to convert guesses
into measurements.

**Pace and camera mix are independent parameters, deliberately.** An earlier
design coupled them behind a single "energy" dial, on the observation that fast
cutting travelled with moving cameras across a five-clip corpus. Measuring nine
clips did not support it — the correlation between average shot length and
static-camera fraction came out at roughly r = 0.4, which at n=9 is not a
finding. Only the fast end held: the quickest edits really are almost never
locked off. The slow end scattered from 30 to 78 percent static at similar pace.
So the dial was removed rather than kept as a plausible-sounding fiction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from assetto_ai_director.domain.camera import RigKind
from assetto_ai_director.domain.provenance import Provenance, Tunable
from assetto_ai_director.domain.scoring import ScoreWeights
from assetto_ai_director.domain.units import Seconds

__all__ = ["CameraMixProfile", "RhythmProfile", "StyleProfile"]


@dataclass(frozen=True, slots=True)
class RhythmProfile:
    """How a style paces its cutting.

    Attributes:
        median_shot_length: Target median duration. **Median, not mean**, and the
            distinction is load-bearing for this material: measured shot-length
            distributions are strongly right-skewed and sometimes bimodal. One
            reference clip has a 6.11 s mean against a 2.23 s median, because it
            is mostly short shots punctuated by a few very long ones. Targeting
            the mean there would produce an edit resembling neither.
        spread: Coefficient of variation, sigma over mu. Scale-free, so it
            transfers between a fast style and a slow one.
        spectral_slope: Target log-log slope of the shot-length sequence. Zero is
            white noise — independent lengths, which real editing is not. Around
            minus one is the pink-noise structure observed in film.
        min_shot_length: Floor, below which a shot cannot be read.
        max_shot_length: Ceiling, above which a shot outstays its welcome.
    """

    median_shot_length: Tunable[float]
    spread: Tunable[float]
    spectral_slope: Tunable[float]
    min_shot_length: Tunable[float]
    max_shot_length: Tunable[float]

    @property
    def target_mean(self) -> Seconds:
        """Approximate arithmetic mean implied by the median and spread.

        For a log-normal distribution the mean sits above the median by
        ``exp(sigma^2 / 2)``. Exposed because the rhythm generator parameterises
        on the mean while the profile is authored on the median.

        Returns:
            The implied mean shot length in seconds.
        """
        import math  # noqa: PLC0415 — kept local so the domain has no import-time cost

        sigma_squared = math.log1p(max(self.spread.value, 0.0) ** 2)
        return Seconds(self.median_shot_length.value * math.exp(sigma_squared * 0.5))


@dataclass(frozen=True, slots=True)
class CameraMixProfile:
    """How a style distributes camera behaviour across shots.

    Fractions are targets rather than quotas: the scheduler weights candidate
    generation by them, but geometric feasibility can still override. A trackside
    camera that cannot see the car is not placed merely to hit a percentage.

    Attributes:
        static_fraction: Share of shots that should be locked off.
        axial_fraction: Share travelling along the optical axis — chase, bonnet,
            bumper, follow-drone.
        track_fraction: Share translating laterally alongside the subject.
        pan_fraction: Share rotating in place.
        allowed_rigs: Rigs this style permits at all. A style that forbids drones
            expresses it here rather than through a zero weight, so an
            unsatisfiable request fails loudly instead of silently drifting.
    """

    static_fraction: Tunable[float]
    axial_fraction: Tunable[float]
    track_fraction: Tunable[float]
    pan_fraction: Tunable[float]
    allowed_rigs: frozenset[RigKind] = field(default_factory=lambda: frozenset(RigKind))

    def permits(self, rig: RigKind) -> bool:
        """Test whether a rig may be used by this style.

        Args:
            rig: The rig in question.

        Returns:
            ``True`` if the style allows it.
        """
        return rig in self.allowed_rigs


@dataclass(frozen=True, slots=True)
class StyleProfile:
    """A complete directorial style.

    Attributes:
        name: Identifier, e.g. ``touge_reference``.
        description: Human-readable summary.
        rhythm: Cutting pace and its structure.
        camera_mix: Camera behaviour distribution.
        weights: How the style values each moment score.
        seed: Random seed for the rhythm generator, so an edit is reproducible.
        measured_from: Names of the reference clips this was measured from, empty
            when the profile is hand-authored.
    """

    name: str
    description: str
    rhythm: RhythmProfile
    camera_mix: CameraMixProfile
    weights: ScoreWeights
    seed: int = 0
    measured_from: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_measured(self) -> bool:
        """Whether this profile's pace came from real footage.

        Returns:
            ``True`` if the median shot length carries measured provenance.
        """
        return self.rhythm.median_shot_length.provenance is Provenance.MEASURED
