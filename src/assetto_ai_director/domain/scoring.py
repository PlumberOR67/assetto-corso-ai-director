"""Moment scores: how good is this instant, and good in what way.

The project brief asked for seven scores. Building them as seven peers would have
produced a confused design, because two of them are not moment scores at all:

* **Style** is not a property of a moment. It is the weight vector, the hard
  constraints and the rhythm parameters *applied to* the other scores. A moment
  is not "more Initial D" in itself; Initial D is a way of valuing moments.
* **Smoothness** is not a property of a moment either. It is a cost on a camera
  *path*, and it can only be evaluated once a shot has been chosen and a
  trajectory generated. Scoring it beforehand would mean scoring a camera move
  that does not exist yet.

So this module defines the five genuine moment scores, and the other two live
where they belong: style in the style profile, smoothness as a path cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from assetto_ai_director.domain.units import Score, Seconds

__all__ = ["MomentScore", "ScoreKind", "ScoreVector", "ScoreWeights"]


class ScoreKind(StrEnum):
    """The five orthogonal ways a moment can be worth filming.

    Attributes:
        DRIFT: How far past the limit of adhesion the car is, sustained and
            speed-modulated so a parking-lot spin does not outrank a committed
            slide through a fast sweeper.
        SPEED: How fast, relative to *this session's* own distribution rather
            than an absolute number. Percentile-based so the score means the same
            thing on a mountain pass and on a high-speed circuit.
        RISK: How close to disaster — grip at the limit, cars in proximity and
            closing, damage, violent lateral jerk.
        FLOW: How clean and connected the driving is. The inverse of jerk over a
            window, weighted by how monotonic the throttle trace is. This is the
            score that values a beautifully linked sequence of corners, and it is
            what television-style coverage is actually looking for.
        CINEMATIC: How well the moment *photographs*, which is not the same as
            how dramatic it is. Two cars at a framable separation, headings
            spread enough to read as distinct, elevation change in the
            background, a manoeuvre legible from outside the car.
    """

    DRIFT = "drift"
    SPEED = "speed"
    RISK = "risk"
    FLOW = "flow"
    CINEMATIC = "cinematic"


@dataclass(frozen=True, slots=True)
class ScoreVector:
    """The five scores for one moment, each normalised to ``[0, 1]``.

    Attributes:
        drift: Drift score.
        speed: Speed score.
        risk: Risk score.
        flow: Flow score.
        cinematic: Cinematic score.
    """

    drift: Score = Score(0.0)
    speed: Score = Score(0.0)
    risk: Score = Score(0.0)
    flow: Score = Score(0.0)
    cinematic: Score = Score(0.0)

    def __post_init__(self) -> None:
        """Validate that every component is a normalised score.

        Raises:
            ValueError: If any component falls outside ``[0, 1]``. Scores are
                combined by weighted sum, so an out-of-range component silently
                distorts every comparison it takes part in.
        """
        for kind in ScoreKind:
            value = self[kind]
            if not 0.0 <= value <= 1.0:
                message = f"Score {kind.value} must be in [0, 1], got {value}."
                raise ValueError(message)

    def __getitem__(self, kind: ScoreKind) -> Score:
        """Read one component by kind.

        Args:
            kind: Which score to read.

        Returns:
            That component's value.
        """
        match kind:
            case ScoreKind.DRIFT:
                return self.drift
            case ScoreKind.SPEED:
                return self.speed
            case ScoreKind.RISK:
                return self.risk
            case ScoreKind.FLOW:
                return self.flow
            case ScoreKind.CINEMATIC:
                return self.cinematic

    @property
    def peak(self) -> Score:
        """The strongest single component.

        Useful for "was anything interesting happening at all", independent of
        which style is asking.

        Returns:
            The maximum component value.
        """
        return Score(max(self[kind] for kind in ScoreKind))


@dataclass(frozen=True, slots=True)
class ScoreWeights:
    """How a style values each score, plus its tolerance for illegibility.

    **These are the least defensible numbers in the system, and deliberately not
    wrapped in :class:`~assetto_ai_director.domain.provenance.Tunable`.** Nothing
    recoverable from reference footage says how much a style should value risk
    against flow — that is an editorial claim about intent, not a measurable
    property of frames. Wrapping them in a provenance type would let them be
    labelled ``MEASURED`` by some future code path and inherit authority they
    cannot have. Leaving them as plain floats keeps them visibly outside the
    audit, and the measurement command says so explicitly when it writes a
    profile.

    Attributes:
        drift: Weight on the drift score.
        speed: Weight on the speed score.
        risk: Weight on the risk score.
        flow: Weight on the flow score.
        cinematic: Weight on the cinematic score.
        legibility: Penalty applied to moments whose defining technique would be
            lost under aggressive coverage.

            This encodes the sharpest stylistic distinction in the whole system.
            A static, held, legible frame is what lets an audience actually read
            a drift; frenetic coverage of the same moment converts craft into
            noise. Styles that depend on the viewer understanding *what the car
            is doing* pay a high penalty for illegibility; styles that trade
            comprehension for energy pay almost none.
    """

    drift: float = 0.2
    speed: float = 0.2
    risk: float = 0.2
    flow: float = 0.2
    cinematic: float = 0.2
    legibility: float = 0.5

    def __post_init__(self) -> None:
        """Validate weight signs.

        Raises:
            ValueError: If any weight is negative. A negative weight would mean
                a style actively seeks *worse* moments of that kind, which is
                never the intent and is almost always a typo.
        """
        for name in ("drift", "speed", "risk", "flow", "cinematic", "legibility"):
            value: float = getattr(self, name)
            if value < 0.0:
                message = f"Weight {name!r} must be non-negative, got {value}."
                raise ValueError(message)

    @property
    def total(self) -> float:
        """The sum of the five score weights, excluding the legibility penalty.

        Returns:
            The weight total, used for normalisation.
        """
        return self.drift + self.speed + self.risk + self.flow + self.cinematic

    def weight_of(self, kind: ScoreKind) -> float:
        """Read the weight for one score kind.

        Args:
            kind: Which score.

        Returns:
            That score's weight.
        """
        match kind:
            case ScoreKind.DRIFT:
                return self.drift
            case ScoreKind.SPEED:
                return self.speed
            case ScoreKind.RISK:
                return self.risk
            case ScoreKind.FLOW:
                return self.flow
            case ScoreKind.CINEMATIC:
                return self.cinematic

    def combine(self, scores: ScoreVector, illegibility: float = 0.0) -> Score:
        """Reduce a score vector to the single number this style cares about.

        Implements ``S(m) = sum_k w_k * score_k(m) - lambda * illegibility(m)``,
        normalised by the weight total so styles with different weight scales
        stay comparable.

        Args:
            scores: The moment's five component scores.
            illegibility: How poorly this moment would survive aggressive
                coverage, in ``[0, 1]``.

        Returns:
            The combined score, clamped to ``[0, 1]``.
        """
        total = self.total
        if total <= 0.0:
            return Score(0.0)
        weighted = sum(self.weight_of(kind) * scores[kind] for kind in ScoreKind) / total
        penalised = weighted - self.legibility * illegibility
        return Score(min(1.0, max(0.0, penalised)))


@dataclass(frozen=True, slots=True)
class MomentScore:
    """A scored instant, ready to be turned into shot candidates.

    Attributes:
        moment: When, in seconds from session start.
        subjects: The cars this score is about, most important first.
        scores: The five component scores.
        illegibility: How poorly the moment would survive aggressive coverage.
    """

    moment: Seconds
    subjects: tuple[int, ...]
    scores: ScoreVector
    illegibility: float = 0.0

    def value_for(self, weights: ScoreWeights) -> Score:
        """Evaluate this moment under a particular style's weighting.

        Args:
            weights: The style's weights.

        Returns:
            The combined score.
        """
        return weights.combine(self.scores, self.illegibility)
