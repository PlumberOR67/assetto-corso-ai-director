"""Measured statistics of real reference footage.

This module exists to answer a question the literature could not: **how long is a
shot, actually, in the styles this project imitates?**

No shot log, average-shot-length study or lens table exists for Initial D, MF
Ghost or Best Motoring in any language. That was a confirmed negative finding,
not a gap in searching. So rather than invent numbers or borrow them from an
unrelated corpus, the project measures footage directly and stamps the result
with :attr:`~assetto_ai_director.domain.provenance.Provenance.MEASURED`.

Everything here is a pure summary of numbers produced elsewhere. The extraction —
decoding video, detecting cuts, computing optical flow — lives in the
infrastructure layer where it belongs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

from assetto_ai_director.domain.units import Seconds

__all__ = [
    "CameraMotion",
    "MotionSummary",
    "ShotMeasurement",
    "StyleMeasurement",
]

MIN_SHOTS_FOR_SPECTRUM = 32
"""Fewest shots from which a 1/f spectral slope is worth reporting.

Below this the periodogram is dominated by its own variance and will happily
return a confident-looking slope for white noise. Reporting one anyway would
manufacture exactly the false precision this project is built to avoid.
"""

_MIN_SHOTS_FOR_STATS = 2


class CameraMotion(StrEnum):
    """What the camera was doing during a shot.

    Classified from dense optical flow, which measures apparent motion in the
    image without knowing anything about the scene. That is a real limitation: a
    locked-off camera watching a car cross frame produces flow, and so does a
    panning camera holding a stationary car. The classifier separates them by
    whether the flow is *global* — filling the frame coherently — or *local*.

    Attributes:
        STATIC: Locked off. Almost no global flow, or flow confined to part of
            the frame — a subject crossing a held frame is editorially a static
            shot even though the pixels move.
        PAN: Coherent lateral or vertical flow across the whole frame, without
            radial expansion. The camera turns; it does not travel.
        TRACK: Coherent lateral translation — a camera moving alongside the
            subject, keeping pace with it across the frame.
        AXIAL: Radial flow expanding from or converging on a point: the camera is
            travelling along its own optical axis.

            **This class deliberately merges two things** that optical flow
            cannot separate — a lens zoom and a forward or backward dolly both
            expand the image from a centre, and no amount of threshold tuning
            distinguishes them without scene geometry. In racing footage the
            dolly case dominates overwhelmingly, since chase, bonnet and bumper
            cameras are all travelling forward with the car.

            An earlier version of this classifier called the class ``ZOOM`` and
            reported that 60% of touge footage consisted of zoom shots, which is
            obviously false. Merging the classes reports less, but reports it
            truthfully, and the merged class is still exactly what the director
            needs to know: this is a forward-travelling shot, so map it to a
            chase, bonnet or follow rig.
        COMPLEX: Coherent but not dominated by any single mode, e.g. a drone move
            combining translation and rotation.
    """

    STATIC = "static"
    PAN = "pan"
    TRACK = "track"
    AXIAL = "axial"
    COMPLEX = "complex"

    @property
    def is_moving(self) -> bool:
        """Whether the camera itself moves during the shot.

        Returns:
            ``True`` for everything but a locked-off camera.
        """
        return self is not CameraMotion.STATIC


@dataclass(frozen=True, slots=True)
class ShotMeasurement:
    """One measured shot from reference footage.

    Attributes:
        index: Position in the clip, counting from zero.
        start: Start time in seconds from the clip's beginning.
        end: End time.
        motion: How the camera behaved.
        motion_magnitude: Camera speed in frame heights per second, from optical
            flow normalised by both frame height and frame rate so clips of
            different resolutions *and* different frame rates are comparable.
        motion_coherence: Fraction of flow vectors agreeing with the dominant
            direction, in ``[0, 1]``. High values mean the whole frame moves
            together, which is what distinguishes a camera move from a subject
            moving through a still frame.
    """

    index: int
    start: Seconds
    end: Seconds
    motion: CameraMotion
    motion_magnitude: float
    motion_coherence: float

    @property
    def duration(self) -> Seconds:
        """How long the shot lasted.

        Returns:
            The span in seconds.
        """
        return Seconds(self.end - self.start)


@dataclass(frozen=True, slots=True)
class MotionSummary:
    """How a style distributes its camera behaviour across shots.

    Attributes:
        counts: Number of shots of each motion class.
        mean_magnitude: Mean normalised flow magnitude across all shots.
    """

    counts: dict[CameraMotion, int] = field(default_factory=dict)
    mean_magnitude: float = 0.0

    @property
    def total(self) -> int:
        """Total shots counted.

        Returns:
            The shot count.
        """
        return sum(self.counts.values())

    def fraction(self, motion: CameraMotion) -> float:
        """Proportion of shots using a given camera behaviour.

        Args:
            motion: The behaviour to look up.

        Returns:
            Its share of all shots, or ``0.0`` when nothing was measured.
        """
        total = self.total
        return self.counts.get(motion, 0) / total if total else 0.0

    @property
    def static_fraction(self) -> float:
        """Share of shots with a locked-off camera.

        The most stylistically discriminating single number available from
        footage: it separates a grammar built on held, legible frames from one
        built on constant motion.

        Returns:
            The static share in ``[0, 1]``.
        """
        return self.fraction(CameraMotion.STATIC)


@dataclass(frozen=True, slots=True)
class StyleMeasurement:
    """Aggregate statistics for one style, measured from real footage.

    Attributes:
        style_name: Which style this describes.
        source_names: The clips measured.
        shots: Every measured shot, in order across the concatenated clips.
        total_duration: Combined length of the measured footage.
        clip_count: How many clips contributed.
    """

    style_name: str
    source_names: tuple[str, ...]
    shots: tuple[ShotMeasurement, ...]
    total_duration: Seconds
    clip_count: int

    @property
    def shot_count(self) -> int:
        """How many shots were measured.

        Returns:
            The shot count.
        """
        return len(self.shots)

    @property
    def durations(self) -> tuple[float, ...]:
        """Every shot duration, in order.

        Returns:
            The durations in seconds.
        """
        return tuple(float(shot.duration) for shot in self.shots)

    @property
    def average_shot_length(self) -> Seconds:
        """Mean shot duration — the standard measure of an edit's pace.

        Returns:
            The mean in seconds, or zero if nothing was measured.
        """
        values = self.durations
        return Seconds(sum(values) / len(values)) if values else Seconds(0.0)

    @property
    def median_shot_length(self) -> Seconds:
        """Median shot duration.

        Reported alongside the mean because shot-length distributions are
        strongly right-skewed: a handful of long establishing shots drag the mean
        well above what most cuts actually feel like.

        Returns:
            The median in seconds, or zero if nothing was measured.
        """
        values = sorted(self.durations)
        if not values:
            return Seconds(0.0)
        middle = len(values) // 2
        if len(values) % 2:
            return Seconds(values[middle])
        return Seconds((values[middle - 1] + values[middle]) * 0.5)

    @property
    def duration_spread(self) -> float:
        """Coefficient of variation of shot length, i.e. sigma over mu.

        Scale-free, so a fast style and a slow style with equally varied cutting
        report the same spread. Drives the variance of the generated rhythm.

        Returns:
            The ratio, or zero when it cannot be computed.
        """
        values = self.durations
        if len(values) < _MIN_SHOTS_FOR_STATS:
            return 0.0
        mean = sum(values) / len(values)
        if mean <= 0.0:
            return 0.0
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        return math.sqrt(variance) / mean

    def percentile(self, fraction: float) -> Seconds:
        """Look up a shot-length percentile.

        Args:
            fraction: Position in ``[0, 1]``; ``0.5`` is the median.

        Returns:
            The duration at that percentile, or zero if nothing was measured.
        """
        values = sorted(self.durations)
        if not values:
            return Seconds(0.0)
        position = min(len(values) - 1, max(0, round(fraction * (len(values) - 1))))
        return Seconds(values[position])

    @property
    def has_enough_shots_for_spectrum(self) -> bool:
        """Whether a spectral slope can honestly be reported.

        Returns:
            ``True`` once enough shots were measured for the periodogram to mean
            anything.
        """
        return self.shot_count >= MIN_SHOTS_FOR_SPECTRUM

    @property
    def motion(self) -> MotionSummary:
        """How camera behaviour is distributed across the measured shots.

        Returns:
            The motion summary.
        """
        counts: dict[CameraMotion, int] = {}
        for shot in self.shots:
            counts[shot.motion] = counts.get(shot.motion, 0) + 1
        magnitudes = [shot.motion_magnitude for shot in self.shots]
        mean = sum(magnitudes) / len(magnitudes) if magnitudes else 0.0
        return MotionSummary(counts, mean)

    @property
    def cuts_per_minute(self) -> float:
        """Cutting rate, the reciprocal view of average shot length.

        Returns:
            Cuts per minute, or zero when nothing was measured.
        """
        if self.total_duration <= 0.0:
            return 0.0
        return self.shot_count / (self.total_duration / 60.0)
