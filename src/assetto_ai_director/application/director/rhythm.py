"""Cutting rhythm: measuring it, and generating it.

The finding this module is built on is the most robust quantitative fact in the
whole cinematography literature this project could verify: **shot lengths in real
films are 1/f-correlated, not independent**. Cutting, DeLong and Nothelfer showed
that Hollywood shot-length sequences drift toward a 1/f (pink noise) spectrum
over time, with action films showing the steepest approach.

Why it matters here: an edit whose shot durations are drawn independently from
the right distribution will have the correct average shot length, the correct
variance, and still feel mechanically wrong. Viewers notice the difference
without being able to name it. Independent draws produce white noise — no
correlation between neighbouring shots — whereas real editing has runs: a
sequence of quick cuts, then a longer held shot, then quick again.

So this module does two things:

* :func:`spectral_slope` measures the correlation structure of an observed
  sequence, used to characterise reference footage.
* :func:`pink_noise_durations` generates a target sequence with that structure,
  used to drive the scheduler.

The scheduler cannot enforce a spectral property directly — a spectrum is global
and dynamic programming is local. The trick is to pre-sample a target sequence
here, then have the scheduler pay a local cost for deviating from the k-th
target. That converts a global requirement into an additive local term.
"""

from __future__ import annotations

import math

import numpy as np

from assetto_ai_director.domain.units import Seconds

__all__ = ["pink_noise_durations", "spectral_slope"]

_MIN_POINTS_FOR_SPECTRUM = 8
_VOSS_OCTAVES = 5
_STD_EPSILON = 1e-12
"""Below this a standard deviation is numerically indistinguishable from zero."""


def spectral_slope(durations: tuple[float, ...] | list[float]) -> float | None:
    """Estimate the log-log spectral slope of a shot-length sequence.

    Computes a periodogram of the mean-removed log-durations and fits a straight
    line to it in log-log space. Log-durations rather than raw durations because
    shot lengths are approximately log-normally distributed, so the log domain is
    where the additive structure lives.

    Interpretation:

    * ``0`` means white noise — successive shot lengths are independent. This is
      what a naive random generator produces, and what real editing is not.
    * ``-1`` is pink noise, the 1/f structure observed in real films.
    * More negative still means heavily correlated, drifting pacing.

    Args:
        durations: Shot durations in seconds, in edit order. Order carries the
            entire signal; sorting the input destroys what is being measured.

    Returns:
        The fitted slope, or ``None`` if there are too few shots for the estimate
        to mean anything. Returning ``None`` rather than a number is deliberate:
        a short sequence yields a confident-looking slope that is pure sampling
        noise, and reporting it would manufacture false precision.
    """
    values = [value for value in durations if value > 0.0]
    if len(values) < _MIN_POINTS_FOR_SPECTRUM:
        return None

    series = np.log(np.asarray(values, dtype=np.float64))
    series -= series.mean()
    if not np.any(series):
        return None

    spectrum = np.abs(np.fft.rfft(series)) ** 2
    frequencies = np.fft.rfftfreq(series.size, d=1.0)

    # Drop the DC term (zero frequency has no log) and any zero-power bins.
    mask = (frequencies > 0.0) & (spectrum > 0.0)
    if mask.sum() < _MIN_POINTS_FOR_SPECTRUM // 2:
        return None

    slope, _ = np.polyfit(np.log(frequencies[mask]), np.log(spectrum[mask]), 1)
    return float(slope)


def pink_noise_durations(
    count: int,
    mean: Seconds,
    spread: float,
    *,
    seed: int = 0,
    minimum: Seconds = Seconds(0.3),
    maximum: Seconds = Seconds(30.0),
) -> tuple[Seconds, ...]:
    """Generate a target shot-duration sequence with 1/f structure.

    Uses the Voss-McCartney algorithm, which sums several independent random
    sources updated at geometrically decreasing rates. The slow sources create
    long-range correlation while the fast ones supply local variation, producing
    an approximately 1/f spectrum cheaply and in order — no FFT, no need to know
    the sequence length in advance.

    Operates in the log domain so durations stay positive and the distribution
    comes out right-skewed, matching real shot-length distributions where a few
    long establishing shots sit above a mass of shorter ones.

    Args:
        count: How many durations to generate.
        mean: Target average shot length.
        spread: Target coefficient of variation, i.e. sigma over mu.
        seed: Seed for reproducibility. Two runs of the director on the same
            session with the same seed must produce the same edit.
        minimum: Floor, so no shot is too brief to read.
        maximum: Ceiling, so no shot outstays its welcome.

    Returns:
        The target durations, in order.

    Raises:
        ValueError: If ``count`` is negative or ``mean`` is not positive.
    """
    if count < 0:
        message = f"count must be non-negative, got {count}."
        raise ValueError(message)
    if mean <= 0.0:
        message = f"mean duration must be positive, got {mean}."
        raise ValueError(message)
    if count == 0:
        return ()

    generator = np.random.default_rng(seed)
    series = _voss_mccartney(count, generator)

    # Rescale to zero mean, unit standard deviation before mapping to log-space.
    deviation = series.std()
    normalised = (series - series.mean()) / deviation if deviation > _STD_EPSILON else series * 0.0

    # For a log-normal, sigma_log ~ sqrt(log(1 + cv^2)); mu_log offsets so the
    # arithmetic mean lands on the target rather than the median.
    sigma_log = math.sqrt(math.log1p(max(spread, 0.0) ** 2))
    mu_log = math.log(mean) - 0.5 * sigma_log**2

    durations = np.exp(mu_log + sigma_log * normalised)
    clipped = np.clip(durations, minimum, maximum)
    return tuple(Seconds(float(value)) for value in clipped)


def _voss_mccartney(count: int, generator: np.random.Generator) -> np.ndarray:
    """Generate a 1/f-correlated series.

    Sums ``_VOSS_OCTAVES`` random sources, where source ``k`` is refreshed every
    ``2**k`` steps. The rarely-refreshed sources hold their value across many
    steps, which is what produces long-range correlation; the frequently-refreshed
    ones vary every step.

    Args:
        count: Sequence length.
        generator: Random source.

    Returns:
        The generated series, shape ``(count,)``.
    """
    series = np.zeros(count, dtype=np.float64)
    for octave in range(_VOSS_OCTAVES):
        period = 2**octave
        # One value per period, repeated to fill the sequence.
        holds = generator.standard_normal((count + period - 1) // period)
        series += np.repeat(holds, period)[:count]
    return series
