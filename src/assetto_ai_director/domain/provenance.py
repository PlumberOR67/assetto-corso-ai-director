"""Provenance tracking: where every number in this system came from.

This module is the project's honesty rule made mechanical.

An AI director is a machine for turning judgement into numbers. The temptation
is to write ``shot_duration = 2.4`` and move on. But a reader cannot tell whether
``2.4`` came from a peer-reviewed shot-length study, from arithmetic on a
verified constant, from measuring real footage, or from taste.

So every tunable carries its origin. :func:`Tunable.guesses` and the
``aidirector provenance-report`` command turn that into an auditable inventory,
and CI fails if a tunable ships without provenance.

The practical consequence: when someone asks "why is an Initial D shot 2.4
seconds?", the answer is in the type, not in a code archaeology session.

Example:
    >>> drone_speed_cap = Tunable(
    ...     50.0,
    ...     Provenance.VERIFIED,
    ...     source="MF Ghost: drones are canonically capped at 180 km/h",
    ... )
    >>> drone_speed_cap.value
    50.0
    >>> drone_speed_cap.is_trustworthy
    True
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["Provenance", "Tunable"]


class Provenance(StrEnum):
    """Where a numeric value came from.

    Ordered loosely from most to least defensible.

    Attributes:
        VERIFIED: Traceable to a cited primary source — a paper, a standard, a
            vendor SDK, or canon from the reference work itself. ``source`` must
            name it.
        MEASURED: Produced by this project's own measurement harness against real
            data. ``source`` records the run that produced it, so it is
            reproducible.
        DERIVED: Computed from a verified or measured value. ``source`` states
            the derivation, so the arithmetic can be checked.
        GUESS: Engineering judgement with no source. Legitimate and unavoidable
            for a system that has to pick *something* — but it is labelled, it is
            counted, and it is the backlog for the measurement harness.
    """

    VERIFIED = "verified"
    MEASURED = "measured"
    DERIVED = "derived"
    GUESS = "guess"

    @property
    def is_trustworthy(self) -> bool:
        """Whether a value with this provenance rests on evidence.

        Returns:
            ``True`` for verified, measured and derived values; ``False`` for
            guesses.
        """
        return self is not Provenance.GUESS


@dataclass(frozen=True, slots=True)
class Tunable[T]:
    """A configurable value bundled with the reason it has that value.

    Generic over the value type so a tunable can hold a float threshold, an
    integer count, a boolean switch or a tuple range without losing type
    information.

    Attributes:
        value: The value itself.
        provenance: Where it came from.
        source: A citation, derivation or measurement run id. Required for
            everything except a :attr:`Provenance.GUESS`.
        note: Optional free text — typically what would be needed to upgrade a
            guess into a measurement.

    Raises:
        ValueError: If a non-guess value is constructed without a source. An
            unsourced "verified" claim is worse than an honest guess, because it
            invites trust it has not earned.
    """

    value: T
    provenance: Provenance
    source: str | None = None
    note: str | None = None

    def __post_init__(self) -> None:
        """Validate that evidence-backed values actually cite their evidence.

        Raises:
            ValueError: If provenance is not ``GUESS`` and no source was given.
        """
        if self.provenance.is_trustworthy and not self.source:
            message = (
                f"A {self.provenance.value!r} tunable must cite its source. "
                f"Value {self.value!r} was given none. If it is judgement, mark "
                f"it Provenance.GUESS honestly."
            )
            raise ValueError(message)

    @property
    def is_trustworthy(self) -> bool:
        """Whether this value rests on evidence rather than judgement.

        Returns:
            ``True`` unless the provenance is :attr:`Provenance.GUESS`.
        """
        return self.provenance.is_trustworthy

    def measured(self, value: T, run_id: str) -> Tunable[T]:
        """Return a copy replaced by a measured value.

        The intended lifecycle of a guess: the measurement harness analyses real
        reference footage and upgrades the tunable in place, turning taste into
        evidence.

        Args:
            value: The newly measured value.
            run_id: Identifier of the measurement run, for reproducibility.

        Returns:
            A new tunable carrying the measured value and provenance.
        """
        return Tunable(value, Provenance.MEASURED, source=f"measured: {run_id}")

    def __str__(self) -> str:
        """Render the value with a provenance marker.

        Returns:
            A short human-readable form such as ``2.4 [guess]``.
        """
        return f"{self.value} [{self.provenance.value}]"


def iter_tunables(container: object) -> Iterator[tuple[str, Tunable[object]]]:
    """Walk an object's fields and yield every tunable it holds.

    Powers ``aidirector provenance-report``. Reflection is used deliberately here
    rather than a registry: a registry can be forgotten when a field is added,
    whereas reflection cannot, and a guess that escapes the inventory defeats the
    purpose of having one.

    Args:
        container: Any dataclass instance, typically a style profile.

    Yields:
        ``(field_name, tunable)`` for each :class:`Tunable` field found. Nested
        dataclasses are walked recursively with dotted names.
    """
    fields = getattr(container, "__dataclass_fields__", None)
    if fields is None:
        return
    for name in fields:
        value = getattr(container, name, None)
        if isinstance(value, Tunable):
            yield name, value
        elif hasattr(value, "__dataclass_fields__"):
            for inner_name, inner in iter_tunables(value):
                yield f"{name}.{inner_name}", inner


@dataclass(frozen=True, slots=True)
class ProvenanceReport:
    """A tally of where a configuration's numbers came from.

    Attributes:
        entries: Every tunable found, as ``(dotted_name, tunable)``.
    """

    entries: tuple[tuple[str, Tunable[object]], ...] = field(default_factory=tuple)

    @classmethod
    def of(cls, container: object) -> ProvenanceReport:
        """Build a report by walking a configuration object.

        Args:
            container: A dataclass instance, typically a style profile.

        Returns:
            The report for that object.
        """
        return cls(tuple(iter_tunables(container)))

    def count(self, provenance: Provenance) -> int:
        """Count entries with a given provenance.

        Args:
            provenance: The provenance to tally.

        Returns:
            How many tunables carry it.
        """
        return sum(1 for _, tunable in self.entries if tunable.provenance is provenance)

    @property
    def guesses(self) -> tuple[tuple[str, Tunable[object]], ...]:
        """Every unsourced tunable, i.e. the measurement backlog.

        Returns:
            The entries whose provenance is :attr:`Provenance.GUESS`.
        """
        return tuple(
            (name, tunable)
            for name, tunable in self.entries
            if tunable.provenance is Provenance.GUESS
        )

    @property
    def trust_ratio(self) -> float:
        """The fraction of tunables that rest on evidence.

        A blunt single number for tracking whether the project is getting more
        honest over time.

        Returns:
            Evidence-backed tunables divided by the total, or ``1.0`` when there
            are no tunables at all.
        """
        if not self.entries:
            return 1.0
        trusted = sum(1 for _, tunable in self.entries if tunable.is_trustworthy)
        return trusted / len(self.entries)
