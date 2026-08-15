"""Tests for the provenance mechanism.

These tests protect the project's core rule. If provenance can be bypassed, the
codebase loses the ability to distinguish a cited constant from a guess, and the
whole honesty argument becomes decoration.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from assetto_ai_director.domain.provenance import Provenance, ProvenanceReport, Tunable


class TestProvenance:
    def test_guess_is_not_trustworthy(self) -> None:
        assert not Provenance.GUESS.is_trustworthy

    @pytest.mark.parametrize(
        "provenance", [Provenance.VERIFIED, Provenance.MEASURED, Provenance.DERIVED]
    )
    def test_evidence_backed_provenances_are_trustworthy(self, provenance: Provenance) -> None:
        assert provenance.is_trustworthy


class TestTunable:
    def test_guess_needs_no_source(self) -> None:
        tunable = Tunable(2.4, Provenance.GUESS)
        assert tunable.value == 2.4
        assert not tunable.is_trustworthy

    def test_verified_without_source_is_refused(self) -> None:
        # An unsourced "verified" claim is worse than an honest guess, because
        # it invites trust it has not earned.
        with pytest.raises(ValueError, match="must cite its source"):
            Tunable(2.4, Provenance.VERIFIED)

    def test_verified_with_source_is_accepted(self) -> None:
        tunable = Tunable(50.0, Provenance.VERIFIED, source="MF Ghost drone cap, 180 km/h")
        assert tunable.is_trustworthy

    def test_measured_upgrade_replaces_value_and_provenance(self) -> None:
        guess = Tunable(2.4, Provenance.GUESS, note="needs measurement")
        measured = guess.measured(3.1, run_id="initial_d_2026_08_06")
        assert measured.value == 3.1
        assert measured.provenance is Provenance.MEASURED
        assert measured.is_trustworthy
        assert "initial_d_2026_08_06" in (measured.source or "")

    def test_str_shows_provenance(self) -> None:
        assert str(Tunable(2.4, Provenance.GUESS)) == "2.4 [guess]"

    def test_is_generic_over_value_type(self) -> None:
        integer: Tunable[int] = Tunable(5, Provenance.GUESS)
        flag: Tunable[bool] = Tunable(True, Provenance.GUESS)
        assert integer.value == 5
        assert flag.value is True


@dataclass(frozen=True)
class _Nested:
    inner_threshold: Tunable[float]


@dataclass(frozen=True)
class _Profile:
    shot_length: Tunable[float]
    drone_cap: Tunable[float]
    nested: _Nested
    name: str = "example"


class TestProvenanceReport:
    def _profile(self) -> _Profile:
        return _Profile(
            shot_length=Tunable(2.4, Provenance.GUESS),
            drone_cap=Tunable(50.0, Provenance.VERIFIED, source="MF Ghost canon"),
            nested=_Nested(inner_threshold=Tunable(0.1, Provenance.GUESS)),
        )

    def test_finds_nested_tunables(self) -> None:
        report = ProvenanceReport.of(self._profile())
        names = {name for name, _ in report.entries}
        assert names == {"shot_length", "drone_cap", "nested.inner_threshold"}

    def test_counts_by_provenance(self) -> None:
        report = ProvenanceReport.of(self._profile())
        assert report.count(Provenance.GUESS) == 2
        assert report.count(Provenance.VERIFIED) == 1

    def test_guesses_lists_the_measurement_backlog(self) -> None:
        report = ProvenanceReport.of(self._profile())
        assert {name for name, _ in report.guesses} == {"shot_length", "nested.inner_threshold"}

    def test_trust_ratio(self) -> None:
        report = ProvenanceReport.of(self._profile())
        assert report.trust_ratio == pytest.approx(1 / 3)

    def test_empty_container_is_fully_trusted(self) -> None:
        assert ProvenanceReport.of(object()).trust_ratio == 1.0
