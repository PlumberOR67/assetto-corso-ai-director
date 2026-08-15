"""Round-trip tests for style profile TOML.

The property that matters is that **provenance survives the file**. If a measured
value came back as a guess, or worse a guess came back as measured, the whole
audit trail would be decorative. The escaping tests exist because this module
writes TOML itself rather than delegating to an encoder.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from assetto_ai_director.application.director.styles import (
    BUILTIN_STYLES,
    profile_from_measurement,
)
from assetto_ai_director.domain.camera import RigKind
from assetto_ai_director.domain.measurement import (
    CameraMotion,
    ShotMeasurement,
    StyleMeasurement,
)
from assetto_ai_director.domain.provenance import Provenance
from assetto_ai_director.domain.scoring import ScoreWeights
from assetto_ai_director.domain.units import Seconds
from assetto_ai_director.infrastructure.style.toml_io import read_profile, write_profile


def _measurement(shot_count: int = 64) -> StyleMeasurement:
    shots = tuple(
        ShotMeasurement(
            index=i,
            start=Seconds(float(i)),
            end=Seconds(float(i) + 1.0 + (i % 3)),
            motion=CameraMotion.STATIC if i % 3 == 0 else CameraMotion.AXIAL,
            motion_magnitude=0.4,
            motion_coherence=0.7,
        )
        for i in range(shot_count)
    )
    return StyleMeasurement(
        style_name="touge_reference",
        source_names=("a.mp4", "b.mp4"),
        shots=shots,
        total_duration=Seconds(float(shot_count) * 2.0),
        clip_count=2,
    )


class TestProfileFromMeasurement:
    def test_pace_is_marked_measured(self) -> None:
        profile = profile_from_measurement(_measurement())
        assert profile.rhythm.median_shot_length.provenance is Provenance.MEASURED
        assert profile.is_measured

    def test_measured_values_cite_the_corpus(self) -> None:
        profile = profile_from_measurement(_measurement())
        source = profile.rhythm.median_shot_length.source or ""
        assert "clips" in source
        assert "shots" in source

    def test_small_corpus_falls_back_to_the_cited_slope(self) -> None:
        # Too few shots to estimate a slope: the profile must fall back to the
        # published 1/f value and say so, not report sampling noise as measured.
        profile = profile_from_measurement(_measurement(shot_count=8))
        slope = profile.rhythm.spectral_slope
        assert slope.provenance is Provenance.VERIFIED
        assert slope.value == pytest.approx(-1.0)
        assert "too small" in (slope.note or "")

    def test_weights_are_not_claimed_as_measured(self) -> None:
        # Nothing in a frame says how a style values risk against flow.
        profile = profile_from_measurement(_measurement())
        assert profile.weights == ScoreWeights()

    def test_source_clips_are_recorded(self) -> None:
        assert profile_from_measurement(_measurement()).measured_from == ("a.mp4", "b.mp4")


class TestRoundTrip:
    def test_written_file_is_valid_toml(self, tmp_path: Path) -> None:
        destination = tmp_path / "profile.toml"
        write_profile(profile_from_measurement(_measurement()), destination)
        with destination.open("rb") as handle:
            assert tomllib.load(handle)["name"] == "touge_reference"

    def test_values_survive(self, tmp_path: Path) -> None:
        original = profile_from_measurement(_measurement())
        destination = tmp_path / "profile.toml"
        write_profile(original, destination)
        restored = read_profile(destination)
        assert restored.rhythm.median_shot_length.value == pytest.approx(
            original.rhythm.median_shot_length.value
        )
        assert restored.camera_mix.static_fraction.value == pytest.approx(
            original.camera_mix.static_fraction.value
        )

    def test_provenance_survives(self, tmp_path: Path) -> None:
        original = profile_from_measurement(_measurement())
        destination = tmp_path / "profile.toml"
        write_profile(original, destination)
        restored = read_profile(destination)
        assert restored.rhythm.median_shot_length.provenance is Provenance.MEASURED
        assert restored.is_measured

    def test_guesses_stay_guesses(self, tmp_path: Path) -> None:
        original = BUILTIN_STYLES["initial_d"]
        destination = tmp_path / "initial_d.toml"
        write_profile(original, destination)
        restored = read_profile(destination)
        assert restored.rhythm.median_shot_length.provenance is Provenance.GUESS
        assert not restored.is_measured

    @pytest.mark.parametrize("name", sorted(BUILTIN_STYLES))
    def test_every_builtin_round_trips(self, tmp_path: Path, name: str) -> None:
        destination = tmp_path / f"{name}.toml"
        write_profile(BUILTIN_STYLES[name], destination)
        restored = read_profile(destination)
        assert restored.name == name
        assert restored.weights == BUILTIN_STYLES[name].weights

    def test_allowed_rigs_survive(self, tmp_path: Path) -> None:
        original = BUILTIN_STYLES["initial_d"]
        destination = tmp_path / "p.toml"
        write_profile(original, destination)
        restored = read_profile(destination)
        assert restored.camera_mix.allowed_rigs == original.camera_mix.allowed_rigs
        assert not restored.camera_mix.permits(RigKind.DRONE)


class TestEncoding:
    """This module writes TOML by hand, so escaping is its own responsibility."""

    @pytest.mark.parametrize(
        "text",
        [
            'quotes " inside',
            "backslash \\ inside",
            'both \\ and " together',
            "tab\there",
            "newline\nhere",
            "unicode: ドリフト и кириллица",
            "control\x01char",
        ],
    )
    def test_awkward_descriptions_round_trip(self, tmp_path: Path, text: str) -> None:
        from dataclasses import replace  # noqa: PLC0415

        original = replace(BUILTIN_STYLES["initial_d"], description=text)
        destination = tmp_path / "p.toml"
        write_profile(original, destination)
        assert read_profile(destination).description == text

    def test_floats_stay_floats(self, tmp_path: Path) -> None:
        # A whole-number float written as "3" would parse back as an int and
        # change the field's type.
        from dataclasses import replace  # noqa: PLC0415

        from assetto_ai_director.domain.provenance import Tunable  # noqa: PLC0415

        style = BUILTIN_STYLES["initial_d"]
        rhythm = replace(style.rhythm, median_shot_length=Tunable(3.0, Provenance.GUESS))
        destination = tmp_path / "p.toml"
        write_profile(replace(style, rhythm=rhythm), destination)
        with destination.open("rb") as handle:
            value = tomllib.load(handle)["rhythm"]["median_shot_length"]["value"]
        assert isinstance(value, float)


class TestReadingFailures:
    def test_unknown_schema_is_refused(self, tmp_path: Path) -> None:
        destination = tmp_path / "p.toml"
        write_profile(BUILTIN_STYLES["initial_d"], destination)
        destination.write_text(
            destination.read_text(encoding="utf-8").replace(
                "schema_version = 1", "schema_version = 99"
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Unsupported style profile schema"):
            read_profile(destination)

    def test_bare_number_is_read_as_a_guess(self, tmp_path: Path) -> None:
        # Hand-editing a value to try it must not launder it into something
        # more authoritative than a guess.
        destination = tmp_path / "p.toml"
        write_profile(BUILTIN_STYLES["initial_d"], destination)
        text = destination.read_text(encoding="utf-8")
        start = text.index("median_shot_length = ")
        end = text.index("\n", start)
        patched = text[:start] + "median_shot_length = 4.2" + text[end:]
        destination.write_text(patched, encoding="utf-8")
        restored = read_profile(destination)
        assert restored.rhythm.median_shot_length.value == pytest.approx(4.2)
        assert restored.rhythm.median_shot_length.provenance is Provenance.GUESS

    def test_missing_field_is_reported_by_name(self, tmp_path: Path) -> None:
        destination = tmp_path / "p.toml"
        write_profile(BUILTIN_STYLES["initial_d"], destination)
        text = destination.read_text(encoding="utf-8")
        start = text.index("spread = ")
        end = text.index("\n", start)
        destination.write_text(text[:start] + text[end + 1 :], encoding="utf-8")
        with pytest.raises(ValueError, match="spread"):
            read_profile(destination)
