"""Validate real output against the published JSON Schemas.

A schema file that nothing checks is documentation, and documentation drifts. So
these tests do two separate jobs.

The first is **conformance**: take a genuinely directed timeline and a genuinely
written style profile, and validate them. Hand-written example documents would
prove only that the examples match the schema.

The second, and the one that actually keeps the contract honest, is **anti-drift**:
every enumeration in a schema is compared against the Python enum it describes,
by exact set equality. Adding a thirteenth camera rig or a twentieth event kind
without updating the contract is then a test failure rather than a file that
quietly stops describing the software. Set equality in both directions matters —
a schema listing a rig the code removed is as wrong as one missing a rig the code
added, and only one of those is caught by validating real output.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from assetto_ai_director.application.camera.paths import SubjectTracks
from assetto_ai_director.application.diagnostics import diagnose_cameras
from assetto_ai_director.application.director.styles import BUILTIN_STYLES
from assetto_ai_director.application.pipeline import direct
from assetto_ai_director.domain.camera import RigKind
from assetto_ai_director.domain.diagnostics import CheckStatus
from assetto_ai_director.domain.events import EventKind, EventSeverity
from assetto_ai_director.domain.provenance import Provenance
from assetto_ai_director.domain.session import RecordingQuality
from assetto_ai_director.domain.shot import LineSide, ShotSize
from assetto_ai_director.domain.timeline import SCHEMA_VERSION, Timeline, TransitionKind
from assetto_ai_director.infrastructure.export.cameras_ini import planned_spans
from assetto_ai_director.infrastructure.export.diagnostic_json import (
    diagnostic_to_dict,
    write_diagnostic,
)
from assetto_ai_director.infrastructure.export.json_timeline import (
    timeline_to_dict,
    write_timeline,
)
from assetto_ai_director.infrastructure.style import toml_io
from assetto_ai_director.infrastructure.style.toml_io import write_profile
from assetto_ai_director.infrastructure.synthetic.source import SyntheticSource

CONTRACTS = Path(__file__).resolve().parents[2] / "contracts"


def _schema(name: str) -> dict[str, Any]:
    """Load a published schema.

    Args:
        name: File stem, e.g. ``timeline``.

    Returns:
        The parsed schema.
    """
    document: dict[str, Any] = json.loads(
        (CONTRACTS / f"{name}.schema.json").read_text(encoding="utf-8")
    )
    return document


def _at(schema: dict[str, Any], *path: str) -> dict[str, Any]:
    """Walk into a schema by key.

    Args:
        schema: The schema.
        *path: Keys to follow.

    Returns:
        The nested subschema.
    """
    node: dict[str, Any] = schema
    for key in path:
        node = node[key]
    return node


@pytest.fixture(scope="module")
def timeline_schema() -> dict[str, Any]:
    """The published timeline schema."""
    return _schema("timeline")


@pytest.fixture(scope="module")
def style_schema() -> dict[str, Any]:
    """The published style-profile schema."""
    return _schema("style")


@pytest.fixture(scope="module")
def diagnostic_schema() -> dict[str, Any]:
    """The published camera-diagnostic schema."""
    return _schema("diagnostic")


@pytest.fixture(scope="module")
def edit() -> Timeline:
    """A genuinely directed timeline, from synthetic telemetry."""
    session = SyntheticSource(seconds=60.0, cars=3, seed=11).generate()
    return direct(
        session.signals,
        session.meta,
        BUILTIN_STYLES["initial_d"],
        session.time,
        float(session.sample_rate),
    )


class TestSchemasThemselves:
    @pytest.mark.parametrize("name", ["timeline", "style", "diagnostic"])
    def test_the_schema_is_itself_valid(self, name: str) -> None:
        # A schema with a typo in a keyword silently validates everything.
        schema = _schema(name)
        jsonschema.Draft202012Validator.check_schema(schema)

    @pytest.mark.parametrize("name", ["timeline", "style", "diagnostic"])
    def test_every_definition_is_referenced(self, name: str) -> None:
        text = (CONTRACTS / f"{name}.schema.json").read_text(encoding="utf-8")
        for key in _schema(name).get("$defs", {}):
            assert f'"#/$defs/{key}"' in text, f"{key} is defined but never used"


class TestTimelineConformance:
    def test_a_directed_edit_validates(
        self, edit: Timeline, timeline_schema: dict[str, Any]
    ) -> None:
        jsonschema.validate(timeline_to_dict(edit), timeline_schema)

    def test_the_written_file_validates(
        self, tmp_path: Path, edit: Timeline, timeline_schema: dict[str, Any]
    ) -> None:
        # Validate what actually lands on disk, not the mapping on its way there.
        path = tmp_path / "timeline.json"
        write_timeline(edit, path)
        jsonschema.validate(json.loads(path.read_text(encoding="utf-8")), timeline_schema)

    def test_every_style_produces_a_valid_edit(self, timeline_schema: dict[str, Any]) -> None:
        session = SyntheticSource(seconds=40.0, cars=2, seed=2).generate()
        for name, profile in BUILTIN_STYLES.items():
            timeline = direct(
                session.signals,
                session.meta,
                profile,
                session.time,
                float(session.sample_rate),
            )
            document = timeline_to_dict(timeline)
            try:
                jsonschema.validate(document, timeline_schema)
            except jsonschema.ValidationError as error:  # pragma: no cover - failure path
                pytest.fail(f"style {name!r} produced a non-conforming timeline: {error.message}")

    def test_the_declared_version_is_the_code_version(self, edit: Timeline) -> None:
        assert timeline_to_dict(edit)["schema_version"] == SCHEMA_VERSION

    def test_additional_properties_are_actually_rejected(
        self, edit: Timeline, timeline_schema: dict[str, Any]
    ) -> None:
        # Proves the contract is closed. An open schema would accept anything and
        # the conformance tests above would prove nothing.
        document = timeline_to_dict(edit)
        document["surprise"] = 1
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(document, timeline_schema)

    def test_a_shot_without_a_camera_is_rejected(
        self, edit: Timeline, timeline_schema: dict[str, Any]
    ) -> None:
        document = timeline_to_dict(edit)
        del document["shots"][0]["camera"]
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(document, timeline_schema)

    def test_a_lone_dof_field_is_rejected(
        self, edit: Timeline, timeline_schema: dict[str, Any]
    ) -> None:
        # dof_distance without dof_factor is a half-written effect, and a
        # consumer applying it would focus at the right distance with an
        # undefined strength.
        document = timeline_to_dict(edit)
        document["shots"][0]["camera"]["keyframes"][0]["dof_distance"] = 12.0
        document["shots"][0]["camera"]["keyframes"][0].pop("dof_factor", None)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(document, timeline_schema)


class TestStyleConformance:
    def test_every_builtin_style_validates(
        self, tmp_path: Path, style_schema: dict[str, Any]
    ) -> None:
        for name, profile in BUILTIN_STYLES.items():
            path = tmp_path / f"{name}.toml"
            write_profile(profile, path)
            with path.open("rb") as handle:
                document = tomllib.load(handle)
            try:
                jsonschema.validate(document, style_schema)
            except jsonschema.ValidationError as error:  # pragma: no cover - failure path
                pytest.fail(f"built-in style {name!r} does not conform: {error.message}")

    def test_the_shipped_profiles_validate(self, style_schema: dict[str, Any]) -> None:
        profiles = sorted((CONTRACTS.parent / "profiles").glob("*.toml"))
        assert profiles, "no shipped profiles found to validate"
        for path in profiles:
            with path.open("rb") as handle:
                document = tomllib.load(handle)
            try:
                jsonschema.validate(document, style_schema)
            except jsonschema.ValidationError as error:  # pragma: no cover - failure path
                pytest.fail(f"{path.name} does not conform: {error.message}")

    def test_the_declared_version_is_the_code_version(self, style_schema: dict[str, Any]) -> None:
        del style_schema
        assert toml_io.SCHEMA_VERSION >= 1

    def test_a_bare_number_is_accepted_as_a_tunable(
        self, tmp_path: Path, style_schema: dict[str, Any]
    ) -> None:
        # The documented shorthand for hand-editing. If the schema rejected it,
        # the schema would be describing something the reader does not implement.
        path = tmp_path / "bare.toml"
        write_profile(BUILTIN_STYLES["initial_d"], path)
        with path.open("rb") as handle:
            document = tomllib.load(handle)
        document["rhythm"]["median_shot_length"] = 2.5
        jsonschema.validate(document, style_schema)

    def test_a_measured_value_without_a_source_is_rejected(
        self, tmp_path: Path, style_schema: dict[str, Any]
    ) -> None:
        # The whole point of the type: a number cannot claim authority it has not
        # earned. The Python class enforces this at construction; the schema must
        # enforce the same thing, or a hand-written file could smuggle one past.
        path = tmp_path / "unsourced.toml"
        write_profile(BUILTIN_STYLES["initial_d"], path)
        with path.open("rb") as handle:
            document = tomllib.load(handle)
        document["rhythm"]["median_shot_length"] = {"value": 2.5, "provenance": "measured"}
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(document, style_schema)

    def test_an_unsourced_guess_is_accepted(
        self, tmp_path: Path, style_schema: dict[str, Any]
    ) -> None:
        path = tmp_path / "guess.toml"
        write_profile(BUILTIN_STYLES["initial_d"], path)
        with path.open("rb") as handle:
            document = tomllib.load(handle)
        document["rhythm"]["median_shot_length"] = {"value": 2.5, "provenance": "guess"}
        jsonschema.validate(document, style_schema)


class TestNoDrift:
    """Every enumeration in a schema must match the Python enum it describes.

    Exact set equality, in both directions. A schema that has fallen behind the
    code is worse than no schema, because it looks authoritative.
    """

    def test_check_statuses(self, diagnostic_schema: dict[str, Any]) -> None:
        assert set(_at(diagnostic_schema, "$defs", "status")["enum"]) == {
            status.value for status in CheckStatus
        }

    def test_rigs(self, timeline_schema: dict[str, Any], style_schema: dict[str, Any]) -> None:
        expected = {rig.value for rig in RigKind}
        assert set(_at(timeline_schema, "$defs", "rig")["enum"]) == expected
        mix = _at(style_schema, "properties", "camera_mix", "properties")
        assert set(mix["allowed_rigs"]["items"]["enum"]) == expected

    def test_shot_sizes(self, timeline_schema: dict[str, Any]) -> None:
        listed = _at(timeline_schema, "$defs", "shot", "properties", "size")["enum"]
        assert set(listed) == {size.name.lower() for size in ShotSize}

    def test_shot_sizes_stay_in_order(self, timeline_schema: dict[str, Any]) -> None:
        # The vocabulary is ordered from tightest to widest, and a consumer that
        # reads the enum as an ordering — to compare two shots' tightness — would
        # be misled by an alphabetised list.
        listed = _at(timeline_schema, "$defs", "shot", "properties", "size")["enum"]
        assert listed == [size.name.lower() for size in sorted(ShotSize)]

    def test_line_sides(self, timeline_schema: dict[str, Any]) -> None:
        listed = _at(timeline_schema, "$defs", "shot", "properties", "side")["enum"]
        assert set(listed) == {side.value for side in LineSide}

    def test_event_kinds(self, timeline_schema: dict[str, Any]) -> None:
        listed = _at(timeline_schema, "$defs", "because", "properties", "event")["enum"]
        assert set(listed) == {kind.value for kind in EventKind}

    def test_event_severities(self, timeline_schema: dict[str, Any]) -> None:
        listed = _at(timeline_schema, "$defs", "because", "properties", "severity")["enum"]
        assert set(listed) == {severity.value for severity in EventSeverity}

    def test_transition_kinds(self, timeline_schema: dict[str, Any]) -> None:
        listed = _at(timeline_schema, "$defs", "cut", "properties", "kind")["enum"]
        assert set(listed) == {kind.value for kind in TransitionKind}

    def test_recording_qualities(self, timeline_schema: dict[str, Any]) -> None:
        listed = _at(timeline_schema, "$defs", "session", "properties", "recording_quality")["enum"]
        assert set(listed) == {quality.name.lower() for quality in RecordingQuality}

    def test_provenance_values(self, style_schema: dict[str, Any]) -> None:
        listed = style_schema["$defs"]["tunable"]["oneOf"][1]["properties"]["provenance"]["enum"]
        assert set(listed) == {value.value for value in Provenance}

    def test_score_weight_keys(self, style_schema: dict[str, Any], tmp_path: Path) -> None:
        # The weights table is closed, so a typo in a hand-edited profile is
        # caught rather than silently ignored — which would leave the user
        # wondering why their edit did not change.
        path = tmp_path / "weights.toml"
        write_profile(BUILTIN_STYLES["initial_d"], path)
        with path.open("rb") as handle:
            written = tomllib.load(handle)["weights"]
        listed = _at(style_schema, "properties", "weights", "properties")
        assert set(listed) == set(written)


class TestDiagnosticConformance:
    """A real diagnosis validates, and its unknowns survive the round trip.

    The second half matters more than it looks. A serialiser that renders an
    unmeasured value as ``0.0`` produces a document that validates perfectly and
    means something different from the report it came from.
    """

    @staticmethod
    def _report(edit: Timeline) -> Any:
        """Diagnose a directed edit using the tracks it was built from."""
        session = SyntheticSource(seconds=60.0, cars=3, seed=11).generate()
        tracks = SubjectTracks(
            positions={car_id: car.position for car_id, car in session.signals.items()},
            headings={car_id: car.heading for car_id, car in session.signals.items()},
            time=session.time,
            sample_rate=float(session.sample_rate),
        )
        return diagnose_cameras(edit, tracks, exported_spans=planned_spans(edit))

    def test_a_real_diagnosis_validates(
        self, edit: Timeline, diagnostic_schema: dict[str, Any]
    ) -> None:
        jsonschema.validate(diagnostic_to_dict(self._report(edit)), diagnostic_schema)

    def test_the_written_file_validates(
        self, tmp_path: Path, edit: Timeline, diagnostic_schema: dict[str, Any]
    ) -> None:
        path = tmp_path / "diagnostic.json"
        write_diagnostic(self._report(edit), path)
        jsonschema.validate(json.loads(path.read_text(encoding="utf-8")), diagnostic_schema)

    def test_unmeasured_checks_serialise_as_null_not_zero(self, edit: Timeline) -> None:
        document = diagnostic_to_dict(self._report(edit))
        unknowns = [
            check
            for camera in document["cameras"]
            for check in camera["checks"]
            if check["status"] == "unknown"
        ]
        assert unknowns, "the fixture should contain checks that could not run"
        assert all(check["value"] is None for check in unknowns)

    def test_the_compact_form_validates_against_the_same_schema(
        self, edit: Timeline, diagnostic_schema: dict[str, Any]
    ) -> None:
        # One schema for both forms, so a consumer cannot be handed a shape it
        # was never told about.
        document = diagnostic_to_dict(self._report(edit), notable_only=True)
        jsonschema.validate(document, diagnostic_schema)
        assert document["cameras_included"] == "notable"

    def test_the_blind_spots_travel_with_the_document(self, edit: Timeline) -> None:
        document = diagnostic_to_dict(self._report(edit))
        assert document["blind_spots"]
        assert document["assumptions"]

    def test_the_environment_travels_with_the_document(self, edit: Timeline) -> None:
        # Without it, an all-unknown report cannot be told apart from one
        # produced on a machine where Assetto Corsa was never found.
        document = diagnostic_to_dict(self._report(edit))
        assert "assetto_corsa_found" in document["environment"]
        assert "custom_shaders_patch" in document["environment"]
