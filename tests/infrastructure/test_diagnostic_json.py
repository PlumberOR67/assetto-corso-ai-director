"""Tests for writing a camera diagnosis.

The document is the deliverable — it is what travels from the machine that can
see the cameras to the one that can fix them — so these tests are about what a
reader can conclude from it, not about field names for their own sake.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from assetto_ai_director.domain.camera import RigKind
from assetto_ai_director.domain.diagnostics import (
    CameraDiagnostic,
    Check,
    CheckStatus,
    DiagnosticReport,
)
from assetto_ai_director.domain.geometry import Vec3
from assetto_ai_director.domain.shot import LineSide, ShotSize
from assetto_ai_director.domain.units import Seconds
from assetto_ai_director.infrastructure import ac_install
from assetto_ai_director.infrastructure.export.diagnostic_json import (
    diagnostic_to_dict,
    diagnostic_to_text,
    write_diagnostic,
)


@pytest.fixture(autouse=True)
def _no_machine_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the suite from depending on whether the developer owns the game.

    Args:
        monkeypatch: Pytest's patcher.
    """
    monkeypatch.delenv(ac_install.ENV_VAR, raising=False)
    monkeypatch.setattr(ac_install, "_DEFAULT_STEAM_DIRS", ())
    monkeypatch.setattr(ac_install, "_steam_path_from_registry", lambda: None)


def _camera(index: int, *checks: Check) -> CameraDiagnostic:
    """Build one camera diagnostic.

    Args:
        index: The shot index.
        *checks: Its checks.

    Returns:
        The diagnostic.
    """
    return CameraDiagnostic(
        shot_index=index,
        rig=RigKind.ROADSIDE,
        size=ShotSize.FULL,
        side=LineSide.LEFT,
        start=Seconds(float(index)),
        end=Seconds(float(index) + 2.0),
        subjects=(0,),
        position=Vec3(1.0, 2.0, 3.0),
        look=Vec3(0.0, 0.0, 1.0),
        fov=40.0,
        checks=checks,
    )


def _passed(name: str = "subject_in_frame") -> Check:
    """A measured, passing check."""
    return Check(name, CheckStatus.PASS, 1.0, "fraction", "fail below 0.60", "All good.")


def _failed(name: str = "subject_behind_camera") -> Check:
    """A measured, failing check."""
    return Check(name, CheckStatus.FAIL, 0.8, "fraction", "fail above 0.00", "Facing away.")


def _report(*cameras: CameraDiagnostic, blind: tuple[str, ...] = ()) -> DiagnosticReport:
    """Build a report.

    Args:
        *cameras: Its cameras.
        blind: Its blind spots.

    Returns:
        The report.
    """
    return DiagnosticReport(
        track_id="pk_akina",
        layout="akina_downhill",
        style_name="initial_d",
        cameras=cameras,
        blind_spots=blind,
        assumptions=("Car size is a guess.",),
    )


class TestSummary:
    def test_the_tally_covers_every_camera(self) -> None:
        document = diagnostic_to_dict(
            _report(_camera(0, _passed()), _camera(1, _failed()), _camera(2, _passed()))
        )
        summary = document["summary"]
        assert summary["cameras"] == 3
        assert summary["failing"] == 1
        assert summary["passing"] == 2
        assert summary["status"] == "fail"

    def test_an_unmeasured_camera_is_counted_apart_from_a_passing_one(self) -> None:
        document = diagnostic_to_dict(
            _report(_camera(0, Check.unknown("camera_on_surface", "No track geometry.")))
        )
        assert document["summary"]["unmeasured"] == 1
        assert document["summary"]["passing"] == 0
        assert document["summary"]["status"] == "unknown"

    def test_measured_status_reads_past_the_checks_that_could_not_run(self) -> None:
        # Two checks stay unmeasurable until track geometry can be read, so
        # without this a reader sees "0 passing" on every report and learns
        # nothing from it.
        document = diagnostic_to_dict(
            _report(_camera(0, _passed(), Check.unknown("camera_on_surface", "No geometry.")))
        )
        assert document["summary"]["status"] == "unknown"
        assert document["summary"]["measured_status"] == "pass"

    def test_measured_status_still_reports_a_real_failure(self) -> None:
        document = diagnostic_to_dict(
            _report(_camera(0, _failed(), Check.unknown("camera_on_surface", "No geometry.")))
        )
        assert document["summary"]["measured_status"] == "fail"


class TestFailuresAreEasyToFind:
    def test_failures_appear_before_the_per_camera_detail(self) -> None:
        document = diagnostic_to_dict(_report(_camera(0, _passed()), _camera(1, _failed())))
        keys = list(document)
        assert keys.index("failures") < keys.index("cameras")

    def test_only_failing_checks_are_listed_under_failures(self) -> None:
        document = diagnostic_to_dict(_report(_camera(0, _passed(), _failed())))
        assert len(document["failures"]) == 1
        listed = document["failures"][0]["failed"]
        assert [entry["check"] for entry in listed] == ["subject_behind_camera"]

    def test_a_clean_report_lists_no_failures(self) -> None:
        assert diagnostic_to_dict(_report(_camera(0, _passed())))["failures"] == []


class TestHonesty:
    def test_an_unmeasured_value_is_null_not_zero(self) -> None:
        # Rendering it as 0.0 would make "not measured" indistinguishable from
        # "measured, and it was zero" — the confusion UNKNOWN exists to prevent.
        document = diagnostic_to_dict(
            _report(_camera(0, Check.unknown("sight_line_clear", "No collision geometry.")))
        )
        assert document["cameras"][0]["checks"][0]["value"] is None

    def test_the_report_and_the_environment_both_contribute_blind_spots(self) -> None:
        document = diagnostic_to_dict(_report(_camera(0, _passed()), blind=("A gap.",)))
        assert "A gap." in document["blind_spots"]
        # No Assetto Corsa on this machine, so the environment adds its own.
        assert len(document["blind_spots"]) > 1

    def test_a_missing_game_is_stated_rather_than_left_to_inference(self) -> None:
        document = diagnostic_to_dict(_report(_camera(0, _passed())))
        assert document["environment"]["assetto_corsa_found"] is False
        assert document["environment"]["custom_shaders_patch"] == "not installed"

    def test_thresholds_travel_so_a_reader_can_disagree(self) -> None:
        document = diagnostic_to_dict(_report(_camera(0, _passed())))
        check = document["cameras"][0]["checks"][0]
        assert check["threshold"] == "fail below 0.60"
        assert check["unit"] == "fraction"


class TestEnvironmentEmbedding:
    def test_a_found_install_and_its_track_are_recorded(self, tmp_path: Path) -> None:
        root = tmp_path / "ac"
        layout = root / "content" / "tracks" / "pk_akina" / "akina_downhill" / "data"
        layout.mkdir(parents=True)
        (layout / "fast_lane.ai").write_bytes(b"\x00" * 16)

        document = diagnostic_to_dict(_report(_camera(0, _passed())), ac_root=root)
        environment = document["environment"]
        assert environment["assetto_corsa_found"] is True
        assert environment["track_files"]["present"]["fast_lane.ai"] == 16
        assert "map.ini" in environment["track_files"]["missing"]


class TestCompactForm:
    """The compact form exists so the report can be pasted at all."""

    def test_it_keeps_only_the_cameras_worth_looking_at(self) -> None:
        report = _report(
            _camera(0, _passed()),
            _camera(1, _failed()),
            _camera(2, _passed()),
        )
        document = diagnostic_to_dict(report, notable_only=True)
        assert [camera["shot"] for camera in document["cameras"]] == [1]
        assert document["cameras_included"] == "notable"

    def test_the_summary_still_describes_the_whole_cut(self) -> None:
        # Otherwise a short cameras array reads as a short cut, and the reader
        # concludes the director produced three shots instead of forty-seven.
        report = _report(_camera(0, _passed()), _camera(1, _failed()), _camera(2, _passed()))
        document = diagnostic_to_dict(report, notable_only=True)
        assert document["summary"]["cameras"] == 3
        assert len(document["cameras"]) == 1

    def test_failures_survive_the_trim(self) -> None:
        report = _report(_camera(0, _passed()), _camera(1, _failed()))
        document = diagnostic_to_dict(report, notable_only=True)
        assert len(document["failures"]) == 1

    def test_the_full_form_says_so(self) -> None:
        document = diagnostic_to_dict(_report(_camera(0, _passed())))
        assert document["cameras_included"] == "all"
        assert len(document["cameras"]) == 1

    def test_it_is_substantially_smaller(self) -> None:
        cameras = [_camera(index, _passed(), _passed("subject_framing")) for index in range(40)]
        report = _report(*cameras, _camera(99, _failed()))
        full = diagnostic_to_text(report)
        compact = diagnostic_to_text(report, notable_only=True)
        assert len(compact) < len(full) / 4


class TestCheckTally:
    def test_it_counts_every_check_across_every_camera(self) -> None:
        report = _report(_camera(0, _passed()), _camera(1, _passed()), _camera(2, _failed()))
        tally = diagnostic_to_dict(report)["check_tally"]
        assert tally["subject_in_frame"] == {"pass": 2, "warn": 0, "fail": 0, "unknown": 0}
        assert tally["subject_behind_camera"]["fail"] == 1

    def test_it_survives_the_compact_form(self) -> None:
        # The tally is the reason the compact form is usable at all: it carries
        # what the dropped cameras would have said.
        report = _report(
            *(_camera(index, _passed()) for index in range(10)), _camera(10, _failed())
        )
        tally = diagnostic_to_dict(report, notable_only=True)["check_tally"]
        assert tally["subject_in_frame"]["pass"] == 10


class TestWriting:
    def test_the_file_is_valid_json_and_ends_with_a_newline(self, tmp_path: Path) -> None:
        destination = tmp_path / "nested" / "diagnostic.json"
        write_diagnostic(_report(_camera(0, _passed())), destination)
        text = destination.read_text(encoding="utf-8")
        assert text.endswith("\n")
        assert json.loads(text)["session"]["track"] == "pk_akina"

    def test_parent_directories_are_created(self, tmp_path: Path) -> None:
        destination = tmp_path / "a" / "b" / "c" / "diagnostic.json"
        write_diagnostic(_report(_camera(0, _passed())), destination)
        assert destination.is_file()
