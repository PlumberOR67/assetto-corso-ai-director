"""Tests for the MCP server.

Two things are worth guarding here. That the tools work at all — they are the
only surface a model sees, so a broken one fails silently from the user's point
of view. And that every tool returning an analysis also returns its **notes**: a
model reading these results has no other way to learn what the analysis was blind
to, and a confident-looking result with the caveats stripped is worse than no
result.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from itertools import pairwise
from pathlib import Path

import pytest

from assetto_ai_director.infrastructure.synthetic.source import SyntheticSource
from assetto_ai_director.mcp import server
from assetto_ai_director.mcp.session import Session, SessionStore


@pytest.fixture
def synthetic_handle() -> Iterator[str]:
    """Register a synthetic session directly, bypassing replay decoding."""
    source = SyntheticSource(seconds=40.0, cars=2, seed=5)
    generated = source.generate()
    session = Session(
        handle="testhandle",
        path=Path("synthetic"),
        meta=generated.meta,
        signals=generated.signals,
        time=generated.time,
        sample_rate=float(generated.sample_rate),
    )
    server.store._sessions[session.handle] = session
    yield session.handle
    server.store.close(session.handle)


class TestSdkContract:
    def test_the_server_uses_the_two_point_oh_entry_point(self) -> None:
        # The 2.0 SDK renamed FastMCP to MCPServer with no compatibility alias,
        # so anything written against the old name silently fails to import.
        from mcp.server import MCPServer  # noqa: PLC0415

        assert isinstance(server.build_server(), MCPServer)

    def test_every_promised_tool_exists(self) -> None:
        expected = {
            "open_session",
            "load_replay",
            "load_telemetry",
            "find_drifts",
            "find_transitions",
            "find_best_moments",
            "generate_shotlist",
            "generate_style",
            "preview",
            "export_json",
        }
        for name in expected:
            assert callable(getattr(server, name)), f"{name} is missing"


class TestSessionStore:
    def test_unknown_handle_lists_the_open_ones(self) -> None:
        # A client that lost a handle otherwise has no way to recover short of
        # reopening everything.
        store = SessionStore()
        with pytest.raises(KeyError, match="Open sessions"):
            store.get("nope")

    def test_closing_forgets_the_session(self, synthetic_handle: str) -> None:
        assert synthetic_handle in server.store.handles
        server.store.close(synthetic_handle)
        assert synthetic_handle not in server.store.handles

    def test_detection_is_computed_once(self, synthetic_handle: str) -> None:
        session = server.store.get(synthetic_handle)
        assert session.detection is session.detection

    def test_scores_are_computed_once(self, synthetic_handle: str) -> None:
        session = server.store.get(synthetic_handle)
        assert session.scores is session.scores


class TestAnalysisTools:
    def test_find_drifts_returns_structured_data(self, synthetic_handle: str) -> None:
        result = server.find_drifts(synthetic_handle)
        assert isinstance(result["count"], int)
        assert isinstance(result["drifts"], list)
        if result["drifts"]:
            entry = result["drifts"][0]
            assert isinstance(entry["at_s"], float)
            assert "detail" in entry

    def test_find_drifts_discloses_its_blind_spots(self, synthetic_handle: str) -> None:
        notes = server.find_drifts(synthetic_handle)["notes"]
        assert any("Not detected" in note for note in notes)

    def test_min_angle_filters(self, synthetic_handle: str) -> None:
        loose = server.find_drifts(synthetic_handle, min_angle_deg=5.0)["count"]
        strict = server.find_drifts(synthetic_handle, min_angle_deg=90.0)["count"]
        assert strict <= loose

    def test_find_transitions_returns_a_list(self, synthetic_handle: str) -> None:
        assert isinstance(server.find_transitions(synthetic_handle)["transitions"], list)

    def test_load_telemetry_summarises_a_span(self, synthetic_handle: str) -> None:
        result = server.load_telemetry(synthetic_handle, car_id=0, start_s=5.0, end_s=20.0)
        assert result["car_id"] == 0
        assert result["speed_kmh"]["median"] > 0.0
        assert "position" in result["available_channels"]
        assert result["notes"]

    def test_load_telemetry_refuses_a_parked_car(self, synthetic_handle: str) -> None:
        with pytest.raises(ValueError, match="did not move"):
            server.load_telemetry(synthetic_handle, car_id=99)

    def test_best_moments_are_ranked_and_spaced(self, synthetic_handle: str) -> None:
        result = server.find_best_moments(synthetic_handle, limit=5)
        moments = result["moments"]
        assert len(moments) <= 5
        totals = [m["total"] for m in moments]
        assert totals == sorted(totals, reverse=True)
        times = sorted(m["at_s"] for m in moments)
        for earlier, later in pairwise(times):
            assert later - earlier > 1.0


class TestDirectingTools:
    def test_shotlist_tiles_the_horizon(self, synthetic_handle: str) -> None:
        result = server.generate_shotlist(synthetic_handle, style="initial_d")
        assert result["shot_count"] > 0
        assert result["tiles_without_gaps"]
        assert result["notes"]

    def test_shotlist_records_why_shots_exist(self, synthetic_handle: str) -> None:
        shots = server.generate_shotlist(synthetic_handle)["shots"]
        assert any(shot["because"] for shot in shots)

    def test_the_timeline_is_cached_per_style(self, synthetic_handle: str) -> None:
        server.generate_shotlist(synthetic_handle, style="initial_d")
        session = server.store.get(synthetic_handle)
        assert "initial_d" in session.timelines

    def test_preview_returns_camera_geometry(self, synthetic_handle: str) -> None:
        result = server.preview(synthetic_handle, shot_index=0)
        assert result["keyframe_count"] > 0
        assert len(result["keyframes"][0]["position"]) == 3

    def test_preview_rejects_an_out_of_range_shot(self, synthetic_handle: str) -> None:
        with pytest.raises(ValueError, match="out of range"):
            server.preview(synthetic_handle, shot_index=99999)

    def test_diagnose_reports_every_camera_and_what_it_could_not_see(
        self, synthetic_handle: str
    ) -> None:
        result = server.diagnose_cameras_tool(synthetic_handle)
        assert result["summary"]["cameras"] > 0
        assert result["check_tally"]
        # The blind spots are the point: an assistant reading this must not
        # conclude the cameras are fine from an absence of failures.
        assert result["blind_spots"]
        assert result["assumptions"]

    def test_diagnose_defaults_to_the_compact_form(self, synthetic_handle: str) -> None:
        compact = server.diagnose_cameras_tool(synthetic_handle)
        full = server.diagnose_cameras_tool(synthetic_handle, notable_only=False)
        assert compact["cameras_included"] == "notable"
        assert full["cameras_included"] == "all"
        assert len(compact["cameras"]) <= len(full["cameras"])
        # The counts describe the whole cut either way, so a trimmed list is
        # never mistaken for a short edit.
        assert compact["summary"] == full["summary"]

    def test_export_writes_a_readable_file(self, synthetic_handle: str, tmp_path: Path) -> None:
        destination = tmp_path / "t.json"
        result = server.export_json(synthetic_handle, str(destination))
        assert destination.is_file()
        assert result["shot_count"] > 0
        document = json.loads(destination.read_text(encoding="utf-8"))
        assert document["shots"]


class TestStyleTool:
    def test_describes_a_builtin_style(self) -> None:
        result = server.generate_style("initial_d")
        assert result["name"] == "initial_d"
        assert result["weights"]["drift"] > 0.0
        assert "roadside" in result["allowed_rigs"]

    def test_reports_provenance_rather_than_bare_numbers(self) -> None:
        # A model reading a style needs to know which figures are measured and
        # which are taste.
        rhythm = server.generate_style("initial_d")["rhythm"]
        assert rhythm["median_shot_length_s"]["provenance"] == "guess"
        assert not server.generate_style("initial_d")["is_measured"]

    def test_discloses_that_weights_are_judgement(self) -> None:
        assert any("judgement" in note for note in server.generate_style("initial_d")["notes"])

    def test_initial_d_forbids_drones(self) -> None:
        assert "drone" not in server.generate_style("initial_d")["allowed_rigs"]

    def test_unknown_style_lists_the_available_ones(self) -> None:
        with pytest.raises(ValueError, match="Available"):
            server.generate_style("nonexistent")


REPLAY_DIR = Path(__file__).resolve().parents[2] / "replays"
FIXTURES = sorted(REPLAY_DIR.glob("*.acreplay")) if REPLAY_DIR.is_dir() else []


@pytest.mark.fixtures
@pytest.mark.skipif(not FIXTURES, reason="no .acreplay fixtures in replays/")
class TestOnRealReplay:
    def test_load_replay_reads_metadata(self) -> None:
        result = server.load_replay(str(FIXTURES[0]))
        assert result["layout_verified"]
        assert result["cars"] > 0

    def test_open_session_reports_skipped_cars(self) -> None:
        # The real fixture is a two-player session with fifteen empty slots; a
        # client should be told they were skipped, not left to wonder.
        result = server.open_session(str(FIXTURES[0]), seconds=30.0)
        try:
            assert result["cars_analysed"]
            assert result["cars_skipped"]
            assert result["recording_quality"]
        finally:
            server.store.close(result["handle"])
