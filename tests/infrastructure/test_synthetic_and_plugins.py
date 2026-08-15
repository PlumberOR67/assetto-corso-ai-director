"""Tests for the synthetic provider and plugin discovery.

The synthetic provider's value is that its answers are *known*. Real telemetry
can only be checked for plausibility; here a detector can be asked whether it
found the specific drift that was scripted, at the time it was scripted for.
"""

from __future__ import annotations

import numpy as np
import pytest

from assetto_ai_director.application.director.styles import BUILTIN_STYLES
from assetto_ai_director.application.events.detect import detect_all
from assetto_ai_director.application.pipeline import direct
from assetto_ai_director.domain.events import EventKind
from assetto_ai_director.domain.telemetry import Channel
from assetto_ai_director.domain.units import Seconds
from assetto_ai_director.infrastructure.plugins import PluginGroup, discover
from assetto_ai_director.infrastructure.synthetic.source import (
    ScriptedDrift,
    ScriptedOvertake,
    SyntheticSource,
)


class TestSyntheticGeneration:
    def test_produces_the_requested_shape(self) -> None:
        session = SyntheticSource(seconds=30.0, cars=3).generate()
        assert len(session.signals) == 3
        assert session.time.shape[0] == session.signals[0].position.shape[0]
        assert float(session.duration) == pytest.approx(30.0, abs=0.1)

    def test_signals_are_finite_and_plausible(self) -> None:
        session = SyntheticSource(seconds=40.0).generate()
        for car in session.signals.values():
            assert np.isfinite(car.position).all()
            assert np.isfinite(car.heading).all()
            assert float(car.speed.min()) > 0.0
            assert float(car.speed.max()) * 3.6 < 250.0

    def test_is_reproducible(self) -> None:
        first = SyntheticSource(seconds=20.0, seed=7).generate()
        second = SyntheticSource(seconds=20.0, seed=7).generate()
        assert np.allclose(first.signals[0].sideslip, second.signals[0].sideslip)

    def test_different_seeds_differ(self) -> None:
        first = SyntheticSource(seconds=20.0, seed=1).generate()
        second = SyntheticSource(seconds=20.0, seed=2).generate()
        assert not np.allclose(first.signals[0].sideslip, second.signals[0].sideslip)

    def test_declares_only_the_channels_the_replay_reader_offers(self) -> None:
        # Being more generous than reality would let tests pass here and fail on
        # a real file.
        availability = SyntheticSource().availability()
        assert availability.has(Channel.POSITION)
        assert not availability.has(Channel.WHEEL_SLIP_ANGLE)
        assert not availability.has(Channel.THROTTLE)

    def test_descends_so_the_scenery_term_has_signal(self) -> None:
        session = SyntheticSource(seconds=30.0).generate()
        elevation = session.signals[0].position[:, 1]
        assert float(elevation[-1]) < float(elevation[0])


class TestGroundTruth:
    def test_scripted_drifts_are_detected_where_they_were_placed(self) -> None:
        # The point of the synthetic source: not "a drift was found somewhere"
        # but "the drift that was scripted was found at its time".
        drifts = (
            ScriptedDrift(Seconds(10.0), Seconds(4.0), 35.0, 0),
            ScriptedDrift(Seconds(30.0), Seconds(3.0), 26.0, 0),
        )
        source = SyntheticSource(seconds=45.0, cars=1, drifts=drifts, overtakes=())
        session = source.generate()
        result = detect_all(session.signals, session.time, float(session.sample_rate))

        found = [e for e in result.events if e.kind is EventKind.DRIFT_START]
        assert len(found) >= len(drifts)
        for scripted in drifts:
            assert any(abs(float(event.focus) - float(scripted.peak)) < 1.5 for event in found), (
                f"no drift detected near the scripted peak at {float(scripted.peak):.1f}s"
            )

    def test_a_session_without_drifts_yields_none(self) -> None:
        source = SyntheticSource(seconds=30.0, cars=1, drifts=(), overtakes=())
        session = source.generate()
        result = detect_all(session.signals, session.time, float(session.sample_rate))
        assert not [e for e in result.events if e.kind is EventKind.DRIFT_START]

    def test_scripted_overtake_is_detected(self) -> None:
        overtakes = (ScriptedOvertake(Seconds(25.0), overtaker=1, overtaken=0),)
        source = SyntheticSource(seconds=50.0, cars=2, drifts=(), overtakes=overtakes)
        session = source.generate()
        result = detect_all(session.signals, session.time, float(session.sample_rate))
        passes = [e for e in result.events if e.kind is EventKind.OVERTAKE_COMPLETE]
        assert passes

    def test_corners_are_found_on_the_synthetic_road(self) -> None:
        source = SyntheticSource(seconds=60.0, cars=1, drifts=(), overtakes=())
        session = source.generate()
        result = detect_all(session.signals, session.time, float(session.sample_rate))
        assert len([e for e in result.events if e.kind is EventKind.APEX]) >= 3


class TestPipelineOnSynthetic:
    def test_the_whole_director_runs_without_a_replay(self) -> None:
        # A fresh clone has no .acreplay fixtures; this is what proves the
        # pipeline works there rather than skipping silently.
        source = SyntheticSource(seconds=45.0, cars=2)
        session = source.generate()
        timeline = direct(
            session.signals,
            session.meta,
            BUILTIN_STYLES["initial_d"],
            session.time,
            float(session.sample_rate),
        )
        assert timeline.shot_count > 0
        assert timeline.tiles_without_gaps()
        assert timeline.notes


class TestPluginDiscovery:
    @pytest.mark.parametrize("group", list(PluginGroup))
    def test_every_group_is_queryable(self, group: PluginGroup) -> None:
        registry = discover(group)
        assert registry.group is group
        assert isinstance(registry.loaded, dict)

    def test_built_in_exporters_are_registered(self) -> None:
        registry = discover(PluginGroup.EXPORTERS)
        assert "json" in registry.names
        assert "cameras-ini" in registry.names
        assert registry.healthy, registry.failures

    def test_built_in_telemetry_provider_is_registered(self) -> None:
        registry = discover(PluginGroup.TELEMETRY_PROVIDERS)
        assert "synthetic" in registry.names

    def test_loaded_exporter_is_usable(self) -> None:
        sink = discover(PluginGroup.EXPORTERS).get("json")
        assert sink is not None
        assert sink().name == "json"

    def test_missing_plugin_returns_none(self) -> None:
        assert discover(PluginGroup.EXPORTERS).get("nonexistent") is None
