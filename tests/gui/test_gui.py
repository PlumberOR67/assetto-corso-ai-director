"""Tests for the desktop application.

Run headless via ``QT_QPA_PLATFORM=offscreen``, which is what lets them run in CI
at all. A GUI that is only tested by someone looking at it is a GUI that breaks
silently between releases.

What is worth asserting is not appearance but **behaviour a user depends on**:
that opening a session does not start twenty-four minutes of work, that the
progress display names what it is doing, that a stop is reported as a choice
rather than a fault, that a failed write does not cost the finished edit, and that
the notes panel actually carries the caveats — the GUI being the easiest place to
convey unearned confidence by accident.

Where an older test pinned a *widget name*, it has been re-pointed. Where it
pinned *behaviour*, it has been strengthened rather than moved:
``test_nothing_that_acts_is_enabled_before_a_session`` iterates a list of the
controls that start work or write files, so a newly added button is covered
automatically — strictly more than naming two buttons was.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from PySide6.QtGui import QGuiApplication, QPalette
from PySide6.QtWidgets import QApplication

from assetto_ai_director.application.director.styles import BUILTIN_STYLES
from assetto_ai_director.application.pipeline import analyse, direct
from assetto_ai_director.application.ports.progress import PIPELINE_STAGES, Stage
from assetto_ai_director.domain.timeline import Timeline
from assetto_ai_director.infrastructure.csvio.source import write_csv
from assetto_ai_director.infrastructure.loader import load_session
from assetto_ai_director.infrastructure.synthetic.source import SyntheticSession, SyntheticSource
from assetto_ai_director.presentation.gui.app import MainWindow
from assetto_ai_director.presentation.gui.artifacts import Artifact, ArtifactsPanel
from assetto_ai_director.presentation.gui.header import SessionHeader, format_clock
from assetto_ai_director.presentation.gui.output import ArtifactWriter, run_folder, safe_name
from assetto_ai_director.presentation.gui.project import Phase, Project
from assetto_ai_director.presentation.gui.stages import DONE, PENDING, RUNNING, StageList
from assetto_ai_director.presentation.gui.theme import COLOURS
from assetto_ai_director.presentation.gui.timeline_view import TimelineView
from assetto_ai_director.presentation.gui.trackmap import TrackMapView
from assetto_ai_director.presentation.gui.welcome import STEPS
from assetto_ai_director.presentation.gui.workers import (
    MIN_EMIT_INTERVAL_S,
    DirectorWorker,
    SignalProgress,
)

if TYPE_CHECKING:
    from assetto_ai_director.presentation.gui.settings import Settings

pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def session() -> SyntheticSession:
    """A synthetic session, so no replay file is needed."""
    return SyntheticSource(seconds=40.0, cars=2, seed=3).generate()


@pytest.fixture(scope="module")
def edit(session: SyntheticSession) -> Timeline:
    """A directed timeline from that session."""
    return direct(
        session.signals,
        session.meta,
        BUILTIN_STYLES["initial_d"],
        session.time,
        float(session.sample_rate),
    )


@pytest.fixture
def csv_session(tmp_path: Path, session: SyntheticSession) -> Path:
    """That session on disk as a CSV the window can open."""
    path = tmp_path / "session.csv"
    write_csv(path, session.signals, session.time)
    return path


@pytest.fixture
def window(app: QApplication, settings: Settings) -> MainWindow:
    """A window with private settings, past the welcome page."""
    del app
    settings.first_run_done = True
    return MainWindow(settings)


class TestTheme:
    def test_the_stylesheet_layer_is_applied(self, app: QApplication) -> None:
        """The style itself cannot be identified once a stylesheet is set.

        Setting a stylesheet wraps the active style in a ``QStyleSheetStyle``
        proxy whose ``name()`` is empty — the Fusion base is still underneath,
        but it is no longer reachable through the public API. So this asserts
        what is actually observable, and the palette tests below carry the real
        weight.
        """
        assert app.styleSheet()

    def test_palette_is_dark(self, app: QApplication) -> None:
        assert app.palette().window().color().lightness() < 80

    def test_disabled_text_stays_readable(self, app: QApplication) -> None:
        # Leaving the disabled group at its default is the most common way a
        # hand-rolled dark theme becomes grey-on-grey.
        disabled = app.palette().color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text)
        background = app.palette().window().color()
        assert abs(disabled.lightness() - background.lightness()) > 30

    def test_every_named_colour_is_valid(self) -> None:
        assert all(colour.isValid() for colour in COLOURS.values())

    def test_status_colours_differ_in_lightness(self) -> None:
        # Every use of these also carries a shape or a word, but they should not
        # rely on hue alone even so.
        lightness = {name: COLOURS[name].lightness() for name in ("ok", "warning", "error")}
        assert len(set(lightness.values())) == len(lightness)


class TestOpeningDoesNotRun:
    def test_opening_a_session_does_not_start_the_pipeline(
        self, window: MainWindow, csv_session: Path
    ) -> None:
        # The central new behaviour. Opening used to launch the whole analysis
        # straight from the file dialog, so a user could not look at a replay
        # before committing twenty-four minutes to it.
        window.open_session(csv_session)
        assert window.project.phase is Phase.INSPECTED
        assert window.project.session is None
        assert window.project.timeline is None
        window.close()

    def test_opening_shows_what_the_session_is(self, window: MainWindow, csv_session: Path) -> None:
        window.open_session(csv_session)
        assert csv_session.name in window.header.filename.text()
        assert window.header.detail.text()
        window.close()

    def test_opening_names_the_file_in_the_title(
        self, window: MainWindow, csv_session: Path
    ) -> None:
        # The filename used to appear only transiently in the status bar and was
        # gone the moment a run finished.
        window.open_session(csv_session)
        assert csv_session.name in window.windowTitle()
        window.close()

    def test_the_window_can_open_a_csv(self, window: MainWindow, csv_session: Path) -> None:
        # The CLI branched on the suffix for weeks; the GUI could not.
        window.open_session(csv_session)
        assert window.project.path == csv_session
        window.close()

    def test_a_bad_file_does_not_leave_a_half_open_project(
        self, window: MainWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "assetto_ai_director.presentation.gui.app.QMessageBox.warning",
            lambda *args, **kwargs: None,
        )
        broken = tmp_path / "broken.csv"
        broken.write_text("nonsense\n", encoding="utf-8")
        window.open_session(broken)
        assert window.project.phase is Phase.EMPTY
        window.close()


class TestControlsGateOnPhase:
    def test_nothing_that_acts_is_enabled_before_a_session(self, window: MainWindow) -> None:
        # Iterating the controls rather than naming two of them: a newly added
        # button that starts work or writes a file is covered automatically.
        actors = (window.run_panel.run_button, window.export_action)
        assert not any(control.isEnabled() for control in actors)
        window.close()

    def test_run_becomes_available_once_a_session_is_open(
        self, window: MainWindow, csv_session: Path
    ) -> None:
        window.open_session(csv_session)
        assert window.run_panel.run_button.isEnabled()
        window.close()

    def test_lists_every_builtin_style(self, window: MainWindow) -> None:
        box = window.run_panel.style_box
        listed = {box.itemText(index) for index in range(box.count())}
        assert listed == set(BUILTIN_STYLES)
        window.close()

    def test_each_style_says_what_it_does(self, window: MainWindow) -> None:
        # StyleProfile.description existed since the styles did and was never
        # shown, leaving five names with no way to tell them apart.
        for name in sorted(BUILTIN_STYLES):
            window.run_panel.style_box.setCurrentText(name)
            assert window.run_panel.style_description.text() == BUILTIN_STYLES[name].description
        window.close()

    def test_the_span_control_exists_and_defaults_to_part_of_the_session(
        self, window: MainWindow
    ) -> None:
        # DirectorWorker has always accepted `seconds` and the GUI never exposed
        # it, so the only way to try a style was to commit to the whole session.
        assert 0.0 < window.run_panel.scope_seconds <= 600.0
        window.close()

    def test_whole_session_means_no_limit(self, window: MainWindow) -> None:
        window.run_panel.whole_radio.setChecked(True)
        assert window.run_panel.scope_seconds == 0.0
        window.close()


class TestStaleness:
    def test_changing_style_marks_the_result_stale(
        self, window: MainWindow, csv_session: Path, edit: Timeline
    ) -> None:
        # Changing a dropdown must not leave an edit built with the previous
        # style on screen with no indication.
        window.open_session(csv_session)
        window.project = window.project.completed(
            (window.run_panel.style_name, window.run_panel.scope_seconds), edit
        )
        other = next(name for name in sorted(BUILTIN_STYLES) if name != window.run_panel.style_name)
        window.run_panel.style_box.setCurrentText(other)
        assert window.project.phase is Phase.STALE
        window.close()

    def test_a_stale_run_button_names_the_style_it_would_use(
        self, window: MainWindow, csv_session: Path, edit: Timeline
    ) -> None:
        window.open_session(csv_session)
        window.project = window.project.completed(
            (window.run_panel.style_name, window.run_panel.scope_seconds), edit
        )
        other = next(name for name in sorted(BUILTIN_STYLES) if name != window.run_panel.style_name)
        window.run_panel.style_box.setCurrentText(other)
        assert other in window.run_panel.run_button.text()
        window.close()

    def test_a_cached_result_is_shown_without_re_running(
        self, window: MainWindow, csv_session: Path, edit: Timeline
    ) -> None:
        window.open_session(csv_session)
        key = (window.run_panel.style_name, window.run_panel.scope_seconds)
        window.project = window.project.completed(key, edit)
        window._run()
        assert window.project.phase is Phase.READY
        assert window._thread is None
        window.close()


class TestProjectStateMachine:
    def test_a_fresh_project_is_empty(self) -> None:
        assert Project().phase is Phase.EMPTY

    def test_opening_clears_derived_state(self, edit: Timeline) -> None:
        project = Project().opened(Path("a.acreplay")).completed(("initial_d", 0.0), edit)
        reopened = project.opened(Path("b.acreplay"))
        assert reopened.timelines == {}
        assert reopened.analysis is None

    def test_a_scope_change_drops_the_analysis(
        self, session: SyntheticSession, edit: Timeline
    ) -> None:
        # Style keeps it; scope must not. Getting this backwards would direct a
        # three-minute analysis over a whole-session horizon.
        project = Project().opened(Path("a.acreplay"))
        project.analysis = analyse(session.signals, session.time, float(session.sample_rate))
        project = project.completed(("initial_d", 180.0), edit)
        assert project.invalidated_by_scope().analysis is None

    def test_cancelling_keeps_the_previous_result(self, edit: Timeline) -> None:
        # Pressing Stop is a choice, and it must not cost the user an edit they
        # already had.
        project = Project().opened(Path("a.acreplay")).completed(("initial_d", 0.0), edit)
        stopped = project.stopped("Cancelled during selecting shots.")
        assert stopped.timeline is edit
        assert stopped.phase is Phase.STALE

    def test_cancelling_with_no_result_returns_to_inspected(self) -> None:
        project = Project().opened(Path("a.acreplay"))
        assert project.stopped("Cancelled.").phase is Phase.INSPECTED

    def test_a_failure_is_distinguishable_from_a_stop(self, edit: Timeline) -> None:
        project = Project().opened(Path("a.acreplay")).completed(("initial_d", 0.0), edit)
        assert project.failed("boom").phase is Phase.FAILED
        assert project.stopped("Cancelled.").phase is not Phase.FAILED


class TestStageList:
    def test_every_stage_has_a_row_before_anything_runs(self) -> None:
        # A checklist that grows a row at a time is barely better than a spinner.
        listing = StageList()
        assert set(listing.rows) == {str(stage) for stage in PIPELINE_STAGES}

    def test_rows_start_pending(self) -> None:
        listing = StageList()
        assert all(row.glyph.text() == PENDING for row in listing.rows.values())

    def test_a_started_stage_shows_as_running(self) -> None:
        listing = StageList()
        listing.report(str(Stage.DETECT), 0, 4)
        assert listing.rows[str(Stage.DETECT)].glyph.text() == RUNNING

    def test_a_completed_stage_shows_as_done(self) -> None:
        listing = StageList()
        listing.report(str(Stage.DETECT), 0, 4)
        listing.report(str(Stage.DETECT), 4, 4)
        assert listing.rows[str(Stage.DETECT)].glyph.text() == DONE

    def test_starting_a_later_stage_closes_an_earlier_one(self) -> None:
        # The scheduler ticks in batches and can finish without its last update
        # landing on the total. Without this the row would spin forever.
        listing = StageList()
        listing.report(str(Stage.SCHEDULE), 0, 5000)
        listing.report(str(Stage.SCHEDULE), 4608, 5000)
        listing.report(str(Stage.CAMERA), 0, 30)
        assert listing.rows[str(Stage.SCHEDULE)].glyph.text() == DONE

    def test_a_counter_is_shown_while_running(self) -> None:
        listing = StageList()
        listing.report(str(Stage.CANDIDATES), 0, 100)
        listing.report(str(Stage.CANDIDATES), 40, 100)
        assert listing.rows[str(Stage.CANDIDATES)].counter.text() == "40/100"

    def test_elapsed_time_survives_the_stage_finishing(self) -> None:
        # So the user can see afterwards which stage ate the time.
        listing = StageList()
        listing.report(str(Stage.DETECT), 0, 2)
        listing.report(str(Stage.DETECT), 2, 2)
        assert listing.rows[str(Stage.DETECT)].elapsed >= 0.0

    def test_no_estimate_before_there_is_evidence_for_one(self) -> None:
        # A figure that jumps from four seconds to nine minutes teaches the user
        # to ignore it.
        listing = StageList()
        listing.report(str(Stage.SCHEDULE), 0, 10_000)
        listing.report(str(Stage.SCHEDULE), 2, 10_000)
        assert listing.remaining is None

    def test_an_unknown_stage_is_ignored(self) -> None:
        listing = StageList()
        listing.report("Something else", 0, 5)


class TestProgressCoalescing:
    def test_start_and_finish_always_emit(self) -> None:
        # The existing specification of the coalescer, kept verbatim from before
        # it existed: start and finish must never be dropped.
        reporter = SignalProgress()
        received: list[tuple[str, int, int]] = []
        reporter.changed.connect(lambda d, done, total: received.append((d, done, total)))
        reporter.start("Working", 4)
        reporter.advance()
        reporter.finish()
        assert received[0] == ("Working", 0, 4)
        assert received[-1] == ("Working", 4, 4)

    def test_a_burst_of_advances_is_collapsed(self) -> None:
        # The correctness constraint: the scheduler ticks hundreds of thousands
        # of times, and a queued cross-thread signal per tick would arrive faster
        # than the GUI thread can drain its queue.
        reporter = SignalProgress()
        received: list[int] = []
        reporter.changed.connect(lambda _d, done, _t: received.append(done))
        reporter.start("Working", 100_000)
        for _ in range(5000):
            reporter.advance()
        assert len(received) < 100

    def test_the_final_advance_still_lands(self) -> None:
        # A bar left at 4999/5000 forever is its own kind of lie.
        reporter = SignalProgress()
        received: list[int] = []
        reporter.changed.connect(lambda _d, done, _t: received.append(done))
        reporter.start("Working", 20)
        for _ in range(20):
            reporter.advance()
        assert received[-1] == 20

    def test_the_interval_is_not_so_long_that_the_bar_looks_frozen(self) -> None:
        assert 0.0 < MIN_EMIT_INTERVAL_S <= 0.1


class TestOutput:
    def test_the_destination_is_deterministic(self, tmp_path: Path) -> None:
        # "Where does it save" should not have a different answer each time.
        first = run_folder(tmp_path, Path("a/b/session.acreplay"), "initial_d")
        second = run_folder(tmp_path, Path("a/b/session.acreplay"), "initial_d")
        assert first == second
        assert first == tmp_path / "session" / "initial_d"

    def test_style_and_session_separate_the_folders(self, tmp_path: Path) -> None:
        one = run_folder(tmp_path, Path("s.acreplay"), "initial_d")
        two = run_folder(tmp_path, Path("s.acreplay"), "mf_ghost")
        assert one != two

    @pytest.mark.parametrize("raw", ["a/b", "a:b", 'a"b', "a|b", "  ", "."])
    def test_names_that_are_not_valid_folders_are_sanitised(self, raw: str) -> None:
        cleaned = safe_name(raw)
        assert not set(cleaned) & set('<>:"/\\|?*')
        assert cleaned

    def test_a_run_writes_the_timeline_and_the_camera_set(
        self, tmp_path: Path, edit: Timeline
    ) -> None:
        written = ArtifactWriter(tmp_path).write_all(edit)
        kinds = {artifact.kind for artifact in written}
        assert kinds == {"timeline", "cameras-ini"}
        assert all(artifact.error is None for artifact in written)
        assert all(artifact.path.is_file() for artifact in written)

    def test_a_lossy_export_is_written_with_its_limitations_attached(
        self, tmp_path: Path, edit: Timeline
    ) -> None:
        # The CLI refuses without --force because a terminal scrolls away. A
        # panel keeps the caveat beside the file permanently, so refusing would
        # only add a click before the same outcome.
        written = ArtifactWriter(tmp_path).write_all(edit)
        cameras = next(a for a in written if a.kind == "cameras-ini")
        assert cameras.path.is_file()
        assert cameras.limitations

    def test_a_write_failure_becomes_a_row_rather_than_an_exception(
        self, tmp_path: Path, edit: Timeline, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # write_timeline has no try/except and the old window called it bare, so
        # a full disk produced an unhandled exception in the Qt event loop —
        # losing the window while the finished edit sat in memory, perfectly good.
        def explode(*args: object, **kwargs: object) -> None:
            message = "disk full"
            raise OSError(message)

        monkeypatch.setattr(
            "assetto_ai_director.infrastructure.export.json_timeline.JsonTimelineSink.write",
            explode,
        )
        written = ArtifactWriter(tmp_path).write_all(edit)
        timeline_row = next(a for a in written if a.kind == "timeline")
        assert timeline_row.error is not None
        assert "disk full" in timeline_row.error

    def test_the_folder_is_only_made_when_something_is_written(self, tmp_path: Path) -> None:
        target = tmp_path / "never"
        ArtifactWriter(target)
        assert not target.exists()


class TestArtifactsPanel:
    def test_every_kind_written_has_an_explanation(
        self, app: QApplication, tmp_path: Path, edit: Timeline
    ) -> None:
        del app
        for artifact in ArtifactWriter(tmp_path).write_all(edit):
            assert artifact.explanation, f"{artifact.kind} has no explanation"

    def test_rows_appear_for_each_file(
        self, app: QApplication, tmp_path: Path, edit: Timeline
    ) -> None:
        del app
        panel = ArtifactsPanel()
        written = ArtifactWriter(tmp_path).write_all(edit)
        panel.show_artifacts(written)
        assert len(panel.rows) == len(written)

    def test_showing_again_replaces_rather_than_appends(
        self, app: QApplication, tmp_path: Path, edit: Timeline
    ) -> None:
        del app
        panel = ArtifactsPanel()
        written = ArtifactWriter(tmp_path).write_all(edit)
        panel.show_artifacts(written)
        panel.show_artifacts(written)
        assert len(panel.rows) == len(written)

    def test_the_destination_is_stated(self, app: QApplication, tmp_path: Path) -> None:
        del app
        panel = ArtifactsPanel()
        panel.show_destination(tmp_path)
        assert str(tmp_path) in panel.destination.text()

    def test_a_failed_artifact_says_so(self, app: QApplication, tmp_path: Path) -> None:
        del app
        panel = ArtifactsPanel()
        panel.show_artifacts([Artifact(tmp_path / "x.json", "timeline", error="disk full")])
        assert len(panel.rows) == 1


class TestSessionHeader:
    def test_it_starts_empty(self, app: QApplication) -> None:
        del app
        assert "No session" in SessionHeader().filename.text()

    def test_it_names_the_track_and_the_driver(
        self, app: QApplication, session: SyntheticSession, tmp_path: Path
    ) -> None:
        del app
        header = SessionHeader()
        header.show_session(tmp_path / "x.acreplay", session.meta)
        text = header.detail.text()
        assert session.meta.track_label in text
        player = session.meta.player()
        assert player is not None
        assert player.display_name in text

    def test_it_reports_parked_cars(
        self, app: QApplication, session: SyntheticSession, tmp_path: Path
    ) -> None:
        del app
        header = SessionHeader()
        header.show_session(tmp_path / "x.acreplay", session.meta, skipped=15)
        assert "15 parked" in header.detail.text()

    def test_recording_quality_is_shown_with_its_advice(
        self, app: QApplication, session: SyntheticSession, tmp_path: Path
    ) -> None:
        # domain/session.py claims the rate is "surfaced in the GUI". It was not.
        del app
        header = SessionHeader()
        header.show_session(tmp_path / "x.acreplay", session.meta)
        assert header.quality.text() == session.meta.quality.name
        assert header.quality.toolTip() == session.meta.quality.advice

    @pytest.mark.parametrize(
        ("seconds", "expected"), [(0, "0:00"), (61, "1:01"), (3661, "1:01:01")]
    )
    def test_durations_read_as_clocks(self, seconds: int, expected: str) -> None:
        assert format_clock(seconds) == expected


class TestResultViews:
    def test_the_timeline_view_shows_an_edit(self, app: QApplication, edit: Timeline) -> None:
        del app
        view = TimelineView()
        view.show_timeline(edit)
        assert len(view.scene().items()) >= edit.shot_count

    def test_one_item_per_shot_not_per_frame(
        self, app: QApplication, edit: Timeline, session: SyntheticSession
    ) -> None:
        del app
        view = TimelineView()
        view.show_timeline(edit)
        assert len(view.scene().items()) < session.time.shape[0] / 10

    def test_the_trace_is_a_single_item(self, app: QApplication, session: SyntheticSession) -> None:
        # Giving each sample its own item is the standard way to make Qt unusable.
        del app
        car = session.signals[0]
        view = TrackMapView()
        view.show_session(car.position, session.time, car.sideslip)
        assert len(view.scene().items()) == 1

    def test_shot_markers_are_added(
        self, app: QApplication, session: SyntheticSession, edit: Timeline
    ) -> None:
        del app
        car = session.signals[0]
        view = TrackMapView()
        view.show_session(car.position, session.time, car.sideslip)
        before = len(view.scene().items())
        view.highlight_shots(edit)
        assert len(view.scene().items()) == before + edit.shot_count

    def test_an_empty_map_is_safe(self, app: QApplication) -> None:
        del app
        view = TrackMapView()
        view.mark_time(1.0)
        view.show_session(np.zeros((0, 3)), np.zeros(0), np.zeros(0))

    def test_an_empty_timeline_is_safe(self, app: QApplication) -> None:
        del app
        TimelineView().mark_time(3.0)


class TestMainWindowResults:
    def test_the_notes_panel_shows_the_caveats(self, window: MainWindow, edit: Timeline) -> None:
        # The GUI is the easiest place to convey unearned confidence, so the
        # blind spots must be visible next to the result.
        window._show_result(edit)
        assert "Not detected" in window.notes.toPlainText()
        window.close()

    def test_the_shot_table_is_filled(self, window: MainWindow, edit: Timeline) -> None:
        window._show_result(edit)
        assert window.shot_table.rowCount() == edit.shot_count
        window.close()

    def test_export_is_enabled_only_once_there_is_something_to_export(
        self, window: MainWindow, csv_session: Path, edit: Timeline
    ) -> None:
        window.open_session(csv_session)
        assert not window.export_action.isEnabled()
        window.project = window.project.completed(("initial_d", 0.0), edit)
        window._apply_phase()
        assert window.export_action.isEnabled()
        window.close()


class TestWorkerReporting:
    def test_a_missing_session_reports_rather_than_hanging(self, tmp_path: Path) -> None:
        # An exception escaping a worker slot kills the thread silently and
        # leaves the window waiting for a result that never arrives.
        worker = DirectorWorker(tmp_path / "nope.acreplay", BUILTIN_STYLES["initial_d"])
        failures: list[str] = []
        worker.failed.connect(failures.append)
        worker.run()
        assert failures

    def test_a_cancelled_run_reports_cancelled_not_failed(self, csv_session: Path) -> None:
        # A user who pressed Stop and is shown "Could not direct this replay" has
        # been told their machine broke.
        worker = DirectorWorker(csv_session, BUILTIN_STYLES["initial_d"])
        stopped: list[str] = []
        failed: list[str] = []
        worker.cancelled.connect(stopped.append)
        worker.failed.connect(failed.append)
        worker.cancel()
        worker.run()
        assert stopped
        assert not failed

    def test_the_stop_message_names_the_stage(self, csv_session: Path) -> None:
        worker = DirectorWorker(csv_session, BUILTIN_STYLES["initial_d"])
        stopped: list[str] = []
        worker.cancelled.connect(stopped.append)
        worker.cancel()
        worker.run()
        assert "cancelled" in stopped[0].lower()


class TestSettingsPersistence:
    def test_the_output_root_survives_a_new_store(self, settings: Settings, tmp_path: Path) -> None:
        settings.output_root = tmp_path / "chosen"
        settings.sync()
        assert settings.output_root == tmp_path / "chosen"

    def test_recent_sessions_are_remembered_most_recent_first(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        first = tmp_path / "one.csv"
        second = tmp_path / "two.csv"
        for path in (first, second):
            path.write_text("x", encoding="utf-8")
        settings.remember(first)
        settings.remember(second)
        assert settings.recent[0] == second

    def test_a_recent_session_that_no_longer_exists_is_dropped(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        gone = tmp_path / "gone.csv"
        gone.write_text("x", encoding="utf-8")
        settings.remember(gone)
        gone.unlink()
        assert gone not in settings.recent

    def test_the_default_span_is_a_few_minutes_not_the_whole_session(
        self, settings: Settings
    ) -> None:
        # A first run on a twenty-four minute replay produces an edit the user
        # has no basis to judge, minutes later.
        assert 0.0 < settings.scope_seconds <= 600.0

    def test_timing_estimates_blend_rather_than_jump(self, settings: Settings) -> None:
        settings.record_timing("run", 1.0)
        settings.record_timing("run", 2.0)
        blended = settings.seconds_per_unit("run", 0.0)
        assert 1.0 < blended < 2.0

    def test_a_nonsense_stored_value_falls_back(self, settings: Settings) -> None:
        settings._store.setValue("run/scopeSeconds", "not a number")
        assert settings.scope_seconds > 0.0


class TestWelcome:
    def test_it_explains_the_whole_flow(self) -> None:
        assert len(STEPS) == 4
        assert all(heading and detail for heading, detail in STEPS)

    def test_it_warns_about_recording_quality(self, app: QApplication) -> None:
        # The honesty rule reaching the user before they spend twenty minutes on
        # an 8 Hz recording that cannot support event timing at all.
        del app
        from assetto_ai_director.presentation.gui.welcome import (  # noqa: PLC0415
            _QUALITY_NOTE,
        )

        assert "quality" in _QUALITY_NOTE.lower()

    def test_dismissing_it_sticks(self, app: QApplication, settings: Settings) -> None:
        del app
        settings.first_run_done = False
        window = MainWindow(settings)
        window._dismiss_welcome()
        assert settings.first_run_done
        window.close()


class TestCamerasTab:
    """The one-click hand-off, which is the whole reason the tab exists."""

    @staticmethod
    def _run(window: MainWindow, csv_session: Path, edit: Timeline) -> None:
        """Put a finished edit on screen the way a real run would.

        Args:
            window: The window.
            csv_session: A session file to open.
            edit: The directed timeline.
        """
        window.open_session(csv_session)
        # A real run decodes the session on the worker and stores it; opening
        # alone reads only the header. The checks need the car positions.
        window.project.session = load_session(csv_session)
        window.project = window.project.completed(
            (window.run_panel.style_name, window.run_panel.scope_seconds), edit
        )
        window._show_result(edit)

    def test_the_tab_exists(self, window: MainWindow) -> None:
        labels = [window.tabs.tabText(index) for index in range(window.tabs.count())]
        assert "Cameras" in labels
        window.close()

    def test_it_is_empty_and_its_buttons_are_dead_before_a_run(self, window: MainWindow) -> None:
        assert not window.camera_inspector.copy_button.isEnabled()
        assert not window.camera_inspector.save_button.isEnabled()
        window.close()

    def test_a_finished_run_fills_it_without_being_asked(
        self, window: MainWindow, csv_session: Path, edit: Timeline
    ) -> None:
        # No dialog, no path, no second button: the report must exist by the time
        # the run finishes, or the person verifying will never produce one.
        self._run(window, csv_session, edit)
        assert window._diagnostic is not None
        assert len(window._previews) == edit.shot_count
        assert window.camera_inspector.copy_button.isEnabled()
        window.close()

    def test_copying_puts_a_usable_report_on_the_clipboard(
        self, window: MainWindow, csv_session: Path, edit: Timeline
    ) -> None:
        self._run(window, csv_session, edit)
        window._copy_diagnostic()
        pasted = QGuiApplication.clipboard().text()
        document = json.loads(pasted)
        assert document["summary"]["cameras"] == edit.shot_count
        assert document["cameras_included"] == "notable"
        assert document["blind_spots"]
        window.close()

    def test_the_copied_report_is_small_enough_to_paste(
        self, window: MainWindow, csv_session: Path, edit: Timeline
    ) -> None:
        # The full document runs to hundreds of kilobytes. If the button ever
        # starts copying that, the hand-off silently stops working.
        self._run(window, csv_session, edit)
        window._copy_diagnostic()
        assert len(QGuiApplication.clipboard().text()) < 200_000

    def test_selecting_a_camera_shows_every_check_behind_it(
        self, window: MainWindow, csv_session: Path, edit: Timeline
    ) -> None:
        self._run(window, csv_session, edit)
        items = window.camera_inspector.view.scene().items()
        assert items
        items[0].setSelected(True)
        assert "threshold:" in window.camera_inspector.detail.toPlainText()
        window.close()

    def test_the_contact_sheet_is_written_where_the_run_went(
        self, window: MainWindow, csv_session: Path, edit: Timeline
    ) -> None:
        self._run(window, csv_session, edit)
        window._save_contact_sheet()
        assert (window._current_folder() / "cameras.svg").is_file()
        window.close()

    def test_opening_another_session_clears_it(
        self, window: MainWindow, csv_session: Path, edit: Timeline
    ) -> None:
        self._run(window, csv_session, edit)
        window.open_session(csv_session)
        assert window._diagnostic is None
        assert not window.camera_inspector.copy_button.isEnabled()
        window.close()

    def test_a_timeline_without_its_session_says_so_rather_than_looking_clean(
        self, window: MainWindow, csv_session: Path, edit: Timeline
    ) -> None:
        # A blank tab and a tab full of passing cameras look identical at a
        # glance, and a person here to verify something would believe either.
        window.open_session(csv_session)
        window.project = window.project.completed(
            (window.run_panel.style_name, window.run_panel.scope_seconds), edit
        )
        window._show_result(edit)
        assert window._diagnostic is None
        assert "cannot be checked" in window.camera_inspector.summary.text()
        assert not window.camera_inspector.copy_button.isEnabled()
        window.close()
