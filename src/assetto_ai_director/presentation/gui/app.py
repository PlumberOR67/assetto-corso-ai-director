"""The main window: composition and the state machine, and not much else.

Every panel lives in its own module. What is left here is what only the window
can do — hold the project, own the worker thread, and decide which phase the
application is in.

**Opening a session no longer runs it.** That single change is what most of the
rest hangs off. The old window started the entire analysis straight from the file
dialog, so a user who wanted to look at a replay before committing twenty-four
minutes to it had no way to; now opening reads only the header, the identity card
fills in, and nothing expensive happens until Run.

**The notes panel is not decoration.** Everything the analysis could not see —
undecoded channels, skipped cars, unmeasured style pacing — is shown next to the
result rather than buried. A director that looks confident about a session it only
half understood is the failure this whole project is arranged against, and a GUI
is where that confidence is easiest to convey by accident.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt, QThread, QUrl
from PySide6.QtGui import QAction, QCloseEvent, QDesktopServices, QGuiApplication
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from assetto_ai_director.application.camera.paths import SubjectTracks
from assetto_ai_director.application.diagnostics import diagnose_cameras
from assetto_ai_director.application.diagnostics.scene import Preview, build_previews
from assetto_ai_director.application.director.styles import BUILTIN_STYLES
from assetto_ai_director.domain.diagnostics import DiagnosticReport
from assetto_ai_director.domain.session import SessionMeta
from assetto_ai_director.domain.timeline import Timeline
from assetto_ai_director.domain.track import TrackSpline
from assetto_ai_director.infrastructure.acreplay.structure import read_structure
from assetto_ai_director.infrastructure.csvio.source import read_csv
from assetto_ai_director.infrastructure.export.cameras_ini import planned_spans
from assetto_ai_director.infrastructure.export.diagnostic_json import diagnostic_to_text
from assetto_ai_director.infrastructure.export.preview_svg import write_contact_sheet
from assetto_ai_director.infrastructure.loader import LoadedSession
from assetto_ai_director.infrastructure.track.resolve import resolve_spline
from assetto_ai_director.presentation.gui.artifacts import ArtifactsPanel
from assetto_ai_director.presentation.gui.camera_inspector import CameraInspector
from assetto_ai_director.presentation.gui.header import SessionHeader
from assetto_ai_director.presentation.gui.output import ArtifactWriter, run_folder
from assetto_ai_director.presentation.gui.project import Phase, Project, RunKey
from assetto_ai_director.presentation.gui.runpanel import RunPanel
from assetto_ai_director.presentation.gui.settings import Settings
from assetto_ai_director.presentation.gui.stages import StageList, format_duration
from assetto_ai_director.presentation.gui.theme import apply_theme
from assetto_ai_director.presentation.gui.timeline_view import TimelineView
from assetto_ai_director.presentation.gui.trackmap import TrackMapView
from assetto_ai_director.presentation.gui.welcome import WelcomePage
from assetto_ai_director.presentation.gui.workers import DirectorWorker, run_in_thread

__all__ = ["MainWindow", "main"]

_SHOT_COLUMNS = ("#", "start", "dur", "rig", "size", "side", "score", "because")

_SESSION_FILTER = "Sessions (*.acreplay *.csv);;Replays (*.acreplay);;Telemetry (*.csv);;All (*)"

_STOP_GRACE_MS = 5000
"""How long to wait for a worker to notice a cancellation when closing.

Cooperative cancellation is bounded by the slowest checkpoint, which is one car's
frame decode. Five seconds covers that on a large replay; past it the user is
asked rather than made to wait.
"""


class MainWindow(QMainWindow):
    """The application's only window."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Build the window.

        Args:
            settings: Persisted preferences. A default store is opened when
                omitted; tests pass one backed by a temporary file.
        """
        super().__init__()
        self.settings = settings or Settings()
        self.project = Project()
        self._worker: DirectorWorker | None = None
        self._thread: QThread | None = None
        self._closing = False
        self._diagnostic: DiagnosticReport | None = None
        self._previews: tuple[Preview, ...] = ()
        self._track: TrackSpline | None = None

        self.setWindowTitle("Assetto AI Director")
        self.resize(1400, 900)

        self._build_ui()
        self._build_menu()
        self._restore()
        self._apply_phase()

    # --- Construction ------------------------------------------------------

    def _build_ui(self) -> None:
        """Assemble the window's widgets."""
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.header = SessionHeader()
        outer.addWidget(self.header)

        body = QHBoxLayout()
        body.setContentsMargins(16, 12, 16, 12)
        body.setSpacing(18)

        body.addWidget(self._build_left_column(), 0)
        body.addWidget(self._build_results(), 1)
        outer.addLayout(body, 1)

        self.setCentralWidget(root)
        self.setStatusBar(self.statusBar())
        self._build_status_bar()

    def _build_left_column(self) -> QWidget:
        """Build the steps, the Run button and the stage list.

        Returns:
            The column widget.
        """
        column = QWidget()
        column.setFixedWidth(320)
        layout = QVBoxLayout(column)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(18)

        self.run_panel = RunPanel()
        self.run_panel.open_requested.connect(self._choose_session)
        self.run_panel.run_requested.connect(self._run)
        self.run_panel.cancel_requested.connect(self._cancel)
        self.run_panel.output_change_requested.connect(self._choose_output)
        self.run_panel.open_output_requested.connect(self._open_output)
        self.run_panel.settings_changed.connect(self._on_settings_changed)
        layout.addWidget(self.run_panel)

        self.stage_list = StageList()
        layout.addWidget(self.stage_list)
        layout.addStretch(1)
        return column

    def _build_results(self) -> QWidget:
        """Build the result tabs.

        Returns:
            The tab widget, with the welcome page in front of it until dismissed.
        """
        self.tabs = QTabWidget()

        self.map_view = TrackMapView()
        self.tabs.addTab(self.map_view, "Map")

        self.timeline_view = TimelineView()
        timeline_page = QWidget()
        timeline_layout = QVBoxLayout(timeline_page)
        timeline_layout.setContentsMargins(0, 0, 0, 0)
        timeline_layout.addWidget(self.timeline_view)
        timeline_layout.addStretch(1)
        self.tabs.addTab(timeline_page, "Timeline")

        self.shot_table = QTableWidget(0, len(_SHOT_COLUMNS))
        self.shot_table.setHorizontalHeaderLabels(list(_SHOT_COLUMNS))
        self.shot_table.verticalHeader().setVisible(False)
        self.shot_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.shot_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.shot_table.itemSelectionChanged.connect(self._on_shot_selected)
        self.tabs.addTab(self.shot_table, "Shots")

        self.camera_inspector = CameraInspector()
        self.camera_inspector.copy_requested.connect(self._copy_diagnostic)
        self.camera_inspector.save_requested.connect(self._save_contact_sheet)
        self.tabs.addTab(self.camera_inspector, "Cameras")

        self.notes = QPlainTextEdit()
        self.notes.setReadOnly(True)
        self.notes.setPlaceholderText(
            "What the analysis could not see will appear here after a run."
        )
        self.tabs.addTab(self.notes, "What we could not see")

        self.artifacts = ArtifactsPanel()
        self.tabs.addTab(self.artifacts, "Results")

        self.timeline_view.scrubbed.connect(self.map_view.mark_time)
        self.timeline_view.shot_selected.connect(self._select_shot)
        self.map_view.hovered.connect(self.timeline_view.mark_time)

        self.welcome = WelcomePage()
        self.welcome.open_requested.connect(self._choose_session)
        self.welcome.output_change_requested.connect(self._choose_output)
        self.welcome.dismissed.connect(self._dismiss_welcome)

        self.results_splitter = QSplitter(Qt.Orientation.Vertical)
        self.results_splitter.addWidget(self.welcome)
        self.results_splitter.addWidget(self.tabs)
        self.results_splitter.setSizes([1, 0])
        return self.results_splitter

    def _build_status_bar(self) -> None:
        """Put the live progress line and the Stop button in the status bar."""
        self.status_label = QLabel("Open a session to begin.")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setVisible(False)
        self.stop_button.clicked.connect(self._cancel)
        self.statusBar().addWidget(self.status_label, 1)
        self.statusBar().addPermanentWidget(self.stop_button)

    def _build_menu(self) -> None:
        """Build the menu bar."""
        file_menu = self.menuBar().addMenu("&File")

        open_action = QAction("&Open session…", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._choose_session)
        file_menu.addAction(open_action)

        self.recent_menu = file_menu.addMenu("Open &recent")
        self._rebuild_recent()

        file_menu.addSeparator()
        self.export_action = QAction("&Export elsewhere…", self)
        self.export_action.setShortcut("Ctrl+Shift+S")
        self.export_action.triggered.connect(self._export_elsewhere)
        file_menu.addAction(self.export_action)

        reveal_action = QAction("Open results &folder", self)
        reveal_action.triggered.connect(self._open_output)
        file_menu.addAction(reveal_action)

        file_menu.addSeparator()
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        help_menu = self.menuBar().addMenu("&Help")
        about_action = QAction("What the outputs are", self)
        about_action.triggered.connect(self._show_help)
        help_menu.addAction(about_action)

    def _restore(self) -> None:
        """Apply persisted settings to the window."""
        geometry = self.settings.geometry()
        if geometry:
            self.restoreGeometry(geometry)
        splitter = self.settings.splitter("results")
        if splitter:
            self.results_splitter.restoreState(splitter)
        self.run_panel.restore(self.settings.style, self.settings.scope_seconds)
        self._show_welcome(not self.settings.first_run_done)
        self._refresh_output_labels()

    # --- Opening -----------------------------------------------------------

    def _choose_session(self) -> None:
        """Ask for a session file and open it."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open session", str(self.settings.last_input_folder), _SESSION_FILTER
        )
        if path:
            self.open_session(Path(path))

    def open_session(self, path: Path) -> None:
        """Read a session's header and show what it is.

        Deliberately cheap: this reads the structure only, without decoding a
        single frame. Opening and running are separate acts, and this is the half
        that must stay instant.

        Args:
            path: The session file.
        """
        if self._busy():
            return
        try:
            meta = self._describe(path)
        except Exception as error:
            self._show_error(f"Could not read {path.name}", f"{type(error).__name__}: {error}")
            return

        self.project = self.project.opened(path)
        self.settings.last_input_folder = path.parent
        self.settings.remember(path)
        self._rebuild_recent()
        self._dismiss_welcome()

        self.header.show_session(path, meta)
        self.run_panel.show_session_length(float(meta.duration))
        self.setWindowTitle(f"{path.name} — Assetto AI Director")
        self.stage_list.reset()
        self._clear_results()
        self._refresh_output_labels()
        self.status_label.setText("Ready. Press Run.")
        self._apply_phase()

    @staticmethod
    def _describe(path: Path) -> SessionMeta:
        """Read a session's metadata without decoding telemetry.

        For a replay this is the structural walk only — no frames are read, which
        is what keeps opening instant on a 233 MB file. A CSV has no header to
        walk separately, so it is parsed in full; it is a text file and finishes
        before the dialog has closed.

        Args:
            path: The session file.

        Returns:
            The session metadata.
        """
        if path.suffix.lower() == ".csv":
            return read_csv(path).meta
        return read_structure(path).to_session_meta()

    # --- Running -----------------------------------------------------------

    def _run(self) -> None:
        """Direct the open session with the current settings."""
        source = self.project.path
        if source is None or self._busy():
            return

        key: RunKey = (self.run_panel.style_name, self.run_panel.scope_seconds)
        cached = self.project.cached(key)
        if cached is not None:
            # Already computed with exactly these settings. Showing it costs
            # nothing, and re-running would be minutes for a known answer.
            self.project = self.project.matching(key)
            self._show_result(cached)
            self._apply_phase()
            return

        self.project.phase = Phase.RUNNING
        self.stage_list.reset()
        self.status_label.setText("Starting…")

        worker = DirectorWorker(
            source,
            BUILTIN_STYLES[self.run_panel.style_name],
            seconds=self.run_panel.scope_seconds,
            analysis=self.project.analysis,
        )
        worker.progress.connect(self._on_progress)
        worker.loaded.connect(self._on_loaded)
        worker.finished.connect(self._on_finished)
        worker.failed.connect(self._on_failed)
        worker.cancelled.connect(self._on_cancelled)
        self._worker = worker
        self._thread = run_in_thread(worker)
        self._apply_phase()

    def _cancel(self) -> None:
        """Ask the running analysis to stop."""
        if self._worker is not None:
            self._worker.cancel()
            self.status_label.setText("Stopping…")
            self.stop_button.setEnabled(False)

    def _busy(self) -> bool:
        """Report whether a run is in flight.

        Returns:
            ``True`` when a worker owns the project.
        """
        if self._thread is not None and self._thread.isRunning():
            self.status_label.setText("Already working. Press Stop first.")
            return True
        return False

    # --- Worker signals ----------------------------------------------------

    def _on_progress(self, description: str, done: int, total: int) -> None:
        """Show one progress update.

        Args:
            description: The stage name.
            done: Units completed.
            total: Units in the stage.
        """
        self.stage_list.report(description, done, total)
        remaining = self.stage_list.remaining
        suffix = f"  ·  ≈{format_duration(remaining)} left" if remaining else ""
        # Never a bare percentage: the stage name is what turns a number into
        # information, and its absence is what produced the original complaint.
        share = f" — {done * 100 // total}%" if total else ""
        self.status_label.setText(f"{description}{share}{suffix}")

    def _on_loaded(self, session: object) -> None:
        """Record the decoded session and show what it turned out to contain.

        Args:
            session: The loaded session.
        """
        loaded: LoadedSession = session  # type: ignore[assignment]
        self.project.session = loaded
        self.header.show_session(loaded.path, loaded.meta, skipped=len(loaded.skipped))
        self.map_view.show_session(
            next(iter(loaded.signals.values())).position,
            loaded.time,
            next(iter(loaded.signals.values())).sideslip,
        )

    def _on_finished(self, timeline: object) -> None:
        """Show a finished edit and write its files.

        Args:
            timeline: The directed timeline.
        """
        result: Timeline = timeline  # type: ignore[assignment]
        key: RunKey = (self.run_panel.style_name, self.run_panel.scope_seconds)
        if self._worker is not None:
            self.project.analysis = self._worker.analysis
        self.project = self.project.completed(key, result)
        self._show_result(result)
        self._write_artifacts(result)
        self.status_label.setText(
            f"Done — {result.shot_count} shots, average {float(result.average_shot_length):.1f}s."
        )
        self._apply_phase()

    def _on_failed(self, message: str) -> None:
        """Report a failed run.

        Args:
            message: What went wrong.
        """
        self.stage_list.fail_running()
        self.project = self.project.failed(message)
        self.status_label.setText(message)
        self._show_error("Could not direct this session", message)
        self._apply_phase()

    def _on_cancelled(self, message: str) -> None:
        """Report a stopped run.

        Never as a failure, and never destroying a result the user already had.
        Pressing Stop is a choice; presenting it as a fault tells the user their
        machine broke when it did exactly what they asked.

        Args:
            message: Which stage was interrupted.
        """
        self.stage_list.fail_running()
        self.project = self.project.stopped(message)
        self.status_label.setText(message)
        self.stop_button.setEnabled(True)
        self._apply_phase()

    # --- Results -----------------------------------------------------------

    def _show_result(self, timeline: Timeline) -> None:
        """Draw a timeline in every view.

        Args:
            timeline: The edit to show.
        """
        events = self.project.analysis.detection.events if self.project.analysis else ()
        self.timeline_view.show_timeline(timeline, events)
        self.map_view.highlight_shots(timeline)
        self._fill_shot_table(timeline)
        self._check_cameras(timeline)
        self.notes.setPlainText("\n\n".join(timeline.notes))
        self.tabs.setCurrentIndex(0)

    def _clear_results(self) -> None:
        """Empty every result view."""
        self.shot_table.setRowCount(0)
        self.notes.setPlainText("")
        self.camera_inspector.clear()
        self._diagnostic = None
        self._previews = ()
        self._track = None
        self.artifacts.show_artifacts([])

    def _fill_shot_table(self, timeline: Timeline) -> None:
        """Populate the shot list.

        Args:
            timeline: The edit to list.
        """
        self.shot_table.setRowCount(timeline.shot_count)
        for row, shot in enumerate(timeline.shots):
            candidate = shot.candidate
            values = (
                str(shot.index),
                f"{float(shot.start):.1f}s",
                f"{float(shot.duration):.1f}s",
                candidate.rig.value,
                candidate.size.name.lower(),
                candidate.side.value,
                f"{float(candidate.score):.3f}",
                candidate.anchor.kind.value if candidate.anchor else "",
            )
            for column, value in enumerate(values):
                self.shot_table.setItem(row, column, QTableWidgetItem(value))
        self.shot_table.resizeColumnsToContents()

    def _check_cameras(self, timeline: Timeline) -> None:
        """Diagnose the cut's cameras and fill the Cameras tab.

        Runs inline rather than on a worker: the checks are arithmetic over
        arrays the window already holds, taking a fraction of the time the
        scheduler that produced the timeline just spent. Moving it to a thread
        would add a second lifecycle to manage for no gain a user can perceive.

        Args:
            timeline: The finished edit.
        """
        session = self.project.session
        if session is None:
            # Reachable when a timeline is shown without the session it came
            # from — a restored result, or a file opened but never run. Leaving
            # the tab blank would look exactly like a clean bill of health.
            self._diagnostic = None
            self._previews = ()
            self.camera_inspector.show_unavailable(
                "The cameras cannot be checked without the session's car positions, "
                "which this edit does not carry on its own. Run the session to "
                "produce them."
            )
            return
        tracks = SubjectTracks.from_signals(session.signals, session.time, session.sample_rate)
        resolved = resolve_spline(
            timeline.session.track_id,
            timeline.session.layout,
            next(iter(tracks.positions.values()), None),
            ac_root=self.settings.ac_root,
        )
        report = diagnose_cameras(
            timeline,
            tracks,
            exported_spans=planned_spans(timeline, track=resolved.spline),
            track=resolved.spline,
            notes=(resolved.note,) if resolved.note else (),
        )
        previews = build_previews(report, tracks)
        self._diagnostic = report
        self._previews = tuple(previews)
        self._track = resolved.spline
        self.camera_inspector.show_report(report, previews)

    def _copy_diagnostic(self) -> None:
        """Put the compact camera report on the clipboard.

        The compact form, not the full one: a three-minute cut produces a couple
        of hundred kilobytes of JSON, and being pasteable is the entire point of
        the button. The full report is already on disk beside the timeline.
        """
        if self._diagnostic is None:
            return
        report_text = diagnostic_to_text(self._diagnostic, notable_only=True)
        QGuiApplication.clipboard().setText(report_text)
        self.status_label.setText(
            f"Camera report copied — {len(report_text):,} characters. Paste it into the chat."
        )

    def _save_contact_sheet(self) -> None:
        """Write the contact sheet into this run's output folder and open it."""
        if self._diagnostic is None:
            return
        destination = self._current_folder() / "cameras.svg"
        try:
            write_contact_sheet(
                self._previews,
                destination,
                title=(
                    f"{self._diagnostic.track_id} / "
                    f"{self._diagnostic.layout or 'default'} - {self._diagnostic.style_name}"
                ),
            )
        except OSError as error:
            self._show_error("Could not write the contact sheet", str(error))
            return
        self.status_label.setText(f"Contact sheet written to {destination}.")
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(destination)))

    def _write_artifacts(self, timeline: Timeline) -> None:
        """Write the run's files into the output folder.

        Args:
            timeline: The finished edit.
        """
        folder = self._current_folder()
        writer = ArtifactWriter(folder)
        self.artifacts.show_destination(folder)
        written = writer.write_all(timeline, self._track)
        if self._diagnostic is not None:
            written.append(writer.write_diagnostic_report(self._diagnostic))
        self.artifacts.show_artifacts(written)

    def _current_folder(self) -> Path:
        """Where this run's files belong.

        Returns:
            The per-run output folder.
        """
        return run_folder(
            self.settings.output_root,
            self.project.path,
            self.run_panel.style_name,
        )

    # --- Interaction -------------------------------------------------------

    def _select_shot(self, index: int) -> None:
        """Select a shot in the table.

        Args:
            index: The shot's index.
        """
        if 0 <= index < self.shot_table.rowCount():
            self.shot_table.selectRow(index)

    def _on_shot_selected(self) -> None:
        """Move the playhead to the selected shot."""
        timeline = self.project.timeline
        rows = self.shot_table.selectionModel().selectedRows()
        if timeline is None or not rows:
            return
        index = rows[0].row()
        if 0 <= index < timeline.shot_count:
            moment = float(timeline.shots[index].start)
            self.timeline_view.mark_time(moment)
            self.map_view.mark_time(moment)

    def _on_settings_changed(self) -> None:
        """React to a change of style or span."""
        self.settings.style = self.run_panel.style_name
        self.settings.scope_seconds = self.run_panel.scope_seconds
        self._refresh_output_labels()

        key: RunKey = (self.run_panel.style_name, self.run_panel.scope_seconds)
        if self.project.cached(key) is not None:
            self.project = self.project.matching(key)
            cached = self.project.timeline
            if cached is not None:
                self._show_result(cached)
        elif self.project.current is not None and key[1] != self.project.current[1]:
            # A change of span invalidates the decode and the analysis, not just
            # the edit. Keeping the analysis would direct a three-minute
            # analysis over a twenty-four-minute horizon.
            self.project = self.project.invalidated_by_scope()
        else:
            self.project = self.project.marked_stale()
        self._apply_phase()

    def _choose_output(self) -> None:
        """Ask for a new output root."""
        chosen = QFileDialog.getExistingDirectory(
            self, "Where should results go?", str(self.settings.output_root)
        )
        if chosen:
            self.settings.output_root = Path(chosen)
            self._refresh_output_labels()

    def _open_output(self) -> None:
        """Show the output folder in the system file manager."""
        folder = self._current_folder() if self.project.path else self.settings.output_root
        folder.mkdir(parents=True, exist_ok=True)
        self.artifacts.open_folder(folder)

    def _export_elsewhere(self) -> None:
        """Write the current edit to a location the user picks."""
        timeline = self.project.timeline
        if timeline is None:
            self.status_label.setText("Nothing to export yet — run first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export timeline", str(self._current_folder() / "timeline.json"), "JSON (*.json)"
        )
        if not path:
            return
        artifact = ArtifactWriter(Path(path).parent).write_timeline_to(timeline, Path(path))
        self.artifacts.show_artifacts([artifact])
        self.status_label.setText(
            f"Wrote {path}" if artifact.error is None else f"Could not write: {artifact.error}"
        )

    def _show_help(self) -> None:
        """Explain what each output is for."""
        from assetto_ai_director.presentation.gui.artifacts import EXPLANATIONS  # noqa: PLC0415

        body = "\n\n".join(f"{name}\n{text}" for name, text in EXPLANATIONS.items())
        QMessageBox.information(self, "What the outputs are", body)

    def _show_error(self, title: str, detail: str) -> None:
        """Report a failure without losing anything already on screen.

        Args:
            title: Short summary.
            detail: The message.
        """
        QMessageBox.warning(self, title, detail)

    # --- Phase -------------------------------------------------------------

    def _apply_phase(self) -> None:
        """Enable the controls that make sense in the current phase."""
        phase = self.project.phase
        self.run_panel.apply_phase(phase, self._estimate())
        self.stop_button.setVisible(phase is Phase.RUNNING)
        self.export_action.setEnabled(self.project.timeline is not None)
        if phase is Phase.STALE:
            self.status_label.setText(
                "Settings changed — this edit was made with the previous ones. Run again."
            )

    def _estimate(self) -> float | None:
        """Guess how long the next run will take.

        Returns:
            Seconds, or ``None`` before there is anything to base it on.
        """
        if self.project.path is None:
            return None
        span = self.run_panel.scope_seconds
        if span <= 0.0:
            session = self.project.session
            span = session.duration if session else 0.0
        if span <= 0.0:
            return None
        # Scheduling dominates and is linear in the horizon; the rest is roughly
        # proportional too. Deliberately a single blended rate rather than a
        # per-stage model, which would need measurements this window does not
        # have on a first run.
        rate = self.settings.seconds_per_unit("run", 0.10)
        if self.project.analysis is not None:
            rate *= 0.55  # decode, detection and scoring are already done
        return span * rate

    def _refresh_output_labels(self) -> None:
        """Show where results will go, before the run rather than after."""
        folder = self._current_folder() if self.project.path else self.settings.output_root
        self.run_panel.show_output(folder)
        self.artifacts.show_destination(folder)
        self.welcome.show_output(self.settings.output_root)

    def _rebuild_recent(self) -> None:
        """Refill the recent-sessions menu."""
        self.recent_menu.clear()
        recent = self.settings.recent
        self.recent_menu.setEnabled(bool(recent))
        for path in recent:
            action = QAction(path.name, self)
            action.setToolTip(str(path))
            action.triggered.connect(lambda _=False, target=path: self.open_session(target))
            self.recent_menu.addAction(action)

    def _show_welcome(self, visible: bool) -> None:
        """Show or hide the first-run page.

        Args:
            visible: Whether to show it.
        """
        self.welcome.setVisible(visible)
        self.tabs.setVisible(not visible)
        self.results_splitter.setSizes([1, 0] if visible else [0, 1])

    def _dismiss_welcome(self) -> None:
        """Put the welcome page away for good."""
        if not self.settings.first_run_done:
            self.settings.first_run_done = True
        self._show_welcome(False)

    # --- Shutdown ----------------------------------------------------------

    def closeEvent(self, event: QCloseEvent) -> None:
        """Stop any running analysis before the window goes away.

        There was no handler at all before this, so closing mid-run destroyed a
        live ``QThread`` and took the process with it. Never ``terminate()``:
        that kills the worker mid-allocation and mid-read, with memory and file
        handles in a state nobody has reasoned about.

        Args:
            event: The close event.
        """
        self.settings.set_geometry(bytes(self.saveGeometry().data()))
        self.settings.set_splitter("results", bytes(self.results_splitter.saveState().data()))
        self.settings.sync()

        if self._thread is not None and self._thread.isRunning():
            if self._worker is not None:
                self._worker.cancel()
            self._thread.quit()
            if not self._thread.wait(_STOP_GRACE_MS):
                answer = QMessageBox.question(
                    self,
                    "Still stopping",
                    "The analysis has not stopped yet. Close anyway?",
                )
                if answer is not QMessageBox.StandardButton.Yes:
                    event.ignore()
                    return
        self._closing = True
        event.accept()


def main() -> None:
    """Run the desktop application."""
    app = QApplication(sys.argv)
    app.setOrganizationName("AssettoAIDirector")
    app.setApplicationName("AssettoAIDirector")
    apply_theme(app)

    window = MainWindow()
    window.show()

    if len(sys.argv) > 1:
        window.open_session(Path(sys.argv[1]))

    sys.exit(app.exec())


if __name__ == "__main__":  # pragma: no cover - exercised by the console script
    main()
