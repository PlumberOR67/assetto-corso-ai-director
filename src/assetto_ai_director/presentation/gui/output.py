"""Writing a run's files, and never losing a result to a failed write.

Two things this module exists to stop.

**A write that raises on the GUI thread.** ``write_timeline`` has no ``try``, and
the old window called it bare — a full disk or a file open in another program
produced an unhandled exception in the Qt event loop. Worse, the timeline it
failed to write was still in memory and perfectly good, and the user lost the
window instead of losing the file. Every write here returns an
:class:`~assetto_ai_director.presentation.gui.artifacts.Artifact` describing what
happened, including the failures.

**A result the user cannot find.** The destination is deterministic and stated
before the run: ``<root>/<session name>/<style>/``. There is no save dialog in the
normal path, because "where did it go" should not be a question that has a
different answer each time.

**Lossy exports are written rather than refused**, with the sink's own limitations
attached to the row. The CLI refuses without ``--force`` and is right to — a
terminal prints once and scrolls away. A panel keeps the caveat beside the file
permanently, so refusing would only add a click before the same outcome.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from assetto_ai_director.infrastructure.export.cameras_ini import CamerasIniSink
from assetto_ai_director.infrastructure.export.diagnostic_json import write_diagnostic
from assetto_ai_director.infrastructure.export.json_timeline import JsonTimelineSink
from assetto_ai_director.presentation.gui.artifacts import Artifact

if TYPE_CHECKING:
    from pathlib import Path

    from assetto_ai_director.application.ports.timeline_sink import TimelineSink
    from assetto_ai_director.domain.diagnostics import DiagnosticReport
    from assetto_ai_director.domain.timeline import Timeline
    from assetto_ai_director.domain.track import TrackSpline

__all__ = ["ArtifactWriter", "run_folder", "safe_name"]

_CAMERA_SET_INDEX = 9
"""Which camera set to write.

High on purpose: sets are ``cameras.ini``, ``cameras_1.ini`` and upward, and a
high index avoids colliding with whatever the track already ships.
"""


def safe_name(text: str) -> str:
    """Turn arbitrary text into something usable as a folder name.

    Args:
        text: The name to sanitise.

    Returns:
        The name with path separators and characters Windows forbids replaced.
        Empty input becomes ``session`` rather than an empty path component,
        which would silently move the output one level up.
    """
    cleaned = "".join("_" if character in '<>:"/\\|?*' else character for character in text)
    cleaned = cleaned.strip(" .")
    return cleaned or "session"


def run_folder(root: Path, session: Path | None, style: str) -> Path:
    """Work out where one run's files belong.

    Args:
        root: The configured output root.
        session: The session being directed, or ``None``.
        style: The style being directed in.

    Returns:
        ``<root>/<session>/<style>``. Deterministic, so the answer to "where does
        it save" is the same every time and can be shown before the run.
    """
    if session is None:
        return root
    return root / safe_name(session.stem) / safe_name(style)


class ArtifactWriter:
    """Writes a run's outputs, reporting rather than raising."""

    def __init__(self, folder: Path) -> None:
        """Target a folder.

        Args:
            folder: Where the files go. Created on first write, not here — a
                window that made directories merely by having settings open
                would litter the disk with folders for runs that never happened.
        """
        self.folder = folder

    def write_all(self, timeline: Timeline, track: TrackSpline | None = None) -> list[Artifact]:
        """Write every output this run produces.

        Args:
            timeline: The finished edit.
            track: The track's centreline, when one was found. Passed through to
                the camera set, where it decides whether the file switches
                cameras where they stand or at a fraction of the elapsed time.

        Returns:
            One artifact per file, successes and failures alike, in the order
            they should be shown.
        """
        return [
            self._write(JsonTimelineSink(), timeline, self.folder / "timeline.json", "timeline"),
            self._write(
                CamerasIniSink(track),
                timeline,
                self.folder / f"cameras_{_CAMERA_SET_INDEX}.ini",
                "cameras-ini",
            ),
        ]

    def write_diagnostic_report(self, report: DiagnosticReport) -> Artifact:
        """Write the camera diagnosis beside the run's other files.

        Separate from :meth:`write_all` because it needs the session's car
        positions, which a timeline does not carry, so the window supplies it
        once the checks have run rather than this class reaching for it.

        Args:
            report: The diagnosis.

        Returns:
            The artifact, carrying an error rather than raising one.
        """
        destination = self.folder / "diagnostic.json"
        try:
            write_diagnostic(report, destination)
            size = destination.stat().st_size
        except OSError as error:
            return Artifact(path=destination, kind="diagnostic", error=str(error))
        return Artifact(path=destination, kind="diagnostic", bytes=size)

    def write_timeline_to(self, timeline: Timeline, destination: Path) -> Artifact:
        """Write the timeline to a location the user chose.

        Args:
            timeline: The finished edit.
            destination: Where to write it.

        Returns:
            The artifact, carrying an error rather than raising one.
        """
        return self._write(JsonTimelineSink(), timeline, destination, "timeline")

    def _write(
        self, sink: TimelineSink, timeline: Timeline, destination: Path, kind: str
    ) -> Artifact:
        """Write one file through a sink, capturing what it could not represent.

        Args:
            sink: The exporter.
            timeline: The edit to write.
            destination: Where it goes.
            kind: Which explanation applies to the result.

        Returns:
            The artifact.
        """
        report = sink.supports(timeline)
        if not report.supported:
            return Artifact(
                path=destination,
                kind=kind,
                limitations=report.limitations,
                error="this edit cannot be expressed in that format",
            )
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            sink.write(timeline, destination)
            size = destination.stat().st_size
        except OSError as error:
            # A failed export must never cost the user the finished timeline,
            # which is still in memory and still correct.
            return Artifact(
                path=destination,
                kind=kind,
                limitations=report.limitations,
                error=str(error),
            )
        return Artifact(
            path=destination,
            kind=kind,
            bytes=size,
            limitations=report.limitations if report.lossy else (),
        )
