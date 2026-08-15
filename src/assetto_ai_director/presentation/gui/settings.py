"""What the application remembers between runs.

The window used to remember nothing: not its size, not the folder you opened a
replay from, not where results went, not which style you had chosen. Every
session started from the same blank state, which for a tool whose input lives in
one folder and whose output belongs in another is a small tax paid over and over.

**Typed accessors rather than raw keys.** ``QSettings.value`` is typed as
returning ``object``, and the stubs are being honest: the Windows registry
backend really does hand back a string where a number was written, an INI file
collapses a one-element list into a bare string, and a settings file can be edited
by hand into anything at all. A misread setting is the kind of bug that appears
once, on someone else's machine, months later. Every key here is read through a
function that knows its type and its default and degrades to the default rather
than raising three call frames away.

**INI format, deliberately.** The Windows default is the registry, where settings
are invisible, hard to reset, and survive uninstalling the application. A file
under the user's config directory can be inspected, deleted, copied to another
machine, and — the practical one — redirected in tests, so a test run never
mutates the developer's own preferences.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from PySide6.QtCore import QSettings, QStandardPaths

__all__ = [
    "APPLICATION_NAME",
    "DEFAULT_SCOPE_SECONDS",
    "MAX_RECENT",
    "ORGANISATION_NAME",
    "Settings",
    "default_output_root",
    "default_replay_folder",
]

ORGANISATION_NAME: Final = "AssettoAIDirector"
APPLICATION_NAME: Final = "AssettoAIDirector"

DEFAULT_SCOPE_SECONDS: Final = 180.0
"""How much of a session to direct until the user says otherwise.

Three minutes rather than everything. A first run on a twenty-four minute replay
takes minutes and produces an edit the user has no basis to judge yet; three
minutes is long enough to tell whether the style is right and short enough to try
another one.
"""

MAX_RECENT: Final = 8
"""How many recent sessions to remember. Long enough to cover a working session,
short enough that the menu stays scannable."""


def _as_text(value: object, fallback: str = "") -> str:
    """Coerce a stored value to text.

    ``QSettings.value`` is typed as returning ``object``, and that is not
    pedantry on the stubs' part: the Windows registry backend genuinely returns
    a string where an int was written, and a settings file edited by hand can
    contain anything at all. Every read goes through one of these so a corrupt
    or foreign value degrades to the default instead of surfacing as a
    ``TypeError`` three call frames away.

    Args:
        value: What the store returned.
        fallback: Used when the value is missing or unusable.

    Returns:
        The text.
    """
    return value if isinstance(value, str) else fallback


def _as_float(value: object, fallback: float) -> float:
    """Coerce a stored value to a float.

    Args:
        value: What the store returned.
        fallback: Used when the value is missing or unusable.

    Returns:
        The number.
    """
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return fallback


def _as_bool(value: object, fallback: bool) -> bool:
    """Coerce a stored value to a boolean.

    Args:
        value: What the store returned.
        fallback: Used when the value is missing or unusable.

    Returns:
        The flag. The string forms matter: an INI file stores ``true``, and a
        naive ``bool("false")`` is ``True``.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return fallback


def _as_bytes(value: object) -> bytes:
    """Coerce a stored value to bytes.

    Args:
        value: What the store returned.

    Returns:
        The blob, empty when there is none.
    """
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray | memoryview):
        return bytes(value)
    return b""


def _as_paths(value: object) -> list[str]:
    """Coerce a stored value to a list of path strings.

    A single-element list written to an INI file reads back as a bare string,
    which is a real behaviour of the format rather than a hypothetical.

    Args:
        value: What the store returned.

    Returns:
        The entries.
    """
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def default_output_root() -> Path:
    """Where results go until the user says otherwise.

    Documents rather than the install directory: a PyInstaller build lands in
    Program Files, which is not writable by the user who runs it. And not the
    working directory either — that is how a 22 MB timeline ended up committed
    to this repository.

    Returns:
        The default output root.
    """
    documents = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation)
    base = Path(documents) if documents else Path.home()
    return base / "Assetto AI Director"


def default_replay_folder() -> Path:
    """Where Assetto Corsa keeps replays, if that folder exists.

    Only a starting point for the file dialog. It is not validated beyond
    existence and nothing depends on it — a user with a non-standard install
    simply starts in their documents folder instead.

    Returns:
        The replay folder if present, otherwise the documents folder.
    """
    documents = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation)
    base = Path(documents) if documents else Path.home()
    replays = base / "Assetto Corsa" / "replay"
    return replays if replays.is_dir() else base


class Settings:
    """Typed access to everything the application remembers.

    Wraps one ``QSettings`` instance. Construct it once and pass it around rather
    than creating them ad hoc — each construction re-reads the backing file.
    """

    def __init__(self, backing: QSettings | None = None) -> None:
        """Open the settings store.

        Args:
            backing: An existing store, for tests. A default one is created when
                omitted, using whatever format and path have been configured
                process-wide.
        """
        self._store = backing or QSettings(ORGANISATION_NAME, APPLICATION_NAME)

    # --- Output ------------------------------------------------------------

    @property
    def output_root(self) -> Path:
        """Where results are written.

        Returns:
            The configured root, or the default when unset.
        """
        stored = _as_text(self._store.value("output/root"))
        return Path(stored) if stored else default_output_root()

    @output_root.setter
    def output_root(self, value: Path) -> None:
        self._store.setValue("output/root", str(value))

    @property
    def ac_root(self) -> Path | None:
        """Where Assetto Corsa is, when the user had to say.

        Normally unset, and normally should stay that way: the camera checks
        find the installation through Steam's library index and the registry, and
        a setting that has to be filled in is a setting the person verifying
        somebody else's cameras will not know to fill in. This exists for the
        install those sources cannot see — a portable copy, a content tree with
        no game beside it.

        Returns:
            The configured directory, or ``None`` to leave it to discovery.
        """
        stored = _as_text(self._store.value("assettoCorsa/root"))
        return Path(stored) if stored else None

    @ac_root.setter
    def ac_root(self, value: Path | None) -> None:
        if value is None:
            self._store.remove("assettoCorsa/root")
            return
        self._store.setValue("assettoCorsa/root", str(value))

    # --- Input -------------------------------------------------------------

    @property
    def last_input_folder(self) -> Path:
        """Where the file dialog should open.

        Returns:
            The last folder a session was opened from, or Assetto Corsa's replay
            folder on first run.
        """
        stored = _as_text(self._store.value("input/lastFolder"))
        if stored and Path(stored).is_dir():
            return Path(stored)
        return default_replay_folder()

    @last_input_folder.setter
    def last_input_folder(self, value: Path) -> None:
        self._store.setValue("input/lastFolder", str(value))

    @property
    def recent(self) -> tuple[Path, ...]:
        """Recently opened sessions, most recent first.

        Paths that no longer exist are filtered out on read rather than on write:
        a file on a disconnected drive should not be forgotten merely because the
        application happened to start while it was unplugged.

        Returns:
            The surviving recent paths.
        """
        stored = _as_paths(self._store.value("input/recent"))
        return tuple(Path(item) for item in stored if Path(item).is_file())

    def remember(self, path: Path) -> None:
        """Add a session to the recent list, most recent first.

        Args:
            path: The session that was opened.
        """
        stored = _as_paths(self._store.value("input/recent"))
        entries = [str(path), *(item for item in stored if item != str(path))]
        self._store.setValue("input/recent", entries[:MAX_RECENT])

    # --- Run controls ------------------------------------------------------

    @property
    def style(self) -> str:
        """The last style chosen.

        Returns:
            The style name, defaulting to ``initial_d``.
        """
        return _as_text(self._store.value("run/style"), "initial_d")

    @style.setter
    def style(self, value: str) -> None:
        self._store.setValue("run/style", value)

    @property
    def scope_seconds(self) -> float:
        """How much of the session to direct. Zero means all of it.

        Defaults to three minutes rather than everything. A first run on a
        twenty-four minute replay takes minutes and produces an edit the user has
        no basis to judge yet; three minutes is long enough to tell whether the
        style is right and short enough to try another.

        Returns:
            The span in seconds.
        """
        return _as_float(self._store.value("run/scopeSeconds"), DEFAULT_SCOPE_SECONDS)

    @scope_seconds.setter
    def scope_seconds(self, value: float) -> None:
        self._store.setValue("run/scopeSeconds", float(value))

    # --- Window ------------------------------------------------------------

    @property
    def first_run_done(self) -> bool:
        """Whether the welcome screen has been dismissed.

        Returns:
            ``True`` once the user has got past it.
        """
        return _as_bool(self._store.value("firstRun/completed"), False)

    @first_run_done.setter
    def first_run_done(self, value: bool) -> None:
        self._store.setValue("firstRun/completed", value)

    def geometry(self) -> bytes:
        """The saved window geometry.

        Returns:
            The blob ``QWidget.restoreGeometry`` expects, empty when unset.
        """
        return _as_bytes(self._store.value("window/geometry"))

    def set_geometry(self, value: bytes) -> None:
        """Store the window geometry.

        Args:
            value: The blob from ``QWidget.saveGeometry``.
        """
        self._store.setValue("window/geometry", value)

    def splitter(self, name: str) -> bytes:
        """A saved splitter layout.

        Args:
            name: Which splitter.

        Returns:
            The blob ``QSplitter.restoreState`` expects, empty when unset.
        """
        return _as_bytes(self._store.value(f"splitter/{name}"))

    def set_splitter(self, name: str, value: bytes) -> None:
        """Store a splitter layout.

        Args:
            name: Which splitter.
            value: The blob from ``QSplitter.saveState``.
        """
        self._store.setValue(f"splitter/{name}", value)

    # --- Timing estimates --------------------------------------------------

    def seconds_per_unit(self, stage: str, fallback: float) -> float:
        """How long one unit of a stage took, averaged over recent runs.

        Used to estimate a run's length before it starts. Seeded with a
        conservative built-in guess so the first estimate is not wild.

        Args:
            stage: The stage name.
            fallback: Estimate to use when nothing has been measured.

        Returns:
            Seconds per unit of work.
        """
        return _as_float(self._store.value(f"timings/{stage}"), fallback)

    def record_timing(self, stage: str, seconds_per_unit: float, weight: float = 0.3) -> None:
        """Fold a measurement into the running estimate for a stage.

        An exponential moving average rather than a plain mean: the machine's
        speed is a slowly moving property and the most recent runs describe it
        best, while a plain mean would let one run on a busy laptop skew the
        estimate for a long time.

        Args:
            stage: The stage name.
            seconds_per_unit: What this run measured.
            weight: How much to trust the new measurement.
        """
        if seconds_per_unit <= 0.0:
            return
        previous = _as_float(self._store.value(f"timings/{stage}"), 0.0)
        blended = (
            seconds_per_unit
            if previous <= 0.0
            else (1.0 - weight) * previous + weight * seconds_per_unit
        )
        self._store.setValue(f"timings/{stage}", blended)

    def sync(self) -> None:
        """Flush pending writes to disk."""
        self._store.sync()
