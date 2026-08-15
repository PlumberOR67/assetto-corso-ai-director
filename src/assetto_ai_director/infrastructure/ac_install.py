r"""Finding an Assetto Corsa installation, so nobody has to type a path.

The whole point of this module is that the person verifying a camera set is not
the person who wrote it. They open the application, press one button and send
back a report. Every path in that flow has to be discovered rather than asked
for, because each question is somewhere the hand-off can fail silently.

**Every candidate is validated before it is offered.** A discovery source
proposes a directory; the directory only becomes a result if it actually
contains ``content/tracks``. That makes a wrong guess harmless — it yields no
candidate rather than a plausible-looking path that later reads as an empty
track folder. It is the same self-verification principle the replay reader uses,
applied to the filesystem.

What was established, per source:

* **Environment variable** — this project's own, documented here and nowhere
  else. An escape hatch for installs the other sources cannot see.
* **Steam library folders** — ``steamapps/libraryfolders.vdf`` lists every
  library on the machine. Assetto Corsa installs to
  ``<library>/steamapps/common/assettocorsa``. The file is Valve's KeyValues
  text format; this module extracts quoted path values rather than implementing
  a parser for a format whose full grammar it does not need.
* **Windows registry** — ``HKCU\\Software\\Valve\\Steam`` carries ``SteamPath``,
  which locates the *default* library. Used only to seed the library search
  above, never as an Assetto Corsa path in itself.

What was **not** established, and therefore is not implemented: Content
Manager's own record of the install. CM stores its settings under
``%APPDATA%/AcTools Content Manager``, but the file that holds the configured
Assetto Corsa root, and its encoding, could not be confirmed. Guessing at it
would mean shipping a reader for a format this project has not seen — exactly
what the honesty rule forbids. A user whose install is invisible to Steam sets
:data:`ENV_VAR` instead, and the report says which source found the root so the
gap is visible rather than mysterious.
"""

from __future__ import annotations

import configparser
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

if sys.platform == "win32":  # pragma: no cover - exercised only on Windows
    import winreg

__all__ = [
    "ENV_VAR",
    "AcInstall",
    "CspManifest",
    "EnvironmentSnapshot",
    "TrackFiles",
    "discover_ac_root",
    "environment_snapshot",
    "find_track_data",
]

ENV_VAR = "ASSETTO_CORSA_ROOT"
"""Environment variable that overrides discovery. Checked first."""

_CONTENT_MARKER = Path("content") / "tracks"
"""What makes a directory an Assetto Corsa root, for this project's purposes.

Deliberately not the executable. A user may point at a copied content tree with
no game beside it, and everything this project reads lives under ``content``.
"""

_EXECUTABLES = ("acs.exe", "AssettoCorsa.exe")
"""Names that confirm a root is a real install rather than a content copy."""

_CSP_MANIFEST = Path("extension") / "config" / "data_manifest.ini"
"""Custom Shaders Patch writes a manifest here.

Its ``[VERSION]`` keys are reported verbatim rather than interpreted: which key
carries the patch version was not confirmed against a real install, and naming
one would be a guess dressed as a fact.
"""

_TEXT_ENCODING = "utf-8-sig"
"""How every configuration file here is decoded.

``utf-8-sig`` rather than ``utf-8``, because Windows tools routinely write a
byte-order mark and ``utf-8`` leaves it in the first key, which makes a section
header unparseable. This was not theoretical: the first manual run of
``aidirector doctor`` against an install whose manifest carried a BOM reported
Custom Shaders Patch as *not installed*. ``utf-8-sig`` strips a BOM when present
and behaves exactly like ``utf-8`` when it is not.
"""

_VDF_PATH_VALUE = re.compile(r'"(?:path|\d+)"\s*"([^"]+)"', re.IGNORECASE)
"""Matches a library path in ``libraryfolders.vdf``.

Two shapes exist in the wild: modern Steam writes ``"path" "D:\\\\SteamLibrary"``
inside a numbered block, older Steam wrote ``"1" "D:\\\\SteamLibrary"`` directly.
Both are covered; anything else simply yields no match and costs nothing.
"""

_STEAM_SUBPATH = Path("steamapps") / "common" / "assettocorsa"
_LIBRARY_INDEX = Path("steamapps") / "libraryfolders.vdf"

_DEFAULT_STEAM_DIRS = (
    Path("C:/Program Files (x86)/Steam"),
    Path("C:/Program Files/Steam"),
)
"""Where Steam lands without the registry. Tried only if the registry is silent."""


@dataclass(frozen=True, slots=True)
class AcInstall:
    """A validated Assetto Corsa root.

    Attributes:
        root: The install directory. Contains ``content/tracks``.
        found_by: Which discovery source proposed it, for the report. A user
            whose install was found by the environment variable rather than by
            Steam has a non-standard setup, and that is worth knowing before
            anything else is diagnosed.
        has_executable: Whether a game executable sits beside the content. False
            means a content copy, which is enough to read track data but not to
            run anything.
    """

    root: Path
    found_by: str
    has_executable: bool


@dataclass(frozen=True, slots=True)
class CspManifest:
    """Custom Shaders Patch's version manifest, as found on disk.

    Separates "not installed" from "installed but the manifest could not be
    read", because collapsing those two into one absent value produces a
    confident false negative — the report says CSP is missing when it is sitting
    right there, and the reader believes it.

    Attributes:
        path: Where the manifest was found.
        values: The ``[VERSION]`` section as plain key-value pairs, empty when
            the file carries no such section, or ``None`` when the file exists
            but could not be parsed.
    """

    path: Path
    values: dict[str, str] | None

    @property
    def readable(self) -> bool:
        """Whether the manifest parsed.

        Returns:
            ``True`` when the file yielded key-value pairs, or yielded nothing
            because it genuinely has no ``[VERSION]`` section.
        """
        return self.values is not None


@dataclass(frozen=True, slots=True)
class TrackFiles:
    """Which of a track layout's data files are actually present.

    Absence is the normal case for several of these — plenty of tracks ship no
    ``map.ini``, and a layout with no AI line is a layout the game cannot run
    AI on. So every field is optional and the report says which were missing
    rather than treating a gap as a failure.

    Attributes:
        track_id: The track folder name, as the replay header records it.
        layout: The layout name, empty for a single-layout track.
        layout_dir: The directory the data was looked for in.
        fast_lane: ``data/fast_lane.ai`` — the AI line, and the only source of a
            true normalised track position.
        pit_lane: ``data/pit_lane.ai``.
        camera_sets: Every ``data/cameras*.ini``, sorted. Reference material for
            checking this project's own exporter against real files.
        map_image: ``map.png``.
        map_ini: ``map.ini`` — the world-to-map transform.
        surfaces: ``data/surfaces.ini``.
    """

    track_id: str
    layout: str
    layout_dir: Path
    fast_lane: Path | None
    pit_lane: Path | None
    camera_sets: tuple[Path, ...]
    map_image: Path | None
    map_ini: Path | None
    surfaces: Path | None

    def sizes(self) -> dict[str, int]:
        """Report the size of every file that was found.

        Sizes travel in the report because they distinguish "the file is there"
        from "the file is there and has something in it", and a zero-byte
        ``fast_lane.ai`` is a real thing that happens to modded tracks.

        Returns:
            File name to size in bytes, for present files only.
        """
        found: dict[str, int] = {}
        singles = (
            ("fast_lane.ai", self.fast_lane),
            ("pit_lane.ai", self.pit_lane),
            ("map.png", self.map_image),
            ("map.ini", self.map_ini),
            ("surfaces.ini", self.surfaces),
        )
        for name, path in singles:
            if path is not None:
                found[name] = _size_of(path)
        for path in self.camera_sets:
            found[path.name] = _size_of(path)
        return found

    @property
    def missing(self) -> tuple[str, ...]:
        """Name the files that were looked for and not found.

        Returns:
            The missing file names, in a fixed order so two reports diff cleanly.
        """
        absent: list[str] = []
        if self.fast_lane is None:
            absent.append("data/fast_lane.ai")
        if self.map_ini is None:
            absent.append("map.ini")
        if self.map_image is None:
            absent.append("map.png")
        if not self.camera_sets:
            absent.append("data/cameras*.ini")
        if self.surfaces is None:
            absent.append("data/surfaces.ini")
        return tuple(absent)


@dataclass(frozen=True, slots=True)
class EnvironmentSnapshot:
    """Everything about the machine that a camera report needs to carry.

    This is embedded in every diagnostic report. Without it, a report showing
    every track-dependent check as unknown is indistinguishable from a report
    produced on a machine where the track data was simply never found — and
    those two need completely different answers.

    Attributes:
        install: The chosen Assetto Corsa root, or ``None``.
        candidates: Every validated root that was found, in the order the
            sources were tried. More than one is not an error; it is worth
            seeing, because the wrong one may have been picked.
        csp: Custom Shaders Patch's manifest, or ``None`` when CSP is not
            installed. Keys are reported rather than interpreted.
        track: The track data for the session being diagnosed, when a track was
            named and an install was found.
    """

    install: AcInstall | None
    candidates: tuple[AcInstall, ...]
    csp: CspManifest | None
    track: TrackFiles | None

    @property
    def blind_spots(self) -> tuple[str, ...]:
        """Say plainly what could not be established, and why it matters.

        Returns:
            One sentence per gap, written for the person reading the report
            rather than for the person who wrote this module.
        """
        if self.install is None:
            absent = (
                "No Assetto Corsa installation was found, so no track data could "
                f"be read. Set the {ENV_VAR} environment variable to point at it."
            )
            return (absent,)

        gaps: list[str] = []
        if self.csp is not None and not self.csp.readable:
            gaps.append(
                f"Custom Shaders Patch is installed but its manifest at {self.csp.path} "
                "could not be parsed, so its version is unknown."
            )
        if self.track is None:
            gaps.append("No track was named for this session, so its data was not looked up.")
            return tuple(gaps)
        if self.track.missing:
            gaps.append(f"Missing from {self.track.layout_dir}: {', '.join(self.track.missing)}.")
        return tuple(gaps)


def discover_ac_root(explicit: Path | None = None) -> tuple[AcInstall, ...]:
    """Find every Assetto Corsa installation on this machine.

    Sources are tried in order of how much the user meant them: an explicit
    setting, then the environment variable, then Steam. Each candidate is
    validated before it is returned, so an unreadable registry or a stale
    library entry produces nothing rather than a broken path.

    Args:
        explicit: A root the user configured by hand, which wins over discovery.

    Returns:
        Validated installs, best first, without duplicates. Empty when Assetto
        Corsa is not installed — which is a normal state, not an error.
    """
    proposals: list[tuple[Path, str]] = []
    if explicit is not None:
        proposals.append((explicit, "configured by hand"))

    from_env = os.environ.get(ENV_VAR)
    if from_env:
        proposals.append((Path(from_env), f"{ENV_VAR} environment variable"))

    proposals.extend((path, "Steam library") for path in _steam_candidates())

    found: list[AcInstall] = []
    seen: set[Path] = set()
    for path, source in proposals:
        install = _validate(path, source)
        if install is None or install.root in seen:
            continue
        seen.add(install.root)
        found.append(install)
    return tuple(found)


def find_track_data(root: Path, track_id: str, layout: str = "") -> TrackFiles | None:
    """Locate a track layout's data files inside an installation.

    The arguments come straight from the replay header, which already records
    ``track_id`` and ``track_config``, so nothing here needs to be typed by a
    user.

    **The data directory is layout-scoped.** A multi-layout track keeps its
    files under ``content/tracks/<track>/<layout>/``, and looking in the track
    root instead finds a directory that exists but holds another layout's data.
    The map, by contrast, may sit in either place, so it is searched in the
    layout first and then the track root.

    Args:
        root: An Assetto Corsa root, as returned by :func:`discover_ac_root`.
        track_id: The track folder name.
        layout: The layout name, empty for a single-layout track.

    Returns:
        What was found, or ``None`` when the track folder does not exist at all
        — which usually means the replay was recorded with a track that is no
        longer installed.
    """
    track_root = root / "content" / "tracks" / track_id
    if not track_root.is_dir():
        return None

    layout_dir = track_root / layout if layout else track_root
    if not layout_dir.is_dir():
        # The replay names a layout the install does not have. Reporting the
        # track root would quietly diagnose a different layout's cameras.
        return None

    data = layout_dir / "data"
    return TrackFiles(
        track_id=track_id,
        layout=layout,
        layout_dir=layout_dir,
        fast_lane=_file_or_none(data / "fast_lane.ai"),
        pit_lane=_file_or_none(data / "pit_lane.ai"),
        camera_sets=_camera_sets(data),
        map_image=_first_existing(layout_dir / "map.png", track_root / "map.png"),
        map_ini=_first_existing(layout_dir / "map.ini", track_root / "map.ini"),
        surfaces=_file_or_none(data / "surfaces.ini"),
    )


def environment_snapshot(
    track_id: str = "",
    layout: str = "",
    *,
    explicit_root: Path | None = None,
) -> EnvironmentSnapshot:
    """Describe this machine, for embedding in a diagnostic report.

    Called automatically whenever a report is produced. It is not a step the
    user performs — a diagnostic flow with a prerequisite command is a flow
    where the prerequisite gets skipped and the report arrives unexplained.

    Args:
        track_id: The session's track, from the replay header. Empty when
            unknown, in which case no track lookup happens.
        layout: The session's layout, from the replay header.
        explicit_root: A root the user configured by hand.

    Returns:
        The snapshot. Never raises: a machine with no Assetto Corsa produces a
        snapshot saying exactly that, which is what continuous integration sees.
    """
    candidates = discover_ac_root(explicit_root)
    install = candidates[0] if candidates else None
    csp = _csp_manifest(install.root) if install is not None else None
    track = (
        find_track_data(install.root, track_id, layout)
        if install is not None and track_id
        else None
    )
    return EnvironmentSnapshot(install=install, candidates=candidates, csp=csp, track=track)


# --- Discovery sources ------------------------------------------------------


def _steam_candidates() -> list[Path]:
    """Propose an Assetto Corsa directory for every Steam library found.

    Returns:
        Candidate roots, unvalidated. Order follows the library index, which
        puts the default library first.
    """
    return [library / _STEAM_SUBPATH for library in _steam_libraries()]


def _steam_libraries() -> list[Path]:
    """Find every Steam library folder on this machine.

    Returns:
        Library roots, without duplicates. The Steam install itself is a library
        and is always included when found.
    """
    roots: list[Path] = []
    registry = _steam_path_from_registry()
    if registry is not None:
        roots.append(registry)
    roots.extend(_DEFAULT_STEAM_DIRS)

    libraries: list[Path] = []
    seen: set[Path] = set()
    for steam in roots:
        if not steam.is_dir():
            continue
        for library in [steam, *_libraries_from_index(steam / _LIBRARY_INDEX)]:
            resolved = _normalise(library)
            if resolved in seen:
                continue
            seen.add(resolved)
            libraries.append(library)
    return libraries


def _libraries_from_index(index: Path) -> list[Path]:
    """Read library paths out of ``libraryfolders.vdf``.

    Args:
        index: Path to the index file.

    Returns:
        The library roots it lists, or an empty list when the file is absent or
        unreadable. An unreadable index is not worth an exception: it costs one
        discovery source, and the others still run.
    """
    try:
        text = index.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return [Path(match.replace("\\\\", "\\")) for match in _VDF_PATH_VALUE.findall(text)]


def _steam_path_from_registry() -> Path | None:
    r"""Read Steam's own install path from the Windows registry.

    Used only to seed the library search. Steam records ``SteamPath`` under
    ``HKCU\Software\Valve\Steam``; nothing there names Assetto Corsa, so no
    Assetto-specific registry key is invented here.

    Returns:
        Steam's directory, or ``None`` off Windows or when the key is absent.
    """
    if sys.platform != "win32":
        return None

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam") as key:
            value, kind = winreg.QueryValueEx(key, "SteamPath")
    except OSError:
        return None
    if kind != winreg.REG_SZ or not isinstance(value, str) or not value:
        return None
    return Path(value)


# --- Validation and small helpers -------------------------------------------


def _validate(path: Path, source: str) -> AcInstall | None:
    """Accept a proposed root only if it really holds Assetto Corsa content.

    Args:
        path: The proposed directory.
        source: Which discovery source proposed it.

    Returns:
        The validated install, or ``None``. A path that does not exist, is not a
        directory, or lacks ``content/tracks`` is rejected silently — discovery
        proposing something wrong is expected, not exceptional.
    """
    try:
        if not (path / _CONTENT_MARKER).is_dir():
            return None
        root = _normalise(path)
        has_exe = any((path / name).is_file() for name in _EXECUTABLES)
    except OSError:
        return None
    return AcInstall(root=root, found_by=source, has_executable=has_exe)


def _csp_manifest(root: Path) -> CspManifest | None:
    """Read Custom Shaders Patch's version manifest, verbatim.

    Args:
        root: The Assetto Corsa root.

    Returns:
        ``None`` when CSP is not installed, otherwise a manifest whose values
        are the ``[VERSION]`` section as plain key-value pairs — or ``None``
        values when the file is there but will not parse. Keys are passed
        through untouched: which one carries the patch version was not confirmed
        against a real install, so naming one would be a guess.
    """
    manifest = root / _CSP_MANIFEST
    if not manifest.is_file():
        return None
    parser = configparser.ConfigParser(strict=False)
    try:
        parser.read_string(manifest.read_text(encoding=_TEXT_ENCODING, errors="replace"))
    except (OSError, configparser.Error):
        return CspManifest(path=manifest, values=None)
    values = dict(parser["VERSION"]) if parser.has_section("VERSION") else {}
    return CspManifest(path=manifest, values=values)


def _camera_sets(data: Path) -> tuple[Path, ...]:
    """List a layout's camera set files.

    Args:
        data: The layout's ``data`` directory.

    Returns:
        Every ``cameras*.ini``, sorted by name, or empty when the directory does
        not exist.
    """
    if not data.is_dir():
        return ()
    return tuple(sorted(data.glob("cameras*.ini")))


def _file_or_none(path: Path) -> Path | None:
    """Return a path only if it is an existing file.

    Args:
        path: The candidate.

    Returns:
        The path, or ``None``.
    """
    return path if path.is_file() else None


def _first_existing(*candidates: Path) -> Path | None:
    """Return the first candidate that is an existing file.

    Args:
        *candidates: Paths to try, in order of preference.

    Returns:
        The first existing file, or ``None``.
    """
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _size_of(path: Path) -> int:
    """Report a file's size, treating an unreadable file as empty.

    Args:
        path: The file.

    Returns:
        Size in bytes, or ``0`` when it cannot be read. A report is not worth
        failing over a file that vanished between listing and measuring.
    """
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _normalise(path: Path) -> Path:
    """Resolve a path for comparison, tolerating one that does not exist.

    Args:
        path: The path to normalise.

    Returns:
        The resolved path, or the original when resolution fails.
    """
    try:
        return path.resolve()
    except OSError:
        return path
