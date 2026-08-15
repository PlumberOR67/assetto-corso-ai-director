"""The ``aidirector install`` command: put a camera set where the game reads it.

``suggested_path`` has been able to work out the right destination since the
exporter was written, and nothing ever called it. The user was left to build
``content/tracks/<track>/<layout>/data/cameras_9.ini`` by hand, and the
documentation had to warn that putting it in the track root instead produces a
file the game silently never loads — which is the shape of a step that should not
have been manual.

**The default is a dry run.** Writing into a game installation is the one thing
this project does that is not confined to its own output folder, and a command
that overwrites a track's data because somebody mistyped a name is not a command
worth having. It prints what it would do; ``--write`` makes it do it.

**An existing file is copied aside, never replaced silently.** A camera set the
user hand-authored and a camera set this project generated look identical from
the command line.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console

from assetto_ai_director.infrastructure.ac_install import ENV_VAR, discover_ac_root
from assetto_ai_director.infrastructure.timeline_io import read_timeline
from assetto_ai_director.infrastructure.track.cameras_ini_read import read_cameras_ini

if TYPE_CHECKING:
    from assetto_ai_director.infrastructure.ac_install import AcInstall

__all__ = ["install"]

console = Console()

_BACKUP_SUFFIX = ".aidirector-backup"


def install(
    cameras: Annotated[
        Path,
        typer.Argument(help="The cameras_N.ini to install.", exists=True, dir_okay=False),
    ],
    track: Annotated[
        str, typer.Option("--track", help="Track folder name. Read from --timeline if omitted.")
    ] = "",
    layout: Annotated[
        str, typer.Option("--layout", help="Layout name, for a multi-layout track.")
    ] = "",
    *,
    timeline_path: Annotated[
        Path | None,
        typer.Option("--timeline", help="Timeline the set came from, to name the track."),
    ] = None,
    ac_root: Annotated[
        Path | None,
        typer.Option("--ac-root", help="Assetto Corsa directory, when discovery cannot find it."),
    ] = None,
    set_index: Annotated[
        int, typer.Option("--set-index", help="Which camera set number to write.")
    ] = 9,
    write: Annotated[
        bool, typer.Option("--write", help="Actually write. Without it, only report.")
    ] = False,
) -> None:
    """Copy a generated camera set into an Assetto Corsa track."""
    track_id, layout_name = _identify(track, layout, timeline_path)
    if not track_id:
        console.print(
            "[red]No track named.[/] Pass --track, or --timeline pointing at the "
            "timeline this set was exported from."
        )
        raise typer.Exit(code=1)

    install_root = _installation(ac_root)
    destination = _destination(install_root, track_id, layout_name, set_index)

    console.print(f"[bold]Camera set[/] {cameras}")
    _describe(cameras)
    console.print(f"[bold]Destination[/] {destination}")

    if not destination.parent.is_dir():
        console.print(
            f"\n[red]{destination.parent} does not exist.[/]\n"
            "  [dim]A multi-layout track keeps its cameras under the layout, not the "
            "track root. Check --layout against the folders the track actually has.[/]"
        )
        raise typer.Exit(code=1)

    if not write:
        console.print("\n  [dim]Dry run. Pass --write to install.[/]")
        return

    _install(cameras, destination)
    console.print(
        f"\n[green]Installed.[/] Choose camera set {set_index} in game — "
        f"press F3 until it comes round, or pick it in Content Manager."
    )


def _identify(track: str, layout: str, timeline_path: Path | None) -> tuple[str, str]:
    """Work out which track this set belongs to.

    Args:
        track: An explicitly named track.
        layout: An explicitly named layout.
        timeline_path: A timeline to read the names from instead.

    Returns:
        ``(track_id, layout)``, either of which may be empty.
    """
    if track:
        return track, layout
    if timeline_path is None:
        return "", layout
    try:
        timeline = read_timeline(timeline_path)
    except (ValueError, OSError) as error:
        console.print(f"[yellow]Could not read {timeline_path}: {error}[/]")
        return "", layout
    return timeline.session.track_id, layout or timeline.session.layout


def _installation(ac_root: Path | None) -> AcInstall:
    """Find the installation to write into.

    Args:
        ac_root: A directory the user named.

    Returns:
        The installation.

    Raises:
        typer.Exit: If none was found, or more than one was and none was chosen.
            Picking one silently is how a camera set ends up in a copy of the
            game the user is not running.
    """
    found = discover_ac_root(ac_root)
    if not found:
        console.print(
            f"[red]No Assetto Corsa installation found.[/] Pass --ac-root, or set {ENV_VAR}."
        )
        raise typer.Exit(code=1)
    if len(found) > 1 and ac_root is None:
        console.print("[red]More than one installation found.[/] Pass --ac-root to choose:")
        for candidate in found:
            console.print(f"  [dim]{candidate.root}  ({candidate.found_by})[/]")
        raise typer.Exit(code=1)
    return found[0]


def _destination(install_root: AcInstall, track: str, layout: str, set_index: int) -> Path:
    """Build the layout-scoped path the game reads.

    Args:
        install_root: The installation.
        track: The track folder name.
        layout: The layout name, empty for a single-layout track.
        set_index: Which camera set number.

    Returns:
        Where the file goes.
    """
    base = install_root.root / "content" / "tracks" / track
    data = base / layout / "data" if layout else base / "data"
    name = "cameras.ini" if set_index == 0 else f"cameras_{set_index}.ini"
    return data / name


def _describe(cameras: Path) -> None:
    """Say what is in the file being installed.

    Reading it back through this project's own parser before writing it into a
    game is worth the two lines: a set with no cameras, or one whose spans do not
    meet, is a set the game will behave oddly with, and finding that out now is
    better than finding it out in the replay.

    Args:
        cameras: The file.
    """
    try:
        parsed = read_cameras_ini(cameras)
    except (ValueError, OSError) as error:
        console.print(f"  [yellow]Could not parse it back: {error}[/]")
        return
    console.print(
        f"  [dim]{len(parsed.cameras)} cameras, "
        f"{'contiguous' if parsed.spans_are_contiguous else 'with gaps between'} spans"
        f"{'' if parsed.count_agrees else ', header count disagrees with the body'}[/]"
    )


def _install(cameras: Path, destination: Path) -> None:
    """Copy the set in, keeping whatever was there.

    Args:
        cameras: The file to install.
        destination: Where it goes.

    Raises:
        typer.Exit: If the copy fails.
    """
    try:
        if destination.exists():
            backup = destination.with_suffix(destination.suffix + _BACKUP_SUFFIX)
            shutil.copy2(destination, backup)
            console.print(f"  [dim]Existing set kept as {backup.name}[/]")
        shutil.copy2(cameras, destination)
    except OSError as error:
        console.print(f"[red]Could not write {destination}: {error}[/]")
        raise typer.Exit(code=1) from error
