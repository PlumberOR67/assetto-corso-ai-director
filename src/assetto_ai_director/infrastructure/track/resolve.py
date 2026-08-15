"""Getting the best centreline available, and saying which one that was.

Three outcomes, in order of preference, and the caller is told which it got
because the checks behave differently for each:

1. The track's own ``fast_lane.ai``. Carries the road's width, so a camera can
   be told it is standing in a ditch.
2. A line averaged from the session's own driving. Always available, never
   exact, and knows nothing about width.
3. Nothing, when the session is too short to describe a line at all.

**A file that fails to parse degrades to the driven line rather than failing the
run.** The AI-line reader refuses on a layout mismatch, and that refusal is a
finding about the file, not a reason a camera report cannot be produced. The
reason travels back so it reaches the report's blind spots instead of vanishing
into a log nobody reads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from assetto_ai_director.application.camera.spline import driven_spline
from assetto_ai_director.domain.errors import AssettoDirectorError
from assetto_ai_director.infrastructure.ac_install import discover_ac_root, find_track_data
from assetto_ai_director.infrastructure.track.fast_lane import read_fast_lane

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

    from assetto_ai_director.domain.track import TrackSpline
    from assetto_ai_director.infrastructure.ac_install import TrackFiles

__all__ = ["ResolvedSpline", "resolve_spline"]


@dataclass(frozen=True, slots=True)
class ResolvedSpline:
    """A centreline and the story of where it came from.

    Attributes:
        spline: The centreline, or ``None`` when none could be had.
        note: What happened, in a sentence, or empty when the best outcome was
            reached with nothing worth saying. Written for the report's blind
            spots rather than for a log: the person reading it is trying to work
            out why a check said unknown.
    """

    spline: TrackSpline | None
    note: str = ""


def resolve_spline(
    track_id: str,
    layout: str,
    position: np.ndarray | None,
    *,
    ac_root: Path | None = None,
) -> ResolvedSpline:
    """Find the best centreline for a session.

    Args:
        track_id: The track, from the replay header.
        layout: The layout, from the replay header.
        position: One car's world positions, used to derive a line when the
            track's own is unavailable. ``None`` skips the fallback.
        ac_root: An Assetto Corsa directory the user configured by hand.

    Returns:
        The centreline and why it is the one that was chosen.
    """
    files = _track_files(track_id, layout, ac_root)
    if files is not None and files.fast_lane is not None:
        try:
            return ResolvedSpline(read_fast_lane(files.fast_lane))
        except (AssettoDirectorError, ValueError, OSError) as error:
            return _derived(
                position,
                f"{files.fast_lane} could not be read ({error}), so the centreline "
                f"was derived from the session's own driving instead.",
            )

    if files is None:
        return _derived(
            position,
            f"No installed copy of {track_id}/{layout or 'default'} was found, so the "
            f"centreline was derived from the session's own driving.",
        )
    return _derived(
        position,
        f"{files.layout_dir} has no fast_lane.ai, so the centreline was derived "
        f"from the session's own driving.",
    )


def _track_files(track_id: str, layout: str, ac_root: Path | None) -> TrackFiles | None:
    """Locate a track's data across every installation found.

    More than one installation is unusual but not an error, and the track being
    diagnosed may be modded into only one of them. Trying each is cheaper than
    asking the user which one they meant.

    Args:
        track_id: The track.
        layout: The layout.
        ac_root: A configured Assetto Corsa directory.

    Returns:
        The track files, or ``None`` when no installation has this layout.
    """
    if not track_id:
        return None
    for install in discover_ac_root(ac_root):
        found = find_track_data(install.root, track_id, layout)
        if found is not None:
            return found
    return None


def _derived(position: np.ndarray | None, note: str) -> ResolvedSpline:
    """Fall back to a line the cars drove.

    Args:
        position: One car's world positions, or ``None``.
        note: Why the fallback was reached.

    Returns:
        The derived centreline and the note, or nothing at all with the reason
        extended to say the fallback failed too.
    """
    if position is None:
        return ResolvedSpline(None, note)
    spline = driven_spline(position)
    if spline is None:
        return ResolvedSpline(
            None, f"{note} That failed as well: the session covers too little ground."
        )
    return ResolvedSpline(spline, note)
