"""The whole director, end to end.

The only place the ports meet. Everything above this is a use case; everything
below is a detail. A caller hands in decoded signals and a style, and gets back a
finished timeline.

The pipeline also collects **honest notes** as it goes — degraded channels, cars
that never moved, event kinds this source cannot detect, a recording rate too
coarse for reliable event timing. They travel with the timeline into every export,
so a user is told what the analysis was blind to rather than left to assume it
saw everything.

**Progress is reported by each stage, not by this module.** An earlier version
announced the stages from here and passed a total only for the first one, so a
guard of ``total is not None`` silently dropped three of the four labels: a user
watching a 24-minute session saw "Detecting events" climb to 100% and then sit
there, unmoving and misnamed, through candidate generation, scheduling and camera
paths. Each stage now owns its own start/advance/finish, because the total —
cars that moved, cut points, candidates, selected shots — is only knowable inside
the function that computes it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from assetto_ai_director.application.camera.paths import SubjectTracks, build_camera_path
from assetto_ai_director.application.director.candidates import build_candidates
from assetto_ai_director.application.director.continuity import ContinuityRules
from assetto_ai_director.application.director.schedule import schedule
from assetto_ai_director.application.events.detect import detect_all
from assetto_ai_director.application.ports.progress import NullProgress, Stage
from assetto_ai_director.application.scoring.features import compute_features
from assetto_ai_director.application.scoring.moments import RISK_LIMITATION_NOTE, score_session
from assetto_ai_director.domain.errors import OperationCancelledError
from assetto_ai_director.domain.shot import Shot
from assetto_ai_director.domain.timeline import Cut, Timeline, TransitionKind
from assetto_ai_director.domain.units import Seconds

if TYPE_CHECKING:
    from collections.abc import Mapping

    from assetto_ai_director.application.events.detect import CarSignals, DetectionResult
    from assetto_ai_director.application.ports.cancellation import CancelToken
    from assetto_ai_director.application.ports.progress import ProgressReporter
    from assetto_ai_director.domain.scoring import ScoreVector
    from assetto_ai_director.domain.session import SessionMeta
    from assetto_ai_director.domain.style import StyleProfile
    from assetto_ai_director.domain.track import TrackSpline

__all__ = ["SessionAnalysis", "analyse", "direct"]

SLOW_MOTION_FACTOR = 0.45
"""Playback rate for a shot marked for slow motion."""

SLOW_MOTION_SCORE = 0.75
"""Score a shot must reach before it earns a speed ramp.

Slow motion is a strong statement. Applied to every good moment it stops meaning
anything, so it is reserved for the shots that clear a high bar.
"""


@dataclass(frozen=True, slots=True)
class SessionAnalysis:
    """Everything derived from a session that does not depend on the style.

    The split is not arbitrary and it is not a caching convenience — it follows
    the code. Style weights are applied in ``candidates.py``, *after* scoring, so
    events and score vectors are the same whichever style is chosen. Recomputing
    them on a style change is pure waste: on the fixture replays it means
    re-reading 233 MB from disk and re-running every detector to change one
    dropdown.

    Attributes:
        detection: Detected events and what could not be detected.
        scores: Per-frame score vectors per car id.
        notes: Caveats about the analysis itself, carried into the timeline.
    """

    detection: DetectionResult
    scores: dict[int, list[ScoreVector]]
    notes: tuple[str, ...] = ()


def analyse(
    signals: Mapping[int, CarSignals],
    time: np.ndarray,
    sample_rate: float,
    *,
    progress: ProgressReporter | None = None,
    cancel: CancelToken | None = None,
) -> SessionAnalysis:
    """Detect events and score every moment.

    The expensive, style-independent half of directing. Callers that will direct
    the same session in more than one style should call this once and pass the
    result to :func:`direct`.

    Args:
        signals: Decoded per-car signals.
        time: Time per frame, seconds from session start.
        sample_rate: Frames per second.
        progress: Optional reporter for the detection and scoring stages.
        cancel: Optional cancellation token.

    Returns:
        The analysis.

    Raises:
        OperationCancelledError: If cancellation is requested part-way through.
    """
    detection = detect_all(signals, time, sample_rate, progress=progress, cancel=cancel)
    features = compute_features(signals, time, sample_rate)
    scores = score_session(features, progress=progress, cancel=cancel)
    return SessionAnalysis(
        detection=detection,
        scores=scores,
        notes=(*detection.notes, RISK_LIMITATION_NOTE),
    )


def direct(
    signals: Mapping[int, CarSignals],
    session: SessionMeta,
    style: StyleProfile,
    time: np.ndarray,
    sample_rate: float,
    *,
    horizon: tuple[float, float] | None = None,
    rules: ContinuityRules | None = None,
    progress: ProgressReporter | None = None,
    cancel: CancelToken | None = None,
    analysis: SessionAnalysis | None = None,
    track: TrackSpline | None = None,
) -> Timeline:
    """Analyse a session and produce a directed timeline.

    Args:
        signals: Decoded per-car signals.
        session: Session metadata.
        style: The style to direct in.
        time: Time per frame, seconds from session start.
        sample_rate: Frames per second.
        horizon: ``(start, end)`` to cover. Defaults to the whole session.
        rules: Continuity strictness. Defaults apply when omitted.
        progress: Optional reporter for the long stages.
        cancel: Optional cancellation token. Polled inside every long loop, so a
            full session can be stopped without waiting it out.
        analysis: A previously computed :func:`analyse` result for this same
            session. Supplying it skips detection and scoring, which is the
            whole cost of changing style. Computed here when omitted, so every
            existing caller is unaffected.
        track: The road the cars are on, when it could be read. Trackside rigs
            use it to stay beside the road instead of tens of metres into
            whatever is next to it; every other rig ignores it.

    Returns:
        The finished timeline.

    Raises:
        NoCandidatesError: If no edit can cover the horizon.
        OperationCancelledError: If cancellation is requested part-way through.
            Nothing has been written at that point — this function is pure — so
            the caller's previous result, if any, is still valid.
    """
    span = horizon or (float(time[0]), float(time[-1]))

    if analysis is None:
        analysis = analyse(signals, time, sample_rate, progress=progress, cancel=cancel)
    notes: list[str] = list(analysis.notes)

    grid = build_candidates(
        list(analysis.detection.events),
        analysis.scores,
        time,
        style,
        span,
        progress=progress,
        cancel=cancel,
    )
    result = schedule(grid, style, span, rules, progress=progress, cancel=cancel)

    tracks = SubjectTracks.from_signals(signals, time, sample_rate, track)

    # A loop rather than the generator expression this used to be, because this
    # is the longest stage on a full session and it reported nothing at all: the
    # bar sat at 100% under the previous stage's name until the window filled in.
    reporter: ProgressReporter = progress if progress is not None else NullProgress()
    reporter.start(str(Stage.CAMERA), len(result.shots))
    shots_built: list[Shot] = []
    for index, candidate in enumerate(result.shots):
        if cancel is not None and cancel.cancelled():
            raise OperationCancelledError(str(Stage.CAMERA))
        shots_built.append(
            Shot(
                candidate=candidate,
                path=build_camera_path(candidate, tracks, style),
                index=index,
            )
        )
        reporter.advance()
    reporter.finish()

    shots = tuple(shots_built)
    cuts = _cuts_for(shots)

    if not style.is_measured:
        notes.append(
            f"Style {style.name!r} uses unmeasured pacing. Run `aidirector measure` on "
            f"reference footage to replace its guesses with measurements."
        )
    if not session.quality.supports_event_timing:
        notes.append(session.quality.advice)

    return Timeline(
        shots=shots,
        cuts=cuts,
        session=session,
        style_name=style.name,
        notes=tuple(notes),
    )


def _cuts_for(shots: tuple[Shot, ...]) -> tuple[Cut, ...]:
    """Build the transitions between consecutive shots.

    Args:
        shots: The selected shots, in order.

    Returns:
        One cut per shot boundary.
    """
    cuts: list[Cut] = []
    for index in range(len(shots) - 1):
        outgoing = shots[index]
        incoming = shots[index + 1]
        slow = float(incoming.candidate.score) >= SLOW_MOTION_SCORE
        cuts.append(
            Cut(
                time=incoming.start,
                kind=TransitionKind.SPEED_RAMP if slow else TransitionKind.CUT,
                duration=Seconds(0.0),
                from_index=outgoing.index,
                to_index=incoming.index,
                speed_factor=SLOW_MOTION_FACTOR if slow else 1.0,
            )
        )
    return tuple(cuts)
