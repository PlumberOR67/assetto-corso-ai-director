"""The MCP server: the whole director, exposed to an MCP client.

Built against the official SDK's 2.x API. That release renamed the high-level
entry point from ``FastMCP`` to ``MCPServer`` and moved it to ``mcp.server``,
with **no compatibility alias**, so anything written against the older name does
not import. The version is pinned in ``pyproject.toml`` and the import is
verified by the test suite rather than assumed.

**Tools return structured data, never rendered text.** A client that receives
``{"drifts": [{"at": 41.2, "peak_angle_deg": 34.1}]}`` can filter, sort and act
on it; one that receives a formatted table has to parse prose back into numbers.
The CLI is where output gets pretty.

**Every tool that can mislead says so in its result.** Analyses carry the same
notes the timeline does — which channels this source cannot see, which cars were
skipped, whether the style's pacing is measured or guessed. A model reading these
results has no other way to know what the analysis was blind to.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mcp.server import MCPServer

from assetto_ai_director.application.camera.paths import SubjectTracks
from assetto_ai_director.application.diagnostics import diagnose_cameras as run_diagnosis
from assetto_ai_director.application.director.styles import BUILTIN_STYLES
from assetto_ai_director.application.pipeline import direct as run_director
from assetto_ai_director.domain.events import EventKind
from assetto_ai_director.domain.scoring import ScoreKind
from assetto_ai_director.infrastructure.export.cameras_ini import planned_spans
from assetto_ai_director.infrastructure.export.diagnostic_json import diagnostic_to_dict
from assetto_ai_director.infrastructure.export.json_timeline import (
    timeline_to_dict,
    write_timeline,
)
from assetto_ai_director.infrastructure.style.toml_io import read_profile
from assetto_ai_director.infrastructure.track.resolve import resolve_spline
from assetto_ai_director.mcp.session import SessionStore

__all__ = ["build_server", "mcp", "store"]

store = SessionStore()

mcp: MCPServer = MCPServer(
    name="assetto-ai-director",
    instructions=(
        "Analyse Assetto Corsa replays and direct cinematic camera timelines. "
        "Call open_session first; every other tool takes the handle it returns. "
        "Results carry a 'notes' field listing what the analysis could not see — "
        "read it before drawing conclusions about what did not happen."
    ),
)


@mcp.tool()
def open_session(path: str, seconds: float = 0.0) -> dict[str, Any]:
    """Open a replay and prepare it for analysis.

    Args:
        path: Path to an .acreplay file.
        seconds: Analyse only the first N seconds. 0 analyses everything.

    Returns:
        The session handle and what the replay contains.
    """
    session = store.open(Path(path), seconds)
    meta = session.meta
    return {
        "handle": session.handle,
        "track": meta.track_label,
        "weather": meta.weather_id,
        "sample_rate_hz": round(float(meta.sample_rate), 2),
        "recording_quality": meta.quality.name.lower(),
        "quality_advice": meta.quality.advice,
        "analysed_seconds": round(session.duration, 1),
        "cars_analysed": sorted(session.signals),
        "cars_skipped": [car.car_id for car in meta.cars if car.car_id not in session.signals],
        "entry_list": [
            {"id": car.car_id, "model": car.model, "driver": car.driver_name} for car in meta.cars
        ],
    }


@mcp.tool()
def load_replay(path: str) -> dict[str, Any]:
    """Read a replay's metadata without decoding telemetry.

    Fast enough to call on a directory of replays. Use it to choose which one to
    open rather than opening each in turn.

    Args:
        path: Path to an .acreplay file.

    Returns:
        The replay's metadata and whether its layout verified.
    """
    from assetto_ai_director.infrastructure.acreplay.structure import (  # noqa: PLC0415
        read_structure,
    )

    structure = read_structure(Path(path))
    meta = structure.to_session_meta()
    return {
        "track": meta.track_label,
        "weather": meta.weather_id,
        "cars": meta.car_count,
        "frames": structure.frame_count,
        "sample_rate_hz": round(float(meta.sample_rate), 2),
        "duration_s": round(float(meta.duration), 1),
        "recording_quality": meta.quality.name.lower(),
        "layout_verified": structure.verified,
        "ring_buffer_wrapped": structure.wrapped,
    }


@mcp.tool()
def load_telemetry(
    handle: str, car_id: int, start_s: float = 0.0, end_s: float = 0.0
) -> dict[str, Any]:
    """Summarise one car's telemetry over a span.

    Args:
        handle: Session handle from open_session.
        car_id: Which car.
        start_s: Span start. 0 starts at the beginning.
        end_s: Span end. 0 runs to the end.

    Returns:
        Summary statistics for the requested span.
    """
    import numpy as np  # noqa: PLC0415

    session = store.get(handle)
    if car_id not in session.signals:
        message = f"Car {car_id} did not move. Analysed cars: {sorted(session.signals)}."
        raise ValueError(message)

    car = session.signals[car_id]
    first = int(np.searchsorted(session.time, start_s))
    last = int(np.searchsorted(session.time, end_s)) if end_s > 0.0 else session.time.shape[0]
    last = max(first + 1, min(last, session.time.shape[0]))

    speed = car.speed[first:last] * 3.6
    sideslip = np.degrees(np.abs(car.sideslip[first:last]))
    elevation = car.position[first:last, 1]
    return {
        "car_id": car_id,
        "span_s": [round(float(session.time[first]), 2), round(float(session.time[last - 1]), 2)],
        "speed_kmh": {
            "median": round(float(np.median(speed)), 1),
            "p95": round(float(np.percentile(speed, 95)), 1),
            "max": round(float(speed.max()), 1),
        },
        "sideslip_deg": {
            "median": round(float(np.median(sideslip)), 2),
            "p95": round(float(np.percentile(sideslip, 95)), 2),
            "max": round(float(sideslip.max()), 2),
        },
        "elevation_change_m": round(float(elevation[-1] - elevation[0]), 1),
        "available_channels": ["position", "speed", "heading", "sideslip"],
        "notes": [
            (
                "Throttle, brake, gear, rpm and per-wheel slip are present in "
                "the replay format but have not been located, so they are not "
                "reported."
            )
        ],
    }


@mcp.tool()
def find_drifts(handle: str, min_angle_deg: float = 10.0) -> dict[str, Any]:
    """Find sustained slides.

    Args:
        handle: Session handle from open_session.
        min_angle_deg: Only report slides whose peak exceeds this angle.

    Returns:
        The detected drifts, strongest first.
    """
    session = store.get(handle)
    drifts = [
        _event_to_dict(event)
        for event in session.detection.of_kind(EventKind.DRIFT_START, EventKind.SPIN)
        if event.detail.get("peak_angle_deg", 0.0) >= min_angle_deg
    ]
    drifts.sort(key=lambda entry: -entry["detail"].get("peak_angle_deg", 0.0))
    return {"count": len(drifts), "drifts": drifts, "notes": list(session.notes)}


@mcp.tool()
def find_transitions(handle: str) -> dict[str, Any]:
    """Find linked transitions — the flick from one slide directly into another.

    Args:
        handle: Session handle from open_session.

    Returns:
        The detected transitions, in time order.
    """
    session = store.get(handle)
    events = [_event_to_dict(e) for e in session.detection.of_kind(EventKind.DRIFT_TRANSITION)]
    return {"count": len(events), "transitions": events}


@mcp.tool()
def find_best_moments(handle: str, style: str = "initial_d", limit: int = 10) -> dict[str, Any]:
    """Rank the session's moments under a style's weighting.

    Args:
        handle: Session handle from open_session.
        style: Which style's priorities to rank by.
        limit: How many moments to return.

    Returns:
        The highest-scoring moments, best first.
    """
    import numpy as np  # noqa: PLC0415

    session = store.get(handle)
    profile = _style(style)
    scores = session.scores
    subject = max(scores, key=lambda car: float(session.signals[car].speed.sum()))

    combined = np.array([profile.weights.combine(vector) for vector in scores[subject]])
    spacing = max(int(session.sample_rate * 3.0), 1)
    chosen: list[int] = []
    for index in np.argsort(combined)[::-1]:
        position = int(index)
        if all(abs(position - kept) > spacing for kept in chosen):
            chosen.append(position)
        if len(chosen) >= limit:
            break

    moments = [
        {
            "at_s": round(float(session.time[index]), 2),
            "car_id": subject,
            "total": round(float(combined[index]), 4),
            "scores": {
                kind.value: round(float(scores[subject][index][kind]), 3) for kind in ScoreKind
            },
        }
        for index in sorted(chosen, key=lambda i: -combined[i])
    ]
    return {
        "style": style,
        "moments": moments,
        "notes": ["Ranking reflects the style's weights, which are editorial judgement."],
    }


@mcp.tool()
def generate_shotlist(handle: str, style: str = "initial_d") -> dict[str, Any]:
    """Direct a cut and return its shot list without camera geometry.

    Use this to inspect the editorial decisions. Call preview or export_json for
    the full timeline including camera paths.

    Args:
        handle: Session handle from open_session.
        style: Which style to direct in.

    Returns:
        The shot list and a summary of the edit.
    """
    session = store.get(handle)
    timeline = _timeline(handle, style)
    return {
        "style": style,
        "shot_count": timeline.shot_count,
        "duration_s": round(float(timeline.duration), 2),
        "average_shot_length_s": round(float(timeline.average_shot_length), 2),
        "tiles_without_gaps": timeline.tiles_without_gaps(),
        "shots": [
            {
                "index": shot.index,
                "start_s": round(float(shot.start), 2),
                "duration_s": round(float(shot.duration), 2),
                "rig": shot.candidate.rig.value,
                "size": shot.candidate.size.name.lower(),
                "side": shot.candidate.side.value,
                "subjects": list(shot.candidate.subjects),
                "score": round(float(shot.candidate.score), 3),
                "because": shot.candidate.anchor.kind.value if shot.candidate.anchor else None,
            }
            for shot in timeline.shots
        ],
        "notes": list(timeline.notes),
        "session": session.meta.track_label,
    }


@mcp.tool()
def generate_style(name: str) -> dict[str, Any]:
    """Describe a style: how it values moments, how it paces, what it forbids.

    Args:
        name: A built-in style name, or a path to a TOML profile.

    Returns:
        The style's parameters, each with its provenance.
    """
    from assetto_ai_director.domain.provenance import ProvenanceReport  # noqa: PLC0415

    profile = _style(name)
    report = ProvenanceReport.of(profile)
    return {
        "name": profile.name,
        "description": profile.description,
        "is_measured": profile.is_measured,
        "rhythm": {
            "median_shot_length_s": _tunable(profile.rhythm.median_shot_length),
            "spread": _tunable(profile.rhythm.spread),
            "spectral_slope": _tunable(profile.rhythm.spectral_slope),
        },
        "camera_mix": {
            "static_fraction": _tunable(profile.camera_mix.static_fraction),
            "axial_fraction": _tunable(profile.camera_mix.axial_fraction),
        },
        "allowed_rigs": sorted(rig.value for rig in profile.camera_mix.allowed_rigs),
        "weights": {
            "drift": profile.weights.drift,
            "speed": profile.weights.speed,
            "risk": profile.weights.risk,
            "flow": profile.weights.flow,
            "cinematic": profile.weights.cinematic,
            "legibility": profile.weights.legibility,
        },
        "provenance": {
            "guessed": report.count(_guess()),
            "trust_ratio": round(report.trust_ratio, 3),
        },
        "notes": [
            "The six weights are editorial judgement and are not provenance-tracked.",
        ],
        "available": sorted(BUILTIN_STYLES),
    }


@mcp.tool()
def preview(handle: str, style: str = "initial_d", shot_index: int = 0) -> dict[str, Any]:
    """Inspect one shot's camera path in detail.

    Args:
        handle: Session handle from open_session.
        style: Which style's edit to inspect.
        shot_index: Which shot.

    Returns:
        The shot's camera specification and a sample of its keyframes.
    """
    timeline = _timeline(handle, style)
    if not 0 <= shot_index < timeline.shot_count:
        message = f"Shot {shot_index} is out of range; the edit has {timeline.shot_count}."
        raise ValueError(message)

    shot = timeline.shots[shot_index]
    keyframes = shot.path.keyframes
    sample = [keyframes[0], keyframes[len(keyframes) // 2], keyframes[-1]]
    return {
        "index": shot.index,
        "start_s": round(float(shot.start), 2),
        "duration_s": round(float(shot.duration), 2),
        "rig": shot.candidate.rig.value,
        "fov_degrees_vertical": round(float(shot.path.spec.fov), 1),
        "keyframe_count": len(keyframes),
        "keyframes": [
            {
                "t": round(float(frame.time), 2),
                "position": [
                    round(frame.basis.position.x, 2),
                    round(frame.basis.position.y, 2),
                    round(frame.basis.position.z, 2),
                ],
                "fov": round(float(frame.fov), 1),
                "own_share": round(frame.own_share, 2),
            }
            for frame in sample
        ],
    }


@mcp.tool()
def export_json(handle: str, path: str, style: str = "initial_d") -> dict[str, Any]:
    """Write a directed timeline to a JSON file.

    Args:
        handle: Session handle from open_session.
        path: Where to write.
        style: Which style to direct in.

    Returns:
        Where it was written and what it contains.
    """
    timeline = _timeline(handle, style)
    destination = Path(path)
    write_timeline(timeline, destination)
    document = timeline_to_dict(timeline)
    return {
        "written_to": str(destination.resolve()),
        "shot_count": document["summary"]["shot_count"],
        "duration_s": document["summary"]["duration_s"],
        "notes": document["notes"],
    }


@mcp.tool()
def diagnose_cameras_tool(
    handle: str, style: str = "initial_d", notable_only: bool = True
) -> dict[str, Any]:
    """Check what a directed cut's cameras would actually see.

    Answers, per camera and in numbers: is the car in front of the lens, in
    frame, a sensible size, at a sensible range — and does the track span the
    cameras.ini export would write actually contain the car when the shot
    happens. Checks that cannot run without the track's geometry report as
    unknown rather than being omitted, so an unexamined camera is never
    mistaken for a good one.

    Args:
        handle: Session handle from open_session.
        style: Which style to direct in.
        notable_only: Return only the cameras whose measured checks warned or
            failed. The summary and the per-check tally always describe the
            whole cut either way. Turning this off on a long session returns a
            very large document.

    Returns:
        The diagnosis, in the schema published at contracts/diagnostic.schema.json.
    """
    session = store.get(handle)
    timeline = _timeline(handle, style)
    tracks = SubjectTracks.from_signals(session.signals, session.time, session.sample_rate)
    resolved = resolve_spline(
        timeline.session.track_id,
        timeline.session.layout,
        next(iter(tracks.positions.values()), None),
    )
    report = run_diagnosis(
        timeline,
        tracks,
        exported_spans=planned_spans(timeline, track=resolved.spline),
        track=resolved.spline,
        notes=(resolved.note,) if resolved.note else (),
    )
    return diagnostic_to_dict(report, notable_only=notable_only)


def _timeline(handle: str, style: str) -> Any:
    """Direct a cut, reusing the cached one when the style is unchanged.

    Args:
        handle: The session.
        style: The style name or profile path.

    Returns:
        The timeline.
    """
    session = store.get(handle)
    if style in session.timelines:
        return session.timelines[style]
    profile = _style(style)
    timeline = run_director(
        session.signals,
        session.meta,
        profile,
        session.time,
        session.sample_rate,
    )
    session.timelines[style] = timeline
    return timeline


def _style(name: str) -> Any:
    """Resolve a style name or profile path.

    Args:
        name: A built-in name, or a path to a TOML profile.

    Returns:
        The style profile.

    Raises:
        ValueError: If the name is neither.
    """
    candidate = Path(name)
    if candidate.suffix == ".toml" and candidate.is_file():
        return read_profile(candidate)
    if name not in BUILTIN_STYLES:
        available = ", ".join(sorted(BUILTIN_STYLES))
        message = f"Unknown style {name!r}. Available: {available}."
        raise ValueError(message)
    return BUILTIN_STYLES[name]


def _event_to_dict(event: Any) -> dict[str, Any]:
    """Render an event as structured data.

    Args:
        event: The event.

    Returns:
        The mapping.
    """
    return {
        "kind": event.kind.value,
        "at_s": round(float(event.focus), 2),
        "start_s": round(float(event.start), 2),
        "end_s": round(float(event.end), 2),
        "duration_s": round(float(event.duration), 2),
        "subjects": list(event.subjects),
        "severity": event.severity.value,
        "confidence": round(event.confidence, 3),
        "detail": {key: round(value, 3) for key, value in event.detail.items()},
    }


def _tunable(value: Any) -> dict[str, Any]:
    """Render a tunable with its provenance intact.

    Args:
        value: The tunable.

    Returns:
        The mapping.
    """
    return {
        "value": round(float(value.value), 4),
        "provenance": value.provenance.value,
        "source": value.source,
    }


def _guess() -> Any:
    """Return the guess provenance without importing it at module scope.

    Returns:
        ``Provenance.GUESS``.
    """
    from assetto_ai_director.domain.provenance import Provenance  # noqa: PLC0415

    return Provenance.GUESS


def build_server() -> MCPServer:
    """Return the configured server.

    Returns:
        The server, with every tool registered.
    """
    return mcp


def main() -> None:
    """Run the server over stdio."""
    mcp.run()


if __name__ == "__main__":  # pragma: no cover - exercised by the console script
    main()
