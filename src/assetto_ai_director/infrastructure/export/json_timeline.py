"""The canonical JSON timeline — the product everything else is derived from.

Written to be **read by a person as well as a machine**. A directed edit is an
editorial artefact: someone will want to see why a shot is where it is, diff two
versions of a cut, or hand-correct one camera. So every shot carries the event
that motivated it and the scores that justified it, not merely its geometry.

The schema is versioned, and a reader refuses a version it does not know rather
than guessing at a field's meaning.

Camera keyframes store **look as a direction, not a target point**. That is what
Custom Shaders Patch's ``GrabbedCamera`` consumes, and the conversion is lossy in
the other order — a target can always be turned into a direction, but recovering
a target from one requires a distance nothing records.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from assetto_ai_director.application.ports.timeline_sink import SupportReport
from assetto_ai_director.domain.timeline import SCHEMA_VERSION

if TYPE_CHECKING:
    from pathlib import Path

    from assetto_ai_director.domain.camera import CameraKeyframe, CameraPath
    from assetto_ai_director.domain.geometry import Vec3
    from assetto_ai_director.domain.shot import Shot
    from assetto_ai_director.domain.timeline import Timeline

__all__ = ["JsonTimelineSink", "timeline_to_dict", "write_timeline"]


class JsonTimelineSink:
    """Writes a timeline as canonical JSON.

    Satisfies the ``TimelineSink`` protocol structurally.
    """

    @property
    def name(self) -> str:
        """A short identifier for this sink.

        Returns:
            The sink name.
        """
        return "json"

    @property
    def extension(self) -> str:
        """The file extension this sink writes.

        Returns:
            ``".json"``.
        """
        return ".json"

    def supports(self, timeline: Timeline) -> SupportReport:
        """Report what would survive exporting this timeline.

        Args:
            timeline: The timeline to assess.

        Returns:
            A lossless support report — this is the canonical format, so nothing
            is dropped by definition.
        """
        del timeline
        return SupportReport.full()

    def write(self, timeline: Timeline, destination: Path) -> None:
        """Write the timeline.

        Args:
            timeline: The timeline to export.
            destination: Where to write it.

        Raises:
            OSError: If the destination cannot be written.
        """
        write_timeline(timeline, destination)


def write_timeline(timeline: Timeline, destination: Path) -> None:
    """Write a timeline to a JSON file.

    Args:
        timeline: The timeline to export.
        destination: Where to write it. Parent directories are created.

    Raises:
        OSError: If the destination cannot be written.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(timeline_to_dict(timeline), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def timeline_to_dict(timeline: Timeline) -> dict[str, Any]:
    """Convert a timeline to a JSON-serialisable mapping.

    Args:
        timeline: The timeline.

    Returns:
        The document.
    """
    session = timeline.session
    return {
        "schema_version": SCHEMA_VERSION,
        "style": timeline.style_name,
        "session": {
            "track": session.track_id,
            "layout": session.layout,
            "weather": session.weather_id,
            "sample_rate_hz": round(float(session.sample_rate), 4),
            "recording_quality": session.quality.name.lower(),
            "cars": [
                {
                    "id": car.car_id,
                    "model": car.model,
                    "driver": car.driver_name,
                    "skin": car.skin_id,
                    "is_player": car.is_player,
                }
                for car in session.cars
            ],
        },
        "summary": {
            "shot_count": timeline.shot_count,
            "duration_s": round(float(timeline.duration), 3),
            "average_shot_length_s": round(float(timeline.average_shot_length), 3),
            "tiles_without_gaps": timeline.tiles_without_gaps(),
        },
        # Carried into every export so a consumer is told what the analysis could
        # not see, rather than inferring completeness from a confident-looking file.
        "notes": list(timeline.notes),
        "shots": [_shot_to_dict(shot) for shot in timeline.shots],
        "cuts": [
            {
                "time_s": round(float(cut.time), 3),
                "kind": cut.kind.value,
                "duration_s": round(float(cut.duration), 3),
                "from_shot": cut.from_index,
                "to_shot": cut.to_index,
                "speed_factor": round(cut.speed_factor, 3),
            }
            for cut in timeline.cuts
        ],
    }


def _shot_to_dict(shot: Shot) -> dict[str, Any]:
    """Serialise one shot, including why it was chosen.

    Args:
        shot: The shot.

    Returns:
        The shot's mapping.
    """
    candidate = shot.candidate
    entry: dict[str, Any] = {
        "index": shot.index,
        "start_s": round(float(shot.start), 3),
        "end_s": round(float(shot.end), 3),
        "duration_s": round(float(shot.duration), 3),
        "subjects": list(candidate.subjects),
        "rig": candidate.rig.value,
        "size": candidate.size.name.lower(),
        "side": candidate.side.value,
        "score": round(float(candidate.score), 4),
        "camera": _path_to_dict(shot.path),
    }
    if candidate.anchor is not None:
        anchor = candidate.anchor
        entry["because"] = {
            "event": anchor.kind.value,
            "at_s": round(float(anchor.focus), 3),
            "severity": anchor.severity.value,
            "confidence": round(anchor.confidence, 3),
            "detail": {key: round(value, 4) for key, value in anchor.detail.items()},
        }
    return entry


def _path_to_dict(path: CameraPath) -> dict[str, Any]:
    """Serialise a camera path.

    Args:
        path: The path.

    Returns:
        The path's mapping.
    """
    spec = path.spec
    return {
        "rig": spec.rig.value,
        "fov_degrees_vertical": round(float(spec.fov), 2),
        # Carried explicitly: without them a round trip loses zoom tracking, and
        # the cameras.ini exporter — whose whole MIN_FOV/MAX_FOV mechanism is
        # zoom tracking — silently writes a fixed lens.
        "fov_near": None if spec.fov_near is None else round(float(spec.fov_near), 2),
        "fov_far": None if spec.fov_far is None else round(float(spec.fov_far), 2),
        "smoothing_s": round(float(spec.smoothing), 3),
        "shake_amplitude_m": round(float(spec.shake.amplitude), 4),
        "max_speed_mps": None if spec.max_speed is None else round(float(spec.max_speed), 2),
        "keyframes": [_keyframe_to_dict(keyframe) for keyframe in path.keyframes],
    }


def _keyframe_to_dict(keyframe: CameraKeyframe) -> dict[str, Any]:
    """Serialise one camera keyframe.

    Args:
        keyframe: The keyframe.

    Returns:
        The keyframe's mapping.
    """
    entry: dict[str, Any] = {
        "t": round(float(keyframe.time), 3),
        "position": _vec(keyframe.basis.position),
        "look": _vec(keyframe.basis.look),
        "up": _vec(keyframe.basis.up),
        "fov": round(float(keyframe.fov), 2),
        "own_share": round(keyframe.own_share, 3),
    }
    if keyframe.dof_distance is not None:
        entry["dof_distance"] = round(float(keyframe.dof_distance), 2)
        entry["dof_factor"] = round(keyframe.dof_factor, 3)
    return entry


def _vec(vector: Vec3) -> list[float]:
    """Serialise a vector as a three-element list.

    Args:
        vector: The vector.

    Returns:
        ``[x, y, z]`` rounded to millimetre precision, which is far finer than
        any camera move needs and keeps the file readable.
    """
    return [round(vector.x, 4), round(vector.y, 4), round(vector.z, 4)]
