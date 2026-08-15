"""Reading a timeline back from its canonical JSON.

The write side lives in :mod:`~assetto_ai_director.infrastructure.export.json_timeline`;
this is the inverse, and it exists so a timeline is a **document rather than a
pipeline artefact**. A user can direct a cut once, hand-edit a camera position in
a text editor, and export it to another format without re-running the analysis —
which is the whole reason the JSON was made readable in the first place.

A file whose schema version is not recognised is refused outright. Guessing at a
field's meaning across a version boundary is exactly how a tool starts producing
confident nonsense.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from assetto_ai_director.domain.camera import (
    CameraKeyframe,
    CameraPath,
    CameraSpec,
    RigKind,
    ShakeProfile,
)
from assetto_ai_director.domain.geometry import CameraBasis, Vec3
from assetto_ai_director.domain.identity import CarIdentity
from assetto_ai_director.domain.session import SessionMeta
from assetto_ai_director.domain.shot import LineSide, Shot, ShotCandidate, ShotSize
from assetto_ai_director.domain.timeline import SCHEMA_VERSION, Cut, Timeline, TransitionKind
from assetto_ai_director.domain.units import FovDegrees, Hertz, Metres, Score, Seconds

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["read_timeline", "timeline_from_dict"]

_VECTOR_COMPONENTS = 3


def read_timeline(source: Path) -> Timeline:
    """Read a timeline from a JSON file.

    Args:
        source: The file to read.

    Returns:
        The parsed timeline.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If the schema version is unsupported or a field is missing.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    return timeline_from_dict(json.loads(source.read_text(encoding="utf-8")))


def timeline_from_dict(document: dict[str, Any]) -> Timeline:
    """Reconstruct a timeline from its serialised form.

    Args:
        document: The parsed JSON document.

    Returns:
        The timeline.

    Raises:
        ValueError: If the schema version is unsupported or a field is missing.
    """
    version = document.get("schema_version")
    if version != SCHEMA_VERSION:
        message = (
            f"Unsupported timeline schema {version!r}; this build reads version {SCHEMA_VERSION}."
        )
        raise ValueError(message)

    try:
        session = _session_from(document["session"])
        shots = tuple(_shot_from(entry) for entry in document["shots"])
        cuts = tuple(_cut_from(entry) for entry in document.get("cuts", ()))
    except KeyError as error:
        message = f"Timeline is missing required field {error.args[0]!r}."
        raise ValueError(message) from error

    return Timeline(
        shots=shots,
        cuts=cuts,
        session=session,
        style_name=document.get("style", "unknown"),
        notes=tuple(document.get("notes", ())),
    )


def _session_from(entry: dict[str, Any]) -> SessionMeta:
    """Rebuild session metadata.

    Args:
        entry: The session mapping.

    Returns:
        The session metadata.
    """
    cars = tuple(
        CarIdentity(
            car_id=car["id"],
            model=car["model"],
            driver_name=car.get("driver"),
            skin_id=car.get("skin"),
            is_player=bool(car.get("is_player", False)),
        )
        for car in entry.get("cars", ())
    )
    rate = float(entry.get("sample_rate_hz", 0.0))
    return SessionMeta(
        track_id=entry.get("track", ""),
        layout=entry.get("layout", ""),
        weather_id=entry.get("weather", ""),
        sample_rate=Hertz(rate),
        frame_count=0,
        cars=cars,
        source_name="timeline-json",
    )


def _shot_from(entry: dict[str, Any]) -> Shot:
    """Rebuild one shot, including its camera path.

    Args:
        entry: The shot mapping.

    Returns:
        The shot.
    """
    camera = entry["camera"]
    near = camera.get("fov_near")
    far = camera.get("fov_far")
    spec = CameraSpec(
        rig=RigKind(camera["rig"]),
        fov=FovDegrees(float(camera["fov_degrees_vertical"])),
        fov_near=None if near is None else FovDegrees(float(near)),
        fov_far=None if far is None else FovDegrees(float(far)),
        smoothing=Seconds(float(camera.get("smoothing_s", 0.0))),
        shake=ShakeProfile(amplitude=Metres(float(camera.get("shake_amplitude_m", 0.0)))),
    )
    keyframes = tuple(_keyframe_from(frame) for frame in camera["keyframes"])
    path = CameraPath(keyframes=keyframes, spec=spec)

    candidate = ShotCandidate(
        start=Seconds(float(entry["start_s"])),
        end=Seconds(float(entry["end_s"])),
        subjects=tuple(entry["subjects"]),
        rig=RigKind(entry["rig"]),
        size=ShotSize[entry["size"].upper()],
        side=LineSide(entry["side"]),
        score=Score(float(entry.get("score", 0.0))),
        spec=spec,
    )
    return Shot(candidate=candidate, path=path, index=int(entry["index"]))


def _keyframe_from(entry: dict[str, Any]) -> CameraKeyframe:
    """Rebuild one camera keyframe.

    Args:
        entry: The keyframe mapping.

    Returns:
        The keyframe.
    """
    position = _vec(entry["position"])
    look = _vec(entry["look"])
    up = _vec(entry["up"])
    dof = entry.get("dof_distance")
    return CameraKeyframe(
        time=Seconds(float(entry["t"])),
        basis=CameraBasis(position, look, up),
        fov=FovDegrees(float(entry["fov"])),
        dof_distance=None if dof is None else Metres(float(dof)),
        dof_factor=float(entry.get("dof_factor", 0.0)),
        own_share=float(entry.get("own_share", 1.0)),
    )


def _vec(values: list[float]) -> Vec3:
    """Rebuild a world-frame vector from a serialised triple.

    Args:
        values: ``[x, y, z]``.

    Returns:
        The vector.

    Raises:
        ValueError: If fewer than three components are present. A truncated
            vector would otherwise be silently completed with a default and
            place a camera somewhere no one chose.
    """
    if len(values) < _VECTOR_COMPONENTS:
        message = f"Expected three vector components, got {len(values)}."
        raise ValueError(message)
    return Vec3(float(values[0]), float(values[1]), float(values[2]))


def _cut_from(entry: dict[str, Any]) -> Cut:
    """Rebuild one transition.

    Args:
        entry: The cut mapping.

    Returns:
        The cut.
    """
    return Cut(
        time=Seconds(float(entry["time_s"])),
        kind=TransitionKind(entry.get("kind", "cut")),
        duration=Seconds(float(entry.get("duration_s", 0.0))),
        from_index=entry.get("from_shot"),
        to_index=entry.get("to_shot"),
        speed_factor=float(entry.get("speed_factor", 1.0)),
    )
