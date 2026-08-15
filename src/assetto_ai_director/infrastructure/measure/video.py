"""Extract shot boundaries and camera behaviour from reference footage.

Two measurements, from two different techniques:

**Where the cuts are** comes from PySceneDetect's content detector, which
compares consecutive frames in HSV and flags a boundary when the difference
spikes. This is well-suited to the material: hard cuts between visually distinct
shots. It will miss dissolves, and a whip pan can read as a cut — both are noted
in the output rather than hidden.

**What the camera was doing** comes from dense optical flow (Farneback). Flow
alone cannot distinguish a locked-off camera watching a car cross frame from a
panning camera holding a still car, because both produce motion in the image. The
classifier separates them by *coherence*: a camera move shifts the whole frame
together, whereas a subject moving through a static frame produces flow in only
part of it. That distinction is the single most stylistically informative thing
recoverable from footage without scene geometry.

**What cannot be recovered, and is therefore not claimed:** true focal length,
camera height in metres, or subject distance. Those need scene geometry or camera
metadata that footage does not carry. The project marks them ``GUESS`` rather
than deriving a confident-looking number from pixels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import cv2
import numpy as np
from scenedetect import ContentDetector, detect

from assetto_ai_director.domain.measurement import (
    CameraMotion,
    ShotMeasurement,
    StyleMeasurement,
)
from assetto_ai_director.domain.units import Seconds

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import numpy.typing as npt

    from assetto_ai_director.application.ports.progress import ProgressReporter

__all__ = ["measure_clip", "measure_style"]

# --- Detection parameters ---------------------------------------------------
# All GUESS provenance: engineering judgement, chosen to be conservative rather
# than tuned to any particular clip. Documented here so they can be revisited
# against a hand-labelled ground truth rather than silently trusted.

CONTENT_THRESHOLD: Final = 27.0
"""PySceneDetect's own default. Left alone deliberately.

Tuning it per clip until the shot count "looks right" would fit the detector to
the answer, which defeats the purpose of measuring.
"""

MIN_SHOT_FRAMES: Final = 12
"""Shortest span accepted as a shot.

At 60 fps this is 0.2 s. Below that, a detection is far more likely to be a whip
pan, a flash frame or compression noise than a real editorial choice.
"""

FLOW_SAMPLES_PER_SHOT: Final = 5
"""Frame pairs sampled per shot for optical flow.

Dense flow is expensive and camera behaviour within a single shot is largely
constant, so a handful of samples spread across the shot captures it. Sampling
every frame would multiply cost by two orders of magnitude for no new
information.
"""

FLOW_WIDTH: Final = 320
"""Width that frames are downscaled to before computing flow.

Camera motion is a low-spatial-frequency signal; full resolution adds cost and
sensor noise without adding signal.
"""

STATIC_MAGNITUDE: Final = 0.20
"""Flow speed, in frame heights per **second**, below which a camera is static.

Per second, not per frame, and that distinction is a correctness requirement
rather than a preference. Optical flow measures displacement between *adjacent
frames*, so identical physical camera motion produces half as much per-frame flow
at 60 fps as at 30 fps. A corpus mixing frame rates — which any real collection of
reference clips does — would then have its 60 fps material systematically
classified as calmer than its 30 fps material, for no reason but the encoder.

Multiplying by frame rate makes the measurement a property of the camera instead
of a property of the file.

Deliberately not zero: encoder noise and grain produce a small non-zero floor
even on a genuinely locked-off frame.
"""

COHERENCE_FOR_CAMERA_MOVE: Final = 0.55
"""Fraction of the frame that must move together to call it a camera move.

Below this, the motion is localised — a subject crossing an otherwise still
frame, not the camera moving.
"""

TRACK_MAGNITUDE: Final = 0.90
"""Flow speed, in frame heights per second, above which a coherent move is a track.

A pan rotates about a point; a tracking camera translates through the world
alongside a fast subject and sweeps markedly more image past the sensor.
"""

AXIAL_RADIAL_FRACTION: Final = 0.6
"""Share of flow energy that must point to or from frame centre to call a shot axial.

Above this the camera is travelling along its own optical axis. The classifier
does **not** try to split that into "zoom" and "dolly", because optical flow
genuinely cannot: both expand the image from a centre, and separating them needs
scene geometry the footage does not carry.

That merge is a deliberate reduction in claimed precision. An earlier version
split them by magnitude and duly reported that 60% of touge footage was zoom
shots — a conclusion no one who has watched the footage would accept. The split
was measuring nothing; only the label was doing work.
"""

_MIN_FRAMES_FOR_FLOW = 2
_DIRECTION_EPSILON = 1e-6
"""Below this vector length a direction is numerically meaningless."""

_ENERGY_EPSILON = 1e-9
"""Below this total flow energy a share of it would be a division by noise."""


def measure_clip(path: Path, *, progress_label: str | None = None) -> list[ShotMeasurement]:
    """Measure one video's shots and camera behaviour.

    Args:
        path: The video file.
        progress_label: Unused placeholder kept so callers can pass a label
            without the signature changing when progress reporting is added.

    Returns:
        The measured shots, in order.

    Raises:
        OSError: If the video cannot be opened.
    """
    del progress_label
    scenes = detect(
        str(path),
        ContentDetector(threshold=CONTENT_THRESHOLD, min_scene_len=MIN_SHOT_FRAMES),
    )
    spans = [(scene[0].seconds, scene[1].seconds) for scene in scenes]
    if not spans:
        # A clip with no detected cut is one continuous take, not an error.
        spans = [(0.0, _clip_duration(path))]

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        message = f"Cannot open video: {path}"
        raise OSError(message)
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        return [
            _measure_shot(capture, index, start, end, fps)
            for index, (start, end) in enumerate(spans)
        ]
    finally:
        capture.release()


def measure_style(
    style_name: str, paths: Sequence[Path], progress: ProgressReporter | None = None
) -> StyleMeasurement:
    """Measure every clip for a style and aggregate the result.

    Shots are concatenated across clips in the order given. That is correct for
    distribution statistics such as average shot length, but it does introduce
    artificial boundaries in the *sequence*, which matters for spectral analysis
    of cutting rhythm. Callers doing spectral work should weigh that against the
    sample size they gain.

    Args:
        style_name: Name of the style being measured.
        paths: The clips to measure.
        progress: Optional reporter. Dense optical flow over a long corpus runs
            for minutes, and a silent terminal reads as a hung process.

    Returns:
        The aggregate measurement.
    """
    shots: list[ShotMeasurement] = []
    total = 0.0
    if progress is not None:
        progress.start("Measuring clips", total=len(paths))
    for path in paths:
        clip_shots = measure_clip(path)
        offset = len(shots)
        shots.extend(
            ShotMeasurement(
                index=offset + shot.index,
                start=shot.start,
                end=shot.end,
                motion=shot.motion,
                motion_magnitude=shot.motion_magnitude,
                motion_coherence=shot.motion_coherence,
            )
            for shot in clip_shots
        )
        total += _clip_duration(path)
        if progress is not None:
            progress.advance()
    if progress is not None:
        progress.finish()
    return StyleMeasurement(
        style_name=style_name,
        source_names=tuple(path.name for path in paths),
        shots=tuple(shots),
        total_duration=Seconds(total),
        clip_count=len(paths),
    )


def _clip_duration(path: Path) -> float:
    """Read a video's duration in seconds.

    Args:
        path: The video file.

    Returns:
        The duration, or ``0.0`` if it cannot be determined.
    """
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return 0.0
        fps = capture.get(cv2.CAP_PROP_FPS)
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        return frames / fps if fps > 0 else 0.0
    finally:
        capture.release()


def _measure_shot(
    capture: cv2.VideoCapture, index: int, start: float, end: float, fps: float
) -> ShotMeasurement:
    """Measure one shot's camera behaviour.

    Args:
        capture: An open video capture.
        index: The shot's position in the clip.
        start: Shot start in seconds.
        end: Shot end in seconds.
        fps: Frame rate.

    Returns:
        The measured shot.
    """
    magnitudes: list[float] = []
    coherences: list[float] = []
    radial_shares: list[float] = []

    for first, second in _sample_frame_pairs(capture, start, end, fps):
        magnitude, coherence, radial = _analyse_flow(first, second)
        # Convert per-frame displacement to per-second speed so clips of
        # different frame rates are directly comparable.
        magnitudes.append(magnitude * fps)
        coherences.append(coherence)
        radial_shares.append(radial)

    magnitude = float(np.median(magnitudes)) if magnitudes else 0.0
    coherence = float(np.median(coherences)) if coherences else 0.0
    radial = float(np.median(radial_shares)) if radial_shares else 0.0

    return ShotMeasurement(
        index=index,
        start=Seconds(start),
        end=Seconds(end),
        motion=_classify(magnitude, coherence, radial),
        motion_magnitude=magnitude,
        motion_coherence=coherence,
    )


def _sample_frame_pairs(
    capture: cv2.VideoCapture, start: float, end: float, fps: float
) -> list[tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]]:
    """Grab a few consecutive frame pairs spread across a shot.

    Args:
        capture: An open video capture.
        start: Shot start in seconds.
        end: Shot end in seconds.
        fps: Frame rate.

    Returns:
        Greyscale, downscaled frame pairs.
    """
    first_frame = int(start * fps)
    last_frame = max(first_frame + 1, int(end * fps) - 1)
    span = last_frame - first_frame
    if span < _MIN_FRAMES_FOR_FLOW:
        return []

    count = min(FLOW_SAMPLES_PER_SHOT, span - 1)
    positions = np.linspace(first_frame, last_frame - 1, num=max(count, 1), dtype=int)

    pairs: list[tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]] = []
    for position in positions:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(position))
        ok_a, frame_a = capture.read()
        ok_b, frame_b = capture.read()
        if not (ok_a and ok_b):
            continue
        # Narrow OpenCV's loosely-typed frames once, here at the read boundary.
        pairs.append(
            (
                _prepare(np.asarray(frame_a, dtype=np.uint8)),
                _prepare(np.asarray(frame_b, dtype=np.uint8)),
            )
        )
    return pairs


def _prepare(frame: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Downscale and greyscale a frame for flow computation.

    Args:
        frame: A colour frame.

    Returns:
        The prepared single-channel frame.
    """
    height, width = frame.shape[:2]
    # OpenCV's type stubs describe returns loosely, as "some integer or float
    # array". Narrowing back with np.asarray at the boundary keeps the loose type
    # out of the rest of the module rather than sprinkling ignores through it.
    scaled = frame
    if width > FLOW_WIDTH:
        scale = FLOW_WIDTH / width
        scaled = np.asarray(
            cv2.resize(frame, (FLOW_WIDTH, max(1, int(height * scale)))), dtype=np.uint8
        )
    return np.asarray(cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY), dtype=np.uint8)


def _analyse_flow(
    first: npt.NDArray[np.uint8], second: npt.NDArray[np.uint8]
) -> tuple[float, float, float]:
    """Compute motion magnitude, coherence and radial share for a frame pair.

    Args:
        first: Earlier frame.
        second: Later frame.

    Returns:
        ``(magnitude, coherence, radial_share)`` where magnitude is normalised by
        frame height, coherence is the share of flow aligned with the dominant
        direction, and radial share is the fraction of flow energy pointing
        toward or away from frame centre.
    """
    # An explicit output buffer rather than None: OpenCV accepts either at
    # runtime, but the typed stubs only describe the buffer form.
    buffer = np.zeros((*first.shape[:2], 2), dtype=np.float32)
    flow = np.asarray(
        cv2.calcOpticalFlowFarneback(first, second, buffer, 0.5, 3, 15, 3, 5, 1.2, 0),
        dtype=np.float32,
    )
    height, width = flow.shape[:2]
    vectors = flow.reshape(-1, 2)
    magnitudes = np.linalg.norm(vectors, axis=1)
    mean_magnitude = float(magnitudes.mean()) / height

    # Coherence: how much of the flow agrees with the average direction. A
    # camera move carries the whole frame; a subject crossing a still frame does
    # not.
    mean_vector = vectors.mean(axis=0)
    mean_norm = float(np.linalg.norm(mean_vector))
    if mean_norm < _DIRECTION_EPSILON:
        coherence = 0.0
    else:
        unit = mean_vector / mean_norm
        projections = vectors @ unit
        total = float(magnitudes.sum())
        coherence = float(np.abs(projections).sum() / total) if total > _ENERGY_EPSILON else 0.0

    # Radial share: flow pointing to or from frame centre means the camera is
    # travelling along its own optical axis (or the lens is zooming — the two are
    # indistinguishable here, and the AXIAL class says so).
    grid_y, grid_x = np.mgrid[0:height, 0:width]
    centre_x, centre_y = width / 2.0, height / 2.0
    radial = np.stack([(grid_x - centre_x).ravel(), (grid_y - centre_y).ravel()], axis=1).astype(
        np.float32
    )
    radial_norms = np.linalg.norm(radial, axis=1)
    valid = radial_norms > _DIRECTION_EPSILON
    if not valid.any():
        return mean_magnitude, coherence, 0.0
    radial_unit = radial[valid] / radial_norms[valid, None]
    radial_component = np.abs(np.einsum("ij,ij->i", vectors[valid], radial_unit))
    total_valid = float(magnitudes[valid].sum())
    radial_share = (
        float(radial_component.sum() / total_valid) if total_valid > _ENERGY_EPSILON else 0.0
    )

    return mean_magnitude, coherence, radial_share


def _classify(magnitude: float, coherence: float, radial_share: float) -> CameraMotion:
    """Assign a camera-motion class from flow statistics.

    Args:
        magnitude: Flow magnitude normalised by frame height.
        coherence: Share of flow aligned with the dominant direction.
        radial_share: Share of flow energy pointing to or from frame centre.

    Returns:
        The motion class.
    """
    if magnitude < STATIC_MAGNITUDE:
        return CameraMotion.STATIC
    if coherence < COHERENCE_FOR_CAMERA_MOVE:
        # Motion exists but does not fill the frame: the subject is moving
        # through a held frame, which is a static camera editorially.
        return CameraMotion.STATIC

    if radial_share > AXIAL_RADIAL_FRACTION:
        # Travelling along the optical axis. Not split further into "zoom" and
        # "dolly": flow cannot tell them apart, and inventing a threshold to
        # pretend otherwise would be fitting the classifier to expectations
        # rather than measuring anything.
        return CameraMotion.AXIAL
    if magnitude > TRACK_MAGNITUDE:
        return CameraMotion.TRACK
    return CameraMotion.PAN
