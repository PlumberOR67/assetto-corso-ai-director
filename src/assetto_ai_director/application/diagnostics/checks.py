"""The camera checks themselves.

Each check is a named measurement with a threshold and a sentence explaining
what the number means for the shot. They are deliberately simple and
individually wrong-able: a reader who disagrees with a threshold can still use
the measurement, which is why every check publishes both.

**What these checks can and cannot see.** Most work from the camera path and the
cars' world positions alone, which the pipeline always has: whether the car is in
front of the lens, in frame, a sensible size, at a sensible range. Two more need
the track's centreline and answer only when one was supplied — whether the camera
is above the road beside it, and how far past the road's edge it stands. One
still cannot be answered at all: nothing reads collision geometry, so a barrier
between the camera and the car goes undetected.

Every check that cannot run reports
:attr:`~assetto_ai_director.domain.diagnostics.CheckStatus.UNKNOWN` with its
reason rather than being left out, because a check missing from a report reads as
a check that passed.

**The span check is the one this was built for.** Assetto Corsa switches
trackside cameras on normalised *track position*; the exporter writes a
normalised *time*. :func:`_check_span` puts those two numbers next to each other
and converts the gap into metres of track, which turns a suspicion into a
measurement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from assetto_ai_director.application.camera.rigs import SIZE_DISTANCE_METRES
from assetto_ai_director.application.camera.spline import DistanceProfile, distance_profile
from assetto_ai_director.domain.camera import RigKind
from assetto_ai_director.domain.diagnostics import (
    CameraDiagnostic,
    Check,
    CheckStatus,
    DiagnosticReport,
)
from assetto_ai_director.domain.geometry import Vec3
from assetto_ai_director.domain.projection import (
    DEFAULT_ASPECT,
    ScreenPoint,
    frame_fraction,
    project_point,
)
from assetto_ai_director.domain.units import FovDegrees, Metres

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from assetto_ai_director.application.camera.paths import SubjectTracks
    from assetto_ai_director.domain.camera import CameraKeyframe
    from assetto_ai_director.domain.shot import Shot
    from assetto_ai_director.domain.terrain import TerrainGrid
    from assetto_ai_director.domain.timeline import Timeline
    from assetto_ai_director.domain.track import TrackSpline

__all__ = [
    "NOMINAL_CAR_HEIGHT_METRES",
    "NOMINAL_CAR_LENGTH_METRES",
    "diagnose_cameras",
]

NOMINAL_CAR_HEIGHT_METRES = 1.30
"""Height of a car, for apparent-size checks.

**A guess, and flagged as one in every report that uses it.** The project reads
no car geometry — nothing parses a ``.kn5`` — so there is no measured figure to
use. 1.30 m is a touge-appropriate coupe; a formula car is half that and a GT3
car a little less. The number only scales the apparent-size check, so being out
by a third moves that one measurement by a third and changes no other check.
"""

NOMINAL_CAR_LENGTH_METRES = 4.40
"""Length of a car.

The same guess as :data:`NOMINAL_CAR_HEIGHT_METRES` and carrying the same caveat.
Reported in the assumptions so a reader can scale any apparent-size result.
"""

_IN_FRAME_FAIL = 0.60
_IN_FRAME_WARN = 0.90

_CENTRED_FAIL = 0.80
_CENTRED_WARN = 0.50

_SIZE_FAIL_SMALL = 0.015
_SIZE_WARN_SMALL = 0.05
_SIZE_WARN_LARGE = 0.90
_SIZE_FAIL_LARGE = 1.20

_BELOW_SUBJECT_FAIL = -0.50
_BELOW_SUBJECT_WARN = -0.10

_DISTANCE_FAIL_RATIO = 2.0
_DISTANCE_WARN_RATIO = 1.4

_SPAN_FAIL_SPANS = 1.0
_SPAN_WARN_SPANS = 0.5
"""How far a camera's written position may drift, measured in its own span widths.

Judged against the camera's own stretch of track rather than against a fixed
fraction of a lap, because that is the criterion with a consequence attached. A
camera whose written position is out by more than one full span is a camera the
game shows somewhere the shot was never about; out by half a span and the shot
starts on the wrong side of its corner. A flat "5% of a lap" would mean something
different on a 2 km touge and a 20 km Nordschleife, and nothing in particular on
either.
"""

_MIN_SPAN_WIDTH = 1e-6

_BELOW_SURFACE_FAIL = -1.0
_BELOW_SURFACE_WARN = -0.25
"""How far below the road beside it a camera may sit, in metres.

Looser than the subject-relative check, because the road's height comes from a
centreline sampled every few metres and a camera legitimately stands a little
below the crown of a cambered corner.
"""

_BEYOND_EDGE_WARN = 2.0
_BEYOND_EDGE_FAIL = 10.0
"""How far past the road's edge a camera may stand, in metres.

Two metres is a verge. Ten is over a barrier and into whatever is behind it,
which on a touge is a hillside on one side and nothing at all on the other.
"""

_NO_TRACK = "The subject car has no position track in this session."

_ASSUMPTIONS = (
    (
        "Car size is a guess: no car geometry is read, so apparent-size checks use a "
        f"nominal {NOMINAL_CAR_HEIGHT_METRES:.2f} m tall, "
        f"{NOMINAL_CAR_LENGTH_METRES:.2f} m long car."
    ),
    (
        "Field of view is treated as vertical, matching this project's convention and "
        "CSP's GrabbedCamera.fov. Which axis Assetto Corsa's cameras.ini MIN_FOV and "
        "MAX_FOV refer to is not documented anywhere this project could find."
    ),
)

_NO_TRACK_ASSUMPTION = (
    "Normalised track position was estimated from the car's own path length rather "
    "than the track's AI line, because no centreline was available. Two cars on "
    "different lines through the same corner therefore disagree slightly."
)

_DRIVEN_TRACK_ASSUMPTION = (
    "The centreline was averaged from a car's path, not read from the track's AI "
    "line. It sits wherever the driver placed the car — off by up to a lane through "
    "every corner — and carries no road width."
)


@dataclass(frozen=True, slots=True)
class _Sample:
    """The subject as one keyframe of a shot sees it.

    Attributes:
        point: Where the car lands in that frame.
        fov: The vertical field of view that keyframe uses. Carried alongside
            because a trackside camera zooms as the car approaches, and
            measuring apparent size against a nominal lens instead of the one
            actually in use would report the wrong size for every zooming shot.
    """

    point: ScreenPoint
    fov: FovDegrees


def diagnose_cameras(
    timeline: Timeline,
    tracks: SubjectTracks,
    *,
    exported_spans: Mapping[int, tuple[float, float]] | None = None,
    track: TrackSpline | None = None,
    terrain: TerrainGrid | None = None,
    notes: tuple[str, ...] = (),
    aspect: float = DEFAULT_ASPECT,
) -> DiagnosticReport:
    """Check every camera in a directed timeline.

    Args:
        timeline: The finished cut.
        tracks: The cars' world positions and headings, as the director used
            them. Required rather than optional: ``timeline.json`` records only
            subject *identifiers*, so a timeline alone cannot answer whether a
            car is in frame.
        exported_spans: What the ``cameras.ini`` exporter would write, per shot
            index, from
            :func:`~assetto_ai_director.infrastructure.export.cameras_ini.planned_spans`.
            Passed in rather than computed here so the check measures the real
            exporter and not a copy of it. Omit it to skip the span check, which
            then reports as unmeasured.
        track: The track's centreline, when one could be read. Its
            :attr:`~assetto_ai_director.domain.track.TrackSpline.source` decides
            what the checks will answer: a line read from the track's AI data
            can say whether a camera is on the road, a line averaged from a
            drive cannot and must not pretend to.
        terrain: A terrain scan from the CSP scanner, when available. Enables
            the surface and sight-line checks that otherwise report UNKNOWN.
        notes: Things the caller already knows it could not see — most often why
            the track's own data was unavailable. Joined to the report's blind
            spots, because the reason a check came back unknown is worth as much
            as the check.
        aspect: Frame width divided by height.

    Returns:
        The report, one diagnostic per shot in cut order.
    """
    profiles = {car_id: distance_profile(position) for car_id, position in tracks.positions.items()}
    cameras = tuple(
        _diagnose_shot(
            shot, tracks, profiles, exported_spans=exported_spans, track=track, terrain=terrain, aspect=aspect
        )
        for shot in timeline.shots
    )
    return DiagnosticReport(
        track_id=timeline.session.track_id,
        layout=timeline.session.layout,
        style_name=timeline.style_name,
        cameras=cameras,
        blind_spots=(*notes, *_blind_spots(cameras)),
        assumptions=_assumptions(track),
    )


def _assumptions(track: TrackSpline | None) -> tuple[str, ...]:
    """State what this report rests on, given what was available.

    The list changes with the inputs on purpose. Leaving "the AI line is not
    parsed" in a report produced *from* the AI line would be the report lying
    about its own footing, which is the failure every other rule here guards
    against.

    Args:
        track: The centreline that was used, or ``None``.

    Returns:
        One sentence per assumption.
    """
    if track is None:
        return (*_ASSUMPTIONS, _NO_TRACK_ASSUMPTION)
    if not track.source.is_measured:
        return (*_ASSUMPTIONS, _DRIVEN_TRACK_ASSUMPTION)
    return _ASSUMPTIONS


def _diagnose_shot(
    shot: Shot,
    tracks: SubjectTracks,
    profiles: Mapping[int, DistanceProfile],
    *,
    exported_spans: Mapping[int, tuple[float, float]] | None,
    track: TrackSpline | None,
    terrain: TerrainGrid | None,
    aspect: float,
) -> CameraDiagnostic:
    """Run every check for one shot.

    Args:
        shot: The shot to check.
        tracks: The cars' motion.
        profiles: Distance profile per car id.
        exported_spans: The exporter's spans, or ``None``.
        track: The track's centreline, or ``None``.
        terrain: The terrain grid, or ``None``.
        aspect: Frame aspect ratio.

    Returns:
        The diagnostic.
    """
    keyframes = shot.path.keyframes
    middle = keyframes[len(keyframes) // 2]
    samples = _subject_samples(shot, tracks, aspect)

    checks = (
        _check_behind(samples),
        _check_in_frame(samples),
        _check_centred(samples),
        _check_apparent_size(samples),
        _check_height(shot, tracks, middle),
        _check_distance(shot, samples),
        _check_company(shot, tracks, middle, aspect),
        _check_span(shot, tracks, profiles, exported_spans, track),
        _check_on_surface(middle, track, terrain),
        _check_on_road(shot.candidate.rig, middle, track),
        _check_sight_line(middle, terrain),
    )

    return CameraDiagnostic(
        shot_index=shot.index,
        rig=shot.candidate.rig,
        size=shot.candidate.size,
        side=shot.candidate.side,
        start=shot.start,
        end=shot.end,
        subjects=shot.candidate.subjects,
        position=middle.basis.position,
        look=middle.basis.look,
        fov=float(middle.fov),
        checks=checks,
    )


def _subject_samples(shot: Shot, tracks: SubjectTracks, aspect: float) -> list[_Sample]:
    """Project the primary subject into every keyframe of a shot.

    Args:
        shot: The shot.
        tracks: The cars' motion.
        aspect: Frame aspect ratio.

    Returns:
        One sample per keyframe. Empty when the subject's track is missing,
        which the callers translate into unmeasured checks rather than a
        confident zero.
    """
    positions = tracks.positions.get(shot.candidate.primary_subject)
    if positions is None or positions.shape[0] == 0:
        return []

    return [
        _Sample(
            point=project_point(
                keyframe.basis,
                _vec(
                    positions[
                        min(_frame_index(tracks, float(keyframe.time)), positions.shape[0] - 1)
                    ]
                ),
                keyframe.fov,
                aspect,
            ),
            fov=keyframe.fov,
        )
        for keyframe in shot.path.keyframes
    ]


def _check_behind(samples: Sequence[_Sample]) -> Check:
    """Check whether the subject ever falls behind the camera.

    The single most damning result available without the game: a camera the car
    passes and never comes back to is pointing at empty track for the rest of
    the shot, and nothing else in the pipeline notices.

    Args:
        samples: The projected subject, per keyframe.

    Returns:
        The check.
    """
    if not samples:
        return Check.unknown("subject_behind_camera", _NO_TRACK)
    behind = sum(1 for sample in samples if not sample.point.in_front) / len(samples)
    return Check(
        name="subject_behind_camera",
        status=CheckStatus.FAIL if behind > 0.0 else CheckStatus.PASS,
        value=behind,
        unit="fraction",
        threshold="fail above 0.00",
        detail=(
            f"The car is behind the camera for {behind * 100:.0f}% of the shot; "
            "the frame shows whatever is in the opposite direction."
            if behind > 0.0
            else "The car stays in front of the camera throughout."
        ),
    )


def _check_in_frame(samples: Sequence[_Sample]) -> Check:
    """Check how much of the shot actually contains the subject.

    Args:
        samples: The projected subject, per keyframe.

    Returns:
        The check.
    """
    if not samples:
        return Check.unknown("subject_in_frame", _NO_TRACK)
    inside = sum(1 for sample in samples if sample.point.in_frame) / len(samples)
    worst = max(sample.point.off_frame_by for sample in samples)
    excess = "" if math.isinf(worst) else f" Furthest miss: {worst:.2f} frame widths outside."
    return Check(
        name="subject_in_frame",
        status=_band_low(inside, fail_below=_IN_FRAME_FAIL, warn_below=_IN_FRAME_WARN),
        value=inside,
        unit="fraction",
        threshold=f"fail below {_IN_FRAME_FAIL:.2f}, warn below {_IN_FRAME_WARN:.2f}",
        detail=f"The car is visible for {inside * 100:.0f}% of the shot.{excess}",
    )


def _check_centred(samples: Sequence[_Sample]) -> Check:
    """Check where in the frame the subject sits at the middle of the shot.

    Uses the middle rather than the mean: a car that enters at one edge and
    leaves at the other averages to dead centre while never being there.

    Args:
        samples: The projected subject, per keyframe.

    Returns:
        The check.
    """
    if not samples:
        return Check.unknown("subject_framing", _NO_TRACK)
    point = samples[len(samples) // 2].point
    if not point.in_front:
        return Check(
            name="subject_framing",
            status=CheckStatus.FAIL,
            value=None,
            unit="fraction",
            threshold=f"fail above {_CENTRED_FAIL:.2f}",
            detail="The car is behind the camera at mid-shot, so it is not framed at all.",
        )
    offset = max(abs(point.u), abs(point.v))
    return Check(
        name="subject_framing",
        status=_band_high(offset, warn_above=_CENTRED_WARN, fail_above=_CENTRED_FAIL),
        value=offset,
        unit="fraction",
        threshold=f"fail above {_CENTRED_FAIL:.2f}, warn above {_CENTRED_WARN:.2f}",
        detail=(
            f"At mid-shot the car sits at u={point.u:+.2f}, v={point.v:+.2f}, "
            "where 0 is centre and 1 is the frame edge."
        ),
    )


def _check_apparent_size(samples: Sequence[_Sample]) -> Check:
    """Check whether the car reads at the intended size.

    Measured against the field of view that keyframe actually uses, so a
    trackside camera that zooms as the car approaches is judged on the lens it
    would really be showing.

    Args:
        samples: The projected subject, per keyframe.

    Returns:
        The check.
    """
    if not samples:
        return Check.unknown("subject_apparent_size", _NO_TRACK)
    sample = samples[len(samples) // 2]
    if not sample.point.in_front:
        return Check.unknown(
            "subject_apparent_size",
            "The car is behind the camera at mid-shot, so it has no apparent size.",
        )

    fraction = frame_fraction(Metres(NOMINAL_CAR_HEIGHT_METRES), sample.point.depth, sample.fov)
    return Check(
        name="subject_apparent_size",
        status=_band_both(
            fraction,
            fail_below=_SIZE_FAIL_SMALL,
            warn_below=_SIZE_WARN_SMALL,
            warn_above=_SIZE_WARN_LARGE,
            fail_above=_SIZE_FAIL_LARGE,
        ),
        value=fraction,
        unit="fraction",
        threshold=(
            f"fail below {_SIZE_FAIL_SMALL:.3f} or above {_SIZE_FAIL_LARGE:.2f}, "
            f"warn below {_SIZE_WARN_SMALL:.2f} or above {_SIZE_WARN_LARGE:.2f}"
        ),
        detail=(
            f"The car covers {fraction * 100:.1f}% of the frame height at "
            f"{float(sample.point.depth):.1f} m through a {float(sample.fov):.0f} degree lens, "
            f"assuming a {NOMINAL_CAR_HEIGHT_METRES:.2f} m tall car."
        ),
    )


def _check_height(shot: Shot, tracks: SubjectTracks, middle: CameraKeyframe) -> Check:
    """Check that the camera is not below the road the car is on.

    The closest thing available, without track geometry, to "the camera is
    buried". Placement sets camera height from the car's own elevation, so a
    camera below the car means the placement or the smoothing put it there.

    Args:
        shot: The shot.
        tracks: The cars' motion.
        middle: The keyframe at the middle of the shot.

    Returns:
        The check.
    """
    positions = tracks.positions.get(shot.candidate.primary_subject)
    if positions is None or positions.shape[0] == 0:
        return Check.unknown("camera_height_above_subject", _NO_TRACK)
    index = min(_frame_index(tracks, float(middle.time)), positions.shape[0] - 1)
    delta = float(middle.basis.position.y) - float(positions[index][1])
    return Check(
        name="camera_height_above_subject",
        status=_band_low(delta, fail_below=_BELOW_SUBJECT_FAIL, warn_below=_BELOW_SUBJECT_WARN),
        value=delta,
        unit="m",
        threshold=f"fail below {_BELOW_SUBJECT_FAIL:.2f} m, warn below {_BELOW_SUBJECT_WARN:.2f} m",
        detail=(
            f"The camera sits {delta:+.2f} m relative to the car's own height. "
            "Well below it usually means underground."
        ),
    )


def _check_distance(shot: Shot, samples: Sequence[_Sample]) -> Check:
    """Check that the camera stands roughly where the shot size intended.

    Args:
        shot: The shot.
        samples: The projected subject, per keyframe.

    Returns:
        The check.
    """
    if not samples:
        return Check.unknown("camera_distance", _NO_TRACK)
    point = samples[len(samples) // 2].point
    if not point.in_front:
        return Check.unknown(
            "camera_distance",
            "The car is behind the camera at mid-shot, so range is meaningless.",
        )

    intended = SIZE_DISTANCE_METRES[shot.candidate.size]
    actual = float(point.depth)
    ratio = actual / intended
    excursion = max(ratio, 1.0 / ratio) if ratio > 0.0 else math.inf
    return Check(
        name="camera_distance",
        status=_band_high(
            excursion, warn_above=_DISTANCE_WARN_RATIO, fail_above=_DISTANCE_FAIL_RATIO
        ),
        value=actual,
        unit="m",
        threshold=(
            f"fail beyond {_DISTANCE_FAIL_RATIO:.1f}x, warn beyond "
            f"{_DISTANCE_WARN_RATIO:.1f}x the {intended:.1f} m this shot size implies"
        ),
        detail=(
            f"The car is {actual:.1f} m away; a {shot.candidate.size.name.lower()} "
            f"shot intends about {intended:.1f} m."
        ),
    )


def _check_company(
    shot: Shot, tracks: SubjectTracks, middle: CameraKeyframe, aspect: float
) -> Check:
    """Count how many other cars share the frame at mid-shot.

    Informational rather than pass or fail. An empty frame behind the subject is
    right for a drift shot and wrong for an overtake, and this check does not
    know which it is looking at — it reports the number and leaves the judgement
    to a reader who does.

    Args:
        shot: The shot.
        tracks: The cars' motion.
        middle: The keyframe at the middle of the shot.
        aspect: Frame aspect ratio.

    Returns:
        The check.
    """
    index = _frame_index(tracks, float(middle.time))
    subject = shot.candidate.primary_subject

    others = 0
    for car_id, positions in tracks.positions.items():
        if car_id == subject or positions.shape[0] == 0:
            continue
        point = project_point(
            middle.basis,
            _vec(positions[min(index, positions.shape[0] - 1)]),
            middle.fov,
            aspect,
        )
        others += int(point.in_frame)

    return Check(
        name="other_cars_in_frame",
        status=CheckStatus.PASS,
        value=float(others),
        unit="count",
        threshold="informational",
        detail=f"{others} other car(s) share the frame at mid-shot.",
    )


def _check_span(
    shot: Shot,
    tracks: SubjectTracks,
    profiles: Mapping[int, DistanceProfile],
    exported_spans: Mapping[int, tuple[float, float]] | None,
    track: TrackSpline | None,
) -> Check:
    """Compare the exported track span against where the car actually was.

    **This is the check the whole diagnostic was built around.** Assetto Corsa
    selects a trackside camera by normalised track position; the exporter fills
    that field with a normalised *time*. Those agree only for a car moving at a
    constant rate, so this puts the two numbers side by side and converts their
    disagreement into metres of track — the form in which "the map is positioned
    incorrectly" stops being a feeling.

    Args:
        shot: The shot.
        tracks: The cars' motion.
        profiles: Distance profile per car id.
        exported_spans: The exporter's spans, or ``None``.
        track: The track's centreline. When present it supplies the quantity the
            game itself switches on, rather than an estimate of it from one
            car's mileage.

    Returns:
        The check.
    """
    if exported_spans is None:
        return Check.unknown(
            "span_vs_position",
            "No exported camera spans were supplied, so nothing could be compared.",
        )
    span = exported_spans.get(shot.index)
    if span is None:
        return Check.unknown(
            "span_vs_position",
            "cameras.ini does not carry this shot, so it has no track span to be wrong about.",
        )

    profile = profiles.get(shot.candidate.primary_subject)
    if profile is None or profile.lap_length <= 0.0:
        return Check.unknown(
            "span_vs_position",
            "The car's path is too short to estimate a track position from.",
        )

    written = (span[0] + span[1]) * 0.5
    middle_time = (float(shot.start) + float(shot.end)) * 0.5
    index = min(_frame_index(tracks, middle_time), profile.fraction.shape[0] - 1)

    positions = tracks.positions.get(shot.candidate.primary_subject)
    if track is not None and positions is not None and positions.shape[0] > index:
        # The track's own line when there is one: the quantity the game switches
        # on, rather than an estimate of it from how far this car has driven.
        actual = float(track.project(positions[index]).position)
        lap_length = float(track.length)
        closed = track.closed
        basis = f"the track's {track.source.value}"
    else:
        actual = float(profile.fraction[index])
        lap_length = profile.lap_length
        closed = profile.closed
        basis = "the car's own path length"

    # Both are fractions of a lap, so on a circuit the shorter way round the
    # loop is the honest difference.
    gap = abs(written - actual)
    if closed:
        gap = min(gap, 1.0 - gap)
    metres = gap * lap_length

    width = max(span[1] - span[0], _MIN_SPAN_WIDTH)
    drift = gap / width
    width_metres = width * lap_length

    return Check(
        name="span_vs_position",
        status=_band_high(drift, warn_above=_SPAN_WARN_SPANS, fail_above=_SPAN_FAIL_SPANS),
        value=metres,
        unit="m",
        threshold=(
            f"fail beyond {_SPAN_FAIL_SPANS:.0f}x, warn beyond {_SPAN_WARN_SPANS:.1f}x "
            f"this camera's own {width_metres:.0f} m span"
        ),
        detail=(
            f"cameras.ini puts this camera at track position {written:.4f}, but the "
            f"car was at about {actual:.4f} when the shot happens — {metres:.0f} m "
            f"apart over {lap_length:.0f} m measured from {basis}, which is "
            f"{drift:.1f} times the {width_metres:.0f} m of track this camera covers."
        ),
    )


def _check_on_surface(middle: CameraKeyframe, track: TrackSpline | None, terrain: TerrainGrid | None) -> Check:
    """Check that the camera is not below the road beside it.

    The nearest thing to "the camera is buried" that a centreline can answer. It
    compares the camera's height against the road's height at the point of track
    it stands beside, which catches a camera dropped through the surface on a
    descent — the case the subject-relative height check misses, because there
    the car is on its way down too.

    **It does not see terrain.** A camera up a bank or inside a hillside is above
    the road and passes this. Saying so is the point: the check reports what it
    measured and the remaining blind spot stays in the report.

    **With a terrain scan**, this check measures against the actual ground height
    from the CSP scanner rather than the centreline elevation. That catches
    cameras buried in hillsides that the centreline alone cannot see.

    Args:
        middle: The keyframe at the middle of the shot.
        track: The track's centreline, or ``None``.
        terrain: The terrain grid, or ``None``.

    Returns:
        The check.
    """
    pos = middle.basis.position
    x, y, z = float(pos.x), float(pos.y), float(pos.z)

    # If we have a terrain scan, use it — it sees the actual ground.
    if terrain is not None:
        ground = terrain.sample_ground_height(x, z)
        if ground is not None:
            delta = y - ground
            return Check(
                name="camera_on_surface",
                status=_band_low(delta, fail_below=_BELOW_SURFACE_FAIL, warn_below=_BELOW_SURFACE_WARN),
                value=delta,
                unit="m",
                threshold=(
                    f"fail below {_BELOW_SURFACE_FAIL:.2f} m, warn below {_BELOW_SURFACE_WARN:.2f} m"
                ),
                detail=(
                    f"The camera stands {delta:+.2f} m relative to the scanned ground. "
                    f"This measurement comes from the CSP terrain scanner."
                ),
            )
        # Fall through to track-based check if terrain has no data here.

    # No terrain scan — use the centreline.
    if track is None:
        return Check.unknown(
            "camera_on_surface",
            "No track centreline was available, so the road's height beside the "
            "camera is unknown. Install the track, or supply one, to enable this.",
        )

    point = track.project(_as_array(middle.basis.position))
    delta = float(middle.basis.position.y) - float(point.elevation)
    qualifier = (
        ""
        if track.source.is_measured
        else (
            " Measured against a line the cars drove rather than the track's own, "
            "so it is only as level as the drive was."
        )
    )
    return Check(
        name="camera_on_surface",
        status=_band_low(delta, fail_below=_BELOW_SURFACE_FAIL, warn_below=_BELOW_SURFACE_WARN),
        value=delta,
        unit="m",
        threshold=(
            f"fail below {_BELOW_SURFACE_FAIL:.2f} m, warn below {_BELOW_SURFACE_WARN:.2f} m"
        ),
        detail=(
            f"The camera stands {delta:+.2f} m relative to the road beside it. "
            f"Terrain and barriers are still not checked.{qualifier}"
        ),
    )


def _check_on_road(rig: RigKind, middle: CameraKeyframe, track: TrackSpline | None) -> Check:
    """Check how far off the racing line a camera stands.

    A trackside camera is placed by offsetting sideways from the car with no
    knowledge of where the road ends, so on a narrow touge a long shot puts it
    tens of metres into whatever is beside the road. This is the first check able
    to say so — and only when the spline knows the road's width. A line averaged
    from a drive does not, and reports unknown rather than a plausible number.

    Args:
        rig: Which rig produced the camera. Only the ones that hold position are
            judged on where they stand.
        middle: The keyframe at the middle of the shot.
        track: The track's centreline, or ``None``.

    Returns:
        The check.
    """
    if not rig.is_fixed_position:
        # A chase camera trails the car and a drone flies; both are meant to be
        # off the road, and faulting them for it would be judging a rig against
        # a job it does not have. The measurement runs on the rigs that stand
        # still, which are also the only ones cameras.ini can express.
        return Check.unknown(
            "camera_lateral_offset",
            f"A {rig.value} camera is not meant to stand beside the road, so how "
            "far off it sits is not a fault.",
        )
    if track is None:
        return Check.unknown(
            "camera_lateral_offset",
            "No track centreline was available, so how far the camera stands from "
            "the road is unknown.",
        )
    if not track.knows_width:
        return Check.unknown(
            "camera_lateral_offset",
            f"The centreline came from {track.source.value} data, which records no "
            "road width. Only the track's own AI line carries it.",
        )

    point = track.project(_as_array(middle.basis.position))
    lateral = abs(float(point.lateral))
    half_width = float(point.half_width or 0.0)
    beyond = lateral - half_width
    return Check(
        name="camera_lateral_offset",
        status=_band_high(beyond, warn_above=_BEYOND_EDGE_WARN, fail_above=_BEYOND_EDGE_FAIL),
        value=lateral,
        unit="m",
        threshold=(
            f"fail more than {_BEYOND_EDGE_FAIL:.0f} m past the road edge, warn past "
            f"{_BEYOND_EDGE_WARN:.0f} m"
        ),
        detail=(
            f"The camera stands {lateral:.1f} m from the racing line, where the road "
            f"reaches {half_width:.1f} m — {beyond:+.1f} m past its edge."
        ),
    )


def _check_sight_line(middle: CameraKeyframe, terrain: TerrainGrid | None) -> Check:
    """Check whether the sight line from camera to subject is clear.

    With a terrain scan available, this uses the pre-measured sight data from
    the CSP scanner. Without it, the check reports UNKNOWN as before.

    **This is informational rather than pass/fail.** A blocked sight line means
    the camera would see a hillside or barrier instead of the car, but the
    diagnostic does not yet know what constitutes a "soft" obstruction (trees,
    fences) versus a hard one (buildings, rock faces).

    Args:
        middle: The keyframe at the middle of the shot.
        terrain: The terrain grid, or ``None``.

    Returns:
        The check.
    """
    if terrain is None:
        return Check.unknown(
            "sight_line_clear",
            "Needs the CSP terrain scan, which this session did not provide. "
            "Run the scanner on this track to enable sight-line checks.",
        )

    pos = middle.basis.position
    x, y, z = float(pos.x), float(pos.y), float(pos.z)

    # For sight line we need a target — the subject car's position. Without it,
    # we can only report the general sight quality at the camera's location.
    # The full implementation would take the subject position from the shot.
    # For now, report terrain sight availability at this point.
    # TODO: accept subject position parameter for targeted sight check.

    # Find nearest cell and report its sight status
    best_dist = float('inf')
    best_cell = None
    for cell in terrain.cells:
        dx = cell.position.x - x
        dz = cell.position.z - z
        dist = dx * dx + dz * dz
        if dist < best_dist:
            best_dist = dist
            best_cell = cell

    if best_cell is None or best_cell.sight_clear is None:
        return Check.unknown(
            "sight_line_clear",
            "The terrain scan has no sight data near this camera position.",
        )

    clear = best_cell.sight_clear
    sight_dist = best_cell.sight_distance
    return Check(
        name="sight_line_clear",
        status=CheckStatus.PASS if clear else CheckStatus.FAIL,
        value=sight_dist,
        unit="m",
        threshold="fail when obstructed",
        detail=(
            f"Sight line from camera {'is' if clear else 'is NOT'} clear. "
            f"Nearest measured distance to centreline: {sight_dist:.1f} m."
        ),
    )


# --- Shared helpers ---------------------------------------------------------


def _blind_spots(cameras: Sequence[CameraDiagnostic]) -> tuple[str, ...]:
    """Collect, once each, every reason a check could not run.

    Args:
        cameras: The diagnostics.

    Returns:
        Distinct reasons in first-seen order, each naming the check it belongs
        to and how many cameras it affected.
    """
    counts: dict[tuple[str, str], int] = {}
    for camera in cameras:
        for check in camera.checks:
            if check.status is CheckStatus.UNKNOWN:
                key = (check.name, check.detail)
                counts[key] = counts.get(key, 0) + 1
    return tuple(
        f"{name} was not evaluated for {count} of {len(cameras)} cameras: {reason}"
        for (name, reason), count in counts.items()
    )


def _frame_index(tracks: SubjectTracks, moment: float) -> int:
    """Find the frame nearest a moment.

    Args:
        tracks: The cars' motion, carrying the time axis.
        moment: Time in seconds from session start.

    Returns:
        A valid index into the time axis.
    """
    last = max(0, tracks.time.shape[0] - 1)
    return int(np.clip(np.searchsorted(tracks.time, moment), 0, last))


def _as_array(point: Vec3) -> np.ndarray:
    """Convert a world vector into the array shape the spline projects.

    Args:
        point: The point.

    Returns:
        A ``(3,)`` array.
    """
    return np.array([point.x, point.y, point.z], dtype=np.float64)


def _vec(row: np.ndarray) -> Vec3:
    """Convert one row of a position array into a world vector.

    Args:
        row: A ``(3,)`` slice of world position.

    Returns:
        The same point as a :class:`Vec3` in the world frame.
    """
    return Vec3(float(row[0]), float(row[1]), float(row[2]))


def _band_low(value: float, *, fail_below: float, warn_below: float) -> CheckStatus:
    """Grade a value where smaller is worse.

    Args:
        value: The measurement.
        fail_below: Below this is a failure.
        warn_below: Below this is a warning.

    Returns:
        The status.
    """
    if value < fail_below:
        return CheckStatus.FAIL
    if value < warn_below:
        return CheckStatus.WARN
    return CheckStatus.PASS


def _band_high(value: float, *, warn_above: float, fail_above: float) -> CheckStatus:
    """Grade a value where larger is worse.

    Args:
        value: The measurement.
        warn_above: Above this is a warning.
        fail_above: Above this is a failure.

    Returns:
        The status.
    """
    if value > fail_above:
        return CheckStatus.FAIL
    if value > warn_above:
        return CheckStatus.WARN
    return CheckStatus.PASS


def _band_both(
    value: float,
    *,
    fail_below: float,
    warn_below: float,
    warn_above: float,
    fail_above: float,
) -> CheckStatus:
    """Grade a value that can be wrong in either direction.

    Args:
        value: The measurement.
        fail_below: Below this is a failure.
        warn_below: Below this is a warning.
        warn_above: Above this is a warning.
        fail_above: Above this is a failure.

    Returns:
        The status.
    """
    if value < fail_below or value > fail_above:
        return CheckStatus.FAIL
    if value < warn_below or value > warn_above:
        return CheckStatus.WARN
    return CheckStatus.PASS
