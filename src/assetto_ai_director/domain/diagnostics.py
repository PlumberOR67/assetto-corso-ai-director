"""What a camera check found, and how sure it is.

The report these types describe is the project's answer to a specific problem:
the person who can see the cameras is not the person who can fix them. A
screenshot travels badly through a conversation and a screenshot cannot be
diffed. A list of named checks with numbers attached can be pasted, compared
between runs, and argued with.

**Four statuses, not three.** :attr:`CheckStatus.UNKNOWN` is the load-bearing
one. A check that could not run — because the track's AI line was not available,
or the rig is one this export cannot express — must not report ``PASS``. A green
report that is green because nothing was measured is worse than a red one, and
this project's whole posture is against exactly that.

**Every check carries the number it measured and the threshold it compared
against.** A status alone invites the reader to trust the thresholds, which are
mostly engineering judgement. Publishing the measurement lets a reader disagree
with the threshold and still use the report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from assetto_ai_director.domain.camera import RigKind
from assetto_ai_director.domain.geometry import Vec3
from assetto_ai_director.domain.shot import LineSide, ShotSize
from assetto_ai_director.domain.units import Seconds

__all__ = [
    "SCHEMA_VERSION",
    "CameraDiagnostic",
    "Check",
    "CheckStatus",
    "DiagnosticReport",
]

SCHEMA_VERSION = 1
"""Version of the report document this module describes.

Published in ``contracts/diagnostic.schema.json`` and checked against real
output in CI, so the format cannot drift away from what consumers were told.
"""


class CheckStatus(StrEnum):
    """How a single check came out.

    Attributes:
        PASS: Measured, and within the threshold.
        WARN: Measured, outside the comfortable range but not obviously broken.
            Worth a look when several land on the same camera.
        FAIL: Measured, and wrong. Something is visibly not what was intended.
        UNKNOWN: **Not measured.** The inputs the check needs were not
            available. Never treat this as a pass — it is the report saying it
            could not see, which is the one thing a diagnostic must never hide.
    """

    PASS = "pass"  # noqa: S105 - a check verdict, not a credential
    WARN = "warn"
    FAIL = "fail"
    UNKNOWN = "unknown"

    @property
    def is_measured(self) -> bool:
        """Whether the check actually ran.

        Returns:
            ``True`` for anything but :attr:`UNKNOWN`.
        """
        return self is not CheckStatus.UNKNOWN

    @property
    def severity(self) -> int:
        """Rank for choosing the worst status in a group.

        ``UNKNOWN`` ranks above ``PASS`` and below ``WARN``: it is not a
        success, but it is not evidence of a defect either.

        Returns:
            A comparable integer, larger being worse.
        """
        return _SEVERITY[self]


_SEVERITY = {
    CheckStatus.PASS: 0,
    CheckStatus.UNKNOWN: 1,
    CheckStatus.WARN: 2,
    CheckStatus.FAIL: 3,
}


@dataclass(frozen=True, slots=True)
class Check:
    """One named measurement and its verdict.

    Attributes:
        name: Stable identifier, snake case. Stable because reports get diffed
            between runs and a renamed check reads as a check that vanished.
        status: How it came out.
        value: What was measured, or ``None`` when the check did not run.
        unit: What ``value`` is in — ``"m"``, ``"fraction"``, ``"count"``,
            ``"degrees"``. Present so a reader never has to infer it from the
            name.
        threshold: The comparison in words, for example ``"fail below 0.60"``.
            Written out rather than stored as numbers, because several checks
            compare against a band rather than a single bound.
        detail: One sentence a person can act on. Not a restatement of the
            name — it says what the number means for the shot.
    """

    name: str
    status: CheckStatus
    value: float | None
    unit: str
    threshold: str
    detail: str

    @classmethod
    def unknown(cls, name: str, reason: str) -> Check:
        """Build a check that could not run.

        Args:
            name: The check's identifier.
            reason: Why it could not run, in words, ending in a full stop.

        Returns:
            An unmeasured check carrying the reason as its detail.
        """
        return cls(
            name=name,
            status=CheckStatus.UNKNOWN,
            value=None,
            unit="",
            threshold="not evaluated",
            detail=reason,
        )


@dataclass(frozen=True, slots=True)
class CameraDiagnostic:
    """Every check for one shot's camera.

    Attributes:
        shot_index: Position in the cut, matching ``Shot.index``.
        rig: Which camera behaviour produced it.
        size: The intended shot size.
        side: Which side of the line of action.
        start: When the shot begins, seconds from session start.
        end: When it ends.
        subjects: The cars the shot frames, most important first.
        position: Where the camera sits at the middle of the shot. Carried in
            the report so a reader can put it on a map, or paste it into a
            camera set by hand.
        look: The direction it faces at the middle of the shot.
        fov: Vertical field of view in degrees at that instant.
        checks: The measurements, in a fixed order so two reports diff cleanly.
    """

    shot_index: int
    rig: RigKind
    size: ShotSize
    side: LineSide
    start: Seconds
    end: Seconds
    subjects: tuple[int, ...]
    position: Vec3
    look: Vec3
    fov: float
    checks: tuple[Check, ...] = ()

    @property
    def status(self) -> CheckStatus:
        """The worst status among this camera's checks.

        Returns:
            The overall verdict, or :attr:`CheckStatus.UNKNOWN` when there are
            no checks at all — which is itself a thing worth seeing.
        """
        if not self.checks:
            return CheckStatus.UNKNOWN
        return max((check.status for check in self.checks), key=lambda s: s.severity)

    @property
    def measured_status(self) -> CheckStatus:
        """The worst status among the checks that actually ran.

        Reported *alongside* :attr:`status`, never instead of it. Two checks are
        unmeasurable until the track's geometry can be read, which means every
        camera's overall status is at best ``UNKNOWN`` and a reader sees a
        report where nothing passes. That is honest and also unreadable — this
        answers "of what could be measured, is anything wrong", and
        :attr:`status` keeps answering "has this camera been shown to be good".

        Returns:
            The worst measured status, or :attr:`CheckStatus.UNKNOWN` when
            nothing was measured at all.
        """
        measured = [check.status for check in self.checks if check.status.is_measured]
        if not measured:
            return CheckStatus.UNKNOWN
        return max(measured, key=lambda s: s.severity)

    def failed(self) -> tuple[Check, ...]:
        """The checks that came out as failures.

        Returns:
            Failing checks in their original order.
        """
        return tuple(check for check in self.checks if check.status is CheckStatus.FAIL)


@dataclass(frozen=True, slots=True)
class DiagnosticReport:
    """The whole camera diagnosis for one directed session.

    Attributes:
        schema_version: The document version, :data:`SCHEMA_VERSION`.
        track_id: The track the session was recorded on.
        layout: The track layout, empty when the track has none.
        style_name: The style the cut was directed in.
        cameras: One diagnostic per shot, in cut order.
        blind_spots: What could not be checked and why, in plain sentences.
            Populated from the ``UNKNOWN`` checks and from whatever the caller
            already knows it is missing. **A report with no failures and a long
            blind-spot list has not established that the cameras are good.**
        assumptions: Statements the report depends on that were not verified —
            the nominal car size, the axis the exported field of view refers to.
            Listed so a reader can weigh a result rather than take it on faith.
    """

    schema_version: int = SCHEMA_VERSION
    track_id: str = ""
    layout: str = ""
    style_name: str = ""
    cameras: tuple[CameraDiagnostic, ...] = ()
    blind_spots: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = field(default_factory=tuple)

    @property
    def status(self) -> CheckStatus:
        """The worst status across every camera.

        Returns:
            The overall verdict, or :attr:`CheckStatus.PASS` for an empty
            report — nothing was found wrong because nothing was examined, and
            the blind-spot list is where that is said.
        """
        if not self.cameras:
            return CheckStatus.PASS
        return max((camera.status for camera in self.cameras), key=lambda s: s.severity)

    @property
    def measured_status(self) -> CheckStatus:
        """The worst status across every camera, counting only checks that ran.

        Returns:
            The verdict on what could be measured. ``PASS`` here together with a
            long blind-spot list means "nothing measurable is wrong", which is a
            weaker claim than "the cameras are good" and must not be read as it.
        """
        if not self.cameras:
            return CheckStatus.PASS
        return max((camera.measured_status for camera in self.cameras), key=lambda s: s.severity)

    def tally(self) -> dict[CheckStatus, int]:
        """Count cameras by their overall status.

        Returns:
            Every status mapped to how many cameras carry it, including zeros,
            so a report always has the same shape.
        """
        counts = dict.fromkeys(CheckStatus, 0)
        for camera in self.cameras:
            counts[camera.status] += 1
        return counts

    def failing(self) -> tuple[CameraDiagnostic, ...]:
        """The cameras with at least one failed check.

        Returns:
            Failing cameras in cut order.
        """
        return tuple(camera for camera in self.cameras if camera.status is CheckStatus.FAIL)

    def notable(self) -> tuple[CameraDiagnostic, ...]:
        """The cameras a reader needs to look at.

        Every camera carries checks that cannot run until the track's geometry
        is readable, so "not fully passing" describes all of them and selects
        nothing. This selects on the *measured* verdict instead.

        Returns:
            Cameras whose measured checks came out as a warning or a failure, in
            cut order.
        """
        return tuple(
            camera
            for camera in self.cameras
            if camera.measured_status in (CheckStatus.WARN, CheckStatus.FAIL)
        )

    def check_tally(self) -> dict[str, dict[CheckStatus, int]]:
        """Count each named check by status, across every camera.

        The most informative few lines in a report: one glance says whether a
        problem is one bad camera or a policy that is wrong everywhere, and
        those need completely different fixes.

        Returns:
            Check name to a full status count, in the order the checks appear on
            a camera. Every status is present, including zeros, so the shape is
            stable between runs and two reports diff cleanly.
        """
        tally: dict[str, dict[CheckStatus, int]] = {}
        for camera in self.cameras:
            for check in camera.checks:
                counts = tally.setdefault(check.name, dict.fromkeys(CheckStatus, 0))
                counts[check.status] += 1
        return tally
