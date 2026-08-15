"""Writing a camera diagnosis as a document that can be pasted into a conversation.

**The report is the deliverable, not the picture.** A person looking at the game
can see that a camera is wrong; they cannot easily say *how* wrong, and a
screenshot cannot be diffed between two runs. This document can: every check
carries the number it measured, so two reports subtract.

It is written to be read by both a person and an assistant, which is why it
leads with a summary and a list of what failed rather than burying those under
per-camera detail. A reader who stops after the first screen should already know
whether anything is wrong and, if so, what.

**The environment travels with it.** A report whose track-dependent checks are
all unknown is meaningless without knowing whether that is because the track has
no AI line or because Assetto Corsa was never found on the machine that produced
it. Those need opposite responses, so the snapshot from
:mod:`assetto_ai_director.infrastructure.ac_install` is embedded rather than left
to a separate command somebody has to remember to run.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from assetto_ai_director.domain.diagnostics import CheckStatus
from assetto_ai_director.infrastructure.ac_install import environment_snapshot

if TYPE_CHECKING:
    from pathlib import Path

    from assetto_ai_director.domain.diagnostics import (
        CameraDiagnostic,
        Check,
        DiagnosticReport,
    )
    from assetto_ai_director.domain.geometry import Vec3
    from assetto_ai_director.infrastructure.ac_install import EnvironmentSnapshot

__all__ = ["diagnostic_to_dict", "diagnostic_to_text", "write_diagnostic"]


def write_diagnostic(
    report: DiagnosticReport,
    destination: Path,
    *,
    ac_root: Path | None = None,
    notable_only: bool = False,
) -> None:
    """Write a diagnosis to a JSON file.

    Args:
        report: The diagnosis.
        destination: Where to write it. Parent directories are created.
        ac_root: An Assetto Corsa directory the user configured by hand, when
            discovery cannot find theirs.
        notable_only: Keep only the cameras worth looking at. See
            :func:`diagnostic_to_dict`.

    Raises:
        OSError: If the destination cannot be written.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        diagnostic_to_text(report, ac_root=ac_root, notable_only=notable_only),
        encoding="utf-8",
    )


def diagnostic_to_text(
    report: DiagnosticReport,
    *,
    ac_root: Path | None = None,
    notable_only: bool = False,
) -> str:
    """Render a diagnosis as the exact text that goes in a file or a clipboard.

    Separate from :func:`write_diagnostic` because the compact form's whole
    purpose is to be pasted, and a caller that has to write a file first only to
    read it back has been given the wrong tool.

    Args:
        report: The diagnosis.
        ac_root: An Assetto Corsa directory configured by hand.
        notable_only: Keep only the cameras worth looking at.

    Returns:
        The document as JSON text, newline-terminated.
    """
    document = diagnostic_to_dict(report, ac_root=ac_root, notable_only=notable_only)
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"


def diagnostic_to_dict(
    report: DiagnosticReport,
    *,
    ac_root: Path | None = None,
    notable_only: bool = False,
) -> dict[str, Any]:
    """Convert a diagnosis to a JSON-serialisable mapping.

    Args:
        report: The diagnosis.
        ac_root: An Assetto Corsa directory configured by hand.
        notable_only: Keep only the cameras whose measured checks came out as a
            warning or a failure. **The document is otherwise identical**, same
            schema and same summary counts, so a reader cannot mistake one form
            for the other — the summary still says how many cameras there were.
            Exists because the full form runs to a couple of hundred kilobytes
            for a three-minute cut, which is too much to paste into a
            conversation, and pasting it is how this report reaches anyone.

    Returns:
        The document.
    """
    environment = environment_snapshot(report.track_id, report.layout, explicit_root=ac_root)
    tally = report.tally()
    cameras = report.notable() if notable_only else report.cameras
    return {
        "schema_version": report.schema_version,
        "summary": {
            "status": report.status.value,
            # Reported alongside, never instead of. Two checks stay unmeasurable
            # until track geometry can be read, so `status` is at best "unknown"
            # for every camera and a reader would see a report where nothing
            # passes. This says whether anything that *could* be measured is
            # wrong; `status` keeps saying whether the cameras have been shown
            # to be good, which is a different and stronger claim.
            "measured_status": report.measured_status.value,
            "cameras": len(report.cameras),
            "failing": tally[CheckStatus.FAIL],
            "warning": tally[CheckStatus.WARN],
            "unmeasured": tally[CheckStatus.UNKNOWN],
            "passing": tally[CheckStatus.PASS],
        },
        "session": {
            "track": report.track_id,
            "layout": report.layout,
            "style": report.style_name,
        },
        "environment": _environment(environment),
        # Before the detail, not after: a reader who stops at the first screen
        # must still learn what the report could not see.
        "blind_spots": list(report.blind_spots) + list(environment.blind_spots),
        "assumptions": list(report.assumptions),
        # One glance says whether a problem is one bad camera or a policy that
        # is wrong everywhere, and those need different fixes.
        "check_tally": {
            name: {status.value: count for status, count in counts.items()}
            for name, counts in report.check_tally().items()
        },
        "failures": [_failure(camera) for camera in report.failing()],
        "cameras_included": "notable" if notable_only else "all",
        "cameras": [_camera(camera) for camera in cameras],
    }


def _environment(snapshot: EnvironmentSnapshot) -> dict[str, Any]:
    """Serialise what was found on this machine.

    Args:
        snapshot: The environment snapshot.

    Returns:
        The document fragment.
    """
    entry: dict[str, Any] = {
        "assetto_corsa_found": snapshot.install is not None,
    }
    if snapshot.install is not None:
        entry["assetto_corsa_root"] = str(snapshot.install.root)
        entry["found_by"] = snapshot.install.found_by
        entry["has_executable"] = snapshot.install.has_executable

    if snapshot.csp is None:
        entry["custom_shaders_patch"] = "not installed"
    elif snapshot.csp.values is None:
        # Distinguished from absence on purpose: reporting an unreadable
        # manifest as "not installed" is a confident false negative.
        entry["custom_shaders_patch"] = "installed, manifest unreadable"
    else:
        entry["custom_shaders_patch"] = snapshot.csp.values

    if snapshot.track is not None:
        entry["track_files"] = {
            "directory": str(snapshot.track.layout_dir),
            "present": snapshot.track.sizes(),
            "missing": list(snapshot.track.missing),
        }
    return entry


def _failure(camera: CameraDiagnostic) -> dict[str, Any]:
    """Summarise one failing camera, without its passing checks.

    Args:
        camera: The diagnostic.

    Returns:
        The document fragment.
    """
    return {
        "shot": camera.shot_index,
        "at": round(float(camera.start), 2),
        "rig": camera.rig.value,
        "failed": [
            {"check": check.name, "value": _number(check.value), "detail": check.detail}
            for check in camera.failed()
        ],
    }


def _camera(camera: CameraDiagnostic) -> dict[str, Any]:
    """Serialise one camera's full diagnostic.

    Args:
        camera: The diagnostic.

    Returns:
        The document fragment.
    """
    return {
        "shot": camera.shot_index,
        "status": camera.status.value,
        "start": round(float(camera.start), 3),
        "end": round(float(camera.end), 3),
        "rig": camera.rig.value,
        "size": camera.size.name.lower(),
        "side": camera.side.value,
        "subjects": list(camera.subjects),
        "position": _vec(camera.position),
        "look": _vec(camera.look),
        "fov": round(camera.fov, 2),
        "checks": [_check(check) for check in camera.checks],
    }


def _check(check: Check) -> dict[str, Any]:
    """Serialise one check.

    Args:
        check: The check.

    Returns:
        The document fragment.
    """
    return {
        "name": check.name,
        "status": check.status.value,
        "value": _number(check.value),
        "unit": check.unit,
        "threshold": check.threshold,
        "detail": check.detail,
    }


def _number(value: float | None) -> float | None:
    """Round a measurement, keeping ``None`` distinct from zero.

    A check that did not run reports ``None``. Rendering that as ``0.0`` would
    make an unmeasured value indistinguishable from a measured one that happened
    to be zero, which is the exact confusion the ``UNKNOWN`` status exists to
    prevent.

    Args:
        value: The measurement, or ``None``.

    Returns:
        The rounded value, or ``None``.
    """
    return None if value is None else round(value, 4)


def _vec(vector: Vec3) -> list[float]:
    """Serialise a vector as a three-element list.

    Args:
        vector: The vector.

    Returns:
        ``[x, y, z]`` rounded to millimetre precision.
    """
    return [round(vector.x, 3), round(vector.y, 3), round(vector.z, 3)]
