"""Serialise style profiles to and from TOML.

TOML because a style profile is meant to be read, edited and reviewed by a
person, and to sit in version control where a diff shows what someone changed
about the look of a cut.

**Every tunable round-trips with its provenance.** A value written as measured
comes back as measured, carrying the clips it came from; a guess comes back
labelled a guess. Dropping provenance on the way through a file would let a
number quietly gain authority it never earned simply by being written down —
which is the failure this project is organised against.

Each tunable is emitted as an inline table so a profile stays readable::

    [rhythm]
    median_shot_length = { value = 2.11, provenance = "measured", source = "..." }

Reading uses the standard library's :mod:`tomllib`. Writing is done here rather
than through a third-party encoder: the schema is small, fixed and entirely under
this module's control — floats, integers, strings and lists of strings — so a
dependency would buy nothing but generality this file does not need. The one real
hazard, string escaping, is handled explicitly in :func:`_encode_string` and
covered by round-trip tests including quotes, backslashes and control characters.
"""

from __future__ import annotations

import tomllib
from typing import TYPE_CHECKING, Any

from assetto_ai_director.domain.camera import RigKind
from assetto_ai_director.domain.provenance import Provenance, Tunable
from assetto_ai_director.domain.scoring import ScoreWeights
from assetto_ai_director.domain.style import CameraMixProfile, RhythmProfile, StyleProfile

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["read_profile", "write_profile"]

SCHEMA_VERSION = 1
"""Bumped when the file shape changes incompatibly, so a reader can refuse rather
than misinterpret."""


def write_profile(profile: StyleProfile, destination: Path) -> None:
    """Write a style profile to a TOML file.

    Args:
        profile: The profile to write.
        destination: Where to write it. Parent directories are created.

    Raises:
        OSError: If the file cannot be written.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_encode_document(_to_document(profile)), encoding="utf-8")


def read_profile(source: Path) -> StyleProfile:
    """Read a style profile from a TOML file.

    Args:
        source: The file to read.

    Returns:
        The parsed profile.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If the schema version is unsupported or a field is missing.
        tomllib.TOMLDecodeError: If the file is not valid TOML.
    """
    with source.open("rb") as handle:
        document = tomllib.load(handle)

    version = document.get("schema_version")
    if version != SCHEMA_VERSION:
        message = (
            f"Unsupported style profile schema {version!r}; this build reads "
            f"version {SCHEMA_VERSION}."
        )
        raise ValueError(message)

    rhythm_doc = document["rhythm"]
    mix_doc = document["camera_mix"]
    weights_doc = document.get("weights", {})

    rhythm = RhythmProfile(
        median_shot_length=_tunable(rhythm_doc, "median_shot_length"),
        spread=_tunable(rhythm_doc, "spread"),
        spectral_slope=_tunable(rhythm_doc, "spectral_slope"),
        min_shot_length=_tunable(rhythm_doc, "min_shot_length"),
        max_shot_length=_tunable(rhythm_doc, "max_shot_length"),
    )

    allowed = mix_doc.get("allowed_rigs")
    camera_mix = CameraMixProfile(
        static_fraction=_tunable(mix_doc, "static_fraction"),
        axial_fraction=_tunable(mix_doc, "axial_fraction"),
        track_fraction=_tunable(mix_doc, "track_fraction"),
        pan_fraction=_tunable(mix_doc, "pan_fraction"),
        allowed_rigs=(
            frozenset(RigKind(name) for name in allowed) if allowed else frozenset(RigKind)
        ),
    )

    return StyleProfile(
        name=document["name"],
        description=document.get("description", ""),
        rhythm=rhythm,
        camera_mix=camera_mix,
        weights=ScoreWeights(**weights_doc) if weights_doc else ScoreWeights(),
        seed=int(document.get("seed", 0)),
        measured_from=tuple(document.get("measured_from", ())),
    )


def _to_document(profile: StyleProfile) -> dict[str, Any]:
    """Convert a profile to a TOML-serialisable mapping.

    Args:
        profile: The profile.

    Returns:
        The document.
    """
    weights = profile.weights
    return {
        "schema_version": SCHEMA_VERSION,
        "name": profile.name,
        "description": profile.description,
        "seed": profile.seed,
        "measured_from": list(profile.measured_from),
        "rhythm": {
            "median_shot_length": _dump(profile.rhythm.median_shot_length),
            "spread": _dump(profile.rhythm.spread),
            "spectral_slope": _dump(profile.rhythm.spectral_slope),
            "min_shot_length": _dump(profile.rhythm.min_shot_length),
            "max_shot_length": _dump(profile.rhythm.max_shot_length),
        },
        "camera_mix": {
            "static_fraction": _dump(profile.camera_mix.static_fraction),
            "axial_fraction": _dump(profile.camera_mix.axial_fraction),
            "track_fraction": _dump(profile.camera_mix.track_fraction),
            "pan_fraction": _dump(profile.camera_mix.pan_fraction),
            "allowed_rigs": sorted(rig.value for rig in profile.camera_mix.allowed_rigs),
        },
        "weights": {
            "drift": weights.drift,
            "speed": weights.speed,
            "risk": weights.risk,
            "flow": weights.flow,
            "cinematic": weights.cinematic,
            "legibility": weights.legibility,
        },
    }


def _encode_document(document: dict[str, Any]) -> str:
    """Render a document mapping as TOML text.

    Scalars and arrays are emitted first, then each nested table as its own
    ``[section]`` — TOML requires that ordering, since every key after a table
    header belongs to that table.

    Args:
        document: The mapping to render. Values may be ``int``, ``float``,
            ``str``, ``list[str]`` or a nested mapping one level deep.

    Returns:
        The TOML text, newline-terminated.
    """
    lines: list[str] = []
    tables: list[tuple[str, dict[str, Any]]] = []

    for key, value in document.items():
        if isinstance(value, dict):
            tables.append((key, value))
        else:
            lines.append(f"{key} = {_encode_value(value)}")

    for name, table in tables:
        lines.append("")
        lines.append(f"[{name}]")
        lines.extend(f"{key} = {_encode_value(value)}" for key, value in table.items())

    return "\n".join(lines) + "\n"


def _encode_value(value: Any) -> str:
    """Render a single TOML value.

    Args:
        value: A bool, int, float, string, list of scalars, or inline mapping.

    Returns:
        The encoded value.

    Raises:
        TypeError: If the value has no TOML representation here. Raising beats
            emitting something that parses but means the wrong thing.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        # repr round-trips exactly; ensure it still reads as a TOML float rather
        # than an integer, which would change the parsed type.
        text = repr(value)
        return text if ("." in text or "e" in text or "E" in text) else f"{text}.0"
    if isinstance(value, str):
        return _encode_string(value)
    if isinstance(value, list):
        return "[" + ", ".join(_encode_value(item) for item in value) + "]"
    if isinstance(value, dict):
        inner = ", ".join(f"{key} = {_encode_value(item)}" for key, item in value.items())
        return "{ " + inner + " }"
    message = f"Cannot encode {type(value).__name__} as TOML."
    raise TypeError(message)


def _encode_string(value: str) -> str:
    """Render a TOML basic string with the required escapes.

    Args:
        value: The string.

    Returns:
        The quoted, escaped string.
    """
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    escaped = (
        escaped.replace("\b", "\\b")
        .replace("\t", "\\t")
        .replace("\n", "\\n")
        .replace("\f", "\\f")
        .replace("\r", "\\r")
    )
    # Any remaining control character must be escaped as \uXXXX; TOML forbids
    # them raw inside a basic string.
    escaped = "".join(
        character if character >= " " or character == "\\" else f"\\u{ord(character):04X}"
        for character in escaped
    )
    return f'"{escaped}"'


def _dump(tunable: Tunable[float]) -> dict[str, Any]:
    """Serialise one tunable, provenance included.

    Args:
        tunable: The tunable.

    Returns:
        An inline-table mapping. Optional fields are omitted rather than written
        as empty strings, so a hand-edited file stays tidy.
    """
    entry: dict[str, Any] = {
        "value": float(tunable.value),
        "provenance": tunable.provenance.value,
    }
    if tunable.source:
        entry["source"] = tunable.source
    if tunable.note:
        entry["note"] = tunable.note
    return entry


def _tunable(section: dict[str, Any], key: str) -> Tunable[float]:
    """Parse one tunable from a TOML section.

    Accepts a bare number as shorthand for a guess. Hand-editing a profile to try
    a value should not require writing a provenance block — but the shorthand
    resolves to ``GUESS``, never to something more authoritative, so convenience
    cannot launder an unsourced number into a measured one.

    Args:
        section: The enclosing table.
        key: The tunable's key.

    Returns:
        The parsed tunable.

    Raises:
        ValueError: If the key is absent or the provenance is unrecognised.
    """
    if key not in section:
        message = f"Style profile is missing required field {key!r}."
        raise ValueError(message)

    entry = section[key]
    if isinstance(entry, int | float):
        return Tunable(float(entry), Provenance.GUESS, note="bare value in profile file")

    try:
        provenance = Provenance(entry["provenance"])
    except (KeyError, ValueError) as error:
        message = f"Field {key!r} has an unrecognised provenance: {entry!r}"
        raise ValueError(message) from error

    return Tunable(
        float(entry["value"]),
        provenance,
        source=entry.get("source"),
        note=entry.get("note"),
    )
