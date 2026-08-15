"""Typed exceptions.

Every failure mode this project can hit has a named exception. The reason is
specific rather than stylistic: the worst possible outcome for a telemetry tool
is not a crash, it is *plausible garbage*. A misaligned binary read produces
smooth, believable numbers that are entirely wrong. So the reader is built to
fail loudly and by name at the first sign of a broken assumption, and callers
are given exception types precise enough to explain the problem to a user.
"""

from __future__ import annotations

__all__ = [
    "AssettoDirectorError",
    "CalibrationFailedError",
    "ChannelUnavailableError",
    "DirectorError",
    "NoCandidatesError",
    "OperationCancelledError",
    "PluginError",
    "ReplayError",
    "StrideMismatchError",
    "TruncatedReplayError",
    "UnsupportedReplayVersionError",
]


class AssettoDirectorError(Exception):
    """Base class for every error raised by this package."""


# --- Replay reading ---------------------------------------------------------


class ReplayError(AssettoDirectorError):
    """Base class for failures while reading a replay file."""


class UnsupportedReplayVersionError(ReplayError):
    """The replay's version field is not one this reader can decode.

    The reader hard-gates on version 16. Versions 14 and 15 exist in the wild
    and have a different header layout; decoding them "best effort" without a
    test fixture would produce exactly the plausible-garbage failure this
    project refuses to ship.

    Attributes:
        version: The version number actually found in the file.
        supported: The versions this reader accepts.
    """

    def __init__(self, version: int, supported: tuple[int, ...]) -> None:
        self.version = version
        self.supported = supported
        supported_text = ", ".join(str(v) for v in supported)
        super().__init__(
            f"Replay version {version} is not supported; this reader decodes "
            f"version {supported_text} only. Refusing to guess at the layout."
        )


class TruncatedReplayError(ReplayError):
    """The file ends before a structure it declares would fit.

    Raised during bounds-checking, *before* any allocation. A replay that claims
    68771 frames of 17 cars must actually contain that many bytes; trusting the
    declared count and allocating first turns a corrupt file into an out-of-memory
    kill instead of a clear error message.

    Attributes:
        what: The structure being read when the shortfall was found.
        needed: Bytes required.
        available: Bytes actually remaining.
    """

    def __init__(self, what: str, needed: int, available: int) -> None:
        self.what = what
        self.needed = needed
        self.available = available
        super().__init__(
            f"Truncated replay: reading {what} needs {needed} bytes but only {available} remain."
        )


class StrideMismatchError(ReplayError):
    """A car's frame records did not consume the space they should have.

    This is the reader's self-verification failing, and it is the single most
    important error in the codebase. The per-frame car record is
    ``276 + 4 * numWings`` bytes, derived empirically rather than documented, so
    the reader checks its own arithmetic against the file: after consuming every
    declared frame for every car, it must land exactly on the offset recorded in
    the trailing Custom Shaders Patch footer.

    If that check fails, the layout assumption is wrong and every channel is
    silently shifted. Raising here is the difference between a tool that reports
    a problem and a tool that invents a race.

    Attributes:
        expected_end: Offset the parse should have finished at.
        actual_end: Offset it actually finished at.
        detail: Human-readable context.
    """

    def __init__(self, expected_end: int, actual_end: int, detail: str = "") -> None:
        self.expected_end = expected_end
        self.actual_end = actual_end
        self.detail = detail
        drift = actual_end - expected_end
        suffix = f" ({detail})" if detail else ""
        super().__init__(
            f"Frame stride verification failed: parse ended at {actual_end} but "
            f"should have ended at {expected_end} (off by {drift:+d} bytes){suffix}. "
            f"The record layout assumption is wrong; refusing to emit telemetry."
        )


# --- Analysis ---------------------------------------------------------------


class ChannelUnavailableError(AssettoDirectorError):
    """Code asked a telemetry source for a channel it does not carry.

    Sources declare their capabilities rather than promising a fixed struct.
    Assetto Corsa's shared memory has no per-wheel slip angle; the replay file
    does. Rather than let a provider fabricate a zero, asking for an absent
    channel is an error and callers are expected to check availability first.

    Attributes:
        channel: Name of the requested channel.
        source: Name of the source that does not provide it.
    """

    def __init__(self, channel: str, source: str) -> None:
        self.channel = channel
        self.source = source
        super().__init__(
            f"Channel {channel!r} is not available from source {source!r}. "
            f"Check availability() before reading; this source will not invent it."
        )


class CalibrationFailedError(AssettoDirectorError):
    """Yaw calibration could not establish the rotation convention.

    The replay stores rotation in a field whose Euler order, handedness and zero
    reference are undocumented. Sideslip is only meaningful once heading has been
    validated against path tangent on near-zero-slip frames. When that fit fails,
    the pipeline degrades to a rear-wheel slip-angle proxy rather than emitting a
    confident, wrong angle.

    Attributes:
        residual: The fit residual that failed the acceptance threshold.
        threshold: The threshold it had to meet.
    """

    def __init__(self, residual: float, threshold: float) -> None:
        self.residual = residual
        self.threshold = threshold
        super().__init__(
            f"Yaw calibration failed: residual {residual:.4f} exceeds threshold "
            f"{threshold:.4f}. Sideslip-derived features are not trustworthy for "
            f"this session."
        )


# --- Direction --------------------------------------------------------------


class DirectorError(AssettoDirectorError):
    """Base class for failures while directing a cut."""


class NoCandidatesError(DirectorError):
    """No shot candidate survived filtering, so no timeline can be built.

    Usually means the style's constraints excluded every rig, or the session
    contained no detectable events in the requested time range.

    Attributes:
        style: Name of the style profile in use.
        window: The (start, end) seconds that produced nothing.
    """

    def __init__(self, style: str, window: tuple[float, float]) -> None:
        self.style = style
        self.window = window
        super().__init__(
            f"No shot candidates for style {style!r} in window [{window[0]:.2f}, {window[1]:.2f}]s."
        )


# --- Control flow -----------------------------------------------------------


class OperationCancelledError(AssettoDirectorError):
    """The caller asked for the work to stop.

    **Not a failure**, and everything that catches it must treat it as such. A
    user who presses Cancel and is shown "Could not direct this replay" has been
    told their machine broke, when in fact it did exactly what they asked.

    It subclasses :class:`AssettoDirectorError` so existing handlers do not let
    it escape as an unhandled exception, which means every such handler must
    catch it explicitly *before* the general case.

    Attributes:
        stage: Which stage was interrupted, so the user can be told how far it
            got rather than merely that it stopped.
    """

    def __init__(self, stage: str) -> None:
        self.stage = stage
        super().__init__(f"Cancelled during {stage.lower()}.")


# --- Plugins ----------------------------------------------------------------


class PluginError(AssettoDirectorError):
    """A plugin could not be loaded or did not satisfy its protocol.

    Attributes:
        group: The entry-point group.
        name: The entry-point name.
    """

    def __init__(self, group: str, name: str, reason: str) -> None:
        self.group = group
        self.name = name
        super().__init__(f"Plugin {name!r} in group {group!r} failed to load: {reason}")
