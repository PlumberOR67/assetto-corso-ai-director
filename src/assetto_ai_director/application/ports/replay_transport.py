"""The replay transport port — documenting an absence.

**No API for this was found to exist.** This module records the negative result
so it is searchable, rather than being rediscovered by everyone who reasonably
assumes a replay system must expose a way to seek.

What was searched: Kunos's official Python app reference contains no replay
functions at all. Custom Shaders Patch's Lua SDK — ``ac_gameplay.lua``,
``ac_gameplay_replaystream.lua``, ``ac_state.lua`` — exposes ``sim.isReplayActive``
but no accessor for playback position, frame index or seek. Shared memory carries
``graphics.replayTimeMultiplier``, but every library found treats it as read-only.
The CSP SDK's own comment on the replay-start callback says a record/playback API
"will be added later, hopefully".

The only lever that exists is launching the game with the replay flagged active
in its session configuration, which starts playback from the beginning and offers
no further control.

**The consequence for this project's design.** A camera timeline cannot be keyed
to replay time, because replay time is not observable. If in-game playback is ever
targeted, the timeline must instead be keyed to something a script *can* see —
the focused car's normalised spline position, or lap and sector, or a clock the
companion script starts for itself when playback begins. That is a real design
decision with real trade-offs, and it must be made deliberately rather than
discovered when a seek call turns out not to exist.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from assetto_ai_director.domain.units import Seconds

__all__ = ["ReplayTransport"]


@runtime_checkable
class ReplayTransport(Protocol):
    """Controls replay playback.

    No implementation ships with this project, and as far as could be
    established, none is possible. See the module docstring.
    """

    def position(self) -> Seconds:
        """Read the current playback position.

        Returns:
            Seconds from the start of the replay.
        """
        ...

    def seek(self, moment: Seconds) -> None:
        """Jump to a moment.

        Args:
            moment: Target position, in seconds from the start of the replay.
        """
        ...

    def set_speed(self, multiplier: float) -> None:
        """Change the playback rate.

        Args:
            multiplier: ``1.0`` for real time, below for slow motion.
        """
        ...
