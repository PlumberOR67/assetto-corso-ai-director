"""Reference frames, and the tagging of vectors with the frame they live in.

Assetto Corsa's replay format stores a position vector and a velocity vector per
car per frame. No source we could find states whether the velocity is expressed
in world axes or in the car's own axes, nor the handedness of either.

That ambiguity is not a detail. Sideslip — the quantity the entire drift analysis
rests on — is the angle between where the car points and where it is going. Get
the frame wrong and you get a confident number that is meaningless.

So vectors carry their frame in the type. A function that needs a car-local
velocity cannot silently accept a world-frame one, and a value whose frame was
never established is :attr:`ReferenceFrame.UNKNOWN` and must be resolved
explicitly before use.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["ReferenceFrame", "WheelIndex"]


class ReferenceFrame(StrEnum):
    """The coordinate frame a vector quantity is expressed in.

    Attributes:
        WORLD: Track-fixed axes. Positions are always in this frame.
        CAR_LOCAL: Axes fixed to the car body: x lateral, z longitudinal, y up,
            following the convention this codebase uses internally. A velocity in
            this frame yields sideslip directly as ``atan2(x, z)``.
        PATH: Frenet-like frame following the racing line: tangent, normal, up.
            Used for expressing camera offsets relative to direction of travel.
        UNKNOWN: The frame could not be established from any source. Values
            tagged this way must not be used in angular computations until a
            calibration step resolves them.
    """

    WORLD = "world"
    CAR_LOCAL = "car_local"
    PATH = "path"
    UNKNOWN = "unknown"


class WheelIndex(StrEnum):
    """Wheel ordering for per-wheel channels.

    The ordering is front-left, front-right, rear-left, rear-right, matching
    Assetto Corsa's own ``acsys.WHEELS`` constants where ``FL, FR, RL, RR`` are
    ``range(4)``.

    This is asserted at every provider boundary rather than assumed. A wheel-order
    mix-up is silent and catastrophic for per-corner analysis: swapping the rear
    pair inverts the sign of every drift measurement without changing its
    magnitude, so the numbers stay plausible while the conclusions invert.
    """

    FRONT_LEFT = "FL"
    FRONT_RIGHT = "FR"
    REAR_LEFT = "RL"
    REAR_RIGHT = "RR"

    @property
    def slot(self) -> int:
        """The positional index of this wheel in a 4-element channel array.

        Named ``slot`` rather than ``index`` because :class:`enum.StrEnum`
        inherits :meth:`str.index`, and shadowing it would change the meaning of
        an inherited method — the kind of quiet incompatibility that only
        surfaces once something calls the base method and gets nonsense.

        Returns:
            ``0`` for FL, ``1`` for FR, ``2`` for RL, ``3`` for RR.
        """
        return _WHEEL_ORDER.index(self)

    @property
    def is_rear(self) -> bool:
        """Whether this is a rear wheel.

        Drift angle is conventionally measured from the rear axle, so the rear
        pair is selected often enough to deserve a named predicate.

        Returns:
            ``True`` for the rear-left and rear-right wheels.
        """
        return self in (WheelIndex.REAR_LEFT, WheelIndex.REAR_RIGHT)


_WHEEL_ORDER: tuple[WheelIndex, ...] = (
    WheelIndex.FRONT_LEFT,
    WheelIndex.FRONT_RIGHT,
    WheelIndex.REAR_LEFT,
    WheelIndex.REAR_RIGHT,
)
