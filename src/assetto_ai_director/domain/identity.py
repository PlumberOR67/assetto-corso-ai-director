"""Car and session identity.

These fields come straight from the replay's per-car header, which stores five
length-prefixed strings: car model, driver name, nationality code, team name and
skin id. Empty strings are common — AI cars usually have no driver name and most
entries have no team — so every optional field is modelled as such rather than
being defaulted to a placeholder that would later be mistaken for real data.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["CarIdentity"]

_MAX_CONTENT_PREFIX_LENGTH = 3
"""Longest leading token treated as an author prefix, e.g. ``ks_`` or ``j8_``.

Anything longer is part of the car's actual name — ``ddm_mugen_civic_ek9`` keeps
``mugen_civic_ek9``, but ``mazda_rx7`` must not lose ``mazda``.
"""


@dataclass(frozen=True, slots=True)
class CarIdentity:
    """Who a car is, as recorded in the session.

    Attributes:
        car_id: Index within the session. Stable for the session's lifetime and
            the key every telemetry array is addressed by.
        model: Car folder name, e.g. ``ks_toyota_ae86_tuned``. Never empty.
        driver_name: Driver's display name, or ``None`` for an unnamed AI entry.
        nationality: Three-letter nation code such as ``RUS``, or ``None``.
        team: Team name, or ``None``. Usually absent outside league racing.
        skin_id: Livery folder name, e.g. ``09_panda_b``, or ``None``.
        is_player: Whether this is the human-driven car. The player's entry
            carries extra trailing records that AI entries do not, which is how
            the reader identifies it structurally rather than by guessing that
            car zero is always the player.
    """

    car_id: int
    model: str
    driver_name: str | None = None
    nationality: str | None = None
    team: str | None = None
    skin_id: str | None = None
    is_player: bool = False

    @property
    def display_name(self) -> str:
        """A human-readable label for logs, the GUI and shot metadata.

        Prefers the driver's name, falling back to the car model when a car has
        no named driver.

        Returns:
            The driver name if present, otherwise the car model, suffixed with
            the car id to keep entries distinguishable when several AI cars share
            a model — which is exactly what happens in a one-make field.
        """
        base = self.driver_name or self.model
        return f"{base} (#{self.car_id})"

    @property
    def short_model(self) -> str:
        """The car model without its content-pack prefix.

        Assetto Corsa folder names carry an author prefix — ``ks_`` for Kunos,
        and arbitrary prefixes for third-party content. Stripping it makes GUI
        labels readable without losing the full name, which stays in
        :attr:`model`.

        Returns:
            The model name with a leading ``<prefix>_`` removed when one is
            present, otherwise the model unchanged.
        """
        head, separator, tail = self.model.partition("_")
        if separator and len(head) <= _MAX_CONTENT_PREFIX_LENGTH:
            return tail
        return self.model
