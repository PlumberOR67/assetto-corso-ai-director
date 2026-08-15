"""Checking a directed cut's cameras against what they would actually see.

The pipeline can place a camera but has never been able to say whether the
camera works. This package answers that in numbers, without the game, so a
defect can be described precisely enough to fix rather than shown in a
screenshot and argued about.
"""

from __future__ import annotations

from assetto_ai_director.application.diagnostics.checks import (
    NOMINAL_CAR_HEIGHT_METRES,
    NOMINAL_CAR_LENGTH_METRES,
    diagnose_cameras,
)

__all__ = [
    "NOMINAL_CAR_HEIGHT_METRES",
    "NOMINAL_CAR_LENGTH_METRES",
    "diagnose_cameras",
]
