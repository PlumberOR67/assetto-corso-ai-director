"""Reading what a track knows about itself.

Three files, each closing a specific gap the camera diagnostic reports as
unmeasurable: ``fast_lane.ai`` for the centreline and the road's width,
``map.ini`` for the world-to-map transform, and ``cameras*.ini`` for a reference
camera set this project can finally check its own writer against.

Everything here degrades by declining rather than by guessing. A track with no
AI line, a map with no scale, a camera set with a section this reader does not
understand — each produces an absence a caller must handle, not a default that
looks like data.
"""

from __future__ import annotations

from assetto_ai_director.infrastructure.track.cameras_ini_read import (
    CameraSet,
    TrackCamera,
    read_cameras_ini,
)
from assetto_ai_director.infrastructure.track.fast_lane import read_fast_lane
from assetto_ai_director.infrastructure.track.map_ini import MapTransform, read_map_ini

__all__ = [
    "CameraSet",
    "MapTransform",
    "TrackCamera",
    "read_cameras_ini",
    "read_fast_lane",
    "read_map_ini",
]
