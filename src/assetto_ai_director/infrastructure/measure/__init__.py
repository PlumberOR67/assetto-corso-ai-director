"""Measurement of real reference footage.

Turns video into numbers the director can use: where the cuts fall, how long
shots run, and what the camera was doing. Requires the optional ``measure``
extra (OpenCV and PySceneDetect).

The point of this package is to replace guesses with measurements. Values it
produces are stamped
:attr:`~assetto_ai_director.domain.provenance.Provenance.MEASURED` and carry the
name of the footage they came from, so any number in a style profile can be
traced back to the frames that produced it.
"""

from __future__ import annotations
