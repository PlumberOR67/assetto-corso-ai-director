"""The application layer: use cases, and the ports they talk through.

This layer owns the analysis and the direction — features, events, scoring, shot
selection, camera paths. It depends on :mod:`assetto_ai_director.domain` and on
nothing else in this package.

It is deliberately framework-free: no Qt, no MCP, no CLI framework, no file
formats. Everything the outside world provides arrives through a Protocol defined
in :mod:`assetto_ai_director.application.ports`, which is what lets the whole
director be exercised against a synthetic session with no game installed.
"""

from __future__ import annotations
