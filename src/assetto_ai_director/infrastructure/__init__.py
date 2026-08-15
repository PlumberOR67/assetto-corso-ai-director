"""The infrastructure layer: everything that touches the outside world.

File parsing, shared memory, exporters and plugin discovery live here. This is
the only layer allowed to know about byte layouts, file paths and external
libraries.

It depends on :mod:`assetto_ai_director.application` and
:mod:`assetto_ai_director.domain`, and implements their ports. Nothing in the
inner layers imports from here — dependencies point inward, and
``import-linter`` fails the build if that is ever violated.
"""

from __future__ import annotations
