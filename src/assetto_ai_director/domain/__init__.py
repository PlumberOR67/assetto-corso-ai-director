"""The domain layer: pure data and pure functions.

This layer depends on nothing but the standard library and NumPy. It contains no
I/O, no Qt, no logging framework, no configuration parsing and no knowledge of
where telemetry comes from. ``import-linter`` enforces that mechanically.

The NumPy dependency is a deliberate, documented exception to "standard library
only". Telemetry is inherently columnar — a session is millions of samples across
tens of channels — and modelling it as lists of objects would misrepresent the
data and make the analysis layer unusably slow.
"""

from __future__ import annotations
