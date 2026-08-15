"""Assetto AI Director — an AI cinematographer for Assetto Corsa.

The package is organised as a Clean Architecture with a strictly one-directional
dependency graph, mechanically enforced by ``import-linter``:

``presentation`` / ``mcp`` -> ``infrastructure`` -> ``application`` -> ``domain``

``domain`` depends on nothing but the standard library and NumPy. The NumPy
dependency is a deliberate, documented exception: telemetry is inherently
columnar and modelling it as anything else would be dishonest about the data.

Design rule that governs the whole codebase: **never invent an API and never
invent a file format**. Where the real world could not be established, this
package defines an interface and says so in the docstring, rather than shipping
a plausible-looking implementation that silently produces wrong numbers.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
