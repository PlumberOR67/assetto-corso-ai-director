"""The PySide6 desktop application.

Qt lives only here. The application and domain layers know nothing about it, and
``import-linter`` enforces that — so the CLI, the MCP server and the test suite
all run without a graphical toolkit installed.
"""

from __future__ import annotations
