"""The MCP server: the whole pipeline, exposed to an MCP client.

Built against the official Model Context Protocol Python SDK. Note that the SDK's
2.0 release renamed the high-level entry point from ``FastMCP`` to ``MCPServer``
and moved it to ``mcp.server``, with no compatibility alias — code written
against the earlier name does not import. The version in use is pinned in
``pyproject.toml`` and verified at runtime by the test suite rather than assumed.
"""

from __future__ import annotations
