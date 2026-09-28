"""Lua syntax checker built on lupa (a LuaJIT binding).

Checks syntax with the real Lua parser, without running the code.

We write the code to a temporary file and call ``loadfile`` on it. This is the
only reliable way to check arbitrary Lua code through lupa without escaping
problems and without running the code.
"""

import contextlib
import os
import re
import tempfile
from pathlib import Path

try:
    from lupa import LuaRuntime
except ImportError:
    LuaRuntime = None


class LuaSyntaxError:
    """One syntax error that the Lua parser reported."""

    def __init__(self, line: int, message: str) -> None:
        self.line = line
        self.message = message

    def __repr__(self) -> str:
        """Return the error as one report line."""
        return f"Line {self.line}: {self.message}"


class LuaSyntaxChecker:
    """Checks Lua syntax with lupa (LuaJIT), the real Lua parser.

    What check_file returns:
      - An empty list: the syntax is correct.
      - A list of errors: syntax errors were found.
      - When lupa is not installed, check_file returns an empty list, and the
        caller checks is_available to report UNAVAILABLE.
    """

    def __init__(self) -> None:
        self._lua = None
        if LuaRuntime is not None:
            self._lua = LuaRuntime()

    @property
    def is_available(self) -> bool:
        """Whether lupa is installed, so the checks actually run."""
        return self._lua is not None

    def check_file(self, filepath: Path) -> list[LuaSyntaxError]:
        """Check one file with the real Lua parser.

        Args:
            filepath: The Lua file to check.

        Returns:
            An empty list when the syntax is correct (or lupa is not installed).
            A list of errors when syntax problems were found.
        """
        if not self.is_available:
            return []

        content = filepath.read_text(encoding="utf-8")
        return self._check(content)

    def check_content(self, content: str) -> list[LuaSyntaxError]:
        """Check Lua source text with the real Lua parser.

        Args:
            content: The Lua source.

        Returns:
            An empty list when the syntax is correct (or lupa is not installed).
            A list of errors when syntax problems were found.
        """
        if not self.is_available:
            return []
        return self._check(content)

    def _check(self, content: str) -> list[LuaSyntaxError]:
        """Write the source to a temporary file and call loadfile on it.

        Args:
            content: The Lua source.

        Returns:
            An empty list when the syntax is correct, otherwise one error.
        """
        fd, tmp_path = tempfile.mkstemp(suffix=".lua", prefix="aidirector_luacheck_")
        try:
            os.write(fd, content.encode("utf-8"))
            os.close(fd)
            fd = -1

            # Escape the path for a Lua string (backslashes on Windows)
            lua_path = tmp_path.replace("\\", "\\\\")

            # Lua code: calls loadfile and returns a status string.
            # loadfile does NOT run the code, it only parses it.
            # Returns "OK" or "ERR:<line_number>:<message>".
            # Wrapped in an IIFE because lupa.eval() needs an expression.
            lua_code = f'''
            (function()
                local f, err = loadfile("{lua_path}")
                if f then
                    return "OK"
                end
                local line = 0
                local m = tostring(err):match(":(%d+):")
                if m then line = tonumber(m) end
                return "ERR:" .. tostring(line) .. ":" .. tostring(err)
            end)()
            '''

            result = self._lua.eval(lua_code)
            result = str(result)

            if result == "OK":
                return []

            # Parse "ERR:<line>:<message>"
            m = re.match(r"^ERR:(\d+):(.*)$", result, re.DOTALL)
            if m:
                line = int(m.group(1))
                message = m.group(2).strip()
                return [LuaSyntaxError(line, message)]

            return [LuaSyntaxError(0, result)]

        except Exception as e:
            return [LuaSyntaxError(0, f"unexpected error from lupa: {e}")]
        finally:
            if fd >= 0:
                with contextlib.suppress(OSError):
                    os.close(fd)
            with contextlib.suppress(OSError):
                Path(tmp_path).unlink()

    @staticmethod
    def _parse_error(err: str) -> list[LuaSyntaxError]:
        """Parse a Lua error message.

        Formats that LuaJIT returns:
          [string "tmp..."]:12: syntax error near 'end'
          tmp.lua:5: unexpected symbol near '1'

        Args:
            err: The error message.

        Returns:
            One error, with line 0 when the message has no line number.
        """
        # Try to extract the line number
        m = re.search(r":(\d+):\s*(.*)", err)
        if m:
            return [LuaSyntaxError(int(m.group(1)), m.group(2).strip())]
        # Fallback: the whole string is the message, line 0
        return [LuaSyntaxError(0, err)]


def check_lupa_available() -> bool:
    """Check whether the real Lua parser is available.

    Returns:
        True when lupa is installed.
    """
    if LuaRuntime is None:
        return False
    LuaRuntime()
    return True
