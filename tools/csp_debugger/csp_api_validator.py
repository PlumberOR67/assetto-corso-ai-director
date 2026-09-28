"""CSP API validator.

Checks how a Lua file uses CSP functions.
"""

import re
from pathlib import Path

from tools.csp_debugger.csp_api_registry import CONTEXT_RESTRICTED, CSP_API_REGISTRY


class CSPAPIIssue:
    """One problem with how the code uses the CSP API."""

    def __init__(self, line: int, function: str, issue: str, severity: str) -> None:
        self.line = line
        self.function = function
        self.issue = issue
        self.severity = severity  # "FAIL", "WARNING", "REQUIRES_GAME"

    def __repr__(self) -> str:
        """Return the issue as one report line."""
        return f"Line {self.line} [{self.severity}] {self.function}: {self.issue}"


class CSPAPIValidator:
    """Validator for CSP API usage."""

    def __init__(self) -> None:
        self.issues: list[CSPAPIIssue] = []

    def validate_file(self, filepath: Path) -> list[CSPAPIIssue]:
        """Check the CSP API calls in one Lua file.

        Args:
            filepath: The Lua file to read.

        Returns:
            Every issue found, empty when there is none.
        """
        content = filepath.read_text(encoding="utf-8")
        return self.validate_content(content)

    def validate_content(self, content: str) -> list[CSPAPIIssue]:
        """Check the CSP API calls in Lua source text.

        Args:
            content: The Lua source.

        Returns:
            Every issue found, empty when there is none.
        """
        self.issues = []
        lines = content.split("\n")

        # Find every CSP function call
        used_functions = self._find_api_calls(lines)

        # Check each function
        for func, line_num in used_functions:
            self._check_function(func, line_num)

        return self.issues

    def _find_api_calls(self, lines: list[str]) -> list[tuple[str, int]]:
        """Find every CSP API call in the source.

        Args:
            lines: The Lua source, one entry per line.

        Returns:
            Pairs of the called function and its 1-based line number.
        """
        calls = []

        for i, line in enumerate(lines, 1):
            # Skip comments
            if line.strip().startswith("--"):
                continue

            # Find ac.*, render.*, ui.* calls
            for match in re.finditer(r"\b(ac|render|ui)\.(\w+)", line):
                full_call = f"{match.group(1)}.{match.group(2)}"
                calls.append((full_call, i))

            # Find ray:method calls
            for match in re.finditer(r"\bray:(\w+)", line):
                full_call = f"ray:{match.group(1)}"
                calls.append((full_call, i))

            # Find sim.* calls
            for match in re.finditer(r"\bsim\.(\w+)", line):
                full_call = f"sim.{match.group(1)}"
                calls.append((full_call, i))

        return calls

    def _check_function(self, func: str, line: int) -> None:
        """Record the issues for one CSP function call.

        Args:
            func: The called function, for example ``ac.getSim``.
            line: The 1-based line number of the call.
        """
        if func not in CSP_API_REGISTRY:
            self.issues.append(CSPAPIIssue(line, func, "Unknown CSP API function", "WARNING"))
            return

        info = CSP_API_REGISTRY[func]

        # Check "required"
        if not info.get("required", False):
            # Optional function - a guard around the call is expected
            pass

        # Check context
        if func in CONTEXT_RESTRICTED:
            allowed_contexts = CONTEXT_RESTRICTED[func]
            # TODO: check that the code tests the context before the call
            self.issues.append(
                CSPAPIIssue(
                    line, func, f"Requires context: {', '.join(allowed_contexts)}", "REQUIRES_GAME"
                )
            )
