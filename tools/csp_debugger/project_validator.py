"""Project structure validator.

Checks the structure of a CSP Lua app.
"""

import re
from pathlib import Path


class ProjectIssue:
    """One problem with the structure of the app."""

    def __init__(self, category: str, message: str, severity: str) -> None:
        self.category = category
        self.message = message
        self.severity = severity


class ProjectValidator:
    """Validator for the structure of a CSP Lua app."""

    def __init__(self, app_dir: Path) -> None:
        self.app_dir = app_dir
        self.issues: list[ProjectIssue] = []

    def validate(self) -> list[ProjectIssue]:
        """Check the structure of the app.

        Returns:
            Every issue found, empty when there is none.
        """
        self.issues = []

        self._check_manifest()
        self._check_lua_file()
        self._check_entry_points()
        self._check_error_handling()

        return self.issues

    def _check_manifest(self) -> None:
        """Check manifest.ini."""
        manifest_path = self.app_dir / "manifest.ini"

        if not manifest_path.exists():
            self.issues.append(ProjectIssue("structure", "Missing manifest.ini", "FAIL"))
            return

        content = manifest_path.read_text(encoding="utf-8")

        # Check the required sections
        if "[ABOUT]" not in content:
            self.issues.append(ProjectIssue("manifest", "Missing [ABOUT] section", "FAIL"))

        if "[CORE]" not in content:
            self.issues.append(ProjectIssue("manifest", "Missing [CORE] section", "FAIL"))

        # Check LUA_FILE
        lua_match = re.search(r"LUA_FILE\s*=\s*(.+)", content)
        if not lua_match:
            self.issues.append(
                ProjectIssue("manifest", "Missing LUA_FILE in [CORE] section", "FAIL")
            )
        else:
            lua_file = lua_match.group(1).strip()
            lua_path = self.app_dir / lua_file
            if not lua_path.exists():
                self.issues.append(
                    ProjectIssue(
                        "manifest", f"LUA_FILE points to non-existent file: {lua_file}", "FAIL"
                    )
                )

    def _check_lua_file(self) -> None:
        """Check the Lua files."""
        # Find the Lua files
        lua_files = list(self.app_dir.glob("*.lua"))

        if not lua_files:
            self.issues.append(
                ProjectIssue("structure", "No .lua files found in app directory", "FAIL")
            )
            return

        for lua_file in lua_files:
            self._check_single_lua_file(lua_file)

    def _check_single_lua_file(self, filepath: Path) -> None:
        """Check one Lua file.

        Args:
            filepath: The Lua file to read.
        """
        content = filepath.read_text(encoding="utf-8")

        # Check for script.windowMain
        if "script.windowMain" not in content:
            self.issues.append(
                ProjectIssue(
                    "entry_point", f"{filepath.name}: Missing script.windowMain function", "FAIL"
                )
            )

        # Check that ac.log is used for diagnostics
        if "ac.log" not in content:
            self.issues.append(
                ProjectIssue("diagnostics", f"{filepath.name}: No ac.log calls found", "WARNING")
            )

    def _check_entry_points(self) -> None:
        """Check the entry points."""
        lua_files = list(self.app_dir.glob("*.lua"))

        for lua_file in lua_files:
            content = lua_file.read_text(encoding="utf-8")

            # windowMain is mentioned, so check that it is a function
            if "windowMain" in content and (
                "function script.windowMain" not in content
                and "function scriptWindowMain" not in content
            ):
                self.issues.append(
                    ProjectIssue(
                        "entry_point",
                        f"{lua_file.name}: windowMain mentioned but not defined as function",
                        "WARNING",
                    )
                )

    def _check_error_handling(self) -> None:
        """Check error handling."""
        lua_files = list(self.app_dir.glob("*.lua"))

        for lua_file in lua_files:
            content = lua_file.read_text(encoding="utf-8")

            # Check for pcall
            pcall_count = content.count("pcall(")
            if pcall_count == 0:
                self.issues.append(
                    ProjectIssue(
                        "error_handling", f"{lua_file.name}: No pcall usage found", "WARNING"
                    )
                )

            # Check for nil checks
            if "== nil" not in content and "~= nil" not in content:
                self.issues.append(
                    ProjectIssue(
                        "error_handling", f"{lua_file.name}: No nil checks found", "WARNING"
                    )
                )
