"""Main validation pipeline.

Runs every check and collects one report.
"""

import argparse
import json
import os
import re
import sys
from enum import Enum
from pathlib import Path

# Force UTF-8 encoding for Windows console
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    sys.stdout.reconfigure(encoding="utf-8")

from tools.csp_debugger.csp_api_validator import CSPAPIValidator
from tools.csp_debugger.lua_syntax_checker import LuaSyntaxChecker
from tools.csp_debugger.project_validator import ProjectValidator


class ValidationStatus(Enum):
    """The verdict of one check."""

    PASS = "PASS"  # noqa: S105 - a check verdict, not a credential
    FAIL = "FAIL"
    WARNING = "WARNING"
    UNKNOWN = "UNKNOWN"
    REQUIRES_GAME = "REQUIRES_GAME"


class ValidationResult:
    """The result of one check."""

    def __init__(self, check_name: str, status: ValidationStatus, details: list[str]) -> None:
        self.check_name = check_name
        self.status = status
        self.details = details


class ValidationReport:
    """The full validation report."""

    def __init__(self) -> None:
        self.results: list[ValidationResult] = []
        self.summary = {
            "PASS": 0,
            "FAIL": 0,
            "WARNING": 0,
            "UNKNOWN": 0,
            "REQUIRES_GAME": 0,
        }

    def add_result(self, result: ValidationResult) -> None:
        """Add one check result and count its status.

        Args:
            result: The result to add.
        """
        self.results.append(result)
        self.summary[result.status.value] += 1

    def has_failures(self) -> bool:
        """Whether any check failed.

        Returns:
            True when at least one result is FAIL.
        """
        return self.summary["FAIL"] > 0

    def print_report(self) -> int:
        """Print the report.

        Returns:
            The process exit code: 1 when a check failed, otherwise 0.
        """
        print("=" * 80)
        print("CSP LUA VALIDATION REPORT")
        print("=" * 80)
        print()

        for result in self.results:
            status_icon = {
                "PASS": "[OK]",
                "FAIL": "[FAIL]",
                "WARNING": "[WARN]",
                "UNKNOWN": "[??]",
                "REQUIRES_GAME": "[GAME]",
            }.get(result.status.value, "[??]")

            print(f"{status_icon} {result.check_name}: {result.status.value}")
            for detail in result.details:
                print(f"  - {detail}")
            print()

        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        for status, count in self.summary.items():
            print(f"{status}: {count}")
        print("=" * 80)

        if self.has_failures():
            print("\nValidation FAILED")
            return 1
        print("\nValidation PASSED (with warnings)")
        return 0


def validate_lua_app(app_dir: Path) -> ValidationReport:
    """Run the full validation of a CSP Lua app.

    Args:
        app_dir: Path to the app directory (the one with manifest.ini).

    Returns:
        A ValidationReport with the results of every check.
    """
    report = ValidationReport()

    lua_file = _find_lua_file(app_dir, report)
    if lua_file is None:
        return report

    _check_syntax(lua_file, report)
    _check_api_usage(lua_file, report)
    _check_project(app_dir, report)
    _check_error_handling(lua_file, report)

    return report


def _find_lua_file(app_dir: Path, report: ValidationReport) -> Path | None:
    """Find the main Lua file through manifest.ini.

    Args:
        app_dir: Path to the app directory.
        report: Receives a FAIL result when the file cannot be found.

    Returns:
        The main Lua file, or None when it cannot be found.
    """
    manifest_path = app_dir / "manifest.ini"
    if not manifest_path.exists():
        report.add_result(
            ValidationResult(
                "Project Structure", ValidationStatus.FAIL, [f"manifest.ini not found in {app_dir}"]
            )
        )
        return None

    # Read the manifest to get the Lua file name
    manifest_content = manifest_path.read_text(encoding="utf-8")

    lua_match = re.search(r"LUA_FILE\s*=\s*(.+)", manifest_content)
    if not lua_match:
        report.add_result(
            ValidationResult(
                "Manifest", ValidationStatus.FAIL, ["LUA_FILE not specified in manifest.ini"]
            )
        )
        return None

    lua_file = app_dir / lua_match.group(1).strip()
    if not lua_file.exists():
        report.add_result(
            ValidationResult("Lua File", ValidationStatus.FAIL, [f"Lua file not found: {lua_file}"])
        )
        return None

    return lua_file


def _check_syntax(lua_file: Path, report: ValidationReport) -> None:
    """Step 1: check the Lua syntax.

    Args:
        lua_file: The main Lua file.
        report: Receives the result.
    """
    print(f"\n[1/4] Checking syntax: {lua_file.name}", file=sys.stderr)
    syntax_checker = LuaSyntaxChecker()

    if not syntax_checker.is_available:
        report.add_result(
            ValidationResult(
                "Syntax Check",
                ValidationStatus.UNAVAILABLE,
                ["lupa (LuaJIT binding) not installed — run: uv pip install lupa"],
            )
        )
        return

    syntax_errors = syntax_checker.check_file(lua_file)
    if syntax_errors:
        details = [str(err) for err in syntax_errors]
        report.add_result(ValidationResult("Syntax Check", ValidationStatus.FAIL, details))
    else:
        report.add_result(
            ValidationResult(
                "Syntax Check",
                ValidationStatus.PASS,
                ["Lua syntax verified via lupa (LuaJIT parser)"],
            )
        )


def _check_api_usage(lua_file: Path, report: ValidationReport) -> None:
    """Step 2: check the CSP API calls.

    Args:
        lua_file: The main Lua file.
        report: Receives the results.
    """
    print("[2/4] Validating CSP API usage", file=sys.stderr)
    api_validator = CSPAPIValidator()
    api_issues = api_validator.validate_file(lua_file)

    if not api_issues:
        report.add_result(
            ValidationResult(
                "CSP API Validation", ValidationStatus.PASS, ["All API calls are valid"]
            )
        )
        return

    # Split by severity
    fails = [i for i in api_issues if i.severity == "FAIL"]
    requires_game = [i for i in api_issues if i.severity == "REQUIRES_GAME"]
    warnings = [i for i in api_issues if i.severity == "WARNING"]

    if fails:
        report.add_result(
            ValidationResult("CSP API Validation", ValidationStatus.FAIL, [str(i) for i in fails])
        )
    if requires_game:
        report.add_result(
            ValidationResult(
                "CSP API Context",
                ValidationStatus.REQUIRES_GAME,
                [str(i) for i in requires_game],
            )
        )
    if warnings:
        report.add_result(
            ValidationResult(
                "CSP API Warnings", ValidationStatus.WARNING, [str(i) for i in warnings]
            )
        )


def _check_project(app_dir: Path, report: ValidationReport) -> None:
    """Step 3: check the project structure.

    Args:
        app_dir: Path to the app directory.
        report: Receives the results.
    """
    print("[3/4] Validating project structure", file=sys.stderr)
    project_validator = ProjectValidator(app_dir)
    project_issues = project_validator.validate()

    if not project_issues:
        report.add_result(
            ValidationResult(
                "Project Structure", ValidationStatus.PASS, ["Project structure is valid"]
            )
        )
        return

    fails = [i for i in project_issues if i.severity == "FAIL"]
    warnings = [i for i in project_issues if i.severity == "WARNING"]

    if fails:
        report.add_result(
            ValidationResult(
                "Project Structure",
                ValidationStatus.FAIL,
                [f"{i.category}: {i.message}" for i in fails],
            )
        )
    if warnings:
        report.add_result(
            ValidationResult(
                "Project Warnings",
                ValidationStatus.WARNING,
                [f"{i.category}: {i.message}" for i in warnings],
            )
        )


def _check_error_handling(lua_file: Path, report: ValidationReport) -> None:
    """Step 4: count pcall() calls and nil checks.

    Args:
        lua_file: The main Lua file.
        report: Receives the result.
    """
    print("[4/4] Checking error handling", file=sys.stderr)
    content = lua_file.read_text(encoding="utf-8")

    pcall_count = content.count("pcall(")
    nil_checks = content.count("== nil") + content.count("~= nil")

    error_details = []
    if pcall_count > 0:
        error_details.append(f"Found {pcall_count} pcall() calls")
    else:
        error_details.append("No pcall() calls found")

    if nil_checks > 0:
        error_details.append(f"Found {nil_checks} nil checks")
    else:
        error_details.append("No nil checks found")

    if pcall_count > 0 and nil_checks > 0:
        report.add_result(ValidationResult("Error Handling", ValidationStatus.PASS, error_details))
    else:
        report.add_result(
            ValidationResult("Error Handling", ValidationStatus.WARNING, error_details)
        )


def main() -> None:
    """Run the validation from the command line and exit with its result."""
    parser = argparse.ArgumentParser(description="CSP Lua validation pipeline")
    parser.add_argument("app_dir", type=Path, help="Path to CSP Lua app directory")
    parser.add_argument(
        "--json", action="store_true", help="Output JSON to stdout (human-readable to stderr)"
    )

    args = parser.parse_args()

    if not args.app_dir.exists():
        print(f"Error: Directory not found: {args.app_dir}", file=sys.stderr)
        sys.exit(1)

    report = validate_lua_app(args.app_dir)

    if args.json:
        # JSON output to stdout, progress to stderr
        data = {
            "results": [
                {
                    "check": r.check_name,
                    "status": r.status.value,
                    "details": r.details,
                }
                for r in report.results
            ],
            "summary": report.summary,
            "has_failures": report.has_failures(),
        }
        print(json.dumps(data, indent=2))
        sys.exit(1 if report.has_failures() else 0)
    else:
        exit_code = report.print_report()
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
