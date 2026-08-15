"""
Main validation pipeline.
Запускает все проверки и собирает единый отчёт.
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any
from enum import Enum

# Force UTF-8 encoding for Windows console
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    sys.stdout.reconfigure(encoding='utf-8')

from .lua_syntax_checker import LuaSyntaxChecker
from .csp_api_validator import CSPAPIValidator, CSPAPIIssue
from .project_validator import ProjectValidator, ProjectIssue


class ValidationStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    UNKNOWN = "UNKNOWN"
    REQUIRES_GAME = "REQUIRES_GAME"


class ValidationResult:
    """Результат одной проверки."""
    def __init__(self, check_name: str, status: ValidationStatus, details: List[str]):
        self.check_name = check_name
        self.status = status
        self.details = details


class ValidationReport:
    """Полный отчёт валидации."""
    def __init__(self):
        self.results = []
        self.summary = {
            "PASS": 0,
            "FAIL": 0,
            "WARNING": 0,
            "UNKNOWN": 0,
            "REQUIRES_GAME": 0,
        }

    def add_result(self, result: ValidationResult):
        self.results.append(result)
        self.summary[result.status.value] += 1

    def has_failures(self) -> bool:
        return self.summary["FAIL"] > 0

    def print_report(self):
        """Вывести отчёт."""
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
                "REQUIRES_GAME": "[GAME]"
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
        else:
            print("\nValidation PASSED (with warnings)")
            return 0


def validate_lua_app(app_dir: Path) -> ValidationReport:
    """Запустить полную валидацию CSP Lua приложения.

    Args:
        app_dir: Путь к директории приложения (где manifest.ini)

    Returns:
        ValidationReport с результатами всех проверок
    """
    report = ValidationReport()

    # Найти основной Lua файл
    manifest_path = app_dir / "manifest.ini"
    if not manifest_path.exists():
        report.add_result(ValidationResult(
            "Project Structure",
            ValidationStatus.FAIL,
            [f"manifest.ini not found in {app_dir}"]
        ))
        return report

    # Прочитать manifest для получения имени Lua файла
    manifest_content = manifest_path.read_text(encoding='utf-8')
    import re
    lua_match = re.search(r'LUA_FILE\s*=\s*(.+)', manifest_content)
    if not lua_match:
        report.add_result(ValidationResult(
            "Manifest",
            ValidationStatus.FAIL,
            ["LUA_FILE not specified in manifest.ini"]
        ))
        return report

    lua_file = app_dir / lua_match.group(1).strip()
    if not lua_file.exists():
        report.add_result(ValidationResult(
            "Lua File",
            ValidationStatus.FAIL,
            [f"Lua file not found: {lua_file}"]
        ))
        return report

    # 1. Синтаксическая проверка
    print(f"\n[1/4] Checking syntax: {lua_file.name}", file=sys.stderr)
    syntax_checker = LuaSyntaxChecker()

    if not syntax_checker.is_available:
        report.add_result(ValidationResult(
            "Syntax Check",
            ValidationStatus.UNAVAILABLE,
            ["lupa (LuaJIT binding) not installed — run: uv pip install lupa"]
        ))
    else:
        syntax_errors = syntax_checker.check_file(lua_file)
        if syntax_errors:
            details = [str(err) for err in syntax_errors]
            report.add_result(ValidationResult(
                "Syntax Check",
                ValidationStatus.FAIL,
                details
            ))
        else:
            report.add_result(ValidationResult(
                "Syntax Check",
                ValidationStatus.PASS,
                ["Lua syntax verified via lupa (LuaJIT parser)"]
            ))

    # 2. CSP API валидация
    print(f"[2/4] Validating CSP API usage", file=sys.stderr)
    api_validator = CSPAPIValidator()
    api_issues = api_validator.validate_file(lua_file)

    if api_issues:
        # Разделить по severity
        fails = [i for i in api_issues if i.severity == "FAIL"]
        requires_game = [i for i in api_issues if i.severity == "REQUIRES_GAME"]
        warnings = [i for i in api_issues if i.severity == "WARNING"]

        if fails:
            report.add_result(ValidationResult(
                "CSP API Validation",
                ValidationStatus.FAIL,
                [str(i) for i in fails]
            ))
        if requires_game:
            report.add_result(ValidationResult(
                "CSP API Context",
                ValidationStatus.REQUIRES_GAME,
                [str(i) for i in requires_game]
            ))
        if warnings:
            report.add_result(ValidationResult(
                "CSP API Warnings",
                ValidationStatus.WARNING,
                [str(i) for i in warnings]
            ))
    else:
        report.add_result(ValidationResult(
            "CSP API Validation",
            ValidationStatus.PASS,
            ["All API calls are valid"]
        ))

    # 3. Project validation
    print(f"[3/4] Validating project structure", file=sys.stderr)
    project_validator = ProjectValidator(app_dir)
    project_issues = project_validator.validate()

    if project_issues:
        fails = [i for i in project_issues if i.severity == "FAIL"]
        warnings = [i for i in project_issues if i.severity == "WARNING"]

        if fails:
            report.add_result(ValidationResult(
                "Project Structure",
                ValidationStatus.FAIL,
                [f"{i.category}: {i.message}" for i in fails]
            ))
        if warnings:
            report.add_result(ValidationResult(
                "Project Warnings",
                ValidationStatus.WARNING,
                [f"{i.category}: {i.message}" for i in warnings]
            ))
    else:
        report.add_result(ValidationResult(
            "Project Structure",
            ValidationStatus.PASS,
            ["Project structure is valid"]
        ))

    # 4. Error handling check
    print(f"[4/4] Checking error handling", file=sys.stderr)
    content = lua_file.read_text(encoding='utf-8')

    pcall_count = content.count('pcall(')
    nil_checks = content.count('== nil') + content.count('~= nil')

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
        report.add_result(ValidationResult(
            "Error Handling",
            ValidationStatus.PASS,
            error_details
        ))
    else:
        report.add_result(ValidationResult(
            "Error Handling",
            ValidationStatus.WARNING,
            error_details
        ))

    return report


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="CSP Lua validation pipeline")
    parser.add_argument("app_dir", type=Path, help="Path to CSP Lua app directory")
    parser.add_argument("--json", action="store_true", help="Output JSON to stdout (human-readable to stderr)")

    args = parser.parse_args()

    if not args.app_dir.exists():
        print(f"Error: Directory not found: {args.app_dir}", file=sys.stderr)
        sys.exit(1)

    report = validate_lua_app(args.app_dir)

    if args.json:
        # JSON output to stdout, progress to stderr
        import json
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
