"""
Autonomous debug loop for CSP Lua validation.

Runs the validation pipeline, analyzes failures, applies conservative fixes,
and re-validates. Stops after MAX_ATTEMPTS or when no safe fix is possible.

Safety rules:
- Never weakens a check to achieve PASS.
- Never deletes tests or test code.
- Never modifies files outside the Lua app directory.
- Only fixes patterns where intent is unambiguous.
- REQUIRES_GAME is NOT treated as a failure.
- UNAVAILABLE from syntax checker means the tool is missing, not that code is wrong.
"""

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


MAX_ATTEMPTS = 5


class DebugLoop:
    """Autonomous CSP Lua debug loop with safety limits."""

    def __init__(self, app_dir: Path, project_root: Path, python_bin: Optional[str] = None,
                 max_attempts: int = 5, skip_tests: bool = False):
        self.app_dir = Path(app_dir).resolve()
        self.project_root = Path(project_root).resolve()
        self.python_bin = python_bin or sys.executable
        self.max_attempts = max_attempts
        self.skip_tests = skip_tests
        self.attempt_count = 0
        self.attempt_history: List[Dict] = []
        self._lua_file: Optional[Path] = None

    def _find_lua_file(self) -> Path:
        """Find the main Lua file from manifest.ini."""
        if self._lua_file is not None:
            return self._lua_file
        manifest = self.app_dir / "manifest.ini"
        if manifest.exists():
            content = manifest.read_text(encoding='utf-8')
            m = re.search(r'LUA_FILE\s*=\s*(.+)', content)
            if m:
                self._lua_file = self.app_dir / m.group(1).strip()
                return self._lua_file
        # Fallback: find any .lua file
        lua_files = list(self.app_dir.glob("*.lua"))
        if lua_files:
            self._lua_file = lua_files[0]
            return self._lua_file
        raise FileNotFoundError(f"No Lua file found in {self.app_dir}")

    def run_pipeline(self) -> Dict:
        """Run validation pipeline and return JSON results."""
        pipeline_cmd = [
            self.python_bin, "-X", "utf8",
            "-m", "tools.csp_debugger.validate_pipeline",
            "--json",
            str(self.app_dir),
        ]

        result = subprocess.run(
            pipeline_cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=str(self.project_root),
        )

        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            print("[ERROR] Failed to parse pipeline output", file=sys.stderr)
            print(f"stdout: {result.stdout[:500]}", file=sys.stderr)
            print(f"stderr: {result.stderr[:500]}", file=sys.stderr)
            return {"has_failures": True, "results": [], "summary": {}}

    def run_python_tests(self) -> bool:
        """Run the project's Python tests from the project root."""
        print("\n=== Running Python tests ===", file=sys.stderr)
        try:
            result = subprocess.run(
                [self.python_bin, "-m", "pytest", "-x", "--tb=short", "-q"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                cwd=str(self.project_root),
                timeout=180,
            )
            if result.stdout:
                print(result.stdout, file=sys.stderr)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            print("[ERROR] Python tests timed out after 180 seconds", file=sys.stderr)
            return False
        except Exception as e:
            print(f"[ERROR] Failed to run Python tests: {e}", file=sys.stderr)
            return False

    @staticmethod
    def _parse_syntax_error(detail: str) -> Optional[Dict]:
        """Parse a lupa syntax error detail into (line, expected, near).

        Format: "<tmpfile>:<line>: '<X>' expected (to close '<Y>' at line N) near '<Z>'"
        or:     "<tmpfile>:<line>: unexpected symbol near '<Z>'"
        """
        # Try: '<X>' expected near '<Z>'
        m = re.search(r':(\d+):\s*[\'\"]([^\']+)[\'\"]\s*expected', detail)
        if not m:
            # Try: unexpected symbol near '<Z>'
            m = re.search(r':(\d+):\s*unexpected symbol', detail)
            if m:
                line = int(m.group(1))
                near_m = re.search(r"near\s*[\'\"]([^\']+)[\'\"]", detail)
                near = near_m.group(1) if near_m else "?"
                return {"line": line, "expected": "symbol", "near": near}
            return None

        line = int(m.group(1))
        expected = m.group(2)

        near_m = re.search(r"near\s*[\'\"]([^\']+)[\'\"]", detail)
        near = near_m.group(1) if near_m else "?"
        return {"line": line, "expected": expected, "near": near}

    def _apply_fix(self, detail: str) -> bool:
        """Try to fix a syntax error. Returns True if a fix was applied.

        Only applies fixes where the intent is unambiguous:
        - Missing ')' → append ')' to the end of the previous non-empty line
        - Missing 'then' → append ' then' to the end of the previous line
        - Missing 'do' → append ' do' to the end of the previous line
        """
        parsed = self._parse_syntax_error(detail)
        if not parsed:
            print(f"  [SKIP] Cannot parse error: {detail[:80]}", file=sys.stderr)
            return False

        line = parsed["line"]
        expected = parsed["expected"]
        near = parsed["near"]
        lua_file = self._find_lua_file()

        content = lua_file.read_text(encoding='utf-8')
        lines = content.split('\n')

        # Lupa reports 1-based line. The error is ON that line.
        # For missing ')', 'then', 'do' — the fix goes at the end of the
        # previous line (the line that should have had the closing token).
        target_idx = line - 2  # 0-based index of previous line
        if target_idx < 0 or target_idx >= len(lines):
            print(f"  [SKIP] Line {line} out of range for fix", file=sys.stderr)
            return False

        fix_map = {
            ')': ')',
            'then': ' then',
            'do': ' do',
            'end': ' end',
            ']': ']',
            '}': '}',
            'symbol': None,  # ambiguous — needs manual review
        }

        fix_token = fix_map.get(expected)
        if fix_token is None:
            print(f"  [SKIP] '{expected}' near '{near}' — needs manual review", file=sys.stderr)
            return False

        # Don't apply if the line already has the token at the end (avoid doubles)
        target_line = lines[target_idx].rstrip()
        if target_line.endswith(fix_token):
            print(f"  [SKIP] Line already ends with '{fix_token}'", file=sys.stderr)
            return False

        lines[target_idx] = target_line + fix_token
        lua_file.write_text('\n'.join(lines), encoding='utf-8')
        print(f"  [FIX] Added '{fix_token.strip()}' at end of line {target_idx + 1}", file=sys.stderr)
        return True

    def analyze_and_fix(self, results: List[Dict]) -> bool:
        """Attempt to fix FAIL results. Returns True if any fix was applied."""
        fixed_any = False
        for result in results:
            if result.get("status") != "FAIL":
                continue

            check_name = result.get("check", "")
            details = result.get("details", [])

            if check_name == "Syntax Check":
                for detail in details:
                    if self._apply_fix(detail):
                        fixed_any = True
                        break  # one fix per iteration, re-validate before next
                if not fixed_any and details:
                    print(f"  [MANUAL] {check_name}: {details[0][:80]}", file=sys.stderr)

            elif check_name in ("CSP API Validation", "Project Structure"):
                # These are structural issues — auto-fix is risky
                for detail in details:
                    print(f"  [MANUAL] {check_name}: {detail[:80]}", file=sys.stderr)

            else:
                for detail in details:
                    print(f"  [MANUAL] {check_name}: {detail[:80]}", file=sys.stderr)

        return fixed_any

    def run(self) -> bool:
        """Main debug loop. Returns True if everything passed."""
        print(f"\n{'=' * 60}", file=sys.stderr)
        print(f"CSP Lua Debug Loop", file=sys.stderr)
        print(f"App: {self.app_dir}", file=sys.stderr)
        print(f"Max attempts: {MAX_ATTEMPTS}", file=sys.stderr)
        print(f"{'=' * 60}\n", file=sys.stderr)

        while self.attempt_count < MAX_ATTEMPTS:
            self.attempt_count += 1
            print(f"\n--- Attempt {self.attempt_count}/{MAX_ATTEMPTS} ---", file=sys.stderr)

            results_json = self.run_pipeline()
            has_failures = results_json.get("has_failures", False)
            results = results_json.get("results", [])

            self.attempt_history.append({
                "attempt": self.attempt_count,
                "has_failures": has_failures,
                "results_count": len(results),
            })

            if not has_failures:
                print(f"\n[LUA PASS] Validation passed on attempt {self.attempt_count}", file=sys.stderr)
                # Show REQUIRES_GAME notes
                for r in results:
                    if r.get("status") == "REQUIRES_GAME":
                        count = len(r.get("details", []))
                        print(f"  Note: {count} check(s) require game context", file=sys.stderr)
                break

            # Show failures
            print(f"\n[FAIL] Validation failed:", file=sys.stderr)
            for r in results:
                if r.get("status") == "FAIL":
                    print(f"  - {r['check']}", file=sys.stderr)
                    for d in r.get("details", [])[:3]:
                        print(f"    {d[:100]}", file=sys.stderr)

            # Try to fix
            fixed = self.analyze_and_fix(results)
            if not fixed:
                print(f"\n[STOP] No safe fix available — manual intervention required", file=sys.stderr)
                return False
            print(f"\n-> Re-validating...", file=sys.stderr)

        if has_failures and self.attempt_count >= MAX_ATTEMPTS:
            print(f"\n[STOP] Max attempts ({MAX_ATTEMPTS}) reached", file=sys.stderr)
            return False

        # Lua passed — now run Python tests (unless skipped)
        if self.skip_tests:
            print(f"\n[SKIP] Python tests skipped (--skip-tests)", file=sys.stderr)
            return True

        tests_ok = self.run_python_tests()
        if not tests_ok:
            print(f"\n[FAIL] Python tests failed", file=sys.stderr)
            return False

        print(f"\n[DONE] Lua validation + Python tests passed", file=sys.stderr)
        return True

    def get_report(self) -> Dict:
        return {
            "attempts": self.attempt_count,
            "max_attempts": MAX_ATTEMPTS,
            "history": self.attempt_history,
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Autonomous CSP Lua debug loop")
    parser.add_argument("app_dir", type=Path, help="Path to CSP Lua app directory")
    parser.add_argument("--project-root", type=Path, default=None, help="Project root (default: cwd)")
    parser.add_argument("--python", type=str, default=None, help="Python binary")
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--skip-tests", action="store_true", help="Skip Python tests (Lua validation only)")

    args = parser.parse_args()

    if not args.app_dir.exists():
        print(f"Error: Directory not found: {args.app_dir}", file=sys.stderr)
        sys.exit(1)

    project_root = args.project_root or Path.cwd()

    loop = DebugLoop(
        args.app_dir,
        project_root,
        args.python,
        args.max_attempts,
        args.skip_tests
    )
    success = loop.run()
    report = loop.get_report()

    print("\n" + json.dumps(report, indent=2))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
