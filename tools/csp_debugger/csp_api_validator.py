"""
CSP API validator.
Проверяет использование CSP функций в коде.
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple

from .csp_api_registry import CSP_API_REGISTRY, CONTEXT_RESTRICTED

class CSPAPIIssue:
    """Проблема с использованием CSP API."""
    def __init__(self, line: int, function: str, issue: str, severity: str):
        self.line = line
        self.function = function
        self.issue = issue
        self.severity = severity  # "FAIL", "WARNING", "REQUIRES_GAME"

    def __repr__(self):
        return f"Line {self.line} [{self.severity}] {self.function}: {self.issue}"

class CSPAPIValidator:
    """Валидатор использования CSP API."""

    def __init__(self):
        self.issues = []

    def validate_file(self, filepath: Path) -> List[CSPAPIIssue]:
        """Валидировать файл на использование CSP API."""
        content = filepath.read_text(encoding='utf-8')
        return self.validate_content(content)

    def validate_content(self, content: str) -> List[CSPAPIIssue]:
        """Валидировать содержимое."""
        self.issues = []
        lines = content.split('\n')

        # Найти все вызовы CSP функций
        used_functions = self._find_api_calls(lines)

        # Проверить каждую функцию
        for func, line_num in used_functions:
            self._check_function(func, line_num)

        return self.issues

    def _find_api_calls(self, lines: List[str]) -> List[Tuple[str, int]]:
        """Найти все вызовы CSP API функций."""
        calls = []

        for i, line in enumerate(lines, 1):
            # Пропустить комментарии
            if line.strip().startswith('--'):
                continue

            # Найти вызовы ac.*, render.*, ui.*
            for match in re.finditer(r'\b(ac|render|ui)\.(\w+)', line):
                full_call = f"{match.group(1)}.{match.group(2)}"
                calls.append((full_call, i))

            # Найти вызовы ray:methods
            for match in re.finditer(r'\bray:(\w+)', line):
                full_call = f"ray:{match.group(1)}"
                calls.append((full_call, i))

            # Найти вызовы sim.*
            for match in re.finditer(r'\bsim\.(\w+)', line):
                full_call = f"sim.{match.group(1)}"
                calls.append((full_call, i))

        return calls

    def _check_function(self, func: str, line: int):
        """Проверить одну функцию."""
        if func not in CSP_API_REGISTRY:
            self.issues.append(CSPAPIIssue(
                line, func,
                "Unknown CSP API function",
                "WARNING"
            ))
            return

        info = CSP_API_REGISTRY[func]

        # Проверить required
        if not info.get("required", False):
            # Необязательная функция - проверяем наличие защиты
            pass

        # Проверить context
        if func in CONTEXT_RESTRICTED:
            allowed_contexts = CONTEXT_RESTRICTED[func]
            # TODO: Проверить, что код проверяет контекст перед вызовом
            self.issues.append(CSPAPIIssue(
                line, func,
                f"Requires context: {', '.join(allowed_contexts)}",
                "REQUIRES_GAME"
            ))
