"""
Project structure validator.
Проверяет структуру CSP Lua приложения.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any

class ProjectIssue:
    """Проблема со структурой проекта."""
    def __init__(self, category: str, message: str, severity: str):
        self.category = category
        self.message = message
        self.severity = severity

class ProjectValidator:
    """Валидатор структуры CSP Lua приложения."""

    def __init__(self, app_dir: Path):
        self.app_dir = app_dir
        self.issues = []

    def validate(self) -> List[ProjectIssue]:
        """Валидировать структуру приложения."""
        self.issues = []

        self._check_manifest()
        self._check_lua_file()
        self._check_entry_points()
        self._check_error_handling()

        return self.issues

    def _check_manifest(self):
        """Проверить manifest.ini."""
        manifest_path = self.app_dir / "manifest.ini"

        if not manifest_path.exists():
            self.issues.append(ProjectIssue(
                "structure",
                "Missing manifest.ini",
                "FAIL"
            ))
            return

        content = manifest_path.read_text(encoding='utf-8')

        # Проверить обязательные секции
        if '[ABOUT]' not in content:
            self.issues.append(ProjectIssue(
                "manifest",
                "Missing [ABOUT] section",
                "FAIL"
            ))

        if '[CORE]' not in content:
            self.issues.append(ProjectIssue(
                "manifest",
                "Missing [CORE] section",
                "FAIL"
            ))

        # Проверить LUA_FILE
        lua_match = re.search(r'LUA_FILE\s*=\s*(.+)', content)
        if not lua_match:
            self.issues.append(ProjectIssue(
                "manifest",
                "Missing LUA_FILE in [CORE] section",
                "FAIL"
            ))
        else:
            lua_file = lua_match.group(1).strip()
            lua_path = self.app_dir / lua_file
            if not lua_path.exists():
                self.issues.append(ProjectIssue(
                    "manifest",
                    f"LUA_FILE points to non-existent file: {lua_file}",
                    "FAIL"
                ))

    def _check_lua_file(self):
        """Проверить Lua файл."""
        # Найти основной Lua файл
        lua_files = list(self.app_dir.glob("*.lua"))

        if not lua_files:
            self.issues.append(ProjectIssue(
                "structure",
                "No .lua files found in app directory",
                "FAIL"
            ))
            return

        for lua_file in lua_files:
            self._check_single_lua_file(lua_file)

    def _check_single_lua_file(self, filepath: Path):
        """Проверить один Lua файл."""
        content = filepath.read_text(encoding='utf-8')

        # Проверить наличие script.windowMain
        if 'script.windowMain' not in content:
            self.issues.append(ProjectIssue(
                "entry_point",
                f"{filepath.name}: Missing script.windowMain function",
                "FAIL"
            ))

        # Проверить использование ac.log для диагностики
        if 'ac.log' not in content:
            self.issues.append(ProjectIssue(
                "diagnostics",
                f"{filepath.name}: No ac.log calls found",
                "WARNING"
            ))

    def _check_entry_points(self):
        """Проверить точки входа."""
        lua_files = list(self.app_dir.glob("*.lua"))

        for lua_file in lua_files:
            content = lua_file.read_text(encoding='utf-8')

            # Проверить наличие windowMain
            if 'windowMain' in content:
                # Проверить, что это функция
                if 'function script.windowMain' not in content and 'function scriptWindowMain' not in content:
                    self.issues.append(ProjectIssue(
                        "entry_point",
                        f"{lua_file.name}: windowMain mentioned but not defined as function",
                        "WARNING"
                    ))

    def _check_error_handling(self):
        """Проверить обработку ошибок."""
        lua_files = list(self.app_dir.glob("*.lua"))

        for lua_file in lua_files:
            content = lua_file.read_text(encoding='utf-8')

            # Проверить использование pcall
            pcall_count = content.count('pcall(')
            if pcall_count == 0:
                self.issues.append(ProjectIssue(
                    "error_handling",
                    f"{lua_file.name}: No pcall usage found",
                    "WARNING"
                ))

            # Проверить наличие проверки на nil
            if '== nil' not in content and '~= nil' not in content:
                self.issues.append(ProjectIssue(
                    "error_handling",
                    f"{lua_file.name}: No nil checks found",
                    "WARNING"
                ))
