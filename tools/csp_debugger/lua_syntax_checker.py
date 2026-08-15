"""
Lua syntax checker с использованием lupa (LuaJIT binding).
Проверяет синтаксис через настоящий Lua-парсер без выполнения кода.

Используем tempfile + loadfile — это единственный надёжный способ
проверить синтаксис произвольного Lua-кода через lupa без риска
экранирования и без выполнения кода.
"""

import os
import re
import tempfile
from pathlib import Path
from typing import List


class LuaSyntaxError:
    def __init__(self, line: int, message: str):
        self.line = line
        self.message = message

    def __repr__(self):
        return f"Line {self.line}: {self.message}"


class LuaSyntaxChecker:
    """Проверяет Lua-синтаксис через lupa (LuaJIT) — настоящий Lua-парсер.

    Статусы, которые возвращает check_file:
      - Пустой список     → синтаксис корректен
      - Список с ошибками  → найдены синтаксические ошибки
      - Если lupa недоступен, check_file возвращает пустой список,
        а caller проверяет is_available() для отчёта UNAVAILABLE.
    """

    def __init__(self):
        self._lua = None
        try:
            from lupa import LuaRuntime
            self._lua = LuaRuntime()
        except ImportError:
            pass

    @property
    def is_available(self) -> bool:
        return self._lua is not None

    def check_file(self, filepath: Path) -> List[LuaSyntaxError]:
        """Проверить файл через настоящий Lua-парсер.

        Returns:
            Пустой список если синтаксис корректен (или lupa недоступен).
            Список ошибок если найдены синтаксические проблемы.
        """
        if not self.is_available:
            return []

        content = filepath.read_text(encoding='utf-8')
        return self._check(content)

    def check_content(self, content: str) -> List[LuaSyntaxError]:
        if not self.is_available:
            return []
        return self._check(content)

    def _check(self, content: str) -> List[LuaSyntaxError]:
        """Ядро проверки: записываем во временный файл и вызываем loadfile."""
        fd, tmp_path = tempfile.mkstemp(suffix=".lua", prefix="aidirector_luacheck_")
        try:
            os.write(fd, content.encode('utf-8'))
            os.close(fd)
            fd = -1

            # Экранируем путь для Lua-строки (обратные слеши на Windows)
            lua_path = tmp_path.replace('\\', '\\\\')

            # Lua-код: вызывает loadfile и возвращает строку-статус.
            # loadfile НЕ выполняет код — только парсит.
            # Возвращает "OK" или "ERR:<номер_строки>:<сообщение>".
            # Заворачиваем в IIFE, т.к. lupa.eval() требует выражение.
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

            # Парсим "ERR:<line>:<message>"
            m = re.match(r'^ERR:(\d+):(.*)$', result, re.DOTALL)
            if m:
                line = int(m.group(1))
                message = m.group(2).strip()
                return [LuaSyntaxError(line, message)]

            return [LuaSyntaxError(0, result)]

        except Exception as e:
            return [LuaSyntaxError(0, f"unexpected error from lupa: {e}")]
        finally:
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    @staticmethod
    def _parse_error(err: str) -> List[LuaSyntaxError]:
        """Распарсить сообщение об ошибке Lua.

        Форматы, которые возвращает LuaJIT:
          [string "tmp..."]:12: syntax error near 'end'
          tmp.lua:5: unexpected symbol near '1'
        """
        # Попытка вытащить номер строки
        m = re.search(r':(\d+):\s*(.*)', err)
        if m:
            return [LuaSyntaxError(int(m.group(1)), m.group(2).strip())]
        # Fallback: вся строка — сообщение, строка 0
        return [LuaSyntaxError(0, err)]


def check_lupa_available() -> bool:
    """Проверить доступность настоящего Lua-парсера."""
    try:
        from lupa import LuaRuntime
        LuaRuntime()
        return True
    except ImportError:
        return False
