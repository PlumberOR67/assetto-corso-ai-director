# CSP Lua Debug Loop — Итоговый отчёт

## Выполненные задачи

### Task 17: Тестирование debug_loop.py ✅
- Протестирован автоматический цикл на существующем spike
- Исправлена проблема с JSON parsing (validate_pipeline должен выводить в stderr, а stdout — только JSON)
- Добавлен `--skip-tests` флаг для пропуска длительных Python-тестов
- Проверена корректная работа с таймаутом (180 сек для pytest)
- Убедился, что REQUIRES_GAME не считается ошибкой

### Task 18: Обновление skill ✅
- Полностью переписан `.claude/skills/csp-lua-debug.md`
- Добавлено описание автономного workflow: CODE → VALIDATE → ANALYZE → FIX → VALIDATE → PYTHON TESTS → REPORT
- Документированы все статусы: PASS, FAIL, WARNING, UNKNOWN, REQUIRES_GAME, UNAVAILABLE
- Добавлены примеры использования
- Описаны ограничения (что можно проверить автоматически, что требует реального запуска игры)

### Task 19: Демонстрация FAIL → FIX → PASS ✅
Создана тестовая копия spike и внесена синтаксическая ошибка:
- Удалено ключевое слово `then` из `if` оператора (строка 107)
- Запущен debug_loop
- Результат:
  ```
  Attempt 1/5: FAIL — 'then' expected near 'result'
  [FIX] Added 'then' at end of line 107
  Attempt 2/5: PASS
  ```
- Автоматический фиксер корректно восстановил код
- Тестовая директория удалена
- Production код не затронут

## Архитектура системы

```
tools/csp_debugger/
├── __init__.py
├── validate_pipeline.py     # Основная валидация (4 проверки)
│   ├── LuaSyntaxChecker     # Настоящий Lua-парсер через lupa (LuaJIT)
│   ├── CSPAPIValidator      # Проверка CSP API usage
│   ├── ProjectValidator     # Проверка структуры проекта
│   └── JSON mode            # --json для машинного парсинга
├── debug_loop.py            # Автономный цикл с max 5 попыток
│   ├── Автофикс синтаксиса  # Только безопасные паттерны (missing then/do/end)
│   ├── Python tests runner  # pytest с timeout
│   └── JSON report          # История попыток
├── csp_api_registry.py      # Реестр CSP функций с версиями
├── csp_api_validator.py     # Валидатор API usage
├── lua_syntax_checker.py    # Обёртка над lupa
├── mock_runtime.py          # Моки для API (ac, render, ui)
└── project_validator.py     # Проверка manifest.ini и структуры

.claude/skills/
└── csp-lua-debug.md         # Skill для Claude Code
```

## Что Claude Code может проверять автономно

### ✅ Автоматическая проверка
1. **Синтаксис Lua** — через настоящий LuaJIT парсер (lupa)
   - Обнаруживает все синтаксические ошибки
   - Показывает номер строки и описание
   - Автоматически исправляет простые случаи (missing then/do/end)

2. **CSP API usage**
   - Проверяет использование известных функций из реестра
   - Помечает функции, требующие特定ной версии CSP
   - Помечает функции, требующие запущенной игры

3. **Структура проекта**
   - Наличие manifest.ini
   - Корректность LUA_FILE
   - Entry points (script.windowMain)
   - Error handling (pcall, nil checks)

4. **Python-интеграция**
   - Запуск pytest для проверки Python-кода
   - Timeout 180 сек, можно пропустить через --skip-tests

### ❌ Что требует реального запуска Assetto Corsa
Все проверки, помеченные как **REQUIRES_GAME**:
- ac.grabCamera — работает только в replay/live сессии
- ac.getTrackSpline — требует запущенной игры
- render.createRay — ray casting работает только в игре
- ray:physics, ray:track — проверка коллизий в реальной сцене

**Важно:** REQUIRES_GAME НЕ считается ошибкой. Это легитимный статус, означающий, что функция требует реального контекста игры.

## Ограничения автоматического фиксера

### Может исправить ✅
- Missing `then` в `if` statements
- Missing `do` в `for`/`while` loops
- Missing `end` (добавление в конец блока)
- Unclosed brackets (если ошибка в предыдущей строке)

### НЕ может исправить ❌
- Логические ошибки (неправильные условия)
- Semantic errors (неправильное использование API)
- Unclosed strings (слишком опасно автофиксить)
- Complex syntax errors (требуют понимания контекста)

Если фиксер не может безопасно исправить ошибку, он останавливается и сообщает:
```
[STOP] No safe fix available — manual intervention required
```

## Использование в задачах

### Ручная валидация
```bash
# Human-readable
.venv/Scripts/python.exe -X utf8 -m tools.csp_debugger.validate_pipeline <app_dir>

# JSON для машинного парсинга
.venv/Scripts/python.exe -X utf8 -m tools.csp_debugger.validate_pipeline --json <app_dir>
```

### Автономный цикл
```bash
# Полный цикл (Lua + Python tests)
.venv/Scripts/python.exe -X utf8 -m tools.csp_debugger.debug_loop <app_dir>

# Только Lua (быстрее)
.venv/Scripts/python.exe -X utf8 -m tools.csp_debugger.debug_loop <app_dir> --skip-tests

# Ограничить попытки
.venv/Scripts/python.exe -X utf8 -m tools.csp_debugger.debug_loop <app_dir> --max-attempts 3
```

## Пример вывода

### Успешная валидация
```
================================================================================
CSP LUA VALIDATION REPORT
================================================================================

[OK] Syntax Check: PASS
  - Lua syntax verified via lupa (LuaJIT parser)

[GAME] CSP API Context: REQUIRES_GAME
  - Line 73 [REQUIRES_GAME] ac.grabCamera: Requires context: replay, live
  - Line 165 [REQUIRES_GAME] ac.getTrackSpline: Requires context: replay, live
  - ...

[OK] Project Structure: PASS
  - Project structure is valid

[OK] Error Handling: PASS
  - Found 3 pcall() calls
  - Found 13 nil checks

================================================================================
SUMMARY
================================================================================
PASS: 3
FAIL: 0
WARNING: 0
UNKNOWN: 0
REQUIRES_GAME: 1

Validation PASSED (with warnings)
```

### Автономный цикл с автофиксом
```
============================================================
CSP Lua Debug Loop
App: spikes/test_debug_demo
Max attempts: 5
============================================================

--- Attempt 1/5 ---

[FAIL] Validation failed:
  - Syntax Check
    Line 108: 'then' expected near 'result'
  [FIX] Added 'then' at end of line 107

-> Re-validating...

--- Attempt 2/5 ---

[LUA PASS] Validation passed on attempt 2
  Note: 14 check(s) require game context

[SKIP] Python tests skipped (--skip-tests)

{
  "attempts": 2,
  "max_attempts": 5,
  "history": [
    {
      "attempt": 1,
      "has_failures": true,
      "results_count": 4
    },
    {
      "attempt": 2,
      "has_failures": false,
      "results_count": 4
    }
  ]
}
```

## Зависимости

Установлено через uv:
```
lupa==2.8  # LuaJIT binding для Python
```

## Итог

Создана полноценная система автономной валидации CSP Lua приложений:

✅ **Настоящий Lua-парсер** (не regex) — обнаруживает все синтаксические ошибки
✅ **Автономный цикл** — до 5 попыток автоматического исправления
✅ **Безопасный фиксер** — исправляет только очевидные ошибки, не ломает код
✅ **Разделение статусов** — REQUIRES_GAME не считается ошибкой
✅ **JSON mode** — интеграция с другими инструментами
✅ **Python-тесты** — проверка целостности проекта
✅ **Skill для Claude** — готов к использованию в задачах

Система готова к использованию. Claude Code может самостоятельно проверять CSP Lua код, обнаруживать и исправлять синтаксические ошибки, и гарантировать, что production код соответствует контракту проекта.
