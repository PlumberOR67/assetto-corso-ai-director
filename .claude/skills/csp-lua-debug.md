---
name: csp-lua-debug
description: Автоматический debug loop для CSP Lua приложений. Проверяет синтаксис, API usage, структуру проекта и запускает Python-тесты.
---

# CSP Lua Debug Loop

Автоматическая валидация CSP Lua приложений без запуска Assetto Corsa.

## Быстрый старт

### Ручная валидация

```bash
# Human-readable output
.venv/Scripts/python.exe -X utf8 -m tools.csp_debugger.validate_pipeline <app_dir>

# JSON output (для машинного парсинга)
.venv/Scripts/python.exe -X utf8 -m tools.csp_debugger.validate_pipeline --json <app_dir>
```

Пример:
```bash
.venv/Scripts/python.exe -X utf8 -m tools.csp_debugger.validate_pipeline spikes/csp_replay_camera/aidirector_spike
```

### Автоматический debug loop

```bash
# С Python-тестами (полный цикл)
.venv/Scripts/python.exe -X utf8 -m tools.csp_debugger.debug_loop <app_dir>

# Без Python-тестов (только Lua validation)
.venv/Scripts/python.exe -X utf8 -m tools.csp_debugger.debug_loop <app_dir> --skip-tests
```

## Автономный workflow для Claude Code

При изменении Lua-кода в CSP spike:

```
CODE (изменение Lua-файла)
  ↓
VALIDATE (запуск validate_pipeline)
  ↓
Анализ результатов:
  - PASS → перейти к PYTHON TESTS
  - FAIL → ANALYZE FAILURE
  ↓
ANALYZE FAILURE
  - Извлечь номер строки и описание ошибки
  - Определить тип ошибки (синтаксис, API, структура)
  ↓
FIX (автоматический или ручной)
  - Синтаксические ошибки → попытка автоматического исправления
  - API/структурные ошибки → ручное исправление Claude
  ↓
VALIDATE AGAIN (повторная валидация)
  ↓
Если FAIL → повторить цикл (max 5 попыток)
Если PASS → перейти к PYTHON TESTS
  ↓
PYTHON TESTS (запуск pytest)
  - Если FAIL → анализ и исправление Python-кода
  - Если PASS → финальный отчёт
  ↓
FINAL REPORT (итоговый отчёт)
```

### Использование в задачах

После изменения Lua-кода:

```
1. Запустить: .venv/Scripts/python.exe -X utf8 -m tools.csp_debugger.debug_loop <app_dir>
2. Если Lua FAIL → попытаться исправить синтаксис (до 5 попыток)
3. Если Python tests FAIL → исправить Python-код
4. Показать финальный отчёт
```

**Важно:**
- `REQUIRES_GAME` НЕ считается ошибкой — это легитимный статус
- `UNAVAILABLE` означает отсутствие инструмента, а не ошибку кода
- Автоматический фиксер исправляет только очевидные синтаксические ошибки
- Для сложных ошибок требуется ручное вмешательство Claude

## Что проверяется

### 1. Syntax Check (Синтаксис)
- **Инструмент:** lupa (LuaJIT binding) — настоящий Lua-парсер
- **Проверяет:** все синтаксические ошибки через `loadfile()`
- **Без выполнения кода** — только парсинг

### 2. CSP API Usage
- Использование известных CSP функций из реестра
- Контекстные ограничения (replay/live/always)
- Версионные ограничения
- Необязательные функции должны иметь защиту

### 3. Project Structure
- Наличие manifest.ini
- Корректность LUA_FILE в manifest
- Наличие entry points (script.windowMain)
- Обработка ошибок (pcall, nil checks)

### 4. Python Tests
- Запуск существующих pytest-тестов проекта
- Проверяет, что изменения Lua не сломали Python-интеграцию
- Можно пропустить через `--skip-tests`

## Статусы проверок

- **PASS** [OK] — проверка прошла успешно
- **FAIL** [FAIL] — критическая ошибка, требует исправления
- **WARNING** [WARN] — потенциальная проблема, рекомендуется проверить
- **UNKNOWN** [??] — не удалось проверить
- **REQUIRES_GAME** [GAME] — проверка требует запущенной Assetto Corsa/CSP
- **UNAVAILABLE** — инструмент недоступен (например, lupa не установлен)

**Важно:**
- `REQUIRES_GAME` — это НЕ ошибка. Это означает, что проверка требует реального запуска игры.
- `UNAVAILABLE` — это NOT ошибка кода. Это означает отсутствие инструмента проверки.
- Только `FAIL` требует исправления кода.

## Примеры использования

### Пример 1: Ручная проверка

```bash
.venv/Scripts/python.exe -X utf8 -m tools.csp_debugger.validate_pipeline spikes/csp_replay_camera/aidirector_spike
```

Вывод:
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
REQUIRES_GAME: 1 (group)

Validation PASSED (with warnings)
================================================================================
```

### Пример 2: Автоматический debug loop

```bash
.venv/Scripts/python.exe -X utf8 -m tools.csp_debugger.debug_loop spikes/csp_replay_camera/aidirector_spike --skip-tests
```

Вывод:
```
============================================================
CSP Lua Debug Loop
App: C:\...\aidirector_spike
Max attempts: 5
============================================================

--- Attempt 1/5 ---

[LUA PASS] Validation passed on attempt 1
  Note: 14 check(s) require game context

[SKIP] Python tests skipped (--skip-tests)

{
  "attempts": 1,
  "max_attempts": 5,
  "history": [
    {
      "attempt": 1,
      "has_failures": false,
      "results_count": 4
    }
  ]
}
```

## Ограничения

### Что может проверять автоматически:
- ✅ Синтаксис Lua (через настоящий парсер)
- ✅ Использование CSP API (проверка по реестру)
- ✅ Структуру проекта (manifest, entry points)
- ✅ Обработку ошибок (pcall, nil checks)
- ✅ Python-интеграцию (через pytest)

### Что требует реального запуска Assetto Corsa:
- 🎮 Работу `ac.grabCamera()` во время replay
- 🎮 Работу `ac.getTrackSpline()` и ray casting
- 🎮 Реальную высоту земли и sight lines
- 🎮 Поведение CSP API в разных версиях

Все `REQUIRES_GAME` проверки помечены в отчёте и НЕ считаются ошибками.

## Архитектура

```
tools/csp_debugger/
├── validate_pipeline.py      # Основная валидация
│   ├── LuaSyntaxChecker      # lupa-based parser
│   ├── CSPAPIValidator       # проверка API usage
│   └── ProjectValidator      # проверка структуры
├── debug_loop.py             # автономный цикл
│   ├── Автоматический фиксер # только синтаксис
│   ├── Python tests          # pytest integration
│   └── Защита от цикла       # max 5 попыток
└── csp_api_registry.py       # реестр CSP функций
```

## Расширение

### Добавить новую CSP функцию в реестр:
Отредактировать `tools/csp_debugger/csp_api_registry.py`:
```python
"ac.newFunction": {
    "version": "0.2.0+",
    "context": "always",  # или "replay", "live"
    "description": "...",
    "required": False,
},
```

### Добавить новую проверку:
Создать новый валидатор в `tools/csp_debugger/` и добавить в `validate_pipeline.py`.
