---
name: ci-check
description: Make the code pass CI before pushing. Formats the code, applies safe automatic fixes, writes the missing docstrings and type annotations by hand, fixes the remaining lint errors, then runs every CI check. Use it before every push, and whenever git push prints "WARNING - push completed, but CI on GitHub WILL FAIL".
---

# /ci-check — make the code pass CI

CI on GitHub runs five checks. This skill makes all five pass **by fixing the code**, never by
weakening the checks. Read `AGENTS.md` first — the code rules there are what CI enforces.

## Hard rules for this skill

- **Do not use `# noqa` or `# type: ignore` to skip work.** Fix the code instead. The only
  exception is a false positive — a rule that is wrong about one exact line. Then silence only
  that code, on that line, with the reason: `# noqa: N802 - CSP API name`. Step 6 lists the
  known cases.
- **Do not edit the rule lists in `pyproject.toml`.** The rules stay as they are.
- **Do not rename public functions, methods or classes**, and do not change behaviour. This is an
  annotation and documentation pass, not a refactor. When a rule forces you to split a long
  function, the output must stay exactly the same.
- **Do not invent facts in a docstring.** Describe only what the code visibly does. If the purpose
  of a function is unclear, read its callers before writing about it.

## How to run the tools

Use whichever works on this machine:

| Tool | With `uv` | Windows, no `uv` on PATH |
|---|---|---|
| Python module | `uv run python -m <module>` | `.venv\Scripts\python.exe -m <module>` |
| Layer check | `uv run lint-imports` | `.venv\Scripts\lint-imports.exe` |

Below, `PY` means whichever Python command works.

## Steps

### 1. Format the code

```
PY -m ruff format .
```

This alone fixes every "Format check" failure.

### 2. Apply the safe automatic fixes

```
PY -m ruff check . --fix
```

This removes unused imports, sorts imports, drops empty f-strings and similar.

### 3. List what is left

```
PY -m ruff check . --output-format=concise
```

Each line is `file:line:column: CODE message`. Work through them file by file.

### 4. Fix missing docstrings — codes starting with `D`

Open the file at that line. Write a **Google-style** docstring directly under the `def` or `class`:

```python
def check_file(self, filepath: Path) -> list[LuaSyntaxError]:
    """Check one Lua file with the real Lua parser.

    Args:
        filepath: The Lua file to check.

    Returns:
        Every syntax error found, empty when the file is valid.
    """
```

- First line: one sentence, what it does, ending with a full stop.
- Then, when it helps, why it exists or what is easy to get wrong.
- `Args:` for every parameter except `self` and `cls`.
- `Returns:` when it returns something. `Raises:` when it raises on purpose.
- A class gets a docstring describing what one instance represents.

### 5. Fix missing type annotations — codes starting with `ANN`

Add the type the code **actually** uses:

- Read the function body and its callers to find the real types.
- A function that returns nothing gets `-> None`.
- `__init__` gets `-> None`.
- Prefer modern syntax: `list[str]`, `dict[str, int]`, `X | None` — not `List`, `Dict`, `Optional`.
- Do **not** use `Any` just to silence the error. Use it only where a value really can be
  anything, and say why in the docstring.

### 6. Fix everything else by the meaning of the rule

Common ones in this repository:

| Code | What to do |
|---|---|
| `EM101`, `EM102` | Put the message in a variable, then `raise SomeError(message)`. |
| `E501` | Break the line; keep it at most 100 characters. |
| `PLC0415` | Move the `import` to the top of the file. An optional package goes in a top-level `try: … except ImportError:` block. |
| `PLR2004` | Replace the bare number with a named module constant. |
| `F841` | Remove the unused variable. |
| `ARG001`, `ARG002`, `ARG005` | If something outside our Python code decides the arguments (Lua calls it, a callback, an override), keep the argument and prefix it with `_`. If only our own code passes it, remove it and update every caller. |
| `UP006`, `UP035` | Use `list`, `dict`, `X | None` instead of `typing.List`, `Dict`, `Optional`. |
| `TID252` | Use an absolute import: `from tools.csp_debugger.x import y`. |
| `RUF001`, `RUF002`, `RUF003` | Translate the Russian text to English. |
| `PLR0912`, `PLR0915` | Split the function into private helpers, one per step. Keep every message and every result the same. |
| `TRY300` | Move the `return` out of the `try` into an `else:` block. |
| `TRY004` | Raise `TypeError` for a wrong type, and add it to the docstring's `Raises:`. |
| `SIM102`, `SIM105` | Merge the nested `if`; use `contextlib.suppress(Error)` instead of `try`/`except`/`pass`. |
| `PTH108` and other `PTH` | Use `pathlib`: `Path(p).unlink()` instead of `os.unlink(p)`. |
| `PLW1510` | Add `check=False` to `subprocess.run` when the code reads the return code itself. |

**Known false positives.** Only these get a `# noqa` with a reason:

| Code | When |
|---|---|
| `N802` | A mock or wrapper that must keep a camelCase name because Lua or Qt calls it by that name. |
| `S603` | `subprocess.run` with a fixed argument list and no shell, run by the developer. |
| `S105` | An enum value like `PASS = "PASS"` — a verdict, not a password. |
| `S108` | A fake `/tmp/...` path in a mock that nothing writes to. |

Anything else: fix the code. If you believe a new case is a false positive, stop and ask the user.

When unsure what a code means, run `PY -m ruff rule CODE`.

### 7. Repeat steps 3 to 6

Until `PY -m ruff check .` prints `All checks passed!`.

### 8. Run every CI check

```
PY -m ruff format --check .
PY -m ruff check .
PY -m mypy
lint-imports
PY -m pytest -q
```

GUI tests need `QT_QPA_PLATFORM=offscreen` set when there is no display.

If `mypy` or `pytest` fails after your edits, an annotation you added is wrong or a change touched
behaviour. Fix that before anything else.

### 9. Report

Tell the user, briefly:

- how many problems there were and how many are left (should be zero);
- which files changed;
- anything you could not fix, and why.

Then commit. The pre-push hook will confirm the result when you push.
