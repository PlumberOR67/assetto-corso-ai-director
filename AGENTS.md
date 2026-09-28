# Agent rules — Assetto AI Director

This file is read by Antigravity (Gemini) and, through `CLAUDE.md`, by Claude Code.
Follow it for every change in this repository.

## What this project is

An AI cinematographer for Assetto Corsa. Python 3.12+ in `src/assetto_ai_director`: it reads
replays, detects events, directs a shot list and exports camera sets. In-game parts are Custom
Shaders Patch Lua apps in `spikes/csp_replay_camera/`. Development tooling lives in `tools/`.

The Python code is layered — `domain -> application -> infrastructure -> presentation/mcp` — and
`lint-imports` fails the build if a layer imports one it should not.

## The one rule that overrides everything

**Never invent an API and never invent a file format.** If you cannot confirm that a function,
field or file layout exists, say so and describe what you searched. A plausible-looking guess is
the worst possible outcome here. Parsers must verify themselves against the file they read.

## Code rules CI enforces — including in `tools/`

CI runs `ruff` with a strict rule set. Code that works but breaks these rules turns CI red.

1. **Every public function, method and class needs a docstring**, Google style:

   ```python
   def lap_time(distance: float, speed: float) -> float:
       """Estimate how long one lap takes.

       Args:
           distance: Lap length in metres.
           speed: Average speed in metres per second.

       Returns:
           The lap time in seconds.

       Raises:
           ValueError: If speed is not positive.
       """
   ```

   Say what the function does and **why it exists**. Do not restate the function's name.

2. **Every parameter and every return value needs a type annotation.** Use the real type read
   from the code. Do not reach for `Any` to make an error go away. A function that returns
   nothing is `-> None`.

3. **Error messages go into a variable before `raise`**, never inline:

   ```python
   message = f"Expected 3 components, got {len(parts)}."
   raise ValueError(message)
   ```

4. **Lines are at most 100 characters.** Imports go at the top of the file. Numbers with meaning
   become named constants. Imports are absolute: `from tools.csp_debugger.x import y`, never
   `from .x import y`.

5. **Write comments, docstrings and strings in English.** Cyrillic letters look like Latin ones,
   and ruff (`RUF001`–`RUF003`) flags every one of them.

6. **Do not silence the checker to skip work.** Never use `# noqa` or `# type: ignore` to skip a
   docstring, an annotation or any problem you can fix. Never edit the rule lists in
   `pyproject.toml`.

   The one exception is a rule that is wrong about one exact line — a false positive. For
   example, a mock of the CSP API must keep the name `getSim`, because Lua calls it by that name.
   Then silence only that one code, on that one line, and always give the reason:

   ```python
   def getSim(self) -> MockSim:  # noqa: N802 - CSP API name
   ```

   The code in `src/` already follows this pattern.

## Checks — run all five before you push

With `uv`:

```
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run lint-imports
uv run pytest -q
```

On Windows without `uv` on the PATH:

```
.venv\Scripts\python.exe -m ruff format --check .
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m mypy
.venv\Scripts\lint-imports.exe
.venv\Scripts\python.exe -m pytest -q
```

GUI tests need `QT_QPA_PLATFORM=offscreen` when no display is available.

**The easy way: run `/ci-check` in the agent chat.** It formats the code, fixes what it can,
writes the missing docstrings and annotations, and runs every check.

## When `git push` prints a warning

This repository has a pre-push hook. It runs the fast checks every time you push.

- **It never blocks the push.** The push always completes.
- If it prints `WARNING: push completed, but CI on GitHub WILL FAIL`, the push reached GitHub
  but CI will go red there.
- Do not ignore it. Run `/ci-check`, commit the fixes, and push again.

The hook is switched on once per clone:

```
git config core.hooksPath .githooks
```

If `git config --get core.hooksPath` prints nothing, run that command first.

## Commit messages

Follow the repository's existing style: `feat(scope): …`, `fix(scope): …`, with a body that says
why the change was needed.
