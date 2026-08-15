# Contributing

## The gate

Run this before opening a pull request. CI runs the same thing on Windows and
Linux across Python 3.12 and 3.13.

```bash
uv run ruff format --check . && uv run ruff check . && uv run mypy && uv run lint-imports && uv run pytest
```

## The one rule that is not negotiable

**Never invent an API, and never invent a file format.**

Sim-racing tooling is full of confidently-wrong struct layouts copied between
projects until they look like documentation. This codebase does not participate.
In practice that means:

- If you cannot establish a fact, write an interface and a docstring saying what
  you searched and what you did not find. `CameraDriver` and `ReplayTransport`
  are the pattern to follow.
- If you decode part of a binary format, **validate it against something
  independent** before trusting it. Position was accepted because the trajectory
  descends 158 m on a track named "downhill"; wheel positions because they
  reproduce an AE86's 2400 mm wheelbase to within five millimetres. "It looks
  plausible" is not validation — plausible garbage is the failure this project is
  organised against.
- Regions you do not understand stay opaque byte ranges. Do not give them names.

## Every number needs a provenance

Numeric constants that shape output belong in a `Tunable`, with one of
`VERIFIED`, `MEASURED`, `DERIVED` or `GUESS`. Constructing anything but a guess
without a source raises, on purpose.

Guesses are fine. Unlabelled guesses are not. Run `aidirector styles
--provenance` to see the current audit.

## Tests

Prefer tests that could actually fail for a real reason.

- **Assert correctness where it can be known exactly.** A car on a 50 m circle
  must read 0.02 1/m of curvature; yaw rate on a circle must equal `v / r`. These
  catch algebra errors that a plausibility check never would.
- **Assert invariants, not values, for anything that is taste.** Which moment the
  director prefers will change as weights are tuned. That the edit tiles its
  horizon, never crosses the line of action and establishes before going tight
  will not.
- **Use the synthetic provider for anything needing a session.** Replay fixtures
  are user-owned and not in the repository, so tests that require them are marked
  `@pytest.mark.fixtures` and skip on a fresh clone. If your test can run on
  synthetic data, make it.
- **Test the failure modes.** Roughly half the value of the `.acreplay` reader's
  test suite is proving it raises on a bad stride rather than absorbing it.

## Commit messages

Explain **why**, not what. The diff already says what changed.

The commits worth imitating are the ones that record a decision and its reason —
why the frame stride is variable, why zoom and axial dolly were merged into one
class, why the score weights are deliberately outside the provenance audit. In
six months that reasoning is not recoverable from the code, and the commit is the
only place it exists.

If you found a defect while writing the change, say what it was and what it would
have produced. "Fixed a bug" tells a future reader nothing.

## Architecture

Dependencies point inward: `presentation`/`mcp` → `infrastructure` →
`application` → `domain`. `import-linter` enforces it, so a violation fails the
build rather than reaching review. See [docs/architecture.md](docs/architecture.md).

If you find yourself wanting to import Qt into the application layer, or a file
format into the domain, the design is telling you the logic belongs somewhere
else.

## Adding a plugin

Five entry-point groups exist: `replay_providers`, `telemetry_providers`,
`camera_packs`, `style_packs`, `exporters`. Register against one in your own
package's metadata:

```toml
[project.entry-points."assetto_ai_director.exporters"]
my-format = "my_package.exporter:MyExporterSink"
```

Installing your package is the whole installation procedure. A plugin that fails
to load is reported rather than taking the host down, so a broken one degrades
the tool instead of breaking it.

## What would help most

- **Decoding more frame channels.** Throttle, brake, gear, rpm and per-wheel slip
  are in the record but not located. Finding them the same way — hypothesis, then
  physical validation — would immediately improve countersteer detection,
  trail-braking and grip-based risk scoring.
- **Making the scheduler fast.** Ten seconds per 180 s of horizon is fine for a
  CLI and will not do for a GUI.
- **Answering the two CSP questions.** Does `ac.grabCamera` hold during replay
  playback, and can Lua observe a replay clock? Both are unresolved, and both
  gate the in-game camera exporter.
- **Measured style profiles.** Every built-in style's pacing is a guess. Footage
  plus `aidirector measure` turns them into measurements.
