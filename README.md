# Assetto AI Director

**An AI cinematographer for Assetto Corsa.** Point it at a replay; it analyses the session, decides which
moments deserve a camera, and directs a cinematic shot list and camera timeline — in the visual language of
*Initial D*, *MF Ghost*, *Best Motoring*, *Option Video*, *Fast & Furious* and *Gran Turismo* replay theatre.

> **Status: alpha, under active development.** See [Roadmap](#roadmap) for what works today.

---

## What makes this different

This is not a replay analyser with a camera menu bolted on. It is a *director*: it makes editorial decisions —
where to put the camera, when to cut, what deserves a shot at all — and it makes them using the published
research on automated cinematography rather than ad-hoc heuristics.

Shot selection is a **minimum-cost path through an editing graph**, solved exactly by dynamic programming under
a semi-Markov assumption, following Galvane, Ronfard, Lino & Christie, *Continuity Editing for 3D Animation*
(AAAI-15). Continuity rules — the line of action, the 30° rule, establish-before-tight — come from
He, Cohen & Salesin, *The Virtual Cinematographer* (SIGGRAPH '96). Two-car framing uses the **Toric space**
parameterisation of Lino & Christie (ACM TOG 2015). Cutting rhythm is sampled as **1/f (pink) noise** rather
than i.i.d. durations, because real film shot lengths are 1/f-correlated (Cutting, DeLong & Nothelfer, 2010).

## The honesty rule

The project has one non-negotiable engineering rule: **never invent an API, and never invent a file format.**

Sim-racing tooling is full of confidently-wrong struct layouts copied between projects until they look like
documentation. This codebase refuses to participate. Concretely:

- Every tunable in a style profile carries a `Provenance` — `VERIFIED`, `DERIVED`, `MEASURED` or `GUESS` — and
  `aidirector provenance-report` prints an inventory of every guess in the system. CI fails if a tunable has
  no provenance at all.
- The `.acreplay` reader **verifies itself against the file**: a correct parse must terminate exactly at the
  offset recorded in the trailing Custom Shaders Patch footer. Unknown regions stay opaque byte slices with no
  invented field names.
- Telemetry providers **declare their capabilities** instead of promising a fixed struct. AC's shared memory
  has no per-wheel slip angle; the replay file does. An interface with an unconditional `slip_angle` field
  would force one of them to fabricate a number, so there isn't one.
- Where the answer genuinely could not be established, you will find an interface and a docstring saying what
  was searched and what was not found — not a plausible implementation.

### A worked example

Community parsers assume a fixed-size per-frame car record. Decoding four real replays showed the record
stride is **`276 + 4 × numWings` bytes** — 288 for most cars, 296 for a car with five wing elements. A
fixed-stride reader silently misaligns every channel for those cars and still produces smooth, believable
telemetry. That failure mode — plausible garbage — is exactly what the self-verification exists to catch.

## Features

- **Event detection** — drift initiation and transitions, countersteer, apex, braking point, corner entry/exit,
  overtakes, contact, near-limit moments.
- **Moment scoring** — Drift, Speed, Risk, Flow and Cinematic scores, all normalised and style-weighted.
- **Direction** — candidate generation across camera rigs, sizes and sides; DP shot selection under continuity
  constraints; pink-noise cutting rhythm; camera path generation with critically-damped smoothing.
- **Camera rigs** — static, follow, roadside, wheel, drone, interior, bonnet, suspension, roof, helicopter,
  spline and orbital, each with FOV, offset, height, smoothing, shake, speed and duration.
- **Camera diagnosis** — every camera checked against what it would actually see, without the game: is the
  car in front of the lens, in frame, a sensible size, and does the exported track span contain it when the
  shot happens. Checks that cannot run yet report `unknown` rather than being omitted.
- **Outputs** — canonical versioned `timeline.json`, an Assetto Corsa `cameras.ini` track camera set, telemetry
  CSV, a camera diagnostic report with an SVG contact sheet, and a PySide6 GUI to inspect the edit.
- **MCP server** — drive the whole pipeline from an MCP client.
- **Plugins** — replay providers, telemetry providers, camera packs, style packs and exporters, all via
  standard `importlib.metadata` entry points.
- **Published contracts** — JSON Schemas for the timeline and the style profile in [`contracts/`](contracts),
  checked in CI against real output *and* against the Python enums, so they cannot quietly fall behind.

## Requirements

- Python 3.12+ — managed for you by [uv](https://docs.astral.sh/uv/)
- Assetto Corsa (optional — the synthetic provider runs the full pipeline without it)

This project does **not** currently render video. It produces a camera timeline and an Assetto Corsa
camera set; getting pictures out of the game is done by the game. An earlier version of this list
claimed FFmpeg was an optional requirement "for video export" — nothing in the code has ever invoked
it, and a README that overstates the tool is the same failure as a timeline that overstates the
analysis.

## Install

```bash
uv sync
```

That is the whole setup. The GUI, the MCP server and the measurement harness are
published as optional extras — someone installing the package to use the CLI over SSH should not be made to
take Qt and OpenCV with it — but a checkout of this repository pulls all of them in, because `uv run`
re-resolves the environment on every invocation and an extra that is not in the default set gets removed
between the sync and the run.

## Usage

Inspect a replay's metadata:

```bash
uv run aidirector inspect replays/my_session.acreplay
```

Direct a cut and write the timeline:

```bash
uv run aidirector direct replays/my_session.acreplay --style initial_d -o timeline.json
```

Export an Assetto Corsa trackside camera set:

```bash
uv run aidirector export cameras-ini timeline.json -o out/cameras_9.ini
```

Open the GUI:

```bash
uv run aidirector-gui replays/my_session.acreplay
```

Check what the cameras would actually see:

```bash
uv run aidirector diagnose replays/my_session.acreplay --out out/
```

See what the analysis found, and what it valued:

```bash
uv run aidirector events replays/my_session.acreplay
```

```bash
uv run aidirector score replays/my_session.acreplay --style best_motoring
```

Dump the decoded telemetry as a table you can open in a spreadsheet:

```bash
uv run aidirector csv replays/my_session.acreplay out/telemetry.csv
```

That CSV is also an input. `direct` accepts one wherever it accepts a replay, which is how a simulator other
than Assetto Corsa reaches the director — anything that can emit position and time is a session:

```bash
uv run aidirector direct out/telemetry.csv --style mf_ghost -o timeline.json
```

The schema is fixed and documented rather than sniffed: `time_s, car_id, x, y, z` are required, and
`speed_mps, heading_rad, sideslip_rad` are derived when absent. A file without `heading_rad` produces a session
whose sideslip is identically zero, so drift analysis finds nothing — the reader says so in its notes rather
than letting a missing column read as clean driving.

## Architecture

```
presentation / mcp  ->  infrastructure  ->  application  ->  domain
```

`domain` depends on nothing but the standard library and NumPy. The direction is enforced in CI by
`import-linter`, so a circular import or a layer violation fails the build rather than the code review.

```
src/assetto_ai_director/
├── domain/          pure data and pure functions — no I/O, no Qt, no frameworks
├── application/     ports (Protocols), features, events, scoring, director, camera
├── infrastructure/  .acreplay reader, synthetic provider, CSV, exporters
├── presentation/    CLI and PySide6 GUI
└── mcp/             MCP server
contracts/           published JSON Schemas for what this project writes
```

## Status

Built and covered by tests: the toolchain and layer enforcement, the domain model and ports, the synthetic
provider, the self-verifying `.acreplay` reader, feature extraction, event detection, scoring, style profiles,
the director (candidates, DP scheduler, rhythm), camera paths, `timeline.json` and `cameras.ini` export, the
CSV reader and writer, the MCP server, the PySide6 GUI, and the style measurement harness.

### Known limitations

These are recorded rather than hidden, because a director that looks confident about a session it only half
understood is the failure this project is arranged against.

- **Most frame channels are undecoded.** Throttle, brake, gear, rpm and per-wheel slip are present in the
  record but their offsets have not been established. Every detector that would need them says so in its notes
  instead of inferring them.
- **The scheduler is not interactive.** Roughly 10 s of work per 180 s of horizon, and linear in it — a
  30-minute replay takes minutes. The GUI runs it on a worker thread for that reason.
- **`cameras.ini` cannot cut on events.** The format has no notion of time, only of track position, so the
  exporter reports what the conversion drops rather than pretending it survived.
- **Style pacing is largely unmeasured.** Values measured from reference footage are stamped as such and cite
  the clips; the rest are guesses and are counted as guesses by CI. Run `aidirector measure` on footage to
  convert them.

Deferred until two open questions are settled — whether CSP's `ac.grabCamera` holds during replay playback,
and whether Lua can observe a replay clock — are the CSP Lua exporter, the FFmpeg EDL exporter and the live
shared-memory provider. Interfaces for all three exist; implementations do not, and the docstrings say why.

The experiment that settles them is in [`spikes/csp_replay_camera/`](spikes/csp_replay_camera): a Lua app that
grabs the camera during a replay, moves it a metre, gives it back and prints four lines. It needs somebody with
the game to run it once. Its answer decides whether this project can ever cut on time, or whether `cameras.ini`
and its switch-on-position limits are the permanent ceiling.

## Contributing

Run the full gate before opening a PR:

```bash
uv run ruff format --check . && uv run ruff check . && uv run mypy && uv run lint-imports && uv run pytest
```

## Acknowledgements

This project stands on published work in automated cinematography — He, Cohen & Salesin (SIGGRAPH '96);
Christianson et al. (AAAI '96); Lino & Christie (ACM TOG 2015); Galvane et al. (AAAI 2015, MIG 2013) — and on
the Assetto Corsa modding community's reverse-engineering of the game's data formats.

Assetto Corsa is a trademark of Kunos Simulazioni. This project is unaffiliated with and unendorsed by Kunos.

## License

MIT — see [LICENSE](LICENSE).
