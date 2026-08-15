# Usage

## Install

```bash
uv sync
```

That is everything: the CLI, the GUI, the MCP server and the measurement harness.

Those last three are declared as optional extras, because someone who installs
the published package to use the CLI over SSH should not be made to take Qt and
OpenCV with it. A checkout of *this repository* pulls them in anyway, via a
`local` dependency group that `[tool.uv] default-groups` marks as default.

That indirection is not decoration. `uv run` re-resolves the environment on every
invocation, down to the default set — and extras are never in it. So
`uv sync --all-extras` followed by `uv run aidirector-gui` installs PySide6 and
then removes it again, and the command fails with `ModuleNotFoundError` for a
dependency you just watched it install. Putting the extras behind a default group
is what makes a bare `uv run` work.

If you want the published CLI's dependency set instead — to check that nothing
has quietly grown a dependency on a graphical toolkit, which is what CI's
provenance job does — ask for it explicitly:

```bash
uv sync --no-group local
```

## The short version

```bash
uv run aidirector direct replays/session.acreplay --style initial_d -o timeline.json
```

That reads the replay, detects events, scores moments, selects shots, generates
camera paths and writes the timeline.

## Commands

### `inspect` — what is in this replay?

```bash
uv run aidirector inspect replays/*.acreplay
```

Reads only the file's structure, so it returns in well under a second on a 327 MB
replay. Reports the entry list, recording rate, and whether the layout
**verified** — meaning the structural walk landed exactly where the file's own
footer says the data ends. A replay that fails that check is reported as unusable
rather than half-parsed.

Watch the recording rate. Assetto Corsa's replay quality is a user setting, and
at the low end the file simply does not contain enough temporal resolution to
locate a braking point or the onset of a drift. If `inspect` says the quality is
coarse, raise it in the game and record again — no analysis recovers information
the file never stored.

### `events` — what happened?

```bash
uv run aidirector events replays/session.acreplay --seconds 300
```

Detects drifts, linked transitions, spins, corners, apexes, braking points and
overtakes. Cars that never moved are skipped and reported — a replay from an
online session records every empty grid slot.

The output ends with a list of what the detectors **cannot** see. Read it: a
missing contact event means the tool has no damage channel, not that nobody
touched.

### `score` — how good is each moment, and to whom?

```bash
uv run aidirector score replays/session.acreplay --style initial_d
```

Prints the five moment scores as distributions plus the highest-scoring moments
under a style's weighting. Its job is to make the director's taste inspectable
*before* any camera is placed: if the top moments look wrong to you watching the
same replay, the weights are wrong, and that is far cheaper to discover here than
after a full edit.

### `direct` — make the cut

```bash
uv run aidirector direct replays/session.acreplay \
    --style initial_d --seconds 300 -o out/timeline.json
```

`--style` takes a built-in name or a path to a TOML profile.

The input may also be a `.csv` in this project's schema — see below. Past the
loader nothing downstream knows which one it came from.

### `csv` — get the telemetry out, and back in

```bash
uv run aidirector csv replays/session.acreplay out/telemetry.csv --seconds 120
```

Two jobs.

**Verification.** A binary reader nobody can check is a binary reader you have to
trust, and this project's whole position on the `.acreplay` format is that trust
is not a method. The export is a table you can open, sort and plot, and see for
yourself whether the car goes where the track goes.

**Interchange.** The same table read back is a session, so a simulator this
project has never heard of can drive the director:

```bash
uv run aidirector direct out/telemetry.csv --style mf_ghost -o out/timeline.json
```

The schema is fixed and documented rather than sniffed — every sim-racing logger
emits a different shape, and guessing which column is which produces smooth,
plausible, entirely wrong telemetry that nothing downstream would catch.

| Column | Required | Meaning |
|---|---|---|
| `time_s` | yes | Seconds from session start. Every car shares one time base. |
| `car_id` | yes | Integer index. Rows are grouped by it. |
| `x`, `y`, `z` | yes | World position in metres, `y` up. |
| `speed_mps` | no | Ground speed. Derived by differencing position when absent. |
| `heading_rad` | no | Chassis yaw. Derived from the path when absent. |
| `sideslip_rad` | no | Heading minus direction of travel. |

The one consequence worth understanding: **derived heading is the path tangent**,
so sideslip computed from it is identically zero. A file without `heading_rad`
yields a session in which drift detection finds nothing — not because nothing
happened, but because the file could not say. The reader puts that in its notes
and marks the session's rotation uncalibrated rather than letting a missing
column read as clean driving.

### `doctor` — what does this machine have?

```bash
uv run aidirector doctor
```

No required arguments. Finds Assetto Corsa through Steam's library index and the
registry, reports whether Custom Shaders Patch is installed, and — with
`--track` — lists which of a layout's data files are present and how large they
are.

**Not a prerequisite.** The same snapshot is embedded in every camera report
automatically. This exists for the moment a report comes back with every
track-dependent check skipped and nobody can tell whether that means the track
has no AI line or the game was never found. On a machine without Assetto Corsa
it succeeds and says so.

### `diagnose` — are these cameras any good?

```bash
uv run aidirector diagnose replays/session.acreplay --out out/
```

One required argument. Directs the session, then checks every camera it produced
against what it would actually see: is the car in front of the lens, in frame, a
sensible size, at a sensible range, and does the track span the `cameras.ini`
export would write actually contain the car when the shot happens.

Writes three files, because they answer different questions:

| File | What it is for |
|---|---|
| `diagnostic.json` | Every camera, every check. The record. |
| `diagnostic_compact.json` | Only the cameras worth looking at. The one you paste into a conversation. |
| `cameras.svg` | What each camera sees, as a contact sheet. Open it in a browser. |

Two checks depend on the track's own data. When Assetto Corsa is installed the
command finds it and reads `fast_lane.ai`, which gives the road's centreline and
its width, so `camera_on_surface` and `camera_lateral_offset` can answer. When it
is not, a centreline is derived from the session's own driving — good enough to
say roughly where along the track something is, useless for road width, and the
report says so rather than reporting a plausible number.

One check cannot be answered at all: `sight_line_clear` needs collision geometry,
which nothing here reads, so a barrier between the camera and the car goes
undetected. It reports `unknown` rather than being left out, because a check
missing from a report reads as a check that passed. The consequence is that **no
camera reports an overall pass** while it is unmeasurable, so the summary carries
`measured_status` beside `status`. One says whether anything that could be
measured is wrong; the other says whether the cameras have been shown to be good,
which is a stronger claim.

The same thing is in the desktop application under the **Cameras** tab, where
**Copy report** puts the compact form on the clipboard. That is the intended
route when the person checking the cameras is not the person who will fix them.

### `export` — convert a timeline

```bash
uv run aidirector export cameras-ini out/timeline.json -o out/cameras_9.ini --force
```

Reports what the target format loses **before** writing, and refuses to write a
lossy export without `--force`. That is not pedantry: `cameras.ini` switches
cameras on track position rather than time, so it cannot reproduce a single
event-triggered cut, and telling you afterwards would be too late.

### `install` — put a camera set where the game reads it

```bash
uv run aidirector install out/cameras_9.ini --timeline out/timeline.json
```

Works out the layout-scoped destination, finds the installation, reads the set
back to say what is in it, and prints what it would do. Nothing is written until
`--write`, and an existing set is copied aside first — this is the only thing the
tool does outside its own output folder.

A multi-layout track keeps cameras under the layout, not the track root. The
command refuses a layout the install does not have rather than creating the
folder, because the track root is the common mistake and it produces a file the
game silently never loads.

### `styles` — what is the director's taste made of?

```bash
uv run aidirector styles
uv run aidirector styles --provenance
```

`--provenance` audits where every number came from: `verified`, `measured`,
`derived` or `guess`. Guesses are legitimate — a director has to pick something —
but they are the measurement backlog, and this is how they stay visible.

### `measure` — replace guesses with measurements

```bash
uv run aidirector measure references/ --style touge --out profiles/touge.toml
```

Point it at footage you legally own. It detects shot boundaries, measures average
shot length, spread and the spectral structure of the cutting rhythm, classifies
camera behaviour per shot, and writes a style profile whose pacing carries
`provenance = "measured"` and cites the clips it came from.

Give it several minutes of material. Below about thirty shots the rhythm spectrum
is dominated by its own sampling noise, and the command refuses to report a slope
rather than printing one that means nothing.

What it **cannot** recover is true focal length or camera height in metres —
those need scene geometry footage does not carry. They stay `GUESS`.

## The MCP server

```bash
uv run aidirector-mcp
```

Speaks stdio. Ten tools: `open_session`, `load_replay`, `load_telemetry`,
`find_drifts`, `find_transitions`, `find_best_moments`, `generate_shotlist`,
`generate_style`, `preview`, `export_json`.

Call `open_session` first; every other tool takes the handle it returns.
Decoding is expensive and happens once per session, then detection, scoring and
each style's timeline are computed lazily on first use.

Results carry a `notes` field listing what the analysis could not see. Read it
before concluding that something did not happen.

## Without a replay

Every part of the pipeline runs on the synthetic provider, which generates a
session with events at known times:

```python
from assetto_ai_director.application.director.styles import BUILTIN_STYLES
from assetto_ai_director.application.pipeline import direct
from assetto_ai_director.infrastructure.synthetic.source import SyntheticSource

session = SyntheticSource(seconds=60.0, cars=2).generate()
timeline = direct(
    session.signals,
    session.meta,
    BUILTIN_STYLES["initial_d"],
    session.time,
    float(session.sample_rate),
)
print(timeline.shot_count, float(timeline.average_shot_length))
```

This is what the test suite uses, and it is why a fresh clone with no replay
files still gets a meaningful build rather than a green one that skipped
everything.

## Getting good input

Two settings in Assetto Corsa matter more than anything this tool does.

**Replay quality.** Set it as high as your disk allows. The recording rate is the
hard ceiling on how precisely any event can be located.

**Session length.** The game records into a ring buffer, and once it wraps the
oldest frames are gone — `inspect` reports when that has happened. If you want a
whole session, keep it inside the buffer.
