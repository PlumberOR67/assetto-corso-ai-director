# Architecture

## The dependency rule

```
presentation / mcp  ──▶  infrastructure  ──▶  application  ──▶  domain
```

Dependencies point inward. `domain` imports nothing from the other layers and
nothing from any framework — no Qt, no MCP SDK, no CLI library, not even SciPy.

This is **enforced, not documented**. `import-linter` runs in CI with five
contracts, and a circular import or a domain module reaching for Qt fails the
build rather than surviving to code review.

| Layer | What lives there | May import |
|---|---|---|
| `domain` | Units, geometry, telemetry, events, scores, shots, camera, timeline, provenance | stdlib, NumPy |
| `application` | Ports (Protocols), features, event detection, scoring, director, camera paths | `domain`, SciPy |
| `infrastructure` | `.acreplay` reader, synthetic provider, exporters, style TOML, video measurement, plugin discovery | `application`, `domain` |
| `presentation` | CLI, and the GUI when it lands | everything inward |
| `mcp` | The MCP server | everything inward |

NumPy in the domain is a deliberate, documented exception. Telemetry is
inherently columnar — millions of samples across tens of channels — and modelling
it as lists of objects would misrepresent the data and make analysis unusably
slow.

## The pipeline

```
.acreplay ─▶ structure ─▶ frames ─▶ signals
                                      │
                        ┌─────────────┴─────────────┐
                        ▼                           ▼
                   detection                     features
                   (events)                    (kinematics)
                        │                           │
                        └─────────────┬─────────────┘
                                      ▼
                                 moment scores
                                      │
                                      ▼
                              shot candidates
                                      │
                                      ▼
                          DP schedule (editing graph)
                                      │
                                      ▼
                               camera paths
                                      │
                                      ▼
                                  Timeline
                                      │
                        ┌─────────────┴─────────────┐
                        ▼                           ▼
                   timeline.json              cameras.ini
```

Every stage is a pure function of the stage before it. The only stateful object
in the system is the MCP server's session store, and that exists solely to avoid
re-decoding a 300 MB file on every tool call.

## Three ideas that shape everything

### Sources declare their capabilities

`TelemetrySource.availability()` returns the set of channels a source actually
provides. Assetto Corsa's shared memory has one car and no per-wheel slip angle;
the replay file has every car with slip angle, ratio and load. An interface with
an unconditional `slip_angle` field would force one of them to invent a number —
and an invented zero is indistinguishable from a car tracking perfectly straight.

Consumers ask before they read. Analyses that need a missing channel degrade to a
documented proxy or refuse to run, and say which.

### Every tunable carries its provenance

`Tunable[T]` pairs a value with where it came from: `VERIFIED`, `MEASURED`,
`DERIVED` or `GUESS`. Constructing a non-guess without a source raises — an
unsourced "verified" claim is worse than an honest guess, because it invites
trust it has not earned.

`aidirector styles --provenance` prints the audit, and CI runs it on every push.

The six score weights in each style are deliberately **not** wrapped in
`Tunable`. Nothing recoverable from footage says how much a style should value
risk against flow; wrapping them would let some future code path stamp them
`MEASURED` and inherit authority they cannot have. Leaving them as plain floats
keeps them visibly outside the audit, and the tools say so.

### Unknowns stay unknown

Where a fact could not be established, the code says so and stops.
`CameraDriver` and `ReplayTransport` ship with no implementation and docstrings
recording exactly what was searched and what was found. Regions of the replay
format whose meaning is unknown are skipped as opaque byte ranges rather than
decoded into invented field names.

## The `.acreplay` format

Kunos publishes no specification. The layout was derived by probing four real
replays and then **proved** against the files themselves: a complete structural
walk terminates at precisely the offset each file records in its own Custom
Shaders Patch footer, with zero drift across all seventeen cars.

```
u32   version (== 16)
f64   recording_interval_ms
str   weather, track, track_config          (int32 length prefix, raw bytes)
u32   car_count, current_recording_index, frame_count, track_object_count

frame_count × (4 + 12 × track_object_count) bytes        ← opaque

per car:
  str   model, driver, nation, team, skin
  u32   car_frame_count, wing_count
  car_frame_count × (276 + 4 × wing_count) bytes         ← frame records
  u32   extra_count
  extra_count × 8 bytes                                  ← opaque, player only

u32   lap_record_count
car_count × lap_record_count × 20 bytes                  ← opaque

═══ must land exactly here, at the CSP block offset ═══

last 31 bytes: "__AC_SHADERS_PATCH_v1__" + u32 offset + u32 version
```

**The record is not a fixed size.** It is `276 + 4 × wing_count` bytes — 288 for
most cars, 296 for one with five wing elements. Published parsers assume a fixed
record and therefore silently misalign every channel for those cars while still
producing smooth, believable telemetry. That failure mode is why the reader
verifies itself and raises rather than absorbing a mismatch.

Two fields inside the record are decoded, each validated against an independent
physical fact:

| Offset | Field | Validated by |
|---|---|---|
| 20 | body position, 3×`float32` | 4504 m of path in 180 s → 90 km/h mean; elevation falls 801.6 → 643.3 m on a track named `akina_downhill` |
| 40 | 4 wheel positions, 12×`float32` | distances to the body constant within 5 mm; rectangle 1.40 m × **2.405 m** — the fixture cars are AE86s, wheelbase 2400 mm |

The wheel positions dissolve what was expected to be the project's biggest risk.
Heading was going to require decoding an undocumented rotation field with unknown
Euler order and handedness; instead the rear-to-front axle vector *is* the car's
longitudinal axis, by geometry rather than convention, and the front/rear
assignment falls out of the data — one orientation gives a median sideslip of
1.34°, the other 178.66°.

## The director

Shot selection is a longest path through an editing graph, following Galvane,
Ronfard, Lino and Christie, *Continuity Editing for 3D Animation* (AAAI-15).

```
maximise   Σ shot score  −  Σ transition cost
```

Candidates are generated on a shared grid of cut points, so any chain whose ends
meet tiles the horizon by construction and no gap repair is ever needed. The grid
is every event boundary, peak and end — so the director can cut *on* the action —
plus a regular background grid for the quiet stretches.

Crossing the line of action costs infinity; everything else is finite. Cut from
the left of a car to the right and it appears to reverse direction on screen,
which an audience reads as a different car. No score is worth that.

**Why a DP and not a ranked list:** greedily taking the best moments produces an
edit that crosses the line, jump-cuts and holds one camera for a minute. None of
those faults are visible in a per-moment score — they exist only *between*
consecutive shots — so any method scoring shots independently is structurally
blind to them.

## Plugins

Five entry-point groups: `replay_providers`, `telemetry_providers`,
`camera_packs`, `style_packs`, `exporters`.

Entry points rather than a bespoke registry, because a registry needs the plugin
imported before it can register itself — which means the host must already know
about it, which is a list with extra steps. A broken plugin is isolated and
reported as data rather than taking the host down.

## Known limitations

- **The scheduler is slow.** About ten seconds per 180 s of horizon, linear in
  session length, so a full thirty-minute replay runs for minutes. Correct but
  not yet interactive.
- **Most frame channels are undecoded.** Throttle, brake, gear, rpm and per-wheel
  slip are present in the record but have not been located. Countersteer,
  trail-braking and grip-based risk wait on them.
- **`cameras.ini` cannot cut on events.** The format switches on track position,
  not time. This is structural, not an implementation gap.
- **`ac.grabCamera` during replay is unverified.** Whether CSP's absolute camera
  placement works during playback, and whether Lua can read a replay clock, are
  open questions. Until they are answered, the CSP Lua exporter stays unwritten
  rather than guessed at.
