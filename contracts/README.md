# Contracts

Machine-readable descriptions of the formats this project **writes**. They exist so
another tool can consume the output without reading the Python, and so a change to
the output is a change to a file someone reviews rather than a surprise in a
consumer's parser.

The schemas are not generated from the code and are not decorative. `tests/contracts/`
validates real output against them, and — the part that matters — compares every
enumeration in a schema against the Python enum it describes by exact set equality
in both directions. Adding a thirteenth camera rig without updating the contract is
a test failure. A schema that has fallen behind the code is worse than no schema,
because it looks authoritative.

| File | Describes | Written by |
| --- | --- | --- |
| [`timeline.schema.json`](timeline.schema.json) | The canonical directed edit: shots, camera paths, the event behind each shot, and the analysis's caveats | `aidirector direct --out`, `aidirector export json` |
| [`style.schema.json`](style.schema.json) | A style profile — cutting pace, camera mix, score weights, each number with its provenance | `aidirector measure --out`, and the profiles in [`profiles/`](../profiles) |

Style profiles are authored as TOML rather than JSON, because a style is meant to be
read and reviewed by a person and to sit in version control where a diff shows what
someone changed about the look of a cut. TOML shares JSON's data model, so the schema
describes the *parsed* document: validate a profile by parsing it and checking the
result.

## What is deliberately not here

**The `.acreplay` format.** This project reads it; it does not define it. The layout
constants that were established empirically live in
`src/assetto_ai_director/infrastructure/acreplay/layout.py` with the evidence for
each one recorded beside it, and everything unvalidated stays an opaque byte range
with no invented name. Publishing a schema for someone else's undocumented binary
format would be presenting a reverse-engineering result as a specification.

**The telemetry CSV.** It is a table, not a JSON document, and JSON Schema has
nothing useful to say about it. Its columns are defined and documented in
`src/assetto_ai_director/infrastructure/csvio/source.py`, which is also the only
thing that reads them — there is no second implementation for a schema to keep in
agreement with.

**`cameras.ini`.** The format belongs to Assetto Corsa. What this project can and
cannot express through it is recorded in the exporter's support report, which is
returned to the caller at export time rather than documented and forgotten.
