"""Reader for Assetto Corsa ``.acreplay`` files.

**Provenance of this module's knowledge.** Kunos publishes no specification for
this format. The layout implemented here was derived from community
reverse-engineering and then **verified byte-exactly against four real replay
files** — a 17-car session on ``pk_akina/akina_downhill``, 233-327 MB each — by
checking that a complete structural parse terminates at precisely the offset the
file itself records in its trailing Custom Shaders Patch footer.

That self-check is not incidental; it is the whole reason this reader is allowed
to exist under the project's "never invent a format" rule. The file carries its
own oracle, so the parse can be *proved* correct on any input rather than assumed
correct because it looked plausible on one.

**One finding worth stating loudly**, because published parsers get it wrong: the
per-frame car record is **not a fixed size**. It is ``276 + 4 * numWings`` bytes,
so a car with five wing elements has a 296-byte record where most cars have 288.
A fixed-stride reader misaligns every channel for those cars and still produces
smooth, entirely believable telemetry. Plausible garbage is the worst failure
mode an analysis tool can have, and it is exactly what the verification catches.

Regions whose meaning could not be established stay **opaque byte ranges with no
invented field names**. They are measured, bounds-checked and skipped — never
decoded into something that looks like data.
"""

from __future__ import annotations
