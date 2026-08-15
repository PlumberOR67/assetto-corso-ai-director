"""Console output that survives a legacy Windows code page.

This is not a cosmetic concern for this project. Assetto Corsa runs on Windows,
so a Windows console is the *primary* place this tool is used — and a console
whose code page is not UTF-8 is still the default on a non-English install. On
one such machine ``aidirector score`` did not merely look wrong, it terminated
with ``UnicodeEncodeError: 'charmap' codec can't encode character '█'``:
the sparkline drawn to summarise the scores could not be represented in cp1251,
and Rich's legacy-console writer raised rather than substituting.

Losing a report because of a bar chart is an absurd way to fail, and it fails
worst for exactly the users most likely to have replays to analyse.

Three measures, each for a different kind of character.

:func:`configure_stdio` installs an error handler that **transliterates** the
typographic punctuation this project actually emits — em dashes, arrows,
ellipses, curly quotes — into ASCII that means the same thing. A sentence reading
``unmeasured weights - run aidirector measure`` is correct English; the same
sentence with a substitution mark in it looks like corrupted output and invites a
bug report about the wrong thing.

Anything the transliteration does not know about still degrades to ``?`` rather
than raising. That covers text this project does not control — a driver name in
Cyrillic, a track name in Japanese — where losing the character is unfortunate
but losing the whole report is worse.

:data:`SUPPORTS_BLOCK_GLYPHS` is the honest fallback for the one place a
character is *load-bearing*. A sparkline made of question marks conveys nothing,
so the sparkline asks first and switches to an ASCII ladder when the answer is
no. Degrading to something legible beats degrading to noise.
"""

from __future__ import annotations

import codecs
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["SUPPORTS_BLOCK_GLYPHS", "configure_stdio", "supports"]

ERROR_HANDLER = "aidirector-ascii"
"""Name this module registers its codec error handler under."""

_PROBE = "█▁"
"""Full block and lower one-eighth block: the extremes of the sparkline ladder."""

_TRANSLITERATIONS: Mapping[str, str] = {
    "—": "-",
    "–": "-",
    "…": "...",
    "→": "->",
    "←": "<-",
    "×": "x",
    "·": "-",
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "≈": "~",
    "≤": "<=",
    "≥": ">=",
    "°": " deg",
    "±": "+/-",
    "μ": "u",
    "σ": "sigma",
    "β": "beta",
}
"""Typographic characters this project emits, and ASCII that carries the meaning.

Deliberately short. It covers what the CLI's own strings contain rather than
attempting general Unicode folding — a broad transliteration table would quietly
mangle a driver's name, which is the one kind of text that should be left alone
and reported as unrepresentable.
"""


def _handle(error: UnicodeError) -> tuple[str, int]:
    """Replace unencodable characters, transliterating the ones we know.

    Args:
        error: The encoding error raised by the stream.

    Returns:
        The replacement text and the position to resume encoding from.
    """
    if not isinstance(error, UnicodeEncodeError):  # pragma: no cover - decode path
        raise error
    original = error.object[error.start : error.end]
    return "".join(_TRANSLITERATIONS.get(character, "?") for character in original), error.end


def configure_stdio() -> None:
    """Make standard output tolerant of characters it cannot represent.

    Called once from the CLI entry point. Idempotent, and a no-op on a stream
    that does not support reconfiguration — a redirected pipe under some test
    harnesses, for instance.
    """
    try:
        codecs.lookup_error(ERROR_HANDLER)
    except LookupError:
        codecs.register_error(ERROR_HANDLER, _handle)

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(errors=ERROR_HANDLER)
        except (ValueError, OSError):  # pragma: no cover - platform dependent
            # A stream that refuses reconfiguration is left alone. The glyph
            # fallback below still applies, so load-bearing output stays correct.
            continue


def supports(text: str) -> bool:
    """Report whether standard output can represent a string.

    Args:
        text: The characters to test.

    Returns:
        ``True`` if the current output encoding can encode them. Unknown
        encodings answer ``False``: assuming success and crashing is worse than
        assuming failure and printing ASCII.
    """
    encoding = getattr(sys.stdout, "encoding", None)
    if not encoding:
        return False
    try:
        text.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return False
    return True


SUPPORTS_BLOCK_GLYPHS = supports(_PROBE)
"""Whether block-drawing characters can be printed on this console.

Evaluated once at import. A console's code page does not change mid-run, and a
sparkline that switched styles halfway down a table would be worse than either
choice applied consistently.
"""
