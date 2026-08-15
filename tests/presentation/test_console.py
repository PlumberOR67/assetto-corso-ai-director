"""Tests for console output on a console that cannot print Unicode.

These exist because of a real crash, on the platform this tool primarily runs on.
A Russian-locale Windows console uses code page 1251, and ``aidirector score``
terminated on it with ``UnicodeEncodeError`` — the sparkline it draws to summarise
the scores uses a block-drawing character cp1251 has no room for, and Rich's
legacy-console writer raised rather than substituting.

The whole report was lost to a bar chart, for the users most likely to have
replays worth analysing.

So what is asserted here is the two-part remedy: that prose degrades to ASCII
that still reads as English, and that the one place a character carries
information switches ladders rather than printing a row of question marks.
"""

from __future__ import annotations

import codecs
import io
import sys
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from assetto_ai_director.presentation.cli import score_cmd
from assetto_ai_director.presentation.cli.console import (
    ERROR_HANDLER,
    SUPPORTS_BLOCK_GLYPHS,
    configure_stdio,
    supports,
)


@contextmanager
def _stdout_encoded_as(encoding: str) -> Iterator[io.TextIOWrapper]:
    """Run with standard output on a chosen encoding.

    Args:
        encoding: The codec to install, e.g. ``cp437``.

    Yields:
        The stream, so a test can read back what was written.
    """
    original = sys.stdout
    buffer = io.BytesIO()
    stream = io.TextIOWrapper(buffer, encoding=encoding, errors="strict", newline="")
    sys.stdout = stream
    try:
        yield stream
    finally:
        sys.stdout = original


class TestSupports:
    def test_ascii_is_always_supported(self) -> None:
        assert supports("plain text")

    def test_a_code_page_without_the_character_says_so(self) -> None:
        with _stdout_encoded_as("cp437"):
            # cp437 has block-drawing characters but no em dash.
            assert not supports("—")
            assert supports("█")

    def test_cp1251_cannot_draw_blocks(self) -> None:
        # The exact combination that caused the crash.
        with _stdout_encoded_as("cp1251"):
            assert not supports("█")

    def test_an_unknown_encoding_answers_no(self) -> None:
        # Guessing yes and crashing is worse than guessing no and printing ASCII.
        class _Nameless:
            encoding = "not-a-real-codec"

        original = sys.stdout
        sys.stdout = _Nameless()
        try:
            assert not supports("x")
        finally:
            sys.stdout = original

    def test_a_stream_without_an_encoding_answers_no(self) -> None:
        class _Bare:
            encoding = None

        original = sys.stdout
        sys.stdout = _Bare()
        try:
            assert not supports("x")
        finally:
            sys.stdout = original


class TestTransliteration:
    def test_the_handler_is_registered(self) -> None:
        configure_stdio()
        assert codecs.lookup_error(ERROR_HANDLER) is not None

    def test_registering_twice_is_harmless(self) -> None:
        configure_stdio()
        configure_stdio()
        assert codecs.lookup_error(ERROR_HANDLER) is not None

    def test_an_em_dash_becomes_a_hyphen(self) -> None:
        # The sentence must still read as English. A substitution mark here looks
        # like corrupted output and invites a bug report about the wrong thing.
        with _stdout_encoded_as("cp437") as stream:
            configure_stdio()
            stream.write("unmeasured weights — run measure")
            stream.flush()
            written = stream.buffer.getvalue().decode("cp437")  # type: ignore[attr-defined]
        assert written == "unmeasured weights - run measure"

    @pytest.mark.parametrize(
        ("character", "expected"),
        # Every one of these is genuinely absent from cp437. Characters it does
        # have — it carries ≈, ≤ and ± natively — would pass through unchanged
        # and prove nothing about the handler.
        [("—", "-"), ("…", "..."), ("→", "->"), ("μ", "u"), ("’", "'")],
    )
    def test_known_punctuation_carries_its_meaning(self, character: str, expected: str) -> None:
        with _stdout_encoded_as("cp437") as stream:
            configure_stdio()
            stream.write(character)
            stream.flush()
            written = stream.buffer.getvalue().decode("cp437")  # type: ignore[attr-defined]
        assert written == expected

    def test_an_unknown_character_degrades_rather_than_raising(self) -> None:
        # Text this project does not control — a driver name, a track name.
        # Losing the character is unfortunate; losing the report is worse.
        with _stdout_encoded_as("cp437") as stream:
            configure_stdio()
            stream.write("driver 中村")
            stream.flush()
            written = stream.buffer.getvalue().decode("cp437")  # type: ignore[attr-defined]
        assert written == "driver ??"

    def test_nothing_is_changed_when_the_encoding_can_cope(self) -> None:
        with _stdout_encoded_as("utf-8") as stream:
            configure_stdio()
            stream.write("weights — μ ≈ 1")
            stream.flush()
            written = stream.buffer.getvalue().decode("utf-8")  # type: ignore[attr-defined]
        assert written == "weights — μ ≈ 1"


class TestSparkline:
    def test_both_ladders_have_the_same_number_of_steps(self) -> None:
        # A shorter fallback ladder would quantise the bars differently, so the
        # same data would read as a different distribution on a legacy console.
        assert len(score_cmd._ASCII_LADDER) == len(score_cmd._BLOCK_LADDER)

    def test_the_ascii_ladder_is_printable_anywhere(self) -> None:
        assert score_cmd._ASCII_LADDER.isascii()

    def test_both_ladders_run_from_empty_to_full(self) -> None:
        for ladder in (score_cmd._ASCII_LADDER, score_cmd._BLOCK_LADDER):
            assert ladder[0] == " "
            assert ladder[-1] != " "

    def test_a_full_bar_is_the_heaviest_character(self) -> None:
        assert score_cmd._block(1.0) in {
            score_cmd._ASCII_LADDER[-1],
            score_cmd._BLOCK_LADDER[-1],
        }

    def test_an_empty_bar_is_blank(self) -> None:
        assert score_cmd._block(0.0) == " "

    def test_the_bar_is_monotonic(self) -> None:
        ladder = score_cmd._BLOCK_LADDER if SUPPORTS_BLOCK_GLYPHS else score_cmd._ASCII_LADDER
        heights = [ladder.index(score_cmd._block(step / 20.0)) for step in range(21)]
        assert heights == sorted(heights)
