"""Tests for rendering camera previews as SVG.

The document has to parse. That sounds obvious until you notice the whole file is
built by string concatenation, which is exactly the technique that produces
almost-valid markup — a label containing an ampersand, a path with a stray quote
— and browsers render almost-valid markup as a blank page with no error.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from assetto_ai_director.application.diagnostics.scene import Preview, Stroke, StrokeKind
from assetto_ai_director.domain.diagnostics import CheckStatus
from assetto_ai_director.infrastructure.export.preview_svg import (
    contact_sheet,
    render_preview,
    write_contact_sheet,
)

_SVG = "{http://www.w3.org/2000/svg}svg"


def _parse(document: str) -> ET.Element:
    """Parse a document this project generated.

    One place to do it, with one suppression: the input is a string this
    repository just built, not untrusted data off a network, so the usual
    external-entity warning does not apply — and repeating the suppression at
    every call site is how a genuinely untrusted parse later slips in unnoticed.

    Args:
        document: SVG text.

    Returns:
        The root element.
    """
    return ET.fromstring(document)  # noqa: S314 - our own output, not untrusted input


def _preview(index: int, *, label: str = "shot", status: CheckStatus = CheckStatus.PASS) -> Preview:
    """Build a preview with a little geometry in it.

    Args:
        index: The shot index.
        label: Caption.
        status: The measured verdict, which sets the border colour.

    Returns:
        The preview.
    """
    return Preview(
        shot_index=index,
        label=label,
        status=CheckStatus.UNKNOWN,
        measured_status=status,
        strokes=(
            Stroke(points=((0.0, 0.0),), kind=StrokeKind.FRAME_CENTRE),
            Stroke(points=((-0.5, -0.2), (0.0, 0.0), (0.5, 0.3)), kind=StrokeKind.SUBJECT_PATH),
            Stroke(points=((-0.1, -0.1), (0.1, -0.1), (0.1, 0.1)), kind=StrokeKind.SUBJECT_CAR),
        ),
    )


class TestWellFormed:
    def test_a_single_preview_parses(self) -> None:
        root = _parse(render_preview(_preview(0).strokes, label="shot 0"))
        assert root.tag == _SVG

    def test_a_contact_sheet_parses(self) -> None:
        root = _parse(contact_sheet([_preview(i) for i in range(5)], title="t"))
        assert root.tag == _SVG

    def test_an_empty_sheet_still_parses(self) -> None:
        root = _parse(contact_sheet([]))
        assert root.tag == _SVG

    def test_a_label_with_markup_in_it_is_escaped(self) -> None:
        # A track called "Ebisu <East> & West" is not hypothetical.
        document = contact_sheet([_preview(0, label="Ebisu <East> & West")], title="a & b")
        root = _parse(document)
        texts = [element.text for element in root.iter() if element.tag.endswith("}text")]
        assert "Ebisu <East> & West" in texts
        assert "a & b" in texts


class TestLayout:
    def test_one_cell_per_preview(self) -> None:
        root = _parse(contact_sheet([_preview(i) for i in range(7)]))
        # The outer element plus one nested svg per preview.
        assert len(list(root.iter(_SVG))) == 8

    def test_the_grid_grows_downward_not_sideways(self) -> None:
        four = _parse(contact_sheet([_preview(i) for i in range(4)], columns=2))
        eight = _parse(contact_sheet([_preview(i) for i in range(8)], columns=2))
        assert four.get("width") == eight.get("width")
        assert int(eight.get("height", "0")) > int(four.get("height", "0"))

    def test_a_title_makes_room_for_itself(self) -> None:
        bare = _parse(contact_sheet([_preview(0)]))
        titled = _parse(contact_sheet([_preview(0)], title="pk_akina"))
        assert int(titled.get("height", "0")) > int(bare.get("height", "0"))


class TestTrimming:
    def test_geometry_far_outside_the_frame_is_dropped(self) -> None:
        # A car's path spans the whole track while a preview shows a few dozen
        # metres of it, so without trimming every cell carries hundreds of
        # invisible coordinates and the sheet balloons.
        near = Preview(
            shot_index=0,
            label="near",
            status=CheckStatus.UNKNOWN,
            measured_status=CheckStatus.PASS,
            strokes=(
                Stroke(points=tuple((0.0, 0.0) for _ in range(200)), kind=StrokeKind.OTHER_PATH),
            ),
        )
        far = Preview(
            shot_index=0,
            label="far",
            status=CheckStatus.UNKNOWN,
            measured_status=CheckStatus.PASS,
            strokes=(
                Stroke(points=tuple((50.0, 50.0) for _ in range(200)), kind=StrokeKind.OTHER_PATH),
            ),
        )
        assert len(contact_sheet([far])) < len(contact_sheet([near]))

    def test_a_line_entering_the_frame_still_arrives_from_outside(self) -> None:
        # Keeping one point either side of a visible run means a line reaches the
        # edge rather than starting at it, which is what makes it read as
        # continuing past the frame.
        stroke = Stroke(points=((-4.0, 0.0), (-0.5, 0.0), (0.5, 0.0)), kind=StrokeKind.OTHER_PATH)
        preview = Preview(
            shot_index=0,
            label="",
            status=CheckStatus.UNKNOWN,
            measured_status=CheckStatus.PASS,
            strokes=(stroke,),
        )
        document = contact_sheet([preview])
        paths = [
            element.get("d", "")
            for element in _parse(document).iter()
            if element.tag.endswith("}path")
        ]
        drawn = [d for d in paths if d.count("L") >= 2]
        assert drawn, "the run and its lead-in should both be drawn"


class TestWriting:
    def test_the_file_is_written_and_parses(self, tmp_path: Path) -> None:
        destination = tmp_path / "nested" / "cameras.svg"
        write_contact_sheet([_preview(0), _preview(1)], destination, title="pk_akina")
        root = _parse(destination.read_text(encoding="utf-8"))
        assert root.tag == _SVG

    @pytest.mark.parametrize(
        "status",
        [CheckStatus.PASS, CheckStatus.WARN, CheckStatus.FAIL, CheckStatus.UNKNOWN],
    )
    def test_every_verdict_has_a_colour(self, status: CheckStatus) -> None:
        # A KeyError here would mean a new status shipped without a colour, and
        # the sheet would fail to render for exactly the runs worth looking at.
        assert contact_sheet([_preview(0, status=status)])
