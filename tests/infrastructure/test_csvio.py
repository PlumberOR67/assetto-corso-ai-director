"""Tests for the CSV reader and writer.

The reader's job is to be **boring and loud**: read exactly the schema it
documents, derive only what it says it derives, and refuse everything else with a
message that names the problem. So most of what is asserted here is refusal —
missing columns, non-numeric values, time that does not advance — because a CSV
reader that quietly coerces bad input is worse than one that has none, since the
resulting telemetry is smooth, plausible and wrong.

The round trip matters for the same reason: if writing and reading are not
inverses, the debug export is not evidence about the decode, it is evidence about
the exporter.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from assetto_ai_director.application.director.styles import BUILTIN_STYLES
from assetto_ai_director.application.events.detect import CarSignals
from assetto_ai_director.application.pipeline import direct
from assetto_ai_director.infrastructure.csvio.source import (
    CSV_COLUMNS,
    read_csv,
    write_csv,
)
from assetto_ai_director.infrastructure.synthetic.source import (
    SyntheticSession,
    SyntheticSource,
)


@pytest.fixture(scope="module")
def session() -> SyntheticSession:
    """A synthetic session, so the tests need no replay file."""
    return SyntheticSource(seconds=30.0, cars=2, seed=5).generate()


@pytest.fixture
def written(tmp_path: Path, session: SyntheticSession) -> Path:
    """That session written out as CSV."""
    path = tmp_path / "session.csv"
    write_csv(path, session.signals, session.time)
    return path


class TestWrite:
    def test_header_is_the_documented_schema(self, written: Path) -> None:
        first = written.read_text(encoding="utf-8").splitlines()[0]
        assert first.split(",") == list(CSV_COLUMNS)

    def test_one_row_per_car_per_frame(self, written: Path, session: SyntheticSession) -> None:
        lines = written.read_text(encoding="utf-8").strip().splitlines()
        expected = len(session.signals) * session.time.shape[0]
        assert len(lines) - 1 == expected

    def test_creates_missing_directories(self, tmp_path: Path, session: SyntheticSession) -> None:
        path = tmp_path / "a" / "b" / "session.csv"
        write_csv(path, session.signals, session.time)
        assert path.is_file()

    def test_a_short_car_does_not_borrow_another_cars_timestamps(
        self, tmp_path: Path, session: SyntheticSession
    ) -> None:
        # A car that joined late has fewer samples than the time base. Writing
        # past its end would pair its last position with someone else's instant.
        car = session.signals[0]
        half = car.position.shape[0] // 2
        short = CarSignals(
            0, car.position[:half], car.heading[:half], car.speed[:half], car.sideslip[:half]
        )
        path = tmp_path / "short.csv"
        write_csv(path, {0: short}, session.time)
        assert len(path.read_text(encoding="utf-8").strip().splitlines()) - 1 == half


class TestRoundTrip:
    def test_positions_survive(self, written: Path, session: SyntheticSession) -> None:
        parsed = read_csv(written)
        for car_id, original in session.signals.items():
            # Written at four decimals: a tenth of a millimetre.
            assert np.allclose(parsed.signals[car_id].position, original.position, atol=1e-3)

    def test_angles_survive(self, written: Path, session: SyntheticSession) -> None:
        parsed = read_csv(written)
        for car_id, original in session.signals.items():
            assert np.allclose(parsed.signals[car_id].heading, original.heading, atol=1e-5)
            assert np.allclose(parsed.signals[car_id].sideslip, original.sideslip, atol=1e-5)

    def test_sample_rate_is_recovered(self, written: Path, session: SyntheticSession) -> None:
        parsed = read_csv(written)
        assert float(parsed.sample_rate) == pytest.approx(float(session.sample_rate), rel=1e-3)

    def test_car_count_is_recovered(self, written: Path, session: SyntheticSession) -> None:
        parsed = read_csv(written)
        assert set(parsed.signals) == set(session.signals)

    def test_a_full_file_derives_nothing(self, written: Path) -> None:
        parsed = read_csv(written)
        assert parsed.derived == ()
        assert parsed.notes == ()

    def test_a_full_file_counts_as_calibrated(self, written: Path) -> None:
        # Heading came from the file, so sideslip means what it says and drift
        # analysis is entitled to run.
        assert read_csv(written).availability.supports_true_sideslip


class TestDerivation:
    def test_missing_speed_is_derived_and_declared(self, tmp_path: Path, written: Path) -> None:
        path = _drop_columns(tmp_path, written, {"speed_mps"})
        parsed = read_csv(path)
        assert "speed_mps" in parsed.derived
        assert parsed.signals[0].speed.max() > 5.0

    def test_derived_speed_matches_the_written_speed(
        self, tmp_path: Path, written: Path, session: SyntheticSession
    ) -> None:
        path = _drop_columns(tmp_path, written, {"speed_mps"})
        parsed = read_csv(path)
        # Differencing position cannot recover the first sample exactly, and the
        # synthetic path is generated at finite resolution, so compare the bulk.
        original = session.signals[0].speed[2:-2]
        assert np.allclose(parsed.signals[0].speed[2:-2], original, rtol=0.05)

    def test_missing_heading_yields_zero_sideslip_and_says_so(
        self, tmp_path: Path, written: Path
    ) -> None:
        # This is the consequence that must never be silent: without heading the
        # file cannot express a slide, and an analysis told nothing would report
        # clean driving rather than missing data.
        path = _drop_columns(tmp_path, written, {"heading_rad", "sideslip_rad"})
        parsed = read_csv(path)
        assert np.allclose(parsed.signals[0].sideslip, 0.0, atol=1e-9)
        assert any("drift" in note for note in parsed.notes)

    def test_derived_heading_is_not_calibrated(self, tmp_path: Path, written: Path) -> None:
        path = _drop_columns(tmp_path, written, {"heading_rad"})
        assert not read_csv(path).availability.supports_true_sideslip

    def test_a_blank_column_counts_as_absent(self, tmp_path: Path, written: Path) -> None:
        # Reading a blank as 0.0 would put a car at a due-north heading and call
        # it measured.
        path = _blank_column(tmp_path, written, "heading_rad")
        assert "heading_rad" in read_csv(path).derived


class TestRefusal:
    @pytest.mark.parametrize("column", ["time_s", "car_id", "x", "y", "z"])
    def test_a_missing_required_column_is_named(
        self, tmp_path: Path, written: Path, column: str
    ) -> None:
        path = _drop_columns(tmp_path, written, {column})
        with pytest.raises(ValueError, match=column):
            read_csv(path)

    def test_the_error_states_the_expected_schema(self, tmp_path: Path, written: Path) -> None:
        path = _drop_columns(tmp_path, written, {"x"})
        with pytest.raises(ValueError, match="expected schema"):
            read_csv(path)

    def test_an_empty_file_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.csv"
        path.write_text(",".join(CSV_COLUMNS) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="no data rows"):
            read_csv(path)

    def test_a_single_sample_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "one.csv"
        path.write_text(
            ",".join(CSV_COLUMNS) + "\n0.0,0,1.0,2.0,3.0,10.0,0.0,0.0\n", encoding="utf-8"
        )
        with pytest.raises(ValueError, match="sampling rate"):
            read_csv(path)

    def test_a_non_numeric_value_names_the_column(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.csv"
        path.write_text(
            ",".join(CSV_COLUMNS)
            + "\n0.00,0,1.0,2.0,3.0,10.0,0.0,0.0\n0.03,0,oops,2.0,3.0,10.0,0.0,0.0\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="'x'"):
            read_csv(path)

    def test_a_non_numeric_car_id_names_the_line(self, tmp_path: Path) -> None:
        path = tmp_path / "badcar.csv"
        path.write_text(
            ",".join(CSV_COLUMNS)
            + "\n0.00,0,1.0,2.0,3.0,10.0,0.0,0.0\n0.03,car,1.0,2.0,3.0,10.0,0.0,0.0\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="line 3"):
            read_csv(path)

    def test_a_stalled_clock_is_refused(self, tmp_path: Path) -> None:
        # A wrong rate silently rescales every speed and every event time.
        path = tmp_path / "stalled.csv"
        rows = "".join(f"0.0,0,{index}.0,0.0,0.0,10.0,0.0,0.0\n" for index in range(5))
        path.write_text(",".join(CSV_COLUMNS) + "\n" + rows, encoding="utf-8")
        with pytest.raises(ValueError, match="non-increasing"):
            read_csv(path)


class TestParsing:
    def test_a_byte_order_mark_does_not_break_the_header(
        self, tmp_path: Path, written: Path
    ) -> None:
        # Excel writes one, and it would otherwise make the first column
        # unfindable and the file look like it was missing time_s.
        path = tmp_path / "bom.csv"
        path.write_text(written.read_text(encoding="utf-8"), encoding="utf-8-sig")
        assert read_csv(path).signals

    def test_the_time_base_comes_from_the_longest_car(
        self, tmp_path: Path, session: SyntheticSession
    ) -> None:
        # A car that joined late must not truncate the session for everyone else.
        car = session.signals[0]
        half = car.position.shape[0] // 2
        short = CarSignals(
            1, car.position[:half], car.heading[:half], car.speed[:half], car.sideslip[:half]
        )
        path = tmp_path / "uneven.csv"
        write_csv(path, {0: car, 1: short}, session.time)
        assert read_csv(path).time.shape[0] == session.time.shape[0]

    def test_metadata_is_labelled_by_the_caller(self, written: Path) -> None:
        parsed = read_csv(written, track="akina", layout="downhill")
        assert parsed.meta.track_label == "akina/downhill"
        assert parsed.meta.source_name == "csv"

    def test_a_player_is_always_nominated(self, written: Path) -> None:
        # A CSV carries no player flag. Leaving every car unmarked would send the
        # scoring layer looking for a default that is not there.
        assert read_csv(written).meta.player() is not None

    def test_duration_matches_the_written_session(
        self, written: Path, session: SyntheticSession
    ) -> None:
        assert float(read_csv(written).duration) == pytest.approx(float(session.time[-1]), abs=0.05)


class TestAsDirectorInput:
    def test_a_csv_can_be_directed(self, written: Path) -> None:
        # The point of the reader: a table of positions is a session, and the
        # director cannot tell where it came from.
        parsed = read_csv(written)
        timeline = direct(
            parsed.signals,
            parsed.meta,
            BUILTIN_STYLES["initial_d"],
            parsed.time,
            float(parsed.sample_rate),
        )
        assert timeline.shot_count > 0
        assert timeline.tiles_without_gaps()


def _drop_columns(tmp_path: Path, source: Path, drop: set[str]) -> Path:
    """Rewrite a CSV without some of its columns.

    Args:
        tmp_path: Temporary directory.
        source: The file to rewrite.
        drop: Column names to remove.

    Returns:
        The rewritten file.
    """
    lines = source.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    keep = [index for index, name in enumerate(header) if name not in drop]
    rows = [",".join(line.split(",")[index] for index in keep) for line in lines]
    path = tmp_path / f"without_{'_'.join(sorted(drop))}.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _blank_column(tmp_path: Path, source: Path, column: str) -> Path:
    """Rewrite a CSV with one column present but empty.

    Args:
        tmp_path: Temporary directory.
        source: The file to rewrite.
        column: The column to blank.

    Returns:
        The rewritten file.
    """
    lines = source.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    at = header.index(column)
    rows = [lines[0]]
    for line in lines[1:]:
        values = line.split(",")
        values[at] = ""
        rows.append(",".join(values))
    path = tmp_path / f"blank_{column}.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path
