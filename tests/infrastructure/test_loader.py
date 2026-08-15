"""Tests for the shared session loader.

This code existed four times before it existed once, and the copies had already
drifted: only the CLI could read a CSV, only the GUI reported progress, and none
of them could be cancelled. What is asserted here is the behaviour all four
callers now share, and in particular the two things a copy is most likely to lose.

**That parked cars are dropped**, because a full grid is mostly empty slots — the
fixture replays carry seventeen cars of which fifteen never move — and a
stationary car has a heading of pure numerical noise that the risk score would
happily send a camera to.

**That dropping them is reported.** The filter used to live inside `detect_all`,
which named the cars it skipped. Moving the filter without moving the note would
have left a user silently reading a two-car analysis of a seventeen-car race,
which is the exact class of quiet narrowing this project reports rather than
performs. `test_skipped_cars_are_named` is the guard.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from assetto_ai_director.application.ports.progress import Stage
from assetto_ai_director.domain.errors import OperationCancelledError
from assetto_ai_director.infrastructure.csvio.source import write_csv
from assetto_ai_director.infrastructure.loader import (
    MIN_TRAVELLED_METRES,
    load_session,
    skipped_note,
)
from assetto_ai_director.infrastructure.synthetic.source import SyntheticSource


class _Recorder:
    """Records stage labels, so a test can say what was reported."""

    def __init__(self) -> None:
        self.labels: list[str] = []
        self.advances = 0

    def start(self, description: str, total: int | None = None) -> None:
        del total
        self.labels.append(description)

    def advance(self, amount: int = 1) -> None:
        self.advances += amount

    def finish(self) -> None:
        pass


class _AlwaysCancelled:
    def cancelled(self) -> bool:
        return True


@pytest.fixture
def csv_session(tmp_path: Path) -> Path:
    """A two-car session written out as CSV."""
    generated = SyntheticSource(seconds=30.0, cars=2, seed=6).generate()
    path = tmp_path / "session.csv"
    write_csv(path, generated.signals, generated.time)
    return path


@pytest.fixture
def csv_with_a_parked_car(tmp_path: Path) -> Path:
    """A session where one car never leaves its grid box."""
    generated = SyntheticSource(seconds=30.0, cars=2, seed=6).generate()
    moving = generated.signals[0]
    parked = generated.signals[1]
    frozen = type(parked)(
        car_id=1,
        position=np.repeat(parked.position[:1], parked.position.shape[0], axis=0),
        heading=parked.heading,
        speed=np.zeros_like(parked.speed),
        sideslip=parked.sideslip,
    )
    path = tmp_path / "with_parked.csv"
    write_csv(path, {0: moving, 1: frozen}, generated.time)
    return path


class TestCsvInput:
    def test_a_csv_loads(self, csv_session: Path) -> None:
        # The GUI could not do this at all before the loader existed; only the
        # CLI knew how.
        loaded = load_session(csv_session)
        assert loaded.signals
        assert loaded.sample_rate > 0.0

    def test_the_suffix_selects_the_reader(self, csv_session: Path) -> None:
        assert load_session(csv_session).meta.source_name == "csv"

    def test_seconds_limits_the_span(self, csv_session: Path) -> None:
        whole = load_session(csv_session)
        clipped = load_session(csv_session, seconds=10.0)
        assert clipped.time.shape[0] < whole.time.shape[0]
        assert clipped.duration == pytest.approx(10.0, abs=0.5)

    def test_every_car_keeps_the_same_frame_count_as_the_time_base(self, csv_session: Path) -> None:
        loaded = load_session(csv_session, seconds=10.0)
        for car in loaded.signals.values():
            assert car.position.shape[0] == loaded.time.shape[0]

    def test_source_notes_are_carried(self, tmp_path: Path, csv_session: Path) -> None:
        # A CSV without a heading column produces zero sideslip, and the reader
        # says so. That has to survive the trip through the loader.
        lines = csv_session.read_text(encoding="utf-8").splitlines()
        header = lines[0].split(",")
        keep = [i for i, name in enumerate(header) if name not in {"heading_rad", "sideslip_rad"}]
        stripped = tmp_path / "no_heading.csv"
        stripped.write_text(
            "\n".join(",".join(line.split(",")[i] for i in keep) for line in lines) + "\n",
            encoding="utf-8",
        )
        assert any("drift" in note for note in load_session(stripped).notes)


class TestParkedCars:
    def test_a_parked_car_is_dropped(self, csv_with_a_parked_car: Path) -> None:
        loaded = load_session(csv_with_a_parked_car)
        assert set(loaded.signals) == {0}

    def test_it_is_recorded_as_skipped(self, csv_with_a_parked_car: Path) -> None:
        assert load_session(csv_with_a_parked_car).skipped == (1,)

    def test_skipped_cars_are_named(self, csv_with_a_parked_car: Path) -> None:
        # The regression guard. Filtering silently would show a one-car analysis
        # of a two-car session with no explanation.
        notes = load_session(csv_with_a_parked_car).notes
        assert any("never moved" in note for note in notes)
        assert any("1" in note for note in notes)

    def test_nothing_is_said_when_nothing_was_skipped(self, csv_session: Path) -> None:
        loaded = load_session(csv_session)
        assert loaded.skipped == ()
        assert not any("never moved" in note for note in loaded.notes)

    def test_a_session_of_only_parked_cars_is_refused(self, tmp_path: Path) -> None:
        generated = SyntheticSource(seconds=20.0, cars=1, seed=2).generate()
        car = generated.signals[0]
        frozen = type(car)(
            car_id=0,
            position=np.repeat(car.position[:1], car.position.shape[0], axis=0),
            heading=car.heading,
            speed=np.zeros_like(car.speed),
            sideslip=car.sideslip,
        )
        path = tmp_path / "all_parked.csv"
        write_csv(path, {0: frozen}, generated.time)
        with pytest.raises(ValueError, match="nothing to analyse"):
            load_session(path)

    def test_the_note_names_every_id_in_order(self) -> None:
        note = skipped_note((9, 2, 5))
        assert "2, 5, 9" in note

    def test_the_threshold_is_generous_enough_to_keep_a_short_run(self) -> None:
        # A car that pulled off after one corner is a participant with a short
        # story; a car that never left its grid box is furniture.
        assert MIN_TRAVELLED_METRES == 50.0


class TestRefusal:
    def test_a_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises((OSError, ValueError)):
            load_session(tmp_path / "nope.acreplay")

    def test_an_unparseable_csv_raises_a_value_error(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.csv"
        path.write_text("not,a,session\n1,2,3\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing required column"):
            load_session(path)


class TestReportingAndCancellation:
    def test_loading_a_csv_needs_no_progress(self, csv_session: Path) -> None:
        # A single pass over a text file finishes before a bar would paint.
        recorder = _Recorder()
        load_session(csv_session, progress=recorder)
        assert recorder.labels == []

    def test_a_cancelled_csv_load_still_completes(self, csv_session: Path) -> None:
        # Nothing to interrupt, and refusing to load a file that takes
        # milliseconds would be worse than ignoring the request.
        assert load_session(csv_session, cancel=_AlwaysCancelled()).signals


@pytest.mark.fixtures
class TestReplayInput:
    """Only runs where real replays are present; the rest of the suite does not."""

    @staticmethod
    def _replay() -> Path:
        directory = Path(__file__).resolve().parents[2] / "replays"
        found = sorted(directory.glob("*.acreplay")) if directory.is_dir() else []
        if not found:
            pytest.skip("no .acreplay fixtures in replays/")
        return found[0]

    SPAN_SECONDS = 60.0
    """Long enough that the player has covered the fifty-metre threshold.

    A five-second window looks like it should be enough and is not: the fixtures
    begin on a standing start, so at five seconds nobody has moved far enough to
    count as a participant and the loader correctly refuses the whole session.
    """

    def test_decoding_reports_the_decode_stage(self) -> None:
        recorder = _Recorder()
        load_session(self._replay(), seconds=self.SPAN_SECONDS, progress=recorder)
        assert recorder.labels == [str(Stage.DECODE)]

    def test_it_advances_once_per_car(self) -> None:
        recorder = _Recorder()
        loaded = load_session(self._replay(), seconds=self.SPAN_SECONDS, progress=recorder)
        assert recorder.advances == len(loaded.signals) + len(loaded.skipped)

    def test_decoding_can_be_cancelled(self) -> None:
        with pytest.raises(OperationCancelledError) as caught:
            load_session(self._replay(), seconds=self.SPAN_SECONDS, cancel=_AlwaysCancelled())
        assert caught.value.stage == str(Stage.DECODE)

    def test_a_full_grid_mostly_never_moves(self) -> None:
        # The reason parked-car filtering exists at all, asserted against reality
        # rather than against a fixture built to prove it.
        loaded = load_session(self._replay(), seconds=self.SPAN_SECONDS)
        assert loaded.skipped
        assert len(loaded.skipped) > len(loaded.signals)
