"""Tests for stage reporting across a full run.

These exist because of a specific failure, and they are written to make that
failure impossible rather than merely absent.

`_report()` in the pipeline used to skip `start()` whenever the total was
`None`. Only the first of four stages passed a total, so three stage names never
reached any reporter, and the one that did was announcing a count of *pipeline
stages* under the name of a single stage. Camera-path construction — the longest
phase of a full session — reported nothing at all. A user directing a 24-minute
replay watched "Detecting events" climb to 100%, then sit there, unmoving and
misnamed, for ten minutes.

Every assertion here is about **what a person watching the bar would conclude**.
A progress display is a claim about the state of the world, and a wrong one is
worse than none: it converts "I don't know how long this will take" into "this is
finished and hung."

The regression guard is `test_no_stage_reports_an_unknown_total`. The original bug
was not that a stage was missing — it was that a stage declined to say how much
work it had, and the reporting layer silently dropped it for that. A stage whose
total is computable must supply it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from assetto_ai_director.application.director.styles import BUILTIN_STYLES
from assetto_ai_director.application.pipeline import direct
from assetto_ai_director.application.ports.progress import PIPELINE_STAGES, Stage
from assetto_ai_director.infrastructure.synthetic.source import SyntheticSource

# Stages the pipeline itself is responsible for. DECODE happens in the loader,
# before there are any signals to hand over, so `direct()` never reports it.
PIPELINE_OWNED = tuple(stage for stage in PIPELINE_STAGES if stage is not Stage.DECODE)


@dataclass
class _Call:
    """One reporter call, recorded in order."""

    kind: str
    description: str = ""
    total: int | None = None
    amount: int = 0


@dataclass
class Recorder:
    """A reporter that remembers everything, in order.

    Satisfies `ProgressReporter` structurally.
    """

    calls: list[_Call] = field(default_factory=list)

    def start(self, description: str, total: int | None = None) -> None:
        self.calls.append(_Call("start", description, total))

    def advance(self, amount: int = 1) -> None:
        self.calls.append(_Call("advance", amount=amount))

    def finish(self) -> None:
        self.calls.append(_Call("finish"))

    @property
    def started(self) -> list[_Call]:
        return [call for call in self.calls if call.kind == "start"]

    @property
    def labels(self) -> list[str]:
        return [call.description for call in self.started]

    def advances_within(self, description: str) -> int:
        """Total units advanced between one stage's start and its finish.

        Args:
            description: The stage label.

        Returns:
            The summed amount, which should reach the declared total.
        """
        total = 0
        inside = False
        for call in self.calls:
            if call.kind == "start":
                inside = call.description == description
            elif call.kind == "finish" and inside:
                break
            elif call.kind == "advance" and inside:
                total += call.amount
        return total

    def declared_total(self, description: str) -> int | None:
        """The total a stage declared when it started.

        Args:
            description: The stage label.

        Returns:
            The declared total, or ``None`` if it declared none.
        """
        for call in self.started:
            if call.description == description:
                return call.total
        return None


@pytest.fixture(scope="module")
def recorded() -> Recorder:
    """One directed run with every reporter call recorded."""
    session = SyntheticSource(seconds=45.0, cars=3, seed=7).generate()
    recorder = Recorder()
    direct(
        session.signals,
        session.meta,
        BUILTIN_STYLES["initial_d"],
        session.time,
        float(session.sample_rate),
        progress=recorder,
    )
    return recorder


class TestEveryStageIsAnnounced:
    def test_all_pipeline_stages_appear(self, recorded: Recorder) -> None:
        # The direct regression: three of these used to be dropped entirely.
        assert set(recorded.labels) == {str(stage) for stage in PIPELINE_OWNED}

    def test_each_appears_exactly_once(self, recorded: Recorder) -> None:
        for stage in PIPELINE_OWNED:
            assert recorded.labels.count(str(stage)) == 1, f"{stage} announced more than once"

    def test_they_appear_in_order(self, recorded: Recorder) -> None:
        assert recorded.labels == [str(stage) for stage in PIPELINE_OWNED]

    def test_the_camera_stage_is_reported(self, recorded: Recorder) -> None:
        # Called out separately because it is the longest stage of a real
        # session and it used to report nothing whatsoever.
        assert str(Stage.CAMERA) in recorded.labels


class TestTotalsAreHonest:
    def test_no_stage_reports_an_unknown_total(self, recorded: Recorder) -> None:
        # The guard that caused the original bug was `total is not None`. Every
        # stage the pipeline owns can compute its total before starting, so a
        # None here means someone reintroduced a stage that cannot say how much
        # work it has — and the reporting layer has no way to draw that.
        for stage in PIPELINE_OWNED:
            assert recorded.declared_total(str(stage)) is not None, f"{stage} declared no total"

    def test_totals_are_positive(self, recorded: Recorder) -> None:
        for stage in PIPELINE_OWNED:
            total = recorded.declared_total(str(stage))
            assert total is not None
            assert total > 0, f"{stage} declared a total of {total}"

    @pytest.mark.parametrize("stage", [Stage.DETECT, Stage.SCORE, Stage.CANDIDATES, Stage.CAMERA])
    def test_advances_reach_the_declared_total(self, recorded: Recorder, stage: Stage) -> None:
        # Excludes SCHEDULE, which ticks in batches of 512 and so lands short of
        # its total by design — see _PROGRESS_BATCH.
        declared = recorded.declared_total(str(stage))
        assert declared is not None
        assert recorded.advances_within(str(stage)) == declared

    def test_the_scheduler_never_overshoots(self, recorded: Recorder) -> None:
        # Batched ticking may under-report; it must never claim more work done
        # than exists, which would drive a progress bar past 100%.
        declared = recorded.declared_total(str(Stage.SCHEDULE))
        assert declared is not None
        assert recorded.advances_within(str(Stage.SCHEDULE)) <= declared


class TestStagesAreBalanced:
    def test_every_start_has_a_finish(self, recorded: Recorder) -> None:
        # `finish()` was never called by the pipeline at all, so no stage ever
        # cleanly completed and a checklist UI could not tick anything off.
        starts = sum(1 for call in recorded.calls if call.kind == "start")
        finishes = sum(1 for call in recorded.calls if call.kind == "finish")
        assert starts == finishes

    def test_stages_do_not_overlap(self, recorded: Recorder) -> None:
        depth = 0
        for call in recorded.calls:
            if call.kind == "start":
                depth += 1
                assert depth == 1, "a stage started while another was still open"
            elif call.kind == "finish":
                depth -= 1
        assert depth == 0

    def test_nothing_is_reported_after_the_last_finish(self, recorded: Recorder) -> None:
        # The window must not be left believing work continues.
        assert recorded.calls[-1].kind == "finish"


class TestReportingIsOptional:
    def test_the_pipeline_runs_without_a_reporter(self) -> None:
        session = SyntheticSource(seconds=30.0, cars=2, seed=1).generate()
        timeline = direct(
            session.signals,
            session.meta,
            BUILTIN_STYLES["initial_d"],
            session.time,
            float(session.sample_rate),
        )
        assert timeline.shot_count > 0

    def test_reporting_does_not_change_the_edit(self) -> None:
        # Observation must not alter the result. If it did, every bug report
        # against the GUI would be unreproducible from the CLI.
        session = SyntheticSource(seconds=30.0, cars=2, seed=1).generate()
        args = (
            session.signals,
            session.meta,
            BUILTIN_STYLES["initial_d"],
            session.time,
            float(session.sample_rate),
        )
        quiet = direct(*args)
        watched = direct(*args, progress=Recorder())
        assert [shot.candidate for shot in quiet.shots] == [
            shot.candidate for shot in watched.shots
        ]
