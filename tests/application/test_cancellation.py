"""Tests for stopping an analysis that is already running.

A user started a 24-minute replay and had no way out of it. The window's only
answer to a second request was "already working; wait for the current run", which
is not a design so much as an absence.

What matters here is not that cancellation *happens* but that it happens
**everywhere the wait is long**. A cancel button that works during a three-second
detection pass and not during the ninety-second scheduler is worse than none: the
user presses it, nothing happens, and they conclude the application has hung —
which is the exact belief this whole change set exists to prevent.

So the parametrised test asserts every long stage can be interrupted, by name. It
uses a token that is *also* a progress reporter, arming itself when the stage
under test announces itself. That is what makes the test exact rather than
approximate: the alternative — cancel after N polls and see where it lands —
needs a different N per stage, per machine, and per change to any loop bound.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from assetto_ai_director.application.director.styles import BUILTIN_STYLES
from assetto_ai_director.application.pipeline import direct
from assetto_ai_director.application.ports.cancellation import CancelFlag, NeverCancelled
from assetto_ai_director.application.ports.progress import Stage
from assetto_ai_director.domain.errors import AssettoDirectorError, OperationCancelledError
from assetto_ai_director.infrastructure.synthetic.source import SyntheticSession, SyntheticSource


@dataclass
class TripOnStage:
    """Cancels once a named stage has begun. One object, both protocols.

    It is a ``ProgressReporter`` *and* a ``CancelToken``, which is what makes the
    test precise: the stages announce themselves, so the token can wait for the
    one under test instead of guessing at an iteration count. The obvious
    alternative — cancel after N polls and see where it lands — needs a different
    N per stage, per machine, and per change to any loop bound.

    Args:
        target: The stage to interrupt.
    """

    target: str
    started: list[str] = field(default_factory=list)
    armed: bool = False

    def start(self, description: str, total: int | None = None) -> None:
        del total
        self.started.append(description)
        if description == self.target:
            self.armed = True

    def advance(self, amount: int = 1) -> None:
        del amount

    def finish(self) -> None:
        pass

    def cancelled(self) -> bool:
        return self.armed


@dataclass
class StageRecorder:
    """Records which stages started, so a test can say where a run got to."""

    labels: list[str] = field(default_factory=list)

    def start(self, description: str, total: int | None = None) -> None:
        del total
        self.labels.append(description)

    def advance(self, amount: int = 1) -> None:
        del amount

    def finish(self) -> None:
        pass


@pytest.fixture(scope="module")
def session() -> SyntheticSession:
    """A session large enough that every stage has real work to interrupt."""
    return SyntheticSource(seconds=45.0, cars=3, seed=4).generate()


def _run(session: SyntheticSession, **kwargs: object) -> object:
    """Direct the session, forwarding keyword arguments.

    Args:
        session: The generated session.
        **kwargs: Passed through to ``direct``.

    Returns:
        The timeline.
    """
    return direct(
        session.signals,
        session.meta,
        BUILTIN_STYLES["initial_d"],
        session.time,
        float(session.sample_rate),
        **kwargs,  # type: ignore[arg-type]
    )


class TestTheTokenItself:
    def test_a_fresh_flag_is_not_cancelled(self) -> None:
        assert not CancelFlag().cancelled()

    def test_cancelling_sticks(self) -> None:
        # Set-once: a run that was cancelled stays cancelled, so "was this
        # cancelled" does not depend on when you ask.
        flag = CancelFlag()
        flag.cancel()
        assert flag.cancelled()
        assert flag.cancelled()

    def test_cancelling_twice_is_harmless(self) -> None:
        flag = CancelFlag()
        flag.cancel()
        flag.cancel()
        assert flag.cancelled()

    def test_the_null_token_never_cancels(self) -> None:
        assert not NeverCancelled().cancelled()

    def test_flags_are_independent(self) -> None:
        first, second = CancelFlag(), CancelFlag()
        first.cancel()
        assert not second.cancelled()


class TestCancellingARun:
    def test_an_already_cancelled_run_stops_immediately(self, session: SyntheticSession) -> None:
        flag = CancelFlag()
        flag.cancel()
        with pytest.raises(OperationCancelledError):
            _run(session, cancel=flag)

    def test_it_stops_in_the_first_stage(self, session: SyntheticSession) -> None:
        # Not merely "it raised" — it must raise without doing the rest of the
        # work, which is the entire point.
        flag = CancelFlag()
        flag.cancel()
        recorder = StageRecorder()
        with pytest.raises(OperationCancelledError):
            _run(session, cancel=flag, progress=recorder)
        assert recorder.labels == [str(Stage.DETECT)]

    def test_the_error_names_the_stage(self, session: SyntheticSession) -> None:
        flag = CancelFlag()
        flag.cancel()
        with pytest.raises(OperationCancelledError) as caught:
            _run(session, cancel=flag)
        assert caught.value.stage == str(Stage.DETECT)

    def test_the_message_reads_as_a_choice_not_a_fault(self, session: SyntheticSession) -> None:
        # A user who pressed Cancel and is shown "Could not direct this replay"
        # has been told their machine broke.
        flag = CancelFlag()
        flag.cancel()
        with pytest.raises(OperationCancelledError) as caught:
            _run(session, cancel=flag)
        message = str(caught.value).lower()
        assert "cancel" in message
        assert "fail" not in message
        assert "error" not in message

    @pytest.mark.parametrize(
        "stage",
        [Stage.DETECT, Stage.SCORE, Stage.CANDIDATES, Stage.SCHEDULE, Stage.CAMERA],
    )
    def test_every_long_stage_can_be_interrupted(
        self, session: SyntheticSession, stage: Stage
    ) -> None:
        # The whole point. A cancel button that works during the three-second
        # detection pass and not during the ninety-second scheduler is worse than
        # none: the user presses it, nothing happens, and concludes the
        # application has hung. A stage with a long loop and no poll fails here
        # by running to completion instead of raising.
        token = TripOnStage(str(stage))
        with pytest.raises(OperationCancelledError) as caught:
            _run(session, cancel=token, progress=token)
        assert caught.value.stage == str(stage)

    def test_a_late_cancellation_yields_no_partial_timeline(
        self, session: SyntheticSession
    ) -> None:
        # The camera stage is last and has already built most of the shots by the
        # time it is interrupted. None of them may reach the caller — a partial
        # edit that looks complete is the failure this project is arranged
        # against, and `direct` is pure, so raising is the whole cleanup.
        token = TripOnStage(str(Stage.CAMERA))
        with pytest.raises(OperationCancelledError):
            _run(session, cancel=token, progress=token)

    def test_it_does_not_run_stages_after_the_cancelled_one(
        self, session: SyntheticSession
    ) -> None:
        token = TripOnStage(str(Stage.SCORE))
        with pytest.raises(OperationCancelledError):
            _run(session, cancel=token, progress=token)
        assert token.started == [str(Stage.DETECT), str(Stage.SCORE)]


class TestNotCancelling:
    def test_a_run_with_no_token_completes(self, session: SyntheticSession) -> None:
        timeline = _run(session)
        assert timeline.shot_count > 0  # type: ignore[attr-defined]

    def test_a_never_cancelled_token_completes(self, session: SyntheticSession) -> None:
        timeline = _run(session, cancel=NeverCancelled())
        assert timeline.shot_count > 0  # type: ignore[attr-defined]

    def test_an_untripped_flag_does_not_change_the_edit(self, session: SyntheticSession) -> None:
        # Polling must not perturb the result, or a cancellable run and a plain
        # one would produce different edits from the same input.
        quiet = _run(session)
        watched = _run(session, cancel=CancelFlag())
        assert [shot.candidate for shot in quiet.shots] == [  # type: ignore[attr-defined]
            shot.candidate
            for shot in watched.shots  # type: ignore[attr-defined]
        ]


class TestErrorHierarchy:
    def test_it_is_catchable_as_a_project_error(self) -> None:
        # So it can never escape an existing handler as an unhandled exception.
        assert issubclass(OperationCancelledError, AssettoDirectorError)

    def test_it_is_distinguishable_from_a_real_failure(self) -> None:
        # And so every handler can catch it explicitly *before* the general case
        # and report it as a choice rather than a fault.
        assert OperationCancelledError is not AssettoDirectorError
        assert OperationCancelledError("Selecting shots").stage == "Selecting shots"
