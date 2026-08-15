"""Cooperative cancellation, so a long analysis can be stopped.

A user started a 24-minute replay, watched it for ten minutes, and had no way to
stop it. The only response the window could give to a second request was "already
working; wait for the current run" — which is not a design so much as an absence.

**Cooperative, never pre-emptive.** ``QThread.terminate()`` and its equivalents
kill a thread wherever it happens to be, which here means mid-NumPy-allocation or
mid-``read`` on a 233 MB file, leaving memory and file handles in a state nobody
has reasoned about. The long loops poll instead, and the cost of that is a bounded
latency between asking and stopping, documented per call site.

**Why a separate port rather than a return value from
:class:`~assetto_ai_director.application.ports.progress.ProgressReporter`.**
Making ``advance()`` return a "keep going" flag needs no new plumbing at all,
which is exactly what makes it tempting. But it repurposes a published protocol
that the CLI, the MCP server and any third-party plugin already implement, turning
a ``None`` return into a live control signal that every existing implementation
returns implicitly. Two concerns that arrive together are still two concerns.

:class:`CancelFlag` is built on :class:`threading.Event`, which is standard
library — so the application layer stays framework-free and the import-linter
contract that enforces it needs no exception. The GUI thread sets the flag; the
worker thread polls it. No Qt object crosses the boundary in either direction.
"""

from __future__ import annotations

import threading
from typing import Protocol, runtime_checkable

__all__ = ["CancelFlag", "CancelToken", "NeverCancelled"]


@runtime_checkable
class CancelToken(Protocol):
    """Asked, periodically, whether the caller still wants the result."""

    def cancelled(self) -> bool:
        """Report whether the work should stop.

        Must be cheap: it is called from inside loops that run hundreds of
        thousands of times, and must be safe to call from a thread other than the
        one that requested cancellation.

        Returns:
            ``True`` once cancellation has been requested.
        """
        ...


class NeverCancelled:
    """A token that never asks for a stop.

    The default, so the pipeline can poll unconditionally rather than guarding
    every call site with a null check.
    """

    def cancelled(self) -> bool:
        """Report that the work should continue.

        Returns:
            Always ``False``.
        """
        return False


class CancelFlag:
    """A token one thread sets and another polls.

    Set-once and never reset: a run that was cancelled stays cancelled, and a new
    run gets a new flag. Reusing a flag across runs would make "was this cancelled"
    depend on when you asked.
    """

    __slots__ = ("_event",)

    def __init__(self) -> None:
        """Create an uncancelled flag."""
        self._event = threading.Event()

    def cancel(self) -> None:
        """Request that the work stop at the next opportunity."""
        self._event.set()

    def cancelled(self) -> bool:
        """Report whether cancellation has been requested.

        Returns:
            ``True`` once :meth:`cancel` has been called.
        """
        return self._event.is_set()
