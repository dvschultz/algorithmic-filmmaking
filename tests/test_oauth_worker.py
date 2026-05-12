"""Tests for ui/workers/oauth_worker.OAuthWorker.

Focus: the signal-duplication guard pattern (R-S5's UniqueConnection
buddy) and the terminal-emission contract — auth_complete OR
auth_failed fires exactly once per run, never both.

The PySide6 QApplication is required for signal connection. Tests use
QSignalSpy to capture emissions without running an event loop.
"""

from __future__ import annotations

import sys

import pytest

# Skip the whole module on headless CI runs where PySide6 isn't installed.
PySide6 = pytest.importorskip("PySide6", reason="PySide6 not available")

from PySide6.QtCore import QCoreApplication, QObject
from PySide6.QtTest import QSignalSpy

from ui.workers.oauth_worker import (
    AUTH_FAILED_CANCELLED,
    AUTH_FAILED_TIMEOUT,
    AUTH_FAILED_NETWORK,
    AUTH_FAILED_REJECTED,
    AUTH_FAILED_MALFORMED,
    OAuthWorker,
)


@pytest.fixture(scope="module")
def qapp():
    """Module-scoped QCoreApplication for signal handling.

    QCoreApplication is sufficient — no widgets are constructed here, so
    the heavier QApplication isn't necessary.
    """
    app = QCoreApplication.instance() or QCoreApplication(sys.argv)
    yield app


class TestTerminalEmissionGuard:
    """auth_complete / auth_failed must fire at most once per run."""

    def test_complete_fires_once_then_suppresses_repeat(self, qapp):
        worker = OAuthWorker()
        spy_complete = QSignalSpy(worker.auth_complete)

        worker._emit_terminal_complete({"access_token": "at_abc"})
        worker._emit_terminal_complete({"access_token": "at_abc"})

        assert spy_complete.count() == 1

    def test_failed_fires_once_then_suppresses_repeat(self, qapp):
        worker = OAuthWorker()
        spy_failed = QSignalSpy(worker.auth_failed)

        worker._emit_terminal_failed(AUTH_FAILED_CANCELLED, "Cancelled.")
        worker._emit_terminal_failed(AUTH_FAILED_NETWORK, "Network error.")

        assert spy_failed.count() == 1

    def test_complete_blocks_subsequent_failed(self, qapp):
        """After auth_complete, a failed emission is suppressed."""
        worker = OAuthWorker()
        spy_complete = QSignalSpy(worker.auth_complete)
        spy_failed = QSignalSpy(worker.auth_failed)

        worker._emit_terminal_complete({"access_token": "at"})
        worker._emit_terminal_failed(AUTH_FAILED_NETWORK, "Late error.")

        assert spy_complete.count() == 1
        assert spy_failed.count() == 0

    def test_failed_blocks_subsequent_complete(self, qapp):
        worker = OAuthWorker()
        spy_complete = QSignalSpy(worker.auth_complete)
        spy_failed = QSignalSpy(worker.auth_failed)

        worker._emit_terminal_failed(AUTH_FAILED_CANCELLED, "Cancelled.")
        worker._emit_terminal_complete({"access_token": "at"})

        assert spy_failed.count() == 1
        assert spy_complete.count() == 0


class TestGuardResetAcrossRuns:
    """The guard flag resets at the top of each run() so the same worker
    instance can be reused for sequential sign-in attempts.
    """

    def test_finished_handled_resets_on_run(self, qapp, monkeypatch):
        worker = OAuthWorker()

        # Force the flag set, simulating end of a prior run.
        worker._finished_handled = True

        # Replace _async_run with an immediate no-op so run() reaches the
        # guard reset without doing real network work.
        async def _noop():
            return None

        monkeypatch.setattr(worker, "_async_run", _noop)

        worker.run()
        # After the no-op coroutine completes, the flag should have been
        # reset to False at the top of run() — even though _async_run
        # never emitted a terminal signal.
        assert worker._finished_handled is False


class TestFailureCategoryConstants:
    """The constants must be stable strings — U5 branches on them."""

    def test_distinct_values(self):
        values = {
            AUTH_FAILED_CANCELLED,
            AUTH_FAILED_TIMEOUT,
            AUTH_FAILED_NETWORK,
            AUTH_FAILED_REJECTED,
            AUTH_FAILED_MALFORMED,
        }
        # All distinct — no accidental collisions.
        assert len(values) == 5

    def test_values_are_strings(self):
        for v in (
            AUTH_FAILED_CANCELLED,
            AUTH_FAILED_TIMEOUT,
            AUTH_FAILED_NETWORK,
            AUTH_FAILED_REJECTED,
            AUTH_FAILED_MALFORMED,
        ):
            assert isinstance(v, str) and v
