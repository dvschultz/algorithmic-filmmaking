"""Tests for FreeAssociationWorker.

Tests call worker.run() directly (synchronously) rather than start() to
avoid QThread scheduling complications in the test environment. This is
the same pattern used in tests/test_gaze_worker.py.
"""

from unittest.mock import patch

import pytest


def _worker_args():
    """Minimal valid arguments for constructing the worker."""
    return {
        "current_clip_metadata": "Description: a close-up",
        "candidate_digests": [("c1", "CU | warm"), ("c2", "wide | cool")],
        "recent_rationales": [],
        "rejected_short_ids": [],
    }


class TestFreeAssociationWorker:
    def test_happy_path_emits_proposal_ready(self):
        from ui.workers.free_association_worker import FreeAssociationWorker

        proposals = []
        errors = []

        with patch(
            "ui.workers.free_association_worker.propose_next_clip",
            return_value=("c1", "Good transition based on shared warm tones"),
        ):
            worker = FreeAssociationWorker(**_worker_args())
            worker.proposal_ready.connect(
                lambda cid, rat: proposals.append((cid, rat))
            )
            worker.error.connect(lambda msg: errors.append(msg))
            worker.run()

        assert proposals == [("c1", "Good transition based on shared warm tones")]
        assert errors == []

    def test_value_error_emits_error_signal(self):
        """ValueError from propose_next_clip (None content, bad JSON, etc.)."""
        from ui.workers.free_association_worker import FreeAssociationWorker

        proposals = []
        errors = []

        with patch(
            "ui.workers.free_association_worker.propose_next_clip",
            side_effect=ValueError("LLM returned no content"),
        ):
            worker = FreeAssociationWorker(**_worker_args())
            worker.proposal_ready.connect(
                lambda cid, rat: proposals.append((cid, rat))
            )
            worker.error.connect(lambda msg: errors.append(msg))
            worker.run()

        assert proposals == []
        assert errors == ["LLM returned no content"]

    def test_network_exception_emits_error_signal(self):
        """Generic exceptions (network, auth) are caught and routed to error."""
        from ui.workers.free_association_worker import FreeAssociationWorker

        proposals = []
        errors = []

        with patch(
            "ui.workers.free_association_worker.propose_next_clip",
            side_effect=ConnectionError("Connection timed out"),
        ):
            worker = FreeAssociationWorker(**_worker_args())
            worker.proposal_ready.connect(
                lambda cid, rat: proposals.append((cid, rat))
            )
            worker.error.connect(lambda msg: errors.append(msg))
            worker.run()

        assert proposals == []
        assert len(errors) == 1
        assert "LLM call failed" in errors[0]
        assert "Connection timed out" in errors[0]

    def test_cancelled_before_llm_call_does_not_emit_proposal(self):
        """Worker cancelled after LLM call succeeds does not emit."""
        from ui.workers.free_association_worker import FreeAssociationWorker

        proposals = []
        errors = []

        def cancel_then_return(**kwargs):
            # Simulate cancellation happening mid-flight by setting the flag
            # before returning the "successful" result
            worker.cancel()
            return ("c1", "Would be a great proposal")

        with patch(
            "ui.workers.free_association_worker.propose_next_clip",
            side_effect=cancel_then_return,
        ):
            worker = FreeAssociationWorker(**_worker_args())
            worker.proposal_ready.connect(
                lambda cid, rat: proposals.append((cid, rat))
            )
            worker.error.connect(lambda msg: errors.append(msg))
            worker.run()

        assert proposals == []
        assert errors == []

    def test_cancelled_after_error_does_not_emit_error(self):
        """If cancelled during an error path, the error signal is suppressed."""
        from ui.workers.free_association_worker import FreeAssociationWorker

        proposals = []
        errors = []

        def cancel_then_raise(**kwargs):
            worker.cancel()
            raise ValueError("some error")

        with patch(
            "ui.workers.free_association_worker.propose_next_clip",
            side_effect=cancel_then_raise,
        ):
            worker = FreeAssociationWorker(**_worker_args())
            worker.proposal_ready.connect(
                lambda cid, rat: proposals.append((cid, rat))
            )
            worker.error.connect(lambda msg: errors.append(msg))
            worker.run()

        assert proposals == []
        assert errors == []  # Suppressed because cancelled

    def test_passes_constructor_args_to_propose_next_clip(self):
        """Worker forwards all its constructor args to the core function."""
        from ui.workers.free_association_worker import FreeAssociationWorker

        with patch(
            "ui.workers.free_association_worker.propose_next_clip",
            return_value=("c1", "ok"),
        ) as mock_propose:
            worker = FreeAssociationWorker(
                current_clip_metadata="current meta",
                candidate_digests=[("c1", "digest one"), ("c2", "digest two")],
                recent_rationales=["r1", "r2"],
                rejected_short_ids=["c3"],
                model="gemini-custom",
                temperature=0.5,
            )
            worker.run()

            mock_propose.assert_called_once_with(
                current_clip_metadata="current meta",
                candidate_digests=[("c1", "digest one"), ("c2", "digest two")],
                recent_rationales=["r1", "r2"],
                rejected_short_ids=["c3"],
                model="gemini-custom",
                temperature=0.5,
            )
