"""Tests for ForcedAlignmentWorker.

Following the convention from tests/test_embedding_worker.py: call worker.run()
directly (synchronously) rather than start() to avoid QThread scheduling
complications in the test environment.

U3 of the Word Sequencer plan.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qapp_fixture():
    """Qt application required because the worker is a QThread subclass."""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def english_source(tmp_path):
    """A Source with a real file path (used as the clip's source media)."""
    from models.clip import Source

    video_file = tmp_path / "video.mp4"
    video_file.write_bytes(b"\x00" * 100)
    return Source(
        id="src-1",
        file_path=video_file,
        duration_seconds=60.0,
        fps=30.0,
    )


def _make_segment(start, end, text, language="en", words=None):
    """Build a TranscriptSegment with the U1 word/language fields."""
    from core.transcription import TranscriptSegment

    return TranscriptSegment(
        start_time=start,
        end_time=end,
        text=text,
        confidence=0.9,
        words=words,
        language=language,
    )


def _make_clip_with_transcript(
    clip_id: str,
    transcript: list,
    source_id: str = "src-1",
):
    """Build a Clip with a transcript attached (no other analysis fields)."""
    from tests.conftest import make_test_clip

    clip = make_test_clip(clip_id, source_id=source_id)
    clip.transcript = transcript
    return clip


def _make_word(start, end, text, probability=0.9):
    """Build a WordTimestamp."""
    from core.transcription import WordTimestamp

    return WordTimestamp(start=start, end=end, text=text, probability=probability)


# ---------------------------------------------------------------------------
# Patches shared across the tests
# ---------------------------------------------------------------------------


def _patch_feature_ready_and_audio_extract(monkeypatch_audio_path):
    """Return context-manager-style patches:

    1. core.feature_registry.check_feature_ready -> (True, [])
    2. The worker's per-clip audio extraction helper returns a fake WAV path
       (so the worker doesn't actually shell out to FFmpeg).
    """
    from contextlib import ExitStack

    cm = ExitStack()
    cm.enter_context(
        patch(
            "core.feature_registry.check_feature_ready", return_value=(True, [])
        )
    )
    cm.enter_context(
        patch(
            "core.analysis.alignment.extract_audio_to_wav",
            return_value=monkeypatch_audio_path,
        )
    )
    return cm


# ---------------------------------------------------------------------------
# Cancellation contract — the U3 plan's mandatory FIRST failing test.
# ---------------------------------------------------------------------------


class TestForcedAlignmentWorkerCancelContract:
    """`worker.cancel()` mid-run must:

    - stop emitting further `clip_aligned` signals,
    - still emit the lifecycle terminator (`alignment_completed`) exactly once,
    - leave already-aligned clips' partial progress intact (the *handler* writes
      back — the worker just emits; this test asserts no extra emissions).
    """

    def test_cancel_mid_run_stops_clip_aligned_and_emits_alignment_completed_once(
        self, qapp_fixture, english_source, tmp_path
    ):
        from ui.workers.forced_alignment_worker import ForcedAlignmentWorker

        # Three clips, all with English transcripts; words=None so each is a
        # candidate for alignment.
        clip1 = _make_clip_with_transcript(
            "c1", [_make_segment(0.0, 2.0, "hello world")]
        )
        clip2 = _make_clip_with_transcript(
            "c2", [_make_segment(0.0, 2.0, "second clip")]
        )
        clip3 = _make_clip_with_transcript(
            "c3", [_make_segment(0.0, 2.0, "third clip")]
        )

        # Fake audio path returned by the mocked extract helper.
        fake_wav = tmp_path / "fake.wav"
        fake_wav.write_bytes(b"RIFF")

        ready_calls: list[tuple[str, list]] = []
        completions: list[bool] = []

        call_count = {"n": 0}

        def fake_align_words(audio_path, segments):
            # Pretend we did the work and return one word per call.
            call_count["n"] += 1
            words = [_make_word(0.0, 0.5, "fakeword")]
            # Cancel after the FIRST clip has been processed and emitted.
            # The worker's loop checks cancellation at the top of each clip
            # iteration, so this guarantees clip2 and clip3 never emit.
            if call_count["n"] == 1:
                worker.cancel()
            return words

        with _patch_feature_ready_and_audio_extract(fake_wav), patch(
            "core.analysis.alignment.align_words",
            side_effect=fake_align_words,
        ):
            worker = ForcedAlignmentWorker(
                clips=[clip1, clip2, clip3],
                sources_by_id={english_source.id: english_source},
            )
            worker.clip_aligned.connect(
                lambda cid, words: ready_calls.append((cid, words))
            )
            worker.alignment_completed.connect(lambda: completions.append(True))
            worker.run()

        # Only the first clip's signal was emitted; the cancel was honoured
        # at the top of the second iteration.
        assert [cid for cid, _ in ready_calls] == ["c1"]
        # Lifecycle terminator emits exactly once.
        assert completions == [True]
        # align_words was only called once — the worker did not re-enter for c2 or c3.
        assert call_count["n"] == 1


# ---------------------------------------------------------------------------
# Happy path: 3-clip run emits clip_aligned 3 times, progress, then completed.
# ---------------------------------------------------------------------------


class TestForcedAlignmentWorkerHappyPath:
    def test_three_clips_emit_three_clip_aligned_then_alignment_completed(
        self, qapp_fixture, english_source, tmp_path
    ):
        from ui.workers.forced_alignment_worker import ForcedAlignmentWorker

        clips = [
            _make_clip_with_transcript(
                f"c{i}", [_make_segment(0.0, 2.0, f"text {i}")]
            )
            for i in range(3)
        ]

        fake_wav = tmp_path / "fake.wav"
        fake_wav.write_bytes(b"RIFF")

        ready_calls: list[tuple[str, list]] = []
        progress_events: list[tuple[int, int]] = []
        completions: list[bool] = []

        def fake_align_words(audio_path, segments):
            return [_make_word(0.0, 0.5, "word")]

        with _patch_feature_ready_and_audio_extract(fake_wav), patch(
            "core.analysis.alignment.align_words",
            side_effect=fake_align_words,
        ):
            worker = ForcedAlignmentWorker(
                clips=clips,
                sources_by_id={english_source.id: english_source},
            )
            worker.clip_aligned.connect(
                lambda cid, words: ready_calls.append((cid, words))
            )
            worker.progress.connect(
                lambda c, t: progress_events.append((c, t))
            )
            worker.alignment_completed.connect(lambda: completions.append(True))
            worker.run()

        assert [cid for cid, _ in ready_calls] == ["c0", "c1", "c2"]
        # Each word list is the mocked single word per clip.
        for _cid, words in ready_calls:
            assert len(words) == 1
            assert words[0].text == "word"

        # Progress fires incrementally — initial (0, total) plus one per clip.
        assert progress_events == [(0, 3), (1, 3), (2, 3), (3, 3)]
        # Completion fires once at the end.
        assert completions == [True]

    def test_progress_emits_zero_total_at_start_before_first_clip(
        self, qapp_fixture, english_source, tmp_path
    ):
        """Pattern from transcription_worker: emit (0, total) before loop."""
        from ui.workers.forced_alignment_worker import ForcedAlignmentWorker

        clips = [
            _make_clip_with_transcript(
                "c1", [_make_segment(0.0, 2.0, "text one")]
            )
        ]
        fake_wav = tmp_path / "fake.wav"
        fake_wav.write_bytes(b"RIFF")

        progress_events: list[tuple[int, int]] = []

        with _patch_feature_ready_and_audio_extract(fake_wav), patch(
            "core.analysis.alignment.align_words",
            return_value=[_make_word(0.0, 0.5, "word")],
        ):
            worker = ForcedAlignmentWorker(
                clips=clips,
                sources_by_id={english_source.id: english_source},
            )
            worker.progress.connect(
                lambda c, t: progress_events.append((c, t))
            )
            worker.run()

        # First event is (0, 1) (pre-loop preload progress); last is (1, 1).
        assert progress_events[0] == (0, 1)
        assert progress_events[-1] == (1, 1)


# ---------------------------------------------------------------------------
# Empty input list: immediate completion, no clip_aligned, no model load.
# ---------------------------------------------------------------------------


class TestForcedAlignmentWorkerEmptyInput:
    def test_empty_clip_list_emits_completion_without_clip_aligned(
        self, qapp_fixture
    ):
        from ui.workers.forced_alignment_worker import ForcedAlignmentWorker

        ready_calls: list = []
        completions: list[bool] = []

        with patch(
            "core.feature_registry.check_feature_ready", return_value=(True, [])
        ), patch(
            "core.analysis.alignment.align_words"
        ) as mock_align, patch(
            "core.analysis.alignment.extract_audio_to_wav"
        ) as mock_extract:
            worker = ForcedAlignmentWorker(clips=[], sources_by_id={})
            worker.clip_aligned.connect(
                lambda cid, words: ready_calls.append(cid)
            )
            worker.alignment_completed.connect(lambda: completions.append(True))
            worker.run()

        # No alignment work attempted, no audio extracted.
        mock_align.assert_not_called()
        mock_extract.assert_not_called()
        assert ready_calls == []
        assert completions == [True]


# ---------------------------------------------------------------------------
# Skip predicate: a clip whose every segment already has words is skipped.
# ---------------------------------------------------------------------------


class TestForcedAlignmentWorkerSkipPredicate:
    def test_clip_with_all_segments_populated_is_skipped(
        self, qapp_fixture, english_source, tmp_path
    ):
        from ui.workers.forced_alignment_worker import ForcedAlignmentWorker

        already_aligned = _make_clip_with_transcript(
            "c-aligned",
            [
                _make_segment(
                    0.0,
                    1.0,
                    "done",
                    words=[_make_word(0.0, 0.5, "done")],
                )
            ],
        )
        partially_aligned = _make_clip_with_transcript(
            "c-partial",
            [
                _make_segment(
                    0.0,
                    1.0,
                    "first",
                    words=[_make_word(0.0, 0.5, "first")],
                ),
                # Second segment .words is None → clip needs re-alignment.
                _make_segment(1.0, 2.0, "second", words=None),
            ],
        )
        empty_words_clip = _make_clip_with_transcript(
            "c-empty-words",
            [
                # Empty list means "aligned but produced no words" — still aligned.
                _make_segment(0.0, 1.0, "silence", words=[]),
            ],
        )
        unaligned = _make_clip_with_transcript(
            "c-unaligned", [_make_segment(0.0, 2.0, "needs alignment")]
        )

        fake_wav = tmp_path / "fake.wav"
        fake_wav.write_bytes(b"RIFF")

        ready_calls: list[str] = []

        with _patch_feature_ready_and_audio_extract(fake_wav), patch(
            "core.analysis.alignment.align_words",
            return_value=[_make_word(0.0, 0.5, "word")],
        ) as mock_align:
            worker = ForcedAlignmentWorker(
                clips=[already_aligned, partially_aligned, empty_words_clip, unaligned],
                sources_by_id={english_source.id: english_source},
            )
            worker.clip_aligned.connect(
                lambda cid, words: ready_calls.append(cid)
            )
            worker.run()

        # Only the partially-aligned and unaligned clips should run.
        assert sorted(ready_calls) == ["c-partial", "c-unaligned"]
        assert mock_align.call_count == 2

    def test_clip_with_no_transcript_is_skipped(
        self, qapp_fixture, english_source, tmp_path
    ):
        from ui.workers.forced_alignment_worker import ForcedAlignmentWorker

        clip = _make_clip_with_transcript("c1", [])
        clip.transcript = None  # entirely missing transcript

        fake_wav = tmp_path / "fake.wav"
        fake_wav.write_bytes(b"RIFF")

        ready_calls: list[str] = []
        completions: list[bool] = []

        with _patch_feature_ready_and_audio_extract(fake_wav), patch(
            "core.analysis.alignment.align_words"
        ) as mock_align:
            worker = ForcedAlignmentWorker(
                clips=[clip],
                sources_by_id={english_source.id: english_source},
            )
            worker.clip_aligned.connect(
                lambda cid, words: ready_calls.append(cid)
            )
            worker.alignment_completed.connect(lambda: completions.append(True))
            worker.run()

        mock_align.assert_not_called()
        assert ready_calls == []
        assert completions == [True]


# ---------------------------------------------------------------------------
# Error path: align_words raises mid-run → error signal, no alignment_completed
# regression for already-emitted clips.
# ---------------------------------------------------------------------------


class TestForcedAlignmentWorkerErrorPath:
    def test_align_words_exception_emits_error_and_continues(
        self, qapp_fixture, english_source, tmp_path
    ):
        """One bad clip should not poison the whole batch."""
        from ui.workers.forced_alignment_worker import ForcedAlignmentWorker

        clips = [
            _make_clip_with_transcript(
                f"c{i}", [_make_segment(0.0, 2.0, f"text {i}")]
            )
            for i in range(3)
        ]

        fake_wav = tmp_path / "fake.wav"
        fake_wav.write_bytes(b"RIFF")

        call_count = {"n": 0}

        def fake_align_words(audio_path, segments):
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise RuntimeError("Simulated alignment failure")
            return [_make_word(0.0, 0.5, "word")]

        ready_calls: list[str] = []
        errors: list[str] = []
        completions: list[bool] = []

        with _patch_feature_ready_and_audio_extract(fake_wav), patch(
            "core.analysis.alignment.align_words",
            side_effect=fake_align_words,
        ):
            worker = ForcedAlignmentWorker(
                clips=clips,
                sources_by_id={english_source.id: english_source},
            )
            worker.clip_aligned.connect(
                lambda cid, words: ready_calls.append(cid)
            )
            worker.error.connect(lambda msg: errors.append(msg))
            worker.alignment_completed.connect(lambda: completions.append(True))
            worker.run()

        # First and third clips succeeded; second raised.
        assert ready_calls == ["c0", "c2"]
        # At least one error message surfaced and mentions the failure.
        assert len(errors) == 1
        assert "Simulated alignment failure" in errors[0]
        # Completion still fires so the analyze-tab pipeline can advance.
        assert completions == [True]

    def test_unsupported_language_raises_to_error_signal(
        self, qapp_fixture, english_source, tmp_path
    ):
        """UnsupportedLanguageError must surface as an error message."""
        from core.analysis.alignment import UnsupportedLanguageError
        from ui.workers.forced_alignment_worker import ForcedAlignmentWorker

        clip = _make_clip_with_transcript(
            "c1",
            [_make_segment(0.0, 2.0, "salut", language="xx")],
        )
        fake_wav = tmp_path / "fake.wav"
        fake_wav.write_bytes(b"RIFF")

        errors: list[str] = []
        completions: list[bool] = []

        def boom(audio_path, segments):
            raise UnsupportedLanguageError("xx")

        with _patch_feature_ready_and_audio_extract(fake_wav), patch(
            "core.analysis.alignment.align_words",
            side_effect=boom,
        ):
            worker = ForcedAlignmentWorker(
                clips=[clip],
                sources_by_id={english_source.id: english_source},
            )
            worker.error.connect(lambda msg: errors.append(msg))
            worker.alignment_completed.connect(lambda: completions.append(True))
            worker.run()

        assert len(errors) == 1
        assert "xx" in errors[0]
        assert completions == [True]


# ---------------------------------------------------------------------------
# Re-run idempotency: running the worker twice on the same set is a no-op the
# second time (the skip predicate sees populated word lists).
# ---------------------------------------------------------------------------


class TestForcedAlignmentWorkerIdempotentRerun:
    def test_second_run_after_handler_writeback_is_a_noop(
        self, qapp_fixture, english_source, tmp_path
    ):
        """Simulate the analyze-tab handler writing words back between runs."""
        from ui.workers.forced_alignment_worker import ForcedAlignmentWorker

        clip = _make_clip_with_transcript(
            "c1", [_make_segment(0.0, 2.0, "hello world")]
        )
        fake_wav = tmp_path / "fake.wav"
        fake_wav.write_bytes(b"RIFF")

        with _patch_feature_ready_and_audio_extract(fake_wav), patch(
            "core.analysis.alignment.align_words",
            return_value=[_make_word(0.0, 0.5, "hello")],
        ) as mock_align:
            # First run — alignment happens.
            worker = ForcedAlignmentWorker(
                clips=[clip],
                sources_by_id={english_source.id: english_source},
            )
            worker.run()
            assert mock_align.call_count == 1

            # Simulate the analyze-tab handler writing words back to every segment.
            for seg in clip.transcript:
                seg.words = [_make_word(0.0, 0.5, "hello")]

            # Second run on the same clip — should be a no-op because every
            # segment now has populated .words.
            worker2 = ForcedAlignmentWorker(
                clips=[clip],
                sources_by_id={english_source.id: english_source},
            )
            worker2.run()
            assert mock_align.call_count == 1  # no additional calls


# ---------------------------------------------------------------------------
# Integration: after a run, the words payload emitted via clip_aligned matches
# what align_words returned, and the worker delivered them in order.
# ---------------------------------------------------------------------------


class TestForcedAlignmentWorkerWordPayload:
    def test_emitted_words_match_align_words_return(
        self, qapp_fixture, english_source, tmp_path
    ):
        from ui.workers.forced_alignment_worker import ForcedAlignmentWorker

        clip = _make_clip_with_transcript(
            "c1",
            [
                _make_segment(0.0, 1.5, "hello world"),
                _make_segment(1.5, 3.0, "second segment"),
            ],
        )

        fake_wav = tmp_path / "fake.wav"
        fake_wav.write_bytes(b"RIFF")

        words_payload = [
            _make_word(0.0, 0.5, "hello"),
            _make_word(0.5, 1.0, "world"),
            _make_word(1.5, 2.0, "second"),
            _make_word(2.0, 2.8, "segment"),
        ]

        ready_calls: list[tuple[str, list]] = []

        with _patch_feature_ready_and_audio_extract(fake_wav), patch(
            "core.analysis.alignment.align_words",
            return_value=words_payload,
        ):
            worker = ForcedAlignmentWorker(
                clips=[clip],
                sources_by_id={english_source.id: english_source},
            )
            worker.clip_aligned.connect(
                lambda cid, words: ready_calls.append((cid, words))
            )
            worker.run()

        assert len(ready_calls) == 1
        emitted_clip_id, emitted_words = ready_calls[0]
        assert emitted_clip_id == "c1"
        assert [w.text for w in emitted_words] == [
            "hello",
            "world",
            "second",
            "segment",
        ]
