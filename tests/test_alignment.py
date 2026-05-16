"""Tests for the forced-alignment runtime module (U2).

These tests mock the heavy alignment engine (``_run_alignment_engine``) and the
FFmpeg audio-extraction step where appropriate, so they exercise the module's
control flow — language gating, empty-input fast paths, temp-file cleanup,
error propagation — without requiring ``ctc-forced-aligner`` or ``torch`` to
be installed.
"""

from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from core.transcription import TranscriptSegment, WordTimestamp


# ---------------------------------------------------------------------------
# Module-import boundary
# ---------------------------------------------------------------------------


class TestModuleImportBoundary(unittest.TestCase):
    """``core.analysis.alignment`` must import on a clean env."""

    def test_imports_without_ctc_forced_aligner(self):
        # Ensure the module loads cleanly even if the heavy deps are absent
        # from this Python environment. The pure-import path should not pull
        # ``ctc_forced_aligner`` or ``torch`` into sys.modules.
        sys.modules.pop("core.analysis.alignment", None)

        import core.analysis.alignment  # noqa: F401

        self.assertNotIn("ctc_forced_aligner", sys.modules)


# ---------------------------------------------------------------------------
# Public API: align_words
# ---------------------------------------------------------------------------


class TestAlignWordsLanguageGating(unittest.TestCase):
    """Language signal handling on the public entry point."""

    def test_empty_segments_returns_empty_no_engine_load(self):
        """Empty segment list short-circuits before model load + ffmpeg."""
        from core.analysis import alignment

        with patch.object(alignment, "extract_audio_to_wav") as ffmpeg_mock, \
             patch.object(alignment, "_run_alignment_engine") as engine_mock:
            result = alignment.align_words("/path/to/audio.wav", [])

        self.assertEqual(result, [])
        ffmpeg_mock.assert_not_called()
        engine_mock.assert_not_called()

    def test_segments_with_language_none_raises_language_unknown(self):
        """Legacy transcripts (pre-U1) lack a language → explicit error."""
        from core.analysis.alignment import LanguageUnknownError, align_words

        seg = TranscriptSegment(
            start_time=0.0, end_time=1.0, text="hello world",
            confidence=0.9, words=None, language=None,
        )
        with self.assertRaises(LanguageUnknownError):
            align_words("/path/to/audio.wav", [seg])

    def test_unsupported_language_raises_unsupported_language_error(self):
        """An ISO code outside the supported set → UnsupportedLanguageError."""
        from core.analysis.alignment import UnsupportedLanguageError, align_words

        seg = TranscriptSegment(
            start_time=0.0, end_time=1.0, text="hello",
            confidence=0.9, words=None, language="xx",
        )
        with patch("importlib.import_module", side_effect=ModuleNotFoundError), \
             self.assertRaises(UnsupportedLanguageError) as ctx:
            align_words("/path/to/audio.wav", [seg])
        self.assertEqual(ctx.exception.language, "xx")

    def test_runtime_without_supported_languages_skips_static_gate(self):
        """Installed runtime with no exported language list lets model raise later."""
        from core.analysis import alignment

        def _fake_import_module(name):
            if name == "ctc_forced_aligner":
                return types.SimpleNamespace()
            if name == "ctc_forced_aligner.alignment_utils":
                return types.SimpleNamespace()
            raise AssertionError(f"unexpected import: {name}")

        with patch("importlib.import_module", side_effect=_fake_import_module):
            alignment._check_language_supported("xx")
            self.assertTrue(alignment.is_language_supported("xx"))

    def test_runtime_supported_languages_still_rejects_unknown_code(self):
        """If the runtime exposes a language list, keep the explicit error."""
        from core.analysis import alignment

        def _fake_import_module(name):
            if name == "ctc_forced_aligner":
                return types.SimpleNamespace()
            if name == "ctc_forced_aligner.alignment_utils":
                return types.SimpleNamespace(SUPPORTED_LANGUAGES={"en": "eng"})
            raise AssertionError(f"unexpected import: {name}")

        with patch("importlib.import_module", side_effect=_fake_import_module), \
             self.assertRaises(alignment.UnsupportedLanguageError):
            alignment._check_language_supported("xx")

        with patch("importlib.import_module", side_effect=_fake_import_module):
            self.assertFalse(alignment.is_language_supported("xx"))

    def test_language_is_read_from_first_segment(self):
        """Language passes through to the alignment engine call."""
        from core.analysis import alignment

        seg = TranscriptSegment(
            start_time=0.0, end_time=1.0, text="hello world",
            confidence=0.9, words=None, language="en",
        )
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            audio = Path(tmp.name)
        try:
            with patch.object(alignment, "extract_audio_to_wav") as ffmpeg_mock, \
                 patch.object(alignment, "_run_alignment_engine") as engine_mock:
                fake_wav = audio  # reuse the existing file for cleanup symmetry
                ffmpeg_mock.return_value = fake_wav
                engine_mock.return_value = [
                    {"start": 0.0, "end": 0.4, "text": "hello", "score": 0.95},
                    {"start": 0.4, "end": 0.8, "text": "world", "score": 0.92},
                ]
                alignment.align_words(str(audio), [seg])

            # Engine receives the language from segments[0].language.
            kwargs = engine_mock.call_args.kwargs
            self.assertEqual(kwargs["language"], "en")
        finally:
            audio.unlink(missing_ok=True)


class TestAlignWordsHappyPath(unittest.TestCase):
    """Successful alignment runs with mocked engine."""

    def setUp(self):
        # Create a real on-disk audio path so the .exists() check passes.
        self.tmp_audio = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        self.tmp_audio.close()
        self.audio_path = Path(self.tmp_audio.name)

        # Create a fake "extracted WAV" — also a real file so .stat()/.unlink work.
        wav_handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wav_handle.write(b"fake-wav-bytes")
        wav_handle.close()
        self.fake_wav = Path(wav_handle.name)

    def tearDown(self):
        self.audio_path.unlink(missing_ok=True)
        self.fake_wav.unlink(missing_ok=True)

    def _run_with_engine_output(self, segments, engine_output):
        """Helper: run align_words with mocked FFmpeg + engine."""
        from core.analysis import alignment

        with patch.object(alignment, "extract_audio_to_wav", return_value=self.fake_wav) as ffmpeg_mock, \
             patch.object(alignment, "_run_alignment_engine", return_value=engine_output) as engine_mock:
            words = alignment.align_words(str(self.audio_path), segments)
        return words, ffmpeg_mock, engine_mock

    def test_returns_word_timestamps_for_aligned_transcript(self):
        """Engine output maps cleanly to WordTimestamp objects."""
        segments = [
            TranscriptSegment(
                start_time=0.0, end_time=1.0, text="hello world",
                confidence=0.9, words=None, language="en",
            ),
        ]
        engine_output = [
            {"start": 0.05, "end": 0.40, "text": "hello", "score": 0.95},
            {"start": 0.45, "end": 0.95, "text": "world", "score": 0.92},
        ]
        words, ffmpeg_mock, engine_mock = self._run_with_engine_output(
            segments, engine_output,
        )

        self.assertEqual(len(words), 2)
        self.assertIsInstance(words[0], WordTimestamp)
        self.assertEqual(words[0].text, "hello")
        self.assertAlmostEqual(words[0].start, 0.05)
        self.assertAlmostEqual(words[0].end, 0.40)
        self.assertAlmostEqual(words[0].probability, 0.95)
        self.assertEqual(words[1].text, "world")
        ffmpeg_mock.assert_called_once_with(self.audio_path)
        engine_mock.assert_called_once()

    def test_word_count_matches_engine_output(self):
        """The output count tracks the engine — not the transcript."""
        segments = [
            TranscriptSegment(
                start_time=0.0, end_time=2.0, text="one two three four five",
                confidence=0.9, words=None, language="en",
            ),
        ]
        engine_output = [
            {"start": float(i) * 0.4, "end": float(i + 1) * 0.4, "text": w, "score": 0.9}
            for i, w in enumerate(["one", "two", "three", "four", "five"])
        ]
        words, _, _ = self._run_with_engine_output(segments, engine_output)
        self.assertEqual(len(words), 5)

    def test_timestamps_within_clip_audio_bounds(self):
        """Word boundaries returned are >= 0 and ordered."""
        segments = [
            TranscriptSegment(
                start_time=0.0, end_time=3.0, text="a b c",
                confidence=0.9, words=None, language="en",
            ),
        ]
        engine_output = [
            {"start": 0.0, "end": 1.0, "text": "a", "score": 0.9},
            {"start": 1.0, "end": 2.0, "text": "b", "score": 0.9},
            {"start": 2.0, "end": 3.0, "text": "c", "score": 0.9},
        ]
        words, _, _ = self._run_with_engine_output(segments, engine_output)
        self.assertEqual(words[0].start, 0.0)
        self.assertGreaterEqual(words[0].start, 0.0)
        for prev, curr in zip(words, words[1:]):
            self.assertLessEqual(prev.start, curr.start)

    def test_score_none_translates_to_probability_none(self):
        """Missing score in engine output → probability is None on the WordTimestamp."""
        segments = [
            TranscriptSegment(
                start_time=0.0, end_time=1.0, text="hello",
                confidence=0.9, words=None, language="en",
            ),
        ]
        engine_output = [
            {"start": 0.0, "end": 0.5, "text": "hello", "score": None},
        ]
        words, _, _ = self._run_with_engine_output(segments, engine_output)
        self.assertIsNone(words[0].probability)

    def test_short_clip_audio_does_not_crash(self):
        """A very short clip (engine returns one word) returns one WordTimestamp."""
        segments = [
            TranscriptSegment(
                start_time=0.0, end_time=0.3, text="hi",
                confidence=0.9, words=None, language="en",
            ),
        ]
        engine_output = [
            {"start": 0.0, "end": 0.2, "text": "hi", "score": 0.88},
        ]
        words, _, _ = self._run_with_engine_output(segments, engine_output)
        self.assertEqual(len(words), 1)
        self.assertEqual(words[0].text, "hi")

    def test_all_empty_text_segments_returns_empty(self):
        """Segments with only whitespace text → empty result, no engine call."""
        from core.analysis import alignment

        segments = [
            TranscriptSegment(
                start_time=0.0, end_time=1.0, text="   ",
                confidence=0.9, words=None, language="en",
            ),
        ]
        with patch.object(alignment, "extract_audio_to_wav") as ffmpeg_mock, \
             patch.object(alignment, "_run_alignment_engine") as engine_mock:
            result = alignment.align_words(str(self.audio_path), segments)

        self.assertEqual(result, [])
        ffmpeg_mock.assert_not_called()
        engine_mock.assert_not_called()

    def test_ctc_target_too_long_retries_by_segment(self):
        """Dense whole-clip text falls back to per-segment alignment."""
        from core.analysis import alignment

        segments = [
            TranscriptSegment(
                start_time=0.0, end_time=1.0, text="hello",
                confidence=0.9, words=None, language="en",
            ),
            TranscriptSegment(
                start_time=1.0, end_time=2.0, text="world",
                confidence=0.9, words=None, language="en",
            ),
        ]
        wavs = []
        for _ in range(3):
            handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            handle.write(b"fake-wav")
            handle.close()
            wavs.append(Path(handle.name))

        engine_outputs = [
            RuntimeError("targets length is too long for CTC."),
            [{"start": 0.1, "end": 0.4, "text": "hello", "score": 0.9}],
            [{"start": 0.2, "end": 0.6, "text": "world", "score": 0.8}],
        ]

        try:
            with patch.object(alignment, "extract_audio_to_wav", side_effect=wavs) as ffmpeg_mock, \
                 patch.object(alignment, "_run_alignment_engine", side_effect=engine_outputs) as engine_mock:
                words = alignment.align_words(str(self.audio_path), segments)
        finally:
            for wav in wavs:
                wav.unlink(missing_ok=True)

        self.assertEqual([w.text for w in words], ["hello", "world"])
        self.assertAlmostEqual(words[0].start, 0.1)
        self.assertAlmostEqual(words[1].start, 1.2)
        self.assertEqual(ffmpeg_mock.call_count, 3)
        ffmpeg_mock.assert_any_call(self.audio_path)
        ffmpeg_mock.assert_any_call(self.audio_path, start_time=0.0, end_time=1.0)
        ffmpeg_mock.assert_any_call(self.audio_path, start_time=1.0, end_time=2.0)
        self.assertEqual(engine_mock.call_count, 3)

    def test_ctc_target_too_long_segment_gets_approximate_words(self):
        """A still-too-dense segment gets approximate word timings."""
        from core.analysis import alignment

        segments = [
            TranscriptSegment(
                start_time=0.0, end_time=1.0, text="dense dense dense",
                confidence=0.9, words=None, language="en",
            ),
            TranscriptSegment(
                start_time=1.0, end_time=2.0, text="ok",
                confidence=0.9, words=None, language="en",
            ),
        ]
        wavs = []
        for _ in range(3):
            handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            handle.write(b"fake-wav")
            handle.close()
            wavs.append(Path(handle.name))

        engine_outputs = [
            RuntimeError("targets length is too long for CTC."),
            RuntimeError("targets length is too long for CTC."),
            [{"start": 0.0, "end": 0.3, "text": "ok", "score": 0.8}],
        ]

        try:
            with patch.object(alignment, "extract_audio_to_wav", side_effect=wavs), \
                 patch.object(alignment, "_run_alignment_engine", side_effect=engine_outputs):
                words = alignment.align_words(str(self.audio_path), segments)
        finally:
            for wav in wavs:
                wav.unlink(missing_ok=True)

        self.assertEqual([w.text for w in words], ["dense", "dense", "dense", "ok"])
        self.assertAlmostEqual(words[0].start, 0.0)
        self.assertAlmostEqual(words[0].end, 1.0 / 3.0)
        self.assertIsNone(words[0].probability)
        self.assertAlmostEqual(words[-1].start, 1.0)


class TestAlignWordsCleanup(unittest.TestCase):
    """Temp-file lifecycle around the engine call."""

    def test_temp_wav_is_removed_after_successful_run(self):
        """Happy path: the temp WAV path returned by extract_audio_to_wav is unlinked."""
        from core.analysis import alignment

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as src:
            src_path = Path(src.name)
        wav_handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wav_handle.write(b"data")
        wav_handle.close()
        wav_path = Path(wav_handle.name)

        try:
            seg = TranscriptSegment(
                start_time=0.0, end_time=1.0, text="hello",
                confidence=0.9, words=None, language="en",
            )
            with patch.object(alignment, "extract_audio_to_wav", return_value=wav_path), \
                 patch.object(alignment, "_run_alignment_engine", return_value=[
                     {"start": 0.0, "end": 0.5, "text": "hello", "score": 0.9},
                 ]):
                alignment.align_words(str(src_path), [seg])

            self.assertFalse(
                wav_path.exists(),
                "Temp WAV should be removed after successful alignment",
            )
        finally:
            src_path.unlink(missing_ok=True)
            wav_path.unlink(missing_ok=True)

    def test_temp_wav_is_removed_when_engine_raises(self):
        """Engine failure must still trigger temp-WAV cleanup."""
        from core.analysis import alignment

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as src:
            src_path = Path(src.name)
        wav_handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wav_handle.write(b"data")
        wav_handle.close()
        wav_path = Path(wav_handle.name)

        try:
            seg = TranscriptSegment(
                start_time=0.0, end_time=1.0, text="hello",
                confidence=0.9, words=None, language="en",
            )

            def _boom(**kwargs):
                raise alignment.AlignmentError("engine exploded")

            with patch.object(alignment, "extract_audio_to_wav", return_value=wav_path), \
                 patch.object(alignment, "_run_alignment_engine", side_effect=_boom):
                with self.assertRaises(alignment.AlignmentError):
                    alignment.align_words(str(src_path), [seg])

            self.assertFalse(
                wav_path.exists(),
                "Temp WAV must still be removed when alignment raises",
            )
        finally:
            src_path.unlink(missing_ok=True)
            wav_path.unlink(missing_ok=True)


class TestAlignWordsErrors(unittest.TestCase):
    """Error-path behavior."""

    def test_missing_audio_path_raises_file_not_found(self):
        from core.analysis.alignment import align_words

        seg = TranscriptSegment(
            start_time=0.0, end_time=1.0, text="hello",
            confidence=0.9, words=None, language="en",
        )
        with self.assertRaises(FileNotFoundError):
            align_words("/definitely/does/not/exist-12345.wav", [seg])

    def test_install_for_feature_path_triggered_on_missing_engine(self):
        """When the engine import fails, AlignmentError is raised — the worker
        layer (U3) is responsible for invoking install_for_feature.

        This test simulates the missing-dep path by mocking
        ``_run_alignment_engine`` to raise the same kind of error the real
        function raises when ``ctc_forced_aligner`` / ``torch`` are absent.
        """
        from core.analysis import alignment

        seg = TranscriptSegment(
            start_time=0.0, end_time=1.0, text="hello",
            confidence=0.9, words=None, language="en",
        )

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as src:
            src_path = Path(src.name)
        wav_handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wav_handle.write(b"data")
        wav_handle.close()
        wav_path = Path(wav_handle.name)

        try:
            def _fake_engine(**kwargs):
                raise alignment.AlignmentError(
                    "ctc-forced-aligner is not installed. "
                    "Install the 'word_alignment' feature first."
                )

            with patch.object(alignment, "extract_audio_to_wav", return_value=wav_path), \
                 patch.object(alignment, "_run_alignment_engine", side_effect=_fake_engine):
                with self.assertRaises(alignment.AlignmentError) as ctx:
                    alignment.align_words(str(src_path), [seg])
            self.assertIn("word_alignment", str(ctx.exception))
        finally:
            src_path.unlink(missing_ok=True)
            wav_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# FFmpeg integration
# ---------------------------------------------------------------------------


class TestExtractAudioCleanup(unittest.TestCase):
    """``extract_audio_to_wav`` owns its temp file on the unhappy paths."""

    def test_extract_failure_cleans_up_temp_wav(self):
        from core.analysis import alignment
        import subprocess as _subprocess

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as src:
            src_path = Path(src.name)

        try:
            captured: dict[str, Path] = {}

            real_named_tmp = tempfile.NamedTemporaryFile

            def _capturing_tmp(*args, **kwargs):
                handle = real_named_tmp(*args, **kwargs)
                captured["path"] = Path(handle.name)
                return handle

            def _boom(*args, **kwargs):
                # Simulate ffmpeg failing — propagates AlignmentFFmpegError.
                exc = _subprocess.CalledProcessError(1, args[0])
                exc.stderr = b"simulated ffmpeg failure"
                raise exc

            with patch.object(alignment, "_require_ffmpeg", return_value="/usr/bin/ffmpeg"), \
                 patch.object(alignment.tempfile, "NamedTemporaryFile", side_effect=_capturing_tmp), \
                 patch.object(alignment.subprocess, "run", side_effect=_boom):
                with self.assertRaises(alignment.AlignmentFFmpegError):
                    alignment.extract_audio_to_wav(src_path)

            self.assertIn("path", captured)
            self.assertFalse(
                captured["path"].exists(),
                "Failed extraction must clean up its temp WAV",
            )
        finally:
            src_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Feature-registry integration
# ---------------------------------------------------------------------------


class TestFeatureRegistryEntry(unittest.TestCase):
    """The new ``word_alignment`` feature is wired into the registry."""

    def test_word_alignment_present_in_feature_deps(self):
        from core.feature_registry import FEATURE_DEPS
        self.assertIn("word_alignment", FEATURE_DEPS)

    def test_word_alignment_requires_ffmpeg_and_aligner(self):
        from core.feature_registry import FEATURE_DEPS
        deps = FEATURE_DEPS["word_alignment"]
        self.assertIn("ffmpeg", deps.binaries)
        self.assertIn("ctc_forced_aligner", deps.packages)
        self.assertIn("torch", deps.packages)

    def test_word_alignment_uses_native_install(self):
        from core.feature_registry import FEATURE_DEPS
        self.assertTrue(FEATURE_DEPS["word_alignment"].native_install)

    def test_word_alignment_has_size_estimate(self):
        from core.feature_registry import FEATURE_DEPS
        self.assertGreater(FEATURE_DEPS["word_alignment"].size_estimate_mb, 0)

    def test_runtime_validation_branch_calls_ensure(self):
        """The _validate_feature_runtime hook delegates to the alignment module."""
        from core import feature_registry
        from core.analysis import alignment as alignment_mod

        with patch.object(alignment_mod, "ensure_word_alignment_runtime_available") as ensure_mock:
            feature_registry._validate_feature_runtime("word_alignment")
        ensure_mock.assert_called_once()


# ---------------------------------------------------------------------------
# distribute_words_to_segments
# ---------------------------------------------------------------------------


class TestDistributeWordsToSegments(unittest.TestCase):
    """The flat-word-list-to-segments distribution helper."""

    @staticmethod
    def _seg(start: float, end: float) -> TranscriptSegment:
        return TranscriptSegment(
            start_time=start,
            end_time=end,
            text="x",
            confidence=0.9,
            words=None,
            language="en",
        )

    def test_word_midpoint_inside_segment_assigns_there(self):
        from core.analysis.alignment import distribute_words_to_segments

        segments = [self._seg(0.0, 1.0), self._seg(1.0, 2.0), self._seg(2.0, 3.0)]
        words = [
            WordTimestamp(start=0.10, end=0.30, text="a"),   # mid 0.20 → seg0
            WordTimestamp(start=1.20, end=1.80, text="b"),   # mid 1.50 → seg1
            WordTimestamp(start=2.30, end=2.70, text="c"),   # mid 2.50 → seg2
        ]
        distribute_words_to_segments(segments, words)
        self.assertEqual([w.text for w in segments[0].words], ["a"])
        self.assertEqual([w.text for w in segments[1].words], ["b"])
        self.assertEqual([w.text for w in segments[2].words], ["c"])

    def test_word_outside_all_segments_falls_back_to_nearest(self):
        from core.analysis.alignment import distribute_words_to_segments

        segments = [self._seg(0.0, 1.0), self._seg(2.0, 3.0)]
        words = [
            # midpoint 1.5 → equidistant. Nearest-boundary lambda compares
            # min(|1.5-start|, |1.5-end|): seg0 → min(1.5, 0.5)=0.5;
            # seg1 → min(0.5, 1.5)=0.5. Tie → ``min`` picks the first (seg0).
            WordTimestamp(start=1.4, end=1.6, text="left"),
            # midpoint 1.8 → seg1 wins (min(1.8, 0.8)=0.8 vs seg0 min(1.8, 0.8)=0.8;
            # tie again, picks seg0). Use a midpoint that breaks the tie:
            WordTimestamp(start=1.7, end=1.9, text="right"),  # mid 1.8 — actually still ties
        ]
        # Replace with a clearer case: 1.9 → seg1 closer.
        words = [
            WordTimestamp(start=1.85, end=1.95, text="right"),  # mid 1.9 → seg1
            WordTimestamp(start=1.05, end=1.15, text="left"),   # mid 1.10 → seg0
        ]
        distribute_words_to_segments(segments, words)
        self.assertEqual([w.text for w in segments[0].words], ["left"])
        self.assertEqual([w.text for w in segments[1].words], ["right"])

    def test_empty_words_assigns_empty_list_to_every_segment(self):
        from core.analysis.alignment import distribute_words_to_segments

        segments = [self._seg(0.0, 1.0), self._seg(1.0, 2.0)]
        distribute_words_to_segments(segments, [])
        # The contract: every segment gets ``.words = []`` (not None) so the
        # skip predicate counts the clip as aligned.
        for seg in segments:
            self.assertEqual(seg.words, [])

    def test_empty_segments_is_a_noop(self):
        from core.analysis.alignment import distribute_words_to_segments

        # Calling with no segments must not raise — and there's nothing to
        # mutate.
        distribute_words_to_segments([], [
            WordTimestamp(start=0.0, end=0.5, text="orphan"),
        ])


if __name__ == "__main__":
    unittest.main()
