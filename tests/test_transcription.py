"""Regression tests for transcription edge cases."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.transcription import (
    FFmpegNotFoundError,
    TranscriptSegment,
    WordTimestamp,
    _parse_mlx_result,
    transcribe_clip,
    transcribe_video,
)


def test_transcribe_clip_skips_video_without_audio(monkeypatch):
    """Video-only clips should return no transcript without invoking FFmpeg."""
    monkeypatch.setattr("core.transcription._has_audio_stream", lambda _path: False)
    monkeypatch.setattr(
        "core.transcription.subprocess.run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("ffmpeg should not run")),
    )

    result = transcribe_clip(
        Path("/tmp/video_only.mp4"),
        start_time=0.0,
        end_time=5.0,
        backend="mlx-whisper",
    )

    assert result == []


def test_transcribe_video_skips_video_without_audio(monkeypatch):
    """Whole-video transcription should short-circuit when no audio stream exists."""
    progress = []

    monkeypatch.setattr("core.transcription._has_audio_stream", lambda _path: False)
    monkeypatch.setattr(
        "core.transcription._resolve_backend",
        lambda _backend: (_ for _ in ()).throw(AssertionError("backend resolution should not run")),
    )

    result = transcribe_video(
        Path("/tmp/video_only.mp4"),
        backend="auto",
        progress_callback=lambda pct, message: progress.append((pct, message)),
    )

    assert result == []
    assert progress == [(1.0, "No audio track found")]


def test_transcribe_clip_missing_ffmpeg_raises_clear_error(monkeypatch):
    """Missing FFmpeg should fail before subprocess tries the literal command."""
    monkeypatch.setattr("core.transcription._has_audio_stream", lambda _path: None)
    monkeypatch.setattr("core.transcription._resolve_backend", lambda _backend: "faster-whisper")
    monkeypatch.setattr("core.transcription.find_binary", lambda _name: None)
    monkeypatch.setattr(
        "core.transcription.subprocess.run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("subprocess should not run without ffmpeg")
        ),
    )

    with pytest.raises(FFmpegNotFoundError, match="FFmpeg is required for transcription"):
        transcribe_clip(
            Path("/tmp/video.mp4"),
            start_time=0.0,
            end_time=5.0,
            backend="faster-whisper",
        )


# ---------------------------------------------------------------------------
# U1: TranscriptSegment word-data schema + language capture
# ---------------------------------------------------------------------------


def test_word_timestamp_round_trip():
    """WordTimestamp round-trips through to_dict/from_dict with and without probability."""
    word = WordTimestamp(start=0.5, end=0.8, text="hello", probability=0.92)
    round_tripped = WordTimestamp.from_dict(word.to_dict())
    assert round_tripped == word

    bare = WordTimestamp(start=1.0, end=1.2, text="world")
    bare_rt = WordTimestamp.from_dict(bare.to_dict())
    assert bare_rt == bare
    assert bare_rt.probability is None


def test_transcript_segment_round_trip_with_words_and_language():
    """A segment with words and language round-trips identically."""
    seg = TranscriptSegment(
        start_time=0.0,
        end_time=1.5,
        text="hello world",
        confidence=-0.1,
        words=[
            WordTimestamp(start=0.0, end=0.4, text="hello", probability=0.95),
            WordTimestamp(start=0.5, end=1.5, text="world", probability=0.88),
        ],
        language="en",
    )
    round_tripped = TranscriptSegment.from_dict(seg.to_dict())
    assert round_tripped == seg


def test_transcript_segment_round_trip_old_format_back_compat():
    """An old-format dict (no words, no language) deserializes with words=None, language=None."""
    legacy_data = {
        "start_time": 0.0,
        "end_time": 1.0,
        "text": "legacy",
        "confidence": 0.0,
    }
    seg = TranscriptSegment.from_dict(legacy_data)
    assert seg.words is None
    assert seg.language is None
    # And the segment otherwise functions normally.
    assert seg.start_time == 0.0
    assert seg.end_time == 1.0
    assert seg.text == "legacy"


def test_transcript_segment_distinguishes_none_from_empty_words_list():
    """words=None and words=[] are distinct states and must both survive a round trip.

    None == "no word data surfaced yet" (MLX pre-alignment, or legacy projects).
    []   == "alignment ran and produced no words" (silence/instrumental).
    """
    none_seg = TranscriptSegment(
        start_time=0.0,
        end_time=1.0,
        text="silence",
        confidence=0.0,
        words=None,
        language="en",
    )
    empty_seg = TranscriptSegment(
        start_time=0.0,
        end_time=1.0,
        text="silence",
        confidence=0.0,
        words=[],
        language="en",
    )

    none_rt = TranscriptSegment.from_dict(none_seg.to_dict())
    empty_rt = TranscriptSegment.from_dict(empty_seg.to_dict())

    assert none_rt.words is None
    assert empty_rt.words == []
    assert none_rt != empty_rt


def test_transcribe_video_faster_whisper_captures_words_and_language(monkeypatch):
    """faster-whisper segments' .words and info.language must be captured onto every segment."""

    # Build mock faster-whisper segment with .words and info with .language
    word_a = MagicMock()
    word_a.start = 0.0
    word_a.end = 0.4
    word_a.word = "hello"
    word_a.probability = 0.95

    word_b = MagicMock()
    word_b.start = 0.5
    word_b.end = 1.5
    word_b.word = "world"
    word_b.probability = 0.88

    fake_segment = MagicMock()
    fake_segment.start = 0.0
    fake_segment.end = 1.5
    fake_segment.text = "hello world"
    fake_segment.avg_logprob = -0.1
    fake_segment.words = [word_a, word_b]

    fake_info = MagicMock()
    fake_info.language = "en"

    fake_model = MagicMock()
    fake_model.transcribe.return_value = (iter([fake_segment]), fake_info)

    monkeypatch.setattr("core.transcription._has_audio_stream", lambda _p: True)
    monkeypatch.setattr("core.transcription.get_model", lambda _name: fake_model)

    result = transcribe_video(
        Path("/tmp/x.mp4"),
        backend="faster-whisper",
    )

    assert len(result) == 1
    seg = result[0]
    assert seg.language == "en"
    assert seg.words is not None
    assert len(seg.words) == 2
    assert seg.words[0].text == "hello"
    assert seg.words[0].start == 0.0
    assert seg.words[0].end == 0.4
    assert seg.words[0].probability == pytest.approx(0.95)
    assert seg.words[1].text == "world"


def test_transcribe_video_faster_whisper_handles_segment_without_words(monkeypatch):
    """If a faster-whisper segment surfaces no .words (e.g., disabled timestamps), words=None."""
    fake_segment = MagicMock()
    fake_segment.start = 0.0
    fake_segment.end = 1.0
    fake_segment.text = "silent"
    fake_segment.avg_logprob = -0.5
    fake_segment.words = None  # Defensive: word_timestamps may not have produced data

    fake_info = MagicMock()
    fake_info.language = "fr"

    fake_model = MagicMock()
    fake_model.transcribe.return_value = (iter([fake_segment]), fake_info)

    monkeypatch.setattr("core.transcription._has_audio_stream", lambda _p: True)
    monkeypatch.setattr("core.transcription.get_model", lambda _name: fake_model)

    result = transcribe_video(Path("/tmp/x.mp4"), backend="faster-whisper")

    assert len(result) == 1
    assert result[0].words is None
    assert result[0].language == "fr"


def test_parse_mlx_result_captures_language():
    """MLX `result["language"]` must populate TranscriptSegment.language; words stays None.

    U2 will fill words via forced alignment; U1's job is only to surface the language signal
    MLX was previously discarding.
    """
    mlx_result = {
        "text": "bonjour le monde",
        "language": "fr",
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "bonjour"},
            [100, 200, "le monde"],  # frames format
        ],
    }
    segments = _parse_mlx_result(mlx_result)
    assert len(segments) == 2
    assert all(s.language == "fr" for s in segments)
    assert all(s.words is None for s in segments)


def test_parse_mlx_result_language_none_when_absent():
    """If MLX result omits 'language' (unusual), segments have language=None."""
    mlx_result = {
        "text": "hello",
        "segments": [{"start": 0.0, "end": 0.5, "text": "hello"}],
    }
    segments = _parse_mlx_result(mlx_result)
    assert len(segments) == 1
    assert segments[0].language is None
    assert segments[0].words is None


def test_clip_with_no_transcript_round_trips():
    """A Clip with transcript=None is unaffected by the schema change."""
    from models.clip import Clip

    clip = Clip(
        id="c1",
        source_id="s1",
        start_frame=0,
        end_frame=24,
        name="clip-1",
    )
    data = clip.to_dict()
    restored = Clip.from_dict(data)
    assert restored.transcript is None


def test_clip_transcript_round_trips_with_new_fields():
    """An existing Clip.from_dict path forwards new TranscriptSegment fields cleanly."""
    from models.clip import Clip

    clip = Clip(
        id="c1",
        source_id="s1",
        start_frame=0,
        end_frame=24,
        name="clip-1",
        transcript=[
            TranscriptSegment(
                start_time=0.0,
                end_time=1.0,
                text="hello",
                confidence=-0.1,
                words=[WordTimestamp(start=0.0, end=0.5, text="hello", probability=0.9)],
                language="en",
            )
        ],
    )
    data = clip.to_dict()
    restored = Clip.from_dict(data)
    assert restored.transcript is not None
    assert len(restored.transcript) == 1
    assert restored.transcript[0].language == "en"
    assert restored.transcript[0].words is not None
    assert restored.transcript[0].words[0].text == "hello"


def test_clip_transcript_round_trips_legacy_format():
    """A legacy clip dict whose transcript entries lack 'words' and 'language' still loads."""
    from models.clip import Clip

    legacy_data = {
        "id": "c1",
        "source_id": "s1",
        "start_frame": 0,
        "end_frame": 24,
        "name": "clip-1",
        "transcript": [
            {
                "start_time": 0.0,
                "end_time": 1.0,
                "text": "legacy",
                "confidence": 0.0,
            }
        ],
    }
    restored = Clip.from_dict(legacy_data)
    assert restored.transcript is not None
    assert len(restored.transcript) == 1
    assert restored.transcript[0].words is None
    assert restored.transcript[0].language is None
