"""Word readiness verifies alignment or native transcription provenance."""

from dataclasses import replace
from unittest.mock import Mock

import pytest

from core.analysis_availability import alignment_is_complete, word_timing_is_complete
from tests.test_alignment_records import setup as setup, execute


@pytest.fixture
def aligned(request):
    project, _ = request.getfixturevalue("setup")
    execute(project)
    return project


def test_completion_does_not_hash_probe_or_infer(aligned, monkeypatch):
    blocked = Mock(side_effect=AssertionError("completion must remain cheap"))
    monkeypatch.setattr("core.analysis_records.AnalysisFingerprints.identity", blocked)
    monkeypatch.setattr("core.analysis.alignment.align_words", blocked)
    monkeypatch.setattr("core.analysis.alignment.extract_audio_to_wav", blocked)
    assert alignment_is_complete(aligned.clips[0], aligned.sources[0])
    assert word_timing_is_complete(aligned.clips[0], aligned.sources[0])
    blocked.assert_not_called()


@pytest.mark.parametrize("change", ["legacy", "failed", "text", "words", "language", "fps", "media", "path", "revision", "source", "source_id"])
def test_stale_alignment_is_pending(aligned, monkeypatch, tmp_path, change):
    clip, source = aligned.clips[0], aligned.sources[0]
    if change == "legacy":
        clip.analysis_records.clear()
    elif change == "failed":
        clip.analysis_records["align_words"] = replace(clip.analysis_records["align_words"], state="failed")
    elif change == "text":
        clip.transcript[0].text = "edited"
    elif change == "words":
        clip.transcript[0].words = []
    elif change == "language":
        clip.transcript[0].language = "es"
    elif change == "fps":
        source.fps = 24
    elif change == "media":
        source.file_path.write_bytes(b"changed")
    elif change == "path":
        source.file_path = tmp_path / "different.mp4"
    elif change == "revision":
        monkeypatch.setattr("core.operations.alignment_records.alignment_model_revision", lambda: "r2")
    elif change == "source_id":
        source = replace(source, id="other-source")
    else:
        source = None
    assert not alignment_is_complete(clip, source)
    assert not word_timing_is_complete(clip, source)


@pytest.mark.parametrize("empty", [False, True])
def test_native_transcription_words_and_silence_are_ready_without_alignment(aligned, monkeypatch, empty):
    from core.operations.transcription import TranscriptionApplication, TranscriptionOptions, run_transcription
    from core.operations.transcription_records import transcription_task
    from core.settings import Settings
    from ui.dialogs._word_source_picker import classify_source_alignment, BADGE_ALIGNED

    clip, source = aligned.clips[0], aligned.sources[0]
    clip.analysis_records.clear()
    segments = [] if empty else clip.transcript
    monkeypatch.setattr("core.settings.load_settings", lambda: Settings(transcription_backend="faster-whisper", transcription_model="small.en", transcription_language="en"))
    monkeypatch.setattr("core.transcription._has_audio_stream", lambda _: True)
    monkeypatch.setattr("core.transcription.transcribe_clip", Mock(return_value=segments))
    task = transcription_task(clip, source, skip_existing=False)
    options = TranscriptionOptions(backend="faster-whisper")
    application = TranscriptionApplication(aligned, (task,), options)
    assert application.apply(aligned, run_transcription((task,), options)[0])
    monkeypatch.setattr("ui.dialogs._word_source_picker._language_is_supported", Mock(side_effect=AssertionError("native words need no aligner")))
    assert not alignment_is_complete(clip, source)
    assert word_timing_is_complete(clip, source)
    assert classify_source_alignment([(clip, source)])[0] == BADGE_ALIGNED
    assert not word_timing_is_complete(clip, replace(source, id="other-source"))
    clip.analysis_records.clear()
    assert not word_timing_is_complete(clip, source)


def test_picker_reverifies_legacy_words_and_does_not_retry_failures(aligned):
    from ui.dialogs._word_source_picker import classify_source_alignment, alignable_pending_clips, partition_clips_for_sequencing, BADGE_NEEDS_ALIGNMENT

    clip, source = aligned.clips[0], aligned.sources[0]
    clip.analysis_records.clear()
    pairs = [(clip, source)]
    assert classify_source_alignment(pairs)[0] == BADGE_NEEDS_ALIGNMENT
    assert alignable_pending_clips(pairs) == [clip]
    assert partition_clips_for_sequencing(pairs, set()) == ([], [clip], [])
    assert partition_clips_for_sequencing(pairs, {clip.id}) == ([], [], [clip.id])
