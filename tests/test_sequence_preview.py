"""Tests for cached continuous sequence previews."""

from core.sequence_preview import (
    SequencePreviewSettings,
    compute_sequence_preview_signature,
    get_sequence_preview_path,
    render_sequence_preview,
)
from models.clip import Clip, Source
from models.sequence import Sequence, SequenceClip


def _make_sequence_with_clips(tmp_path):
    source_path = tmp_path / "source.mp4"
    source_path.write_bytes(b"source")
    source = Source(
        id="src-1",
        file_path=source_path,
        fps=24.0,
        width=1920,
        height=1080,
    )
    clip_a = Clip(id="clip-a", source_id=source.id, start_frame=0, end_frame=24)
    clip_b = Clip(id="clip-b", source_id=source.id, start_frame=24, end_frame=48)
    sequence = Sequence(id="seq-1", fps=24.0)
    sequence.tracks[0].clips = [
        SequenceClip(
            id="seq-a",
            source_clip_id=clip_a.id,
            source_id=source.id,
            start_frame=0,
            in_point=clip_a.start_frame,
            out_point=clip_a.end_frame,
        ),
        SequenceClip(
            id="seq-b",
            source_clip_id=clip_b.id,
            source_id=source.id,
            start_frame=24,
            in_point=clip_b.start_frame,
            out_point=clip_b.end_frame,
        ),
    ]
    return sequence, {source.id: source}, {
        clip_a.id: (clip_a, source),
        clip_b.id: (clip_b, source),
    }


def test_preview_signature_changes_when_clip_order_changes(tmp_path):
    sequence, sources, clips = _make_sequence_with_clips(tmp_path)
    first = compute_sequence_preview_signature(sequence, sources, clips)

    sequence.tracks[0].clips[0].start_frame = 24
    sequence.tracks[0].clips[1].start_frame = 0
    second = compute_sequence_preview_signature(sequence, sources, clips)

    assert first != second


def test_preview_signature_changes_when_music_changes(tmp_path):
    sequence, sources, clips = _make_sequence_with_clips(tmp_path)
    first = compute_sequence_preview_signature(sequence, sources, clips)

    music = tmp_path / "music.wav"
    music.write_bytes(b"music")
    sequence.music_path = str(music)
    second = compute_sequence_preview_signature(sequence, sources, clips)

    assert first != second


def test_render_sequence_preview_uses_proxy_export_config(monkeypatch, tmp_path):
    sequence, sources, clips = _make_sequence_with_clips(tmp_path)
    music = tmp_path / "music.wav"
    music.write_bytes(b"music")
    sequence.music_path = str(music)
    captured = {}

    class FakeExporter:
        def export(self, *, sequence, sources, clips, config, progress_callback, frames):
            captured["config"] = config
            config.output_path.write_bytes(b"preview")
            return True

    monkeypatch.setattr("core.sequence_preview.SequenceExporter", FakeExporter)

    result = render_sequence_preview(
        sequence=sequence,
        sources=sources,
        clips=clips,
        cache_root=tmp_path / "cache",
        settings=SequencePreviewSettings(width=854, height=480, crf=22),
    )

    config = captured["config"]
    assert result.path.exists()
    assert result.from_cache is False
    assert config.width == 854
    assert config.height == 480
    assert config.crf == 22
    assert config.music_path == music


def test_render_sequence_preview_returns_cache_hit(monkeypatch, tmp_path):
    sequence, sources, clips = _make_sequence_with_clips(tmp_path)
    signature = compute_sequence_preview_signature(sequence, sources, clips)
    cached_path = get_sequence_preview_path(sequence, signature, tmp_path / "cache")
    cached_path.parent.mkdir(parents=True)
    cached_path.write_bytes(b"cached")
    called = []

    class FakeExporter:
        def export(self, **kwargs):
            called.append(kwargs)
            return True

    monkeypatch.setattr("core.sequence_preview.SequenceExporter", FakeExporter)

    result = render_sequence_preview(
        sequence=sequence,
        sources=sources,
        clips=clips,
        cache_root=tmp_path / "cache",
    )

    assert result.path == cached_path
    assert result.from_cache is True
    assert called == []
