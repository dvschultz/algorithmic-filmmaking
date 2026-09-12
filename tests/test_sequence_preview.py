"""Tests for cached continuous sequence previews."""

import pytest

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


def test_preview_signature_changes_when_source_file_changes(tmp_path):
    sequence, sources, clips = _make_sequence_with_clips(tmp_path)
    first = compute_sequence_preview_signature(sequence, sources, clips)

    source = next(iter(sources.values()))
    source.file_path.write_bytes(b"changed source bytes")
    second = compute_sequence_preview_signature(sequence, sources, clips)

    assert first != second


def test_preview_signature_changes_when_proxy_settings_change(tmp_path):
    sequence, sources, clips = _make_sequence_with_clips(tmp_path)
    first = compute_sequence_preview_signature(
        sequence,
        sources,
        clips,
        settings=SequencePreviewSettings(width=1280, height=720),
    )
    second = compute_sequence_preview_signature(
        sequence,
        sources,
        clips,
        settings=SequencePreviewSettings(width=854, height=480),
    )

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
    called = []

    class FakeExporter:
        def export(self, **kwargs):
            called.append(kwargs)
            kwargs["config"].output_path.write_bytes(b"cached")
            return True

    monkeypatch.setattr("core.sequence_preview.SequenceExporter", FakeExporter)

    first = render_sequence_preview(sequence, sources, clips, cache_root=tmp_path / "cache")

    result = render_sequence_preview(
        sequence=sequence,
        sources=sources,
        clips=clips,
        cache_root=tmp_path / "cache",
    )

    assert result.path == first.path
    assert result.from_cache is True
    assert len(called) == 1


def test_unregistered_preview_is_preserved_but_not_reused(monkeypatch, tmp_path):
    from core.sequence_preview import cleanup_sequence_preview_cache

    sequence, sources, clips = _make_sequence_with_clips(tmp_path)
    signature = compute_sequence_preview_signature(sequence, sources, clips)
    legacy = get_sequence_preview_path(sequence, signature, tmp_path / "cache")
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"unverified")

    class FakeExporter:
        def export(self, **kwargs):
            kwargs["config"].output_path.write_bytes(b"verified")
            return True

    monkeypatch.setattr("core.sequence_preview.SequenceExporter", FakeExporter)
    result = render_sequence_preview(sequence, sources, clips, cache_root=tmp_path / "cache")
    assert not result.from_cache and result.path.read_bytes() == b"verified"
    cleanup_sequence_preview_cache(tmp_path / "cache", keep_latest=0)
    assert legacy.read_bytes() == b"unverified"
    assert result.path.read_bytes() == b"verified"


def test_failed_preview_never_publishes_partial_render(monkeypatch, tmp_path):
    sequence, sources, clips = _make_sequence_with_clips(tmp_path)
    calls = []

    class FakeExporter:
        def export(self, **kwargs):
            calls.append(1)
            kwargs["config"].output_path.write_bytes(b"partial")
            return False

    monkeypatch.setattr("core.sequence_preview.SequenceExporter", FakeExporter)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="render failed"):
            render_sequence_preview(sequence, sources, clips, cache_root=tmp_path / "cache")
    assert len(calls) == 2
    assert not list((tmp_path / "cache" / "sequence_previews").glob("render-*"))


def test_corrupt_managed_preview_is_recomputed(monkeypatch, tmp_path):
    sequence, sources, clips = _make_sequence_with_clips(tmp_path)
    calls = []

    class FakeExporter:
        def export(self, **kwargs):
            calls.append(1)
            kwargs["config"].output_path.write_bytes(b"complete render")
            return True

    monkeypatch.setattr("core.sequence_preview.SequenceExporter", FakeExporter)
    first = render_sequence_preview(sequence, sources, clips, cache_root=tmp_path / "cache")
    first.path.write_bytes(b"corrupt")
    second = render_sequence_preview(sequence, sources, clips, cache_root=tmp_path / "cache")
    assert not second.from_cache and len(calls) == 2
    assert second.path.read_bytes() == b"complete render"



def _edit_keeping_mtime(path, data: bytes, before) -> None:
    """Rewrite the file with the same size, restore its mtime, and let the edit register.

    The signature detects this edit through ``st_ctime_ns``. Linux stamps inode
    times from a coarse clock (one scheduler tick, typically 1-4 ms), so an edit
    made in the same tick as the baseline stat carries an identical ctime and is
    genuinely indistinguishable by stat alone -- the property under test is that
    an edit the filesystem recorded is detected, not that stat has sub-tick
    resolution. Wait for the clock to move rather than asserting into that gap.
    """
    import os
    import time

    assert len(data) == before.st_size, "the edit must keep the size to exercise the ctime path"
    for _ in range(400):
        path.write_bytes(data)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        after = path.stat()
        if after.st_ctime_ns != before.st_ctime_ns:
            assert after.st_mtime_ns == before.st_mtime_ns
            return
        time.sleep(0.005)
    raise AssertionError("filesystem never recorded a new ctime for the edit")


def test_source_edit_with_restored_mtime_invalidates_preview_signature(tmp_path):
    sequence, sources, clips = _make_sequence_with_clips(tmp_path)
    source = next(iter(sources.values())).file_path
    before = source.stat()
    first = compute_sequence_preview_signature(sequence, sources, clips)
    _edit_keeping_mtime(source, b"change", before)
    assert first != compute_sequence_preview_signature(sequence, sources, clips)


def test_source_edit_during_render_is_not_published(monkeypatch, tmp_path):
    sequence, sources, clips = _make_sequence_with_clips(tmp_path)
    source = next(iter(sources.values())).file_path
    before = source.stat()

    class FakeExporter:
        def export(self, **kwargs):
            kwargs["config"].output_path.write_bytes(b"stale render")
            _edit_keeping_mtime(source, b"change", before)
            return True

    monkeypatch.setattr("core.sequence_preview.SequenceExporter", FakeExporter)
    with pytest.raises(RuntimeError, match="inputs changed"):
        render_sequence_preview(sequence, sources, clips, cache_root=tmp_path / "cache")


@pytest.mark.parametrize("invalid", ["overlap", "missing_music", "missing_clip"])
def test_cache_hit_cannot_bypass_render_plan_validation(tmp_path, invalid):
    sequence, sources, clips = _make_sequence_with_clips(tmp_path)
    if invalid == "overlap":
        sequence.tracks[0].clips[1].start_frame = 12
    elif invalid == "missing_music":
        sequence.music_path = str(tmp_path / "missing.wav")
    else:
        clips.pop("clip-a")
    signature = compute_sequence_preview_signature(sequence, sources, clips)
    cached = get_sequence_preview_path(sequence, signature, tmp_path)
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"cached but not valid for this edit")
    with pytest.raises(ValueError):
        render_sequence_preview(sequence, sources, clips, cache_root=tmp_path)
    assert cached.exists()
