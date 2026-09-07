"""Shared audio import rejects changed inputs and invalid probe metadata."""

from threading import Event
from unittest.mock import patch
import pytest

from core.operations.audio_import import AudioImportTask, run_audio_import
from core.project import Project
from core.spine.audio_sources import import_audio_source


@pytest.mark.parametrize(
    "change", ["none", "before", "during", "cancel", "nan", "channels"]
)
def test_probe_validates_media_and_metadata(tmp_path, change):
    path = tmp_path / "song.wav"
    path.write_bytes(b"audio")
    task = AudioImportTask.from_path(path)
    cancel = Event()
    if change == "before":
        path.write_bytes(b"changed")

    def probe(*args):
        if change == "during":
            path.write_bytes(b"changed")
        if change == "cancel":
            cancel.set()
        return dict(
            duration=float("nan") if change == "nan" else 5,
            sample_rate=48000,
            channels=-1 if change == "channels" else 2,
        )

    with patch("core.ffmpeg.FFmpegProcessor") as processor:
        processor.return_value.ffprobe_available = True
        processor.return_value.get_audio_info.side_effect = probe
        result = run_audio_import(task, cancel_event=cancel)
        if change == "before":
            processor.assert_not_called()
    assert result.status == (
        "succeeded"
        if change == "none"
        else "unprocessed"
        if change == "cancel"
        else "failed"
    )


def test_spine_deduplicates_canonical_paths(tmp_path):
    path = tmp_path / "song.wav"
    path.write_bytes(b"audio")
    alias = tmp_path / "alias.wav"
    alias.symlink_to(path)
    project = Project.new()
    with patch("core.ffmpeg.FFmpegProcessor") as processor:
        processor.return_value.ffprobe_available = True
        processor.return_value.get_audio_info.return_value = dict(
            duration=5, sample_rate=48000, channels=2
        )
        first = import_audio_source(project, str(path))
        again = import_audio_source(project, str(alias))
    assert first["success"] and again["success"]
    assert again["audio_source_id"] == first["audio_source_id"]
    assert len(project.audio_sources) == 1


def test_cancel_from_final_progress_suppresses_result(tmp_path):
    path = tmp_path / "song.wav"
    path.write_bytes(b"audio")
    cancel = Event()
    with patch("core.ffmpeg.FFmpegProcessor") as processor:
        processor.return_value.ffprobe_available = True
        processor.return_value.get_audio_info.return_value = dict(
            duration=5, sample_rate=48000, channels=2
        )
        result = run_audio_import(
            AudioImportTask.from_path(path),
            cancel_event=cancel,
            progress=lambda current, total: cancel.set() if current == total else None,
        )
    assert result.status == "unprocessed"
