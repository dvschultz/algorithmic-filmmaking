"""Scalar analysis distinguishes failed computation from valid empty media."""

from types import SimpleNamespace
from unittest.mock import Mock
import subprocess

import pytest


@pytest.mark.parametrize("failure", ["open", "decode", "partial"])
def test_brightness_failure_is_not_a_neutral_measurement(
    tmp_path, monkeypatch, failure
):
    from core.analysis import color
    import numpy as np

    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    reads = [(True, frame), (False, None)] if failure == "partial" else [(False, None)]
    cap = SimpleNamespace(
        isOpened=lambda: failure != "open",
        set=Mock(return_value=True),
        read=Mock(side_effect=reads),
        release=Mock(),
    )
    monkeypatch.setattr(color.cv2, "VideoCapture", lambda _: cap)
    with pytest.raises(RuntimeError):
        color.get_average_brightness(tmp_path / "video.mp4", 0, 2, 30)
    cap.release.assert_called_once()


def test_failed_seek_cannot_publish_verified_brightness(tmp_path, monkeypatch):
    import numpy as np
    from core.analysis_availability import scalar_analysis_is_complete
    from core.operations.scalars import ScalarApplication, run_scalars, scalar_task
    from core.project import Project
    from models.clip import Clip, Source

    source = Source(file_path=tmp_path / "video.mp4", fps=30)
    source.file_path.write_bytes(b"media")
    clip = Clip(source_id=source.id, start_frame=240, end_frame=270)
    clip.average_brightness = 0.75
    project = Project(sources=[source], clips=[clip])
    cap = SimpleNamespace(
        isOpened=lambda: True,
        set=Mock(return_value=False),
        read=Mock(return_value=(True, np.zeros((2, 2, 3), dtype=np.uint8))),
        release=Mock(),
    )
    monkeypatch.setattr("core.analysis.color.cv2.VideoCapture", lambda _: cap)
    task = scalar_task(clip, source, "brightness", skip_existing=False)
    application = ScalarApplication(project, task)
    outcome = run_scalars((task,))[0]
    assert outcome.status == "failed"
    assert application.apply(project, outcome)
    assert clip.average_brightness == 0.75
    assert clip.analysis_records["brightness"].state == "failed"
    assert not scalar_analysis_is_complete(clip, source, "brightness")
    cap.read.assert_not_called()
    cap.release.assert_called_once()


def test_brightness_from_unchecked_seek_runtime_requires_recomputation(tmp_path, monkeypatch):
    from dataclasses import replace
    from core.analysis_availability import scalar_analysis_is_complete
    from core.operations.scalars import ScalarApplication, run_scalars, scalar_task
    from core.project import Project
    from models.analysis_record import AnalysisIdentity
    from models.clip import Clip, Source

    source = Source(file_path=tmp_path / "video.mp4", fps=30)
    source.file_path.write_bytes(b"media")
    clip = Clip(source_id=source.id, start_frame=240, end_frame=270)
    project = Project(sources=[source], clips=[clip])
    provider = Mock(return_value=0.25)
    monkeypatch.setattr("core.analysis.color.get_average_brightness", provider)
    task = scalar_task(clip, source, "brightness")
    assert ScalarApplication(project, task).apply(project, run_scalars((task,))[0])
    previous = clip.analysis_records["brightness"]
    identity = previous.identity.to_dict()
    identity["model"]["name"] = "opencv-gray-mean/v1"
    clip.analysis_records["brightness"] = replace(previous, identity=AnalysisIdentity.from_dict(identity))
    assert not scalar_analysis_is_complete(clip, source, "brightness")
    provider.return_value = 0.75
    task = scalar_task(clip, source, "brightness")
    outcome = run_scalars((task,))[0]
    assert outcome.status == "succeeded"
    assert ScalarApplication(project, task).apply(project, outcome)
    assert clip.average_brightness == 0.75
    assert provider.call_count == 2
    assert scalar_analysis_is_complete(clip, source, "brightness")


@pytest.mark.parametrize("failure", ["exit", "missing", "nan", "timeout"])
def test_volume_failure_is_not_no_audio(tmp_path, monkeypatch, failure):
    from core.analysis import audio

    result = SimpleNamespace(
        returncode=1 if failure == "exit" else 0,
        stderr="mean_volume: nan dB"
        if failure == "nan"
        else "mean_volume: -20 dB"
        if failure == "exit"
        else "",
    )
    process = (
        Mock(side_effect=subprocess.TimeoutExpired("ffmpeg", 60))
        if failure == "timeout"
        else Mock(return_value=result)
    )
    monkeypatch.setattr(audio.subprocess, "run", process)
    with pytest.raises((RuntimeError, ValueError, subprocess.TimeoutExpired)):
        audio.extract_clip_volume(tmp_path / "video.mp4", 0.0, 1.0, _has_audio=True)


def test_volume_probe_failure_is_not_no_audio(tmp_path, monkeypatch):
    from core.analysis import audio

    monkeypatch.setattr(
        audio.subprocess,
        "run",
        Mock(return_value=SimpleNamespace(returncode=1, stderr="bad media", stdout="")),
    )
    with pytest.raises(RuntimeError):
        audio.extract_clip_volume(tmp_path / "bad.mp4", 0.0, 1.0)


@pytest.mark.parametrize("operation", ["brightness", "volume"])
def test_sequencing_does_not_cache_a_failed_measurement(
    tmp_path, monkeypatch, operation
):
    from core import remix
    from models.clip import Clip, Source

    clip = Clip(source_id="source", start_frame=0, end_frame=30)
    source = Source(id="source", file_path=tmp_path / "video.mp4", fps=30)
    if operation == "brightness":
        monkeypatch.setattr(
            "core.analysis.color.get_average_brightness",
            Mock(side_effect=RuntimeError("decode failed")),
        )
        compute = remix._auto_compute_brightness
    else:
        monkeypatch.setattr(
            "core.analysis.audio.has_audio_track", lambda *args, **kwargs: True
        )
        monkeypatch.setattr(
            "core.analysis.audio.extract_clip_volume",
            Mock(side_effect=RuntimeError("decode failed")),
        )
        compute = remix._auto_compute_volume
    with pytest.raises(RuntimeError):
        compute([(clip, source)])
    assert clip.average_brightness is None and clip.rms_volume is None
