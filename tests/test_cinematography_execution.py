"""Execution metadata follows actual video fallback and local inference paths."""

from unittest.mock import Mock

import pytest

from core.analysis.cinematography import analyze_cinematography
from models.cinematography import CinematographyAnalysis


@pytest.mark.parametrize(
    "video_error", [None, "ffmpeg extraction failed", "401 unauthorized"]
)
def test_cloud_execution_reports_actual_path_even_on_failure(
    tmp_path, monkeypatch, video_error
):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"video")
    video = Mock(
        return_value=CinematographyAnalysis(
            analysis_model="selected", analysis_mode="video"
        )
    )
    if video_error:
        video.side_effect = RuntimeError(video_error)
    frame = Mock(
        return_value=CinematographyAnalysis(
            analysis_model="selected", analysis_mode="frame"
        )
    )
    monkeypatch.setattr(
        "core.analysis.cinematography.analyze_cinematography_video", video
    )
    monkeypatch.setattr(
        "core.analysis.cinematography.analyze_cinematography_frame", frame
    )
    executions = []

    def run():
        return analyze_cinematography(
            tmp_path / "image.jpg",
            source,
            0,
            30,
            30.0,
            mode="video",
            model="selected",
            tier="cloud",
            local_model="local",
            on_execution=executions.append,
        )

    if video_error == "401 unauthorized":
        with pytest.raises(RuntimeError, match="unauthorized"):
            run()
    else:
        analysis = run()
        assert analysis.analysis_mode == ("frame" if video_error else "video")
    expected = [{"backend": "cloud", "model": "selected", "input_mode": "video"}]
    if video_error == "ffmpeg extraction failed":
        expected.append(
            {"backend": "cloud", "model": "selected", "input_mode": "frame"}
        )
        frame.assert_called_once()
    else:
        frame.assert_not_called()
    assert executions == expected


def test_failed_frame_fallback_retains_frame_execution(tmp_path, monkeypatch):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"video")
    monkeypatch.setattr(
        "core.analysis.cinematography.analyze_cinematography_video",
        Mock(side_effect=RuntimeError("ffmpeg extraction failed")),
    )
    monkeypatch.setattr(
        "core.analysis.cinematography.analyze_cinematography_frame",
        Mock(side_effect=RuntimeError("Invalid answer")),
    )
    executions = []
    with pytest.raises(RuntimeError, match="Invalid answer"):
        analyze_cinematography(
            tmp_path / "image.jpg",
            source,
            0,
            30,
            30.0,
            mode="video",
            model="selected",
            tier="cloud",
            local_model="local",
            on_execution=executions.append,
        )
    assert executions[-1] == {
        "backend": "cloud",
        "model": "selected",
        "input_mode": "frame",
    }


@pytest.mark.parametrize("fail", [False, True])
def test_local_execution_uses_selected_local_model(tmp_path, monkeypatch, fail):
    monkeypatch.setattr("core.analysis.description.is_mlx_vlm_available", lambda: True)
    provider = Mock(return_value='{"shot_size": "CU"}')
    if fail:
        provider.side_effect = RuntimeError("Invalid answer")
    monkeypatch.setattr("core.analysis.description.describe_frame_local", provider)
    executions = []

    def run():
        return analyze_cinematography(
            tmp_path / "image.jpg",
            mode="video",
            model="cloud-model",
            tier="local",
            local_model="local-selected",
            on_execution=executions.append,
        )

    if fail:
        with pytest.raises(RuntimeError, match="Invalid answer"):
            run()
    else:
        result = run()
        assert result.analysis_model == "local-selected"
        assert result.analysis_mode == "frame"
    assert executions == [
        {"backend": "mlx", "model": "local-selected", "input_mode": "frame"}
    ]
    assert provider.call_args.kwargs["model_name"] == "local-selected"
