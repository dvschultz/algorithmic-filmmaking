"""Description requests pin provider settings and local cache identity."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.analysis import description
from core.settings import Settings
from tests.test_description_operations import project_with_thumbnails
from ui.workers.description_worker import DescriptionWorker


def test_worker_snapshots_model_and_input_mode(tmp_path, monkeypatch):
    settings = Settings(
        description_model_cloud="gemini-original", description_input_mode="video"
    )
    monkeypatch.setattr("core.settings.load_settings", lambda: settings)
    project = project_with_thumbnails(tmp_path, 1)
    worker = DescriptionWorker(
        project.clips, sources=project.sources_by_id, tier="cloud"
    )
    settings.description_model_cloud = "gpt-replacement"
    settings.description_input_mode = "frame"
    provider = Mock(return_value=("Description", "gemini-original"))
    monkeypatch.setattr(description, "describe_frame", provider)
    worker.run()
    assert provider.call_args.kwargs["model_name"] == "gemini-original"
    assert provider.call_args.kwargs["input_mode"] == "video"
    assert worker.options.model == "gemini-original"


@pytest.mark.parametrize("mode", ["frame", "video"])
def test_routing_forwards_pinned_cloud_model(tmp_path, monkeypatch, mode):
    settings = Settings(
        description_model_cloud="gpt-replacement", description_input_mode="frame"
    )
    monkeypatch.setattr(description, "load_settings", lambda: settings)
    video = tmp_path / "segment.mp4"
    video.write_bytes(b"fake")
    monkeypatch.setattr(description, "extract_clip_segment", lambda *args: video)
    frame_provider = Mock(return_value="Frame")
    video_provider = Mock(return_value=("Video", "gemini-original (video)"))
    monkeypatch.setattr(description, "describe_frame_cloud", frame_provider)
    monkeypatch.setattr(description, "describe_video_cloud", video_provider)
    result = description.describe_frame(
        tmp_path / "thumb.jpg",
        tier="cloud",
        source_path=tmp_path / "source.mp4",
        start_frame=0,
        end_frame=30,
        fps=30,
        model_name="gemini-original",
        input_mode=mode,
    )
    provider = video_provider if mode == "video" else frame_provider
    assert provider.call_args.kwargs["model_name"] == "gemini-original"
    assert result[1].startswith("gemini-original")
    if mode == "video":
        assert not video.exists()
        frame_provider.assert_not_called()
    else:
        video_provider.assert_not_called()


@pytest.mark.parametrize("video", [False, True])
def test_cloud_provider_uses_explicit_model_without_reloading_settings(
    tmp_path, monkeypatch, video
):
    monkeypatch.setattr(
        description, "load_settings", lambda: pytest.fail("must use pinned model")
    )
    monkeypatch.setattr("core.settings.get_gemini_api_key", lambda: "test-key")
    monkeypatch.setattr(description, "encode_image_base64", lambda path: "image")
    monkeypatch.setattr(description, "encode_video_base64", lambda path: "video")
    completion = Mock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Output"))]
        )
    )
    monkeypatch.setattr("core.llm_client.complete_routed", completion)
    provider = (
        description.describe_video_cloud if video else description.describe_frame_cloud
    )
    (tmp_path / "media").write_bytes(b"fake")
    provider(tmp_path / "media", model_name="gemini-original")
    assert completion.call_args.kwargs["model"] == "gemini/gemini-original"


def test_local_model_cache_is_keyed_by_requested_model_and_backend(monkeypatch):
    monkeypatch.setattr(description, "_LOCAL_MODEL", None)
    monkeypatch.setattr(description, "_LOCAL_PROCESSOR", None)
    monkeypatch.setattr(description, "_LOCAL_MODEL_KEY", None)
    backend = [True]
    monkeypatch.setattr(description, "is_mlx_vlm_available", lambda: backend[0])
    loads = []

    def load(model):
        loads.append((model, backend[0]))
        description._LOCAL_MODEL = object()
        description._LOCAL_PROCESSOR = object()

    monkeypatch.setattr(description, "_load_qwen3_vlm", load)
    monkeypatch.setattr(description, "_load_moondream_fallback", load)
    first = description._load_local_model("model-a")
    assert description._load_local_model("model-a") == first
    assert description.is_model_loaded("model-a")
    assert not description.is_model_loaded("model-b")
    second = description._load_local_model("model-b")
    assert second != first
    backend[0] = False
    assert not description.is_model_loaded("model-b")
    description._load_local_model("model-b")
    assert loads == [("model-a", True), ("model-b", True), ("model-b", False)]


def test_explicit_model_still_snapshots_default_input_mode(tmp_path, monkeypatch):
    from core.operations.description import (
        DescriptionOptions,
        DescriptionTask,
        run_description,
    )

    settings = Settings(description_input_mode="video")
    monkeypatch.setattr("core.settings.load_settings", lambda: settings)
    path = tmp_path / "thumb.jpg"
    path.write_bytes(b"fake")
    calls = []

    def provider(*args, **kwargs):
        calls.append(kwargs["input_mode"])
        settings.description_input_mode = "frame"
        return "Result", "model"

    monkeypatch.setattr(description, "describe_frame", provider)
    tasks = tuple(DescriptionTask(str(i), path, None, 0, 1, None) for i in range(2))
    run_description(tasks, DescriptionOptions("cloud", model="gemini-pinned"))
    assert calls == ["video", "video"]


@pytest.mark.parametrize("tier", ["local", "cpu", "gpu"])
def test_local_fallback_reports_executed_model(tmp_path, monkeypatch, tier):
    monkeypatch.setattr(description, "is_mlx_vlm_available", lambda: False)
    monkeypatch.setattr(
        description, "_load_local_model", lambda model: (object(), object())
    )
    monkeypatch.setattr(
        description, "_describe_with_moondream", lambda *args: "A person"
    )
    result = description.describe_frame(
        tmp_path / "thumb.jpg",
        tier=tier,
        model_name="mlx-community/Qwen3-VL-4B-Instruct-4bit",
    )
    assert result == ("A person", "vikhyatk/moondream2")


@pytest.mark.parametrize("extraction_fails", [False, True])
def test_execution_metadata_records_actual_cloud_input(
    tmp_path, monkeypatch, extraction_fails
):
    video = tmp_path / "segment.mp4"
    video.write_bytes(b"video")

    def extract(*args):
        if extraction_fails:
            raise RuntimeError("extraction failed")
        return video

    monkeypatch.setattr(description, "extract_clip_segment", extract)
    monkeypatch.setattr(
        description,
        "describe_video_cloud",
        lambda *args, **kw: ("Video", "gemini-test (video)"),
    )
    monkeypatch.setattr(
        description, "describe_frame_cloud", lambda *args, **kw: "Frame"
    )
    executions = []
    description.describe_frame(
        tmp_path / "thumb.jpg",
        tier="cloud",
        model_name="gemini-test",
        input_mode="video",
        source_path=tmp_path / "source.mp4",
        start_frame=0,
        end_frame=30,
        fps=30,
        on_execution=executions.append,
    )
    assert executions == [
        {
            "backend": "cloud",
            "model": "gemini-test",
            "input_mode": "frame" if extraction_fails else "video",
        }
    ]


def test_execution_metadata_survives_provider_failure(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("provider failed")

    monkeypatch.setattr(description, "describe_frame_cloud", fail)
    executions = []
    with pytest.raises(RuntimeError, match="provider failed"):
        description.describe_frame(
            tmp_path / "thumb.jpg",
            tier="cloud",
            model_name="gpt-test",
            input_mode="frame",
            on_execution=executions.append,
        )
    assert executions == [
        {"backend": "cloud", "model": "gpt-test", "input_mode": "frame"}
    ]


@pytest.mark.parametrize(
    "mlx, requested, actual",
    [
        (True, "mlx-community/Qwen3-VL", "mlx-community/Qwen3-VL"),
        (False, "mlx-community/Qwen3-VL", "vikhyatk/moondream2"),
        (False, "custom/moondream", "custom/moondream"),
    ],
)
def test_local_execution_and_job_runtime_agree(
    tmp_path, monkeypatch, mlx, requested, actual
):
    from core.jobs.description import _runtime
    from core.operations.description import DescriptionOptions

    monkeypatch.setattr(description, "is_mlx_vlm_available", lambda: mlx)
    monkeypatch.setattr(
        description, "describe_frame_local", lambda *args, **kw: "Description"
    )
    executions = []
    result = description.describe_frame(
        tmp_path / "thumb.jpg",
        tier="local",
        model_name=requested,
        input_mode="video",
        on_execution=executions.append,
    )
    runtime = _runtime(DescriptionOptions("local", model=requested))
    assert result[1] == actual
    assert runtime["model"] == actual
    assert executions == [{**runtime, "input_mode": "frame"}]
