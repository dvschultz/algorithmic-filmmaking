"""Shared OCR must use detached inputs and preserve failure/empty semantics."""

from threading import Event, Thread
from unittest.mock import Mock

import pytest

from core.analysis import ocr
from core.analysis_target import AnalysisTarget
from core.spine.analyze import extract_text
from models.clip import ExtractedText
from tests.test_description_operations import project_with_thumbnails
from ui.workers.text_extraction_worker import TextExtractionWorker


def test_worker_captures_ranges_before_dispatch(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 1)
    clip = project.clips[0]
    start = clip.start_frame
    provider = Mock(return_value=[])
    monkeypatch.setattr(ocr, "extract_text_from_clip", provider)
    worker = TextExtractionWorker(project.clips, project.sources_by_id)
    clip.start_frame += 10
    worker.run()
    assert provider.call_args.kwargs["clip"].start_frame == start
    assert provider.call_args.kwargs["clip"] is not clip


def test_clip_analysis_target_uses_video_not_thumbnail(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 1)
    target = AnalysisTarget.from_clip(project.clips[0], project.sources[0])
    video = Mock(return_value=[])
    image = Mock(return_value=("WRONG", 1.0, "vlm"))
    monkeypatch.setattr(ocr, "extract_text_from_clip", video)
    monkeypatch.setattr(ocr, "extract_text_from_frame", image)
    worker = TextExtractionWorker([], {}, analysis_targets=[target])
    worker.run()
    video.assert_called_once()
    image.assert_not_called()


@pytest.mark.parametrize("raw", [float("nan"), 2.0, True])
def test_invalid_confidence_is_not_published(tmp_path, monkeypatch, raw):
    project = project_with_thumbnails(tmp_path, 1)
    provider = Mock(return_value=[ExtractedText(0, "SIGN", raw, "vlm")])
    monkeypatch.setattr(ocr, "extract_text_from_clip", provider)
    result = extract_text(project)["result"]
    assert not result["succeeded"]
    assert result["failed"][0]["code"] == "text_extraction_failed"
    assert project.clips[0].extracted_texts is None


def test_no_ocr_runtime_is_failure_not_empty_observation(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 1)
    monkeypatch.setattr(ocr, "_check_paddleocr", lambda: False)
    monkeypatch.setattr(
        "core.ffmpeg.extract_frame", lambda a, b, path, d: path.write_bytes(b"frame")
    )
    result = extract_text(project, use_vlm_fallback=False)["result"]
    assert result["failed"]
    assert not result["succeeded"]
    assert project.clips[0].extracted_texts is None


def test_valid_empty_observation_is_success(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 1)
    monkeypatch.setattr(ocr, "extract_text_from_clip", lambda **kw: [])
    result = extract_text(project)["result"]
    assert result["succeeded"] == [{"clip_id": project.clips[0].id, "text_count": 0}]
    assert project.clips[0].extracted_texts == []


def test_verified_empty_ocr_reuses_without_provider(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 1)
    provider = Mock(return_value=[])
    monkeypatch.setattr(ocr, "extract_text_from_clip", provider)
    assert extract_text(project)["result"]["succeeded"]
    record = project.clips[0].analysis_records["extract_text"]
    assert record.state == "succeeded" and record.value == {"extracted_texts": []}
    assert extract_text(project)["result"]["skipped"]
    assert provider.call_count == 1


def test_failed_ocr_attempt_preserves_projection_but_invalidates_reuse(
    tmp_path, monkeypatch
):
    project = project_with_thumbnails(tmp_path, 1)
    provider = Mock(return_value=[])
    monkeypatch.setattr(ocr, "extract_text_from_clip", provider)
    extract_text(project)
    provider.side_effect = RuntimeError("provider unavailable")
    assert extract_text(project, skip_existing=False)["result"]["failed"]
    assert project.clips[0].extracted_texts == []
    assert project.clips[0].analysis_records["extract_text"].state == "failed"
    provider.side_effect = None
    assert extract_text(project)["result"]["succeeded"]
    assert provider.call_count == 3


@pytest.mark.parametrize(
    "change", ["range", "source", "parameters", "model", "projection"]
)
def test_ocr_identity_invalidates_changed_inputs(tmp_path, monkeypatch, change):
    project = project_with_thumbnails(tmp_path, 1)
    provider = Mock(return_value=[])
    monkeypatch.setattr(ocr, "extract_text_from_clip", provider)
    extract_text(project, vlm_model="first")
    kwargs = {"vlm_model": "first"}
    if change == "range":
        project.clips[0].end_frame -= 1
    elif change == "source":
        project.sources[0].file_path.write_bytes(b"replacement video")
    elif change == "parameters":
        kwargs["num_keyframes"] = 1
    elif change == "model":
        kwargs["vlm_model"] = "second"
    else:
        project.clips[0].extracted_texts = [ExtractedText(0, "edited", 1.0, "vlm")]
    assert extract_text(project, **kwargs)["result"]["succeeded"]
    assert provider.call_count == 2


def test_ocr_rebinds_identical_relocated_source_without_inference(
    tmp_path, monkeypatch
):
    project = project_with_thumbnails(tmp_path, 1)
    provider = Mock(return_value=[])
    monkeypatch.setattr(ocr, "extract_text_from_clip", provider)
    extract_text(project)
    original = project.clips[0].analysis_records["extract_text"]
    moved = tmp_path / "moved.mp4"
    moved.write_bytes(project.sources[0].file_path.read_bytes())
    project.sources[0].file_path = moved
    assert extract_text(project)["result"]["skipped"]
    current = project.clips[0].analysis_records["extract_text"]
    assert current.identity == original.identity
    assert current.input_json != original.input_json
    assert provider.call_count == 1


def test_ocr_runtime_change_recomputes(tmp_path, monkeypatch):
    from core.analysis_model_identity import ocr_runtime

    project = project_with_thumbnails(tmp_path, 1)
    provider = Mock(return_value=[])
    monkeypatch.setattr(ocr, "extract_text_from_clip", provider)
    extract_text(project)
    changed = {**ocr_runtime(), "algorithm": "next-version"}
    monkeypatch.setattr("core.operations.ocr.ocr_runtime", lambda: changed)
    assert extract_text(project)["result"]["succeeded"]
    assert provider.call_count == 2


def test_reuse_delivery_error_is_not_recorded_as_provider_failure(
    tmp_path, monkeypatch
):
    from core.operations.ocr import OcrOptions, ocr_task, run_ocr

    project = project_with_thumbnails(tmp_path, 1)
    provider = Mock(return_value=[])
    monkeypatch.setattr(ocr, "extract_text_from_clip", provider)
    extract_text(project)
    deliver = Mock(side_effect=RuntimeError("owner delivery failed"))
    with pytest.raises(RuntimeError, match="owner delivery failed"):
        run_ocr(
            (ocr_task(project.clips[0], project.sources[0]),),
            OcrOptions(),
            on_outcome=deliver,
        )
    deliver.assert_called_once()
    assert deliver.call_args.args[0].status == "skipped"
    assert provider.call_count == 1


def test_frame_ocr_empty_reuse_and_image_invalidation(tmp_path, monkeypatch):
    from core.operations.ocr import OcrApplication, OcrOptions, ocr_task, run_ocr
    from models.frame import Frame

    project = project_with_thumbnails(tmp_path, 1)
    frame = Frame(
        id="frame", file_path=project.clips[0].thumbnail_path, frame_number=12
    )
    project.add_frames([frame])
    provider = Mock(return_value=("", 0.0, "none"))
    monkeypatch.setattr(ocr, "extract_text_from_frame", provider)
    for attempt in ("compute", "reuse", "change"):
        if attempt == "change":
            frame.file_path.write_bytes(b"new frame")
        task = ocr_task(frame)
        application = OcrApplication(project, (task,))
        outcome = run_ocr((task,), OcrOptions())[0]
        assert outcome.status == ("skipped" if attempt == "reuse" else "succeeded")
        assert application.apply(project, outcome)
        assert frame.extracted_texts == []
    assert provider.call_count == 2


def test_provider_cannot_overwrite_concurrent_edit(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 1)
    edited = [ExtractedText(0, "EDIT", 1.0, "vlm")]

    def infer(**kwargs):
        project.clips[0].extracted_texts = edited
        return [ExtractedText(0, "STALE", 1.0, "vlm")]

    monkeypatch.setattr(ocr, "extract_text_from_clip", infer)
    result = extract_text(project)["result"]
    assert project.clips[0].extracted_texts == edited
    assert not result["succeeded"]


def test_changed_source_is_rejected(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 1)

    def infer(**kwargs):
        project.sources[0].file_path.write_bytes(b"changed source")
        return []

    monkeypatch.setattr(ocr, "extract_text_from_clip", infer)
    result = extract_text(project)["result"]
    assert not result["succeeded"]
    assert project.clips[0].extracted_texts is None


def test_cancel_while_waiting_for_shared_model(tmp_path, monkeypatch):
    from core.operations.ocr import OcrOptions, OcrTask, run_ocr

    project = project_with_thumbnails(tmp_path, 1)
    tasks = (OcrTask.from_clip(project.clips[0], project.sources[0]),)
    entered, release, finished, cancel = Event(), Event(), Event(), Event()
    calls = []

    def infer(**kwargs):
        calls.append(kwargs)
        entered.set()
        assert release.wait(5)
        return []

    monkeypatch.setattr(ocr, "extract_text_from_clip", infer)
    first = Thread(target=lambda: run_ocr(tasks, OcrOptions()))
    second_results = []

    def second_run():
        second_results.extend(run_ocr(tasks, OcrOptions(), cancel_event=cancel))
        finished.set()

    second = Thread(target=second_run)
    first.start()
    try:
        assert entered.wait(5)
        second.start()
        cancel.set()
        assert finished.wait(5)
        assert len(calls) == 1
        assert second_results[0].status == "unprocessed"
    finally:
        release.set()
        first.join(5)
        if second.ident is not None:
            second.join(5)


def test_strict_decode_failure_cleans_temporary_file(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 1)
    paths = []
    monkeypatch.setattr("core.ffmpeg.extract_frame", lambda a, b, p, d: paths.append(p))
    result = extract_text(project)["result"]
    assert result["failed"]
    assert project.clips[0].extracted_texts is None
    assert len(paths) == 1 and not paths[0].exists()


def test_vlm_failure_is_not_saved_as_empty(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 1)
    monkeypatch.setattr(
        "core.ffmpeg.extract_frame", lambda a, b, p, d: p.write_bytes(b"image")
    )
    monkeypatch.setattr(
        ocr, "_vlm_text_extraction", Mock(side_effect=RuntimeError("provider failed"))
    )
    result = extract_text(project, vlm_only=True)["result"]
    assert result["failed"] and not result["succeeded"]
    assert project.clips[0].extracted_texts is None


def test_empty_frame_observation_survives_save(tmp_path):
    from core.project import Project
    from models.frame import Frame

    project = project_with_thumbnails(tmp_path, 1)
    frame = Frame(
        id="frame", file_path=project.clips[0].thumbnail_path, extracted_texts=[]
    )
    project.add_frames([frame])
    path = tmp_path / "project.json"
    project.save(path)
    assert Project.load(path).frames_by_id[frame.id].extracted_texts == []


@pytest.mark.parametrize(
    "raw", [(None, 0.5, "vlm"), (0, 0.5, "vlm"), ("", float("nan"), "vlm")]
)
def test_invalid_empty_frame_reply_is_failed(tmp_path, monkeypatch, raw):
    project = project_with_thumbnails(tmp_path, 1)
    target = AnalysisTarget(
        "frame", "frame", image_path=project.clips[0].thumbnail_path
    )
    monkeypatch.setattr(ocr, "extract_text_from_frame", lambda **kw: raw)
    worker = TextExtractionWorker([], {}, analysis_targets=[target])
    worker.run()
    assert worker.result[0].status == "failed"


def test_gui_and_spine_share_provider_options(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 1)
    provider = Mock(return_value=[ExtractedText(0, "SIGN", 0.9, "vlm")])
    monkeypatch.setattr(ocr, "extract_text_from_clip", provider)
    worker = TextExtractionWorker(
        project.clips,
        project.sources_by_id,
        num_keyframes=4,
        vlm_only=True,
        vlm_model="test",
    )
    worker.run()
    assert project.clips[0].extracted_texts is None
    result = extract_text(project, num_keyframes=4, vlm_only=True, vlm_model="test")[
        "result"
    ]
    first, second = [dict(call.kwargs) for call in provider.call_args_list]
    first.pop("cancel_event")
    second.pop("cancel_event")
    assert first == second
    assert result["succeeded"][0]["text_count"] == 1
    assert worker.result[0].to_models() == project.clips[0].extracted_texts


def test_model_download_failure_stops_remaining_clips(tmp_path, monkeypatch):
    from core.errors import ModelDownloadError

    project = project_with_thumbnails(tmp_path, 3)
    provider = Mock(side_effect=ModelDownloadError("download failed"))
    monkeypatch.setattr(ocr, "extract_text_from_clip", provider)
    worker = TextExtractionWorker(
        project.clips, project.sources_by_id, use_vlm_fallback=False
    )
    worker.run()
    provider.assert_called_once()
    assert worker.result[0].code == "model_load_failed"
    assert [value.status for value in worker.result[1:]] == [
        "unprocessed",
        "unprocessed",
    ]
