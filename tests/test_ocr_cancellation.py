"""OCR stays inside a clip and discards work completed after cancellation."""

from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.analysis import ocr


@pytest.fixture
def sampling(monkeypatch):
    paths = []
    frames = []

    def extract(source, frame, path, fps):
        frames.append(frame)
        paths.append(path)
        path.write_bytes(b"image")

    monkeypatch.setattr("core.ffmpeg.extract_frame", extract)
    provider = Mock(return_value=("SIGN", 0.9, "paddleocr"))
    monkeypatch.setattr(ocr, "extract_text_from_frame", provider)
    return frames, paths, provider


@pytest.mark.parametrize(
    "start,end,count,expected",
    [
        (10, 11, 3, [10]),
        (10, 13, 3, [10, 11, 12]),
        (10, 20, 3, [10, 14, 19]),
        (10, 20, 1, [15]),
    ],
)
def test_sampling_excludes_next_scene(sampling, start, end, count, expected):
    frames, paths, _ = sampling
    result = ocr.extract_text_from_clip(
        SimpleNamespace(id="clip", start_frame=start, end_frame=end),
        SimpleNamespace(file_path=Path("movie.mp4"), fps=24),
        num_keyframes=count,
    )
    assert frames == expected
    assert [text.frame_number for text in result] == expected
    assert all(not path.exists() for path in paths)


@pytest.mark.parametrize("stage", ["before", "progress", "decode", "inference"])
def test_clip_cancellation_discards_partial_text(sampling, monkeypatch, stage):
    frames, paths, provider = sampling
    cancel = Event()
    if stage == "before":
        cancel.set()
    if stage == "decode":

        def decode(source, frame, path, fps):
            paths.append(path)
            path.write_bytes(b"image")
            cancel.set()

        monkeypatch.setattr("core.ffmpeg.extract_frame", decode)
    if stage == "inference":
        calls = 0

        def infer(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                cancel.set()
            return "SIGN", 0.9, "paddleocr"

        provider.side_effect = infer
    result = ocr.extract_text_from_clip(
        SimpleNamespace(id="clip", start_frame=10, end_frame=20),
        SimpleNamespace(file_path=Path("movie.mp4"), fps=24),
        cancel_event=cancel,
        progress_callback=lambda *_: cancel.set() if stage == "progress" else None,
    )
    assert result == []
    assert all(not path.exists() for path in paths)
    if stage != "inference":
        provider.assert_not_called()
    else:
        assert provider.call_count == 2


def test_cancelled_local_ocr_does_not_start_vlm(monkeypatch, tmp_path):
    cancel = Event()
    engine = Mock()
    engine.ocr.side_effect = lambda *a, **kw: (cancel.set() or [])
    monkeypatch.setattr(ocr, "_check_paddleocr", lambda: True)
    monkeypatch.setattr(ocr, "_get_ocr_engine", lambda: engine)
    vlm = Mock(return_value=("LATE", 0.8))
    monkeypatch.setattr(ocr, "_vlm_text_extraction", vlm)
    assert ocr.extract_text_from_frame(tmp_path / "frame.jpg", cancel_event=cancel) == (
        "",
        0.0,
        "none",
    )
    vlm.assert_not_called()


def test_cancelled_vlm_result_is_discarded(monkeypatch, tmp_path):
    cancel = Event()
    vlm = Mock(side_effect=lambda *a: (cancel.set() or ("LATE", 0.8)))
    monkeypatch.setattr(ocr, "_vlm_text_extraction", vlm)
    assert ocr.extract_text_from_frame(
        tmp_path / "frame.jpg", vlm_only=True, cancel_event=cancel
    ) == ("", 0.0, "none")


def test_cancel_during_engine_load_prevents_inference(monkeypatch, tmp_path):
    cancel = Event()
    engine = Mock()
    monkeypatch.setattr(ocr, "_check_paddleocr", lambda: True)
    monkeypatch.setattr(ocr, "_get_ocr_engine", lambda: (cancel.set() or engine))
    assert ocr.extract_text_from_frame(tmp_path / "frame.jpg", cancel_event=cancel) == (
        "",
        0.0,
        "none",
    )
    engine.ocr.assert_not_called()


def test_spine_cancellation_preserves_completed_clips(monkeypatch, tmp_path):
    from tests.test_description_operations import project_with_thumbnails
    from models.clip import ExtractedText
    from core.spine.analyze import extract_text

    project = project_with_thumbnails(tmp_path, 3)
    cancel = Event()
    calls = []

    def infer(**kwargs):
        assert kwargs["cancel_event"] is cancel
        calls.append(kwargs["clip"].id)
        if len(calls) == 2:
            cancel.set()
        return [
            ExtractedText(frame_number=0, text="SIGN", confidence=0.9, source="vlm")
        ]

    monkeypatch.setattr(ocr, "extract_text_from_clip", infer)
    result = extract_text(project, cancel_event=cancel)["result"]
    assert len(calls) == 2
    assert result["succeeded"] == [{"clip_id": project.clips[0].id, "text_count": 1}]
    assert project.clips[0].extracted_texts[0].text == "SIGN"
    assert all(clip.extracted_texts is None for clip in project.clips[1:])


@pytest.mark.parametrize("surface", ["spine", "worker", "frame_worker"])
def test_surface_rejects_late_result(monkeypatch, tmp_path, surface):
    from tests.test_description_operations import project_with_thumbnails
    from models.clip import ExtractedText
    from core.spine.analyze import extract_text
    from ui.workers.text_extraction_worker import TextExtractionWorker

    project = project_with_thumbnails(tmp_path, 1)
    cancel = Event()
    texts = [ExtractedText(frame_number=0, text="LATE", confidence=0.9, source="vlm")]
    if surface == "spine":
        monkeypatch.setattr(
            ocr, "extract_text_from_clip", lambda **kw: (cancel.set() or texts)
        )
        result = extract_text(project, cancel_event=cancel)
        assert result["result"]["succeeded"] == []
        assert project.clips[0].extracted_texts is None
    else:
        targets = (
            [SimpleNamespace(id="frame", image_path=project.clips[0].thumbnail_path)]
            if surface == "frame_worker"
            else None
        )
        worker = TextExtractionWorker(
            project.clips, project.sources_by_id, analysis_targets=targets
        )
        monkeypatch.setattr(
            ocr, "extract_text_from_clip", lambda **kw: (worker.cancel() or texts)
        )
        monkeypatch.setattr(
            ocr,
            "extract_text_from_frame",
            lambda **kw: (worker.cancel() or ("LATE", 0.9, "vlm")),
        )
        signals = []
        worker.clip_completed.connect(lambda *a: signals.append(a))
        worker.run()
        assert signals == []
