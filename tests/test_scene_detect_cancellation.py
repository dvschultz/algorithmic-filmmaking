"""Regression tests for issue #162: cancelling detection must stop the frame scan.

Both detection paths previously ignored cancellation until the whole video had
been processed. These tests use fakes for the PySceneDetect / OpenCV layers so
no real video is decoded.
"""

import sys
import threading
import time
from threading import Event
from typing import ClassVar

import numpy as np
import pytest

from core import scene_detect
from core.operations.detection import (
    DetectionCancelled,
    DetectionRequest,
    run_detection,
)
from core.scene_detect import KaraokeDetectionConfig, SceneDetector

TOTAL_FRAMES = 10_000
CANCEL_AT_FRAME = 20
FRAME_SECONDS = 0.001


class _FakeDuration:
    def get_seconds(self):
        return TOTAL_FRAMES / 25.0

    def get_frames(self):
        return TOTAL_FRAMES


class _FakeVideoStream:
    def __init__(self, path):
        self.duration = _FakeDuration()
        self.frame_rate = 25.0
        self.frame_size = (64, 48)


class _FakeSceneManager:
    """Mimics PySceneDetect on a video with no cuts.

    The real ``SceneManager`` only invokes ``callback`` when a cut is detected,
    so on a long shot the scan never calls back into our code. This fake never
    calls it at all; the only way to end the scan early is ``stop()``.
    ``on_frame`` lets a test cancel mid-scan, as a user clicking Cancel would.
    """

    instances: ClassVar[list["_FakeSceneManager"]] = []
    on_frame: ClassVar = None

    def __init__(self):
        self._stop = Event()
        self.stop_calls = 0
        self.frames_processed = 0
        self.callback_calls = 0
        _FakeSceneManager.instances.append(self)

    def add_detector(self, detector):
        pass

    def stop(self):
        self.stop_calls += 1
        self._stop.set()

    def detect_scenes(self, video, show_progress=False, callback=None):
        frame_num = 0
        while not self._stop.is_set() and frame_num < TOTAL_FRAMES:
            if _FakeSceneManager.on_frame is not None:
                _FakeSceneManager.on_frame(frame_num)
            self.frames_processed += 1
            frame_num += 1
            time.sleep(FRAME_SECONDS)
        return frame_num

    def get_scene_list(self):
        return []


@pytest.fixture
def fake_adaptive(monkeypatch):
    _FakeSceneManager.instances = []
    _FakeSceneManager.on_frame = None
    monkeypatch.setattr(scene_detect, "SceneManager", _FakeSceneManager)
    monkeypatch.setattr(scene_detect, "VideoStreamCv2", _FakeVideoStream)
    monkeypatch.setattr(SceneDetector, "_verify_source_timing", staticmethod(lambda source: None))
    monkeypatch.setattr(SceneDetector, "_resolve_luma_only", lambda self, path, source: False)
    yield _FakeSceneManager.instances
    _FakeSceneManager.on_frame = None


def _cancel_mid_scan(cancel):
    def on_frame(frame_num):
        if frame_num == CANCEL_AT_FRAME:
            cancel.set()

    _FakeSceneManager.on_frame = on_frame


def _assert_adaptive_scan_stopped(instances, records):
    assert len(instances) == 1
    manager = instances[0]
    assert manager.stop_calls >= 1
    assert manager.frames_processed < TOTAL_FRAMES // 10, manager.frames_processed
    assert all(value < 0.9 for value, _ in records), records
    assert not any("Found" in message or "Extracting" in message for _, message in records)


def test_adaptive_scan_stops_when_cancelled_without_any_cut(tmp_path, fake_adaptive):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"fake")
    cancel = Event()
    records = []
    _cancel_mid_scan(cancel)

    with pytest.raises(DetectionCancelled):
        SceneDetector().detect_scenes_with_progress(
            video, lambda value, message: records.append((value, message)), cancel_event=cancel
        )

    _assert_adaptive_scan_stopped(fake_adaptive, records)


def test_completed_adaptive_scan_returns_and_stops_watcher(tmp_path, fake_adaptive, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "TOTAL_FRAMES", 50)
    video = tmp_path / "v.mp4"
    video.write_bytes(b"fake")

    _source, clips = SceneDetector().detect_scenes_with_progress(
        video, lambda *_: None, cancel_event=Event()
    )

    assert len(clips) == 1  # full-video fallback clip
    assert fake_adaptive[0].frames_processed == 50
    assert fake_adaptive[0].stop_calls == 0
    assert not any(t.name == "scene-detect-cancel" for t in threading.enumerate())


def test_run_detection_forwards_cancel_event_to_adaptive_scan(tmp_path, fake_adaptive):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"fake")
    request = DetectionRequest.build(video)
    cancel = Event()
    records = []
    _cancel_mid_scan(cancel)

    with pytest.raises(DetectionCancelled):
        run_detection(
            request,
            progress_callback=lambda value, message: records.append((value, message)),
            cancel_event=cancel,
        )

    _assert_adaptive_scan_stopped(fake_adaptive, records)



class _FakeCapture:
    FPS = 25.0
    FRAMES = 25 * 600  # 600 coarse samples at one per second

    def __init__(self, path, on_read=None):
        self.reads = 0
        self.released = False
        self._on_read = on_read

    def isOpened(self):
        return True

    def get(self, prop):
        return {
            scene_detect.cv2.CAP_PROP_FPS: self.FPS,
            scene_detect.cv2.CAP_PROP_FRAME_COUNT: self.FRAMES,
            scene_detect.cv2.CAP_PROP_FRAME_WIDTH: 64,
            scene_detect.cv2.CAP_PROP_FRAME_HEIGHT: 48,
        }[prop]

    def set(self, prop, value):
        return True

    def read(self):
        self.reads += 1
        if self._on_read is not None:
            self._on_read(self.reads)
        return True, np.zeros((48, 64, 3), dtype=np.uint8)

    def release(self):
        self.released = True


class _FakeKaraokeDetector:
    def __init__(self, config=None):
        self._ocr_calls = 0
        self._scene_cuts = []
        self._n = 0

    def _extract_roi(self, frame):
        return frame

    def _extract_text(self, roi):
        # Alternate text so every coarse interval is a "transition".
        self._ocr_calls += 1
        self._n += 1
        return f"line {self._n % 2}"

    def _text_similarity(self, a, b):
        return 1.0 if a == b else 0.0

    def get_stats(self):
        return {"cuts_detected": len(self._scene_cuts)}

    def get_scene_list(self):
        return list(self._scene_cuts)


def test_karaoke_scan_stops_when_cancelled(tmp_path, monkeypatch):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"fake")
    cancel = Event()
    captures = []

    def make_capture(path):
        cap = _FakeCapture(path, on_read=lambda n: cancel.set() if n >= 1 else None)
        captures.append(cap)
        return cap

    monkeypatch.setattr(scene_detect.cv2, "VideoCapture", make_capture)
    monkeypatch.setattr(scene_detect, "KaraokeTextDetector", _FakeKaraokeDetector)
    monkeypatch.setattr(SceneDetector, "_verify_source_timing", staticmethod(lambda source: None))
    records = []

    with pytest.raises(DetectionCancelled):
        SceneDetector().detect_karaoke_scenes_with_progress(
            video,
            lambda value, message: records.append((value, message)),
            KaraokeDetectionConfig(),
            cancel_event=cancel,
        )

    assert len(captures) == 1
    cap = captures[0]
    assert cap.released
    assert cap.reads <= 2
    assert all(value < 0.6 for value, _ in records), records
