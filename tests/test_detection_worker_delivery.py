"""Verify the desktop adapter invokes the shared operation on its worker thread."""

from pathlib import Path
import subprocess
import sys


def test_detection_worker_snapshots_settings_and_suppresses_cancelled_result():
    code = '''
import threading
from pathlib import Path
from unittest.mock import patch
from PySide6.QtCore import QCoreApplication, QEventLoop, QTimer
from core.scene_detect import DetectionConfig
from models.clip import Source
from ui.main_window import DetectionWorker

app = QCoreApplication([])
owner = threading.get_ident()
source = Source(file_path=Path('video.mp4'))
for cancelled in (False, True):
    config = DetectionConfig(threshold=4)
    worker = DetectionWorker(Path('video.mp4'), config)
    config.threshold = 9
    loop = QEventLoop()
    received, errors = [], []
    worker.detection_completed.connect(lambda source, clips: received.append(source))
    worker.error.connect(errors.append)
    worker.finished.connect(loop.quit)
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)

    class Detector:
        def __init__(self, config):
            assert threading.get_ident() != owner
            assert config.threshold == 4

        def detect_scenes_with_progress(self, path, progress):
            if cancelled:
                worker.cancel()
            return source, []

    with patch('core.scene_detect.SceneDetector', Detector):
        timer.start(5000)
        worker.start()
        loop.exec()
        assert worker.wait(5000)
    assert received == ([] if cancelled else [source]), received
    assert errors == [], errors
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
