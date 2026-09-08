"""Exercise actual queued Qt delivery without sharing native test state."""

import subprocess
import sys
from pathlib import Path


def test_color_batch_commits_on_project_thread_before_completion(tmp_path):
    code = """
import sys
import threading
from pathlib import Path
from unittest.mock import patch
from PySide6.QtCore import QCoreApplication, QEventLoop, QObject, QTimer, Slot
from tests.test_spine_analyze import _build_project
from ui.main_window import MainWindow
from ui.workers.color_worker import ColorAnalysisWorker

app = QCoreApplication([])
loop = QEventLoop()
project = _build_project(Path(sys.argv[1]), 2)
project.mark_clean()
owner = threading.get_ident()
received = []

class Receiver(QObject):
    def __init__(self):
        super().__init__()
        self.project = project
        self._title_build_suffix = 'test'
        self.current_project_path = None
        self.current_source = None
        self.title = ''
        self._update_window_title()

    _is_dirty = property(lambda self: self.project.is_dirty)
    _update_window_title = MainWindow._update_window_title

    def setWindowTitle(self, title):
        self.title = title

    @Slot(object, object)
    def apply(self, application, result):
        received.append(('result', threading.get_ident()))
        MainWindow._on_color_result(self, application, result)

    @Slot()
    def completed(self):
        received.append(('completed', threading.get_ident()))
        assert all(c.dominant_colors == [(1, 2, 3)] for c in project.clips)

receiver = Receiver()
worker = ColorAnalysisWorker(project.clips, sources_by_id=project.sources_by_id, project=project)
assert worker.operation.session_id == project.session.session_id
assert worker.operation.input_revision == str(project.mutation_generation)
assert worker.operation.arguments['num_colors'] == 5
started = []
worker.job_started.connect(lambda task, persistence: started.append((task, persistence)))
worker.result_ready.connect(receiver.apply)
worker.analysis_completed.connect(receiver.completed)
worker.finished.connect(loop.quit)
QTimer.singleShot(5000, loop.quit)
with patch('core.analysis.color.extract_dominant_colors', return_value=[(1, 2, 3)]):
    worker.start()
    loop.exec()
    assert worker.wait(5000)
assert received == [('result', owner), ('completed', owner)], received
assert started == [(worker.task_id, 'session_only')], started
assert worker.job_status == 'completed'
assert all(c.dominant_colors == [(1, 2, 3)] for c in project.clips)
assert project.is_dirty
assert receiver.title.endswith('*'), receiver.title

# A failed rerun is delivered on the owner thread and marks the record dirty,
# while retaining the last palette for display.
project.mark_clean()
receiver._update_window_title()
received.clear()
worker = ColorAnalysisWorker(project.clips, sources_by_id=project.sources_by_id,
                             project=project, skip_existing=False)
worker.result_ready.connect(receiver.apply)
worker.analysis_completed.connect(receiver.completed)
worker.finished.connect(loop.quit)
with patch('core.analysis.color.extract_dominant_colors', return_value=[]):
    worker.start()
    loop.exec()
    assert worker.wait(5000)
assert received == [('result', owner), ('completed', owner)], received
assert all(c.analysis_records['colors'].state == 'failed' for c in project.clips)
assert project.is_dirty
assert receiver.title.endswith('*'), receiver.title
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_cancelled_desktop_color_job_keeps_partial_results(tmp_path):
    from unittest.mock import patch
    from tests.test_spine_analyze import _build_project
    from ui.workers.color_worker import ColorAnalysisWorker

    project = _build_project(tmp_path, 3)
    worker = ColorAnalysisWorker(
        project.clips,
        parallelism=1,
        sources_by_id=project.sources_by_id,
        project=project,
    )

    def extract(**kwargs):
        worker.cancel()
        return [(1, 2, 3)]

    with patch("core.analysis.color.extract_dominant_colors", side_effect=extract):
        worker.run()
    assert worker.job_status == "cancelled"
    assert worker.result.outcomes[0].status == "succeeded"
    assert [o.status for o in worker.result.outcomes[1:]] == [
        "unprocessed",
        "unprocessed",
    ]
    assert all(c.dominant_colors is None for c in project.clips)
    worker.application.apply(worker.result)
    assert project.clips[0].dominant_colors == [(1, 2, 3)]


def test_cancel_before_start_skips_desktop_color_computation(tmp_path):
    from unittest.mock import patch
    from tests.test_spine_analyze import _build_project
    from ui.workers.color_worker import ColorAnalysisWorker

    project = _build_project(tmp_path, 1)
    worker = ColorAnalysisWorker(project.clips, sources_by_id=project.sources_by_id)
    worker.cancel()
    with patch("core.analysis.color.extract_dominant_colors") as extract:
        worker.run()
    extract.assert_not_called()
    assert worker.job_status == "cancelled"
    assert worker.result.outcomes[0].status == "unprocessed"
