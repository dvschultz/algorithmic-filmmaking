"""Combined analysis owns its reply and never advances a retired run."""

import os
import subprocess
import sys


def test_pipeline_request_and_thread_lifetimes():
    code = r'''
from types import SimpleNamespace
from unittest.mock import Mock
from PySide6.QtCore import QCoreApplication, QObject, QThread, Signal
from core.project import Project
from ui.main_window import MainWindow
from ui.workers.gui_tool_reply import GuiToolReply
from ui.workers.gui_tool_mailbox import GuiToolMailbox
from ui.workers.text_extraction_worker import TextExtractionWorker
assert 'finished' not in TextExtractionWorker.__dict__, 'result signal shadows native thread completion'
from ui.workers.analysis_pipeline_delivery import AnalysisPipelineRun, bind_pipeline_completion
app = QCoreApplication([])
class Window(QObject):
    pass
window = Window()
window.project = Project.new()
mailbox = GuiToolMailbox()
window._chat_worker = SimpleNamespace(_stop_requested=False,
    is_gui_tool_pending=mailbox.is_pending, set_gui_tool_result=mailbox.submit)
reply = GuiToolReply.capture(window, 'analyze_all_live', mailbox.begin('analyze_all_live'))
window._analysis_run = AnalysisPipelineRun(window.project.session.session_id, reply)
window._analysis_completed_ops = []
window._analysis_phase_remaining = 1
window._analysis_current_phase = 'sequential'
window._analysis_sequential_queue = ['describe']
window._analysis_pending_phases = ['cloud']
window._transcription_source_queue = [('source', [])]
window._analysis_clips = []
window._launch_analysis_worker = Mock()
window._start_next_analysis_phase = Mock()
window._gui_state = Mock(); window.analyze_tab = Mock(); window.progress_bar = Mock()
# A timed-out request cannot start the next operation or source.
assert mailbox.wait(0) is None
MainWindow._on_analysis_phase_worker_finished(window, 'transcribe')
assert not window._analysis_completed_ops
window._launch_analysis_worker.assert_not_called()
assert not window._transcription_source_queue
# A replaced pipeline cannot receive old completions even on the same channel.
class Worker(QThread):
    result = Signal()
worker = Worker(); window.worker = worker
window._analysis_run = AnalysisPipelineRun(window.project.session.session_id, None)
seen = []
bind_pipeline_completion(window, worker, 'worker', worker.result, lambda: seen.append('old'))
window._analysis_run = AnalysisPipelineRun(window.project.session.session_id, None)
worker.result.emit()
assert seen == []
worker.finished.emit(); app.processEvents()
# Actual worker results must not shadow native thread completion.
from ui.workers.text_extraction_worker import TextExtractionWorker
from ui.workers.cinematography_worker import CinematographyWorker
for cls, signal in [(TextExtractionWorker, 'extraction_completed'), (CinematographyWorker, 'analysis_completed')]:
    assert 'finished' not in cls.__dict__
    assert signal in cls.__dict__
# Real queued delivery happens on the owner thread; cleanup waits for run() exit.
import threading, time
from shiboken6 import isValid
owner = threading.get_ident()
for cls, result_signal in [(TextExtractionWorker, 'extraction_completed'), (CinematographyWorker, 'analysis_completed')]:
    release = threading.Event()
    class RunningWorker(cls):
        def run(self):
            getattr(self, result_signal).emit({})
            getattr(self, result_signal).emit({})
            assert release.wait(5)
    live = RunningWorker(clips=[], sources_by_id={})
    window.worker = live
    window._analysis_run = AnalysisPipelineRun(window.project.session.session_id, None)
    calls = []
    bind_pipeline_completion(window, live, 'worker', getattr(live, result_signal),
        lambda: calls.append(threading.get_ident()))
    live.start()
    deadline = time.monotonic() + 5
    try:
        while not calls and time.monotonic() < deadline:
            app.processEvents(); time.sleep(.001)
        assert calls == [owner]
        assert live.isRunning() and isValid(live)
        replacement = Worker(); window.worker = replacement
    finally:
        release.set(); assert live.wait(5000)
    app.processEvents()
    assert window.worker is replacement
# Every changed consumer receives the domain result, including Exquisite Corpus.
from unittest.mock import patch
from ui.dialogs.exquisite_corpus_dialog import ExquisiteCorpusDialog
from PySide6.QtCore import Slot
class Corpus(QObject):
    @Slot(int, int, str)
    def _on_extraction_progress(self, *args): pass
    @Slot(str)
    def _on_extraction_error(self, message): pass
    @Slot(dict)
    def _on_extraction_finished(self, results): self.results.append(results)
corpus = Corpus(); corpus.clips = []; corpus.sources_by_id = {}; corpus.results = []
with patch('core.settings.load_settings', return_value=SimpleNamespace(
        text_extraction_method='vlm', text_extraction_vlm_model='test')):
    with patch.object(TextExtractionWorker, 'start', lambda worker: worker.extraction_completed.emit({})):
        ExquisiteCorpusDialog._start_extraction(corpus)
assert corpus.results == [{}]
for cls, op, signal in [(TextExtractionWorker, 'extract_text', 'extraction_completed'),
                         (CinematographyWorker, 'cinematography', 'analysis_completed')]:
    window.settings = SimpleNamespace(description_parallelism=2,
        text_extraction_method='vlm', text_extraction_vlm_model='test')
    window._on_text_extraction_progress = Mock()
    window._on_text_extraction_error = Mock(); window._on_cinematography_progress = Mock()
    window._on_cinematography_clip_ready = Mock(); window._on_cinematography_error = Mock()
    window._on_frame_analysis_op_finished = Mock()
    with patch.object(cls, 'start', lambda worker: getattr(worker, signal).emit({})):
        MainWindow._launch_frame_analysis_worker(window, op, [])
    if op == 'extract_text':
        assert window._frame_text_worker.options.vlm_only is True
        assert window._frame_text_worker.options.vlm_model == 'test'
    window._on_frame_analysis_op_finished.assert_called_once_with(op)
# Empty or filtered dispatch must report failure instead of waiting forever.
harness = SimpleNamespace(project=window.project,
    _filter_available_analysis_operations=lambda _: [], _custom_query_text=None)
assert MainWindow._run_analysis_pipeline(harness, [], ['colors']) is False
assert MainWindow._run_analysis_pipeline(harness, [SimpleNamespace(disabled=False)], ['colors']) is False
# A modal dependency prompt may expire the request before dispatch.
expired = GuiToolReply.capture(window, 'analyze_all_live', mailbox.begin('analyze_all_live'))
harness._dispatch_gui_reply = expired
harness._chat_worker = window._chat_worker
def expire(ops):
    mailbox.wait(0)
    return ops
harness._filter_available_analysis_operations = expire
assert MainWindow._run_analysis_pipeline(harness, [SimpleNamespace(disabled=False)], ['colors']) is False
# Completion answers the captured request, despite unrelated pending IDs.
reply = GuiToolReply.capture(window, 'analyze_all_live', mailbox.begin('analyze_all_live'))
window._analysis_run = AnalysisPipelineRun(window.project.session.session_id, reply)
window._analysis_clips = []; window._analysis_completed_ops = ['colors']
window._pending_agent_tool_call_id = 'unrelated'; window._pending_agent_tool_name = 'other'
window._get_completed_analysis_error_labels = lambda _: []
window.status_bar = Mock(); window.collect_tab = Mock()
window._update_chat_project_state = Mock()
window._build_agent_analysis_result = lambda *args: {'count': 0}
MainWindow._on_analysis_pipeline_complete(window)
result = mailbox.wait(0)
assert result['tool_call_id'] == reply.token and result['name'] == reply.name
assert window._pending_agent_tool_call_id == 'unrelated'
assert window._analysis_run.finished
'''
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"}, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
