"""Export input and reply identity must remain stable after dispatch."""

import os
from pathlib import Path
import subprocess
import sys


def test_export_workers_snapshot_models_and_deliver_original_request():
    code = """
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from PySide6.QtCore import QCoreApplication
from core.project import Project, ProjectMetadata
from models.sequence import Sequence
from models.clip import Source
from ui.main_window import MainWindow, SequenceExportWorker
from ui.workers.export_worker import ExportBundleWorker
from ui.workers.gui_tool_reply import GuiToolReply
app = QCoreApplication([])
sequence = SimpleNamespace(name='original', get_all_clips=lambda: [])
config = SimpleNamespace(output_path=Path('original.mp4'))
source = Source(file_path=Path('original.mp4'))
worker = SequenceExportWorker(sequence, {'source': source}, {}, config)
sequence.name = 'edited'; source.fps = 60; config.output_path = Path('edited.mp4')
assert worker.sequence.name == 'original'
assert worker.sources['source'].fps == 30
assert worker.config.output_path == Path('original.mp4')
project = Project(metadata=ProjectMetadata(name='original'),
    sequences=[Sequence(name='first'), Sequence(name='second')], active_sequence_index=1)
project.add_source(Source(file_path=Path('bundle.mp4')))
bundle = ExportBundleWorker(project, Path('bundle'))
project.sequence.name = 'edited sequence'
project.clear()
seen = []
def export(**kwargs):
    snapshot = kwargs['project']
    seen.append((snapshot.metadata.name, [s.file_path.name for s in snapshot.sources]))
    assert [sequence.name for sequence in snapshot.sequences] == ['first', 'second']
    assert snapshot.active_sequence_index == 1
    return SimpleNamespace(dest_dir=Path('bundle'))
with patch('core.project_export.export_project_bundle', side_effect=export): bundle.run()
assert seen == [('original', ['bundle.mp4'])]
chat = Mock(_stop_requested=False)
window = SimpleNamespace(project=Project.new(), _chat_worker=chat, progress_bar=Mock(),
    sequence_tab=Mock(), status_bar=Mock(), _gui_state=Mock(),
    _pending_agent_tool_call_id='other', _pending_agent_tool_name='analyze_all')
reply = GuiToolReply.capture(window, 'export_sequence', 'original-token')
with patch('ui.main_window.QDesktopServices.openUrl') as open_folder:
    MainWindow._on_sequence_export_finished(window, Path('original.mp4'), reply=reply, clip_count=3)
    open_folder.assert_not_called()
response = chat.set_gui_tool_result.call_args.args[0]
assert response['tool_call_id'] == 'original-token' and response['result']['clip_count'] == 3
assert window._pending_agent_tool_call_id == 'other'
with patch('ui.main_window.QMessageBox.critical') as dialog:
    MainWindow._on_sequence_export_error(window, 'failed', reply=reply)
    dialog.assert_not_called()
assert chat.set_gui_tool_result.call_args.args[0]['error'] == 'failed'
bundle_result = SimpleNamespace(dest_dir=Path('bundle'), sources_copied=1, clips_exported=2,
    frames_copied=0, sources_skipped=[], clips_skipped=0, frames_skipped=[], include_clips=True)
bundle_reply = GuiToolReply.capture(window, 'export_bundle', 'bundle-token')
with patch('ui.main_window.QMessageBox.information') as dialog, patch('ui.main_window.QDesktopServices.openUrl') as folder:
    MainWindow._on_export_bundle_finished(window, bundle_result, reply=bundle_reply)
    dialog.assert_not_called(); folder.assert_not_called()
assert chat.set_gui_tool_result.call_args.args[0]['tool_call_id'] == 'bundle-token'
with patch('ui.main_window.QMessageBox.warning') as dialog:
    MainWindow._on_export_bundle_error(window, 'failed bundle', reply=bundle_reply)
    dialog.assert_not_called()
assert chat.set_gui_tool_result.call_args.args[0]['error'] == 'failed bundle'
new_chat = Mock(_stop_requested=False)
window._chat_worker = new_chat
with patch('ui.main_window.QMessageBox.information') as dialog, patch('ui.main_window.QDesktopServices.openUrl') as folder:
    MainWindow._on_export_bundle_finished(window, bundle_result, reply=bundle_reply)
    dialog.assert_not_called(); folder.assert_not_called()
new_chat.set_gui_tool_result.assert_not_called()
with patch('ui.main_window.QMessageBox.information') as dialog, patch('ui.main_window.QDesktopServices.openUrl') as folder:
    MainWindow._on_sequence_export_finished(window, Path('manual.mp4'))
    MainWindow._on_export_bundle_finished(window, bundle_result)
    assert folder.call_count == 2 and dialog.call_count == 1
# Both export channels deliver on the owner thread and retain replacements.
import threading
from PySide6.QtCore import QObject, QThread, Signal
from ui.workers.export_delivery import ExportDelivery
owner_thread = threading.get_ident()
class SequenceWorker(QThread):
    progress = Signal(float, str)
    export_completed = Signal(object)
    error = Signal(str)
    def run(self):
        self.progress.emit(1., 'done')
        self.export_completed.emit('done'); self.export_completed.emit('duplicate')
        self.error.emit('late')
class BundleWorker(SequenceWorker):
    progress = Signal(int, int, str)
    def run(self):
        self.progress.emit(1, 1, 'file')
        self.export_completed.emit('done'); self.export_completed.emit('duplicate')
        self.error.emit('late')
for attribute, kind in [('export_worker', SequenceWorker), ('export_bundle_worker', BundleWorker)]:
    owner = QObject(); owner.project = Project.new(); received = []
    def bind():
        worker = kind()
        setattr(owner, attribute, worker)
        ExportDelivery(owner, attribute, worker, {
            'progress': lambda *args: received.append(('progress', threading.get_ident())),
            'result': lambda value: received.append((value, threading.get_ident())),
            'error': lambda value: received.append((value, threading.get_ident())),
        })
        worker.start(); assert worker.wait(3000)
        return worker
    old = bind(); current = bind()
    for _ in range(10): app.processEvents()
    assert received == [('progress', owner_thread), ('done', owner_thread)]
    assert getattr(owner, attribute) is None
    received.clear(); bind(); owner.project.clear()
    for _ in range(10): app.processEvents()
    assert not received and getattr(owner, attribute) is None
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=25,
    )
    assert result.returncode == 0, result.stdout + result.stderr
