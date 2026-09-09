"""Project artifact verification must leave the GUI event loop responsive."""

import os
import subprocess
import sys


def test_native_project_load_hydrates_off_thread_and_guards_installation():
    code = r'''
import json
import sqlite3
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock, patch
from PySide6.QtCore import QThread, QTimer
from PySide6.QtWidgets import QApplication, QWidget
from core.artifacts import ArtifactStore
from core.project import Project
from core.project_lock import ProjectWriter
from models.clip import Source
from ui.main_window import MainWindow
from ui.workers.project_artifact_loader import ArtifactLoadDialog
app = QApplication([])
class StopAfterLoad(Exception): pass
with TemporaryDirectory() as directory:
    root = Path(directory)
    with patch('core.paths.get_artifact_store_dir', return_value=root/'artifacts'):
        store = ArtifactStore()
        source = root/'source.mp4'; source.write_bytes(b'source')
        project = Project(sources=[Source(file_path=source, fps=30)])
        path = root/'project.json'; project.save(path); project.session.close()
        with store.pin() as pin:
            ref = store.put_bytes(b'cached-render', pin=pin, media_type='video/mp4')
            document = json.loads(path.read_text())
            document['custom_data'] = {'artifact': ref.to_dict()}
            path.write_text(json.dumps(document))
            for mode in ('current', 'cancel', 'close', 'edit', 'replace', 'failure', 'start'):
                window = QWidget()
                window.project = Project.new()
                original = window.project
                window.status_bar = SimpleNamespace(showMessage=Mock())
                window._clear_project_state = Mock(side_effect=StopAfterLoad)
                window._check_unsaved_changes = lambda: True
                reading, release = Event(), Event()
                ticks=[]; calls=[]
                verify = ArtifactStore._verify
                def observed(self, db, value):
                    assert QThread.currentThread() != app.thread(), 'artifact hash ran on GUI thread'
                    calls.append(value.digest); reading.set()
                    assert release.wait(5), 'GUI did not process its timer'
                    if mode == 'failure': raise sqlite3.OperationalError('artifact storage unavailable')
                    return verify(self, db, value)
                def heartbeat():
                    if not reading.is_set(): return
                    ticks.append(1)
                    if len(ticks) == 3:
                        if mode == 'cancel':
                            window.findChild(ArtifactLoadDialog).cancel_button.click()
                        elif mode == 'close':
                            event = Mock()
                            MainWindow.closeEvent(window, event)
                            event.ignore.assert_called_once()
                        elif mode == 'edit': original.mark_dirty()
                        elif mode == 'replace': window.project = Project.new()
                        release.set()
                timer = QTimer(); timer.setInterval(5); timer.timeout.connect(heartbeat); timer.start()
                start_patch = patch('ui.workers.project_artifact_loader.ProjectArtifactWorker.start', side_effect=RuntimeError('could not start worker')) if mode == 'start' else nullcontext()
                with patch.object(ArtifactStore, '_verify', observed), patch('ui.main_window.QMessageBox.warning') as warning, start_patch:
                    try: MainWindow._load_project_file(window, path)
                    except StopAfterLoad: pass
                    finally: timer.stop(); release.set()
                assert warning.called == (mode in ('failure', 'start')), mode
                assert (calls and len(ticks) >= 3) if mode != 'start' else not calls, mode
                assert window._clear_project_state.called == (mode == 'current'), mode
                assert not window._active_project_loads
                assert window.project is original if mode != 'replace' else window.project is not original
                with ProjectWriter(path): pass
                window.project.session.close()
                if original is not window.project: original.session.close()
                window.deleteLater(); app.processEvents()
'''
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=40,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen", "HF_HUB_OFFLINE": "1"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
