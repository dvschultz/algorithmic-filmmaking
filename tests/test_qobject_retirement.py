"""Native completion must not recursively destroy a retained Qt parent."""

import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("kind", ["agent", "chat", "description", "custom_query"])
def test_delivery_releases_last_window_reference_after_native_exit(kind):
    code = r'''
import gc, sys
from PySide6.QtCore import QObject, QCoreApplication, QEvent, QThread, Signal
from core.project import Project
from ui.workers.gui_tool_reply import AgentAnalysisCompletion
from ui.workers.chat_delivery import ChatDelivery
from ui.workers.description_delivery import DescriptionDelivery
from ui.workers.custom_query_delivery import CustomQueryDelivery

app = QCoreApplication([])
destroyed = []
removed_on_owner_thread = []
kind = sys.argv[1]
class Window(QObject):
    def childEvent(self, event):
        if event.removed():
            removed_on_owner_thread.append(QThread.currentThread() == app.thread())

class Worker(QThread):
    description_ready = Signal(str, str)
    query_result_ready = Signal(str, str, bool, float, str)
    tasks = ()
    def run(self):
        pass

def complete_request():
    window = Window()
    window.project = Project.new()
    window.destroyed.connect(lambda: destroyed.append('window'))
    worker = Worker()
    if kind == 'agent':
        window.worker = worker
        delivery = AgentAnalysisCompletion(window, worker, 'worker', lambda **kw: None)
    elif kind == 'chat':
        window._active_chat_workers = set()
        window._chat_worker = worker
        delivery = ChatDelivery(window, worker, {})
    elif kind == 'description':
        delivery = DescriptionDelivery(window, worker)
    else:
        delivery = CustomQueryDelivery(window, worker)
    assert delivery.parent() is window
    worker.start()
    assert worker.wait(10000)
    app.processEvents()

complete_request()
QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
gc.collect()
app.processEvents()
assert destroyed == ['window'], destroyed
assert removed_on_owner_thread == [True], removed_on_owner_thread
'''
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", code, kind],
        capture_output=True,
        text=True,
        timeout=15,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
