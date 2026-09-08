"""Deferred grid updates must end with their native browser widget."""

import os
import subprocess
import sys


def test_rebuild_coalesces_and_cannot_outlive_browser():
    code = r'''
import gc
from PySide6.QtWidgets import QApplication
import shiboken6
from ui.clip_browser import ClipBrowser
app = QApplication([])
calls = []
class Browser(ClipBrowser):
    def _do_rebuild_grid(self):
        calls.append(shiboken6.isValid(self))
        if shiboken6.isValid(self):
            super()._do_rebuild_grid()
browser = Browser()
browser._rebuild_grid()
browser._rebuild_grid()
app.processEvents()
assert calls == [True], calls
shiboken6.delete(browser)
calls.clear()
for _ in range(5):
    browser = Browser()
    browser._rebuild_grid()
    shiboken6.delete(browser)
    del browser
    gc.collect()
    app.processEvents()
assert not calls, 'A rebuild ran after native browser deletion'
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=15,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
