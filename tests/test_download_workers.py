"""Qt adapters snapshot inputs and retain shared scheduler outcomes."""

import subprocess
import sys
from pathlib import Path


def test_workers_snapshot_inputs_and_keep_result_contract():
    code = """
from pathlib import Path
from types import SimpleNamespace
from PySide6.QtCore import QCoreApplication
from core.downloader import DownloadResult
from ui.workers.download_workers import URLBulkDownloadWorker, BulkDownloadWorker
import core.operations.downloads as operation
app = QCoreApplication([])
seen = []
def download(request, **kwargs):
    seen.append(request)
    if request.url.endswith('bad'):
        return DownloadResult(success=False, error='provider failure')
    return DownloadResult(success=True, file_path=Path('video.mp4'), title=request.url, duration=3)
operation.run_download = download
urls = ['https://youtube.com/first', 'https://youtube.com/bad', 'https://youtube.com/first']
worker = URLBulkDownloadWorker(urls, Path('.'))
urls[0] = 'https://youtube.com/changed'
results, ready = [], []
worker.all_finished.connect(results.append)
worker.video_finished.connect(lambda url, result: ready.append(url))
worker.run()
assert [r['url'] for r in results[0]] == ['https://youtube.com/first', 'https://youtube.com/bad', 'https://youtube.com/first']
assert [r['success'] for r in results[0]] == [True, False, True]
assert len(ready) == 2
assert all(r.adaptive_timeout for r in seen)
video = SimpleNamespace(video_id='original-id', youtube_url='https://youtube.com/bad')
worker = BulkDownloadWorker([video], Path('.'))
video.video_id = 'changed-id'; video.youtube_url = 'https://youtube.com/changed'
errors, finished = [], []
worker.video_error.connect(lambda key, error: errors.append((key, error)))
worker.all_finished.connect(lambda: finished.append(True))
worker.run()
assert errors == [('original-id', 'provider failure')] and finished
assert not seen[-1].adaptive_timeout
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
