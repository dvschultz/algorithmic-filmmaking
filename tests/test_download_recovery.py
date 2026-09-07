"""Download receipts survive job failure and only reuse verified files."""

from threading import Event
from unittest.mock import Mock
import pytest
from core.jobs.store import JobStore
from core.jobs.downloads import run_saved_downloads
from core.downloader import DownloadResult


def test_retry_reuses_success_and_redownloads_missing_file(tmp_path, monkeypatch):
    store = JobStore(tmp_path / "jobs.db")
    target = tmp_path / "downloads"
    calls = []

    def download(request, **kwargs):
        calls.append(request.url)
        path = target / (request.url.rsplit("/", 1)[-1] + ".mp4")
        path.write_bytes(request.url.encode())
        return DownloadResult(success=True, file_path=path, title="video", duration=3)

    monkeypatch.setattr("core.operations.downloads.run_download", download)
    urls = ["https://youtube.com/a", "https://youtube.com/b"]
    assert run_saved_downloads(store, urls, target)["success"]
    reopened = JobStore(store.db_path)
    result = run_saved_downloads(reopened, urls, target)
    assert len(result["result"]["succeeded"]) == 2 and len(calls) == 2
    (target / "a.mp4").unlink()
    run_saved_downloads(reopened, urls, target)
    assert calls == urls + [urls[0]]


def test_progress_failure_keeps_previous_receipt(tmp_path, monkeypatch):
    store = JobStore(tmp_path / "jobs.db")
    target = tmp_path / "downloads"
    calls = []

    def download(request, **kwargs):
        calls.append(request.url)
        path = target / (request.url.rsplit("/", 1)[-1] + ".mp4")
        path.write_bytes(b"video")
        return DownloadResult(success=True, file_path=path)

    monkeypatch.setattr("core.operations.downloads.run_download", download)

    def progress(value, message):
        if value > 0:
            raise RuntimeError("delivery failed")

    urls = ["https://youtube.com/a", "https://youtube.com/b"]
    with pytest.raises(RuntimeError, match="delivery failed"):
        run_saved_downloads(store, urls, target, progress_callback=progress)
    assert calls == [urls[0]]
    run_saved_downloads(store, urls, target)
    assert calls == urls


def test_changed_output_is_preserved_as_conflict(tmp_path, monkeypatch):
    store = JobStore(tmp_path / "jobs.db")
    target = tmp_path / "downloads"
    path = target / "video.mp4"

    def download(request, **kwargs):
        path.write_bytes(b"original")
        return DownloadResult(success=True, file_path=path)

    native = Mock(side_effect=download)
    monkeypatch.setattr("core.operations.downloads.run_download", native)
    urls = ["https://youtube.com/a"]
    run_saved_downloads(store, urls, target)
    path.write_bytes(b"user edit")
    result = run_saved_downloads(store, urls, target)
    assert not result["result"]["succeeded"]
    assert result["result"]["failed"][0]["error_code"] == "download_output_changed"
    assert path.read_bytes() == b"user edit" and native.call_count == 1


def test_receipt_write_failure_preserves_earlier_success(tmp_path, monkeypatch):
    store = JobStore(tmp_path / "jobs.db")
    target = tmp_path / "downloads"
    calls = []

    def download(request, **kwargs):
        calls.append(request.url)
        path = target / (request.url.rsplit("/", 1)[-1] + ".mp4")
        path.write_bytes(b"video")
        return DownloadResult(success=True, file_path=path)

    monkeypatch.setattr("core.operations.downloads.run_download", download)
    record = store.record_download_receipt
    writes = []

    def fail_second(*args):
        writes.append(args)
        if len(writes) == 2:
            raise OSError("database write failed")
        return record(*args)

    monkeypatch.setattr(store, "record_download_receipt", fail_second)
    urls = ["https://youtube.com/a", "https://youtube.com/b"]
    with pytest.raises(OSError, match="database write failed"):
        run_saved_downloads(store, urls, target)
    monkeypatch.setattr(store, "record_download_receipt", record)
    result = run_saved_downloads(store, urls, target)
    assert len(result["result"]["succeeded"]) == 2
    assert calls == urls + [urls[1]]


def test_cancel_while_waiting_for_directory_lock(tmp_path, monkeypatch):
    from threading import Thread
    from core.project_lock import acquire_lock_record

    store = JobStore(tmp_path / "jobs.db")
    target = tmp_path / "downloads"
    target.mkdir()
    stat = target.stat()
    lease = acquire_lock_record(f"download-directory:{(stat.st_dev, stat.st_ino)}")
    attempted, cancel = Event(), Event()

    def acquire(key):
        try:
            return acquire_lock_record(key)
        finally:
            attempted.set()

    monkeypatch.setattr("core.jobs.downloads.acquire_lock_record", acquire)
    native = Mock()
    monkeypatch.setattr("core.operations.downloads.run_download", native)
    results = []
    thread = Thread(
        target=lambda: results.append(
            run_saved_downloads(
                store, ["https://youtube.com/a"], target, cancel_event=cancel
            )
        )
    )
    try:
        thread.start()
        assert attempted.wait(3)
        cancel.set()
        thread.join(3)
        assert not thread.is_alive()
        assert results[0]["result"]["cancelled"] == ["https://youtube.com/a"]
        native.assert_not_called()
    finally:
        cancel.set()
        lease.close()
        thread.join(3)


def test_schema_upgrade_keeps_existing_jobs(tmp_path):
    import sqlite3
    from core.jobs.schema import INITIAL_SCHEMA

    path = tmp_path / "old.db"
    with sqlite3.connect(path) as conn:
        conn.executescript(INITIAL_SCHEMA)
        conn.execute(
            "INSERT INTO jobs(id,kind,status,args_json,created_at,updated_at) VALUES('old','download_videos','completed','{}',0,0)"
        )
    store = JobStore(path)
    assert store.get("old").kind == "download_videos"
    store.record_download_receipt("key", "{}", "{}", "digest")
    assert store.get_download_receipt("key")["payload_json"] == "{}"
