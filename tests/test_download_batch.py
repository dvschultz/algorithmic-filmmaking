"""One scheduler owns bounded dispatch, result ordering, and cancellation."""

from threading import Barrier, Event, Lock, get_ident
from types import SimpleNamespace
import pytest
from core.operations.downloads import DownloadRequest, run_download_batch


def test_parallel_limit_and_ordered_results(monkeypatch):
    lock = Lock()
    barrier = Barrier(2)
    active = maximum = 0
    owner = get_ident()
    callbacks = []
    requests = [DownloadRequest(f"https://youtube.com/{i}") for i in range(6)]

    def execute(request, **kwargs):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        barrier.wait(timeout=3)
        with lock:
            active -= 1
        return SimpleNamespace(success=True)

    monkeypatch.setattr("core.operations.downloads.run_download", execute)
    results = run_download_batch(
        requests,
        max_workers=2,
        item_callback=lambda result: callbacks.append((result.index, get_ident())),
    )
    assert maximum == 2
    assert [r.index for r in results] == list(range(6))
    assert all(r.status == "succeeded" for r in results)
    assert sorted(i for i, _ in callbacks) == list(range(6))
    assert all(thread == owner for _, thread in callbacks)


def test_cancel_from_callback_preserves_success_and_skips_dispatch(monkeypatch):
    cancel = Event()
    calls = []

    def execute(request, **kwargs):
        calls.append(request.url)
        return SimpleNamespace(success=True)

    monkeypatch.setattr("core.operations.downloads.run_download", execute)
    results = run_download_batch(
        [DownloadRequest("https://youtube.com/same")] * 3,
        cancel_event=cancel,
        item_callback=lambda result: cancel.set(),
    )
    assert len(calls) == 1
    assert [r.status for r in results] == ["succeeded", "cancelled", "cancelled"]


def test_failure_does_not_lose_siblings_and_invalid_urls_never_dispatch(monkeypatch):
    def execute(request, **kwargs):
        if request.url.endswith("bad"):
            raise RuntimeError("network failure")
        return SimpleNamespace(success=True)

    monkeypatch.setattr("core.operations.downloads.run_download", execute)
    requests = [
        DownloadRequest(url)
        for url in ["https://youtube.com/bad", "file:///bad", "https://youtube.com/ok"]
    ]
    results = run_download_batch(requests)
    assert [r.status for r in results] == ["failed", "failed", "succeeded"]
    assert [r.error_code for r in results[:2]] == ["download_exception", "invalid_url"]


def test_callback_failure_stops_further_dispatch(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "core.operations.downloads.run_download",
        lambda request, **kwargs: calls.append(request)
        or SimpleNamespace(success=True),
    )

    def fail(result):
        raise RuntimeError("publication failed")

    with pytest.raises(RuntimeError, match="publication failed"):
        run_download_batch(
            [DownloadRequest("https://youtube.com/a")] * 3, item_callback=fail
        )
    assert len(calls) == 1
