"""Verified file receipts for restart-safe headless download batches."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
from threading import Event
from typing import Callable

from core.jobs.spec import encode_object
from core.jobs.store import JobStore
from core.operations.downloads import (
    DownloadRequest,
    DownloadOutcome,
    run_download_batch,
)
from core.project_lock import acquire_lock_record, LockUnavailableError
from core.spine.downloads import format_download_results


class DownloadOutputChanged(RuntimeError):
    pass


def _stamp(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _fingerprint(path: Path, target: Path, *, sync: bool = False) -> tuple[dict, tuple]:
    resolved = path.resolve()
    if not resolved.is_relative_to(target):
        raise DownloadOutputChanged("Download output is outside its target directory")
    before = _stamp(resolved)
    if not resolved.is_file():
        raise DownloadOutputChanged("Download output is no longer a file")
    digest = sha256()
    with resolved.open("r+b" if sync else "rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
        if sync:
            os.fsync(stream.fileno())
    if _stamp(path) != before or path.resolve() != resolved:
        raise DownloadOutputChanged("Download output changed during verification")
    return {"sha256": digest.hexdigest(), "size": before[2]}, before


def _identity(request: DownloadRequest) -> tuple[str, str]:
    spec = encode_object(
        {
            "kind": "download_file",
            "version": 1,
            "url": request.url,
            "output_dir": str(request.download_dir),
            "resolution": request.resolution,
            "adaptive_timeout": request.adaptive_timeout,
        }
    )
    return sha256(spec.encode()).hexdigest(), spec


def run_saved_downloads(
    store: JobStore,
    urls: list[str] | tuple[str, ...],
    target_dir: Path,
    progress_callback: Callable[[float, str], None] | None = None,
    cancel_event: Event | None = None,
) -> dict:
    """Persist each verified success before progress delivery or later work."""
    from core.downloader import DownloadResult

    if store.persistence != "job_history":
        raise ValueError("Download recovery requires a durable job store")
    target = Path(target_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    directory_identity = _stamp(target)[:2]
    requests = tuple(DownloadRequest(url, target) for url in urls)
    cancel = cancel_event if cancel_event is not None else Event()
    lease = None
    while not cancel.is_set():
        try:
            lease = acquire_lock_record(f"download-directory:{directory_identity}")
            break
        except LockUnavailableError:
            cancel.wait(0.05)
    if lease is None:
        return format_download_results(
            tuple(DownloadOutcome(i, r, "cancelled") for i, r in enumerate(requests)),
            target,
        )
    try:

        def validate_directory() -> None:
            if _stamp(target)[:2] != directory_identity:
                raise DownloadOutputChanged("Download directory was replaced")

        validate_directory()
        cached = {}
        verified = {}
        for index, request in enumerate(requests):
            if cancel.is_set():
                break
            request_id, spec = _identity(request)
            receipt = store.get_download_receipt(request_id)
            if receipt is None:
                continue
            payload_json = receipt["payload_json"]
            if (
                receipt["spec_json"] != spec
                or sha256(payload_json.encode()).hexdigest()
                != receipt["payload_digest"]
            ):
                raise ValueError("Download receipt is corrupt")
            payload = json.loads(payload_json)
            result = DownloadResult(success=True, **payload["result"])
            if result.file_path is None:
                raise ValueError("Download receipt has no output file")
            path = Path(result.file_path)
            result.file_path = path
            try:
                fingerprint, stamp = _fingerprint(path, target)
                if fingerprint != payload["file"]:
                    raise DownloadOutputChanged(
                        "Previously downloaded file was modified"
                    )
                cached[index] = DownloadOutcome(
                    index, request, "succeeded", result=result
                )
                verified[index] = stamp
            except FileNotFoundError:
                continue
            except DownloadOutputChanged as exc:
                cached[index] = DownloadOutcome(
                    index,
                    request,
                    "failed",
                    error_code="download_output_changed",
                    error_message=str(exc),
                )

        completed = 0

        def on_item(outcome: DownloadOutcome) -> None:
            nonlocal completed
            validate_directory()
            if outcome.status == "succeeded":
                result = outcome.result
                if result is None or result.file_path is None:
                    raise DownloadOutputChanged(
                        "Successful download has no output file"
                    )
                path = Path(result.file_path)
                if outcome.index in verified:
                    if _stamp(path) != verified[outcome.index]:
                        raise DownloadOutputChanged(
                            "Cached download changed before delivery"
                        )
                else:
                    fingerprint, _ = _fingerprint(path, target, sync=True)
                    payload = encode_object(
                        {
                            "result": {
                                "file_path": str(path.resolve()),
                                "title": result.title,
                                "duration": result.duration,
                            },
                            "file": fingerprint,
                        }
                    )
                    request_id, spec = _identity(outcome.request)
                    store.record_download_receipt(
                        request_id, spec, payload, sha256(payload.encode()).hexdigest()
                    )
            completed += 1
            if progress_callback is not None:
                progress_callback(
                    completed / max(len(requests), 1),
                    f"Processed {completed}/{len(requests)} downloads",
                )

        if progress_callback is not None:
            progress_callback(0.0, f"Starting {len(requests)} downloads")
        outcomes = run_download_batch(
            requests, cancel_event=cancel, cached_outcomes=cached, item_callback=on_item
        )
        return format_download_results(outcomes, target)
    finally:
        lease.close()
