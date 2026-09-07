"""Detection job metadata and JSON results shared by runtime adapters."""

import json
from threading import Event
from typing import Callable

from core.jobs.spec import OperationSpec
from core.operations.detection import (
    DetectionGuard,
    DetectionRequest,
    run_detection,
)


def detection_job_spec(
    request: DetectionRequest, guard: DetectionGuard | None = None
) -> OperationSpec:
    """Describe live-project work without claiming restart-safe publication."""
    return OperationSpec.build(
        kind="detect_scenes",
        version=1,
        arguments={
            "mode": request.mode,
            "config": json.loads(request.config_json),
            "karaoke_config": json.loads(request.karaoke_config_json),
        },
        inputs={
            "video_path": str(request.video_path),
            "media_stamp": request.media_stamp,
            "source_id": guard.source_id if guard else None,
            "target_digest": guard.target_digest if guard else None,
        },
        persistence="session_only",
        session_id=guard.session_id if guard else None,
    )


def run_detection_job(
    request: DetectionRequest,
    progress: Callable[[float, str], None],
    cancel: Event,
) -> dict:
    """Return a serializable result; let the runtime own terminal state."""
    try:
        source, clips = run_detection(
            request, progress_callback=progress, cancel_event=cancel
        )
    except Exception:
        # Native backends may report their own exception while stopping.
        # An accepted cancellation must retain the cancelled terminal state.
        if not cancel.is_set():
            raise
        return {"cancelled": True}
    return {
        "success": True,
        "source": source.to_dict(),
        "clips": [clip.to_dict() for clip in clips],
    }
