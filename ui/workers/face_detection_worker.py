"""Background worker for face detection using InsightFace.

Runs face detection and embedding extraction on clips sequentially,
as the InsightFace model requires video file access for frame sampling.
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.clip import Clip, Source

from PySide6.QtCore import Signal

from ui.workers.base import CancellableWorker

logger = logging.getLogger(__name__)


class FaceDetectionWorker(CancellableWorker):
    """Background worker for face detection and embedding extraction.

    Processes clips sequentially (requires video file access for frame
    sampling). Mutates clip.face_embeddings in-place so partial results
    persist on cancel.

    Signals:
        progress: Emitted with (current, total) during processing
        faces_ready: Emitted with (clip_id, face_embeddings_list)
        detection_completed: Emitted when all clips are processed
        error: Emitted with error message string on failure (inherited)
    """

    progress = Signal(int, int)  # current, total
    faces_ready = Signal(str, list)  # clip_id, face_embeddings
    detection_completed = Signal()

    def __init__(
        self,
        clips: list["Clip"],
        sources_by_id: dict[str, "Source"],
        sample_interval: float = 1.0,
        skip_existing: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        self._clips = clips
        self._sources_by_id = sources_by_id
        self._sample_interval = sample_interval
        self._skip_existing = skip_existing

    def run(self):
        """Execute face detection on all clips."""
        self._log_start()

        from core.analysis.faces import extract_faces_from_clip

        # Filter clips needing processing
        clips_to_process = []
        for clip in self._clips:
            if self._skip_existing and clip.face_embeddings is not None:
                continue
            source = self._sources_by_id.get(clip.source_id)
            if not source or not source.file_path.exists():
                logger.warning(f"Skipping clip {clip.id}: source not found")
                continue
            clips_to_process.append((clip, source))

        total = len(clips_to_process)
        if total == 0:
            logger.info("No clips to process for face detection")
            self.detection_completed.emit()
            self._log_complete()
            return

        logger.info(f"Starting face detection: {total} clips")

        # Pre-load InsightFace model so user sees download status
        try:
            from core.analysis.faces import is_model_loaded, _load_insightface

            if not is_model_loaded():
                self.progress.emit(0, total)
                _load_insightface()
        except Exception as e:
            self.error.emit(f"Failed to load face detection model: {e}")
            self._log_complete()
            return

        if self.is_cancelled():
            self._log_cancelled()
            return

        # Group by source to minimize repeated file opens
        clips_to_process.sort(key=lambda cs: cs[1].file_path.name)

        for i, (clip, source) in enumerate(clips_to_process):
            if self.is_cancelled():
                self._log_cancelled()
                break

            try:
                faces = extract_faces_from_clip(
                    source_path=source.file_path,
                    start_frame=clip.start_frame,
                    end_frame=clip.end_frame,
                    fps=source.fps,
                    sample_interval=self._sample_interval,
                )
                # Mutate in-place (partial results persist on cancel)
                clip.face_embeddings = faces if faces else []
                self.faces_ready.emit(clip.id, faces)
            except Exception as e:
                logger.error(f"Face detection failed for clip {clip.id}: {e}")
                clip.face_embeddings = []

            self.progress.emit(i + 1, total)

        from core.analysis.faces import unload_model
        unload_model()
        self.detection_completed.emit()
        self._log_complete()
