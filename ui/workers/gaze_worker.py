"""Background worker for gaze direction analysis using MediaPipe Face Mesh.

Runs gaze estimation on clips sequentially, as the MediaPipe model
requires video file access for frame sampling.
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.clip import Clip, Source

from PySide6.QtCore import Signal, Slot

from ui.workers.base import CancellableWorker

logger = logging.getLogger(__name__)


class GazeAnalysisWorker(CancellableWorker):
    """Background worker for gaze direction analysis.

    Processes clips sequentially (requires video file access for frame
    sampling). Mutates clip gaze fields in-place so partial results
    persist on cancel.

    Signals:
        progress: Emitted with (current, total) during processing
        gaze_ready: Emitted with (clip_id, yaw, pitch, category) for
            clips where gaze was successfully detected
        detection_completed: Emitted when all clips are processed
        error: Emitted with error message string on failure (inherited)
    """

    progress = Signal(int, int)  # current, total
    gaze_ready = Signal(str, float, float, str)  # clip_id, yaw, pitch, category
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

    @Slot()
    def run(self):
        """Execute gaze analysis on all clips."""
        self._log_start()

        # Filter clips needing processing
        clips_to_process = []
        for clip in self._clips:
            if self._skip_existing and clip.gaze_category is not None:
                continue
            source = self._sources_by_id.get(clip.source_id)
            if not source or not source.file_path.exists():
                logger.warning(f"Skipping clip {clip.id}: source not found")
                continue
            clips_to_process.append((clip, source))

        total = len(clips_to_process)
        if total == 0:
            logger.info("No clips to process for gaze analysis")
            self.detection_completed.emit()
            self._log_complete()
            return

        logger.info(f"Starting gaze analysis: {total} clips")

        # Pre-load MediaPipe FaceMesh model
        try:
            from core.analysis.gaze import is_model_loaded, _load_face_mesh

            if not is_model_loaded():
                self.progress.emit(0, total)
                _load_face_mesh()
        except Exception as e:
            self.error.emit(f"Failed to load gaze detection model: {e}")
            self._log_complete()
            return

        if self.is_cancelled():
            self._log_cancelled()
            return

        # Sort by source_id to minimize repeated file opens
        clips_to_process.sort(key=lambda cs: cs[1].file_path.name)

        from core.analysis.gaze import extract_gaze_from_clip

        for i, (clip, source) in enumerate(clips_to_process):
            if self.is_cancelled():
                self._log_cancelled()
                break

            try:
                result = extract_gaze_from_clip(
                    source_path=str(source.file_path),
                    start_frame=clip.start_frame,
                    end_frame=clip.end_frame,
                    fps=source.fps,
                    sample_interval=self._sample_interval,
                )
                if result is not None:
                    # Mutate in-place (partial results persist on cancel)
                    clip.gaze_yaw = result["gaze_yaw"]
                    clip.gaze_pitch = result["gaze_pitch"]
                    clip.gaze_category = result["gaze_category"]
                    self.gaze_ready.emit(
                        clip.id,
                        result["gaze_yaw"],
                        result["gaze_pitch"],
                        result["gaze_category"],
                    )
                # If result is None: no face found, skip emission
            except Exception as e:
                logger.warning(f"Gaze analysis failed for clip {clip.id}: {e}")

            self.progress.emit(i + 1, total)

        from core.analysis.gaze import unload_model
        unload_model()
        self.detection_completed.emit()
        self._log_complete()
