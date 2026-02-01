"""Background worker for OCR text extraction.

Runs text extraction on multiple clips in a background thread,
emitting progress signals to keep the UI responsive.
"""

import logging
from typing import Optional

from PySide6.QtCore import Signal

from ui.workers.base import CancellableWorker

logger = logging.getLogger(__name__)


class TextExtractionWorker(CancellableWorker):
    """Extract text from multiple clips in background.

    Processes clips sequentially, running OCR on keyframes of each clip
    and emitting progress signals for UI updates.

    Signals:
        progress: Emitted with (current, total, clip_id) during processing
        clip_completed: Emitted with (clip_id, extracted_texts) when a clip finishes
        finished: Emitted with {clip_id: [ExtractedText, ...]} when all complete
        error: Emitted with error message string on failure (inherited)
    """

    progress = Signal(int, int, str)  # current, total, clip_id
    clip_completed = Signal(str, list)  # clip_id, extracted_texts
    finished = Signal(dict)  # {clip_id: [ExtractedText, ...]}

    def __init__(
        self,
        clips: list,
        sources_by_id: dict,
        num_keyframes: int = 3,
        use_vlm_fallback: bool = True,
        vlm_model: Optional[str] = None,
        vlm_only: bool = False,
        use_text_detection: bool = True,
        parent=None,
    ):
        """Initialize the text extraction worker.

        Args:
            clips: List of Clip objects to process
            sources_by_id: Dict mapping source_id to Source objects
            num_keyframes: Number of frames to sample per clip (1-5)
            use_vlm_fallback: Whether to use VLM for low-confidence results
            vlm_model: VLM model to use (default: from settings)
            vlm_only: If True, skip Tesseract and only use VLM
            use_text_detection: Whether to use EAST pre-filter (default: True)
            parent: Optional parent QObject
        """
        super().__init__(parent)
        self.clips = clips
        self.sources_by_id = sources_by_id
        self.num_keyframes = min(max(1, num_keyframes), 5)
        self.use_vlm_fallback = use_vlm_fallback
        self.vlm_model = vlm_model
        self.vlm_only = vlm_only
        self.use_text_detection = use_text_detection

    def run(self):
        """Execute text extraction on all clips.

        This runs in a background thread. Results are communicated
        via signals to the main thread.
        """
        from core.analysis.ocr import extract_text_from_clip

        self._log_start()

        results = {}
        total = len(self.clips)

        logger.info(f"Starting text extraction for {total} clips")

        for i, clip in enumerate(self.clips):
            if self.is_cancelled():
                self._log_cancelled()
                break

            source = self.sources_by_id.get(clip.source_id)
            if not source:
                logger.warning(f"Source not found for clip {clip.id}")
                continue

            self.progress.emit(i + 1, total, clip.id)

            try:
                extracted = extract_text_from_clip(
                    clip=clip,
                    source=source,
                    num_keyframes=self.num_keyframes,
                    use_vlm_fallback=self.use_vlm_fallback,
                    vlm_model=self.vlm_model,
                    vlm_only=self.vlm_only,
                    use_text_detection=self.use_text_detection,
                )
                results[clip.id] = extracted
                self.clip_completed.emit(clip.id, extracted)

                logger.debug(f"Extracted {len(extracted)} text segments from clip {clip.id}")

            except Exception as e:
                self._log_error(str(e), clip.id)
                self.error.emit(f"Error extracting text from clip {clip.id}: {e}")
                # Continue with next clip rather than failing entirely
                results[clip.id] = []

        if not self.is_cancelled():
            logger.info(f"Text extraction complete: {len(results)} clips processed")
            self.finished.emit(results)
            self._log_complete()
