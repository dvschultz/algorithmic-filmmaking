"""Background worker for rich cinematography analysis.

Runs cinematography analysis on multiple clips in a background thread,
emitting progress signals to keep the UI responsive.
"""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from PySide6.QtCore import QThread, Signal

from core.analysis.cinematography import analyze_cinematography
from models.cinematography import CinematographyAnalysis

logger = logging.getLogger(__name__)


class CinematographyWorker(QThread):
    """Analyze cinematography for multiple clips in background.

    Processes clips with light parallelism (configurable), running
    VLM-based cinematography analysis and emitting progress signals.

    Signals:
        progress: Emitted with (current, total, clip_id) during processing
        clip_completed: Emitted with (clip_id, CinematographyAnalysis) when a clip finishes
        finished: Emitted with {clip_id: CinematographyAnalysis} when all complete
        error: Emitted with error message string on failure
    """

    # Signals
    progress = Signal(int, int, str)  # current, total, clip_id
    clip_completed = Signal(str, object)  # clip_id, CinematographyAnalysis
    finished = Signal(dict)  # {clip_id: CinematographyAnalysis}
    error = Signal(str)  # error message

    def __init__(
        self,
        clips: list,
        sources_by_id: dict,
        mode: Optional[str] = None,
        model: Optional[str] = None,
        parallelism: int = 2,
        skip_existing: bool = True,
        parent=None,
    ):
        """Initialize the cinematography analysis worker.

        Args:
            clips: List of Clip objects to process
            sources_by_id: Dict mapping source_id to Source objects
            mode: Input mode ("frame" or "video"). If None, uses settings default.
            model: VLM model to use (default: from settings)
            parallelism: Number of concurrent VLM requests (1-5, default: 2)
            skip_existing: Skip clips that already have cinematography data
            parent: Optional parent QObject
        """
        super().__init__(parent)
        self.clips = clips
        self.sources_by_id = sources_by_id
        self.mode = mode
        self.model = model
        self.parallelism = min(max(1, parallelism), 5)
        self.skip_existing = skip_existing
        self._cancel_event = threading.Event()

    def cancel(self):
        """Request cancellation of the analysis (thread-safe)."""
        logger.info("Cinematography analysis cancellation requested")
        self._cancel_event.set()

    def _is_cancelled(self) -> bool:
        """Thread-safe check for cancellation."""
        return self._cancel_event.is_set()

    def _analyze_single_clip(self, clip) -> tuple[str, Optional[CinematographyAnalysis], Optional[str]]:
        """Analyze a single clip.

        Args:
            clip: Clip object to analyze

        Returns:
            Tuple of (clip_id, analysis_result, error_message)
            error_message is None on success
        """
        if self._is_cancelled():
            return clip.id, None, "Cancelled"

        source = self.sources_by_id.get(clip.source_id)
        if not source:
            return clip.id, None, f"Source not found for clip {clip.id}"

        if not clip.thumbnail_path or not clip.thumbnail_path.exists():
            return clip.id, None, f"Thumbnail not found for clip {clip.id}"

        # Check cancellation before expensive API call
        if self._is_cancelled():
            return clip.id, None, "Cancelled"

        try:
            analysis = analyze_cinematography(
                thumbnail_path=clip.thumbnail_path,
                source_path=source.file_path if source.file_path.exists() else None,
                start_frame=clip.start_frame,
                end_frame=clip.end_frame,
                fps=source.fps,
                mode=self.mode,
                model=self.model,
            )
            return clip.id, analysis, None
        except Exception as e:
            return clip.id, None, str(e)

    def run(self):
        """Execute cinematography analysis on all clips.

        Uses light parallelism to process multiple clips concurrently
        while respecting rate limits.
        """
        results = {}

        # Filter clips before submitting to avoid wasted work
        if self.skip_existing:
            clips_to_process = [c for c in self.clips if c.cinematography is None]
            skipped_count = len(self.clips) - len(clips_to_process)
            if skipped_count > 0:
                logger.info(f"Skipping {skipped_count} clips that already have cinematography data")
        else:
            clips_to_process = self.clips

        total = len(clips_to_process)
        completed = 0

        if total == 0:
            logger.info("No clips to process for cinematography analysis")
            self.finished.emit({})
            return

        logger.info(f"Starting cinematography analysis for {total} clips (parallelism={self.parallelism})")

        # Use ThreadPoolExecutor for light parallelism
        with ThreadPoolExecutor(max_workers=self.parallelism) as executor:
            # Submit all tasks
            future_to_clip = {
                executor.submit(self._analyze_single_clip, clip): clip
                for clip in clips_to_process
            }

            # Process results as they complete
            for future in as_completed(future_to_clip):
                if self._is_cancelled():
                    logger.info("Cinematography analysis cancelled")
                    # Cancel remaining futures (won't stop running tasks but prevents new ones)
                    for f in future_to_clip:
                        f.cancel()
                    break

                clip = future_to_clip[future]
                try:
                    clip_id, analysis, error_msg = future.result()

                    if error_msg:
                        if error_msg != "Cancelled":
                            logger.warning(f"Cinematography analysis failed for {clip_id}: {error_msg}")
                            self.error.emit(f"Error analyzing clip {clip_id}: {error_msg}")
                        results[clip_id] = None
                    else:
                        results[clip_id] = analysis
                        if analysis:
                            self.clip_completed.emit(clip_id, analysis)
                            logger.debug(
                                f"Cinematography for {clip_id}: {analysis.shot_size}, "
                                f"{analysis.camera_angle}"
                            )

                except Exception as e:
                    logger.error(f"Unexpected error processing {clip.id}: {e}")
                    self.error.emit(f"Error analyzing clip {clip.id}: {e}")
                    results[clip.id] = None

                completed += 1
                self.progress.emit(completed, total, clip.id)

        if not self._is_cancelled():
            # Filter out None values for the finished signal
            successful_results = {k: v for k, v in results.items() if v is not None}
            logger.info(
                f"Cinematography analysis complete: {len(successful_results)}/{total} clips processed"
            )
            self.finished.emit(successful_results)
