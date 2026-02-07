"""Background worker for batch frame extraction and thumbnail generation.

Extracts frames from a video source (optionally scoped to a single clip),
creates Frame model objects, and generates thumbnails for each extracted frame.
"""

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Signal

from models.clip import Clip, Source
from models.frame import Frame
from core.ffmpeg import extract_frames_batch
from core.thumbnail import generate_image_thumbnail
from ui.workers.base import CancellableWorker

logger = logging.getLogger(__name__)


class FrameExtractionWorker(CancellableWorker):
    """Extract frames from a video and build Frame objects with thumbnails.

    Signals:
        progress: ``(current, total)`` during extraction.
        frame_ready: ``(frame_id, thumbnail_path)`` when a frame is processed.
        extraction_completed: Emitted with the full list of :class:`Frame`
            objects once all frames have been extracted and thumbnailed.
        error: Inherited from :class:`CancellableWorker`.
    """

    progress = Signal(int, int)           # current, total
    frame_ready = Signal(str, str)        # frame_id, thumbnail_path
    extraction_completed = Signal(list)   # list[Frame]

    def __init__(
        self,
        source: Source,
        clip: Optional[Clip],
        mode: str,
        interval: int,
        output_dir: Path,
        parent=None,
    ):
        super().__init__(parent)
        self._source = source
        self._clip = clip
        self._mode = mode
        self._interval = interval
        self._output_dir = output_dir

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self):
        self._log_start()

        try:
            frames = self._extract_and_build()
        except Exception as exc:
            self._log_error(str(exc))
            self.error.emit(str(exc))
            self.extraction_completed.emit([])
            return

        if self.is_cancelled():
            self._log_cancelled()
            self.extraction_completed.emit([])
            return

        self.extraction_completed.emit(frames)
        self._log_complete()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_and_build(self) -> list[Frame]:
        """Run FFmpeg extraction, then wrap results as Frame objects."""
        video_path = self._source.file_path
        fps = self._source.fps

        # Determine frame range from clip (if provided)
        start_frame = 0
        end_frame: Optional[int] = None
        clip_id: Optional[str] = None

        if self._clip is not None:
            start_frame = self._clip.start_frame
            end_frame = self._clip.end_frame
            clip_id = self._clip.id

        # Frames output directory (scoped by source + optional clip)
        frames_dir = self._output_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        # Thumbnails output directory
        thumbs_dir = self._output_dir / "thumbnails"
        thumbs_dir.mkdir(parents=True, exist_ok=True)

        # ---- Step 1: extract frames via FFmpeg ----
        def _on_progress(current: int, total: int) -> None:
            self.progress.emit(current, total)

        extracted_paths = extract_frames_batch(
            video_path=video_path,
            output_dir=frames_dir,
            fps=fps,
            mode=self._mode,
            interval=self._interval,
            start_frame=start_frame,
            end_frame=end_frame,
            progress_callback=_on_progress,
            cancel_event=self._cancel_event,
        )

        if self.is_cancelled():
            return []

        # ---- Step 2: build Frame objects + thumbnails ----
        frames: list[Frame] = []
        total = len(extracted_paths)

        for idx, frame_path in enumerate(extracted_paths):
            if self.is_cancelled():
                break

            # Derive frame number from filename (frame_000001.png -> 0-indexed)
            try:
                seq_num = int(frame_path.stem.split("_")[1])
            except (IndexError, ValueError):
                seq_num = idx
            frame_number = start_frame + (seq_num - 1) if seq_num >= 1 else start_frame + idx

            # Generate thumbnail
            thumb_path = thumbs_dir / f"{frame_path.stem}_thumb.jpg"
            try:
                generate_image_thumbnail(frame_path, thumb_path)
            except Exception as exc:
                logger.warning(
                    f"Thumbnail generation failed for {frame_path}: {exc}"
                )
                thumb_path = None

            frame = Frame(
                file_path=frame_path,
                source_id=self._source.id,
                clip_id=clip_id,
                frame_number=frame_number,
                thumbnail_path=thumb_path,
            )

            frames.append(frame)
            if thumb_path is not None:
                self.frame_ready.emit(frame.id, str(thumb_path))
            self.progress.emit(idx + 1, total)

        return frames
