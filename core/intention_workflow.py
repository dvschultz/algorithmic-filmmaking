"""Intention-First Workflow Coordinator.

Orchestrates the multi-step flow when a user clicks a sequence card with no clips:
Import -> Download (if URLs) -> Detect scenes -> Generate thumbnails -> Analyze (if needed) -> Build sequence
"""

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Signal

logger = logging.getLogger(__name__)


class WorkflowState(Enum):
    """States of the intention-first workflow."""

    IDLE = auto()
    DOWNLOADING = auto()
    DETECTING = auto()
    THUMBNAILS = auto()
    ANALYZING = auto()
    BUILDING = auto()
    COMPLETE = auto()
    CANCELLED = auto()
    ERROR = auto()


@dataclass
class WorkflowProgress:
    """Progress information for the current workflow step."""

    state: WorkflowState
    current_step: int  # 1-based step number
    total_steps: int
    step_progress: float  # 0.0 to 1.0
    message: str
    sources_processed: int = 0
    sources_total: int = 0
    clips_created: int = 0


@dataclass
class WorkflowResult:
    """Result of the workflow execution."""

    success: bool
    algorithm: str
    clips_created: int
    sources_processed: int
    sources_failed: int
    error_message: Optional[str] = None
    failed_sources: list = field(default_factory=list)


class IntentionWorkflowCoordinator(QObject):
    """Coordinates the intention-first workflow for sequence creation.

    This class orchestrates the multi-step flow when a user clicks a sequence
    card but has no clips. It manages:
    1. URL downloads (if any URLs provided)
    2. Scene detection for each source
    3. Thumbnail generation
    4. Analysis (if required by the algorithm, e.g., colors for Color sequence)
    5. Sequence building

    The coordinator uses guard flags to prevent duplicate signal handling
    (a documented gotcha from docs/solutions/).
    """

    # Signals for progress updates
    progress_updated = Signal(object)  # WorkflowProgress
    step_started = Signal(str, int, int)  # step_name, current, total
    step_completed = Signal(str)  # step_name
    step_skipped = Signal(str)  # step_name
    workflow_completed = Signal(object)  # WorkflowResult
    workflow_cancelled = Signal()
    workflow_error = Signal(str)  # error message

    # Analysis requirements by algorithm
    ANALYSIS_REQUIREMENTS = {
        "color": ["colors"],  # Needs color analysis
        "duration": [],  # Duration comes from detection
        "shuffle": [],  # No analysis needed
        "sequential": [],  # No analysis needed
        "shot_type": ["shot_type"],  # Needs shot type analysis
        "exquisite_corpus": [],  # Text extraction happens in the dialog
        "storyteller": ["descriptions"],  # Needs description analysis
    }

    def __init__(self, parent=None):
        super().__init__(parent)

        # Workflow state
        self._state = WorkflowState.IDLE
        self._algorithm: str = ""
        self._direction: Optional[str] = None
        self._local_files: list[Path] = []
        self._urls: list[str] = []
        self._cancelled = False

        # Processing state
        self._sources_to_process: list = []  # Sources waiting for detection
        self._sources_processed: list = []  # Successfully processed sources
        self._sources_failed: list[dict] = []  # Failed sources with errors
        self._all_clips: list = []  # All clips created
        self._current_source_index = 0

        # Guard flags to prevent duplicate signal handling
        # (Critical pattern from docs/solutions/qthread-destroyed-duplicate-signal-delivery)
        self._download_finished_handled = False
        self._detection_finished_handled = False
        self._thumbnails_finished_handled = False
        self._analysis_finished_handled = False

        # Worker references (set by MainWindow when connecting)
        self._download_worker = None
        self._detection_worker = None
        self._thumbnail_worker = None
        self._color_worker = None

        logger.info("IntentionWorkflowCoordinator initialized")

    @property
    def state(self) -> WorkflowState:
        """Current workflow state."""
        return self._state

    @property
    def algorithm(self) -> str:
        """The sequence algorithm being built."""
        return self._algorithm

    @property
    def is_running(self) -> bool:
        """Whether the workflow is currently running."""
        return self._state not in (
            WorkflowState.IDLE,
            WorkflowState.COMPLETE,
            WorkflowState.CANCELLED,
            WorkflowState.ERROR,
        )

    def start(
        self,
        algorithm: str,
        local_files: list[Path],
        urls: list[str],
        direction: Optional[str] = None,
    ) -> bool:
        """Start the intention workflow.

        Args:
            algorithm: The sequence algorithm (color, duration, shuffle, sequential)
            local_files: List of local video file paths
            urls: List of YouTube/Vimeo URLs to download
            direction: Optional direction for the algorithm (e.g., "rainbow" for color)

        Returns:
            True if workflow started, False if already running
        """
        if self.is_running:
            logger.warning("Workflow already running, cannot start new workflow")
            return False

        logger.info(f"Starting intention workflow: algorithm={algorithm}, "
                    f"files={len(local_files)}, urls={len(urls)}")

        # Reset state
        self._reset()
        self._algorithm = algorithm
        self._direction = direction
        self._local_files = list(local_files)
        self._urls = list(urls)

        # Determine first step
        if urls:
            self._state = WorkflowState.DOWNLOADING
            self._emit_progress("Starting downloads...")
            self.step_started.emit("downloading", 1, self._calculate_total_steps())
        elif local_files:
            # Skip download, go straight to detection
            self._sources_to_process = [{"path": f, "type": "local"} for f in local_files]
            self._state = WorkflowState.DETECTING
            self._emit_progress("Starting scene detection...")
            self.step_started.emit("detecting", 1, self._calculate_total_steps())
        else:
            # No inputs - error
            self._state = WorkflowState.ERROR
            self.workflow_error.emit("No files or URLs provided")
            return False

        return True

    def cancel(self):
        """Cancel the running workflow."""
        if not self.is_running:
            return

        logger.info("Cancelling intention workflow")
        self._cancelled = True
        self._state = WorkflowState.CANCELLED

        # Cancel any running workers
        if self._download_worker and hasattr(self._download_worker, 'cancel'):
            self._download_worker.cancel()
        if self._detection_worker and hasattr(self._detection_worker, 'cancel'):
            self._detection_worker.cancel()
        if self._color_worker and hasattr(self._color_worker, 'cancel'):
            self._color_worker.cancel()

        self.workflow_cancelled.emit()

    def _reset(self):
        """Reset all workflow state for a new run."""
        self._state = WorkflowState.IDLE
        self._algorithm = ""
        self._direction = None
        self._local_files = []
        self._urls = []
        self._cancelled = False

        self._sources_to_process = []
        self._sources_processed = []
        self._sources_failed = []
        self._all_clips = []
        self._current_source_index = 0

        # Reset guard flags
        self._download_finished_handled = False
        self._detection_finished_handled = False
        self._thumbnails_finished_handled = False
        self._analysis_finished_handled = False

    def _calculate_total_steps(self) -> int:
        """Calculate total number of steps in the workflow."""
        steps = 0
        if self._urls:
            steps += 1  # Download
        steps += 1  # Detection
        steps += 1  # Thumbnails
        if self._needs_analysis():
            steps += 1  # Analysis
        steps += 1  # Build sequence
        return steps

    def _needs_analysis(self) -> bool:
        """Check if the algorithm requires analysis."""
        requirements = self.ANALYSIS_REQUIREMENTS.get(self._algorithm, [])
        return len(requirements) > 0

    def _get_current_step_number(self) -> int:
        """Get the current step number (1-based)."""
        step = 0
        if self._state == WorkflowState.DOWNLOADING:
            return 1
        if self._urls:
            step += 1

        if self._state == WorkflowState.DETECTING:
            return step + 1
        step += 1

        if self._state == WorkflowState.THUMBNAILS:
            return step + 1
        step += 1

        if self._state == WorkflowState.ANALYZING:
            return step + 1
        if self._needs_analysis():
            step += 1

        if self._state == WorkflowState.BUILDING:
            return step + 1

        return step + 1

    def _emit_progress(self, message: str, step_progress: float = 0.0):
        """Emit a progress update."""
        progress = WorkflowProgress(
            state=self._state,
            current_step=self._get_current_step_number(),
            total_steps=self._calculate_total_steps(),
            step_progress=step_progress,
            message=message,
            sources_processed=len(self._sources_processed),
            sources_total=len(self._local_files) + len(self._urls),
            clips_created=len(self._all_clips),
        )
        self.progress_updated.emit(progress)

    # --- Download phase handlers ---

    def on_download_progress(self, current: int, total: int, message: str):
        """Handle download progress from URLBulkDownloadWorker."""
        if self._state != WorkflowState.DOWNLOADING or self._cancelled:
            return
        progress = current / total if total > 0 else 0
        self._emit_progress(message, progress)

    def on_download_video_finished(self, url: str, result):
        """Handle individual video download completion."""
        if self._state != WorkflowState.DOWNLOADING or self._cancelled:
            return

        if result and result.success and result.file_path:
            # Add to sources to process
            self._sources_to_process.append({
                "path": Path(result.file_path),
                "type": "downloaded",
                "url": url,
            })
            logger.info(f"Download complete: {url} -> {result.file_path}")
        else:
            error = result.error if result else "Unknown error"
            self._sources_failed.append({"url": url, "error": error})
            logger.warning(f"Download failed: {url} - {error}")

    def on_download_all_finished(self, results: list):
        """Handle all downloads completed."""
        if self._download_finished_handled:
            logger.warning("Download finished already handled, ignoring duplicate")
            return
        self._download_finished_handled = True

        if self._cancelled:
            return

        logger.info(f"All downloads complete: {len(self._sources_to_process)} succeeded, "
                    f"{len(self._sources_failed)} failed")

        # Add local files to the processing queue
        for f in self._local_files:
            self._sources_to_process.append({"path": f, "type": "local"})

        self.step_completed.emit("downloading")

        # Move to detection phase
        if self._sources_to_process:
            self._state = WorkflowState.DETECTING
            self.step_started.emit("detecting", self._get_current_step_number(),
                                   self._calculate_total_steps())
            self._emit_progress("Starting scene detection...")
            # Signal that detection should start - MainWindow will handle
        else:
            # All downloads failed
            self._complete_with_error("All downloads failed")

    # --- Detection phase handlers ---

    def on_detection_progress(self, progress: float, message: str):
        """Handle detection progress."""
        if self._state != WorkflowState.DETECTING or self._cancelled:
            return

        # Calculate overall progress across all sources
        source_count = len(self._sources_to_process)
        if source_count > 0:
            base_progress = self._current_source_index / source_count
            source_progress = progress / source_count
            overall = base_progress + source_progress
        else:
            overall = progress

        self._emit_progress(message, overall)

    def on_detection_completed(self, source, clips: list):
        """Handle detection completion for a single source."""
        if self._detection_finished_handled:
            logger.warning("Detection finished already handled, ignoring duplicate")
            return

        if self._cancelled:
            return

        logger.info(f"Detection complete for source: {source.id}, {len(clips)} clips")

        # Store results
        self._sources_processed.append(source)
        self._all_clips.extend(clips)

        # Move to next source or next phase
        self._current_source_index += 1

        if self._current_source_index < len(self._sources_to_process):
            # More sources to detect
            self._emit_progress(
                f"Detecting scenes ({self._current_source_index + 1}/{len(self._sources_to_process)})..."
            )
            # MainWindow will start next detection
        else:
            # All sources detected, move to thumbnails
            self._detection_finished_handled = True
            self.step_completed.emit("detecting")
            self._advance_to_thumbnails()

    def on_detection_error(self, error: str):
        """Handle detection error for a source."""
        if self._cancelled:
            return

        logger.warning(f"Detection error: {error}")

        # Record the failure
        if self._current_source_index < len(self._sources_to_process):
            source_info = self._sources_to_process[self._current_source_index]
            self._sources_failed.append({
                "path": str(source_info.get("path", "unknown")),
                "error": error,
            })

        # Move to next source
        self._current_source_index += 1

        if self._current_source_index < len(self._sources_to_process):
            # More sources to detect
            self._emit_progress(
                f"Detecting scenes ({self._current_source_index + 1}/{len(self._sources_to_process)})..."
            )
        else:
            # All sources attempted
            self._detection_finished_handled = True
            self.step_completed.emit("detecting")

            if self._all_clips:
                # We have some clips, continue
                self._advance_to_thumbnails()
            else:
                # No clips at all
                self._complete_with_error("Scene detection failed for all sources")

    def _advance_to_thumbnails(self):
        """Advance workflow to thumbnail generation phase."""
        self._state = WorkflowState.THUMBNAILS
        self.step_started.emit("thumbnails", self._get_current_step_number(),
                               self._calculate_total_steps())
        self._emit_progress("Generating thumbnails...")

    # --- Thumbnail phase handlers ---

    def on_thumbnail_progress(self, current: int, total: int):
        """Handle thumbnail generation progress."""
        if self._state != WorkflowState.THUMBNAILS or self._cancelled:
            return
        progress = current / total if total > 0 else 0
        self._emit_progress(f"Generating thumbnails ({current}/{total})...", progress)

    def on_thumbnails_finished(self):
        """Handle thumbnail generation completion."""
        if self._thumbnails_finished_handled:
            logger.warning("Thumbnails finished already handled, ignoring duplicate")
            return
        self._thumbnails_finished_handled = True

        if self._cancelled:
            return

        logger.info("Thumbnail generation complete")
        self.step_completed.emit("thumbnails")

        # Decide next step
        if self._needs_analysis():
            self._state = WorkflowState.ANALYZING
            self.step_started.emit("analyzing", self._get_current_step_number(),
                                   self._calculate_total_steps())
            self._emit_progress("Analyzing clips...")
        else:
            self.step_skipped.emit("analyzing")
            self._advance_to_building()

    # --- Analysis phase handlers ---

    def on_analysis_progress(self, current: int, total: int):
        """Handle analysis progress."""
        if self._state != WorkflowState.ANALYZING or self._cancelled:
            return
        progress = current / total if total > 0 else 0
        self._emit_progress(f"Analyzing clips ({current}/{total})...", progress)

    def on_analysis_finished(self):
        """Handle analysis completion."""
        if self._analysis_finished_handled:
            logger.warning("Analysis finished already handled, ignoring duplicate")
            return
        self._analysis_finished_handled = True

        if self._cancelled:
            return

        logger.info("Analysis complete")
        self.step_completed.emit("analyzing")
        self._advance_to_building()

    def _advance_to_building(self):
        """Advance workflow to sequence building phase."""
        self._state = WorkflowState.BUILDING
        self.step_started.emit("building", self._get_current_step_number(),
                               self._calculate_total_steps())
        self._emit_progress("Building sequence...")

    # --- Building phase handlers ---

    def on_building_complete(self, sequence_clips: list):
        """Handle sequence building completion."""
        if self._cancelled:
            return

        logger.info(f"Sequence built with {len(sequence_clips)} clips")
        self.step_completed.emit("building")
        self._state = WorkflowState.COMPLETE

        result = WorkflowResult(
            success=True,
            algorithm=self._algorithm,
            clips_created=len(self._all_clips),
            sources_processed=len(self._sources_processed),
            sources_failed=len(self._sources_failed),
            failed_sources=self._sources_failed,
        )

        self.workflow_completed.emit(result)

    def _complete_with_error(self, message: str):
        """Complete workflow with an error."""
        logger.error(f"Workflow error: {message}")
        self._state = WorkflowState.ERROR

        result = WorkflowResult(
            success=False,
            algorithm=self._algorithm,
            clips_created=len(self._all_clips),
            sources_processed=len(self._sources_processed),
            sources_failed=len(self._sources_failed),
            error_message=message,
            failed_sources=self._sources_failed,
        )

        self.workflow_completed.emit(result)

    # --- Methods for MainWindow to query state ---

    def get_sources_to_detect(self) -> list[Path]:
        """Get list of source paths that need detection."""
        return [s["path"] for s in self._sources_to_process]

    def get_current_source_path(self) -> Optional[Path]:
        """Get the path of the current source being processed."""
        if self._current_source_index < len(self._sources_to_process):
            return self._sources_to_process[self._current_source_index]["path"]
        return None

    def get_all_clips(self) -> list:
        """Get all clips created during the workflow."""
        return self._all_clips

    def get_all_sources(self) -> list:
        """Get all sources successfully processed."""
        return self._sources_processed

    def get_algorithm_with_direction(self) -> tuple[str, Optional[str]]:
        """Get the algorithm and direction for sequence generation."""
        return self._algorithm, self._direction
