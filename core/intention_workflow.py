"""Intention-First Workflow Coordinator.

Orchestrates the multi-step flow when a user clicks a sequence card with no clips:
Import -> Download (if URLs) -> Detect scenes -> Generate thumbnails -> Analyze (if needed) -> Build sequence
"""

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

from PySide6.QtCore import QObject, Signal

from core.operations.intention import IntentionPlan

if TYPE_CHECKING:
    from models.clip import Clip, Source

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

    The Qt-free plan owns phase order and consumes each completion once.
    This adapter collects results and projects the active step through signals.
    """

    # Signals for progress updates
    progress_updated = Signal(object)  # WorkflowProgress
    step_started = Signal(str, int, int)  # step_name, current, total
    step_completed = Signal(str)  # step_name
    step_skipped = Signal(str)  # step_name
    workflow_completed = Signal(object)  # WorkflowResult
    workflow_cancelled = Signal()
    workflow_error = Signal(str)  # error message

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)

        # Workflow state
        self.plan: IntentionPlan | None = None
        self._state: WorkflowState = WorkflowState.IDLE
        self._algorithm: str = ""
        self._direction: Optional[str] = None
        self._local_files: list[Path] = []
        self._urls: list[str] = []
        self._download_outcomes: set[str] = set()
        self._cancelled = False

        # Processing state
        self._sources_to_process: list[dict[str, Any]] = []
        self._sources_processed: list["Source"] = []
        self._sources_failed: list[dict] = []  # Failed sources with errors
        self._all_clips: list["Clip"] = []
        self._current_source_index = 0

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

        logger.info(
            f"Starting intention workflow: algorithm={algorithm}, "
            f"files={len(local_files)}, urls={len(urls)}"
        )

        # Reset state
        self._reset()
        self._algorithm = algorithm
        self._direction = direction
        self._local_files = list(dict.fromkeys(local_files))
        self._urls = list(dict.fromkeys(urls))

        if not urls and not local_files:
            self._state = WorkflowState.ERROR
            self.workflow_error.emit("No files or URLs provided")
            return False
        self.plan = IntentionPlan(algorithm, downloads=bool(urls))
        if not urls:
            self._sources_to_process = [
                {"path": f, "type": "local"} for f in self._local_files
            ]
        self._enter_step()

        return True

    def cancel(self) -> None:
        """Cancel the running workflow."""
        if not self.is_running:
            return

        logger.info("Cancelling intention workflow")
        self._cancelled = True
        if self.plan is not None:
            self.plan.cancelled = True
        self._state = WorkflowState.CANCELLED

        self.workflow_cancelled.emit()

    def _reset(self):
        """Reset all workflow state for a new run."""
        self.plan = None
        self._state = WorkflowState.IDLE
        self._algorithm = ""
        self._direction = None
        self._local_files = []
        self._urls = []
        self._download_outcomes = set()
        self._cancelled = False

        self._sources_to_process = []
        self._sources_processed = []
        self._sources_failed = []
        self._all_clips = []
        self._current_source_index = 0

    def _calculate_total_steps(self) -> int:
        return len(self.plan.steps) if self.plan is not None else 0

    def _needs_analysis(self) -> bool:
        return bool(self.plan and self.plan.analysis_requirements)

    def _get_current_step_number(self) -> int:
        if self.plan is None:
            return 0
        return min(len(self.plan.completed) + 1, len(self.plan.steps))

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

    def on_download_video_finished(self, url: str, result: Any) -> None:
        """Handle individual video download completion."""
        if self._state != WorkflowState.DOWNLOADING or self._cancelled:
            return
        if url not in self._urls or url in self._download_outcomes:
            return
        self._download_outcomes.add(url)

        if result and result.success and result.file_path:
            # Add to sources to process
            self._sources_to_process.append(
                {
                    "path": Path(result.file_path),
                    "type": "downloaded",
                    "url": url,
                }
            )
            logger.info(f"Download complete: {url} -> {result.file_path}")
        else:
            error = result.error if result else "Unknown error"
            self._sources_failed.append({"url": url, "error": error})
            logger.warning(f"Download failed: {url} - {error}")

    def on_download_all_finished(self, results: list) -> None:
        """Handle all downloads completed."""
        if not self._active("downloading"):
            return

        # Terminal summaries include failures that have no success-only item signal.
        from core.downloader import DownloadResult

        for item in results:
            if not isinstance(item, dict):
                continue  # Older callers already delivered DownloadResult objects.
            url = item.get("url")
            if url not in self._urls or url in self._download_outcomes:
                continue
            path = item.get("file_path")
            self.on_download_video_finished(
                url,
                DownloadResult(
                    success=bool(item.get("success") and path),
                    file_path=Path(path) if path else None,
                    error=item.get("error") or "Download did not complete",
                ),
            )
        if not self._active("downloading"):
            return

        for url in self._urls:
            if url not in self._download_outcomes:
                self.on_download_video_finished(
                    url,
                    DownloadResult(
                        success=False,
                        error="Download completed without a result",
                    ),
                )

        logger.info(
            f"All downloads complete: {len(self._sources_to_process)} succeeded, "
            f"{len(self._sources_failed)} failed"
        )

        # Add local files to the processing queue
        for f in self._local_files:
            if not any(item["path"] == f for item in self._sources_to_process):
                self._sources_to_process.append({"path": f, "type": "local"})

        if self._sources_to_process:
            self._advance("downloading")
        else:
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

    def on_detection_completed(self, source: "Source", clips: list["Clip"]) -> None:
        """Handle detection completion for a single source."""
        if not self._active("detecting") or any(
            item.id == source.id for item in self._sources_processed
        ):
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
            if self._all_clips:
                self._advance("detecting")
            else:
                self._complete_with_error("Scene detection produced no clips")

    def on_detection_error(self, error: str) -> None:
        """Handle detection error for a source."""
        if not self._active("detecting"):
            return

        logger.warning(f"Detection error: {error}")

        # Record the failure
        if self._current_source_index < len(self._sources_to_process):
            source_info = self._sources_to_process[self._current_source_index]
            self._sources_failed.append(
                {
                    "path": str(source_info.get("path", "unknown")),
                    "error": error,
                }
            )

        # Move to next source
        self._current_source_index += 1

        if self._current_source_index < len(self._sources_to_process):
            # More sources to detect
            self._emit_progress(
                f"Detecting scenes ({self._current_source_index + 1}/{len(self._sources_to_process)})..."
            )
        else:
            # All sources attempted

            if self._all_clips:
                # We have some clips, continue
                self._advance("detecting")
            else:
                # No clips at all
                self._complete_with_error("Scene detection failed for all sources")

    def _active(self, step: str) -> bool:
        return bool(
            self.plan and self.plan.active == step and self._state.name.lower() == step
        )

    def _enter_step(self) -> None:
        plan = self.plan
        if plan is None or plan.active is None:
            return
        step = plan.active
        if step == "building" and not self._require_analysis():
            return
        if step == "detecting":
            sources: dict[Path, dict[str, Any]] = {}
            for source in self._sources_to_process:
                sources.setdefault(source["path"].resolve(), source)
            self._sources_to_process = list(sources.values())
        self._state = WorkflowState[step.upper()]
        messages = {
            "downloading": "Starting downloads...",
            "detecting": "Starting scene detection...",
            "thumbnails": "Generating thumbnails...",
            "analyzing": "Analyzing clips...",
            "building": "Building sequence...",
        }
        self._emit_progress(messages[step])
        if self.plan is plan and self._active(step):
            self.step_started.emit(
                step, self._get_current_step_number(), self._calculate_total_steps()
            )

    def _advance(self, step: str) -> None:
        plan = self.plan
        if plan is None or not self._active(step) or not plan.finish(step):
            return
        self.step_completed.emit(step)
        if self.plan is plan and plan.active is not None:
            self._enter_step()

    # --- Thumbnail phase handlers ---

    def on_thumbnail_progress(self, current: int, total: int):
        """Handle thumbnail generation progress."""
        if self._state != WorkflowState.THUMBNAILS or self._cancelled:
            return
        progress = current / total if total > 0 else 0
        self._emit_progress(f"Generating thumbnails ({current}/{total})...", progress)

    def on_thumbnails_finished(self) -> None:
        if not self._active("thumbnails"):
            return
        if not self._needs_analysis():
            self.step_skipped.emit("analyzing")
        self._advance("thumbnails")

    # --- Analysis phase handlers ---

    def on_analysis_progress(self, current: int, total: int):
        """Handle analysis progress."""
        if self._state != WorkflowState.ANALYZING or self._cancelled:
            return
        progress = current / total if total > 0 else 0
        self._emit_progress(f"Analyzing clips ({current}/{total})...", progress)

    def on_analysis_finished(self) -> None:
        if not self._active("analyzing"):
            return
        if self._require_analysis():
            self._advance("analyzing")

    def _require_analysis(self) -> bool:
        assert self.plan is not None
        missing = self.plan.missing_analysis(self._all_clips)
        if missing:
            self._complete_with_error(
                f"Required analysis is missing for {len(missing)} clip(s)"
            )
            return False
        return True

    def on_analysis_failed(self, message: str) -> None:
        if self._active("analyzing"):
            self._complete_with_error(message)

    # --- Building phase handlers ---

    def on_building_complete(self, sequence_clips: list) -> None:
        plan = self.plan
        if plan is None or not self._active("building"):
            return
        if not self._require_analysis() or not plan.finish("building"):
            return
        self.step_completed.emit("building")
        if self.plan is not plan or plan.cancelled or plan.error:
            return
        self._state = WorkflowState.COMPLETE
        self.workflow_completed.emit(
            WorkflowResult(
                success=True,
                algorithm=self._algorithm,
                clips_created=len(self._all_clips),
                sources_processed=len(self._sources_processed),
                sources_failed=len(self._sources_failed),
                failed_sources=list(self._sources_failed),
            )
        )

    def fail(self, message: str) -> None:
        """Abort the current run when its execution adapter cannot continue."""
        self._complete_with_error(message)

    def _complete_with_error(self, message: str):
        """Complete workflow with an error."""
        if not self.is_running:
            return
        if self.plan is not None:
            self.plan.error = message
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
            return Path(self._sources_to_process[self._current_source_index]["path"])
        return None

    def get_download_urls(self) -> list[str]:
        """Get URL inputs queued for the download phase."""
        return list(self._urls)

    def get_all_clips(self) -> list:
        """Get all clips created during the workflow."""
        return self._all_clips

    def get_all_sources(self) -> list:
        """Get all sources successfully processed."""
        return self._sources_processed

    def get_algorithm_with_direction(self) -> tuple[str, Optional[str]]:
        """Get the algorithm and direction for sequence generation."""
        return self._algorithm, self._direction
