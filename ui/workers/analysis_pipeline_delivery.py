"""Run identity for combined analysis replies and owner-thread handoffs."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import QThread

from ui.workers.gui_tool_reply import AgentAnalysisCompletion, GuiToolReply


@dataclass
class AnalysisPipelineRun:
    session_id: str
    reply: GuiToolReply | None
    finished: bool = False


def pipeline_can_continue(window: Any) -> bool:
    """Retire expired requests before another operation or source is launched."""
    run = getattr(window, "_analysis_run", None)
    if run is None:
        return True  # Standalone launchers do not own a pipeline.
    if run.finished:
        return False
    same_session = window.project.session.session_id == run.session_id
    if same_session and (run.reply is None or run.reply.is_current(window)):
        return True
    run.finished = True
    window._analysis_pending_phases = []
    window._analysis_sequential_queue = []
    window._transcription_source_queue = []
    if same_session:
        window._gui_state.clear_processing("analysis")
        window.analyze_tab.set_analyzing(False)
        window.progress_bar.setVisible(False)
    return False


def bind_pipeline_completion(
    window: Any,
    worker: QThread,
    attribute: str,
    signal: Any,
    handler: Callable[[], None],
) -> None:
    """Keep worker cleanup separate from run-scoped result delivery."""
    run = getattr(window, "_analysis_run", None)

    def deliver(**_: Any) -> None:
        if getattr(window, "_analysis_run", None) is run and pipeline_can_continue(window):
            handler()

    completion = AgentAnalysisCompletion(window, worker, attribute, deliver)
    signal.connect(completion.completed)
