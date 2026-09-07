"""Identity shared by adapters executing one intention plan."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from core.project import Project
    from core.intention_workflow import IntentionWorkflowCoordinator, WorkflowState
    from core.operations.intention import IntentionPlan
    from ui.workers.gui_tool_reply import GuiToolReply


@dataclass(frozen=True)
class IntentionRun:
    project: Project
    session_id: str
    path: Path | None
    workflow: IntentionWorkflowCoordinator
    plan: IntentionPlan | None
    reply: GuiToolReply | None

    @classmethod
    def capture(cls, window: Any) -> IntentionRun:
        window.project.session.assert_owner()
        return cls(
            window.project,
            window.project.session.session_id,
            window.project.path,
            window.intention_workflow,
            window.intention_workflow.plan,
            getattr(window, "_dispatch_gui_reply", None),
        )

    def is_current(self, window: Any, state: WorkflowState) -> bool:
        return bool(
            window.project is self.project
            and self.project.session.session_id == self.session_id
            and self.project.path == self.path
            and window.intention_workflow is self.workflow
            and self.workflow.plan is self.plan
            and self.workflow.state == state
            and (self.reply is None or self.reply.is_current(window))
        )
