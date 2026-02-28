"""Plan execution state machine.

Encapsulates all plan state transitions (present, start, advance, fail)
so tool functions can be thin wrappers.
"""

import logging
from typing import Optional

from core.gui_state import GUIState, NameProjectThenPlanAction

logger = logging.getLogger(__name__)

# Common error messages
_NO_PLAN_ERROR = (
    "No plan exists. You must call the present_plan TOOL (not write text). "
    "DO NOT describe steps in your message - that does not create a plan. "
    "Instead, make a tool call: present_plan(steps=[\"Step 1\", \"Step 2\", ...], summary=\"...\"). "
    "After the user confirms the plan widget, then call start_plan_execution."
)

# Keywords that indicate a plan will modify/use a project
_PROJECT_KEYWORDS = [
    "download", "detect", "scene", "clip", "sequence", "export",
    "analyze", "transcribe", "shuffle", "randomize", "import",
    "add to", "render", "save",
]


class PlanController:
    """Manages plan lifecycle: present -> confirm -> execute -> complete/fail.

    All state validation lives here. Tool functions delegate to this controller.
    """

    def __init__(self, gui_state: GUIState):
        self._gui_state = gui_state

    @property
    def current_plan(self):
        return self._gui_state.current_plan

    def present(self, steps: list[str], summary: str, project=None) -> dict:
        """Present a new plan for user confirmation.

        Args:
            steps: List of step descriptions
            summary: Brief plan summary
            project: Optional project to check naming

        Returns:
            Dict with plan info or action_required directive
        """
        from models.plan import Plan

        if not steps:
            return {"success": False, "error": "No steps provided for the plan"}

        if len(steps) > 20:
            return {
                "success": False,
                "error": f"Plan has {len(steps)} steps, maximum is 20. Please break into smaller plans."
            }

        # Check if plan involves project work and project is unnamed
        steps_text = " ".join(steps).lower()
        involves_project = any(kw in steps_text for kw in _PROJECT_KEYWORDS)

        if involves_project and project is not None:
            if project.metadata.name == "Untitled Project":
                self._gui_state.set_pending_action(
                    NameProjectThenPlanAction(
                        pending_steps=steps,
                        pending_summary=summary,
                    )
                )
                return {
                    "success": True,
                    "action_required": "name_project",
                    "instruction": (
                        "STOP. Do not call any tools. Ask the user directly: "
                        "'What would you like to name this project?' "
                        "Wait for their response. Then call set_project_name with "
                        "the name they provide, and present_plan again with the same steps."
                    ),
                    "pending_plan": {"steps": steps, "summary": summary},
                }

        # Create the plan
        plan = Plan.from_steps(steps, summary)
        self._gui_state.current_plan = plan

        return {
            "_display_plan": True,
            "plan_id": plan.id,
            "summary": summary,
            "steps": steps,
            "step_count": len(steps),
            "message": "Plan presented to user. Waiting for confirmation or edits.",
        }

    def start(self) -> dict:
        """Start executing the current plan.

        Returns:
            Current step info or error
        """
        plan = self.current_plan
        if plan is None:
            return {"success": False, "error": _NO_PLAN_ERROR}

        if plan.status == "executing":
            return {
                "success": True,
                "already_executing": True,
                "current_step_number": plan.current_step_index + 1,
                "total_steps": len(plan.steps),
                "current_step": plan.current_step.description if plan.current_step else None,
                "message": "Plan already executing. Continue with current step.",
            }

        if plan.status == "completed":
            return {
                "success": False,
                "error": "Plan already completed. Create a new plan with present_plan.",
            }

        plan.confirm()
        plan.start_execution()

        return {
            "success": True,
            "plan_id": plan.id,
            "status": "executing",
            "current_step_number": 1,
            "total_steps": len(plan.steps),
            "current_step": plan.current_step.description if plan.current_step else None,
            "remaining_steps": [s.description for s in plan.steps[1:]],
            "message": f"Plan started. Execute step 1: {plan.current_step.description if plan.current_step else 'Unknown'}",
        }

    def advance(self, result_summary: Optional[str] = None) -> dict:
        """Mark current step complete and advance.

        Args:
            result_summary: Brief description of what was accomplished

        Returns:
            Next step info or completion status
        """
        plan = self.current_plan
        if plan is None:
            return {"success": False, "error": _NO_PLAN_ERROR}

        if plan.status != "executing":
            return {
                "success": False,
                "error": f"Plan is not executing (status: {plan.status}). Call start_plan_execution first.",
            }

        completed_step = plan.current_step.description if plan.current_step else "Unknown"
        completed_step_number = plan.current_step_index + 1

        plan.advance_step(result_summary)

        if plan.status == "completed":
            return {
                "success": True,
                "plan_completed": True,
                "completed_step": completed_step,
                "completed_step_number": completed_step_number,
                "total_steps": len(plan.steps),
                "message": f"Step {completed_step_number} complete. All {len(plan.steps)} steps finished! Plan completed successfully.",
            }

        return {
            "success": True,
            "plan_completed": False,
            "completed_step": completed_step,
            "completed_step_number": completed_step_number,
            "current_step_number": plan.current_step_index + 1,
            "total_steps": len(plan.steps),
            "current_step": plan.current_step.description if plan.current_step else None,
            "remaining_steps": [s.description for s in plan.steps[plan.current_step_index + 1:]],
            "progress": plan.get_progress_summary(),
            "message": f"Step {completed_step_number} complete. Now execute step {plan.current_step_index + 1}: {plan.current_step.description if plan.current_step else 'Unknown'}",
        }

    _VALID_FAIL_ACTIONS = {"stop", "retry", "skip"}

    def fail(self, error: str, action: str = "stop") -> dict:
        """Handle a step failure.

        Args:
            error: Description of what went wrong
            action: 'stop', 'retry', or 'skip'

        Returns:
            Updated plan status
        """
        if action not in self._VALID_FAIL_ACTIONS:
            return {
                "success": False,
                "error": f"Invalid action '{action}'. Must be one of: {', '.join(sorted(self._VALID_FAIL_ACTIONS))}",
            }

        plan = self.current_plan
        if plan is None:
            return {"success": False, "error": "No plan exists."}

        if plan.status != "executing":
            return {
                "success": False,
                "error": f"Plan is not executing (status: {plan.status}).",
            }

        failed_step = plan.current_step.description if plan.current_step else "Unknown"
        failed_step_number = plan.current_step_index + 1

        if action == "retry":
            plan.retry_current_step()
            return {
                "success": True,
                "action": "retry",
                "step_number": failed_step_number,
                "current_step": failed_step,
                "message": f"Retrying step {failed_step_number}: {failed_step}",
            }

        if action == "skip":
            plan.skip_current_step(error)
            if plan.status == "completed":
                return {
                    "success": True,
                    "action": "skip",
                    "plan_completed": True,
                    "skipped_step": failed_step,
                    "message": f"Skipped failed step {failed_step_number}. Plan completed (with failures).",
                }
            return {
                "success": True,
                "action": "skip",
                "skipped_step": failed_step,
                "current_step_number": plan.current_step_index + 1,
                "current_step": plan.current_step.description if plan.current_step else None,
                "message": f"Skipped failed step. Now on step {plan.current_step_index + 1}: {plan.current_step.description if plan.current_step else 'Unknown'}",
            }

        # stop
        plan.fail_current_step(error)
        plan.stop_on_failure()
        return {
            "success": True,
            "action": "stop",
            "plan_status": "failed",
            "failed_step_number": failed_step_number,
            "failed_step": failed_step,
            "error": error,
            "message": f"Plan stopped at step {failed_step_number} due to error: {error}",
        }

    def get_status(self) -> dict:
        """Get current plan status.

        Returns:
            Plan status, current step, and remaining steps
        """
        plan = self.current_plan
        if plan is None:
            return {
                "has_plan": False,
                "message": "No active plan. Use present_plan to create one.",
            }

        steps_info = [
            {
                "step_number": i + 1,
                "description": step.description,
                "status": step.status,
                "result_summary": step.result_summary,
            }
            for i, step in enumerate(plan.steps)
        ]

        result = {
            "has_plan": True,
            "plan_id": plan.id,
            "summary": plan.summary,
            "status": plan.status,
            "total_steps": len(plan.steps),
            "steps": steps_info,
            "progress": plan.get_progress_summary(),
        }

        if plan.status == "executing" and plan.current_step:
            result["current_step_number"] = plan.current_step_index + 1
            result["current_step"] = plan.current_step.description
            result["remaining_steps"] = [
                s.description for s in plan.steps[plan.current_step_index + 1:]
            ]
            result["message"] = (
                f"Executing step {plan.current_step_index + 1}/{len(plan.steps)}: "
                f"{plan.current_step.description}"
            )
        elif plan.status == "completed":
            result["message"] = "Plan completed successfully."
        elif plan.status == "draft":
            result["message"] = "Plan awaiting user confirmation. Call start_plan_execution after user confirms."
        else:
            result["message"] = f"Plan status: {plan.status}"

        return result
