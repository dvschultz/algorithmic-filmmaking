"""Data models for agent planning workflows.

Provides:
- PlanStep: Individual step in a plan with status tracking
- Plan: Collection of steps with execution state
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import uuid


@dataclass
class PlanStep:
    """Represents a single step in an execution plan."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str = ""  # Human-readable step description
    status: str = "pending"  # pending, running, completed, failed
    error: Optional[str] = None  # Error message if failed
    result_summary: Optional[str] = None  # Brief summary of what was accomplished

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON export."""
        data = {
            "id": self.id,
            "description": self.description,
            "status": self.status,
        }
        if self.error:
            data["error"] = self.error
        if self.result_summary:
            data["result_summary"] = self.result_summary
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "PlanStep":
        """Deserialize from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            description=data.get("description", ""),
            status=data.get("status", "pending"),
            error=data.get("error"),
            result_summary=data.get("result_summary"),
        )


@dataclass
class Plan:
    """Represents a multi-step execution plan.

    Lifecycle:
    - draft: Initial state, plan can be edited
    - confirmed: User confirmed, ready for execution
    - executing: Currently running steps
    - completed: All steps finished successfully
    - cancelled: User cancelled (before or during execution)
    - failed: Execution stopped due to unrecoverable error
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    summary: str = ""  # Brief description of what plan accomplishes
    steps: list[PlanStep] = field(default_factory=list)
    status: str = "draft"  # draft, confirmed, executing, completed, cancelled, failed
    created_at: datetime = field(default_factory=datetime.now)
    confirmed_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    current_step_index: int = 0  # Index of step currently being executed

    @property
    def is_editable(self) -> bool:
        """Whether the plan can be modified."""
        return self.status == "draft"

    @property
    def is_executing(self) -> bool:
        """Whether the plan is currently running."""
        return self.status == "executing"

    @property
    def current_step(self) -> Optional[PlanStep]:
        """Get the current step being executed."""
        if 0 <= self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None

    def confirm(self):
        """Mark the plan as confirmed and ready for execution."""
        self.status = "confirmed"
        self.confirmed_at = datetime.now()

    def start_execution(self):
        """Begin executing the plan."""
        self.status = "executing"
        self.current_step_index = 0
        if self.steps:
            self.steps[0].status = "running"

    def advance_step(self, result_summary: Optional[str] = None):
        """Mark current step as completed and advance to next.

        Args:
            result_summary: Brief summary of what was accomplished
        """
        if self.current_step:
            self.current_step.status = "completed"
            self.current_step.result_summary = result_summary

        self.current_step_index += 1

        if self.current_step_index >= len(self.steps):
            # All steps completed
            self.status = "completed"
            self.completed_at = datetime.now()
        else:
            # Start next step
            self.steps[self.current_step_index].status = "running"

    def fail_current_step(self, error: str):
        """Mark current step as failed.

        Args:
            error: Error message describing what went wrong
        """
        if self.current_step:
            self.current_step.status = "failed"
            self.current_step.error = error

    def retry_current_step(self):
        """Reset current step to running for a retry attempt."""
        if self.current_step:
            self.current_step.status = "running"
            self.current_step.error = None

    def skip_current_step(self, error: str):
        """Mark current step as failed and skip to next step.

        Args:
            error: Error message describing why the step was skipped
        """
        if self.status != "executing":
            return

        # Mark current step as failed
        self.fail_current_step(error)

        # Advance to next step
        self.current_step_index += 1
        if self.current_step_index >= len(self.steps):
            self.status = "completed"
            self.completed_at = datetime.now()
        else:
            self.steps[self.current_step_index].status = "running"

    def cancel(self):
        """Cancel the plan execution."""
        self.status = "cancelled"
        self.completed_at = datetime.now()
        # Mark any running step as cancelled
        if self.current_step and self.current_step.status == "running":
            self.current_step.status = "pending"  # Reset to pending, not failed

    def stop_on_failure(self):
        """Stop execution after a failure (user chose not to retry)."""
        self.status = "failed"
        self.completed_at = datetime.now()

    def get_progress_summary(self) -> str:
        """Get a human-readable progress summary.

        Returns:
            String like "2/5 steps completed"
        """
        completed = sum(1 for s in self.steps if s.status == "completed")
        total = len(self.steps)
        return f"{completed}/{total} steps completed"

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON export."""
        return {
            "id": self.id,
            "summary": self.summary,
            "steps": [s.to_dict() for s in self.steps],
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "confirmed_at": self.confirmed_at.isoformat() if self.confirmed_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "current_step_index": self.current_step_index,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Plan":
        """Deserialize from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            summary=data.get("summary", ""),
            steps=[PlanStep.from_dict(s) for s in data.get("steps", [])],
            status=data.get("status", "draft"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            confirmed_at=datetime.fromisoformat(data["confirmed_at"]) if data.get("confirmed_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            current_step_index=data.get("current_step_index", 0),
        )

    @classmethod
    def from_steps(cls, steps: list[str], summary: str) -> "Plan":
        """Create a plan from a list of step descriptions.

        Args:
            steps: List of human-readable step descriptions
            summary: Brief description of what plan accomplishes

        Returns:
            New Plan instance in draft status
        """
        return cls(
            summary=summary,
            steps=[PlanStep(description=desc) for desc in steps],
        )
