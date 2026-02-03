"""GUI state tracking for agent context awareness.

Tracks current GUI state so the chat agent can be aware of:
- Recent YouTube search results
- Current tab selection
- Selected clips and sources
- Current execution plan

This state is passed to the agent's system prompt for context.
"""

from dataclasses import dataclass, field
from threading import Lock
from typing import Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from models.plan import Plan


@dataclass
class PendingAction:
    """Base class for pending actions that require follow-up."""
    pass


@dataclass
class NameProjectThenPlanAction(PendingAction):
    """Pending action: name the project, then present a plan."""
    pending_steps: list[str]
    pending_summary: str
    user_response: Optional[str] = None


# Type alias for all pending action types
PendingActionType = Optional[Union[NameProjectThenPlanAction]]


@dataclass
class GUIState:
    """Current GUI state for agent context.

    Thread-safe: All access to mutable state is protected by an internal lock.
    """

    # YouTube search state
    last_search_query: str = ""
    search_results: list[dict] = field(default_factory=list)
    selected_video_ids: list[str] = field(default_factory=list)

    # Tab state
    active_tab: str = "collect"  # collect, cut, analyze, sequence, generate, render

    # Selection state
    selected_clip_ids: list[str] = field(default_factory=list)
    selected_source_id: Optional[str] = None

    # Tab-specific selection state (for Sequence tab to read from)
    analyze_selected_ids: list[str] = field(default_factory=list)
    cut_selected_ids: list[str] = field(default_factory=list)

    # Analyze tab state
    analyze_tab_ids: list[str] = field(default_factory=list)

    # Sequence state
    sequence_ids: list[str] = field(default_factory=list)

    # Plan state
    current_plan: Optional["Plan"] = None

    # Active filters state
    active_filters: dict = field(default_factory=dict)

    # Pending action state - tracks actions agent must complete after user responds
    pending_action: PendingActionType = None

    # Thread safety lock (not included in repr/compare/hash)
    _lock: Lock = field(default_factory=Lock, repr=False, compare=False)

    def to_context_string(self) -> str:
        """Generate context string for agent system prompt.

        Returns:
            Human-readable string describing current GUI state,
            or empty string if no notable state to report.
        """
        with self._lock:
            lines = []

            if self.search_results:
                lines.append(f"RECENT YOUTUBE SEARCH: '{self.last_search_query}'")
                lines.append(f"  Found {len(self.search_results)} videos")
                # Show first 3 results
                for v in self.search_results[:3]:
                    title = v.get("title", "Unknown")
                    duration = v.get("duration", "?")
                    lines.append(f"  - {title} ({duration})")
                if len(self.search_results) > 3:
                    lines.append(f"  - ...and {len(self.search_results) - 3} more")

            if self.selected_video_ids:
                count = len(self.selected_video_ids)
                lines.append(f"VIDEOS SELECTED FOR DOWNLOAD: {count}")

            lines.append(f"ACTIVE TAB: {self.active_tab}")

            if self.analyze_tab_ids:
                lines.append(f"ANALYZE TAB CLIPS: {len(self.analyze_tab_ids)}")
                ids_str = ", ".join(self.analyze_tab_ids[:20])
                if len(self.analyze_tab_ids) > 20:
                    ids_str += ", ..."
                lines.append(f"ANALYZE_IDS: [{ids_str}]")

            if self.sequence_ids:
                lines.append(f"TIMELINE SEQUENCE: {len(self.sequence_ids)} clips")
                ids_str = ", ".join(self.sequence_ids[:20])
                if len(self.sequence_ids) > 20:
                    ids_str += ", ..."
                lines.append(f"SEQUENCE_IDS: [{ids_str}]")

            if self.selected_clip_ids:
                lines.append(f"SELECTED CLIPS: {len(self.selected_clip_ids)}")
                # Add list of IDs (truncated if too many)
                ids_str = ", ".join(self.selected_clip_ids[:20])
                if len(self.selected_clip_ids) > 20:
                    ids_str += ", ..."
                lines.append(f"SELECTED_IDS: [{ids_str}]")

            if self.selected_source_id:
                lines.append(f"ACTIVE SOURCE: {self.selected_source_id[:8]}...")

            # Active filters
            if self.active_filters:
                active = [f"{k}={v}" for k, v in self.active_filters.items() if v is not None]
                if active:
                    lines.append(f"ACTIVE FILTERS: {', '.join(active)}")

            # Pending action (high priority - show first if exists)
            if self.pending_action:
                action = self.pending_action
                lines.insert(0, "=" * 50)
                lines.insert(1, "⚠️  PENDING ACTION REQUIRED - YOU MUST ACT NOW")
                if isinstance(action, NameProjectThenPlanAction):
                    user_name = action.user_response or ''
                    lines.insert(2, "Action: name_project_then_plan")
                    lines.insert(3, f"User provided name: '{user_name}'")
                    lines.insert(4, "")
                    lines.insert(5, "YOU MUST CALL THESE TOOLS IN ORDER:")
                    lines.insert(6, f'  1. set_project_name(name="{user_name}")')
                    lines.insert(7, "  2. present_plan(steps=[...], summary=\"...\")")
                    lines.insert(8, "")
                    if action.pending_steps:
                        lines.insert(9, f"PENDING PLAN ({len(action.pending_steps)} steps):")
                        lines.insert(10, f"  Summary: {action.pending_summary}")
                        lines.insert(11, "  Steps:")
                        for i, step in enumerate(action.pending_steps):
                            lines.insert(12 + i, f"    {i+1}. {step}")
                lines.insert(len(lines), "")
                lines.insert(len(lines), "DO NOT respond with text. Call set_project_name NOW.")
                lines.insert(len(lines), "=" * 50)

            # Plan state
            if self.current_plan:
                plan = self.current_plan
                lines.append(f"CURRENT PLAN: {plan.summary}")
                lines.append(f"  Status: {plan.status}")
                lines.append(f"  Steps: {len(plan.steps)}")
                if plan.status == "executing":
                    lines.append(f"  Current step: {plan.current_step_index + 1}/{len(plan.steps)}")
                    if plan.current_step:
                        lines.append(f"  Executing: {plan.current_step.description}")
                elif plan.status == "draft":
                    lines.append("  Awaiting user confirmation")

            return "\n".join(lines) if lines else ""

    def clear_search_state(self):
        """Clear YouTube search state."""
        with self._lock:
            self.last_search_query = ""
            self.search_results = []
            self.selected_video_ids = []

    def update_from_search(self, query: str, results: list[dict]):
        """Update state from a YouTube search.

        Args:
            query: Search query
            results: List of video dicts
        """
        with self._lock:
            self.last_search_query = query
            self.search_results = results
            self.selected_video_ids = []

    def clear_plan_state(self):
        """Clear the current plan state."""
        with self._lock:
            self.current_plan = None

    def set_plan(self, plan: "Plan"):
        """Set the current plan.

        Args:
            plan: Plan object to track
        """
        with self._lock:
            self.current_plan = plan

    def set_pending_action(self, action: PendingAction):
        """Set a pending action that must be completed.

        Args:
            action: A PendingAction subclass instance
        """
        with self._lock:
            self.pending_action = action

    def clear_pending_action(self):
        """Clear the pending action after it's been handled."""
        with self._lock:
            self.pending_action = None

    def update_pending_action_response(self, user_response: str):
        """Update pending action with user's response.

        Args:
            user_response: The user's response text
        """
        with self._lock:
            if isinstance(self.pending_action, NameProjectThenPlanAction):
                self.pending_action.user_response = user_response

    def update_active_filters(self, filters: dict):
        """Update the active filters state.

        Args:
            filters: Dict of filter names to values (None values are excluded)
        """
        with self._lock:
            # Only keep non-None values
            self.active_filters = {k: v for k, v in filters.items() if v is not None}

    def clear_filters(self):
        """Clear the active filters state."""
        with self._lock:
            self.active_filters = {}

    def clear(self):
        """Clear all GUI state for new project.

        Resets all state to initial values.
        """
        with self._lock:
            # YouTube search state
            self.last_search_query = ""
            self.search_results = []
            self.selected_video_ids = []

            # Tab state
            self.active_tab = "collect"

            # Selection state
            self.selected_clip_ids = []
            self.selected_source_id = None
            self.analyze_selected_ids = []
            self.cut_selected_ids = []

            # Analyze tab state
            self.analyze_tab_ids = []

            # Sequence state
            self.sequence_ids = []

            # Plan state
            self.current_plan = None

            # Active filters
            self.active_filters = {}

            # Pending action
            self.pending_action = None
