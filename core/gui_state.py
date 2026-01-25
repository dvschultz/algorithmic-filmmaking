"""GUI state tracking for agent context awareness.

Tracks current GUI state so the chat agent can be aware of:
- Recent YouTube search results
- Current tab selection
- Selected clips and sources

This state is passed to the agent's system prompt for context.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GUIState:
    """Current GUI state for agent context."""

    # YouTube search state
    last_search_query: str = ""
    search_results: list[dict] = field(default_factory=list)
    selected_video_ids: list[str] = field(default_factory=list)

    # Tab state
    active_tab: str = "collect"  # collect, cut, analyze, sequence, generate, render

    # Selection state
    selected_clip_ids: list[str] = field(default_factory=list)
    selected_source_id: Optional[str] = None

    def to_context_string(self) -> str:
        """Generate context string for agent system prompt.

        Returns:
            Human-readable string describing current GUI state,
            or empty string if no notable state to report.
        """
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

        if self.selected_clip_ids:
            lines.append(f"SELECTED CLIPS: {len(self.selected_clip_ids)}")

        if self.selected_source_id:
            lines.append(f"ACTIVE SOURCE: {self.selected_source_id[:8]}...")

        return "\n".join(lines) if lines else ""

    def clear_search_state(self):
        """Clear YouTube search state."""
        self.last_search_query = ""
        self.search_results = []
        self.selected_video_ids = []

    def update_from_search(self, query: str, results: list[dict]):
        """Update state from a YouTube search.

        Args:
            query: Search query
            results: List of video dicts
        """
        self.last_search_query = query
        self.search_results = results
        self.selected_video_ids = []
