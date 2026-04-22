"""Shared clip-filter state for the Cut and Analyze tab browsers.

``FilterState`` owns the values that drive :py:meth:`ClipBrowser._matches_filter`.
In Unit 1 of the filter-sidebar plan, it's a single source of truth that both
tabs point at — they share filter values by default. Later units (FilterSidebar)
subscribe to :py:attr:`changed` to know when to re-render their widgets.

The public ``apply_dict`` / ``to_dict`` contract mirrors today's
``ClipBrowser.apply_filters`` / ``get_active_filters`` dict shape exactly so
existing tests and any agent tools depending on those keys keep working.
"""

from typing import Any, Iterable, Optional

from PySide6.QtCore import QObject, Signal


_ENUM_ALL = "All"


class FilterState(QObject):
    """Shared filter state for clip browsers.

    Fields mirror the pre-refactor ``ClipBrowser`` filter attributes one-for-one.
    Assigning a field only emits :py:attr:`changed` if the value actually differs
    from the current value. :py:meth:`apply_dict` and :py:meth:`clear_all` batch
    multiple field changes into a single ``changed`` emission.
    """

    changed = Signal()

    _FIELD_NAMES: tuple[str, ...] = (
        "shot_type",
        "color_palette",
        "search_query",
        "selected_custom_queries",
        "min_duration",
        "max_duration",
        "aspect_ratio",
        "gaze_filter",
        "object_search",
        "description_search",
        "min_brightness",
        "max_brightness",
        "similarity_anchor_id",
        "similarity_scores",
    )

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._batching = False
        self._dirty_during_batch = False
        object.__setattr__(self, "shot_type", _ENUM_ALL)
        object.__setattr__(self, "color_palette", _ENUM_ALL)
        object.__setattr__(self, "search_query", "")
        object.__setattr__(self, "selected_custom_queries", set())
        object.__setattr__(self, "min_duration", None)
        object.__setattr__(self, "max_duration", None)
        object.__setattr__(self, "aspect_ratio", _ENUM_ALL)
        object.__setattr__(self, "gaze_filter", None)
        object.__setattr__(self, "object_search", "")
        object.__setattr__(self, "description_search", "")
        object.__setattr__(self, "min_brightness", None)
        object.__setattr__(self, "max_brightness", None)
        object.__setattr__(self, "similarity_anchor_id", None)
        object.__setattr__(self, "similarity_scores", {})

    # ── Dirty-tracking field assignment ─────────────────────────────

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self._FIELD_NAMES and hasattr(self, name):
            current = getattr(self, name)
            if current == value:
                return
            object.__setattr__(self, name, value)
            self._mark_dirty()
            return
        object.__setattr__(self, name, value)

    def _mark_dirty(self) -> None:
        if self._batching:
            self._dirty_during_batch = True
            return
        self.changed.emit()

    # ── Batched mutation helpers ────────────────────────────────────

    def _begin_batch(self) -> None:
        self._batching = True
        self._dirty_during_batch = False

    def _end_batch(self) -> None:
        self._batching = False
        if self._dirty_during_batch:
            self._dirty_during_batch = False
            self.changed.emit()

    # ── Public API mirroring ClipBrowser.apply_filters / get_active_filters ──

    def apply_dict(self, filters: dict) -> None:
        """Apply many filters at once; emits :py:attr:`changed` at most once.

        Accepts the exact dict shape produced by :py:meth:`to_dict` plus the
        backward-compat keys used by the pre-refactor ``ClipBrowser.apply_filters``.
        Unknown keys are ignored.
        """
        self._begin_batch()
        try:
            if "min_duration" in filters or "max_duration" in filters:
                self.min_duration = filters.get("min_duration")
                self.max_duration = filters.get("max_duration")

            if "aspect_ratio" in filters:
                value = filters["aspect_ratio"]
                self.aspect_ratio = value if value else _ENUM_ALL

            if "shot_type" in filters:
                value = filters["shot_type"]
                self.shot_type = value if value else _ENUM_ALL

            if "color_palette" in filters:
                value = filters["color_palette"]
                self.color_palette = value if value else _ENUM_ALL

            if "search_query" in filters:
                value = filters["search_query"] or ""
                self.search_query = value.lower().strip()

            if "custom_query" in filters:
                self.selected_custom_queries = _coerce_custom_query(filters["custom_query"])

            if "gaze" in filters:
                self.gaze_filter = filters["gaze"] or None

            if "object_search" in filters:
                self.object_search = (filters["object_search"] or "").strip()

            if "description_search" in filters:
                self.description_search = (filters["description_search"] or "").strip()

            if "min_brightness" in filters or "max_brightness" in filters:
                self.min_brightness = filters.get("min_brightness")
                self.max_brightness = filters.get("max_brightness")

            if "similarity_anchor" in filters:
                self.similarity_anchor_id = filters["similarity_anchor"]
        finally:
            self._end_batch()

    def to_dict(self) -> dict:
        """Return the active-filters dict matching ``ClipBrowser.get_active_filters``."""
        return {
            "shot_type": self.shot_type if self.shot_type != _ENUM_ALL else None,
            "color_palette": self.color_palette if self.color_palette != _ENUM_ALL else None,
            "search_query": self.search_query if self.search_query else None,
            "custom_query": (
                sorted(self.selected_custom_queries, key=str.lower)
                if self.selected_custom_queries
                else None
            ),
            "min_duration": self.min_duration,
            "max_duration": self.max_duration,
            "aspect_ratio": self.aspect_ratio if self.aspect_ratio != _ENUM_ALL else None,
            "gaze": self.gaze_filter,
            "object_search": self.object_search if self.object_search else None,
            "description_search": self.description_search if self.description_search else None,
            "min_brightness": self.min_brightness,
            "max_brightness": self.max_brightness,
            "similarity_anchor": self.similarity_anchor_id,
        }

    def has_active(self) -> bool:
        return (
            self.shot_type != _ENUM_ALL
            or self.color_palette != _ENUM_ALL
            or bool(self.search_query)
            or bool(self.selected_custom_queries)
            or self.min_duration is not None
            or self.max_duration is not None
            or self.aspect_ratio != _ENUM_ALL
            or self.gaze_filter is not None
            or bool(self.object_search)
            or bool(self.description_search)
            or self.min_brightness is not None
            or self.max_brightness is not None
            or self.similarity_anchor_id is not None
        )

    def clear_all(self) -> None:
        """Reset every field to its default. Emits :py:attr:`changed` at most once."""
        self._begin_batch()
        try:
            self.shot_type = _ENUM_ALL
            self.color_palette = _ENUM_ALL
            self.search_query = ""
            self.selected_custom_queries = set()
            self.min_duration = None
            self.max_duration = None
            self.aspect_ratio = _ENUM_ALL
            self.gaze_filter = None
            self.object_search = ""
            self.description_search = ""
            self.min_brightness = None
            self.max_brightness = None
            self.similarity_anchor_id = None
            self.similarity_scores = {}
        finally:
            self._end_batch()


def _coerce_custom_query(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        if value in {"All", "Match", "No Match"}:
            return set()
        return {value}
    if isinstance(value, (list, tuple, set, frozenset)):
        return {str(v).strip() for v in value if str(v).strip()}
    return set()
