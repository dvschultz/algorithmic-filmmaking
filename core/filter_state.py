"""Shared clip-filter state for the Cut and Analyze tab browsers.

``FilterState`` owns the values that drive :py:meth:`ClipBrowser._matches_filter`.
In Unit 1 of the filter-sidebar plan, it's a single source of truth that both
tabs point at — they share filter values by default. Later units (FilterSidebar)
subscribe to :py:attr:`changed` to know when to re-render their widgets.

The public ``apply_dict`` / ``to_dict`` contract mirrors today's
``ClipBrowser.apply_filters`` / ``get_active_filters`` dict shape, with Unit 4
extending four enum fields (``shot_type``, ``color_palette``, ``aspect_ratio``,
``gaze_filter``) from single-select to multi-select. Internally these are
``set[str]``; ``apply_dict`` coerces string / list / tuple / set / None inputs,
and ``to_dict`` emits ``None`` (no selection), the single string (one value)
or a sorted list (multiple values) for backward-compat with existing callers.
"""

from typing import Any, Iterable, Optional

from PySide6.QtCore import QObject, Signal


_ENUM_ALL = "All"
_ENUM_FIELDS = ("shot_type", "color_palette", "aspect_ratio", "gaze_filter")


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
        # Unit 5 additions
        "person_count",
        "has_audio",
        "has_transcript",
        "has_on_screen_text",
        "on_screen_text_search",
        "min_volume",
        "max_volume",
        "has_analysis_ops",
        "enabled_filter",
        "tag_note_search",
    )

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._batching = False
        self._dirty_during_batch = False
        # Multi-select enum fields — empty set means "All" (no filter)
        object.__setattr__(self, "shot_type", set())
        object.__setattr__(self, "color_palette", set())
        object.__setattr__(self, "aspect_ratio", set())
        object.__setattr__(self, "gaze_filter", set())
        object.__setattr__(self, "search_query", "")
        object.__setattr__(self, "selected_custom_queries", set())
        object.__setattr__(self, "min_duration", None)
        object.__setattr__(self, "max_duration", None)
        object.__setattr__(self, "object_search", "")
        object.__setattr__(self, "description_search", "")
        object.__setattr__(self, "min_brightness", None)
        object.__setattr__(self, "max_brightness", None)
        object.__setattr__(self, "similarity_anchor_id", None)
        object.__setattr__(self, "similarity_scores", {})
        # Unit 5 fields
        object.__setattr__(self, "person_count", None)  # (op, int) | None
        object.__setattr__(self, "has_audio", None)  # True / False / None (don't care)
        object.__setattr__(self, "has_transcript", None)
        object.__setattr__(self, "has_on_screen_text", None)
        object.__setattr__(self, "on_screen_text_search", "")
        object.__setattr__(self, "min_volume", None)
        object.__setattr__(self, "max_volume", None)
        object.__setattr__(self, "has_analysis_ops", set())
        object.__setattr__(self, "enabled_filter", None)
        object.__setattr__(self, "tag_note_search", "")

    # ── Dirty-tracking field assignment ─────────────────────────────

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self._FIELD_NAMES and hasattr(self, name):
            if name in _ENUM_FIELDS:
                value = _coerce_enum_set(value)
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
        Unknown keys are ignored. Enum fields accept a string (single-select
        backward compat), a list/tuple/set (multi-select), or ``None`` / ``"All"``
        (clear).
        """
        self._begin_batch()
        try:
            if "min_duration" in filters or "max_duration" in filters:
                self.min_duration = filters.get("min_duration")
                self.max_duration = filters.get("max_duration")

            if "aspect_ratio" in filters:
                self.aspect_ratio = filters["aspect_ratio"]

            if "shot_type" in filters:
                self.shot_type = filters["shot_type"]

            if "color_palette" in filters:
                self.color_palette = filters["color_palette"]

            if "search_query" in filters:
                value = filters["search_query"] or ""
                self.search_query = value.lower().strip()

            if "custom_query" in filters:
                self.selected_custom_queries = _coerce_custom_query(filters["custom_query"])

            if "gaze" in filters:
                self.gaze_filter = filters["gaze"]

            if "object_search" in filters:
                self.object_search = (filters["object_search"] or "").strip()

            if "description_search" in filters:
                self.description_search = (filters["description_search"] or "").strip()

            if "min_brightness" in filters or "max_brightness" in filters:
                self.min_brightness = filters.get("min_brightness")
                self.max_brightness = filters.get("max_brightness")

            if "similarity_anchor" in filters:
                self.similarity_anchor_id = filters["similarity_anchor"]

            # Unit 5 fields
            if "person_count" in filters:
                value = filters["person_count"]
                if value is None:
                    self.person_count = None
                elif isinstance(value, (list, tuple)) and len(value) == 2:
                    self.person_count = (str(value[0]), int(value[1]))
                else:
                    self.person_count = None

            if "has_audio" in filters:
                self.has_audio = _coerce_tribool(filters["has_audio"])
            if "has_transcript" in filters:
                self.has_transcript = _coerce_tribool(filters["has_transcript"])
            if "has_on_screen_text" in filters:
                self.has_on_screen_text = _coerce_tribool(filters["has_on_screen_text"])

            if "on_screen_text_search" in filters:
                self.on_screen_text_search = (filters["on_screen_text_search"] or "").strip()

            if "min_volume" in filters or "max_volume" in filters:
                self.min_volume = filters.get("min_volume")
                self.max_volume = filters.get("max_volume")

            if "has_analysis_ops" in filters:
                value = filters["has_analysis_ops"]
                if value is None:
                    self.has_analysis_ops = set()
                elif isinstance(value, str):
                    self.has_analysis_ops = {value} if value else set()
                elif isinstance(value, (list, tuple, set, frozenset)):
                    self.has_analysis_ops = {str(v) for v in value if str(v).strip()}
                else:
                    self.has_analysis_ops = set()

            if "enabled_filter" in filters:
                self.enabled_filter = _coerce_tribool(filters["enabled_filter"])

            if "tag_note_search" in filters:
                self.tag_note_search = (filters["tag_note_search"] or "").strip()
        finally:
            self._end_batch()

    def to_dict(self) -> dict:
        """Return the active-filters dict matching ``ClipBrowser.get_active_filters``.

        Enum fields emit ``None`` (no selection), the single string (one value)
        or a sorted list (multiple values) for backward-compat with callers
        that expect a string for single-select filters.
        """
        return {
            "shot_type": _emit_enum(self.shot_type),
            "color_palette": _emit_enum(self.color_palette),
            "search_query": self.search_query if self.search_query else None,
            "custom_query": (
                sorted(self.selected_custom_queries, key=str.lower)
                if self.selected_custom_queries
                else None
            ),
            "min_duration": self.min_duration,
            "max_duration": self.max_duration,
            "aspect_ratio": _emit_enum(self.aspect_ratio),
            "gaze": _emit_enum(self.gaze_filter),
            "object_search": self.object_search if self.object_search else None,
            "description_search": self.description_search if self.description_search else None,
            "min_brightness": self.min_brightness,
            "max_brightness": self.max_brightness,
            "similarity_anchor": self.similarity_anchor_id,
            # Unit 5 fields
            "person_count": list(self.person_count) if self.person_count else None,
            "has_audio": self.has_audio,
            "has_transcript": self.has_transcript,
            "has_on_screen_text": self.has_on_screen_text,
            "on_screen_text_search": self.on_screen_text_search or None,
            "min_volume": self.min_volume,
            "max_volume": self.max_volume,
            "has_analysis_ops": sorted(self.has_analysis_ops) if self.has_analysis_ops else None,
            "enabled_filter": self.enabled_filter,
            "tag_note_search": self.tag_note_search or None,
        }

    def has_active(self) -> bool:
        return (
            bool(self.shot_type)
            or bool(self.color_palette)
            or bool(self.aspect_ratio)
            or bool(self.gaze_filter)
            or bool(self.search_query)
            or bool(self.selected_custom_queries)
            or self.min_duration is not None
            or self.max_duration is not None
            or bool(self.object_search)
            or bool(self.description_search)
            or self.min_brightness is not None
            or self.max_brightness is not None
            or self.similarity_anchor_id is not None
            # Unit 5 fields
            or self.person_count is not None
            or self.has_audio is not None
            or self.has_transcript is not None
            or self.has_on_screen_text is not None
            or bool(self.on_screen_text_search)
            or self.min_volume is not None
            or self.max_volume is not None
            or bool(self.has_analysis_ops)
            or self.enabled_filter is not None
            or bool(self.tag_note_search)
        )

    def clear_all(self) -> None:
        """Reset every field to its default. Emits :py:attr:`changed` at most once."""
        self._begin_batch()
        try:
            self.shot_type = set()
            self.color_palette = set()
            self.aspect_ratio = set()
            self.gaze_filter = set()
            self.search_query = ""
            self.selected_custom_queries = set()
            self.min_duration = None
            self.max_duration = None
            self.object_search = ""
            self.description_search = ""
            self.min_brightness = None
            self.max_brightness = None
            self.similarity_anchor_id = None
            self.similarity_scores = {}
            # Unit 5 fields
            self.person_count = None
            self.has_audio = None
            self.has_transcript = None
            self.has_on_screen_text = None
            self.on_screen_text_search = ""
            self.min_volume = None
            self.max_volume = None
            self.has_analysis_ops = set()
            self.enabled_filter = None
            self.tag_note_search = ""
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


def _coerce_enum_set(value: Any) -> set[str]:
    """Normalize enum-field inputs to a ``set[str]``.

    Accepts ``None`` / empty / ``"All"`` (clears), a single string (one-item
    set), or any iterable of strings (drops ``"All"`` sentinel and empties).
    """
    if value is None:
        return set()
    if isinstance(value, str):
        if value == _ENUM_ALL or not value.strip():
            return set()
        return {value}
    if isinstance(value, (list, tuple, set, frozenset)):
        result: set[str] = set()
        for item in value:
            s = str(item).strip()
            if s and s != _ENUM_ALL:
                result.add(s)
        return result
    return set()


def _emit_enum(value: set[str]):
    """Shape an enum set for ``to_dict`` output (backward-compat with callers)."""
    if not value:
        return None
    if len(value) == 1:
        return next(iter(value))
    return sorted(value, key=str.lower)


def _coerce_tribool(value: Any) -> Optional[bool]:
    """Normalize has_audio / has_transcript / has_on_screen_text / enabled_filter inputs."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"yes", "true", "1"}:
            return True
        if lowered in {"no", "false", "0"}:
            return False
        return None
    return None
