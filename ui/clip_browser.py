"""Clip browser with thumbnail grid view."""

import logging
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np

from PySide6.QtCore import Qt, Signal, QMimeData, QRect, QTimer
from PySide6.QtGui import QPixmap, QDrag, QPainter, QColor, QKeyEvent, QAction
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QScrollArea,
    QGridLayout,
    QLabel,
    QFrame,
    QApplication,
    QComboBox,
    QLineEdit,
    QPushButton,
    QMenu,
    QToolButton,
    QGraphicsOpacityEffect,
    QRubberBand,
    QSizePolicy,
)

from core.analysis.color import (
    get_primary_hue,
    classify_color_palette,
    get_palette_display_name,
    COLOR_PALETTES,
)
from core.analysis.gaze import GAZE_CATEGORY_DISPLAY
from core.analysis.shots import get_display_name, SHOT_TYPES
from core.filter_state import FilterState
from core.film_glossary import get_badge_tooltip
from models.clip import Clip, Source
from models.cinematography import CinematographyAnalysis
from ui.gradient_glow import paint_gradient_glow, paint_card_body, RoundedTopLabel
from ui.theme import theme, UISizes, TypeScale, Spacing, Radii
from ui.widgets.range_slider import RangeSlider
from ui.widgets.source_group_header import SourceGroupHeader

# Reverse map: display label -> internal key (built once at module load)
_GAZE_DISPLAY_TO_KEY = {v: k for k, v in GAZE_CATEGORY_DISPLAY.items()}

# Resolve the has-analysis-of-type predicate at module load so the hot
# `_matches_filter` loop doesn't re-import on every clip check, and so that
# a missing dependency surfaces as a visible startup error rather than a
# silently-bypassed filter.
try:
    from core.analysis_availability import operation_is_complete_for_clip as _op_complete
except ImportError:  # pragma: no cover — analysis_availability is always bundled
    _op_complete = None


def _enum_str_view(value_set, *, empty):
    """Convert a FilterState enum set to a backward-compat string/list view.

    Used by ``ClipBrowser`` proxy properties so existing callers that read
    ``browser._gaze_filter`` / ``browser._current_filter`` still get a string
    for single-select state and ``empty`` (``"All"`` or ``None``) for the
    no-filter case.
    """
    if not value_set:
        return empty
    if len(value_set) == 1:
        return next(iter(value_set))
    return sorted(value_set, key=str.lower)


def _combo_text_for_enum(value_set, *, fallback):
    """Single-string representation of a multi-select enum for legacy QComboBox."""
    if not value_set:
        return fallback
    if len(value_set) == 1:
        return next(iter(value_set))
    return fallback

logger = logging.getLogger(__name__)

VIRTUALIZATION_THRESHOLD = 300
VIRTUAL_CARD_ROW_HEIGHT = 246
VIRTUAL_ROW_BUFFER = 4
VIRTUAL_SCROLL_REBUILD_DELAY_MS = 16
VIRTUAL_WIDGET_CACHE_LIMIT = 192
THUMBNAIL_PIXMAP_CACHE_LIMIT = 512

_THUMBNAIL_PIXMAP_CACHE: OrderedDict[tuple[str, int, int, int], QPixmap] = OrderedDict()


def _scaled_thumbnail_from_cache(path: Path, width: int, height: int) -> QPixmap:
    """Return a cached scaled thumbnail pixmap for card rendering."""
    try:
        stat = path.stat()
        mtime_ns = stat.st_mtime_ns
    except OSError:
        mtime_ns = 0

    key = (str(path), mtime_ns, width, height)
    cached = _THUMBNAIL_PIXMAP_CACHE.get(key)
    if cached is not None:
        _THUMBNAIL_PIXMAP_CACHE.move_to_end(key)
        return cached

    pixmap = QPixmap(str(path))
    if pixmap.isNull():
        return pixmap

    scaled = pixmap.scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    _THUMBNAIL_PIXMAP_CACHE[key] = scaled
    while len(_THUMBNAIL_PIXMAP_CACHE) > THUMBNAIL_PIXMAP_CACHE_LIMIT:
        _THUMBNAIL_PIXMAP_CACHE.popitem(last=False)
    return scaled


def _sort_key_by_color(clip: Clip) -> float:
    """Module-level color sort key for both virtual and non-virtual paths."""
    return get_primary_hue(clip.dominant_colors) if clip.dominant_colors else 0.0


def _sort_key_by_duration(clip: Clip, source: Source) -> float:
    """Module-level duration sort key (seconds) for both browser paths."""
    return clip.duration_seconds(source.fps)


def _sort_key_by_timeline(clip: Clip) -> int:
    """Module-level timeline sort key (start frame) for both browser paths."""
    return clip.start_frame


def clear_thumbnail_pixmap_cache() -> None:
    """Flush the module-level scaled-thumbnail pixmap cache.

    Should be called when a project is closed/reopened so stale pixmaps from
    the previous project don't linger and waste memory.
    """
    _THUMBNAIL_PIXMAP_CACHE.clear()


def get_latest_custom_query_results(custom_queries: list[dict] | None) -> dict[str, dict]:
    """Collapse append-only custom query history to the latest result per query."""
    latest_results: dict[str, dict] = {}
    for query_result in custom_queries or []:
        query = str(query_result.get("query") or "").strip()
        if not query:
            continue
        latest_results[query] = query_result
    return latest_results


class ColorSwatchBar(QWidget):
    """Widget that displays dominant colors as horizontal stripes."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.colors: list[tuple[int, int, int]] = []
        self.setFixedSize(220, 10)

    def set_colors(self, colors: list[tuple[int, int, int]] | None):
        """Set the colors to display."""
        self.colors = colors or []
        self.update()

    def paintEvent(self, event):
        """Paint the color stripes."""
        if not self.colors:
            return

        painter = QPainter(self)
        width = self.width()
        height = self.height()

        # Calculate stripe width based on number of colors
        n_colors = len(self.colors)
        stripe_width = width / n_colors

        for i, rgb in enumerate(self.colors):
            color = QColor(rgb[0], rgb[1], rgb[2])
            painter.fillRect(
                int(i * stripe_width),
                0,
                int(stripe_width) + 1,  # +1 to avoid gaps
                height,
                color,
            )

        painter.end()


class _SelectionContainer(QWidget):
    """Grid container that delegates empty-space marquee events to its browser."""

    def __init__(self, browser: "ClipBrowser"):
        super().__init__()
        self._browser = browser

    def mousePressEvent(self, event):
        self._browser._on_container_mouse_press(event)

    def mouseMoveEvent(self, event):
        self._browser._on_container_mouse_move(event)

    def mouseReleaseEvent(self, event):
        self._browser._on_container_mouse_release(event)


class ClipThumbnail(QFrame):
    """Individual clip thumbnail widget."""

    clicked = Signal(object)  # Clip
    double_clicked = Signal(object)  # Clip
    drag_started = Signal(object)  # Clip
    view_details_requested = Signal(object, object)  # Clip, Source
    export_requested = Signal(object, object)  # Clip, Source
    find_similar_requested = Signal(object)  # Clip

    def __init__(self, clip: Clip, source: Source, drag_enabled: bool = False):
        super().__init__()
        self.clip = clip
        self.source = source
        self.selected = False
        self._drag_enabled = drag_enabled
        self._drag_start_pos = None
        self._show_glow = False
        self._glow_colors: list[tuple[int, int, int]] = []

        self.setFrameStyle(QFrame.NoFrame)
        self.setFixedSize(UISizes.GRID_CARD_MAX_WIDTH, 230)
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.StrongFocus)
        clip_name = clip.name if clip.name else f"Clip {clip.id[:8]}"
        duration = clip.duration_seconds(source.fps)
        self.setAccessibleName(clip_name)
        self.setAccessibleDescription(f"Clip from {source.filename}, duration {duration:.1f}s")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        # Thumbnail container (for overlay positioning)
        self.thumb_container = QWidget()
        self.thumb_container.setFixedSize(220, 124)
        thumb_layout = QVBoxLayout(self.thumb_container)
        thumb_layout.setContentsMargins(0, 0, 0, 0)
        thumb_layout.setSpacing(0)

        # Thumbnail image (rounded top corners to match card radius)
        self.thumbnail_label = RoundedTopLabel(Radii.MD, self.thumb_container)
        self.thumbnail_label.setFixedSize(220, 124)
        self.thumbnail_label.setAlignment(Qt.AlignCenter)
        self.thumbnail_label.setStyleSheet(f"background-color: {theme().thumbnail_background};")

        # Opacity effect for disabled state dimming
        self._thumbnail_opacity = QGraphicsOpacityEffect(self.thumbnail_label)
        self._thumbnail_opacity.setOpacity(1.0)
        self.thumbnail_label.setGraphicsEffect(self._thumbnail_opacity)

        if clip.thumbnail_path and clip.thumbnail_path.exists():
            self._load_thumbnail(clip.thumbnail_path)
        else:
            self.thumbnail_label.setText("Loading...")

        # Transcript overlay (positioned on top of thumbnail)
        self.transcript_overlay = QLabel(self.thumb_container)
        self.transcript_overlay.setFixedSize(220, 124)
        self.transcript_overlay.setAlignment(Qt.AlignCenter)
        self.transcript_overlay.setWordWrap(True)
        self.transcript_overlay.setStyleSheet(
            f"background-color: {theme().overlay_dark}; "
            "color: white; "
            f"font-size: {TypeScale.XS}px; "
            f"padding: {Spacing.SM}px; "
            f"border-radius: {Radii.MD}px;"
        )
        self.transcript_overlay.setGeometry(0, 0, 220, 124)
        self.transcript_overlay.raise_()  # Ensure overlay is on top
        self.transcript_overlay.setVisible(False)
        self._update_transcript_overlay()

        layout.addWidget(self.thumb_container)

        # Color swatch bar
        self.color_bar = ColorSwatchBar()
        if clip.dominant_colors:
            self.color_bar.set_colors(clip.dominant_colors)
        layout.addWidget(self.color_bar)

        # Info row: duration and shot type
        info_layout = QHBoxLayout()
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(4)

        # Duration label
        duration = clip.duration_seconds(source.fps)
        self.duration_label = QLabel(self._format_duration(duration))
        self.duration_label.setAlignment(Qt.AlignLeft)
        self.duration_label.setStyleSheet(f"font-size: {TypeScale.SM}px; color: {theme().text_muted};")
        info_layout.addWidget(self.duration_label)

        info_layout.addStretch()

        # Shot type label
        self.shot_type_label = QLabel()
        self.shot_type_label.setAlignment(Qt.AlignRight)
        self._apply_shot_type_badge_style()
        if clip.shot_type:
            self.shot_type_label.setText(get_display_name(clip.shot_type))
            # Add film language tooltip
            tooltip = get_badge_tooltip(clip.shot_type)
            if tooltip:
                self.shot_type_label.setToolTip(tooltip)
        else:
            self.shot_type_label.setVisible(False)

        # Gaze direction label
        self.gaze_label = QLabel()
        self.gaze_label.setAlignment(Qt.AlignRight)
        self._apply_gaze_badge_style()
        if clip.gaze_category:
            self.set_gaze(clip.gaze_category)
        else:
            self.gaze_label.setVisible(False)

        # Container for one badge per matching custom query (flows right-aligned)
        self.custom_query_container = QWidget()
        self.custom_query_container.setStyleSheet("background: transparent; border: none;")
        custom_query_layout = QHBoxLayout(self.custom_query_container)
        custom_query_layout.setContentsMargins(0, 0, 0, 0)
        custom_query_layout.setSpacing(Spacing.XXS)
        custom_query_layout.addStretch()
        self._custom_query_badges: list[QLabel] = []
        info_layout.addWidget(self.custom_query_container)
        self._update_custom_query_badge()
        info_layout.addWidget(self.gaze_label)
        info_layout.addWidget(self.shot_type_label)

        layout.addLayout(info_layout)

        # Cinematography badges row (shown when rich analysis is available)
        self.cinematography_container = QWidget()
        self.cinematography_container.setVisible(False)
        cinematography_layout = QHBoxLayout(self.cinematography_container)
        cinematography_layout.setContentsMargins(0, 0, 0, 0)
        cinematography_layout.setSpacing(3)

        # Create badge labels (initially hidden)
        self._cinematography_badges: list[QLabel] = []

        # Show badges if clip has cinematography data
        if clip.cinematography:
            self._update_cinematography_badges(clip.cinematography)

        layout.addWidget(self.cinematography_container)

        self._update_style()

        # Connect to theme changes
        if theme().changed:
            theme().changed.connect(self._refresh_theme)

    def _load_thumbnail(self, path: Path):
        """Load thumbnail image."""
        pixmap = _scaled_thumbnail_from_cache(path, 220, 124)
        if not pixmap.isNull():
            self.thumbnail_label.setPixmap(pixmap)

    def _format_duration(self, seconds: float) -> str:
        """Format duration as MM:SS.ms"""
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}:{secs:05.2f}"

    @property
    def disabled(self) -> bool:
        """Whether this clip is disabled (reads from model)."""
        return self.clip.disabled

    def set_selected(self, selected: bool):
        """Set selection state."""
        self.selected = selected
        self._update_style()

    def set_thumbnail(self, path: Path):
        """Update the thumbnail image."""
        self.clip.thumbnail_path = path
        self._load_thumbnail(path)

    def _update_style(self):
        """Update visual style based on state."""
        if self.disabled:
            self._thumbnail_opacity.setOpacity(0.5)
            self._show_glow = False
            self.setStyleSheet("""
                ClipThumbnail {{
                    background-color: transparent;
                    border: none;
                }}
            """)
            self.update()
            return

        self._thumbnail_opacity.setOpacity(1.0)
        if self.selected:
            # Content-aware gradient glow for selected state
            if self.clip and self.clip.dominant_colors:
                self._glow_colors = self.clip.dominant_colors[:3]
            else:
                self._glow_colors = theme().gradient.default_colors
            self._show_glow = True
            self.setStyleSheet("ClipThumbnail { background-color: transparent; border: none; }")
        else:
            self._show_glow = False
            self.setStyleSheet(f"""
                ClipThumbnail {{
                    background-color: {theme().card_background};
                    border: 1px solid {theme().card_border};
                    border-radius: {Radii.MD}px;
                }}
                ClipThumbnail:hover {{
                    background-color: {theme().card_hover};
                    border: 1px solid {theme().border_focus};
                    border-radius: {Radii.MD}px;
                }}
            """)
        self.update()

    def paintEvent(self, event):
        """Custom paint for gradient glow on selected/disabled cards."""
        super().paintEvent(event)
        if self._show_glow or self.disabled:
            from PySide6.QtCore import QRectF
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            card_rect = QRectF(self.rect())
            grad = theme().gradient

            if self.disabled:
                paint_card_body(
                    painter, card_rect,
                    QColor(theme().background_tertiary),
                    QColor(theme().card_border),
                    Radii.MD,
                )
            elif self._show_glow:
                paint_gradient_glow(
                    painter, card_rect,
                    self._glow_colors,
                    grad.active_opacity,
                    grad.glow_spread,
                    Radii.MD,
                )
                paint_card_body(
                    painter, card_rect,
                    QColor(theme().card_background),
                    QColor(theme().accent_blue),
                    Radii.MD, 2,
                )
            painter.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_start_pos = event.pos()
        self.clicked.emit(self.clip)

    def mouseMoveEvent(self, event):
        if not self._drag_enabled or not self._drag_start_pos:
            return

        # Check if drag threshold met
        if (event.pos() - self._drag_start_pos).manhattanLength() < QApplication.startDragDistance():
            return

        # Start drag
        drag = QDrag(self)
        mime_data = QMimeData()
        mime_data.setData("application/x-clip-id", self.clip.id.encode())
        drag.setMimeData(mime_data)

        # Set drag pixmap (thumbnail)
        if self.thumbnail_label.pixmap():
            drag.setPixmap(self.thumbnail_label.pixmap().scaled(80, 45, Qt.KeepAspectRatio))

        # Execute drag
        result = drag.exec_(Qt.CopyAction)
        if result == Qt.CopyAction:
            self.drag_started.emit(self.clip)

    def mouseReleaseEvent(self, event):
        self._drag_start_pos = None

    def mouseDoubleClickEvent(self, event):
        # Emit signal to undo the selection toggle from the first click
        # (double-click should not affect selection, only open details)
        self.double_clicked.emit(self.clip)
        # Trigger view details on double-click
        self.view_details_requested.emit(self.clip, self.source)

    def contextMenuEvent(self, event):
        """Show context menu with clip actions."""
        menu = self._build_context_menu()
        menu.exec_(event.globalPos())

    def keyPressEvent(self, event: QKeyEvent):
        """Handle keyboard events for accessibility."""
        if event.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            self.clicked.emit(self.clip)
        elif event.key() == Qt.Key_Menu or (
            event.key() == Qt.Key_F10 and event.modifiers() & Qt.ShiftModifier
        ):
            # Show context menu at widget center
            center = self.rect().center()
            global_pos = self.mapToGlobal(center)
            menu = self._build_context_menu()
            menu.exec_(global_pos)
        else:
            super().keyPressEvent(event)

    def _build_context_menu(self) -> QMenu:
        """Build the clip context menu."""
        menu = QMenu(self)
        view_details_action = menu.addAction("View Details")
        view_details_action.triggered.connect(
            lambda: self.view_details_requested.emit(self.clip, self.source)
        )
        export_action = menu.addAction("Export Clip...")
        export_action.triggered.connect(self._emit_export_requested)
        find_similar_action = menu.addAction("Find Similar")
        find_similar_action.triggered.connect(
            lambda: self.find_similar_requested.emit(self.clip)
        )
        return menu

    def _emit_export_requested(self) -> None:
        """Emit an export request for this clip."""
        self.export_requested.emit(self.clip, self.source)

    def focusInEvent(self, event):
        """Handle focus gained - add visual indicator."""
        super().focusInEvent(event)
        if not self.selected:
            self.setStyleSheet(f"""
                ClipThumbnail {{
                    background-color: {theme().card_hover};
                    border: 2px solid {theme().border_focus};
                    border-radius: {Radii.MD}px;
                }}
            """)

    def focusOutEvent(self, event):
        """Handle focus lost - remove visual indicator."""
        super().focusOutEvent(event)
        self._update_style()

    def set_drag_enabled(self, enabled: bool):
        """Enable or disable dragging."""
        self._drag_enabled = enabled

    def set_colors(self, colors: list[tuple[int, int, int]] | None):
        """Set the dominant colors for this clip."""
        self.clip.dominant_colors = colors
        self.color_bar.set_colors(colors)

    def set_shot_type(self, shot_type: str | None):
        """Set the shot type for this clip."""
        self.clip.shot_type = shot_type
        if shot_type:
            self.shot_type_label.setText(get_display_name(shot_type))
            # Add film language tooltip
            tooltip = get_badge_tooltip(shot_type)
            if tooltip:
                self.shot_type_label.setToolTip(tooltip)
            self._apply_shot_type_badge_style()
            self.shot_type_label.setVisible(True)
        else:
            self.shot_type_label.setVisible(False)

    def set_gaze(self, category: str | None):
        """Set the gaze direction category for this clip.

        Args:
            category: Gaze category string (e.g. 'looking_left') or None to hide.
        """
        from core.analysis.gaze import GAZE_CATEGORY_SHORT, GAZE_CATEGORY_DISPLAY

        self.clip.gaze_category = category
        if category:
            self.gaze_label.setText(GAZE_CATEGORY_SHORT.get(category, "?"))
            display = GAZE_CATEGORY_DISPLAY.get(category, category)
            self.gaze_label.setToolTip(f"Gaze: {display}")
            self._apply_gaze_badge_style()
            self.gaze_label.setVisible(True)
        else:
            self.gaze_label.setVisible(False)

    def set_transcript(self, segments: list | None):
        """Set the transcript segments for this clip."""
        self.clip.transcript = segments
        self._update_transcript_overlay()

    def set_extracted_text(self, texts: list | None):
        """Set the extracted text for this clip."""
        self.clip.extracted_texts = texts

    def set_custom_queries(self, custom_queries: list[dict] | None):
        """Set the custom query results for this clip."""
        self.clip.custom_queries = custom_queries
        self._update_custom_query_badge()

    def set_cinematography(self, cinematography: CinematographyAnalysis | None):
        """Set the cinematography analysis for this clip."""
        self.clip.cinematography = cinematography
        self._update_cinematography_badges(cinematography)

    def _update_cinematography_badges(self, cinematography: CinematographyAnalysis | None):
        """Update the cinematography badge display."""
        layout = self.cinematography_container.layout()

        # Clear all existing items from the layout (widgets AND stretches)
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._cinematography_badges.clear()

        if not cinematography:
            self.cinematography_container.setVisible(False)
            return

        # Get display badges from cinematography (key, display_text tuples)
        badges = cinematography.get_display_badges_formatted()

        if not badges:
            self.cinematography_container.setVisible(False)
            return

        # Create badge labels (max 4 badges to fit in width)
        badge_style = (
            f"font-size: {TypeScale.XS}px; color: {theme().text_muted}; "
            f"background-color: {theme().card_border}; "
            f"border-radius: {Radii.SM}px; padding: {Spacing.XXS}px {Spacing.XS}px;"
        )

        for key, display_text in badges[:4]:
            badge = QLabel(display_text)
            badge.setStyleSheet(badge_style)

            # Add film language tooltip using the key for lookup
            tooltip_html = get_badge_tooltip(key)
            if tooltip_html:
                badge.setToolTip(tooltip_html)

            layout.addWidget(badge)
            self._cinematography_badges.append(badge)

        layout.addStretch()
        self.cinematography_container.setVisible(True)

    def _update_transcript_overlay(self):
        """Update the transcript overlay text."""
        if self.clip.transcript:
            # Get first ~100 chars of transcript
            full_text = self.clip.get_transcript_text()
            if len(full_text) > 100:
                display_text = full_text[:100] + "..."
            else:
                display_text = full_text
            self.transcript_overlay.setText(f'"{display_text}"')
        else:
            self.transcript_overlay.setText("")

    def enterEvent(self, event):
        """Show transcript overlay on hover if transcript exists."""
        if self.clip.transcript and self.clip.get_transcript_text():
            self.transcript_overlay.setVisible(True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Hide transcript overlay when not hovering."""
        self.transcript_overlay.setVisible(False)
        super().leaveEvent(event)

    def _refresh_theme(self):
        """Refresh all themed styles when theme changes."""
        self._update_style()
        # Update thumbnail background
        self.thumbnail_label.setStyleSheet(f"background-color: {theme().thumbnail_background};")
        # Update duration label
        self.duration_label.setStyleSheet(f"font-size: {TypeScale.SM}px; color: {theme().text_muted};")
        # Update shot type badge
        if self.clip.shot_type:
            self._apply_shot_type_badge_style()
        # Update gaze badge
        if self.clip.gaze_category:
            self._apply_gaze_badge_style()
        self._update_custom_query_badge()

    def _apply_shot_type_badge_style(self):
        """Apply the shared shot type badge style."""
        self.shot_type_label.setStyleSheet(
            f"font-size: {TypeScale.XS}px; color: {theme().text_inverted}; background-color: {theme().shot_type_badge}; "
            f"border-radius: {Radii.SM}px; padding: {Spacing.XXS}px {Spacing.XS}px;"
        )

    def _apply_gaze_badge_style(self):
        """Apply the gaze direction badge style."""
        self.gaze_label.setStyleSheet(
            f"font-size: {TypeScale.XS}px; color: {theme().text_inverted}; background-color: {theme().gaze_badge}; "
            f"border-radius: {Radii.SM}px; padding: {Spacing.XXS}px {Spacing.XS}px;"
        )

    def _custom_query_tooltip(self) -> str:
        """Build a tooltip summary for custom query results."""
        latest_results = get_latest_custom_query_results(self.clip.custom_queries)
        if not latest_results:
            return ""

        lines = ["Custom query results:"]
        for query_result in latest_results.values():
            marker = "YES" if query_result.get("match") else "NO"
            confidence = float(query_result.get("confidence", 0.0))
            model = query_result.get("model", "")
            query = query_result.get("query", "")
            suffix = f" ({confidence:.0%}"
            if model:
                suffix += f", {model}"
            suffix += ")"
            lines.append(f"{marker}: {query}{suffix}")
        return "\n".join(lines)

    def _update_custom_query_badge(self):
        """Render one badge per matching custom query; hide the container when none match."""
        # Clear existing badge widgets to avoid leaks on repeated updates.
        for badge in self._custom_query_badges:
            badge.deleteLater()
        self._custom_query_badges = []

        latest_results = get_latest_custom_query_results(self.clip.custom_queries)
        matching = [
            result for result in latest_results.values()
            if bool(result.get("match"))
        ]

        if not matching:
            self.custom_query_container.setVisible(False)
            self.custom_query_container.setToolTip("")
            return

        # Style matches Collect-tab CUT/ANALYZED badges (ui/source_thumbnail.py).
        badge_style = (
            f"font-size: {TypeScale.XS}px; "
            f"color: {theme().badge_analyzed_text}; "
            f"background-color: {theme().badge_analyzed}; "
            f"border-radius: {Radii.SM}px; "
            f"padding: {Spacing.XXS}px {Spacing.SM}px;"
        )
        tooltip = self._custom_query_tooltip()
        layout = self.custom_query_container.layout()
        for result in matching:
            query = str(result.get("query") or "").strip()
            if not query:
                continue
            badge = QLabel(query)
            badge.setAlignment(Qt.AlignCenter)
            badge.setStyleSheet(badge_style)
            badge.setToolTip(tooltip)
            layout.addWidget(badge)
            self._custom_query_badges.append(badge)

        self.custom_query_container.setToolTip(tooltip)
        self.custom_query_container.setVisible(bool(self._custom_query_badges))


class ClipBrowser(QWidget):
    """Grid browser for viewing detected clips."""

    clip_selected = Signal(object)  # Clip
    clip_double_clicked = Signal(object)  # Clip
    clip_dragged_to_timeline = Signal(object)  # Clip
    selection_changed = Signal(list)  # list[str] selected clip IDs
    filters_changed = Signal()  # Emitted when any filter changes
    view_details_requested = Signal(object, object)  # Clip, Source - request to show clip details
    export_requested = Signal(object, object)  # Clip, Source - request to export a single clip
    disabled_clips_changed = Signal(list)  # list[str] clip IDs whose disabled state was toggled

    _MIN_COLUMNS = 2

    # Aspect ratio tolerance ranges (5% tolerance)
    ASPECT_RATIOS = {
        "16:9": (1.778, 1.69, 1.87),   # 1.778 ± 5%
        "4:3": (1.333, 1.27, 1.40),     # 1.333 ± 5%
        "9:16": (0.5625, 0.53, 0.59),   # 0.5625 ± 5%
    }

    # ── Filter-state proxy properties ───────────────────────────────
    # Each pair delegates to `self._filter_state.<field>`. Having these
    # named with underscores preserves backward compat with tests that
    # assign `browser._gaze_filter = "at_camera"` directly.

    @property
    def _current_filter(self):
        return _enum_str_view(self._filter_state.shot_type, empty="All")

    @_current_filter.setter
    def _current_filter(self, value) -> None:
        self._filter_state.shot_type = value

    @property
    def _current_color_filter(self):
        return _enum_str_view(self._filter_state.color_palette, empty="All")

    @_current_color_filter.setter
    def _current_color_filter(self, value) -> None:
        self._filter_state.color_palette = value

    @property
    def _current_search_query(self) -> str:
        return self._filter_state.search_query

    @_current_search_query.setter
    def _current_search_query(self, value: str) -> None:
        self._filter_state.search_query = value or ""

    @property
    def _selected_custom_queries(self) -> set:
        return self._filter_state.selected_custom_queries

    @_selected_custom_queries.setter
    def _selected_custom_queries(self, value) -> None:
        self._filter_state.selected_custom_queries = set(value) if value else set()

    @property
    def _min_duration(self):
        return self._filter_state.min_duration

    @_min_duration.setter
    def _min_duration(self, value) -> None:
        self._filter_state.min_duration = value

    @property
    def _max_duration(self):
        return self._filter_state.max_duration

    @_max_duration.setter
    def _max_duration(self, value) -> None:
        self._filter_state.max_duration = value

    @property
    def _aspect_ratio_filter(self):
        return _enum_str_view(self._filter_state.aspect_ratio, empty="All")

    @_aspect_ratio_filter.setter
    def _aspect_ratio_filter(self, value) -> None:
        self._filter_state.aspect_ratio = value

    @property
    def _gaze_filter(self):
        return _enum_str_view(self._filter_state.gaze_filter, empty=None)

    @_gaze_filter.setter
    def _gaze_filter(self, value) -> None:
        self._filter_state.gaze_filter = value

    @property
    def _object_search(self) -> str:
        return self._filter_state.object_search

    @_object_search.setter
    def _object_search(self, value: str) -> None:
        self._filter_state.object_search = value or ""

    @property
    def _description_search(self) -> str:
        return self._filter_state.description_search

    @_description_search.setter
    def _description_search(self, value: str) -> None:
        self._filter_state.description_search = value or ""

    @property
    def _min_brightness(self):
        return self._filter_state.min_brightness

    @_min_brightness.setter
    def _min_brightness(self, value) -> None:
        self._filter_state.min_brightness = value

    @property
    def _max_brightness(self):
        return self._filter_state.max_brightness

    @_max_brightness.setter
    def _max_brightness(self, value) -> None:
        self._filter_state.max_brightness = value

    @property
    def _similarity_anchor_id(self):
        return self._filter_state.similarity_anchor_id

    @_similarity_anchor_id.setter
    def _similarity_anchor_id(self, value) -> None:
        self._filter_state.similarity_anchor_id = value

    @property
    def _similarity_scores(self) -> dict:
        return self._filter_state.similarity_scores

    @_similarity_scores.setter
    def _similarity_scores(self, value) -> None:
        self._filter_state.similarity_scores = dict(value) if value else {}

    @property
    def COLUMNS(self) -> int:
        """Calculate number of columns based on available width."""
        return self._calculate_columns()

    def __init__(self, filter_state: Optional[FilterState] = None):
        super().__init__()
        self.thumbnails: list[ClipThumbnail] = []
        self._thumbnail_by_id: dict[str, ClipThumbnail] = {}  # O(1) lookup by clip_id
        self.selected_clips: set[str] = set()  # clip ids
        self._drag_enabled = False
        self._source_lookup: dict[str, Source] = {}  # clip_id -> Source
        self._virtual_mode = False
        self._virtual_entries: list[tuple[Clip, Source]] = []
        self._virtual_display_rows: list[tuple[str, object]] = []
        self._virtual_top_spacer: QWidget | None = None
        self._virtual_bottom_spacer: QWidget | None = None
        self._virtual_render_range: tuple[int, int] | None = None
        self._virtual_rows_dirty = True
        self._virtual_rows_columns = 0
        self._virtual_widget_cache: OrderedDict[str, ClipThumbnail] = OrderedDict()
        self._rebuild_pending: bool = False

        # Shared filter state — all filter values live here. Proxy properties
        # on this class (e.g., `_current_filter`, `_gaze_filter`) read/write
        # through `_filter_state` so existing tests that mutate private attrs
        # directly keep working while the shared-state migration (Unit 1 of
        # `docs/plans/2026-04-21-001-feat-comprehensive-clip-filter-system-plan.md`)
        # is introduced. Both tabs' browsers share one `FilterState` instance
        # when MainWindow injects it here.
        self._filter_state = filter_state if filter_state is not None else FilterState()
        # When anything mutates FilterState (sidebar chip toggle, tribool radio,
        # typeahead commit, programmatic apply_dict), re-filter the grid and
        # forward the change to tab-level listeners. Without this, Unit 4+
        # sidebar controls are silent no-ops — they update the shared state
        # but the grid never refreshes. The legacy hidden widgets still call
        # _rebuild_grid directly from their handlers, so their paths remain
        # redundant but harmless (dirty-check in FilterState.__setattr__
        # prevents duplicate rebuilds when both fire for the same change).
        self._filter_state.changed.connect(self._on_filter_state_changed)

        self._custom_query_filter_actions: dict[str, QAction] = {}
        self._updating_custom_query_filter_menu = False
        self._last_column_count = 0  # Track columns to detect changes on resize

        self._filter_panel_visible = False

        # Source grouping state
        self._group_expanded_state: dict[str, bool] = {}  # source_id -> is_expanded
        self._source_headers: dict[str, SourceGroupHeader] = {}  # source_id -> header widget
        self._marquee_origin = None
        self._marquee_active = False
        self._marquee_additive = False
        self._marquee_base_selection: set[str] = set()

        self._setup_ui()

    def _setup_ui(self):
        """Set up the UI."""
        # Enable keyboard focus for clip details shortcuts
        self.setFocusPolicy(Qt.StrongFocus)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Header with non-filter controls only. Filter UI is now owned by
        # FilterSidebar (see ui/widgets/filter_sidebar.py); the legacy
        # filter widgets below are still instantiated for backward compat
        # with tests that manipulate `browser.filter_combo`, etc. directly,
        # but they are not added to any visible layout.
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(8, 8, 8, 4)

        header_label = QLabel("Detected Scenes")
        header_label.setStyleSheet(f"font-weight: bold; font-size: {TypeScale.MD}px;")
        header_layout.addWidget(header_label)

        header_layout.addStretch()

        # Expand/Collapse all buttons (non-filter)
        self.expand_all_btn = QPushButton("Expand All")
        self.expand_all_btn.setToolTip("Expand all source groups")
        self.expand_all_btn.clicked.connect(self.expand_all_groups)
        header_layout.addWidget(self.expand_all_btn)

        self.collapse_all_btn = QPushButton("Collapse All")
        self.collapse_all_btn.setToolTip("Collapse all source groups")
        self.collapse_all_btn.clicked.connect(self.collapse_all_groups)
        header_layout.addWidget(self.collapse_all_btn)

        header_layout.addSpacing(16)

        # Sort dropdown (non-filter)
        sort_label = QLabel("Order:")
        header_layout.addWidget(sort_label)

        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["Timeline", "Color", "Duration"])
        self.sort_combo.setFixedWidth(100)
        self.sort_combo.currentTextChanged.connect(self._on_sort_changed)
        header_layout.addWidget(self.sort_combo)

        layout.addLayout(header_layout)

        # ── Legacy filter widgets (parented but not added to visible layout) ──
        # These instances preserve backward compat with pre-sidebar tests that
        # read/write `browser.filter_combo`, `browser.gaze_combo`, etc.
        self.filters_btn = QPushButton("Filters", parent=self)
        self.filters_btn.setCheckable(True)
        self.filters_btn.setVisible(False)
        self.filters_btn.clicked.connect(self._toggle_filter_panel)

        self.filter_combo = QComboBox(parent=self)
        self.filter_combo.addItems(["All"] + [get_display_name(st) for st in SHOT_TYPES])
        self.filter_combo.setVisible(False)
        self.filter_combo.currentTextChanged.connect(self._on_filter_changed)

        self.color_filter_combo = QComboBox(parent=self)
        self.color_filter_combo.addItems(["All"] + [get_palette_display_name(cp) for cp in COLOR_PALETTES])
        self.color_filter_combo.setVisible(False)
        self.color_filter_combo.currentTextChanged.connect(self._on_color_filter_changed)

        self.search_input = QLineEdit(parent=self)
        self.search_input.setVisible(False)
        self.search_input.textChanged.connect(self._on_search_changed)

        self.custom_query_filter_btn = QToolButton(parent=self)
        self.custom_query_filter_btn.setPopupMode(QToolButton.InstantPopup)
        self.custom_query_filter_btn.setVisible(False)
        self.custom_query_filter_menu = QMenu(self.custom_query_filter_btn)
        self.custom_query_filter_btn.setMenu(self.custom_query_filter_menu)
        self._sync_custom_query_filter_options()

        # Legacy filter panel (slider widgets, gaze combo, etc.) — created
        # but not added to the visible layout. Tests still access its children
        # like `browser.gaze_combo` directly.
        self.filter_panel = self._create_filter_panel()
        self.filter_panel.setParent(self)
        self.filter_panel.setVisible(False)

        # Scroll area for thumbnails
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.verticalScrollBar().valueChanged.connect(self._on_scroll_changed)

        # Container for grid
        self.container = _SelectionContainer(self)
        self.grid = QGridLayout(self.container)
        self.grid.setSpacing(UISizes.GRID_GUTTER)
        self.grid.setContentsMargins(UISizes.GRID_MARGIN, UISizes.GRID_MARGIN, UISizes.GRID_MARGIN, UISizes.GRID_MARGIN)
        self.grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self._selection_band = QRubberBand(QRubberBand.Rectangle, self.container)
        self._selection_band.hide()

        self.scroll.setWidget(self.container)
        layout.addWidget(self.scroll)

        # Placeholder for empty state (not in grid initially - will be added when needed)
        self.empty_label = QLabel("No scenes detected yet.\nDrop a video or click Import.")
        self.empty_label.setAlignment(Qt.AlignCenter)
        self._empty_label_in_grid = False
        self._show_empty_state()  # Show initially since no clips

    def _calculate_columns(self) -> int:
        """Calculate number of columns that fit in the available width."""
        if not hasattr(self, 'scroll'):
            return self._MIN_COLUMNS
        viewport_width = self.scroll.viewport().width()
        if viewport_width <= 0:
            return self._MIN_COLUMNS
        card_width = UISizes.GRID_CARD_MAX_WIDTH
        gutter = UISizes.GRID_GUTTER
        margin = UISizes.GRID_MARGIN
        available = viewport_width - 2 * margin
        cols = max(self._MIN_COLUMNS, (available + gutter) // (card_width + gutter))
        return cols

    def resizeEvent(self, event):
        """Recalculate grid columns when widget is resized."""
        super().resizeEvent(event)
        new_cols = self._calculate_columns()
        if new_cols != self._last_column_count:
            self._last_column_count = new_cols
            if self.thumbnails or self._virtual_entries:
                self._rebuild_grid()
        else:
            self._last_column_count = new_cols

    def refresh_layout(self):
        """Force a grid rebuild using the current viewport width."""
        if not self.thumbnails and not self._virtual_entries:
            return
        if self._virtual_mode and self.thumbnails:
            current_columns = self._calculate_columns()
            if current_columns == self._last_column_count and not self._virtual_rows_dirty:
                return
        self._last_column_count = 0
        self._rebuild_grid()

    def _on_scroll_changed(self, _value: int) -> None:
        """Refresh realized cards as the virtual browser scrolls."""
        if self._virtual_mode and not getattr(self, "_rebuild_pending", False):
            if self._virtual_display_rows and not self._virtual_rows_dirty:
                first_row, last_row = self._current_virtual_render_range(
                    len(self._virtual_display_rows)
                )
                if (first_row, last_row) == self._virtual_render_range:
                    return
            self._rebuild_grid(
                invalidate_virtual_rows=False,
                delay_ms=VIRTUAL_SCROLL_REBUILD_DELAY_MS,
            )

    def _show_empty_state(self):
        """Show the empty state label in the grid."""
        if not self._empty_label_in_grid:
            self.grid.addWidget(self.empty_label, 0, 0, 1, self.COLUMNS)
            self._empty_label_in_grid = True
        self.empty_label.setVisible(True)

    def _hide_empty_state(self):
        """Hide and remove the empty state label from the grid."""
        if self._empty_label_in_grid:
            self.empty_label.setVisible(False)
            self.grid.removeWidget(self.empty_label)
            self._empty_label_in_grid = False

    def _create_thumbnail(self, clip: Clip, source: Source) -> ClipThumbnail:
        """Create and wire a thumbnail widget for a clip."""
        # Store source reference
        self._source_lookup[clip.id] = source

        # Create thumbnail widget
        thumb = ClipThumbnail(clip, source, drag_enabled=self._drag_enabled)
        thumb.clicked.connect(self._on_thumbnail_clicked)
        thumb.double_clicked.connect(self._on_thumbnail_double_clicked)
        thumb.drag_started.connect(self._on_drag_started)
        thumb.view_details_requested.connect(self._on_view_details_requested)
        thumb.export_requested.connect(self._on_export_requested)
        thumb.find_similar_requested.connect(self._activate_similarity)

        return thumb

    def _all_entries(self) -> list[tuple[Clip, Source]]:
        """Return all browser clips as data pairs."""
        if self._virtual_mode:
            return list(self._virtual_entries)
        return [(thumb.clip, thumb.source) for thumb in self.thumbnails]

    def is_virtualized(self) -> bool:
        """Whether the browser is storing clips as data and rendering a visible window."""
        return self._virtual_mode

    def get_total_clip_count(self) -> int:
        """Return total clips in the browser, including unrealized virtual clips."""
        if self._virtual_mode:
            return len(self._virtual_entries)
        return len(self.thumbnails)

    def get_realized_clip_count(self) -> int:
        """Return the number of live card widgets currently realized."""
        return len(self.thumbnails)

    def is_clip_realized(self, clip_id: str) -> bool:
        """Whether a clip currently has a live card widget realized in the grid."""
        return clip_id in self._thumbnail_by_id

    def set_virtual_clips(self, clip_source_pairs: list[tuple[Clip, Source]]) -> None:
        """Load clips in data-backed mode and realize only visible widgets."""
        self.clear()
        self._virtual_mode = True
        self._virtual_entries = []
        self._source_lookup = {}
        seen: set[str] = set()
        for clip, source in clip_source_pairs:
            if clip.id in seen:
                continue
            seen.add(clip.id)
            self._virtual_entries.append((clip, source))
            self._source_lookup[clip.id] = source

        self._invalidate_virtual_rows()
        self._sync_custom_query_filter_options()
        for clip, _source in self._virtual_entries:
            self._incremental_filter_enable(clip)
        self._update_duration_range()
        self._rebuild_grid()

    def _clear_realized_virtual_widgets(self) -> None:
        """Remove currently realized virtual widgets from the grid.

        Reuses cached `ClipThumbnail` widgets via `_virtual_widget_cache`, and
        also reuses `SourceGroupHeader` widgets registered in
        `_source_headers`. Only headers whose `source_id` is no longer present
        in the current virtual entries are torn down (handled below).
        """
        while self.grid.count():
            item = self.grid.takeAt(0)
            widget = item.widget()
            if widget is None:
                continue
            widget.setVisible(False)
            if isinstance(widget, (ClipThumbnail, SourceGroupHeader)):
                continue
            if widget is not self.empty_label:
                widget.deleteLater()

        # Drop any header whose source_id is no longer represented in the
        # virtual entries; reuse the rest on the next rebuild.
        if self._source_headers:
            current_source_ids = {source.id for _clip, source in self._virtual_entries}
            stale_ids = [
                sid for sid in self._source_headers
                if sid not in current_source_ids
            ]
            for sid in stale_ids:
                header = self._source_headers.pop(sid)
                header.deleteLater()
                self._group_expanded_state.pop(sid, None)

        self._empty_label_in_grid = False
        self.thumbnails = []
        self._thumbnail_by_id = {}
        self._virtual_top_spacer = None
        self._virtual_bottom_spacer = None
        self._virtual_render_range = None

    def _make_virtual_spacer(self, height: int) -> QWidget:
        spacer = QWidget()
        spacer.setFixedHeight(max(0, height))
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        return spacer

    def _invalidate_virtual_rows(self) -> None:
        """Mark cached virtual display rows stale."""
        self._virtual_rows_dirty = True
        self._virtual_render_range = None

    def _get_virtual_thumbnail(self, clip: Clip, source: Source) -> ClipThumbnail:
        """Return a cached virtual thumbnail widget for a clip."""
        thumb = self._virtual_widget_cache.get(clip.id)
        if thumb is not None and (thumb.clip is not clip or thumb.source.id != source.id):
            self._virtual_widget_cache.pop(clip.id, None)
            thumb.deleteLater()
            thumb = None

        if thumb is None:
            thumb = self._create_thumbnail(clip, source)
            self._virtual_widget_cache[clip.id] = thumb
        else:
            self._virtual_widget_cache.move_to_end(clip.id)
            thumb.set_drag_enabled(self._drag_enabled)

        # Walk oldest-to-newest and evict the first non-visible entry. If
        # every cached widget is currently visible, accept that the cache is
        # briefly over-limit rather than looping forever.
        if len(self._virtual_widget_cache) > VIRTUAL_WIDGET_CACHE_LIMIT:
            evict_target_id: str | None = None
            for cached_id, cached_thumb in self._virtual_widget_cache.items():
                if not cached_thumb.isVisible():
                    evict_target_id = cached_id
                    break
            if evict_target_id is not None:
                evict_thumb = self._virtual_widget_cache.pop(evict_target_id)
                evict_thumb.deleteLater()

        return thumb

    def add_clip(self, clip: Clip, source: Source):
        """Add a clip to the browser."""
        if self._virtual_mode:
            if clip.id in self._source_lookup:
                return
            self._virtual_entries.append((clip, source))
            self._source_lookup[clip.id] = source
            self._sync_custom_query_filter_options()
            self._incremental_filter_enable(clip)
            self._update_duration_range()
            # Per-clip thumbnail-ready signals can call this dozens of times in
            # quick succession; coalesce into one rebuild via the existing
            # _rebuild_pending guard inside _rebuild_grid.
            self._invalidate_virtual_rows()
            self._rebuild_grid(
                invalidate_virtual_rows=False,
                delay_ms=VIRTUAL_SCROLL_REBUILD_DELAY_MS,
            )
            return

        if clip.id in self._thumbnail_by_id:
            return

        thumb = self._create_thumbnail(clip, source)
        self.thumbnails.append(thumb)
        self._thumbnail_by_id[clip.id] = thumb  # O(1) lookup

        # Rebuild grid to handle grouping properly
        self._sync_custom_query_filter_options()
        # O(1) incremental filter availability (only enables, never disables)
        # Full _update_filter_availability() scan deferred to remove_clips_* paths
        self._incremental_filter_enable(clip)
        self._rebuild_grid()

        # Update duration range for spinboxes
        self._update_duration_range()

    def add_clips(
        self,
        clip_source_pairs: list[tuple[Clip, Source]],
        defer_rebuild: bool = False,
        defer_filter_sync: bool = False,
    ) -> None:
        """Add multiple clips with one filter sync and one grid rebuild.

        When called repeatedly during batched project loads, set
        `defer_rebuild=True` to skip the per-batch grid rebuild and duration
        refresh. Set `defer_filter_sync=True` to skip rebuilding filter menus
        on every batch. Then call `finalize_batch_load()` once after the final
        batch.
        """
        if self._virtual_mode:
            added: list[Clip] = []
            for clip, source in clip_source_pairs:
                if clip.id in self._source_lookup:
                    continue
                self._virtual_entries.append((clip, source))
                self._source_lookup[clip.id] = source
                added.append(clip)
            if not added:
                return
            if not defer_filter_sync:
                self._sync_custom_query_filter_options()
            for clip in added:
                self._incremental_filter_enable(clip)
            if not defer_rebuild:
                self._rebuild_grid()
                self._update_duration_range()
            return

        added: list[Clip] = []
        for clip, source in clip_source_pairs:
            if clip.id in self._thumbnail_by_id:
                continue
            thumb = self._create_thumbnail(clip, source)
            self.thumbnails.append(thumb)
            self._thumbnail_by_id[clip.id] = thumb
            added.append(clip)

        if not added:
            return

        if not defer_filter_sync:
            self._sync_custom_query_filter_options()
        for clip in added:
            self._incremental_filter_enable(clip)
        if not defer_rebuild:
            self._rebuild_grid()
            self._update_duration_range()

    def finalize_batch_load(self) -> None:
        """Flush deferred rebuilds after a series of `add_clips(defer_rebuild=True)` calls."""
        self._sync_custom_query_filter_options()
        self._rebuild_grid()
        self._update_duration_range()

    def clear(self):
        """Clear all clips."""
        visible_virtual_widget_ids = {id(thumb) for thumb in self.thumbnails}
        for thumb in self.thumbnails:
            self.grid.removeWidget(thumb)
            thumb.deleteLater()

        for thumb in self._virtual_widget_cache.values():
            if id(thumb) not in visible_virtual_widget_ids:
                thumb.deleteLater()
        self._virtual_widget_cache.clear()

        # Clear source headers
        for header in self._source_headers.values():
            self.grid.removeWidget(header)
            header.deleteLater()
        if self._virtual_top_spacer:
            self.grid.removeWidget(self._virtual_top_spacer)
            self._virtual_top_spacer.deleteLater()
        if self._virtual_bottom_spacer:
            self.grid.removeWidget(self._virtual_bottom_spacer)
            self._virtual_bottom_spacer.deleteLater()

        self.thumbnails = []
        self._thumbnail_by_id = {}
        self.selected_clips = set()
        self._source_lookup = {}
        self._source_headers = {}
        self._group_expanded_state = {}
        self._virtual_mode = False
        self._virtual_entries = []
        self._virtual_display_rows = []
        self._virtual_top_spacer = None
        self._virtual_bottom_spacer = None
        self._virtual_render_range = None
        self._virtual_rows_dirty = True
        self._virtual_rows_columns = 0
        self.container.setMinimumHeight(0)
        self._selected_custom_queries = set()
        self._sync_custom_query_filter_options()

        # Show empty state
        self._show_empty_state()

    def remove_clips_for_source(self, source_id: str):
        """Remove all clips for a specific source (used when re-analyzing)."""
        selection_before = set(self.selected_clips)

        if self._virtual_mode:
            removed_ids = {
                clip.id
                for clip, source in self._virtual_entries
                if source.id == source_id
            }
            if self._similarity_anchor_id in removed_ids:
                self._clear_similarity()
            self._virtual_entries = [
                (clip, source)
                for clip, source in self._virtual_entries
                if source.id != source_id
            ]
            for clip_id in removed_ids:
                self.selected_clips.discard(clip_id)
                self._source_lookup.pop(clip_id, None)
            if removed_ids:
                selection_changed = self._sync_custom_query_filter_options()
                self._rebuild_grid()
                if self.selected_clips != selection_before:
                    self._emit_selection_changed()
                if selection_changed:
                    self.filters_changed.emit()
            return

        # Separate into keep and remove in single pass (O(n) instead of O(n²))
        keep = []
        remove = []
        for thumb in self.thumbnails:
            if thumb.source.id == source_id:
                remove.append(thumb)
            else:
                keep.append(thumb)

        # Clear similarity mode if the anchor clip is being removed
        if self._similarity_anchor_id and any(
            t.clip.id == self._similarity_anchor_id for t in remove
        ):
            self._clear_similarity()

        # Clean up removed widgets
        for thumb in remove:
            self.grid.removeWidget(thumb)
            thumb.deleteLater()
            # Remove from selection if selected
            self.selected_clips.discard(thumb.clip.id)
            # Remove from lookups
            self._thumbnail_by_id.pop(thumb.clip.id, None)
            self._source_lookup.pop(thumb.clip.id, None)

        # Clean up header for this source
        if source_id in self._source_headers:
            header = self._source_headers.pop(source_id)
            self.grid.removeWidget(header)
            header.deleteLater()
            self._group_expanded_state.pop(source_id, None)

        # Replace list in one operation (avoids O(n) list.remove() calls)
        if remove:
            self.thumbnails = keep
            selection_changed = self._sync_custom_query_filter_options()
            self._rebuild_grid()  # This handles empty state
            if self.selected_clips != selection_before:
                self._emit_selection_changed()
            if selection_changed:
                self.filters_changed.emit()

    def remove_clips_by_ids(self, clip_ids: list[str]):
        """Remove specific clips by their IDs."""
        ids_to_remove = set(clip_ids)
        selection_before = set(self.selected_clips)

        if self._virtual_mode:
            if self._similarity_anchor_id in ids_to_remove:
                self._clear_similarity()
            before_count = len(self._virtual_entries)
            self._virtual_entries = [
                (clip, source)
                for clip, source in self._virtual_entries
                if clip.id not in ids_to_remove
            ]
            removed = before_count != len(self._virtual_entries)
            for clip_id in ids_to_remove:
                self.selected_clips.discard(clip_id)
                self._source_lookup.pop(clip_id, None)
            if removed:
                selection_changed = self._sync_custom_query_filter_options()
                self._rebuild_grid()
                if self.selected_clips != selection_before:
                    self._emit_selection_changed()
                if selection_changed:
                    self.filters_changed.emit()
            return

        # Clear similarity mode if the anchor clip is being removed
        if self._similarity_anchor_id and self._similarity_anchor_id in ids_to_remove:
            self._clear_similarity()

        keep = []
        remove = []
        for thumb in self.thumbnails:
            if thumb.clip.id in ids_to_remove:
                remove.append(thumb)
            else:
                keep.append(thumb)

        for thumb in remove:
            self.grid.removeWidget(thumb)
            thumb.deleteLater()
            self.selected_clips.discard(thumb.clip.id)
            self._thumbnail_by_id.pop(thumb.clip.id, None)
            self._source_lookup.pop(thumb.clip.id, None)

        if remove:
            self.thumbnails = keep
            selection_changed = self._sync_custom_query_filter_options()
            self._rebuild_grid()
            if self.selected_clips != selection_before:
                self._emit_selection_changed()
            if selection_changed:
                self.filters_changed.emit()

    def set_drag_enabled(self, enabled: bool):
        """Enable or disable dragging clips to timeline."""
        self._drag_enabled = enabled
        for thumb in self.thumbnails:
            thumb.set_drag_enabled(enabled)

    def get_selected_clips(self) -> list[Clip]:
        """Get list of selected clips.

        Returns only currently-visible (non-filtered-out) selected clips so the
        result is consistent regardless of whether the browser is operating in
        virtual or non-virtual mode (the same project crossing the
        VIRTUALIZATION_THRESHOLD must not return different selection sets).
        """
        if not self.selected_clips:
            return []
        if self._virtual_mode:
            return [
                clip
                for clip, _source in self._ordered_filtered_entries()
                if clip.id in self.selected_clips
            ]
        return [
            t.clip
            for t in self.thumbnails
            if t.clip.id in self.selected_clips and self._matches_filter(t)
        ]

    def set_selection(self, clip_ids: list[str]) -> None:
        """Set the selection to the specified clip IDs.

        Args:
            clip_ids: List of clip IDs to select
        """
        self._set_selected_ids(set(clip_ids))

    def select_all(self) -> None:
        """Select all visible clips."""
        if self._virtual_mode:
            self._set_selected_ids({
                clip.id
                for clip, _source in self._ordered_filtered_entries()
                if not clip.disabled
            })
            return

        # Only select clips that are currently visible (not filtered out) and enabled.
        self._set_selected_ids({
            thumb.clip.id
            for thumb in self.thumbnails
            if thumb.isVisible() and not thumb.disabled
        })

    def select_source(self, source_id: str, visible_only: bool = True) -> None:
        """Select enabled clips from one source.

        Args:
            source_id: Source ID whose clips should become selected.
            visible_only: When True, only clips matching active filters are selected.
        """
        if self._virtual_mode:
            grouped = self._ordered_grouped_entries()
            entries = grouped.get(source_id, [])
            selected_ids = {
                clip.id
                for clip, source in entries
                if source.id == source_id
                and not clip.disabled
                and (not visible_only or self._matches_entry(clip, source))
            }
        else:
            selected_ids = {
                thumb.clip.id
                for thumb in self.thumbnails
                if thumb.source.id == source_id
                and not thumb.disabled
                and (not visible_only or self._matches_filter(thumb))
            }

        self._set_selected_ids(selected_ids)
        self._refresh_source_header_counts()

    def clear_selection(self) -> None:
        """Clear all selections."""
        self._set_selected_ids(set())

    def _set_selected_ids(self, clip_ids: set[str], emit: bool = True) -> None:
        """Apply selection state and refresh thumbnail visuals."""
        self.selected_clips = set(clip_ids)

        for thumb in self.thumbnails:
            thumb.set_selected(thumb.clip.id in self.selected_clips)

        if emit:
            self._emit_selection_changed()

    def _emit_selection_changed(self) -> None:
        """Emit selected IDs in stable display order."""
        if not self.selected_clips:
            self.selection_changed.emit([])
            return
        if self._virtual_mode:
            selected_ids = [
                clip.id
                for clip, _source in self._ordered_filtered_entries()
                if clip.id in self.selected_clips
            ]
        else:
            selected_ids = [
                thumb.clip.id
                for thumb in self.thumbnails
                if thumb.clip.id in self.selected_clips
            ]
        self.selection_changed.emit(selected_ids)

    def _refresh_source_header_counts(self) -> None:
        """Refresh visible source header counts without rebuilding the grid."""
        if not self._source_headers:
            return

        if self._virtual_mode:
            grouped = self._ordered_grouped_entries()
            for source_id, header in self._source_headers.items():
                entries = grouped.get(source_id, [])
                total_count = len(entries)
                visible_count = sum(
                    1 for clip, source in entries if self._matches_entry(clip, source)
                )
                selected_count = sum(
                    1 for clip, _source in entries if clip.id in self.selected_clips
                )
                header.set_clip_counts(total_count, visible_count, selected_count)
            return

        thumbs_by_source: dict[str, list[ClipThumbnail]] = {}
        for thumb in self.thumbnails:
            thumbs_by_source.setdefault(thumb.source.id, []).append(thumb)

        for source_id, header in self._source_headers.items():
            source_thumbs = thumbs_by_source.get(source_id, [])
            total_count = len(source_thumbs)
            visible_count = sum(1 for thumb in source_thumbs if self._matches_filter(thumb))
            selected_count = sum(
                1 for thumb in source_thumbs if thumb.clip.id in self.selected_clips
            )
            header.set_clip_counts(total_count, visible_count, selected_count)

    def _selected_count_for_source(self, source_id: str) -> int:
        """Return selected clip count for a source without depending on row cache."""
        if self._virtual_mode:
            return sum(
                1
                for clip, source in self._virtual_entries
                if source.id == source_id and clip.id in self.selected_clips
            )

        return sum(
            1
            for thumb in self.thumbnails
            if thumb.source.id == source_id and thumb.clip.id in self.selected_clips
        )

    def _on_container_mouse_press(self, event) -> None:
        """Begin tracking an empty-space click or marquee drag."""
        if event.button() != Qt.LeftButton:
            event.ignore()
            return

        self._marquee_origin = event.position().toPoint()
        self._marquee_active = False
        self._marquee_additive = bool(event.modifiers() & Qt.ShiftModifier)
        self._marquee_base_selection = set(self.selected_clips)
        event.accept()

    def _on_container_mouse_move(self, event) -> None:
        """Update marquee selection while dragging across empty space."""
        if self._marquee_origin is None or not (event.buttons() & Qt.LeftButton):
            event.ignore()
            return

        current_pos = event.position().toPoint()
        if not self._marquee_active:
            distance = (current_pos - self._marquee_origin).manhattanLength()
            if distance < QApplication.startDragDistance():
                return
            self._marquee_active = True
            self._selection_band.show()

        self._update_marquee(current_pos)
        event.accept()

    def _on_container_mouse_release(self, event) -> None:
        """Finish marquee selection or clear selection on empty click."""
        if event.button() != Qt.LeftButton or self._marquee_origin is None:
            event.ignore()
            return

        current_pos = event.position().toPoint()
        if self._marquee_active:
            self._update_marquee(current_pos)
            self._selection_band.hide()
        else:
            self.clear_selection()

        self._marquee_origin = None
        self._marquee_active = False
        self._marquee_additive = False
        self._marquee_base_selection.clear()
        event.accept()

    def _update_marquee(self, current_pos) -> None:
        """Refresh marquee rectangle and live selection."""
        if self._marquee_origin is None:
            return

        rect = QRect(self._marquee_origin, current_pos).normalized()
        self._selection_band.setGeometry(rect)
        self._apply_marquee_selection(rect, additive=self._marquee_additive)

    def _apply_marquee_selection(self, rect: QRect, additive: bool) -> None:
        """Apply replace or additive selection for clips intersecting the rect."""
        selected_ids = self._clip_ids_in_rect(rect)
        if additive:
            selected_ids |= self._marquee_base_selection
        self._set_selected_ids(selected_ids)

    def _clip_ids_in_rect(self, rect: QRect) -> set[str]:
        """Return selectable clip IDs whose cards intersect the marquee rect."""
        return {
            thumb.clip.id
            for thumb in self.thumbnails
            if self._is_marquee_selectable(thumb) and rect.intersects(thumb.geometry())
        }

    def _is_marquee_selectable(self, thumb: ClipThumbnail) -> bool:
        """Whether a clip card should respond to marquee selection."""
        return thumb.isVisible() and not thumb.disabled

    def _on_thumbnail_clicked(self, clip: Clip):
        """Handle thumbnail click."""
        # Toggle selection
        if clip.id in self.selected_clips:
            self.selected_clips.remove(clip.id)
        else:
            self.selected_clips.add(clip.id)

        # Update all thumbnail states
        for thumb in self.thumbnails:
            thumb.set_selected(thumb.clip.id in self.selected_clips)

        self._emit_selection_changed()
        self.clip_selected.emit(clip)

    def _on_thumbnail_double_clicked(self, clip: Clip):
        """Handle thumbnail double-click.

        Double-click should open details without affecting selection.
        Since mousePressEvent already toggled selection, we toggle it back.
        """
        # Undo the selection toggle from the first click
        if clip.id in self.selected_clips:
            self.selected_clips.remove(clip.id)
        else:
            self.selected_clips.add(clip.id)

        # Update thumbnail visual state
        thumb = self._thumbnail_by_id.get(clip.id)
        if thumb:
            thumb.set_selected(thumb.clip.id in self.selected_clips)

        self._emit_selection_changed()
        self.clip_double_clicked.emit(clip)

    def _on_drag_started(self, clip: Clip):
        """Handle clip drag to timeline."""
        self.clip_dragged_to_timeline.emit(clip)

    def _on_view_details_requested(self, clip: Clip, source: Source):
        """Handle request to view clip details."""
        self.view_details_requested.emit(clip, source)

    def _on_export_requested(self, clip: Clip, source: Source):
        """Handle request to export a single clip."""
        self.export_requested.emit(clip, source)

    def get_source_for_clip(self, clip_id: str) -> Optional[Source]:
        """Get the source for a clip by ID."""
        return self._source_lookup.get(clip_id)

    def update_clip_colors(self, clip_id: str, colors: list[tuple[int, int, int]]):
        """Update the colors for a specific clip thumbnail (O(1) lookup)."""
        thumb = self._thumbnail_by_id.get(clip_id)
        if thumb:
            thumb.set_colors(colors)

    def update_clip_shot_type(self, clip_id: str, shot_type: str):
        """Update the shot type for a specific clip thumbnail (O(1) lookup)."""
        thumb = self._thumbnail_by_id.get(clip_id)
        if thumb:
            thumb.set_shot_type(shot_type)

    def update_clip_transcript(self, clip_id: str, segments: list):
        """Update the transcript for a specific clip thumbnail (O(1) lookup)."""
        thumb = self._thumbnail_by_id.get(clip_id)
        if thumb:
            thumb.set_transcript(segments)

    def update_clip_extracted_text(self, clip_id: str, texts: list):
        """Update the extracted text for a specific clip thumbnail (O(1) lookup)."""
        thumb = self._thumbnail_by_id.get(clip_id)
        if thumb:
            thumb.set_extracted_text(texts)

    def update_clip_custom_queries(self, clip_id: str, custom_queries: list[dict] | None):
        """Update custom query results for a specific clip thumbnail."""
        thumb = self._thumbnail_by_id.get(clip_id)
        if thumb:
            thumb.set_custom_queries(custom_queries)
            selection_changed = self._sync_custom_query_filter_options()
            if selection_changed or self._clip_updates_require_rebuild():
                self._rebuild_grid()
            else:
                self._refresh_source_header_counts()
            if selection_changed:
                self.filters_changed.emit()
        elif self._virtual_mode and clip_id in self._source_lookup:
            selection_changed = self._sync_custom_query_filter_options()
            if selection_changed or self._clip_updates_require_rebuild():
                self._rebuild_grid()
            else:
                self._refresh_source_header_counts()
            if selection_changed:
                self.filters_changed.emit()

    def update_clip_cinematography(self, clip_id: str, cinematography):
        """Update the cinematography for a specific clip thumbnail (O(1) lookup)."""
        thumb = self._thumbnail_by_id.get(clip_id)
        if thumb:
            thumb.set_cinematography(cinematography)

    def update_clip_gaze(self, clip_id: str, category: str | None):
        """Update the gaze direction badge for a specific clip thumbnail (O(1) lookup)."""
        thumb = self._thumbnail_by_id.get(clip_id)
        if thumb:
            thumb.set_gaze(category)
            self._update_filter_availability()
            if self._clip_updates_require_rebuild():
                self._rebuild_grid()
            else:
                self._refresh_source_header_counts()
        elif self._virtual_mode and clip_id in self._source_lookup:
            self._update_filter_availability()
            if self._clip_updates_require_rebuild():
                self._rebuild_grid()
            else:
                self._refresh_source_header_counts()

    def update_clip_thumbnail(self, clip_id: str, thumb_path: Path):
        """Update the thumbnail image for a specific clip (O(1) lookup)."""
        logger.debug(f"ClipBrowser.update_clip_thumbnail: clip_id={clip_id}, path={thumb_path}")
        thumb = self._thumbnail_by_id.get(clip_id)
        if thumb:
            thumb.set_thumbnail(thumb_path)
        else:
            logger.debug(
                "Thumbnail widget not found for %s; browser has %s thumbnails",
                clip_id,
                len(self._thumbnail_by_id),
            )

    def update_clips(self, clips: list[Clip], preserve_layout: bool = False):
        """Update thumbnails for the given clips (called when clips are edited).

        Args:
            clips: List of clips that were updated
            preserve_layout: When True, refresh realized cards without rebuilding rows.
        """
        updated_by_id = {clip.id: clip for clip in clips}
        if self._virtual_mode and updated_by_id:
            self._virtual_entries = [
                (updated_by_id.get(existing_clip.id, existing_clip), source)
                for existing_clip, source in self._virtual_entries
            ]
            # Replace any stale Clip references in the cached display rows so
            # subsequent renders (when preserve_layout=True keeps the cache)
            # don't show pre-edit data.
            if self._virtual_display_rows:
                for row_index, (row_type, payload) in enumerate(self._virtual_display_rows):
                    if row_type != "clips":
                        continue
                    refreshed = [
                        (updated_by_id.get(clip.id, clip), source)
                        for clip, source in payload
                    ]
                    self._virtual_display_rows[row_index] = (row_type, refreshed)

        for clip in clips:
            thumb = self._thumbnail_by_id.get(clip.id)
            if thumb:
                # Always update all editable fields to reflect changes
                thumb.set_shot_type(clip.shot_type)
                thumb.set_gaze(clip.gaze_category)
                thumb.set_transcript(clip.transcript)
                thumb.set_colors(clip.dominant_colors)
                thumb.set_custom_queries(clip.custom_queries)
                thumb._update_style()  # Refresh disabled visual state

        selection_changed = self._sync_custom_query_filter_options()
        can_preserve_layout = (
            preserve_layout and self._filter_state.enabled_filter is None
        )
        if can_preserve_layout or not self._clip_updates_require_rebuild():
            self._refresh_source_header_counts()
        else:
            self._rebuild_grid()
        if selection_changed:
            self.filters_changed.emit()

    def _clip_updates_require_rebuild(self) -> bool:
        """Whether generic clip data updates can affect current layout/filtering."""
        if self.has_active_filters():
            return True
        if hasattr(self, "sort_combo") and self.sort_combo.currentText() != "Timeline":
            return True
        return False

    def _available_custom_query_names(self) -> list[str]:
        """Return sorted custom query names present across all clips."""
        names: set[str] = set()
        for clip, _source in self._all_entries():
            names.update(get_latest_custom_query_results(clip.custom_queries).keys())
        return sorted(names, key=str.lower)

    def _refresh_custom_query_filter_button(self):
        """Update the custom query filter button summary text and tooltip."""
        selected_queries = sorted(self._selected_custom_queries, key=str.lower)
        if not selected_queries:
            self.custom_query_filter_btn.setText("All")
            self.custom_query_filter_btn.setToolTip("Filter clips by selected custom query matches")
            return

        if len(selected_queries) == 1:
            query = selected_queries[0]
            summary = query if len(query) <= 18 else f"{query[:15]}..."
            tooltip = f"Showing clips matching: {query}"
        else:
            summary = f"{len(selected_queries)} selected"
            tooltip = "Showing clips matching all selected queries:\n" + "\n".join(
                f"- {query}" for query in selected_queries
            )

        self.custom_query_filter_btn.setText(summary)
        self.custom_query_filter_btn.setToolTip(tooltip)

    def _sync_custom_query_filter_options(self) -> bool:
        """Rebuild the custom query filter menu and prune stale selections."""
        available_queries = self._available_custom_query_names()
        available_set = set(available_queries)
        selected_before = set(self._selected_custom_queries)
        self._selected_custom_queries.intersection_update(available_set)

        self._updating_custom_query_filter_menu = True
        self.custom_query_filter_menu.clear()
        self._custom_query_filter_actions = {}

        clear_action = self.custom_query_filter_menu.addAction("Clear Selection")
        clear_action.setEnabled(bool(self._selected_custom_queries))
        clear_action.triggered.connect(self._clear_custom_query_filter_selection)
        self.custom_query_filter_menu.addSeparator()

        if available_queries:
            for query in available_queries:
                action = self.custom_query_filter_menu.addAction(query)
                action.setCheckable(True)
                action.setChecked(query in self._selected_custom_queries)
                action.toggled.connect(
                    lambda checked, query_name=query: self._on_custom_query_query_toggled(
                        query_name, checked
                    )
                )
                self._custom_query_filter_actions[query] = action
        else:
            empty_action = self.custom_query_filter_menu.addAction("No queries yet")
            empty_action.setEnabled(False)

        self._updating_custom_query_filter_menu = False
        self._refresh_custom_query_filter_button()
        return selected_before != self._selected_custom_queries

    def _set_selected_custom_queries(self, selected_queries: set[str], emit: bool = True):
        """Apply the selected custom query names and refresh filtering."""
        self._selected_custom_queries = {
            query.strip() for query in selected_queries if query and query.strip()
        }
        self._sync_custom_query_filter_options()
        self._rebuild_grid()
        if emit:
            self.filters_changed.emit()

    def _clear_custom_query_filter_selection(self):
        """Clear all selected custom query filters."""
        if not self._selected_custom_queries:
            return
        self._set_selected_custom_queries(set())

    def _on_custom_query_query_toggled(self, query_name: str, checked: bool):
        """Handle a custom query checkbox toggle from the filter menu."""
        if self._updating_custom_query_filter_menu:
            return

        selected_queries = set(self._selected_custom_queries)
        if checked:
            selected_queries.add(query_name)
        else:
            selected_queries.discard(query_name)
        self._set_selected_custom_queries(selected_queries)

    def _on_filter_changed(self, filter_option: str):
        """Handle shot type filter dropdown change."""
        self._current_filter = filter_option
        self._rebuild_grid()
        self.filters_changed.emit()

    def _on_color_filter_changed(self, filter_option: str):
        """Handle color palette filter dropdown change."""
        self._current_color_filter = filter_option
        self._rebuild_grid()
        self.filters_changed.emit()

    def _on_search_changed(self, search_text: str):
        """Handle transcript search input change."""
        self._current_search_query = search_text.lower().strip()
        self._rebuild_grid()
        self.filters_changed.emit()

    def _on_sort_changed(self, sort_option: str):
        """Handle sort dropdown change."""
        if sort_option == "Timeline":
            self._sort_by_timeline()
        elif sort_option == "Color":
            self._sort_by_color()
        elif sort_option == "Duration":
            self._sort_by_duration()

    def _sort_by_timeline(self):
        """Sort clips by timeline order (start frame) within each source group."""
        if self._virtual_mode:
            self._rebuild_grid()
            return
        self._sort_within_groups(key=lambda t: _sort_key_by_timeline(t.clip))

    def _sort_by_color(self):
        """Sort clips by primary hue (HSV color wheel order) within each source group."""
        if self._virtual_mode:
            self._rebuild_grid()
            return
        self._sort_within_groups(key=lambda t: _sort_key_by_color(t.clip))

    def _sort_by_duration(self):
        """Sort clips by duration (longest first) within each source group."""
        if self._virtual_mode:
            self._rebuild_grid()
            return
        self._sort_within_groups(
            key=lambda t: _sort_key_by_duration(t.clip, t.source),
            reverse=True,
        )

    def _sort_within_groups(self, key, reverse: bool = False):
        """Sort thumbnails within each source group, preserving group order.

        Args:
            key: Sort key function for thumbnails
            reverse: If True, sort in descending order
        """
        # Group by source_id
        thumbs_by_source: dict[str, list[ClipThumbnail]] = {}
        for thumb in self.thumbnails:
            source_id = thumb.source.id
            if source_id not in thumbs_by_source:
                thumbs_by_source[source_id] = []
            thumbs_by_source[source_id].append(thumb)

        # Sort within each group
        for source_id in thumbs_by_source:
            thumbs_by_source[source_id].sort(key=key, reverse=reverse)

        # Rebuild thumbnails list maintaining source group order
        # (groups are sorted alphabetically by filename in _rebuild_grid)
        sorted_source_ids = sorted(
            thumbs_by_source.keys(),
            key=lambda sid: thumbs_by_source[sid][0].source.filename.lower()
            if thumbs_by_source[sid] else ""
        )

        self.thumbnails = []
        for source_id in sorted_source_ids:
            self.thumbnails.extend(thumbs_by_source[source_id])

        self._rebuild_grid()

    def _on_filter_state_changed(self) -> None:
        """Handle any mutation of the shared FilterState.

        Fires for every change — sidebar chip toggles, tribool radios,
        typeahead commits, programmatic apply_dict calls. Coalesces into a
        single grid rebuild via `_rebuild_grid`'s QTimer, then notifies
        tab-level listeners through `filters_changed`.
        """
        self._rebuild_grid()
        self.filters_changed.emit()

    def _rebuild_grid(
        self,
        *,
        invalidate_virtual_rows: bool = True,
        delay_ms: int = 0,
    ):
        """Schedule a grid rebuild on the next event loop iteration.

        Coalesces multiple calls into a single rebuild, which also ensures
        Qt has processed layouts so viewport dimensions are correct.
        """
        if self._virtual_mode and invalidate_virtual_rows:
            self._invalidate_virtual_rows()
        if not getattr(self, '_rebuild_pending', False):
            self._rebuild_pending = True
            QTimer.singleShot(delay_ms, self._do_rebuild_grid)

    def _do_rebuild_grid(self):
        """Actually rebuild the grid layout with source grouping, current order, and filter."""
        self._rebuild_pending = False
        if self._virtual_mode:
            self._do_rebuild_virtual_grid()
            return
        try:
            # Guard against C++ object deleted (deferred timer can fire after widget teardown)
            self.scroll.viewport()
        except RuntimeError:
            return
        self._last_column_count = self._calculate_columns()

        # Remove all thumbnails and headers from grid
        for thumb in self.thumbnails:
            self.grid.removeWidget(thumb)
            thumb.setVisible(False)

        for header in self._source_headers.values():
            self.grid.removeWidget(header)
            header.setVisible(False)

        # Handle empty state
        if not self.thumbnails:
            self._show_empty_state()
            return
        else:
            self._hide_empty_state()

        # Group thumbnails by source_id
        thumbs_by_source: dict[str, list[ClipThumbnail]] = {}
        for thumb in self.thumbnails:
            source_id = thumb.source.id
            if source_id not in thumbs_by_source:
                thumbs_by_source[source_id] = []
            thumbs_by_source[source_id].append(thumb)

        # Sort groups alphabetically by filename
        sorted_source_ids = sorted(
            thumbs_by_source.keys(),
            key=lambda sid: self._source_lookup.get(
                next(iter(thumbs_by_source[sid])).clip.id, None
            ).filename.lower() if thumbs_by_source[sid] else ""
        )

        # Build the grid row by row
        current_row = 0

        for source_id in sorted_source_ids:
            source_thumbs = thumbs_by_source[source_id]
            if not source_thumbs:
                continue

            source = source_thumbs[0].source

            # Calculate counts for this group
            total_count = len(source_thumbs)
            visible_thumbs = [t for t in source_thumbs if self._matches_filter(t)]

            # Sort by similarity score when similarity mode is active
            if self._similarity_anchor_id is not None:
                visible_thumbs.sort(
                    key=lambda t: self._similarity_scores.get(t.clip.id, 0.0),
                    reverse=True,
                )

            visible_count = len(visible_thumbs)
            selected_count = sum(
                1 for t in source_thumbs if t.clip.id in self.selected_clips
            )

            # Get or create header for this source
            header = self._get_or_create_header(source_id, source.filename, total_count)
            header.set_clip_counts(total_count, visible_count, selected_count)

            # Initialize expansion state if not set
            if source_id not in self._group_expanded_state:
                self._group_expanded_state[source_id] = True
            header.set_expanded(self._group_expanded_state[source_id])

            # Add header spanning all columns
            self.grid.addWidget(header, current_row, 0, 1, self.COLUMNS)
            header.setVisible(True)
            current_row += 1

            # Add clips if expanded
            if self._group_expanded_state[source_id]:
                col = 0
                for thumb in visible_thumbs:
                    self.grid.addWidget(
                        thumb, current_row, col, Qt.AlignTop | Qt.AlignLeft
                    )
                    thumb.setVisible(True)
                    col += 1
                    if col >= self.COLUMNS:
                        col = 0
                        current_row += 1
                # Move to next row if we have remaining clips
                if col > 0:
                    current_row += 1

        # Clean up headers for sources that no longer exist
        existing_source_ids = set(thumbs_by_source.keys())
        stale_ids = set(self._source_headers.keys()) - existing_source_ids
        for stale_id in stale_ids:
            header = self._source_headers.pop(stale_id)
            header.deleteLater()
            self._group_expanded_state.pop(stale_id, None)

    def _ordered_grouped_entries(self) -> dict[str, list[tuple[Clip, Source]]]:
        """Return virtual entries grouped and sorted like the widget grid."""
        grouped: dict[str, list[tuple[Clip, Source]]] = {}
        for clip, source in self._virtual_entries:
            grouped.setdefault(source.id, []).append((clip, source))

        sort_option = self.sort_combo.currentText()
        for entries in grouped.values():
            if self._similarity_anchor_id is not None:
                entries.sort(
                    key=lambda pair: self._similarity_scores.get(pair[0].id, 0.0),
                    reverse=True,
                )
            elif sort_option == "Color":
                entries.sort(key=lambda pair: _sort_key_by_color(pair[0]))
            elif sort_option == "Duration":
                entries.sort(
                    key=lambda pair: _sort_key_by_duration(pair[0], pair[1]),
                    reverse=True,
                )
            else:
                entries.sort(key=lambda pair: _sort_key_by_timeline(pair[0]))

        return grouped

    def _ordered_filtered_entries(self) -> list[tuple[Clip, Source]]:
        """Return virtual entries in display order with active filters applied."""
        grouped = self._ordered_grouped_entries()
        ordered: list[tuple[Clip, Source]] = []
        for source_id in sorted(
            grouped,
            key=lambda sid: grouped[sid][0][1].filename.lower() if grouped[sid] else "",
        ):
            ordered.extend(
                (clip, source)
                for clip, source in grouped[source_id]
                if self._matches_entry(clip, source)
            )
        return ordered

    def _build_virtual_rows(self, columns: int | None = None) -> list[tuple[str, object]]:
        """Build row records for the virtual grid."""
        grouped = self._ordered_grouped_entries()
        rows: list[tuple[str, object]] = []
        columns = columns or self.COLUMNS
        for source_id in sorted(
            grouped,
            key=lambda sid: grouped[sid][0][1].filename.lower() if grouped[sid] else "",
        ):
            source_entries = grouped[source_id]
            if not source_entries:
                continue
            source = source_entries[0][1]
            visible_entries = [
                (clip, src)
                for clip, src in source_entries
                if self._matches_entry(clip, src)
            ]
            total_count = len(source_entries)
            visible_count = len(visible_entries)
            selected_count = sum(
                1 for clip, _source in source_entries if clip.id in self.selected_clips
            )
            rows.append((
                "header",
                (source_id, source, total_count, visible_count, selected_count),
            ))
            if source_id not in self._group_expanded_state:
                self._group_expanded_state[source_id] = True
            if self._group_expanded_state[source_id]:
                for start in range(0, visible_count, columns):
                    rows.append(("clips", visible_entries[start:start + columns]))
        return rows

    def _get_virtual_rows(self) -> list[tuple[str, object]]:
        """Return cached virtual rows, rebuilding only when the data shape changes."""
        columns = self.COLUMNS
        if (
            self._virtual_rows_dirty
            or self._virtual_rows_columns != columns
            or not self._virtual_display_rows
        ):
            self._virtual_display_rows = self._build_virtual_rows(columns)
            self._virtual_rows_columns = columns
            self._virtual_rows_dirty = False
        return self._virtual_display_rows

    def _current_virtual_render_range(self, row_count: int) -> tuple[int, int]:
        """Calculate the virtual row range that should be realized."""
        viewport_height = max(1, self.scroll.viewport().height())
        scroll_value = self.scroll.verticalScrollBar().value()
        row_height = VIRTUAL_CARD_ROW_HEIGHT
        first_row = max(0, (scroll_value // row_height) - VIRTUAL_ROW_BUFFER)
        visible_rows = (viewport_height // row_height) + (VIRTUAL_ROW_BUFFER * 2) + 2
        last_row = min(row_count, first_row + visible_rows)
        return first_row, last_row

    def _do_rebuild_virtual_grid(self):
        """Render only the visible row window for large clip sets."""
        try:
            self.scroll.viewport()
        except RuntimeError:
            return

        self._last_column_count = self._calculate_columns()
        if not self._virtual_entries:
            self._clear_realized_virtual_widgets()
            self._show_empty_state()
            return

        rows = self._get_virtual_rows()
        if not rows:
            self._clear_realized_virtual_widgets()
            self._show_empty_state()
            return

        first_row, last_row = self._current_virtual_render_range(len(rows))

        row_height = VIRTUAL_CARD_ROW_HEIGHT
        total_height = len(rows) * row_height
        self.container.setMinimumHeight(total_height + (UISizes.GRID_MARGIN * 2))

        self.setUpdatesEnabled(False)
        try:
            self._clear_realized_virtual_widgets()
            self._hide_empty_state()
            self._virtual_render_range = (first_row, last_row)

            grid_row = 0
            top_height = first_row * row_height
            if top_height:
                self._virtual_top_spacer = self._make_virtual_spacer(top_height)
                self.grid.addWidget(self._virtual_top_spacer, grid_row, 0, 1, self.COLUMNS)
                grid_row += 1

            for row_type, payload in rows[first_row:last_row]:
                if row_type == "header":
                    source_id, source, total_count, visible_count, _selected_count = payload
                    selected_count = self._selected_count_for_source(source_id)
                    header = self._get_or_create_header(source_id, source.filename, total_count)
                    header.set_clip_counts(total_count, visible_count, selected_count)
                    header.set_expanded(self._group_expanded_state.get(source_id, True))
                    self.grid.addWidget(header, grid_row, 0, 1, self.COLUMNS)
                    header.setVisible(True)
                    grid_row += 1
                    continue

                entries = payload
                for col, (clip, source) in enumerate(entries):
                    thumb = self._get_virtual_thumbnail(clip, source)
                    thumb.set_selected(clip.id in self.selected_clips)
                    self.thumbnails.append(thumb)
                    self._thumbnail_by_id[clip.id] = thumb
                    self.grid.addWidget(thumb, grid_row, col, Qt.AlignTop | Qt.AlignLeft)
                    thumb.setVisible(True)
                grid_row += 1

            bottom_height = max(0, (len(rows) - last_row) * row_height)
            if bottom_height:
                self._virtual_bottom_spacer = self._make_virtual_spacer(bottom_height)
                self.grid.addWidget(
                    self._virtual_bottom_spacer,
                    grid_row,
                    0,
                    1,
                    self.COLUMNS,
                )
        finally:
            self.setUpdatesEnabled(True)

    def _get_or_create_header(
        self, source_id: str, filename: str, clip_count: int
    ) -> SourceGroupHeader:
        """Get existing header or create a new one for the source."""
        if source_id not in self._source_headers:
            header = SourceGroupHeader(source_id, filename, clip_count)
            header.toggled.connect(self._on_header_toggled)
            header.select_requested.connect(self._on_header_select_requested)
            self._source_headers[source_id] = header
        return self._source_headers[source_id]

    def _on_header_toggled(self, source_id: str, is_expanded: bool):
        """Handle source group header toggle."""
        self._group_expanded_state[source_id] = is_expanded
        self._rebuild_grid()

    def _on_header_select_requested(self, source_id: str):
        """Handle request to select visible clips from a source group."""
        self.select_source(source_id, visible_only=True)

    def expand_all_groups(self):
        """Expand all source groups."""
        for source_id in self._source_headers:
            self._group_expanded_state[source_id] = True
        self._rebuild_grid()

    def collapse_all_groups(self):
        """Collapse all source groups."""
        for source_id in self._source_headers:
            self._group_expanded_state[source_id] = False
        self._rebuild_grid()

    def _matches_filter(self, thumb: ClipThumbnail) -> bool:
        """Check if a thumbnail matches all filters (AND logic)."""
        return self._matches_entry(thumb.clip, thumb.source)

    def _matches_entry(self, clip: Clip, source: Source) -> bool:
        """Check if a clip/source data pair matches all filters."""
        # Check similarity mode — exclude clips without valid embeddings
        if self._similarity_anchor_id is not None:
            if clip.id not in self._similarity_scores:
                return False

        # Check shot type filter (multi-select: match any selected value)
        selected_shots = self._filter_state.shot_type
        if selected_shots:
            shot_type = clip.shot_type
            if not shot_type:
                return False
            if get_display_name(shot_type) not in selected_shots:
                return False

        # Check color palette filter (multi-select)
        selected_palettes = self._filter_state.color_palette
        if selected_palettes:
            colors = clip.dominant_colors
            if not colors:
                return False
            palette = classify_color_palette(colors)
            if get_palette_display_name(palette) not in selected_palettes:
                return False

        # Check transcript search
        if self._current_search_query:
            transcript_text = clip.get_transcript_text().lower()
            if self._current_search_query not in transcript_text:
                return False

        # Check custom query result filter
        if self._selected_custom_queries:
            latest_results = get_latest_custom_query_results(clip.custom_queries)
            for query_name in self._selected_custom_queries:
                query_result = latest_results.get(query_name)
                if not query_result or not bool(query_result.get("match")):
                    return False

        # Check duration filter
        if self._min_duration is not None or self._max_duration is not None:
            duration = clip.duration_seconds(source.fps)
            if self._min_duration is not None and duration < self._min_duration:
                return False
            if self._max_duration is not None and duration > self._max_duration:
                return False

        # Check aspect ratio filter (multi-select: any selected aspect must match)
        selected_aspects = self._filter_state.aspect_ratio
        if selected_aspects:
            if source.width == 0 or source.height == 0:
                return False
            aspect = source.aspect_ratio
            matched = False
            for key in selected_aspects:
                if key in self.ASPECT_RATIOS:
                    _, min_ratio, max_ratio = self.ASPECT_RATIOS[key]
                    if min_ratio <= aspect <= max_ratio:
                        matched = True
                        break
            if not matched:
                return False

        # Check gaze direction filter (multi-select)
        selected_gaze = self._filter_state.gaze_filter
        if selected_gaze:
            if not clip.gaze_category:
                return False
            if clip.gaze_category not in selected_gaze:
                return False

        # Check object search filter
        if self._object_search:
            query = self._object_search.lower()
            labels_text = " ".join(clip.object_labels or []).lower()
            detected_text = " ".join(
                d.get("label", "") for d in (clip.detected_objects or [])
            ).lower()
            if query not in labels_text and query not in detected_text:
                return False

        # Check description search filter
        if self._description_search:
            if self._description_search.lower() not in (clip.description or "").lower():
                return False

        # Check brightness range filter
        if self._min_brightness is not None:
            if clip.average_brightness is None or clip.average_brightness < self._min_brightness:
                return False
        if self._max_brightness is not None:
            if clip.average_brightness is None or clip.average_brightness > self._max_brightness:
                return False

        # ── Unit 5 filter predicates ────────────────────────────────
        fs = self._filter_state

        # Person count operator. Unanalyzed clips (clip.person_count is None)
        # are treated as having 0 people; callers using "< 0" would otherwise
        # exclude every clip silently (0 < 0 is False). The CountOperator
        # spinbox clamps to min=0 so the UI can't express n<0 today, but the
        # programmatic API accepts any int.
        if fs.person_count is not None:
            op, n = fs.person_count
            count = clip.person_count or 0
            if op == ">" and not (count > n):
                return False
            if op == "=" and not (count == n):
                return False
            if op == "<" and not (count < n):
                return False

        # Has audio (tribool)
        if fs.has_audio is not None:
            has_audio = bool(clip.transcript) or clip.rms_volume is not None
            if has_audio != fs.has_audio:
                return False

        # Has transcript
        if fs.has_transcript is not None:
            if bool(clip.transcript) != fs.has_transcript:
                return False

        # Has on-screen text
        if fs.has_on_screen_text is not None:
            if bool(clip.extracted_texts) != fs.has_on_screen_text:
                return False

        # On-screen text search
        if fs.on_screen_text_search:
            needle = fs.on_screen_text_search.lower()
            haystack = " ".join(
                (et.text if hasattr(et, "text") else str(et))
                for et in (clip.extracted_texts or [])
            ).lower()
            if needle not in haystack:
                return False

        # RMS volume range
        if fs.min_volume is not None:
            if clip.rms_volume is None or clip.rms_volume < fs.min_volume:
                return False
        if fs.max_volume is not None:
            if clip.rms_volume is None or clip.rms_volume > fs.max_volume:
                return False

        # Has-analysis-of-type — AND across selected operations.
        # Uses the module-level `_op_complete` predicate. If the analysis
        # availability module somehow failed to import, the filter becomes
        # a no-op (pass-through) — but this is extremely unlikely since
        # `core.analysis_availability` is always bundled.
        if fs.has_analysis_ops and _op_complete is not None:
            for op_key in fs.has_analysis_ops:
                if not _op_complete(op_key, clip):
                    return False

        # Enabled / disabled
        if fs.enabled_filter is not None:
            is_enabled = not clip.disabled
            if is_enabled != fs.enabled_filter:
                return False

        # Tag / note search
        if fs.tag_note_search:
            needle = fs.tag_note_search.lower()
            tags_text = " ".join(clip.tags or []).lower()
            notes_text = (clip.notes or "").lower()
            if needle not in tags_text and needle not in notes_text:
                return False

        # ── Unit 6 predicates: objects (ImageNet + YOLO) ────────────
        # ImageNet labels (Any / All semantics)
        if fs.imagenet_labels:
            clip_labels = set(clip.object_labels or [])
            if fs.imagenet_mode == "all":
                if not fs.imagenet_labels.issubset(clip_labels):
                    return False
            else:  # "any"
                if clip_labels.isdisjoint(fs.imagenet_labels):
                    return False

        # YOLO labels (OR within selected)
        if fs.yolo_labels:
            detected_labels = {d.get("label", "") for d in (clip.detected_objects or [])}
            if detected_labels.isdisjoint(fs.yolo_labels):
                return False

        # YOLO total object count operator
        if fs.yolo_total_count is not None:
            op, n = fs.yolo_total_count
            total = len(clip.detected_objects or [])
            if op == ">" and not (total > n):
                return False
            if op == "=" and not (total == n):
                return False
            if op == "<" and not (total < n):
                return False

        # YOLO per-label count rules (AND across rules)
        if fs.yolo_per_label_rules:
            counts = Counter(d.get("label", "") for d in (clip.detected_objects or []))
            for label, op, n in fs.yolo_per_label_rules:
                cnt = counts.get(label, 0)
                if op == ">" and not (cnt > n):
                    return False
                if op == "=" and not (cnt == n):
                    return False
                if op == "<" and not (cnt < n):
                    return False

        return True

    def _create_filter_panel(self) -> QFrame:
        """Create the collapsible filter panel for duration and aspect ratio."""
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel)
        panel.setStyleSheet(f"QFrame {{ background-color: {theme().background_tertiary}; }}")

        layout = QHBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # Duration range slider
        duration_label = QLabel("Duration:")
        layout.addWidget(duration_label)

        self.duration_slider = RangeSlider()
        self.duration_slider.set_suffix("s")
        self.duration_slider.set_range(0.0, 60.0)  # Default range, updated when clips added
        self.duration_slider.setMinimumWidth(200)
        self.duration_slider.setMaximumWidth(350)
        self.duration_slider.range_changed.connect(self._on_duration_slider_changed)
        layout.addWidget(self.duration_slider)

        layout.addSpacing(16)

        # Aspect ratio filter
        layout.addWidget(QLabel("Aspect:"))
        self.aspect_ratio_combo = QComboBox()
        self.aspect_ratio_combo.addItems(["All", "16:9", "4:3", "9:16"])
        self.aspect_ratio_combo.setFixedWidth(80)
        self.aspect_ratio_combo.currentTextChanged.connect(self._on_aspect_filter_changed)
        layout.addWidget(self.aspect_ratio_combo)

        layout.addSpacing(16)

        # Gaze direction filter
        self._gaze_label = QLabel("Gaze:")
        layout.addWidget(self._gaze_label)
        self.gaze_combo = QComboBox()
        gaze_options = ["All Gaze"] + list(GAZE_CATEGORY_DISPLAY.values())
        self.gaze_combo.addItems(gaze_options)
        self.gaze_combo.setFixedWidth(110)
        self.gaze_combo.currentTextChanged.connect(self._on_gaze_filter_changed)
        layout.addWidget(self.gaze_combo)

        layout.addSpacing(16)

        # Object search
        self._object_label = QLabel("Object:")
        layout.addWidget(self._object_label)
        self.object_search_input = QLineEdit()
        self.object_search_input.setPlaceholderText("Search objects...")
        self.object_search_input.setFixedWidth(120)
        self.object_search_input.setToolTip("Filter clips by detected object labels")
        self.object_search_input.textChanged.connect(self._on_object_search_changed)
        layout.addWidget(self.object_search_input)

        layout.addSpacing(16)

        # Description search
        self._description_label = QLabel("Description:")
        layout.addWidget(self._description_label)
        self.description_search_input = QLineEdit()
        self.description_search_input.setPlaceholderText("Search descriptions...")
        self.description_search_input.setFixedWidth(140)
        self.description_search_input.setToolTip("Filter clips by description text")
        self.description_search_input.textChanged.connect(self._on_description_search_changed)
        layout.addWidget(self.description_search_input)

        layout.addSpacing(16)

        # Brightness range slider
        self._brightness_label = QLabel("Brightness:")
        brightness_label = self._brightness_label
        layout.addWidget(brightness_label)

        self.brightness_slider = RangeSlider()
        self.brightness_slider.set_suffix("")
        self.brightness_slider.set_range(0.0, 1.0)
        self.brightness_slider.set_values(0.0, 1.0)
        self.brightness_slider.setMinimumWidth(200)
        self.brightness_slider.setMaximumWidth(350)
        self.brightness_slider.range_changed.connect(self._on_brightness_slider_changed)
        layout.addWidget(self.brightness_slider)

        layout.addSpacing(16)

        # Clear filters button
        self.clear_filters_btn = QPushButton("Clear Filters")
        self.clear_filters_btn.setToolTip("Reset all filters to show all clips")
        self.clear_filters_btn.clicked.connect(self.clear_all_filters)
        layout.addWidget(self.clear_filters_btn)

        # Clear Similarity button (hidden by default, shown when similarity mode is active)
        self._clear_similarity_btn = QPushButton("Clear Similarity")
        self._clear_similarity_btn.setToolTip("Exit similarity mode and restore normal sort order")
        self._clear_similarity_btn.clicked.connect(self._clear_similarity)
        self._clear_similarity_btn.setVisible(False)
        layout.addWidget(self._clear_similarity_btn)

        layout.addStretch()

        return panel

    def _toggle_filter_panel(self, visible: bool):
        """Show or hide the filter panel."""
        self._filter_panel_visible = visible
        self.filter_panel.setVisible(visible)
        self.filters_btn.setChecked(visible)

    def _update_duration_range(self):
        """Update duration slider range based on actual clip durations."""
        entries = self._all_entries()
        if not entries:
            return

        durations = [
            clip.duration_seconds(source.fps)
            for clip, source in entries
        ]
        if durations:
            min_dur = min(durations)
            max_dur = max(durations)
            # Set slider range with a small buffer
            self.duration_slider.set_range(
                max(0.0, min_dur - 0.1),
                max_dur + 0.1
            )
            # Reset to full range if no filter is active
            if self._min_duration is None and self._max_duration is None:
                self.duration_slider.set_values(min_dur - 0.1, max_dur + 0.1)

    def _on_duration_slider_changed(self, min_val: float, max_val: float):
        """Handle duration slider changes."""
        # Get the data range from the slider
        data_min = self.duration_slider._data_min
        data_max = self.duration_slider._data_max

        # Only apply filter if values differ from full range
        # Use small tolerance for float comparison
        at_min = abs(min_val - data_min) < 0.05
        at_max = abs(max_val - data_max) < 0.05

        if at_min and at_max:
            # Full range selected = no filter
            self._min_duration = None
            self._max_duration = None
        else:
            self._min_duration = min_val if not at_min else None
            self._max_duration = max_val if not at_max else None

        self._rebuild_grid()
        self.filters_changed.emit()

    def _on_aspect_filter_changed(self, value: str):
        """Handle aspect ratio filter changes."""
        self._aspect_ratio_filter = value
        self._rebuild_grid()
        self.filters_changed.emit()

    def _on_gaze_filter_changed(self, value: str):
        """Handle gaze direction filter dropdown change."""
        # Translate display label to internal key at the combo boundary
        self._gaze_filter = _GAZE_DISPLAY_TO_KEY.get(value)  # None for "All Gaze"
        self._rebuild_grid()
        self.filters_changed.emit()

    def _on_object_search_changed(self, text: str):
        """Handle object search input change."""
        self._object_search = text.strip()
        self._rebuild_grid()
        self.filters_changed.emit()

    def _on_description_search_changed(self, text: str):
        """Handle description search input change."""
        self._description_search = text.strip()
        self._rebuild_grid()
        self.filters_changed.emit()

    def _on_brightness_slider_changed(self, min_val: float, max_val: float):
        """Handle brightness slider changes."""
        # Get the data range from the slider
        data_min = self.brightness_slider._data_min
        data_max = self.brightness_slider._data_max

        # Only apply filter if values differ from full range
        at_min = abs(min_val - data_min) < 0.005
        at_max = abs(max_val - data_max) < 0.005

        if at_min and at_max:
            self._min_brightness = None
            self._max_brightness = None
        else:
            self._min_brightness = min_val if not at_min else None
            self._max_brightness = max_val if not at_max else None

        self._rebuild_grid()
        self.filters_changed.emit()

    def _activate_similarity(self, clip: Clip):
        """Activate similarity mode: rank all clips by embedding similarity to the anchor.

        Args:
            clip: The anchor clip to compare against.
        """
        # Validate anchor clip has a valid embedding
        if clip.embedding is None or np.linalg.norm(clip.embedding) == 0:
            logger.warning(
                "Cannot activate similarity: clip %s has no valid embedding", clip.id
            )
            return

        anchor = np.array(clip.embedding, dtype=np.float32)
        anchor_dim = anchor.shape[0]
        scores: dict[str, float] = {}

        # Batch-vectorize: stack valid embeddings into a matrix for a single matmul
        valid_ids: list[str] = []
        valid_embs: list[list] = []
        for candidate_clip, _source in self._all_entries():
            emb = candidate_clip.embedding
            if emb is not None and len(emb) == anchor_dim:
                valid_ids.append(candidate_clip.id)
                valid_embs.append(emb)

        if valid_embs:
            matrix = np.array(valid_embs, dtype=np.float32)  # (N, D)
            norms = np.linalg.norm(matrix, axis=1)
            # Mask out zero-vector embeddings
            nonzero = norms > 0
            if nonzero.any():
                dot_scores = matrix[nonzero] @ anchor  # single matmul
                for clip_id, score in zip(
                    (cid for cid, nz in zip(valid_ids, nonzero) if nz),
                    dot_scores,
                ):
                    scores[clip_id] = float(score)

        self._similarity_anchor_id = clip.id
        self._similarity_scores = scores
        self._clear_similarity_btn.setVisible(True)

        # Show the filter panel so the user can see the "Clear Similarity" button
        if not self._filter_panel_visible:
            self._toggle_filter_panel(True)

        self._rebuild_grid()

    def _clear_similarity(self):
        """Exit similarity mode and restore normal sort order."""
        self._similarity_anchor_id = None
        self._similarity_scores = {}
        self._clear_similarity_btn.setVisible(False)
        self._rebuild_grid()

    def clear_all_filters(self):
        """Reset all filters to show all clips.

        Delegates to ``FilterState.clear_all()`` so Unit 5/6 fields
        (person_count, has_audio, imagenet_labels, yolo_*, etc.) are reset
        alongside the legacy 13 — previously this method only touched the
        legacy set and left Unit 5/6 filters active after a "Clear".
        """
        # Block legacy-widget signals so the syncing calls below don't
        # trigger extra grid rebuilds — FilterState.changed already does.
        legacy_widgets = [
            self.filter_combo, self.color_filter_combo, self.search_input,
            self.duration_slider, self.aspect_ratio_combo, self.gaze_combo,
            self.object_search_input, self.description_search_input,
            self.brightness_slider,
        ]
        for w in legacy_widgets:
            w.blockSignals(True)

        self._filter_state.clear_all()
        self._clear_similarity_btn.setVisible(False)

        # Sync legacy (hidden) widget UI to match the cleared state.
        self.filter_combo.setCurrentText("All")
        self.color_filter_combo.setCurrentText("All")
        self.search_input.clear()
        self.duration_slider.reset()
        self.aspect_ratio_combo.setCurrentText("All")
        self.gaze_combo.setCurrentText("All Gaze")
        self.object_search_input.clear()
        self.description_search_input.clear()
        self.brightness_slider.reset()
        self._sync_custom_query_filter_options()

        for w in legacy_widgets:
            w.blockSignals(False)

        # Rebuild once
        self._rebuild_grid()
        self.filters_changed.emit()

    def apply_filters(self, filters: dict) -> None:
        """Apply multiple filters at once via public API.

        Args:
            filters: Dict with optional keys:
                - min_duration: float or None
                - max_duration: float or None
                - aspect_ratio: str ('16:9', '4:3', '9:16') or None
                - shot_type: str ('Wide Shot', 'Medium Shot', 'Close-up', 'Extreme CU') or None
                - color_palette: str ('Warm', 'Cool', 'Neutral', 'Vibrant') or None
                - search_query: str or None
                - custom_query: list[str] | tuple[str, ...] | set[str] | str | None
                - gaze: str (internal key like 'at_camera' or display label like 'At Camera') or None
                - object_search: str or None
                - description_search: str or None
                - min_brightness: float or None
                - max_brightness: float or None
        """
        # Block signals to avoid multiple rebuilds
        self.duration_slider.blockSignals(True)
        self.aspect_ratio_combo.blockSignals(True)
        self.filter_combo.blockSignals(True)
        self.color_filter_combo.blockSignals(True)
        self.search_input.blockSignals(True)
        self.gaze_combo.blockSignals(True)
        self.object_search_input.blockSignals(True)
        self.description_search_input.blockSignals(True)
        self.brightness_slider.blockSignals(True)

        # Apply duration filter
        if 'min_duration' in filters or 'max_duration' in filters:
            self._min_duration = filters.get('min_duration')
            self._max_duration = filters.get('max_duration')
            data_min, data_max = self.duration_slider.get_data_range()
            new_min = self._min_duration if self._min_duration is not None else data_min
            new_max = self._max_duration if self._max_duration is not None else data_max
            self.duration_slider.set_values(new_min, new_max)

        # Apply aspect ratio filter. Enum fields accept strings (backward compat)
        # or list/set (multi-select). When multi-select is active, the legacy
        # QComboBox UI can only reflect one value — show "All" as the fallback
        # since the sidebar owns the full multi-select state from Unit 4 onward.
        if 'aspect_ratio' in filters:
            self._filter_state.aspect_ratio = filters['aspect_ratio']
            self.aspect_ratio_combo.setCurrentText(
                _combo_text_for_enum(self._filter_state.aspect_ratio, fallback="All")
            )

        if 'shot_type' in filters:
            self._filter_state.shot_type = filters['shot_type']
            self.filter_combo.setCurrentText(
                _combo_text_for_enum(self._filter_state.shot_type, fallback="All")
            )

        if 'color_palette' in filters:
            self._filter_state.color_palette = filters['color_palette']
            self.color_filter_combo.setCurrentText(
                _combo_text_for_enum(self._filter_state.color_palette, fallback="All")
            )

        # Apply search query
        if 'search_query' in filters:
            value = filters['search_query'] or ""
            self._current_search_query = value.lower().strip()
            self.search_input.setText(value)

        # Apply custom query filter
        if 'custom_query' in filters:
            value = filters['custom_query']
            if value is None:
                self._selected_custom_queries = set()
            elif isinstance(value, str):
                if value in {"All", "Match", "No Match"}:
                    self._selected_custom_queries = set()
                else:
                    self._selected_custom_queries = {value}
            else:
                self._selected_custom_queries = {
                    str(query).strip() for query in value if str(query).strip()
                }
            self._sync_custom_query_filter_options()

        # Apply gaze filter — accepts internal key, display label, list/set
        # (multi-select), or None/empty. Normalize to the internal key set.
        if 'gaze' in filters:
            value = filters['gaze']
            if value is None or value == "" or value == "All Gaze":
                self._filter_state.gaze_filter = set()
            else:
                if isinstance(value, str):
                    items = [value]
                else:
                    items = list(value)
                keys: set[str] = set()
                for item in items:
                    if item in GAZE_CATEGORY_DISPLAY:
                        keys.add(item)
                    elif item in _GAZE_DISPLAY_TO_KEY:
                        keys.add(_GAZE_DISPLAY_TO_KEY[item])
                self._filter_state.gaze_filter = keys
            # Sync legacy combo box — shows single value or "All Gaze" fallback
            keys = self._filter_state.gaze_filter
            if len(keys) == 1:
                display = GAZE_CATEGORY_DISPLAY.get(next(iter(keys)), "All Gaze")
            else:
                display = "All Gaze"
            self.gaze_combo.setCurrentText(display)

        # Apply object search filter
        if 'object_search' in filters:
            value = filters['object_search'] or ""
            self._object_search = value.strip()
            self.object_search_input.setText(value)

        # Apply description search filter
        if 'description_search' in filters:
            value = filters['description_search'] or ""
            self._description_search = value.strip()
            self.description_search_input.setText(value)

        # Apply brightness filter
        if 'min_brightness' in filters or 'max_brightness' in filters:
            self._min_brightness = filters.get('min_brightness')
            self._max_brightness = filters.get('max_brightness')
            data_min, data_max = self.brightness_slider.get_data_range()
            new_min = self._min_brightness if self._min_brightness is not None else data_min
            new_max = self._max_brightness if self._max_brightness is not None else data_max
            self.brightness_slider.set_values(new_min, new_max)

        # Unblock signals
        self.duration_slider.blockSignals(False)
        self.aspect_ratio_combo.blockSignals(False)
        self.filter_combo.blockSignals(False)
        self.color_filter_combo.blockSignals(False)
        self.search_input.blockSignals(False)
        self.gaze_combo.blockSignals(False)
        self.object_search_input.blockSignals(False)
        self.description_search_input.blockSignals(False)
        self.brightness_slider.blockSignals(False)

        # Delegate Unit 5+ keys to FilterState. The legacy keys above have
        # already been applied (often with display↔key conversion, e.g. for
        # gaze), so strip them before delegation to avoid clobbering.
        _legacy_keys = {
            "min_duration", "max_duration", "aspect_ratio", "shot_type",
            "color_palette", "search_query", "custom_query", "gaze",
            "object_search", "description_search", "min_brightness",
            "max_brightness", "similarity_anchor",
        }
        unit5_filters = {k: v for k, v in filters.items() if k not in _legacy_keys}
        if unit5_filters:
            self._filter_state.apply_dict(unit5_filters)

        # Rebuild grid and emit signal
        self._rebuild_grid()
        self.filters_changed.emit()

        # Show filter panel if hidden
        if not self._filter_panel_visible:
            self._toggle_filter_panel(True)

    def set_sort_order(self, sort_by: str) -> None:
        """Set the clip sort order via public API.

        Args:
            sort_by: One of 'Timeline', 'Color', 'Duration'
        """
        self.sort_combo.setCurrentText(sort_by)

    def get_active_filters(self) -> dict:
        """Return current filter state.

        Returns:
            Dict with filter names and their current values. The exact
            shape is determined by ``FilterState.to_dict`` — legacy keys
            (shot_type, color_palette, search_query, custom_query,
            min/max_duration, aspect_ratio, gaze, object_search,
            description_search, min/max_brightness, similarity_anchor)
            plus Unit 5 additions (person_count, has_audio,
            has_transcript, has_on_screen_text, on_screen_text_search,
            min/max_volume, has_analysis_ops, enabled_filter,
            tag_note_search).
        """
        return self._filter_state.to_dict()

    def has_active_filters(self) -> bool:
        """Check if any filters are currently active.

        Returns:
            True if at least one filter is set. Delegates to
            ``FilterState.has_active`` so Unit 5/6 fields are included —
            this method previously only checked the 13 legacy fields,
            silently returning False when (e.g.) has_audio or imagenet_labels
            was the only active filter.
        """
        return self._filter_state.has_active()

    def _incremental_filter_enable(self, clip) -> None:
        """Enable filter controls based on a single clip's data (O(1)).

        Called from add_clip to avoid O(n) scan on every addition.
        Only enables controls — never disables (use _update_filter_availability for that).
        """
        if clip.gaze_category:
            self._gaze_label.setEnabled(True)
            self.gaze_combo.setEnabled(True)
            self.gaze_combo.setToolTip("")
        if clip.object_labels or clip.detected_objects:
            self._object_label.setEnabled(True)
            self.object_search_input.setEnabled(True)
            self.object_search_input.setToolTip("Filter clips by detected object labels")
        if clip.description:
            self._description_label.setEnabled(True)
            self.description_search_input.setEnabled(True)
            self.description_search_input.setToolTip("Filter clips by description text")
        if clip.average_brightness is not None:
            self._brightness_label.setEnabled(True)
            self.brightness_slider.setEnabled(True)
            self.brightness_slider.setToolTip("")

    def _update_filter_availability(self):
        """Enable/disable analysis-dependent filter controls based on clip data.

        A filter is enabled if ANY clip has the relevant field populated.
        """
        entries = self._all_entries()
        has_gaze = any(clip.gaze_category for clip, _source in entries)
        has_objects = any(
            clip.object_labels or clip.detected_objects for clip, _source in entries
        )
        has_descriptions = any(clip.description for clip, _source in entries)
        has_brightness = any(
            clip.average_brightness is not None for clip, _source in entries
        )

        # Gaze filter
        self._gaze_label.setEnabled(has_gaze)
        self.gaze_combo.setEnabled(has_gaze)
        self.gaze_combo.setToolTip(
            "" if has_gaze else "Requires gaze analysis — run it in the Analyze tab"
        )

        # Object filter
        self._object_label.setEnabled(has_objects)
        self.object_search_input.setEnabled(has_objects)
        self.object_search_input.setToolTip(
            "Filter clips by detected object labels" if has_objects
            else "Requires object detection — run it in the Analyze tab"
        )

        # Description filter
        self._description_label.setEnabled(has_descriptions)
        self.description_search_input.setEnabled(has_descriptions)
        self.description_search_input.setToolTip(
            "Filter clips by description text" if has_descriptions
            else "Requires clip descriptions — run Describe in the Analyze tab"
        )

        # Brightness filter
        self._brightness_label.setEnabled(has_brightness)
        self.brightness_slider.setEnabled(has_brightness)
        self.brightness_slider.setToolTip(
            "" if has_brightness
            else "Requires brightness analysis — run Extract Colors in the Analyze tab"
        )

    def get_visible_clip_count(self) -> int:
        """Get the number of currently visible (non-filtered) clips.

        Returns:
            Number of visible clips
        """
        if self._virtual_mode:
            return len(self._ordered_filtered_entries())
        return sum(1 for thumb in self.thumbnails if self._matches_filter(thumb))

    def toggle_disabled(self, clip_ids: list[str]):
        """Toggle the disabled state of clips by ID via undo stack."""
        selection_before = set(self.selected_clips)
        clips_by_id = {clip.id: clip for clip, _source in self._all_entries()}
        becoming_disabled = {
            clip_id
            for clip_id in clip_ids
            if (clip := clips_by_id.get(clip_id)) is not None and not clip.disabled
        }

        from ui.commands.toggle_clip_disabled import ToggleClipDisabledCommand
        main_win = self.window()
        if hasattr(main_win, 'undo_stack'):
            # Notify listeners (e.g. MainWindow) that these IDs are toggling so the
            # downstream `clips_updated` handler can preserve layout for them.
            self.disabled_clips_changed.emit(list(clip_ids))
            cmd = ToggleClipDisabledCommand(main_win.project, clip_ids)
            main_win.undo_stack.push(cmd)
        else:
            # Fallback when no undo stack (e.g. tests)
            for clip_id in clip_ids:
                clip = clips_by_id.get(clip_id)
                if clip:
                    clip.disabled = not clip.disabled
                if thumb := self._thumbnail_by_id.get(clip_id):
                    thumb._update_style()

        # Disabling a clip should only clear selection for that clip.
        self.selected_clips.difference_update(becoming_disabled)
        for thumb in self.thumbnails:
            thumb.set_selected(thumb.clip.id in self.selected_clips)
        if self._filter_state.enabled_filter is not None:
            self._rebuild_grid()
        else:
            self._refresh_source_header_counts()

        if self.selected_clips != selection_before:
            self._emit_selection_changed()

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts.

        Args:
            event: The key event
        """
        # Delete/Backspace to toggle disable on selected clips
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            selected = self.get_selected_clips()
            if selected:
                self.toggle_disabled([c.id for c in selected])
                event.accept()
                return
        # Enter or 'i' to show details of selected clip
        elif event.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_I):
            selected = self.get_selected_clips()
            if selected and len(selected) == 1:
                clip = selected[0]
                # Look up the source for this clip
                source = self._source_lookup.get(clip.id)
                if source:
                    self.view_details_requested.emit(clip, source)
                    event.accept()
                    return
        super().keyPressEvent(event)
