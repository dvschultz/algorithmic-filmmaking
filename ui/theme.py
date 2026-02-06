"""Centralized theme support for light/dark mode.

Provides a ThemeColors dataclass with all UI colors, auto-detection of system
dark/light mode, and easy access via the theme() convenience function.

Usage:
    from ui.theme import theme

    # In styles:
    label.setStyleSheet(f"color: {theme().text_secondary};")

    # For QColor:
    painter.fillRect(rect, theme().qcolor('background_primary'))

    # Listen for theme changes:
    theme().changed.connect(self._on_theme_changed)
"""

from dataclasses import dataclass, field
from typing import Optional

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QPalette, QColor
from PySide6.QtCore import QObject, Signal


# ---------------------------------------------------------------------------
# Design token scales
# ---------------------------------------------------------------------------

class TypeScale:
    """Consistent font sizes. Use for QFont.setPointSize() and QSS font-size."""
    XS = 9       # Badges, timestamps, fine print
    SM = 11      # Secondary labels, captions, metadata
    BASE = 13    # Body text, form inputs, default
    MD = 14      # Card titles, section labels
    LG = 16      # Dialog headers, prominent labels
    XL = 18      # Section headers, empty state titles
    XXL = 24     # Hero text, main empty state titles
    XXXL = 32    # Splash/about (rare)


class Spacing:
    """Consistent spacing for padding and margins."""
    XXS = 2      # Tight: between badge elements
    XS = 4       # Compact: icon-text gaps
    SM = 8       # Default: between related elements
    MD = 12      # Comfortable: form field spacing
    LG = 16      # Section: card padding, panel margins
    XL = 24      # Group: grid gutter, section breaks
    XXL = 32     # Page: major section separators
    XXXL = 48    # Hero: empty state breathing room


class Radii:
    """Consistent border-radius values."""
    SM = 4       # Badges, checkboxes, progress bars
    MD = 8       # Cards, buttons, inputs
    LG = 12      # Chat bubbles, dialogs
    XL = 16      # Featured cards, modals
    FULL = 9999  # Pill shapes


@dataclass
class GradientPalette:
    """Default gradient colors for glow effects."""
    # Default cool purple-blue (non-selected sorting cards, fallback)
    default_color_1: tuple = (99, 102, 241)    # indigo
    default_color_2: tuple = (139, 92, 246)    # violet
    default_color_3: tuple = (59, 130, 246)    # blue

    # Electric multicolor (selected/active state)
    active_color_1: tuple = (249, 115, 22)     # orange
    active_color_2: tuple = (6, 182, 212)      # cyan
    active_color_3: tuple = (91, 141, 239)     # electric blue

    # Glow intensity
    default_opacity: int = 160    # ~63%
    active_opacity: int = 200     # ~78%
    glow_spread: int = 6          # pixels beyond card edge

    @property
    def default_colors(self) -> list[tuple[int, int, int]]:
        return [self.default_color_1, self.default_color_2, self.default_color_3]

    @property
    def active_colors(self) -> list[tuple[int, int, int]]:
        return [self.active_color_1, self.active_color_2, self.active_color_3]


# Singleton gradient palettes
DARK_GRADIENT = GradientPalette()
LIGHT_GRADIENT = GradientPalette(
    default_opacity=100,  # Lower intensity in light mode
    active_opacity=140,
)


# ---------------------------------------------------------------------------
# Theme colors
# ---------------------------------------------------------------------------

@dataclass
class ThemeColors:
    """Color definitions for a theme."""

    # Backgrounds
    background_primary: str      # Main background
    background_secondary: str    # Cards, panels
    background_tertiary: str     # Nested elements, hover states
    background_elevated: str     # Popovers, menus

    # Text
    text_primary: str            # Main text
    text_secondary: str          # Labels, less important text
    text_muted: str              # Disabled, hints
    text_inverted: str           # Text on accent colors

    # Borders
    border_primary: str          # Main borders
    border_secondary: str        # Subtle borders
    border_focus: str            # Focus rings

    # Accents
    accent_blue: str             # Primary accent (selection, links)
    accent_blue_hover: str       # Blue hover state
    accent_red: str              # Errors, destructive, playhead
    accent_green: str            # Success, analyzed badge
    accent_orange: str           # Warning
    accent_purple: str           # Special highlights

    # Timeline specific
    timeline_background: str     # Timeline view background
    timeline_ruler: str          # Ruler background
    timeline_ruler_border: str   # Ruler bottom border
    timeline_ruler_tick: str     # Major tick color
    timeline_ruler_tick_minor: str  # Minor tick color
    timeline_track: str          # Track background
    timeline_track_highlight: str   # Track drop highlight
    timeline_clip: str           # Clip fill
    timeline_clip_selected: str  # Selected clip fill
    timeline_clip_border: str    # Clip border
    timeline_clip_selected_border: str  # Selected clip border

    # Component specific
    thumbnail_background: str    # Thumbnail placeholder
    card_background: str         # Card/panel background
    card_border: str             # Card border
    card_hover: str              # Card hover state
    badge_analyzed: str          # "Analyzed" badge
    badge_not_analyzed: str      # "Not Analyzed" badge
    shot_type_badge: str         # Shot type label background
    surface_highlight: str       # Highlighted surface (env override indicator)

    # Chat/Plan widget colors (semantic)
    chat_user_bubble: str        # User message bubble
    chat_assistant_bubble: str   # Assistant message bubble
    chat_user_text: str          # User message text
    chat_assistant_text: str     # Assistant message text
    plan_pending_bg: str         # Pending step background
    plan_pending_border: str     # Pending step border
    plan_running_bg: str         # Running step background
    plan_running_border: str     # Running step border
    plan_completed_bg: str       # Completed step background
    plan_completed_border: str   # Completed step border
    plan_failed_bg: str          # Failed step background
    plan_failed_border: str      # Failed step border

    # Overlay tokens
    overlay_dark: str = "rgba(5, 5, 15, 0.85)"
    overlay_medium: str = "rgba(5, 5, 15, 0.65)"
    surface_success: str = "rgba(62, 207, 110, 0.1)"
    surface_error: str = "rgba(240, 78, 94, 0.1)"
    badge_disabled_bg: str = "rgba(170, 35, 35, 0.95)"

    def qcolor(self, attr: str) -> QColor:
        """Get a QColor for the given attribute name."""
        color_str = getattr(self, attr)
        return QColor(color_str)


# UI sizing constants - use these for consistent widget dimensions
class UISizes:
    """Standard sizes for UI elements to ensure consistency across the app."""

    # Widget heights
    COMBO_BOX_MIN_HEIGHT = 32
    LINE_EDIT_MIN_HEIGHT = 32
    BUTTON_MIN_HEIGHT = 32

    # Label widths for form layouts
    FORM_LABEL_WIDTH = 140
    FORM_LABEL_WIDTH_NARROW = 120
    FORM_LABEL_WIDTH_WIDE = 180

    # Minimum widths
    COMBO_BOX_MIN_WIDTH = 200
    COMBO_BOX_MIN_WIDTH_WIDE = 300

    # Grid layout constants
    GRID_GUTTER = 24           # Spacing between grid items
    GRID_CARD_MAX_WIDTH = 240  # Maximum width per card/column
    GRID_MARGIN = 16           # Outer margin around grid

    # Source group header
    SOURCE_HEADER_HEIGHT = 36  # Height of collapsible source headers in clip grid


# Dark theme — deep navy-charcoal palette
DARK_THEME = ThemeColors(
    # Backgrounds — deep navy-charcoal instead of neutral gray
    background_primary="#0d0f14",
    background_secondary="#151820",
    background_tertiary="#1c2030",
    background_elevated="#252a3a",

    # Text — slightly warm white, not pure #ffffff
    text_primary="#e8eaf0",
    text_secondary="#8b92a8",
    text_muted="#525a72",
    text_inverted="#ffffff",

    # Borders — subtle, blends with navy
    border_primary="#2a3045",
    border_secondary="#1e2436",
    border_focus="#5b8def",

    # Accents — electric blue family
    accent_blue="#5b8def",
    accent_blue_hover="#7ba3f7",
    accent_red="#f04e5e",
    accent_green="#3ecf6e",
    accent_orange="#f0a030",
    accent_purple="#a87edb",

    # Timeline
    timeline_background="#0d0f14",
    timeline_ruler="#151820",
    timeline_ruler_border="#2a3045",
    timeline_ruler_tick="#525a72",
    timeline_ruler_tick_minor="#2a3045",
    timeline_track="#131620",
    timeline_track_highlight="#1c2545",
    timeline_clip="#4a7de0",
    timeline_clip_selected="#5b8def",
    timeline_clip_border="#3a65c0",
    timeline_clip_selected_border="#e8eaf0",

    # Components
    thumbnail_background="#1a1d28",
    card_background="#151820",
    card_border="#2a3045",
    card_hover="#1c2030",
    badge_analyzed="#3ecf6e",
    badge_not_analyzed="#525a72",
    shot_type_badge="#2a3045",
    surface_highlight="#1c2545",

    # Chat
    chat_user_bubble="#5b8def",
    chat_assistant_bubble="#1c2030",
    chat_user_text="#ffffff",
    chat_assistant_text="#e8eaf0",
    plan_pending_bg="#151820",
    plan_pending_border="#2a3045",
    plan_running_bg="#14203a",
    plan_running_border="#5b8def",
    plan_completed_bg="#0f2a1a",
    plan_completed_border="#3ecf6e",
    plan_failed_bg="#2a1015",
    plan_failed_border="#f04e5e",

    # Overlays
    overlay_dark="rgba(5, 5, 15, 0.85)",
    overlay_medium="rgba(5, 5, 15, 0.65)",
    surface_success="rgba(62, 207, 110, 0.1)",
    surface_error="rgba(240, 78, 94, 0.1)",
    badge_disabled_bg="rgba(170, 35, 35, 0.95)",
)


# Light theme — cool-white backgrounds
LIGHT_THEME = ThemeColors(
    # Backgrounds — cool-white instead of pure gray
    background_primary="#fafbfe",
    background_secondary="#f0f2f8",
    background_tertiary="#e4e8f0",
    background_elevated="#ffffff",

    # Text
    text_primary="#1a1d28",
    text_secondary="#525a72",
    text_muted="#8b92a8",
    text_inverted="#ffffff",

    # Borders
    border_primary="#c8cdd8",
    border_secondary="#dde0e8",
    border_focus="#5b8def",

    # Accents — same electric blue family
    accent_blue="#5b8def",
    accent_blue_hover="#3a6dd4",
    accent_red="#dc3545",
    accent_green="#28a745",
    accent_orange="#f0a030",
    accent_purple="#9b59b6",

    # Timeline
    timeline_background="#f0f2f8",
    timeline_ruler="#e4e8f0",
    timeline_ruler_border="#c8cdd8",
    timeline_ruler_tick="#525a72",
    timeline_ruler_tick_minor="#8b92a8",
    timeline_track="#e8eaf0",
    timeline_track_highlight="#d0d4e4",
    timeline_clip="#4a7de0",
    timeline_clip_selected="#5b8def",
    timeline_clip_border="#3a65c0",
    timeline_clip_selected_border="#3a6dd4",

    # Components
    thumbnail_background="#e4e8f0",
    card_background="#f0f2f8",
    card_border="#dde0e8",
    card_hover="#e4e8f0",
    badge_analyzed="#28a745",
    badge_not_analyzed="#8b92a8",
    shot_type_badge="#525a72",
    surface_highlight="#e4e8f0",

    # Chat/Plan widget colors
    chat_user_bubble="#5b8def",
    chat_assistant_bubble="#e4e8f0",
    chat_user_text="#ffffff",
    chat_assistant_text="#1a1d28",
    plan_pending_bg="#f0f2f8",
    plan_pending_border="#dde0e8",
    plan_running_bg="#e3f0ff",
    plan_running_border="#5b8def",
    plan_completed_bg="#e8f5e9",
    plan_completed_border="#28a745",
    plan_failed_bg="#ffebee",
    plan_failed_border="#dc3545",

    # Overlays — lighter versions for light theme
    overlay_dark="rgba(0, 0, 0, 0.75)",
    overlay_medium="rgba(0, 0, 0, 0.50)",
    surface_success="rgba(40, 167, 69, 0.1)",
    surface_error="rgba(220, 53, 69, 0.1)",
    badge_disabled_bg="rgba(170, 35, 35, 0.90)",
)


class ThemeSignals(QObject):
    """Signal emitter for theme changes."""
    changed = Signal()  # Emitted when theme preference changes


class Theme:
    """Singleton theme manager with auto-detection of system dark/light mode."""

    _instance: Optional["Theme"] = None
    _signals: Optional[ThemeSignals] = None

    def __new__(cls) -> "Theme":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._preference: str = "system"  # "system", "light", "dark"
        self._cached_is_dark: Optional[bool] = None
        # Create signals object (needs QApplication to exist)
        if Theme._signals is None and QApplication.instance():
            Theme._signals = ThemeSignals()

    @property
    def changed(self) -> Signal:
        """Signal emitted when theme changes. Connect to refresh styles."""
        if Theme._signals is None and QApplication.instance():
            Theme._signals = ThemeSignals()
        return Theme._signals.changed if Theme._signals else None

    def _emit_changed(self):
        """Emit the changed signal if available."""
        if Theme._signals:
            Theme._signals.changed.emit()

    def _detect_system_dark_mode(self) -> bool:
        """Detect if system is in dark mode using QPalette luminance."""
        app = QApplication.instance()
        if not app:
            return True  # Default to dark if no app

        palette = app.palette()
        bg_color = palette.color(QPalette.Window)

        # Calculate luminance - dark mode if background is dark
        luminance = (
            0.299 * bg_color.red() +
            0.587 * bg_color.green() +
            0.114 * bg_color.blue()
        ) / 255

        return luminance < 0.5

    @property
    def preference(self) -> str:
        """Get current theme preference."""
        return self._preference

    @property
    def is_dark(self) -> bool:
        """Check if currently using dark theme."""
        if self._preference == "dark":
            return True
        elif self._preference == "light":
            return False
        else:  # "system"
            return self._detect_system_dark_mode()

    @property
    def colors(self) -> ThemeColors:
        """Get the current theme colors."""
        return DARK_THEME if self.is_dark else LIGHT_THEME

    @property
    def gradient(self) -> GradientPalette:
        """Get the current gradient palette."""
        return DARK_GRADIENT if self.is_dark else LIGHT_GRADIENT

    def set_preference(self, preference: str):
        """Set theme preference: 'system', 'light', or 'dark'."""
        if preference in ("system", "light", "dark"):
            old_is_dark = self.is_dark
            self._preference = preference
            self._cached_is_dark = None
            # Emit changed signal if the effective theme changed
            if self.is_dark != old_is_dark:
                self._emit_changed()

    def set_dark_mode(self, dark: Optional[bool]):
        """Force dark or light mode, or None to auto-detect.

        Deprecated: Use set_preference() instead.
        """
        if dark is None:
            self._preference = "system"
        elif dark:
            self._preference = "dark"
        else:
            self._preference = "light"
        self._cached_is_dark = None

    def invalidate_cache(self):
        """Clear cached detection (call when system theme might have changed)."""
        self._cached_is_dark = None

    def refresh(self):
        """Force emit changed signal to refresh all connected components."""
        self.apply_to_app()
        self._emit_changed()

    def get_app_stylesheet(self) -> str:
        """Generate a comprehensive stylesheet for the entire application."""
        c = self.colors
        r = Radii
        s = Spacing
        return f"""
            /* Main window and widgets */
            QMainWindow, QDialog, QWidget {{
                background-color: {c.background_primary};
                color: {c.text_primary};
            }}

            /* Tab widget */
            QTabWidget::pane {{
                background-color: {c.background_primary};
                border: 1px solid {c.border_secondary};
            }}
            QTabBar::tab {{
                background-color: {c.background_secondary};
                color: {c.text_primary};
                padding: {s.SM}px {s.LG}px;
                border: 1px solid {c.border_secondary};
                border-bottom: none;
                border-top-left-radius: {r.MD}px;
                border-top-right-radius: {r.MD}px;
                margin-right: 2px;
            }}
            QTabBar::tab:selected {{
                background-color: {c.background_primary};
                border-bottom: 2px solid {c.accent_blue};
            }}
            QTabBar::tab:hover:!selected {{
                background-color: {c.background_tertiary};
            }}

            /* Labels */
            QLabel {{
                color: {c.text_primary};
                background-color: transparent;
            }}

            /* Buttons */
            QPushButton {{
                background-color: {c.background_tertiary};
                color: {c.text_primary};
                border: 1px solid {c.border_primary};
                padding: {s.SM}px {s.LG}px;
                border-radius: {r.MD}px;
                min-height: {UISizes.BUTTON_MIN_HEIGHT}px;
            }}
            QPushButton:hover {{
                background-color: {c.background_elevated};
                border-color: {c.border_focus};
            }}
            QPushButton:pressed {{
                background-color: {c.accent_blue};
            }}
            QPushButton:disabled {{
                background-color: {c.background_secondary};
                color: {c.text_muted};
                border-color: {c.border_secondary};
            }}

            /* Line edits and text inputs */
            QLineEdit, QTextEdit, QPlainTextEdit {{
                background-color: {c.background_secondary};
                color: {c.text_primary};
                border: 1px solid {c.border_primary};
                border-radius: {r.MD}px;
                padding: {s.XS}px {s.SM}px;
            }}
            QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
                border-color: {c.border_focus};
            }}
            QLineEdit:disabled, QTextEdit:disabled, QPlainTextEdit:disabled {{
                background-color: {c.background_tertiary};
                color: {c.text_muted};
                border-color: {c.border_secondary};
            }}

            /* Combo boxes */
            QComboBox {{
                background-color: {c.background_secondary};
                color: {c.text_primary};
                border: 1px solid {c.border_primary};
                border-radius: {r.MD}px;
                padding: {s.XS}px {s.SM}px;
                min-height: {UISizes.COMBO_BOX_MIN_HEIGHT}px;
            }}
            QComboBox:hover {{
                border-color: {c.border_focus};
            }}
            QComboBox:disabled {{
                background-color: {c.background_tertiary};
                color: {c.text_muted};
                border-color: {c.border_secondary};
            }}
            QComboBox::drop-down {{
                border: none;
                padding-right: {s.SM}px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {c.background_elevated};
                color: {c.text_primary};
                border: 1px solid {c.border_primary};
                border-radius: {r.MD}px;
                selection-background-color: {c.accent_blue};
            }}

            /* Spin boxes */
            QSpinBox, QDoubleSpinBox {{
                background-color: {c.background_secondary};
                color: {c.text_primary};
                border: 1px solid {c.border_primary};
                border-radius: {r.MD}px;
                padding: {s.XS}px {s.SM}px;
            }}
            QSpinBox:disabled, QDoubleSpinBox:disabled {{
                background-color: {c.background_tertiary};
                color: {c.text_muted};
                border-color: {c.border_secondary};
            }}

            /* Sliders - use explicit background colors to prevent native styling bleed-through on macOS */
            QSlider {{
                background-color: {c.background_secondary};
                border: none;
                padding: 0px;
                min-height: 20px;
                max-height: 20px;
            }}
            QSlider::groove:horizontal {{
                background-color: {c.background_tertiary};
                border: none;
                height: 6px;
                border-radius: 3px;
                margin: 7px 0;
            }}
            QSlider::sub-page:horizontal {{
                background-color: {c.background_secondary};
                border: none;
                height: 6px;
                border-radius: 3px;
                margin: 7px 0;
            }}
            QSlider::add-page:horizontal {{
                background-color: {c.background_secondary};
                border: none;
                height: 6px;
                border-radius: 3px;
                margin: 7px 0;
            }}
            QSlider::handle:horizontal {{
                background-color: {c.accent_blue};
                border: none;
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: {r.MD}px;
            }}
            QSlider::handle:horizontal:hover {{
                background-color: {c.accent_blue_hover};
            }}
            QSlider:disabled::groove:horizontal {{
                background-color: {c.border_secondary};
            }}
            QSlider:disabled::handle:horizontal {{
                background-color: {c.text_muted};
            }}
            /* Vertical sliders */
            QSlider::groove:vertical {{
                background-color: {c.background_tertiary};
                border: none;
                width: 6px;
                border-radius: 3px;
                margin: 0 7px;
            }}
            QSlider::handle:vertical {{
                background-color: {c.accent_blue};
                border: none;
                height: 16px;
                width: 16px;
                margin: 0 -5px;
                border-radius: {r.MD}px;
            }}
            QSlider::handle:vertical:hover {{
                background-color: {c.accent_blue_hover};
            }}
            QSlider:focus::handle:horizontal, QSlider:focus::handle:vertical {{
                border: 2px solid {c.border_focus};
            }}

            /* Checkboxes */
            QCheckBox {{
                color: {c.text_primary};
                spacing: {s.SM}px;
            }}
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border: 1px solid {c.border_primary};
                border-radius: {r.SM}px;
                background-color: {c.background_secondary};
            }}
            QCheckBox::indicator:checked {{
                background-color: {c.accent_blue};
                border-color: {c.accent_blue};
            }}

            /* Group boxes */
            QGroupBox {{
                color: {c.text_primary};
                border: 1px solid {c.border_secondary};
                border-radius: {r.MD}px;
                margin-top: {s.MD}px;
                padding-top: {s.SM}px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 {s.SM}px;
                color: {c.text_primary};
            }}

            /* Scroll areas */
            QScrollArea {{
                background-color: {c.background_primary};
                border: none;
            }}
            QScrollArea > QWidget > QWidget {{
                background-color: {c.background_primary};
            }}

            /* Scroll bars — thinner, more transparent */
            QScrollBar:vertical {{
                background-color: transparent;
                width: 10px;
                border-radius: 5px;
            }}
            QScrollBar::handle:vertical {{
                background-color: {c.background_elevated};
                border-radius: 5px;
                min-height: 20px;
            }}
            QScrollBar::handle:vertical:hover {{
                background-color: {c.border_primary};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            QScrollBar:horizontal {{
                background-color: transparent;
                height: 10px;
                border-radius: 5px;
            }}
            QScrollBar::handle:horizontal {{
                background-color: {c.background_elevated};
                border-radius: 5px;
                min-width: 20px;
            }}
            QScrollBar::handle:horizontal:hover {{
                background-color: {c.border_primary};
            }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                width: 0px;
            }}

            /* Status bar */
            QStatusBar {{
                background-color: {c.background_secondary};
                color: {c.text_secondary};
                border-top: 1px solid {c.border_secondary};
            }}

            /* Menu bar */
            QMenuBar {{
                background-color: {c.background_secondary};
                color: {c.text_primary};
            }}
            QMenuBar::item:selected {{
                background-color: {c.background_tertiary};
            }}

            /* Menus */
            QMenu {{
                background-color: {c.background_elevated};
                color: {c.text_primary};
                border: 1px solid {c.border_primary};
                border-radius: {r.LG}px;
                padding: {s.XS}px;
            }}
            QMenu::item {{
                padding: {s.SM}px {s.LG}px;
                border-radius: {r.SM}px;
            }}
            QMenu::item:selected {{
                background-color: {c.accent_blue};
            }}

            /* Progress bars */
            QProgressBar {{
                background-color: {c.background_tertiary};
                border: 1px solid {c.border_secondary};
                border-radius: {r.MD}px;
                text-align: center;
                color: {c.text_primary};
            }}
            QProgressBar::chunk {{
                background-color: {c.accent_blue};
                border-radius: {r.MD - 1}px;
            }}

            /* Tool tips */
            QToolTip {{
                background-color: {c.background_elevated};
                color: {c.text_primary};
                border: 1px solid {c.border_primary};
                border-radius: {r.MD}px;
                padding: {s.XS}px {s.SM}px;
            }}

            /* Splitters */
            QSplitter::handle {{
                background-color: {c.border_secondary};
            }}
            QSplitter::handle:hover {{
                background-color: {c.border_focus};
            }}

            /* Dialog buttons */
            QDialogButtonBox QPushButton {{
                min-width: 80px;
            }}
        """

    def apply_to_app(self):
        """Apply the current theme stylesheet to the application."""
        app = QApplication.instance()
        if app:
            app.setStyleSheet(self.get_app_stylesheet())

    # Convenience properties for direct color access
    def __getattr__(self, name: str):
        """Allow direct attribute access to colors, e.g., theme().text_secondary."""
        return getattr(self.colors, name)


def theme() -> Theme:
    """Get the global Theme instance.

    Usage:
        from ui.theme import theme

        # Access colors directly:
        color = theme().text_secondary
        bg = theme().background_primary

        # Check mode:
        if theme().is_dark:
            ...

        # Get QColor:
        qcolor = theme().colors.qcolor('accent_blue')

        # Access gradient palette:
        glow_colors = theme().gradient.default_colors
    """
    return Theme()
