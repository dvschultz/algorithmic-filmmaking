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

from dataclasses import dataclass
from typing import Optional

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QPalette, QColor
from PySide6.QtCore import QObject, Signal


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

    def qcolor(self, attr: str) -> QColor:
        """Get a QColor for the given attribute name."""
        color_str = getattr(self, attr)
        return QColor(color_str)


# Dark theme color definitions
DARK_THEME = ThemeColors(
    # Backgrounds
    background_primary="#1e1e1e",
    background_secondary="#2a2a2a",
    background_tertiary="#3a3a3a",
    background_elevated="#454545",

    # Text
    text_primary="#ffffff",
    text_secondary="#aaaaaa",
    text_muted="#888888",
    text_inverted="#ffffff",

    # Borders
    border_primary="#555555",
    border_secondary="#444444",
    border_focus="#4a90d9",

    # Accents
    accent_blue="#4a90d9",
    accent_blue_hover="#5a9eff",
    accent_red="#ff4444",
    accent_green="#4CAF50",
    accent_orange="#f0ad4e",
    accent_purple="#9b59b6",

    # Timeline
    timeline_background="#1e1e1e",
    timeline_ruler="#2a2a2a",
    timeline_ruler_border="#444444",
    timeline_ruler_tick="#888888",
    timeline_ruler_tick_minor="#555555",
    timeline_track="#2d2d2d",
    timeline_track_highlight="#3d3d4d",
    timeline_clip="#4a7dc9",
    timeline_clip_selected="#5a9eff",
    timeline_clip_border="#3a6db9",
    timeline_clip_selected_border="#ffffff",

    # Components
    thumbnail_background="#333333",
    card_background="#3a3a3a",
    card_border="#555555",
    card_hover="#454545",
    badge_analyzed="#4CAF50",
    badge_not_analyzed="#999999",
    shot_type_badge="#666666",
)


# Light theme color definitions
LIGHT_THEME = ThemeColors(
    # Backgrounds
    background_primary="#ffffff",
    background_secondary="#f5f5f5",
    background_tertiary="#e8e8e8",
    background_elevated="#ffffff",

    # Text
    text_primary="#333333",
    text_secondary="#666666",
    text_muted="#999999",
    text_inverted="#ffffff",

    # Borders
    border_primary="#cccccc",
    border_secondary="#dddddd",
    border_focus="#4a90d9",

    # Accents
    accent_blue="#4a90d9",
    accent_blue_hover="#2d6cb5",
    accent_red="#dc3545",
    accent_green="#28a745",
    accent_orange="#f0ad4e",
    accent_purple="#9b59b6",

    # Timeline
    timeline_background="#f0f0f0",
    timeline_ruler="#e0e0e0",
    timeline_ruler_border="#cccccc",
    timeline_ruler_tick="#666666",
    timeline_ruler_tick_minor="#aaaaaa",
    timeline_track="#e5e5e5",
    timeline_track_highlight="#d0d0e0",
    timeline_clip="#4a7dc9",
    timeline_clip_selected="#5a9eff",
    timeline_clip_border="#3a6db9",
    timeline_clip_selected_border="#2d6cb5",

    # Components
    thumbnail_background="#e0e0e0",
    card_background="#f5f5f5",
    card_border="#dddddd",
    card_hover="#e8e8e8",
    badge_analyzed="#28a745",
    badge_not_analyzed="#999999",
    shot_type_badge="#666666",
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
                padding: 8px 16px;
                border: 1px solid {c.border_secondary};
                border-bottom: none;
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
                padding: 6px 12px;
                border-radius: 4px;
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
                border-radius: 4px;
                padding: 4px 8px;
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
                border-radius: 4px;
                padding: 4px 8px;
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
                padding-right: 8px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {c.background_elevated};
                color: {c.text_primary};
                border: 1px solid {c.border_primary};
                selection-background-color: {c.accent_blue};
            }}

            /* Spin boxes */
            QSpinBox, QDoubleSpinBox {{
                background-color: {c.background_secondary};
                color: {c.text_primary};
                border: 1px solid {c.border_primary};
                border-radius: 4px;
                padding: 4px 8px;
            }}
            QSpinBox:disabled, QDoubleSpinBox:disabled {{
                background-color: {c.background_tertiary};
                color: {c.text_muted};
                border-color: {c.border_secondary};
            }}

            /* Sliders */
            QSlider {{
                background: transparent;
            }}
            QSlider::groove:horizontal {{
                background-color: {c.background_tertiary};
                height: 6px;
                border-radius: 3px;
            }}
            QSlider::sub-page:horizontal {{
                background: transparent;
            }}
            QSlider::add-page:horizontal {{
                background: transparent;
            }}
            QSlider::handle:horizontal {{
                background-color: {c.accent_blue};
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
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

            /* Checkboxes */
            QCheckBox {{
                color: {c.text_primary};
                spacing: 8px;
            }}
            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
                border: 1px solid {c.border_primary};
                border-radius: 3px;
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
                border-radius: 4px;
                margin-top: 12px;
                padding-top: 8px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 8px;
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

            /* Scroll bars */
            QScrollBar:vertical {{
                background-color: {c.background_secondary};
                width: 12px;
                border-radius: 6px;
            }}
            QScrollBar::handle:vertical {{
                background-color: {c.background_elevated};
                border-radius: 6px;
                min-height: 20px;
            }}
            QScrollBar::handle:vertical:hover {{
                background-color: {c.border_primary};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            QScrollBar:horizontal {{
                background-color: {c.background_secondary};
                height: 12px;
                border-radius: 6px;
            }}
            QScrollBar::handle:horizontal {{
                background-color: {c.background_elevated};
                border-radius: 6px;
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
            }}
            QMenu::item:selected {{
                background-color: {c.accent_blue};
            }}

            /* Progress bars */
            QProgressBar {{
                background-color: {c.background_tertiary};
                border: 1px solid {c.border_secondary};
                border-radius: 4px;
                text-align: center;
                color: {c.text_primary};
            }}
            QProgressBar::chunk {{
                background-color: {c.accent_blue};
                border-radius: 3px;
            }}

            /* Tool tips */
            QToolTip {{
                background-color: {c.background_elevated};
                color: {c.text_primary};
                border: 1px solid {c.border_primary};
                padding: 4px 8px;
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
    """
    return Theme()
