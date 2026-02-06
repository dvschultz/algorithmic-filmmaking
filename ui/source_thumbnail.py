"""Thumbnail widget for video sources in the library grid."""

from pathlib import Path

from PySide6.QtWidgets import (
    QFrame,
    QVBoxLayout,
    QLabel,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QKeyEvent

from models.clip import Source
from ui.theme import theme, UISizes, TypeScale, Spacing, Radii
from ui.gradient_glow import RoundedTopLabel


class SourceThumbnail(QFrame):
    """Individual source video thumbnail widget."""

    clicked = Signal(object)  # Source
    double_clicked = Signal(object)  # Source

    def __init__(self, source: Source):
        super().__init__()
        self.source = source
        self.selected = False

        self.setFrameStyle(QFrame.NoFrame)
        self.setFixedSize(UISizes.GRID_CARD_MAX_WIDTH, 200)
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAccessibleName(source.filename)
        self.setAccessibleDescription(f"Video source: {source.filename}")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(Spacing.XS, Spacing.XS, Spacing.XS, Spacing.XS)
        layout.setSpacing(Spacing.XS)

        # Thumbnail image (rounded top corners to match card radius)
        self.thumbnail_label = RoundedTopLabel(Radii.MD)
        self.thumbnail_label.setFixedSize(220, 124)
        self.thumbnail_label.setAlignment(Qt.AlignCenter)
        self.thumbnail_label.setStyleSheet(f"background-color: {theme().thumbnail_background};")

        if source.thumbnail_path and source.thumbnail_path.exists():
            self._load_thumbnail(source.thumbnail_path)
        else:
            self.thumbnail_label.setText("No Preview")
            self.thumbnail_label.setStyleSheet(
                f"background-color: {theme().thumbnail_background}; color: {theme().text_muted}; font-size: {TypeScale.SM}px;"
            )

        layout.addWidget(self.thumbnail_label)

        # Filename label
        self.filename_label = QLabel(source.filename)
        self.filename_label.setAlignment(Qt.AlignCenter)
        self.filename_label.setStyleSheet(f"font-size: {TypeScale.SM}px; color: {theme().text_primary};")
        self.filename_label.setWordWrap(True)
        self.filename_label.setMaximumHeight(30)
        # Truncate long filenames
        metrics = self.filename_label.fontMetrics()
        elided = metrics.elidedText(source.filename, Qt.ElideMiddle, 220)
        self.filename_label.setText(elided)
        self.filename_label.setToolTip(source.filename)
        layout.addWidget(self.filename_label)

        # Analyzed badge
        self.badge_label = QLabel()
        self.badge_label.setAlignment(Qt.AlignCenter)
        self.badge_label.setFixedHeight(16)
        self._update_badge()
        layout.addWidget(self.badge_label)

        self._update_style()

        # Connect to theme changes
        if theme().changed:
            theme().changed.connect(self._refresh_theme)

    def _load_thumbnail(self, path: Path):
        """Load thumbnail image."""
        pixmap = QPixmap(str(path))
        if not pixmap.isNull():
            scaled = pixmap.scaled(
                220, 124,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.thumbnail_label.setPixmap(scaled)

    def set_thumbnail(self, path: Path):
        """Update the thumbnail image."""
        self.source.thumbnail_path = path
        self._load_thumbnail(path)

    def set_analyzed(self, analyzed: bool):
        """Update the analyzed status."""
        self.source.analyzed = analyzed
        self._update_badge()

    def _update_badge(self):
        """Update the analyzed badge display."""
        if self.source.analyzed:
            self.badge_label.setText("Analyzed")
            self.badge_label.setStyleSheet(
                f"font-size: {TypeScale.XS}px; color: {theme().badge_analyzed_text}; background-color: {theme().badge_analyzed}; "
                f"border-radius: {Radii.SM}px; padding: {Spacing.XXS}px {Spacing.SM}px;"
            )
        else:
            self.badge_label.setText("Not Analyzed")
            self.badge_label.setStyleSheet(
                f"font-size: {TypeScale.XS}px; color: {theme().text_inverted}; background-color: {theme().badge_not_analyzed}; "
                f"border-radius: {Radii.SM}px; padding: {Spacing.XXS}px {Spacing.SM}px;"
            )

    def set_selected(self, selected: bool):
        """Set selection state."""
        self.selected = selected
        self._update_style()

    def _update_style(self):
        """Update visual style based on state."""
        if self.selected:
            self.setStyleSheet(f"""
                SourceThumbnail {{
                    background-color: {theme().card_background};
                    border: 2px solid {theme().accent_blue};
                    border-radius: {Radii.MD}px;
                }}
            """)
        else:
            self.setStyleSheet(f"""
                SourceThumbnail {{
                    background-color: {theme().card_background};
                    border: 1px solid {theme().card_border};
                    border-radius: {Radii.MD}px;
                }}
                SourceThumbnail:hover {{
                    background-color: {theme().card_hover};
                    border: 1px solid {theme().border_focus};
                    border-radius: {Radii.MD}px;
                }}
            """)

    def _refresh_theme(self):
        """Refresh all themed styles when theme changes."""
        self._update_style()
        self._update_badge()
        # Update thumbnail background if no preview
        if not self.thumbnail_label.pixmap():
            self.thumbnail_label.setStyleSheet(
                f"background-color: {theme().thumbnail_background}; color: {theme().text_muted}; font-size: {TypeScale.SM}px;"
            )
        else:
            self.thumbnail_label.setStyleSheet(f"background-color: {theme().thumbnail_background};")
        # Update filename label
        self.filename_label.setStyleSheet(f"font-size: {TypeScale.SM}px; color: {theme().text_primary};")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.source)

    def mouseDoubleClickEvent(self, event):
        self.double_clicked.emit(self.source)

    def keyPressEvent(self, event: QKeyEvent):
        """Handle keyboard events for accessibility."""
        if event.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            self.clicked.emit(self.source)
        else:
            super().keyPressEvent(event)

    def focusInEvent(self, event):
        """Handle focus gained - add visual indicator."""
        super().focusInEvent(event)
        if not self.selected:
            self.setStyleSheet(f"""
                SourceThumbnail {{
                    background-color: {theme().card_hover};
                    border: 2px solid {theme().border_focus};
                    border-radius: {Radii.MD}px;
                }}
            """)

    def focusOutEvent(self, event):
        """Handle focus lost - remove visual indicator."""
        super().focusOutEvent(event)
        self._update_style()
