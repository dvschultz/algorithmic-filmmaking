"""Individual YouTube search result thumbnail with selection."""

from PySide6.QtWidgets import (
    QFrame,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QCheckBox,
    QWidget,
)
from typing import Optional

from PySide6.QtCore import Qt, Signal, Slot, QUrl
from PySide6.QtGui import QPixmap, QCloseEvent
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply

from ui.theme import theme, UISizes
from core.youtube_api import YouTubeVideo


class YouTubeResultThumbnail(QFrame):
    """Thumbnail widget for a YouTube search result."""

    selection_changed = Signal(str, bool)  # video_id, selected

    def __init__(self, video: YouTubeVideo, network_manager: QNetworkAccessManager, parent=None):
        super().__init__(parent)
        self.video = video
        self._selected = False
        self._network_manager = network_manager  # Shared manager from parent
        self._pending_reply: Optional[QNetworkReply] = None

        self.setFixedSize(UISizes.GRID_CARD_MAX_WIDTH, 186)
        self.setFrameStyle(QFrame.Panel | QFrame.Raised)
        self.setCursor(Qt.PointingHandCursor)

        self._setup_ui()
        self._load_thumbnail()
        self._apply_theme()

        if theme().changed:
            theme().changed.connect(self._apply_theme)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        # Thumbnail with checkbox overlay
        thumb_container = QWidget()
        thumb_container.setFixedSize(228, 128)

        self.thumbnail_label = QLabel(thumb_container)
        self.thumbnail_label.setFixedSize(228, 128)
        self.thumbnail_label.setAlignment(Qt.AlignCenter)
        self.thumbnail_label.setStyleSheet(
            f"background-color: {theme().thumbnail_background};"
        )

        # Checkbox in top-left corner
        self.checkbox = QCheckBox(thumb_container)
        self.checkbox.setGeometry(4, 4, 20, 20)
        self.checkbox.stateChanged.connect(self._on_checkbox_changed)
        self.checkbox.raise_()

        # Duration badge in bottom-right
        self.duration_label = QLabel(thumb_container)
        self.duration_label.setAlignment(Qt.AlignCenter)
        self.duration_label.setStyleSheet(
            "background-color: rgba(0, 0, 0, 0.8); "
            "color: white; font-size: 10px; padding: 2px 4px; "
            "border-radius: 2px;"
        )
        if self.video.duration_str:
            self.duration_label.setText(self.video.duration_str)
            self.duration_label.adjustSize()
            self.duration_label.move(
                228 - self.duration_label.width() - 4,
                128 - self.duration_label.height() - 4,
            )
        else:
            self.duration_label.hide()

        # Resolution badge in top-right (below checkbox area)
        self.resolution_label = QLabel(thumb_container)
        self.resolution_label.setAlignment(Qt.AlignCenter)
        self.resolution_label.setStyleSheet(
            f"background-color: {theme().accent_purple}; "
            "color: white; font-size: 9px; font-weight: bold; padding: 2px 4px; "
            "border-radius: 2px;"
        )
        self.resolution_label.hide()  # Hidden until metadata is fetched
        self.resolution_label.raise_()

        layout.addWidget(thumb_container)

        # Title (truncated)
        self.title_label = QLabel()
        self.title_label.setWordWrap(True)
        self.title_label.setMaximumHeight(28)
        metrics = self.title_label.fontMetrics()
        elided = metrics.elidedText(self.video.title, Qt.ElideRight, 460)
        self.title_label.setText(elided)
        self.title_label.setToolTip(self.video.title)
        self.title_label.setStyleSheet(
            f"font-size: 10px; color: {theme().text_primary};"
        )
        layout.addWidget(self.title_label)

        # Metadata row
        meta_layout = QHBoxLayout()
        meta_layout.setContentsMargins(0, 0, 0, 0)

        # View count
        if self.video.view_count:
            views_text = self._format_view_count(self.video.view_count)
            views_label = QLabel(views_text)
            views_label.setStyleSheet(f"font-size: 9px; color: {theme().text_muted};")
            meta_layout.addWidget(views_label)

        meta_layout.addStretch()

        # HD badge
        if self.video.definition == "hd":
            hd_label = QLabel("HD")
            hd_label.setStyleSheet(
                f"font-size: 9px; color: {theme().text_inverted}; "
                f"background-color: {theme().accent_blue}; "
                "padding: 1px 3px; border-radius: 2px;"
            )
            meta_layout.addWidget(hd_label)

        layout.addLayout(meta_layout)

    def _load_thumbnail(self):
        """Load thumbnail image from URL."""
        if not self.video.thumbnail_url:
            self.thumbnail_label.setText("No thumbnail")
            return

        request = QNetworkRequest(QUrl(self.video.thumbnail_url))
        self._pending_reply = self._network_manager.get(request)
        self._pending_reply.finished.connect(self._on_thumbnail_loaded)

    @Slot()
    def _on_thumbnail_loaded(self):
        """Handle thumbnail download completion."""
        reply = self._pending_reply
        if reply is None:
            return

        if reply.error() == QNetworkReply.NoError:
            data = reply.readAll()
            pixmap = QPixmap()
            pixmap.loadFromData(data)
            if not pixmap.isNull():
                scaled = pixmap.scaled(
                    228, 128, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.thumbnail_label.setPixmap(scaled)

        reply.deleteLater()
        self._pending_reply = None

    def closeEvent(self, event: QCloseEvent):
        """Clean up pending network requests."""
        if self._pending_reply and self._pending_reply.isRunning():
            self._pending_reply.abort()
        super().closeEvent(event)

    def _format_view_count(self, count: int) -> str:
        """Format view count with K/M suffixes."""
        if count >= 1_000_000:
            return f"{count / 1_000_000:.1f}M views"
        elif count >= 1_000:
            return f"{count / 1_000:.0f}K views"
        return f"{count} views"

    @Slot(int)
    def _on_checkbox_changed(self, state: int):
        """Handle checkbox state change."""
        self._selected = state == Qt.CheckState.Checked.value
        self._apply_theme()
        self.selection_changed.emit(self.video.video_id, self._selected)

    def set_selected(self, selected: bool):
        """Set selection state."""
        self.checkbox.setChecked(selected)

    def update_metadata_display(self):
        """Update display after metadata is fetched."""
        if self.video.has_detailed_info and self.video.resolution_str:
            self.resolution_label.setText(self.video.resolution_str)
            self.resolution_label.adjustSize()
            # Position in top-right, below the checkbox
            self.resolution_label.move(
                228 - self.resolution_label.width() - 4,
                4,
            )
            self.resolution_label.show()

    def mousePressEvent(self, event):
        """Toggle selection on click."""
        if event.button() == Qt.LeftButton:
            self.checkbox.setChecked(not self.checkbox.isChecked())

    @Slot()
    def _apply_theme(self):
        """Apply theme-aware styles."""
        if self._selected:
            self.setStyleSheet(
                f"""
                YouTubeResultThumbnail {{
                    background-color: {theme().accent_blue};
                    border: 2px solid {theme().accent_blue_hover};
                }}
            """
            )
        else:
            self.setStyleSheet(
                f"""
                YouTubeResultThumbnail {{
                    background-color: {theme().card_background};
                    border: 1px solid {theme().card_border};
                }}
                YouTubeResultThumbnail:hover {{
                    background-color: {theme().card_hover};
                }}
            """
            )
