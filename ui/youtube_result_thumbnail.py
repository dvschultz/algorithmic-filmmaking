"""Individual YouTube search result thumbnail with selection."""

from PySide6.QtWidgets import (
    QFrame,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QCheckBox,
    QWidget,
)
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QPixmap
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply

from ui.theme import theme
from core.youtube_api import YouTubeVideo


class YouTubeResultThumbnail(QFrame):
    """Thumbnail widget for a YouTube search result."""

    selection_changed = Signal(str, bool)  # video_id, selected

    def __init__(self, video: YouTubeVideo, parent=None):
        super().__init__(parent)
        self.video = video
        self._selected = False
        self._network_manager = None

        self.setFixedSize(180, 140)
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
        thumb_container.setFixedSize(172, 97)

        self.thumbnail_label = QLabel(thumb_container)
        self.thumbnail_label.setFixedSize(172, 97)
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
                172 - self.duration_label.width() - 4,
                97 - self.duration_label.height() - 4,
            )
        else:
            self.duration_label.hide()

        layout.addWidget(thumb_container)

        # Title (truncated)
        self.title_label = QLabel()
        self.title_label.setWordWrap(True)
        self.title_label.setMaximumHeight(28)
        metrics = self.title_label.fontMetrics()
        elided = metrics.elidedText(self.video.title, Qt.ElideRight, 340)
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

        self._network_manager = QNetworkAccessManager(self)
        self._network_manager.finished.connect(self._on_thumbnail_loaded)

        request = QNetworkRequest(self.video.thumbnail_url)
        self._network_manager.get(request)

    @Slot(QNetworkReply)
    def _on_thumbnail_loaded(self, reply: QNetworkReply):
        """Handle thumbnail download completion."""
        if reply.error() == QNetworkReply.NoError:
            data = reply.readAll()
            pixmap = QPixmap()
            pixmap.loadFromData(data)
            if not pixmap.isNull():
                scaled = pixmap.scaled(
                    172, 97, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.thumbnail_label.setPixmap(scaled)
        reply.deleteLater()

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
