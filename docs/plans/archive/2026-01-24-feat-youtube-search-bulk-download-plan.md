---
title: "feat: YouTube Search and Bulk Download"
type: feat
date: 2026-01-24
---

# YouTube Search and Bulk Download

## Overview

Add YouTube video search functionality to the Collect tab using YouTube Data API v3, allowing users to search for videos by title, preview results with thumbnails, select multiple videos, and download them in bulk with parallel processing.

## Problem Statement / Motivation

Currently, users must manually find YouTube videos in their browser, copy URLs one at a time, and import them individually via "Import from URL...". This workflow is tedious for users who want to collect multiple videos on a topic for their collage filmmaking projects.

**User story:** As a video artist, I want to search YouTube directly from Scene Ripper and download multiple videos at once, so I can quickly build my source material library without switching between applications.

## Proposed Solution

Add an expandable search panel to the Collect tab that:
1. Accepts a search query and displays results in a grid with thumbnails
2. Shows video metadata (title, channel, duration, view count)
3. Allows multi-selection via checkboxes
4. Downloads selected videos in parallel (2-3 concurrent) with aggregate progress

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Collect Tab                             │
├─────────────────────────────────────────────────────────────┤
│ [▼ Search YouTube]  [Import from URL...]  [Cut New Videos]  │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────────────────────┐ │
│ │  Search: [________________________] [Search] [Settings] │ │
│ │                                                         │ │
│ │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │ │
│ │  │ [✓]     │ │ [ ]     │ │ [✓]     │ │ [ ]     │       │ │
│ │  │ thumb   │ │ thumb   │ │ thumb   │ │ thumb   │       │ │
│ │  │ Title   │ │ Title   │ │ Title   │ │ Title   │       │ │
│ │  │ 5:32 HD │ │ 12:45   │ │ 3:21 HD │ │ 8:15    │       │ │
│ │  └─────────┘ └─────────┘ └─────────┘ └─────────┘       │ │
│ │                                                         │ │
│ │  [Select All] [Clear] 2 selected    [Download Selected] │ │
│ └─────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                    Video Library                             │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐                       │
│  │ + Add   │ │ Video 1 │ │ Video 2 │                       │
│  │ Video   │ │         │ │         │                       │
│  └─────────┘ └─────────┘ └─────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

## Technical Approach

### New Dependencies

```
# requirements.txt additions
google-api-python-client>=2.100.0
```

### New Files

| File | Purpose |
|------|---------|
| `core/youtube_api.py` | YouTube Data API v3 client wrapper |
| `ui/youtube_search_panel.py` | Expandable search panel widget |
| `ui/youtube_result_thumbnail.py` | Individual result thumbnail with checkbox |

### Modified Files

| File | Changes |
|------|---------|
| `core/settings.py` | Add `youtube_api_key`, `youtube_results_count` settings |
| `ui/settings_dialog.py` | Add "API Keys" tab with YouTube API key field |
| `ui/tabs/collect_tab.py` | Add search panel toggle, integrate YouTubeSearchPanel |
| `ui/main_window.py` | Add `BulkDownloadWorker`, connect signals |

---

## Implementation Phases

### Phase 1: YouTube API Client

**Goal:** Create a robust API client with proper error handling and quota awareness.

#### `core/youtube_api.py`

```python
"""YouTube Data API v3 client for video search."""

from dataclasses import dataclass
from typing import Optional
import re
from datetime import timedelta

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


@dataclass
class YouTubeVideo:
    """Video metadata from YouTube search."""
    video_id: str
    title: str
    description: str
    channel_title: str
    thumbnail_url: str
    duration: Optional[timedelta] = None
    view_count: Optional[int] = None
    definition: Optional[str] = None  # 'hd' or 'sd'
    published_at: Optional[str] = None

    @property
    def youtube_url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}"

    @property
    def duration_str(self) -> str:
        if not self.duration:
            return ""
        total_seconds = int(self.duration.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"


@dataclass
class YouTubeSearchResult:
    """Search result with pagination info."""
    videos: list[YouTubeVideo]
    next_page_token: Optional[str] = None
    total_results: int = 0


class YouTubeAPIError(Exception):
    """Base exception for YouTube API errors."""
    pass


class QuotaExceededError(YouTubeAPIError):
    """Raised when daily quota is exhausted."""
    pass


class InvalidAPIKeyError(YouTubeAPIError):
    """Raised when API key is invalid."""
    pass


class YouTubeSearchClient:
    """Client for YouTube Data API v3 search operations."""

    def __init__(self, api_key: str):
        if not api_key or not api_key.strip():
            raise InvalidAPIKeyError("API key is required")

        self._api_key = api_key
        self._youtube = build(
            'youtube', 'v3',
            developerKey=api_key,
            cache_discovery=False
        )

    def search(
        self,
        query: str,
        max_results: int = 25,
        page_token: Optional[str] = None,
        order: str = 'relevance',
        video_duration: Optional[str] = None,
    ) -> YouTubeSearchResult:
        """
        Search YouTube for videos.

        Args:
            query: Search term
            max_results: 1-50 results per page
            page_token: For pagination
            order: relevance, date, rating, viewCount, title
            video_duration: short (<4min), medium (4-20min), long (>20min)

        Returns:
            YouTubeSearchResult with videos and pagination token

        Raises:
            QuotaExceededError: Daily quota exhausted
            InvalidAPIKeyError: API key is invalid
            YouTubeAPIError: Other API errors
        """
        try:
            # Step 1: Search (100 quota units)
            search_params = {
                'part': 'snippet',
                'q': query,
                'type': 'video',
                'maxResults': min(max_results, 50),
                'order': order,
            }
            if page_token:
                search_params['pageToken'] = page_token
            if video_duration:
                search_params['videoDuration'] = video_duration

            search_response = self._youtube.search().list(**search_params).execute()

            # Extract video IDs
            video_ids = [
                item['id']['videoId']
                for item in search_response.get('items', [])
                if item['id'].get('kind') == 'youtube#video'
            ]

            if not video_ids:
                return YouTubeSearchResult(videos=[], total_results=0)

            # Step 2: Get video details (1 quota unit for up to 50 videos)
            details_response = self._youtube.videos().list(
                part='snippet,contentDetails,statistics',
                id=','.join(video_ids)
            ).execute()

            # Build result objects
            details_map = {v['id']: v for v in details_response.get('items', [])}
            videos = []

            for item in search_response['items']:
                video_id = item['id'].get('videoId')
                if not video_id or video_id not in details_map:
                    continue

                detail = details_map[video_id]
                snippet = item['snippet']

                videos.append(YouTubeVideo(
                    video_id=video_id,
                    title=snippet['title'],
                    description=snippet.get('description', ''),
                    channel_title=snippet.get('channelTitle', ''),
                    thumbnail_url=self._get_thumbnail_url(snippet),
                    duration=self._parse_duration(detail['contentDetails']['duration']),
                    view_count=int(detail['statistics'].get('viewCount', 0)),
                    definition=detail['contentDetails'].get('definition', 'sd'),
                    published_at=snippet.get('publishedAt'),
                ))

            return YouTubeSearchResult(
                videos=videos,
                next_page_token=search_response.get('nextPageToken'),
                total_results=search_response.get('pageInfo', {}).get('totalResults', 0),
            )

        except HttpError as e:
            self._handle_http_error(e)

    def _get_thumbnail_url(self, snippet: dict) -> str:
        """Get best available thumbnail URL."""
        thumbnails = snippet.get('thumbnails', {})
        for size in ['high', 'medium', 'default']:
            if size in thumbnails:
                return thumbnails[size]['url']
        return ''

    def _parse_duration(self, iso_duration: str) -> Optional[timedelta]:
        """Parse ISO 8601 duration (PT1H2M3S) to timedelta."""
        pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?'
        match = re.match(pattern, iso_duration)
        if not match:
            return None

        hours = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        seconds = int(match.group(3) or 0)

        return timedelta(hours=hours, minutes=minutes, seconds=seconds)

    def _handle_http_error(self, error: HttpError):
        """Convert HTTP errors to specific exceptions."""
        status = error.resp.status
        reason = ''
        if error.error_details:
            reason = error.error_details[0].get('reason', '')

        if status == 403:
            if reason == 'quotaExceeded':
                raise QuotaExceededError(
                    "YouTube API daily quota exceeded. Try again tomorrow."
                )
            elif reason in ('keyInvalid', 'forbidden'):
                raise InvalidAPIKeyError(
                    "Invalid YouTube API key. Check your key in Settings."
                )
        elif status == 400:
            raise YouTubeAPIError(f"Invalid request: {error}")

        raise YouTubeAPIError(f"YouTube API error: {error}")
```

**Acceptance Criteria:**
- [x] `YouTubeSearchClient` can search and return results with duration/view count
- [x] Proper exception hierarchy for quota, auth, and general errors
- [x] ISO 8601 duration parsing works correctly
- [x] Unit tests for client with mocked API responses

---

### Phase 2: Settings Integration

**Goal:** Add API key configuration to Settings dialog.

#### `core/settings.py` additions

```python
# Add to Settings dataclass:
youtube_api_key: str = ""
youtube_results_count: int = 25  # 10-50
youtube_parallel_downloads: int = 2  # 1-3
```

#### `ui/settings_dialog.py` additions

Add new "API Keys" tab:

```python
def _create_api_keys_tab(self) -> QWidget:
    """Create the API Keys settings tab."""
    tab = QWidget()
    layout = QVBoxLayout(tab)

    # YouTube API group
    youtube_group = QGroupBox("YouTube Data API")
    youtube_layout = QVBoxLayout(youtube_group)

    # API Key input
    key_layout = QHBoxLayout()
    key_layout.addWidget(QLabel("API Key:"))

    self.youtube_api_key_edit = QLineEdit()
    self.youtube_api_key_edit.setEchoMode(QLineEdit.Password)
    self.youtube_api_key_edit.setPlaceholderText("Enter your YouTube Data API v3 key")
    key_layout.addWidget(self.youtube_api_key_edit)

    self.show_key_btn = QPushButton("Show")
    self.show_key_btn.setCheckable(True)
    self.show_key_btn.toggled.connect(self._toggle_api_key_visibility)
    key_layout.addWidget(self.show_key_btn)

    youtube_layout.addLayout(key_layout)

    # Help text
    help_label = QLabel(
        '<a href="https://console.cloud.google.com/apis/credentials">'
        'Get an API key from Google Cloud Console</a>'
    )
    help_label.setOpenExternalLinks(True)
    help_label.setStyleSheet(f"color: {theme().text_secondary};")
    youtube_layout.addWidget(help_label)

    # Results count
    results_layout = QHBoxLayout()
    results_layout.addWidget(QLabel("Search results:"))

    self.youtube_results_spin = QSpinBox()
    self.youtube_results_spin.setRange(10, 50)
    self.youtube_results_spin.setValue(25)
    self.youtube_results_spin.setToolTip("Number of results per search (affects API quota)")
    results_layout.addWidget(self.youtube_results_spin)
    results_layout.addStretch()

    youtube_layout.addLayout(results_layout)

    # Parallel downloads
    parallel_layout = QHBoxLayout()
    parallel_layout.addWidget(QLabel("Parallel downloads:"))

    self.youtube_parallel_spin = QSpinBox()
    self.youtube_parallel_spin.setRange(1, 3)
    self.youtube_parallel_spin.setValue(2)
    self.youtube_parallel_spin.setToolTip("Number of simultaneous downloads")
    parallel_layout.addWidget(self.youtube_parallel_spin)
    parallel_layout.addStretch()

    youtube_layout.addLayout(parallel_layout)

    layout.addWidget(youtube_group)
    layout.addStretch()

    return tab
```

**Acceptance Criteria:**
- [x] API key stored securely in QSettings
- [x] Password field with show/hide toggle
- [x] Results count configurable (10-50)
- [x] Parallel downloads configurable (1-3)
- [x] Help link to Google Cloud Console

---

### Phase 3: Search Panel UI

**Goal:** Create expandable search panel with results grid.

#### `ui/youtube_search_panel.py`

```python
"""Expandable YouTube search panel for Collect tab."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLineEdit, QPushButton, QLabel, QScrollArea,
    QFrame, QCheckBox, QProgressBar,
)
from PySide6.QtCore import Qt, Signal, Slot, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QPixmap

from ui.theme import theme
from ui.youtube_result_thumbnail import YouTubeResultThumbnail
from core.youtube_api import YouTubeVideo


class YouTubeSearchPanel(QWidget):
    """Collapsible panel for YouTube search and results."""

    # Signals
    download_requested = Signal(list)  # list of YouTubeVideo
    search_started = Signal()
    search_finished = Signal()
    error_occurred = Signal(str)

    COLUMNS = 4

    def __init__(self, parent=None):
        super().__init__(parent)
        self._expanded = False
        self._results: list[YouTubeVideo] = []
        self._thumbnails: list[YouTubeResultThumbnail] = []
        self._selected_videos: set[str] = set()  # video_ids

        self._setup_ui()
        self._connect_theme()

    def _setup_ui(self):
        """Set up the panel UI."""
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # Toggle button
        self.toggle_btn = QPushButton("▶ Search YouTube")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.clicked.connect(self._on_toggle)
        self.main_layout.addWidget(self.toggle_btn)

        # Content container (for animation)
        self.content = QWidget()
        self.content.setMaximumHeight(0)
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(8, 8, 8, 8)

        # Search row
        search_row = QHBoxLayout()

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search YouTube videos...")
        self.search_input.returnPressed.connect(self._on_search)
        search_row.addWidget(self.search_input, 1)

        self.search_btn = QPushButton("Search")
        self.search_btn.clicked.connect(self._on_search)
        search_row.addWidget(self.search_btn)

        self.content_layout.addLayout(search_row)

        # Results scroll area
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setMinimumHeight(200)
        self.scroll.setMaximumHeight(400)

        self.results_container = QWidget()
        self.results_grid = QGridLayout(self.results_container)
        self.results_grid.setSpacing(8)
        self.results_grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        self.scroll.setWidget(self.results_container)
        self.content_layout.addWidget(self.scroll)

        # Status/action row
        action_row = QHBoxLayout()

        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(self._on_select_all)
        action_row.addWidget(self.select_all_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._on_clear_selection)
        action_row.addWidget(self.clear_btn)

        self.status_label = QLabel("")
        action_row.addWidget(self.status_label)

        action_row.addStretch()

        self.download_btn = QPushButton("Download Selected")
        self.download_btn.setEnabled(False)
        self.download_btn.clicked.connect(self._on_download)
        action_row.addWidget(self.download_btn)

        self.content_layout.addLayout(action_row)

        self.main_layout.addWidget(self.content)

        # Animation
        self._animation = QPropertyAnimation(self.content, b"maximumHeight")
        self._animation.setDuration(200)
        self._animation.setEasingCurve(QEasingCurve.InOutQuad)

    def _on_toggle(self, checked: bool):
        """Handle panel expand/collapse."""
        self._expanded = checked

        if checked:
            self.toggle_btn.setText("▼ Search YouTube")
            self._animation.setStartValue(0)
            self._animation.setEndValue(500)  # Max expanded height
        else:
            self.toggle_btn.setText("▶ Search YouTube")
            self._animation.setStartValue(self.content.height())
            self._animation.setEndValue(0)

        self._animation.start()

    def _on_search(self):
        """Emit search request."""
        query = self.search_input.text().strip()
        if query:
            self.search_started.emit()
            # MainWindow will handle the actual search

    def display_results(self, videos: list[YouTubeVideo]):
        """Display search results in the grid."""
        self._clear_results()
        self._results = videos

        for i, video in enumerate(videos):
            thumb = YouTubeResultThumbnail(video)
            thumb.selection_changed.connect(self._on_selection_changed)

            row = i // self.COLUMNS
            col = i % self.COLUMNS
            self.results_grid.addWidget(thumb, row, col, Qt.AlignTop)
            self._thumbnails.append(thumb)

        self._update_status()
        self.search_finished.emit()

    def _clear_results(self):
        """Clear all result thumbnails."""
        for thumb in self._thumbnails:
            self.results_grid.removeWidget(thumb)
            thumb.deleteLater()
        self._thumbnails = []
        self._results = []
        self._selected_videos = set()
        self._update_status()

    def _on_selection_changed(self, video_id: str, selected: bool):
        """Handle thumbnail selection change."""
        if selected:
            self._selected_videos.add(video_id)
        else:
            self._selected_videos.discard(video_id)
        self._update_status()

    def _on_select_all(self):
        """Select all results."""
        for thumb in self._thumbnails:
            thumb.set_selected(True)

    def _on_clear_selection(self):
        """Clear all selections."""
        for thumb in self._thumbnails:
            thumb.set_selected(False)

    def _on_download(self):
        """Request download of selected videos."""
        selected = [v for v in self._results if v.video_id in self._selected_videos]
        if selected:
            self.download_requested.emit(selected)

    def _update_status(self):
        """Update status label and button state."""
        count = len(self._selected_videos)
        if count == 0:
            self.status_label.setText("")
            self.download_btn.setEnabled(False)
        else:
            self.status_label.setText(f"{count} selected")
            self.download_btn.setEnabled(True)

    def set_searching(self, searching: bool):
        """Update UI state during search."""
        self.search_btn.setEnabled(not searching)
        self.search_input.setEnabled(not searching)
        if searching:
            self.search_btn.setText("Searching...")
        else:
            self.search_btn.setText("Search")

    def _connect_theme(self):
        """Connect to theme changes."""
        if theme().changed:
            theme().changed.connect(self._apply_theme)
        self._apply_theme()

    def _apply_theme(self):
        """Apply current theme colors."""
        self.setStyleSheet(f"""
            YouTubeSearchPanel {{
                background-color: {theme().background_secondary};
                border: 1px solid {theme().border_secondary};
                border-radius: 4px;
            }}
        """)
```

#### `ui/youtube_result_thumbnail.py`

```python
"""Individual YouTube search result thumbnail with selection."""

from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox,
)
from PySide6.QtCore import Qt, Signal
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
                97 - self.duration_label.height() - 4
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
        self.title_label.setStyleSheet(f"font-size: 10px; color: {theme().text_primary};")
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
        if self.video.definition == 'hd':
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

    def _on_thumbnail_loaded(self, reply: QNetworkReply):
        """Handle thumbnail download completion."""
        if reply.error() == QNetworkReply.NoError:
            data = reply.readAll()
            pixmap = QPixmap()
            pixmap.loadFromData(data)
            if not pixmap.isNull():
                scaled = pixmap.scaled(
                    172, 97,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
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

    def _on_checkbox_changed(self, state: int):
        """Handle checkbox state change."""
        self._selected = state == Qt.Checked
        self._apply_theme()
        self.selection_changed.emit(self.video.video_id, self._selected)

    def set_selected(self, selected: bool):
        """Set selection state."""
        self.checkbox.setChecked(selected)

    def mousePressEvent(self, event):
        """Toggle selection on click."""
        if event.button() == Qt.LeftButton:
            self.checkbox.setChecked(not self.checkbox.isChecked())

    def _apply_theme(self):
        """Apply theme-aware styles."""
        if self._selected:
            self.setStyleSheet(f"""
                YouTubeResultThumbnail {{
                    background-color: {theme().accent_blue};
                    border: 2px solid {theme().accent_blue_hover};
                }}
            """)
        else:
            self.setStyleSheet(f"""
                YouTubeResultThumbnail {{
                    background-color: {theme().card_background};
                    border: 1px solid {theme().card_border};
                }}
                YouTubeResultThumbnail:hover {{
                    background-color: {theme().card_hover};
                }}
            """)
```

**Acceptance Criteria:**
- [x] Panel expands/collapses with smooth animation
- [x] Search input triggers on Enter key and button click
- [x] Results display in 4-column grid with thumbnails
- [x] Checkboxes allow multi-selection
- [x] "Select All" and "Clear" work correctly
- [x] Selection count updates live
- [x] Theme changes apply correctly

---

### Phase 4: Bulk Download Worker

**Goal:** Implement parallel download queue with progress aggregation.

#### `ui/main_window.py` additions

```python
class BulkDownloadWorker(QThread):
    """Background worker for parallel bulk downloads."""

    # Signals
    progress = Signal(int, int, str)  # current, total, message
    video_finished = Signal(object)  # DownloadResult
    video_error = Signal(str, str)  # video_id, error message
    all_finished = Signal()

    def __init__(self, videos: list[YouTubeVideo], max_parallel: int = 2):
        super().__init__()
        self.videos = videos
        self.max_parallel = max_parallel
        self._cancelled = False
        self._completed = 0
        self._lock = threading.Lock()

    def cancel(self):
        """Request cancellation."""
        self._cancelled = True

    def run(self):
        """Run parallel downloads."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        total = len(self.videos)
        self.progress.emit(0, total, f"Starting download of {total} videos...")

        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            # Submit all downloads
            future_to_video = {
                executor.submit(self._download_one, video): video
                for video in self.videos
            }

            # Process completions
            for future in as_completed(future_to_video):
                if self._cancelled:
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

                video = future_to_video[future]
                try:
                    result = future.result()
                    if result.success:
                        self.video_finished.emit(result)
                    else:
                        self.video_error.emit(video.video_id, result.error or "Download failed")
                except Exception as e:
                    self.video_error.emit(video.video_id, str(e))

                with self._lock:
                    self._completed += 1
                    self.progress.emit(
                        self._completed, total,
                        f"Downloaded {self._completed}/{total}"
                    )

        self.all_finished.emit()

    def _download_one(self, video: YouTubeVideo) -> DownloadResult:
        """Download a single video."""
        downloader = VideoDownloader()
        return downloader.download(
            video.youtube_url,
            cancel_check=lambda: self._cancelled,
        )
```

**Acceptance Criteria:**
- [x] Downloads run in parallel (configurable 1-3)
- [x] Progress shows completed/total count
- [x] Individual video errors don't stop the batch
- [x] Cancellation stops pending downloads
- [x] Each completed video is added to library immediately

---

### Phase 5: Integration

**Goal:** Wire everything together in MainWindow and CollectTab.

#### Signal connections in `ui/main_window.py`:

```python
# In __init__:
self.youtube_client: Optional[YouTubeSearchClient] = None

# Connect search panel signals
self.collect_tab.youtube_search_panel.search_started.connect(self._on_youtube_search)
self.collect_tab.youtube_search_panel.download_requested.connect(self._on_bulk_download)

def _on_youtube_search(self):
    """Handle YouTube search request."""
    if not self.settings.youtube_api_key:
        QMessageBox.warning(
            self, "API Key Required",
            "Please configure your YouTube API key in Settings → API Keys."
        )
        return

    # Initialize client if needed
    if not self.youtube_client:
        try:
            self.youtube_client = YouTubeSearchClient(self.settings.youtube_api_key)
        except InvalidAPIKeyError as e:
            QMessageBox.critical(self, "Invalid API Key", str(e))
            return

    query = self.collect_tab.youtube_search_panel.search_input.text().strip()
    if not query:
        return

    self.collect_tab.youtube_search_panel.set_searching(True)

    # Run search in thread
    self.youtube_search_worker = YouTubeSearchWorker(
        self.youtube_client, query, self.settings.youtube_results_count
    )
    self.youtube_search_worker.finished.connect(self._on_youtube_search_finished)
    self.youtube_search_worker.error.connect(self._on_youtube_search_error)
    self.youtube_search_worker.start()

def _on_youtube_search_finished(self, result: YouTubeSearchResult):
    """Handle search completion."""
    self.collect_tab.youtube_search_panel.set_searching(False)
    self.collect_tab.youtube_search_panel.display_results(result.videos)

def _on_youtube_search_error(self, error: str):
    """Handle search error."""
    self.collect_tab.youtube_search_panel.set_searching(False)
    QMessageBox.critical(self, "Search Failed", error)

def _on_bulk_download(self, videos: list[YouTubeVideo]):
    """Start bulk download of selected videos."""
    self.progress_bar.setVisible(True)
    self.progress_bar.setRange(0, len(videos))
    self.progress_bar.setValue(0)

    self.bulk_download_worker = BulkDownloadWorker(
        videos, self.settings.youtube_parallel_downloads
    )
    self.bulk_download_worker.progress.connect(self._on_bulk_progress)
    self.bulk_download_worker.video_finished.connect(self._on_bulk_video_finished)
    self.bulk_download_worker.video_error.connect(self._on_bulk_video_error)
    self.bulk_download_worker.all_finished.connect(self._on_bulk_finished)
    self.bulk_download_worker.start()

def _on_bulk_progress(self, current: int, total: int, message: str):
    """Update bulk download progress."""
    self.progress_bar.setValue(current)
    self.status_bar.showMessage(message)

def _on_bulk_video_finished(self, result: DownloadResult):
    """Handle single video download completion."""
    if result.file_path and result.file_path.exists():
        self._load_video(result.file_path)

def _on_bulk_video_error(self, video_id: str, error: str):
    """Log individual video download error."""
    logger.warning(f"Failed to download {video_id}: {error}")

def _on_bulk_finished(self):
    """Handle bulk download completion."""
    self.progress_bar.setVisible(False)
    self.status_bar.showMessage("Bulk download complete")
```

**Acceptance Criteria:**
- [x] Search triggers API call with configured parameters
- [x] Missing API key shows helpful error with link to Settings
- [x] Search errors display user-friendly messages
- [x] Bulk download adds each video to library as it completes
- [x] Progress bar shows overall progress
- [x] Failed downloads are logged but don't stop batch

---

## Critical Implementation Notes

### From Institutional Learnings (MUST APPLY)

1. **QThread Signal Guards** (from `qthread-destroyed-duplicate-signal-delivery-20260124.md`)
   - Use guard flags in all signal handlers that create resources
   - Use `Qt.UniqueConnection` when connecting signals
   - Add `@Slot()` decorators on all handlers

2. **Subprocess Cleanup** (from `subprocess-cleanup-on-exception.md`)
   - Wrap all yt-dlp calls in try/finally
   - Implement terminate → wait(5s) → kill sequence
   - The existing `VideoDownloader` already handles this

3. **URL Validation** (from `url-scheme-validation-bypass.md`)
   - YouTube URLs constructed internally are safe
   - The existing domain whitelist covers youtube.com/youtu.be

---

## Acceptance Criteria (Full Feature)

### Functional Requirements
- [x] User can expand/collapse search panel in Collect tab
- [x] Search returns results with thumbnails, titles, duration, view count
- [x] Multi-select via checkboxes works correctly
- [x] "Select All" and "Clear" buttons function
- [x] Bulk download processes videos in parallel
- [x] Each downloaded video appears in library immediately
- [x] Progress shows current/total count
- [x] Cancellation stops pending downloads gracefully

### Non-Functional Requirements
- [x] API quota usage minimized (search + videos.list pattern)
- [x] Errors handled gracefully with user-friendly messages
- [x] Theme changes apply to all new components
- [x] No memory leaks from network requests or workers

### Quality Gates
- [x] Unit tests for YouTubeSearchClient
- [ ] Integration test for search → select → download flow
- [ ] Manual test: search, select 5 videos, download, verify in library

---

## Dependencies & Prerequisites

- Google Cloud project with YouTube Data API v3 enabled
- User must obtain their own API key
- `google-api-python-client` package

---

## References

### Internal References
- Existing download pattern: `core/downloader.py`
- Worker pattern: `ui/main_window.py:122-151` (DownloadWorker)
- Source browser grid: `ui/source_browser.py`
- Settings system: `core/settings.py`, `ui/settings_dialog.py`

### External References
- [YouTube Data API v3 Search](https://developers.google.com/youtube/v3/docs/search/list)
- [YouTube Data API v3 Videos](https://developers.google.com/youtube/v3/docs/videos)
- [API Quota Calculator](https://developers.google.com/youtube/v3/determine_quota_cost)
- [google-api-python-client](https://github.com/googleapis/google-api-python-client)

### Institutional Learnings Applied
- `docs/solutions/runtime-errors/qthread-destroyed-duplicate-signal-delivery-20260124.md`
- `docs/solutions/reliability-issues/subprocess-cleanup-on-exception.md`
- `docs/solutions/security-issues/url-scheme-validation-bypass.md`
