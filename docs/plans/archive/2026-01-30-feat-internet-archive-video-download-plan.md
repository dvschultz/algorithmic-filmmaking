---
title: "feat: Add Internet Archive Video Search and Download"
type: feat
date: 2026-01-30
---

# feat: Add Internet Archive Video Search and Download

## Overview

Extend the existing URL import feature to support Internet Archive (archive.org) video downloads. This includes adding a source selector combobox to the existing YouTube search panel so users can switch between YouTube and Internet Archive as their search source.

## Problem Statement / Motivation

Users currently can only search and bulk-download videos from YouTube. The Internet Archive hosts a vast collection of public domain films, documentaries, and archival footage that would be valuable for video editing projects. By adding Internet Archive support:

- Users get access to copyright-free public domain content
- Same familiar search/filter/bulk-download workflow
- Unified UI reduces complexity vs. multiple separate panels

## Proposed Solution

### High-Level Approach

1. **Refactor YouTubeSearchPanel → UnifiedSearchPanel** with a source selector combobox
2. **Create InternetArchiveClient** for searching and getting video metadata from archive.org
3. **Add archive.org domains to VideoDownloader whitelist** (yt-dlp already supports archive.org)
4. **Create InternetArchiveVideo data model** parallel to YouTubeVideo
5. **Update result thumbnail widget** to handle both video types

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    UnifiedSearchPanel                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ [Source: YouTube ▼]  [Search: ___________] [Search]  │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Filters: [Aspect ▼] [Resolution ▼] [Size ▼]          │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                   Results Grid                        │   │
│  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                     │   │
│  │  │ vid │ │ vid │ │ vid │ │ vid │  ...                │   │
│  │  └─────┘ └─────┘ └─────┘ └─────┘                     │   │
│  └──────────────────────────────────────────────────────┘   │
│  [Select All] [Clear]  "3 selected"    [Download Selected]  │
└─────────────────────────────────────────────────────────────┘
         │                              │
         │ search_requested(source, query)
         │                              │
         ▼                              ▼
┌─────────────────┐          ┌─────────────────────┐
│YouTubeSearchClient│         │InternetArchiveClient│
│(core/youtube_api.py)│       │(core/internet_archive_api.py)│
└─────────────────┘          └─────────────────────┘
         │                              │
         └──────────────┬───────────────┘
                        ▼
              ┌─────────────────┐
              │ VideoDownloader │
              │ (yt-dlp)        │
              └─────────────────┘
```

## Technical Approach

### Phase 1: Internet Archive API Client

Create `core/internet_archive_api.py` following the pattern from `core/youtube_api.py`:

```python
# core/internet_archive_api.py

@dataclass
class InternetArchiveVideo:
    """Data model for an Internet Archive video item."""
    identifier: str  # IA item identifier (unique)
    title: str
    description: str
    creator: str | None
    date: str | None
    duration_seconds: float | None
    thumbnail_url: str | None
    item_url: str  # https://archive.org/details/{identifier}
    download_url: str  # Direct MP4/video URL

    # Metadata (populated after detail fetch)
    width: int | None = None
    height: int | None = None
    filesize_approx: int | None = None
    has_detailed_info: bool = False

    @property
    def duration_str(self) -> str:
        """Format duration as HH:MM:SS."""
        ...


class InternetArchiveClient:
    """Client for searching Internet Archive videos."""

    # Default to video media types
    VIDEO_MEDIATYPES = ["movies", "feature_films", "short_films", "animation"]

    SCRAPE_API_URL = "https://archive.org/services/search/v1/scrape"

    def search(
        self,
        query: str,
        max_results: int = 25,
        mediatype: str | None = None,
    ) -> list[InternetArchiveVideo]:
        """Search for videos on Internet Archive."""
        ...

    def get_item_details(self, identifier: str) -> dict:
        """Get detailed metadata for an item."""
        ...
```

**Key Implementation Details:**

- Use the [scraping API](https://archive.org/services/search/v1/scrape) for searches
- Query format: `mediatype:(movies OR feature_films) AND {user_query}`
- No API key required for public searches
- Fields to request: `identifier,title,description,creator,date,runtime`
- Thumbnail URL pattern: `https://archive.org/services/img/{identifier}`

**Files to create:**
- `core/internet_archive_api.py` (~150 lines)

### Phase 2: Update VideoDownloader Domain Whitelist

Add archive.org domains to the allowed list in `core/downloader.py`:

```python
# core/downloader.py:62-72

ALLOWED_DOMAINS = {
    # YouTube
    "youtube.com",
    "www.youtube.com",
    "youtu.be",
    # Vimeo
    "vimeo.com",
    "www.vimeo.com",
    "player.vimeo.com",
    # Internet Archive (NEW)
    "archive.org",
    "www.archive.org",
    "ia800.us.archive.org",  # CDN pattern
    "ia900.us.archive.org",
    "ia*.us.archive.org",    # Wildcard for CDN servers
}
```

**Note:** yt-dlp already supports archive.org downloads. We just need to whitelist the domains.

**Files to modify:**
- `core/downloader.py:62-72` - Add archive.org domains

### Phase 3: Refactor Search Panel with Source Selector

Rename and extend `ui/youtube_search_panel.py` to support multiple sources:

```python
# ui/video_search_panel.py (renamed from youtube_search_panel.py)

class VideoSearchPanel(QWidget):
    """Collapsible panel for video search with source selection."""

    # Signals
    search_requested = Signal(str, str)  # source, query
    download_requested = Signal(str, list)  # source, list of videos

    class SearchSource(Enum):
        YOUTUBE = "youtube"
        INTERNET_ARCHIVE = "internet_archive"

    def _setup_ui(self):
        ...
        # Source selector (NEW - before search input)
        source_row = QHBoxLayout()

        source_label = QLabel("Source:")
        source_row.addWidget(source_label)

        self.source_combo = QComboBox()
        self.source_combo.addItem("YouTube", SearchSource.YOUTUBE)
        self.source_combo.addItem("Internet Archive", SearchSource.INTERNET_ARCHIVE)
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        source_row.addWidget(self.source_combo)

        self.content_layout.addLayout(source_row)
        ...

    def _on_source_changed(self):
        """Handle source selection change - update placeholder and filters."""
        source = self.source_combo.currentData()
        if source == self.SearchSource.YOUTUBE:
            self.search_input.setPlaceholderText("Search YouTube videos...")
            self.toggle_btn.setText("Search Videos")
        else:
            self.search_input.setPlaceholderText("Search Internet Archive...")
            self.toggle_btn.setText("Search Videos")

        # Clear results when source changes
        self._clear_results()
```

**Files to modify:**
- `ui/youtube_search_panel.py` → rename to `ui/video_search_panel.py`
- Update imports in `ui/tabs/collect_tab.py`
- Update imports in `ui/main_window.py`

### Phase 4: Create Generic Video Result Model

Create a protocol/base class that both YouTubeVideo and InternetArchiveVideo implement:

```python
# models/video_result.py

from typing import Protocol, runtime_checkable

@runtime_checkable
class VideoResult(Protocol):
    """Protocol for video search result items."""

    @property
    def id(self) -> str:
        """Unique identifier for this video."""
        ...

    @property
    def title(self) -> str:
        ...

    @property
    def thumbnail_url(self) -> str | None:
        ...

    @property
    def duration_str(self) -> str:
        ...

    @property
    def download_url(self) -> str:
        """URL to pass to VideoDownloader."""
        ...

    # Filter properties
    width: int | None
    height: int | None
    aspect_ratio: float | None
    filesize_approx: int | None
    has_detailed_info: bool

    def matches_aspect_ratio(self, filter_value: str) -> bool:
        ...

    def matches_resolution(self, filter_value: str) -> bool:
        ...

    def matches_max_size(self, filter_value: str) -> bool:
        ...
```

**Files to create:**
- `models/video_result.py` (~50 lines)

**Files to modify:**
- `core/youtube_api.py` - Update YouTubeVideo to implement protocol
- `core/internet_archive_api.py` - InternetArchiveVideo implements protocol

### Phase 5: Update Result Thumbnail Widget

Modify `ui/youtube_result_thumbnail.py` to work with the generic `VideoResult` protocol:

```python
# ui/video_result_thumbnail.py (renamed from youtube_result_thumbnail.py)

class VideoResultThumbnail(QWidget):
    """Thumbnail widget for video search results from any source."""

    def __init__(self, video: VideoResult, network_manager: QNetworkAccessManager):
        ...
        self.video = video  # Now accepts any VideoResult
```

**Files to modify:**
- `ui/youtube_result_thumbnail.py` → rename to `ui/video_result_thumbnail.py`
- Update type hints to use `VideoResult` protocol
- Update imports throughout

### Phase 6: Wire Up MainWindow

Update `ui/main_window.py` to handle both search sources:

```python
# ui/main_window.py

class InternetArchiveSearchWorker(QThread):
    """Background worker for Internet Archive searches."""
    results_ready = Signal(list)  # list of InternetArchiveVideo
    error_occurred = Signal(str)

    def __init__(self, query: str, max_results: int = 25):
        super().__init__()
        self.query = query
        self.max_results = max_results

    def run(self):
        from core.internet_archive_api import InternetArchiveClient
        client = InternetArchiveClient()
        try:
            results = client.search(self.query, self.max_results)
            self.results_ready.emit(results)
        except Exception as e:
            self.error_occurred.emit(str(e))

# In MainWindow:
def _on_video_search(self, source: str, query: str):
    """Handle search request from unified search panel."""
    if source == "youtube":
        self._on_youtube_search(query)  # existing
    elif source == "internet_archive":
        self._on_internet_archive_search(query)  # new

def _on_internet_archive_search(self, query: str):
    """Search Internet Archive."""
    self._ia_search_worker = InternetArchiveSearchWorker(query)
    self._ia_search_worker.results_ready.connect(
        self.video_search_panel.display_results
    )
    self._ia_search_worker.error_occurred.connect(self._on_search_error)
    self._ia_search_worker.start()
```

**Files to modify:**
- `ui/main_window.py` - Add IA search worker, update signal handlers

### Phase 7: Settings Integration (Optional)

Add Internet Archive settings to preferences:

```python
# core/settings.py - additions

# Internet Archive settings
internet_archive_results_count: int = 25  # 10-50
internet_archive_default_mediatype: str = "movies"  # or "any"
```

**Files to modify:**
- `core/settings.py` - Add IA settings
- `ui/settings_dialog.py` - Add IA settings UI (optional for MVP)

## Acceptance Criteria

### Functional Requirements

- [x] Source selector combobox in search panel with YouTube and Internet Archive options
- [x] Switching sources clears current results, updates placeholder text, and cancels in-progress searches
- [x] Internet Archive search returns video results with thumbnails
- [x] Results display in same grid layout as YouTube results
- [x] Filters (aspect ratio, resolution, size) work with IA results once metadata is fetched
- [x] Bulk download works for selected IA videos
- [x] Single URL import accepts archive.org URLs
- [x] Error handling for IA API failures displays user-friendly messages
- [x] Handle missing thumbnails gracefully (some IA items have no thumbnail)
- [x] Show "No results found" message when IA search returns empty results
- [x] Show loading indicator during IA search

### State Management

- [x] Selected search source persists when panel is collapsed/expanded
- [x] Selected search source persists across tab switches
- [x] Switching sources cancels any in-progress metadata fetch
- [x] Starting a new search cancels any in-progress search

### Result Display

- [x] IA results show creator name where YouTube shows channel name
- [x] IA results show date where applicable
- [x] Handle IA descriptions that are HTML-formatted (strip tags) or very long (truncate)

### Non-Functional Requirements

- [ ] Search response time < 5s for typical queries
- [ ] No API key required for Internet Archive searches
- [ ] Memory usage comparable to YouTube search (no leaks)
- [ ] Background metadata fetch doesn't block UI

### Quality Gates

- [ ] Unit tests for InternetArchiveClient
  - [ ] Test search with empty query
  - [ ] Test search with no results
  - [ ] Test runtime parsing edge cases (HH:MM:SS, MM:SS, seconds, decimals, empty)
  - [ ] Test API timeout handling
  - [ ] Test malformed JSON response handling
- [ ] Integration tests for search → download flow
- [ ] Manual testing of edge cases (no results, API timeout, invalid URLs)

## Dependencies & Prerequisites

- yt-dlp already supports archive.org (verified)
- No additional Python packages required (use `urllib` or existing `requests`)
- Internet Archive scraping API is public, no authentication needed

## Risk Analysis & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| IA API rate limiting | Low | Medium | Add retry logic with exponential backoff; set User-Agent header |
| IA API changes | Low | High | Abstract API calls, document version used |
| yt-dlp IA extractor breaks | Medium | High | Pin yt-dlp version, monitor releases |
| Some IA items have no video files | Medium | Low | Filter by file format in search, graceful error handling |
| IA CDN domain pattern changes | Low | Medium | Use `.endswith(".us.archive.org")` pattern matching |
| IA items requiring authentication | Low | Low | Detect and show user-friendly error for lending library items |
| Description contains HTML | Medium | Low | Strip HTML tags before display, truncate long descriptions |
| Unicode/encoding issues in metadata | Low | Low | Use `.decode('utf-8', errors='replace')` |

## Future Considerations

- **Additional sources**: Vimeo search, Pexels, Pixabay (stock footage)
- **Collection browsing**: Let users browse IA collections/playlists
- **Favorites/history**: Remember recent searches per source
- **Source-specific filters**: IA has unique filters (year range, collection, creator)
- **Pagination**: Add "Load More" button using IA cursor-based pagination
- **Agent integration**: Add `search_internet_archive` agent tool or extend existing `search_youtube` to support source parameter

## References & Research

### Internal References
- Pattern reference: `ui/youtube_search_panel.py` (entire file)
- Downloader: `core/downloader.py:62-72` (domain whitelist)
- API client pattern: `core/youtube_api.py` (entire file)
- URL validation security: `docs/solutions/security-issues/url-scheme-validation-bypass.md`

### External References
- [Internet Archive Python Library](https://archive.org/developers/internetarchive/)
- [IA Scraping API](https://archive.org/services/search/v1/scrape)
- [IA Advanced Search Syntax](https://archive.org/advancedsearch.php#raw)
- [yt-dlp Supported Sites](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md) (includes archive.org)

## MVP

### Phase 1: core/internet_archive_api.py

```python
"""Internet Archive API client for video search."""

import logging
import urllib.request
import urllib.parse
import json
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class InternetArchiveVideo:
    """Data model for an Internet Archive video item."""

    identifier: str
    title: str
    description: str
    creator: Optional[str] = None
    date: Optional[str] = None
    duration_seconds: Optional[float] = None
    thumbnail_url: Optional[str] = None

    # Detailed metadata (populated via yt-dlp)
    width: Optional[int] = None
    height: Optional[int] = None
    aspect_ratio: Optional[float] = None
    filesize_approx: Optional[int] = None
    has_detailed_info: bool = False

    @property
    def id(self) -> str:
        return self.identifier

    @property
    def item_url(self) -> str:
        return f"https://archive.org/details/{self.identifier}"

    @property
    def download_url(self) -> str:
        return self.item_url  # yt-dlp handles extraction

    @property
    def duration_str(self) -> str:
        if self.duration_seconds is None:
            return "Unknown"
        total_seconds = int(self.duration_seconds)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"

    def matches_aspect_ratio(self, filter_value: str) -> bool:
        # Same logic as YouTubeVideo
        ...

    def matches_resolution(self, filter_value: str) -> bool:
        ...

    def matches_max_size(self, filter_value: str) -> bool:
        ...


class InternetArchiveClient:
    """Client for searching Internet Archive videos."""

    SCRAPE_API_URL = "https://archive.org/services/search/v1/scrape"
    VIDEO_MEDIATYPES = ["movies", "feature_films", "short_films", "animation"]

    def __init__(self):
        self._timeout = 30

    def search(
        self,
        query: str,
        max_results: int = 25,
    ) -> list[InternetArchiveVideo]:
        """Search for videos on Internet Archive."""
        # Build query with video mediatype filter
        mediatype_filter = " OR ".join(f"mediatype:{mt}" for mt in self.VIDEO_MEDIATYPES)
        full_query = f"({mediatype_filter}) AND ({query})"

        params = {
            "q": full_query,
            "fields": "identifier,title,description,creator,date,runtime",
            "count": str(max_results),
        }

        url = f"{self.SCRAPE_API_URL}?{urllib.parse.urlencode(params)}"

        # Create request with User-Agent header to avoid rate limiting
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Scene-Ripper/1.0 (video editing tool)"}
        )

        with urllib.request.urlopen(request, timeout=self._timeout) as response:
            data = json.loads(response.read().decode('utf-8', errors='replace'))

        results = []
        for item in data.get("items", []):
            # Handle description: could be string, list, or None
            raw_desc = item.get("description", "")
            if isinstance(raw_desc, list):
                raw_desc = " ".join(str(d) for d in raw_desc)
            description = self._truncate(self._strip_html(str(raw_desc)))

            video = InternetArchiveVideo(
                identifier=item["identifier"],
                title=item.get("title", item["identifier"]),
                description=description,
                creator=item.get("creator"),
                date=item.get("date"),
                duration_seconds=self._parse_runtime(item.get("runtime")),
                thumbnail_url=f"https://archive.org/services/img/{item['identifier']}",
            )
            results.append(video)

        return results

    def _parse_runtime(self, runtime: str | None) -> float | None:
        """Parse runtime string to seconds.

        Handles formats:
        - "HH:MM:SS" (e.g., "1:30:45")
        - "MM:SS" (e.g., "5:30")
        - Plain seconds as string (e.g., "90")
        - Decimal seconds (e.g., "123.45")
        """
        if not runtime:
            return None
        try:
            if ":" in runtime:
                parts = runtime.split(":")
                if len(parts) == 3:
                    h, m, s = map(float, parts)
                    return h * 3600 + m * 60 + s
                elif len(parts) == 2:
                    m, s = map(float, parts)
                    return m * 60 + s
            return float(runtime)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _strip_html(text: str) -> str:
        """Strip HTML tags from text."""
        import re
        clean = re.sub(r'<[^>]+>', '', text)
        # Also decode common HTML entities
        clean = clean.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
        clean = clean.replace('&quot;', '"').replace('&#39;', "'")
        return clean.strip()

    @staticmethod
    def _truncate(text: str, max_length: int = 200) -> str:
        """Truncate text to max length with ellipsis."""
        if len(text) <= max_length:
            return text
        return text[:max_length - 3].rsplit(' ', 1)[0] + '...'
```

### Phase 2: Domain whitelist update

```python
# core/downloader.py - update ALLOWED_DOMAINS

ALLOWED_DOMAINS = {
    # YouTube
    "youtube.com",
    "www.youtube.com",
    "youtu.be",
    # Vimeo
    "vimeo.com",
    "www.vimeo.com",
    "player.vimeo.com",
    # Internet Archive
    "archive.org",
    "www.archive.org",
}
```

**Note:** For CDN URLs (ia800.us.archive.org, etc.), update `is_valid_url()` to handle wildcard matching:

```python
def is_valid_url(self, url: str) -> bool:
    ...
    # Check for Internet Archive CDN pattern
    if hostname.endswith(".us.archive.org"):
        return True
    ...
```
