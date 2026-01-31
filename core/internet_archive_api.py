"""Internet Archive API client for video search."""

import json
import logging
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Optional

from core.youtube_api import ASPECT_RATIO_RANGES, RESOLUTION_THRESHOLDS, SIZE_LIMITS

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
        """Unique identifier - matches video_id pattern from YouTubeVideo."""
        return self.identifier

    @property
    def video_id(self) -> str:
        """Alias for identifier to match YouTubeVideo interface."""
        return self.identifier

    @property
    def item_url(self) -> str:
        """URL to the Internet Archive item page."""
        return f"https://archive.org/details/{self.identifier}"

    @property
    def download_url(self) -> str:
        """URL for yt-dlp to download from."""
        return self.item_url  # yt-dlp handles extraction

    @property
    def duration_str(self) -> str:
        """Format duration as HH:MM:SS or MM:SS."""
        if self.duration_seconds is None:
            return ""
        total_seconds = int(self.duration_seconds)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"

    @property
    def channel_title(self) -> str:
        """Return creator for compatibility with YouTubeVideo interface."""
        return self.creator or ""

    @property
    def resolution_str(self) -> str:
        """Return resolution string (e.g., '1080p', '4K')."""
        if not self.height:
            return ""
        if self.height >= 2160:
            return "4K"
        elif self.height >= 1080:
            return "1080p"
        elif self.height >= 720:
            return "720p"
        elif self.height >= 480:
            return "480p"
        return f"{self.height}p"

    @property
    def aspect_ratio_str(self) -> str:
        """Return aspect ratio as a string (e.g., '16:9')."""
        if not self.aspect_ratio:
            return ""
        for name, (low, high) in ASPECT_RATIO_RANGES.items():
            if low <= self.aspect_ratio <= high:
                return name
        return f"{self.aspect_ratio:.2f}"

    def matches_aspect_ratio(self, filter_value: str) -> bool:
        """Check if video matches the given aspect ratio filter."""
        if filter_value == "any" or not filter_value:
            return True
        if not self.aspect_ratio:
            return False
        if filter_value not in ASPECT_RATIO_RANGES:
            return False
        low, high = ASPECT_RATIO_RANGES[filter_value]
        return low <= self.aspect_ratio <= high

    def matches_resolution(self, filter_value: str) -> bool:
        """Check if video meets minimum resolution requirement."""
        if filter_value == "any" or not filter_value:
            return True
        if not self.height:
            return False
        if filter_value not in RESOLUTION_THRESHOLDS:
            return False
        return self.height >= RESOLUTION_THRESHOLDS[filter_value]

    def matches_max_size(self, filter_value: str) -> bool:
        """Check if video is under the max file size."""
        if filter_value == "any" or not filter_value:
            return True
        if not self.filesize_approx:
            return False
        if filter_value not in SIZE_LIMITS:
            return False
        return self.filesize_approx <= SIZE_LIMITS[filter_value]


class InternetArchiveError(Exception):
    """Base exception for Internet Archive API errors."""

    pass


class InternetArchiveClient:
    """Client for searching Internet Archive videos."""

    # Use Advanced Search API - more reliable than scrape API
    SEARCH_API_URL = "https://archive.org/advancedsearch.php"
    VIDEO_MEDIATYPES = ["movies", "feature_films", "short_films", "animation"]
    USER_AGENT = "Scene-Ripper/1.0 (video editing tool; +https://github.com/)"

    def __init__(self, timeout: int = 30):
        self._timeout = timeout

    @staticmethod
    def _escape_query(text: str) -> str:
        """Escape Lucene special characters in user query."""
        special_chars = r'+-&|!(){}[]^"~*?:\/'
        escaped = []
        for char in text:
            if char in special_chars:
                escaped.append(f"\\{char}")
            else:
                escaped.append(char)
        return "".join(escaped)

    def search(
        self,
        query: str,
        max_results: int = 25,
    ) -> list[InternetArchiveVideo]:
        """
        Search for videos on Internet Archive.

        Args:
            query: Search term
            max_results: Maximum number of results to return

        Returns:
            List of InternetArchiveVideo objects

        Raises:
            InternetArchiveError: On API errors
        """
        if not query.strip():
            return []

        # Escape special characters in user query
        escaped_query = self._escape_query(query)

        # Build query with video mediatype filter
        mediatype_filter = " OR ".join(
            f"mediatype:{mt}" for mt in self.VIDEO_MEDIATYPES
        )
        full_query = f"({mediatype_filter}) AND ({escaped_query})"

        params = {
            "q": full_query,
            "fl[]": "identifier,title,description,creator,date,runtime",
            "rows": max_results,
            "output": "json",
        }

        url = f"{self.SEARCH_API_URL}?{urllib.parse.urlencode(params, doseq=True)}"

        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": self.USER_AGENT},
            )

            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                data = json.loads(response.read().decode("utf-8", errors="replace"))
                # Advanced Search API returns results under 'response.docs'
                data = {"items": data.get("response", {}).get("docs", [])}

        except urllib.error.URLError as e:
            logger.error(f"Internet Archive API error: {e}")
            raise InternetArchiveError(f"Failed to search Internet Archive: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response from Internet Archive: {e}")
            raise InternetArchiveError("Invalid response from Internet Archive")

        results = []
        for item in data.get("items", []):
            # Handle description: could be string, list, or None
            raw_desc = item.get("description", "")
            if isinstance(raw_desc, list):
                raw_desc = " ".join(str(d) for d in raw_desc)
            description = self._truncate(self._strip_html(str(raw_desc)))

            # Handle title: could be string or list
            raw_title = item.get("title", item["identifier"])
            if isinstance(raw_title, list):
                raw_title = raw_title[0] if raw_title else item["identifier"]

            # Handle creator: could be string or list
            raw_creator = item.get("creator")
            if isinstance(raw_creator, list):
                raw_creator = ", ".join(str(c) for c in raw_creator[:3])
                if len(item.get("creator", [])) > 3:
                    raw_creator += "..."

            video = InternetArchiveVideo(
                identifier=item["identifier"],
                title=str(raw_title),
                description=description,
                creator=str(raw_creator) if raw_creator else None,
                date=item.get("date"),
                duration_seconds=self._parse_runtime(item.get("runtime")),
                thumbnail_url=f"https://archive.org/services/img/{item['identifier']}",
            )
            results.append(video)

        return results

    def _parse_runtime(self, runtime) -> Optional[float]:
        """
        Parse runtime to seconds.

        Handles formats:
        - "HH:MM:SS" (e.g., "1:30:45")
        - "MM:SS" (e.g., "5:30")
        - Plain seconds as string (e.g., "90")
        - Decimal seconds (e.g., "123.45")
        - List of runtimes (takes first)
        """
        if runtime is None:
            return None

        # Handle list (take first value)
        if isinstance(runtime, list):
            if not runtime:
                return None
            runtime = runtime[0]

        # Convert to string
        runtime = str(runtime).strip()
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
        clean = re.sub(r"<[^>]+>", "", text)
        # Decode common HTML entities
        clean = clean.replace("&amp;", "&")
        clean = clean.replace("&lt;", "<")
        clean = clean.replace("&gt;", ">")
        clean = clean.replace("&quot;", '"')
        clean = clean.replace("&#39;", "'")
        clean = clean.replace("&nbsp;", " ")
        return clean.strip()

    @staticmethod
    def _truncate(text: str, max_length: int = 200) -> str:
        """Truncate text to max length with ellipsis."""
        if len(text) <= max_length:
            return text
        # Try to break at word boundary
        truncated = text[: max_length - 3]
        last_space = truncated.rfind(" ")
        if last_space > max_length // 2:
            truncated = truncated[:last_space]
        return truncated + "..."
