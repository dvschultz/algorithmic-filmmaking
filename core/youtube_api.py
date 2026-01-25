"""YouTube Data API v3 client for video search."""

import re
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

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
            "youtube",
            "v3",
            developerKey=api_key,
            cache_discovery=False,
        )

    def search(
        self,
        query: str,
        max_results: int = 25,
        page_token: Optional[str] = None,
        order: str = "relevance",
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
                "part": "snippet",
                "q": query,
                "type": "video",
                "maxResults": min(max_results, 50),
                "order": order,
            }
            if page_token:
                search_params["pageToken"] = page_token
            if video_duration:
                search_params["videoDuration"] = video_duration

            search_response = self._youtube.search().list(**search_params).execute()

            # Extract video IDs
            video_ids = [
                item["id"]["videoId"]
                for item in search_response.get("items", [])
                if item["id"].get("kind") == "youtube#video"
            ]

            if not video_ids:
                return YouTubeSearchResult(videos=[], total_results=0)

            # Step 2: Get video details (1 quota unit for up to 50 videos)
            details_response = (
                self._youtube.videos()
                .list(
                    part="snippet,contentDetails,statistics",
                    id=",".join(video_ids),
                )
                .execute()
            )

            # Build result objects
            details_map = {v["id"]: v for v in details_response.get("items", [])}
            videos = []

            for item in search_response["items"]:
                video_id = item["id"].get("videoId")
                if not video_id or video_id not in details_map:
                    continue

                detail = details_map[video_id]
                snippet = item["snippet"]

                videos.append(
                    YouTubeVideo(
                        video_id=video_id,
                        title=snippet["title"],
                        description=snippet.get("description", ""),
                        channel_title=snippet.get("channelTitle", ""),
                        thumbnail_url=self._get_thumbnail_url(snippet),
                        duration=self._parse_duration(
                            detail["contentDetails"]["duration"]
                        ),
                        view_count=int(detail["statistics"].get("viewCount", 0)),
                        definition=detail["contentDetails"].get("definition", "sd"),
                        published_at=snippet.get("publishedAt"),
                    )
                )

            return YouTubeSearchResult(
                videos=videos,
                next_page_token=search_response.get("nextPageToken"),
                total_results=search_response.get("pageInfo", {}).get(
                    "totalResults", 0
                ),
            )

        except HttpError as e:
            self._handle_http_error(e)
            raise  # Unreachable, but makes control flow explicit for type checkers

    def _get_thumbnail_url(self, snippet: dict) -> str:
        """Get best available thumbnail URL."""
        thumbnails = snippet.get("thumbnails", {})
        for size in ["high", "medium", "default"]:
            if size in thumbnails:
                return thumbnails[size]["url"]
        return ""

    def _parse_duration(self, iso_duration: str) -> Optional[timedelta]:
        """Parse ISO 8601 duration (PT1H2M3S) to timedelta."""
        pattern = r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?"
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
        reason = ""
        if error.error_details:
            reason = error.error_details[0].get("reason", "")

        if status == 403:
            if reason == "quotaExceeded":
                raise QuotaExceededError(
                    "YouTube API daily quota exceeded. Try again tomorrow."
                )
            elif reason in ("keyInvalid", "forbidden"):
                raise InvalidAPIKeyError(
                    "Invalid YouTube API key. Check your key in Settings."
                )
        elif status == 400:
            raise YouTubeAPIError(f"Invalid request: {error}")

        raise YouTubeAPIError(f"YouTube API error: {error}")
