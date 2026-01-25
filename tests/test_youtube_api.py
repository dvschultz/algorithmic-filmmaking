"""Unit tests for YouTube API client."""

import pytest
from datetime import timedelta
from unittest.mock import Mock, patch, MagicMock

from core.youtube_api import (
    YouTubeSearchClient,
    YouTubeVideo,
    YouTubeSearchResult,
    YouTubeAPIError,
    QuotaExceededError,
    InvalidAPIKeyError,
)


class TestYouTubeVideo:
    """Tests for YouTubeVideo dataclass."""

    def test_youtube_url(self):
        """Test youtube_url property."""
        video = YouTubeVideo(
            video_id="dQw4w9WgXcQ",
            title="Test Video",
            description="A test video",
            channel_title="Test Channel",
            thumbnail_url="https://example.com/thumb.jpg",
        )
        assert video.youtube_url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_duration_str_with_hours(self):
        """Test duration_str with hours, minutes, seconds."""
        video = YouTubeVideo(
            video_id="test",
            title="Test",
            description="",
            channel_title="",
            thumbnail_url="",
            duration=timedelta(hours=1, minutes=23, seconds=45),
        )
        assert video.duration_str == "1:23:45"

    def test_duration_str_without_hours(self):
        """Test duration_str with only minutes and seconds."""
        video = YouTubeVideo(
            video_id="test",
            title="Test",
            description="",
            channel_title="",
            thumbnail_url="",
            duration=timedelta(minutes=5, seconds=32),
        )
        assert video.duration_str == "5:32"

    def test_duration_str_empty(self):
        """Test duration_str when duration is None."""
        video = YouTubeVideo(
            video_id="test",
            title="Test",
            description="",
            channel_title="",
            thumbnail_url="",
            duration=None,
        )
        assert video.duration_str == ""

    def test_duration_str_seconds_only(self):
        """Test duration_str with only seconds."""
        video = YouTubeVideo(
            video_id="test",
            title="Test",
            description="",
            channel_title="",
            thumbnail_url="",
            duration=timedelta(seconds=45),
        )
        assert video.duration_str == "0:45"


class TestYouTubeSearchClient:
    """Tests for YouTubeSearchClient."""

    def test_init_requires_api_key(self):
        """Test that empty API key raises InvalidAPIKeyError."""
        with pytest.raises(InvalidAPIKeyError):
            YouTubeSearchClient("")

        with pytest.raises(InvalidAPIKeyError):
            YouTubeSearchClient("   ")

    @patch("core.youtube_api.build")
    def test_init_creates_client(self, mock_build):
        """Test that valid API key creates client."""
        client = YouTubeSearchClient("valid_api_key")
        mock_build.assert_called_once_with(
            "youtube",
            "v3",
            developerKey="valid_api_key",
            cache_discovery=False,
        )

    @patch("core.youtube_api.build")
    def test_parse_duration_full(self, mock_build):
        """Test ISO 8601 duration parsing with all components."""
        client = YouTubeSearchClient("test_key")
        duration = client._parse_duration("PT1H23M45S")
        assert duration == timedelta(hours=1, minutes=23, seconds=45)

    @patch("core.youtube_api.build")
    def test_parse_duration_minutes_seconds(self, mock_build):
        """Test duration parsing with minutes and seconds only."""
        client = YouTubeSearchClient("test_key")
        duration = client._parse_duration("PT5M32S")
        assert duration == timedelta(minutes=5, seconds=32)

    @patch("core.youtube_api.build")
    def test_parse_duration_hours_only(self, mock_build):
        """Test duration parsing with hours only."""
        client = YouTubeSearchClient("test_key")
        duration = client._parse_duration("PT2H")
        assert duration == timedelta(hours=2)

    @patch("core.youtube_api.build")
    def test_parse_duration_invalid(self, mock_build):
        """Test duration parsing with invalid format."""
        client = YouTubeSearchClient("test_key")
        duration = client._parse_duration("invalid")
        assert duration is None

    @patch("core.youtube_api.build")
    def test_get_thumbnail_url_high(self, mock_build):
        """Test thumbnail URL selection prefers high quality."""
        client = YouTubeSearchClient("test_key")
        snippet = {
            "thumbnails": {
                "default": {"url": "default.jpg"},
                "medium": {"url": "medium.jpg"},
                "high": {"url": "high.jpg"},
            }
        }
        assert client._get_thumbnail_url(snippet) == "high.jpg"

    @patch("core.youtube_api.build")
    def test_get_thumbnail_url_medium_fallback(self, mock_build):
        """Test thumbnail URL falls back to medium."""
        client = YouTubeSearchClient("test_key")
        snippet = {
            "thumbnails": {
                "default": {"url": "default.jpg"},
                "medium": {"url": "medium.jpg"},
            }
        }
        assert client._get_thumbnail_url(snippet) == "medium.jpg"

    @patch("core.youtube_api.build")
    def test_get_thumbnail_url_default_fallback(self, mock_build):
        """Test thumbnail URL falls back to default."""
        client = YouTubeSearchClient("test_key")
        snippet = {
            "thumbnails": {
                "default": {"url": "default.jpg"},
            }
        }
        assert client._get_thumbnail_url(snippet) == "default.jpg"

    @patch("core.youtube_api.build")
    def test_get_thumbnail_url_empty(self, mock_build):
        """Test thumbnail URL returns empty string when none available."""
        client = YouTubeSearchClient("test_key")
        snippet = {"thumbnails": {}}
        assert client._get_thumbnail_url(snippet) == ""

    @patch("core.youtube_api.build")
    def test_search_success(self, mock_build):
        """Test successful search returns results."""
        # Set up mock responses
        mock_youtube = MagicMock()
        mock_build.return_value = mock_youtube

        # Mock search response
        mock_search_response = {
            "items": [
                {
                    "id": {"kind": "youtube#video", "videoId": "video123"},
                    "snippet": {
                        "title": "Test Video",
                        "description": "A test video",
                        "channelTitle": "Test Channel",
                        "thumbnails": {"high": {"url": "https://example.com/thumb.jpg"}},
                        "publishedAt": "2024-01-01T00:00:00Z",
                    },
                }
            ],
            "nextPageToken": "next_token",
            "pageInfo": {"totalResults": 100},
        }
        mock_youtube.search().list().execute.return_value = mock_search_response

        # Mock videos response
        mock_videos_response = {
            "items": [
                {
                    "id": "video123",
                    "contentDetails": {"duration": "PT5M32S", "definition": "hd"},
                    "statistics": {"viewCount": "1000000"},
                }
            ]
        }
        mock_youtube.videos().list().execute.return_value = mock_videos_response

        # Execute search
        client = YouTubeSearchClient("test_key")
        result = client.search("test query")

        # Verify result
        assert isinstance(result, YouTubeSearchResult)
        assert len(result.videos) == 1
        assert result.next_page_token == "next_token"
        assert result.total_results == 100

        video = result.videos[0]
        assert video.video_id == "video123"
        assert video.title == "Test Video"
        assert video.channel_title == "Test Channel"
        assert video.duration == timedelta(minutes=5, seconds=32)
        assert video.view_count == 1000000
        assert video.definition == "hd"

    @patch("core.youtube_api.build")
    def test_search_empty_results(self, mock_build):
        """Test search with no results."""
        mock_youtube = MagicMock()
        mock_build.return_value = mock_youtube

        mock_search_response = {"items": [], "pageInfo": {"totalResults": 0}}
        mock_youtube.search().list().execute.return_value = mock_search_response

        client = YouTubeSearchClient("test_key")
        result = client.search("nonexistent query")

        assert isinstance(result, YouTubeSearchResult)
        assert len(result.videos) == 0
        assert result.total_results == 0


class TestQuotaExceededError:
    """Tests for QuotaExceededError."""

    def test_is_youtube_api_error(self):
        """Test that QuotaExceededError inherits from YouTubeAPIError."""
        error = QuotaExceededError("Quota exceeded")
        assert isinstance(error, YouTubeAPIError)


class TestInvalidAPIKeyError:
    """Tests for InvalidAPIKeyError."""

    def test_is_youtube_api_error(self):
        """Test that InvalidAPIKeyError inherits from YouTubeAPIError."""
        error = InvalidAPIKeyError("Invalid key")
        assert isinstance(error, YouTubeAPIError)
