"""Tests for the stem separation module."""

import hashlib
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.analysis.stem_separation import (
    STEM_NAMES,
    get_cached_stems,
    get_stem_cache_key,
)


class TestGetStemCacheKey:
    """Tests for file hashing used as cache key."""

    def test_returns_16_char_hex(self, tmp_path):
        f = tmp_path / "test.mp3"
        f.write_bytes(b"fake audio content " * 100)
        key = get_stem_cache_key(f)
        assert len(key) == 16
        assert all(c in "0123456789abcdef" for c in key)

    def test_same_content_same_key(self, tmp_path):
        f1 = tmp_path / "a.mp3"
        f2 = tmp_path / "b.mp3"
        content = b"identical content " * 100
        f1.write_bytes(content)
        f2.write_bytes(content)
        assert get_stem_cache_key(f1) == get_stem_cache_key(f2)

    def test_different_content_different_key(self, tmp_path):
        f1 = tmp_path / "a.mp3"
        f2 = tmp_path / "b.mp3"
        f1.write_bytes(b"content A " * 100)
        f2.write_bytes(b"content B " * 100)
        assert get_stem_cache_key(f1) != get_stem_cache_key(f2)


class TestGetCachedStems:
    """Tests for stem cache lookup."""

    def _create_cached_stems(self, cache_dir: Path, music_path: Path) -> Path:
        """Helper to create a complete set of cached stems."""
        key = get_stem_cache_key(music_path)
        stem_dir = cache_dir / key
        stem_dir.mkdir(parents=True)
        for name in STEM_NAMES:
            (stem_dir / f"{name}.wav").write_bytes(b"fake wav data")
        return stem_dir

    def test_cache_hit(self, tmp_path):
        music = tmp_path / "song.mp3"
        music.write_bytes(b"audio " * 100)
        cache_dir = tmp_path / "cache"
        self._create_cached_stems(cache_dir, music)

        result = get_cached_stems(music, cache_dir)
        assert result is not None
        assert set(result.keys()) == set(STEM_NAMES)
        for name, path in result.items():
            assert path.is_file()
            assert path.name == f"{name}.wav"

    def test_cache_miss_no_dir(self, tmp_path):
        music = tmp_path / "song.mp3"
        music.write_bytes(b"audio " * 100)
        cache_dir = tmp_path / "cache"

        result = get_cached_stems(music, cache_dir)
        assert result is None

    def test_cache_miss_partial_stems(self, tmp_path):
        """If any stem is missing, it's a cache miss."""
        music = tmp_path / "song.mp3"
        music.write_bytes(b"audio " * 100)
        cache_dir = tmp_path / "cache"

        key = get_stem_cache_key(music)
        stem_dir = cache_dir / key
        stem_dir.mkdir(parents=True)
        # Only write 3 of 4 stems
        for name in ("drums", "bass", "vocals"):
            (stem_dir / f"{name}.wav").write_bytes(b"fake wav data")

        result = get_cached_stems(music, cache_dir)
        assert result is None

    def test_different_files_different_cache(self, tmp_path):
        """Two different music files should use different cache dirs."""
        music_a = tmp_path / "a.mp3"
        music_b = tmp_path / "b.mp3"
        music_a.write_bytes(b"song A " * 100)
        music_b.write_bytes(b"song B " * 100)
        cache_dir = tmp_path / "cache"

        self._create_cached_stems(cache_dir, music_a)

        assert get_cached_stems(music_a, cache_dir) is not None
        assert get_cached_stems(music_b, cache_dir) is None


class TestStemConstants:
    """Tests for module constants."""

    def test_stem_names(self):
        assert STEM_NAMES == ("drums", "bass", "vocals", "other")

    def test_four_stems(self):
        assert len(STEM_NAMES) == 4
