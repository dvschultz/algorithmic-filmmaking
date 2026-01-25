"""Unit tests for core settings module."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from core.settings import (
    Settings,
    load_settings,
    save_settings,
    is_from_environment,
    get_env_overridden_settings,
    get_default_settings,
    _get_config_dir,
    _get_config_path,
    _get_cache_dir,
    _apply_env_overrides,
    ENV_YOUTUBE_API_KEY,
    ENV_CACHE_DIR,
    ENV_DOWNLOAD_DIR,
    ENV_EXPORT_DIR,
    ENV_CONFIG_PATH,
    ENV_SENSITIVITY,
    ENV_WHISPER_MODEL,
)


class TestSettingsDefaults:
    """Tests for default settings values."""

    def test_default_sensitivity(self):
        """Test default sensitivity value."""
        settings = Settings()
        assert settings.default_sensitivity == 3.0

    def test_default_transcription_model(self):
        """Test default transcription model."""
        settings = Settings()
        assert settings.transcription_model == "small.en"

    def test_default_export_quality(self):
        """Test default export quality."""
        settings = Settings()
        assert settings.export_quality == "medium"

    def test_default_theme_preference(self):
        """Test default theme preference."""
        settings = Settings()
        assert settings.theme_preference == "system"

    def test_get_default_settings(self):
        """Test get_default_settings factory function."""
        settings = get_default_settings()
        assert isinstance(settings, Settings)
        assert settings.default_sensitivity == 3.0


class TestSettingsHelpers:
    """Tests for settings helper methods."""

    def test_get_quality_preset_high(self):
        """Test quality preset lookup for high."""
        settings = Settings(export_quality="high")
        preset = settings.get_quality_preset()
        assert preset["crf"] == 18
        assert preset["preset"] == "slow"

    def test_get_quality_preset_medium(self):
        """Test quality preset lookup for medium."""
        settings = Settings(export_quality="medium")
        preset = settings.get_quality_preset()
        assert preset["crf"] == 23
        assert preset["preset"] == "medium"

    def test_get_quality_preset_low(self):
        """Test quality preset lookup for low."""
        settings = Settings(export_quality="low")
        preset = settings.get_quality_preset()
        assert preset["crf"] == 28
        assert preset["preset"] == "fast"

    def test_get_resolution_1080p(self):
        """Test resolution preset lookup."""
        settings = Settings(export_resolution="1080p")
        width, height = settings.get_resolution()
        assert width == 1920
        assert height == 1080

    def test_get_resolution_original(self):
        """Test resolution preset for original."""
        settings = Settings(export_resolution="original")
        width, height = settings.get_resolution()
        assert width is None
        assert height is None

    def test_get_fps_30(self):
        """Test FPS preset lookup."""
        settings = Settings(export_fps="30")
        fps = settings.get_fps()
        assert fps == 30.0

    def test_get_fps_original(self):
        """Test FPS preset for original."""
        settings = Settings(export_fps="original")
        fps = settings.get_fps()
        assert fps is None

    def test_min_scene_length_frames(self):
        """Test frame count calculation from seconds."""
        settings = Settings(min_scene_length_seconds=1.0)
        frames = settings.min_scene_length_frames(fps=30.0)
        assert frames == 30


class TestConfigPaths:
    """Tests for config path functions."""

    def test_get_config_dir_xdg(self):
        """Test XDG config directory on Unix."""
        with patch.dict(os.environ, {"XDG_CONFIG_HOME": "/custom/config"}, clear=False):
            with patch("os.name", "posix"):
                config_dir = _get_config_dir()
                assert config_dir == Path("/custom/config/scene-ripper")

    def test_get_config_dir_default_unix(self):
        """Test default config directory on Unix."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("os.name", "posix"):
                config_dir = _get_config_dir()
                assert "scene-ripper" in str(config_dir)
                assert ".config" in str(config_dir)

    def test_get_config_path_default(self):
        """Test default config file path."""
        with patch.dict(os.environ, {}, clear=True):
            config_path = _get_config_path()
            assert config_path.name == "config.json"
            assert "scene-ripper" in str(config_path)

    def test_get_config_path_custom(self):
        """Test custom config file path via env var."""
        with patch.dict(os.environ, {ENV_CONFIG_PATH: "/custom/path/settings.json"}):
            config_path = _get_config_path()
            assert config_path == Path("/custom/path/settings.json")

    def test_get_cache_dir_xdg(self):
        """Test XDG cache directory."""
        with patch.dict(os.environ, {"XDG_CACHE_HOME": "/custom/cache"}, clear=False):
            with patch("os.name", "posix"):
                cache_dir = _get_cache_dir()
                assert cache_dir == Path("/custom/cache/scene-ripper")


class TestEnvironmentVariables:
    """Tests for environment variable support."""

    def test_env_youtube_api_key(self):
        """Test YOUTUBE_API_KEY env var."""
        with patch.dict(os.environ, {ENV_YOUTUBE_API_KEY: "test_key_123"}):
            settings = Settings()
            _apply_env_overrides(settings)
            assert settings.youtube_api_key == "test_key_123"
            assert is_from_environment("youtube_api_key")

    def test_env_cache_dir(self):
        """Test SCENE_RIPPER_CACHE_DIR env var."""
        with patch.dict(os.environ, {ENV_CACHE_DIR: "/env/cache/dir"}):
            settings = Settings()
            _apply_env_overrides(settings)
            assert settings.thumbnail_cache_dir == Path("/env/cache/dir")
            assert is_from_environment("thumbnail_cache_dir")

    def test_env_download_dir(self):
        """Test SCENE_RIPPER_DOWNLOAD_DIR env var."""
        with patch.dict(os.environ, {ENV_DOWNLOAD_DIR: "/env/download"}):
            settings = Settings()
            _apply_env_overrides(settings)
            assert settings.download_dir == Path("/env/download")
            assert is_from_environment("download_dir")

    def test_env_export_dir(self):
        """Test SCENE_RIPPER_EXPORT_DIR env var."""
        with patch.dict(os.environ, {ENV_EXPORT_DIR: "/env/export"}):
            settings = Settings()
            _apply_env_overrides(settings)
            assert settings.export_dir == Path("/env/export")
            assert is_from_environment("export_dir")

    def test_env_sensitivity(self):
        """Test SCENE_RIPPER_SENSITIVITY env var."""
        with patch.dict(os.environ, {ENV_SENSITIVITY: "7.5"}):
            settings = Settings()
            _apply_env_overrides(settings)
            assert settings.default_sensitivity == 7.5
            assert is_from_environment("default_sensitivity")

    def test_env_sensitivity_invalid(self):
        """Test invalid SCENE_RIPPER_SENSITIVITY is ignored."""
        with patch.dict(os.environ, {ENV_SENSITIVITY: "not_a_number"}):
            settings = Settings()
            original_sensitivity = settings.default_sensitivity
            _apply_env_overrides(settings)
            # Should keep default value
            assert settings.default_sensitivity == original_sensitivity
            assert not is_from_environment("default_sensitivity")

    def test_env_whisper_model(self):
        """Test SCENE_RIPPER_WHISPER_MODEL env var."""
        with patch.dict(os.environ, {ENV_WHISPER_MODEL: "large-v3"}):
            settings = Settings()
            _apply_env_overrides(settings)
            assert settings.transcription_model == "large-v3"
            assert is_from_environment("transcription_model")

    def test_get_env_overridden_settings(self):
        """Test getting list of overridden settings."""
        with patch.dict(os.environ, {
            ENV_YOUTUBE_API_KEY: "key",
            ENV_SENSITIVITY: "5.0",
        }):
            settings = Settings()
            _apply_env_overrides(settings)
            overridden = get_env_overridden_settings()
            assert "youtube_api_key" in overridden
            assert "default_sensitivity" in overridden
            assert "download_dir" not in overridden

    def test_is_from_environment_false(self):
        """Test is_from_environment returns False for non-overridden settings."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            _apply_env_overrides(settings)
            assert not is_from_environment("youtube_api_key")
            assert not is_from_environment("default_sensitivity")


class TestJSONSettings:
    """Tests for JSON settings persistence."""

    def test_save_and_load_settings(self):
        """Test saving and loading settings to/from JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"

            with patch("core.settings._get_config_path", return_value=config_path):
                with patch("core.settings._get_api_key_from_keyring", return_value=""):
                    with patch("core.settings._set_api_key_in_keyring", return_value=True):
                        # Save settings
                        settings = Settings()
                        settings.default_sensitivity = 8.5
                        settings.export_quality = "high"
                        settings.transcription_model = "medium.en"

                        assert save_settings(settings) is True
                        assert config_path.exists()

                        # Load settings back
                        loaded = load_settings()
                        assert loaded.default_sensitivity == 8.5
                        assert loaded.export_quality == "high"
                        assert loaded.transcription_model == "medium.en"

    def test_json_schema_structure(self):
        """Test saved JSON has expected structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"

            with patch("core.settings._get_config_path", return_value=config_path):
                with patch("core.settings._set_api_key_in_keyring", return_value=True):
                    settings = Settings()
                    save_settings(settings)

                    data = json.loads(config_path.read_text())

                    # Check schema version
                    assert "version" in data
                    assert data["version"] == "1.0"

                    # Check sections exist
                    assert "paths" in data
                    assert "detection" in data
                    assert "transcription" in data
                    assert "export" in data
                    assert "appearance" in data
                    assert "youtube" in data

                    # API key should NOT be in JSON
                    assert "api_key" not in json.dumps(data)

    def test_load_settings_nonexistent_file(self):
        """Test loading settings when config file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "nonexistent.json"

            with patch("core.settings._get_config_path", return_value=config_path):
                with patch("core.settings._get_api_key_from_keyring", return_value=""):
                    settings = load_settings()
                    # Should return defaults
                    assert settings.default_sensitivity == 3.0
                    assert settings.transcription_model == "small.en"

    def test_load_settings_invalid_json(self):
        """Test loading settings with invalid JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text("not valid json {{{")

            with patch("core.settings._get_config_path", return_value=config_path):
                with patch("core.settings._get_api_key_from_keyring", return_value=""):
                    settings = load_settings()
                    # Should return defaults on error
                    assert settings.default_sensitivity == 3.0

    def test_env_vars_override_json(self):
        """Test that environment variables override JSON config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"

            # Save settings to JSON
            with patch("core.settings._get_config_path", return_value=config_path):
                with patch("core.settings._set_api_key_in_keyring", return_value=True):
                    settings = Settings()
                    settings.default_sensitivity = 5.0
                    save_settings(settings)

            # Load with env var override
            with patch("core.settings._get_config_path", return_value=config_path):
                with patch("core.settings._get_api_key_from_keyring", return_value=""):
                    with patch.dict(os.environ, {ENV_SENSITIVITY: "9.0"}):
                        loaded = load_settings()
                        # Env var should win
                        assert loaded.default_sensitivity == 9.0


class TestFilePermissions:
    """Tests for file permission handling."""

    @pytest.mark.skipif(os.name == "nt", reason="Unix permissions only")
    def test_config_file_permissions(self):
        """Test that config file has restrictive permissions on Unix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"

            with patch("core.settings._get_config_path", return_value=config_path):
                with patch("core.settings._set_api_key_in_keyring", return_value=True):
                    settings = Settings()
                    save_settings(settings)

                    # Check file permissions (0600 = owner read/write only)
                    stat_result = config_path.stat()
                    permissions = stat_result.st_mode & 0o777
                    assert permissions == 0o600

    @pytest.mark.skipif(os.name == "nt", reason="Unix permissions only")
    def test_config_dir_permissions(self):
        """Test that config directory has restrictive permissions on Unix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / "scene-ripper"
            config_path = config_dir / "config.json"

            with patch("core.settings._get_config_path", return_value=config_path):
                with patch("core.settings._set_api_key_in_keyring", return_value=True):
                    settings = Settings()
                    save_settings(settings)

                    # Check directory permissions (0700 = owner rwx only)
                    stat_result = config_dir.stat()
                    permissions = stat_result.st_mode & 0o777
                    assert permissions == 0o700
