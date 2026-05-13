"""Unit tests for core settings module."""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from core.settings import (
    Settings,
    KEYRING_SERVICE,
    LEGACY_KEYRING_SERVICES,
    load_settings,
    save_settings,
    is_from_environment,
    get_env_overridden_settings,
    get_default_settings,
    validate_download_dir,
    get_default_download_dir,
    is_download_dir_from_env,
    _get_config_dir,
    _get_config_path,
    _get_cache_dir,
    _sync_model_cache_env,
    _get_api_key_from_keyring,
    _apply_env_overrides,
    ENV_YOUTUBE_API_KEY,
    ENV_CACHE_DIR,
    ENV_DOWNLOAD_DIR,
    ENV_EXPORT_DIR,
    ENV_CONFIG_PATH,
    ENV_SENSITIVITY,
    ENV_WHISPER_MODEL,
    ENV_GROQ_API_KEY,
    get_groq_api_key,
    set_groq_api_key,
    is_api_key_from_env,
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
        assert settings.transcription_model == "medium.en"

    def test_default_export_quality(self):
        """Test default export quality."""
        settings = Settings()
        assert settings.export_quality == "medium"

    def test_default_theme_preference(self):
        """Test default theme preference."""
        settings = Settings()
        assert settings.theme_preference == "system"

    def test_default_update_settings(self):
        """Update settings should have stable defaults for future native updater support."""
        settings = Settings()
        assert settings.check_for_updates is True
        assert settings.automatically_download_updates is False
        assert settings.skipped_update_version == ""
        assert settings.last_prompted_update_version == ""
        assert settings.update_channel == "stable"
        assert settings.last_update_status == "never_checked"
        assert settings.last_update_version == ""
        assert settings.last_update_error == ""

    def test_get_default_settings(self):
        """Test get_default_settings factory function."""
        settings = get_default_settings()
        assert isinstance(settings, Settings)
        assert settings.default_sensitivity == 3.0

    def test_default_text_detection_enabled(self):
        """Test default text detection enabled value."""
        settings = Settings()
        assert settings.text_detection_enabled is True

    def test_default_text_detection_confidence(self):
        """Test default text detection confidence threshold."""
        settings = Settings()
        assert settings.text_detection_confidence == 0.5


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

    @pytest.mark.skipif(sys.platform == "win32", reason="PosixPath not available on Windows")
    def test_get_config_dir_xdg(self):
        """Test XDG config directory on Unix."""
        with patch.dict(os.environ, {"XDG_CONFIG_HOME": "/custom/config"}, clear=False):
            with patch("os.name", "posix"):
                config_dir = _get_config_dir()
                assert config_dir == Path("/custom/config/scene-ripper")

    @pytest.mark.skipif(sys.platform == "win32", reason="PosixPath not available on Windows")
    def test_get_config_dir_default_unix(self):
        """Test default config directory on Unix."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("os.name", "posix"):
                config_dir = _get_config_dir()
                assert "scene-ripper" in str(config_dir)
                assert ".config" in str(config_dir)

    def test_get_config_path_default(self):
        """Test default config file path."""
        # Preserve HOME-related vars so Path.home() works on all platforms
        home_vars = {k: v for k, v in os.environ.items()
                     if k in ("HOME", "USERPROFILE", "HOMEDRIVE", "HOMEPATH",
                              "APPDATA", "LOCALAPPDATA", "SYSTEMROOT", "WINDIR")}
        with patch.dict(os.environ, home_vars, clear=True):
            config_path = _get_config_path()
            assert config_path.name == "config.json"
            assert "scene-ripper" in str(config_path)

    def test_get_config_path_custom(self):
        """Test custom config file path via env var."""
        with patch.dict(os.environ, {ENV_CONFIG_PATH: "/custom/path/settings.json"}):
            config_path = _get_config_path()
            assert config_path == Path("/custom/path/settings.json")

    @pytest.mark.skipif(sys.platform == "win32", reason="PosixPath not available on Windows")
    def test_get_cache_dir_xdg(self):
        """Test XDG cache directory."""
        with patch.dict(os.environ, {"XDG_CACHE_HOME": "/custom/cache"}, clear=False):
            with patch("os.name", "posix"):
                cache_dir = _get_cache_dir()
                assert cache_dir == Path("/custom/cache/scene-ripper")

    def test_get_cache_dir_frozen_macos_uses_bundle_identifier(self):
        """Frozen macOS builds should use the release bundle identifier cache path."""
        with patch("core.settings.is_frozen", return_value=True), \
             patch("core.settings.sys.platform", "darwin"):
            cache_dir = _get_cache_dir()
            assert cache_dir == Path.home() / "Library" / "Caches" / KEYRING_SERVICE


class TestKeyringCompatibility:
    """Tests for keyring service migration compatibility."""

    def test_reads_current_keyring_service_first(self):
        """The new bundle identifier service should be the primary lookup target."""
        keyring = MagicMock()
        keyring.get_password.side_effect = ["new-value"]

        with patch.dict(sys.modules, {"keyring": keyring}):
            assert _get_api_key_from_keyring() == "new-value"

        keyring.get_password.assert_called_once_with(KEYRING_SERVICE, "youtube_api_key")

    def test_falls_back_to_legacy_keyring_service(self):
        """Legacy keyring entries should still be readable after the service rename."""
        keyring = MagicMock()
        keyring.get_password.side_effect = ["", "legacy-value"]

        with patch.dict(sys.modules, {"keyring": keyring}):
            assert _get_api_key_from_keyring() == "legacy-value"

        assert keyring.get_password.call_args_list[0].args == (KEYRING_SERVICE, "youtube_api_key")
        assert keyring.get_password.call_args_list[1].args == (
            LEGACY_KEYRING_SERVICES[0],
            "youtube_api_key",
        )


class TestGroqAPIKey:
    """Tests for Groq API key storage and environment override behavior."""

    def test_get_groq_api_key_prefers_environment(self):
        with patch.dict(os.environ, {ENV_GROQ_API_KEY: "env-groq-key"}):
            assert get_groq_api_key() == "env-groq-key"
            assert is_api_key_from_env("groq")

    def test_get_groq_api_key_falls_back_to_keyring(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch(
                "core.settings._get_provider_api_key_from_keyring",
                return_value="keyring-groq-key",
            ) as get_key:
                assert get_groq_api_key() == "keyring-groq-key"
                get_key.assert_called_once_with("groq_api_key")

    def test_set_groq_api_key_uses_keyring(self):
        with patch(
            "core.settings._set_provider_api_key_in_keyring",
            return_value=True,
        ) as set_key:
            assert set_groq_api_key("gsk-test") is True
            set_key.assert_called_once_with("groq_api_key", "gsk-test")


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


class TestModelCacheEnvSync:
    """Tests for Hugging Face / Torch cache environment synchronization."""

    def test_sync_model_cache_env_sets_huggingface_and_torch_paths(self):
        cache_root = Path("/tmp/scene-ripper-model-cache")

        with patch.dict(os.environ, {}, clear=False):
            _sync_model_cache_env(cache_root)

            assert os.environ["HF_HOME"] == str(cache_root / "huggingface")
            assert os.environ["HF_HUB_CACHE"] == str(cache_root / "huggingface")
            assert os.environ["HF_MODULES_CACHE"] == str(cache_root / "huggingface" / "modules")
            assert os.environ["TORCH_HOME"] == str(cache_root)

    def test_sync_model_cache_env_updates_imported_transformers_modules(self):
        cache_root = Path("/tmp/scene-ripper-model-cache")
        fake_dynamic = MagicMock(HF_MODULES_CACHE="/old/modules")
        fake_hub = MagicMock(HF_MODULES_CACHE="/old/modules")

        with patch.dict(
            sys.modules,
            {
                "transformers.dynamic_module_utils": fake_dynamic,
                "transformers.utils.hub": fake_hub,
            },
            clear=False,
        ):
            _sync_model_cache_env(cache_root)

        expected = str(cache_root / "huggingface" / "modules")
        assert fake_dynamic.HF_MODULES_CACHE == expected
        assert fake_hub.HF_MODULES_CACHE == expected

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
        # Preserve HOME-related vars so Path.home() works on all platforms
        home_vars = {k: v for k, v in os.environ.items()
                     if k in ("HOME", "USERPROFILE", "HOMEDRIVE", "HOMEPATH",
                              "APPDATA", "LOCALAPPDATA", "SYSTEMROOT", "WINDIR")}
        with patch.dict(os.environ, home_vars, clear=True):
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
                        settings.last_update_status = "update_available"
                        settings.last_update_version = "0.1.1"

                        assert save_settings(settings) is True
                        assert config_path.exists()

                        # Load settings back
                        loaded = load_settings()
                        assert loaded.default_sensitivity == 8.5
                        assert loaded.export_quality == "high"
                        assert loaded.transcription_model == "medium.en"
                        assert loaded.last_update_status == "update_available"
                        assert loaded.last_update_version == "0.1.1"

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

                    # Credential field NAMES should NOT be in JSON — keys live in keyring.
                    # The literal string "api_key" can appear as a value (auth.mode = "api_key"
                    # is a legitimate mode name distinct from any credential).
                    json_str = json.dumps(data)
                    for credential_field in (
                        "youtube_api_key",
                        "openai_api_key",
                        "anthropic_api_key",
                        "gemini_api_key",
                        "groq_api_key",
                        "openrouter_api_key",
                        "replicate_api_key",
                        "chatgpt_oauth_token",
                    ):
                        assert credential_field not in json_str, (
                            f"Sensitive field {credential_field!r} leaked into JSON"
                        )

    def test_load_settings_nonexistent_file(self):
        """Test loading settings when config file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "nonexistent.json"

            with patch("core.settings._get_config_path", return_value=config_path):
                with patch("core.settings._get_api_key_from_keyring", return_value=""):
                    settings = load_settings()
                    # Should return defaults
                    assert settings.default_sensitivity == 3.0
                    assert settings.transcription_model == "medium.en"

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


class TestAnalysisSelectedOperations:
    """Tests for analysis_selected_operations persistence."""

    def test_default_analysis_selected_operations(self):
        """Test default selected operations value."""
        settings = Settings()
        assert settings.analysis_selected_operations == ["colors", "shots", "transcribe"]

    def test_save_and_load_analysis_selected_operations(self):
        """Test round-trip save/load of analysis_selected_operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"

            with patch("core.settings._get_config_path", return_value=config_path):
                with patch("core.settings._get_api_key_from_keyring", return_value=""):
                    with patch("core.settings._set_api_key_in_keyring", return_value=True):
                        # Save with custom selection
                        settings = Settings()
                        settings.analysis_selected_operations = [
                            "colors", "shots", "describe", "cinematography"
                        ]
                        assert save_settings(settings) is True

                        # Load and verify
                        loaded = load_settings()
                        assert loaded.analysis_selected_operations == [
                            "colors", "shots", "describe", "cinematography"
                        ]

    def test_save_and_load_empty_analysis_selected_operations(self):
        """Test round-trip with empty selection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"

            with patch("core.settings._get_config_path", return_value=config_path):
                with patch("core.settings._get_api_key_from_keyring", return_value=""):
                    with patch("core.settings._set_api_key_in_keyring", return_value=True):
                        settings = Settings()
                        settings.analysis_selected_operations = []
                        assert save_settings(settings) is True

                        loaded = load_settings()
                        assert loaded.analysis_selected_operations == []

    def test_analysis_selected_in_json_schema(self):
        """Test that selected_operations appears in the analysis section of JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"

            with patch("core.settings._get_config_path", return_value=config_path):
                with patch("core.settings._set_api_key_in_keyring", return_value=True):
                    settings = Settings()
                    settings.analysis_selected_operations = ["colors", "shots"]
                    save_settings(settings)

                    data = json.loads(config_path.read_text())
                    assert "analysis" in data
                    assert "selected_operations" in data["analysis"]
                    assert data["analysis"]["selected_operations"] == ["colors", "shots"]


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


class TestDownloadDirectoryValidation:
    """Tests for download directory validation functions."""

    def test_validate_download_dir_existing(self):
        """Test validation of existing writable directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            valid, error = validate_download_dir(Path(tmpdir))
            assert valid is True
            assert error == ""

    def test_validate_download_dir_creates_new(self):
        """Test that validation creates directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = Path(tmpdir) / "new_subdir" / "nested"
            assert not new_dir.exists()

            valid, error = validate_download_dir(new_dir)
            assert valid is True
            assert error == ""
            assert new_dir.exists()

    @pytest.mark.skipif(
        sys.platform == "win32" or (hasattr(os, "getuid") and os.getuid() == 0),
        reason="Skip permission test on Windows or when root",
    )
    def test_validate_download_dir_permission_denied(self):
        """Test validation fails for non-writable paths."""
        # /root is typically not writable by non-root users
        invalid_path = Path("/root/scene_ripper_test_invalid")
        valid, error = validate_download_dir(invalid_path)
        assert valid is False
        assert "Permission" in error or "Cannot" in error

    def test_get_default_download_dir_contains_app_name(self):
        """Test that default download dir includes app identifier."""
        default = get_default_download_dir()
        assert "Scene Ripper" in str(default)

    def test_is_download_dir_from_env_true(self):
        """Test detection when download dir is set via env var."""
        with patch.dict(os.environ, {ENV_DOWNLOAD_DIR: "/tmp/test_downloads"}):
            assert is_download_dir_from_env() is True

    def test_is_download_dir_from_env_false(self):
        """Test detection when download dir is not set via env var."""
        env_copy = os.environ.copy()
        env_copy.pop(ENV_DOWNLOAD_DIR, None)
        with patch.dict(os.environ, env_copy, clear=True):
            assert is_download_dir_from_env() is False


class TestSequenceSelectedCategory:
    """Tests for sequence_selected_category persistence."""

    def test_default_sequence_selected_category(self):
        """Default value is 'All' on a fresh Settings instance."""
        settings = Settings()
        assert settings.sequence_selected_category == "All"

    def test_round_trip_sequence_selected_category(self):
        """Round-trip: set to 'Audio', save to JSON, load from JSON, value is 'Audio'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"

            with patch("core.settings._get_config_path", return_value=config_path):
                with patch("core.settings._get_api_key_from_keyring", return_value=""):
                    with patch("core.settings._set_api_key_in_keyring", return_value=True):
                        settings = Settings()
                        settings.sequence_selected_category = "Audio"
                        assert save_settings(settings) is True

                        loaded = load_settings()
                        assert loaded.sequence_selected_category == "Audio"

    def test_missing_sequence_section_falls_back_to_all(self):
        """Loading with missing 'sequence' section falls back to 'All'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            # Write JSON without a "sequence" section
            data = {"version": "1.0", "detection": {"default_sensitivity": 3.0}}
            config_path.write_text(json.dumps(data))

            with patch("core.settings._get_config_path", return_value=config_path):
                with patch("core.settings._get_api_key_from_keyring", return_value=""):
                    loaded = load_settings()
                    assert loaded.sequence_selected_category == "All"

    def test_unrecognized_category_falls_back_to_all(self):
        """Loading with unrecognized category value falls back to 'All'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            data = {
                "version": "1.0",
                "sequence": {"selected_category": "deleted_category"},
            }
            config_path.write_text(json.dumps(data))

            with patch("core.settings._get_config_path", return_value=config_path):
                with patch("core.settings._get_api_key_from_keyring", return_value=""):
                    loaded = load_settings()
                    assert loaded.sequence_selected_category == "All"

    def test_non_string_category_falls_back_to_all(self):
        """Loading with non-string category value falls back to 'All'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            data = {
                "version": "1.0",
                "sequence": {"selected_category": 42},
            }
            config_path.write_text(json.dumps(data))

            with patch("core.settings._get_config_path", return_value=config_path):
                with patch("core.settings._get_api_key_from_keyring", return_value=""):
                    loaded = load_settings()
                    assert loaded.sequence_selected_category == "All"


class TestChatGPTOAuthToken:
    """Tests for the ChatGPT subscription OAuth token blob storage.

    The token is a JSON-serialized blob under a single keyring key.
    Plaintext backends (Linux without Secret Service) must be refused —
    OAuth refresh tokens warrant fail-closed behavior on insecure storage.
    """

    def _make_fake_keyring(self, *, backend_module="keyring.backends.macOS", backend_name="Keyring"):
        """Build a MagicMock standing in for the `keyring` module."""
        backend_cls = type(backend_name, (), {})
        backend_cls.__module__ = backend_module
        backend_instance = backend_cls()

        keyring = MagicMock()
        keyring.get_keyring.return_value = backend_instance
        # Storage simulating successful set/get with no prior value
        keyring.get_password.side_effect = [None]
        return keyring

    def test_round_trip_set_then_get(self):
        """Setting a blob then reading it back returns an equal dict."""
        from core.settings import (
            get_chatgpt_oauth_token,
            set_chatgpt_oauth_token,
            KEYRING_CHATGPT_OAUTH_TOKEN,
            KEYRING_SERVICE,
        )

        keyring = self._make_fake_keyring()
        stored = {"value": None}

        def fake_set(service, key, value):
            stored["value"] = value
        def fake_get(service, key):
            return stored["value"]

        keyring.set_password.side_effect = fake_set
        keyring.get_password.side_effect = lambda service, key: fake_get(service, key)

        blob = {
            "access_token": "at_abc",
            "refresh_token": "rt_xyz",
            "id_token": "id_def",
            "expires_at_unix": 1747000000,
            "account_email": "derrick@example.com",
        }

        with patch.dict(sys.modules, {"keyring": keyring}):
            assert set_chatgpt_oauth_token(blob) is True
            keyring.set_password.assert_called_once()
            args = keyring.set_password.call_args.args
            assert args[0] == KEYRING_SERVICE
            assert args[1] == KEYRING_CHATGPT_OAUTH_TOKEN
            # Stored value is JSON-serialized
            assert json.loads(args[2]) == blob

            assert get_chatgpt_oauth_token() == blob

    def test_set_none_deletes_entry(self):
        """Setting None clears the stored token via the delete path."""
        from core.settings import set_chatgpt_oauth_token

        keyring = self._make_fake_keyring()
        # Make a mock errors module so delete-path doesn't blow up
        keyring.errors = MagicMock()
        keyring.errors.PasswordDeleteError = type("PasswordDeleteError", (Exception,), {})

        with patch.dict(sys.modules, {"keyring": keyring}):
            assert set_chatgpt_oauth_token(None) is True
            assert keyring.delete_password.called
            keyring.set_password.assert_not_called()

    def test_get_returns_none_when_no_token(self):
        """get_chatgpt_oauth_token returns None when no entry is stored."""
        from core.settings import get_chatgpt_oauth_token

        keyring = MagicMock()
        keyring.get_password.return_value = None

        with patch.dict(sys.modules, {"keyring": keyring}):
            assert get_chatgpt_oauth_token() is None

    def test_get_self_clears_on_malformed_json(self):
        """Malformed JSON is logged and the entry is cleared to break retry loops."""
        from core.settings import get_chatgpt_oauth_token

        keyring = MagicMock()
        keyring.errors = MagicMock()
        keyring.errors.PasswordDeleteError = type("PasswordDeleteError", (Exception,), {})
        keyring.get_password.return_value = "this is not json {"

        with patch.dict(sys.modules, {"keyring": keyring}):
            assert get_chatgpt_oauth_token() is None
            assert keyring.delete_password.called

    def test_set_refuses_plaintext_keyring_backend(self):
        """A PlaintextKeyring backend is refused; no entry is created."""
        from core.settings import set_chatgpt_oauth_token

        keyring = self._make_fake_keyring(
            backend_module="keyrings.alt.file",
            backend_name="PlaintextKeyring",
        )

        with patch.dict(sys.modules, {"keyring": keyring}):
            assert set_chatgpt_oauth_token({"access_token": "at_abc"}) is False
            keyring.set_password.assert_not_called()

    def test_set_refuses_any_keyrings_alt_backend(self):
        """Anything from keyrings.alt is treated as plaintext-equivalent."""
        from core.settings import set_chatgpt_oauth_token

        keyring = self._make_fake_keyring(
            backend_module="keyrings.alt.pyfs",
            backend_name="EncryptedKeyring",
        )

        with patch.dict(sys.modules, {"keyring": keyring}):
            assert set_chatgpt_oauth_token({"access_token": "at_abc"}) is False
            keyring.set_password.assert_not_called()

    def test_set_accepts_macos_keychain_backend(self):
        """The macOS Keychain backend is accepted; the blob is written."""
        from core.settings import set_chatgpt_oauth_token

        keyring = self._make_fake_keyring(
            backend_module="keyring.backends.macOS",
            backend_name="Keyring",
        )

        with patch.dict(sys.modules, {"keyring": keyring}):
            assert set_chatgpt_oauth_token({"access_token": "at_abc"}) is True
            keyring.set_password.assert_called_once()


class TestAuthModeSettings:
    """Tests for the auth_mode and chatgpt_account_email Settings fields."""

    def test_auth_mode_defaults_to_api_key(self):
        """New Settings instances default to api_key mode for backward compat."""
        s = Settings()
        assert s.auth_mode == "api_key"
        assert s.chatgpt_account_email == ""

    def test_auth_mode_persists_through_save_and_load(self):
        """Setting subscription mode survives the JSON round trip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            with patch("core.settings._get_config_path", return_value=config_path):
                with patch("core.settings._get_api_key_from_keyring", return_value=""):
                    s = Settings()
                    s.auth_mode = "subscription"
                    s.chatgpt_account_email = "derrick@example.com"
                    assert save_settings(s) is True

                    loaded = load_settings()
                    assert loaded.auth_mode == "subscription"
                    assert loaded.chatgpt_account_email == "derrick@example.com"

    def test_load_without_auth_section_keeps_defaults(self):
        """A config.json without 'auth' loads cleanly with api_key mode default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(json.dumps({"version": "1.0"}))

            with patch("core.settings._get_config_path", return_value=config_path):
                with patch("core.settings._get_api_key_from_keyring", return_value=""):
                    loaded = load_settings()
                    assert loaded.auth_mode == "api_key"
                    assert loaded.chatgpt_account_email == ""

    def test_load_rejects_unknown_auth_mode(self):
        """Unknown mode strings are ignored; the default applies."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(json.dumps({
                "version": "1.0",
                "auth": {"mode": "garbage"},
            }))

            with patch("core.settings._get_config_path", return_value=config_path):
                with patch("core.settings._get_api_key_from_keyring", return_value=""):
                    loaded = load_settings()
                    assert loaded.auth_mode == "api_key"
