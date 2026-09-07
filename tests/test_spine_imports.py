"""Spine import-boundary tests.

These tests defend the spine layering rule: ``core/spine/*`` modules MUST NOT
pull GUI or heavy-runtime dependencies (PySide6, mpv, av, faster_whisper,
paddleocr, mlx_vlm) into ``sys.modules``. Both the GUI agent and the MCP
server depend on this — the spine must remain importable from a headless
process.

Pattern mirrors ``tests/test_transcription_runtime_imports.py``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# Modules that must never appear in ``sys.modules`` after a spine import.
FORBIDDEN_MODULES: tuple[str, ...] = (
    "PySide6",
    "mpv",
    "av",
    "faster_whisper",
    "paddleocr",
    "mlx_vlm",
)

# Spine modules under test. Add new spine modules here as they land.
SPINE_MODULES: tuple[str, ...] = (
    "core.operations.shots",
    "core.operations.audio_import",
    "core.operations.image_import",
    "core.jobs.image_import",
    "core.jobs.gui_image_import",
    "core.jobs.audio_import",
    "core.jobs.gui_audio_import",
    "core.operations.frame_extraction",
    "core.jobs.gui_frame_extraction",
    "core.jobs.frame_extraction",
    "core.operations.audio_transcription",
    "core.jobs.gui_audio_transcription",
    "core.jobs.audio_transcription",
    "core.project_session",
    "core.jobs",
    "core.jobs.commits",
    "core.jobs.colors",
    "core.jobs.detection",
    "core.jobs.transcription",
    "core.jobs.alignment",
    "core.jobs.description",
    "core.jobs.media",
    "core.jobs.gui_alignment",
    "core.jobs.gui_results",
    "core.jobs.gui_checkpoints",
    "core.jobs.gui_transcription",
    "core.jobs.gui_description",
    "core.jobs.gui_custom_query",
    "core.jobs.custom_query",
    "core.jobs.cinematography",
    "core.jobs.classification",
    "core.jobs.object_detection",
    "core.jobs.faces",
    "core.jobs.gaze",
    "core.jobs.embeddings",
    "core.jobs.boundary_embeddings",
    "core.jobs.sequence_embeddings",
    "core.operations.ocr",
    "core.jobs.ocr",
    "core.jobs.gui_ocr",
    "core.jobs.gui_boundary_embeddings",
    "core.jobs.gui_cinematography",
    "core.jobs.gui_classification",
    "core.jobs.gui_object_detection",
    "core.jobs.gui_faces",
    "core.jobs.gui_gaze",
    "core.jobs.gui_embeddings",
    "core.operations.boundary_embeddings",
    "core.jobs.analysis",
    "core.operations.analysis_plan",
    "core.project_migrations",
    "core.project_lock",
    "core.commands.clip_disabled",
    "core.commands.sequence_clips",
    "core.commands.sequences",
    "core.commands.sources",
    "core.commands.metadata",
    "core.spine.metadata",
    "core.spine.sequences",
    "core.spine.history",
    "core.operations.contracts",
    "core.operations.classification",
    "core.operations.faces",
    "core.operations.gaze",
    "core.operations.embeddings",
    "core.operations.object_detection",
    "core.operations.colors",
    "core.operations.detection",
    "core.operations.transcription",
    "core.operations.alignment",
    "core.operations.description",
    "core.operations.custom_query",
    "core.operations.cinematography",
    "core.provider_errors",
    "core.spine",
    "core.spine.security",
    "core.spine.url_security",
    "core.spine.project_io",
    "core.spine.project_sessions",
    "core.spine.timeline",
    "core.spine._agent_formatting",
    "core.spine.audio_sources",
    "core.spine.chatgpt_auth",
    "core.spine.chatgpt_oauth_flow",
    "core.spine.clips",
    "core.spine.exports",
    "core.spine.frames",
    "core.spine.glossary",
    "core.spine.log_redaction",
    "core.spine.project_save",
    "core.spine.queries",
    "core.spine.sequence_analysis",
    "core.spine.settings_io",
    "core.spine.sources",
    "core.spine.detect",
    "core.spine.analyze",
    "core.spine.thumbnails",
    "core.spine.downloads",
    "core.spine.words",
)


def test_spine_modules_do_not_load_gui_or_runtime_deps():
    """Test in a fresh interpreter; never unload native modules in pytest.

    Removing PySide6 from sys.modules does not unload its native libraries and
    can segfault a later Qt import in the same test process.
    """
    code = f"""
import importlib
import sys
for name in {SPINE_MODULES!r}:
    importlib.import_module(name)
for forbidden in {FORBIDDEN_MODULES!r}:
    if any(name == forbidden or name.startswith(forbidden + '.') for name in sys.modules):
        raise RuntimeError('Spine import pulled in forbidden module: ' + forbidden)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_import_boundary_detects_a_forbidden_dependency(monkeypatch):
    monkeypatch.setitem(globals(), "SPINE_MODULES", SPINE_MODULES + ("PySide6.QtCore",))
    with pytest.raises(AssertionError, match="forbidden module: PySide6"):
        test_spine_modules_do_not_load_gui_or_runtime_deps()


def test_spine_security_validates_safe_path(tmp_path):
    """Smoke test — happy path through ``validate_path``."""
    from core.spine.security import validate_path

    target = tmp_path / "proj.sceneripper"
    target.write_text("{}")

    valid, err, resolved = validate_path(
        str(target), must_exist=True, must_be_file=True
    )
    assert valid, err
    assert resolved == target.resolve()


def test_spine_security_rejects_traversal():
    from core.spine.security import validate_path

    valid, err, _ = validate_path("/Users/foo/../../etc/passwd")
    assert not valid
    assert "traversal" in err.lower()


def test_spine_security_rejects_relative_paths():
    from core.spine.security import validate_path

    valid, err, _ = validate_path("relative/path")
    assert not valid
    assert "absolute" in err.lower()


def test_spine_security_rejects_empty_and_none_like():
    from core.spine.security import validate_path

    valid, err, _ = validate_path("")
    assert not valid
    assert "empty" in err.lower()


def test_spine_security_rejects_outside_safe_roots():
    import sys as _sys

    if _sys.platform == "win32":
        # On Windows every absolute path lives under a drive-letter root
        # (C:\, D:\, ...) and SAFE_ROOTS includes those drive roots, so a
        # cross-platform "absolute but outside safe roots" probe doesn't
        # exist. The Linux/macOS path below covers the rejection logic; the
        # platform-specific Windows behavior is exercised by
        # tests/test_windows_compat.py.
        import pytest

        pytest.skip("All absolute Windows paths are under drive-letter SAFE_ROOTS")

    from core.spine.security import validate_path

    valid, err, _ = validate_path("/etc/passwd")
    assert not valid
    assert "home" in err.lower() or "safe" in err.lower() or "temp" in err.lower()


def test_spine_security_must_be_file_rejects_directory(tmp_path):
    from core.spine.security import validate_path

    valid, err, _ = validate_path(str(tmp_path), must_be_file=True)
    assert not valid
    assert "not a file" in err.lower()


def test_spine_security_must_be_dir_rejects_file(tmp_path):
    from core.spine.security import validate_path

    target = tmp_path / "f.txt"
    target.write_text("x")

    valid, err, _ = validate_path(str(target), must_be_dir=True)
    assert not valid
    assert "not a directory" in err.lower()


def test_spine_security_video_path_extension_check(tmp_path):
    from core.spine.security import validate_video_path

    bad = tmp_path / "f.txt"
    bad.write_text("x")
    valid, err, _ = validate_video_path(str(bad))
    assert not valid
    assert "video" in err.lower()

    good = tmp_path / "f.mp4"
    good.write_text("x")
    valid, _, _ = validate_video_path(str(good))
    assert valid


def test_spine_security_project_path_extension_check(tmp_path):
    from core.spine.security import validate_project_path

    bad = tmp_path / "f.txt"
    bad.write_text("x")
    valid, err, _ = validate_project_path(str(bad))
    assert not valid
    assert ".sceneripper" in err

    good = tmp_path / "f.sceneripper"
    good.write_text("{}")
    valid, _, _ = validate_project_path(str(good))
    assert valid


def test_url_security_rejects_bad_schemes():
    from core.spine.url_security import validate_url

    for url in (
        "javascript://example.com/x",
        "file:///etc/passwd",
        "ftp://example.com/x",
        "data:text/plain,hello",
    ):
        valid, err = validate_url(url)
        assert not valid, f"expected reject for {url}, got {err!r}"


def test_url_security_accepts_youtube_and_subdomains():
    from core.spine.url_security import validate_url

    for url in (
        "https://youtube.com/watch?v=abc",
        "https://www.youtube.com/watch?v=abc",
        "https://m.youtube.com/watch?v=abc",
        "https://youtu.be/abc",
        "https://vimeo.com/123",
        "https://archive.org/details/foo",
        "https://ia800.us.archive.org/foo",
    ):
        valid, err = validate_url(url)
        assert valid, f"expected accept for {url}, got {err!r}"


def test_url_security_rejects_lookalike_subdomains():
    from core.spine.url_security import validate_url

    # Suffix match without the leading-dot anchor would accept this.
    valid, _ = validate_url("https://evil.notyoutube.com/x")
    assert not valid


def test_url_security_strips_credentials_and_ports():
    from core.spine.url_security import validate_url

    # Credentials should be stripped before the host check; the underlying
    # host is still youtube.com so this remains valid.
    valid, _ = validate_url("https://user:pass@youtube.com/x")
    assert valid

    # Default port is allowed implicitly.
    valid, _ = validate_url("https://youtube.com:443/x")
    assert valid

    # Non-default ports are rejected — SSRF defence.
    valid, _ = validate_url("https://youtube.com:9999/x")
    assert not valid

    valid, _ = validate_url("http://youtube.com:80/x")
    assert valid


def test_url_security_rejects_empty():
    from core.spine.url_security import validate_url

    for url in ("", None):
        valid, _ = validate_url(url)  # type: ignore[arg-type]
        assert not valid
