"""Spine import-boundary tests.

These tests defend the spine layering rule: ``core/spine/*`` modules MUST NOT
pull GUI or heavy-runtime dependencies (PySide6, mpv, av, faster_whisper,
paddleocr, mlx_vlm) into ``sys.modules``. Both the GUI agent and the MCP
server depend on this — the spine must remain importable from a headless
process.

Pattern mirrors ``tests/test_transcription_runtime_imports.py``.
"""

from __future__ import annotations

import importlib
import sys

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
    "core.spine",
    "core.spine.security",
    "core.spine.url_security",
    "core.spine.project_io",
    "core.spine._agent_formatting",
    "core.spine.audio_sources",
    "core.spine.chatgpt_auth",
    "core.spine.chatgpt_oauth_flow",
    "core.spine.clips",
    "core.spine.exports",
    "core.spine.frames",
    "core.spine.glossary",
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


def _purge_modules(*names: str) -> None:
    for name in list(sys.modules):
        for prefix in names:
            if name == prefix or name.startswith(prefix + "."):
                sys.modules.pop(name, None)


def _assert_no_forbidden_modules() -> None:
    for forbidden in FORBIDDEN_MODULES:
        assert forbidden not in sys.modules, (
            f"Spine import pulled in forbidden module: {forbidden}"
        )


def test_spine_modules_do_not_load_gui_or_runtime_deps():
    """Importing every spine module must not load any forbidden module."""
    _purge_modules(*SPINE_MODULES, *FORBIDDEN_MODULES)

    for module_name in SPINE_MODULES:
        importlib.import_module(module_name)

    _assert_no_forbidden_modules()


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
