"""Tests for optional packaging runtime helpers."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_module(name: str, relative_path: str):
    module_path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


build_support = _load_module("scene_ripper_build_support_tests", "packaging/build_support.py")
collect_macos_sparkle_datas = build_support.collect_macos_sparkle_datas
collect_macos_ffmpeg_binaries = build_support.collect_macos_ffmpeg_binaries
collect_macos_model_datas = build_support.collect_macos_model_datas
collect_windows_ffmpeg_binaries = build_support.collect_windows_ffmpeg_binaries
collect_windows_winsparkle_binaries = build_support.collect_windows_winsparkle_binaries
get_core_pyinstaller_collect_targets = build_support.get_core_pyinstaller_collect_targets
get_core_pyinstaller_hiddenimports = build_support.get_core_pyinstaller_hiddenimports
get_core_pyinstaller_metadata = build_support.get_core_pyinstaller_metadata
get_pyinstaller_data_excludes = build_support.get_pyinstaller_data_excludes
get_pyinstaller_hiddenimport_excludes = build_support.get_pyinstaller_hiddenimport_excludes
read_core_requirement_distributions = build_support.read_core_requirement_distributions
resolve_macos_sparkle_metadata = build_support.resolve_macos_sparkle_metadata
resolve_update_public_ed_key = build_support.resolve_update_public_ed_key
use_full_package_collection = build_support.use_full_package_collection


def test_collect_macos_sparkle_datas_returns_empty_when_not_staged(tmp_path):
    """Optional Sparkle runtime collection should be empty when no runtime is staged."""
    assert collect_macos_sparkle_datas(tmp_path) == []


def test_collect_macos_sparkle_datas_preserves_relative_layout(tmp_path):
    """Staged Sparkle assets should bundle only the canonical versioned framework tree."""
    sparkle_cli = (
        tmp_path
        / "packaging"
        / "runtime"
        / "sparkle"
        / "macos"
        / "Sparkle.framework"
        / "Versions"
        / "B"
        / "Resources"
        / "bin"
        / "sparkle"
    )
    sparkle_cli.parent.mkdir(parents=True)
    sparkle_cli.write_text("binary", encoding="utf-8")

    collected = collect_macos_sparkle_datas(tmp_path)

    assert len(collected) == 1
    source, destination = collected[0]
    assert source == str(sparkle_cli)
    assert destination.replace("\\", "/") == "Sparkle.framework/Versions/B/Resources/bin"


def test_collect_windows_winsparkle_binaries_returns_empty_when_not_staged(tmp_path):
    """Optional WinSparkle collection should be empty when no runtime is staged."""
    assert collect_windows_winsparkle_binaries(tmp_path) == []


def test_collect_windows_winsparkle_binaries_collects_staged_dll(tmp_path):
    """Staged WinSparkle DLLs should be copied to the executable directory."""
    winsparkle_dll = (
        tmp_path
        / "packaging"
        / "runtime"
        / "winsparkle"
        / "windows"
        / "WinSparkle.dll"
    )
    winsparkle_dll.parent.mkdir(parents=True)
    winsparkle_dll.write_text("binary", encoding="utf-8")

    collected = collect_windows_winsparkle_binaries(tmp_path)

    assert collected == [(str(winsparkle_dll), ".")]


def test_collect_windows_ffmpeg_binaries_collects_staged_runtime(tmp_path):
    """Staged Windows FFmpeg runtime files should bundle under bin/."""
    runtime_dir = (
        tmp_path
        / "packaging"
        / "runtime"
        / "ffmpeg"
        / "windows"
    )
    runtime_dir.mkdir(parents=True)
    (runtime_dir / "ffmpeg.exe").write_text("binary", encoding="utf-8")
    (runtime_dir / "ffprobe.exe").write_text("binary", encoding="utf-8")
    (runtime_dir / "avcodec-61.dll").write_text("binary", encoding="utf-8")

    collected = collect_windows_ffmpeg_binaries(tmp_path)

    bundled = {(Path(src).name, destination.replace("\\", "/")) for src, destination in collected}
    assert ("ffmpeg.exe", "bin") in bundled
    assert ("ffprobe.exe", "bin") in bundled
    assert ("avcodec-61.dll", "bin") in bundled


def test_collect_windows_ffmpeg_binaries_rejects_duplicate_required_binaries(tmp_path):
    """Windows packaging should not bundle ambiguous duplicate FFmpeg executables."""
    runtime_dir = (
        tmp_path
        / "packaging"
        / "runtime"
        / "ffmpeg"
        / "windows"
    )
    nested_dir = runtime_dir / "nested"
    nested_dir.mkdir(parents=True)
    (runtime_dir / "ffmpeg.exe").write_text("binary", encoding="utf-8")
    (runtime_dir / "ffprobe.exe").write_text("binary", encoding="utf-8")
    (nested_dir / "ffmpeg.exe").write_text("duplicate", encoding="utf-8")

    import pytest

    with pytest.raises(RuntimeError, match="Expected exactly one ffmpeg.exe"):
        collect_windows_ffmpeg_binaries(tmp_path)


def test_windows_runtime_manifest_assets_match_download_urls():
    """Windows runtime pins should point at the exact archives they stage."""
    manifest_path = PROJECT_ROOT / "packaging" / "windows" / "runtime-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for name, entry in manifest.items():
        asset_name = entry["asset_name"]
        url = entry["url"]
        assert url.startswith("https://github.com/")
        assert url.endswith(f"/{asset_name}"), name
        assert len(entry["sha256"]) == 64
        assert all(char in "0123456789abcdef" for char in entry["sha256"].lower())


def test_windows_runtime_stager_rejects_html_downloads():
    """Runtime staging should fail clearly when GitHub returns an HTML error page."""
    script = (PROJECT_ROOT / "packaging" / "windows" / "stage-runtimes.ps1").read_text(
        encoding="utf-8"
    )

    assert "MaximumRedirection" in script
    assert "Downloaded HTML instead of archive" in script
    assert "runtime-manifest.json URL" in script


def test_collect_macos_ffmpeg_binaries_collects_staged_runtime(tmp_path):
    """Staged macOS FFmpeg runtime files should bundle under bin/."""
    runtime_dir = (
        tmp_path
        / "packaging"
        / "runtime"
        / "ffmpeg"
        / "macos"
    )
    runtime_dir.mkdir(parents=True)
    (runtime_dir / "ffmpeg").write_text("binary", encoding="utf-8")
    (runtime_dir / "ffprobe").write_text("binary", encoding="utf-8")

    collected = collect_macos_ffmpeg_binaries(tmp_path)

    bundled = {(Path(src).name, destination.replace("\\", "/")) for src, destination in collected}
    assert ("ffmpeg", "bin") in bundled
    assert ("ffprobe", "bin") in bundled


def test_collect_macos_ffmpeg_binaries_rejects_duplicate_required_binaries(tmp_path):
    """macOS packaging should not bundle ambiguous duplicate FFmpeg binaries."""
    runtime_dir = (
        tmp_path
        / "packaging"
        / "runtime"
        / "ffmpeg"
        / "macos"
    )
    nested_dir = runtime_dir / "nested"
    nested_dir.mkdir(parents=True)
    (runtime_dir / "ffmpeg").write_text("binary", encoding="utf-8")
    (runtime_dir / "ffprobe").write_text("binary", encoding="utf-8")
    (nested_dir / "ffprobe").write_text("duplicate", encoding="utf-8")

    import pytest

    with pytest.raises(RuntimeError, match="Expected exactly one ffprobe"):
        collect_macos_ffmpeg_binaries(tmp_path)


def test_collect_macos_model_datas_collects_staged_runtime(tmp_path):
    """Staged bundled-model assets should be bundled under models/."""
    runtime_dir = tmp_path / "packaging" / "runtime" / "models" / "macos"
    (runtime_dir / "huggingface").mkdir(parents=True)
    (runtime_dir / "huggingface" / "weights.bin").write_text("binary", encoding="utf-8")
    (runtime_dir / "manifest.json").write_text("{}", encoding="utf-8")

    collected = collect_macos_model_datas(tmp_path)

    bundled = {(Path(src).name, destination.replace("\\", "/")) for src, destination in collected}
    assert ("manifest.json", "models") in bundled
    assert ("weights.bin", "models/huggingface") in bundled


def test_core_requirement_distributions_follow_requirements_file(tmp_path):
    """Frozen bundle mappings should be derived from requirements-core.txt."""
    distributions = read_core_requirement_distributions(PROJECT_ROOT)
    assert "certifi" in distributions
    assert "litellm" in distributions
    assert "scikit-learn" in distributions
    assert "opencv-python" in distributions
    assert "google-api-python-client" in distributions
    assert "pyside6" in distributions
    # rapidfuzz is required by the Cassette Tape sequencer (core)
    assert "rapidfuzz" in distributions
    # paddleocr is optional (on-demand)
    assert "paddleocr" not in distributions


def test_core_requirement_distributions_support_direct_reference_requirements(tmp_path):
    """PEP 508 direct references should still resolve to the package distribution name."""
    requirements_file = tmp_path / "requirements-core.txt"
    requirements_file.write_text(
        "litellm @ https://files.pythonhosted.org/packages/02/6c/5327667e6dbe9e98cbfbd4261c8e91386a52e38f41419575854248bbab6a/litellm-1.82.6-py3-none-any.whl#sha256=164a3ef3e19f309e3cabc199bef3d2045212712fefdfa25fc7f75884a5b5b205\n",
        encoding="utf-8",
    )

    distributions = read_core_requirement_distributions(tmp_path)

    assert distributions == ("litellm",)


def test_core_requirement_distributions_support_local_wheel_requirements(tmp_path):
    """Local vendored wheel paths should still resolve to the package distribution name."""
    requirements_file = tmp_path / "requirements-core.txt"
    requirements_file.write_text(
        "./vendor/wheels/litellm-1.82.6-py3-none-any.whl\n",
        encoding="utf-8",
    )

    distributions = read_core_requirement_distributions(tmp_path)

    assert distributions == ("litellm",)


def test_core_requirement_distributions_follow_nested_requirements_file(tmp_path):
    """Nested -r includes should contribute their distribution names."""
    (tmp_path / "requirements-core.txt").write_text(
        "certifi>=2024.0.0\n",
        encoding="utf-8",
    )
    (tmp_path / "requirements-core-macos.txt").write_text(
        "-r requirements-core.txt\ntransformers>=4.50,<5\ntorchaudio>=2.4,<2.7\npaddlepaddle>=3.3.0,<4\n",
        encoding="utf-8",
    )

    distributions = read_core_requirement_distributions(tmp_path, "requirements-core-macos.txt")

    assert distributions == ("certifi", "transformers", "torchaudio", "paddlepaddle")


def test_core_pyinstaller_collect_targets_cover_packaged_runtime_dependencies(tmp_path):
    """Frozen builds should explicitly collect dynamic-import package trees."""
    targets = get_core_pyinstaller_collect_targets(PROJECT_ROOT)
    assert "certifi" in targets
    assert "googleapiclient" in targets
    assert "google_auth_httplib2" in targets
    assert "httplib2" in targets
    assert "sklearn" in targets
    assert "scipy" in targets
    assert "cv2" in targets
    assert "numpy" in targets
    assert "mpv" in targets
    # rapidfuzz is required by the Cassette Tape sequencer (core)
    assert "rapidfuzz" in targets
    # paddleocr is optional (on-demand)
    assert "paddleocr" not in targets


def test_core_pyinstaller_metadata_covers_core_requirements(tmp_path):
    """Frozen builds should carry distribution metadata for bundled core requirements."""
    metadata = get_core_pyinstaller_metadata(PROJECT_ROOT)
    assert "certifi" in metadata
    assert "google-api-python-client" in metadata
    assert "scikit-learn" in metadata
    assert "scipy" in metadata
    assert "pillow" in metadata
    assert "opencv-python" in metadata
    # paddleocr is in requirements-core-macos.txt, not core
    assert "paddleocr" not in metadata


def test_core_pyinstaller_hiddenimports_include_on_demand_stdlib_dependencies():
    """Frozen builds should carry stdlib modules needed by on-demand packages."""
    hiddenimports = get_core_pyinstaller_hiddenimports()
    assert "pickletools" in hiddenimports
    assert "filecmp" in hiddenimports
    assert "modulefinder" in hiddenimports
    assert "cProfile" in hiddenimports
    assert "profile" in hiddenimports
    assert "pstats" in hiddenimports
    assert "html" in hiddenimports
    assert "html.parser" in hiddenimports
    assert "html.entities" in hiddenimports
    assert "_markupbase" in hiddenimports
    assert "wave" in hiddenimports
    assert "aifc" in hiddenimports
    assert "sunau" in hiddenimports


def test_resolve_update_public_ed_key_prefers_explicit_value():
    """Explicitly configured updater keys should pass through unchanged."""
    assert resolve_update_public_ed_key("public-key", "ignored") == "public-key"


def test_resolve_update_public_ed_key_derives_from_private_seed():
    """Updater packaging should derive the public key from a raw Ed25519 seed."""
    import base64

    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519

    private = ed25519.Ed25519PrivateKey.generate()
    private_seed = private.private_bytes(
        serialization.Encoding.Raw,
        serialization.PrivateFormat.Raw,
        serialization.NoEncryption(),
    )
    expected_public = base64.b64encode(
        private.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )
    ).decode("ascii")

    derived_public = resolve_update_public_ed_key(
        "",
        base64.b64encode(private_seed).decode("ascii"),
    )

    assert derived_public == expected_public


def test_resolve_update_public_ed_key_derives_from_private_pkcs8_der():
    """Updater packaging should also accept base64-encoded PKCS8 DER private keys."""
    import base64

    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519

    private = ed25519.Ed25519PrivateKey.generate()
    private_der = private.private_bytes(
        serialization.Encoding.DER,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    expected_public = base64.b64encode(
        private.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )
    ).decode("ascii")

    derived_public = resolve_update_public_ed_key(
        "",
        base64.b64encode(private_der).decode("ascii"),
    )

    assert derived_public == expected_public


def test_resolve_macos_sparkle_metadata_accepts_canonical_env_names():
    """macOS packaging should read Sparkle-specific metadata env vars."""
    metadata = resolve_macos_sparkle_metadata(
        {
            "SPARKLE_FEED_URL": "https://example.com/appcast-macos.xml",
            "SPARKLE_PUBLIC_ED_KEY": "public-key",
        }
    )

    assert metadata.feed_url == "https://example.com/appcast-macos.xml"
    assert metadata.public_ed_key == "public-key"


def test_resolve_macos_sparkle_metadata_accepts_release_env_aliases():
    """Local macOS builds should support the documented updater env names."""
    metadata = resolve_macos_sparkle_metadata(
        {
            "UPDATE_FEED_URL": "https://example.com/appcast-macos.xml",
            "UPDATE_PUBLIC_ED_KEY": "public-key",
        }
    )

    assert metadata.feed_url == "https://example.com/appcast-macos.xml"
    assert metadata.public_ed_key == "public-key"


def test_litellm_uses_curated_collection_rules():
    """Frozen builds should not collect LiteLLM proxy/test payloads wholesale."""
    assert not use_full_package_collection("litellm")
    assert "**/proxy/**" in get_pyinstaller_data_excludes("litellm")
    assert "litellm.proxy" in get_pyinstaller_hiddenimport_excludes("litellm")


def test_collect_runtime_worker_datas_stages_the_worker_package_as_source(tmp_path):
    root = tmp_path / "repo"
    package = root / "core" / "runtime_worker"
    package.mkdir(parents=True)
    for name in ("__init__.py", "__main__.py", "protocol.py", "tasks.py"):
        (package / name).write_text("# stub\n")
    (package / "notes.txt").write_text("not python")
    collected = build_support.collect_runtime_worker_datas(root)
    assert {Path(src).name for src, _ in collected} == {"__init__.py", "__main__.py", "protocol.py", "tasks.py"}
    assert {dest for _, dest in collected} == {str(Path("runtime_worker_src") / "runtime_worker")}
    assert build_support.runtime_worker_staged_layout() == ("runtime_worker_src", "runtime_worker")


def test_collect_runtime_worker_datas_requires_the_package(tmp_path):
    with pytest.raises(RuntimeError, match="runtime_worker"):
        build_support.collect_runtime_worker_datas(tmp_path)


def test_real_runtime_worker_package_is_stdlib_only_at_import():
    """The staged worker must import without project or model dependencies."""
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1] / "core"
    code = (
        "import sys; sys.modules.pop('core', None); import runtime_worker, runtime_worker.tasks;"
        "bad=[m for m in sys.modules if m.split('.')[0] in ('core','models','PySide6','torch','numpy','faster_whisper')];"
        "assert not bad, bad; print('ok')"
    )
    # -I ignores PYTHONPATH; put the worker root on the path explicitly instead.
    code = f"import sys; sys.path.insert(0, {str(root)!r}); " + code
    result = subprocess.run([sys.executable, "-I", "-c", code], cwd=str(root), capture_output=True, text=True)
    assert result.returncode == 0 and "ok" in result.stdout, result.stderr
