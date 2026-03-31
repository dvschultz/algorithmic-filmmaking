# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for Scene Ripper macOS .app bundle.

Build with:
    pyinstaller packaging/macos/scene_ripper.spec

Requires: pyinstaller, requirements-core.txt installed.
"""

import importlib.util
import os
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)

block_cipher = None

# Version from environment variable or default
VERSION = os.environ.get("APP_VERSION", "0.2.0")
BUILD_VERSION = os.environ.get("APP_BUILD_VERSION", VERSION)
UPDATE_CHANNEL = os.environ.get("APP_UPDATE_CHANNEL", "stable")
SPARKLE_FEED_URL = os.environ.get("SPARKLE_FEED_URL", "")
SPARKLE_PUBLIC_ED_KEY = os.environ.get("SPARKLE_PUBLIC_ED_KEY", "")
SPARKLE_PRIVATE_ED_KEY = os.environ.get("UPDATE_PRIVATE_ED_KEY", "")
PROJECT_ROOT = Path.cwd()
BUILD_ROOT = Path(os.environ.get("SCENE_RIPPER_BUILD_ROOT", str(PROJECT_ROOT / "build")))
VERSION_FILE = BUILD_ROOT / "release-metadata" / "app_version.txt"
BUILD_VERSION_FILE = BUILD_ROOT / "release-metadata" / "app_build_version.txt"
UPDATE_CHANNEL_FILE = BUILD_ROOT / "release-metadata" / "app_update_channel.txt"
VERSION_FILE.parent.mkdir(parents=True, exist_ok=True)
VERSION_FILE.write_text(VERSION, encoding="utf-8")
BUILD_VERSION_FILE.write_text(BUILD_VERSION, encoding="utf-8")
UPDATE_CHANNEL_FILE.write_text(UPDATE_CHANNEL, encoding="utf-8")

build_support_spec = importlib.util.spec_from_file_location(
    "scene_ripper_build_support",
    PROJECT_ROOT / "packaging" / "build_support.py",
)
build_support = importlib.util.module_from_spec(build_support_spec)
build_support_spec.loader.exec_module(build_support)
SPARKLE_PUBLIC_ED_KEY = build_support.resolve_update_public_ed_key(
    SPARKLE_PUBLIC_ED_KEY,
    SPARKLE_PRIVATE_ED_KEY,
)
binaries = (
    build_support.collect_macos_mpv_binaries(PROJECT_ROOT)
    + build_support.collect_macos_ffmpeg_binaries(PROJECT_ROOT)
)
sparkle_datas = build_support.collect_macos_sparkle_datas(PROJECT_ROOT)


def _unique(items):
    return list(dict.fromkeys(items))


def _collect_module_payload(module_name):
    if build_support.use_full_package_collection(module_name):
        return collect_all(module_name)

    excluded_hiddenimports = build_support.get_pyinstaller_hiddenimport_excludes(module_name)
    return (
        collect_data_files(
            module_name,
            excludes=list(build_support.get_pyinstaller_data_excludes(module_name)),
        ),
        collect_dynamic_libs(module_name),
        collect_submodules(
            module_name,
            filter=lambda name: not any(
                name == prefix or name.startswith(prefix + ".")
                for prefix in excluded_hiddenimports
            ),
        ),
    )


core_requirement_hiddenimports = []
core_requirement_datas = []
core_requirement_binaries = []
for module_name in build_support.get_core_pyinstaller_collect_targets(
    PROJECT_ROOT,
    "requirements-core.txt",
):
    module_datas, module_binaries, module_hiddenimports = _collect_module_payload(module_name)
    core_requirement_datas.extend(module_datas)
    core_requirement_binaries.extend(module_binaries)
    core_requirement_hiddenimports.extend(module_hiddenimports)

for metadata_name in build_support.get_core_pyinstaller_metadata(
    PROJECT_ROOT,
    "requirements-core.txt",
):
    core_requirement_datas.extend(copy_metadata(metadata_name))

a = Analysis(
    ["../../main.py"],
    pathex=[],
    binaries=binaries + core_requirement_binaries,
    datas=[
        ("../../core/package_manifest.json", "core"),
        (str(VERSION_FILE), "core"),
        (str(BUILD_VERSION_FILE), "core"),
        (str(UPDATE_CHANNEL_FILE), "core"),
    ] + sparkle_datas + core_requirement_datas,
    hiddenimports=_unique([
        # PySide6 modules actually used by the app
        "PySide6.QtWidgets",
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtNetwork",
        # Qt platform plugin
        "PySide6.QtSvg",
        # shiboken6 (PySide6 binding layer)
        "shiboken6",
        # keyring backends
        "keyring.backends.macOS",
        # scenedetect
        "scenedetect",
        "scenedetect.detectors",
    ] + list(build_support.get_core_pyinstaller_hiddenimports()) + core_requirement_hiddenimports),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Unused PySide6 modules (saves ~200-400 MB)
        "PySide6.QtWebEngine",
        "PySide6.QtWebEngineCore",
        "PySide6.QtWebEngineWidgets",
        "PySide6.QtWebChannel",
        "PySide6.Qt3DCore",
        "PySide6.Qt3DRender",
        "PySide6.Qt3DInput",
        "PySide6.Qt3DLogic",
        "PySide6.Qt3DExtras",
        "PySide6.Qt3DAnimation",
        "PySide6.QtQuick",
        "PySide6.QtQuickWidgets",
        "PySide6.QtQuickControls2",
        "PySide6.QtQml",
        "PySide6.QtBluetooth",
        "PySide6.QtCharts",
        "PySide6.QtDataVisualization",
        "PySide6.QtSensors",
        "PySide6.QtSerialPort",
        "PySide6.QtSerialBus",
        "PySide6.QtTest",
        "PySide6.QtPdf",
        "PySide6.QtPdfWidgets",
        "PySide6.QtPositioning",
        "PySide6.QtLocation",
        "PySide6.QtNfc",
        "PySide6.QtRemoteObjects",
        "PySide6.QtScxml",
        "PySide6.QtStateMachine",
        "PySide6.QtTextToSpeech",
        "PySide6.QtVirtualKeyboard",
        "PySide6.QtDesigner",
        "PySide6.QtHelp",
        "PySide6.QtDBus",
        # No longer used (video playback via MPV, not Qt Multimedia)
        "PySide6.QtMultimedia",
        "PySide6.QtMultimediaWidgets",
        # On-demand packages (downloaded at runtime, not bundled).
        # Excluding these prevents unsigned binaries from leaking into the
        # app bundle (e.g., torch/bin/protoc) which Apple notarization rejects.
        "torch",
        "torchvision",
        "torchaudio",
        "transformers",
        "ultralytics",
        "faster_whisper",
        "lightning_whisper_mlx",
        "mlx_vlm",
        "mlx",
        "librosa",
        "demucs_infer",
        "insightface",
        "onnxruntime",
        "paddleocr",
        "paddlepaddle",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Scene Ripper",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,  # Don't strip — breaks macOS code signing
    upx=False,    # Don't use UPX — breaks macOS code signing
    console=False,  # Windowed app (no terminal)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,  # Inherit runner arch (arm64 on Apple Silicon CI)
    codesign_identity=None,  # Sign separately after build
    entitlements_file=None,  # Apply entitlements during manual signing
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Scene Ripper",
)

info_plist = {
    "CFBundleDisplayName": "Scene Ripper",
    "CFBundleShortVersionString": VERSION,
    "CFBundleVersion": BUILD_VERSION,
    "NSHighResolutionCapable": True,
    "LSMinimumSystemVersion": "13.0",
    "NSHumanReadableCopyright": "Copyright 2024-2026 Algorithmic Filmmaking",
    # Permissions descriptions
    "NSAppleEventsUsageDescription": "Scene Ripper needs automation access.",
    # Dark mode support
    "NSRequiresAquaSystemAppearance": False,
}

if SPARKLE_FEED_URL:
    info_plist["SUFeedURL"] = SPARKLE_FEED_URL
if SPARKLE_PUBLIC_ED_KEY:
    info_plist["SUPublicEDKey"] = SPARKLE_PUBLIC_ED_KEY

app = BUNDLE(
    coll,
    name="Scene Ripper.app",
    icon="../../assets/icon.icns",
    bundle_identifier="com.algorithmic-filmmaking.scene-ripper",
    info_plist=info_plist,
)
