# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for Scene Ripper Windows .exe.

Build with:
    pyinstaller packaging/windows/scene_ripper.spec --distpath dist --workpath build --noconfirm

Requires: pyinstaller, requirements-core.txt installed.
"""

import importlib.util
import os
from pathlib import Path

block_cipher = None

# Version from environment variable or default
VERSION = os.environ.get("APP_VERSION", "0.2.0")
BUILD_VERSION = os.environ.get("APP_BUILD_VERSION", VERSION)
UPDATE_CHANNEL = os.environ.get("APP_UPDATE_CHANNEL", "stable")
WINSPARKLE_FEED_URL = os.environ.get("WINSPARKLE_APPCAST_URL", "")
WINSPARKLE_BETA_FEED_URL = os.environ.get("WINSPARKLE_APPCAST_BETA_URL", "")
WINSPARKLE_PUBLIC_ED_KEY = os.environ.get("WINSPARKLE_PUBLIC_ED_KEY", "")
PROJECT_ROOT = Path.cwd()
VERSION_FILE = PROJECT_ROOT / "build" / "release-metadata" / "app_version.txt"
BUILD_VERSION_FILE = PROJECT_ROOT / "build" / "release-metadata" / "app_build_version.txt"
UPDATE_CHANNEL_FILE = PROJECT_ROOT / "build" / "release-metadata" / "app_update_channel.txt"
UPDATE_FEED_FILE = PROJECT_ROOT / "build" / "release-metadata" / "app_update_feed_url.txt"
UPDATE_BETA_FEED_FILE = PROJECT_ROOT / "build" / "release-metadata" / "app_update_feed_url_beta.txt"
UPDATE_PUBLIC_KEY_FILE = PROJECT_ROOT / "build" / "release-metadata" / "app_update_public_key.txt"
VERSION_FILE.parent.mkdir(parents=True, exist_ok=True)
VERSION_FILE.write_text(VERSION, encoding="utf-8")
BUILD_VERSION_FILE.write_text(BUILD_VERSION, encoding="utf-8")
UPDATE_CHANNEL_FILE.write_text(UPDATE_CHANNEL, encoding="utf-8")
UPDATE_FEED_FILE.write_text(WINSPARKLE_FEED_URL, encoding="utf-8")
UPDATE_BETA_FEED_FILE.write_text(WINSPARKLE_BETA_FEED_URL, encoding="utf-8")
UPDATE_PUBLIC_KEY_FILE.write_text(WINSPARKLE_PUBLIC_ED_KEY, encoding="utf-8")

build_support_spec = importlib.util.spec_from_file_location(
    "scene_ripper_build_support",
    PROJECT_ROOT / "packaging" / "build_support.py",
)
build_support = importlib.util.module_from_spec(build_support_spec)
build_support_spec.loader.exec_module(build_support)
binaries = (
    build_support.collect_windows_mpv_binaries(PROJECT_ROOT)
    + build_support.collect_windows_winsparkle_binaries(PROJECT_ROOT)
)

a = Analysis(
    ["../../main.py"],
    pathex=[],
    binaries=binaries,
    datas=[
        ("../../core/package_manifest.json", "core"),
        (str(VERSION_FILE), "core"),
        (str(BUILD_VERSION_FILE), "core"),
        (str(UPDATE_CHANNEL_FILE), "core"),
        (str(UPDATE_FEED_FILE), "core"),
        (str(UPDATE_BETA_FEED_FILE), "core"),
        (str(UPDATE_PUBLIC_KEY_FILE), "core"),
    ],
    hiddenimports=[
        # PySide6 modules actually used by the app
        "PySide6.QtWidgets",
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtNetwork",
        # Qt platform plugin
        "PySide6.QtSvg",
        # shiboken6 (PySide6 binding layer)
        "shiboken6",
        # keyring backends — Windows uses WinVaultKeyring
        "keyring.backends.Windows",
        # scenedetect
        "scenedetect",
        "scenedetect.detectors",
    ],
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
        "PySide6.QtOpenGL",
        "PySide6.QtOpenGLWidgets",
        "PySide6.QtDBus",
        # No longer used (video playback via MPV, not Qt Multimedia)
        "PySide6.QtMultimedia",
        "PySide6.QtMultimediaWidgets",
        # macOS-only backends
        "keyring.backends.macOS",
        "lightning_whisper_mlx",
        "mlx_vlm",
        "mlx",
        # On-demand packages (downloaded at runtime, not bundled)
        "torch",
        "torchvision",
        "transformers",
        "ultralytics",
        "paddleocr",
        "paddlepaddle",
        "librosa",
        "einops",
        # Unused stdlib
        "tkinter",
        "unittest",
        "test",
        "distutils",
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
    strip=False,
    upx=False,
    console=False,  # Windowed app (no terminal)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="../../assets/icon.ico",
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
