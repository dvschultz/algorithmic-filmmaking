# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for Scene Ripper macOS .app bundle.

Build with:
    pyinstaller packaging/macos/scene_ripper.spec

Requires: pyinstaller, requirements-core.txt installed.
"""

import os
import sys

block_cipher = None

# Version from environment variable or default
VERSION = os.environ.get("APP_VERSION", "0.2.0")

a = Analysis(
    ["../../main.py"],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        # PySide6 modules actually used by the app
        "PySide6.QtWidgets",
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtMultimedia",
        "PySide6.QtMultimediaWidgets",
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
        # On-demand packages (downloaded at runtime, not bundled)
        "torch",
        "torchvision",
        "transformers",
        "ultralytics",
        "paddleocr",
        "paddlepaddle",
        "faster_whisper",
        "lightning_whisper_mlx",
        "mlx_vlm",
        "mlx",
        "librosa",
        "scipy",
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

app = BUNDLE(
    coll,
    name="Scene Ripper.app",
    icon="../../assets/icon.icns",
    bundle_identifier="com.algorithmic-filmmaking.scene-ripper",
    info_plist={
        "CFBundleDisplayName": "Scene Ripper",
        "CFBundleShortVersionString": VERSION,
        "CFBundleVersion": VERSION,
        "NSHighResolutionCapable": True,
        "LSMinimumSystemVersion": "13.0",
        "NSHumanReadableCopyright": "Copyright 2024-2026 Algorithmic Filmmaking",
        # Permissions descriptions
        "NSAppleEventsUsageDescription": "Scene Ripper needs automation access.",
        # Dark mode support
        "NSRequiresAquaSystemAppearance": False,
    },
)
