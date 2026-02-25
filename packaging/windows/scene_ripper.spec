# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for Scene Ripper Windows .exe.

Build with:
    pyinstaller packaging/windows/scene_ripper.spec --distpath dist --workpath build --noconfirm

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
    datas=[
        ("../../core/package_manifest.json", "core"),
    ],
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
