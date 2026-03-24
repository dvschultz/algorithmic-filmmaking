# Scene Ripper Windows build script.
#
# Usage:
#   .\packaging\windows\build.ps1
#
# Prerequisites:
#   - Python 3.11+
#   - 7-Zip available on PATH
#   - Inno Setup available on PATH for installer builds

$ErrorActionPreference = "Stop"

Write-Host "=== Scene Ripper Windows Build ===" -ForegroundColor Cyan

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Resolve-Path (Join-Path $scriptDir "..\..")
Push-Location $projectRoot

if (-not $env:APP_VERSION) {
    $gitVersion = (git describe --tags --abbrev=0 2>$null)
    if ($gitVersion) {
        $env:APP_VERSION = $gitVersion.TrimStart("v")
    } else {
        $env:APP_VERSION = "0.0.0"
    }
}
if (-not $env:APP_BUILD_VERSION) {
    $env:APP_BUILD_VERSION = $env:APP_VERSION
}
if (-not $env:APP_UPDATE_CHANNEL) {
    $env:APP_UPDATE_CHANNEL = "stable"
}
if (-not $env:WINSPARKLE_APPCAST_BETA_URL -and $env:WINSPARKLE_APPCAST_URL) {
    $env:WINSPARKLE_APPCAST_BETA_URL = $env:WINSPARKLE_APPCAST_URL
}

if (-not $env:WINSPARKLE_PUBLIC_ED_KEY -and $env:UPDATE_PRIVATE_ED_KEY) {
    $resolvedPublicKey = python -c @"
import importlib.util
from pathlib import Path
import os
spec = importlib.util.spec_from_file_location("scene_ripper_build_support", Path("packaging/build_support.py"))
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
print(module.resolve_update_public_ed_key("", os.environ.get("UPDATE_PRIVATE_ED_KEY", "")), end="")
"@
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to derive WINSPARKLE_PUBLIC_ED_KEY from UPDATE_PRIVATE_ED_KEY"
    }
    if ($resolvedPublicKey) {
        $env:WINSPARKLE_PUBLIC_ED_KEY = $resolvedPublicKey
    }
}

Write-Host "Building Scene Ripper version $($env:APP_VERSION)" -ForegroundColor Yellow

& (Join-Path $scriptDir "stage-runtimes.ps1") -ProjectRoot $projectRoot

# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements-core.txt pyinstaller cryptography

# Build with PyInstaller
Write-Host "Building with PyInstaller..." -ForegroundColor Yellow
pyinstaller packaging/windows/scene_ripper.spec --distpath dist --workpath build --noconfirm

if ($LASTEXITCODE -ne 0) {
    Write-Host "PyInstaller build failed!" -ForegroundColor Red
    exit 1
}

Write-Host "Build complete! Output in dist/Scene Ripper/" -ForegroundColor Green
if (-not (Test-Path (Join-Path $projectRoot "dist\Scene Ripper\Scene Ripper.exe"))) {
    Write-Host "Built executable missing from dist/Scene Ripper/" -ForegroundColor Red
    exit 1
}
$bundledMpvDll = Get-ChildItem -Path (Join-Path $projectRoot "dist\Scene Ripper") -Recurse -File -ErrorAction SilentlyContinue | Where-Object {
    $_.Name -in @("mpv-2.dll","libmpv-2.dll","mpv-1.dll")
} | Select-Object -First 1
if (-not $bundledMpvDll) {
    Write-Host "Bundled mpv runtime DLL missing from dist/Scene Ripper/" -ForegroundColor Red
    exit 1
}
$bundledFfmpegExe = Get-ChildItem -Path (Join-Path $projectRoot "dist\Scene Ripper") -Recurse -File -ErrorAction SilentlyContinue | Where-Object {
    $_.Name -eq "ffmpeg.exe"
} | Select-Object -First 1
if (-not $bundledFfmpegExe) {
    Write-Host "Bundled ffmpeg.exe missing from dist/Scene Ripper/" -ForegroundColor Red
    exit 1
}
$bundledFfprobeExe = Get-ChildItem -Path (Join-Path $projectRoot "dist\Scene Ripper") -Recurse -File -ErrorAction SilentlyContinue | Where-Object {
    $_.Name -eq "ffprobe.exe"
} | Select-Object -First 1
if (-not $bundledFfprobeExe) {
    Write-Host "Bundled ffprobe.exe missing from dist/Scene Ripper/" -ForegroundColor Red
    exit 1
}
$bundledWinSparkleDll = Get-ChildItem -Path (Join-Path $projectRoot "dist\Scene Ripper") -Recurse -File -ErrorAction SilentlyContinue | Where-Object {
    $_.Name -eq "WinSparkle.dll"
} | Select-Object -First 1
if (-not $bundledWinSparkleDll) {
    Write-Host "Bundled WinSparkle.dll missing from dist/Scene Ripper/" -ForegroundColor Red
    exit 1
}
$bundledFeedFile = Get-ChildItem -Path (Join-Path $projectRoot "dist\Scene Ripper") -Recurse -File -ErrorAction SilentlyContinue | Where-Object {
    $_.Name -eq "app_update_feed_url.txt"
} | Select-Object -First 1
$bundledPublicKeyFile = Get-ChildItem -Path (Join-Path $projectRoot "dist\Scene Ripper") -Recurse -File -ErrorAction SilentlyContinue | Where-Object {
    $_.Name -eq "app_update_public_key.txt"
} | Select-Object -First 1
if ($env:WINSPARKLE_APPCAST_URL -and (-not $bundledFeedFile -or [string]::IsNullOrWhiteSpace((Get-Content $bundledFeedFile.FullName -Raw)))) {
    Write-Host "Bundled WinSparkle feed metadata file missing or empty from dist/Scene Ripper/" -ForegroundColor Red
    exit 1
}
if ($env:WINSPARKLE_PUBLIC_ED_KEY -and (-not $bundledPublicKeyFile -or [string]::IsNullOrWhiteSpace((Get-Content $bundledPublicKeyFile.FullName -Raw)))) {
    Write-Host "Bundled WinSparkle public-key metadata file missing or empty from dist/Scene Ripper/" -ForegroundColor Red
    exit 1
}

# Check if Inno Setup is available
$iscc = Get-Command iscc -ErrorAction SilentlyContinue
if ($iscc) {
    Write-Host "Building installer with Inno Setup..." -ForegroundColor Yellow
    iscc packaging/windows/scene_ripper.iss
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Inno Setup build failed!" -ForegroundColor Red
        exit 1
    }
    if (-not (Get-ChildItem -Path (Join-Path $scriptDir "Output") -Filter "*.exe" -ErrorAction SilentlyContinue)) {
        Write-Host "Installer build completed but no installer was created." -ForegroundColor Red
        exit 1
    }
    Write-Host "Installer built! Output in packaging/windows/Output/" -ForegroundColor Green
} else {
    Write-Host "Inno Setup not found — skipping installer. Install from https://jrsoftware.org/isdown.php" -ForegroundColor Yellow
}

Pop-Location
