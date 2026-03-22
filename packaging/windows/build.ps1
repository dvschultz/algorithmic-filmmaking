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
$runtimeDir = Join-Path $projectRoot "packaging\runtime\mpv\windows"
$ffmpegRuntimeDir = Join-Path $projectRoot "packaging\runtime\ffmpeg\windows"
Push-Location $projectRoot

Write-Host "Staging mpv runtime..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null
Remove-Item (Join-Path $runtimeDir "*.dll") -Force -ErrorAction SilentlyContinue

$mpvArchive = Join-Path $projectRoot "mpv.7z"
$release = Invoke-RestMethod `
    -Uri "https://api.github.com/repos/shinchiro/mpv-winbuild-cmake/releases/latest" `
    -Headers @{ "User-Agent" = "Scene-Ripper-Build" }

$asset = $release.assets | Where-Object { $_.name -like "mpv-dev-x86_64-*.7z" } | Select-Object -First 1
if (-not $asset) {
    Write-Host "Could not find a matching mpv development archive in the latest GitHub release." -ForegroundColor Red
    exit 1
}

Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $mpvArchive

$extractDir = Join-Path $projectRoot "tmp\mpv"
New-Item -ItemType Directory -Force -Path $extractDir | Out-Null
7z x $mpvArchive "-o$extractDir" | Out-Null

$mpvDll = Get-ChildItem -Path $extractDir -Recurse -Include "mpv-2.dll","libmpv-2.dll","mpv-1.dll" | Select-Object -First 1
if (-not $mpvDll) {
    Write-Host "No supported mpv runtime DLL found in downloaded runtime archive." -ForegroundColor Red
    exit 1
}

$dllDir = $mpvDll.Directory.FullName
Get-ChildItem -Path $dllDir -Filter "*.dll" | ForEach-Object {
    Copy-Item $_.FullName -Destination $runtimeDir -Force
}

Write-Host "Staging FFmpeg runtime..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $ffmpegRuntimeDir | Out-Null
Remove-Item (Join-Path $ffmpegRuntimeDir "*") -Force -ErrorAction SilentlyContinue

$ffmpegArchive = Join-Path $projectRoot "ffmpeg.zip"
Invoke-WebRequest `
    -Uri "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip" `
    -OutFile $ffmpegArchive

$ffmpegExtractDir = Join-Path $projectRoot "tmp\winffmpeg"
New-Item -ItemType Directory -Force -Path $ffmpegExtractDir | Out-Null
Expand-Archive -Path $ffmpegArchive -DestinationPath $ffmpegExtractDir -Force

$ffmpegExe = Get-ChildItem -Path $ffmpegExtractDir -Recurse -Filter "ffmpeg.exe" | Select-Object -First 1
$ffprobeExe = Get-ChildItem -Path $ffmpegExtractDir -Recurse -Filter "ffprobe.exe" | Select-Object -First 1
if (-not $ffmpegExe -or -not $ffprobeExe) {
    Write-Host "FFmpeg runtime executables not found in downloaded archive." -ForegroundColor Red
    exit 1
}

$ffmpegBinDir = $ffmpegExe.Directory.FullName
Get-ChildItem -Path $ffmpegBinDir -File | ForEach-Object {
    Copy-Item $_.FullName -Destination $ffmpegRuntimeDir -Force
}

# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements-core.txt pyinstaller

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
