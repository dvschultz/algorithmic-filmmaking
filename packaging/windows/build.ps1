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

Write-Host "Staging mpv runtime..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null
Remove-Item (Join-Path $runtimeDir "*.dll") -Force -ErrorAction SilentlyContinue

$mpvArchive = Join-Path $projectRoot "mpv.7z"
Invoke-WebRequest `
    -Uri "https://sourceforge.net/projects/mpv-player-windows/files/libmpv/mpv-dev-x86_64-latest.7z/download" `
    -OutFile $mpvArchive

$extractDir = Join-Path $projectRoot "tmp\mpv"
New-Item -ItemType Directory -Force -Path $extractDir | Out-Null
7z x $mpvArchive "-o$extractDir" | Out-Null

$mpvDll = Get-ChildItem -Path $extractDir -Recurse -Filter "mpv-2.dll" | Select-Object -First 1
if (-not $mpvDll) {
    Write-Host "mpv-2.dll not found in downloaded runtime archive." -ForegroundColor Red
    exit 1
}

$dllDir = $mpvDll.Directory.FullName
Get-ChildItem -Path $dllDir -Filter "*.dll" | ForEach-Object {
    Copy-Item $_.FullName -Destination $runtimeDir -Force
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
if (-not (Test-Path (Join-Path $projectRoot "dist\Scene Ripper\mpv-2.dll"))) {
    Write-Host "Bundled mpv-2.dll missing from dist/Scene Ripper/" -ForegroundColor Red
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
