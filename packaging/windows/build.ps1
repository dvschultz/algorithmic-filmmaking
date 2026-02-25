# Scene Ripper Windows build script.
#
# Usage:
#   .\packaging\windows\build.ps1
#
# Prerequisites:
#   - Python 3.11+
#   - pip install -r requirements-core.txt pyinstaller

$ErrorActionPreference = "Stop"

Write-Host "=== Scene Ripper Windows Build ===" -ForegroundColor Cyan

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

# Check if Inno Setup is available
$iscc = Get-Command iscc -ErrorAction SilentlyContinue
if ($iscc) {
    Write-Host "Building installer with Inno Setup..." -ForegroundColor Yellow
    iscc packaging/windows/scene_ripper.iss
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Inno Setup build failed!" -ForegroundColor Red
        exit 1
    }
    Write-Host "Installer built! Output in packaging/windows/Output/" -ForegroundColor Green
} else {
    Write-Host "Inno Setup not found — skipping installer. Install from https://jrsoftware.org/isdown.php" -ForegroundColor Yellow
}
