param(
    [string]$ProjectRoot = ""
)

$ErrorActionPreference = "Stop"

if (-not $ProjectRoot) {
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    $ProjectRoot = Resolve-Path (Join-Path $scriptDir "..\..")
}

$manifestPath = Join-Path $ProjectRoot "packaging\windows\runtime-manifest.json"
if (-not (Test-Path $manifestPath)) {
    throw "Runtime manifest not found: $manifestPath"
}

$manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json

function Download-AndVerifyAsset {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$Destination,
        [Parameter(Mandatory = $true)][string]$ExpectedSha256
    )

    Invoke-WebRequest -Uri $Url -OutFile $Destination
    $actualSha256 = (Get-FileHash -Path $Destination -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualSha256 -ne $ExpectedSha256.ToLowerInvariant()) {
        throw "SHA256 mismatch for $(Split-Path $Destination -Leaf). Expected $ExpectedSha256, got $actualSha256"
    }
}

$mpvRuntimeDir = Join-Path $ProjectRoot "packaging\runtime\mpv\windows"
$ffmpegRuntimeDir = Join-Path $ProjectRoot "packaging\runtime\ffmpeg\windows"
$winSparkleRuntimeDir = Join-Path $ProjectRoot "packaging\runtime\winsparkle\windows"
$winSparkleToolsDir = Join-Path $winSparkleRuntimeDir "tools"

Write-Host "Staging mpv runtime..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $mpvRuntimeDir | Out-Null
Remove-Item (Join-Path $mpvRuntimeDir "*.dll") -Force -ErrorAction SilentlyContinue
Remove-Item (Join-Path $ProjectRoot "tmp\mpv") -Recurse -Force -ErrorAction SilentlyContinue

$mpvArchive = Join-Path $ProjectRoot $manifest.mpv.asset_name
Download-AndVerifyAsset -Url $manifest.mpv.url -Destination $mpvArchive -ExpectedSha256 $manifest.mpv.sha256

$mpvExtractDir = Join-Path $ProjectRoot "tmp\mpv"
New-Item -ItemType Directory -Force -Path $mpvExtractDir | Out-Null
7z x $mpvArchive "-o$mpvExtractDir" | Out-Null

$mpvDll = Get-ChildItem -Path $mpvExtractDir -Recurse -Include "mpv-2.dll","libmpv-2.dll","mpv-1.dll" | Select-Object -First 1
if (-not $mpvDll) {
    throw "No supported mpv runtime DLL found in extracted archive"
}

$mpvDllDir = $mpvDll.Directory.FullName
Get-ChildItem -Path $mpvDllDir -Filter "*.dll" | ForEach-Object {
    Copy-Item $_.FullName -Destination $mpvRuntimeDir -Force
}

$stagedMpvDll = Get-ChildItem -Path $mpvRuntimeDir -File | Where-Object {
    $_.Name -in @("mpv-2.dll", "libmpv-2.dll", "mpv-1.dll")
} | Select-Object -First 1
if (-not $stagedMpvDll) {
    throw "Failed to stage an mpv runtime DLL"
}

Write-Host "Staging FFmpeg runtime..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $ffmpegRuntimeDir | Out-Null
Remove-Item (Join-Path $ffmpegRuntimeDir "*") -Force -ErrorAction SilentlyContinue
Remove-Item (Join-Path $ProjectRoot "tmp\winffmpeg") -Recurse -Force -ErrorAction SilentlyContinue

$ffmpegArchive = Join-Path $ProjectRoot $manifest.ffmpeg.asset_name
Download-AndVerifyAsset -Url $manifest.ffmpeg.url -Destination $ffmpegArchive -ExpectedSha256 $manifest.ffmpeg.sha256

$ffmpegExtractDir = Join-Path $ProjectRoot "tmp\winffmpeg"
New-Item -ItemType Directory -Force -Path $ffmpegExtractDir | Out-Null
Expand-Archive -Path $ffmpegArchive -DestinationPath $ffmpegExtractDir -Force

$ffmpegExe = Get-ChildItem -Path $ffmpegExtractDir -Recurse -Filter "ffmpeg.exe" | Select-Object -First 1
$ffprobeExe = Get-ChildItem -Path $ffmpegExtractDir -Recurse -Filter "ffprobe.exe" | Select-Object -First 1
if (-not $ffmpegExe -or -not $ffprobeExe) {
    throw "Failed to locate FFmpeg runtime executables in extracted archive"
}

$ffmpegBinDir = $ffmpegExe.Directory.FullName
Get-ChildItem -Path $ffmpegBinDir -File | ForEach-Object {
    Copy-Item $_.FullName -Destination $ffmpegRuntimeDir -Force
}

if (-not (Test-Path (Join-Path $ffmpegRuntimeDir "ffmpeg.exe"))) {
    throw "Failed to stage ffmpeg.exe runtime"
}
if (-not (Test-Path (Join-Path $ffmpegRuntimeDir "ffprobe.exe"))) {
    throw "Failed to stage ffprobe.exe runtime"
}

Write-Host "Staging WinSparkle runtime..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $winSparkleToolsDir | Out-Null
Remove-Item (Join-Path $winSparkleRuntimeDir "*.dll") -Force -ErrorAction SilentlyContinue
Remove-Item (Join-Path $winSparkleToolsDir "*") -Force -ErrorAction SilentlyContinue
Remove-Item (Join-Path $ProjectRoot "tmp\winsparkle") -Recurse -Force -ErrorAction SilentlyContinue

$winSparkleArchive = Join-Path $ProjectRoot $manifest.winsparkle.asset_name
Download-AndVerifyAsset -Url $manifest.winsparkle.url -Destination $winSparkleArchive -ExpectedSha256 $manifest.winsparkle.sha256

$winSparkleExtractDir = Join-Path $ProjectRoot "tmp\winsparkle"
New-Item -ItemType Directory -Force -Path $winSparkleExtractDir | Out-Null
Expand-Archive -Path $winSparkleArchive -DestinationPath $winSparkleExtractDir -Force

$dllCandidates = Get-ChildItem -Path $winSparkleExtractDir -Recurse -Filter "WinSparkle.dll"
$winSparkleDll = $dllCandidates | Where-Object { $_.FullName -match '(?i)(x64|amd64)' } | Select-Object -First 1
if (-not $winSparkleDll) {
    $winSparkleDll = $dllCandidates | Select-Object -First 1
}
if (-not $winSparkleDll) {
    throw "Failed to locate WinSparkle.dll in extracted release assets"
}

$winSparkleTool = Get-ChildItem -Path $winSparkleExtractDir -Recurse -Filter "winsparkle-tool.exe" | Select-Object -First 1
if (-not $winSparkleTool) {
    throw "Failed to locate winsparkle-tool.exe in extracted release assets"
}

Copy-Item $winSparkleDll.FullName -Destination $winSparkleRuntimeDir -Force
Copy-Item $winSparkleTool.FullName -Destination $winSparkleToolsDir -Force
