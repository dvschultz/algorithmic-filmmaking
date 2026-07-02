# Runtime Acquisition Inventory

**Generated:** 2026-07-01 (one-shot, date-stamped snapshot — not maintained)
**Git commit:** `3cec9f6` (`3cec9f6b2f2a727bd37d272cf486240e13e7a889`)
**Platform hashed:** macOS ARM64 (Apple Silicon). Other platforms' URLs are recorded but not hashed.

This document inventories every artifact Scene Ripper acquires at **runtime** (downloaded on first use of a feature) or **build time** (bundled into the distributed `.app` by CI). It is a point-in-time audit; the underlying sources are not kept in sync with this file.

---

## Mutable references summary

Every artifact below whose reference can change out from under a pinned build. These are the supply-chain "floating" points — the same URL/spec can resolve to different bytes over time.

**GitHub `/latest/` release redirects** (each resolves to whatever the newest release is at download time):
- `yt-dlp_macos` — `https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_macos`
- `yt-dlp.exe` (Windows) — `.../releases/latest/download/yt-dlp.exe`
- `yt-dlp_linux`, `yt-dlp_linux_aarch64` (Linux) — `.../releases/latest/download/...`
- Deno (all platforms) — `https://github.com/denoland/deno/releases/latest/download/deno-<triple>.zip`

**GitHub `latest` / `master`-branch build tags** (rolling build artifacts, not a fixed version):
- FFmpeg Windows/Linux (BtbN) — `https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-<platform>-gpl.<ext>` (runtime download path in `dependency_manager.py`). Note: the **build-time** Windows/macOS FFmpeg is pinned to a specific asset (see build-time table); this `latest` reference is the on-demand runtime fallback.

**`master`-branch raw file references** (follow the branch head, not a tag/commit):
- EAST text-detection model — `https://github.com/oyyd/frozen_east_text_detection.pb/raw/master/frozen_east_text_detection.pb`
- ImageNet class labels — `https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt`

**Git-HEAD pip specifier** (installs from the default branch of a git repo — no tag/commit pin):
- `ctc-forced-aligner` — `ctc-forced-aligner @ git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git` (T6 addresses commit-pinning this)

**Bare-name pip specifiers** (no version constraint — pip resolves to newest compatible at install time):
- `mediapipe` — used by the `gaze_detect` feature in `feature_registry.py`; it has **no entry in `core/package_manifest.json`**, so `get_pip_specifier()` falls back to the bare name `mediapipe` (unconstrained).

**Floating HuggingFace `main`-branch model references** (no revision pin — resolve to repo `main` at download time):
- `mlx-community/Qwen3-VL-4B-Instruct-4bit` (local VLM, describe_local) — no revision pin
- `google/siglip2-base-patch16-224` (shot classification) — no revision pin
- `facebook/dinov2-base` (embeddings) — no revision pin
- InsightFace `buffalo_l` model pack (face_detect) — auto-downloaded by the `insightface` library, no revision pin

**Pinned HuggingFace reference** (for contrast — this one IS pinned):
- `vikhyatk/moondream2` — pinned to git revision `2025-06-21` via `MOONDREAM_REVISION` in `core/analysis/description.py`

**Version-floating on-demand pip specifiers** (range specs — resolve within the range at install time). All packages in `package_manifest.json` and `requirements-optional.txt` use `>=`/`<` ranges, e.g. `torch>=2.4,<2.7`, `transformers>=4.50,<5`. These are "floating within a range," not fully pinned.

---

## Artifact inventory

Legend for **Mutable?**: `MUTABLE` = flagged above (latest/master/git-HEAD/bare-name/unpinned-HF); `range` = version range spec; `pinned` = fixed version/tag/asset.

### Direct-download binaries (managed by `core/dependency_manager.py`)

| Name | Acquisition (URL) | Version | License | Install path | Bundled / on-demand / build-time | Mutable? | SHA-256 |
|---|---|---|---|---|---|---|---|
| FFmpeg (macOS ARM64) | `https://www.osxexperts.net/ffmpeg7arm.zip` | FFmpeg 7.x (osxexperts static build; "7arm") | GPL / LGPL (FFmpeg) — verify build's GPL flags | Managed bin dir (`~/Library/Application Support/Scene Ripper/bin/`) | on-demand (also build-time bundled, see below) | pinned URL (unversioned filename — content can change) | `563111a239fe70d2e5c84a5382204a7d0bf0a332385a92a44baff36d313e27f2` |
| FFprobe (macOS ARM64) | `https://www.osxexperts.net/ffprobe7arm.zip` | FFmpeg 7.x (osxexperts static build) | GPL / LGPL (FFmpeg) — verify | Managed bin dir | on-demand (also build-time bundled) | pinned URL (unversioned filename) | `e5ae34ee2f0b3594892a695fd733646904bbc7eb40af3b359ed91538ddcb5513` |
| yt-dlp (macOS) | `https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_macos` | floating (`/latest/`) | Unlicense (public domain) — verify | Managed bin dir | on-demand | **MUTABLE** (`/latest/`) | `b82c3626952e6c14eaf654cc565866775ffd0b9ffb7021628ac59b42c2f4f244` (as of 2026-07-01) |
| Deno (macOS ARM64) | `https://github.com/denoland/deno/releases/latest/download/deno-aarch64-apple-darwin.zip` | floating (`/latest/`) | MIT (Deno) — verify | Managed bin dir | on-demand (required by yt-dlp for modern YouTube extraction) | **MUTABLE** (`/latest/`) | `ee3473502118eab301eca93aa6b31d6b0b6c1602d0f59e4cb89d4a262b12f6e7` (as of 2026-07-01) |
| FFmpeg (Windows x64) | `https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip` | floating (master/latest) | GPL (BtbN gpl build) — verify | Managed bin dir | on-demand | **MUTABLE** (master/latest) | not hashed (other platform) |
| yt-dlp (Windows) | `https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp.exe` | floating (`/latest/`) | Unlicense — verify | Managed bin dir | on-demand | **MUTABLE** (`/latest/`) | not hashed (other platform) |
| Deno (Windows x64) | `https://github.com/denoland/deno/releases/latest/download/deno-x86_64-pc-windows-msvc.zip` | floating (`/latest/`) | MIT — verify | Managed bin dir | on-demand | **MUTABLE** (`/latest/`) | not hashed (other platform) |
| FFmpeg (Linux x64) | `https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz` | floating (master/latest) | GPL — verify | Managed bin dir | on-demand | **MUTABLE** (master/latest) | not hashed (other platform) |
| FFmpeg (Linux ARM64) | `https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linuxarm64-gpl.tar.xz` | floating (master/latest) | GPL — verify | Managed bin dir | on-demand | **MUTABLE** (master/latest) | not hashed (other platform) |
| yt-dlp (Linux x64) | `https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_linux` | floating (`/latest/`) | Unlicense — verify | Managed bin dir | on-demand | **MUTABLE** (`/latest/`) | not hashed (other platform) |
| yt-dlp (Linux ARM64) | `https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_linux_aarch64` | floating (`/latest/`) | Unlicense — verify | Managed bin dir | on-demand | **MUTABLE** (`/latest/`) | not hashed (other platform) |
| Deno (Linux x64) | `https://github.com/denoland/deno/releases/latest/download/deno-x86_64-unknown-linux-gnu.zip` | floating (`/latest/`) | MIT — verify | Managed bin dir | on-demand | **MUTABLE** (`/latest/`) | not hashed (other platform) |
| Deno (Linux ARM64) | `https://github.com/denoland/deno/releases/latest/download/deno-aarch64-unknown-linux-gnu.zip` | floating (`/latest/`) | MIT — verify | Managed bin dir | on-demand | **MUTABLE** (`/latest/`) | not hashed (other platform) |
| Standalone Python (macOS ARM64) | `https://github.com/astral-sh/python-build-standalone/releases/download/20250317/cpython-3.11.11+20250317-aarch64-apple-darwin-install_only.tar.gz` | 3.11.11 (build tag 20250317) | PSF / python-build-standalone (see repo) — verify | Managed python dir | on-demand (used for `pip install --target`) | pinned (version + build tag) | not hashed (>200 MB extracted; ~30 MB+ archive, but skipped per size/scope rule) |
| Standalone Python (Windows x64) | `.../20250317/cpython-3.11.11+20250317-x86_64-pc-windows-msvc-install_only.tar.gz` | 3.11.11 / 20250317 | PSF / pbs — verify | Managed python dir | on-demand | pinned | not hashed (other platform) |
| Standalone Python (Linux x64) | `.../20250317/cpython-3.11.11+20250317-x86_64-unknown-linux-gnu-install_only.tar.gz` | 3.11.11 / 20250317 | PSF / pbs — verify | Managed python dir | on-demand | pinned | not hashed (other platform) |
| Standalone Python (Linux ARM64) | `.../20250317/cpython-3.11.11+20250317-aarch64-unknown-linux-gnu-install_only.tar.gz` | 3.11.11 / 20250317 | PSF / pbs — verify | Managed python dir | on-demand | pinned | not hashed (other platform) |

**Note on `_CHECKSUMS`:** `core/dependency_manager.py` defines a `_CHECKSUMS` dict mapping every binary URL above to `None` (verification disabled). Per task constraints, this doc does NOT populate it. The hashes above are recorded here for reference only; they are not wired into any verification.

### Direct-download model / data files (managed by `core/analysis/*`)

| Name | Acquisition (URL) | Version | License | Install path | Bundled / on-demand / build-time | Mutable? | SHA-256 |
|---|---|---|---|---|---|---|---|
| YOLO26 nano weights (`yolo26n.pt`) | `https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt` | v8.4.0 (release tag pinned; default `n` size) | AGPL-3.0 (Ultralytics) — verify | `settings.model_cache_dir` | on-demand (object_detect) | pinned tag (filename varies by requested size: `yolo26{n,s,m,l,x}.pt`) | `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef` |
| YOLOE open-vocab weights (`yoloe-26s.pt`) | `https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26s.pt` (URL template in `detection.py`) | referenced as `_YOLOE_MODEL_NAME` | AGPL-3.0 (Ultralytics) — verify | `settings.model_cache_dir` | on-demand (custom-class detection) | pinned tag | **unreachable — HTTP 404 at the code's `v8.4.0/{name}` URL** (this filename is not an asset of the `v8.4.0` release; ultralytics likely resolves it from a different release at runtime) |
| EAST text-detection model (`frozen_east_text_detection.pb`) | `https://github.com/oyyd/frozen_east_text_detection.pb/raw/master/frozen_east_text_detection.pb` | floating (master branch) | verify (third-party mirror of OpenCV EAST model) | `<model cache>/...` (text detection) | on-demand | **MUTABLE** (master branch) | `9b486f3c3eee77b4c8cc91a83892c37026cca7d29b79bf3b93772ccd2db58454` |
| ImageNet class labels (`imagenet_classes.txt`) | `https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt` | floating (master branch) | BSD-3-Clause (pytorch/hub) — verify | `<cache_dir>/imagenet_classes.txt` | on-demand (image_classify; falls back to numeric labels if download fails) | **MUTABLE** (master branch) | `1f386e0d1cb6e28b9c2dac651c3dea6801e98ad1b41a14ce6bb1a093d72069f5` |
| MediaPipe FaceLandmarker (`face_landmarker.task`) | `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task` | float16 v1 (versioned path) | Apache-2.0 (MediaPipe) — verify | `<model_cache_dir>/mediapipe/face_landmarker.task` | on-demand (gaze_detect) | pinned (versioned GCS path) | not hashed (Google GCS — versioned but not re-hashed this pass; small file, hashable next audit if needed) |

### HuggingFace model repos (multi-GB — not hashed; repo id + revision status recorded)

| Name | Acquisition (HF repo) | Revision | License | Install path | Bundled / on-demand / build-time | Mutable? |
|---|---|---|---|---|---|---|
| Local VLM (default, Apple Silicon) | `mlx-community/Qwen3-VL-4B-Instruct-4bit` | none (resolves to repo `main`) | verify (Qwen license + mlx-community quant) | HF cache | on-demand (describe_local) | **MUTABLE** (no revision pin) |
| Local VLM (non-Apple-Silicon fallback) | `vikhyatk/moondream2` | **pinned** `2025-06-21` (`MOONDREAM_REVISION`) | verify (Moondream) | HF cache | on-demand (describe_local_cpu) | pinned |
| Shot classifier | `google/siglip2-base-patch16-224` | none (resolves to repo `main`) | verify (SigLIP2 / Google) | HF cache | on-demand (shot_classify) | **MUTABLE** (no revision pin) |
| Embeddings model | `facebook/dinov2-base` | none (resolves to repo `main`) | Apache-2.0 (DINOv2) — verify | HF cache | on-demand (embeddings) | **MUTABLE** (no revision pin) |
| Face detection/recognition pack | InsightFace `buffalo_l` (auto-downloaded by `insightface` lib) | none (library-managed) | verify (InsightFace models — non-commercial terms on some packs) | `<cache_dir>/insightface/` | on-demand (face_detect) | **MUTABLE** (library-managed, no pin) |

### Bundled Python wheel (vendored in-repo)

| Name | Acquisition | Version | License | Install path | Bundled / on-demand / build-time | Mutable? | SHA-256 |
|---|---|---|---|---|---|---|---|
| LiteLLM (vendored wheel) | `./vendor/wheels/litellm-1.82.6-py3-none-any.whl` (checked into repo; referenced by `requirements-core.txt` and `requirements.txt`) | 1.82.6 (pinned) | MIT (LiteLLM) — verify | site-packages | bundled (in core deps) | pinned | not hashed (in-repo file; `resolved by pip` from local path) |

### On-demand pip packages (`core/package_manifest.json` — `resolved by pip`)

All specs below are version-**range** ("floating within a range"), not fully pinned, except where noted. SHA-256 not applicable — `resolved by pip`. License column: `verify` unless a well-known permissive license is noted.

| Package | pip specifier | Version | License | Bundled / on-demand | Mutable? |
|---|---|---|---|---|---|
| torch | `torch>=2.4,<2.7` | range | BSD-3-Clause — verify | on-demand | range |
| torchaudio | `torchaudio>=2.4,<2.7` | range | BSD-2-Clause — verify | on-demand | range |
| transformers | `transformers>=4.50,<5` | range | Apache-2.0 — verify | on-demand | range |
| huggingface_hub | `huggingface-hub>=0.34.0,<1.0` | range | Apache-2.0 — verify | on-demand | range |
| tokenizers | `tokenizers>=0.21,<0.24` | range | Apache-2.0 — verify | on-demand | range |
| torchvision | `torchvision>=0.19,<0.22` | range | BSD-3-Clause — verify | on-demand | range |
| einops | `einops>=0.7.0,<1.0` | range | MIT — verify | on-demand | range |
| sentencepiece | `sentencepiece>=0.2.0,<1.0` | range | Apache-2.0 — verify | on-demand | range |
| protobuf | `protobuf>=4.25,<6` | range | BSD-3-Clause — verify | on-demand | range |
| ultralytics | `ultralytics>=8.4.0,<9` | range | AGPL-3.0 — verify | on-demand | range |
| faster_whisper | `faster-whisper>=1.0.0,<2` | range | MIT — verify | on-demand | range |
| lightning_whisper_mlx | `lightning-whisper-mlx>=0.0.10,<1.0` | range | verify | on-demand (macOS ARM) | range |
| mlx_vlm | `mlx-vlm>=0.1.0,<1.0` | range | verify | on-demand (macOS ARM) | range |
| paddleocr | `paddleocr>=3.0.0,<4` | range | Apache-2.0 — verify | on-demand | range |
| librosa | `librosa>=0.10.0,<1.0` | range | ISC — verify | on-demand | range |
| insightface | `insightface>=0.7.3,<1.0` | range | MIT (code; model weights differ) — verify | on-demand | range |
| onnxruntime | `onnxruntime>=1.16.0,<2.0` | range | MIT — verify | on-demand | range |
| tiktoken | `tiktoken>=0.7.0,<0.8` | range | MIT — verify | on-demand | range |
| demucs_infer | `demucs-infer>=4.1.0,<5` | range | MIT — verify | on-demand | range |
| ctc_forced_aligner | `ctc-forced-aligner @ git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git` | git-HEAD (default branch) | verify | on-demand (word_alignment) | **MUTABLE** (git-HEAD, no commit pin — T6) |

### Bare-name on-demand pip packages (referenced in `feature_registry.py`, NOT in the manifest)

| Package | pip specifier | Version | License | Bundled / on-demand | Mutable? |
|---|---|---|---|---|---|
| mediapipe | `mediapipe` (bare name — no manifest entry, `get_pip_specifier` falls back to bare name) | unconstrained | Apache-2.0 — verify | on-demand (gaze_detect) | **MUTABLE** (bare-name, unconstrained) |

### Build-time bundled binaries (baked into distributed builds by CI)

| Name | Acquisition (URL) | Version | License | Install path | Bundled / on-demand / build-time | Mutable? | SHA-256 |
|---|---|---|---|---|---|---|---|
| FFmpeg (macOS build) | `https://www.osxexperts.net/ffmpeg7arm.zip` (`.github/workflows/build-macos.yml`) | FFmpeg 7.x (osxexperts) | GPL/LGPL — verify | `packaging/runtime/ffmpeg/macos/`, bundled into `.app` | build-time | pinned URL (unversioned filename) | `563111a239fe70d2e5c84a5382204a7d0bf0a332385a92a44baff36d313e27f2` (same file as runtime macOS FFmpeg) |
| FFprobe (macOS build) | `https://www.osxexperts.net/ffprobe7arm.zip` | FFmpeg 7.x (osxexperts) | GPL/LGPL — verify | `packaging/runtime/ffmpeg/macos/`, bundled into `.app` | build-time | pinned URL (unversioned filename) | `e5ae34ee2f0b3594892a695fd733646904bbc7eb40af3b359ed91538ddcb5513` |
| Sparkle (macOS auto-updater) | `https://github.com/sparkle-project/Sparkle/releases/download/2.8.1/Sparkle-2.8.1.tar.xz` | 2.8.1 (`SPARKLE_VERSION`) | MIT (Sparkle) — verify | `packaging/runtime/sparkle/macos/`, bundled into `.app` | build-time | pinned (version) | not hashed (build-time; version-pinned tag, not this-platform runtime download) |
| mpv (Homebrew, macOS) | `brew install mpv` (`.github/workflows/build-macos.yml`) | Homebrew formula (floating) | GPL/LGPL (mpv) — verify | Homebrew, `libmpv*.dylib` bundled into `.app` | build-time | **MUTABLE** (Homebrew formula — whatever version brew resolves) | n/a (Homebrew) |
| create-dmg (Homebrew, macOS) | `brew install create-dmg` | Homebrew formula (floating) | MIT — verify | Homebrew (build tool only, not bundled) | build-time | **MUTABLE** (Homebrew formula) | n/a (Homebrew) |
| Python ML wheels (macOS build) | `pip install torch/torchaudio/torchvision/transformers/huggingface-hub/sentencepiece/protobuf/einops/ultralytics/insightface/onnxruntime/librosa/demucs-infer/mlx-vlm/paddleocr/paddlepaddle/rapidfuzz` (range specs, plus `lightning-whisper-mlx --no-deps`) | ranges | see per-package rows above | site-packages, bundled into `.app` via PyInstaller | build-time | range (`resolved by pip` at build time) | not hashed (`resolved by pip`) |
| PyInstaller + cryptography (build tools) | `pip install pyinstaller cryptography` | unconstrained (bare) | GPL-with-exception / Apache+BSD — verify | build environment only (not bundled) | build-time | **MUTABLE** (bare-name, unconstrained) | not hashed (`resolved by pip`) |
| mpv (Windows build) | `https://github.com/shinchiro/mpv-winbuild-cmake/releases/download/20260524/mpv-dev-x86_64-20260524-git-9e06c32.7z` (`packaging/windows/runtime-manifest.json`) | 20260524 git-9e06c32 (pinned asset) | GPL/LGPL (mpv) — verify | bundled into Windows build | build-time | pinned (asset + sha256 in manifest) | `3dd0754d2a95ba2f09b030d9a21d437c901e2103c0bdaf3e988763461ebc5447` (from manifest — not re-hashed; other platform) |
| FFmpeg (Windows build) | `https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-05-24-13-16/ffmpeg-n8.1.1-8-gb21e00eda5-win64-gpl-8.1.zip` | n8.1.1-8-gb21e00eda5 (pinned asset) | GPL — verify | bundled into Windows build | build-time | pinned (asset + sha256 in manifest) | `fe99b6a7e667000943d5d60fcbe156d988726d0a1c88ff4204e2665827d6f643` (from manifest — not re-hashed; other platform) |
| WinSparkle (Windows auto-updater) | `https://github.com/vslavik/winsparkle/releases/download/v0.9.2/WinSparkle-0.9.2.zip` | 0.9.2 (pinned asset) | MIT — verify | bundled into Windows build | build-time | pinned (asset + sha256 in manifest) | `2a5facf5a22056edf7afc4d0a4bf05620b7aa7cc1619c3078b281e374cdb9996` (from manifest — not re-hashed; other platform) |

---

## Counts

- **Total artifacts inventoried:** 55
  - Direct-download binaries (dependency_manager): 17 (ffmpeg/ffprobe/yt-dlp/deno per platform + 4 standalone-Python variants)
  - Direct-download model/data files: 5 (YOLO26, YOLOE, EAST, ImageNet labels, MediaPipe)
  - HuggingFace model repos: 5
  - Vendored wheel: 1 (LiteLLM)
  - On-demand pip packages (manifest): 21
  - Bare-name pip package (not in manifest): 1 (mediapipe)
  - Build-time bundled/tool artifacts: 10
- **Mutable references (rows flagged MUTABLE): 22**
  - `/latest/` GitHub redirects: 8 — yt-dlp (macOS, Windows, Linux x64, Linux ARM64) + Deno (macOS, Windows, Linux x64, Linux ARM64)
  - FFmpeg master/latest (runtime): 3 — Windows, Linux x64, Linux ARM64
  - master-branch raw files: 2 — EAST model, ImageNet labels
  - git-HEAD pip: 1 — ctc-forced-aligner
  - bare-name / unconstrained pip: 3 — mediapipe, pyinstaller, cryptography
  - unpinned HuggingFace repos: 4 — Qwen3-VL, SigLIP2, DINOv2, InsightFace buffalo_l
  - Homebrew floating formulae: 2 — mpv, create-dmg

  (8 + 3 + 2 + 1 + 3 + 4 + 2 = 22. The per-row **Mutable?** column is authoritative.)

- **Download failures:** 1
  - `yoloe-26s.pt` — HTTP 404 at the code's `https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26s.pt` URL. Recorded, not guessed.
