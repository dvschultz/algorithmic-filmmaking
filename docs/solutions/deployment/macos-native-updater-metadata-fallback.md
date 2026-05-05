---
title: macOS native updater falls back because metadata is missing
date: 2026-05-05
category: deployment
module: packaging
problem_type: build_configuration
component: macos_updater
severity: medium
symptoms:
  - "Native updater metadata is not configured in this build."
  - "Falling back to browser-based update checks."
root_cause: missing_bundle_metadata
resolution_type: packaging_configuration
tags:
  - macos
  - sparkle
  - updater
  - pyinstaller
  - release
---

# macOS Native Updater Falls Back Because Metadata Is Missing

## Problem

macOS builds can show this update-check message:

```text
Native updater metadata is not configured in this build.

Falling back to browser-based update checks.
```

## Root Cause

The macOS updater adapter requires both `SUFeedURL` and `SUPublicEDKey` in the app bundle `Info.plist`. The local macOS build path previously only embedded those values when the Sparkle-specific environment variables were set directly, while the release docs and Windows build path also use shared `UPDATE_*` updater variable names.

## Solution

Resolve macOS Sparkle metadata through `packaging/build_support.py`, accepting both:

- `SPARKLE_FEED_URL` / `SPARKLE_PUBLIC_ED_KEY`
- `UPDATE_FEED_URL` / `UPDATE_PUBLIC_ED_KEY`

The local `packaging/macos/build.sh` path also derives `SPARKLE_PUBLIC_ED_KEY` from `UPDATE_PRIVATE_ED_KEY` when needed and prints whether native updater metadata will be embedded.

## Verification

Run:

```bash
python -m pytest tests/test_build_support.py::test_resolve_macos_sparkle_metadata_accepts_canonical_env_names tests/test_build_support.py::test_resolve_macos_sparkle_metadata_accepts_release_env_aliases tests/test_macos_updater.py::test_get_status_requires_bundle_metadata_for_native_updates tests/test_macos_updater.py::test_get_status_reports_native_check_when_cli_and_metadata_exist -v
ruff check packaging/build_support.py tests/test_build_support.py core/macos_updater.py tests/test_macos_updater.py
bash -n packaging/macos/build.sh
```
