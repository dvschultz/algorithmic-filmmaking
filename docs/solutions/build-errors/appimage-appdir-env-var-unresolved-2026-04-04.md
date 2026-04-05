---
title: AppImage build fails — APPDIR environment variable unresolved
date: 2026-04-04
category: build-errors
module: packaging
problem_type: build_error
component: development_workflow
severity: medium
symptoms:
  - "Linux CI 'build-appimage' job fails on every push"
  - "appimagebuilder.recipe.errors.RecipeError: Unable to resolve environment variable: APPDIR"
  - "Test job passes; only AppImage packaging step fails"
root_cause: config_error
resolution_type: environment_setup
tags:
  - appimage
  - linux
  - ci
  - github-actions
  - appimage-builder
  - environment-variable
  - yaml-loader
---

# AppImage build fails — APPDIR environment variable unresolved

## Problem

The Linux CI `build-appimage` job failed on every push because `appimage-builder`'s YAML loader tried to resolve `${APPDIR}` as an environment variable at recipe parse time, but the build script never exported it.

## Symptoms

- Every push triggered a failing "Linux Build" workflow on GitHub Actions
- The `build-appimage` job produced:
  ```
  appimagebuilder.recipe.errors.RecipeError: Unable to resolve environment variable: APPDIR
  ```
- The test job within the same workflow passed — only the AppImage packaging step failed
- Windows CI was unaffected

## What Didn't Work

N/A — diagnosed correctly on first attempt by reading the traceback, then tracing the `${APPDIR}` references in the recipe back to the missing export.

## Solution

Added `export APPDIR="$PROJECT_ROOT/AppDir"` to `packaging/linux/build-appimage.sh` before the `appimage-builder` invocation.

**Before** (`packaging/linux/build-appimage.sh`, line 73):

```bash
# Build AppImage using appimage-builder
echo "Building AppImage with appimage-builder..."
export VERSION="$VERSION"
appimage-builder --recipe "$SCRIPT_DIR/AppImageBuilder.yml" --skip-test
```

**After**:

```bash
# Build AppImage using appimage-builder
echo "Building AppImage with appimage-builder..."
export VERSION="$VERSION"
export APPDIR="$PROJECT_ROOT/AppDir"
appimage-builder --recipe "$SCRIPT_DIR/AppImageBuilder.yml" --skip-test
```

## Why This Works

`appimage-builder` uses a custom YAML loader (`appimagebuilder.recipe.loader`) that eagerly resolves **all** `${...}` patterns as environment variable lookups at parse time — before any build steps execute. The recipe file `packaging/linux/AppImageBuilder.yml` references `${APPDIR}` in its `script` section (pip install target) and `runtime.env` section (PYTHONPATH, QT_PLUGIN_PATH).

The build script already exported `VERSION` (which the recipe also references via `!ENV ${VERSION:-0.1.0}`), but `APPDIR` was missed. By exporting `APPDIR` with the correct path before invoking `appimage-builder`, the YAML loader can resolve the variable successfully during parsing.

This is a non-obvious behavior: `${APPDIR}` looks like it should be a runtime variable (resolved when the AppImage runs), but `appimage-builder`'s loader resolves it at build time during YAML parsing.

## Prevention

- When adding or modifying `${...}` references in `AppImageBuilder.yml`, ensure every referenced variable is explicitly exported in `build-appimage.sh` before the `appimage-builder` call
- The `VERSION` export (line 75) serves as an existing example of the pattern — any new `${VAR}` in the recipe needs a matching `export VAR` in the build script
- Consider adding a comment block in the build script listing all variables the recipe expects:
  ```bash
  # Variables required by AppImageBuilder.yml (resolved at YAML parse time):
  #   VERSION  — build version string
  #   APPDIR   — path to AppDir directory
  ```

## Related Issues

- Complements: `docs/solutions/deployment/linux-pyside6-distribution-packaging.md` (same domain, different problem)
- Extends: `docs/linux-distribution-guide.md` (adds build-time env var guidance to existing runtime guidance)
- Commit: `3d949ab` on main
