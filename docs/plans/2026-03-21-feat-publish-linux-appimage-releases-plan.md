---
title: "feat: Publish Linux AppImage Releases"
type: feat
status: proposed
date: 2026-03-21
---

# feat: Publish Linux AppImage Releases

## Overview

Publish Linux AppImage builds to the same GitHub Releases flow used for macOS and Windows so Linux users can actually download a versioned release artifact from the public release page.

The repo already has Linux packaging and CI coverage. The missing work is release publication, verification, and documentation alignment.

## Problem Statement / Motivation

The current repository state is inconsistent:

- `README.md:26` tells users to download a Linux AppImage from GitHub Releases.
- `.github/workflows/linux-build.yml:70` builds an AppImage.
- `.github/workflows/linux-build.yml:102` uploads only a GitHub Actions artifact, not a release asset.
- `docs/releases.md:26` defines release success as only macOS and Windows assets being present.

That means Linux support exists in packaging terms, but not as a real downloadable release channel. Users following the README can land on a release page with no Linux asset, while maintainers have no documented release checklist for Linux.

## Proposed Solution

Extend the existing Linux workflow from CI-only packaging to first-class release publishing.

### Chosen Scope

Ship one Linux release artifact for now:

- `Scene_Ripper-<version>-x86_64.AppImage`

Keep the work narrowly focused on release delivery:

- build on tag push and `workflow_dispatch`
- upload the AppImage to the matching `vX.Y.Z` GitHub release
- verify the expected file exists at the exact upload path
- update maintainer docs and README so Linux release expectations match reality

### Explicitly Out of Scope

- Linux in-app auto-updates or appcast publishing
- additional Linux formats such as `.deb`, `.rpm`, or Flatpak
- signing the AppImage
- multi-architecture Linux builds beyond current `x86_64`
- broader Linux runtime-hardening work unless release publication exposes a concrete blocker

## Research Summary

### Internal References

- Linux release workflow exists but is CI-oriented:
  - `.github/workflows/linux-build.yml:11`
  - `.github/workflows/linux-build.yml:70`
  - `.github/workflows/linux-build.yml:102`
- Linux packaging already exists:
  - `packaging/linux/build-appimage.sh:18`
  - `packaging/linux/build-appimage.sh:73`
  - `packaging/linux/AppImageBuilder.yml:10`
  - `packaging/linux/AppImageBuilder.yml:63`
- Public docs already imply downloadable Linux releases:
  - `README.md:26`
  - `README.md:30`
- Maintainer release docs currently describe only macOS and Windows release assets:
  - `docs/releases.md:26`
  - `docs/releases.md:47`

### Institutional Learnings

- `docs/solutions/deployment/linux-pyside6-distribution-packaging.md`
  - Linux packaging should stay XDG-aware and distribution-specific runtime assumptions must be explicit.
  - Linux media/runtime validation matters because Qt and multimedia dependencies can fail silently.
- Existing desktop release work in `docs/releases.md` reinforces that artifact existence alone is not enough; release workflows need runtime-aware verification.

### External Research

Not needed for this plan. The repo already has an AppImage packaging path and the problem is integration with the existing GitHub release workflow, not choosing a new packaging technology.

## SpecFlow Analysis

This feature introduces a few concrete maintainer and user flows that the implementation should make explicit.

### User Flows

1. A Linux user opens `vX.Y.Z` on GitHub Releases and expects an AppImage download beside the macOS and Windows assets.
2. A maintainer pushes tag `vX.Y.Z` and expects the Linux build to publish to that same release automatically.
3. A maintainer manually dispatches the Linux workflow with version `X.Y.Z` and expects it to upload to release tag `vX.Y.Z`.
4. A maintainer inspects the release page and can tell whether Linux release publication succeeded without opening workflow artifacts.

### Failure Flows

1. The AppImage build succeeds but the release upload fails.
2. The workflow uploads an asset with the wrong filename or wrong version.
3. The AppImage exists as a workflow artifact but never becomes a public release asset.
4. The Linux build is marked green despite a broken or missing AppImage because the workflow is currently tolerant of build failure.

### Gaps the Plan Must Close

- No release upload step exists for Linux.
- No Linux asset is listed in the maintainer release checklist.
- The Linux workflow currently uses `continue-on-error: true` for the build step, which is too weak for a publishing workflow.
- There is no explicit Linux release verification step comparable to macOS/Windows artifact checks.
- The Linux workflow does not currently declare the GitHub release permissions/auth needed for `gh release` commands.
- The plan must preserve the current `push: main` and `pull_request` Linux CI coverage while adding release behavior.

## Technical Approach

### Architecture

Keep Linux aligned with the existing desktop release pattern:

1. `linux-build.yml` remains the Linux CI and packaging entry point.
2. Existing `push: main` and `pull_request` Linux validation remains intact.
3. Tag pushes `v*` and manual `workflow_dispatch` become supported release triggers.
4. The workflow builds exactly one versioned AppImage at repo root: `./Scene_Ripper-<version>-x86_64.AppImage`.
5. The workflow uses the same GitHub release pattern as macOS and Windows:
   - `permissions: contents: write`
   - `GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}`
   - `gh release view || gh release create`
   - `gh release upload --clobber`
6. The workflow fails if the expected upload path is missing, but still uploads build logs on failure.

### Proposed Workflow Shape

#### Phase 1: Promote Linux Workflow to Release Workflow

**Files**

- `.github/workflows/linux-build.yml`

**Work**

- Keep current branch/PR Linux CI triggers and test job behavior.
- Add tag trigger parity with macOS and Windows release workflows.
- Normalize version resolution so tag pushes and manual dispatch both produce the same `X.Y.Z` value.
- Add GitHub release permissions/auth wiring for Linux:
  - `permissions: contents: write`
  - `GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}`
- Add a release creation/upload step using the same GitHub CLI pattern used elsewhere in the repo:
  - `gh release view "${RELEASE_TAG}" || gh release create "${RELEASE_TAG}" ...`
  - `gh release upload "${RELEASE_TAG}" "./Scene_Ripper-${VERSION}-x86_64.AppImage" --clobber`
- Replace `continue-on-error: true` with explicit control flow that still preserves failure diagnostics:
  - capture build failure status
  - upload logs/artifacts with `if: always()`
  - fail the job after diagnostic uploads if the AppImage is missing or the build failed
- Add a hard failure if `./Scene_Ripper-<version>-x86_64.AppImage` does not exist.

**Success Criteria**

- A tag push `vX.Y.Z` publishes a Linux AppImage to release `vX.Y.Z`.
- A manual dispatch with `version=X.Y.Z` publishes to release `vX.Y.Z`.
- The workflow fails red if the AppImage is not built or not uploaded.

#### Phase 2: Minimal Release Verification

**Files**

- `.github/workflows/linux-build.yml`
- `packaging/linux/build-appimage.sh`

**Work**

- Add explicit filename verification so the build output matches release naming.
- Ensure the release job uploads logs and any partial artifacts with `if: always()` so diagnostics survive a failed build.
- Keep any runnable smoke check optional unless release publication exposes a concrete launch blocker.

**Success Criteria**

- The Linux release workflow verifies artifact presence at the exact upload path.
- Maintainers still get actionable failure logs when Linux packaging breaks.

#### Phase 3: Align Documentation and Release Checklist

**Files**

- `README.md`
- `docs/releases.md`

**Work**

- Update release docs so Linux is included in the expected assets list.
- Add Linux workflow to the documented release workflow files.
- Add Linux checks to the post-tag release checklist.
- Ensure README download instructions match the actual published asset name and release behavior.

**Success Criteria**

- User-facing Linux install docs match the real release page.
- Maintainer docs treat Linux as a first-class published artifact rather than CI-only output.

## Alternative Approaches Considered

### Keep Linux as CI-only

Rejected because it preserves the current mismatch between public docs and actual release availability.

### Publish Linux workflow artifacts only

Rejected because workflow artifacts are not the same as public versioned release downloads and are poor as an end-user distribution surface.

### Add Flatpak or `.deb` first

Rejected for now because the repo already has an AppImage path. The shortest path to a trustworthy Linux release is to publish the artifact that already exists.

## Acceptance Criteria

### Functional Requirements

- [ ] Tag push `vX.Y.Z` triggers Linux AppImage publishing to GitHub Releases.
- [ ] Manual dispatch with `version=X.Y.Z` uploads to release `vX.Y.Z`.
- [ ] GitHub release `vX.Y.Z` shows `Scene_Ripper-<version>-x86_64.AppImage`.
- [ ] Linux workflow fails if the AppImage output is missing or incorrectly named.
- [ ] Linux workflow preserves existing `push: main` and `pull_request` CI behavior.
- [ ] Linux workflow has the permissions/auth needed to create or update a release.

### Non-Functional Requirements

- [ ] Linux release documentation is consistent across README and maintainer docs.
- [ ] Failure logs remain available after a broken Linux release build.

### Quality Gates

- [ ] Linux build passes on GitHub Actions with release upload enabled.
- [ ] Maintainer checklist in `docs/releases.md` includes Linux verification.

## Success Metrics

- Linux assets appear on versioned GitHub releases without manual post-processing.
- README Linux download instructions work against a real release page.
- Maintainers can verify all desktop release assets from `gh release view vX.Y.Z`.

## Dependencies & Risks

### Dependencies

- GitHub Actions runner support for the existing AppImage build path
- `gh` CLI access in the Linux workflow
- `GITHUB_TOKEN`-backed release permissions in the Linux workflow
- Stable artifact naming from `packaging/linux/build-appimage.sh`

### Risks

- AppImage builds may be more fragile than macOS/Windows packaging and will need stronger CI feedback.
- The current AppImage runtime assumptions may not be sufficient for all distros even if the release publishes successfully.
- Linux release publication could expose gaps that were previously hidden because artifacts were not publicly consumed.

## Documentation Plan

- Update `docs/releases.md` to include Linux in the release process, expected assets, workflow list, and post-release checklist.
- Update `README.md` only as needed to keep asset names and release expectations accurate.
- Do not broaden user-facing Linux promises beyond the single `x86_64` AppImage until additional formats or architectures actually ship.

## References & Research

### Internal References

- `.github/workflows/linux-build.yml:11`
- `.github/workflows/linux-build.yml:70`
- `.github/workflows/linux-build.yml:95`
- `.github/workflows/linux-build.yml:102`
- `packaging/linux/build-appimage.sh:20`
- `packaging/linux/build-appimage.sh:73`
- `packaging/linux/AppImageBuilder.yml:10`
- `packaging/linux/AppImageBuilder.yml:63`
- `README.md:26`
- `README.md:30`
- `docs/releases.md:26`
- `docs/releases.md:47`

### Related Work

- `docs/solutions/deployment/linux-pyside6-distribution-packaging.md`
- `docs/plans/archive/2026-01-24-feat-linux-app-distribution-plan.md`
