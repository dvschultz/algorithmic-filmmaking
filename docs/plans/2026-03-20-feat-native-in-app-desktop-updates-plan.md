---
title: "feat: Native In-App Desktop Updates"
type: feat
status: active
date: 2026-03-20
---

# feat: Native In-App Desktop Updates

## Overview

Deliver true in-app updates for installed macOS and Windows builds so users do not need to manually re-download every release from GitHub. The existing release-page update check becomes the fallback path, while production builds use native platform updaters that can discover, verify, download, and apply signed releases.

The recommended architecture is:

- macOS: Sparkle-backed updates from a signed appcast feed
- Windows: WinSparkle-backed updates from the same signed appcast model
- Shared Python-side update domain for settings, version policy, diagnostics, and fallback behavior
- GitHub Actions extended to publish signed update metadata in addition to the existing GitHub Releases artifacts

## Problem Statement

Scene Ripper currently has only a lightweight release check:

- `core/update_checker.py` checks GitHub Releases directly and compares tags
- `ui/main_window.py` shows a banner or dialog and opens the release page in the browser
- `ui/settings_dialog.py` exposes a launch-time "Check for updates" toggle

That is useful, but it is still manual distribution. Users must:

1. Leave the app
2. Open GitHub Releases
3. Download a DMG or installer manually
4. Replace or reinstall the app themselves

This creates friction for non-technical users and blocks smoother delivery of fixes. It also leaves the app without platform-native concepts such as:

- signed update feeds
- phased rollouts
- critical updates
- skip/remind-later state
- installer handoff and restart behavior

## Proposed Solution

Adopt a staged native-updater architecture instead of writing a custom downloader/installer from scratch.

### Chosen Approach

- Keep the existing Python release checker as a compatibility fallback.
- Add a shared update domain in Python for settings, logging, version normalization, and updater capability detection.
- Integrate Sparkle on macOS because it is the standard update framework for macOS apps, supports DMG/ZIP/pkg delivery, EdDSA-signed appcasts, phased rollouts, channels, and manual/background checks.
- Integrate WinSparkle on Windows because it uses the same appcast model and EdDSA signatures as Sparkle and is designed for Windows desktop apps.
- Publish update metadata to a stable hosted appcast URL, while continuing to ship the actual DMG and installer binaries via GitHub Releases.

### Important Product Decision

This should ship in two milestones:

1. **Guided native updates**
   - Native updater frameworks are embedded and configured.
   - The app can discover updates and launch the platform updater UI.
   - macOS and Windows can still fall back to browser/release-page download when trust or installation prerequisites are not met.

2. **Full in-place updates**
   - macOS uses Sparkle to install signed updates directly.
   - Windows uses WinSparkle to hand off to a signed installer and restart cleanly.

This is the safer path because Windows auto-update quality depends heavily on installer signing and restart behavior, while macOS update quality depends on proper signing, notarization, appcast hosting, and bundle metadata.

## Research Summary

### Internal References

- Existing lightweight release check:
  - `core/update_checker.py:17`
  - `core/update_checker.py:87`
- Existing Help menu and manual update action:
  - `ui/main_window.py:1087`
  - `ui/main_window.py:1122`
- Existing launch-time update preference:
  - `ui/settings_dialog.py:1126`
- Existing bundled app version resolution:
  - `core/app_version.py:19`
- Existing release process and artifact naming:
  - `docs/releases.md:5`
  - `docs/releases.md:26`
- Existing macOS release pipeline:
  - `.github/workflows/build-macos.yml:35`
  - `.github/workflows/build-macos.yml:116`
  - `.github/workflows/build-macos.yml:165`
  - `.github/workflows/build-macos.yml:211`
- Existing Windows release pipeline:
  - `.github/workflows/build-windows.yml:35`
  - `.github/workflows/build-windows.yml:106`
  - `.github/workflows/build-windows.yml:131`
- Prior packaging plans already assumed future update checking:
  - `docs/plans/2026-02-21-feat-macos-app-bundle-distribution-plan.md:252`
  - `docs/plans/2026-02-24-feat-windows-version-plan.md:13`

### Institutional Learnings

- Platform-specific distribution behavior should live behind explicit platform abstractions rather than leaking into generic app logic.
  - `docs/solutions/deployment/linux-pyside6-distribution-packaging.md`
- CI should validate distribution artifacts directly, not only unit tests.
  - reinforced by the recent release workflow hardening in `docs/releases.md`

### External Research

- Sparkle documentation recommends Developer ID signed distribution, `SUFeedURL`, proper `CFBundleVersion`, and appcast generation with signatures:
  - https://sparkle-project.org/documentation/
  - https://sparkle-project.org/documentation/publishing/
- Sparkle supports DMG, ZIP, tar, Apple Archive, and installer package updates, and supports informational updates, phased rollouts, and channels:
  - https://sparkle-project.org/documentation/publishing/
- Sparkle notes that updates may not proceed if the app is running from a read-only mount or under app translocation:
  - https://sparkle-project.org/documentation/
- WinSparkle uses the same appcast format and EdDSA signing model as Sparkle:
  - https://github.com/vslavik/winsparkle
- GitHub Releases API is still useful for fallback/manual checks, but it is not a full native-update feed:
  - https://docs.github.com/en/rest/releases/releases
- Apple notarization remains a hard requirement for a credible macOS update path:
  - https://developer.apple.com/documentation/security/notarizing_macos_software_before_distribution
  - https://developer.apple.com/documentation/security/customizing-the-notarization-workflow

## SpecFlow Analysis

The feature is not just "download latest build." It introduces several distinct user flows that the implementation must make explicit.

### Core User Flows

1. Installed user launches the app and a background check finds a newer version.
2. Installed user manually chooses `Help > Check for Updates...`.
3. User is already on latest version and expects a clear confirmation.
4. User is on a skipped version and wants to re-enable update prompts.
5. User is offline or GitHub/appcast hosting is temporarily unavailable.
6. User is running the app from a DMG or a translocated macOS path.
7. Windows user has an update downloaded and needs a clean installer handoff and restart.
8. Critical update should bypass "remind me later" behavior.

### Gaps The Plan Must Close

- There is no canonical appcast feed yet.
- There is no machine-readable release metadata beyond GitHub Releases JSON.
- There is no skipped-version persistence model.
- There is no rollout policy for critical vs normal vs phased updates.
- There is no platform bridge from Python to Sparkle/WinSparkle.
- There is no Windows signing policy strong enough to safely promise automatic installer execution.

## Technical Approach

### Architecture

Use a split architecture:

1. **Shared Python update domain**
   - owns settings, version parsing, compatibility checks, fallback behavior, logging, and app UI state
2. **Native platform updater adapters**
   - macOS adapter wraps Sparkle
   - Windows adapter wraps WinSparkle
3. **Release metadata publisher**
   - CI generates appcast XML, signatures, release notes, and stable URLs
4. **GitHub Releases as binary origin**
   - DMG and installer remain downloadable release assets
   - appcast points at those immutable asset URLs

### Proposed Module Layout

```text
core/
  app_version.py
  update_checker.py              # fallback/manual GitHub release checks
  update_models.py               # UpdateInfo, UpdateChannel, UpdateState
  update_settings.py             # skipped version, last prompt, channel prefs
  update_service.py              # shared app-facing update coordinator
  update_fallback.py             # browser / GitHub-release fallback path

platform/
  macos_updater.py               # Sparkle bridge / subprocess wrapper
  windows_updater.py             # WinSparkle bridge / ctypes wrapper

packaging/release/
  generate_appcast.py            # appcast XML + release notes generation
  publish_update_feed.py         # deploy signed appcast to hosted URL
  verify_update_feed.py          # CI validation
```

### Versioning Model

The app needs both a human release version and a machine-safe updater version.

- `CFBundleShortVersionString`: keep semantic release version such as `0.2.0`
- `CFBundleVersion`: add a strictly monotonic machine-readable build version
- Windows installer metadata and WinSparkle appcast item must use the same canonical machine-readable version value
- `core/app_version.py` should remain the single place the Python app reads the current version, but it will need to expose both display and machine versions

Without this, updater frameworks can compare versions inconsistently across platforms.

## Implementation Phases

### Phase 1: Update Metadata Foundation

Create the shared metadata model before embedding any native updater framework.

**Files**

- `core/app_version.py`
- `core/update_models.py`
- `core/update_service.py`
- `core/update_checker.py`
- `ui/settings_dialog.py`
- `ui/main_window.py`
- `packaging/macos/scene_ripper.spec`
- `packaging/windows/scene_ripper.spec`

**Work**

- Add a canonical update model:
  - `UpdateInfo`
  - `UpdateAvailability`
  - `UpdateChannel`
  - `UpdateCapability`
- Extend version handling to expose:
  - display version
  - machine version
  - release channel
- Add persisted update settings:
  - `check_for_updates`
  - `automatically_download_updates`
  - `skipped_update_version`
  - `last_prompted_update_version`
  - `update_channel`
- Refactor `core/update_checker.py` so GitHub Releases becomes the fallback provider, not the only provider.
- Keep the existing manual menu action and banner behavior operational during the transition.

**Success Criteria**

- [x] The app can represent update state independently of any specific backend.
- [x] Version comparison is stable for semver display versions and monotonic build versions.
- [x] Settings can persist skipped-version state and update channel.
- [x] Current browser-based update flow still works as fallback.

### Phase 2: Appcast Generation and Release Publishing

Build the release metadata pipeline required by Sparkle and WinSparkle.

**Files**

- `packaging/release/generate_appcast.py`
- `packaging/release/publish_update_feed.py`
- `packaging/release/verify_update_feed.py`
- `.github/workflows/build-macos.yml`
- `.github/workflows/build-windows.yml`
- `docs/releases.md`

**Work**

- Generate a signed appcast feed after successful release builds.
- Publish release notes in a stable web location.
- Host the appcast and release notes at a stable URL, preferably GitHub Pages for this repo.
- Add EdDSA signing to update assets:
  - Sparkle signing for macOS archives
  - WinSparkle-compatible signing for Windows installer assets
- Keep GitHub Releases as the source of binary downloads referenced by the appcast.
- Add CI validation that:
  - the appcast is well-formed
  - the enclosure URLs exist
  - the signatures and lengths are present
  - the newest item matches the release version

**Required Secrets / Infrastructure**

- `SPARKLE_PRIVATE_KEY`
- `SPARKLE_PUBLIC_KEY`
- `WINSPARKLE_PRIVATE_KEY`
- `WINSPARKLE_PUBLIC_KEY`
- existing macOS signing and notarization secrets from `docs/releases.md`
- Windows code-signing certificate and password before unattended Windows install is enabled

**Success Criteria**

- [ ] Every tagged release publishes DMG and installer artifacts plus signed appcast metadata.
- [ ] The appcast is hosted at a stable public URL.
- [ ] CI fails if appcast generation or signature validation fails.
- [ ] Rollback is possible by publishing a newer appcast item or removing a bad one.

### Phase 3: macOS Sparkle Integration

Add native macOS update installation on top of the signed appcast feed.

**Files**

- `core/macos_updater.py`
- `ui/main_window.py`
- `packaging/macos/scene_ripper.spec`
- `.github/workflows/build-macos.yml`

**Work**

- Bundle Sparkle into the macOS app distribution output.
- Add required updater metadata to the macOS bundle:
  - `SUFeedURL`
  - `SUPublicEDKey`
  - automatic-check preferences as appropriate
- Expose `Check for Updates...` through the native updater path for macOS installed builds.
- Preserve the existing Python fallback if Sparkle is unavailable or misconfigured.
- Re-sign and re-notarize the final app after Sparkle helpers/frameworks are embedded.
- Detect unsupported install states:
  - app running from DMG
  - read-only mount
  - app translocation
- In unsupported states, show a targeted prompt like "Move Scene Ripper to Applications to enable updates" and fall back to informational/browser updates.

**Success Criteria**

- [ ] Installed notarized macOS builds can perform a manual Sparkle update check.
- [ ] Background checks surface native update UI for newer versions.
- [ ] App updates do not break bundle signing/notarization.
- [x] Running-from-DMG/translocated states degrade gracefully.

### Phase 4: Windows WinSparkle Integration

Add native Windows update checks and installer handoff.

**Files**

- `core/windows_updater.py`
- `ui/main_window.py`
- `packaging/windows/scene_ripper.spec`
- `.github/workflows/build-windows.yml`

**Work**

- Bundle `WinSparkle.dll` with the Windows app.
- Wrap the WinSparkle C API with `ctypes` or another thin bridge.
- Set:
  - appcast URL
  - app identity / registry path
  - EdDSA public key
  - shutdown/restart callback for installer handoff
- Route `Help > Check for Updates...` through WinSparkle on supported Windows builds.
- Keep the current browser fallback available if:
  - WinSparkle is unavailable
  - the feed is invalid
  - installer signing prerequisites are missing
- Add a hard gate: do not enable unattended install/update execution for unsigned installers.

**Success Criteria**

- [ ] Installed Windows builds can check for updates without opening GitHub first.
- [ ] WinSparkle can discover the latest signed installer through the appcast.
- [ ] Installer handoff closes the app cleanly and restarts into the updated build.
- [ ] Unsigned Windows releases fall back to informational/manual update behavior.

### Phase 5: Shared UX and Settings

Move update behavior from ad hoc dialogs into a clear update subsystem visible to users.

**Files**

- `ui/main_window.py`
- `ui/settings_dialog.py`
- `ui/widgets/dependency_widgets.py`
- `core/update_service.py`

**Work**

- Expand Settings > Updates to include:
  - check automatically
  - download automatically when supported
  - update channel
  - skipped version reset
  - diagnostics / last successful check
- Keep the top-of-window banner for fallback informational updates.
- Add explicit states for:
  - update available
  - downloading
  - waiting for restart
  - up to date
  - failed check
- Ensure the Help menu always works:
  - installed native-updater build -> native updater
  - unsupported/development build -> fallback release checker

**Success Criteria**

- [ ] Users can understand whether updates are native, informational, or disabled.
- [ ] "Skip this version" and "Remind me later" are persisted.
- [ ] Support/debugging can inspect the last update outcome from app logs/settings.

### Phase 6: Release Validation and Rollout Policy

Treat updater delivery as release infrastructure, not only as app UI.

**Files**

- `.github/workflows/build-macos.yml`
- `.github/workflows/build-windows.yml`
- `docs/releases.md`
- `README.md`
- new smoke-test docs if needed

**Work**

- Add release-candidate smoke checks for:
  - manual update check
  - update discovery from appcast
  - macOS install from a prior version
  - Windows installer handoff from a prior version
- Define rollout policy:
  - stable only at first
  - phased rollout later on macOS where supported
  - critical update handling
- Define rollback policy:
  - yank broken feed item
  - publish newer hotfix feed item
  - disable automatic checks temporarily if feed is bad

**Success Criteria**

- [ ] A clean-machine test can update from `vN` to `vN+1` on macOS.
- [ ] A clean-machine test can update from `vN` to `vN+1` on Windows.
- [ ] Broken release metadata can be rolled back without shipping a new app binary.

## Alternative Approaches Considered

### 1. Keep the current browser-based release flow

Rejected because it does not materially reduce user friction and does not create a trusted update pipeline.

### 2. Build a custom cross-platform updater around GitHub Releases JSON

Rejected because it would require reinventing:

- feed semantics
- signed metadata
- phased rollout policy
- installer handoff
- version skip behavior
- release-note presentation

Sparkle and WinSparkle already solve most of this.

### 3. Ship only a "download latest" button inside the app

Rejected as the end state, but kept as the fallback path. It is valuable when native updater prerequisites fail or when running unsupported builds.

## Acceptance Criteria

### Functional Requirements

- [ ] Installed macOS builds can check for, discover, and apply a newer signed update from inside the app.
- [ ] Installed Windows builds can check for, discover, and hand off to a signed installer update from inside the app.
- [x] Development/source builds keep the current fallback GitHub-release flow.
- [x] Settings allow users to enable/disable automatic checks and reset skipped updates.
- [ ] The updater honors skipped-version state and can surface critical updates appropriately.

### Non-Functional Requirements

- [ ] Update metadata is signed and hosted at a stable URL.
- [ ] macOS updates preserve Developer ID signing and notarization integrity.
- [ ] Windows auto-install behavior is gated behind Authenticode-signed installers.
- [x] Update checks do not block the UI thread.
- [ ] Failure modes degrade to a safe manual download path.

### Quality Gates

- [x] Unit tests cover version comparison, update state transitions, and fallback logic.
- [ ] Integration tests validate appcast parsing and feed selection.
- [ ] CI validates generated appcast entries against release artifacts.
- [ ] Manual smoke tests are performed on clean macOS and Windows environments before enabling automatic install broadly.

## Success Metrics

- Reduced support burden around "where do I download the latest version?"
- Majority of installed users can update without leaving the app
- No release where the appcast points to missing or unsigned assets
- No regression in notarized macOS launch behavior or Windows installer execution

## Dependencies & Prerequisites

- Active Apple Developer signing/notarization setup
- Windows code-signing certificate if unattended Windows install is desired
- Stable public hosting for appcast XML and release notes
- Release workflows capable of signing assets and publishing metadata
- Time for manual clean-machine testing on both platforms

## Risk Analysis & Mitigation

| Risk | Impact | Mitigation |
| --- | --- | --- |
| PyInstaller + Sparkle packaging is harder than a standard Xcode app | High | Ship Sparkle behind a dedicated macOS adapter and validate post-build re-signing in CI before enabling |
| Windows installer handoff is brittle or unsafe without Authenticode | High | Gate unattended install behind signing; keep fallback informational flow |
| Appcast feed becomes stale or malformed | High | Validate in CI and keep a rollback path separate from app binaries |
| Version comparison drifts across bundle/appcast/installer metadata | High | Introduce one canonical version model and stamp every artifact from it |
| Users run app from a DMG or translocated path | Medium | Detect and fall back to "move to Applications" guidance plus informational updates |
| GitHub Releases URLs or hosting assumptions change | Medium | Keep appcast hosting separate and validate asset URLs on every release |

## Resource Requirements

- Python application work
- Native platform integration work:
  - macOS Sparkle embedding/signing
  - Windows WinSparkle binding and installer coordination
- CI/CD work for appcast generation and publishing
- Manual QA on clean machines

## Future Considerations

- phased rollouts
- beta/stable channels
- critical-update policy
- release-note rendering inside the app
- update telemetry or anonymous adoption metrics if ever desired

## Documentation Plan

- Update `docs/releases.md` with appcast/signing/publishing steps
- Add operator docs for update-key rotation and feed rollback
- Add user-facing docs for:
  - update settings
  - how in-app updates behave on macOS
  - how installer-based updates behave on Windows

## References & Research

### Internal References

- `core/update_checker.py:17`
- `core/update_checker.py:87`
- `ui/main_window.py:1087`
- `ui/main_window.py:1122`
- `ui/settings_dialog.py:1126`
- `core/app_version.py:19`
- `.github/workflows/build-macos.yml:35`
- `.github/workflows/build-macos.yml:116`
- `.github/workflows/build-macos.yml:165`
- `.github/workflows/build-macos.yml:211`
- `.github/workflows/build-windows.yml:35`
- `.github/workflows/build-windows.yml:106`
- `.github/workflows/build-windows.yml:131`
- `docs/releases.md:5`
- `docs/releases.md:93`
- `docs/plans/2026-02-21-feat-macos-app-bundle-distribution-plan.md:252`
- `docs/plans/2026-02-24-feat-windows-version-plan.md:13`

### External References

- Sparkle documentation: https://sparkle-project.org/documentation/
- Sparkle publishing updates: https://sparkle-project.org/documentation/publishing/
- WinSparkle project and signing model: https://github.com/vslavik/winsparkle
- GitHub Releases API: https://docs.github.com/en/rest/releases/releases
- Apple notarization overview: https://developer.apple.com/documentation/security/notarizing_macos_software_before_distribution
- Apple notarization automation: https://developer.apple.com/documentation/security/customizing-the-notarization-workflow

### Related Work

- No separate issue tracked yet
- Existing release automation work lives in `docs/releases.md`
