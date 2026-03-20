# Releases

This repo ships desktop releases from Git tags.

## Standard Release

Use this flow when the packaging workflows are already healthy.

1. Make sure `main` contains the exact commit you want to ship.
2. Push `main`.
3. Create a version tag in `vX.Y.Z` format.
4. Push the tag.
5. Wait for the macOS and Windows workflows to finish.
6. Confirm the GitHub release contains both release assets.

```bash
git checkout main
git pull --ff-only origin main
git tag v0.1.1
git push origin main
git push origin v0.1.1
gh run list --limit 10
gh release view v0.1.1
```

## Expected Assets

- macOS: `Scene-Ripper-<version>-arm64.dmg`
- Windows: `SceneRipper-Setup-<version>.exe`

The release workflows upload both assets to the GitHub release for the matching tag.

They also generate per-platform update-feed artifacts in workflow artifacts:

- macOS: `Scene-Ripper-macOS-Appcast`
- Windows: `SceneRipper-Windows-Appcast`

The workflows also publish stable feed assets to a dedicated GitHub release tag named `update-feed`:

- `appcast-macos.xml`
- `release-notes-macos.html`
- `appcast-windows.xml`
- `release-notes-windows.html`

These assets are the stable public URLs used by native in-app updates, while the DMG and installer downloads still come from the versioned `vX.Y.Z` releases.

## Workflow Files

- [build-macos.yml](../.github/workflows/build-macos.yml)
- [build-windows.yml](../.github/workflows/build-windows.yml)

Both workflows run on:

- tag push matching `v*`
- manual `workflow_dispatch`

## Monitoring

Check workflow status:

```bash
gh run list --limit 10
gh run watch
```

Inspect the published release:

```bash
gh release view v0.1.1
```

## If a Tag Release Fails

Use this flow if the first tag-triggered release fails and you need to fix packaging or workflow logic.

1. Fix the issue on `main`.
2. Push the fix.
3. Rerun the platform workflows manually with the same version number.
4. Confirm both successful runs uploaded the correct assets to the existing release.
5. Move the tag to the exact commit used by the successful release builds.

Example:

```bash
git checkout main
git pull --ff-only origin main
git log --oneline -5
git tag -f v0.1.1 <good_commit_sha>
git push origin -f v0.1.1
```

This matters because a GitHub release can contain assets built from a newer commit than the tag currently points to.

## Manual Workflow Dispatch

Use manual dispatch when you want to rebuild a release without creating a new tag yet, or after fixing a broken workflow.

From GitHub Actions:

1. Open `macOS Build`.
2. Run workflow on `main` with `version=0.1.1`.
3. Open `Windows Build`.
4. Run workflow on `main` with `version=0.1.1`.

The workflows are configured to upload assets to release tag `v0.1.1` when dispatched with version `0.1.1`.

## macOS Signing and Notarization

The macOS workflow signs and notarizes only when these GitHub secrets are configured:

- `MACOS_CERTIFICATE`
- `MACOS_CERTIFICATE_PWD`
- `KEYCHAIN_PASSWORD`
- `CODESIGN_IDENTITY`
- `APPLE_ID`
- `APPLE_TEAM_ID`
- `APPLE_APP_PASSWORD`

Without those secrets, the DMG still builds, but it will not be signed or notarized.

## Updater Signing

Native updater feeds use EdDSA signatures for the DMG and Windows installer when these secrets are configured:

- `UPDATE_PUBLIC_ED_KEY`
- `UPDATE_PRIVATE_ED_KEY`

The public key is embedded into packaged builds. The private key is only used in CI to sign release artifacts before generating appcast metadata.

## Local Build Fallback

Use local builds to test packaging before tagging.

macOS:

```bash
APP_VERSION=0.1.1 ./packaging/macos/build.sh --dmg
```

Windows:

```powershell
$env:APP_VERSION="0.1.1"
.\packaging\windows\build.ps1
```

Scripts:

- [build.sh](../packaging/macos/build.sh)
- [build.ps1](../packaging/windows/build.ps1)

## Release Checklist

Before tagging:

- `main` is pushed and green enough to ship
- release notes, README, or docs updates are included if needed
- macOS signing secrets are present if you want a signed/notarized DMG

After tagging:

- macOS workflow succeeds
- Windows workflow succeeds
- `gh release view vX.Y.Z` shows both assets
- `gh release view update-feed` shows refreshed appcast and release-notes assets
- the tag points to the exact commit used by the successful release builds
