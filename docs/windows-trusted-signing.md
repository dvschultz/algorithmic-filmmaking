# Windows Trusted Signing

This document captures the recommended Authenticode path for Scene Ripper Windows releases.

## Goal

Use Microsoft Trusted Signing to Authenticode-sign the Windows app and installer so the release is trusted as a signed Windows publisher.

This is separate from WinSparkle updater signing:

- Trusted Signing: Windows Authenticode trust, SmartScreen reputation, signed `.exe` files
- `UPDATE_PUBLIC_ED_KEY` / `UPDATE_PRIVATE_ED_KEY`: WinSparkle update-feed integrity

You likely want both.

## Recommended Model

Use Microsoft Artifact Signing with the `Public Trust` model.

As of March 22, 2026, Microsoft documents `Public Trust` availability for:

- organizations in the USA, Canada, the European Union, and the United Kingdom
- individual developers in the USA and Canada

## Azure Setup

1. Register `Microsoft.CodeSigning` in your Azure subscription.
2. Create an Artifact Signing account.
3. Complete identity validation in the Azure portal.
4. Create a `Public Trust` certificate profile.
5. Grant the CI identity the `Artifact Signing Certificate Profile Signer` role.

## CI Requirements

On the Windows runner, install or make available:

- Windows SDK `signtool.exe` `10.0.2261.755+`
- `.NET 8 Runtime`
- Microsoft Artifact Signing client tools / dlib

Prefer GitHub OIDC to Azure over storing a long-lived client secret.

## Signing Metadata

Create a `metadata.json` file for Trusted Signing:

```json
{
  "Endpoint": "https://eus.codesigning.azure.net",
  "CodeSigningAccountName": "your-account",
  "CertificateProfileName": "your-public-trust-profile",
  "CorrelationId": "optional-build-id"
}
```

Use the endpoint for the region where your signing account exists.

## Signing Commands

Sign the packaged app executable:

```powershell
signtool sign /v /debug /fd SHA256 /tr "http://timestamp.acs.microsoft.com" /td SHA256 `
  /dlib "C:\\path\\to\\Azure.CodeSigning.Dlib.dll" `
  /dmdf ".\\metadata.json" `
  "dist\\Scene Ripper\\Scene Ripper.exe"
```

Sign the installer:

```powershell
signtool sign /v /debug /fd SHA256 /tr "http://timestamp.acs.microsoft.com" /td SHA256 `
  /dlib "C:\\path\\to\\Azure.CodeSigning.Dlib.dll" `
  /dmdf ".\\metadata.json" `
  "packaging\\windows\\Output\\SceneRipper-Setup-0.2.4.exe"
```

Verify signatures:

```powershell
signtool verify /pa /v "dist\\Scene Ripper\\Scene Ripper.exe"
signtool verify /pa /v "packaging\\windows\\Output\\SceneRipper-Setup-0.2.4.exe"
```

## Repo Integration Plan

For this repo, the Windows workflow should do all of the following:

1. Authenticate GitHub Actions to Azure.
2. Install Trusted Signing prerequisites on the Windows runner.
3. Build the PyInstaller app.
4. Authenticode-sign `dist/Scene Ripper/Scene Ripper.exe`.
5. Build the Inno Setup installer.
6. Authenticode-sign `packaging/windows/Output/SceneRipper-Setup-<version>.exe`.
7. Verify both signatures in CI.
8. Upload the signed installer to the GitHub release.

## Inno Setup Note

The current Inno Setup script does not define a `SignTool` or `SignedUninstaller` path.

To fully sign the Windows installation experience, update `packaging/windows/scene_ripper.iss` so the generated uninstaller is also signed.

## Important Distinction

Trusted Signing does not replace WinSparkle signing.

- Trusted Signing signs Windows executables and installers.
- WinSparkle signing signs updater metadata and installer payload references for native in-app updates.

If you want:

- fewer Windows trust warnings: you need Trusted Signing
- secure native updater feeds: you need `UPDATE_PUBLIC_ED_KEY` and `UPDATE_PRIVATE_ED_KEY`

## References

- Microsoft quickstart: <https://learn.microsoft.com/en-us/azure/artifact-signing/quickstart>
- Microsoft signing integrations: <https://learn.microsoft.com/en-us/azure/artifact-signing/how-to-signing-integrations>
- Microsoft trust models: <https://learn.microsoft.com/en-us/azure/artifact-signing/concept-trust-models>
