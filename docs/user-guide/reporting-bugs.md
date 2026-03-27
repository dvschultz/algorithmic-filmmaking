# Reporting Bugs

Found a problem? This guide explains how to file a useful bug report and where to find the information that helps us fix it quickly.

## Where to Report

File bugs on GitHub Issues: [github.com/dvschultz/algorithmic-filmmaking/issues](https://github.com/dvschultz/algorithmic-filmmaking/issues)

## What to Include

A good bug report has three things:

1. **What happened** -- describe what you saw (error dialog, crash, wrong behavior)
2. **What you expected** -- what should have happened instead
3. **Steps to reproduce** -- how to trigger it (which tab, which button, what kind of video)

If you can, also include:

- Your OS and Scene Ripper version (shown in the title bar or **Help > About**)
- Whether it happens every time or intermittently
- Screenshots of error dialogs

## Attaching Log Files

Log files are the single most useful thing you can attach. They capture every operation, error, and warning behind the scenes.

### macOS

Log location:

```
~/Library/Logs/Scene Ripper/scene-ripper.log
```

To open this folder:

1. Open **Finder**
2. Press **Cmd + Shift + G** (Go to Folder)
3. Paste `~/Library/Logs/Scene Ripper/` and press Enter
4. Drag `scene-ripper.log` into the GitHub issue

Or from Terminal:

```bash
open ~/Library/Logs/Scene\ Ripper/
```

### Windows

Log location:

```
%LOCALAPPDATA%\Scene Ripper\logs\scene-ripper.log
```

To open this folder:

1. Press **Win + R** to open the Run dialog
2. Paste `%LOCALAPPDATA%\Scene Ripper\logs` and press Enter
3. Drag `scene-ripper.log` into the GitHub issue

Or from PowerShell:

```powershell
explorer "$env:LOCALAPPDATA\Scene Ripper\logs"
```

### Running from source

When running from source (`python main.py`), logs are printed to the terminal. Copy the relevant section into your bug report.

## Tips for Better Reports

- **Reproduce first, then grab the log.** The log overwrites each session, so the freshest run is the most useful.
- **Include the full error message.** If you see an error dialog, screenshot it or copy the text.
- **Mention what you were analyzing.** Some bugs only happen with certain video formats, resolutions, or codecs.
- **Note which analysis operations were running.** "I clicked Classify Shots on 50 clips" is more useful than "analysis failed."
- **If a model download failed**, include whether you're on WiFi, VPN, or behind a corporate firewall -- network restrictions are a common cause.
