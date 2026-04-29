"""Audio file format constants shared across the app."""

# Extensions accepted as audio sources. Keep aligned with AUDIO_FILE_DIALOG_FILTER.
AUDIO_EXTENSIONS = frozenset({".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg"})

# Filter string for QFileDialog.getOpenFileName / getOpenFileNames.
AUDIO_FILE_DIALOG_FILTER = (
    "Audio Files (*.mp3 *.wav *.flac *.m4a *.aac *.ogg);;All Files (*)"
)


def is_audio_file(path) -> bool:
    """Return True if the given path's extension is a supported audio format."""
    from pathlib import Path

    return Path(path).suffix.lower() in AUDIO_EXTENSIONS
