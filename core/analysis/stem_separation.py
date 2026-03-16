"""Source separation using Demucs for isolating individual stems.

Separates a music file into 4 stems: drums, bass, vocals, other.
Uses the htdemucs model from Meta via the demucs-infer package.
"""

import hashlib
import logging
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

STEM_NAMES = ("drums", "bass", "vocals", "other")


def get_stem_cache_key(music_path: Path) -> str:
    """Generate a cache key from the first 64KB of the file.

    Args:
        music_path: Path to the music file

    Returns:
        16-character hex string based on SHA-256
    """
    h = hashlib.sha256()
    with open(music_path, "rb") as f:
        h.update(f.read(65536))
    return h.hexdigest()[:16]


def get_cached_stems(music_path: Path, cache_dir: Path) -> Optional[dict[str, Path]]:
    """Check if separated stems are already cached.

    Args:
        music_path: Path to the original music file
        cache_dir: Base cache directory for stems

    Returns:
        Dict mapping stem names to WAV file paths if all 4 stems cached,
        None if cache miss.
    """
    key = get_stem_cache_key(music_path)
    stem_dir = cache_dir / key

    if not stem_dir.is_dir():
        return None

    stems = {}
    for name in STEM_NAMES:
        path = stem_dir / f"{name}.wav"
        if not path.is_file():
            return None
        stems[name] = path

    return stems


def separate_stems(
    music_path: Path,
    output_dir: Path,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> dict[str, Path]:
    """Separate a music file into individual stems using Demucs.

    Args:
        music_path: Path to the music file (MP3, WAV, FLAC, etc.)
        output_dir: Directory to write stem WAV files to
        progress_cb: Optional callback(message: str) for progress updates

    Returns:
        Dict mapping stem names to output WAV file paths

    Raises:
        ImportError: If demucs-infer is not installed
        RuntimeError: If separation fails
    """
    try:
        from demucs.api import Separator, save_audio
    except ImportError:
        raise ImportError(
            "demucs-infer is required for stem separation. "
            "Install it with: pip install demucs-infer"
        )

    import torch

    output_dir.mkdir(parents=True, exist_ok=True)

    if progress_cb:
        progress_cb("Loading separation model...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    separator = Separator(
        model="htdemucs",
        device=device,
        progress=False,
    )

    if progress_cb:
        progress_cb("Separating stems...")

    try:
        _original, separated = separator.separate_audio_file(str(music_path))
    except Exception as e:
        raise RuntimeError(f"Stem separation failed: {e}") from e

    # Save each stem to WAV
    stems = {}
    for name in STEM_NAMES:
        if name not in separated:
            logger.warning(f"Stem '{name}' not found in separation output")
            continue

        out_path = output_dir / f"{name}.wav"
        if progress_cb:
            progress_cb(f"Saving {name} stem...")

        save_audio(separated[name], str(out_path), samplerate=separator.samplerate)
        stems[name] = out_path

    logger.info(f"Separated {len(stems)} stems to {output_dir}")
    return stems
