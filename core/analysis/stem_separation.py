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
        from demucs_infer.pretrained import get_model
        from demucs_infer.apply import apply_model
        from demucs_infer.audio import save_audio
    except ImportError:
        raise ImportError(
            "demucs-infer is required for stem separation. "
            "Install it with: pip install demucs-infer"
        )

    import torch
    import numpy as np
    from core.analysis.audio import _get_librosa

    output_dir.mkdir(parents=True, exist_ok=True)

    if progress_cb:
        progress_cb("Loading separation model...")

    # Device detection: prefer CUDA, then MPS (Apple Silicon), then CPU
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    try:
        model = get_model("htdemucs")
        model.to(device)
    except Exception as e:
        from core.errors import ModelDownloadError

        raise ModelDownloadError(
            f"Failed to load Demucs htdemucs model: {e}"
        ) from e

    if progress_cb:
        progress_cb("Loading audio...")

    librosa = _get_librosa()
    # Load at model's sample rate, mono=False to preserve stereo
    y, sr = librosa.load(str(music_path), sr=model.samplerate, mono=False)
    # librosa returns (samples,) for mono or (channels, samples) for stereo
    if y.ndim == 1:
        y = np.stack([y, y])  # mono -> stereo
    wav = torch.from_numpy(y).float()

    # Normalize (canonical demucs normalization with epsilon guard)
    ref = wav.mean(0)
    ref_mean = ref.mean()
    ref_std = ref.std() + 1e-8
    wav = (wav - ref_mean) / ref_std
    mix = wav.unsqueeze(0).to(device)

    if progress_cb:
        progress_cb("Separating stems...")

    try:
        sources = apply_model(model, mix, device=device, progress=False)
        # sources shape: (1, num_sources, channels, samples)
        sources = sources.squeeze(0).cpu()  # (num_sources, channels, samples)
        # Undo normalization (matching epsilon from above)
        sources = sources * ref_std + ref_mean
    except Exception as e:
        raise RuntimeError(f"Stem separation failed: {e}") from e

    # Save each stem to WAV
    stems = {}
    source_names = model.sources  # e.g., ['drums', 'bass', 'other', 'vocals']
    for i, name in enumerate(source_names):
        if name not in STEM_NAMES:
            logger.warning("Unexpected stem name '%s' from model", name)
            continue

        out_path = output_dir / f"{name}.wav"
        if progress_cb:
            progress_cb(f"Saving {name} stem...")

        save_audio(sources[i], str(out_path), samplerate=model.samplerate)
        stems[name] = out_path

    if len(stems) < len(STEM_NAMES):
        missing = set(STEM_NAMES) - set(stems.keys())
        logger.warning("Missing stems from separation output: %s", missing)

    logger.info(f"Separated {len(stems)} stems to {output_dir}")
    return stems
