"""Audio analysis for beat detection and rhythm-based sequencing.

This module provides audio analysis functions using librosa:
- Beat detection and tempo (BPM) estimation
- Onset/transient detection for cut points
- Audio extraction from video files

Primary use case: Aligning video cuts to music beats for
music videos, montages, and rhythm-driven editing.
"""

import bisect
import logging
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Optional

from core.binary_resolver import find_binary, get_subprocess_kwargs

import numpy as np

logger = logging.getLogger(__name__)

# Lazy import librosa to avoid startup cost
_librosa = None


def ensure_audio_analysis_runtime_available():
    """Validate that the audio-analysis runtime imports cleanly."""
    try:
        import librosa

        return librosa
    except Exception as e:
        raise RuntimeError(f"audio analysis runtime is incomplete: {e}") from e


def _get_librosa():
    """Lazy load librosa module."""
    global _librosa
    if _librosa is None:
        _librosa = ensure_audio_analysis_runtime_available()
    return _librosa


@dataclass
class AudioAnalysis:
    """Results from audio beat/rhythm analysis.

    Attributes:
        tempo_bpm: Detected tempo in beats per minute
        beat_times: Timestamps (seconds) of detected beats (sorted)
        onset_times: Timestamps of transients/hits (good cut points, sorted)
        downbeat_times: Strong beats (measure starts, assumes 4/4 time)
        duration_seconds: Total audio duration
        sample_rate: Audio sample rate used for analysis
    """
    tempo_bpm: float = 0.0
    beat_times: list[float] = field(default_factory=list)
    onset_times: list[float] = field(default_factory=list)
    onset_strengths: list[float] = field(default_factory=list)  # normalized [0,1] per onset
    downbeat_times: list[float] = field(default_factory=list)
    duration_seconds: float = 0.0
    sample_rate: int = 22050

    def to_dict(self) -> dict:
        """Serialize for JSON storage."""
        return {
            "tempo_bpm": self.tempo_bpm,
            "beat_times": self.beat_times,
            "onset_times": self.onset_times,
            "onset_strengths": self.onset_strengths,
            "downbeat_times": self.downbeat_times,
            "duration_seconds": self.duration_seconds,
            "sample_rate": self.sample_rate,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AudioAnalysis":
        """Deserialize from dictionary."""
        if data is None:
            return cls()
        return cls(
            tempo_bpm=data.get("tempo_bpm", 0.0),
            beat_times=data.get("beat_times", []),
            onset_times=data.get("onset_times", []),
            onset_strengths=data.get("onset_strengths", []),
            downbeat_times=data.get("downbeat_times", []),
            duration_seconds=data.get("duration_seconds", 0.0),
            sample_rate=data.get("sample_rate", 22050),
        )

    def nearest_beat(self, time: float) -> float:
        """Find the beat timestamp nearest to a given time.

        Uses binary search for O(log n) performance.

        Args:
            time: Target time in seconds

        Returns:
            Nearest beat timestamp, or the input time if no beats
        """
        return _find_nearest(self.beat_times, time)

    def nearest_onset(self, time: float) -> float:
        """Find the onset timestamp nearest to a given time.

        Uses binary search for O(log n) performance.

        Args:
            time: Target time in seconds

        Returns:
            Nearest onset timestamp, or the input time if no onsets
        """
        return _find_nearest(self.onset_times, time)

    def onset_strength_at(self, time: float) -> float:
        """Get the onset strength at or nearest to a given time.

        Args:
            time: Target time in seconds

        Returns:
            Normalized onset strength [0, 1], or 0.5 if no onset data
        """
        if not self.onset_times or not self.onset_strengths:
            return 0.5

        idx = bisect.bisect_left(self.onset_times, time)

        if idx == 0:
            return self.onset_strengths[0]
        if idx >= len(self.onset_times):
            return self.onset_strengths[-1]

        # Return strength of nearest onset
        before = self.onset_times[idx - 1]
        after = self.onset_times[idx]
        nearest_idx = idx - 1 if (time - before) <= (after - time) else idx
        if nearest_idx < len(self.onset_strengths):
            return self.onset_strengths[nearest_idx]
        return 0.5


@dataclass(frozen=True)
class OnsetDetectionConfig:
    """Tunable onset detector settings for rhythm-heavy editing workflows."""

    profile: str = "balanced"
    hop_length: int = 512
    delta: float = 0.07
    wait_seconds: float = 0.03
    pre_max_seconds: float = 0.03
    post_max_seconds: float = 0.00
    pre_avg_seconds: float = 0.10
    post_avg_seconds: float = 0.10
    backtrack: bool = False
    superflux: bool = False
    lag: int = 2
    max_size: int = 3


_ONSET_PROFILE_PRESETS: dict[str, dict[str, object]] = {
    "balanced": {
        "delta": 0.07,
        "hop_length": 512,
        "wait_seconds": 0.06,
        "backtrack": False,
        "superflux": False,
    },
    "drums": {
        "delta": 0.04,
        "hop_length": 256,
        "wait_seconds": 0.06,
        "backtrack": True,
        "superflux": True,
    },
    "dense": {
        "delta": 0.025,
        "hop_length": 256,
        "wait_seconds": 0.03,
        "backtrack": True,
        "superflux": True,
    },
    "sparse": {
        "delta": 0.12,
        "hop_length": 512,
        "wait_seconds": 0.12,
        "backtrack": False,
        "superflux": False,
    },
}


def make_onset_detection_config(
    profile: str = "balanced",
    *,
    cut_density: int = 5,
    min_gap_ms: Optional[int] = None,
    backtrack: Optional[bool] = None,
    hop_length: Optional[int] = None,
    delta: Optional[float] = None,
    superflux: Optional[bool] = None,
) -> OnsetDetectionConfig:
    """Build an onset detection config from editor-facing controls.

    Args:
        profile: One of ``balanced``, ``drums``, ``dense``, ``sparse``, or ``custom``.
        cut_density: 1-10 where higher means more sensitive onset picking.
        min_gap_ms: Minimum interval between detected onsets.
        backtrack: Whether to align detected events to transient starts.
        hop_length: Analysis hop length. Lower values improve timing precision.
        delta: Raw peak-picking threshold override.
        superflux: Whether to use a SuperFlux-style onset envelope.
    """
    normalized_profile = (profile or "balanced").lower()
    preset = dict(_ONSET_PROFILE_PRESETS.get(normalized_profile, _ONSET_PROFILE_PRESETS["balanced"]))

    density = max(1, min(10, int(cut_density)))
    # Density 1 keeps only clear attacks; density 10 admits much weaker peaks.
    density_factor = float(np.interp(density, [1, 10], [1.8, 0.45]))
    base_delta = float(preset["delta"])

    return OnsetDetectionConfig(
        profile=normalized_profile,
        hop_length=int(hop_length if hop_length is not None else preset["hop_length"]),
        delta=float(delta if delta is not None else base_delta * density_factor),
        wait_seconds=(
            float(min_gap_ms) / 1000.0
            if min_gap_ms is not None
            else float(preset["wait_seconds"])
        ),
        backtrack=bool(backtrack if backtrack is not None else preset["backtrack"]),
        superflux=bool(superflux if superflux is not None else preset["superflux"]),
    )


def _seconds_to_frames(seconds: float, sr: int, hop_length: int) -> int:
    return max(1, int(round(seconds * sr / hop_length)))


def _detect_tuned_onsets(
    librosa,
    y,
    sr: int,
    config: OnsetDetectionConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect onsets using an explicitly configured onset envelope."""
    hop_length = config.hop_length

    if config.superflux:
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            hop_length=hop_length,
            fmin=27.5,
            n_mels=138,
        )
        onset_env = librosa.onset.onset_strength(
            S=librosa.power_to_db(mel, ref=np.max),
            sr=sr,
            hop_length=hop_length,
            lag=config.lag,
            max_size=config.max_size,
        )
    else:
        onset_env = librosa.onset.onset_strength(
            y=y,
            sr=sr,
            hop_length=hop_length,
            aggregate=np.median,
        )

    peak_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        backtrack=False,
        units="frames",
        pre_max=_seconds_to_frames(config.pre_max_seconds, sr, hop_length),
        post_max=max(1, _seconds_to_frames(config.post_max_seconds, sr, hop_length)),
        pre_avg=_seconds_to_frames(config.pre_avg_seconds, sr, hop_length),
        post_avg=_seconds_to_frames(config.post_avg_seconds, sr, hop_length),
        delta=config.delta,
        wait=_seconds_to_frames(config.wait_seconds, sr, hop_length),
    )
    onset_frames = peak_frames
    if config.backtrack and len(peak_frames) > 0:
        onset_frames = librosa.onset.onset_backtrack(peak_frames, onset_env)

    return onset_frames, onset_env, peak_frames


def _find_nearest(sorted_times: list[float], time: float) -> float:
    """Find the nearest timestamp using binary search.

    Args:
        sorted_times: Sorted list of timestamps
        time: Target time to find nearest match for

    Returns:
        Nearest timestamp, or the input time if list is empty
    """
    if not sorted_times:
        return time

    idx = bisect.bisect_left(sorted_times, time)

    # Handle edge cases
    if idx == 0:
        return sorted_times[0]
    if idx == len(sorted_times):
        return sorted_times[-1]

    # Compare neighbors and return closest
    before = sorted_times[idx - 1]
    after = sorted_times[idx]
    return before if (time - before) <= (after - time) else after


def has_audio_track(file_path: Path) -> bool:
    """Check if a file has an audio stream.

    Args:
        file_path: Path to video or audio file

    Returns:
        True if file has at least one audio stream
    """
    _ffprobe = find_binary("ffprobe") or "ffprobe"
    cmd = [
        _ffprobe, "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=codec_type",
        "-of", "csv=p=0",
        str(file_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30,
                                **get_subprocess_kwargs())
        # Check return code - ffprobe returns non-zero for errors
        if result.returncode != 0:
            logger.warning(f"ffprobe returned non-zero: {result.stderr.strip()}")
            return False
        return bool(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning(f"ffprobe check failed: {e}")
        return False


@contextmanager
def extracted_audio(
    video_path: Path,
    sample_rate: int = 22050,
) -> Generator[Path, None, None]:
    """Context manager for temporary audio extraction.

    Guarantees cleanup of temp files even if analysis fails.

    Args:
        video_path: Path to video file
        sample_rate: Target sample rate

    Yields:
        Path to extracted WAV file

    Example:
        with extracted_audio(video_path) as audio_path:
            result = analyze_audio(audio_path)
    """
    audio_path = extract_audio(video_path, sample_rate=sample_rate)
    try:
        yield audio_path
    finally:
        if audio_path.exists():
            audio_path.unlink()
        # Try to remove temp directory if empty
        try:
            audio_path.parent.rmdir()
        except OSError:
            pass


def extract_audio(
    video_path: Path,
    output_path: Optional[Path] = None,
    sample_rate: int = 22050,
) -> Path:
    """Extract audio from video file to WAV format.

    Args:
        video_path: Path to video file
        output_path: Output WAV path (default: temp file)
        sample_rate: Target sample rate (default: 22050 for librosa)

    Returns:
        Path to extracted WAV file

    Raises:
        RuntimeError: If extraction fails
    """
    if output_path is None:
        temp_dir = tempfile.mkdtemp(prefix="audio_")
        output_path = Path(temp_dir) / "audio.wav"

    _ffmpeg = find_binary("ffmpeg") or "ffmpeg"
    cmd = [
        _ffmpeg, "-y",
        "-i", str(video_path),
        "-vn",  # No video
        "-acodec", "pcm_s16le",  # PCM 16-bit
        "-ar", str(sample_rate),  # Sample rate
        "-ac", "1",  # Mono
        str(output_path)
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            **get_subprocess_kwargs(),
        )
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg audio extraction failed: {result.stderr}")
        return output_path
    except subprocess.TimeoutExpired:
        raise RuntimeError("Audio extraction timed out")


def analyze_audio(
    audio_path: Path,
    sample_rate: int = 22050,
    include_onsets: bool = True,
    onset_config: Optional[OnsetDetectionConfig] = None,
) -> AudioAnalysis:
    """Analyze audio file for beats, tempo, and onsets.

    Args:
        audio_path: Path to audio file (WAV, MP3, etc.)
        sample_rate: Sample rate for analysis
        include_onsets: Whether to detect onsets (adds processing time)
        onset_config: Optional tuned onset detector settings

    Returns:
        AudioAnalysis with beat times, tempo, and optionally onsets

    Raises:
        RuntimeError: If analysis fails
    """
    librosa = _get_librosa()

    try:
        logger.info(f"Loading audio: {audio_path}")
        y, sr = librosa.load(str(audio_path), sr=sample_rate)
        duration = len(y) / sr

        logger.info(f"Detecting beats (duration: {duration:.1f}s)")

        # Beat tracking
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

        # Convert tempo to float (may be array)
        if hasattr(tempo, '__iter__'):
            tempo = float(tempo[0]) if len(tempo) > 0 else 0.0
        else:
            tempo = float(tempo)

        # Estimate downbeats (every 4th beat for 4/4 time)
        downbeat_times = beat_times[::4] if beat_times else []

        # Onset detection (transients - good for cut points)
        onset_times = []
        onset_strengths = []
        if include_onsets:
            logger.info("Detecting onsets")
            if onset_config is None:
                onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
                onset_env = librosa.onset.onset_strength(y=y, sr=sr)
                onset_times = librosa.frames_to_time(onset_frames, sr=sr).tolist()
            else:
                onset_frames, onset_env, strength_frames = _detect_tuned_onsets(
                    librosa, y, sr, onset_config
                )
                onset_times = librosa.frames_to_time(
                    onset_frames,
                    sr=sr,
                    hop_length=onset_config.hop_length,
                ).tolist()

            # Sample peak frames for strength, even when displayed onsets are backtracked.
            if len(onset_frames) > 0:
                if onset_config is None:
                    strength_frames = onset_frames
                # Clamp frame indices to envelope length
                valid_frames = np.clip(strength_frames, 0, len(onset_env) - 1)
                raw_strengths = onset_env[valid_frames].tolist()
                # Normalize to [0, 1]
                max_str = max(raw_strengths) if raw_strengths else 0.0
                if max_str > 0:
                    onset_strengths = [s / max_str for s in raw_strengths]
                else:
                    onset_strengths = [0.0] * len(raw_strengths)

        logger.info(
            f"Analysis complete: {tempo:.1f} BPM, "
            f"{len(beat_times)} beats, {len(onset_times)} onsets"
        )

        return AudioAnalysis(
            tempo_bpm=tempo,
            beat_times=beat_times,
            onset_times=onset_times,
            onset_strengths=onset_strengths,
            downbeat_times=downbeat_times,
            duration_seconds=duration,
            sample_rate=sr,
        )

    except Exception as e:
        logger.error(f"Audio analysis failed: {e}")
        raise RuntimeError(f"Audio analysis failed: {e}") from e


def analyze_audio_from_video(
    video_path: Path,
    sample_rate: int = 22050,
    include_onsets: bool = True,
    cleanup: bool = True,
    onset_config: Optional[OnsetDetectionConfig] = None,
) -> AudioAnalysis:
    """Extract and analyze audio from a video file.

    Convenience function that combines extract_audio and analyze_audio.

    Args:
        video_path: Path to video file
        sample_rate: Sample rate for analysis
        include_onsets: Whether to detect onsets
        cleanup: Whether to delete extracted audio after analysis
        onset_config: Optional tuned onset detector settings

    Returns:
        AudioAnalysis with beat times, tempo, and onsets

    Raises:
        ValueError: If video has no audio track
        RuntimeError: If extraction or analysis fails
    """
    if not has_audio_track(video_path):
        raise ValueError(f"Video has no audio track: {video_path}")

    if cleanup:
        # Use context manager for guaranteed cleanup
        with extracted_audio(video_path, sample_rate=sample_rate) as audio_path:
            return analyze_audio(
                audio_path,
                sample_rate=sample_rate,
                include_onsets=include_onsets,
                onset_config=onset_config,
            )
    else:
        # Caller is responsible for cleanup
        audio_path = extract_audio(video_path, sample_rate=sample_rate)
        return analyze_audio(
            audio_path,
            sample_rate=sample_rate,
            include_onsets=include_onsets,
            onset_config=onset_config,
        )


def extract_clip_volume(
    source_path: Path,
    start_seconds: float,
    duration_seconds: float,
    *,
    _has_audio: Optional[bool] = None,
) -> Optional[float]:
    """Extract mean RMS volume level for a clip segment using FFmpeg volumedetect.

    Args:
        source_path: Path to the source video file
        start_seconds: Start time of the clip in seconds
        duration_seconds: Duration of the clip in seconds
        _has_audio: Pre-computed has_audio_track result to skip redundant ffprobe

    Returns:
        Mean volume in dB (typically -60 to 0, higher = louder),
        or None if the source has no audio track.
    """
    if _has_audio is None:
        _has_audio = has_audio_track(source_path)
    if not _has_audio:
        return None

    _ffmpeg = find_binary("ffmpeg") or "ffmpeg"
    null_target = "NUL" if sys.platform == "win32" else "-"
    cmd = [
        _ffmpeg, "-y",
        "-ss", str(start_seconds),
        "-t", str(duration_seconds),
        "-i", str(source_path),
        "-vn",
        "-af", "volumedetect",
        "-f", "null",
        null_target,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            **get_subprocess_kwargs(),
        )

        # Parse mean_volume from stderr
        for line in result.stderr.splitlines():
            if "mean_volume:" in line:
                # Format: "mean_volume: -23.4 dB"
                parts = line.split("mean_volume:")
                if len(parts) >= 2:
                    vol_str = parts[1].strip().replace("dB", "").strip()
                    return float(vol_str)

        logger.warning(f"No volume data found for {source_path} at {start_seconds}s")
        return None

    except subprocess.TimeoutExpired:
        logger.warning(f"Volume extraction timed out for {source_path}")
        return None
    except (ValueError, IndexError) as e:
        logger.warning(f"Failed to parse volume data: {e}")
        return None


def analyze_music_file(
    music_path: Path,
    include_onsets: bool = True,
    onset_config: Optional[OnsetDetectionConfig] = None,
) -> AudioAnalysis:
    """Analyze a music file (MP3, WAV, FLAC, etc.).

    Args:
        music_path: Path to music file
        include_onsets: Whether to detect onsets
        onset_config: Optional tuned onset detector settings

    Returns:
        AudioAnalysis with beat times, tempo, and onsets

    Raises:
        FileNotFoundError: If file doesn't exist
        RuntimeError: If analysis fails
    """
    if not music_path.exists():
        raise FileNotFoundError(f"Music file not found: {music_path}")

    return analyze_audio(
        music_path,
        include_onsets=include_onsets,
        onset_config=onset_config,
    )
