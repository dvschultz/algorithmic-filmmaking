"""Audio analysis for beat detection and rhythm-based sequencing.

This module provides audio analysis functions using librosa:
- Beat detection and tempo (BPM) estimation
- Onset/transient detection for cut points
- Audio extraction from video files

Primary use case: Aligning video cuts to music beats for
music videos, montages, and rhythm-driven editing.
"""

import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Lazy import librosa to avoid startup cost
_librosa = None


def _get_librosa():
    """Lazy load librosa module."""
    global _librosa
    if _librosa is None:
        import librosa
        _librosa = librosa
    return _librosa


@dataclass
class AudioAnalysis:
    """Results from audio beat/rhythm analysis.

    Attributes:
        tempo_bpm: Detected tempo in beats per minute
        beat_times: Timestamps (seconds) of detected beats
        onset_times: Timestamps of transients/hits (good cut points)
        downbeat_times: Strong beats (measure starts in 4/4 time)
        duration_seconds: Total audio duration
        sample_rate: Audio sample rate used for analysis
    """
    tempo_bpm: float = 0.0
    beat_times: list[float] = None
    onset_times: list[float] = None
    downbeat_times: list[float] = None
    duration_seconds: float = 0.0
    sample_rate: int = 22050

    def __post_init__(self):
        if self.beat_times is None:
            self.beat_times = []
        if self.onset_times is None:
            self.onset_times = []
        if self.downbeat_times is None:
            self.downbeat_times = []

    def to_dict(self) -> dict:
        """Serialize for JSON storage."""
        return {
            "tempo_bpm": self.tempo_bpm,
            "beat_times": self.beat_times,
            "onset_times": self.onset_times,
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
            downbeat_times=data.get("downbeat_times", []),
            duration_seconds=data.get("duration_seconds", 0.0),
            sample_rate=data.get("sample_rate", 22050),
        )

    def nearest_beat(self, time: float) -> float:
        """Find the beat timestamp nearest to a given time.

        Args:
            time: Target time in seconds

        Returns:
            Nearest beat timestamp, or the input time if no beats
        """
        if not self.beat_times:
            return time
        return min(self.beat_times, key=lambda b: abs(b - time))

    def nearest_onset(self, time: float) -> float:
        """Find the onset timestamp nearest to a given time.

        Args:
            time: Target time in seconds

        Returns:
            Nearest onset timestamp, or the input time if no onsets
        """
        if not self.onset_times:
            return time
        return min(self.onset_times, key=lambda o: abs(o - time))


def has_audio_track(file_path: Path) -> bool:
    """Check if a file has an audio stream.

    Args:
        file_path: Path to video or audio file

    Returns:
        True if file has at least one audio stream
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=codec_type",
        "-of", "csv=p=0",
        str(file_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return bool(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning(f"ffprobe check failed: {e}")
        return False


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

    cmd = [
        "ffmpeg", "-y",
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
) -> AudioAnalysis:
    """Analyze audio file for beats, tempo, and onsets.

    Args:
        audio_path: Path to audio file (WAV, MP3, etc.)
        sample_rate: Sample rate for analysis
        include_onsets: Whether to detect onsets (adds processing time)

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
        if include_onsets:
            logger.info("Detecting onsets")
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
            onset_times = librosa.frames_to_time(onset_frames, sr=sr).tolist()

        logger.info(
            f"Analysis complete: {tempo:.1f} BPM, "
            f"{len(beat_times)} beats, {len(onset_times)} onsets"
        )

        return AudioAnalysis(
            tempo_bpm=tempo,
            beat_times=beat_times,
            onset_times=onset_times,
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
) -> AudioAnalysis:
    """Extract and analyze audio from a video file.

    Convenience function that combines extract_audio and analyze_audio.

    Args:
        video_path: Path to video file
        sample_rate: Sample rate for analysis
        include_onsets: Whether to detect onsets
        cleanup: Whether to delete extracted audio after analysis

    Returns:
        AudioAnalysis with beat times, tempo, and onsets

    Raises:
        ValueError: If video has no audio track
        RuntimeError: If extraction or analysis fails
    """
    if not has_audio_track(video_path):
        raise ValueError(f"Video has no audio track: {video_path}")

    audio_path = extract_audio(video_path, sample_rate=sample_rate)

    try:
        return analyze_audio(audio_path, sample_rate=sample_rate, include_onsets=include_onsets)
    finally:
        if cleanup and audio_path.exists():
            audio_path.unlink()
            # Try to remove temp directory if empty
            try:
                audio_path.parent.rmdir()
            except OSError:
                pass


def analyze_music_file(
    music_path: Path,
    include_onsets: bool = True,
) -> AudioAnalysis:
    """Analyze a music file (MP3, WAV, FLAC, etc.).

    Args:
        music_path: Path to music file
        include_onsets: Whether to detect onsets

    Returns:
        AudioAnalysis with beat times, tempo, and onsets

    Raises:
        FileNotFoundError: If file doesn't exist
        RuntimeError: If analysis fails
    """
    if not music_path.exists():
        raise FileNotFoundError(f"Music file not found: {music_path}")

    return analyze_audio(music_path, include_onsets=include_onsets)
