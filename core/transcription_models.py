"""Shared transcription data types and errors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


class TranscriptionError(Exception):
    """Base exception for transcription errors."""


class FFmpegNotFoundError(TranscriptionError):
    """Raised when FFmpeg is required but unavailable."""

    def __init__(self):
        super().__init__(
            "FFmpeg is required for transcription but was not found. "
            "Install FFmpeg from Settings > Dependencies and try again."
        )


class FasterWhisperNotInstalledError(TranscriptionError):
    """Raised when faster-whisper is not installed."""

    def __init__(self):
        super().__init__(
            "faster-whisper is not installed. "
            "Install it with: pip install faster-whisper"
        )


class ModelDownloadError(TranscriptionError):
    """Raised when model download fails."""


@dataclass
class WordTimestamp:
    """A single word with start/end timestamps from ASR or forced alignment."""

    start: float
    end: float
    text: str
    probability: Optional[float] = None

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON export."""
        data: dict = {
            "start": self.start,
            "end": self.end,
            "text": self.text,
        }
        if self.probability is not None:
            data["probability"] = self.probability
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "WordTimestamp":
        """Deserialize from dictionary."""
        probability = data.get("probability")
        return cls(
            start=data.get("start", 0.0),
            end=data.get("end", 0.0),
            text=data.get("text", ""),
            probability=probability if probability is None else float(probability),
        )


@dataclass
class TranscriptSegment:
    """A segment of transcribed speech."""

    start_time: float
    end_time: float
    text: str
    confidence: float = 0.0
    words: Optional[list[WordTimestamp]] = None
    language: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON export."""
        data: dict = {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "text": self.text,
            "confidence": self.confidence,
        }
        if self.words is not None:
            data["words"] = [w.to_dict() for w in self.words]
        if self.language is not None:
            data["language"] = self.language
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "TranscriptSegment":
        """Deserialize from dictionary while preserving missing word-data state."""
        if "words" in data:
            raw_words = data["words"]
            words: Optional[list[WordTimestamp]] = (
                [WordTimestamp.from_dict(w) for w in raw_words]
                if raw_words is not None
                else None
            )
        else:
            words = None

        return cls(
            start_time=data.get("start_time", 0.0),
            end_time=data.get("end_time", 0.0),
            text=data.get("text", ""),
            confidence=data.get("confidence", 0.0),
            words=words,
            language=data.get("language"),
        )
