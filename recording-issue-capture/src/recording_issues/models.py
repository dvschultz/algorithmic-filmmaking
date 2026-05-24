from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

AUDIO_EXTENSIONS = {".m4a", ".mp3", ".wav", ".aac", ".flac", ".ogg", ".webm"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}
ISSUE_LABELS = ["idea", "recording-capture"]
PROMPT_VERSION = "provenance-v2"
PROVENANCE_KINDS = {"explicit", "observed", "inferred"}


@dataclass(frozen=True)
class RecordingMedia:
    path: Path
    kind: str
    dedupe_key: str


@dataclass(frozen=True)
class MediaChunk:
    index: int
    start_time: float
    end_time: float


@dataclass
class TranscriptSegment:
    source_path: str
    source_name: str
    start_time: float
    end_time: float
    text: str
    confidence: float = 0.0
    language: str | None = None


@dataclass
class TranscriptChunk:
    source_name: str
    source_path: str
    start_time: float
    end_time: float
    text: str
    segments: list[TranscriptSegment] = field(default_factory=list)


@dataclass
class IssueCandidate:
    title: str
    summary: str
    source_path: str
    source_name: str
    start_time: float | None = None
    end_time: float | None = None
    evidence: str = ""
    exact_quote: str = ""
    provenance_kind: str = "unverified"
    provenance_notes: str = ""
    validation_status: str = "unvalidated"
    validation_errors: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=lambda: ISSUE_LABELS.copy())
    acceptance_criteria: list[str] = field(default_factory=list)
    implementation_notes: str = ""
    confidence: float = 0.0
