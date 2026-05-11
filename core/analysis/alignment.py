"""Word-level forced alignment via ``ctc-forced-aligner``.

Given a clip's audio and the transcript segments produced for that clip, this
module returns word-level timestamps (``WordTimestamp[]``) in the *same* frame
of reference as the input segments (clip-relative seconds). Useful as a
post-step on transcripts produced by either ``faster-whisper`` (already has
word data — alignment refines accuracy) or ``lightning-whisper-mlx`` (no word
data at all — alignment is the only path to words).

Design notes:

- **Pure module**: no Qt imports, no project model imports. Inputs are an audio
  path plus pure-data ``TranscriptSegment`` objects; output is pure-data
  ``WordTimestamp`` objects.
- **Lazy heavy imports**: ``ctc_forced_aligner`` and ``torch`` are imported
  inside function bodies so the module loads on a clean Python environment
  (e.g. for unit tests that mock the engine). The model and its checkpoint(s)
  install on demand through ``core.feature_registry.install_for_feature(
  "word_alignment", ...)``.
- **Per-clip audio strategy**: this module mirrors ``core/transcription.py``'s
  ``transcribe_clip`` pattern — a small FFmpeg pass extracts the clip's audio
  to a temporary WAV (16 kHz mono pcm_s16le) before alignment. Returned
  timestamps are clip-relative seconds; no source-vs-clip offset translation
  is needed downstream.
- **Apple Silicon constraint**: forced CPU + fp32. MPS-backed ``wav2vec2``
  alignment is unreliable in 2026; promising MPS would silently degrade
  accuracy. Users on Apple Silicon pay the CPU latency cost; the worker layer
  is responsible for surfacing this to the UI.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from core.binary_resolver import find_binary, get_subprocess_env, get_subprocess_kwargs

if TYPE_CHECKING:
    # Type-only import; the runtime import lives inside ``align_words``.
    from core.transcription import TranscriptSegment, WordTimestamp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class AlignmentError(Exception):
    """Base exception for forced-alignment failures."""


class LanguageUnknownError(AlignmentError):
    """Raised when the transcript carries no language signal.

    Only legacy transcripts produced before U1 should hit this. Re-running
    transcription on the source repopulates ``TranscriptSegment.language``.
    """

    def __init__(self) -> None:
        super().__init__(
            "Cannot align: transcript carries no language information. "
            "Re-run transcription to populate the language field before aligning."
        )


class UnsupportedLanguageError(AlignmentError):
    """Raised when the alignment model does not support the transcript's language."""

    def __init__(self, language: str) -> None:
        self.language = language
        super().__init__(
            f"Forced alignment does not support language '{language}'. "
            "This source is unavailable for word-level sequencing."
        )


class AlignmentFFmpegError(AlignmentError):
    """Raised when FFmpeg fails to extract the clip audio for alignment."""


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def distribute_words_to_segments(
    segments: "list[TranscriptSegment]",
    words: "list[WordTimestamp]",
) -> None:
    """Assign a flat ``list[WordTimestamp]`` back onto transcript segments.

    The forced-alignment worker emits one flat word list per clip (alignment
    runs over the concatenated text). Callers need those words distributed
    back onto the parent ``TranscriptSegment[]`` so the existing per-segment
    word-data shape is preserved.

    For each word: assign to the first segment whose
    ``[start_time, end_time]`` contains the word's midpoint; if no segment
    contains it, fall back to the segment whose nearest boundary is closest
    to the midpoint. After distribution, *every* segment in ``segments``
    has its ``.words`` attribute assigned (to a — possibly empty — list).
    That empty-list-vs-None distinction is load-bearing: it lets the skip
    predicate treat a re-aligned clip as fully aligned even when individual
    segments end up wordless.

    Args:
        segments: The clip's transcript segments. Mutated in place — each
            segment's ``.words`` is reassigned. Empty input → no-op.
        words: Flat word list from alignment. Empty list is fine (every
            segment ends up with ``.words = []``).
    """
    if not segments:
        return

    per_segment: list[list] = [[] for _ in segments]
    n = len(segments)

    # Both inputs are time-sorted in practice; the per-word linear scan is
    # cheap enough that we keep the simple shape here (correctness over a
    # micro-optimization).
    for word in words:
        midpoint = (float(word.start) + float(word.end)) / 2.0
        chosen_idx: Optional[int] = None
        for i, seg in enumerate(segments):
            if seg.start_time <= midpoint <= seg.end_time:
                chosen_idx = i
                break
        if chosen_idx is None:
            chosen_idx = min(
                range(n),
                key=lambda i: min(
                    abs(midpoint - segments[i].start_time),
                    abs(midpoint - segments[i].end_time),
                ),
            )
        per_segment[chosen_idx].append(word)

    for seg, seg_words in zip(segments, per_segment):
        # Always assign — even an empty list means "alignment ran for this
        # segment", which is what the skip predicate checks.
        seg.words = seg_words


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


# Languages supported by the MMS-based ``ctc-forced-aligner`` model.
#
# Mirrors the ISO 639-1 / ISO 639-3 set covered by Meta's MMS-1B-ALL alignment
# model (the default backend the library ships with). The runtime canonical
# list lives inside ``ctc_forced_aligner`` (queried via
# ``ctc_forced_aligner.alignment_utils.SUPPORTED_LANGUAGES`` when the package is
# importable), but we keep a small, conservative subset here so the language
# pre-check can run without importing the heavy module.
#
# This static set is treated as authoritative for the unit-test path; if the
# real runtime can prove a language is also supported, ``_check_language_supported``
# delegates to the runtime when available.
_FALLBACK_SUPPORTED_ISO_639_1: frozenset[str] = frozenset(
    {
        # A small, high-confidence subset — used only when the runtime
        # SUPPORTED_LANGUAGES lookup is unavailable.
        "en", "fr", "de", "es", "it", "pt", "nl", "pl", "ru", "uk", "cs",
        "sv", "no", "da", "fi", "tr", "el", "ro", "hu", "bg", "hr", "sk",
        "sl", "lt", "lv", "et", "ar", "fa", "he", "hi", "bn", "ta", "te",
        "ml", "kn", "mr", "gu", "pa", "ur", "ja", "ko", "zh", "vi", "th",
        "id", "ms", "tl", "sw", "yo", "ig", "ha", "am",
    }
)


def align_words(
    audio_path: str,
    transcript_segments: "list[TranscriptSegment]",
) -> "list[WordTimestamp]":
    """Produce word-level timestamps for a clip's audio.

    Args:
        audio_path: Path to the clip's source media (video or audio file). The
            function extracts a 16 kHz mono WAV from this path via FFmpeg
            before running the alignment model. ``WordTimestamp.start`` /
            ``.end`` returned here are clip-relative seconds — the same frame
            of reference as ``TranscriptSegment.start_time``.
        transcript_segments: The transcript for the clip. Language is read off
            ``transcript_segments[0].language``; passing legacy segments with
            ``language is None`` raises ``LanguageUnknownError``.

    Returns:
        A list of ``WordTimestamp`` objects in chronological order. The list is
        flat across the input segments — segment boundaries are not preserved
        in the output (alignment operates on the concatenated text). Empty
        input segments → empty result; the alignment model is not loaded and
        no FFmpeg extraction is attempted.

    Raises:
        LanguageUnknownError: ``transcript_segments[0].language is None``.
        UnsupportedLanguageError: language is not supported by the model.
        AlignmentFFmpegError: FFmpeg failed to extract audio.
        FileNotFoundError: audio path does not exist.
        AlignmentError: any other alignment failure.
    """
    # Fast paths first: avoid loading the heavy model when there's nothing to do.
    if not transcript_segments:
        return []

    # Use the local import so test code can monkeypatch the model entry points
    # without dragging the heavy deps into ``sys.modules`` at module load.
    from core.transcription import WordTimestamp  # local-import: avoid cycles at module load

    language = transcript_segments[0].language
    if language is None:
        raise LanguageUnknownError()

    # Validate the language up-front so users don't pay the FFmpeg + model-load
    # cost before discovering the source can't be aligned.
    _check_language_supported(language)

    # Strip empty segments — alignment requires text. If every segment is
    # empty, return [] (consistent with the empty-input fast path).
    non_empty_segments = [
        seg for seg in transcript_segments if (seg.text or "").strip()
    ]
    if not non_empty_segments:
        return []

    audio_path_obj = Path(audio_path)
    if not audio_path_obj.exists():
        raise FileNotFoundError(f"Audio path does not exist: {audio_path}")

    # Build the concatenated transcript for the clip. The library expects a
    # single text string per audio; segment boundaries are reconstructed via
    # the returned word timings.
    full_text = " ".join((seg.text or "").strip() for seg in non_empty_segments).strip()
    if not full_text:
        return []

    wav_path = extract_audio_to_wav(audio_path_obj)
    try:
        raw_word_timings = _run_alignment_engine(
            wav_path=wav_path,
            text=full_text,
            language=language,
        )
    finally:
        # Always clean up the temp WAV, even if alignment raises.
        try:
            wav_path.unlink(missing_ok=True)
        except OSError as cleanup_exc:
            logger.warning("Failed to remove temp alignment WAV %s: %s", wav_path, cleanup_exc)

    return [
        WordTimestamp(
            start=float(entry["start"]),
            end=float(entry["end"]),
            text=str(entry["text"]),
            probability=(
                float(entry["score"]) if entry.get("score") is not None else None
            ),
        )
        for entry in raw_word_timings
    ]


# ---------------------------------------------------------------------------
# Runtime gating (called from feature_registry._validate_feature_runtime)
# ---------------------------------------------------------------------------


def ensure_word_alignment_runtime_available():
    """Validate that the forced-alignment runtime imports cleanly.

    Called from ``core.feature_registry._validate_feature_runtime`` for the
    ``word_alignment`` feature. Mirrors the shape of
    ``ensure_audio_analysis_runtime_available`` and
    ``ensure_faster_whisper_runtime_available``.
    """
    try:
        import ctc_forced_aligner  # noqa: F401
        import torch  # noqa: F401

        return ctc_forced_aligner
    except Exception as e:  # pragma: no cover - exercised via mock
        raise RuntimeError(f"word alignment runtime is incomplete: {e}") from e


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _check_language_supported(language: str) -> None:
    """Raise ``UnsupportedLanguageError`` when the language can't be aligned.

    Tries the runtime's authoritative ``SUPPORTED_LANGUAGES`` map first; falls
    back to a conservative static list when the runtime isn't installed. This
    lets the test path validate language gating without requiring the heavy
    dependency.
    """
    code = (language or "").lower().strip()
    if not code:
        raise UnsupportedLanguageError(language)

    try:
        # Local import: heavy dep is optional at module load time.
        from ctc_forced_aligner.alignment_utils import (
            SUPPORTED_LANGUAGES,  # type: ignore[attr-defined]
        )
    except Exception:
        # ctc-forced-aligner isn't installed (or its API changed). Use the
        # static fallback so language gating still works in the test / pre-install
        # path. The runtime will raise its own clear error later if the language
        # is rejected by the real model.
        if code not in _FALLBACK_SUPPORTED_ISO_639_1:
            raise UnsupportedLanguageError(language)
        return

    # Runtime present: SUPPORTED_LANGUAGES is typically a dict whose keys cover
    # both ISO 639-1 and ISO 639-3 codes. Accept either; library raises KeyError
    # internally on unsupported codes, but we want to fail with our own
    # exception type so callers can branch cleanly.
    try:
        if code in SUPPORTED_LANGUAGES:
            return
        # Some versions key by ISO 639-3 only; check the values too.
        try:
            if any(code == str(v).lower() for v in SUPPORTED_LANGUAGES.values()):
                return
        except Exception:
            pass
    except TypeError:
        # SUPPORTED_LANGUAGES is not a mapping/iterable as expected. Fall
        # through to the unsupported branch.
        pass

    raise UnsupportedLanguageError(language)


def _require_ffmpeg() -> str:
    """Return the resolved FFmpeg path or raise ``AlignmentFFmpegError``."""
    ffmpeg_path = find_binary("ffmpeg")
    if ffmpeg_path is None:
        raise AlignmentFFmpegError(
            "FFmpeg is required for forced alignment but was not found. "
            "Install FFmpeg from Settings > Dependencies and try again."
        )
    return ffmpeg_path


def extract_audio_to_wav(
    source_path: Path,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> Path:
    """Extract source audio to a temporary 16 kHz mono PCM WAV.

    Mirrors the pattern in ``core.transcription.transcribe_clip``. The temp
    file is created here; the caller is responsible for cleanup via the
    ``try/finally`` around the alignment call.

    Args:
        source_path: Path to the source media (video or audio file).
        start_time: Optional start time (seconds) for a sub-range extraction.
            When provided alongside ``end_time``, ``-ss`` / ``-to`` are
            passed to FFmpeg so only the named range is decoded. ``None``
            extracts the whole file.
        end_time: Optional end time (seconds) — see ``start_time``.

    Returns:
        Path to a newly-created temporary WAV file. The caller owns cleanup.

    Raises:
        AlignmentFFmpegError: FFmpeg is missing, failed, or produced no audio.
    """
    ffmpeg = _require_ffmpeg()

    # NamedTemporaryFile with ``delete=False`` because we need the path
    # after the file handle closes. Cleanup happens in the caller's finally.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    cmd: list[str] = [ffmpeg, "-y"]
    if start_time is not None:
        cmd += ["-ss", str(float(start_time))]
    if end_time is not None:
        cmd += ["-to", str(float(end_time))]
    cmd += [
        "-i", str(source_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(tmp_path),
    ]

    try:
        subprocess.run(
            cmd,
            capture_output=True,
            check=True,
            env=get_subprocess_env(),
            **get_subprocess_kwargs(),
        )
    except subprocess.CalledProcessError as exc:
        # Clean up the empty/partial temp file before re-raising — we own it.
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
        raise AlignmentFFmpegError(
            f"FFmpeg failed to extract audio from {source_path}: {stderr.strip() or exc}"
        ) from exc
    except FileNotFoundError as exc:
        # FFmpeg vanished between resolution and exec — rare but possible.
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise AlignmentFFmpegError(f"FFmpeg binary not found: {exc}") from exc

    if tmp_path.stat().st_size == 0:
        # Defensive: FFmpeg can succeed but produce no audio (e.g. video-only
        # input). Treat the same as a failed extract.
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise AlignmentFFmpegError(
            f"FFmpeg produced an empty audio file for {source_path} "
            "(source may have no audio stream)."
        )

    return tmp_path


def _run_alignment_engine(
    wav_path: Path,
    text: str,
    language: str,
) -> list[dict]:
    """Run ``ctc-forced-aligner`` against the extracted audio + text.

    All heavy imports happen here so the module-level import of
    ``core.analysis.alignment`` does not pay the ``torch`` / ``ctc_forced_aligner``
    startup cost — and so unit tests can monkeypatch this function to bypass
    the real engine entirely.

    Returns:
        A list of ``{"start": float, "end": float, "text": str, "score": float | None}``
        dicts in chronological order. ``start`` / ``end`` are clip-relative
        seconds. ``score`` is the per-word alignment score when the library
        provides one; otherwise ``None``.
    """
    # Local imports so module-level load is cheap.
    try:
        import torch  # type: ignore[import-not-found]
    except Exception as exc:
        raise AlignmentError(
            "PyTorch is required for forced alignment but is not installed. "
            "Install the 'word_alignment' feature first."
        ) from exc

    try:
        from ctc_forced_aligner import (  # type: ignore[import-not-found]
            generate_emissions,
            get_alignments,
            get_spans,
            load_alignment_model,
            load_audio,
            postprocess_results,
            preprocess_text,
        )
    except Exception as exc:
        raise AlignmentError(
            "ctc-forced-aligner is not installed. "
            "Install the 'word_alignment' feature first."
        ) from exc

    # CPU + fp32 — see module docstring. MPS wav2vec2 forced alignment is
    # unreliable in 2026; do not promise it.
    device = "cpu"
    dtype = torch.float32

    model, alignment_tokenizer = load_alignment_model(device=device, dtype=dtype)

    audio_waveform = load_audio(str(wav_path), dtype=dtype, device=device)

    emissions, stride = generate_emissions(model, audio_waveform)

    # ``preprocess_text`` signature varies slightly across versions of the
    # library; pass the parameters the public docs call out and rely on the
    # library's defaults for the rest.
    tokens_starred, text_starred = preprocess_text(
        text,
        romanize=True,
        language=language,
    )

    segments, scores, blank_id = get_alignments(
        emissions,
        tokens_starred,
        alignment_tokenizer,
    )

    spans = get_spans(tokens_starred, segments, blank_id)

    word_timestamps = postprocess_results(
        text_starred,
        spans,
        stride,
        scores,
    )

    return word_timestamps


__all__ = [
    "AlignmentError",
    "AlignmentFFmpegError",
    "LanguageUnknownError",
    "UnsupportedLanguageError",
    "align_words",
    "distribute_words_to_segments",
    "ensure_word_alignment_runtime_available",
    "extract_audio_to_wav",
]
