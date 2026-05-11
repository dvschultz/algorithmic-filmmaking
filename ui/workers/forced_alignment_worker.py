"""Background worker for word-level forced alignment.

Given a list of clips that already have transcripts (from U1's transcription
pass), this worker runs ``ctc-forced-aligner`` per clip to produce per-word
timestamps. Output is emitted to the main thread via ``clip_aligned``; the
*handler* on the main thread is responsible for writing the words back into
the project (per Scene Ripper's worker discipline — workers never mutate the
``Project`` or ``Clip`` model directly).

Why serial-per-clip? ``ctc-forced-aligner`` is CPU-bound on Apple Silicon
(MPS wav2vec2 is unreliable in 2026; the alignment runtime forces CPU + fp32
— see ``core/analysis/alignment.py``) and not thread-safe. Mirroring the
``TranscriptionWorker`` MLX backend constraint, this worker runs one clip at a
time.

Lazy imports: the heavy alignment dependencies (``torch``,
``ctc_forced_aligner``) are imported only inside ``core.analysis.alignment``,
and that module itself defers them to function bodies. This module is
importable on a clean Python environment without those deps installed, which
keeps the test path light.

Audio strategy: each clip's audio range is extracted to a temporary WAV via
FFmpeg (mirroring ``core.transcription.transcribe_clip`` at lines 663–685).
The temp file is cleaned up after alignment runs (success or failure).
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from PySide6.QtCore import Signal, Slot

from ui.workers.base import CancellableWorker, summarize_clip_errors

if TYPE_CHECKING:
    from models.clip import Clip, Source

logger = logging.getLogger(__name__)


def _extract_clip_audio_wav(
    source_path: Path,
    start_time: float,
    end_time: float,
) -> Path:
    """Extract a clip's audio range to a temporary 16 kHz mono PCM WAV.

    Mirrors the pattern in ``core.transcription.transcribe_clip``. The temp
    file is created here; the caller is responsible for cleanup via the
    ``try/finally`` around the alignment call.

    Raises:
        RuntimeError: FFmpeg is missing, fails to extract, or produces an
            empty output (e.g. source has no audio track).
    """
    from core.binary_resolver import (
        find_binary,
        get_subprocess_env,
        get_subprocess_kwargs,
    )

    ffmpeg = find_binary("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "FFmpeg is required for word-level alignment but was not found. "
            "Install FFmpeg from Settings > Dependencies and try again."
        )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    cmd = [
        ffmpeg, "-y",
        "-ss", str(float(start_time)),
        "-to", str(float(end_time)),
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
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        stderr = (
            exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
        )
        raise RuntimeError(
            f"FFmpeg failed to extract clip audio from {source_path}: "
            f"{stderr.strip() or exc}"
        ) from exc
    except FileNotFoundError as exc:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise RuntimeError(f"FFmpeg binary not found: {exc}") from exc

    if tmp_path.stat().st_size == 0:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise RuntimeError(
            f"FFmpeg produced an empty audio file for {source_path} "
            "(source may have no audio stream)."
        )

    return tmp_path


def _clip_needs_alignment(clip) -> bool:
    """True when any of the clip's transcript segments lacks word data.

    Per the plan's per-clip skip predicate: re-align the whole clip when any
    ``TranscriptSegment.words is None``. ``words == []`` (alignment ran and
    produced no words) is treated as "already aligned" — we deliberately
    don't conflate ``None`` with ``[]``.
    """
    transcript = getattr(clip, "transcript", None)
    if not transcript:
        # No transcript at all → cannot align; the worker skips this clip.
        return False
    for seg in transcript:
        if seg.words is None:
            return True
    return False


class ForcedAlignmentWorker(CancellableWorker):
    """Background worker for word-level forced alignment.

    Runs ``ctc-forced-aligner`` over a list of clips that already have
    transcripts. Each clip's audio is extracted to a temp WAV, alignment runs,
    and the resulting ``list[WordTimestamp]`` is emitted to the main thread.

    The worker does NOT mutate the ``Project`` or ``Clip`` model — the main
    thread's ``clip_aligned`` handler is responsible for writing the words
    back through the project (matching the ``TranscriptionWorker`` /
    ``transcript_ready`` pattern in ``ui/main_window.py``).

    Signals:
        progress: ``(current, total)`` after each clip is processed (and once
            before the loop with ``(0, total)`` so progress UIs can render
            their initial state, mirroring ``TranscriptionWorker``).
        clip_aligned: ``(clip_id, words)`` where ``words`` is a flat
            ``list[WordTimestamp]`` produced by ``align_words``. Segment
            boundaries are not preserved in the payload — the receiver
            distributes them across segments on the main thread.
        alignment_completed: Emitted exactly once when the worker finishes
            (success, cancellation, or fatal error — the pipeline can always
            advance on this signal).
        error: Inherited from ``CancellableWorker``. Emitted at most once at
            the end of the run with a summarized per-clip error message when
            one or more clips failed.

    Notes:
        - Serial per-clip; ``ctc-forced-aligner`` is not thread-safe.
        - Pre-loads the alignment model with progress reporting BEFORE the
          per-clip loop (pattern from ``TranscriptionWorker.run`` lines
          162-174). On Apple Silicon this front-loads any required model
          download/import cost.
        - Calls ``core.feature_registry.check_feature_ready('word_alignment')``
          and ``install_for_feature`` from inside ``run()``, never from the
          dialog/tab — this matches the staccato pattern.
    """

    progress = Signal(int, int)  # current, total
    clip_aligned = Signal(str, list)  # clip_id, list[WordTimestamp]
    alignment_completed = Signal()

    def __init__(
        self,
        clips: list["Clip"],
        sources_by_id: dict[str, "Source"],
        skip_existing: bool = True,
        parent=None,
    ) -> None:
        """
        Args:
            clips: Clips to align. Clips without transcripts, or whose every
                segment already has populated ``words`` (per the skip
                predicate), are filtered out before processing.
            sources_by_id: Lookup of ``Source.id`` → ``Source`` so the worker
                can resolve each clip's source media path and frame rate.
                Required; clips whose source is missing are skipped with a
                warning.
            skip_existing: When True (default), clips whose every transcript
                segment already has ``.words`` populated are skipped. When
                False, every clip with a transcript is re-aligned (the data
                is idempotent — alignment produces the same boundaries on
                repeated runs).
            parent: Qt parent.
        """
        super().__init__(parent)
        self._clips = list(clips or [])
        self._sources_by_id = sources_by_id or {}
        self._skip_existing = skip_existing

    @Slot()
    def run(self) -> None:
        """Execute alignment over the (filtered) clip set."""
        self._log_start()

        # Filter the input set: drop clips that have nothing to align, are
        # missing a source, or already have full word data (when
        # skip_existing).
        clips_to_process: list["Clip"] = []
        for clip in self._clips:
            if not getattr(clip, "transcript", None):
                continue
            if self._skip_existing and not _clip_needs_alignment(clip):
                continue
            source = self._sources_by_id.get(clip.source_id)
            if source is None:
                logger.warning(
                    "ForcedAlignmentWorker: clip %s has no source in sources_by_id; skipping",
                    clip.id,
                )
                continue
            clips_to_process.append(clip)

        total = len(clips_to_process)
        if total == 0:
            logger.info(
                "ForcedAlignmentWorker: no clips require alignment (input %d, after filtering 0)",
                len(self._clips),
            )
            self.alignment_completed.emit()
            self._log_complete()
            return

        # Feature-registry preflight INSIDE the worker (per staccato pattern).
        try:
            from core import feature_registry

            ready, missing = feature_registry.check_feature_ready("word_alignment")
            if not ready:
                logger.info(
                    "ForcedAlignmentWorker: word_alignment not ready (missing %s); "
                    "calling install_for_feature",
                    missing,
                )
                installed = feature_registry.install_for_feature("word_alignment")
                if not installed:
                    self.error.emit(
                        "Could not install word-level alignment dependencies. "
                        "Check Settings > Dependencies and try again."
                    )
                    self.alignment_completed.emit()
                    self._log_complete()
                    return
        except Exception as exc:  # noqa: BLE001 — feature_registry errors are user-facing
            logger.exception("ForcedAlignmentWorker: feature_registry preflight raised")
            self.error.emit(f"Word alignment dependencies unavailable: {exc}")
            self.alignment_completed.emit()
            self._log_complete()
            return

        # Pre-loop progress emission so progress UIs can show 0/total before
        # any per-clip work happens (mirrors TranscriptionWorker). The
        # alignment runtime itself was validated by
        # ``check_feature_ready('word_alignment')`` above — it internally
        # calls ``ensure_word_alignment_runtime_available()``. The model
        # checkpoints lazy-load on the first ``align_words`` call.
        self.progress.emit(0, total)

        if self.is_cancelled():
            self._log_cancelled()
            self.alignment_completed.emit()
            self._log_complete()
            return

        # Import the alignment entry point once. Module load is cheap; the
        # heavy imports live inside ``align_words`` itself.
        from core.analysis import alignment as alignment_mod

        errors: list[tuple[str, str]] = []
        completed = 0
        for clip in clips_to_process:
            # Cancellation gate at the TOP of each iteration — the moment
            # cancel() fires (from any thread, including the main thread via
            # a Cancel button), we stop emitting further clip_aligned signals
            # and exit cleanly. Already-aligned clips persist (partial
            # progress is real progress; the handler has already written
            # them back to the project).
            if self.is_cancelled():
                self._log_cancelled()
                break

            source = self._sources_by_id[clip.source_id]
            wav_path: Optional[Path] = None
            try:
                wav_path = _extract_clip_audio_wav(
                    Path(source.file_path),
                    clip.start_time(source.fps),
                    clip.end_time(source.fps),
                )

                if self.is_cancelled():
                    self._log_cancelled()
                    break

                words = alignment_mod.align_words(str(wav_path), clip.transcript)
                self.clip_aligned.emit(clip.id, list(words))
            except Exception as exc:  # noqa: BLE001 — surface per-clip errors
                self._log_error(str(exc), clip.id)
                errors.append((clip.id, str(exc)))
            finally:
                if wav_path is not None:
                    try:
                        wav_path.unlink(missing_ok=True)
                    except OSError as cleanup_exc:
                        logger.warning(
                            "Failed to remove temp alignment WAV %s: %s",
                            wav_path,
                            cleanup_exc,
                        )

            completed += 1
            self.progress.emit(completed, total)

        if errors:
            self.error.emit(
                summarize_clip_errors(errors, operation_label="Word alignment")
            )

        self.alignment_completed.emit()
        self._log_complete()


__all__ = ["ForcedAlignmentWorker"]
