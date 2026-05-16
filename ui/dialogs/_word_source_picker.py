"""Shared source-picker scaffolding for the two word-sequencer dialogs.

This module hosts the parts the ``WordSequencerDialog`` (U6 preset modes)
and ``WordLLMComposerDialog`` (U6 LLM composer) used to duplicate verbatim:

- The four per-source badge constants (``BADGE_ALIGNED`` etc.) and the
  ``classify_source_alignment`` helper that picks one for a given source.
- ``WordAlignmentController`` — a small ``QObject`` that owns the
  ``ForcedAlignmentWorker`` lifecycle. The two dialogs each instantiate one
  of these instead of duplicating the worker wiring (start, progress,
  per-clip distribution back onto transcript segments, completion, error,
  cancel). The QListWidget population stays per-dialog because the row
  formatting is the same shape but minor layout details belong to the
  dialog that owns the widget.

Why ``QObject`` and not a free-standing utility? The controller needs to
emit Qt signals (``progress``, ``error``, ``completed``) back to the
dialog, and we want ``Qt.UniqueConnection`` semantics on those signals so
the dialog's duplicate-signal guard pattern still works.

The module imports PySide6 — it is a UI-layer module, not part of the
GUI-agnostic spine.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from PySide6.QtCore import QObject, Qt, Signal, Slot

logger = logging.getLogger(__name__)


__all__ = [
    "BADGE_ALIGNED",
    "BADGE_MISSING_FPS",
    "BADGE_NEEDS_ALIGNMENT",
    "BADGE_UNSUPPORTED_LANGUAGE",
    "WordAlignmentController",
    "alignable_pending_clips",
    "classify_source_alignment",
    "format_source_row",
]


# ---------------------------------------------------------------------------
# Per-source badge constants — replace the underscore-private names that
# were previously imported across dialog modules.
# ---------------------------------------------------------------------------

BADGE_ALIGNED = "aligned"
BADGE_NEEDS_ALIGNMENT = "needs_alignment"
BADGE_UNSUPPORTED_LANGUAGE = "unsupported_language"
BADGE_MISSING_FPS = "missing_fps"


def _language_is_supported(language: str) -> bool:
    """Return whether the alignment stack supports ``language``.

    This indirection keeps tests able to stub the gate without importing the
    heavy optional alignment runtime.
    """
    from core.analysis.alignment import is_language_supported
    return bool(is_language_supported(language))


def classify_source_alignment(
    clips_for_source: list[tuple[Any, Any]],
) -> tuple[str, Optional[str]]:
    """Return ``(badge_key, language)`` describing a source's alignment status.

    The badge key is one of the ``BADGE_*`` constants. The language string
    is the first transcript language found across the source's clips, or
    ``None`` when no clip has surfaced one.

    Args:
        clips_for_source: ``[(Clip, Source), ...]`` for a single source.

    Returns:
        ``(badge_key, language)``. Possible badges, in priority order:

        - ``BADGE_MISSING_FPS`` — the source has no ``fps`` (we can't convert
          word boundaries to frames; the source is hard-unavailable).
        - ``BADGE_UNSUPPORTED_LANGUAGE`` — the alignment model rejects the
          detected language; source is unavailable for word sequencing.
        - ``BADGE_NEEDS_ALIGNMENT`` — at least one segment has
          ``words is None`` (i.e. needs alignment before sequencing).
        - ``BADGE_ALIGNED`` — every segment already has word data.
    """
    language: Optional[str] = None
    needs_alignment = False
    for clip, source in clips_for_source:
        if getattr(source, "fps", None) in (None, 0):
            return BADGE_MISSING_FPS, language
        transcript = getattr(clip, "transcript", None) or []
        for seg in transcript:
            seg_lang = getattr(seg, "language", None)
            if seg_lang and language is None:
                language = seg_lang
            if getattr(seg, "words", None) is None:
                needs_alignment = True

    if language:
        try:
            if not _language_is_supported(language):
                return BADGE_UNSUPPORTED_LANGUAGE, language
        except Exception:
            # Fail open — if the gate itself errored we don't want to lock
            # users out of every source.
            pass

    if needs_alignment:
        return BADGE_NEEDS_ALIGNMENT, language
    return BADGE_ALIGNED, language


def format_source_row(
    source: Any,
    src_clips: list[tuple[Any, Any]],
    badge_key: str,
    language: Optional[str],
) -> str:
    """Format the shared word-source picker row label."""
    filename = (
        getattr(source, "filename", None)
        or str(getattr(source, "file_path", "")) or "unknown"
    )
    duration = getattr(source, "duration_seconds", None)
    if duration is None:
        duration_str = getattr(source, "duration_str", "?")
    else:
        duration_str = f"{float(duration):.1f}s"

    if badge_key == BADGE_ALIGNED:
        badge = "✓ aligned"
    elif badge_key == BADGE_NEEDS_ALIGNMENT:
        badge = "… needs alignment"
    elif badge_key == BADGE_UNSUPPORTED_LANGUAGE:
        badge = f"⚠ unsupported language ({language})"
    else:
        badge = "⚠ missing fps"
    return f"{filename}  ·  {duration_str}  ·  {badge}  ·  {len(src_clips)} clip(s)"


def alignable_pending_clips(checked_clips: list[tuple[Any, Any]]) -> list:
    """Return checked ``Clip`` objects missing word-level data."""
    pending: list = []
    for clip, _source in checked_clips:
        transcript = getattr(clip, "transcript", None) or []
        if any(getattr(seg, "words", None) is None for seg in transcript):
            pending.append(clip)
    return pending


# ---------------------------------------------------------------------------
# WordAlignmentController — wraps the worker lifecycle the two dialogs
# share.
# ---------------------------------------------------------------------------


class WordAlignmentController(QObject):
    """Run forced alignment over a list of clips and emit lifecycle signals.

    Both word dialogs need exactly the same alignment-handling logic: spin
    up a ``ForcedAlignmentWorker``, route per-clip ``clip_aligned`` payloads
    back onto transcript segments via
    ``core.analysis.alignment.distribute_words_to_segments``, expose
    progress / error / completion signals to the dialog, and support
    cancellation.

    The controller owns the worker's lifetime. The dialog connects to
    ``progress``, ``error``, and ``completed`` and remains responsible for
    its own UI state (which stack page is showing, what error label says,
    etc.). The controller does NOT mutate the dialog.

    Signals:
        progress: ``(current, total)`` — proxied straight from the worker.
        error: ``(message)`` — proxied straight from the worker. May fire
            zero or one times before ``completed``.
        completed: emitted exactly once at the end of a run (success,
            cancellation, or fatal error).

    Lifecycle methods are :meth:`start` and :meth:`cancel`. :meth:`wait`
    is a convenience for the dialog's ``closeEvent``.
    """

    progress = Signal(int, int)
    error = Signal(str)
    completed = Signal()

    def __init__(self, dialog_clips: list[tuple[Any, Any]], parent: QObject = None):
        """
        Args:
            dialog_clips: The dialog's full ``[(Clip, Source), ...]`` set.
                The controller uses it to look up the ``Clip`` object for a
                payload's ``clip_id`` (so it can mutate the matching
                clip's transcript). Workers can't carry the project model
                themselves, so each dialog passes in its known selection.
            parent: Qt parent (typically the dialog).
        """
        super().__init__(parent)
        self._dialog_clips = list(dialog_clips or [])
        self._worker = None
        self._finished_handled = False

    def is_running(self) -> bool:
        worker = self._worker
        return worker is not None and worker.isRunning()

    def start(
        self,
        pending_clips: list,
        sources_by_id: dict,
    ) -> None:
        """Spawn a ``ForcedAlignmentWorker`` over the given clips.

        Args:
            pending_clips: Clips that need alignment (filtered by the
                dialog from its checked selection).
            sources_by_id: Source-id → ``Source`` lookup the worker uses to
                resolve clip audio paths.
        """
        from ui.workers.forced_alignment_worker import ForcedAlignmentWorker

        self._finished_handled = False

        worker = ForcedAlignmentWorker(
            clips=pending_clips,
            sources_by_id=sources_by_id,
            skip_existing=True,
            parent=self,
        )
        worker.progress.connect(self._on_worker_progress, Qt.UniqueConnection)
        worker.clip_aligned.connect(self._on_worker_clip_aligned, Qt.UniqueConnection)
        worker.alignment_completed.connect(
            self._on_worker_completed, Qt.UniqueConnection,
        )
        worker.error.connect(self._on_worker_error, Qt.UniqueConnection)
        self._worker = worker
        worker.start()

    def cancel(self) -> None:
        worker = self._worker
        if worker is not None and worker.isRunning():
            worker.cancel()

    def wait(self, msecs: int = 50) -> None:
        worker = self._worker
        if worker is None:
            return
        try:
            worker.wait(msecs)
        except Exception:  # noqa: BLE001 — best-effort, dialog is closing
            pass

    # -- Slots --------------------------------------------------------------

    @Slot(int, int)
    def _on_worker_progress(self, current: int, total: int) -> None:
        self.progress.emit(current, total)

    @Slot(str, list)
    def _on_worker_clip_aligned(self, clip_id: str, words: list) -> None:
        """Find the matching clip and distribute words onto its segments."""
        clip = None
        for c, _ in self._dialog_clips:
            if getattr(c, "id", None) == clip_id:
                clip = c
                break
        if clip is None or not getattr(clip, "transcript", None):
            logger.warning(
                "WordAlignmentController: clip %s no longer present; dropping payload",
                clip_id,
            )
            return

        # The actual distribute logic lives in core.analysis.alignment so the
        # analyze-tab path and both dialogs share one implementation.
        from core.analysis.alignment import distribute_words_to_segments
        distribute_words_to_segments(list(clip.transcript), words)

    @Slot(str)
    def _on_worker_error(self, message: str) -> None:
        self.error.emit(message)

    @Slot()
    def _on_worker_completed(self) -> None:
        if self._finished_handled:
            return
        self._finished_handled = True
        self.wait(50)
        self._worker = None
        self.completed.emit()
