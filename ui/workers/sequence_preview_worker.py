"""Background worker for cached sequence preview rendering."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Signal

from core.sequence_preview import (
    SequencePreviewSettings,
    cleanup_sequence_preview_cache,
    render_sequence_preview,
)
from ui.workers.base import CancellableWorker


class SequencePreviewWorker(CancellableWorker):
    """Render a continuous timeline preview in the background."""

    progress = Signal(float, str)
    preview_completed = Signal(object, str, str, bool)  # path, signature, profile label, from cache

    def __init__(
        self,
        sequence,
        sources: dict,
        clips: dict,
        frames: Optional[dict] = None,
        cache_root: Optional[Path] = None,
        settings: Optional[SequencePreviewSettings] = None,
    ):
        super().__init__()
        self.sequence = copy.deepcopy(sequence)
        self.sources = dict(sources)
        self.clips = dict(clips)
        self.frames = dict(frames or {})
        self.cache_root = cache_root
        self.settings = settings or SequencePreviewSettings()

    def run(self):
        self._log_start()
        try:
            if self.is_cancelled():
                self._log_cancelled()
                return
            result = render_sequence_preview(
                sequence=self.sequence,
                sources=self.sources,
                clips=self.clips,
                frames=self.frames,
                cache_root=self.cache_root,
                settings=self.settings,
                progress_callback=lambda p, m: self.progress.emit(p, m),
            )
            if self.is_cancelled():
                self._log_cancelled()
                return
            cleanup_sequence_preview_cache(self.cache_root)
            self.preview_completed.emit(
                result.path,
                result.signature,
                result.profile_label,
                result.from_cache,
            )
            self._log_complete()
        except Exception as exc:
            if not self.is_cancelled():
                self._log_error(str(exc))
                self.error.emit(str(exc))
