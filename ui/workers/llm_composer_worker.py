"""Background worker for the LLM Word Composer.

Runs ``core.remix.word_llm_composer.generate_llm_word_sequence`` off the GUI
thread. The composer makes a single HTTP call to a local Ollama server (via
the ``compose_with_llm`` helper) which can take many seconds at corpus scale,
so a worker is required to keep the dialog responsive.

Cancellation contract: ``httpx``'s blocking client does not expose a cancel
token to non-cooperative callers. We approximate cancellation by checking
``self.is_cancelled()`` immediately before and after the spine call and
silently dropping the emitted ``SequenceClip`` payload if the user has asked
to stop. The in-flight HTTP request itself runs to completion (or to its
configured timeout); we trade a small amount of wasted compute for a simple,
reliable cancellation semantic.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from PySide6.QtCore import Signal, Slot

from ui.workers.base import CancellableWorker

logger = logging.getLogger(__name__)


__all__ = ["LLMComposerWorker"]


class LLMComposerWorker(CancellableWorker):
    """Background worker for LLM-composed word sequencing.

    Signals:
        progress: ``(current, total)``. Emits ``(0, 1)`` before the call and
            ``(1, 1)`` immediately after a successful call. Cancellation
            paths skip the final tick.
        sequence_ready: Emits ``list[SequenceClip]`` produced by
            ``generate_llm_word_sequence``. Suppressed when cancelled.
        error: Inherited from ``CancellableWorker``. Emits a single
            human-readable message for any of: ``MissingWordDataError``,
            ``OllamaUnreachableError``, ``LLMEmptyResponseError``, plain
            ``ValueError`` (empty corpus / OOV / bad fps), or any other
            unexpected failure.
    """

    progress = Signal(int, int)
    sequence_ready = Signal(list)

    def __init__(
        self,
        clips: list[tuple[Any, Any]],
        prompt: str,
        target_length: int,
        repeat_policy: str = "round-robin",
        seed: Optional[int] = None,
        handle_frames: int = 0,
        *,
        model: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.7,
        timeout: float = 120.0,
        system_prompt: Optional[str] = None,
        think: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._clips = list(clips or [])
        self._prompt = prompt
        self._target_length = int(target_length)
        self._repeat_policy = repeat_policy
        self._seed = seed
        self._handle_frames = int(handle_frames)
        self._model = model
        self._api_base = api_base
        self._temperature = float(temperature)
        self._timeout = float(timeout)
        self._system_prompt = system_prompt
        self._think = think

    @Slot()
    def run(self) -> None:
        self._log_start()
        try:
            if self.is_cancelled():
                self._log_cancelled()
                return

            # Coarse progress: a single tick before and after the call. The
            # underlying composer doesn't expose intermediate progress.
            self.progress.emit(0, 1)

            from core.remix.word_llm_composer import generate_llm_word_sequence

            sequence_clips = generate_llm_word_sequence(
                self._clips,
                prompt=self._prompt,
                target_length=self._target_length,
                repeat_policy=self._repeat_policy,
                seed=self._seed,
                handle_frames=self._handle_frames,
                model=self._model,
                api_base=self._api_base,
                temperature=self._temperature,
                timeout=self._timeout,
                system_prompt=self._system_prompt,
                think=self._think,
            )

            if self.is_cancelled():
                self._log_cancelled()
                return

            self.progress.emit(1, 1)
            # PySide6 marshals ``list`` over queued connections by value;
            # the receiver gets its own copy.
            self.sequence_ready.emit(list(sequence_clips))
        except Exception as exc:  # noqa: BLE001
            if self.is_cancelled():
                self._log_cancelled()
                return
            logger.exception("LLMComposerWorker run failed")
            self.error.emit(str(exc))
        finally:
            self._log_complete()
