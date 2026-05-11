"""LLM Word Composer dialog — prompt-driven, corpus-constrained sequencing.

Lets the user pick the source(s) whose word inventory to draw from, type a
generation prompt, and run a local LLM (Ollama) whose decoding is constrained
to that exact word vocabulary. The result is emitted as
``list[SequenceClip]`` materialized through
``core.remix.word_llm_composer.generate_llm_word_sequence``.

Behaviour mirrors ``WordSequencerDialog`` — same source picker, same
missing-alignment auto-run, same inline error / progress UX — but the
generation step calls into a separate worker
(``ui.workers.llm_composer_worker.LLMComposerWorker``) since the Ollama call
can take many seconds at corpus scale.

Ollama health is probed once at dialog construction via
``check_ollama_health``. When Ollama is unreachable the dialog renders an
explanatory message + a "Recheck" button instead of the normal form, so the
user gets a clear next step without a modal trap.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from core.remix.word_llm_composer import generate_llm_word_sequence  # noqa: F401 - re-exported symbol used in tests
from core.remix.word_sequencer import MissingWordDataError
from ui.dialogs.word_sequencer_dialog import (
    _BADGE_ALIGNED,
    _BADGE_MISSING_FPS,
    _BADGE_NEEDS_ALIGNMENT,
    _BADGE_UNSUPPORTED_LANGUAGE,
    _classify_source,
)
from ui.theme import theme, Spacing, TypeScale, UISizes

logger = logging.getLogger(__name__)


__all__ = ["WordLLMComposerDialog"]


_REPEAT_POLICIES = [
    ("round-robin", "Round-robin"),
    ("random", "Random (with seed)"),
    ("first", "First"),
    ("longest", "Longest"),
    ("shortest", "Shortest"),
]


def _check_ollama_health_sync(api_base: str = "http://localhost:11434") -> tuple[bool, str]:
    """Synchronously call the async ``check_ollama_health`` helper."""
    try:
        from core.llm_client import check_ollama_health
    except Exception as exc:  # noqa: BLE001
        return False, f"LLM client unavailable: {exc}"
    try:
        return asyncio.run(check_ollama_health(api_base))
    except RuntimeError:
        # Already inside an event loop (e.g. async test harness). Skip the
        # probe — the worker call will surface a clear error if needed.
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


class WordLLMComposerDialog(QDialog):
    """Modal dialog for the LLM Word Composer."""

    sequence_ready = Signal(list)

    def __init__(
        self,
        clips: list[tuple[Any, Any]],
        project: Any = None,
        parent=None,
        *,
        _ollama_health_fn=None,
    ) -> None:
        super().__init__(parent)
        self._clips = list(clips or [])
        self._project = project
        self._health_probe = _ollama_health_fn or _check_ollama_health_sync

        # Worker state.
        self._alignment_worker = None
        self._alignment_finished_handled = False
        self._compose_worker = None
        self._compose_finished_handled = False
        self._pending_after_alignment = False
        self._ollama_healthy = False

        self._sources_by_id: dict[str, Any] = {}
        self._clips_by_source_id: dict[str, list[tuple[Any, Any]]] = {}
        for clip, source in self._clips:
            source_id = getattr(source, "id", None)
            if source_id is None:
                continue
            self._sources_by_id.setdefault(source_id, source)
            self._clips_by_source_id.setdefault(source_id, []).append((clip, source))

        self._source_status: dict[str, tuple[str, Optional[str]]] = {
            sid: _classify_source(s_clips)
            for sid, s_clips in self._clips_by_source_id.items()
        }

        self.setWindowTitle("LLM Word Composer")
        self.setMinimumWidth(560)
        self.setMinimumHeight(580)
        self._setup_ui()
        self._populate_source_picker()
        self._on_repeat_policy_changed(self._repeat_combo.currentIndex())
        self._probe_ollama()
        self._refresh_validation()

    # ------------------------------------------------------------------ UI

    def _setup_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 24, 24, 24)
        outer.setSpacing(Spacing.MD)

        title = QLabel("LLM Word Composer")
        title.setStyleSheet(f"font-size: {TypeScale.XL}px; font-weight: bold;")
        outer.addWidget(title)

        desc = QLabel(
            "Write a prompt; a local LLM drafts a sequence using only words "
            "from your corpus. Decoding is constrained to your vocabulary, so "
            "every word maps to a real clip."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color: {theme().text_secondary};")
        outer.addWidget(desc)

        self._stack = QStackedWidget()
        outer.addWidget(self._stack, 1)

        self._stack.addWidget(self._build_form_page())
        self._stack.addWidget(self._build_progress_page())
        self._stack.addWidget(self._build_ollama_absent_page())
        self._stack.setCurrentIndex(0)

    def _build_form_page(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Spacing.MD)

        # --- Source picker ------------------------------------------------
        sources_label = QLabel("Sources")
        sources_label.setStyleSheet(
            f"color: {theme().text_secondary}; font-size: {TypeScale.SM}px;"
        )
        layout.addWidget(sources_label)
        self._source_list = QListWidget()
        self._source_list.setMinimumHeight(120)
        self._source_list.itemChanged.connect(self._refresh_validation)
        layout.addWidget(self._source_list)

        # --- Prompt -------------------------------------------------------
        prompt_label = QLabel("Prompt")
        prompt_label.setStyleSheet(
            f"color: {theme().text_secondary}; font-size: {TypeScale.SM}px;"
        )
        layout.addWidget(prompt_label)
        self._prompt_input = QPlainTextEdit()
        self._prompt_input.setPlaceholderText(
            "e.g., 'compose a sentence about silence'"
        )
        self._prompt_input.setMinimumHeight(80)
        self._prompt_input.textChanged.connect(self._refresh_validation)
        layout.addWidget(self._prompt_input)

        # --- Target length -----------------------------------------------
        length_row = QHBoxLayout()
        length_label = QLabel("Target length")
        length_label.setMinimumWidth(UISizes.FORM_LABEL_WIDTH)
        length_row.addWidget(length_label)
        self._length_spin = QSpinBox()
        self._length_spin.setMinimumHeight(UISizes.LINE_EDIT_MIN_HEIGHT)
        self._length_spin.setRange(1, 200)
        self._length_spin.setValue(20)
        length_row.addWidget(self._length_spin)
        length_row.addStretch()
        layout.addLayout(length_row)

        # --- Repeat policy ------------------------------------------------
        policy_row = QHBoxLayout()
        policy_label = QLabel("Repeat policy")
        policy_label.setMinimumWidth(UISizes.FORM_LABEL_WIDTH)
        policy_row.addWidget(policy_label)
        self._repeat_combo = QComboBox()
        self._repeat_combo.setMinimumHeight(UISizes.COMBO_BOX_MIN_HEIGHT)
        self._repeat_combo.setMinimumWidth(UISizes.COMBO_BOX_MIN_WIDTH)
        for key, label in _REPEAT_POLICIES:
            self._repeat_combo.addItem(label, key)
        self._repeat_combo.currentIndexChanged.connect(
            self._on_repeat_policy_changed
        )
        policy_row.addWidget(self._repeat_combo, 1)
        layout.addLayout(policy_row)

        # --- Seed (visible only with random policy) ----------------------
        self._seed_container = QWidget()
        seed_row = QHBoxLayout(self._seed_container)
        seed_row.setContentsMargins(0, 0, 0, 0)
        seed_label = QLabel("Random seed")
        seed_label.setMinimumWidth(UISizes.FORM_LABEL_WIDTH)
        seed_row.addWidget(seed_label)
        self._seed_spin = QSpinBox()
        self._seed_spin.setMinimumHeight(UISizes.LINE_EDIT_MIN_HEIGHT)
        self._seed_spin.setRange(0, 2 ** 31 - 1)
        self._seed_spin.setValue(0)
        seed_row.addWidget(self._seed_spin)
        seed_row.addStretch()
        layout.addWidget(self._seed_container)

        # --- Inline error -------------------------------------------------
        self._error_label = QLabel("")
        self._error_label.setWordWrap(True)
        self._error_label.setStyleSheet(
            f"color: {theme().accent_red}; font-size: {TypeScale.SM}px;"
        )
        self._error_label.setVisible(False)
        layout.addWidget(self._error_label)

        layout.addStretch()

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setMinimumHeight(UISizes.BUTTON_MIN_HEIGHT)
        self._cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(self._cancel_btn)
        self._accept_btn = QPushButton("Generate")
        self._accept_btn.setMinimumHeight(UISizes.BUTTON_MIN_HEIGHT)
        self._accept_btn.setStyleSheet("font-weight: bold;")
        self._accept_btn.clicked.connect(self._on_accept)
        btn_row.addWidget(self._accept_btn)
        layout.addLayout(btn_row)

        scroll.setWidget(page)
        return scroll

    def _build_progress_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Spacing.MD)

        self._progress_title = QLabel("Generating...")
        self._progress_title.setStyleSheet(
            f"font-size: {TypeScale.LG}px; font-weight: bold;"
        )
        layout.addWidget(self._progress_title)

        self._progress_label = QLabel("")
        self._progress_label.setStyleSheet(f"color: {theme().text_secondary};")
        layout.addWidget(self._progress_label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setMinimum(0)
        self._progress_bar.setMaximum(100)
        layout.addWidget(self._progress_bar)

        layout.addStretch()

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._progress_cancel_btn = QPushButton("Cancel")
        self._progress_cancel_btn.setMinimumHeight(UISizes.BUTTON_MIN_HEIGHT)
        self._progress_cancel_btn.clicked.connect(self._on_cancel_workers)
        btn_row.addWidget(self._progress_cancel_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)
        return page

    def _build_ollama_absent_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Spacing.MD)

        title = QLabel("Local LLM not detected")
        title.setStyleSheet(
            f"font-size: {TypeScale.LG}px; font-weight: bold;"
        )
        layout.addWidget(title)

        self._ollama_message_label = QLabel("")
        self._ollama_message_label.setWordWrap(True)
        self._ollama_message_label.setStyleSheet(
            f"color: {theme().text_secondary};"
        )
        layout.addWidget(self._ollama_message_label)

        layout.addStretch()
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.setMinimumHeight(UISizes.BUTTON_MIN_HEIGHT)
        close_btn.clicked.connect(self.reject)
        btn_row.addWidget(close_btn)
        recheck_btn = QPushButton("Recheck")
        recheck_btn.setMinimumHeight(UISizes.BUTTON_MIN_HEIGHT)
        recheck_btn.clicked.connect(self._on_recheck_ollama)
        btn_row.addWidget(recheck_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)
        return page

    # -------------------------------------------------------- Source picker

    def _populate_source_picker(self) -> None:
        self._source_list.blockSignals(True)
        self._source_list.clear()
        for source_id, src_clips in self._clips_by_source_id.items():
            source = self._sources_by_id.get(source_id)
            badge_key, language = self._source_status.get(
                source_id, (_BADGE_ALIGNED, None)
            )
            label = self._format_source_row(source, src_clips, badge_key, language)
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, source_id)
            disabled = badge_key in (_BADGE_UNSUPPORTED_LANGUAGE, _BADGE_MISSING_FPS)
            if disabled:
                item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
                item.setCheckState(Qt.Unchecked)
                if badge_key == _BADGE_UNSUPPORTED_LANGUAGE:
                    item.setToolTip(
                        f"alignment model does not support {language!r}; "
                        "source unavailable for word-level sequencing"
                    )
                else:
                    item.setToolTip("source missing fps metadata")
            else:
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked)
            self._source_list.addItem(item)
        self._source_list.blockSignals(False)

    def _format_source_row(
        self,
        source: Any,
        src_clips: list[tuple[Any, Any]],
        badge_key: str,
        language: Optional[str],
    ) -> str:
        filename = (
            getattr(source, "filename", None)
            or str(getattr(source, "file_path", "")) or "unknown"
        )
        duration = getattr(source, "duration_seconds", None)
        if duration is None:
            duration_str = getattr(source, "duration_str", "?")
        else:
            duration_str = f"{float(duration):.1f}s"
        if badge_key == _BADGE_ALIGNED:
            badge = "✓ aligned"
        elif badge_key == _BADGE_NEEDS_ALIGNMENT:
            badge = "… needs alignment"
        elif badge_key == _BADGE_UNSUPPORTED_LANGUAGE:
            badge = f"⚠ unsupported language ({language})"
        else:
            badge = "⚠ missing fps"
        return f"{filename}  ·  {duration_str}  ·  {badge}  ·  {len(src_clips)} clip(s)"

    def _checked_clips(self) -> list[tuple[Any, Any]]:
        out: list[tuple[Any, Any]] = []
        for i in range(self._source_list.count()):
            item = self._source_list.item(i)
            if item.checkState() != Qt.Checked:
                continue
            source_id = item.data(Qt.UserRole)
            out.extend(self._clips_by_source_id.get(source_id, []))
        return out

    def _alignable_pending_clips(self) -> list:
        pending: list = []
        for clip, _source in self._checked_clips():
            transcript = getattr(clip, "transcript", None) or []
            if any(getattr(seg, "words", None) is None for seg in transcript):
                pending.append(clip)
        return pending

    # ------------------------------------------------------------ Slots

    @Slot(int)
    def _on_repeat_policy_changed(self, _index: int) -> None:
        policy = self._repeat_combo.currentData()
        self._seed_container.setVisible(policy == "random")

    def _set_error(self, message: Optional[str]) -> None:
        if message:
            self._error_label.setText(message)
            self._error_label.setVisible(True)
        else:
            self._error_label.clear()
            self._error_label.setVisible(False)

    def _refresh_validation(self) -> None:
        if not self._ollama_healthy:
            self._accept_btn.setEnabled(False)
            return
        checked = self._checked_clips()
        if not checked:
            self._set_error("Select at least one source to continue.")
            self._accept_btn.setEnabled(False)
            return

        prompt_text = self._prompt_input.toPlainText().strip()
        if not prompt_text:
            self._set_error("Enter a prompt.")
            self._accept_btn.setEnabled(False)
            return

        try:
            from core.spine.words import build_inventory
            inv = build_inventory(checked)
            corpus_size = len(inv.by_word)
        except Exception:
            corpus_size = 0
        any_needs_alignment = any(
            self._source_status.get(sid, (_BADGE_ALIGNED, None))[0]
            == _BADGE_NEEDS_ALIGNMENT
            for sid in self._checked_source_ids()
        )
        if corpus_size == 0 and not any_needs_alignment:
            self._set_error(
                "no words in selected sources — adjust your selection or include list"
            )
            self._accept_btn.setEnabled(False)
            return

        self._set_error(None)
        self._accept_btn.setEnabled(True)

    def _checked_source_ids(self) -> list[str]:
        out: list[str] = []
        for i in range(self._source_list.count()):
            item = self._source_list.item(i)
            if item.checkState() == Qt.Checked:
                src_id = item.data(Qt.UserRole)
                if src_id:
                    out.append(src_id)
        return out

    # --------------------------------------------------------- Ollama gate

    def _probe_ollama(self) -> None:
        healthy, message = self._health_probe()
        self._ollama_healthy = bool(healthy)
        if not healthy:
            self._ollama_message_label.setText(
                "Local LLM not detected — install or start Ollama, then click "
                "Recheck.\n\nDetails: " + (message or "(no diagnostic message)")
            )
            self._stack.setCurrentIndex(2)
            self._accept_btn.setEnabled(False)
        else:
            self._stack.setCurrentIndex(0)

    @Slot()
    def _on_recheck_ollama(self) -> None:
        self._probe_ollama()
        self._refresh_validation()

    # ------------------------------------------------------------ Accept

    @Slot()
    def _on_accept(self) -> None:
        if not self._ollama_healthy:
            self._set_error("Local LLM not detected. Start Ollama and recheck.")
            return
        checked = self._checked_clips()
        if not checked:
            return
        pending = self._alignable_pending_clips()
        if pending:
            self._start_alignment(pending)
            return
        self._start_compose()

    # --------------------------------------------------------- Composition

    def _start_compose(self) -> None:
        from ui.workers.llm_composer_worker import LLMComposerWorker

        self._compose_finished_handled = False
        self._stack.setCurrentIndex(1)
        self._progress_title.setText("Composing with LLM")
        self._progress_label.setText(
            f"Calling local LLM to generate {self._length_spin.value()} words..."
        )
        self._progress_bar.setValue(0)
        # Lock the prompt while generating.
        self._prompt_input.setReadOnly(True)

        policy = self._repeat_combo.currentData()
        seed = self._seed_spin.value() if policy == "random" else None

        worker = LLMComposerWorker(
            clips=self._checked_clips(),
            prompt=self._prompt_input.toPlainText().strip(),
            target_length=self._length_spin.value(),
            repeat_policy=policy,
            seed=seed,
            handle_frames=0,
            parent=self,
        )
        worker.progress.connect(self._on_compose_progress, Qt.UniqueConnection)
        worker.sequence_ready.connect(
            self._on_compose_ready, Qt.UniqueConnection,
        )
        worker.error.connect(self._on_compose_error, Qt.UniqueConnection)
        self._compose_worker = worker
        worker.start()

    @Slot(int, int)
    def _on_compose_progress(self, current: int, total: int) -> None:
        if total > 0:
            self._progress_bar.setValue(int(current / total * 100))

    @Slot(list)
    def _on_compose_ready(self, sequence_clips: list) -> None:
        if self._compose_finished_handled:
            return
        self._compose_finished_handled = True
        self._prompt_input.setReadOnly(False)
        worker = self._compose_worker
        self._compose_worker = None
        if worker is not None:
            try:
                worker.wait(50)
            except Exception:
                pass
        if not sequence_clips:
            self._set_error(
                "LLM produced no sequence. Adjust the prompt and try again."
            )
            self._stack.setCurrentIndex(0)
            return
        self.sequence_ready.emit(sequence_clips)
        self.accept()

    @Slot(str)
    def _on_compose_error(self, message: str) -> None:
        if self._compose_finished_handled:
            return
        self._compose_finished_handled = True
        self._prompt_input.setReadOnly(False)
        worker = self._compose_worker
        self._compose_worker = None
        if worker is not None:
            try:
                worker.wait(50)
            except Exception:
                pass
        self._set_error(message)
        self._stack.setCurrentIndex(0)

    # --------------------------------------------------------- Alignment

    def _start_alignment(self, pending_clips: list) -> None:
        from ui.workers.forced_alignment_worker import ForcedAlignmentWorker

        self._alignment_finished_handled = False
        self._pending_after_alignment = True

        sources_by_id = {
            src.id: src for src in self._sources_by_id.values() if src is not None
        }

        self._stack.setCurrentIndex(1)
        self._progress_title.setText("Aligning words")
        self._progress_label.setText(
            f"Running word-level alignment on {len(pending_clips)} clip(s)..."
        )
        self._progress_bar.setValue(0)

        worker = ForcedAlignmentWorker(
            clips=pending_clips,
            sources_by_id=sources_by_id,
            skip_existing=True,
            parent=self,
        )
        worker.progress.connect(self._on_alignment_progress, Qt.UniqueConnection)
        worker.clip_aligned.connect(self._on_clip_aligned, Qt.UniqueConnection)
        worker.alignment_completed.connect(
            self._on_alignment_completed, Qt.UniqueConnection,
        )
        worker.error.connect(self._on_alignment_error, Qt.UniqueConnection)
        self._alignment_worker = worker
        worker.start()

    @Slot(int, int)
    def _on_alignment_progress(self, current: int, total: int) -> None:
        if total > 0:
            self._progress_bar.setValue(int(current / total * 100))
        self._progress_label.setText(f"Aligning clip {current} of {total}...")

    @Slot(str, list)
    def _on_clip_aligned(self, clip_id: str, words: list) -> None:
        clip = None
        for c, _ in self._clips:
            if getattr(c, "id", None) == clip_id:
                clip = c
                break
        if clip is None or not getattr(clip, "transcript", None):
            return
        segments = list(clip.transcript)
        per_segment: list[list] = [[] for _ in segments]
        for word in words:
            midpoint = (float(word.start) + float(word.end)) / 2.0
            chosen_idx: Optional[int] = None
            for i, seg in enumerate(segments):
                if seg.start_time <= midpoint <= seg.end_time:
                    chosen_idx = i
                    break
            if chosen_idx is None:
                chosen_idx = min(
                    range(len(segments)),
                    key=lambda i: min(
                        abs(midpoint - segments[i].start_time),
                        abs(midpoint - segments[i].end_time),
                    ),
                )
            per_segment[chosen_idx].append(word)
        for seg, seg_words in zip(segments, per_segment):
            seg.words = seg_words

    @Slot(str)
    def _on_alignment_error(self, message: str) -> None:
        self._set_error(f"Alignment failed: {message}")

    @Slot()
    def _on_alignment_completed(self) -> None:
        if self._alignment_finished_handled:
            return
        self._alignment_finished_handled = True
        for source_id, src_clips in self._clips_by_source_id.items():
            self._source_status[source_id] = _classify_source(src_clips)
        worker = self._alignment_worker
        self._alignment_worker = None
        if worker is not None:
            try:
                worker.wait(50)
            except Exception:
                pass
        if not self._pending_after_alignment:
            self._stack.setCurrentIndex(0)
            return
        self._pending_after_alignment = False
        self._start_compose()

    # ------------------------------------------------------------ Cancel

    @Slot()
    def _on_cancel_workers(self) -> None:
        cancelled = False
        for worker in (self._alignment_worker, self._compose_worker):
            if worker is not None and worker.isRunning():
                worker.cancel()
                cancelled = True
        self._pending_after_alignment = False
        if not cancelled:
            self._stack.setCurrentIndex(0)

    # --------------------------------------------------------- Lifecycle

    def closeEvent(self, event) -> None:  # noqa: D401
        for worker in (self._alignment_worker, self._compose_worker):
            if worker is not None and worker.isRunning():
                worker.cancel()
                worker.wait(2000)
        super().closeEvent(event)
