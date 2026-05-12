"""Word Sequencer dialog — preset-mode word-level sequencing.

Lets the user pick the source(s) whose word inventory to draw from, choose one
of five preset modes (alphabetical / chosen-words / by-frequency / by-property
/ user-curated ordered list), and emit a ``list[SequenceClip]`` materialized
through ``core.remix.word_sequencer.generate_word_sequence``.

If any of the user's selected clips lack word-level alignment data, accepting
the dialog auto-runs the ``ForcedAlignmentWorker`` (U3) on the offending
clips. Word data is distributed back onto each clip's transcript on the main
thread before the dialog re-attempts the sequencer call.

Follows the staccato-dialog pattern:
  - modal ``QDialog`` subclass,
  - owns its worker(s) internally,
  - feature-registry preflight runs inside the worker, not here,
  - builds a ``list[SequenceClip]`` and emits via ``sequence_ready``; the
    Sequence tab is responsible for wrapping the result in a ``Sequence`` and
    assigning it to ``project.sequence``.
"""

from __future__ import annotations

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

from core.remix.word_sequencer import (
    MissingWordDataError,
    generate_word_sequence,
)
from ui.dialogs._word_source_picker import (
    BADGE_ALIGNED,
    BADGE_MISSING_FPS,
    BADGE_NEEDS_ALIGNMENT,
    BADGE_UNSUPPORTED_LANGUAGE,
    WordAlignmentController,
    classify_source_alignment,
)
from ui.theme import theme, Spacing, TypeScale, UISizes

logger = logging.getLogger(__name__)


__all__ = ["WordSequencerDialog"]


# Mode display labels keyed by spine-level mode names. Order matches the
# brainstorm's enumeration.
_MODE_OPTIONS: list[tuple[str, str]] = [
    ("alphabetical", "Alphabetical"),
    ("by_chosen_words", "Chosen Words"),
    ("by_frequency", "By Frequency"),
    ("by_property", "By Property"),
    ("from_word_list", "User-Curated Ordered List"),
]


# Backward-compatible private aliases for the per-source badge constants.
# The canonical names live in ``ui.dialogs._word_source_picker``; these
# aliases keep existing tests / imports working.
_BADGE_ALIGNED = BADGE_ALIGNED
_BADGE_NEEDS_ALIGNMENT = BADGE_NEEDS_ALIGNMENT
_BADGE_UNSUPPORTED_LANGUAGE = BADGE_UNSUPPORTED_LANGUAGE
_BADGE_MISSING_FPS = BADGE_MISSING_FPS

# Backward-compatible alias for ``_classify_source`` (the previous
# underscore-private name). New code should use
# ``classify_source_alignment`` directly.
_classify_source = classify_source_alignment


def _parse_word_list(text: str) -> list[str]:
    """Tokenize a user-entered word list (comma / whitespace / newline)."""
    if not text:
        return []
    parts: list[str] = []
    for line in text.splitlines():
        for piece in line.replace(",", " ").split():
            piece = piece.strip()
            if piece:
                parts.append(piece)
    return parts


def _normalize_for_lookup(word: str) -> str:
    """Lazy import of the spine normalizer to keep this module light."""
    from core.spine.words import normalize_word
    return normalize_word(word)


class WordSequencerDialog(QDialog):
    """Modal dialog for the Word Sequencer (preset modes)."""

    # Emits ``list[SequenceClip]`` ordered as the preset mode produced them.
    sequence_ready = Signal(list)

    def __init__(
        self,
        clips: list[tuple[Any, Any]],
        project: Any = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._clips = list(clips or [])
        self._project = project

        # Worker state. The controller owns the worker lifetime; the
        # dialog only needs to know whether an alignment is currently in
        # flight (via ``self._alignment_ctrl is not None``).
        self._pending_after_alignment = False

        # Per-source classification cache, keyed by Source.id.
        self._source_status: dict[str, tuple[str, Optional[str]]] = {}
        # Source id -> list of (Clip, Source) tuples for that source from the
        # user's selection.
        self._clips_by_source_id: dict[str, list[tuple[Any, Any]]] = {}
        self._sources_by_id: dict[str, Any] = {}

        for clip, source in self._clips:
            source_id = getattr(source, "id", None)
            if source_id is None:
                continue
            self._sources_by_id.setdefault(source_id, source)
            self._clips_by_source_id.setdefault(source_id, []).append((clip, source))

        for source_id, src_clips in self._clips_by_source_id.items():
            self._source_status[source_id] = classify_source_alignment(src_clips)

        # Shared alignment controller — wraps the ``ForcedAlignmentWorker``
        # lifecycle and the per-clip "distribute words back" mutation. The
        # dialog reacts to the controller's ``completed`` / ``error``
        # signals (see :meth:`_start_alignment`).
        self._alignment_ctrl: Optional[WordAlignmentController] = None

        # Validation-time inventory cache. Keyed by frozenset of checked
        # source ids; invalidated on source-picker toggles and after
        # alignment runs. Without this, every keystroke in the chosen-words
        # or user-list inputs rebuilds the full inventory.
        self._inventory_cache_key: Optional[frozenset] = None
        self._inventory_cache: Optional[Any] = None

        self.setWindowTitle("Word Sequencer")
        self.setMinimumWidth(560)
        self.setMinimumHeight(560)
        self._setup_ui()
        self._populate_source_picker()
        self._on_mode_changed(self._mode_combo.currentIndex())
        self._refresh_validation()

    # ------------------------------------------------------------------ UI

    def _setup_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 24, 24, 24)
        outer.setSpacing(Spacing.MD)

        title = QLabel("Word Sequencer")
        title.setStyleSheet(f"font-size: {TypeScale.XL}px; font-weight: bold;")
        outer.addWidget(title)

        desc = QLabel(
            "Compose a film from individual spoken words. Order by alphabet, "
            "frequency, a property like word length, or a list you supply."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color: {theme().text_secondary};")
        outer.addWidget(desc)

        self._stack = QStackedWidget()
        outer.addWidget(self._stack, 1)

        self._stack.addWidget(self._build_form_page())
        self._stack.addWidget(self._build_progress_page())
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
        source_label = QLabel("Sources")
        source_label.setStyleSheet(
            f"color: {theme().text_secondary}; font-size: {TypeScale.SM}px;"
        )
        layout.addWidget(source_label)

        self._source_list = QListWidget()
        self._source_list.setMinimumHeight(120)
        self._source_list.itemChanged.connect(self._on_source_checked_changed)
        layout.addWidget(self._source_list)

        # --- Mode select --------------------------------------------------
        mode_row = QHBoxLayout()
        mode_label_widget = QLabel("Mode")
        mode_label_widget.setMinimumWidth(UISizes.FORM_LABEL_WIDTH)
        mode_row.addWidget(mode_label_widget)

        self._mode_combo = QComboBox()
        self._mode_combo.setMinimumHeight(UISizes.COMBO_BOX_MIN_HEIGHT)
        self._mode_combo.setMinimumWidth(UISizes.COMBO_BOX_MIN_WIDTH)
        for _key, label in _MODE_OPTIONS:
            self._mode_combo.addItem(label)
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self._mode_combo, 1)
        layout.addLayout(mode_row)

        # --- Chosen-words container --------------------------------------
        self._chosen_container = QWidget()
        chosen_layout = QVBoxLayout(self._chosen_container)
        chosen_layout.setContentsMargins(0, 0, 0, 0)
        chosen_layout.setSpacing(Spacing.SM)
        chosen_label = QLabel("Words to include (one per line, comma or space-separated):")
        chosen_label.setStyleSheet(
            f"color: {theme().text_secondary}; font-size: {TypeScale.SM}px;"
        )
        chosen_layout.addWidget(chosen_label)
        self._chosen_words_input = QPlainTextEdit()
        self._chosen_words_input.setPlaceholderText("the, sky, light")
        self._chosen_words_input.setMinimumHeight(80)
        self._chosen_words_input.textChanged.connect(self._refresh_validation)
        chosen_layout.addWidget(self._chosen_words_input)
        self._chosen_match_label = QLabel("")
        self._chosen_match_label.setStyleSheet(
            f"color: {theme().text_muted}; font-size: {TypeScale.XS}px;"
        )
        chosen_layout.addWidget(self._chosen_match_label)
        layout.addWidget(self._chosen_container)

        # --- Property select container -----------------------------------
        self._property_container = QWidget()
        prop_layout = QHBoxLayout(self._property_container)
        prop_layout.setContentsMargins(0, 0, 0, 0)
        prop_layout.setSpacing(Spacing.SM)
        prop_layout.addWidget(QLabel("Property:"))
        self._property_combo = QComboBox()
        self._property_combo.setMinimumHeight(UISizes.COMBO_BOX_MIN_HEIGHT)
        self._property_combo.addItem("length", "length")
        self._property_combo.addItem("duration", "duration")
        self._property_combo.addItem("log_frequency", "log_frequency")
        self._property_combo.currentIndexChanged.connect(self._on_property_changed)
        prop_layout.addWidget(self._property_combo)
        prop_layout.addWidget(QLabel("Order:"))
        self._property_order_combo = QComboBox()
        self._property_order_combo.setMinimumHeight(UISizes.COMBO_BOX_MIN_HEIGHT)
        self._property_order_combo.addItem("ascending", "ascending")
        self._property_order_combo.addItem("descending", "descending")
        prop_layout.addWidget(self._property_order_combo)
        prop_layout.addStretch()
        layout.addWidget(self._property_container)

        # --- Frequency order container -----------------------------------
        self._frequency_container = QWidget()
        freq_layout = QHBoxLayout(self._frequency_container)
        freq_layout.setContentsMargins(0, 0, 0, 0)
        freq_layout.setSpacing(Spacing.SM)
        freq_layout.addWidget(QLabel("Order:"))
        self._frequency_order_combo = QComboBox()
        self._frequency_order_combo.setMinimumHeight(UISizes.COMBO_BOX_MIN_HEIGHT)
        self._frequency_order_combo.addItem("descending", "descending")
        self._frequency_order_combo.addItem("ascending", "ascending")
        freq_layout.addWidget(self._frequency_order_combo)
        freq_layout.addStretch()
        layout.addWidget(self._frequency_container)

        # --- User-curated word list container ----------------------------
        self._userlist_container = QWidget()
        ul_layout = QVBoxLayout(self._userlist_container)
        ul_layout.setContentsMargins(0, 0, 0, 0)
        ul_layout.setSpacing(Spacing.SM)
        ul_label = QLabel(
            "Word sequence (repeats allowed; one slot per entry):"
        )
        ul_label.setStyleSheet(
            f"color: {theme().text_secondary}; font-size: {TypeScale.SM}px;"
        )
        ul_layout.addWidget(ul_label)
        self._userlist_input = QPlainTextEdit()
        self._userlist_input.setPlaceholderText("the the the sky")
        self._userlist_input.setMinimumHeight(80)
        self._userlist_input.textChanged.connect(self._refresh_validation)
        ul_layout.addWidget(self._userlist_input)
        self._userlist_match_label = QLabel("")
        self._userlist_match_label.setStyleSheet(
            f"color: {theme().text_muted}; font-size: {TypeScale.XS}px;"
        )
        ul_layout.addWidget(self._userlist_match_label)
        layout.addWidget(self._userlist_container)

        # --- Frame padding -----------------------------------------------
        handle_row = QHBoxLayout()
        handle_label = QLabel("Frame Padding")
        handle_label.setMinimumWidth(UISizes.FORM_LABEL_WIDTH)
        handle_row.addWidget(handle_label)
        self._handle_spin = QSpinBox()
        self._handle_spin.setRange(0, 10)
        self._handle_spin.setValue(0)
        self._handle_spin.setMinimumHeight(UISizes.LINE_EDIT_MIN_HEIGHT)
        handle_row.addWidget(self._handle_spin)
        handle_row.addStretch()
        layout.addLayout(handle_row)

        # --- Inline error label ------------------------------------------
        self._error_label = QLabel("")
        self._error_label.setWordWrap(True)
        self._error_label.setStyleSheet(
            f"color: {theme().accent_red}; font-size: {TypeScale.SM}px;"
        )
        self._error_label.setVisible(False)
        layout.addWidget(self._error_label)

        layout.addStretch()

        # --- Buttons ------------------------------------------------------
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

        self._progress_title = QLabel("Aligning words...")
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

        btns = QHBoxLayout()
        btns.addStretch()
        self._progress_cancel_btn = QPushButton("Cancel")
        self._progress_cancel_btn.setMinimumHeight(UISizes.BUTTON_MIN_HEIGHT)
        self._progress_cancel_btn.clicked.connect(self._on_cancel_alignment)
        btns.addWidget(self._progress_cancel_btn)
        btns.addStretch()
        layout.addLayout(btns)
        return page

    # ----------------------------------------------------------- Helpers

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
            # Try a derived duration_str if present.
            duration_str = getattr(source, "duration_str", "?")
        else:
            duration_str = f"{float(duration):.1f}s"

        if badge_key == _BADGE_ALIGNED:
            badge = "✓ aligned"
        elif badge_key == _BADGE_NEEDS_ALIGNMENT:
            badge = "… needs alignment"
        elif badge_key == _BADGE_UNSUPPORTED_LANGUAGE:
            badge = f"⚠ unsupported language ({language})"
        else:  # missing fps
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
        """Return ``Clip`` objects from checked sources missing word data."""
        pending: list = []
        for clip, _source in self._checked_clips():
            transcript = getattr(clip, "transcript", None) or []
            if any(getattr(seg, "words", None) is None for seg in transcript):
                pending.append(clip)
        return pending

    def _selected_mode_key(self) -> str:
        return _MODE_OPTIONS[self._mode_combo.currentIndex()][0]

    # ----------------------------------------------------------- Slots

    @Slot(int)
    def _on_mode_changed(self, _index: int) -> None:
        mode = self._selected_mode_key()
        self._chosen_container.setVisible(mode == "by_chosen_words")
        self._property_container.setVisible(mode == "by_property")
        self._frequency_container.setVisible(mode == "by_frequency")
        self._userlist_container.setVisible(mode == "from_word_list")
        self._refresh_validation()

    @Slot(int)
    def _on_property_changed(self, _index: int) -> None:
        # Plan default: ascending for length, descending for log_frequency.
        key = self._property_combo.currentData()
        if key == "log_frequency":
            self._property_order_combo.setCurrentIndex(1)  # descending
        elif key == "length":
            self._property_order_combo.setCurrentIndex(0)  # ascending

    @Slot()
    def _on_source_checked_changed(self, _item) -> None:
        # The set of checked sources changed → corpus changed.
        self._invalidate_inventory_cache()
        self._refresh_validation()

    def _invalidate_inventory_cache(self) -> None:
        """Drop the cached ``WordInventory`` so the next access rebuilds it."""
        self._inventory_cache_key = None
        self._inventory_cache = None

    def _get_cached_inventory(self, checked_clips: list[tuple[Any, Any]]):
        """Build (or return cached) ``WordInventory`` for the checked set.

        Cache key is the frozenset of checked source ids — the only thing
        that affects which clips are included in the inventory. Text
        inputs (chosen-words / user-list) do NOT change the corpus, so
        their textChanged signals never trigger a rebuild.
        """
        key = frozenset(self._checked_source_ids())
        if self._inventory_cache_key == key and self._inventory_cache is not None:
            return self._inventory_cache
        try:
            from core.spine.words import build_inventory
            inv = build_inventory(checked_clips)
        except Exception:
            inv = None
        self._inventory_cache_key = key
        self._inventory_cache = inv
        return inv

    def _refresh_validation(self) -> None:
        """Re-evaluate Accept enablement and update the inline error label."""
        checked_clips = self._checked_clips()
        mode = self._selected_mode_key()

        if not checked_clips:
            self._set_error("Select at least one source to continue.")
            self._accept_btn.setEnabled(False)
            return

        # Mode-specific corpus / list checks. These are cheap and only run
        # over the current selection's already-aligned words; clips pending
        # alignment are conservatively counted as "potentially in corpus".
        #
        # The inventory is cached by checked-source-id set so that typing
        # in the chosen-words / user-list inputs doesn't rebuild it on
        # every keystroke (textChanged fires very frequently).
        inv = self._get_cached_inventory(checked_clips)
        corpus = inv.by_word if inv is not None else {}

        # If every clip is missing words and the corpus is empty, the dialog
        # will fall back to triggering alignment. That's fine — we let
        # Accept stay enabled in that case so the user can opt into the run.
        any_needs_alignment = any(
            self._source_status.get(item_source_id, (_BADGE_ALIGNED, None))[0]
            == _BADGE_NEEDS_ALIGNMENT
            for item_source_id in self._checked_source_ids()
        )

        if not corpus and not any_needs_alignment:
            self._set_error(
                "no words in selected sources — adjust your selection or include list"
            )
            self._accept_btn.setEnabled(False)
            return

        if mode == "by_chosen_words":
            entries = _parse_word_list(self._chosen_words_input.toPlainText())
            normalized = [_normalize_for_lookup(w) for w in entries]
            normalized = [w for w in normalized if w]
            found = [w for w in normalized if w in corpus]
            if normalized:
                self._chosen_match_label.setText(
                    f"in corpus: {len(found)} / {len(normalized)}"
                )
            else:
                self._chosen_match_label.setText("")
            if normalized and not found and not any_needs_alignment:
                self._set_error(
                    f"0 of {len(normalized)} include-list words found in corpus"
                )
                self._accept_btn.setEnabled(False)
                return
            if not normalized:
                self._set_error("Enter at least one word to include.")
                self._accept_btn.setEnabled(False)
                return

        if mode == "from_word_list":
            entries = _parse_word_list(self._userlist_input.toPlainText())
            normalized = [_normalize_for_lookup(w) for w in entries]
            normalized = [w for w in normalized if w]
            unrecognized = [w for w in normalized if w not in corpus]
            if normalized:
                self._userlist_match_label.setText(
                    f"{len(normalized)} slots; {len(unrecognized)} unrecognized"
                )
            else:
                self._userlist_match_label.setText("")
            if not normalized:
                self._set_error("Enter at least one word.")
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

    def _set_error(self, text: Optional[str]) -> None:
        if text:
            self._error_label.setText(text)
            self._error_label.setVisible(True)
        else:
            self._error_label.clear()
            self._error_label.setVisible(False)

    # ----------------------------------------------------------- Accept

    @Slot()
    def _on_accept(self) -> None:
        """Build the sequence; auto-run alignment if needed."""
        checked_clips = self._checked_clips()
        if not checked_clips:
            return

        pending = self._alignable_pending_clips()
        if pending:
            # Auto-run alignment over the pending clips. Once the worker
            # completes, re-enter accept logic via _pending_after_alignment.
            self._start_alignment(pending)
            return

        self._try_generate()

    def _try_generate(self) -> None:
        try:
            sequence_clips = generate_word_sequence(
                self._checked_clips(),
                mode=self._selected_mode_key(),
                mode_params=self._collect_mode_params(),
                handle_frames=self._handle_spin.value(),
            )
        except MissingWordDataError as exc:
            # Alignment must have failed for some clips; fall through to
            # surfacing the message without dismissing the dialog.
            self._set_error(str(exc))
            self._stack.setCurrentIndex(0)
            return
        except ValueError as exc:
            self._set_error(str(exc))
            self._stack.setCurrentIndex(0)
            return
        except Exception as exc:  # noqa: BLE001
            logger.exception("Word Sequencer generation failed")
            self._set_error(f"Sequencer error: {exc}")
            self._stack.setCurrentIndex(0)
            return

        if not sequence_clips:
            self._set_error(
                "Sequencer produced no clips — check your word list / corpus."
            )
            self._stack.setCurrentIndex(0)
            return

        self.sequence_ready.emit(sequence_clips)
        self.accept()

    def _collect_mode_params(self) -> dict:
        mode = self._selected_mode_key()
        if mode == "by_chosen_words":
            return {"include": _parse_word_list(self._chosen_words_input.toPlainText())}
        if mode == "by_frequency":
            return {"order": self._frequency_order_combo.currentData()}
        if mode == "by_property":
            return {
                "key": self._property_combo.currentData(),
                "order": self._property_order_combo.currentData(),
            }
        if mode == "from_word_list":
            return {
                "sequence": _parse_word_list(self._userlist_input.toPlainText()),
                "on_missing": "skip",
            }
        return {}

    # --------------------------------------------------------- Alignment

    def _start_alignment(self, pending_clips: list) -> None:
        """Spawn a ``WordAlignmentController`` over the pending clips."""
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

        ctrl = WordAlignmentController(self._clips, parent=self)
        ctrl.progress.connect(self._on_alignment_progress, Qt.UniqueConnection)
        ctrl.completed.connect(self._on_alignment_completed, Qt.UniqueConnection)
        ctrl.error.connect(self._on_alignment_error, Qt.UniqueConnection)
        self._alignment_ctrl = ctrl
        ctrl.start(pending_clips, sources_by_id)

    @Slot(int, int)
    def _on_alignment_progress(self, current: int, total: int) -> None:
        if total > 0:
            self._progress_bar.setValue(int(current / total * 100))
        self._progress_label.setText(f"Aligning clip {current} of {total}...")

    @Slot(str)
    def _on_alignment_error(self, message: str) -> None:
        logger.warning("WordSequencerDialog alignment error: %s", message)
        self._set_error(f"Alignment failed: {message}")

    @Slot()
    def _on_alignment_completed(self) -> None:
        # Refresh per-source status caches so the picker would render
        # correctly if the user backs out and re-opens the dialog.
        for source_id, src_clips in self._clips_by_source_id.items():
            self._source_status[source_id] = classify_source_alignment(src_clips)

        self._alignment_ctrl = None
        # The controller mutated Clip.transcript[*].words in place; mark the
        # project dirty so the new alignment data survives save.
        if self._project is not None:
            self._project.mark_dirty()
        # The clips' word data has changed — invalidate the inventory
        # cache so the next ``_refresh_validation`` rebuilds.
        self._invalidate_inventory_cache()

        if not self._pending_after_alignment:
            # User cancelled — bounce back to form.
            self._stack.setCurrentIndex(0)
            return

        self._pending_after_alignment = False
        # Re-attempt generation after alignment.
        self._try_generate()

    @Slot()
    def _on_cancel_alignment(self) -> None:
        ctrl = self._alignment_ctrl
        if ctrl is not None and ctrl.is_running():
            self._pending_after_alignment = False
            ctrl.cancel()
            # The controller's ``completed`` signal will bounce the stack
            # back to the form page.
        else:
            self._stack.setCurrentIndex(0)

    # --------------------------------------------------------- Lifecycle

    def closeEvent(self, event) -> None:  # noqa: D401 - Qt override
        ctrl = self._alignment_ctrl
        if ctrl is not None and ctrl.is_running():
            ctrl.cancel()
            ctrl.wait(2000)
        super().closeEvent(event)
