"""Staccato dialog — beat-driven sequencing with onset-strength visual contrast.

User selects a music file, previews the waveform with beat markers,
then generates a sequence where onset strength drives visual contrast
between consecutive clips via DINOv2 embedding distance.
"""

import logging
from pathlib import Path

import numpy as np

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices

from core.analysis.audio import AudioAnalysis, analyze_music_file
from core.remix.staccato import generate_staccato_sequence
from ui.theme import theme, Spacing, TypeScale, UISizes
from ui.widgets.waveform_widget import WaveformWidget
from ui.workers.base import CancellableWorker

logger = logging.getLogger(__name__)

_AUDIO_FORMATS = "Audio Files (*.mp3 *.wav *.flac *.m4a *.aac *.ogg);;All Files (*)"


class StaccatoAnalyzeWorker(CancellableWorker):
    """Background worker for analyzing a music file or separated stem."""

    audio_ready = Signal(object, object)  # AudioAnalysis, np.ndarray (samples)
    progress_message = Signal(str)

    def __init__(
        self,
        music_path: Path,
        stem_name: str | None = None,
        stems_cache_dir: Path | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._music_path = music_path
        self._stem_name = stem_name
        self._stems_cache_dir = stems_cache_dir

    def run(self):
        self._log_start()
        try:
            # Ensure librosa is installed before attempting audio analysis
            from core.feature_registry import check_feature_ready, install_for_feature

            available, _missing = check_feature_ready("audio_analysis")
            if not available:
                self.progress_message.emit("Installing audio analysis dependencies...")
                if not install_for_feature("audio_analysis"):
                    self.error.emit("Failed to install audio analysis dependencies (librosa)")
                    return

            audio_path = self._music_path

            # If stem separation requested, ensure deps are installed then separate
            if self._stem_name and self._stems_cache_dir:
                stem_available, _missing = check_feature_ready("stem_separation")
                if not stem_available:
                    self.progress_message.emit("Installing stem separation dependencies...")
                    if not install_for_feature("stem_separation"):
                        self.error.emit("Failed to install stem separation dependencies (demucs)")
                        return

                audio_path = self._get_or_separate_stem()
                if audio_path is None:
                    return  # cancelled or error already emitted

            if self.is_cancelled():
                self._log_cancelled()
                return

            self.progress_message.emit(
                f"Analyzing {self._stem_name} track..."
                if self._stem_name else "Analyzing audio..."
            )
            analysis = analyze_music_file(audio_path, include_onsets=True)

            if self.is_cancelled():
                self._log_cancelled()
                return

            # Load raw samples for waveform display
            from core.analysis.audio import _get_librosa
            librosa = _get_librosa()
            y, _sr = librosa.load(str(audio_path), sr=22050)

            self.audio_ready.emit(analysis, y)
        except Exception as e:
            if not self.is_cancelled():
                logger.error("Staccato audio analysis failed: %s", e)
                self.error.emit(str(e))
        self._log_complete()

    def _get_or_separate_stem(self) -> Path | None:
        """Get the stem audio path, running Demucs if not cached."""
        from core.analysis.stem_separation import (
            get_cached_stems,
            get_stem_cache_key,
            separate_stems,
        )

        cache_key = get_stem_cache_key(self._music_path)
        stem_dir = self._stems_cache_dir / cache_key

        # Check cache first
        cached = get_cached_stems(self._music_path, self._stems_cache_dir)
        if cached and self._stem_name in cached:
            logger.info(f"Using cached {self._stem_name} stem")
            return cached[self._stem_name]

        # Run separation
        self.progress_message.emit("Separating stems (this may take a minute)...")

        try:
            stems = separate_stems(
                self._music_path,
                stem_dir,
                progress_cb=lambda msg: self.progress_message.emit(msg),
            )
        except ImportError as e:
            self.error.emit(str(e))
            return None
        except RuntimeError as e:
            self.error.emit(str(e))
            return None

        if self.is_cancelled():
            self._log_cancelled()
            return None

        if self._stem_name in stems:
            return stems[self._stem_name]

        self.error.emit(f"Stem '{self._stem_name}' not found in separation output")
        return None


class StaccatoGenerateWorker(CancellableWorker):
    """Background worker for generating the beat-driven sequence."""

    progress_update = Signal(int, int)
    progress_message = Signal(str)
    finished_sequence = Signal(object)  # StaccatoResult (list-like)

    def __init__(
        self,
        clips: list,
        audio_analysis: AudioAnalysis,
        strategy: str,
        cut_times: list[float] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._clips = clips
        self._audio_analysis = audio_analysis
        self._strategy = strategy
        self._cut_times = cut_times

    def run(self):
        self._log_start()
        try:
            # Step 1: Auto-compute missing embeddings
            self.progress_message.emit("Computing clip embeddings...")
            self._auto_compute_embeddings()

            if self.is_cancelled():
                self._log_cancelled()
                return

            # Step 2: Generate beat-driven sequence
            self.progress_message.emit("Matching clips to beats...")
            result = generate_staccato_sequence(
                clips=self._clips,
                audio_analysis=self._audio_analysis,
                strategy=self._strategy,
                cut_times=self._cut_times,
                progress_cb=self._on_progress,
            )

            if self.is_cancelled():
                self._log_cancelled()
                return

            self.finished_sequence.emit(result)
        except Exception as e:
            if not self.is_cancelled():
                logger.error("Staccato generation failed: %s", e)
                self.error.emit(str(e))
        self._log_complete()

    def _auto_compute_embeddings(self):
        """Compute DINOv2 embeddings for clips that don't have them."""
        needs_embedding = [
            (clip, source) for clip, source in self._clips
            if clip.embedding is None and clip.thumbnail_path
        ]
        if not needs_embedding:
            return

        from core.feature_registry import check_feature

        available, missing = check_feature("embeddings")
        if not available:
            raise RuntimeError(
                "DINOv2 embeddings require torch and transformers. "
                f"Missing: {', '.join(missing)}. "
                "Run embedding analysis first or install dependencies via Settings."
            )

        from core.analysis.embeddings import extract_clip_embeddings_batch, _EMBEDDING_MODEL_TAG

        self.progress_message.emit(
            f"Computing embeddings for {len(needs_embedding)} clips..."
        )
        thumbnail_paths = [clip.thumbnail_path for clip, _ in needs_embedding]
        try:
            embeddings = extract_clip_embeddings_batch(thumbnail_paths)
            for (clip, _), emb in zip(needs_embedding, embeddings):
                clip.embedding = emb
                clip.embedding_model = _EMBEDDING_MODEL_TAG
        except Exception as e:
            raise RuntimeError(f"Failed to compute clip embeddings: {e}") from e

        still_missing = []
        for clip, _source in self._clips:
            if getattr(clip, "embedding", None) is None:
                still_missing.append(getattr(clip, "id", "<unknown>"))
        if still_missing:
            raise RuntimeError(
                "Missing DINOv2 embeddings for "
                f"{len(still_missing)} clips. Run embedding analysis first or "
                "ensure thumbnails exist before generating Staccato."
            )

    def _on_progress(self, current: int, total: int):
        self.progress_update.emit(current, total)


class StaccatoDialog(QDialog):
    """Dialog for Staccato beat-driven sequencing.

    Page 0: Config — music file picker, waveform, sensitivity, strategy, generate
    Page 1: Progress — progress bar during generation
    """

    sequence_ready = Signal(object)  # list of (Clip, Source)

    def __init__(self, clips: list, parent=None):
        super().__init__(parent)
        self._clips = clips
        self._analyze_worker = None
        self._generate_worker = None
        self._audio_analysis: AudioAnalysis | None = None
        self._audio_samples: np.ndarray | None = None
        self._music_path: Path | None = None
        self._handler_executed = False
        self._sequence_data = None
        self._debug_info = None

        self.setWindowTitle("Staccato")
        self.setMinimumWidth(520)
        self.setMinimumHeight(420)
        self._setup_ui()

    @property
    def music_path(self) -> Path | None:
        """The music file used for this staccato sequence."""
        return self._music_path

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)

        self._stack = QStackedWidget()
        layout.addWidget(self._stack)

        self._config_page = self._create_config_page()
        self._stack.addWidget(self._config_page)

        self._progress_page = self._create_progress_page()
        self._stack.addWidget(self._progress_page)

        self._results_page = self._create_results_page()
        self._stack.addWidget(self._results_page)

        self._stack.setCurrentIndex(0)

    def _create_config_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Spacing.MD)

        title = QLabel("Staccato")
        title.setStyleSheet(f"font-size: {TypeScale.XL}px; font-weight: bold;")
        layout.addWidget(title)

        desc = QLabel(
            "Cut clips to the rhythm of a music track.\n"
            "Stronger beats trigger bigger visual jumps between clips."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color: {theme().text_secondary};")
        layout.addWidget(desc)

        layout.addSpacing(Spacing.SM)

        # Music file picker
        file_row = QHBoxLayout()
        self._file_label = QLabel("No file selected")
        self._file_label.setStyleSheet(f"color: {theme().text_muted};")
        file_row.addWidget(self._file_label, 1)

        file_btn = QPushButton("Select Music File...")
        file_btn.clicked.connect(self._on_select_file)
        file_row.addWidget(file_btn)
        layout.addLayout(file_row)

        # Stem separation controls
        stem_row = QHBoxLayout()
        self._stem_checkbox = QCheckBox("Separate stems")
        self._stem_checkbox.setToolTip(
            "Use Demucs to isolate a specific instrument track for beat detection"
        )
        self._stem_checkbox.toggled.connect(self._on_stem_toggled)
        stem_row.addWidget(self._stem_checkbox)

        self._stem_combo = QComboBox()
        self._stem_combo.setMinimumHeight(UISizes.COMBO_BOX_MIN_HEIGHT)
        self._stem_combo.addItems(["Drums", "Bass", "Vocals", "Other"])
        self._stem_combo.setVisible(False)
        self._stem_combo.currentTextChanged.connect(self._on_stem_changed)
        stem_row.addWidget(self._stem_combo)
        stem_row.addStretch()
        layout.addLayout(stem_row)

        # Waveform
        self._waveform = WaveformWidget()
        layout.addWidget(self._waveform)

        # Controls row
        controls = QHBoxLayout()
        controls.setSpacing(Spacing.LG)

        # Sensitivity slider
        sens_layout = QVBoxLayout()
        sens_label = QLabel("Sensitivity")
        sens_label.setStyleSheet(f"color: {theme().text_secondary}; font-size: {TypeScale.SM}px;")
        sens_layout.addWidget(sens_label)

        sens_row = QHBoxLayout()
        fewer_label = QLabel("Fewer Cuts")
        fewer_label.setStyleSheet(f"color: {theme().text_muted}; font-size: {TypeScale.XS}px;")
        sens_row.addWidget(fewer_label)

        self._sensitivity_slider = QSlider(Qt.Horizontal)
        self._sensitivity_slider.setRange(1, 10)
        self._sensitivity_slider.setValue(5)
        self._sensitivity_slider.setTickPosition(QSlider.TicksBelow)
        self._sensitivity_slider.setTickInterval(1)
        self._sensitivity_slider.valueChanged.connect(self._on_sensitivity_or_strategy_changed)
        sens_row.addWidget(self._sensitivity_slider)

        more_label = QLabel("More Cuts")
        more_label.setStyleSheet(f"color: {theme().text_muted}; font-size: {TypeScale.XS}px;")
        sens_row.addWidget(more_label)

        sens_layout.addLayout(sens_row)
        controls.addLayout(sens_layout, 2)

        # Strategy dropdown
        strat_layout = QVBoxLayout()
        strat_label = QLabel("Beat Strategy")
        strat_label.setStyleSheet(f"color: {theme().text_secondary}; font-size: {TypeScale.SM}px;")
        strat_layout.addWidget(strat_label)

        self._strategy_combo = QComboBox()
        self._strategy_combo.setMinimumHeight(UISizes.COMBO_BOX_MIN_HEIGHT)
        self._strategy_combo.addItems(["Onsets", "Beats", "Downbeats"])
        self._strategy_combo.currentTextChanged.connect(self._on_sensitivity_or_strategy_changed)
        strat_layout.addWidget(self._strategy_combo)
        controls.addLayout(strat_layout, 1)

        layout.addLayout(controls)

        # Info labels
        self._info_label = QLabel("")
        self._info_label.setStyleSheet(f"color: {theme().text_muted}; font-size: {TypeScale.SM}px;")
        layout.addWidget(self._info_label)

        layout.addStretch()

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        self._generate_btn = QPushButton("Generate")
        self._generate_btn.setEnabled(False)
        self._generate_btn.setStyleSheet(f"""
            QPushButton {{
                padding: {Spacing.SM}px {Spacing.XL}px;
                font-weight: bold;
            }}
        """)
        self._generate_btn.clicked.connect(self._on_generate)
        btn_layout.addWidget(self._generate_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        return page

    def _create_progress_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Spacing.MD)

        title = QLabel("Staccato")
        title.setStyleSheet(f"font-size: {TypeScale.XL}px; font-weight: bold;")
        layout.addWidget(title)

        self._progress_label = QLabel("Preparing...")
        self._progress_label.setStyleSheet(f"color: {theme().text_secondary};")
        layout.addWidget(self._progress_label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setMinimum(0)
        self._progress_bar.setMaximum(100)
        layout.addWidget(self._progress_bar)

        layout.addStretch()

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self._on_cancel)
        btn_layout.addWidget(cancel_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        return page

    @Slot()
    def _on_select_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Music File", "", _AUDIO_FORMATS,
        )
        if not path:
            return

        self._music_path = Path(path)
        self._file_label.setText(self._music_path.name)
        self._file_label.setStyleSheet(f"color: {theme().text_primary};")
        self._generate_btn.setEnabled(False)
        self._info_label.setText("Analyzing audio...")
        self._waveform.clear()

        self._analyze_audio()

    @Slot(bool)
    def _on_stem_toggled(self, checked: bool):
        """Show/hide stem dropdown and check dependency availability."""
        self._stem_combo.setVisible(checked)
        if checked:
            # Check if demucs-infer is available — the worker thread will
            # install on demand if needed, but warn the user upfront about
            # the download size so they can opt out before waiting.
            from core.feature_registry import check_feature_ready
            available, _missing = check_feature_ready("stem_separation")
            if not available:
                reply = QMessageBox.question(
                    self,
                    "Stem Separation",
                    "Stem separation requires demucs + torch (~2 GB download).\n\n"
                    "Dependencies will be installed automatically when you "
                    "generate the sequence. Continue?",
                    QMessageBox.Yes | QMessageBox.No,
                )
                if reply != QMessageBox.Yes:
                    self._stem_checkbox.setChecked(False)
                    return

        # Re-analyze if we already have a file loaded
        if self._music_path:
            self._info_label.setText("Re-analyzing...")
            self._generate_btn.setEnabled(False)
            self._waveform.clear()
            self._analyze_audio()

    @Slot(str)
    def _on_stem_changed(self, _stem_text: str):
        """Re-analyze when the user changes the stem selection."""
        if self._music_path and self._stem_checkbox.isChecked():
            self._info_label.setText("Re-analyzing...")
            self._generate_btn.setEnabled(False)
            self._waveform.clear()
            self._analyze_audio()

    def _get_stem_name(self) -> str | None:
        """Return the selected stem name if separation is enabled."""
        if self._stem_checkbox.isChecked():
            return self._stem_combo.currentText().lower()
        return None

    def _analyze_audio(self):
        """Start background audio analysis."""
        if self._analyze_worker and self._analyze_worker.isRunning():
            self._analyze_worker.cancel()
            self._analyze_worker.wait(2000)

        self._handler_executed = False

        stem_name = self._get_stem_name()
        stems_cache_dir = None
        if stem_name:
            from core.settings import load_settings
            settings = load_settings()
            stems_cache_dir = settings.stems_cache_dir

        self._analyze_worker = StaccatoAnalyzeWorker(
            self._music_path,
            stem_name=stem_name,
            stems_cache_dir=stems_cache_dir,
            parent=self,
        )
        self._analyze_worker.audio_ready.connect(
            self._on_audio_ready, Qt.UniqueConnection,
        )
        self._analyze_worker.error.connect(
            self._on_analyze_error, Qt.UniqueConnection,
        )
        self._analyze_worker.progress_message.connect(
            self._on_analyze_progress, Qt.UniqueConnection,
        )
        self._analyze_worker.start()

    @Slot(str)
    def _on_analyze_progress(self, message: str):
        """Update info label with progress from the analyze worker."""
        self._info_label.setText(message)

    @Slot(object, object)
    def _on_audio_ready(self, analysis: AudioAnalysis, samples: np.ndarray):
        if self._handler_executed:
            return
        self._handler_executed = True

        self._audio_analysis = analysis
        self._audio_samples = samples

        # Update waveform and info using current sensitivity/strategy
        self._on_sensitivity_or_strategy_changed()
        self._generate_btn.setEnabled(True)

    def _get_filtered_markers(self) -> list[float]:
        """Get cut-point markers filtered by current strategy and sensitivity.

        Sensitivity 1 = only the strongest onsets, 10 = all onsets.
        For beats/downbeats, sensitivity has no effect (all are used).
        """
        analysis = self._audio_analysis
        if not analysis:
            return []

        strategy = self._strategy_combo.currentText().lower()
        if strategy == "downbeats":
            return analysis.downbeat_times
        elif strategy == "beats":
            return analysis.beat_times

        # Onsets: filter by strength threshold based on sensitivity
        sensitivity = self._sensitivity_slider.value()
        if sensitivity >= 10 or not analysis.onset_strengths:
            return analysis.onset_times

        # Map slider 1-10 to threshold 0.9-0.0
        # Slider 1 (fewer cuts) = threshold 0.9 (only strongest)
        # Slider 10 (more cuts) = threshold 0.0 (all onsets)
        threshold = (10 - sensitivity) / 10.0

        filtered = [
            t for t, s in zip(analysis.onset_times, analysis.onset_strengths)
            if s >= threshold
        ]
        return filtered if filtered else analysis.onset_times[:1]

    @Slot()
    def _on_sensitivity_or_strategy_changed(self):
        """Update waveform and info when sensitivity or strategy changes."""
        if not self._audio_analysis or self._audio_samples is None:
            return

        markers = self._get_filtered_markers()
        self._waveform.set_audio_data(
            samples=self._audio_samples,
            duration=self._audio_analysis.duration_seconds,
            beat_times=self._audio_analysis.beat_times,
            onset_times=markers,
        )

        # Update info label
        duration_str = f"{self._audio_analysis.duration_seconds:.1f}s"
        stem_label = ""
        stem_name = self._get_stem_name()
        if stem_name:
            stem_label = f" ({stem_name} stem)"
        self._info_label.setText(
            f"{self._audio_analysis.tempo_bpm:.0f} BPM · {len(markers)} cut points · "
            f"{duration_str}{stem_label} · {len(self._clips)} clips available"
        )

    @Slot(str)
    def _on_analyze_error(self, error_msg: str):
        self._info_label.setText(f"Analysis failed: {error_msg}")
        self._generate_btn.setEnabled(False)

    @Slot()
    def _on_generate(self):
        if not self._audio_analysis:
            return

        self._handler_executed = False
        self._stack.setCurrentIndex(1)
        self._progress_bar.setValue(0)
        self._progress_label.setText("Preparing...")

        strategy_text = self._strategy_combo.currentText().lower()
        cut_times = self._get_filtered_markers()

        self._generate_worker = StaccatoGenerateWorker(
            clips=self._clips,
            audio_analysis=self._audio_analysis,
            strategy=strategy_text,
            cut_times=cut_times,
            parent=self,
        )
        self._generate_worker.progress_update.connect(self._on_progress_update)
        self._generate_worker.progress_message.connect(self._on_progress_message)
        self._generate_worker.finished_sequence.connect(
            self._on_finished, Qt.UniqueConnection,
        )
        self._generate_worker.error.connect(
            self._on_generate_error, Qt.UniqueConnection,
        )
        self._generate_worker.start()

    @Slot(int, int)
    def _on_progress_update(self, current: int, total: int):
        if total > 0:
            pct = int(current / total * 100)
            self._progress_bar.setValue(pct)

    @Slot(str)
    def _on_progress_message(self, message: str):
        self._progress_label.setText(message)

    def _create_results_page(self) -> QWidget:
        """Create the results page shown after generation completes."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Spacing.MD)

        title = QLabel("Staccato")
        title.setStyleSheet(f"font-size: {TypeScale.XL}px; font-weight: bold;")
        layout.addWidget(title)

        self._results_summary = QLabel("")
        self._results_summary.setWordWrap(True)
        self._results_summary.setStyleSheet(f"color: {theme().text_secondary};")
        layout.addWidget(self._results_summary)

        layout.addStretch()

        # Action buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        save_report_btn = QPushButton("Save Debug Report")
        save_report_btn.setToolTip(
            "Export an interactive HTML report with onset/distance analysis"
        )
        save_report_btn.clicked.connect(self._on_save_debug_report)
        btn_layout.addWidget(save_report_btn)

        use_btn = QPushButton("Use Sequence")
        use_btn.setStyleSheet(f"""
            QPushButton {{
                padding: {Spacing.SM}px {Spacing.XL}px;
                font-weight: bold;
            }}
        """)
        use_btn.clicked.connect(self._on_use_sequence)
        btn_layout.addWidget(use_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        return page

    @Slot(object)
    def _on_finished(self, sequence_data):
        if self._handler_executed:
            return
        self._handler_executed = True

        self._sequence_data = sequence_data

        # Extract debug info from StaccatoResult or from worker
        if hasattr(sequence_data, 'debug'):
            self._debug_info = sequence_data.debug
        elif self._generate_worker and hasattr(self._generate_worker, '_debug_info'):
            self._debug_info = self._generate_worker._debug_info

        # Update results summary
        n_slots = len(sequence_data)
        unique_clips = len({
            getattr(entry[0], 'id', i) for i, entry in enumerate(sequence_data)
        })
        summary = f"Generated {n_slots} slots using {unique_clips} unique clips."
        if self._debug_info:
            summary += f"\nStrategy: {self._debug_info.strategy}"
        self._results_summary.setText(summary)

        self._stack.setCurrentIndex(2)

    @Slot()
    def _on_save_debug_report(self):
        """Save the debug report as an interactive HTML file."""
        if not self._debug_info:
            QMessageBox.warning(self, "No Debug Data", "No debug data available.")
            return

        from core.settings import load_settings
        settings = load_settings()
        default_dir = str(settings.export_dir) if settings.export_dir else ""
        default_path = Path(default_dir) / "staccato_debug_report.html"

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Debug Report", str(default_path),
            "HTML Files (*.html);;All Files (*)",
        )
        if not path:
            return

        try:
            from core.remix.staccato_report import save_staccato_report
            save_staccato_report(self._debug_info, Path(path))
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))
        except Exception as e:
            logger.error(f"Failed to save debug report: {e}")
            QMessageBox.critical(
                self, "Save Failed", f"Could not save report:\n{e}",
            )

    @Slot()
    def _on_use_sequence(self):
        """Emit the sequence and close the dialog."""
        if self._sequence_data is not None:
            self.sequence_ready.emit(self._sequence_data)
        self.accept()

    @Slot(str)
    def _on_generate_error(self, error_msg: str):
        logger.error("Staccato generation error: %s", error_msg)
        self._stack.setCurrentIndex(0)
        QMessageBox.critical(
            self, "Staccato Error", f"Generation failed:\n{error_msg}",
        )

    @Slot()
    def _on_cancel(self):
        for worker in (self._analyze_worker, self._generate_worker):
            if worker and worker.isRunning():
                worker.cancel()
                worker.wait(3000)
        self._analyze_worker = None
        self._generate_worker = None
        self.reject()

    def closeEvent(self, event):
        for worker in (self._analyze_worker, self._generate_worker):
            if worker and worker.isRunning():
                worker.cancel()
                worker.wait(3000)
        super().closeEvent(event)
