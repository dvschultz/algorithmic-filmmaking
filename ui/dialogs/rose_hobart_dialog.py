"""Rose Hobart dialog — face-filter sequencer.

Users provide 1-3 reference images of a person, and the system uses
InsightFace/ArcFace face embeddings to keep only clips where that
person's face appears. Named after Joseph Cornell's 1936 film.
"""

import logging
from pathlib import Path

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QFont, QImage, QPainter, QPen, QPixmap

from core.analysis.faces import SENSITIVITY_PRESETS, order_matched_clips
from ui.theme import theme, UISizes
from ui.workers.base import CancellableWorker

logger = logging.getLogger(__name__)

# Map dialog display names to faces.py preset keys
_SENSITIVITY_DISPLAY_TO_KEY = {
    "Strict": "strict",
    "Balanced": "balanced",
    "Loose": "loose",
}

_ORDERING_OPTIONS = [
    "Original Order",
    "By Duration",
    "By Color",
    "By Brightness",
    "By Confidence",
    "Random",
]


class RoseHobartWorker(CancellableWorker):
    """Background worker for Rose Hobart face matching and sequence generation."""

    progress_message = Signal(str)  # Status message
    match_found = Signal(int)  # Running match count
    finished_sequence = Signal(list)  # list of (Clip, Source)

    def __init__(
        self,
        reference_image_paths: list[Path],
        clips: list[tuple],  # list of (Clip, Source)
        sensitivity_preset: str,
        ordering: str,
        sample_interval: float,
        parent=None,
    ):
        super().__init__(parent)
        self._reference_paths = reference_image_paths
        self._clips = clips
        self._sensitivity = sensitivity_preset
        self._ordering = ordering
        self._sample_interval = sample_interval

    def run(self):
        """Run face matching pipeline."""
        self._log_start()
        try:
            from core.analysis.faces import (
                average_embeddings,
                compare_faces,
                extract_faces_from_clip,
                extract_faces_from_image,
            )

            # Step 1: Extract reference face embeddings
            self.progress_message.emit("Extracting reference face embeddings...")
            ref_embeddings = []
            for path in self._reference_paths:
                faces = extract_faces_from_image(path)
                if faces:
                    best = max(faces, key=lambda f: f["confidence"])
                    ref_embeddings.append(best["embedding"])

            if not ref_embeddings:
                self.error.emit("No faces detected in reference images.")
                return

            if len(ref_embeddings) > 1:
                ref_embeddings = [average_embeddings(ref_embeddings)]

            preset_key = _SENSITIVITY_DISPLAY_TO_KEY.get(self._sensitivity, "balanced")
            threshold = SENSITIVITY_PRESETS[preset_key]

            if self.is_cancelled():
                self._log_cancelled()
                return

            # Step 2: Process clips
            total = len(self._clips)
            matched = []
            match_count = 0

            for i, (clip, source) in enumerate(self._clips):
                if self.is_cancelled():
                    self._log_cancelled()
                    return

                self.progress_message.emit(f"Processing clip {i + 1} of {total}...")

                if clip.face_embeddings is not None:
                    clip_faces = clip.face_embeddings
                else:
                    clip_faces = extract_faces_from_clip(
                        source_path=source.file_path,
                        start_frame=clip.start_frame,
                        end_frame=clip.end_frame,
                        fps=source.fps,
                        sample_interval=self._sample_interval,
                    )
                    clip.face_embeddings = clip_faces if clip_faces else []

                is_match, confidence = compare_faces(
                    ref_embeddings, clip_faces, threshold
                )
                if is_match:
                    matched.append((clip, source, confidence))
                    match_count += 1
                    self.match_found.emit(match_count)

            if self.is_cancelled():
                self._log_cancelled()
                return

            if not matched:
                self.finished_sequence.emit([])
                return

            # Step 3: Order matched clips (shared function)
            ordered = order_matched_clips(matched, self._ordering)

            if not self.is_cancelled():
                self.finished_sequence.emit(ordered)

            self._log_complete()

        except Exception as e:
            if not self.is_cancelled():
                logger.error(f"Rose Hobart generation error: {e}", exc_info=True)
                self.error.emit("Face matching failed. Check logs for details.")


class _RefImageExtractWorker(CancellableWorker):
    """Tiny worker that extracts faces from a single reference image off the main thread."""

    faces_extracted = Signal(str, list)  # image_path_str, faces_list

    def __init__(self, image_path: Path, parent=None):
        super().__init__(parent)
        self._image_path = image_path

    def run(self):
        from core.analysis.faces import extract_faces_from_image
        faces = extract_faces_from_image(self._image_path)
        self.faces_extracted.emit(str(self._image_path), faces)


class _ReferenceImageWidget(QWidget):
    """Widget displaying a reference image thumbnail with face bbox overlay."""

    remove_requested = Signal()

    def __init__(self, image_path: Path, faces: list[dict], parent=None):
        super().__init__(parent)
        self._image_path = image_path
        self._faces = faces

        self.setFixedSize(120, 120)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Image label with face overlay
        self._image_label = QLabel()
        self._image_label.setFixedSize(110, 96)
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setStyleSheet("border: 1px solid #555;")
        self._load_image_with_overlay()

        # Remove button
        remove_btn = QPushButton("X")
        remove_btn.setFixedSize(20, 20)
        remove_btn.clicked.connect(self.remove_requested.emit)

        top = QHBoxLayout()
        top.addWidget(self._image_label, 1)
        top.addWidget(remove_btn, 0, Qt.AlignTop)

        layout.addLayout(top)

        # Warning if no face
        if not faces:
            warn = QLabel("No face detected")
            warn.setStyleSheet("color: #ff6b6b; font-size: 10px;")
            warn.setAlignment(Qt.AlignCenter)
            layout.addWidget(warn)

    def _load_image_with_overlay(self):
        """Load image and draw face bounding box overlay."""
        pixmap = QPixmap(str(self._image_path))
        if pixmap.isNull():
            self._image_label.setText("Error")
            return

        if self._faces:
            # Draw bbox for highest confidence face
            best = max(self._faces, key=lambda f: f["confidence"])
            painter = QPainter(pixmap)
            pen = QPen(Qt.green)
            pen.setWidth(max(2, pixmap.width() // 100))
            painter.setPen(pen)
            x, y, w, h = best["bbox"]
            painter.drawRect(x, y, w, h)
            painter.end()

        scaled = pixmap.scaled(
            self._image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self._image_label.setPixmap(scaled)

    @property
    def has_face(self) -> bool:
        return len(self._faces) > 0


class RoseHobartDialog(QDialog):
    """Face-filter sequencer dialog.

    Opens as a modal dialog where users select reference images of a person,
    configure sensitivity and ordering, and generate a filtered sequence.

    Signals:
        sequence_ready: Emitted with list of (Clip, Source) tuples
    """

    sequence_ready = Signal(list)

    PAGE_CONFIG = 0
    PAGE_PROGRESS = 1

    def __init__(
        self,
        clips,
        sources_by_id,
        parent=None,
    ):
        super().__init__(parent)
        self.clips = clips
        self.sources_by_id = sources_by_id
        self.worker: RoseHobartWorker | None = None
        self._ref_extract_worker: _RefImageExtractWorker | None = None
        self._ref_widgets: list[_ReferenceImageWidget] = []

        self.setWindowTitle("Rose Hobart")
        self.setModal(True)
        self.setMinimumSize(500, 400)
        self.resize(600, 500)

        self._setup_ui()
        self._apply_theme()

        if theme().changed:
            theme().changed.connect(self._apply_theme)

    def _setup_ui(self):
        """Build the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Stacked widget for pages
        self.stack = QStackedWidget()
        layout.addWidget(self.stack)

        # Page 0: Config
        self.config_page = self._create_config_page()
        self.stack.addWidget(self.config_page)

        # Page 1: Progress
        self.progress_page = self._create_progress_page()
        self.stack.addWidget(self.progress_page)

        # Bottom buttons
        btn_layout = QHBoxLayout()
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._on_cancel)

        self.generate_btn = QPushButton("Generate")
        self.generate_btn.setEnabled(False)
        self.generate_btn.clicked.connect(self._on_generate)

        btn_layout.addWidget(self.cancel_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(self.generate_btn)
        layout.addLayout(btn_layout)

    def _create_config_page(self) -> QWidget:
        """Create the configuration page."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # Title
        title = QLabel("Select reference images of the person to isolate")
        title_font = QFont()
        title_font.setPointSize(12)
        title.setFont(title_font)
        layout.addWidget(title)

        # Reference images section
        ref_section = QHBoxLayout()
        ref_section.setSpacing(8)

        self._ref_container = QHBoxLayout()
        self._ref_container.setSpacing(8)
        ref_section.addLayout(self._ref_container)

        self.add_ref_btn = QPushButton("+ Add Reference Image")
        self.add_ref_btn.setMinimumHeight(UISizes.BUTTON_MIN_HEIGHT)
        self.add_ref_btn.clicked.connect(self._on_add_reference)
        ref_section.addWidget(self.add_ref_btn, 0, Qt.AlignTop)
        ref_section.addStretch()

        layout.addLayout(ref_section)

        # Settings row
        settings = QHBoxLayout()
        settings.setSpacing(16)

        # Sensitivity
        sens_label = QLabel("Sensitivity:")
        sens_label.setFixedWidth(UISizes.FORM_LABEL_WIDTH_NARROW)
        settings.addWidget(sens_label)

        self.sensitivity_combo = QComboBox()
        self.sensitivity_combo.setMinimumHeight(UISizes.COMBO_BOX_MIN_HEIGHT)
        self.sensitivity_combo.addItems(list(_SENSITIVITY_DISPLAY_TO_KEY.keys()))
        self.sensitivity_combo.setCurrentText("Balanced")
        self.sensitivity_combo.setToolTip(
            "Strict: frontal faces only\n"
            "Balanced: good accuracy, allows angled faces\n"
            "Loose: permissive, may include ambiguous matches"
        )
        settings.addWidget(self.sensitivity_combo)

        # Ordering
        order_label = QLabel("Ordering:")
        order_label.setFixedWidth(UISizes.FORM_LABEL_WIDTH_NARROW)
        settings.addWidget(order_label)

        self.ordering_combo = QComboBox()
        self.ordering_combo.setMinimumHeight(UISizes.COMBO_BOX_MIN_HEIGHT)
        self.ordering_combo.addItems(_ORDERING_OPTIONS)
        settings.addWidget(self.ordering_combo)

        settings.addStretch()
        layout.addLayout(settings)

        # Sampling interval
        sample_row = QHBoxLayout()
        sample_label = QLabel("Sample interval:")
        sample_label.setFixedWidth(UISizes.FORM_LABEL_WIDTH_NARROW)
        sample_row.addWidget(sample_label)

        self.sample_spin = QDoubleSpinBox()
        self.sample_spin.setMinimumHeight(UISizes.COMBO_BOX_MIN_HEIGHT)
        self.sample_spin.setRange(0.25, 5.0)
        self.sample_spin.setValue(1.0)
        self.sample_spin.setSingleStep(0.25)
        self.sample_spin.setSuffix(" sec")
        self.sample_spin.setToolTip("How often to sample frames for face detection")
        sample_row.addWidget(self.sample_spin)
        sample_row.addStretch()
        layout.addLayout(sample_row)

        layout.addStretch()
        return page

    def _create_progress_page(self) -> QWidget:
        """Create the progress page."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)

        header = QLabel("Searching for faces...")
        header_font = QFont()
        header_font.setPointSize(14)
        header_font.setBold(True)
        header.setFont(header_font)
        layout.addWidget(header)

        layout.addSpacing(20)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(0)  # Indeterminate
        layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("Starting...")
        self.progress_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.progress_label)

        self.match_count_label = QLabel("Matches found: 0")
        self.match_count_label.setAlignment(Qt.AlignCenter)
        match_font = QFont()
        match_font.setPointSize(12)
        self.match_count_label.setFont(match_font)
        layout.addWidget(self.match_count_label)

        layout.addStretch()
        return page

    # ──────────────────────────────────────────────────────────
    # Reference images
    # ──────────────────────────────────────────────────────────

    def _on_add_reference(self):
        """Add a reference image via file picker (async face extraction)."""
        if len(self._ref_widgets) >= 3:
            QMessageBox.information(self, "Limit Reached", "Maximum 3 reference images.")
            return

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Reference Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp)",
        )
        if not path:
            return

        # Disable button while extracting
        self.add_ref_btn.setEnabled(False)
        self.add_ref_btn.setText("Detecting face...")

        # Run face extraction off the main thread
        worker = _RefImageExtractWorker(Path(path), parent=self)
        worker.faces_extracted.connect(self._on_ref_faces_extracted)
        worker.finished.connect(worker.deleteLater)
        worker.start()
        self._ref_extract_worker = worker  # prevent GC

    @Slot(str, list)
    def _on_ref_faces_extracted(self, image_path_str: str, faces: list):
        """Handle async face extraction result."""
        self._ref_extract_worker = None
        image_path = Path(image_path_str)

        widget = _ReferenceImageWidget(image_path, faces, parent=self)
        widget.remove_requested.connect(lambda w=widget: self._remove_reference(w))
        self._ref_container.addWidget(widget)
        self._ref_widgets.append(widget)

        self._update_generate_enabled()

        if len(self._ref_widgets) >= 3:
            self.add_ref_btn.setEnabled(False)
            self.add_ref_btn.setText("+ Add Reference Image")
        else:
            self.add_ref_btn.setEnabled(True)
            self.add_ref_btn.setText("+ Add Reference Image")

    def _remove_reference(self, widget: _ReferenceImageWidget):
        """Remove a reference image."""
        if widget in self._ref_widgets:
            self._ref_widgets.remove(widget)
            self._ref_container.removeWidget(widget)
            widget.deleteLater()
            self.add_ref_btn.setEnabled(True)
            self._update_generate_enabled()

    def _update_generate_enabled(self):
        """Enable Generate only when at least 1 reference has a detected face."""
        has_face = any(w.has_face for w in self._ref_widgets)
        self.generate_btn.setEnabled(has_face)

    # ──────────────────────────────────────────────────────────
    # Generation
    # ──────────────────────────────────────────────────────────

    def _on_generate(self):
        """Start face matching."""
        # Collect reference image paths (only those with faces)
        ref_paths = [
            w._image_path for w in self._ref_widgets if w.has_face
        ]
        if not ref_paths:
            return

        # Build clip pairs
        clip_pairs = []
        for clip in self.clips:
            source = self.sources_by_id.get(clip.source_id)
            if source:
                clip_pairs.append((clip, source))

        if not clip_pairs:
            QMessageBox.warning(self, "No Clips", "No clips available for processing.")
            return

        # Switch to progress page
        self.stack.setCurrentIndex(self.PAGE_PROGRESS)
        self.generate_btn.setEnabled(False)
        self.generate_btn.setText("Generating...")

        # Start worker
        self.worker = RoseHobartWorker(
            reference_image_paths=ref_paths,
            clips=clip_pairs,
            sensitivity_preset=self.sensitivity_combo.currentText(),
            ordering=self.ordering_combo.currentText(),
            sample_interval=self.sample_spin.value(),
            parent=self,
        )
        self.worker.progress_message.connect(self._on_progress, Qt.UniqueConnection)
        self.worker.match_found.connect(self._on_match_found, Qt.UniqueConnection)
        self.worker.finished_sequence.connect(self._on_finished, Qt.UniqueConnection)
        self.worker.error.connect(self._on_error, Qt.UniqueConnection)
        self.worker.start()

    @Slot(str)
    def _on_progress(self, message: str):
        """Update progress label."""
        self.progress_label.setText(message)

    @Slot(int)
    def _on_match_found(self, count: int):
        """Update match count."""
        self.match_count_label.setText(f"Matches found: {count}")

    @Slot(list)
    def _on_finished(self, sequence: list):
        """Handle generation completion."""
        if not sequence:
            # Zero matches — let user adjust sensitivity
            self.progress_label.setText("No clips matched the reference person.")
            self.match_count_label.setText("0 matches")
            self.stack.setCurrentIndex(self.PAGE_CONFIG)
            self.generate_btn.setEnabled(True)
            self.generate_btn.setText("Retry")
            QMessageBox.information(
                self,
                "No Matches",
                "No clips matched the reference person.\n\n"
                "Try adjusting the sensitivity to 'Loose' or adding more reference images.",
            )
            return

        self.sequence_ready.emit(sequence)
        self.accept()

    @Slot(str)
    def _on_error(self, message: str):
        """Handle generation error."""
        self.stack.setCurrentIndex(self.PAGE_CONFIG)
        self.generate_btn.setEnabled(True)
        self.generate_btn.setText("Generate")
        QMessageBox.warning(self, "Error", message)

    def _on_cancel(self):
        """Cancel and close."""
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(3000)
        self.reject()

    # ──────────────────────────────────────────────────────────
    # Theme
    # ──────────────────────────────────────────────────────────

    def _apply_theme(self):
        """Apply current theme colors."""
        t = theme()
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {t.colors.background_primary};
                color: {t.colors.text_primary};
            }}
            QLabel {{
                color: {t.colors.text_primary};
            }}
            QPushButton {{
                background-color: {t.colors.background_secondary};
                color: {t.colors.text_primary};
                border: 1px solid {t.colors.border_primary};
                border-radius: 4px;
                padding: 6px 12px;
                min-height: {UISizes.BUTTON_MIN_HEIGHT}px;
            }}
            QPushButton:hover {{
                background-color: {t.colors.background_tertiary};
            }}
            QPushButton:disabled {{
                opacity: 0.5;
            }}
            QComboBox {{
                background-color: {t.colors.background_secondary};
                color: {t.colors.text_primary};
                border: 1px solid {t.colors.border_primary};
                border-radius: 4px;
                padding: 4px;
            }}
            QDoubleSpinBox {{
                background-color: {t.colors.background_secondary};
                color: {t.colors.text_primary};
                border: 1px solid {t.colors.border_primary};
                border-radius: 4px;
                padding: 4px;
            }}
            QProgressBar {{
                background-color: {t.colors.background_secondary};
                border: 1px solid {t.colors.border_primary};
                border-radius: 4px;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background-color: {t.colors.accent_blue};
                border-radius: 3px;
            }}
        """)
