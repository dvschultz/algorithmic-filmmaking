"""Rose Hobart dialog — face-filter sequencer.

Users provide 1-3 reference images of a person, and the system uses
InsightFace/ArcFace face embeddings to keep only clips where that
person's face appears. Named after Joseph Cornell's 1936 film.
"""

import logging
import json
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

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
from PySide6.QtCore import Qt, Signal, Slot, QTimer
from PySide6.QtGui import QFont, QPainter, QPen, QPixmap

from core.analysis.faces import SENSITIVITY_PRESETS, order_matched_clips
from core.operations.faces import (
    FaceApplication,
    FaceOptions,
    FaceOutcome,
    face_task,
    run_faces,
)
from core.jobs.media import media_stamp
from models.analysis_record import AnalysisRecord
from ui.theme import theme, UISizes
from ui.workers.base import CancellableWorker

if TYPE_CHECKING:
    from core.project import Project
    from models.clip import Clip, Source

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
        clips: list[tuple["Clip", "Source"]],
        sensitivity_preset: str,
        ordering: str,
        sample_interval: float,
        parent=None,
    ):
        super().__init__(parent)
        self._reference_paths = tuple(reference_image_paths)
        self._reference_stamps = {
            path: media_stamp(path) for path in self._reference_paths
        }
        self._clips = deepcopy(clips)
        self.tasks = tuple(face_task(clip, source) for clip, source in clips)
        self.options = FaceOptions(sample_interval)
        self.outcomes: tuple[FaceOutcome, ...] = ()
        self.result: list[tuple["Clip", "Source"]] | None = None
        self.failure: str | None = None
        self._sensitivity = sensitivity_preset
        self._ordering = ordering
        self._sample_interval = sample_interval

    def run(self) -> None:
        """Run face matching pipeline."""
        self._log_start()
        try:
            # Ensure InsightFace is installed before attempting face detection
            from core.feature_registry import check_feature_ready, install_for_feature

            available, _missing = check_feature_ready("face_detect")
            if not available:
                self.progress_message.emit("Installing face detection dependencies...")
                if not install_for_feature("face_detect"):
                    self.failure = (
                        "Failed to install face detection dependencies (insightface)"
                    )
                    self.error.emit(self.failure)
                    return

            from core.analysis.faces import (
                average_embeddings,
                compare_faces,
                extract_faces_from_image,
            )

            # Step 1: Extract reference face embeddings
            if not self.references_current():
                raise ValueError("Reference images changed while queued")
            self.progress_message.emit("Extracting reference face embeddings...")
            ref_embeddings = []
            for path in self._reference_paths:
                if self.is_cancelled():
                    return
                faces = extract_faces_from_image(path)
                if faces:
                    best = max(faces, key=lambda f: f["confidence"])
                    ref_embeddings.append(best["embedding"])

            if not ref_embeddings:
                self.failure = "No faces detected in reference images."
                self.error.emit(self.failure)
                return

            if len(ref_embeddings) > 1:
                ref_embeddings = [average_embeddings(ref_embeddings)]

            preset_key = _SENSITIVITY_DISPLAY_TO_KEY.get(self._sensitivity, "balanced")
            threshold = SENSITIVITY_PRESETS[preset_key]

            if self.is_cancelled():
                self._log_cancelled()
                return

            if not self.references_current():
                raise ValueError("Reference images changed during matching")

            # Step 2: Process clips
            total = len(self._clips)
            matched = []
            match_count = 0
            self.outcomes = run_faces(
                self.tasks,
                self.options,
                cancel_event=self._cancel_event,
                progress=lambda current, count: self.progress_message.emit(
                    f"Analyzing clip {current} of {count}..."
                ),
            )

            for i, (clip, source) in enumerate(self._clips):
                if self.is_cancelled():
                    self._log_cancelled()
                    return

                self.progress_message.emit(f"Processing clip {i + 1} of {total}...")

                outcome = self.outcomes[i]
                if not outcome.has_result:
                    if not self.is_cancelled():
                        raise ValueError(
                            outcome.message or "Face analysis did not complete"
                        )
                    return
                clip_faces = outcome.face_dicts()

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
                if not self.references_current():
                    raise ValueError("Reference images changed during matching")
                self.result = []
                self.finished_sequence.emit([])
                return

            # Step 3: Order matched clips (shared function)
            ordered = order_matched_clips(matched, self._ordering)

            if not self.is_cancelled():
                if not self.references_current():
                    raise ValueError("Reference images changed during matching")
                self.result = ordered
                self.finished_sequence.emit(ordered)

            self._log_complete()

        except Exception as e:
            if not self.is_cancelled():
                self.failure = "Face matching failed. Check logs for details."
                logger.error(f"Rose Hobart generation error: {e}", exc_info=True)
                self.error.emit(self.failure)

    def references_current(self) -> bool:
        return all(
            stamp is not None and media_stamp(path) == stamp
            for path, stamp in self._reference_stamps.items()
        )


class _RefImageExtractWorker(CancellableWorker):
    """Tiny worker that extracts faces from a single reference image off the main thread."""

    faces_extracted = Signal(str, list)  # image_path_str, faces_list

    def __init__(self, image_path: Path, parent=None):
        super().__init__(parent)
        self._image_path = image_path
        self.result: list[dict] = []

    def run(self) -> None:
        from core.analysis.faces import extract_faces_from_image

        try:
            faces = extract_faces_from_image(self._image_path)
            if not self.is_cancelled():
                self.result = faces
                self.faces_extracted.emit(str(self._image_path), faces)
        except Exception:
            logger.exception("Reference face extraction failed")


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
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setStyleSheet("border: 1px solid #555;")
        self._load_image_with_overlay()

        # Remove button
        remove_btn = QPushButton("X")
        remove_btn.setFixedSize(20, 20)
        remove_btn.clicked.connect(self.remove_requested.emit)

        top = QHBoxLayout()
        top.addWidget(self._image_label, 1)
        top.addWidget(remove_btn, 0, Qt.AlignmentFlag.AlignTop)

        layout.addLayout(top)

        # Warning if no face
        if not faces:
            warn = QLabel("No face detected")
            warn.setStyleSheet("color: #ff6b6b; font-size: 10px;")
            warn.setAlignment(Qt.AlignmentFlag.AlignCenter)
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
        *,
        project: "Project | None" = None,
    ):
        super().__init__(parent)
        self.clips = clips
        self.sources_by_id = sources_by_id
        self.project = project
        self._application: FaceApplication | None = None
        self._closing = False
        self._retiring = False
        self._retire_timer = QTimer(self)
        self._retire_timer.setInterval(10)
        self._retire_timer.timeout.connect(self._retire_workers)
        self.worker: RoseHobartWorker | None = None
        self._ref_extract_worker: _RefImageExtractWorker | None = None
        self._ref_widgets: list[_ReferenceImageWidget] = []

        self.setWindowTitle("Rose Hobart")
        self.setModal(True)
        self.setMinimumSize(500, 400)
        self.resize(600, 500)

        self._setup_ui()
        self._apply_theme()

        changed = getattr(theme(), "changed", None)
        if changed:
            changed.connect(self._apply_theme)

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
        ref_section.addWidget(self.add_ref_btn, 0, Qt.AlignmentFlag.AlignTop)
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
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.progress_label)

        self.match_count_label = QLabel("Matches found: 0")
        self.match_count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
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
        if (
            self.worker is not None
            or self._ref_extract_worker is not None
            or self._closing
        ):
            return
        if len(self._ref_widgets) >= 3:
            QMessageBox.information(
                self, "Limit Reached", "Maximum 3 reference images."
            )
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
        self._ref_extract_worker = worker  # prevent GC
        worker.finished.connect(self._retire_workers)
        worker.start()

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

    def _on_generate(self) -> None:
        """Start face matching."""
        if (
            self.worker is not None
            or self._ref_extract_worker is not None
            or self._closing
        ):
            return
        # Collect reference image paths (only those with faces)
        ref_paths = [w._image_path for w in self._ref_widgets if w.has_face]
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
        self._application = (
            FaceApplication(self.project, self.worker.tasks, self.worker.options)
            if self.project is not None
            else None
        )
        self._submitted_pairs = clip_pairs
        self._submitted_values = [
            (deepcopy(c.to_dict()), deepcopy(s.to_dict())) for c, s in clip_pairs
        ]
        self.worker.progress_message.connect(
            self._on_progress, Qt.ConnectionType.UniqueConnection
        )
        self.worker.match_found.connect(
            self._on_match_found, Qt.ConnectionType.UniqueConnection
        )
        self.worker.finished.connect(self._retire_workers)
        self.worker.start()

    def _inputs_current(self) -> bool:
        if self._closing:
            return False
        if self._application is not None and (
            self.project is None
            or self.project.path != self._application.path
            or self.project.session.session_id != self._application.session_id
            or getattr(self.parent(), "_project", self.project) is not self.project
        ):
            return False
        return all(
            c.to_dict() == cv
            and s.to_dict() == sv
            and (
                self.project is None
                or (
                    self.project.clips_by_id.get(c.id) is c
                    and self.project.sources_by_id.get(s.id) is s
                )
            )
            for (c, s), (cv, sv) in zip(self._submitted_pairs, self._submitted_values)
        )

    @Slot()
    def _retire_workers(self) -> None:
        """Retain workers until native exit, then publish on the owner thread."""
        if self._retiring:
            return
        self._retiring = True
        try:
            self._finish_workers()
        finally:
            self._retiring = False

    def _finish_workers(self) -> None:
        worker = self.worker
        if worker is not None:
            if worker.isRunning() or not worker.wait(0):
                self._retire_timer.start()
                return
            self.worker = None
            worker.deleteLater()
            if not self._closing and not worker.is_cancelled():
                current = self._inputs_current() and worker.references_current()
                applied = current
                if (
                    current
                    and self._application is not None
                    and self.project is not None
                ):
                    try:
                        for index, outcome in enumerate(worker.outcomes):
                            if (
                                not self._inputs_current()
                                or not outcome.can_apply
                                or outcome.record_json is None
                            ):
                                applied = False
                                break
                            expected, source = deepcopy(worker._clips[index])
                            expected.analysis_records["face_embeddings"] = (
                                AnalysisRecord.from_dict(
                                    json.loads(outcome.record_json)
                                )
                            )
                            if outcome.status == "succeeded":
                                expected.face_embeddings = outcome.face_dicts()
                            self._submitted_values[index] = (
                                expected.to_dict(),
                                source.to_dict(),
                            )
                            if (
                                not self._application.apply(self.project, outcome)
                                or not self._inputs_current()
                            ):
                                applied = False
                                break
                    except Exception:
                        logger.exception("Rose Hobart publication failed")
                        applied = False
                if self._closing:
                    pass
                elif not applied:
                    self._on_error("Project inputs changed. Run face matching again.")
                elif worker.failure is not None or worker.result is None:
                    self._on_error(worker.failure or "Face matching did not complete.")
                else:
                    pairs = {c.id: (c, s) for c, s in self._submitted_pairs}
                    self._on_finished([pairs[c.id] for c, _ in worker.result])
        reference = self._ref_extract_worker
        if reference is not None:
            if reference.isRunning() or not reference.wait(0):
                self._retire_timer.start()
                return
            self._ref_extract_worker = None
            reference.deleteLater()
            if not self._closing:
                self._on_ref_faces_extracted(
                    str(reference._image_path), reference.result
                )
        self._retire_timer.stop()
        if self._closing:
            super().reject()

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
        self.reject()

    def reject(self) -> None:
        self._closing = True
        for worker in (self.worker, self._ref_extract_worker):
            if worker is not None:
                worker.cancel()
        self._retire_workers()

    def closeEvent(self, event):
        if self.worker is not None or self._ref_extract_worker is not None:
            event.ignore()
            self.reject()
        else:
            super().closeEvent(event)

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
