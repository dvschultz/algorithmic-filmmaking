"""Transcription settings UI section for SettingsDialog."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
)


def build_transcription_settings_group(owner, groq_models: list[str]) -> QGroupBox:
    """Build the transcription settings group and attach controls to the dialog."""
    transcription_group = QGroupBox("Transcription")
    transcription_layout = QVBoxLayout(transcription_group)

    backend_layout = QHBoxLayout()
    backend_layout.addWidget(QLabel("Backend:"))

    owner.transcription_backend_combo = QComboBox()
    owner.transcription_backend_combo.addItem(
        "Auto - MLX on Apple Silicon, otherwise faster-whisper",
        "auto",
    )
    owner.transcription_backend_combo.addItem(
        "faster-whisper - Local CPU/CUDA",
        "faster-whisper",
    )
    owner.transcription_backend_combo.addItem(
        "MLX Whisper - Local Apple Silicon GPU",
        "mlx-whisper",
    )
    owner.transcription_backend_combo.addItem("Groq - Cloud Whisper API", "groq")
    owner.transcription_backend_combo.setToolTip(
        "Choose where transcription runs.\n"
        "Groq uses the cloud and requires a Groq API key in Settings > API Keys."
    )
    owner.transcription_backend_combo.currentIndexChanged.connect(
        owner._on_transcription_backend_changed
    )
    backend_layout.addWidget(owner.transcription_backend_combo)
    transcription_layout.addLayout(backend_layout)

    model_layout = QHBoxLayout()
    owner.whisper_model_lbl = QLabel("Whisper Model:")
    model_layout.addWidget(owner.whisper_model_lbl)

    owner.transcription_model_combo = QComboBox()
    owner.transcription_model_combo.addItems(
        [
            "tiny.en - Fast, basic accuracy (39MB)",
            "small.en - Good balance (244MB)",
            "medium.en - Better accuracy (769MB)",
            "large-v3 - Best accuracy (1.5GB)",
        ]
    )
    owner.transcription_model_combo.setToolTip(
        "Larger models are more accurate but slower.\n"
        "Models are downloaded on first use."
    )
    model_layout.addWidget(owner.transcription_model_combo)
    transcription_layout.addLayout(model_layout)

    disk_layout = QHBoxLayout()
    disk_layout.addWidget(QLabel("Minimum free disk:"))
    owner.transcription_disk_spin = QDoubleSpinBox()
    owner.transcription_disk_spin.setRange(0.5, 100.0)
    owner.transcription_disk_spin.setSingleStep(0.5)
    owner.transcription_disk_spin.setDecimals(1)
    owner.transcription_disk_spin.setSuffix(" GB")
    owner.transcription_disk_spin.setToolTip(
        "Warn before local Whisper transcription if model cache storage is below this threshold."
    )
    disk_layout.addWidget(owner.transcription_disk_spin)
    disk_layout.addStretch()
    transcription_layout.addLayout(disk_layout)

    segmentation_layout = QHBoxLayout()
    segmentation_layout.addWidget(QLabel("Segments:"))
    owner.transcription_segmentation_combo = QComboBox()
    owner.transcription_segmentation_combo.addItem("Backend boundaries", "backend")
    owner.transcription_segmentation_combo.addItem("Sentences", "sentence")
    owner.transcription_segmentation_combo.addItem("Phrases", "phrase")
    owner.transcription_segmentation_combo.addItem("Fixed duration", "fixed")
    owner.transcription_segmentation_combo.setToolTip(
        "Choose how transcript segments are split after transcription."
    )
    owner.transcription_segmentation_combo.currentIndexChanged.connect(
        owner._on_transcript_segmentation_changed
    )
    segmentation_layout.addWidget(owner.transcription_segmentation_combo)

    owner.transcription_segment_seconds_spin = QDoubleSpinBox()
    owner.transcription_segment_seconds_spin.setRange(2.0, 120.0)
    owner.transcription_segment_seconds_spin.setSingleStep(1.0)
    owner.transcription_segment_seconds_spin.setDecimals(0)
    owner.transcription_segment_seconds_spin.setSuffix(" sec")
    owner.transcription_segment_seconds_spin.setToolTip(
        "Maximum segment length used when Segments is set to Fixed duration."
    )
    segmentation_layout.addWidget(owner.transcription_segment_seconds_spin)
    transcription_layout.addLayout(segmentation_layout)

    cloud_model_layout = QHBoxLayout()
    owner.transcription_cloud_model_lbl = QLabel("Groq Model:")
    cloud_model_layout.addWidget(owner.transcription_cloud_model_lbl)

    owner.transcription_cloud_model_combo = QComboBox()
    owner.transcription_cloud_model_combo.addItems(groq_models)
    owner.transcription_cloud_model_combo.setToolTip(
        "Groq Whisper model used when Backend is set to Groq."
    )
    cloud_model_layout.addWidget(owner.transcription_cloud_model_combo)
    transcription_layout.addLayout(cloud_model_layout)

    lang_layout = QHBoxLayout()
    lang_layout.addWidget(QLabel("Language:"))

    owner.transcription_lang_combo = QComboBox()
    owner.transcription_lang_combo.addItems(["English", "Auto-detect"])
    owner.transcription_lang_combo.setToolTip(
        "Select 'Auto-detect' for multi-language content.\n"
        "English is faster for English-only content."
    )
    lang_layout.addWidget(owner.transcription_lang_combo)
    lang_layout.addStretch()
    transcription_layout.addLayout(lang_layout)

    return transcription_group
