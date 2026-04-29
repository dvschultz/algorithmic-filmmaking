"""Simple list view for imported audio sources in the Collect tab."""

from typing import Optional

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)

from models.audio_source import AudioSource


class AudioLibraryList(QWidget):
    """A simple table-based list of imported audio sources.

    Shows filename and duration, with a per-row Remove button.

    Signals:
        audio_source_selected: Emitted when a row is clicked (audio: AudioSource)
        remove_requested: Emitted when a row's Remove button is clicked (audio_source_id: str)
    """

    audio_source_selected = Signal(object)  # AudioSource
    remove_requested = Signal(str)  # audio source id
    transcribe_requested = Signal(str)  # audio source id

    _COL_FILENAME = 0
    _COL_DURATION = 1
    _COL_TRANSCRIBE = 2
    _COL_REMOVE = 3

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._sources: list[AudioSource] = []
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._table = QTableWidget(0, 4, self)
        self._table.setHorizontalHeaderLabels(["Filename", "Duration", "", ""])
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setShowGrid(False)
        self._table.setAlternatingRowColors(True)

        header = self._table.horizontalHeader()
        header.setSectionResizeMode(self._COL_FILENAME, QHeaderView.Stretch)
        header.setSectionResizeMode(self._COL_DURATION, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(self._COL_TRANSCRIBE, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(self._COL_REMOVE, QHeaderView.ResizeToContents)

        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self._table)

    def set_sources(self, sources: list[AudioSource]) -> None:
        """Replace all rows with the given audio sources."""
        self._sources = list(sources)
        self._table.setRowCount(0)
        for audio in self._sources:
            self._append_row(audio)

    def _append_row(self, audio: AudioSource) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)

        filename_item = QTableWidgetItem(audio.filename)
        filename_item.setData(Qt.UserRole, audio.id)
        self._table.setItem(row, self._COL_FILENAME, filename_item)

        duration_item = QTableWidgetItem(audio.duration_str)
        self._table.setItem(row, self._COL_DURATION, duration_item)

        transcribe_btn = QPushButton("Transcribed" if audio.transcript else "Transcribe")
        if audio.transcript:
            transcribe_btn.setEnabled(False)
            transcribe_btn.setToolTip(f"{len(audio.transcript)} segments")
        else:
            transcribe_btn.setToolTip("Run Whisper on this audio source")
        transcribe_btn.clicked.connect(
            lambda _checked=False, aid=audio.id: self.transcribe_requested.emit(aid)
        )
        self._table.setCellWidget(row, self._COL_TRANSCRIBE, transcribe_btn)

        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(lambda _checked=False, aid=audio.id: self.remove_requested.emit(aid))
        self._table.setCellWidget(row, self._COL_REMOVE, remove_btn)

    def get_sources(self) -> list[AudioSource]:
        return list(self._sources)

    def count(self) -> int:
        return len(self._sources)

    def selected_audio_source(self) -> Optional[AudioSource]:
        row = self._table.currentRow()
        if row < 0 or row >= len(self._sources):
            return None
        return self._sources[row]

    def _on_selection_changed(self) -> None:
        audio = self.selected_audio_source()
        if audio is not None:
            self.audio_source_selected.emit(audio)
