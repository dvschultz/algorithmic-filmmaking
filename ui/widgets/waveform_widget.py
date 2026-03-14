"""Waveform visualization widget for audio with beat/onset markers."""

import numpy as np

from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QPainter, QPen, QColor, QPainterPath

from ui.theme import theme


class WaveformWidget(QWidget):
    """Displays an audio waveform with beat and onset markers overlaid.

    Call set_audio_data() to provide waveform samples and marker positions.
    The widget repaints automatically when data changes.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._samples: np.ndarray | None = None
        self._beat_times: list[float] = []
        self._onset_times: list[float] = []
        self._duration: float = 0.0
        self.setMinimumHeight(80)
        self.setMaximumHeight(120)

    def set_audio_data(
        self,
        samples: np.ndarray,
        duration: float,
        beat_times: list[float] | None = None,
        onset_times: list[float] | None = None,
    ):
        """Update the waveform data and markers.

        Args:
            samples: Audio samples (mono, any sample rate)
            duration: Total audio duration in seconds
            beat_times: List of beat timestamps (seconds)
            onset_times: List of onset timestamps (seconds)
        """
        self._samples = samples
        self._duration = duration
        self._beat_times = beat_times or []
        self._onset_times = onset_times or []
        self.update()

    def clear(self):
        """Clear all waveform data."""
        self._samples = None
        self._beat_times = []
        self._onset_times = []
        self._duration = 0.0
        self.update()

    def paintEvent(self, event):
        if self._samples is None or self._duration <= 0:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        mid_y = h / 2

        t = theme()
        bg_color = QColor(t.background_secondary)
        painter.fillRect(self.rect(), bg_color)

        # Downsample waveform to pixel width
        samples = self._samples
        n_samples = len(samples)
        if n_samples == 0:
            return

        bins = min(w, n_samples)
        chunk_size = max(1, n_samples // bins)
        peaks = []
        for i in range(bins):
            start = i * chunk_size
            end = min(start + chunk_size, n_samples)
            chunk = np.abs(samples[start:end])
            peaks.append(float(np.max(chunk)) if len(chunk) > 0 else 0.0)

        max_peak = max(peaks) if peaks else 1.0
        if max_peak == 0:
            max_peak = 1.0

        # Draw waveform as filled shape
        waveform_color = QColor(t.accent_primary)
        waveform_color.setAlpha(140)

        top_path = QPainterPath()
        bottom_path = QPainterPath()
        top_path.moveTo(0, mid_y)
        bottom_path.moveTo(0, mid_y)

        x_scale = w / max(len(peaks), 1)
        for i, peak in enumerate(peaks):
            x = i * x_scale
            amplitude = (peak / max_peak) * (h * 0.45)
            top_path.lineTo(x, mid_y - amplitude)
            bottom_path.lineTo(x, mid_y + amplitude)

        top_path.lineTo(w, mid_y)
        bottom_path.lineTo(w, mid_y)

        painter.setPen(Qt.NoPen)
        painter.setBrush(waveform_color)
        painter.drawPath(top_path)
        painter.drawPath(bottom_path)

        # Draw beat markers (subtle vertical lines)
        if self._beat_times and self._duration > 0:
            beat_pen = QPen(QColor(t.text_muted))
            beat_pen.setWidth(1)
            beat_pen.setStyle(Qt.DashLine)
            painter.setPen(beat_pen)
            for bt in self._beat_times:
                x = (bt / self._duration) * w
                if 0 <= x <= w:
                    painter.drawLine(int(x), 0, int(x), h)

        # Draw onset markers (prominent vertical lines)
        if self._onset_times and self._duration > 0:
            onset_color = QColor(t.accent_secondary)
            onset_color.setAlpha(180)
            onset_pen = QPen(onset_color)
            onset_pen.setWidth(1)
            painter.setPen(onset_pen)
            for ot in self._onset_times:
                x = (ot / self._duration) * w
                if 0 <= x <= w:
                    painter.drawLine(int(x), 0, int(x), h)

        painter.end()
