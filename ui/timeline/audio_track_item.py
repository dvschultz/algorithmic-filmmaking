"""Audio waveform track item for the timeline."""

import numpy as np

from PySide6.QtWidgets import QGraphicsRectItem, QGraphicsItem, QStyleOptionGraphicsItem
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QPen, QPainter, QPainterPath

from ui.theme import theme


AUDIO_TRACK_HEIGHT = 50
AUDIO_TRACK_LABEL = "Audio"


class AudioTrackItem(QGraphicsRectItem):
    """A non-interactive waveform track that displays below video tracks.

    Renders audio samples as a filled waveform shape, scaled to the
    timeline's pixels_per_second zoom level.
    """

    def __init__(self, y_position: float, width: float = 10000):
        super().__init__()

        self._samples: np.ndarray | None = None
        self._duration: float = 0.0
        self._pixels_per_second: float = 100.0
        self._cached_peaks: list[float] | None = None
        self._cached_width: int = 0

        self.setRect(0, y_position, width, AUDIO_TRACK_HEIGHT)
        self.setBrush(QBrush(theme().colors.qcolor("timeline_track")))
        self.setPen(QPen(Qt.NoPen))
        self.setZValue(-1)

        self.setFlag(QGraphicsItem.ItemIsSelectable, False)
        self.setFlag(QGraphicsItem.ItemIsMovable, False)

    def set_audio_data(self, samples: np.ndarray, duration: float):
        """Set the waveform data.

        Args:
            samples: Mono audio samples (1D numpy array)
            duration: Total audio duration in seconds
        """
        self._samples = samples
        self._duration = duration
        self._cached_peaks = None
        self.update()

    def set_pixels_per_second(self, pps: float):
        """Update zoom level and invalidate peak cache."""
        self._pixels_per_second = pps
        self._cached_peaks = None
        self.update()

    def set_width(self, width: float):
        """Update track width."""
        rect = self.rect()
        rect.setWidth(width)
        self.setRect(rect)

    def clear(self):
        """Clear waveform data."""
        self._samples = None
        self._duration = 0.0
        self._cached_peaks = None
        self.update()

    def paint(self, painter: QPainter, option, widget=None):
        """Draw the track background and waveform."""
        # Draw track background
        super().paint(painter, option, widget)

        if self._samples is None or self._duration <= 0:
            return

        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect()
        y_top = rect.y()
        h = rect.height()
        mid_y = y_top + h / 2

        # Calculate waveform width in pixels
        waveform_width = int(self._duration * self._pixels_per_second)
        if waveform_width <= 0:
            return

        # Compute peaks (cached)
        if self._cached_peaks is None or self._cached_width != waveform_width:
            self._cached_peaks = self._compute_peaks(waveform_width)
            self._cached_width = waveform_width

        peaks = self._cached_peaks
        if not peaks:
            return

        max_peak = max(peaks) if peaks else 1.0
        if max_peak == 0:
            max_peak = 1.0

        # Clip to visible portion for performance
        exposed = option.exposedRect
        start_x = max(0, int(exposed.left()))
        end_x = min(len(peaks), int(exposed.right()) + 1)
        if start_x >= end_x:
            return

        # Draw waveform (visible portion only)
        waveform_color = QColor(theme().accent_blue)
        waveform_color.setAlpha(140)

        top_path = QPainterPath()
        bottom_path = QPainterPath()
        first_amp = (peaks[start_x] / max_peak) * (h * 0.4) if start_x < len(peaks) else 0
        top_path.moveTo(start_x, mid_y - first_amp)
        bottom_path.moveTo(start_x, mid_y + first_amp)

        for i in range(start_x + 1, end_x):
            x = float(i)
            amplitude = (peaks[i] / max_peak) * (h * 0.4)
            top_path.lineTo(x, mid_y - amplitude)
            bottom_path.lineTo(x, mid_y + amplitude)

        top_path.lineTo(end_x - 1, mid_y)
        bottom_path.lineTo(end_x - 1, mid_y)

        painter.setPen(Qt.NoPen)
        painter.setBrush(waveform_color)
        painter.drawPath(top_path)
        painter.drawPath(bottom_path)

    def _compute_peaks(self, target_width: int) -> list[float]:
        """Downsample audio to target pixel width."""
        samples = self._samples
        n_samples = len(samples)
        bins = min(target_width, n_samples)
        if bins <= 0:
            return []

        chunk_size = max(1, n_samples // bins)
        peaks = []
        for i in range(bins):
            start = i * chunk_size
            end = min(start + chunk_size, n_samples)
            chunk = np.abs(samples[start:end])
            peaks.append(float(np.max(chunk)) if len(chunk) > 0 else 0.0)
        return peaks
