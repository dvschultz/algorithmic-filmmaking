"""Video player component using MPV (libmpv) via python-mpv."""

from __future__ import annotations

import locale
import logging
import threading
import time
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QStyle,
    QComboBox,
)
from ui.widgets.styled_slider import StyledSlider
from PySide6.QtCore import Qt, Slot, Signal, QObject

try:
    import mpv
except OSError:
    mpv = None  # libmpv not available (e.g. Windows CI without DLL)

from core.constants import PLAYBACK_SPEEDS, DEFAULT_SPEED_INDEX
from ui.theme import theme, TypeScale, Spacing, UISizes

logger = logging.getLogger(__name__)


class MpvSignalBridge(QObject):
    """Bridges MPV property observer callbacks (event thread) to Qt signals (main thread).

    MPV property observers fire on MPV's event thread. Emitting Qt signals from
    that thread is safe — Qt auto-queues cross-thread signal emissions. This class
    keeps all observer function references alive to prevent garbage collection
    (which would silently stop updates).
    """

    position_changed = Signal(float)   # seconds
    duration_changed = Signal(float)   # seconds
    pause_changed = Signal(bool)
    media_loaded = Signal()
    eof_reached = Signal()

    # Throttle position updates to max 5 Hz to avoid flooding the main thread
    _POSITION_EMIT_INTERVAL = 0.2  # seconds

    def __init__(self, mpv_instance: mpv.MPV):
        super().__init__()
        self._mpv = mpv_instance
        self._last_position_emit: float = 0.0
        self._throttle_lock = threading.Lock()
        # Store observer references to prevent GC
        self._observers: list = []
        self._register_observers()

    def _register_observers(self):
        """Register MPV property observers."""
        def on_time_pos(_name, value):
            if value is not None:
                with self._throttle_lock:
                    now = time.monotonic()
                    if now - self._last_position_emit < self._POSITION_EMIT_INTERVAL:
                        return
                    self._last_position_emit = now
                self.position_changed.emit(value)

        def on_duration(_name, value):
            if value is not None:
                self.duration_changed.emit(value)

        def on_pause(_name, value):
            if value is not None:
                self.pause_changed.emit(value)

        def on_idle(_name, value):
            # core-idle True + eof-reached True = file ended
            if value and self._mpv.eof_reached:
                self.eof_reached.emit()

        # Store references
        self._observers.extend([on_time_pos, on_duration, on_pause, on_idle])

        self._mpv.observe_property('time-pos', on_time_pos)
        self._mpv.observe_property('duration', on_duration)
        self._mpv.observe_property('pause', on_pause)
        self._mpv.observe_property('core-idle', on_idle)

    def cleanup(self):
        """Remove observers before shutdown."""
        for obs in self._observers:
            try:
                self._mpv.unobserve_property(obs)
            except Exception:
                pass
        self._observers.clear()


class VideoPlayer(QWidget):
    """Video player with playback controls, powered by MPV."""

    # Signals
    position_updated = Signal(int)  # position in milliseconds
    duration_changed = Signal(int)  # duration in milliseconds
    media_loaded = Signal()  # fires when file is ready to play
    playback_state_changed = Signal(bool)  # True=playing, False=paused/stopped

    def __init__(self):
        super().__init__()
        # Assert locale — PySide6 can corrupt LC_NUMERIC (only safe on main thread)
        if threading.current_thread() is threading.main_thread():
            try:
                locale.setlocale(locale.LC_NUMERIC, 'C')
            except locale.Error:
                logger.warning("Failed to set LC_NUMERIC to 'C'")

        # Range playback (clip mode)
        self._clip_start_ms: Optional[int] = None
        self._clip_end_ms: Optional[int] = None

        # Internal state
        self._duration_s: float = 0.0
        self._shutdown_event = threading.Event()
        self._cached_speed: float = 1.0
        self._ab_a_seconds: Optional[float] = None
        self._ab_b_seconds: Optional[float] = None

        self._setup_ui()
        self._setup_player()

    @property
    def _shutting_down(self) -> bool:
        return self._shutdown_event.is_set()

    def _setup_ui(self):
        """Set up the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Header
        header = QLabel("Preview")
        header.setStyleSheet(f"font-weight: bold; font-size: {TypeScale.MD}px; padding: {Spacing.SM}px;")
        layout.addWidget(header)

        # Video container — MPV renders into this widget via window ID
        self.video_widget = QWidget()
        self.video_widget.setMinimumSize(400, 300)
        self.video_widget.setStyleSheet("background-color: #000000;")
        self.video_widget.setAttribute(Qt.WA_DontCreateNativeAncestors)
        self.video_widget.setAttribute(Qt.WA_NativeWindow)
        layout.addWidget(self.video_widget, 1)

        # Controls
        controls = QHBoxLayout()
        controls.setContentsMargins(8, 8, 8, 8)

        # Play/Pause button
        self.play_btn = QPushButton()
        self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_btn.setFixedSize(44, 32)
        self.play_btn.setAccessibleName("Play")
        self.play_btn.setToolTip("Play/Pause video")
        self.play_btn.clicked.connect(self._toggle_playback)
        controls.addWidget(self.play_btn)

        # Stop button
        self.stop_btn = QPushButton()
        self.stop_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self.stop_btn.setFixedSize(44, 32)
        self.stop_btn.setAccessibleName("Stop")
        self.stop_btn.setToolTip("Stop video")
        self.stop_btn.clicked.connect(self.stop)
        controls.addWidget(self.stop_btn)

        # Frame step backward button
        self.frame_back_btn = QPushButton()
        self.frame_back_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaSkipBackward))
        self.frame_back_btn.setFixedSize(32, 32)
        self.frame_back_btn.setAccessibleName("Frame back")
        self.frame_back_btn.setToolTip("Step one frame backward")
        self.frame_back_btn.clicked.connect(self.frame_step_backward)
        controls.addWidget(self.frame_back_btn)

        # Frame step forward button
        self.frame_fwd_btn = QPushButton()
        self.frame_fwd_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaSkipForward))
        self.frame_fwd_btn.setFixedSize(32, 32)
        self.frame_fwd_btn.setAccessibleName("Frame forward")
        self.frame_fwd_btn.setToolTip("Step one frame forward")
        self.frame_fwd_btn.clicked.connect(self.frame_step_forward)
        controls.addWidget(self.frame_fwd_btn)

        # Position slider
        self.position_slider = StyledSlider(Qt.Horizontal)
        self.position_slider.setRange(0, 0)
        self.position_slider.setAccessibleName("Video position")
        self.position_slider.sliderMoved.connect(self._set_position)
        controls.addWidget(self.position_slider, 1)

        # Time label
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setStyleSheet("font-family: monospace;")
        controls.addWidget(self.time_label)

        # Speed selector
        self.speed_combo = QComboBox()
        self.speed_combo.setMinimumHeight(UISizes.COMBO_BOX_MIN_HEIGHT)
        self.speed_combo.setFixedWidth(70)
        self.speed_combo.setAccessibleName("Playback speed")
        self.speed_combo.setToolTip("Playback speed")
        self._speed_values = PLAYBACK_SPEEDS
        for speed in self._speed_values:
            self.speed_combo.addItem(f"{speed}x")
        self.speed_combo.setCurrentIndex(DEFAULT_SPEED_INDEX)
        self.speed_combo.currentIndexChanged.connect(self._on_speed_changed)
        controls.addWidget(self.speed_combo)

        layout.addLayout(controls)

        # A/B Loop controls (optional — hidden by default, shown via show_ab_loop_controls)
        self._ab_loop_row = QWidget()
        ab_layout = QHBoxLayout(self._ab_loop_row)
        ab_layout.setContentsMargins(8, 0, 8, 8)
        ab_layout.setSpacing(4)

        ab_label = QLabel("A/B Loop:")
        ab_label.setStyleSheet(f"font-size: {TypeScale.SM}px; color: {theme().text_muted};")
        ab_layout.addWidget(ab_label)

        self.set_a_btn = QPushButton("Set A")
        self.set_a_btn.setFixedHeight(24)
        self.set_a_btn.setToolTip("Set loop start at current position")
        self.set_a_btn.clicked.connect(self._on_set_a)
        ab_layout.addWidget(self.set_a_btn)

        self.set_b_btn = QPushButton("Set B")
        self.set_b_btn.setFixedHeight(24)
        self.set_b_btn.setToolTip("Set loop end at current position")
        self.set_b_btn.clicked.connect(self._on_set_b)
        ab_layout.addWidget(self.set_b_btn)

        self.clear_ab_btn = QPushButton("Clear")
        self.clear_ab_btn.setFixedHeight(24)
        self.clear_ab_btn.setToolTip("Clear A/B loop markers")
        self.clear_ab_btn.clicked.connect(self._on_clear_ab)
        ab_layout.addWidget(self.clear_ab_btn)

        self._ab_loop_label = QLabel("")
        self._ab_loop_label.setStyleSheet(f"font-family: monospace; font-size: {TypeScale.SM}px;")
        ab_layout.addWidget(self._ab_loop_label, 1)

        ab_layout.addStretch()
        self._ab_loop_row.setVisible(False)
        layout.addWidget(self._ab_loop_row)

        # Apply themed styling to all control buttons
        self._apply_control_styles()

    def _apply_control_styles(self):
        """Apply themed styling to all control buttons."""
        t = theme()
        btn_style = f"""
            QPushButton {{
                background-color: {t.background_elevated};
                border: 1px solid {t.border_secondary};
                border-radius: 4px;
                color: {t.text_primary};
            }}
            QPushButton:hover {{
                background-color: {t.background_tertiary};
                border-color: {t.border_focus};
            }}
            QPushButton:pressed {{
                background-color: {t.accent_blue};
            }}
        """
        for btn in (self.play_btn, self.stop_btn, self.frame_back_btn, self.frame_fwd_btn):
            btn.setStyleSheet(btn_style)

        ab_btn_style = f"""
            QPushButton {{
                background-color: {t.background_elevated};
                border: 1px solid {t.border_secondary};
                border-radius: 3px;
                color: {t.text_secondary};
                font-size: {TypeScale.SM}px;
                padding: 2px 8px;
            }}
            QPushButton:hover {{
                background-color: {t.background_tertiary};
                border-color: {t.accent_blue};
                color: {t.text_primary};
            }}
        """
        for btn in (self.set_a_btn, self.set_b_btn, self.clear_ab_btn):
            btn.setStyleSheet(ab_btn_style)

    def show_ab_loop_controls(self, visible: bool = True):
        """Show or hide the A/B loop controls row.

        Enable this on VideoPlayer instances that should allow manual A/B looping
        (e.g., sequence tab player, not sidebar).
        """
        self._ab_loop_row.setVisible(visible)

    def _setup_player(self):
        """Set up MPV player instance."""
        if mpv is None:
            raise RuntimeError(
                "libmpv is not available. Install mpv/libmpv for your platform."
            )
        wid = str(int(self.video_widget.winId()))
        self._mpv = mpv.MPV(
            wid=wid,
            vo='gpu',
            keep_open='yes',
            idle='yes',
            hwdec='auto',
            hr_seek='yes',
            input_default_bindings=False,
            input_vo_keyboard=False,
            osc=False,
            log_handler=self._mpv_log_handler,
        )
        self._mpv.pause = True  # Start paused

        # Signal bridge — MPV callbacks → Qt signals
        self._bridge = MpvSignalBridge(self._mpv)
        self._bridge.position_changed.connect(self._on_position_changed)
        self._bridge.duration_changed.connect(self._on_duration_changed)
        self._bridge.pause_changed.connect(self._on_pause_changed)
        self._bridge.media_loaded.connect(self._on_file_loaded)
        self._bridge.eof_reached.connect(self._on_eof)

        # Register file-loaded event to detect when media is ready
        @self._mpv.event_callback('file-loaded')
        def on_file_loaded(event):
            self._bridge.media_loaded.emit()
        self._file_loaded_cb = on_file_loaded  # prevent GC

    def _mpv_log_handler(self, loglevel: str, component: str, message: str):
        """Route MPV log messages to Python logger."""
        msg = f"[mpv/{component}] {message.strip()}"
        if loglevel in ('fatal', 'error'):
            logger.error(msg)
        elif loglevel == 'warn':
            logger.warning(msg)
        elif loglevel == 'info':
            logger.info(msg)
        else:
            logger.debug(msg)

    # --- Public API (preserved from QMediaPlayer version) ---

    def load_video(self, path: Path):
        """Load a video file."""
        if self._shutting_down:
            return
        self._clip_start_ms = None
        self._clip_end_ms = None
        self._mpv.pause = True
        self._mpv.play(str(path))

    def _safe_mpv_command(self, fn, *args, **kwargs):
        """Execute an MPV command with standard error handling."""
        if self._shutting_down:
            return
        try:
            fn(*args, **kwargs)
        except mpv.ShutdownError:
            pass
        except Exception:
            logger.warning("MPV command failed", exc_info=True)

    def seek_to(self, seconds: float):
        """Seek to a position in seconds."""
        self._safe_mpv_command(self._mpv.seek, seconds, 'absolute', 'exact')

    def set_clip_range(self, start_seconds: float, end_seconds: float):
        """Set playback range to a specific clip.

        Uses MPV's ab-loop properties for frame-accurate looping.
        """
        self._clip_start_ms = int(start_seconds * 1000)
        self._clip_end_ms = int(end_seconds * 1000)
        clip_duration = self._clip_end_ms - self._clip_start_ms
        self.position_slider.setRange(0, clip_duration)

        # Set MPV A/B loop for frame-accurate boundaries
        self._mpv.ab_loop_a = start_seconds
        self._mpv.ab_loop_b = end_seconds

        # Seek to clip start
        self._safe_mpv_command(self._mpv.seek, start_seconds, 'absolute', 'exact')
        self._update_time_label(self._clip_start_ms)

    def clear_clip_range(self):
        """Clear clip range, allowing full video playback."""
        self._clip_start_ms = None
        self._clip_end_ms = None
        self._mpv.ab_loop_a = 'no'
        self._mpv.ab_loop_b = 'no'
        duration_ms = int(self._duration_s * 1000)
        self.position_slider.setRange(0, duration_ms)

    def play_range(self, start_seconds: float, end_seconds: float):
        """Play a specific range of the video."""
        self.set_clip_range(start_seconds, end_seconds)
        self._mpv.pause = False

    def play(self):
        """Start or resume playback."""
        if self._shutting_down:
            return
        self._mpv.pause = False

    def pause(self):
        """Pause playback."""
        if self._shutting_down:
            return
        self._mpv.pause = True

    def stop(self):
        """Stop playback and return to start."""
        if self._shutting_down:
            return
        self._mpv.pause = True
        if self._clip_start_ms is not None:
            start_s = self._clip_start_ms / 1000.0
            self._safe_mpv_command(self._mpv.seek, start_s, 'absolute', 'exact')
        else:
            self._safe_mpv_command(self._mpv.seek, 0, 'absolute')

    def shutdown(self):
        """Clean up MPV resources. Must be called from main thread before app exit."""
        if self._shutting_down:
            return
        self._shutdown_event.set()
        try:
            self._bridge.cleanup()
            self._mpv.terminate()
        except Exception:
            logger.debug("MPV shutdown exception (may be normal)", exc_info=True)

    @property
    def is_playing(self) -> bool:
        """Whether the player is currently playing."""
        if self._shutting_down:
            return False
        try:
            return not self._mpv.pause
        except Exception:
            return False

    @property
    def duration_ms(self) -> int:
        """Total duration in milliseconds."""
        return int(self._duration_s * 1000)

    @property
    def playback_speed(self) -> float:
        """Current playback speed multiplier."""
        return self._cached_speed

    @playback_speed.setter
    def playback_speed(self, speed: float):
        """Set playback speed multiplier."""
        try:
            self._mpv.speed = speed
            self._cached_speed = speed
        except Exception:
            if not self._shutting_down:
                logger.debug("Failed to set playback speed", exc_info=True)

    @property
    def mute(self) -> bool:
        """Whether audio is muted."""
        try:
            return self._mpv.mute
        except Exception:
            return False

    @mute.setter
    def mute(self, value: bool):
        """Set mute state."""
        try:
            self._mpv.mute = value
        except Exception:
            if not self._shutting_down:
                logger.debug("Failed to set mute state", exc_info=True)

    # --- New feature methods ---

    def frame_step_forward(self):
        """Advance one frame forward."""
        if self._shutting_down:
            return
        self._mpv.pause = True
        self._mpv.frame_step()

    def frame_step_backward(self):
        """Step one frame backward."""
        if self._shutting_down:
            return
        self._mpv.pause = True
        self._mpv.frame_back_step()

    def set_ab_loop(self, a_seconds: float, b_seconds: float):
        """Set manual A/B loop markers.

        Only use when clip range is NOT active.
        """
        if self._clip_start_ms is not None:
            return  # Clip range owns the ab-loop properties
        self._mpv.ab_loop_a = a_seconds
        self._mpv.ab_loop_b = b_seconds
        self._ab_a_seconds = a_seconds
        self._ab_b_seconds = b_seconds
        self._update_ab_label()

    def clear_ab_loop(self):
        """Clear manual A/B loop markers."""
        if self._clip_start_ms is not None:
            return  # Clip range owns the ab-loop properties
        self._mpv.ab_loop_a = 'no'
        self._mpv.ab_loop_b = 'no'
        self._ab_a_seconds = None
        self._ab_b_seconds = None
        self._update_ab_label()

    def set_speed(self, speed: float):
        """Set playback speed, updating both MPV engine and UI combo box."""
        self.playback_speed = speed
        for i in range(self.speed_combo.count()):
            if abs(PLAYBACK_SPEEDS[i] - speed) < 0.001:
                self.speed_combo.setCurrentIndex(i)
                break

    def set_speed_control_enabled(self, enabled: bool):
        """Enable or disable the speed control widget.

        Disable during automated sequence playback to avoid timeline sync issues.
        """
        self.speed_combo.setEnabled(enabled)
        if not enabled:
            # Reset to 1x during automated playback
            self.speed_combo.setCurrentIndex(DEFAULT_SPEED_INDEX)
            self.playback_speed = 1.0

    # --- Internal handlers ---

    def _toggle_playback(self):
        """Toggle play/pause."""
        if self._shutting_down:
            return
        if not self._mpv.pause:
            self._mpv.pause = True
        else:
            self._mpv.pause = False

    @Slot(int)
    def _on_speed_changed(self, index: int):
        """Handle speed combo box change."""
        if 0 <= index < len(self._speed_values):
            self.playback_speed = self._speed_values[index]

    def _on_set_a(self):
        """Set A/B loop start at current position."""
        if self._clip_start_ms is not None:
            return  # Clip range owns the ab-loop
        try:
            pos = self._mpv.time_pos
            if pos is not None:
                self._ab_a_seconds = pos
                self._mpv.ab_loop_a = pos
                self._update_ab_label()
        except Exception:
            if not self._shutting_down:
                logger.debug("Failed to set A loop point", exc_info=True)

    def _on_set_b(self):
        """Set A/B loop end at current position."""
        if self._clip_start_ms is not None:
            return
        try:
            pos = self._mpv.time_pos
            if pos is not None:
                self._ab_b_seconds = pos
                self._mpv.ab_loop_b = pos
                self._update_ab_label()
        except Exception:
            if not self._shutting_down:
                logger.debug("Failed to set B loop point", exc_info=True)

    def _on_clear_ab(self):
        """Clear A/B loop markers."""
        self._ab_a_seconds = None
        self._ab_b_seconds = None
        self.clear_ab_loop()
        self._update_ab_label()

    def _update_ab_label(self):
        """Update the A/B loop status label."""
        parts = []
        if self._ab_a_seconds is not None:
            parts.append(f"A: {self._format_time(int(self._ab_a_seconds * 1000))}")
        if self._ab_b_seconds is not None:
            parts.append(f"B: {self._format_time(int(self._ab_b_seconds * 1000))}")
        self._ab_loop_label.setText("  ".join(parts) if parts else "")

    def _set_position(self, position: int):
        """Set playback position from slider.

        In clip mode, slider position is relative to clip start.
        """
        if self._shutting_down:
            return
        if self._clip_start_ms is not None:
            absolute_ms = self._clip_start_ms + position
            seconds = absolute_ms / 1000.0
        else:
            seconds = position / 1000.0
        try:
            self._mpv.seek(seconds, 'absolute', 'exact')
        except mpv.ShutdownError:
            pass
        except Exception:
            if not self._shutting_down:
                logger.warning("_set_position seek failed", exc_info=True)

    @Slot(float)
    def _on_position_changed(self, seconds: float):
        """Handle position update from MPV."""
        if self._shutting_down:
            return
        position_ms = int(seconds * 1000)
        self.position_updated.emit(position_ms)

        if self._clip_start_ms is not None and self._clip_end_ms is not None:
            relative_ms = position_ms - self._clip_start_ms
            clip_duration = self._clip_end_ms - self._clip_start_ms
            relative_ms = max(0, min(relative_ms, clip_duration))
            self.position_slider.setValue(relative_ms)
            self._update_time_label(position_ms)
        else:
            self.position_slider.setValue(position_ms)
            current = self._format_time(position_ms)
            total = self._format_time(int(self._duration_s * 1000))
            self.time_label.setText(f"{current} / {total}")

    @Slot(float)
    def _on_duration_changed(self, seconds: float):
        """Handle duration update from MPV."""
        self._duration_s = seconds
        duration_ms = int(seconds * 1000)
        if self._clip_start_ms is None:
            self.position_slider.setRange(0, duration_ms)
        self.duration_changed.emit(duration_ms)

    @Slot(bool)
    def _on_pause_changed(self, paused: bool):
        """Handle pause state change from MPV."""
        if paused:
            self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        else:
            self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        self.playback_state_changed.emit(not paused)

    @Slot()
    def _on_file_loaded(self):
        """Handle file loaded event — media is ready to play."""
        self.media_loaded.emit()

    @Slot()
    def _on_eof(self):
        """Handle end-of-file reached."""
        # In clip mode, ab-loop handles looping automatically.
        # In full video mode, just pause at the end.
        if self._clip_start_ms is None:
            self._mpv.pause = True

    def _update_time_label(self, absolute_position_ms: int):
        """Update time label for clip mode."""
        if self._clip_start_ms is not None and self._clip_end_ms is not None:
            relative_pos = absolute_position_ms - self._clip_start_ms
            clip_duration = self._clip_end_ms - self._clip_start_ms
            relative_pos = max(0, min(relative_pos, clip_duration))
            current = self._format_time(relative_pos)
            total = self._format_time(clip_duration)
            self.time_label.setText(f"{current} / {total}")

    def _format_time(self, ms: int) -> str:
        """Format milliseconds as MM:SS."""
        seconds = ms // 1000
        mins = seconds // 60
        secs = seconds % 60
        return f"{mins:02d}:{secs:02d}"
