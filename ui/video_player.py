"""Video player component using MPV (libmpv) via python-mpv.

Uses mpv's OpenGL render API with QOpenGLWidget for reliable cross-platform
embedding. The --wid approach is unreliable on macOS where mpv falls back to
creating a separate pop-out window.
"""

from __future__ import annotations

import ctypes
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
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtGui import QOpenGLContext, QSurfaceFormat
from PySide6.QtCore import Qt, Slot, Signal, QObject, QMetaObject
from ui.widgets.styled_slider import StyledSlider

try:
    import mpv
except OSError:
    mpv = None  # libmpv not available (e.g. Windows CI without DLL)

from core.constants import PLAYBACK_SPEEDS, DEFAULT_SPEED_INDEX
from ui.theme import theme, TypeScale, Spacing, UISizes

logger = logging.getLogger(__name__)


class MpvGLWidget(QOpenGLWidget):
    """OpenGL widget that renders mpv video output via the render API.

    mpv renders each frame into Qt's OpenGL framebuffer, so the video appears
    inline as a normal widget — no separate window.
    """

    _frame_ready = Signal()  # Internal signal: mpv thread -> main thread repaint

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        # Match app-level default format in main.py on macOS to maximize odds of
        # successful Qt shared-context creation with QOpenGLWidget.
        fmt = QSurfaceFormat()
        fmt.setVersion(3, 2)
        fmt.setProfile(QSurfaceFormat.CoreProfile)
        self.setFormat(fmt)

        self._mpv: Optional[mpv.MPV] = None
        self._ctx = None  # MpvRenderContext
        self._proc_addr_func = None  # prevent GC of ctypes callback

    def set_mpv(self, mpv_instance: mpv.MPV):
        """Assign the mpv instance. Must be called before the widget is shown."""
        self._mpv = mpv_instance

    def initializeGL(self):
        """Create the mpv render context once the GL context is ready."""
        if self._mpv is None or self._ctx is not None:
            return

        def _get_proc_address(_ctx, name):
            gl_ctx = QOpenGLContext.currentContext()
            if gl_ctx is None:
                return 0
            gl_fmt = gl_ctx.format()
            profile_name = {
                QSurfaceFormat.NoProfile: "NoProfile",
                QSurfaceFormat.CoreProfile: "CoreProfile",
                QSurfaceFormat.CompatibilityProfile: "CompatibilityProfile",
            }.get(gl_fmt.profile(), str(gl_fmt.profile()))
            major, minor = gl_fmt.majorVersion(), gl_fmt.minorVersion()
            logger.debug(
                "MpvGLWidget current GL context: profile=%s version=%d.%d depth=%d",
                profile_name,
                major,
                minor,
                gl_fmt.depthBufferSize(),
            )
            if gl_fmt.profile() != QSurfaceFormat.CoreProfile or (major, minor) < (3, 2):
                logger.warning(
                    "Potentially incompatible GL context for mpv rendering: "
                    "profile=%s version=%d.%d (expected CoreProfile >= 3.2)",
                    profile_name,
                    major,
                    minor,
                )
            # PySide6 getProcAddress accepts bytes, returns int
            addr = gl_ctx.getProcAddress(name)
            return addr or 0

        self._proc_addr_func = mpv.MpvGlGetProcAddressFn(_get_proc_address)

        self._ctx = mpv.MpvRenderContext(
            self._mpv, 'opengl',
            opengl_init_params={'get_proc_address': self._proc_addr_func},
        )

        # mpv calls update_cb from its render thread when a new frame is ready.
        # We bridge to the main thread via a signal so update() is thread-safe.
        self._frame_ready.connect(self.update, Qt.QueuedConnection)
        self._ctx.update_cb = self._on_mpv_frame_ready

    def _on_mpv_frame_ready(self):
        """Called from mpv's render thread — emit signal to repaint on main thread."""
        self._frame_ready.emit()

    def paintGL(self):
        """Render the current mpv frame into Qt's framebuffer.

        Always renders — even when mpv has no new frame — to prevent Qt's FBO
        from showing stale compositor data (appears as a 'screen mirror' on macOS).
        """
        if self._ctx is None:
            return

        fbo = self.defaultFramebufferObject()
        ratio = self.devicePixelRatio()
        w = int(self.width() * ratio)
        h = int(self.height() * ratio)

        self._ctx.render(
            flip_y=True,
            opengl_fbo={'fbo': fbo, 'w': w, 'h': h, 'internal_format': 0},
        )
        self._ctx.report_swap()

    def cleanup(self):
        """Free the render context. Must be called while GL context is current."""
        if self._ctx is not None:
            self._ctx.update_cb = None
            self._ctx.free()
            self._ctx = None


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
    play_requested = Signal()  # emitted in sequence mode when play is clicked
    stop_requested = Signal()  # emitted in sequence mode when stop is clicked

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
        self._loop_clip = True  # Whether clip-range playback loops

        # Internal state
        self._duration_s: float = 0.0
        self._shutdown_event = threading.Event()
        self._cached_speed: float = 1.0
        self._ab_a_seconds: Optional[float] = None
        self._ab_b_seconds: Optional[float] = None
        self._chromatic_bar_color: Optional[tuple[int, int, int]] = None
        self._media_loaded = False
        self._pending_seek_seconds: Optional[float] = None
        self._pending_clip_range: Optional[tuple[float, float]] = None
        self._pending_play_on_load = False
        self._player_ready = False  # True once mpv is initialized
        self._mpv = None
        self._bridge = None
        self._pending_load: Optional[Path] = None  # queued load_video before init
        self._sequence_mode = False  # When True, play/stop emit signals instead of controlling MPV

        self._setup_ui()

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

        # Video container — mpv renders via OpenGL render API into this widget
        self.video_widget = MpvGLWidget()
        self.video_widget.setMinimumSize(400, 300)
        self.video_widget.setStyleSheet("background-color: #000000;")
        layout.addWidget(self.video_widget, 1)

        self._chromatic_color_bar = QWidget()
        self._chromatic_color_bar.setFixedHeight(16)
        self._chromatic_color_bar.setVisible(False)
        layout.addWidget(self._chromatic_color_bar, 0)

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
        self.stop_btn.clicked.connect(self._on_stop_clicked)
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

    def set_loop(self, loop: bool):
        """Set whether clip-range playback should loop.

        When False, playback pauses at the end of the clip range instead
        of looping back to the start.  Default is True.
        """
        self._loop_clip = loop

    def show_ab_loop_controls(self, visible: bool = True):
        """Show or hide the A/B loop controls row.

        Enable this on VideoPlayer instances that should allow manual A/B looping
        (e.g., sequence tab player, not sidebar).
        """
        self._ab_loop_row.setVisible(visible)

    def showEvent(self, event):
        """Initialize mpv when the widget is first shown.

        The OpenGL context must be ready before creating the render context,
        so we defer until the widget is visible.
        """
        super().showEvent(event)
        if not self._player_ready:
            self._setup_player()

    def _setup_player(self):
        """Set up MPV player instance with OpenGL render API."""
        if self._player_ready:
            return
        if mpv is None:
            raise RuntimeError(
                "libmpv is not available. Install mpv/libmpv for your platform."
            )

        # Create mpv with vo=libmpv — rendering is handled by MpvGLWidget
        self._mpv = mpv.MPV(
            vo='libmpv',
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

        # Connect mpv to the GL widget for rendering
        self.video_widget.set_mpv(self._mpv)
        # Force GL initialization if the context is already ready
        self.video_widget.makeCurrent()
        self.video_widget.initializeGL()
        self.video_widget.doneCurrent()

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
        self._player_ready = True

        # Replay any load_video call that arrived before init completed
        if self._pending_load is not None:
            pending = self._pending_load
            self._pending_load = None
            self.load_video(pending)

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
        """Load a video file.

        If mpv hasn't initialized yet (widget not shown), the path is queued
        and loaded automatically once the player is ready.
        """
        if self._shutting_down:
            return
        if not self._player_ready:
            self._pending_load = path
            return
        self._pending_load = None
        self._media_loaded = False
        self._pending_seek_seconds = None
        self._pending_clip_range = None
        self._pending_play_on_load = False
        self._clip_start_ms = None
        self._clip_end_ms = None
        self._safe_mpv_command(setattr, self._mpv, 'pause', True)
        self._safe_mpv_command(self._mpv.play, str(path))

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
        if not self._player_ready:
            return
        if not self._media_loaded:
            self._pending_seek_seconds = seconds
            return
        self._safe_mpv_command(self._mpv.seek, seconds, 'absolute', 'exact')

    def set_clip_range(self, start_seconds: float, end_seconds: float):
        """Set playback range to a specific clip.

        Uses MPV's ab-loop properties for frame-accurate looping when
        ``_loop_clip`` is True.  When looping is disabled, the ab-loop
        is not set and playback pauses at the clip end via ``_on_eof``.
        """
        if not self._player_ready:
            return
        self._clip_start_ms = int(start_seconds * 1000)
        self._clip_end_ms = int(end_seconds * 1000)
        clip_duration = self._clip_end_ms - self._clip_start_ms
        self.position_slider.setRange(0, clip_duration)
        self._pending_clip_range = (start_seconds, end_seconds)

        if not self._media_loaded:
            return

        self._apply_clip_range(start_seconds, end_seconds)

    def _apply_clip_range(self, start_seconds: float, end_seconds: float):
        """Apply clip range directly to MPV once media is ready."""
        if self._loop_clip:
            # Set MPV A/B loop for frame-accurate boundaries
            self._mpv.ab_loop_a = start_seconds
            self._mpv.ab_loop_b = end_seconds
        else:
            self._mpv.ab_loop_a = 'no'
            self._mpv.ab_loop_b = 'no'

        # Seek to clip start
        self._safe_mpv_command(self._mpv.seek, start_seconds, 'absolute', 'exact')
        self._update_time_label(self._clip_start_ms)

    def clear_clip_range(self):
        """Clear clip range, allowing full video playback."""
        if not self._player_ready:
            return
        self._clip_start_ms = None
        self._clip_end_ms = None
        self._pending_clip_range = None
        self._pending_play_on_load = False
        self._pending_seek_seconds = None
        if not self._media_loaded:
            return
        self._mpv.ab_loop_a = 'no'
        self._mpv.ab_loop_b = 'no'
        duration_ms = int(self._duration_s * 1000)
        self.position_slider.setRange(0, duration_ms)

    def play_range(self, start_seconds: float, end_seconds: float):
        """Play a specific range of the video."""
        self.set_clip_range(start_seconds, end_seconds)
        if not self._media_loaded:
            self._pending_play_on_load = True
            return
        self._mpv.pause = False

    def play(self):
        """Start or resume playback."""
        if self._shutting_down or not self._player_ready:
            return
        if not self._media_loaded:
            self._pending_play_on_load = True
            return
        self._mpv.pause = False

    def pause(self):
        """Pause playback."""
        if self._shutting_down or not self._player_ready:
            return
        self._mpv.pause = True

    def stop(self):
        """Stop playback and return to start."""
        if self._shutting_down or not self._player_ready:
            return
        self._mpv.pause = True
        if not self._media_loaded:
            return
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
        if not self._player_ready:
            return
        try:
            self._bridge.cleanup()
            # Free render context before terminating mpv
            self.video_widget.makeCurrent()
            self.video_widget.cleanup()
            self.video_widget.doneCurrent()
            self._mpv.terminate()
        except Exception:
            logger.debug("MPV shutdown exception (may be normal)", exc_info=True)

    @property
    def is_playing(self) -> bool:
        """Whether the player is currently playing."""
        if self._shutting_down or not self._player_ready:
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
        self._cached_speed = speed
        if not self._player_ready:
            return
        try:
            self._mpv.speed = speed
        except Exception:
            if not self._shutting_down:
                logger.debug("Failed to set playback speed", exc_info=True)

    @property
    def mute(self) -> bool:
        """Whether audio is muted."""
        if not self._player_ready:
            return False
        try:
            return self._mpv.mute
        except Exception:
            return False

    @mute.setter
    def mute(self, value: bool):
        """Set mute state."""
        if not self._player_ready:
            return
        try:
            self._mpv.mute = value
        except Exception:
            if not self._shutting_down:
                logger.debug("Failed to set mute state", exc_info=True)

    # --- New feature methods ---

    def frame_step_forward(self):
        """Advance one frame forward."""
        if self._shutting_down or not self._player_ready:
            return
        self._mpv.pause = True
        self._mpv.frame_step()

    def frame_step_backward(self):
        """Step one frame backward."""
        if self._shutting_down or not self._player_ready:
            return
        self._mpv.pause = True
        self._mpv.frame_back_step()

    def set_ab_loop(self, a_seconds: float, b_seconds: float):
        """Set manual A/B loop markers.

        Only use when clip range is NOT active.
        """
        if not self._player_ready or self._clip_start_ms is not None:
            return
        self._mpv.ab_loop_a = a_seconds
        self._mpv.ab_loop_b = b_seconds
        self._ab_a_seconds = a_seconds
        self._ab_b_seconds = b_seconds
        self._update_ab_label()

    def clear_ab_loop(self):
        """Clear manual A/B loop markers."""
        if not self._player_ready or self._clip_start_ms is not None:
            return
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

    def set_chromatic_color_bar(self, color: Optional[tuple[int, int, int]]):
        """Set the full-width chromatic color bar shown below the video.

        Args:
            color: RGB tuple for the active clip color, or None to hide the bar.
        """
        if color is None:
            self._chromatic_bar_color = None
            self._chromatic_color_bar.setVisible(False)
            return

        r, g, b = (max(0, min(255, int(c))) for c in color)
        self._chromatic_bar_color = (r, g, b)
        self._chromatic_color_bar.setStyleSheet(
            f"background-color: rgb({r}, {g}, {b}); border-top: 1px solid rgba(255,255,255,0.08);"
        )
        self._chromatic_color_bar.setVisible(True)

    def set_transforms(self, hflip: bool = False, vflip: bool = False, reverse: bool = False):
        """Apply per-clip transforms for sequence preview.

        Args:
            hflip: Horizontal flip
            vflip: Vertical flip
            reverse: Reverse playback direction
        """
        if not self._player_ready or self._shutting_down:
            return

        # Build video filter chain
        vf_parts = []
        if hflip:
            vf_parts.append("hflip")
        if vflip:
            vf_parts.append("vflip")

        try:
            self._mpv.vf = ",".join(vf_parts) if vf_parts else ""
        except Exception:
            logger.debug("Failed to set mpv vf", exc_info=True)

        # Reverse playback direction
        try:
            self._mpv["play-direction"] = "backward" if reverse else "forward"
        except Exception:
            logger.debug("Failed to set mpv play-direction", exc_info=True)

    def clear_transforms(self):
        """Remove all per-clip transforms."""
        self.set_transforms()

    # --- Internal handlers ---

    def _toggle_playback(self):
        """Toggle play/pause."""
        if self._sequence_mode:
            self.play_requested.emit()
            return
        if self._shutting_down or not self._player_ready:
            return
        if not self._mpv.pause:
            self._mpv.pause = True
        else:
            self._mpv.pause = False

    def _on_stop_clicked(self):
        """Handle stop button click."""
        if self._sequence_mode:
            self.stop_requested.emit()
            return
        self.stop()

    def set_sequence_mode(self, enabled: bool):
        """Enable sequence mode — play/stop emit signals instead of controlling MPV."""
        self._sequence_mode = enabled

    def set_playing(self, playing: bool):
        """Update play/pause icon to reflect sequence playback state."""
        if playing:
            self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

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
        if self._shutting_down or not self._player_ready or not self._media_loaded:
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
        # In sequence mode, icon is controlled by set_playing() from the sequence engine
        if not self._sequence_mode:
            if paused:
                self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            else:
                self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        self.playback_state_changed.emit(not paused)

    @Slot()
    def _on_file_loaded(self):
        """Handle file loaded event — media is ready to play."""
        self._media_loaded = True
        if self._pending_clip_range is not None:
            start_seconds, end_seconds = self._pending_clip_range
            self._apply_clip_range(start_seconds, end_seconds)
        if self._pending_seek_seconds is not None:
            self._safe_mpv_command(
                self._mpv.seek,
                self._pending_seek_seconds,
                'absolute',
                'exact',
            )
            self._pending_seek_seconds = None
        if self._pending_play_on_load:
            self._mpv.pause = False
            self._pending_play_on_load = False
        self.media_loaded.emit()

    @Slot()
    def _on_eof(self):
        """Handle end-of-file reached."""
        # In clip mode with looping, ab-loop handles restart automatically.
        # Otherwise pause at the end.
        if self._clip_start_ms is None or not self._loop_clip:
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
