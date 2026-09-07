"""Read-only controls preserve inspection and resist selection-driven enabling."""

from typing import Any
from PySide6.QtCore import QObject, QEvent
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QWidget


def is_read_only(widget: Any) -> bool:
    """Find the owning project, including graphics items and floating docks."""
    if not isinstance(widget, QObject) and hasattr(widget, "scene"):
        scene = widget.scene()
        widget = scene.views()[0] if scene and scene.views() else None
    while widget is not None:
        project = getattr(widget, "project", None)
        if getattr(project, "is_read_only", False) is True:
            return True
        if widget.property("projectReadOnly") is True:
            return True
        widget = widget.parent()
    return False


class ReadOnlyControls(QObject):
    def __init__(self, parent: QObject):
        super().__init__(parent)
        self._read_only = False
        self._controls: dict[QWidget | QAction, bool] = {}

    def watch(self, control: QWidget | QAction) -> None:
        if control in self._controls:
            return
        self._controls[control] = control.isEnabled()
        if isinstance(control, QAction):
            control.changed.connect(lambda: self._enforce(control))
        else:
            control.installEventFilter(self)
        self._enforce(control)

    def _enforce(self, control: QWidget | QAction) -> None:
        if self._read_only and control.isEnabled():
            control.setEnabled(False)

    def eventFilter(self, watched, event):
        if event.type() == QEvent.EnabledChange:
            self._enforce(watched)
        return False

    def set_read_only(self, read_only: bool) -> None:
        if read_only == self._read_only:
            return
        if read_only:
            self._controls = {
                control: control.isEnabled() for control in self._controls
            }
        self._read_only = read_only
        for control, enabled in self._controls.items():
            if read_only:
                control.setEnabled(False)
            else:
                control.setEnabled(enabled)


MUTATING_CONTROLS = (
    "save_project_action",
    "save_project_as_action",
    "export_bundle_action",
    "import_project_action",
    "collect_tab.audio_btn",
    "collect_tab.url_btn",
    "collect_tab.analyze_btn",
    "collect_tab.source_browser.add_card",
    "cut_tab.cut_standard_btn",
    "cut_tab.cut_fast_btn",
    "analyze_tab.analyze_btn",
    "analyze_tab.quick_run_btn",
    "analyze_tab.alignment_btn",
    "frames_tab.extract_btn",
    "frames_tab.import_btn",
    "frames_tab.analyze_btn",
    "frames_tab.add_to_seq_btn",
    "sequence_tab.new_seq_btn",
    "sequence_tab.card_grid",
    "sequence_tab.algorithm_dropdown",
    "sequence_tab._confirm_generate_btn",
    "sequence_tab.timeline.clear_btn",
    "sequence_tab.timeline.add_track_btn",
    "clip_details_sidebar.name_edit",
    "clip_details_sidebar.shot_type_dropdown",
    "clip_details_sidebar.transcript_edit",
    "clip_details_sidebar.object_labels_edit",
    "clip_details_sidebar.description_edit",
)


def refresh_project_access(window: Any) -> None:
    guard = getattr(window, "_read_only_controls", None)
    if guard is None:
        guard = window._read_only_controls = ReadOnlyControls(window)
    for path in MUTATING_CONTROLS:
        control = window
        for part in path.split("."):
            control = getattr(control, part, None)
        if isinstance(control, (QWidget, QAction)):
            guard.watch(control)
    for action in getattr(window, "_project_import_actions", ()):
        guard.watch(action)
    guard.set_read_only(window.project.is_read_only)
