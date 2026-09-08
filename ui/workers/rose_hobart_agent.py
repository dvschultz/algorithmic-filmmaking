"""Agent-owned Rose Hobart progress dialog and asynchronous completion."""

from copy import deepcopy
from pathlib import Path
from typing import Any

from PySide6.QtCore import QTimer, Slot

from core.spine._agent_formatting import add_sequence_summary_for_agent
from ui.dialogs.rose_hobart_dialog import RoseHobartDialog


class AgentRoseHobart(RoseHobartDialog):
    def __init__(self, window: Any, request: dict) -> None:
        self._window = window
        self.reply = window._dispatch_gui_reply
        self.sequence_tab = window.sequence_tab
        self.request = request
        self._replied = False
        project = window.project
        self._sequence = project.sequence
        self._sequence_value = (
            deepcopy(self._sequence.to_dict()) if self._sequence else None
        )
        clips = [project.clips_by_id[cid] for cid in request["clip_ids"]]
        super().__init__(clips, project.sources_by_id, parent=window, project=project)
        self.sensitivity_combo.setCurrentText(request["sensitivity"].title())
        ordering = {
            "original": "Original Order",
            "duration": "By Duration",
            "color": "By Color",
            "brightness": "By Brightness",
            "confidence": "By Confidence",
            "random": "Random",
        }
        self.ordering_combo.setCurrentText(ordering[request["ordering"]])
        self.sample_spin.setValue(request["sampling_interval"])
        self._reply_timer = QTimer(self)
        self._reply_timer.setInterval(100)
        self._reply_timer.timeout.connect(self._check_request)
        self.finished.connect(self._retire_agent)

    def start(self) -> bool:
        if not self.reply.is_current(self._window):
            self.reject()
            return False
        self.show()
        self._reply_timer.start()
        try:
            if self.start_matching(
                [Path(p) for p in self.request["reference_image_paths"]],
                sample_interval=self.request["sampling_interval"],
            ):
                return True
            self._on_error("Could not start Rose Hobart generation")
        except Exception as exc:
            self._on_error(str(exc))
        return False

    def _inputs_current(self) -> bool:
        return (
            super()._inputs_current()
            and self.project is not None
            and self._window.project is self.project
            and self._window.sequence_tab is self.sequence_tab
            and self.reply.is_current(self._window)
            and self.project.sequence is self._sequence
            and (
                self._sequence is None
                or self._sequence.to_dict() == self._sequence_value
            )
        )

    @Slot()
    def _check_request(self) -> None:
        if (
            not self.reply.is_current(self._window)
            or self._window.project is not self.project
        ):
            self.reject()

    def _send(self, result: dict) -> None:
        if not self._replied:
            self._replied = True
            self.reply.send(self._window, result)

    @Slot(list)
    def _on_finished(self, sequence: list) -> None:
        if not self._inputs_current():
            self._on_error("Project or sequence changed during face matching")
            return
        if sequence and not self.sequence_tab._apply_dialog_sequence(
            sequence, "rose_hobart", "Rose Hobart"
        ):
            self._on_error("Could not commit the generated Rose Hobart sequence")
            return
        result = {
            "success": True,
            "matched_count": len(sequence),
            "total_clips": len(self.clips),
            "sensitivity": self.request["sensitivity"],
            "ordering": self.request["ordering"],
        }
        if not sequence:
            result["message"] = (
                "No clips matched the reference person. Try 'loose' sensitivity."
            )
        self._send(add_sequence_summary_for_agent(self.project, result, sequence))
        self.accept()

    @Slot(str)
    def _on_error(self, message: str) -> None:
        self._send({"success": False, "error": message})
        self.reject()

    def reject(self) -> None:
        self._send({"success": False, "error": "Rose Hobart generation cancelled"})
        super().reject()

    @Slot(int)
    def _retire_agent(self, result: int) -> None:
        self._reply_timer.stop()
        if getattr(self._window, "_rose_hobart_dialog", None) is self:
            self._window._rose_hobart_dialog = None
        self.setParent(None)
        self.deleteLater()
