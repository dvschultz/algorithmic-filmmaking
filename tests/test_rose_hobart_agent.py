"""Agent Rose Hobart uses verified background work and owned GUI replies."""

from types import SimpleNamespace
from unittest.mock import Mock
import threading
import time

import pytest
from PySide6.QtWidgets import QWidget, QApplication

from core.chat_tools import generate_rose_hobart
from core.project import Project
from tests.test_face_records import setup as face_setup  # noqa: F401
from tests.test_rose_hobart_analysis import analyzed_project as rose_setup  # noqa: F401
from ui.dialogs.rose_hobart_dialog import RoseHobartWorker
from ui.main_window import MainWindow
from ui.workers.gui_tool_mailbox import GuiToolMailbox
from ui.workers.gui_tool_reply import GuiToolReply


@pytest.fixture
def setup(request, monkeypatch):
    project, provider, directory = request.getfixturevalue("rose_setup")
    monkeypatch.setattr(
        "core.feature_registry.check_feature_ready", lambda _: (True, [])
    )
    return project, provider, directory


def test_tool_only_validates_and_dispatches(setup, monkeypatch):
    project, provider, _ = setup
    blocked = Mock(side_effect=AssertionError("inference must not run during dispatch"))
    monkeypatch.setattr("core.analysis.faces.extract_faces_from_image", blocked)
    result = generate_rose_hobart(
        project,
        SimpleNamespace(sequence_tab=object()),
        reference_image_path=str(project.sources[0].file_path),
    )
    assert result["_wait_for_worker"] == "rose_hobart"
    assert result["clip_ids"] == [project.clips[0].id]
    blocked.assert_not_called()
    provider.assert_not_called()


@pytest.mark.parametrize(
    "options",
    [
        {"sampling_interval": True},
        {"sampling_interval": float("nan")},
        {"sampling_interval": 0},
        {"sensitivity": "unknown"},
        {"ordering": "unknown"},
        {"reference_image_paths": ["x"] * 4},
    ],
)
def test_tool_rejects_invalid_options(setup, options):
    project, _, _ = setup
    result = generate_rose_hobart(
        project,
        SimpleNamespace(sequence_tab=object()),
        reference_image_path=str(project.sources[0].file_path),
        **options,
    )
    assert result["success"] is False


@pytest.mark.parametrize(
    "mode", ["current", "cancel", "timeout", "project", "sequence"]
)
def test_agent_waits_for_native_exit_and_rejects_stale_requests(
    setup, monkeypatch, mode
):
    project, _, _ = setup
    window = QWidget()
    window.project = project
    window.sequence_tab = SimpleNamespace(
        _apply_dialog_sequence=Mock(return_value=True)
    )
    mailbox = GuiToolMailbox()
    window._chat_worker = SimpleNamespace(
        _stop_requested=False,
        is_gui_tool_pending=mailbox.is_pending,
        set_gui_tool_result=mailbox.submit,
    )
    name = "generate_rose_hobart"
    window._dispatch_gui_reply = GuiToolReply.capture(window, name, mailbox.begin(name))
    request = generate_rose_hobart(
        project, window, reference_image_paths=[str(project.sources[0].file_path)]
    )
    entered, release = threading.Event(), threading.Event()
    original = RoseHobartWorker.run

    def held(self):
        original(self)
        entered.set()
        assert release.wait(5)

    monkeypatch.setattr(RoseHobartWorker, "run", held)
    assert MainWindow._start_worker_for_tool(window, "rose_hobart", request)
    dialog = window._rose_hobart_dialog
    worker = dialog.worker
    try:
        assert entered.wait(5)
        QApplication.instance().processEvents()
        window.sequence_tab._apply_dialog_sequence.assert_not_called()
        assert project.clips[0].face_embeddings is None
        if mode == "cancel":
            dialog.reject()
            assert dialog.worker is worker
        elif mode == "timeout":
            assert mailbox.wait(0) is None
        elif mode == "project":
            window.project = Project.new()
        elif mode == "sequence":
            from models.sequence import Sequence

            project.sequence = Sequence()
    finally:
        release.set()
        assert worker.wait(5000)
    deadline = time.monotonic() + 5
    while (
        getattr(window, "_rose_hobart_dialog", None) is not None
        and time.monotonic() < deadline
    ):
        QApplication.instance().processEvents()
        time.sleep(0.002)
    assert window._rose_hobart_dialog is None
    result = mailbox.wait(0)
    if mode == "current":
        assert result["success"], result
        assert result["matched_count"] == 1
        pairs = window.sequence_tab._apply_dialog_sequence.call_args.args[0]
        assert pairs[0][0] is project.clips[0]
        assert project.clips[0].analysis_records["face_embeddings"].state == "succeeded"
    else:
        window.sequence_tab._apply_dialog_sequence.assert_not_called()
        assert project.clips[0].face_embeddings is None
        if mode in ("cancel", "sequence"):
            assert result["success"] is False
    window.close()


def test_tool_selects_requested_enabled_clips(setup):
    from models.clip import Clip

    project, _, _ = setup
    original = project.clips[0]
    other = Clip(source_id=original.source_id, start_frame=60, end_frame=90)
    disabled = Clip(
        source_id=original.source_id, start_frame=90, end_frame=120, disabled=True
    )
    project.add_clips([other, disabled])
    result = generate_rose_hobart(
        project,
        SimpleNamespace(sequence_tab=object()),
        reference_image_path=str(project.sources[0].file_path),
        clip_ids=[other.id, other.id, disabled.id],
    )
    assert result["clip_ids"] == [other.id]
    assert not generate_rose_hobart(
        project,
        SimpleNamespace(sequence_tab=object()),
        reference_image_path=str(project.sources[0].file_path),
        clip_ids=["missing"],
    )["success"]


def test_start_failure_replies_and_retires_dialog(setup, monkeypatch):
    project, _, _ = setup
    window = QWidget()
    window.project = project
    window.sequence_tab = SimpleNamespace()
    mailbox = GuiToolMailbox()
    window._chat_worker = SimpleNamespace(
        _stop_requested=False,
        is_gui_tool_pending=mailbox.is_pending,
        set_gui_tool_result=mailbox.submit,
    )
    name = "generate_rose_hobart"
    window._dispatch_gui_reply = GuiToolReply.capture(window, name, mailbox.begin(name))
    request = generate_rose_hobart(
        project,
        window,
        reference_image_path=str(project.sources[0].file_path),
        sampling_interval=0.333,
    )

    def failed(self):
        assert self.options.sample_interval == 0.333
        raise RuntimeError("start failed")

    monkeypatch.setattr(RoseHobartWorker, "start", failed)
    assert not MainWindow._start_worker_for_tool(window, "rose_hobart", request)
    assert window._rose_hobart_dialog is None
    assert mailbox.wait(0)["error"] == "start failed"
    window.close()
