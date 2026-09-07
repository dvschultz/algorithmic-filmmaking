"""Cancellation belongs to the native worker's captured request."""

from types import SimpleNamespace
from unittest.mock import Mock

from core.project import Project
from ui.workers.gui_tool_reply import GuiToolReply


def test_timeout_cancels_only_matching_request_and_chat_retirement_cancels_owned_work():
    from ui.workers.gui_tool_cancellation import cancel_gui_tool_work

    chat = object()
    window = SimpleNamespace(project=Project.new(), _chat_worker=chat)
    old = GuiToolReply.capture(window, "download_video", "old")
    new = GuiToolReply.capture(window, "download_video", "new")
    old_worker = Mock(gui_tool_reply=old)
    new_worker = Mock(gui_tool_reply=new)
    manual = Mock(gui_tool_reply=None)
    window._active_download_workers = {old_worker, new_worker, manual}
    window.detection_worker = None
    cancel_gui_tool_work(window, name=old.name, token=old.token)
    old_worker.cancel.assert_called_once()
    new_worker.cancel.assert_not_called()
    manual.cancel.assert_not_called()
    cancel_gui_tool_work(window)
    new_worker.cancel.assert_called_once()
    manual.cancel.assert_not_called()
    window._chat_worker = object()
    cancel_gui_tool_work(window)
    assert new_worker.cancel.call_count == 1
    window._chat_worker = chat
    window.project.clear()
    cancel_gui_tool_work(window)
    assert new_worker.cancel.call_count == 1


def test_detection_requires_matching_token_and_running_worker():
    from ui.workers.gui_tool_cancellation import cancel_gui_tool_work

    window = SimpleNamespace(project=Project.new(), _chat_worker=object())
    reply = GuiToolReply.capture(window, "detect_scenes_live", "request")
    worker = Mock(gui_tool_reply=reply)
    window.detection_worker = worker
    cancel_gui_tool_work(window, name=reply.name, token="old")
    worker.cancel.assert_not_called()
    worker.isRunning.return_value = False
    cancel_gui_tool_work(window, name=reply.name, token=reply.token)
    worker.cancel.assert_not_called()
    worker.isRunning.return_value = True
    cancel_gui_tool_work(window, name=reply.name, token=reply.token)
    worker.cancel.assert_called_once()
