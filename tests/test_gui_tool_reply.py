"""Operation replies retain their requester rather than reading current pending IDs."""

from types import SimpleNamespace
from unittest.mock import Mock

from core.project import Project
from ui.workers.gui_tool_reply import GuiToolReply, gui_reply_scope
from ui.workers.gui_tool_mailbox import GuiToolMailbox


def test_reply_preserves_owner_and_identity():
    worker = Mock(_stop_requested=False)
    window = SimpleNamespace(project=Project.new(), _chat_worker=worker)
    reply = GuiToolReply.capture(window, "describe", "original")
    window._pending_agent_tool_call_id = "unrelated"
    assert reply.send(window, {"success": True, "result": {"count": 2}})
    assert worker.set_gui_tool_result.call_args.args[0] == {
        "tool_call_id": "original",
        "name": "describe",
        "success": True,
        "result": {"count": 2},
    }
    window._chat_worker = Mock()
    assert not reply.send(window, {"success": True})
    window._chat_worker.set_gui_tool_result.assert_not_called()
    window._chat_worker = worker
    window.project.clear()
    assert not reply.send(window, {"success": True})


def test_scope_restores_after_nested_dispatch_and_failure():
    window = SimpleNamespace(
        project=Project.new(), _chat_worker=Mock(_stop_requested=False)
    )
    outer = GuiToolReply.capture(window, "first", "one")
    inner = GuiToolReply.capture(window, "second", "two")
    with gui_reply_scope(window, outer):
        try:
            with gui_reply_scope(window, inner):
                assert window._dispatch_gui_reply is inner
                raise RuntimeError("failed dispatch")
        except RuntimeError:
            pass
        assert window._dispatch_gui_reply is outer
    assert window._dispatch_gui_reply is None
    window._chat_worker._stop_requested = True
    assert not outer.send(window, {"success": True})


def test_old_operation_cannot_complete_new_request_on_same_chat():
    mailbox = GuiToolMailbox()
    worker = SimpleNamespace(_stop_requested=False, set_gui_tool_result=mailbox.submit)
    window = SimpleNamespace(project=Project.new(), _chat_worker=worker)
    old = GuiToolReply.capture(window, "describe", mailbox.begin("describe"))
    assert mailbox.wait(0) is None
    current = GuiToolReply.capture(window, "describe", mailbox.begin("describe"))
    window._pending_agent_tool_call_id = current.token
    assert not old.send(window, {"success": True, "result": {"old": True}})
    assert current.send(window, {"success": True, "result": {"new": True}})
    assert mailbox.wait(0)["result"] == {"new": True}
