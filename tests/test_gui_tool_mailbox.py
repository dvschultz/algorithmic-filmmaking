"""Replies must match exactly one live GUI request, even across reused LLM IDs."""

from concurrent.futures import ThreadPoolExecutor

from ui.workers.gui_tool_mailbox import GuiToolMailbox


def test_late_wrong_and_duplicate_replies_are_rejected():
    mailbox = GuiToolMailbox()
    first = mailbox.begin("edit")
    assert mailbox.wait(0) is None
    second = mailbox.begin("edit")
    assert first != second
    assert not mailbox.submit({"tool_call_id": first, "name": "edit"})
    assert not mailbox.submit({"tool_call_id": second, "name": "other"})
    reply = {"tool_call_id": second, "name": "edit", "result": {"count": 1}}
    assert mailbox.submit(reply)
    reply["result"]["count"] = 9
    assert not mailbox.submit(reply)
    assert mailbox.wait(0)["result"] == {"count": 1}
    assert not mailbox.submit(reply)


def test_cancellation_wakes_waiter_and_prevents_new_requests():
    mailbox = GuiToolMailbox()
    token = mailbox.begin("edit")
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(mailbox.wait, 10)
        mailbox.cancel()
        assert result.result(timeout=1) is None
    assert not mailbox.submit({"tool_call_id": token, "name": "edit"})
    assert mailbox.begin("next") is None


def test_result_crosses_threads_without_lost_wakeup():
    mailbox = GuiToolMailbox()
    token = mailbox.begin("edit")
    reply = {"tool_call_id": token, "name": "edit", "success": True}
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(mailbox.wait, 10)
        assert mailbox.submit(reply)
        assert result.result(timeout=1) == reply


def test_reply_from_another_worker_and_cancelled_accepted_reply():
    old, current = GuiToolMailbox(), GuiToolMailbox()
    old_token = old.begin("edit")
    current_token = current.begin("edit")
    assert not current.submit({"tool_call_id": old_token, "name": "edit"})
    assert current.submit({"tool_call_id": current_token, "name": "edit"})
    current.cancel()
    assert current.wait(0) is None
    assert old.wait(0) is None
