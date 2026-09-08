"""Cancel native GUI work by captured request identity, including expired tokens."""

from typing import Any

from ui.workers.gui_tool_reply import GuiToolReply


def cancel_gui_tool_work(
    window: Any, *, name: str | None = None, token: str | None = None,
) -> None:
    """Cancel matching work, or all work owned by the retiring conversation."""
    if window._chat_worker is None:
        return
    for controller in tuple(getattr(window, "_active_clip_analyses", ())):
        reply = controller.reply
        if (
            isinstance(reply, GuiToolReply)
            and reply.worker is window._chat_worker
            and reply.session_id == window.project.session.session_id
            and (name is None or reply.name == name)
            and (token is None or reply.token == token)
        ):
            controller.cancel()
    workers = list(getattr(window, "_active_download_workers", ()))
    workers.extend(getattr(window, "_active_audio_imports", ()))
    image_import = getattr(window, "_image_import_worker", None)
    if image_import is not None:
        workers.append(image_import)
    detection = getattr(window, "detection_worker", None)
    if detection is not None:
        workers.append(detection)
    for worker in workers:
        reply = getattr(worker, "gui_tool_reply", None)
        if (
            not isinstance(reply, GuiToolReply)
            or reply.worker is not window._chat_worker
            or reply.session_id != window.project.session.session_id
            or (name is not None and reply.name != name)
            or (token is not None and reply.token != token)
        ):
            continue
        # Pending admission is already closed on timeout. Match the captured
        # identity instead of is_current(), which intentionally rejects expiry.
        if worker.isRunning():
            worker.cancel()
