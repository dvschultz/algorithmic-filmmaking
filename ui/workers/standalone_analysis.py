"""Standalone GUI analysis gates and legacy response formatting."""

from typing import Any

from core.operations.analysis_inputs import clip_input
from core.operations.clip_analysis import ClipAnalysisOptions


def start_standalone_analysis(
    window: Any,
    clip_ids: list[str],
    operation: str,
    options: ClipAnalysisOptions | None = None,
) -> bool:
    """Capture the request before dependency dialogs can reenter the GUI."""
    from ui.workers.clip_analysis import ClipAnalysisController, WORKER_ATTRIBUTES

    options = options or ClipAnalysisOptions()
    project = window.project
    clips = [
        project.clips_by_id[cid]
        for cid in dict.fromkeys(clip_ids)
        if cid in project.clips_by_id
    ]
    if not clips:
        return False
    session_id, path = project.session.session_id, project.path
    inputs = tuple(clip_input(project, clip) for clip in clips)
    reply = getattr(window, "_dispatch_gui_reply", None)
    previous = getattr(window, "_clip_analysis_controller", None)

    def valid() -> bool:
        return (
            window.project is project
            and project.session.session_id == session_id
            and project.path == path
            and (reply is None or reply.is_current(window))
            and getattr(window, "_clip_analysis_controller", None) is previous
            and all(project.clips_by_id.get(c.id) is c for c in clips)
            and tuple(clip_input(project, c) for c in clips) == inputs
        )

    def busy() -> bool:
        worker = getattr(window, WORKER_ATTRIBUTES[operation], None)
        return (worker is not None and worker.isRunning()) or any(
            not run.finished
            and not run.plan.cancelled
            and (not run.standalone or operation in run.plan.operations)
            for run in getattr(window, "_active_clip_analyses", ())
        )

    if not valid() or busy():
        return False
    gate_options = {"description_tier": options.tier} if operation == "describe" else {}
    if not window._ensure_analysis_operation_available(operation, **gate_options):
        return False
    if not valid() or busy():
        return False
    controller = ClipAnalysisController(
        window,
        clips,
        [operation],
        options=options,
        standalone=True,
    )
    controller.progress.connect(window._on_clip_analysis_progress)
    controller.status.connect(window._on_clip_analysis_status)
    for update, args in (
        (window.analyze_tab.add_clips, ([c.id for c in clips],)),
        (window._switch_to_tab, ("analyze",)),
        (
            window._gui_state.set_processing,
            ("analysis", f"{operation} on {len(clips)} clips"),
        ),
        (window.analyze_tab.set_analyzing, (True, operation)),
        (window.progress_bar.setVisible, (True,)),
        (window.progress_bar.setRange, (0, 100)),
    ):
        if not valid() or busy():
            controller.retire()
            return False
        update(*args)
    if not valid() or busy():
        controller.retire()
        return False
    controller.start()
    return True


def finish_standalone_analysis(window: Any, controller: Any, result: dict) -> None:
    """Reply to this request even when another operation owns the progress UI."""
    from PySide6.QtWidgets import QMessageBox

    if not controller.owns_project():
        return
    other_runs = any(
        run is not controller
        and not run.finished
        and not run.plan.cancelled
        and run.owns_project()
        for run in getattr(window, "_active_clip_analyses", ())
    )
    if controller.owns_view() and not other_runs:
        window._gui_state.clear_processing("analysis")
        if not controller.owns_view():
            return
        window.analyze_tab.set_analyzing(False)
        if not controller.owns_view():
            return
        window.progress_bar.setVisible(False)
    if not controller.owns_project():
        return
    accepted = [controller.clips[cid] for cid in result["succeeded"]]
    total = len(controller.plan.clip_ids)
    failed = len(result["failed"])
    operation = controller.plan.operations[0]
    message = (
        "Analysis cancelled"
        if result["cancelled"]
        else f"{operation}: {len(accepted)} of {total} clips completed"
    )
    if controller.owns_view():
        window.status_bar.showMessage(message)
    for source_id in result["analyzed_sources"]:
        if not controller.owns_project():
            return
        window.collect_tab.update_source_has_analysis(source_id, True)
    if not controller.owns_project():
        return
    window._update_chat_project_state()
    extra = {
        **result,
        "success": not failed and not result["cancelled"],
        "clip_count": total,
        "clip_ids": list(controller.plan.clip_ids),
        "total_clips": total,
    }
    if operation == "shots":
        counts: dict[str, int] = {}
        for clip in accepted:
            label = clip.shot_type or "unknown"
            counts[label] = counts.get(label, 0) + 1
        extra["shot_type_summary"] = counts
    elif operation == "transcribe":
        extra["transcribed_count"] = sum(bool(c.transcript) for c in accepted)
    elif operation == "classify":
        extra.update(
            classified_clips=sum(bool(c.object_labels) for c in accepted),
            sample_labels=[
                {"clip_id": c.id, "labels": c.object_labels[:5]}
                for c in accepted[:3]
                if c.object_labels
            ],
        )
    elif operation == "describe":
        extra.update(
            described_clips=sum(bool(c.description) for c in accepted),
            error_count=failed,
            sample_descriptions=[
                {"clip_id": c.id, "description": c.description}
                for c in accepted[:3]
                if c.description
            ],
        )
        # Preserve the standalone description API's partial-success contract.
        extra["success"] = not result["cancelled"] and (not failed or bool(accepted))
        if result["errors"]:
            extra["last_error"] = result["errors"][-1]
    elif operation == "detect_objects":
        extra.update(
            analyzed_clips=len(accepted),
            total_people_detected=sum(c.person_count or 0 for c in accepted),
        )
        if controller.options.detect_all:
            labels: dict[str, int] = {}
            for clip in accepted:
                for detection in clip.detected_objects or ():
                    label = detection["label"]
                    labels[label] = labels.get(label, 0) + 1
            extra["object_counts"] = labels
    if controller.reply is not None and controller.owns_project():
        payload = window._build_agent_analysis_result(
            accepted, [operation], message, extra
        )
        controller.reply.send(
            window,
            {
                "success": extra["success"] if operation == "describe" else True,
                "result": payload,
            },
        )
    elif failed and not result["cancelled"] and controller.owns_view():
        details = "\n".join(dict.fromkeys(result["errors"])) or message
        QMessageBox.warning(window, "Analysis Incomplete", details)
