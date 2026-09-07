"""Saved-project analysis plans with durable transcription steps."""

import json
from pathlib import Path
from threading import Event

from core.jobs.commits import StaleJobResult
from core.jobs.spec import OperationSpec
from core.jobs.store import JobStore
from core.jobs.transcription import run_transcription_job, transcription_job_spec
from core.operations.analysis_plan import Progress, run_analysis_plan
from core.operations.transcription import TranscriptionOptions
from core.project import Project
from core.project_revision import ProjectRevisionConflict
from core.spine.project_io import load_with_mtime, project_writer, save_with_mtime_check


def analysis_job_spec(project: Project, *, arguments: dict) -> OperationSpec:
    inputs = {}
    if "describe" in (arguments.get("operations") or []):
        from core.jobs.description import description_job_spec
        from core.operations.description import resolve_options

        description = description_job_spec(
            project, arguments.get("clip_ids"), resolve_options(), arguments={},
        )
        inputs["description"] = json.loads(description.inputs_json)
    if "transcribe" in (arguments.get("operations") or []):
        transcription = transcription_job_spec(
            project,
            arguments.get("clip_ids"),
            TranscriptionOptions(model="base", language=None),
            arguments={},
        )
        inputs["transcription"] = json.loads(transcription.inputs_json)
    revision = project.session.file_revision
    return OperationSpec.build(
        kind="analyze_clips",
        version=1,
        arguments=arguments,
        inputs=inputs,
        persistence="job_history",
        cancellable=True,
        session_id=project.session.session_id,
        input_revision=revision.digest if revision else None,
    )


def run_analysis_job(
    store: JobStore,
    path: Path,
    operation: OperationSpec,
    progress: Progress,
    cancel: Event,
) -> dict:
    from core.spine.analyze import ANALYZE_CLIP_OPERATION_MAP

    arguments = operation.arguments
    ids = arguments.get("clip_ids")
    with project_writer(path):
        project, _ = load_with_mtime(path)
        revision = project.session.file_revision
        if operation.input_revision is not None and (
            revision is None or revision.digest != operation.input_revision
        ):
            raise ProjectRevisionConflict(path)
        # Resolve the submitted backend once; changes in runtime availability
        # must not silently retarget an already queued transcription request.
        captured = json.loads(operation.inputs_json)
        transcription = captured.get("transcription")
        description = captured.get("description")
        options = (
            TranscriptionOptions(**transcription["options"]) if transcription else None
        )
        if options is not None:
            live = transcription_job_spec(project, ids, options, arguments={})
            if json.loads(live.inputs_json) != transcription:
                raise StaleJobResult("Analysis inputs changed while the job was queued")

        def execute(op: str, report: Progress | None) -> dict:
            if op == "describe":
                from core.jobs.description import description_job_spec, run_description_job
                from core.operations.description import DescriptionOptions

                if description is None:
                    raise StaleJobResult("Analysis job has no captured description options")
                current, _ = load_with_mtime(path)
                step = description_job_spec(
                    current, ids, DescriptionOptions(**description["options"]), arguments={},
                )
                if json.loads(step.inputs_json) != description:
                    raise StaleJobResult("Description inputs changed before analysis")
                return run_description_job(
                    store, path, ids, report or (lambda *_: None), cancel, operation=step,
                )
            if op == "transcribe":
                assert options is not None
                current, _ = load_with_mtime(path)
                step = transcription_job_spec(current, ids, options, arguments={})
                if json.loads(step.inputs_json) != transcription:
                    raise StaleJobResult("Analysis inputs changed before transcription")
                return run_transcription_job(
                    store,
                    path,
                    ids,
                    options,
                    report or (lambda *_: None),
                    cancel,
                    operation=step,
                    skip_existing=True,
                )
            # Reload after each durable step; never save an older model over
            # its result receipts. Unmigrated steps retain their spine provider.
            current, mtime = load_with_mtime(path)
            kwargs: dict[str, object] = {"skip_existing": True}
            if op == "custom_query":
                kwargs = {"skip_existing": False, "query": arguments.get("query")}
            result = ANALYZE_CLIP_OPERATION_MAP[op](
                current,
                ids,
                progress_callback=report,
                cancel_event=cancel,
                **kwargs,
            )
            save_with_mtime_check(current, path, mtime)
            return result

        return run_analysis_plan(
            arguments.get("operations"), execute, progress=progress, cancel=cancel
        )
