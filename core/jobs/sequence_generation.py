"""Durable, cancellable sequence generation for headless job runtimes.

Runs a registry algorithm in a job: the algorithm (including any provider
calls) computes a recipe first, the recipe payload is recorded in the job
store, and only then is it published into the project as one edit. A retry
after a crash replays the recorded recipe instead of paying for inference
again; a cancelled run publishes nothing.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from threading import Event
from typing import Callable

from core.jobs.commits import ResultSpec, StaleJobResult, result_batch
from core.jobs.spec import OperationSpec
from core.jobs.store import JobStore
from core.project import Project
from models.recipe import SequenceRecipe

OPERATION_VERSION = 1


def _request_arguments(
    algorithm: str, clip_ids: list[str] | None, parameters: dict | None, seed: int | None, name: str | None,
    parent_recipe_id: str | None = None,
) -> dict:
    return {
        "algorithm": algorithm,
        "clip_ids": list(clip_ids) if clip_ids is not None else None,
        "parameters": dict(parameters or {}),
        "seed": seed,
        "name": name,
        "parent_recipe_id": parent_recipe_id,
    }


def _inputs(project: Project, arguments: dict) -> dict:
    """Identity of the clips the run will read, so a stale project is refused."""
    clip_ids = arguments["clip_ids"]
    clips = [project.clips_by_id[c] for c in clip_ids if c in project.clips_by_id] if clip_ids else [
        clip for clip in project.clips if not clip.disabled
    ]
    return {
        "project_id": project.metadata.id,
        "clips": [
            {"id": clip.id, "source_id": clip.source_id, "start": clip.start_frame, "end": clip.end_frame}
            for clip in clips
        ],
    }


def sequence_generation_job_spec(
    project: Project,
    algorithm: str,
    *,
    clip_ids: list[str] | None = None,
    parameters: dict | None = None,
    seed: int | None = None,
    name: str | None = None,
    parent_recipe_id: str | None = None,
) -> OperationSpec:
    from core.remix.registry import registry, resolve_seed
    from core.spine.sequences import prepare_generation_parameters

    definition = registry.require(algorithm)
    normalized = prepare_generation_parameters(project, algorithm, parameters)
    # Draw the seed now so a retried job reproduces the same run.
    resolved_seed = resolve_seed(definition, seed)
    arguments = _request_arguments(algorithm, clip_ids, normalized, resolved_seed, name, parent_recipe_id)
    revision = project.session.file_revision
    return OperationSpec.build(
        kind="generate_sequence",
        version=OPERATION_VERSION,
        arguments=arguments,
        inputs=_inputs(project, arguments),
        persistence="job_history",
        session_id=project.session.session_id,
        input_revision=revision.digest if revision else None,
    )


def sequence_generation_retry_spec(
    store: JobStore,
    path: Path,
    project: Project,
    algorithm: str,
    *,
    idempotency_key: str | None,
    clip_ids: list[str] | None = None,
    parameters: dict | None = None,
    seed: int | None = None,
    name: str | None = None,
    parent_recipe_id: str | None = None,
) -> OperationSpec:
    """Restore exact metadata for a keyed retry, or build an independent request."""
    restored = restored_sequence_generation_operation(store, path, project, idempotency_key)
    if restored is not None:
        return restored
    return sequence_generation_job_spec(
        project,
        algorithm,
        clip_ids=clip_ids,
        parameters=parameters,
        seed=seed,
        name=name,
        parent_recipe_id=parent_recipe_id,
    )


def restored_sequence_generation_operation(
    store: JobStore,
    path: Path,
    project: Project,
    idempotency_key: str | None,
) -> OperationSpec | None:
    """Return the exact failed operation named by an explicit retry key."""
    from core.jobs.store import TERMINAL_ERROR_STATUSES

    canonical_path = str(path.expanduser().resolve())
    if idempotency_key is not None:
        previous = store.find_by_idempotency(
            kind="generate_sequence",
            project_path=canonical_path,
            idempotency_key=idempotency_key,
        )
        if (
            previous is not None
            and previous.status in TERMINAL_ERROR_STATUSES
            and previous.operation_json is not None
        ):
            operation = OperationSpec.from_json(previous.operation_json)
            if operation.kind != "generate_sequence" or operation.version != OPERATION_VERSION:
                raise ValueError("Stored retry operation is not supported by this build")
            from core.spine.sequences import prepare_generation_parameters

            validated = prepare_generation_parameters(
                project,
                operation.arguments["algorithm"],
                operation.arguments["parameters"],
            )
            if validated != operation.arguments["parameters"]:
                raise ValueError("Stored retry parameters are not canonical")
            if _inputs(project, operation.arguments) != json.loads(operation.inputs_json):
                raise StaleJobResult("Sequence generation inputs changed before retry")
            return operation
    return None


def run_sequence_generation_job(
    store: JobStore,
    path: Path,
    operation: OperationSpec,
    progress: Callable[[float, str], None],
    cancel: Event,
) -> dict:
    """Compute the recipe under the job, then publish it as one saved edit."""
    from core.remix.registry import registry, run_algorithm
    from core.spine.sequences import _default_name, _sequence_result, publish_recipe

    arguments = operation.arguments
    definition = registry.require(arguments["algorithm"])
    with result_batch(store, path, max_items=1) as batch:
        project = batch.project
        expected_inputs = json.loads(operation.inputs_json)
        if _inputs(project, arguments) != expected_inputs:
            raise StaleJobResult("Sequence generation inputs changed while the job was queued")
        if cancel.is_set():
            raise StaleJobResult("Sequence generation was cancelled before it started")
        spec = ResultSpec.build(
            path, kind="generate_sequence", version=OPERATION_VERSION,
            target_id=operation.session_id or "project",
            arguments=arguments, inputs=expected_inputs,
        )
        def compute() -> dict:
            progress(0.0, f"Generating {arguments['algorithm']}...")
            candidates = _candidates(project, arguments)
            for pname in definition.source_parameters:
                source_id = arguments["parameters"].get(pname)
                source = project.sources_by_id.get(source_id)
                if source is None:
                    raise ValueError(f"Parameter {pname!r} must name a source in this project")
                present = {clip.id for clip, _ in candidates}
                candidates.extend(
                    (clip, source) for clip in project.clips_by_source.get(source_id, [])
                    if clip.id not in present and not clip.disabled
                )
            run = run_algorithm(
                definition, deepcopy(candidates), arguments["parameters"],
                seed=arguments["seed"], cancel_event=cancel,
                parent_recipe_id=arguments.get("parent_recipe_id"),
                progress=lambda message: progress(0.5, message),
            )
            if run is None:
                raise StaleJobResult("Sequence generation was cancelled")
            if not run.recipe.realized:
                raise ValueError("Algorithm produced an empty sequence: " + "; ".join(run.proposal.notes))
            return {
                "recipe": run.recipe.to_dict(),
                "notes": list(run.proposal.notes),
                "sequence_settings": dict(run.proposal.sequence_settings),
            }

        def validate_input(current: Project) -> bool:
            return not cancel.is_set() and _inputs(current, arguments) == expected_inputs

        def apply(current: Project, payload: dict) -> None:
            if cancel.is_set():
                raise StaleJobResult("Sequence generation was cancelled before publication")
            recipe = SequenceRecipe.from_dict(payload["recipe"])
            label = arguments["name"] or _default_name(definition.key)
            publish_recipe(
                current, recipe, name=label, sequence_settings=payload.get("sequence_settings") or {},
            )

        def is_applied(current: Project, payload: dict) -> bool:
            recipe_id = payload["recipe"]["id"]
            return any(
                s.readable_recipe is not None and s.readable_recipe.id == recipe_id for s in current.sequences
            )

        outcome = batch.commit(
            spec, compute=compute, validate_input=validate_input, apply=apply, is_applied=is_applied,
        )
        recipe_id = outcome["payload"]["recipe"]["id"]
        sequence = next(s for s in project.sequences if s.readable_recipe and s.readable_recipe.id == recipe_id)
        progress(1.0, "Sequence published")
        result = _sequence_result(sequence, project, tuple(outcome["payload"].get("notes", [])))
        result["result_id"] = outcome["result_id"]
        result["replayed"] = not outcome["applied"]
        return result


def run_sequence_generation_runtime_job(
    store: JobStore,
    path: Path,
    operation: OperationSpec,
    progress: Callable[[float, str], None],
    cancel: Event,
) -> dict:
    """Adapt publication cancellation to the JobRuntime terminal contract."""
    try:
        return run_sequence_generation_job(store, path, operation, progress, cancel)
    except StaleJobResult:
        if cancel.is_set():
            return {"success": False, "error": "Sequence generation was cancelled", "cancelled": True}
        raise


def _candidates(project: Project, arguments: dict) -> list:
    clip_ids = arguments["clip_ids"]
    if clip_ids:
        missing = [c for c in clip_ids if c not in project.clips_by_id]
        if missing:
            raise ValueError(f"Unknown clip IDs: {', '.join(missing)}")
        clips = [project.clips_by_id[c] for c in clip_ids]
    else:
        clips = [clip for clip in project.clips if not clip.disabled]
    if not clips:
        raise ValueError("No clips available. Detect scenes first.")
    return [(clip, project.sources_by_id[clip.source_id]) for clip in clips if clip.source_id in project.sources_by_id]
