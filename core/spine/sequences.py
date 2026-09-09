"""Sequence management shared by desktop agent and retained headless sessions."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, Any, Mapping

from models.recipe import SequenceRecipe
from models.sequence import Sequence
from core.spine.security import validate_path

if TYPE_CHECKING:
    from threading import Event

    from core.project import Project
    from models.clip import Clip, Source


@dataclass
class SequenceDraft:
    """Detached generation output committed once, without replaying computation."""

    project: Project
    session_id: str
    sequence: Sequence
    origin: Sequence
    origin_state: dict[str, Any]
    replace: bool
    reuse: bool
    committed: bool = False

    @classmethod
    def prepare(
        cls,
        project: Project,
        algorithm: str,
        name: str,
        *,
        replace_sequence_id: str | None = None,
        show_chromatic_color_bar: bool = False,
    ) -> SequenceDraft:
        project.session.assert_owner()
        origin = (
            next((s for s in project.sequences if s.id == replace_sequence_id), None)
            if replace_sequence_id
            else project.sequence
        )
        if origin is None:
            raise ValueError("Generation target no longer exists")
        reuse = not origin.get_all_clips()
        sequence = deepcopy(origin) if reuse else Sequence()
        sequence.name = name
        sequence.algorithm = algorithm
        sequence.recipe = None  # Provenance belongs to the new generation, never the reused shell.
        sequence.show_chromatic_color_bar = (
            show_chromatic_color_bar and algorithm == "color"
        )
        return cls(
            project,
            project.session.session_id,
            sequence,
            origin,
            deepcopy(origin.to_dict()),
            replace_sequence_id is not None,
            reuse,
        )

    def validate_session(self, project: Project) -> None:
        project.session.assert_owner()
        if project is not self.project or project.session.session_id != self.session_id:
            raise ValueError("Generation belongs to a previous project session")
        if self.committed:
            raise ValueError("Generation was already committed")

    def commit(self, project: Project) -> Sequence:
        from core.commands.sequences import EditSequences

        self.validate_session(project)
        if self.reuse or self.replace:
            if (
                not any(s is self.origin for s in project.sequences)
                or self.origin.to_dict() != self.origin_state
            ):
                raise ValueError(
                    "Generation target changed while the result was being prepared"
                )
        if not isfinite(self.sequence.fps) or self.sequence.fps <= 0:
            raise ValueError("Generated sequence FPS must be positive and finite")
        if not self.sequence.tracks:
            raise ValueError("Generated sequence requires a track")
        clip_ids = set()
        for index, track in enumerate(self.sequence.tracks):
            for clip in track.clips:
                if clip.id in clip_ids:
                    raise ValueError("Generated sequence has duplicate clip IDs")
                clip_ids.add(clip.id)
                if (
                    clip.track_index != index
                    or clip.start_frame < 0
                    or clip.in_point < 0
                    or clip.duration_frames <= 0
                ):
                    raise ValueError("Generated sequence has an invalid clip placement")
        project.session.execute(
            EditSequences.generated(
                project,
                self.sequence,
                replace=self.origin if self.reuse or self.replace else None,
                reuse=self.reuse,
            )
        )
        self.committed = True
        return self.sequence


def unique_sequence_name(project: Project, name: str) -> str:
    names = {sequence.name for sequence in project.sequences}
    label = name
    suffix = 2
    while label in names:
        label = f"{name} #{suffix}"
        suffix += 1
    return label


# --- Registry-backed generation -------------------------------------------

SEQUENCE_SETTING_FIELDS = frozenset({"music_path", "reference_source_id", "dimension_weights", "allow_repeats"})


def recipe_input_problems(project: Project, recipe: SequenceRecipe) -> list[str]:
    """Explain why a recipe's realized clips cannot be replayed in this project."""
    problems: list[str] = []
    by_clip = {item.clip_id: item for item in recipe.inputs}
    for entry in recipe.realized:
        snapshot = by_clip[entry.clip_id]
        clip = project.clips_by_id.get(entry.clip_id)
        source = project.sources_by_id.get(entry.source_id)
        if clip is None:
            problems.append(f"clip {entry.clip_id} is no longer in the project")
        elif source is None or clip.source_id != source.id:
            problems.append(f"clip {entry.clip_id} no longer belongs to source {entry.source_id}")
        elif (clip.start_frame, clip.end_frame) != (snapshot.start_frame, snapshot.end_frame):
            problems.append(
                f"clip {entry.clip_id} was re-cut ({snapshot.start_frame}-{snapshot.end_frame}"
                f" became {clip.start_frame}-{clip.end_frame})"
            )
        elif float(source.fps) != snapshot.source_fps:
            problems.append(f"source {source.id} changed frame rate")
        elif entry.out_offset > clip.duration_frames:
            problems.append(f"clip {entry.clip_id} is shorter than the recipe expects")
    return sorted(set(problems), key=problems.index)


def publish_recipe(
    project: Project,
    recipe: SequenceRecipe,
    *,
    name: str,
    replace_sequence_id: str | None = None,
    show_chromatic_color_bar: bool = False,
    fps: float | None = None,
    sequence_settings: Mapping[str, Any] | None = None,
) -> Sequence:
    """Build a timeline from realized recipe entries and publish it as one edit.

    Performs no algorithm or provider work. Raises ``ValueError`` with every
    input problem when the project no longer matches the recipe. ``fps`` sets
    the timeline rate; it defaults to the first realized source's rate, which
    is what desktop generation uses.
    """
    from fractions import Fraction
    from core.sequence_time import video_entry

    if not recipe.realized:
        raise ValueError("Recipe realized no clips")
    problems = recipe_input_problems(project, recipe)
    if problems:
        raise ValueError("Recipe inputs changed: " + "; ".join(problems))
    if fps is None:
        fps = project.sources_by_id[recipe.realized[0].source_id].fps
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not isfinite(fps) or fps <= 0:
        raise ValueError("Sequence FPS must be positive and finite")
    draft = SequenceDraft.prepare(
        project, recipe.algorithm, unique_sequence_name(project, name),
        replace_sequence_id=replace_sequence_id,
        show_chromatic_color_bar=show_chromatic_color_bar,
    )
    draft.sequence.fps = float(fps)
    for field_name, value in (sequence_settings or {}).items():
        if field_name not in SEQUENCE_SETTING_FIELDS:
            raise ValueError(f"Algorithm set unknown sequence field {field_name!r}")
        setattr(draft.sequence, field_name, deepcopy(value))
    position = Fraction(0)
    for realized in recipe.realized:
        clip = project.clips_by_id[realized.clip_id]
        source = project.sources_by_id[realized.source_id]
        entry = video_entry(
            clip, source, timeline_fps=draft.sequence.fps,
            start=position, relative_range=realized.relative_range,
        )
        entry.hflip, entry.vflip, entry.reverse = realized.hflip, realized.vflip, realized.reverse
        entry.rationale = realized.rationale
        draft.sequence.tracks[0].add_clip(entry)
        position = entry.timeline_range.end
    draft.sequence.recipe = recipe
    return draft.commit(project)


def list_algorithms() -> dict:
    """Registry schemas for agent surfaces; no UI imports."""
    from core.remix.registry import registry

    return {"success": True, "algorithms": registry.describe()}


def _generation_candidates(
    project: Project, clip_ids: list[str] | None,
) -> tuple[list[tuple[Clip, Source]], str | None]:
    if clip_ids is not None:
        if not isinstance(clip_ids, list) or not clip_ids or not all(isinstance(c, str) for c in clip_ids):
            return [], "Provide a nonempty list of clip IDs"
        if len(set(clip_ids)) != len(clip_ids):
            return [], "Clip IDs must not repeat"
        missing = [c for c in clip_ids if c not in project.clips_by_id]
        if missing:
            return [], f"Unknown clip IDs: {', '.join(missing)}"
        clips = [project.clips_by_id[c] for c in clip_ids]
    else:
        clips = [clip for clip in project.clips if not clip.disabled]
    candidates = []
    for clip in clips:
        source = project.sources_by_id.get(clip.source_id)
        if source is None:
            return [], f"Clip {clip.id} has no source in this project"
        candidates.append((clip, source))
    if not candidates:
        return [], "No clips available. Detect scenes first."
    return candidates, None


def _sequence_result(sequence: Sequence, project: Project, notes: tuple[str, ...] = ()) -> dict:
    recipe = sequence.readable_recipe
    return {
        "success": True,
        "sequence_id": sequence.id,
        "name": sequence.name,
        "algorithm": sequence.algorithm,
        "algorithm_version": recipe.algorithm_version if recipe else None,
        "recipe_id": recipe.id if recipe else None,
        "seed": recipe.seed if recipe else None,
        "parameters": dict(recipe.parameters) if recipe else {},
        "clip_count": len(sequence.get_all_clips()),
        "clip_ids": [entry.source_clip_id for entry in sequence.get_all_clips()],
        "duration_seconds": sequence.duration_seconds,
        "active": project.sequence is sequence,
        "notes": list(notes),
    }


def generate_sequence(
    project: Project,
    algorithm: str,
    *,
    clip_ids: list[str] | None = None,
    parameters: dict[str, Any] | None = None,
    seed: int | None = None,
    name: str | None = None,
    replace_sequence_id: str | None = None,
    parent_recipe_id: str | None = None,
    show_chromatic_color_bar: bool = False,
    cancel_event: Event | None = None,
) -> dict:
    """Run a registry algorithm on project clips and publish its recipe.

    Returns a result dict; never raises for caller mistakes. Inputs are
    snapshotted so generation cannot mutate library clips.
    """
    from copy import deepcopy

    from core.remix.registry import registry, run_algorithm

    definition = registry.get(algorithm) if isinstance(algorithm, str) else None
    if definition is None:
        return {
            "success": False,
            "error": f"Algorithm {algorithm!r} is not available through the registry. "
                     f"Registry algorithms: {', '.join(registry.keys())}",
        }
    candidates, error = _generation_candidates(project, clip_ids)
    if error:
        return {"success": False, "error": error}
    if parameters is not None and not isinstance(parameters, dict):
        return {"success": False, "error": "Parameters must be an object"}
    if seed is not None and (isinstance(seed, bool) or type(seed) is not int or seed < 0):
        return {"success": False, "error": "Seed must be a non-negative integer"}
    parameters = dict(parameters or {})
    for name in definition.asset_parameters:
        value = parameters.get(name)
        if value is None or value == "" or value == []:
            continue
        values = value if isinstance(value, list) else [value]
        resolved = []
        for item in values:
            if not isinstance(item, str):
                return {"success": False, "error": f"Parameter {name!r} must hold file paths"}
            ok, error, path = validate_path(item, must_exist=True, must_be_file=True)
            if not ok:
                return {"success": False, "error": f"Parameter {name!r}: {error}"}
            resolved.append(str(path))
        parameters[name] = resolved if isinstance(value, list) else resolved[0]
    for name in definition.source_parameters:
        source_id = (parameters or {}).get(name)
        if not isinstance(source_id, str) or source_id not in project.sources_by_id:
            return {"success": False, "error": f"Parameter {name!r} must name a source in this project"}
        present = {clip.id for clip, _ in candidates}
        source = project.sources_by_id[source_id]
        candidates.extend(
            (clip, source) for clip in project.clips_by_source.get(source_id, [])
            if clip.id not in present and not clip.disabled
        )
    snapshots = deepcopy(candidates)
    try:
        run = run_algorithm(
            definition, snapshots, parameters, seed=seed,
            cancel_event=cancel_event, parent_recipe_id=parent_recipe_id,
        )
    except (ValueError, RuntimeError, OSError) as exc:
        # Provider, loader and prerequisite failures surface as results, never tracebacks.
        return {"success": False, "error": str(exc)}
    if run is None:
        return {"success": False, "error": "Generation was cancelled", "cancelled": True}
    if not run.recipe.realized:
        return {
            "success": False,
            "error": "Algorithm produced an empty sequence: " + "; ".join(run.proposal.notes),
            "notes": list(run.proposal.notes),
        }
    label = name.strip() if isinstance(name, str) and name.strip() else _default_name(definition.key)
    try:
        sequence = publish_recipe(
            project, run.recipe, name=label,
            replace_sequence_id=replace_sequence_id,
            show_chromatic_color_bar=show_chromatic_color_bar,
            sequence_settings=run.proposal.sequence_settings,
        )
    except (ValueError, RuntimeError) as exc:
        return {"success": False, "error": str(exc)}
    return _sequence_result(sequence, project, run.proposal.notes)


def _default_name(algorithm: str) -> str:
    return algorithm.replace("_", " ").title()


def _find_sequence(project: Project, sequence_id: str | None) -> Sequence | None:
    if sequence_id is None:
        return project.sequence
    return next((s for s in project.sequences if s.id == sequence_id), None)


def recipe_matches_timeline(sequence: Sequence, recipe: SequenceRecipe) -> bool:
    """Whether track 0 still holds exactly the realized entries, in order."""
    placed = sequence.tracks[0].clips if sequence.tracks else []
    if len(placed) != len(recipe.realized) or len(sequence.tracks) > 1 and any(
        track.clips for track in sequence.tracks[1:]
    ):
        return False
    for entry, realized in zip(placed, recipe.realized):
        if entry.source_clip_id != realized.clip_id or entry.source_id != realized.source_id:
            return False
        if entry.hflip != realized.hflip or entry.vflip != realized.vflip or entry.reverse != realized.reverse:
            return False
        if entry.out_point - entry.in_point != realized.out_offset - realized.in_offset:
            return False
    return True


def get_sequence_recipe(project: Project, sequence_id: str | None = None) -> dict:
    """Inspect the stored recipe without touching the project."""
    sequence = _find_sequence(project, sequence_id)
    if sequence is None:
        return {"success": False, "error": "Sequence not found"}
    recipe = sequence.readable_recipe
    if recipe is None:
        if sequence.recipe is not None:
            return {
                "success": False,
                "error": "This sequence's recipe was written by a newer build and cannot be read here",
            }
        return {"success": False, "error": "This sequence has no recipe (manual or pre-recipe generation)"}
    problems = recipe_input_problems(project, recipe)
    return {
        "success": True,
        "sequence_id": sequence.id,
        "name": sequence.name,
        "recipe": recipe.to_dict(),
        "generation_fingerprint": recipe.generation_fingerprint,
        "uses_provider": recipe.uses_provider,
        "reconstructable": not problems,
        "problems": problems,
        "matches_timeline": recipe_matches_timeline(sequence, recipe),
    }


def reconstruct_sequence(
    project: Project, sequence_id: str | None = None, *, name: str | None = None,
) -> dict:
    """Rebuild a sequence from its realized recipe entries; no provider calls."""
    sequence = _find_sequence(project, sequence_id)
    if sequence is None:
        return {"success": False, "error": "Sequence not found"}
    recipe = sequence.readable_recipe
    if recipe is None:
        return get_sequence_recipe(project, sequence.id)
    label = name.strip() if isinstance(name, str) and name.strip() else f"{sequence.name} (reconstructed)"
    try:
        rebuilt = publish_recipe(
            project, recipe.derive(), name=label,
            show_chromatic_color_bar=sequence.show_chromatic_color_bar,
            fps=sequence.fps,
            sequence_settings={
                field: getattr(sequence, field) for field in SEQUENCE_SETTING_FIELDS
                if getattr(sequence, field) not in (None, False, {})
            },
        )
    except (ValueError, RuntimeError) as exc:
        return {"success": False, "error": str(exc)}
    return _sequence_result(rebuilt, project)


def create_sequence(
    project: Project, name: str | None = None, fps: float | None = None
) -> dict:
    name = (name or project.metadata.name or "Untitled Sequence").strip()
    fps = (
        fps
        if fps is not None
        else (project.sources[0].fps if project.sources else 30.0)
    )
    if (
        not name
        or isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not isfinite(fps)
        or fps <= 0
    ):
        return {
            "success": False,
            "error": "Provide a nonempty name and a finite positive fps",
        }
    sequence = Sequence(name=name, fps=fps)
    try:
        project.add_sequence(sequence, activate=True)
    except (ValueError, RuntimeError) as exc:
        return {"success": False, "error": str(exc)}
    return {
        "success": True,
        "message": f"Created sequence '{name}' at {fps} fps (now active)",
        "name": name,
        "fps": fps,
        "sequence_index": project.active_sequence_index,
        "sequence_id": sequence.id,
    }


def update_sequence(
    project: Project, sequence_id: str | None = None, **changes: Any
) -> dict:
    sequence = (
        project.sequence
        if sequence_id is None
        else next((s for s in project.sequences if s.id == sequence_id), None)
    )
    if sequence is None:
        return {"success": False, "error": "Sequence not found"}
    if "music_path" in changes and changes["music_path"] is not None:
        valid, error, path = validate_path(changes["music_path"], must_exist=True)
        if not valid:
            return {"success": False, "error": error}
        changes["music_path"] = str(path)
    try:
        project.update_sequence_metadata(sequence, **changes)
    except (ValueError, RuntimeError) as exc:
        return {"success": False, "error": str(exc)}
    return {
        "success": True,
        "message": f"Updated sequence: {', '.join(changes)}",
        "updated_fields": changes,
    }


def delete_sequence(project: Project, sequence_id: str) -> dict:
    index = next(
        (i for i, s in enumerate(project.sequences) if s.id == sequence_id), None
    )
    if index is None:
        return {"success": False, "error": "Sequence not found"}
    try:
        project.remove_sequence(index)
    except (ValueError, RuntimeError) as exc:
        return {"success": False, "error": str(exc)}
    return {
        "success": True,
        "deleted": sequence_id,
        "active_sequence_id": project.sequences[project.active_sequence_index].id,
    }


def list_sequences(project: Project) -> dict:
    return {
        "success": True,
        "sequences": [
            {
                "id": s.id,
                "name": s.name,
                "active": i == project.active_sequence_index,
                "clip_count": len(s.get_all_clips()),
                "algorithm": s.algorithm,
                "duration_seconds": s.duration_seconds,
                "recipe_id": s.readable_recipe.id if s.readable_recipe else None,
                "parent_recipe_id": s.readable_recipe.parent_id if s.readable_recipe else None,
            }
            for i, s in enumerate(project.sequences)
        ],
    }


# --- Variation commands -----------------------------------------------------


def activate_sequence(project: Project, sequence_id: str) -> dict:
    """Make a sequence active without editing it (view state, not history)."""
    index = next((i for i, s in enumerate(project.sequences) if s.id == sequence_id), None)
    if index is None:
        return {"success": False, "error": "Sequence not found"}
    project.set_active_sequence(index)
    return {"success": True, "sequence_id": sequence_id, "name": project.sequences[index].name, "sequence_index": index}


def duplicate_sequence(project: Project, sequence_id: str | None = None, *, name: str | None = None) -> dict:
    """Copy a sequence's timeline and recipe as a new sequence; nothing is recomputed."""
    import uuid

    sequence = _find_sequence(project, sequence_id)
    if sequence is None:
        return {"success": False, "error": "Sequence not found"}
    copy = deepcopy(sequence)
    copy.id = str(uuid.uuid4())
    for track in copy.tracks:
        track.id = str(uuid.uuid4())
        for entry in track.clips:
            entry.id = str(uuid.uuid4())
    label = name.strip() if isinstance(name, str) and name.strip() else f"{sequence.name} copy"
    copy.name = unique_sequence_name(project, label)
    recipe = sequence.readable_recipe
    copy.recipe = recipe.derive() if recipe is not None else sequence.recipe
    try:
        project.add_sequence(copy, activate=True)
    except (ValueError, RuntimeError) as exc:
        return {"success": False, "error": str(exc)}
    result = _sequence_result(copy, project)
    result["source_sequence_id"] = sequence.id
    return result


@dataclass(frozen=True)
class RegenerationPlan:
    """Everything a variation run needs, validated against the current project.

    ``prepare_regeneration`` builds it; ``regenerate_sequence`` runs it in
    place, and the desktop tab runs the algorithm half off the GUI thread
    before publishing with ``publish_recipe``.
    """

    sequence_id: str
    algorithm: str
    clip_ids: tuple[str, ...]
    parameters: dict[str, Any]
    seed: int | None
    name: str
    parent_recipe_id: str
    show_chromatic_color_bar: bool


def prepare_regeneration(
    project: Project,
    sequence_id: str | None = None,
    *,
    parameters: dict[str, Any] | None = None,
    seed: int | None = None,
    keep_seed: bool = False,
    name: str | None = None,
) -> RegenerationPlan | dict:
    """Validate a variation request; returns a plan or a ``{"success": False}`` error."""
    from core.remix.registry import registry

    sequence = _find_sequence(project, sequence_id)
    if sequence is None:
        return {"success": False, "error": "Sequence not found"}
    recipe = sequence.readable_recipe
    if recipe is None:
        return get_sequence_recipe(project, sequence.id)
    definition = registry.get(recipe.algorithm)
    if definition is None:
        return {
            "success": False,
            "error": f"Algorithm {recipe.algorithm!r} is not available in this build; "
                     "reconstruct_sequence can still replay the stored result",
        }
    if definition.version != recipe.algorithm_version:
        return {
            "success": False,
            "error": (
                f"Recipe was made with {recipe.algorithm} version {recipe.algorithm_version}; "
                f"this build has version {definition.version}. Reconstruct to replay the stored "
                "result, or generate_sequence to start a new recipe with the current version."
            ),
            "available_version": definition.version,
        }
    problems = recipe_input_problems(project, recipe)
    missing = [item.clip_id for item in recipe.inputs if item.clip_id not in project.clips_by_id]
    if problems or missing:
        return {
            "success": False,
            "error": "Recipe inputs changed: " + "; ".join(problems + [f"clip {c} is no longer in the project" for c in missing]),
        }
    merged = dict(recipe.parameters)
    for key in definition.variation_resets:
        merged.pop(key, None)  # a manual edit of the previous run must not override the new one
    if parameters is not None:
        if not isinstance(parameters, dict):
            return {"success": False, "error": "Parameters must be an object"}
        merged.update(parameters)
    if seed is not None and (isinstance(seed, bool) or type(seed) is not int or seed < 0):
        return {"success": False, "error": "Seed must be a non-negative integer"}
    if definition.seeded and seed is None and keep_seed:
        seed = recipe.seed
    if not definition.seeded:
        seed = None
    label = name.strip() if isinstance(name, str) and name.strip() else f"{sequence.name} variation"
    return RegenerationPlan(
        sequence_id=sequence.id, algorithm=recipe.algorithm,
        clip_ids=tuple(item.clip_id for item in recipe.inputs),
        parameters=merged, seed=seed, name=label, parent_recipe_id=recipe.id,
        show_chromatic_color_bar=sequence.show_chromatic_color_bar,
    )


def regenerate_sequence(
    project: Project,
    sequence_id: str | None = None,
    *,
    parameters: dict[str, Any] | None = None,
    seed: int | None = None,
    keep_seed: bool = False,
    name: str | None = None,
    cancel_event: Event | None = None,
) -> dict:
    """Run a recipe's algorithm again as a new variation.

    Uses the recipe's ordered inputs and parameters with any ``parameters``
    overrides. Seeded algorithms draw a fresh seed unless ``seed`` is given or
    ``keep_seed`` is true. The original sequence is never modified. Provider-
    assisted algorithms make new provider calls; use ``reconstruct_sequence``
    to replay without them.
    """
    plan = prepare_regeneration(
        project, sequence_id, parameters=parameters, seed=seed, keep_seed=keep_seed, name=name,
    )
    if isinstance(plan, dict):
        return plan
    return generate_sequence(
        project, plan.algorithm,
        clip_ids=list(plan.clip_ids),
        parameters=plan.parameters, seed=plan.seed, name=plan.name,
        parent_recipe_id=plan.parent_recipe_id,
        show_chromatic_color_bar=plan.show_chromatic_color_bar,
        cancel_event=cancel_event,
    )


def _sequence_summary(sequence: Sequence) -> dict:
    recipe = sequence.readable_recipe
    return {
        "sequence_id": sequence.id,
        "name": sequence.name,
        "algorithm": sequence.algorithm,
        "clip_count": len(sequence.get_all_clips()),
        "duration_seconds": round(sequence.duration_seconds, 3),
        "has_recipe": recipe is not None,
        "unreadable_recipe": sequence.recipe is not None and recipe is None,
        "recipe_id": recipe.id if recipe else None,
        "parent_recipe_id": recipe.parent_id if recipe else None,
        "seed": recipe.seed if recipe else None,
        "algorithm_version": recipe.algorithm_version if recipe else None,
        "uses_provider": recipe.uses_provider if recipe else False,
    }


def _timeline_identity(entry: Any) -> tuple:
    """Everything that makes two placed entries the same cut: media, range, place, transforms."""
    return (
        entry.source_clip_id, entry.frame_id, entry.track_index, entry.start_frame,
        entry.in_point, entry.out_point, entry.hold_frames, entry.hflip, entry.vflip, entry.reverse,
    )


def regeneration_candidates(project: Project, plan: RegenerationPlan) -> list[tuple[Clip, Source]]:
    """Candidate (Clip, Source) pairs for a plan, including source-parameter clips.

    Mirrors ``generate_sequence`` so the desktop worker and the headless path
    feed the algorithm the same inputs. Raises ``ValueError`` when a clip or
    parameter source has left the project.
    """
    from core.remix.registry import registry

    definition = registry.require(plan.algorithm)
    candidates: list[tuple[Clip, Source]] = []
    for clip_id in plan.clip_ids:
        clip = project.clips_by_id.get(clip_id)
        source = project.sources_by_id.get(clip.source_id) if clip is not None else None
        if clip is None or source is None:
            raise ValueError(f"Clip {clip_id} is no longer in the project")
        candidates.append((clip, source))
    for name in definition.source_parameters:
        source_id = plan.parameters.get(name)
        source = project.sources_by_id.get(source_id) if isinstance(source_id, str) else None
        if source is None:
            raise ValueError(f"Parameter {name!r} must name a source in this project")
        present = {clip.id for clip, _ in candidates}
        candidates.extend(
            (clip, source) for clip in project.clips_by_source.get(source.id, [])
            if clip.id not in present and not clip.disabled
        )
    return candidates


def compare_sequences(project: Project, sequence_a: str, sequence_b: str) -> dict:
    """Side-by-side summary of two sequences and their recipe differences.

    ``parameter_differences`` lists every recipe parameter whose value
    differs (``{"key", "a", "b"}``, a missing key reported as ``None``).
    ``inputs_equal`` says whether both recipes drew from the same ordered
    clip inputs; ``related`` whether one recipe derives from the other.
    """
    first = _find_sequence(project, sequence_a)
    second = _find_sequence(project, sequence_b)
    if first is None or second is None:
        return {"success": False, "error": "Sequence not found"}
    if first is second:
        return {"success": False, "error": "Choose two different sequences to compare"}
    summary_a, summary_b = _sequence_summary(first), _sequence_summary(second)
    recipe_a, recipe_b = first.readable_recipe, second.readable_recipe
    differences: list[dict] = []
    same_algorithm = (recipe_a.algorithm == recipe_b.algorithm) if recipe_a and recipe_b else (
        (first.algorithm or "") == (second.algorithm or "")
    )
    if recipe_a is not None and recipe_b is not None:
        for key in sorted(set(recipe_a.parameters) | set(recipe_b.parameters)):
            left, right = recipe_a.parameters.get(key), recipe_b.parameters.get(key)
            if left != right:
                differences.append({"key": key, "a": deepcopy(left), "b": deepcopy(right)})
    inputs_equal = (
        recipe_a is not None and recipe_b is not None
        and [i.clip_id for i in recipe_a.inputs] == [i.clip_id for i in recipe_b.inputs]
    )
    related = bool(
        recipe_a is not None and recipe_b is not None
        and (recipe_a.parent_id == recipe_b.id or recipe_b.parent_id == recipe_a.id
             or (recipe_a.parent_id is not None and recipe_a.parent_id == recipe_b.parent_id))
    )
    entries_a = [_timeline_identity(e) for e in first.get_all_clips()]
    entries_b = [_timeline_identity(e) for e in second.get_all_clips()]
    return {
        "success": True,
        "a": summary_a,
        "b": summary_b,
        "same_algorithm": same_algorithm,
        "seed_changed": (recipe_a.seed != recipe_b.seed) if recipe_a and recipe_b else None,
        "parameter_differences": differences,
        "inputs_equal": inputs_equal,
        "related": related,
        "timelines_identical": entries_a == entries_b,
        "duration_delta_seconds": round(summary_b["duration_seconds"] - summary_a["duration_seconds"], 3),
        "clip_count_delta": summary_b["clip_count"] - summary_a["clip_count"],
        "comparable_seconds": round(min(summary_a["duration_seconds"], summary_b["duration_seconds"]), 3),
    }
