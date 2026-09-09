"""Sequence management shared by desktop agent and retained headless sessions."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, Any

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


def apply_generated_order(
    project: Project,
    entries: list[tuple[Clip, Source]],
    algorithm: str,
    name: str,
    *,
    relative_ranges: list[tuple[int, int]] | None = None,
    recipe: SequenceRecipe | None = None,
) -> Sequence:
    """Publish resolved algorithm output in one edit, with no provider calls."""
    from fractions import Fraction
    from core.sequence_time import video_entry

    if not entries:
        raise ValueError("Generated sequence has no clips")
    if relative_ranges is not None and len(relative_ranges) != len(entries):
        raise ValueError("Provide one range for each generated clip")
    draft = SequenceDraft.prepare(project, algorithm, unique_sequence_name(project, name))
    position = Fraction(0)
    for index, (clip, source) in enumerate(entries):
        if (
            project.clips_by_id.get(clip.id) is not clip
            or project.sources_by_id.get(source.id) is not source
        ):
            raise ValueError("Generated clip inputs no longer belong to this project")
        start, end = (
            relative_ranges[index]
            if relative_ranges is not None
            else (0, clip.duration_frames)
        )
        if start < 0 or end <= start or end > clip.duration_frames:
            raise ValueError("Generated range falls outside its source clip")
        entry = video_entry(
            clip, source, timeline_fps=draft.sequence.fps,
            start=position, relative_range=(start, end),
        )
        draft.sequence.tracks[0].add_clip(entry)
        position = entry.timeline_range.end
    draft.sequence.recipe = recipe
    return draft.commit(project)


# --- Registry-backed generation -------------------------------------------


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
) -> Sequence:
    """Build a timeline from realized recipe entries and publish it as one edit.

    Performs no algorithm or provider work. Raises ``ValueError`` with every
    input problem when the project no longer matches the recipe.
    """
    from fractions import Fraction
    from core.sequence_time import video_entry

    if not recipe.realized:
        raise ValueError("Recipe realized no clips")
    problems = recipe_input_problems(project, recipe)
    if problems:
        raise ValueError("Recipe inputs changed: " + "; ".join(problems))
    draft = SequenceDraft.prepare(
        project, recipe.algorithm, unique_sequence_name(project, name),
        replace_sequence_id=replace_sequence_id,
        show_chromatic_color_bar=show_chromatic_color_bar,
    )
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
    snapshots = deepcopy(candidates)
    try:
        run = run_algorithm(
            definition, snapshots, parameters, seed=seed,
            cancel_event=cancel_event, parent_recipe_id=parent_recipe_id,
        )
    except ValueError as exc:
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
            }
            for i, s in enumerate(project.sequences)
        ],
    }
