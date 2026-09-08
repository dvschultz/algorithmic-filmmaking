"""Sequence management shared by desktop agent and retained headless sessions."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, Any

from models.sequence import Sequence
from core.spine.security import validate_path

if TYPE_CHECKING:
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


def apply_generated_order(
    project: Project,
    entries: list[tuple[Clip, Source]],
    algorithm: str,
    name: str,
    *,
    relative_ranges: list[tuple[int, int]] | None = None,
) -> Sequence:
    """Publish resolved algorithm output in one edit, with no provider calls."""
    from fractions import Fraction
    from core.sequence_time import video_entry

    if not entries:
        raise ValueError("Generated sequence has no clips")
    if relative_ranges is not None and len(relative_ranges) != len(entries):
        raise ValueError("Provide one range for each generated clip")
    names = {sequence.name for sequence in project.sequences}
    label = name
    suffix = 2
    while label in names:
        label = f"{name} #{suffix}"
        suffix += 1
    draft = SequenceDraft.prepare(project, algorithm, label)
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
    return draft.commit(project)


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
