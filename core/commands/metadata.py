"""Explicit editorial metadata edits, separate from analysis result updates."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, Any

from core.constants import VALID_SHOT_TYPES

if TYPE_CHECKING:
    from core.project import Project


FIELDS = {
    "clip": {
        "name",
        "notes",
        "tags",
        "shot_type",
        "transcript",
        "object_labels",
        "description",
        "description_model",
        "description_frames",
        "cinematography",
        "custom_queries",
    },
    "frame": {"notes", "tags", "shot_type"},
    "source": {"fps", "color_profile", "analyzed"},
    "project": {"name"},
}


def _lookup(project: Project, kind: str) -> dict[str, Any]:
    if kind not in FIELDS:
        raise ValueError(f"Unknown metadata entity kind: {kind}")
    if kind == "project":
        return {project.metadata.id: project.metadata}
    if kind == "clip":
        return project.clips_by_id
    if kind == "frame":
        return project.frames_by_id
    return project.sources_by_id


def _validate(kind: str, changes: dict[str, Any]) -> None:
    if set(changes) - FIELDS[kind]:
        raise ValueError("Provide supported editorial metadata fields")
    for field, value in changes.items():
        if (
            field in {"description", "description_model"}
            and value is not None
            and not isinstance(value, str)
        ):
            raise ValueError(f"{field} must be text or None")
        if (
            field == "description_frames"
            and value is not None
            and (isinstance(value, bool) or not isinstance(value, int) or value < 1)
        ):
            raise ValueError("description_frames must be a positive integer or None")
        if field == "transcript" and value is not None:
            from core.transcription_models import TranscriptSegment

            if not isinstance(value, list) or not all(
                isinstance(segment, TranscriptSegment) for segment in value
            ):
                raise ValueError("transcript must contain TranscriptSegment values")
        if field == "cinematography" and value is not None:
            from models.cinematography import CinematographyAnalysis

            if not isinstance(value, CinematographyAnalysis):
                raise ValueError(
                    "cinematography must be CinematographyAnalysis or None"
                )
        if field == "custom_queries" and value is not None:
            import json

            if not isinstance(value, list) or not all(
                isinstance(item, dict) for item in value
            ):
                raise ValueError(
                    "custom_queries must be a list of dictionaries or None"
                )
            try:
                json.dumps(value, allow_nan=False)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "custom_queries must contain serializable values"
                ) from exc
        if field in {"name", "notes"} and not isinstance(value, str):
            raise ValueError(f"{field} must be text")
        if field == "name" and kind == "project" and not value.strip():
            raise ValueError("Project name cannot be empty")
        if field in {"tags", "object_labels"}:
            if field == "object_labels" and value is None:
                continue
            if not isinstance(value, list) or not all(
                isinstance(tag, str) for tag in value
            ):
                raise ValueError(f"{field} must be a list of strings")
        if (
            field == "shot_type"
            and value is not None
            and (not isinstance(value, str) or value not in VALID_SHOT_TYPES)
        ):
            raise ValueError(f"Invalid shot type: '{value}'")
        if field == "fps" and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
            or value <= 0
        ):
            raise ValueError("FPS must be positive and finite")
        if field == "color_profile" and (
            not isinstance(value, str) or value not in {"color", "grayscale", "sepia"}
        ):
            raise ValueError("Invalid color profile")
        if field == "analyzed" and not isinstance(value, bool):
            raise ValueError("analyzed must be a boolean")


@dataclass(frozen=True)
class MetadataChange:
    entity_id: str
    entity: Any
    before: dict[str, Any]
    after: dict[str, Any]


@dataclass(frozen=True)
class EditMetadata:
    kind: str
    changes: tuple[MetadataChange, ...]

    @property
    def event_name(self) -> str:
        return {
            "clip": "clips_updated",
            "frame": "frames_updated",
            "source": "source_updated",
            "project": "project_metadata_changed",
        }[self.kind]

    @property
    def event_data(self) -> list[Any]:
        return [change.entity for change in self.changes]

    @property
    def label(self) -> str:
        return f"Edit {self.kind} metadata"

    def notification_events(self, *, undo: bool) -> list[tuple[str, Any]]:
        if self.kind == "source":
            return [(self.event_name, entity) for entity in self.event_data]
        return [(self.event_name, self.event_data)]

    @classmethod
    def capture(
        cls, project: Project, kind: str, updates: dict[str, dict[str, Any]]
    ) -> EditMetadata:
        lookup = _lookup(project, kind)
        changes = []
        for entity_id, fields in updates.items():
            _validate(kind, fields)
            entity = lookup.get(entity_id)
            if entity is None:
                continue
            after = {
                key: deepcopy(value)
                for key, value in fields.items()
                if getattr(entity, key) != value
            }
            if after:
                changes.append(
                    MetadataChange(
                        entity_id,
                        entity,
                        {key: deepcopy(getattr(entity, key)) for key in after},
                        after,
                    )
                )
        return cls(kind, tuple(changes))

    def apply(self, project: Project, *, undo: bool = False) -> list[Any]:
        lookup = _lookup(project, self.kind)
        for change in self.changes:
            expected = change.after if undo else change.before
            if lookup.get(change.entity_id) is not change.entity or any(
                getattr(change.entity, key) != value for key, value in expected.items()
            ):
                raise ValueError(
                    "Metadata changed since this edit; cannot apply history"
                )
        for change in self.changes:
            for key, value in (change.before if undo else change.after).items():
                setattr(change.entity, key, deepcopy(value))
        return self.event_data
