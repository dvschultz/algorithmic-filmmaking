"""Versioned sequence recipes: generation inputs plus realized decisions.

A recipe records what an algorithm was asked to do (algorithm identity and
version, normalized parameters, explicit seed, ordered input clips with the
analysis identities they carried) and what it actually produced (ordered
realized entries with trims, transforms, rationale and any provider output).

Reconstruction replays the realized entries only. It never calls the
algorithm or a provider. Regeneration runs the algorithm again with the same
inputs and parameters but, by default, a fresh seed, and produces a new
recipe. These values import no project models, Qt, or media runtimes.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
import re
from typing import Any
import uuid

RECIPE_SCHEMA_VERSION = 1

_ID = re.compile(r"[A-Za-z0-9_.:-]{1,128}")
_KEY = re.compile(r"[a-z][a-z0-9_]{0,63}")


def canonical_json(value: Any) -> str:
    """Stable JSON used for recipe fingerprints and stored parameters."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _check_json(value: Any, label: str) -> Any:
    """Return ``value`` if it is plain JSON data with a canonical encoding."""
    try:
        return json.loads(canonical_json(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be JSON data: {exc}") from exc


def _check_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError(f"{label} must be a short identifier string")
    return value


def _check_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer of at least {minimum}")
    return value


def _check_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be a boolean")
    return value


@dataclass(frozen=True)
class RecipeInput:
    """One clip offered to the algorithm, as it was when generation ran."""

    clip_id: str
    source_id: str
    start_frame: int
    end_frame: int
    source_fps: float
    analysis: dict[str, str | None] = field(default_factory=dict)
    """Analysis operation → record identity key (``None`` for provenance-unknown)."""

    def __post_init__(self) -> None:
        _check_id(self.clip_id, "Recipe input clip id")
        _check_id(self.source_id, "Recipe input source id")
        _check_int(self.start_frame, "Recipe input start frame")
        _check_int(self.end_frame, "Recipe input end frame", minimum=1)
        if self.end_frame <= self.start_frame:
            raise ValueError("Recipe input end frame must exceed its start frame")
        if isinstance(self.source_fps, bool) or not isinstance(self.source_fps, (int, float)) or not self.source_fps > 0:
            raise ValueError("Recipe input source fps must be positive")
        object.__setattr__(self, "source_fps", float(self.source_fps))
        if not isinstance(self.analysis, dict):
            raise ValueError("Recipe input analysis must be a mapping")
        for operation, key in self.analysis.items():
            if not isinstance(operation, str) or not _KEY.fullmatch(operation):
                raise ValueError("Recipe input analysis operations must be lowercase keys")
            if key is not None and (not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{64}", key)):
                raise ValueError("Recipe input analysis identities must be SHA-256 keys or None")
        object.__setattr__(self, "analysis", dict(sorted(self.analysis.items())))

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame

    def to_dict(self) -> dict:
        return {
            "clip_id": self.clip_id,
            "source_id": self.source_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "source_fps": self.source_fps,
            "analysis": dict(self.analysis),
        }

    @classmethod
    def from_dict(cls, value: dict) -> RecipeInput:
        if not isinstance(value, dict):
            raise ValueError("Recipe input must be an object")
        return cls(
            value["clip_id"], value["source_id"], value["start_frame"], value["end_frame"],
            value["source_fps"], dict(value.get("analysis") or {}),
        )


@dataclass(frozen=True)
class RealizedEntry:
    """One placed timeline entry the algorithm chose."""

    clip_id: str
    source_id: str
    in_offset: int
    """Clip-relative start frame of the used range (0 = clip start)."""
    out_offset: int
    """Clip-relative exclusive end frame of the used range."""
    hflip: bool = False
    vflip: bool = False
    reverse: bool = False
    rationale: str | None = None
    provider_output: Any = None
    """Provider decision needed to replay this entry offline (JSON data)."""

    def __post_init__(self) -> None:
        _check_id(self.clip_id, "Realized entry clip id")
        _check_id(self.source_id, "Realized entry source id")
        _check_int(self.in_offset, "Realized entry in offset")
        _check_int(self.out_offset, "Realized entry out offset", minimum=1)
        if self.out_offset <= self.in_offset:
            raise ValueError("Realized entry out offset must exceed its in offset")
        for name in ("hflip", "vflip", "reverse"):
            _check_bool(getattr(self, name), f"Realized entry {name}")
        if self.rationale is not None and not isinstance(self.rationale, str):
            raise ValueError("Realized entry rationale must be text")
        object.__setattr__(self, "provider_output", _check_json(self.provider_output, "Realized entry provider output"))

    @property
    def relative_range(self) -> tuple[int, int]:
        return self.in_offset, self.out_offset

    @property
    def has_transform(self) -> bool:
        return self.hflip or self.vflip or self.reverse

    def to_dict(self) -> dict:
        data: dict[str, Any] = {
            "clip_id": self.clip_id,
            "source_id": self.source_id,
            "in_offset": self.in_offset,
            "out_offset": self.out_offset,
        }
        for name in ("hflip", "vflip", "reverse"):
            if getattr(self, name):
                data[name] = True
        if self.rationale is not None:
            data["rationale"] = self.rationale
        if self.provider_output is not None:
            data["provider_output"] = self.provider_output
        return data

    @classmethod
    def from_dict(cls, value: dict) -> RealizedEntry:
        if not isinstance(value, dict):
            raise ValueError("Realized entry must be an object")
        return cls(
            value["clip_id"], value["source_id"], value["in_offset"], value["out_offset"],
            value.get("hflip", False), value.get("vflip", False), value.get("reverse", False),
            value.get("rationale"), value.get("provider_output"),
        )


@dataclass(frozen=True)
class SequenceRecipe:
    """Complete, replayable description of one generated sequence."""

    algorithm: str
    algorithm_version: int
    parameters: dict[str, Any]
    inputs: tuple[RecipeInput, ...]
    realized: tuple[RealizedEntry, ...]
    seed: int | None = None
    """Explicit seed used by the run; ``None`` means the algorithm draws no randomness."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    parent_id: str | None = None
    """Recipe this one was duplicated or regenerated from."""
    provider_outputs: dict[str, Any] = field(default_factory=dict)
    """Sequence-level provider results (e.g. a composed poem) kept for offline replay."""
    schema_version: int = RECIPE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RECIPE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported recipe schema version {self.schema_version!r}")
        if not isinstance(self.algorithm, str) or not _KEY.fullmatch(self.algorithm):
            raise ValueError("Recipe algorithm must be a lowercase key")
        _check_int(self.algorithm_version, "Recipe algorithm version", minimum=1)
        if not isinstance(self.parameters, dict):
            raise ValueError("Recipe parameters must be a mapping")
        object.__setattr__(self, "parameters", _check_json(dict(self.parameters), "Recipe parameters"))
        if self.seed is not None:
            _check_int(self.seed, "Recipe seed")
        _check_id(self.id, "Recipe id")
        if self.parent_id is not None:
            _check_id(self.parent_id, "Recipe parent id")
        if not isinstance(self.created_at, str) or not self.created_at:
            raise ValueError("Recipe creation time is required")
        inputs = tuple(self.inputs)
        realized = tuple(self.realized)
        if not all(isinstance(item, RecipeInput) for item in inputs):
            raise ValueError("Recipe inputs must be RecipeInput values")
        if not all(isinstance(item, RealizedEntry) for item in realized):
            raise ValueError("Recipe realized entries must be RealizedEntry values")
        if len({item.clip_id for item in inputs}) != len(inputs):
            raise ValueError("Recipe inputs must not repeat a clip")
        by_clip = {item.clip_id: item for item in inputs}
        for entry in realized:
            source = by_clip.get(entry.clip_id)
            if source is None:
                raise ValueError(f"Realized entry {entry.clip_id} was not a recipe input")
            if source.source_id != entry.source_id:
                raise ValueError(f"Realized entry {entry.clip_id} names a different source than its input")
            if entry.out_offset > source.duration_frames:
                raise ValueError(f"Realized entry {entry.clip_id} extends past its input clip")
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "realized", realized)
        if not isinstance(self.provider_outputs, dict):
            raise ValueError("Recipe provider outputs must be a mapping")
        object.__setattr__(self, "provider_outputs", _check_json(dict(self.provider_outputs), "Recipe provider outputs"))

    @property
    def input_fingerprint(self) -> str:
        """Identity of the ordered inputs, independent of the realized output."""
        return sha256(canonical_json([item.to_dict() for item in self.inputs]).encode()).hexdigest()

    @property
    def generation_fingerprint(self) -> str:
        """Identity of everything that determines the output for a deterministic run."""
        return sha256(canonical_json({
            "algorithm": self.algorithm,
            "algorithm_version": self.algorithm_version,
            "parameters": self.parameters,
            "seed": self.seed,
            "inputs": self.input_fingerprint,
        }).encode()).hexdigest()

    @property
    def uses_provider(self) -> bool:
        return bool(self.provider_outputs) or any(entry.provider_output is not None for entry in self.realized)

    def derive(self, **changes: Any) -> SequenceRecipe:
        """Copy the recipe as a new one that remembers its origin."""
        changes.setdefault("parent_id", self.id)
        changes["id"] = str(uuid.uuid4())
        changes["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return replace(self, **changes)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "algorithm": self.algorithm,
            "algorithm_version": self.algorithm_version,
            "parameters": dict(self.parameters),
            "seed": self.seed,
            "created_at": self.created_at,
            "parent_id": self.parent_id,
            "inputs": [item.to_dict() for item in self.inputs],
            "realized": [item.to_dict() for item in self.realized],
            "provider_outputs": dict(self.provider_outputs),
        }

    @classmethod
    def from_dict(cls, value: dict) -> SequenceRecipe:
        if not isinstance(value, dict):
            raise ValueError("Recipe must be an object")
        version = value.get("schema_version")
        if version != RECIPE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported recipe schema version {version!r}")
        return cls(
            algorithm=value["algorithm"],
            algorithm_version=value["algorithm_version"],
            parameters=dict(value.get("parameters") or {}),
            inputs=tuple(RecipeInput.from_dict(item) for item in value.get("inputs", [])),
            realized=tuple(RealizedEntry.from_dict(item) for item in value.get("realized", [])),
            seed=value.get("seed"),
            id=value["id"],
            created_at=value["created_at"],
            parent_id=value.get("parent_id"),
            provider_outputs=dict(value.get("provider_outputs") or {}),
            schema_version=version,
        )


@dataclass(frozen=True)
class UnreadableRecipe:
    """A stored recipe this build cannot interpret; preserved verbatim on save."""

    document: str

    def to_dict(self) -> Any:
        return json.loads(self.document)


StoredRecipe = SequenceRecipe | UnreadableRecipe


def load_recipe(value: Any) -> StoredRecipe | None:
    """Parse a stored recipe, retaining unknown or future documents unchanged."""
    if value is None:
        return None
    try:
        return SequenceRecipe.from_dict(value)
    except (ValueError, KeyError, TypeError, AttributeError):
        return UnreadableRecipe(canonical_json(value))
