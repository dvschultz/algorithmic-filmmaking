"""Qt-free sequencer algorithm engine: definitions, proposals, runs, and recipes.

Every registered algorithm has exactly one definition built on these types. A
definition owns its parameter schema, the meaning of its prerequisites, input
selection, prerequisite preparation, and pure generation. It knows nothing
about dialogs, labels, icons, or timelines: UI adapters keep those in
``ui/algorithm_config.py``. The registry instance lives in
``core/remix/registry.py`` so definition modules can import this module freely.

Generation produces a :class:`SequenceProposal` plus a
:class:`~models.recipe.SequenceRecipe`; the caller decides where and whether
to publish it. Deterministic runs are guaranteed by ``(algorithm, version,
normalized parameters, seed, ordered inputs)``.

Seed contract: ``seed`` is an explicit non-negative integer, and zero is a
valid seed. ``None`` asks the engine to draw a fresh seed, which is then
recorded. Legacy callers that used ``0`` to mean "random" translate through
:func:`legacy_seed` before reaching this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random
import secrets
from threading import Event
from typing import TYPE_CHECKING, Any, Callable, Iterable, Literal, Mapping, Sequence

from models.recipe import RealizedEntry, RecipeInput, SequenceRecipe, canonical_json

if TYPE_CHECKING:
    from models.clip import Clip, Source

ClipInput = tuple["Clip", "Source"]

ParameterType = Literal["string", "integer", "number", "boolean"]
ProposalKind = Literal["ordering", "timed", "provider"]


@dataclass(frozen=True)
class ParameterSpec:
    """One engine-owned algorithm parameter."""

    name: str
    type: ParameterType
    default: Any
    description: str = ""
    choices: tuple[Any, ...] | None = None
    minimum: int | float | None = None
    maximum: int | float | None = None

    def normalize(self, value: Any) -> Any:
        if value is None:
            value = self.default
        if self.type == "boolean":
            if type(value) is not bool:
                raise ValueError(f"Parameter {self.name!r} must be true or false")
        elif self.type == "integer":
            if type(value) is not int:
                raise ValueError(f"Parameter {self.name!r} must be an integer")
        elif self.type == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Parameter {self.name!r} must be a number")
            value = float(value)
            if value != value or value in (float("inf"), float("-inf")):
                raise ValueError(f"Parameter {self.name!r} must be finite")
        elif self.type == "string":
            if not isinstance(value, str):
                raise ValueError(f"Parameter {self.name!r} must be text")
        if self.choices is not None and value not in self.choices:
            options = ", ".join(str(choice) for choice in self.choices)
            raise ValueError(f"Parameter {self.name!r} must be one of: {options}")
        if self.minimum is not None and value < self.minimum:
            raise ValueError(f"Parameter {self.name!r} must be at least {self.minimum}")
        if self.maximum is not None and value > self.maximum:
            raise ValueError(f"Parameter {self.name!r} must be at most {self.maximum}")
        return value

    def to_dict(self) -> dict:
        data: dict[str, Any] = {
            "name": self.name,
            "type": self.type,
            "default": self.default,
            "description": self.description,
        }
        if self.choices is not None:
            data["choices"] = list(self.choices)
        if self.minimum is not None:
            data["minimum"] = self.minimum
        if self.maximum is not None:
            data["maximum"] = self.maximum
        return data


@dataclass(frozen=True)
class ProposedEntry:
    """A clip placement an algorithm proposes, in clip-relative frames."""

    clip_id: str
    source_id: str
    in_offset: int = 0
    out_offset: int | None = None
    """``None`` means the whole clip; resolved against the input when realized."""
    hflip: bool = False
    vflip: bool = False
    reverse: bool = False
    rationale: str | None = None
    provider_output: Any = None


@dataclass(frozen=True)
class SequenceProposal:
    """Typed generation output shared by simple orderings, timed edits and provider edits."""

    kind: ProposalKind
    entries: tuple[ProposedEntry, ...]
    provider_outputs: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()
    """Human-readable facts about the run (e.g. how many inputs lacked data)."""


@dataclass(frozen=True)
class GenerationRun:
    """A completed algorithm run: the proposal and the recipe that reproduces it."""

    proposal: SequenceProposal
    recipe: SequenceRecipe
    inputs: tuple[ClipInput, ...]
    """Inputs the recipe describes, in recipe order (snapshot objects the caller passed)."""

    @property
    def ordered_clips(self) -> list[ClipInput]:
        """Realized entries as ``(Clip, Source)`` pairs, for legacy consumers."""
        by_id = {clip.id: (clip, source) for clip, source in self.inputs}
        return [by_id[entry.clip_id] for entry in self.recipe.realized]


class AlgorithmDefinition:
    """Base class for one Qt-free algorithm definition.

    Subclasses set the class attributes and override :meth:`generate`.
    ``select_inputs`` and ``prepare`` are optional hooks that separate candidate
    selection and prerequisite work from pure generation.
    """

    key: str = ""
    version: int = 1
    parameters: tuple[ParameterSpec, ...] = ()
    prerequisites: tuple[str, ...] = ()
    """Analysis operations whose records the algorithm reads."""
    seeded: bool = False
    """Whether generation consumes randomness and therefore needs a seed."""
    allow_duplicates: bool = False
    kind: ProposalKind = "ordering"

    def legacy_parameters(
        self,
        *,
        direction: str | None = None,
        no_color_handling: str | None = None,
        transform_options: Mapping[str, bool] | None = None,
    ) -> dict[str, Any]:
        """Translate pre-registry keyword arguments into this algorithm's parameters.

        Compatibility hook for ``core.remix.run_registry_algorithm``; each
        migrated definition owns its own translation so the adapter never
        needs per-algorithm branches. The default ignores every legacy value.
        """
        return {}

    def select_inputs(self, candidates: Sequence[ClipInput], parameters: Mapping[str, Any]) -> list[ClipInput]:
        """Choose which candidates take part; default keeps every candidate in order."""
        return list(candidates)

    def prepare(
        self,
        inputs: Sequence[ClipInput],
        parameters: Mapping[str, Any],
        *,
        cancel_event: Event | None = None,
    ) -> list[ClipInput]:
        """Resolve prerequisites on detached inputs; default needs nothing."""
        return list(inputs)

    def generate(
        self,
        inputs: Sequence[ClipInput],
        parameters: Mapping[str, Any],
        rng: random.Random | None,
    ) -> SequenceProposal:
        raise NotImplementedError

    def describe(self) -> dict:
        return {
            "key": self.key,
            "version": self.version,
            "kind": self.kind,
            "seeded": self.seeded,
            "allow_duplicates": self.allow_duplicates,
            "prerequisites": list(self.prerequisites),
            "parameters": [spec.to_dict() for spec in self.parameters],
        }


class AlgorithmRegistry:
    """Registered definitions keyed by algorithm key."""

    def __init__(self) -> None:
        self._definitions: dict[str, AlgorithmDefinition] = {}

    def register(self, definition: AlgorithmDefinition) -> AlgorithmDefinition:
        key = definition.key
        if not key or key != key.lower():
            raise ValueError("Algorithm keys must be lowercase")
        if key in self._definitions:
            raise ValueError(f"Algorithm {key!r} is already registered")
        if type(definition.version) is not int or definition.version < 1:
            raise ValueError("Algorithm versions must be positive integers")
        names = [spec.name for spec in definition.parameters]
        if len(set(names)) != len(names):
            raise ValueError(f"Algorithm {key!r} repeats a parameter name")
        self._definitions[key] = definition
        return definition

    def get(self, key: str) -> AlgorithmDefinition | None:
        return self._definitions.get(key.lower()) if isinstance(key, str) else None

    def require(self, key: str) -> AlgorithmDefinition:
        definition = self.get(key)
        if definition is None:
            raise ValueError(f"Algorithm {key!r} is not available through the registry")
        return definition

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key.lower() in self._definitions

    def keys(self) -> list[str]:
        return sorted(self._definitions)

    def definitions(self) -> list[AlgorithmDefinition]:
        return [self._definitions[key] for key in self.keys()]

    def describe(self) -> list[dict]:
        """Parameter schemas for agent surfaces; no UI imports."""
        return [definition.describe() for definition in self.definitions()]


def normalize_parameters(definition: AlgorithmDefinition, raw: Mapping[str, Any] | None) -> dict[str, Any]:
    """Fill defaults, validate values, and reject unknown keys."""
    raw = dict(raw or {})
    known = {spec.name for spec in definition.parameters}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(
            f"Algorithm {definition.key!r} does not accept parameters: {', '.join(unknown)}"
        )
    normalized = {spec.name: spec.normalize(raw.get(spec.name)) for spec in definition.parameters}
    # Round-trip through canonical JSON so stored parameters compare stably.
    return dict(sorted(normalized.items()))


def resolve_seed(definition: AlgorithmDefinition, seed: int | None) -> int | None:
    """Apply the explicit-seed contract for one run."""
    if not definition.seeded:
        if seed is not None:
            raise ValueError(f"Algorithm {definition.key!r} does not use a seed")
        return None
    if seed is None:
        return secrets.randbelow(2**31)
    if type(seed) is not int or seed < 0:
        raise ValueError("Seed must be a non-negative integer")
    return seed


def legacy_seed(seed: Any) -> int | None:
    """Translate the pre-registry convention where ``0``/``None`` meant random."""
    if seed is None or isinstance(seed, bool):
        return None
    if type(seed) is not int:
        raise ValueError("Seed must be an integer")
    return None if seed <= 0 else seed


def _analysis_identities(clip: "Clip", operations: Iterable[str]) -> dict[str, str | None]:
    """Record identity keys of the prerequisite records a clip carries."""
    from models.analysis_record import AnalysisRecord

    identities: dict[str, str | None] = {}
    records = getattr(clip, "analysis_records", None) or {}
    for operation in operations:
        record = records.get(operation)
        if record is None:
            continue
        if isinstance(record, AnalysisRecord) and record.identity is not None:
            identities[operation] = record.identity.key
        else:
            identities[operation] = None
    return identities


def recipe_inputs(inputs: Sequence[ClipInput], operations: Iterable[str] = ()) -> tuple[RecipeInput, ...]:
    operations = tuple(operations)
    return tuple(
        RecipeInput(
            clip.id, source.id, clip.start_frame, clip.end_frame, source.fps,
            _analysis_identities(clip, operations),
        )
        for clip, source in inputs
    )


def realize(proposal: SequenceProposal, inputs: Sequence[ClipInput], *, allow_duplicates: bool) -> tuple[RealizedEntry, ...]:
    """Bind proposed entries to their inputs, validating every placement."""
    by_id = {clip.id: (clip, source) for clip, source in inputs}
    seen: set[str] = set()
    realized = []
    for entry in proposal.entries:
        pair = by_id.get(entry.clip_id)
        if pair is None:
            raise ValueError(f"Algorithm proposed clip {entry.clip_id!r} that was not an input")
        clip, source = pair
        if source.id != entry.source_id:
            raise ValueError(f"Algorithm proposed clip {entry.clip_id!r} with the wrong source")
        if not allow_duplicates and entry.clip_id in seen:
            raise ValueError(f"Algorithm proposed clip {entry.clip_id!r} more than once")
        seen.add(entry.clip_id)
        out_offset = clip.duration_frames if entry.out_offset is None else entry.out_offset
        if entry.in_offset < 0 or out_offset <= entry.in_offset or out_offset > clip.duration_frames:
            raise ValueError(f"Algorithm proposed a range outside clip {entry.clip_id!r}")
        realized.append(RealizedEntry(
            entry.clip_id, entry.source_id, entry.in_offset, out_offset,
            entry.hflip, entry.vflip, entry.reverse, entry.rationale, entry.provider_output,
        ))
    return tuple(realized)


def run_algorithm(
    definition: AlgorithmDefinition,
    candidates: Sequence[ClipInput],
    parameters: Mapping[str, Any] | None = None,
    *,
    seed: int | None = None,
    cancel_event: Event | None = None,
    parent_recipe_id: str | None = None,
    prepared: Callable[[Sequence[ClipInput]], None] | None = None,
    resolve_prerequisites: bool = True,
) -> GenerationRun | None:
    """Select, prepare, generate, and describe one run.

    Returns ``None`` when ``cancel_event`` is set before generation. ``prepared``
    receives the prepared inputs so callers can observe prerequisite results.
    Pass ``resolve_prerequisites=False`` when the caller already resolved them
    on the given inputs (e.g. a GUI job that publishes results itself).
    """
    normalized = normalize_parameters(definition, parameters)
    resolved_seed = resolve_seed(definition, seed)
    selected = definition.select_inputs(candidates, normalized)
    if len({clip.id for clip, _ in selected}) != len(selected):
        raise ValueError("Algorithm inputs must not repeat a clip")
    if cancel_event is not None and cancel_event.is_set():
        return None
    inputs = (
        definition.prepare(selected, normalized, cancel_event=cancel_event)
        if resolve_prerequisites else list(selected)
    )
    if cancel_event is not None and cancel_event.is_set():
        return None
    if prepared is not None:
        prepared(inputs)
    rng = random.Random(resolved_seed) if definition.seeded else None
    proposal = definition.generate(inputs, normalized, rng)
    if proposal.kind != definition.kind:
        raise ValueError(f"Algorithm {definition.key!r} produced a {proposal.kind} proposal")
    recipe = SequenceRecipe(
        algorithm=definition.key,
        algorithm_version=definition.version,
        parameters=normalized,
        inputs=recipe_inputs(inputs, definition.prerequisites),
        realized=realize(proposal, inputs, allow_duplicates=definition.allow_duplicates),
        seed=resolved_seed,
        parent_id=parent_recipe_id,
        provider_outputs=dict(proposal.provider_outputs),
    )
    return GenerationRun(proposal, recipe, tuple(inputs))


def parameters_document(definition: AlgorithmDefinition, parameters: Mapping[str, Any] | None) -> str:
    """Canonical text of normalized parameters, for comparisons and logs."""
    return canonical_json(normalize_parameters(definition, parameters))
