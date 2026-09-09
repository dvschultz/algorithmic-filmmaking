"""Rose Hobart: isolate clips featuring one person identified by reference images.

Qt-free pipeline shared by the dialog worker, chat, CLI, and MCP. Reference
images are asset paths (validated by the calling surface); their face
embeddings are extracted with the same verified execution as the clip faces,
and the match confidences are kept as provider output so a recipe can be
inspected and replayed without running face detection again.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from threading import Event
from typing import Any, Callable, Mapping, Sequence

from core.remix.engine import (
    AlgorithmDefinition, ClipInput, ParameterSpec, Prepared, ProposedEntry, SequenceProposal,
)

logger = logging.getLogger(__name__)

ORDERINGS = ("original", "duration", "color", "brightness", "confidence", "random")
SENSITIVITIES = ("strict", "balanced", "loose")


@dataclass(frozen=True)
class PersonMatch:
    clip_id: str
    confidence: float


@dataclass(frozen=True)
class PersonMatches:
    """Result of scanning clips for a reference person."""

    matches: tuple[PersonMatch, ...]
    reference_identity: dict[str, Any] | None
    reference_stamps: dict[str, list[Any]]
    reference_execution: dict[str, Any] | None = None
    outcomes: tuple[Any, ...] = ()
    """Per-clip face outcomes, for callers that publish analysis into a project."""


def references_unchanged(paths: Sequence[str], stamps: Mapping[str, Any]) -> bool:
    from core.jobs.media import media_stamp

    for path in paths:
        current = media_stamp(Path(path))
        if stamps.get(path) is None or current is None or list(current) != list(stamps[path]):
            return False
    return True


def match_person(
    reference_paths: Sequence[str],
    clips: Sequence[ClipInput],
    *,
    threshold: float,
    sample_interval: float,
    cancel_event: Event | None = None,
    progress: Callable[[str], None] | None = None,
    face_cache: Any = None,
    on_match: Callable[[int], None] | None = None,
) -> PersonMatches | None:
    """Scan ``clips`` for the person in ``reference_paths``.

    Returns ``None`` when cancelled. Raises ``ValueError`` when references or
    analysis inputs change mid-run or when no reference face is found.
    ``face_cache`` is an optional GUI result journal used to reuse verified
    face analysis; headless callers run the operation directly.
    """
    from core.analysis.faces import average_embeddings, compare_faces, extract_faces_from_image
    from core.feature_registry import check_feature_ready, install_for_feature
    from core.jobs.media import MediaFingerprints, media_stamp
    from core.operations.face_records import face_environment, saved_execution, verified_face_execution
    from core.operations.faces import FaceOptions, FaceOutcome, face_task, run_faces
    from models.analysis_record import AnalysisRecord

    cancel = cancel_event or Event()
    say = progress or (lambda message: None)

    available, _missing = check_feature_ready("face_detect")
    if not available:
        say("Installing face detection dependencies...")
        if not install_for_feature("face_detect"):
            raise RuntimeError("Failed to install face detection dependencies (insightface)")

    stamps: dict[str, list[Any]] = {}
    for path in reference_paths:
        stamp = media_stamp(Path(path))
        if stamp is None:
            raise ValueError(f"Reference image is missing or unreadable: {path}")
        stamps[path] = list(stamp)

    say("Extracting reference face embeddings...")
    fingerprints = MediaFingerprints(cancel)
    reference_identity: dict[str, Any] | None = None
    reference_execution: dict[str, Any] | None = None
    embeddings = []
    for path in reference_paths:
        if cancel.is_set():
            return None
        executions: list[dict] = []
        faces = extract_faces_from_image(Path(path), on_execution=lambda value: executions.append(deepcopy(value)))
        if not faces:
            continue
        if len(executions) != 1:
            raise ValueError("Reference faces require one verified execution")
        identity = verified_face_execution(executions[0], fingerprints, face_environment())
        if reference_identity is not None and identity != reference_identity:
            raise ValueError("Reference faces used different model executions")
        reference_identity = identity
        reference_execution = executions[0]
        embeddings.append(max(faces, key=lambda f: f["confidence"])["embedding"])
    if not embeddings:
        raise ValueError("No faces detected in reference images.")
    if len(embeddings) > 1:
        embeddings = [average_embeddings(embeddings)]
    if cancel.is_set():
        return None
    if not references_unchanged(reference_paths, stamps):
        raise ValueError("Reference images changed during matching")

    tasks = tuple(face_task(clip, source) for clip, source in clips)
    options = FaceOptions(sample_interval)
    outcomes: list[FaceOutcome] = []

    def deliver(outcome: FaceOutcome) -> None:
        outcomes.append(outcome)

    def analysis_progress(current: int, count: int) -> None:
        say(f"Analyzing clip {current} of {count}...")

    if face_cache is not None:
        outcomes = list(face_cache.run(
            tasks, cancel, lambda: not cancel.is_set() and references_unchanged(reference_paths, stamps),
            deliver, analysis_progress,
        ))
    else:
        run_faces(tasks, options, cancel_event=cancel, on_outcome=deliver, progress=analysis_progress)
    if cancel.is_set():
        return None
    if len(outcomes) != len(clips):
        raise ValueError("Face analysis did not complete for every clip")

    matches = []
    for index, ((clip, _source), outcome) in enumerate(zip(clips, outcomes)):
        if cancel.is_set():
            return None
        say(f"Processing clip {index + 1} of {len(clips)}...")
        if not outcome.has_result:
            raise ValueError(outcome.message or "Face analysis did not complete")
        if outcome.record_json is None:
            raise ValueError("Clip faces require verified execution")
        record = AnalysisRecord.from_dict(json.loads(outcome.record_json))
        if verified_face_execution(saved_execution(record), fingerprints, face_environment()) != reference_identity:
            raise ValueError("Reference and clip faces used different model executions")
        is_match, confidence = compare_faces(embeddings, outcome.face_dicts(), threshold)
        if is_match:
            matches.append(PersonMatch(clip.id, float(confidence)))
            if on_match is not None:
                on_match(len(matches))
    if not references_unchanged(reference_paths, stamps):
        raise ValueError("Reference images changed during matching")
    return PersonMatches(tuple(matches), reference_identity, stamps, reference_execution, tuple(outcomes))


def order_matches(
    matched: list[tuple[ClipInput, float]], ordering: str, rng: Any,
) -> list[ClipInput]:
    """Deterministic orderings over matched clips; ``random`` uses the run's rng."""
    import colorsys

    items = list(matched)
    if ordering == "original":
        items.sort(key=lambda m: (m[0][1].file_path.name, m[0][0].start_frame))
    elif ordering == "duration":
        items.sort(key=lambda m: m[0][0].duration_seconds(m[0][1].fps))
    elif ordering == "color":
        def hue(m):
            colors = m[0][0].dominant_colors
            if colors:
                r, g, b = colors[0]
                return colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)[0]
            return 0.5
        items.sort(key=hue)
    elif ordering == "brightness":
        items.sort(key=lambda m: m[0][0].average_brightness if m[0][0].average_brightness is not None else 0.5)
    elif ordering == "confidence":
        items.sort(key=lambda m: m[1], reverse=True)
    elif ordering == "random":
        rng.shuffle(items)
    return [pair for pair, _ in items]


class RoseHobartDefinition(AlgorithmDefinition):
    key = "rose_hobart"
    version = 1
    kind = "provider"
    seeded = True
    provider = True
    prerequisites = ("face_embeddings",)
    asset_parameters = ("reference_image_paths",)
    parameters = (
        ParameterSpec("reference_image_paths", "array", [], "1-3 reference images of the person"),
        ParameterSpec("sensitivity", "string", "balanced", "Match sensitivity", choices=SENSITIVITIES),
        ParameterSpec("ordering", "string", "original", "Order of matched clips", choices=ORDERINGS),
        ParameterSpec("sampling_interval", "number", 1.0, "Seconds between sampled frames", minimum=0.25, maximum=5.0),
    )

    def prepare(self, inputs, parameters, *, cancel_event=None, progress=None, resources=None):
        from core.analysis.faces import SENSITIVITY_PRESETS

        paths = parameters["reference_image_paths"]
        if not 1 <= len(paths) <= 3 or not all(isinstance(p, str) and p for p in paths):
            raise ValueError("Provide 1-3 reference image paths")
        result = match_person(
            paths, list(inputs),
            threshold=SENSITIVITY_PRESETS[parameters["sensitivity"]],
            sample_interval=parameters["sampling_interval"],
            cancel_event=cancel_event, progress=progress,
            face_cache=(resources or {}).get("face_cache"),
            on_match=(resources or {}).get("on_match"),
        )
        if result is None:
            return Prepared([], {"cancelled": True})
        return Prepared(list(inputs), {
            "matches": {m.clip_id: m.confidence for m in result.matches},
            "reference_identity": result.reference_identity,
            "reference_stamps": result.reference_stamps,
            # Not persisted: handed to callers that publish face analysis.
            "reference_execution": result.reference_execution,
            "face_outcomes": result.outcomes,
        })

    def generate(self, inputs, parameters, rng, context=None):
        context = context or {}
        if context.get("cancelled"):
            return SequenceProposal("ordering", ())
        confidences: dict[str, float] = dict(context.get("matches") or {})
        matched = [(item, confidences[item[0].id]) for item in inputs if item[0].id in confidences]
        ordered = order_matches(matched, parameters["ordering"], rng)
        notes = [f"{len(matched)} of {len(inputs)} clips matched the reference person"]
        return SequenceProposal(
            "provider",
            tuple(
                ProposedEntry(clip.id, source.id, provider_output={"confidence": confidences[clip.id]})
                for clip, source in ordered
            ),
            provider_outputs={
                "reference_identity": context.get("reference_identity"),
                "reference_stamps": context.get("reference_stamps") or {},
            },
            notes=tuple(notes),
        )
