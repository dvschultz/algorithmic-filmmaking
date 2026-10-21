"""Availability flags in the chat agent's sequencer catalog.

``list_sorting_algorithms`` feeds the chat tool of the same name and the
per-turn context in ``ChatAgentWorker``. An algorithm whose analysis has not
been run must be reported unavailable, or the agent picks it and gets an
empty or meaningless ordering.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from core.project import Project
from core.remix.registry import registry
from core.spine.settings_io import AUTO_RESOLVED_PREREQUISITES, list_sorting_algorithms
from core.transcription_models import TranscriptSegment, WordTimestamp
from models.cinematography import CinematographyAnalysis
from models.clip import Clip, ExtractedText, Source


def _bare_project(count: int = 3) -> Project:
    project = Project()
    project.add_source(Source(id="s0", file_path=Path("/tmp/v0.mp4"), fps=30.0, duration_seconds=60))
    project.add_clips([
        Clip(id=f"c{i}", source_id="s0", start_frame=i * 30, end_frame=(i + 1) * 30)
        for i in range(count)
    ])
    return project


# Populates the clip fields each analysis operation writes.
_FILLERS = {
    "colors": lambda clip: setattr(clip, "dominant_colors", [(200, 40, 30)]),
    "shots": lambda clip: setattr(clip, "shot_type", "close-up"),
    "cinematography": lambda clip: setattr(clip, "cinematography", CinematographyAnalysis(shot_size="CU")),
    "extract_text": lambda clip: setattr(
        clip, "extracted_texts", [ExtractedText(frame_number=0, text="EXIT", confidence=0.9, source="vlm")],
    ),
    "describe": lambda clip: setattr(clip, "description", "A person walks past a window"),
    "transcribe": lambda clip: setattr(clip, "transcript", [
        TranscriptSegment(0.0, 1.0, "hello there", words=[WordTimestamp(0.0, 0.5, "hello"), WordTimestamp(0.5, 1.0, "there")]),
    ]),
    "gaze": lambda clip: setattr(clip, "gaze_category", "at_camera"),
    "face_embeddings": lambda clip: setattr(
        clip, "face_embeddings", [{"bbox": [0, 0, 10, 10], "embedding": [0.1] * 4, "confidence": 0.9, "frame_number": 0}],
    ),
    "embeddings": lambda clip: setattr(clip, "embedding", [0.1, 0.2, 0.3]),
    "boundary_embeddings": lambda clip: (
        setattr(clip, "first_frame_embedding", [0.1, 0.2]),
        setattr(clip, "last_frame_embedding", [0.2, 0.1]),
    ),
    "brightness": lambda clip: setattr(clip, "average_brightness", 0.5),
    "volume": lambda clip: setattr(clip, "rms_volume", -20.0),
}


def _project_with(operations) -> Project:
    project = _bare_project()
    for clip in project.clips:
        for operation in operations:
            _FILLERS[operation](clip)
    return project


def _listing(project: Project) -> dict[str, dict]:
    return {entry["key"]: entry for entry in list_sorting_algorithms(project)["algorithms"]}


def _gated(definition) -> set[str]:
    return set(definition.prerequisites) - AUTO_RESOLVED_PREREQUISITES


GATED = [d for d in registry.definitions() if _gated(d)]


def test_every_prerequisite_has_a_presence_check_and_a_filler():
    from core.cost_estimates import METADATA_CHECKS

    declared = {op for d in registry.definitions() for op in d.prerequisites}
    assert declared <= set(METADATA_CHECKS), declared - set(METADATA_CHECKS)
    assert declared <= set(_FILLERS), declared - set(_FILLERS)


def test_review_examples_are_unavailable_without_their_analysis():
    listing = _listing(_bare_project())
    for key in ("eyes_without_a_face", "word_sequencer", "word_llm_composer", "free_association"):
        assert listing[key]["available"] is False, key
        assert isinstance(listing[key]["reason"], str) and listing[key]["reason"], key


@pytest.mark.parametrize("definition", GATED, ids=lambda d: d.key)
def test_gated_algorithm_is_unavailable_with_a_reason_on_a_bare_project(definition):
    entry = _listing(_bare_project())[definition.key]
    assert entry["available"] is False
    assert isinstance(entry["reason"], str) and entry["reason"]


@pytest.mark.parametrize("definition", GATED, ids=lambda d: d.key)
def test_gated_algorithm_becomes_available_once_its_analysis_exists(definition):
    entry = _listing(_project_with(_gated(definition)))[definition.key]
    assert entry["available"] is True
    assert entry["reason"] is None


def test_algorithms_without_gated_prerequisites_stay_available_on_a_bare_project():
    listing = _listing(_bare_project())
    for definition in registry.definitions():
        if not _gated(definition):
            assert listing[definition.key]["available"] is True, definition.key


def test_every_algorithm_is_available_when_all_analysis_exists():
    listing = _listing(_project_with(_FILLERS))
    unavailable = [key for key, entry in listing.items() if not entry["available"]]
    assert unavailable == []


def test_proximity_accepts_shot_types_without_cinematography():
    # Proximity reads either record (core.remix.arrange.proximity_score).
    assert _listing(_project_with(["shots"]))["proximity"]["available"] is True


def test_empty_project_reports_gated_algorithms_unavailable():
    project = Project()
    listing = _listing(project)
    assert listing["shuffle"]["available"] is True
    assert listing["eyes_without_a_face"]["available"] is False


def test_catalog_agrees_with_gui_cost_gate_requirements():
    """A bare project gates exactly the algorithms whose GUI required_analysis needs prior analysis."""
    from ui.algorithm_config import ALGORITHM_CONFIG

    listing = _listing(_bare_project())
    for key, config in ALGORITHM_CONFIG.items():
        needs_analysis = bool(set(config["required_analysis"]) - AUTO_RESOLVED_PREREQUISITES)
        assert listing[key]["available"] is (not needs_analysis), key


@pytest.mark.parametrize("operations", [[], list(_FILLERS)], ids=["bare", "analyzed"])
def test_catalog_blocks_everything_the_sequence_tab_cards_block(operations):
    """Cards the Sequence tab disables are unavailable in the catalog too.

    The card grid is looser for dialog algorithms (their dialogs run the
    missing analysis), so the catalog may block more, never less. The
    embeddings dependency probe is forced ready; the catalog does not probe
    installs.
    """
    from ui.tabs import sequence_tab

    project = _project_with(operations)
    grid = MagicMock()
    stub = SimpleNamespace(
        _clips=list(project.clips), card_grid=grid,
        gaze_filter_label=MagicMock(), gaze_filter_dropdown=MagicMock(),
    )
    with patch.object(sequence_tab, "check_feature_ready", return_value=(True, [])), \
            patch.object(sequence_tab, "load_settings"), \
            patch.object(sequence_tab, "get_operation_feature_candidates", return_value=["embeddings"]):
        sequence_tab.SequenceTab._update_card_availability(stub)

    (cards,), _ = grid.set_algorithm_availability.call_args
    blocked = {key for key, status in cards.items() if (status[0] if isinstance(status, tuple) else status) is False}
    listing = _listing(project)
    assert {key for key in blocked if listing[key]["available"]} == set()
