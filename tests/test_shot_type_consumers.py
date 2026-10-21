"""Every consumer of Clip.shot_type / Frame.shot_type matches by meaning.

Stored values come in several spellings: the canonical "wide shot" that shot
analysis writes, the short "wide"/"ECU" that cinematography analysis writes,
and older title-case labels ("Wide Shot"). Filters, orderings and scores must
treat all of them as the same shot type.
"""

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from PySide6.QtWidgets import QApplication

from models.clip import Clip, Source
from models.frame import Frame

app = QApplication.instance() or QApplication(sys.argv)

SOURCE = Source(id="s", file_path=Path("/tmp/s.mp4"), fps=30.0, width=1920, height=1080)


def _clip(clip_id, shot_type, **kwargs):
    return Clip(id=clip_id, source_id="s", start_frame=0, end_frame=30, shot_type=shot_type, **kwargs)


# --- shot_type_matches -----------------------------------------------------


@pytest.mark.parametrize(
    "stored, wanted, expected",
    [
        ("wide shot", "Wide", True),
        ("Wide Shot", "wide shot", True),
        ("wide", "Wide Shot", True),
        ("ECU", "Extreme CU", True),
        ("ECU", "extreme close-up", True),
        ("close-up", "extreme close-up", False),
        ("custom", "CUSTOM", True),  # both unknown: case-insensitive raw compare
        ("custom", "wide shot", False),
        ("wide shot", "custom", False),
        (None, "wide shot", False),
        ("", "wide shot", False),
    ],
)
def test_shot_type_matches(stored, wanted, expected):
    from core.constants import shot_type_matches

    assert shot_type_matches(stored, wanted) is expected


def test_every_cinematography_simple_type_is_known():
    from core.constants import normalize_shot_type
    from models.cinematography import SHOT_SIZES, CinematographyAnalysis

    for size in SHOT_SIZES:
        simple = CinematographyAnalysis(shot_size=size).get_simple_shot_type()
        assert normalize_shot_type(simple) is not None, (size, simple)


# --- load keeps analysis records consistent --------------------------------


def test_legacy_value_round_trips_and_matches_canonical():
    from core.constants import shot_type_matches

    clip = Clip.from_dict(_clip("a", "Wide Shot").to_dict())
    frame = Frame.from_dict(Frame(id="f", shot_type="Wide Shot").to_dict())
    assert shot_type_matches(clip.shot_type, "wide shot")
    assert shot_type_matches(frame.shot_type, "wide shot")
    assert Clip.from_dict(_clip("b", "custom").to_dict()).shot_type == "custom"


@pytest.mark.parametrize("size", ["ELS", "LS", "MS", "CU", "ECU"])
def test_load_keeps_cinematography_record_value_current(size):
    """A stored cinematography result must still equal the clip after load."""
    from models.analysis_record import AnalysisRecord
    from models.cinematography import CinematographyAnalysis
    from core.operations.cinematography import cinematography_value

    analysis = CinematographyAnalysis(shot_size=size)
    clip = _clip("a", analysis.get_simple_shot_type(), cinematography=analysis)
    clip.analysis_records["cinematography"] = AnalysisRecord.legacy(
        {"cinematography": analysis.to_dict(), "shot_type": analysis.get_simple_shot_type()}
    )
    loaded = Clip.from_dict(clip.to_dict())
    assert loaded.analysis_records["cinematography"].value == cinematography_value(loaded)


# --- intention auto-build filter -------------------------------------------


@pytest.mark.parametrize("stored", ["wide shot", "Wide Shot", "wide"])
def test_intention_auto_build_filter_matches_any_spelling(monkeypatch, stored):
    from core.intention_workflow import WorkflowState
    from core.project import Project
    from ui.main_window import MainWindow

    monkeypatch.setattr("ui.main_window.QMessageBox", Mock())
    keep = _clip("keep", stored)
    drop = _clip("drop", "close-up")
    project = Project.new()
    plan = object()
    workflow = Mock(state=WorkflowState.COMPLETE, plan=plan)
    workflow.get_all_clips.return_value = [keep, drop]
    workflow.get_all_sources.return_value = [SOURCE]
    workflow.get_algorithm_with_direction.return_value = ("duration", None)
    sequence_tab = Mock()
    sequence_tab.apply_intention_workflow_result.return_value = {"success": True, "clip_count": 1}
    window = SimpleNamespace(
        project=project,
        intention_workflow=workflow,
        intention_import_dialog=Mock(),
        sources_by_id={SOURCE.id: SOURCE},
        clips_by_id={},
        collect_tab=Mock(),
        analyze_tab=Mock(),
        sequence_tab=sequence_tab,
        status_bar=Mock(),
        _gui_state=None,
        _intention_pending_shot_type="wide shot",
        _sync_intention_workflow_ui=Mock(),
        _mark_dirty=Mock(),
        _refresh_sequence_tab_clips=Mock(),
    )

    MainWindow._finalize_intention_workflow(
        window, workflow, plan, project, project.session.session_id
    )

    assert sequence_tab.apply_intention_workflow_result.called, "every clip was filtered out"
    kwargs = sequence_tab.apply_intention_workflow_result.call_args.kwargs
    assert [clip.id for clip, _source in kwargs["clips_with_sources"]] == ["keep"]


# --- remix algorithms -------------------------------------------------------


def test_focal_ladder_orders_legacy_and_cinematography_values():
    from core.remix.arrange import ShotTypeDefinition

    inputs = [
        (_clip("ecu", "ECU"), SOURCE),
        (_clip("close", "close-up"), SOURCE),
        (_clip("medium", "Medium Shot"), SOURCE),
        (_clip("wide", "wide"), SOURCE),
        (_clip("unknown", None), SOURCE),
    ]
    proposal = ShotTypeDefinition().generate(inputs, {}, None)
    assert [entry.clip_id for entry in proposal.entries] == [
        "wide", "medium", "close", "ecu", "unknown",
    ]


@pytest.mark.parametrize("stored", ["Wide Shot", "wide"])
def test_proximity_scores_use_canonical_shot_type(stored):
    from core.remix.arrange import proximity_score
    from core.remix.reference_match import _get_proximity_score

    clip = _clip("a", stored)
    assert proximity_score(clip) == 2.0
    assert _get_proximity_score(clip) == 2.0


def test_signature_style_scores_legacy_shot_type():
    from core.remix.drawing_segment import DrawingSegment
    from core.remix.signature_style import _compute_match_score

    segment = DrawingSegment(
        x_start=0, x_end=10, target_duration_seconds=1.0, target_pacing=0.5, shot_type="wide shot"
    )
    canonical = _compute_match_score(segment, _clip("a", "wide shot"), SOURCE)
    for stored in ("Wide Shot", "wide"):
        assert _compute_match_score(segment, _clip("b", stored), SOURCE) == canonical


# --- spine find_similar / group / list_frames / timeline shuffle ----------


def _project(clips):
    from core.project import Project

    return Project(sources=[SOURCE], clips=clips)


def test_find_similar_matches_legacy_and_scores_adjacent_sizes():
    from core.spine.clips import find_similar_clips

    project = _project([
        _clip("ref", "close-up"),
        _clip("same", "Close-up"),
        _clip("adjacent", "ECU"),
        _clip("far", "wide shot"),
    ])
    result = find_similar_clips(project, "ref", criteria=["shot_type"])
    scores = {item["clip_id"]: item["similarity_score"] for item in result["similar_clips"]}
    assert scores == {"same": 1.0, "adjacent": 0.5}


def test_group_clips_by_shot_type_merges_spellings():
    from core.spine.clips import group_clips_by

    project = _project([
        _clip("a", "wide shot"), _clip("b", "Wide Shot"), _clip("c", "wide"), _clip("d", "custom"),
    ])
    groups = group_clips_by(project, "shot_type")["groups"]
    assert set(groups) == {"wide shot", "custom"}
    assert groups["wide shot"]["count"] == 3


@pytest.mark.parametrize("wanted", ["Extreme CU", "extreme close-up", "ECU"])
def test_list_frames_matches_cinematography_value(wanted):
    from core.spine.queries import list_frames

    project = _project([])
    project.add_frames([
        Frame(id="ecu", shot_type="ECU"),
        Frame(id="legacy", shot_type="Extreme Close-up"),
        Frame(id="close", shot_type="close-up"),
    ])
    result = list_frames(project, shot_type=wanted)
    assert {frame["id"] for frame in result["frames"]} == {"ecu", "legacy"}


def test_timeline_shuffle_by_shot_type_uses_canonical_order():
    from core.spine.timeline import shuffle_clips
    from models.sequence import Sequence, SequenceClip

    clips = [
        _clip("ecu", "ECU"), _clip("full", "full shot"), _clip("wide", "Wide Shot"),
        _clip("medium", "medium"),
    ]
    project = _project(clips)
    sequence = Sequence(name="seq")
    for clip in clips:
        sequence.tracks[0].clips.append(
            SequenceClip(id=f"sc-{clip.id}", source_clip_id=clip.id, source_id="s", in_point=0, out_point=30)
        )
    project.add_sequence(sequence, activate=True)
    shuffle_clips(project, sequence.id, "by_shot_type")
    order = [sc.source_clip_id for sc in project.sequence.tracks[0].clips]
    assert order == ["wide", "full", "medium", "ecu"]


# --- agent list_clips, CLI filters ------------------------------------------


def test_agent_list_clips_matches_cinematography_value():
    from core.chat_tools import list_clips

    project = _project([_clip("a", "ECU"), _clip("b", "close-up")])
    result = list_clips(None, project, shot_type="Extreme CU")
    assert [clip["id"] for clip in result["clips"]] == ["a"]


def test_agent_list_clips_sorts_spellings_together():
    from core.chat_tools import list_clips

    project = _project([_clip("a", "Wide Shot"), _clip("b", "close-up"), _clip("c", "wide shot")])
    result = list_clips(None, project, sort_by="shot_type")
    assert [clip["id"] for clip in result["clips"]] == ["b", "a", "c"]


@pytest.mark.parametrize("module", ["cli.commands.project", "cli.commands.export"])
def test_cli_filter_matches_cinematography_value(module):
    import importlib

    apply_filter = importlib.import_module(module)._apply_filter
    clips = [_clip("a", "ECU"), _clip("b", "wide shot")]
    assert [c.id for c in apply_filter(clips, "shot_type=extreme close-up")] == ["a"]
    assert [c.id for c in apply_filter(clips, "shot_type=wide")] == ["b"]


# --- GUI: sequence tab filter, shot type dropdown --------------------------


def _sequence_tab(clips):
    from ui.tabs.sequence_tab import SequenceTab

    tab = SequenceTab.__new__(SequenceTab)
    tab._current_state = SequenceTab.STATE_CARDS
    tab._available_clips = [(clip, SOURCE) for clip in clips]
    return tab


@pytest.mark.parametrize("wanted", ["Wide", "wide shot", "Wide Shot"])
def test_sequence_tab_shot_filter_counts_every_spelling(wanted):
    tab = _sequence_tab([
        _clip("a", "wide shot"), _clip("b", "Wide Shot"), _clip("c", "wide"), _clip("d", "ECU"),
    ])
    assert tab.apply_shot_type_filter(wanted) == 3


def test_sequence_tab_shot_filter_handles_ecu_and_unknown_values():
    tab = _sequence_tab([_clip("a", "ECU"), _clip("b", "custom"), _clip("c", None)])
    assert tab.apply_shot_type_filter("Extreme CU") == 1
    assert tab.apply_shot_type_filter("custom") == 1


@pytest.mark.parametrize("stored, expected", [("Wide Shot", "wide shot"), ("ECU", "extreme close-up")])
def test_shot_type_dropdown_selects_legacy_value(stored, expected):
    from ui.widgets.shot_type_dropdown import ShotTypeDropdown

    dropdown = ShotTypeDropdown()
    dropdown.setValue(stored)
    assert dropdown.currentData() == expected


# --- MCP filter_clips --------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "wanted, expected",
    [("Wide", {"wide", "legacy-wide"}), ("Extreme CU", {"ecu"})],
)
async def test_mcp_filter_clips_matches_display_names(wanted, expected):
    from core.project import Project
    from scene_ripper_mcp.tools.clips import filter_clips

    with tempfile.TemporaryDirectory() as tmpdir:
        video = Path(tmpdir) / "video.mp4"
        video.write_bytes(b"fake")
        project_path = Path(tmpdir) / "p.sceneripper"
        project = Project.new(name="Shots")
        project.add_source(
            Source(id="s", file_path=video, duration_seconds=10.0, fps=30.0, width=1920, height=1080)
        )
        project.add_clips([
            Clip(id="wide", source_id="s", start_frame=0, end_frame=30, shot_type="wide"),
            Clip(id="legacy-wide", source_id="s", start_frame=30, end_frame=60, shot_type="Wide Shot"),
            Clip(id="ecu", source_id="s", start_frame=60, end_frame=90, shot_type="ECU"),
            Clip(id="close", source_id="s", start_frame=90, end_frame=120, shot_type="close-up"),
        ])
        project.save(project_path)

        result = json.loads(
            await filter_clips(project_path=str(project_path), shot_type=wanted, ctx=AsyncMock())
        )
    assert result["success"], result
    assert {clip["id"] for clip in result["clips"]} == expected


@pytest.mark.parametrize("save", [False, True])
def test_cinematography_shot_types_stay_analyzed_after_load(tmp_path, monkeypatch, save):
    """Stored "wide"/"ECU" values must keep matching their analysis records.

    analysis_availability compares record.value with the clip's shot_type
    exactly, so loading must not rewrite the stored spelling.
    """
    from core.analysis_availability import compute_disabled_operations
    from core.project import Project
    from core.settings import Settings
    from tests.analysis_fixtures import verify_clip_analysis

    # Cloud tier: completion of a local tier depends on probing its runtime.
    settings = Settings(cinematography_tier="cloud", cinematography_model="gpt-test", cinematography_input_mode="frame")
    monkeypatch.setattr("core.settings.load_settings", lambda: settings)
    clip = _clip("analyzed", None)
    source = verify_clip_analysis(clip, tmp_path, shots=True, cinematography=True)
    assert clip.shot_type == "wide"
    if save:
        project = Project(sources=[source], clips=[clip])
        assert project.save(tmp_path / "project.json")
        project = Project.load(tmp_path / "project.json")
        clip, source = project.clips[0], project.sources[0]
        assert clip.shot_type == "wide"
    done = compute_disabled_operations([clip], {"shots", "cinematography"}, sources_by_id={source.id: source})
    assert {"shots", "cinematography"} <= set(done)
