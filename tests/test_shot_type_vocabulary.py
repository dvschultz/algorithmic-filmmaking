"""Shot types share one canonical vocabulary across agent, MCP, and GUI seams.

Analysis stores lowercase canonical values ("wide shot"), the GUI filters by
display names ("Wide"), and older code used title-case labels ("Wide Shot").
Every entry point must accept all three and store or compare canonically.
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from PySide6.QtWidgets import QApplication

from core.spine.frames import update_frame
from core.spine.metadata import update_clip
from models.clip import Clip, Source
from models.frame import Frame
from tests.test_clip_disabled import _make_project_with_clips

app = QApplication.instance() or QApplication(sys.argv)


# --- constants -------------------------------------------------------------


def test_constants_match_analysis_vocabulary():
    from core.analysis_model_identity import SHOT_TYPES as ANALYSIS_SHOT_TYPES
    from core.constants import SHOT_TYPES

    assert SHOT_TYPES == ANALYSIS_SHOT_TYPES


def test_cloud_classifier_writes_canonical_values():
    from core.analysis.shots_cloud import SHOT_TYPE_SIMPLIFIED
    from core.constants import SHOT_TYPES

    assert set(SHOT_TYPE_SIMPLIFIED.values()) == set(SHOT_TYPES)


@pytest.mark.parametrize(
    "value, expected",
    [
        ("wide shot", "wide shot"),
        ("Wide", "wide shot"),
        ("Wide Shot", "wide shot"),
        ("  WIDE SHOT ", "wide shot"),
        ("Full", "full shot"),
        ("Medium Shot", "medium shot"),
        ("Close-up", "close-up"),
        ("extreme close-up", "extreme close-up"),
        ("Extreme CU", "extreme close-up"),
        # cinematography get_simple_shot_type() output
        ("wide", "wide shot"),
        ("full", "full shot"),
        ("medium", "medium shot"),
        ("ECU", "extreme close-up"),
        ("ecu", "extreme close-up"),
        # cinematography shot sizes that name exactly one canonical size
        ("ELS", "wide shot"),
        ("LS", "full shot"),
        ("MS", "medium shot"),
        ("CU", "close-up"),
        # in-between sizes and inserts have no single canonical equivalent
        ("VLS", None),
        ("MLS", None),
        ("MCU", None),
        ("BCU", None),
        ("Insert", None),
        ("bogus", None),
        ("", None),
    ],
)
def test_normalize_shot_type(value, expected):
    from core.constants import normalize_shot_type

    assert normalize_shot_type(value) == expected


def test_display_name_handles_legacy_stored_values():
    from core.analysis.shots import get_display_name

    assert get_display_name("wide shot") == "Wide"
    assert get_display_name("Wide Shot") == "Wide"
    assert get_display_name("Extreme CU") == "Extreme CU"


# --- spine update_clip / update_frame -------------------------------------


@pytest.mark.parametrize(
    "value, stored",
    [
        ("close-up", "close-up"),
        ("wide shot", "wide shot"),
        ("Wide Shot", "wide shot"),
        ("Extreme CU", "extreme close-up"),
    ],
)
def test_update_clip_accepts_and_stores_canonical(value, stored):
    project = _make_project_with_clips()
    result = update_clip(project, "c0", shot_type=value)
    assert result["success"], result
    assert project.clips_by_id["c0"].shot_type == stored


def test_update_clip_rejects_unknown_and_lists_canonical_values():
    project = _make_project_with_clips()
    result = update_clip(project, "c0", shot_type="bogus")
    assert not result["success"]
    assert "wide shot" in result["error"]
    assert "Wide Shot" not in result["error"]


def test_update_frame_accepts_and_stores_canonical():
    project = _make_project_with_clips()
    frame = Frame(id="frame")
    project.add_frames([frame])
    for value in ("close-up", "Wide"):
        assert update_frame(project, frame.id, shot_type=value)["success"]
    assert frame.shot_type == "wide shot"


def test_metadata_command_stores_canonical_for_legacy_value():
    project = _make_project_with_clips()
    project.update_clip_metadata("c0", shot_type="Medium Shot")
    assert project.clips_by_id["c0"].shot_type == "medium shot"


# --- agent apply_filters -> GUI clip browser --------------------------------


def _browser_window():
    from ui.clip_browser import ClipBrowser

    browser = ClipBrowser()
    window = Mock()
    window._gui_state = None
    window.get_active_clip_browser.return_value = browser
    return window, browser


@pytest.mark.parametrize("value", ["wide shot", "Wide", "Wide Shot"])
def test_apply_filters_sets_gui_display_name(value):
    from core.chat_tools import apply_filters

    window, browser = _browser_window()
    result = apply_filters(window, shot_type=value)
    assert result["success"], result
    assert browser._filter_state.shot_type == {"Wide"}


def test_apply_filters_list_input_and_visible_match():
    from core.chat_tools import apply_filters

    window, browser = _browser_window()
    source = Source(id="s", file_path=Path("/tmp/s.mp4"), fps=30.0, width=1920, height=1080)
    for clip_id, shot in [("a", "wide shot"), ("b", "extreme close-up"), ("c", "close-up")]:
        browser.add_clip(
            Clip(id=clip_id, source_id="s", start_frame=0, end_frame=30, shot_type=shot),
            source,
        )
    result = apply_filters(window, shot_type=["wide shot", "Extreme CU"])
    assert result["success"], result
    assert browser._filter_state.shot_type == {"Wide", "Extreme CU"}
    assert browser.get_visible_clip_count() == 2


def test_apply_filters_rejects_unknown_with_canonical_values():
    from core.chat_tools import apply_filters

    window, _browser = _browser_window()
    result = apply_filters(window, shot_type="bogus")
    assert not result["success"]
    assert "wide shot" in result["error"]


# --- clip filters (agent spine + MCP tool) ----------------------------------


def _project_with_mixed_shot_values():
    project = _make_project_with_clips()
    project.clips_by_id["c0"].shot_type = "wide shot"
    project.clips_by_id["c1"].shot_type = "Wide Shot"  # legacy stored value
    return project


@pytest.mark.parametrize("value", ["Wide Shot", "WIDE SHOT", "wide shot", "Wide"])
def test_spine_filter_clips_normalizes_both_sides(value):
    from core.spine.clips import filter_clips

    project = _project_with_mixed_shot_values()
    ids = {item["id"] for item in filter_clips(project, shot_type=value)}
    assert ids == {"c0", "c1"}


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["Wide Shot", "WIDE SHOT"])
async def test_mcp_filter_clips_normalizes_both_sides(value):
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
            Clip(id="a", source_id="s", start_frame=0, end_frame=30, shot_type="wide shot"),
            Clip(id="b", source_id="s", start_frame=30, end_frame=60, shot_type="Wide Shot"),
            Clip(id="c", source_id="s", start_frame=60, end_frame=90, shot_type="close-up"),
        ])
        project.save(project_path)

        result = json.loads(
            await filter_clips(project_path=str(project_path), shot_type=value, ctx=AsyncMock())
        )
    assert result["success"], result
    assert {clip["id"] for clip in result["clips"]} == {"a", "b"}
