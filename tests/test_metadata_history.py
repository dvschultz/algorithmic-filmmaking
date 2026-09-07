"""Editorial patches preserve inverse values without owning analysis results."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import pytest

from core.spine.frames import update_frame
from core.spine.metadata import edit_tags, update_clip
from models.frame import Frame
from tests.test_clip_disabled import _make_project_with_clips


def test_batch_tags_is_one_reversible_edit_and_noops_stay_clean():
    project = _make_project_with_clips()
    result = edit_tags(project, ["c0", "c1", "missing"], ["keep"])
    assert result["not_found"] == ["missing"]
    assert project.clips[0].tags == project.clips[1].tags == ["keep"]
    project.session.undo()
    assert project.clips[0].tags == project.clips[1].tags == []
    assert not project.is_dirty
    assert not project.session.can_undo
    project.session.redo()
    project.mark_clean()
    generation = project.mutation_generation
    edit_tags(project, ["c0", "c1"], ["keep"])
    assert not project.is_dirty
    assert project.mutation_generation == generation


def test_invalid_multi_field_edit_is_atomic():
    project = _make_project_with_clips()
    original = project.clips[0].name
    result = update_clip(project, "c0", name="changed", shot_type="invalid")
    assert not result["success"]
    assert project.clips[0].name == original
    assert not project.session.can_undo
    assert not project.is_dirty


def test_undo_metadata_preserves_new_analysis_and_dirty_state():
    project = _make_project_with_clips()
    project.update_clip_metadata("c0", notes="editorial")
    project.clips[0].description = "new analysis"
    project.update_clips([project.clips[0]])
    project.session.undo()
    assert project.clips[0].notes == ""
    assert project.clips[0].description == "new analysis"
    assert project.is_dirty


def test_conflicting_batch_undo_does_not_partially_restore():
    project = _make_project_with_clips()
    edit_tags(project, ["c0", "c1"], ["keep"])
    project.clips[1].tags.append("external")
    project.update_clips([project.clips[1]])
    with pytest.raises(ValueError, match="changed"):
        project.session.undo()
    assert project.clips[0].tags == ["keep"]
    assert project.session.can_undo


def test_mutable_values_are_detached_on_capture_and_redo():
    project = _make_project_with_clips()
    tags = ["keep"]
    project.update_clip_metadata("c0", tags=tags)
    tags.append("not saved")
    assert project.clips[0].tags == ["keep"]
    project.session.undo()
    project.session.redo()
    assert project.clips[0].tags == ["keep"]


def test_frame_fields_publish_one_event_and_undo_together():
    project = _make_project_with_clips()
    frame = Frame(id="frame")
    project.add_frames([frame])
    project.mark_clean()
    events = []
    project.add_observer(lambda event, data: events.append(event))
    assert update_frame(
        project, frame.id, tags=["keep"], notes="note", shot_type="Wide Shot"
    )["success"]
    assert events == ["frames_updated"]
    project.session.undo()
    assert (frame.tags, frame.notes, frame.shot_type) == ([], "", None)
    assert not project.is_dirty


def test_source_and_project_metadata_undo_notifications():
    project = _make_project_with_clips()
    events = []
    project.add_observer(lambda event, data: events.append((event, data)))
    name = project.metadata.name
    project.rename("Editorial")
    project.update_source_metadata("s1", fps=24.0)
    assert events[-1] == ("source_updated", project.sources[0])
    project.session.undo()
    project.session.undo()
    assert project.sources[0].fps == 30.0
    assert project.metadata.name == name
    assert events[-1][0] == "project_metadata_changed"
    assert not project.is_dirty


@pytest.mark.parametrize("fps", [0, -1, float("inf"), float("nan"), True])
def test_invalid_fps_rejected(fps):
    project = _make_project_with_clips()
    with pytest.raises(ValueError):
        project.update_source_metadata("s1", fps=fps)
    assert project.sources[0].fps == 30.0
    assert not project.session.can_undo


def test_metadata_enforces_owner_thread():
    project = _make_project_with_clips()
    project.session
    with ThreadPoolExecutor() as pool:
        with pytest.raises(RuntimeError, match="owner thread"):
            pool.submit(
                project.update_clip_metadata, "c0", notes="wrong thread"
            ).result()
    assert project.clips[0].notes == ""


def test_agent_metadata_commands_share_history():
    from core.chat_tools import add_note, add_tags, update_clip_transcript
    from core.transcription_models import TranscriptSegment

    project = _make_project_with_clips()
    clip = project.clips[0]
    clip.transcript = [TranscriptSegment(0, 1, "before")]
    assert add_note(project, "c0", "note")["success"]
    assert add_tags(project, ["c0", "c1"], ["keep"])["success"]
    assert update_clip_transcript(project, "c0", 0, "after")["new_text"] == "after"
    project.session.undo()
    assert clip.transcript[0].text == "before"
    project.session.undo()
    assert project.clips[0].tags == project.clips[1].tags == []
    project.session.undo()
    assert clip.notes == ""
    assert not project.is_dirty


def test_qt_adapter_forwards_metadata_undo_notifications():
    from ui.project_adapter import ProjectSignalAdapter

    project = _make_project_with_clips()
    frame = Frame(id="frame")
    project.add_frames([frame])
    adapter = ProjectSignalAdapter(project)
    events = []
    adapter.frames_updated.connect(lambda frames: events.append(frames[0].notes))
    adapter.project_metadata_changed.connect(lambda: events.append(project.metadata.name))
    name = project.metadata.name
    project.update_frame_metadata("frame", notes="edit")
    project.rename("renamed")
    project.session.undo()
    project.session.undo()
    assert events == ["edit", "renamed", name, ""]
    adapter.disconnect_from_project()


def test_sidebar_transcript_edit_and_undo_refresh():
    from PySide6.QtWidgets import QApplication
    from core.transcription import TranscriptSegment
    from ui.clip_details_sidebar import ClipDetailsSidebar

    app = QApplication.instance() or QApplication([])
    project = _make_project_with_clips()
    clip = project.clips[0]
    clip.transcript = [TranscriptSegment(start_time=0, end_time=1, text="before")]
    sidebar = ClipDetailsSidebar()
    sidebar.video_player._setup_player = lambda: None
    sidebar.video_player._player_ready = True
    sidebar.metadata_editor = lambda target, fields: project.update_clip_metadata(
        target.id, **fields
    )
    sidebar.show_clip(clip, project.sources[0])
    assert sidebar.transcript_edit._segments[0] is not clip.transcript[0]
    after = deepcopy(sidebar.transcript_edit._segments)
    after[0].text = "after"
    sidebar._on_transcript_changed(after)
    assert clip.transcript[0].text == "after"
    project.session.undo()
    sidebar.video_player.load_video = lambda *args, **kwargs: pytest.fail(
        "metadata refresh reloaded playback"
    )
    sidebar.refresh_editor_fields(clip)
    assert sidebar.transcript_edit._segments[0].text == "before"
    assert not project.is_dirty
    sidebar.name_edit._start_editing()
    sidebar.name_edit.edit.setText("draft still being typed")
    clip.description = "analysis finished"
    project.update_clips([clip])
    sidebar.refresh_editor_fields(clip)
    assert sidebar.name_edit.edit.text() == "draft still being typed"
    sidebar.name_edit._cancel_editing()
    sidebar.close()
    app.processEvents()


@pytest.mark.parametrize(
    "field,value",
    [
        ("description", 123),
        ("transcript", ["invalid"]),
        ("cinematography", {}),
        ("description_frames", "invalid"),
        ("custom_queries", "invalid"),
    ],
)
def test_reject_unserializable_editorial_values(field, value):
    project = _make_project_with_clips()
    with pytest.raises(ValueError):
        project.update_clip_metadata("c0", **{field: value})
    assert not project.is_dirty
