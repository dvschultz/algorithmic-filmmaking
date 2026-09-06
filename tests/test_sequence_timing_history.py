"""Shared timing commands are atomic and do not expose drag previews to saves."""

from dataclasses import replace

import pytest

from core.chat_tools import update_sequence_clip
from core.commands.sequence_clips import Placement
from models.sequence import Track
from tests.test_clip_disabled import _make_project_with_clips


def project_with_sequence():
    project = _make_project_with_clips()
    project.add_to_sequence(["c0", "c1", "c2"])
    project.mark_clean()
    return project


def test_reorder_undo_restores_order_gaps_and_clean_checkpoint():
    project = project_with_sequence()
    entries = project.sequence.get_all_clips()
    entries[1].start_frame += 20
    original = [c.start_frame for c in entries]
    assert project.reorder_sequence([entries[2].id, entries[0].id])
    assert project.sequence.get_all_clips() == [entries[2], entries[0], entries[1]]
    project.session.undo()
    assert project.sequence.get_all_clips() == entries
    assert [c.start_frame for c in entries] == original
    assert not project.is_dirty
    project.session.redo()
    assert project.sequence.get_all_clips()[0] is entries[2]


def test_duplicate_reorder_ids_cannot_duplicate_timeline_objects():
    project = project_with_sequence()
    entries = project.sequence.get_all_clips()
    assert not project.reorder_sequence([entries[0].id, entries[0].id])
    assert project.sequence.get_all_clips() == entries
    assert not project.is_dirty


def test_invalid_combined_edit_is_atomic():
    project = project_with_sequence()
    clip = project.sequence.get_all_clips()[0]
    before = clip.to_dict()
    result = update_sequence_clip(project, clip.id, in_point=10, out_point=5)
    assert not result["success"]
    assert clip.to_dict() == before and not project.is_dirty
    assert not update_sequence_clip(project, clip.id, in_point=clip.out_point)[
        "success"
    ]
    assert clip.to_dict() == before


def test_trim_move_track_and_transform_are_one_reversible_edit():
    project = project_with_sequence()
    project.sequence.tracks.append(Track())
    clip = project.sequence.get_all_clips()[0]
    clip.prerendered_path = "/tmp/rendered.mp4"
    before = clip.to_dict()
    assert update_sequence_clip(
        project,
        clip.id,
        in_point=3,
        out_point=40,
        start_frame=12,
        track_index=1,
        hflip=True,
    )["success"]
    assert clip not in project.sequence.tracks[0].clips
    assert project.sequence.tracks[1].clips == [clip]
    assert clip.prerendered_path is None
    project.session.undo()
    assert clip.to_dict() == before and not project.is_dirty
    assert not project.sequence.tracks[1].clips
    project.session.redo()
    assert clip.hflip and clip.in_point == 3 and clip.track_index == 1


def test_noop_edit_and_reorder_do_not_create_history():
    project = project_with_sequence()
    clip = project.sequence.get_all_clips()[0]
    generation = project.mutation_generation
    project.update_sequence_clip(clip.id, in_point=clip.in_point)
    project.reorder_sequence([c.id for c in project.sequence.get_all_clips()])
    assert project.mutation_generation == generation and not project.is_dirty


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_drag_preview_commits_one_edit_without_mutating_live_model(qapp):
    from ui.timeline.timeline_widget import TimelineWidget

    project = project_with_sequence()
    timeline = TimelineWidget()
    timeline.scene.project = project
    timeline.scene.set_sequence(project.sequence)
    clip = project.sequence.get_all_clips()[0]
    item = timeline.scene._clip_items[clip.id]
    item._edit_source = clip
    item._edit_before = Placement.capture(clip)
    item._edit_project = project
    item._edit_sequence = project.sequence
    item.seq_clip = replace(clip, start_frame=33, in_point=4)
    assert clip.start_frame == 0 and clip.in_point == 0
    generation = project.mutation_generation
    item._commit_history_edit()
    assert clip.start_frame == 33 and clip.in_point == 4
    assert project.mutation_generation == generation + 1
    project.session.undo()
    assert clip.start_frame == 0 and clip.in_point == 0 and not project.is_dirty
    timeline.close()


@pytest.mark.parametrize("gesture", ["move", "left", "right", "frame_right"])
def test_actual_mouse_gesture_keeps_save_snapshot_stable(qapp, gesture):
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtWidgets import QGraphicsSceneMouseEvent
    from ui.project_adapter import ProjectSignalAdapter
    from ui.timeline.timeline_widget import TimelineWidget

    project = project_with_sequence()
    if gesture == "frame_right":
        from models.frame import Frame

        project.add_frames([Frame(id="f")])
        project.add_frames_to_sequence(["f"], hold_frames=12)
        project.mark_clean()
    timeline = TimelineWidget()
    timeline.scene.project = project
    timeline.scene.set_sequence(project.sequence)
    adapter = ProjectSignalAdapter(project)
    adapter.sequence_changed.connect(
        lambda _: timeline.load_sequence(
            project.sequence, project.sources_by_id, project.clips
        )
    )
    timeline.sequence_changed.connect(project.mark_dirty)
    clip = project.sequence.get_all_clips()[-1 if gesture == "frame_right" else 0]
    before = clip.to_dict()
    item = timeline.scene._clip_items[clip.id]
    x = item.rect().width() / 2
    if gesture == "left":
        x = 1
    elif gesture in {"right", "frame_right"}:
        x = item.rect().width() - 1
    point = QPointF(x, 10)
    press = QGraphicsSceneMouseEvent(QEvent.GraphicsSceneMousePress)
    press.setButton(Qt.LeftButton)
    press.setButtons(Qt.LeftButton)
    press.setPos(point)
    press.setScenePos(point)
    item.mousePressEvent(press)
    move = QGraphicsSceneMouseEvent(QEvent.GraphicsSceneMouseMove)
    move.setScenePos(point + QPointF(50, 0))
    item.mouseMoveEvent(move)
    assert item.seq_clip is not clip
    assert clip.to_dict() == before
    saved = project.snapshot_for_save()["sequence"]
    assert next(c for c in saved.get_all_clips() if c.id == clip.id).to_dict() == before
    release = QGraphicsSceneMouseEvent(QEvent.GraphicsSceneMouseRelease)
    release.setButton(Qt.LeftButton)
    item.mouseReleaseEvent(release)
    assert clip.to_dict() != before
    project.session.undo()
    assert clip.to_dict() == before and not project.is_dirty
    timeline.close()


def test_frame_hold_trim_and_external_analysis_survive_history():
    from models.frame import Frame

    project = project_with_sequence()
    project.add_frames([Frame(id="frame")])
    project.add_frames_to_sequence(["frame"], hold_frames=12)
    entry = project.sequence.get_all_clips()[-1]
    project.mark_clean()
    project.update_sequence_clip(entry.id, hold_frames=7, start_frame=9)
    project.clips[0].dominant_colors = [(1, 2, 3)]
    project.update_clips([project.clips[0]])
    project.session.undo()
    assert entry.hold_frames == 12
    assert project.clips[0].dominant_colors == [(1, 2, 3)] and project.is_dirty


def test_noop_on_overlapping_entries_preserves_stable_order():
    project = project_with_sequence()
    entries = project.sequence.get_all_clips()
    entries[1].start_frame = entries[0].start_frame
    generation = project.mutation_generation
    project.update_sequence_clip(entries[0].id, start_frame=entries[0].start_frame)
    assert project.sequence.get_all_clips() == entries
    assert project.mutation_generation == generation
