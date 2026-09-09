"""Shared clip/frame item models (KTD13, plan U15).

Scenarios:
1. Edits update visible cards in both workspaces without rebuilding
   unrelated cards or resetting selection.
2. Filtered-out and offscreen selections keep their semantics across
   model updates.
3. Thumbnail results for removed items are ignored, and model mutations
   are refused off the owning thread.
"""

import threading
from pathlib import Path

import pytest
from PySide6.QtCore import QModelIndex, Qt

from core.project import Project
from models.clip import Source
from models.frame import Frame
from tests.conftest import make_test_clip
from ui.models import ClipLibraryModel, FrameLibraryModel


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _source(source_id: str = "src-1") -> Source:
    return Source(
        id=source_id, file_path=Path(f"/test/{source_id}.mp4"), duration_seconds=10.0,
        fps=30.0, width=1280, height=720,
    )


def _project(clip_count: int = 4) -> Project:
    project = Project.new()
    project.add_source(_source())
    project.add_clips([make_test_clip(f"c{i}") for i in range(clip_count)])
    return project


def _frame(frame_id: str) -> Frame:
    return Frame(id=frame_id, file_path=Path(f"/test/{frame_id}.png"), source_id="src-1")


class TestClipLibraryModel:
    def test_projects_project_clips_and_sources(self, qapp):
        model = ClipLibraryModel()
        model.set_project(_project(3))
        assert model.rowCount() == 3 and model.ids() == ["c0", "c1", "c2"]
        index = model.index(1)
        assert model.data(index, ClipLibraryModel.IdRole) == "c1"
        assert model.data(index, ClipLibraryModel.SourceRole).id == "src-1"
        assert model.data(index, Qt.ItemDataRole.DisplayRole)
        assert model.data(QModelIndex(), ClipLibraryModel.ClipRole) is None
        assert model.source_for("c2").id == "src-1" and model.source_for("nope") is None
        assert model.entries(["c2", "missing", "c0"]) == [
            (model.clip("c2"), model.source("src-1")), (model.clip("c0"), model.source("src-1")),
        ]

    def test_upsert_refresh_remove_emit_row_signals(self, qapp):
        model = ClipLibraryModel()
        model.set_project(_project(5))
        inserted, removed, changed = [], [], []
        model.rowsInserted.connect(lambda _p, first, last: inserted.append((first, last)))
        model.rowsRemoved.connect(lambda _p, first, last: removed.append((first, last)))
        model.dataChanged.connect(lambda tl, br, _r: changed.append((tl.row(), br.row())))
        source = _source()

        added, updated = model.upsert([(make_test_clip("c1"), source), (make_test_clip("c9"), source)])
        assert (added, updated) == (["c9"], ["c1"])
        assert inserted == [(5, 5)] and changed == [(1, 1)]

        changed.clear()
        assert model.refresh([make_test_clip("c3"), make_test_clip("c4"), make_test_clip("zzz")]) == ["c3", "c4"]
        assert changed == [(3, 4)]  # contiguous rows coalesce into one signal

        assert sorted(model.remove(["c0", "c1", "c3", "missing"])) == ["c0", "c1", "c3"]
        assert removed == [(3, 3), (0, 1)]  # runs removed from the end first
        assert model.ids() == ["c2", "c4", "c9"]
        assert model.row_for("c9") == 2 and "c0" not in model

    def test_sync_project_reconciles_without_reset(self, qapp):
        project = _project(3)
        model = ClipLibraryModel()
        model.set_project(project)
        resets = []
        model.modelReset.connect(lambda: resets.append(True))
        project.remove_clips(["c1"])  # model is not observing; reconcile explicitly
        project.add_clips([make_test_clip("c7")])
        added, removed = model.sync_project(project)
        assert (added, removed) == (["c7"], ["c1"])
        assert model.ids() == ["c0", "c2", "c7"] and resets == []

    def test_batch_duplicates_collapse_and_removal_keeps_project_order(self, qapp):
        model = ClipLibraryModel()
        source = _source()
        first, second = make_test_clip("dup"), make_test_clip("dup")
        added, updated = model.upsert([(first, source), (second, source), (make_test_clip("x"), source)])
        assert (added, updated) == (["dup", "x"], []) and model.rowCount() == 2
        assert model.clip("dup") is second
        model.upsert([(make_test_clip("y"), source), (make_test_clip("z"), source)])
        assert model.remove({"z", "dup", "x"}) == ["dup", "x", "z"]
        assert model.ids() == ["y"] and model.row_for("y") == 0

    def test_thumbnail_for_removed_clip_is_ignored(self, qapp, tmp_path):
        model = ClipLibraryModel()
        model.set_project(_project(2))
        landed = []
        model.thumbnail_changed.connect(lambda clip_id, path: landed.append((clip_id, path)))
        model.remove(["c1"])
        assert model.thumbnail_ready("c1", tmp_path / "late.jpg") is False
        assert model.thumbnail_ready("c0", tmp_path / "ok.jpg") is True
        assert landed == [("c0", tmp_path / "ok.jpg")]
        assert model.clip("c0").thumbnail_path == tmp_path / "ok.jpg"

    def test_mutations_refused_off_owner_thread(self, qapp):
        model = ClipLibraryModel()
        model.set_project(_project(1))
        errors = []

        def worker():
            for call in (
                lambda: model.upsert([(make_test_clip("x"), _source())]),
                lambda: model.remove(["c0"]),
                lambda: model.refresh([make_test_clip("c0")]),
                lambda: model.thumbnail_ready("c0", "/tmp/x.jpg"),
                lambda: model.set_project(None),
            ):
                try:
                    call()
                except RuntimeError as exc:
                    errors.append(str(exc))

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
        assert len(errors) == 5 and all("owning" in e for e in errors)
        assert model.ids() == ["c0"]  # nothing leaked through


class TestFrameLibraryModel:
    def test_append_refresh_remove(self, qapp):
        model = FrameLibraryModel()
        model.set_frames([_frame("f0"), _frame("f1"), _frame("f1")])
        assert model.ids() == ["f0", "f1"]
        assert model.append([_frame("f1"), _frame("f2")]) == ["f2"]
        assert model.refresh([_frame("f9")]) == []
        assert model.remove(["f0", "f2"]) and model.ids() == ["f1"]
        index = model.index(0)
        assert model.data(index, FrameLibraryModel.IdRole) == "f1"
        assert model.data(index, FrameLibraryModel.AnalyzedRole) is False
        assert model.data(index, FrameLibraryModel.ThumbnailPathRole) is None
        assert model.frame_at(model.index(5)) is None

    def test_mutations_refused_off_owner_thread(self, qapp):
        model = FrameLibraryModel()
        model.set_frames([_frame("f0")])
        errors = []

        def worker():
            for call in (
                lambda: model.append([_frame("f1")]),
                lambda: model.refresh([_frame("f0")]),
                lambda: model.remove(["f0"]),
                lambda: model.set_frames([]),
                lambda: model.set_project(None),
                lambda: model.sync_project(None),
            ):
                try:
                    call()
                except RuntimeError as exc:
                    errors.append(str(exc))

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
        assert len(errors) == 6 and all("owning" in e for e in errors)
        assert model.ids() == ["f0"]

    def test_batch_duplicates_and_removal_order(self, qapp):
        model = FrameLibraryModel()
        first, second = _frame("dup"), _frame("dup")
        assert model.append([first, second, _frame("x")]) == ["dup", "x"]
        assert model.rowCount() == 2 and model.frame("dup") is second
        model.append([_frame("y"), _frame("z")])
        assert model.remove({"z", "dup", "x"}) == ["dup", "x", "z"]  # project order, not set order
        assert model.get_frame(model.index(0)).id == "y"

    def test_sync_project_keeps_surviving_rows(self, qapp):
        project = _project(0)
        project.add_frames([_frame("f0"), _frame("f1"), _frame("f2")])
        model = FrameLibraryModel()
        model.set_project(project)
        resets, removed_rows = [], []
        model.modelReset.connect(lambda: resets.append(True))
        model.rowsRemoved.connect(lambda _p, first, last: removed_rows.append((first, last)))
        project.remove_frames(["f1"])
        project.add_frames([_frame("f3")])
        assert model.sync_project(project) == (["f3"], ["f1"])
        assert model.ids() == ["f0", "f2", "f3"] and resets == [] and removed_rows == [(1, 1)]


class TestAdapterOwnsModels:
    def test_models_follow_project_events_before_signals(self, qapp):
        from ui.project_adapter import ProjectSignalAdapter

        project = _project(2)
        adapter = ProjectSignalAdapter(project)
        seen = []
        adapter.clips_added.connect(lambda clips: seen.append(("added", adapter.clip_model.ids())))
        adapter.clips_removed.connect(lambda clips: seen.append(("removed", adapter.clip_model.ids())))
        adapter.frames_added.connect(lambda frames: seen.append(("frames", adapter.frame_model.ids())))

        assert adapter.clip_model.ids() == ["c0", "c1"]
        project.add_clips([make_test_clip("c2")])
        project.remove_clips(["c0"])
        project.add_frames([_frame("f0")])
        assert seen == [
            ("added", ["c0", "c1", "c2"]), ("removed", ["c1", "c2"]), ("frames", ["f0"]),
        ]

        replacement = _project(1)
        adapter.set_project(replacement)
        assert adapter.clip_model.ids() == ["c0"] and adapter.frame_model.ids() == []
        adapter.disconnect_from_project()
        assert len(adapter.clip_model) == 0

    def test_redetection_replaces_rows_and_signals_removal(self, qapp):
        from ui.project_adapter import ProjectSignalAdapter

        project = _project(3)
        adapter = ProjectSignalAdapter(project)
        removed = []
        adapter.clips_removed.connect(lambda clips: removed.append([c.id for c in clips]))
        project.replace_source_clips("src-1", [make_test_clip("n0"), make_test_clip("n1")])
        assert removed == [["c0", "c1", "c2"]]
        assert adapter.clip_model.ids() == ["n0", "n1"]
        assert adapter.clip_model.thumbnail_ready("c0", "/tmp/late.jpg") is False

    def test_signal_still_fires_when_model_mirroring_fails(self, qapp, monkeypatch):
        from ui.project_adapter import ProjectSignalAdapter

        project = _project(1)
        adapter = ProjectSignalAdapter(project)
        heard = []
        adapter.clips_added.connect(lambda clips: heard.append([c.id for c in clips]))
        monkeypatch.setattr(adapter.clip_model, "upsert", lambda pairs: (_ for _ in ()).throw(RuntimeError("off thread")))
        project.add_clips([make_test_clip("c9")])
        assert heard == [["c9"]]  # views are told even though the model lagged

    def test_source_removal_and_undo_keep_model_in_sync(self, qapp):
        from ui.project_adapter import ProjectSignalAdapter

        project = _project(2)
        adapter = ProjectSignalAdapter(project)
        project.remove_source("src-1")
        assert adapter.clip_model.ids() == []
        project.session.undo()
        assert adapter.clip_model.ids() == ["c0", "c1"]
        assert adapter.clip_model.source_for("c1").id == "src-1"


class TestBrowsersShareTheModel:
    """Scenarios 1 and 2: two workspaces, one model, separate selection."""

    @pytest.fixture
    def workspaces(self, qapp, monkeypatch):
        from ui.clip_browser import ClipBrowser
        from ui.project_adapter import ProjectSignalAdapter

        monkeypatch.setattr("ui.clip_browser.ClipThumbnail.isVisible", lambda self: True)
        project = _project(6)
        adapter = ProjectSignalAdapter(project)
        cut, analyze = ClipBrowser(), ClipBrowser()
        for browser in (cut, analyze):
            browser.attach_model(adapter.clip_model)
            browser.resize(900, 600)
        source = project.sources_by_id["src-1"]
        cut.set_virtual_clips([(clip, source) for clip in project.clips])
        analyze.add_clips([(clip, source) for clip in project.clips[:3]])
        qapp.processEvents()
        return project, adapter, cut, analyze

    def test_browsers_read_sources_from_shared_model_and_keep_membership(self, workspaces):
        project, adapter, cut, analyze = workspaces
        assert cut.library_model is adapter.clip_model is analyze.library_model
        assert cut.get_total_clip_count() == 6 and analyze.get_total_clip_count() == 3
        assert cut.get_source_for_clip("c5").id == "src-1"
        assert analyze.get_source_for_clip("c5").id == "src-1"  # library lookup, not membership
        assert "c5" not in analyze._virtual_id_set and "c5" not in analyze._thumbnail_by_id

    def test_shared_update_emits_one_data_changed_per_clip(self, workspaces):
        project, adapter, cut, analyze = workspaces
        emitted = []
        adapter.clip_model.dataChanged.connect(lambda tl, br, _r: emitted.append((tl.row(), br.row())))
        clip = project.clips_by_id["c1"]
        project.update_clips([clip])
        cut.update_clips([clip], preserve_layout=True)
        analyze.update_clips([clip], preserve_layout=True)
        assert emitted == [(1, 1)]

    def test_remove_clips_for_source_drops_ids_already_gone_from_model(self, qapp, workspaces):
        project, adapter, cut, analyze = workspaces
        project.replace_source_clips("src-1", [make_test_clip("n0")])  # rows c0..c5 leave the model
        cut.remove_clips_for_source("src-1")
        assert cut.get_total_clip_count() == 0 and cut._virtual_ids == []

    def test_edit_refreshes_cards_in_both_workspaces_without_resetting_selection(self, workspaces):
        project, adapter, cut, analyze = workspaces
        cut.selected_clips = {"c0", "c4"}
        analyze.selected_clips = {"c2"}
        cut_cards_before = {cid: id(t) for cid, t in cut._thumbnail_by_id.items()}
        analyze_cards_before = {cid: id(t) for cid, t in analyze._thumbnail_by_id.items()}

        clip = project.clips_by_id["c2"]
        clip.shot_type = "close-up"
        project.update_clips([clip])
        adapter.clips_updated.emit([clip])  # what MainWindow forwards
        cut.update_clips([clip], preserve_layout=True)
        analyze.update_clips([clip], preserve_layout=True)

        assert cut.selected_clips == {"c0", "c4"} and analyze.selected_clips == {"c2"}
        assert {cid: id(t) for cid, t in cut._thumbnail_by_id.items()} == cut_cards_before
        assert {cid: id(t) for cid, t in analyze._thumbnail_by_id.items()} == analyze_cards_before
        assert cut._thumbnail_by_id["c2"].clip.shot_type == "close-up"
        assert analyze._thumbnail_by_id["c2"].clip.shot_type == "close-up"

    def test_removal_drops_membership_but_not_other_workspace_selection(self, workspaces):
        project, adapter, cut, analyze = workspaces
        cut.selected_clips = {"c1", "c5"}
        analyze.selected_clips = {"c1"}
        project.remove_clips(["c1"])
        cut.remove_clips_by_ids(["c1"])
        analyze.remove_clips_by_ids(["c1"])
        assert "c1" not in adapter.clip_model
        assert cut.selected_clips == {"c5"} and analyze.selected_clips == set()
        assert cut.get_total_clip_count() == 5 and analyze.get_total_clip_count() == 2
        # A workspace clearing its view never removes rows from the shared model.
        analyze.clear()
        assert adapter.clip_model.ids() == ["c0", "c2", "c3", "c4", "c5"]

    def test_filtered_out_and_offscreen_selection_survive_updates(self, qapp, workspaces):
        project, adapter, cut, analyze = workspaces
        cut.selected_clips = {"c0", "c5"}
        cut._filter_state.shot_type = {"wide"}  # nothing matches: all cards filtered out
        qapp.processEvents()
        assert cut.get_visible_clip_count() == 0
        assert cut.selected_clips == {"c0", "c5"}
        clip = project.clips_by_id["c3"]
        project.update_clips([clip])
        cut.update_clips([clip])
        assert cut.selected_clips == {"c0", "c5"}
        cut._filter_state.shot_type = set()
        qapp.processEvents()
        assert {c.id for c in cut.get_selected_clips()} == {"c0", "c5"}

    def test_private_model_still_backs_standalone_browsers(self, qapp):
        from ui.clip_browser import ClipBrowser

        browser = ClipBrowser()
        source = _source("solo")
        browser.add_clip(make_test_clip("a", source_id="solo"), source)
        assert browser.get_source_for_clip("a") is source
        browser.remove_clips_by_ids(["a"])
        assert browser.get_source_for_clip("a") is None


class TestMainWindowThumbnailRouting:
    def test_late_thumbnail_for_removed_clip_never_reaches_the_cut_tab(self, qapp):
        from types import SimpleNamespace
        from unittest.mock import Mock

        from ui.main_window import MainWindow
        from ui.project_adapter import ProjectSignalAdapter

        project = _project(2)
        adapter = ProjectSignalAdapter(project)
        window = SimpleNamespace(
            _project_adapter=adapter, clips_by_id=project.clips_by_id,
            sources_by_id=project.sources_by_id, cut_tab=Mock(), project=project,
        )
        project.remove_clips(["c1"])
        MainWindow._on_thumbnail_ready(window, "c1", "/tmp/late.jpg")
        window.cut_tab.add_clip.assert_not_called()
        MainWindow._on_thumbnail_ready(window, "c0", "/tmp/ok.jpg")
        window.cut_tab.add_clip.assert_called_once()
        assert adapter.clip_model.clip("c0").thumbnail_path == Path("/tmp/ok.jpg")

    def test_project_load_thumbnails_go_through_the_model_too(self, qapp):
        from types import SimpleNamespace
        from unittest.mock import Mock

        from ui.main_window import MainWindow
        from ui.project_adapter import ProjectSignalAdapter

        project = _project(1)
        adapter = ProjectSignalAdapter(project)
        window = SimpleNamespace(
            _project_adapter=adapter, clips_by_id=project.clips_by_id, cut_tab=Mock(), analyze_tab=Mock(),
        )
        MainWindow._on_project_thumbnail_ready(window, "missing", "/tmp/x.jpg")
        window.cut_tab.update_clip_thumbnail.assert_not_called()
        MainWindow._on_project_thumbnail_ready(window, "c0", "/tmp/x.jpg")
        window.cut_tab.update_clip_thumbnail.assert_called_once()
        assert adapter.clip_model.clip("c0").thumbnail_path == Path("/tmp/x.jpg")


class TestFrameBrowserSharesTheModel:
    def test_frames_tab_refresh_never_resets_the_shared_model(self, qapp):
        from ui.project_adapter import ProjectSignalAdapter
        from ui.tabs.frames_tab import FramesTab

        project = _project(0)
        adapter = ProjectSignalAdapter(project)
        tab = FramesTab()
        tab.set_frame_model(adapter.frame_model)
        tab.set_project(project)
        tab.update_frame_browser()
        assert tab.state_stack.currentIndex() == tab.STATE_EMPTY
        project.add_frames([_frame("f0"), _frame("f1")])
        resets = []
        adapter.frame_model.modelReset.connect(lambda: resets.append(True))
        tab.update_frame_browser()
        assert tab.state_stack.currentIndex() == tab.STATE_FRAMES
        browser = tab.frame_browser
        browser._view.selectionModel().select(
            adapter.frame_model.index(0), browser._view.selectionModel().SelectionFlag.Select,
        )
        tab.update_frame_browser()
        browser.set_frames([])  # ignored on a shared model
        assert browser.get_selected_frame_ids() == ["f0"] and resets == []
        browser.clear()  # selection only
        assert browser.get_selected_frame_ids() == [] and adapter.frame_model.ids() == ["f0", "f1"]

    def test_undo_of_unrelated_source_removal_keeps_frame_selection(self, qapp):
        from ui.frame_browser import FrameBrowser
        from ui.project_adapter import ProjectSignalAdapter

        project = _project(1)
        project.add_source(_source("src-2"))
        project.add_frames([_frame("f0"), _frame("f1")])
        adapter = ProjectSignalAdapter(project)
        browser = FrameBrowser()
        browser.set_model(adapter.frame_model)
        browser._view.selectionModel().select(
            adapter.frame_model.index(1), browser._view.selectionModel().SelectionFlag.Select,
        )
        project.remove_source("src-2")
        project.session.undo()
        assert browser.get_selected_frame_ids() == ["f1"]

    def test_view_renders_shared_model_and_keeps_selection_on_refresh(self, qapp):
        from ui.frame_browser import FrameBrowser
        from ui.project_adapter import ProjectSignalAdapter

        project = _project(0)
        adapter = ProjectSignalAdapter(project)
        browser = FrameBrowser()
        browser.set_model(adapter.frame_model)
        project.add_frames([_frame("f0"), _frame("f1"), _frame("f2")])
        assert browser.frame_count() == 3 and browser.uses_shared_model()
        browser._view.selectionModel().select(
            adapter.frame_model.index(1), browser._view.selectionModel().SelectionFlag.Select,
        )
        assert browser.get_selected_frame_ids() == ["f1"]
        project.update_frame_metadata("f1", notes="hello")
        assert browser.get_selected_frame_ids() == ["f1"]
        browser.set_frames(project.frames)  # identical contents: no reset
        assert browser.get_selected_frame_ids() == ["f1"]
        project.remove_frames(["f0"])
        assert browser.get_selected_frame_ids() == ["f1"] and browser.frame_count() == 2


class TestModelScale:
    """Scenario 4 guard: model operations stay cheap at 10k clips.

    The full browser benchmark (population, scroll, filter, memory) lives in
    ``scripts/library_model_benchmark.py``; its recorded numbers and budgets
    are in ``docs/architecture/library-models.md``. This test bounds only the
    model layer so a regression there fails fast without widget timing noise.
    """

    def test_ten_thousand_clip_model_operations_are_fast(self, qapp):
        import time

        model = ClipLibraryModel()
        source = _source()
        clips = [make_test_clip(f"c{i}") for i in range(10_000)]
        start = time.perf_counter()
        model.upsert((clip, source) for clip in clips)
        model.refresh(clips[:500])
        model.entries(model.ids())
        model.remove([f"c{i}" for i in range(0, 10_000, 7)])
        elapsed = time.perf_counter() - start
        assert len(model) == 10_000 - len(range(0, 10_000, 7))
        assert elapsed < 2.0, f"model operations took {elapsed:.2f}s at 10k clips"

    def test_ten_thousand_frame_model_operations_are_fast(self, qapp):
        import time

        model = FrameLibraryModel()
        frames = [_frame(f"f{i}") for i in range(10_000)]
        start = time.perf_counter()
        model.append(frames)
        model.refresh(frames[::3])
        model.remove([f"f{i}" for i in range(0, 10_000, 7)])
        elapsed = time.perf_counter() - start
        assert len(model) == 10_000 - len(range(0, 10_000, 7))
        assert elapsed < 2.0, f"frame model operations took {elapsed:.2f}s at 10k frames"
