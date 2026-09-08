"""Tests for SaveProjectWorker — concurrency safety and signal contract."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from core.project import Project
from models.clip import Source
from tests.conftest import make_test_clip


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _make_project_with_clip(tmp_dir: Path) -> Project:
    """Build a small project ready for save."""
    project = Project.new(name="save-worker-test")
    source = Source(
        id="src-1",
        file_path=tmp_dir / "video.mp4",
        duration_seconds=10.0,
        fps=30.0,
        width=640,
        height=480,
    )
    project.add_source(source)
    project.add_clips([make_test_clip("clip-1", source_id=source.id)])
    return project


class _ActionStub:
    def __init__(self):
        self.enabled = None

    def setEnabled(self, enabled: bool):
        self.enabled = enabled


class _StatusBarStub:
    def __init__(self):
        self.messages = []

    def showMessage(self, message: str, *_args):
        self.messages.append(message)


class _SaveCompletionWindowStub:
    def __init__(self, project: Project, context: dict | None):
        self.project = project
        self._save_project_context = context
        self.save_worker = SimpleNamespace(isRunning=lambda: False)
        self.save_project_action = _ActionStub()
        self.save_project_as_action = _ActionStub()
        self.status_bar = _StatusBarStub()
        self.recent_projects = []
        self.window_title_updates = 0

    def _add_recent_project(self, filepath: Path):
        self.recent_projects.append(filepath)

    def _update_window_title(self):
        self.window_title_updates += 1


def test_snapshot_for_save_is_independent_of_live_project(tmp_path):
    """Snapshot must not see post-snapshot mutations on the live project."""
    project = _make_project_with_clip(tmp_path)
    snapshot = project.snapshot_for_save()

    # Mutate the live project after snapshotting.
    project.add_clips([make_test_clip("clip-2", source_id="src-1")])
    project._clips[0].shot_type = "wide"

    snapshot_clip_ids = {c.id for c in snapshot["clips"]}
    assert snapshot_clip_ids == {"clip-1"}
    assert snapshot["clips"][0].shot_type is None


def test_queued_save_retains_artifacts_after_new_project(qapp, tmp_path, monkeypatch):
    from core.artifacts import ArtifactStore
    from models.analysis_record import AnalysisRecord
    from tests.test_analysis_records import identity
    from ui.main_window import SaveProjectWorker

    root = tmp_path / "artifacts"
    monkeypatch.setattr("core.paths.get_artifact_store_dir", lambda: root)
    store = ArtifactStore(root)
    project = _make_project_with_clip(tmp_path)
    with store.pin() as producer:
        ref = store.put_bytes(b"embedding", pin=producer)
        project.record_analysis("clip", "clip-1", "embeddings", AnalysisRecord.success(identity(operation="embeddings"), artifact=ref))
    worker = SaveProjectWorker(project.snapshot_for_save(), tmp_path / "queued.json")
    project.clear()
    assert store.collect() == []
    worker.run()
    assert store.collect() == []
    assert store.read_bytes(ref) == b"embedding"


@pytest.mark.parametrize("asynchronous", [False, True])
def test_desktop_save_preserves_manual_project_name_and_undo(
    qapp, tmp_path, monkeypatch, asynchronous
):
    import json

    from ui.main_window import MainWindow, SaveProjectWorker

    project = _make_project_with_clip(tmp_path)
    project.mark_clean()
    original_name = project.metadata.name
    project.rename("Editorial title")
    window = _SaveCompletionWindowStub(project, None)
    window.save_worker = None
    window.sequence_tab = SimpleNamespace(_persist_current_sequence=lambda: None)
    window.analyze_tab = SimpleNamespace(get_clip_ids=lambda: [])
    window._on_project_save_finished = (
        lambda *args: MainWindow._on_project_save_finished(window, *args)
    )
    monkeypatch.setattr(SaveProjectWorker, "start", lambda worker: worker.run())
    target = tmp_path / "different-filename.sceneripper"

    MainWindow._save_project_to_file(window, target, asynchronous=asynchronous)

    assert project.metadata.name == "Editorial title"
    assert json.loads(target.read_text())["project_name"] == "Editorial title"
    assert not project.is_dirty
    project.session.undo()
    assert project.metadata.name == original_name
    assert project.is_dirty
    project.session.redo()
    assert project.metadata.name == "Editorial title"
    assert not project.is_dirty


def test_project_mutation_generation_increments_even_when_already_dirty(tmp_path):
    """Save completion needs a monotonic mutation generation, not just dirty bool."""
    project = _make_project_with_clip(tmp_path)
    generation = project.mutation_generation

    project.mark_dirty()
    project.mark_dirty()

    assert project.is_dirty
    assert project.mutation_generation == generation + 2


def test_save_completion_marks_clean_for_current_unchanged_project(tmp_path):
    from ui.main_window import MainWindow

    project = _make_project_with_clip(tmp_path)
    target = tmp_path / "out.sceneripper"
    context = {
        "project": project,
        "mutation_generation": project.mutation_generation,
        "filepath": target,
    }
    window = _SaveCompletionWindowStub(project, context)

    MainWindow._on_project_save_finished(window, True, str(target), "")

    assert project.path == target
    assert not project.is_dirty
    assert window.recent_projects == [target]
    assert window._save_project_context is None
    assert window.save_worker is None


def test_stale_save_completion_does_not_mark_mutated_project_clean(tmp_path):
    from ui.main_window import MainWindow

    project = _make_project_with_clip(tmp_path)
    target = tmp_path / "out.sceneripper"
    context = {
        "project": project,
        "mutation_generation": project.mutation_generation,
        "filepath": target,
    }
    window = _SaveCompletionWindowStub(project, context)

    project.add_clips([make_test_clip("clip-2", source_id="src-1")])
    MainWindow._on_project_save_finished(window, True, str(target), "")

    assert project.path is None
    assert project.is_dirty
    assert window.recent_projects == []
    assert "newer changes remain unsaved" in window.status_bar.messages[-1]


def test_stale_save_completion_does_not_attach_path_to_replaced_project(tmp_path):
    from ui.main_window import MainWindow

    original_project = _make_project_with_clip(tmp_path)
    current_project = Project.new(name="replacement")
    target = tmp_path / "out.sceneripper"
    context = {
        "project": original_project,
        "mutation_generation": original_project.mutation_generation,
        "filepath": target,
    }
    window = _SaveCompletionWindowStub(current_project, context)

    MainWindow._on_project_save_finished(window, True, str(target), "")

    assert current_project.path is None
    assert not current_project.is_dirty
    assert window.recent_projects == []


def test_save_worker_emits_success_signal(qapp, tmp_path):
    """Successful save emits save_finished(True, path, '')."""
    from ui.main_window import SaveProjectWorker

    project = _make_project_with_clip(tmp_path)
    target = tmp_path / "out.sceneripper"
    snapshot = project.snapshot_for_save()

    received = []
    worker = SaveProjectWorker(snapshot, target)
    worker.save_finished.connect(
        lambda success, path, error: received.append((success, path, error))
    )

    worker.run()  # Run inline (no thread) so we can assert deterministically.

    assert received, "save_finished was not emitted"
    success, path, error = received[0]
    assert success is True
    assert path == str(target)
    assert error == ""
    assert target.exists()


def test_save_worker_emits_failure_signal_on_exception(qapp, tmp_path, monkeypatch):
    """When save_project raises, worker emits save_finished(False, path, error)."""
    from ui import main_window
    from ui.main_window import SaveProjectWorker

    project = _make_project_with_clip(tmp_path)
    target = tmp_path / "out.sceneripper"
    snapshot = project.snapshot_for_save()

    def boom(**_kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(main_window, "save_project", boom)

    received = []
    worker = SaveProjectWorker(snapshot, target)
    worker.save_finished.connect(
        lambda success, path, error: received.append((success, path, error))
    )

    worker.run()

    assert received
    success, path, error = received[0]
    assert success is False
    assert path == str(target)
    assert "disk full" in error


def test_save_worker_uses_snapshot_not_live_project(qapp, tmp_path, monkeypatch):
    """Mutations on the live project must not leak into the worker's save call."""
    from ui import main_window

    project = _make_project_with_clip(tmp_path)
    target = tmp_path / "out.sceneripper"
    snapshot = project.snapshot_for_save()

    captured = {}

    def fake_save_project(**kwargs):
        captured["clips"] = kwargs["clips"]
        captured["sources"] = kwargs["sources"]
        return True

    monkeypatch.setattr(main_window, "save_project", fake_save_project)

    # Mutate the live project AFTER snapshotting but BEFORE the worker runs.
    project.add_clips([make_test_clip("clip-2", source_id="src-1")])

    worker = main_window.SaveProjectWorker(snapshot, target)
    worker.run()

    saved_ids = {c.id for c in captured["clips"]}
    assert saved_ids == {"clip-1"}, (
        "Worker must serialize the snapshot, not the post-mutation live project"
    )


@pytest.mark.parametrize("fail", [False, True])
def test_save_worker_uses_explicit_session_writer_on_real_thread(
    qapp, tmp_path, monkeypatch, fail
):
    from core.project_lock import ProjectWriter, ProjectBusyError
    from ui import main_window

    target = tmp_path / "threaded.sceneripper"
    project = _make_project_with_clip(tmp_path)
    writer = ProjectWriter(target).acquire()
    if fail:

        def failed_save(**kwargs):
            raise OSError("disk unavailable")

        monkeypatch.setattr(main_window, "save_project", failed_save)
    worker = None
    try:
        worker = main_window.SaveProjectWorker(
            project.snapshot_for_save(), target, writer=writer
        )
        received = []
        worker.save_finished.connect(lambda *args: received.append(args))
        worker.start()
        assert worker.wait(10000)
        qapp.processEvents()
        assert received and received[0][0] is (not fail)
        assert target.exists() is (not fail)
        with pytest.raises(ProjectBusyError):
            with ProjectWriter(target):
                pass
        # Worker scope is gone on success and failure; session ownership remains.
        with writer.activate():
            pass
    finally:
        if worker is not None and worker.isRunning():
            worker.wait()
        writer.close()


def test_save_worker_rejects_writer_for_another_destination(qapp, tmp_path):
    from core.project_lock import ProjectWriter
    from ui.main_window import SaveProjectWorker

    protected = tmp_path / "protected.sceneripper"
    target = tmp_path / "other.sceneripper"
    with ProjectWriter(protected) as writer:
        received = []
        worker = SaveProjectWorker(
            _make_project_with_clip(tmp_path).snapshot_for_save(), target, writer=writer
        )
        worker.save_finished.connect(lambda *args: received.append(args))
        worker.run()
        assert received and not received[0][0]
        assert not target.exists()


def test_owned_async_save_as_keeps_new_path_and_newer_edits(
    qapp, tmp_path, monkeypatch
):
    from core.project_lock import ProjectBusyError, ProjectWriter
    from ui.main_window import MainWindow, SaveProjectWorker

    project = _make_project_with_clip(tmp_path)
    project._retain_writer = True
    old, target = tmp_path / "old.sceneripper", tmp_path / "new.sceneripper"
    assert project.save(old)
    window = _SaveCompletionWindowStub(project, None)
    window.save_worker = None
    window.sequence_tab = SimpleNamespace(_persist_current_sequence=lambda: None)
    window.analyze_tab = SimpleNamespace(get_clip_ids=lambda: [])
    window._on_project_save_finished = (
        lambda *args: MainWindow._on_project_save_finished(window, *args)
    )

    def start(worker):
        for path in (old, target):
            with pytest.raises(ProjectBusyError):
                with ProjectWriter(path):
                    pass
        project.rename("Newer edit")
        worker.run()

    monkeypatch.setattr(SaveProjectWorker, "start", start)
    try:
        MainWindow._save_project_to_file(window, target)
        assert project.path == target
        assert project.is_dirty
        assert not project.save_in_progress
        with ProjectWriter(old):
            pass
        with pytest.raises(ProjectBusyError):
            with ProjectWriter(target):
                pass
    finally:
        project.close_writer()


def test_owned_worker_start_failure_preserves_old_project(qapp, tmp_path, monkeypatch):
    from core.project_lock import ProjectBusyError, ProjectWriter
    from ui.main_window import MainWindow, SaveProjectWorker, QMessageBox

    project = _make_project_with_clip(tmp_path)
    project._retain_writer = True
    old, target = tmp_path / "old.sceneripper", tmp_path / "new.sceneripper"
    assert project.save(old)
    window = _SaveCompletionWindowStub(project, None)
    window.save_worker = None
    window.sequence_tab = SimpleNamespace(_persist_current_sequence=lambda: None)
    window.analyze_tab = SimpleNamespace(get_clip_ids=lambda: [])
    window._on_project_save_finished = lambda *args: None

    def fail(worker):
        raise RuntimeError("cannot start worker")

    monkeypatch.setattr(SaveProjectWorker, "start", fail)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: None)
    try:
        MainWindow._save_project_to_file(window, target)
        assert project.path == old
        assert not project.save_in_progress
        assert window._save_project_context is None
        with ProjectWriter(target):
            pass
        with pytest.raises(ProjectBusyError):
            with ProjectWriter(old):
                pass
    finally:
        project.close_writer()
