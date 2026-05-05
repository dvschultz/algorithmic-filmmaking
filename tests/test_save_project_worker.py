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
        self.save_worker = SimpleNamespace()
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
