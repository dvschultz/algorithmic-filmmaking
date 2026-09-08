"""Rose Hobart uses detached verified analysis before owner publication."""

from copy import deepcopy
from unittest.mock import Mock

import pytest
from PySide6.QtWidgets import QApplication
from tests.test_face_records import setup as face_setup, run  # noqa: F401


@pytest.fixture
def analyzed_project(request, monkeypatch):
    setup = request.getfixturevalue("face_setup")
    provider = setup[1]

    def reference(path, *, on_execution):
        return provider.side_effect(on_execution=on_execution, start_frame=0)

    monkeypatch.setattr("core.analysis.faces.extract_faces_from_image", reference)
    return setup


def test_worker_does_not_mutate_project_faces(analyzed_project, monkeypatch):
    from ui.dialogs.rose_hobart_dialog import RoseHobartWorker

    project, provider, _ = analyzed_project
    monkeypatch.setattr(
        "core.feature_registry.check_feature_ready", lambda _: (True, [])
    )
    clip, source = project.clips[0], project.sources[0]
    original = deepcopy(clip.to_dict())
    worker = RoseHobartWorker(
        [source.file_path], [(clip, source)], "Balanced", "Original Order", 1.0
    )
    worker.run()
    assert provider.call_count == 1
    assert clip.to_dict() == original
    assert worker.outcomes[0].record_json is not None
    assert worker.result[0][0].id == clip.id


@pytest.mark.parametrize(
    "mode", ["current", "cancel", "range", "session", "path", "observer_cancel"]
)
def test_dialog_publishes_only_after_native_exit(analyzed_project, monkeypatch, mode):
    import time
    import threading
    from types import SimpleNamespace
    from ui.dialogs.rose_hobart_dialog import RoseHobartDialog, RoseHobartWorker
    from core.operations.faces import run_faces

    project, _, _ = analyzed_project
    clip, source = project.clips[0], project.sources[0]
    app = QApplication.instance()
    entered, release = threading.Event(), threading.Event()
    monkeypatch.setattr("PySide6.QtWidgets.QMessageBox.warning", Mock())

    def held(self):
        self.outcomes = run_faces(self.tasks, self.options)
        self.result = self._clips
        self.finished_sequence.emit(self.result)
        entered.set()
        assert release.wait(5)

    monkeypatch.setattr(RoseHobartWorker, "run", held)
    dialog = RoseHobartDialog([clip], {source.id: source}, project=project)
    dialog._ref_widgets = [SimpleNamespace(_image_path=source.file_path, has_face=True)]
    sequences = []
    dialog.sequence_ready.connect(sequences.append)
    if mode == "observer_cancel":
        project.add_observer(
            lambda event, data: dialog.reject() if event == "clips_updated" else None
        )
    dialog._on_generate()
    worker = dialog.worker
    try:
        assert entered.wait(5)
        app.processEvents()
        assert not sequences and clip.face_embeddings is None
        assert dialog.worker is worker
        if mode == "cancel":
            dialog.reject()
            assert dialog.worker is worker
        elif mode == "range":
            clip.end_frame -= 1
        elif mode == "session":
            project.session.session_id = "replaced"
        elif mode == "path":
            project.path = source.file_path.parent / "other.sceneripper"
    finally:
        release.set()
        assert worker.wait(5000)
    deadline = time.monotonic() + 5
    while dialog.worker is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.002)
    assert dialog.worker is None
    if mode == "current":
        assert len(sequences) == 1 and len(sequences[0]) == 1
        assert sequences[0][0][0] is clip and sequences[0][0][1] is source
        assert clip.analysis_records["face_embeddings"].state == "succeeded"
    else:
        assert not sequences
        if mode != "observer_cancel":
            assert clip.face_embeddings is None
    dialog._ref_widgets = []
    dialog.reject()


@pytest.mark.parametrize("change", ["none", "legacy", "interval"])
def test_worker_verifies_reuse(analyzed_project, monkeypatch, change):
    from ui.dialogs.rose_hobart_dialog import RoseHobartWorker

    project, provider, _ = analyzed_project
    run(project)
    clip, source = project.clips[0], project.sources[0]
    if change == "legacy":
        clip.analysis_records.clear()
    monkeypatch.setattr(
        "core.feature_registry.check_feature_ready", lambda _: (True, [])
    )
    worker = RoseHobartWorker(
        [source.file_path],
        [(clip, source)],
        "Balanced",
        "Original Order",
        0.5 if change == "interval" else 1.0,
    )
    worker.run()
    assert worker.result is not None, worker.failure
    assert provider.call_count == (1 if change == "none" else 2)


@pytest.mark.parametrize("cancel", [False, True])
def test_reference_worker_is_retained_until_native_exit(
    analyzed_project, monkeypatch, cancel
):
    import threading
    import time
    from ui.dialogs.rose_hobart_dialog import RoseHobartDialog, _RefImageExtractWorker

    project, _, _ = analyzed_project
    source = project.sources[0]
    entered, release = threading.Event(), threading.Event()

    def held(self):
        self.result = []
        self.faces_extracted.emit(str(self._image_path), [])
        entered.set()
        assert release.wait(5)

    monkeypatch.setattr(_RefImageExtractWorker, "run", held)
    monkeypatch.setattr(
        "PySide6.QtWidgets.QFileDialog.getOpenFileName",
        lambda *args: (str(source.file_path), ""),
    )
    dialog = RoseHobartDialog(project.clips, project.sources_by_id, project=project)
    dialog._on_add_reference()
    worker = dialog._ref_extract_worker
    try:
        assert entered.wait(5)
        QApplication.instance().processEvents()
        assert dialog._ref_extract_worker is worker and not dialog._ref_widgets
        if cancel:
            dialog.reject()
            assert dialog._ref_extract_worker is worker
    finally:
        release.set()
        assert worker.wait(5000)
    deadline = time.monotonic() + 5
    while dialog._ref_extract_worker is not None and time.monotonic() < deadline:
        QApplication.instance().processEvents()
        time.sleep(0.002)
    assert dialog._ref_extract_worker is None
    assert len(dialog._ref_widgets) == (0 if cancel else 1)
    dialog.reject()


def test_reference_faces_require_execution_identity(analyzed_project, monkeypatch):
    from ui.dialogs.rose_hobart_dialog import RoseHobartWorker

    project, _, _ = analyzed_project
    clip, source = project.clips[0], project.sources[0]
    monkeypatch.setattr(
        "core.feature_registry.check_feature_ready", lambda _: (True, [])
    )
    monkeypatch.setattr(
        "core.analysis.faces.extract_faces_from_image",
        Mock(
            return_value=[
                {
                    "embedding": [0.12346] * 512,
                    "confidence": 0.9,
                }
            ]
        ),
    )
    worker = RoseHobartWorker(
        [source.file_path], [(clip, source)], "Balanced", "Original Order", 1.0
    )
    worker.run()
    assert worker.result is None
    assert worker.failure is not None


@pytest.mark.parametrize(
    "stage",
    ["during_reference", "between_references", "during_clips", "after_completion"],
)
def test_model_changes_do_not_publish_mixed_embeddings(
    analyzed_project, monkeypatch, stage
):
    from ui.dialogs.rose_hobart_dialog import RoseHobartWorker

    project, provider, directory = analyzed_project
    clip, source = project.clips[0], project.sources[0]
    original = provider.side_effect
    calls = []

    def change():
        (directory / "recognition.onnx").write_bytes(b"new recognition model")

    def reference(path, *, on_execution):
        calls.append(path)
        if stage == "between_references" and len(calls) == 2:
            change()
        faces = original(on_execution=on_execution, start_frame=0)
        if stage == "during_reference":
            change()
        return faces

    monkeypatch.setattr("core.analysis.faces.extract_faces_from_image", reference)
    monkeypatch.setattr(
        "core.feature_registry.check_feature_ready", lambda _: (True, [])
    )
    if stage == "during_clips":

        def execute(**kwargs):
            change()
            return original(**kwargs)

        provider.side_effect = execute
    refs = [source.file_path] * (2 if stage == "between_references" else 1)
    worker = RoseHobartWorker(refs, [(clip, source)], "Balanced", "Original Order", 1.0)
    worker.run()
    assert clip.face_embeddings is None
    if stage == "after_completion":
        assert worker.result is not None, worker.failure
        assert worker.references_current()
        change()
        assert not worker.references_current()
    else:
        assert worker.result is None
        assert worker.failure is not None
