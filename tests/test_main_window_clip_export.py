"""Focused tests for single-clip export helpers on MainWindow."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import logging

from core.project import Project
from models.clip import Source, Clip
from models.sequence import Sequence, SequenceClip
from tests.conftest import make_test_clip
from ui.main_window import MainWindow


@pytest.fixture
def source():
    return Source(
        id="src-1",
        file_path=Path("/test/video.mp4"),
        duration_seconds=10.0,
        fps=30.0,
        width=1280,
        height=720,
    )


class _DummyProgressBar:
    def __init__(self):
        self.visible = False
        self.ranges = []

    def setVisible(self, visible: bool):
        self.visible = visible

    def setRange(self, minimum: int, maximum: int):
        self.ranges.append((minimum, maximum))


class _DummyStatusBar:
    def __init__(self):
        self.messages = []

    def showMessage(self, message: str, timeout: int = 0):
        self.messages.append((message, timeout))


class _DummyExportButton:
    def __init__(self):
        self.enabled = True

    def setEnabled(self, enabled: bool):
        self.enabled = enabled


def _make_window(tmp_path):
    window = SimpleNamespace(
        settings=SimpleNamespace(export_dir=tmp_path),
        progress_bar=_DummyProgressBar(),
        status_bar=_DummyStatusBar(),
        sources_by_id={},
        _sanitize_filename=MainWindow._sanitize_filename,
    )
    window._default_clip_export_filename = (
        lambda clip, source, ordinal=None:
        MainWindow._default_clip_export_filename(window, clip, source, ordinal)
    )
    window._get_edl_sources = lambda: MainWindow._get_edl_sources(window)
    window._get_edl_frames = lambda: MainWindow._get_edl_frames(window)
    window._default_sequence_edl_filename = (
        lambda sequence, index=None:
        MainWindow._default_sequence_edl_filename(window, sequence, index)
    )
    window._persist_sequence_tab_state_for_export = (
        lambda: MainWindow._persist_sequence_tab_state_for_export(window)
    )
    window._export_sequence_edl_to_path = (
        lambda sequence, output_path:
        MainWindow._export_sequence_edl_to_path(window, sequence, output_path)
    )
    window._unique_edl_output_path = (
        lambda output_dir, filename:
        MainWindow._unique_edl_output_path(window, output_dir, filename)
    )
    window._get_populated_edl_sequences = (
        lambda: MainWindow._get_populated_edl_sequences(window)
    )
    window._format_sequence_edl_choice_label = (
        lambda index, sequence:
        MainWindow._format_sequence_edl_choice_label(window, index, sequence)
    )
    window._prompt_for_edl_sequence_index = (
        lambda: MainWindow._prompt_for_edl_sequence_index(window)
    )
    window._on_sequence_edl_export_requested = (
        lambda sequence_index:
        MainWindow._on_sequence_edl_export_requested(window, sequence_index)
    )
    return window


def test_single_clip_export_appends_mp4_and_uses_clicked_source(tmp_path, source, monkeypatch):
    clip = make_test_clip("c1")
    clip.source_id = source.id
    window = _make_window(tmp_path)

    requested_paths = []

    monkeypatch.setattr(
        "ui.main_window.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(tmp_path / "chosen_clip"), "Video Files (*.mp4)"),
    )
    monkeypatch.setattr(
        "ui.main_window.QDesktopServices.openUrl",
        lambda url: requested_paths.append(Path(url.toLocalFile())),
    )
    monkeypatch.setattr(
        "ui.main_window.QMessageBox.critical",
        lambda *args, **kwargs: pytest.fail("export should not fail"),
    )

    export_calls = []

    def fake_export(req_clip, req_source, output_path):
        export_calls.append((req_clip, req_source, output_path))
        return True

    window._export_clip_to_path = fake_export

    MainWindow._on_clip_export_requested(window, clip, source)

    assert export_calls == [(clip, source, tmp_path / "chosen_clip.mp4")]
    assert requested_paths == [tmp_path]
    assert window.status_bar.messages[-1] == ("Exported clip to chosen_clip.mp4", 5000)
    assert window.progress_bar.ranges == [(0, 0), (0, 100)]


def test_single_clip_export_cancel_does_nothing(tmp_path, source, monkeypatch):
    clip = make_test_clip("c1")
    window = _make_window(tmp_path)

    monkeypatch.setattr(
        "ui.main_window.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: ("", ""),
    )

    export_calls = []
    window._export_clip_to_path = lambda *args: export_calls.append(args) or True

    MainWindow._on_clip_export_requested(window, clip, source)

    assert export_calls == []
    assert window.status_bar.messages == []


def test_default_clip_export_filename_uses_clip_name_when_present(tmp_path, source):
    clip = make_test_clip("c1")
    clip.name = 'Hero "Moment"'
    window = _make_window(tmp_path)

    filename = MainWindow._default_clip_export_filename(window, clip, source)

    assert filename == "video_Hero _Moment_.mp4"


def test_export_clip_to_path_logs_failure(tmp_path, source, monkeypatch, caplog):
    clip = make_test_clip("c1")
    clip.source_id = source.id
    output_path = tmp_path / "clip.mp4"

    class _FakeProcessor:
        def extract_clip(self, **_kwargs):
            return False

    monkeypatch.setattr("ui.main_window.FFmpegProcessor", lambda: _FakeProcessor())

    window = _make_window(tmp_path)

    with caplog.at_level(logging.INFO):
        success = MainWindow._export_clip_to_path(window, clip, source, output_path)

    assert success is False
    assert "Manual clip export requested" in caplog.text
    assert "Manual clip export failed" in caplog.text


def test_sequence_export_rejects_unwritable_output_directory(tmp_path, source, monkeypatch):
    clip = make_test_clip("c1")
    clip.source_id = source.id
    export_button = _DummyExportButton()

    sequence = SimpleNamespace(
        fps=source.fps,
        algorithm="dice",
        music_path=None,
        get_all_clips=lambda: [clip],
    )
    timeline = SimpleNamespace(
        export_btn=export_button,
        get_sources_lookup=lambda: {source.id: source},
        get_clips_lookup=lambda: {clip.id: (clip, source)},
    )
    window = _make_window(tmp_path)
    window.export_worker = None
    window.current_source = source
    window.clips = [clip]
    window.sources_by_id = {source.id: source}
    window.sequence_tab = SimpleNamespace(get_sequence=lambda: sequence, timeline=timeline)
    window.render_tab = SimpleNamespace(
        get_quality_setting=lambda: "medium",
        get_resolution_setting=lambda: None,
        get_fps_setting=lambda: source.fps,
    )
    window._validate_export_output_path = (
        lambda output_path: MainWindow._validate_export_output_path(window, output_path)
    )

    warnings = []
    monkeypatch.setattr(
        "ui.main_window.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: ("/scene_export.mp4", "Video Files (*.mp4)"),
    )
    monkeypatch.setattr(
        "ui.main_window.QMessageBox.warning",
        lambda *args, **kwargs: warnings.append((args[1], args[2])),
    )
    monkeypatch.setattr(
        "ui.main_window.os.access",
        lambda path, mode: Path(path) != Path("/"),
    )
    monkeypatch.setattr(
        "ui.main_window.SequenceExportWorker",
        lambda *args, **kwargs: pytest.fail("sequence export should not start"),
    )

    MainWindow._on_sequence_export_click(window)

    assert len(warnings) == 1
    assert warnings[0][0] == "Export Sequence"
    # Path separator differs on Windows (\) vs Unix (/)
    root_path = str(Path("/"))
    assert warnings[0][1] == f"Cannot write to export folder:\n{root_path}\n\nChoose a different location."
    assert window.progress_bar.ranges == []
    assert export_button.enabled is True


def test_sequence_tab_edl_export_uses_selected_sequence_and_project_sources(
    tmp_path, source, monkeypatch
):
    project = Project.new(name="EDL Project")
    project.add_source(source)
    clip = Clip(id="clip-1", source_id=source.id, start_frame=0, end_frame=30)
    project.add_clips([clip])

    project.sequences[0].name = "First"
    project.add_to_sequence(["clip-1"])
    second = Sequence(name="Second")
    second.tracks[0].add_clip(
        SequenceClip(
            source_clip_id=clip.id,
            source_id=source.id,
            start_frame=0,
            in_point=0,
            out_point=30,
        )
    )
    project.add_sequence(second)

    window = _make_window(tmp_path)
    window.project = project
    window.project_metadata = project.metadata
    window.current_source = None
    window.sources_by_id = project.sources_by_id
    persisted = []
    window.sequence_tab = SimpleNamespace(
        _persist_current_sequence=lambda: persisted.append(True)
    )

    requested_paths = []
    export_calls = []

    monkeypatch.setattr(
        "ui.main_window.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(tmp_path / "chosen"), "Edit Decision List (*.edl)"),
    )
    monkeypatch.setattr(
        "ui.main_window.QDesktopServices.openUrl",
        lambda url: requested_paths.append(Path(url.toLocalFile())),
    )
    monkeypatch.setattr(
        "ui.main_window.export_edl",
        lambda sequence, sources, config, frames=None: export_calls.append(
            (sequence, sources, config, frames)
        )
        or True,
    )

    MainWindow._on_sequence_edl_export_requested(window, 1)

    assert persisted == [True]
    assert len(export_calls) == 1
    exported_sequence, exported_sources, config, frames = export_calls[0]
    assert exported_sequence is project.sequences[1]
    assert exported_sources == project.sources_by_id
    assert frames == project.frames_by_id
    assert config.title == "Second"
    assert config.output_path == tmp_path / "chosen.edl"
    assert requested_paths == [tmp_path]
    assert window.status_bar.messages[-1] == ("EDL exported to chosen.edl", 5000)


def test_sequence_tab_batch_edl_export_writes_each_populated_sequence(
    tmp_path, source, monkeypatch
):
    project = Project.new(name="EDL Project")
    project.add_source(source)
    clip_a = Clip(id="clip-a", source_id=source.id, start_frame=0, end_frame=30)
    clip_b = Clip(id="clip-b", source_id=source.id, start_frame=30, end_frame=60)
    project.add_clips([clip_a, clip_b])

    project.sequences[0].name = "A Cut"
    project.add_to_sequence(["clip-a"])
    sequence_b = Sequence(name="B Cut")
    sequence_b.tracks[0].add_clip(
        SequenceClip(
            source_clip_id=clip_b.id,
            source_id=source.id,
            start_frame=0,
            in_point=clip_b.start_frame,
            out_point=clip_b.end_frame,
        )
    )
    project.add_sequence(sequence_b)
    project.add_sequence(Sequence(name="Empty"))

    window = _make_window(tmp_path)
    window.project = project
    window.project_metadata = project.metadata
    window.current_source = None
    window.sources_by_id = project.sources_by_id
    window.sequence_tab = SimpleNamespace(_persist_current_sequence=lambda: None)

    opened = []
    monkeypatch.setattr(
        "ui.main_window.QFileDialog.getExistingDirectory",
        lambda *args, **kwargs: str(tmp_path),
    )
    monkeypatch.setattr(
        "ui.main_window.QDesktopServices.openUrl",
        lambda url: opened.append(Path(url.toLocalFile())),
    )

    MainWindow._on_all_sequence_edl_export_requested(window)

    first_edl = tmp_path / "01_A Cut.edl"
    second_edl = tmp_path / "02_B Cut.edl"
    assert first_edl.exists()
    assert second_edl.exists()
    assert not (tmp_path / "03_Empty.edl").exists()
    assert "TITLE: A Cut" in first_edl.read_text()
    assert "TITLE: B Cut" in second_edl.read_text()
    assert opened == [tmp_path]
    assert window.status_bar.messages[-1] == ("Exported 2 sequence EDL(s)", 5000)


def test_file_menu_edl_export_prompts_for_sequence_before_export(
    tmp_path, source, monkeypatch
):
    project = Project.new(name="EDL Project")
    project.add_source(source)
    clip_a = Clip(id="clip-a", source_id=source.id, start_frame=0, end_frame=30)
    clip_b = Clip(id="clip-b", source_id=source.id, start_frame=30, end_frame=60)
    project.add_clips([clip_a, clip_b])

    project.sequences[0].name = "Active Cut"
    project.add_to_sequence(["clip-a"])
    selected_sequence = Sequence(name="Selected Cut")
    selected_sequence.tracks[0].add_clip(
        SequenceClip(
            source_clip_id=clip_b.id,
            source_id=source.id,
            start_frame=0,
            in_point=clip_b.start_frame,
            out_point=clip_b.end_frame,
        )
    )
    project.add_sequence(selected_sequence)

    window = _make_window(tmp_path)
    window.project = project
    window.project_metadata = project.metadata
    window.current_source = None
    window.sources_by_id = project.sources_by_id
    window.sequence_tab = SimpleNamespace(_persist_current_sequence=lambda: None)

    prompts = []
    export_calls = []
    monkeypatch.setattr(
        "ui.main_window.QInputDialog.getItem",
        lambda *args: prompts.append(args) or (args[3][1], True),
    )
    monkeypatch.setattr(
        "ui.main_window.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(tmp_path / "file_menu"), "Edit Decision List (*.edl)"),
    )
    monkeypatch.setattr("ui.main_window.QDesktopServices.openUrl", lambda *_args: None)
    monkeypatch.setattr(
        "ui.main_window.export_edl",
        lambda sequence, sources, config, frames=None: export_calls.append(
            (sequence, sources, config, frames)
        )
        or True,
    )

    MainWindow._on_export_edl_click(window)

    assert len(prompts) == 1
    assert prompts[0][3] == [
        "1. Active Cut (1 clip)",
        "2. Selected Cut (1 clip)",
    ]
    assert len(export_calls) == 1
    assert export_calls[0][0] is selected_sequence
    assert export_calls[0][2].output_path == tmp_path / "file_menu.edl"
