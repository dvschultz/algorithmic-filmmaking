"""Command lifetime ownership, tested through Click's actual cleanup path."""

import json

from click.testing import CliRunner
import pytest

from cli.main import cli, register_commands
from core.project import Project, ProjectLoadError
from core.project_lock import ProjectBusyError, ProjectWriter


@pytest.fixture
def project_file(tmp_path, monkeypatch):
    monkeypatch.setattr("core.project_lock._lock_directory", lambda: tmp_path / "locks")
    path = tmp_path / "edit.sceneripper"
    assert Project.new().save(path)
    register_commands()
    return path


@pytest.mark.parametrize(
    "command",
    [
        ["analyze", name]
        for name in ["colors", "describe", "shots", "classify", "objects", "people"]
    ]
    + [["transcribe"], ["project", "add-to-sequence"]],
)
def test_busy_mutations_fail_before_loading(project_file, monkeypatch, command):
    monkeypatch.setattr(
        Project, "load", lambda *a, **k: pytest.fail("loaded busy file")
    )
    monkeypatch.setattr(
        "core.project.load_project", lambda *a, **k: pytest.fail("loaded busy file")
    )
    args = ["--json", *command, str(project_file)]
    if command == ["project", "add-to-sequence"]:
        args.append("--all")
    with ProjectWriter(project_file):
        result = CliRunner().invoke(cli, args)
    assert result.exit_code == 1, result.output
    assert json.loads(result.stdout)["error"]["code"] == "project_busy"


def test_detect_busy_destination_prevents_computation(project_file, tmp_path):
    video = tmp_path / "video.mp4"
    video.touch()
    with ProjectWriter(project_file):
        result = CliRunner().invoke(
            cli, ["--json", "detect", str(video), "-o", str(project_file), "--force"]
        )
    assert result.exit_code == 1, result.output
    assert json.loads(result.stdout)["error"]["code"] == "project_busy"


@pytest.mark.parametrize("failure", [False, True])
def test_command_owns_before_load_and_releases_on_return_or_error(
    project_file, monkeypatch, failure
):
    load = Project.load
    observed = []

    def checked_load(*args, **kwargs):
        with pytest.raises(ProjectBusyError):
            with ProjectWriter(project_file):
                pass
        observed.append(True)
        if failure:
            raise ProjectLoadError("test load failure")
        return load(*args, **kwargs)

    monkeypatch.setattr(Project, "load", checked_load)
    result = CliRunner().invoke(cli, ["analyze", "colors", str(project_file)])
    assert result.exit_code == (1 if failure else 0), result.output
    assert observed == [True]
    with ProjectWriter(project_file):
        pass


def test_read_only_command_is_available_while_owned(project_file):
    with ProjectWriter(project_file):
        result = CliRunner().invoke(
            cli, ["--json", "project", "info", str(project_file)]
        )
    assert result.exit_code == 0, result.output


def test_sequence_mutation_excludes_process_during_save(project_file, monkeypatch):
    import subprocess
    import sys
    from core.project_lock import _lock_directory
    from models.clip import Source, Clip

    video = project_file.with_suffix(".mp4")
    video.touch()
    source = Source(file_path=video, duration_seconds=1)
    clip = Clip(source_id=source.id, start_frame=0, end_frame=30)
    project = Project.load(project_file)
    project.add_source(source)
    project.add_clips([clip])
    assert project.save()
    save = Project.save
    observed = []

    def checked_save(self, *args, **kwargs):
        script = """
import sys
from pathlib import Path
import core.project_lock as locks
locks._lock_directory = lambda: Path(sys.argv[2])
try:
    with locks.ProjectWriter(sys.argv[1]):
        sys.exit(9)
except locks.ProjectBusyError:
    pass
"""
        result = subprocess.run(
            [sys.executable, "-c", script, str(project_file), str(_lock_directory())],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, result.stderr
        observed.append(True)
        return save(self, *args, **kwargs)

    monkeypatch.setattr(Project, "save", checked_save)
    result = CliRunner().invoke(
        cli, ["project", "add-to-sequence", str(project_file), "--all"]
    )
    assert result.exit_code == 0, result.output
    assert observed == [True]
    assert len(Project.load(project_file).sequence.get_all_clips()) == 1
    with ProjectWriter(project_file):
        pass


def test_download_busy_project_keeps_download_success(project_file, monkeypatch):
    import inspect
    from types import SimpleNamespace
    from unittest.mock import Mock

    from core.jobs.store import JobStore
    monkeypatch.setattr("core.jobs.downloads.open_download_store", lambda: JobStore(project_file.parent / "jobs.db"))
    project_file.with_suffix(".mp4").write_bytes(b"video")
    downloader = Mock()
    downloader.is_valid_url.return_value = (True, "")
    downloader.get_video_info.return_value = {"title": "Video", "duration": 1}
    downloader.download.return_value = SimpleNamespace(
        success=True,
        title="Video",
        file_path=project_file.with_suffix(".mp4"),
        duration=1,
    )
    monkeypatch.setattr("core.downloader.VideoDownloader", lambda **kwargs: downloader)
    monkeypatch.setattr(
        "core.scene_detect.SceneDetector",
        lambda **kwargs: pytest.fail("computed before ownership"),
    )
    # Click 8.2+ always separates stdout; older versions default to mixing it.
    runner_options = (
        {"mix_stderr": False}
        if "mix_stderr" in inspect.signature(CliRunner).parameters
        else {}
    )
    with ProjectWriter(project_file):
        result = CliRunner(**runner_options).invoke(
            cli,
            [
                "--json",
                "download",
                "https://youtube.com/watch?v=test",
                "-o",
                str(project_file.parent),
                "--detect",
            ],
        )
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["success"]
    assert "project_file" not in data
    assert data["detection_error"]["code"] == "project_busy"
