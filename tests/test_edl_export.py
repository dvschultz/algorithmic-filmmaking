"""CMX 3600 EDL regression tests: reel names, clip names and source paths."""

import re
from pathlib import Path

import pytest

from core.edl_export import EDLExportConfig, export_edl
from core.project import Project
from models.clip import Clip, Source
from models.sequence import Sequence

EVENT = re.compile(r"^\d{3}  ")


def build_project(paths, order=None):
    project = Project.new()
    project.sequence = Sequence(fps=30)
    clips = []
    for path in paths:
        Path(path).write_bytes(b"media")
        source = Source(file_path=path, fps=30)
        clip = Clip(source_id=source.id, start_frame=240, end_frame=264)
        project.add_source(source)
        project.add_clips([clip])
        clips.append(clip)
    for index in order if order is not None else range(len(clips)):
        project.add_to_sequence([clips[index].id])
    return project


def export_lines(project, tmp_path):
    output = tmp_path / "out.edl"
    config = EDLExportConfig(output)
    assert export_edl(project.sequence, project.sources_by_id, config), config.error_message
    return output.read_text(encoding="utf-8").splitlines()


def event_fields(lines):
    return [line.split() for line in lines if EVENT.match(line)]


def values(lines, prefix):
    return [line[len(prefix):] for line in lines if line.startswith(prefix)]


DADA = ("Dadaist Filmmaking week2.mov", "Dadaist Filmmaking week3.mov")


@pytest.mark.parametrize("names", [DADA, ("my clip.mov", "my clip (1).mov")])
def test_reels_are_single_token_valid_and_distinct_per_source(tmp_path, names):
    lines = export_lines(build_project([tmp_path / name for name in names]), tmp_path)
    events = event_fields(lines)
    assert len(events) == 2
    for fields in events:
        assert len(fields) == 8, fields
        assert fields[2] == "V" and fields[3] == "C"
        assert re.match(r"^[A-Z0-9_]{1,8}$", fields[1]), fields[1]
    assert events[0][1] != events[1][1]


def test_from_clip_name_is_source_basename(tmp_path):
    lines = export_lines(build_project([tmp_path / name for name in DADA]), tmp_path)
    assert values(lines, "* FROM CLIP NAME: ") == list(DADA)


def test_source_file_is_absolute_posix_for_relative_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    lines = export_lines(build_project([Path("clip a.mp4")]), tmp_path)
    expected = (tmp_path / "clip a.mp4").resolve().as_posix()
    assert values(lines, "* SOURCE FILE: ") == [expected]
    assert all("\\" not in value for value in values(lines, "* SOURCE FILE: "))


def test_same_source_twice_shares_reel(tmp_path):
    lines = export_lines(build_project([tmp_path / DADA[0]], order=[0, 0]), tmp_path)
    events = event_fields(lines)
    assert len(events) == 2
    assert events[0][1] == events[1][1]
