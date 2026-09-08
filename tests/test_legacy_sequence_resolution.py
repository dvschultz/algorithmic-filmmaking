"""Legacy resolution is inspectable, reversible, and available on each surface."""

import asyncio
import json

import pytest

from core.project import Project


@pytest.fixture
def legacy_path(tmp_path):
    media = tmp_path / "source.mp4"
    media.write_bytes(b"media")
    data = {
        "version": "1.5",
        "sources": [{"id": "source", "file_path": str(media), "fps": 24}],
        "clips": [{"id": "clip", "source_id": "source", "start_frame": 240, "end_frame": 720}],
        "sequence": {"id": "sequence", "fps": 30, "tracks": [{"clips": [{
            "id": "entry", "source_clip_id": "clip", "source_id": "source",
            "in_point": 264, "out_point": 288,
        }]}]},
    }
    path = tmp_path / "legacy.sceneripper"
    path.write_text(json.dumps(data))
    return path


def test_resolution_is_reversible_and_preserves_original_bytes(legacy_path):
    original = legacy_path.read_bytes()
    project = Project.load(legacy_path)
    entry = project.sequence.get_all_clips()[0]
    before = entry.to_dict()
    project.resolve_sequence_timing("sequence", "entry", "clip-relative")
    assert entry.in_point == 504 and entry.out_point == 528
    project.session.undo()
    assert entry.to_dict() == before
    project.session.redo()
    assert entry.in_point == 504
    assert project.save()
    assert next(legacy_path.parent.glob("*.bak")).read_bytes() == original
    restored = Project.load(legacy_path).sequence.get_all_clips()[0]
    assert restored.legacy_timing["original"]["in_point"] == 264
    assert restored.source_range.duration == 1


@pytest.mark.parametrize("surface", ["cli", "mcp"])
def test_headless_resolution_uses_same_conversion(legacy_path, surface):
    if surface == "cli":
        from click.testing import CliRunner
        from cli.main import cli, register_commands
        register_commands()
        result = CliRunner().invoke(cli, [
            "--json", "project", "resolve-timing", str(legacy_path),
            "sequence", "entry", "--convention", "clip-relative",
        ])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
    else:
        from scene_ripper_mcp.tools.sequence import resolve_sequence_timing
        payload = json.loads(asyncio.run(resolve_sequence_timing(
            str(legacy_path), "sequence", "entry", "clip-relative",
        )))
    assert payload["success"], payload
    entry = Project.load(legacy_path).sequence.get_all_clips()[0]
    assert entry.in_point == 504
    assert entry.legacy_timing["status"] == "resolved"


def test_sequence_ui_resolution_uses_project_history(legacy_path, monkeypatch):
    from PySide6.QtWidgets import QApplication, QInputDialog
    from ui.tabs.sequence_tab import SequenceTab
    app = QApplication.instance() or QApplication([])
    project = Project.load(legacy_path)
    tab = SequenceTab()
    tab.set_project(project)
    choices = iter([0, 1])
    monkeypatch.setattr(QInputDialog, "getItem", lambda *args: (args[3][next(choices)], True))
    tab._resolve_legacy_timing(project.sequence)
    entry = project.sequence.get_all_clips()[0]
    assert entry.in_point == 504
    project.session.undo()
    assert entry.legacy_timing["status"] == "unresolved"
    tab.close()
    assert app is not None


def test_builtin_agent_can_inspect_and_resolve_legacy_timing(legacy_path):
    from core.chat_tools import tools
    from core.tool_executor import ToolExecutor
    project = Project.load(legacy_path)
    executor = ToolExecutor(project=project)
    state = tools.get("get_sequence_state").func(project)
    assert state["sequence_id"] == "sequence"
    assert state["clips"][0]["legacy_timing"]["status"] == "unresolved"
    tool = tools.get("resolve_sequence_timing")
    assert tool is not None and tool.modifies_project_state
    original_bytes = legacy_path.read_bytes()
    reply = executor.execute({"function": {
        "name": "resolve_sequence_timing", "arguments": json.dumps({
            "sequence_id": "sequence", "entry_id": "entry", "convention": "clip-relative",
        }),
    }})
    assert reply["success"] and reply["result"]["success"], reply
    assert project.sequence.get_all_clips()[0].in_point == 504
    assert legacy_path.read_bytes() == original_bytes
    project.session.undo()
    assert project.sequence.get_all_clips()[0].legacy_timing["status"] == "unresolved"


def test_unresolved_timing_blocks_all_render_outputs(legacy_path, tmp_path, monkeypatch):
    from core.edl_export import EDLExportConfig, export_edl
    from core.sequence_export import ExportConfig, SequenceExporter
    from core.sequence_preview import (
        compute_sequence_preview_signature, get_sequence_preview_path, render_sequence_preview,
    )
    project = Project.load(legacy_path)
    sequence = project.sequence
    sources = project.sources_by_id
    clips = {clip.id: (clip, sources[clip.source_id]) for clip in project.clips}
    output = tmp_path / "existing.mp4"
    output.write_bytes(b"previous export")
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: pytest.fail("Unresolved timing launched an encoder"))
    messages = []
    assert not SequenceExporter(ffmpeg_path="ffmpeg").export(
        sequence, sources, clips, ExportConfig(output_path=output, width=32, height=32),
        progress_callback=lambda value, message: messages.append(message),
    )
    assert "Resolve legacy timing" in messages[0]
    assert output.read_bytes() == b"previous export"
    edl = tmp_path / "existing.edl"
    edl.write_text("previous EDL")
    assert not export_edl(sequence, sources, EDLExportConfig(output_path=edl))
    assert edl.read_text() == "previous EDL"
    signature = compute_sequence_preview_signature(sequence, sources, clips)
    cached = get_sequence_preview_path(sequence, signature, tmp_path)
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(b"cached preview must not bypass unresolved timing")
    with pytest.raises(ValueError, match="Resolve legacy timing"):
        render_sequence_preview(sequence, sources, clips, cache_root=tmp_path)


def test_resolution_dialog_does_not_retarget_replaced_clip(legacy_path, monkeypatch):
    from dataclasses import replace
    from unittest.mock import Mock
    from PySide6.QtWidgets import QApplication, QInputDialog
    from ui.tabs.sequence_tab import SequenceTab
    app = QApplication.instance() or QApplication([])
    project = Project.load(legacy_path)
    tab = SequenceTab()
    tab.set_project(project)
    resolve = Mock(wraps=project.resolve_sequence_timing)
    monkeypatch.setattr(project, "resolve_sequence_timing", resolve)
    calls = []
    def choose(*args):
        calls.append(True)
        if len(calls) == 1:
            old = project.clips[0]
            project.replace_source_clips(old.source_id, [replace(old, start_frame=300, end_frame=780)])
            return args[3][0], True
        return args[3][1], True
    monkeypatch.setattr(QInputDialog, "getItem", choose)
    tab._resolve_legacy_timing(project.sequence)
    resolve.assert_not_called()
    tab.close()
    assert app is not None


def test_resolution_redo_rejects_replaced_source_clip(legacy_path):
    from dataclasses import replace
    project = Project.load(legacy_path)
    project.resolve_sequence_timing("sequence", "entry", "clip-relative")
    project.session.undo()
    old = project.clips[0]
    project.replace_source_clips(old.source_id, [replace(old, start_frame=300, end_frame=780)])
    with pytest.raises(ValueError, match="source changed"):
        project.session.redo()
