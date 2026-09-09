"""U16: recipe inspection, duplicate/regenerate actions and A/B comparison."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from core.project import Project
from core.spine.sequences import compare_sequences, generate_sequence, regenerate_sequence
from models.clip import Clip, Source
from models.sequence import Sequence


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _project(tmp_path: Path, count: int = 6) -> Project:
    project = Project.new()
    path = tmp_path / "v.mp4"
    path.write_bytes(b"0")
    project.add_source(Source(id="s", file_path=path, fps=25.0, duration_seconds=60))
    project.add_clips([
        Clip(id=f"c{i}", source_id="s", start_frame=i * 25, end_frame=(i + 1) * 25 + i * 5,
             dominant_colors=[(i * 40, 20, 20)])
        for i in range(count)
    ])
    project.mark_clean()
    return project


def _ab(tmp_path: Path):
    """A shuffled sequence A and a regenerated variation B with one changed parameter."""
    project = _project(tmp_path)
    first = generate_sequence(project, "shuffle", seed=3, name="A")
    assert first["success"], first
    a = project.sequence
    second = regenerate_sequence(project, a.id, parameters={"hflip": True}, keep_seed=True, name="B")
    assert second["success"], second
    return project, a, project.sequence


# --- spine ---------------------------------------------------------------

class TestCompareSequences:
    def test_reports_changed_parameters_lineage_and_deltas(self, tmp_path):
        project, a, b = _ab(tmp_path)
        result = compare_sequences(project, a.id, b.id)
        assert result["success"] and result["same_algorithm"] and result["seed_changed"] is False
        assert result["parameter_differences"] == [{"key": "hflip", "a": False, "b": True}]
        assert result["inputs_equal"] and result["related"]
        assert result["a"]["clip_count"] == result["b"]["clip_count"] == 6
        assert result["clip_count_delta"] == 0 and result["duration_delta_seconds"] == 0.0
        assert result["comparable_seconds"] == result["a"]["duration_seconds"]
        assert result["a"]["parent_recipe_id"] is None and result["b"]["parent_recipe_id"] == result["a"]["recipe_id"]
        json.dumps(result)  # MCP-safe

    def test_manual_sequences_compare_timelines_only(self, tmp_path):
        project = _project(tmp_path)
        project.add_sequence(Sequence(name="Manual", algorithm="manual"), activate=True)
        manual = project.sequence
        project.add_to_sequence(["c0", "c1"])
        result = compare_sequences(project, project.sequences[0].id, manual.id)
        assert result["success"] and result["parameter_differences"] == [] and result["seed_changed"] is None
        assert result["a"]["has_recipe"] is False and result["clip_count_delta"] == 2
        assert not result["timelines_identical"]
        assert compare_sequences(project, manual.id, manual.id)["success"] is False
        assert compare_sequences(project, manual.id, "nope")["success"] is False

    def test_chat_and_mcp_expose_compare(self, tmp_path):
        from core.chat_tools import tools

        project, a, b = _ab(tmp_path)
        chat = tools.get("compare_sequences")
        assert chat is not None and not chat.modifies_project_state
        assert chat.func(project, a.id, b.id)["parameter_differences"][0]["key"] == "hflip"


@pytest.mark.asyncio
async def test_mcp_compare_sequences_is_read_only(tmp_path):
    from scene_ripper_mcp.tools import sequence as mcp_tools

    project, a, b = _ab(tmp_path)
    path = tmp_path / "p.sceneripper"
    assert project.save(path)
    stamp = path.stat().st_mtime_ns
    result = json.loads(await mcp_tools.compare_sequences(str(path), a.id, b.id))
    assert result["success"] and result["parameter_differences"][0]["key"] == "hflip"
    assert path.stat().st_mtime_ns == stamp
    missing = json.loads(await mcp_tools.compare_sequences(str(path), a.id, "missing"))
    assert missing["success"] is False


# --- panel ---------------------------------------------------------------

class TestComparisonPanel:
    @pytest.fixture
    def panel(self, qapp, tmp_path):
        from ui.widgets.sequence_comparison import SequenceComparisonPanel

        project, a, b = _ab(tmp_path)
        panel = SequenceComparisonPanel()
        panel.set_project(project)
        panel.select("a", a.id)
        panel.select("b", b.id)
        return project, a, b, panel

    def test_empty_state_until_two_sequences_exist(self, qapp, tmp_path):
        from ui.widgets.sequence_comparison import SequenceComparisonPanel

        project = _project(tmp_path)
        panel = SequenceComparisonPanel()
        panel.set_project(project)
        assert panel.empty_label.isVisibleTo(panel) and not panel.body.isVisibleTo(panel)
        generate_sequence(project, "shuffle", seed=1)  # replaces the empty default sequence
        panel.refresh()
        assert panel.empty_label.isVisibleTo(panel)
        generate_sequence(project, "shuffle", seed=2)
        panel.refresh()
        assert panel.body.isVisibleTo(panel) and panel.selected_ids() == (None, None)
        assert "Pick A and B" in panel.difference_label.text()
        assert not panel.side_a.show_btn.isEnabled()

    def test_shows_differences_and_summaries(self, panel):
        project, a, b, panel = panel
        text = panel.difference_label.text()
        assert "hflip: False -> True" in text and "Identical timelines." in text
        assert panel.side_a.values["clips"].text() == "6"
        assert panel.side_a.values["seed"].text() == "3" == panel.side_b.values["seed"].text()
        assert panel.side_a.values["preview"].text() == "not rendered"
        assert panel.side_a.regenerate_btn.isEnabled() and panel.side_a.inspect_btn.isEnabled()

    def test_missing_preview_offers_render_without_blocking_differences(self, panel):
        project, a, b, panel = panel
        panel.set_preview_probe(lambda sequence: sequence.id == a.id)
        assert panel.side_a.render_btn.text() == "Preview ready" and not panel.side_a.render_btn.isEnabled()
        assert panel.side_b.render_btn.text() == "Render preview" and panel.side_b.render_btn.isEnabled()
        asked = []
        panel.render_preview_requested.connect(asked.append)
        panel.side_b.render_btn.click()
        assert asked == [b.id]
        assert "hflip" in panel.difference_label.text()  # differences still readable
        panel.set_preview_probe(lambda sequence: (_ for _ in ()).throw(RuntimeError("probe")))
        assert panel.side_a.values["preview"].text() == "not rendered"  # a failing probe never breaks the panel

    def test_deleting_a_compared_sequence_clears_only_its_slot(self, panel):
        project, a, b, panel = panel
        project.remove_sequence(project.sequences.index(b))
        panel.refresh()
        assert panel.selected_ids() == (a.id, None)
        assert panel.side_b.values["clips"].text() == "--"
        assert "Pick A and B" in panel.difference_label.text()

    def test_keyboard_switches_and_actions_emit_ids(self, panel):
        project, a, b, panel = panel
        switched, inspected, duplicated, regenerated = [], [], [], []
        panel.switch_requested.connect(switched.append)
        panel.inspect_requested.connect(inspected.append)
        panel.duplicate_requested.connect(duplicated.append)
        panel.regenerate_requested.connect(regenerated.append)
        QTest.keyClick(panel, Qt.Key_B)
        QTest.keyClick(panel, Qt.Key_Left)
        panel.side_a.inspect_btn.click()
        panel.side_b.duplicate_btn.click()
        panel.side_a.regenerate_btn.click()
        assert switched == [b.id, a.id] and inspected == [a.id]
        assert duplicated == [b.id] and regenerated == [a.id]

    def test_generation_state_disables_actions_for_that_sequence(self, panel):
        project, a, b, panel = panel
        panel.set_generation_state(a.id, "Regenerating shuffle...", running=True)
        assert panel.status_label.text() == "Regenerating shuffle..."
        assert not panel.side_a.regenerate_btn.isEnabled() and panel.side_b.regenerate_btn.isEnabled()
        panel.set_generation_state(a.id, running=False)
        assert panel.side_a.regenerate_btn.isEnabled() and panel.status_label.text() == ""

    def test_read_only_projects_keep_inspection_but_not_mutation(self, panel, monkeypatch):
        project, a, b, panel = panel
        monkeypatch.setattr(type(project), "is_read_only", property(lambda self: True))
        panel.refresh()
        assert panel.side_a.inspect_btn.isEnabled() and panel.side_a.show_btn.isEnabled()
        assert not panel.side_a.duplicate_btn.isEnabled() and not panel.side_a.regenerate_btn.isEnabled()


# --- tab -----------------------------------------------------------------

class TestSequenceTabVariations:
    @pytest.fixture
    def tab(self, qapp, tmp_path):
        from ui.tabs.sequence_tab import SequenceTab

        project, a, b = _ab(tmp_path)
        tab = SequenceTab()
        tab.set_project(project)
        tab._sources = dict(project.sources_by_id)
        tab._clips = list(project.clips)
        return project, a, b, tab

    def test_switch_keeps_elapsed_time_and_clamps_to_shorter_sequence(self, tab):
        project, a, b, tab = tab
        project.add_sequence(Sequence(name="Short", algorithm="manual", fps=25.0), activate=True)
        short = project.sequence
        project.add_to_sequence(["c0"])
        tab._load_active_sequence()
        assert tab.switch_to_sequence(a.id)
        tab.timeline.set_playhead_time(4.0)
        assert tab.switch_to_sequence(b.id) and project.sequence is b
        assert tab.timeline.get_playhead_time() == pytest.approx(4.0)
        assert tab.switch_to_sequence(short.id) and project.sequence is short
        assert tab.timeline.get_playhead_time() == pytest.approx(short.duration_seconds)
        assert tab.timeline.get_playhead_time() < 4.0
        assert tab.switch_to_sequence("missing") is False

    def test_duplicate_through_tab_uses_spine_and_fills_slot_b(self, tab):
        project, a, b, tab = tab
        tab.comparison_panel.select("a", a.id)
        tab.comparison_panel.select("b", None)
        before = a.to_dict()
        result = tab.duplicate_sequence(a.id)
        assert result["success"] and len(project.sequences) == 3
        assert project.sequence.readable_recipe.parent_id == a.readable_recipe.id
        assert a.to_dict() == before
        assert tab.comparison_panel.selected_ids() == (a.id, project.sequence.id)
        assert tab.sequence_dropdown.currentText() == project.sequence.name

    def test_variation_runs_off_thread_publishes_b_and_leaves_a_untouched(self, qapp, tab):
        project, a, b, tab = tab
        before = a.to_dict()
        tab.comparison_panel.select("a", a.id)
        tab.comparison_panel.select("b", None)
        result = tab.start_variation(a.id, parameters={"reverse": True}, keep_seed=True, name="C")
        assert result["success"] and tab._variation_worker is not None
        worker = tab._variation_worker
        assert not tab.comparison_panel.side_a.regenerate_btn.isEnabled()
        worker.wait(10000)
        deadline = 50
        while tab._variation_worker is not None and deadline:
            qapp.processEvents()
            QTest.qWait(20)
            deadline -= 1
        new = project.sequence
        assert new.name == "C" and new.readable_recipe.parameters["reverse"] is True
        assert new.readable_recipe.parent_id == a.readable_recipe.id and new.readable_recipe.seed == 3
        assert a.to_dict() == before and len(project.sequences) == 3
        assert tab.comparison_panel.selected_ids() == (a.id, new.id)
        assert tab.comparison_panel.side_a.regenerate_btn.isEnabled()
        assert project.session.undo_text == "Generate sequence"

    def test_cancelled_variation_publishes_nothing(self, qapp, tab, monkeypatch):

        project, a, b, tab = tab
        started = threading.Event()
        release = threading.Event()

        def slow_run(definition, candidates, parameters, **kwargs):
            started.set()
            release.wait(5)
            if kwargs["cancel_event"].is_set():
                return None
            raise AssertionError("worker should have been cancelled")

        import core.remix.registry as registry_module
        monkeypatch.setattr(registry_module, "run_algorithm", slow_run)
        tab.comparison_panel.select("a", a.id)
        assert tab.start_variation(a.id, name="Never")["success"]
        assert not tab.comparison_panel.side_a.regenerate_btn.isEnabled()
        worker = tab._variation_worker
        assert started.wait(5)
        assert tab.cancel_variation()
        release.set()
        worker.wait(5000)
        for _ in range(50):
            if tab._variation_worker is None:
                break
            qapp.processEvents()
            QTest.qWait(20)
        assert tab._variation_worker is None
        assert len(project.sequences) == 2 and all(s.name != "Never" for s in project.sequences)
        assert tab.comparison_panel.side_a.regenerate_btn.isEnabled()

    def test_agent_created_variation_shows_up_in_panel(self, tab):
        project, a, b, tab = tab
        assert tab.comparison_panel.isHidden()
        tab.compare_btn.setChecked(True)
        assert not tab.comparison_panel.isHidden()
        assert tab.comparison_panel.selected_ids()[0] == project.sequence.id  # A defaults to the active sequence
        result = regenerate_sequence(project, a.id, name="Agent")
        assert result["success"]
        tab._sync_sequence_dropdown()  # what MainWindow forwards on sequences_changed
        names = [tab.comparison_panel.side_b.selector.itemText(i) for i in range(tab.comparison_panel.side_b.selector.count())]
        assert "Agent" in names
        assert not tab.comparison_panel.empty_label.isVisibleTo(tab.comparison_panel)

    def test_inspect_dialog_renders_recipe(self, qapp, tab):
        from ui.dialogs.recipe_dialogs import RecipeInspectDialog, RegenerateDialog
        from core.remix.registry import registry

        project, a, b, tab = tab
        dialog = RecipeInspectDialog(project, a.id)
        assert "\"algorithm\": \"shuffle\"" in dialog.text.toPlainText()
        project.add_sequence(Sequence(name="Manual", algorithm="manual"), activate=False)
        manual = RecipeInspectDialog(project, project.sequences[-1].id)
        assert manual.text.toPlainText() == ""
        form = RegenerateDialog(registry.require("shuffle"), a.readable_recipe, "A variation")
        form._fields["hflip"].setText("true")
        form._fields["max_consecutive_same_source"].setText("not json")
        with pytest.raises(ValueError):
            form.parameters()
        form._fields["max_consecutive_same_source"].setText("")
        assert form.parameters() == {"hflip": True, "vflip": False, "reverse": False}
        form.explicit_seed.setChecked(True)
        form.seed_spin.setValue(7)
        assert form.seed() == 7 and not form.keeps_seed()


# --- automation walkthrough ---------------------------------------------

@pytest.mark.asyncio
async def test_import_to_variation_to_export_walkthrough_over_mcp_and_cli(tmp_path):
    """U16 verification: import -> generate A -> variation B -> compare -> export both EDLs."""
    import subprocess
    import sys

    from scene_ripper_mcp.tools import export as export_tools
    from scene_ripper_mcp.tools import sequence as mcp_tools

    project = _project(tmp_path)  # stands in for import + detection (covered elsewhere)
    path = tmp_path / "walk.sceneripper"
    assert project.save(path)
    first = json.loads(await mcp_tools.generate_sequence(str(path), "shuffle", seed=5, name="A"))
    assert first["success"]
    second = json.loads(await mcp_tools.regenerate_sequence(
        str(path), first["sequence_id"], parameters={"max_consecutive_same_source": 2}, name="B",
    ))
    assert second["success"] and second["parameters"]["max_consecutive_same_source"] == 2
    comparison = json.loads(await mcp_tools.compare_sequences(str(path), first["sequence_id"], second["sequence_id"]))
    assert comparison["parameter_differences"] == [{"key": "max_consecutive_same_source", "a": 1, "b": 2}]
    assert comparison["related"] and comparison["seed_changed"] is True

    cli = subprocess.run(
        [sys.executable, "-m", "cli.main", "--json", "sequence", "compare", str(path), first["sequence_id"], second["sequence_id"]],
        capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert cli.returncode == 0, cli.stderr
    assert json.loads(cli.stdout)["parameter_differences"][0]["key"] == "max_consecutive_same_source"

    for sequence_id, label in ((first["sequence_id"], "a"), (second["sequence_id"], "b")):
        activated = json.loads(await mcp_tools.activate_sequence(str(path), sequence_id))
        assert activated["success"]
        exported = json.loads(await export_tools.export_edl(str(path), str(tmp_path / f"{label}.edl"), title=label))
        assert exported["success"], exported
        assert (tmp_path / f"{label}.edl").read_text().count("V     C") == 6
