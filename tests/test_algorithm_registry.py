"""Algorithm registry: determinism, seed contract, recipes, and surface parity.

Pilot algorithms are Hatchet Job (``shuffle``) and Chromatics (``color``).
"""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from core.project import Project
from core.remix import generate_sequence, run_registry_algorithm
from core.remix.engine import (
    AlgorithmDefinition, ParameterSpec, ProposedEntry, SequenceProposal,
    legacy_seed, normalize_parameters, resolve_seed, run_algorithm,
)
from core.remix.registry import registry
from models.analysis_record import AnalysisIdentity, AnalysisRecord
from models.clip import Clip, Source
from models.recipe import SequenceRecipe


def _project(count: int = 6, *, sources: int = 2) -> Project:
    project = Project()
    for s in range(sources):
        project.add_source(Source(id=f"s{s}", file_path=Path(f"/tmp/v{s}.mp4"), fps=30.0, duration_seconds=60))
    clips = []
    for i in range(count):
        colors = [(255 - i * 40, i * 40, 30)]
        clip = Clip(id=f"c{i}", source_id=f"s{i % sources}", start_frame=i * 30, end_frame=(i + 1) * 30,
                    dominant_colors=colors)
        clips.append(clip)
    project.add_clips(clips)
    project.mark_clean()
    return project


def _pairs(project: Project):
    return [(clip, project.sources_by_id[clip.source_id]) for clip in project.clips]


# --- Scenario 1: determinism ------------------------------------------------


def test_same_inputs_version_and_seed_reconstruct_deterministic_output():
    project = _project(12)
    first = run_algorithm(registry.require("shuffle"), _pairs(project), {"hflip": True, "reverse": True}, seed=42)
    second = run_algorithm(registry.require("shuffle"), deepcopy(_pairs(project)), {"hflip": True, "reverse": True}, seed=42)
    assert first.recipe.realized == second.recipe.realized
    assert first.recipe.generation_fingerprint == second.recipe.generation_fingerprint
    assert [e.clip_id for e in first.recipe.realized] != [c.id for c in project.clips] or len(project.clips) < 2
    other = run_algorithm(registry.require("shuffle"), _pairs(project), {"hflip": True, "reverse": True}, seed=43)
    assert other.recipe.realized != first.recipe.realized

    color = registry.require("color")
    a = run_algorithm(color, _pairs(project), {"direction": "complementary"})
    b = run_algorithm(color, list(reversed(_pairs(project))), {"direction": "complementary"})
    assert [e.clip_id for e in a.recipe.realized] == [e.clip_id for e in b.recipe.realized]
    assert a.recipe.seed is None and not color.seeded


def test_shuffle_seed_covers_both_order_and_transform_draws():
    project = _project(8)
    run = run_algorithm(registry.require("shuffle"), _pairs(project), {"hflip": True, "vflip": True}, seed=3)
    again = run_algorithm(registry.require("shuffle"), _pairs(project), {"hflip": True, "vflip": True}, seed=3)
    assert [(e.hflip, e.vflip, e.reverse) for e in run.recipe.realized] == [
        (e.hflip, e.vflip, e.reverse) for e in again.recipe.realized
    ]
    assert not any(e.reverse for e in run.recipe.realized)
    assert any(e.hflip or e.vflip for e in run.recipe.realized)


# --- Scenario 2: seed contract ---------------------------------------------


def test_seed_zero_is_explicit_and_legacy_zero_means_draw_fresh():
    shuffle = registry.require("shuffle")
    assert resolve_seed(shuffle, 0) == 0
    drawn = resolve_seed(shuffle, None)
    assert isinstance(drawn, int) and drawn >= 0
    assert legacy_seed(0) is None and legacy_seed(None) is None and legacy_seed(7) == 7
    assert legacy_seed(-4) is None
    with pytest.raises(ValueError, match="non-negative"):
        resolve_seed(shuffle, -1)
    with pytest.raises(ValueError, match="non-negative"):
        resolve_seed(shuffle, True)
    with pytest.raises(ValueError, match="does not use a seed"):
        resolve_seed(registry.require("color"), 5)

    project = _project(8)
    zero = run_algorithm(shuffle, _pairs(project), seed=0)
    assert zero.recipe.seed == 0
    assert run_algorithm(shuffle, _pairs(project), seed=0).recipe.realized == zero.recipe.realized
    fresh = run_algorithm(shuffle, _pairs(project), seed=None)
    assert fresh.recipe.seed is not None  # drawn seed is recorded for replay
    assert run_algorithm(shuffle, _pairs(project), seed=fresh.recipe.seed).recipe.realized == fresh.recipe.realized


def test_legacy_generate_sequence_translates_seed_zero_and_matches_registry():
    project = _project(9)
    pairs = _pairs(project)
    via_legacy = [c.id for c, _ in generate_sequence("shuffle", pairs, len(pairs), seed=11)]
    via_registry = [e.clip_id for e in run_algorithm(registry.require("shuffle"), pairs, seed=11).recipe.realized]
    assert via_legacy == via_registry
    # seed=0 keeps the old "random" meaning: the run still records a real seed.
    run = run_registry_algorithm("shuffle", pairs, seed=0)
    assert run.recipe.seed is not None
    legacy_color = [c.id for c, _ in generate_sequence("color", pairs, len(pairs), direction="warm_to_cool")]
    registry_color = [
        e.clip_id for e in run_algorithm(registry.require("color"), pairs, {"direction": "warm_to_cool"}).recipe.realized
    ]
    assert legacy_color == registry_color


# --- Scenario 3: recipe round-trip -------------------------------------------


def test_recipe_round_trip_retains_selection_analysis_identities_parameters_and_transforms():
    project = _project(4)
    identity = AnalysisIdentity.build(
        operation="colors", sources={"video": "b" * 64}, source_range={"start": 0, "end": 30},
        model={"name": "kmeans"}, parameters={"count": 5}, sampling={},
    )
    project.clips[0].analysis_records["colors"] = AnalysisRecord.success(identity, {"dominant_colors": [[1, 2, 3]]})
    project.clips[1].analysis_records["colors"] = AnalysisRecord.legacy({"dominant_colors": [[1, 2, 3]]})
    pairs = _pairs(project)[:3]
    run = run_algorithm(registry.require("color"), pairs, {"direction": "rainbow", "no_color_handling": "exclude"})
    restored = SequenceRecipe.from_dict(json.loads(json.dumps(run.recipe.to_dict())))
    assert restored == run.recipe
    assert [i.clip_id for i in restored.inputs] == ["c0", "c1", "c2"]
    assert restored.inputs[0].analysis == {"colors": identity.key}
    assert restored.inputs[1].analysis == {"colors": None}  # provenance unknown
    assert restored.inputs[2].analysis == {}
    assert restored.parameters == {"direction": "rainbow", "no_color_handling": "exclude"}
    assert restored.algorithm_version == registry.require("color").version

    shuffle = run_algorithm(registry.require("shuffle"), pairs, {"hflip": True, "vflip": True, "reverse": True}, seed=1)
    restored = SequenceRecipe.from_dict(shuffle.recipe.to_dict())
    assert [(e.hflip, e.vflip, e.reverse) for e in restored.realized] == [
        (e.hflip, e.vflip, e.reverse) for e in shuffle.recipe.realized
    ]


# --- Scenario 4: Qt-free registry with schemas for agent surfaces ------------


def test_registry_import_is_qt_free_and_exposes_schemas_without_ui_imports():
    code = """
import json, sys
from core.remix.registry import registry
from core.spine.sequences import list_algorithms
from core.spine.settings_io import list_sorting_algorithms
forbidden = [m for m in sys.modules if m == 'PySide6' or m.startswith(('PySide6.', 'ui.', 'mpv'))]
assert not forbidden, forbidden
schemas = list_algorithms()['algorithms']
assert [s['key'] for s in schemas] == registry.keys()
print(json.dumps(schemas))
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    schemas = {s["key"]: s for s in json.loads(result.stdout.strip().splitlines()[-1])}
    assert schemas["color"]["seeded"] is False and schemas["shuffle"]["seeded"] is True
    direction = next(p for p in schemas["color"]["parameters"] if p["name"] == "direction")
    assert direction["choices"] == ["rainbow", "warm_to_cool", "cool_to_warm", "complementary"]
    assert {p["name"] for p in schemas["shuffle"]["parameters"]} == {
        "max_consecutive_same_source", "hflip", "vflip", "reverse",
    }


def test_chat_algorithm_listing_carries_engine_schema_for_registry_algorithms():
    from core.spine.settings_io import list_sorting_algorithms

    listing = {a["key"]: a for a in list_sorting_algorithms(_project(2))["algorithms"]}
    assert listing["color"]["engine"]["version"] == 1
    assert listing["shuffle"]["engine"]["seeded"] is True
    assert listing["sequential"]["engine"]["parameters"] == []
    assert listing["storyteller"]["engine"]["provider"] is True


# --- Parameter normalization -------------------------------------------------


def test_parameters_are_normalized_validated_and_unknown_keys_rejected():
    color = registry.require("color")
    assert normalize_parameters(color, None) == {"direction": "rainbow", "no_color_handling": "append_end"}
    assert normalize_parameters(color, {"direction": "complementary"})["direction"] == "complementary"
    with pytest.raises(ValueError, match="must be one of"):
        normalize_parameters(color, {"direction": "sideways"})
    with pytest.raises(ValueError, match="does not accept parameters: seed"):
        normalize_parameters(color, {"seed": 1})
    shuffle = registry.require("shuffle")
    with pytest.raises(ValueError, match="must be true or false"):
        normalize_parameters(shuffle, {"hflip": "yes"})
    with pytest.raises(ValueError, match="at least 1"):
        normalize_parameters(shuffle, {"max_consecutive_same_source": 0})
    with pytest.raises(ValueError, match="must be an integer"):
        normalize_parameters(shuffle, {"max_consecutive_same_source": True})


def test_registry_rejects_duplicate_keys_and_invalid_definitions():
    from core.remix.engine import AlgorithmRegistry

    local = AlgorithmRegistry()

    class Fresh(AlgorithmDefinition):
        key = "fresh"
        parameters = (ParameterSpec("a", "integer", 1), ParameterSpec("a", "integer", 2))

    with pytest.raises(ValueError, match="repeats a parameter"):
        local.register(Fresh())
    Fresh.parameters = (ParameterSpec("a", "integer", 1),)
    local.register(Fresh())
    with pytest.raises(ValueError, match="already registered"):
        local.register(Fresh())
    assert "fresh" in local and "Fresh" in local and "missing" not in local
    with pytest.raises(ValueError, match="not available"):
        local.require("missing")


def test_realization_rejects_foreign_or_repeated_or_out_of_range_entries():
    project = _project(3)
    pairs = _pairs(project)

    class Bad(AlgorithmDefinition):
        key = "bad"
        entries: tuple = ()

        def generate(self, inputs, parameters, rng, context=None):
            return SequenceProposal("ordering", self.entries)

    bad = Bad()
    bad.entries = (ProposedEntry("zz", "s0"),)
    with pytest.raises(ValueError, match="not an input"):
        run_algorithm(bad, pairs)
    bad.entries = (ProposedEntry("c0", "s0"), ProposedEntry("c0", "s0"))
    with pytest.raises(ValueError, match="more than once"):
        run_algorithm(bad, pairs)
    bad.entries = (ProposedEntry("c0", "s0", 0, 31),)
    with pytest.raises(ValueError, match="outside clip"):
        run_algorithm(bad, pairs)
    bad.entries = (ProposedEntry("c0", "s1"),)
    with pytest.raises(ValueError, match="wrong source"):
        run_algorithm(bad, pairs)


def test_prepare_and_select_hooks_run_before_generation_and_honor_cancellation():
    from threading import Event

    project = _project(4)
    pairs = _pairs(project)
    seen = {}

    class Hooked(AlgorithmDefinition):
        key = "hooked"

        def select_inputs(self, candidates, parameters):
            return list(candidates)[:2]

        def prepare(self, inputs, parameters, *, cancel_event=None, progress=None, resources=None):
            seen["prepared"] = [c.id for c, _ in inputs]
            return list(inputs)

        def generate(self, inputs, parameters, rng, context=None):
            return SequenceProposal("ordering", tuple(ProposedEntry(c.id, s.id) for c, s in reversed(inputs)))

    run = run_algorithm(Hooked(), pairs)
    assert seen["prepared"] == ["c0", "c1"]
    assert [e.clip_id for e in run.recipe.realized] == ["c1", "c0"]
    assert [i.clip_id for i in run.recipe.inputs] == ["c0", "c1"]
    cancel = Event()
    cancel.set()
    assert run_algorithm(Hooked(), pairs, cancel_event=cancel) is None


# --- Spine surface: generate, inspect, reconstruct ---------------------------


def test_spine_generation_publishes_recipe_as_one_undoable_edit():
    from core.spine.sequences import generate_sequence as spine_generate, get_sequence_recipe

    project = _project(6)
    project.add_to_sequence(["c0"])
    original = project.sequence
    project.mark_clean()
    result = spine_generate(project, "color", parameters={"direction": "cool_to_warm"}, name="Cool")
    assert result["success"], result
    assert result["recipe_id"] and result["algorithm_version"] == 1 and result["seed"] is None
    generated = project.sequence
    assert generated is not original and generated.name == "Cool"
    assert generated.readable_recipe.id == result["recipe_id"]
    assert [e.source_clip_id for e in generated.get_all_clips()] == result["clip_ids"]
    inspected = get_sequence_recipe(project, generated.id)
    assert inspected["reconstructable"] and inspected["recipe"]["parameters"]["direction"] == "cool_to_warm"
    project.session.undo()
    assert project.sequence is original and len(project.sequences) == 1 and not project.is_dirty
    project.session.redo()
    assert project.sequence.readable_recipe.id == result["recipe_id"]


def test_spine_generation_respects_clip_ids_disabled_clips_and_seed_zero():
    from core.spine.sequences import generate_sequence as spine_generate

    project = _project(6)
    project.set_clips_disabled(["c5"], True)
    everything = spine_generate(project, "shuffle", seed=0)
    assert everything["success"] and "c5" not in everything["clip_ids"] and everything["seed"] == 0
    subset = spine_generate(project, "shuffle", clip_ids=["c1", "c3"], seed=0, parameters={"vflip": True})
    assert subset["success"] and sorted(subset["clip_ids"]) == ["c1", "c3"]
    assert subset["parameters"]["vflip"] is True
    assert not spine_generate(project, "shuffle", clip_ids=["c1", "nope"])["success"]
    assert not spine_generate(project, "shuffle", clip_ids=["c1", "c1"])["success"]
    assert not spine_generate(project, "shuffle", seed=-1)["success"]
    assert "not available through the registry" in spine_generate(project, "no_such_algorithm")["error"]
    bad = spine_generate(project, "color", parameters={"direction": "nope"})
    assert not bad["success"] and "must be one of" in bad["error"]


def test_reconstruction_replays_realized_entries_without_the_algorithm(monkeypatch):
    from core.spine.sequences import generate_sequence as spine_generate, reconstruct_sequence

    project = _project(6)
    result = spine_generate(project, "shuffle", seed=5, parameters={"hflip": True, "reverse": True})
    assert result["success"]
    original = project.sequence
    flags = [(e.hflip, e.vflip, e.reverse) for e in original.get_all_clips()]
    assert flags == [(e.hflip, e.vflip, e.reverse) for e in original.readable_recipe.realized]

    def explode(*args, **kwargs):
        raise AssertionError("reconstruction must not run the algorithm")

    monkeypatch.setattr(type(registry.require("shuffle")), "generate", explode)
    rebuilt = reconstruct_sequence(project, original.id)
    assert rebuilt["success"], rebuilt
    copy = project.sequence
    assert copy is not original and copy.name == f"{original.name} (reconstructed)"
    assert rebuilt["clip_ids"] == result["clip_ids"]
    assert [(e.hflip, e.vflip, e.reverse) for e in copy.get_all_clips()] == flags
    assert copy.readable_recipe.parent_id == original.readable_recipe.id
    assert copy.readable_recipe.realized == original.readable_recipe.realized
    assert original.readable_recipe.id == result["recipe_id"]  # untouched


def test_changed_inputs_give_actionable_reconstruction_errors_without_edits():
    from core.spine.sequences import generate_sequence as spine_generate, get_sequence_recipe, reconstruct_sequence

    project = _project(4)
    assert spine_generate(project, "color")["success"]
    generated = project.sequence
    before = [s.to_dict() for s in project.sequences]
    project.clips_by_id["c1"].end_frame = 45
    inspected = get_sequence_recipe(project)
    assert not inspected["reconstructable"]
    assert any("re-cut" in p for p in inspected["problems"])
    project.mark_clean()
    generation = project.mutation_generation
    result = reconstruct_sequence(project, generated.id)
    assert not result["success"] and "Recipe inputs changed" in result["error"]
    assert "c1" in result["error"]
    assert [s.to_dict() for s in project.sequences] == before
    assert not project.is_dirty and project.mutation_generation == generation


def test_sequences_without_recipes_report_why():
    from core.spine.sequences import get_sequence_recipe, reconstruct_sequence
    from models.sequence import Sequence

    project = _project(2)
    project.add_to_sequence(["c0"])
    assert "no recipe" in get_sequence_recipe(project)["error"]
    assert "no recipe" in reconstruct_sequence(project)["error"]
    future = Sequence.from_dict({"name": "future", "recipe": {"schema_version": 99}})
    project.add_sequence(future, activate=True)
    assert "newer build" in get_sequence_recipe(project)["error"]
    assert not get_sequence_recipe(project, "missing")["success"]


# --- GUI and chat surfaces ---------------------------------------------------


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_sequence_worker_stores_recipe_for_registry_algorithms(qapp):
    from ui.workers.sequence_worker import SequenceWorker

    project = _project(5)
    worker = SequenceWorker("color", _pairs(project), direction="complementary", project=project)
    delivered = []
    worker.sequence_ready.connect(delivered.append)
    worker.run()
    assert delivered and worker.recipe is not None
    assert [c.id for c, _ in delivered[0]] == [e.clip_id for e in worker.recipe.realized]
    assert worker.recipe.parameters["direction"] == "complementary"
    plain = SequenceWorker("sequential", _pairs(project), project=project)
    plain.run()
    assert plain.recipe.algorithm == "sequential" and plain.recipe.parameters == {}


def test_sequence_tab_generate_and_apply_persists_recipe_with_realized_transforms(qapp, monkeypatch):
    from ui.tabs.sequence_tab import SequenceTab

    project = _project(6)
    tab = SequenceTab()
    tab.set_project(project)
    tab.set_available_clips(_pairs(project))
    tab.set_gui_state(SimpleNamespace(analyze_selected_ids=[c.id for c in project.clips], cut_selected_ids=[]))
    monkeypatch.setattr("core.remix.prerender.prerender_batch", lambda clips_with_transforms, output_dir, **kw: [
        (clip, source, None) for clip, source, _ in clips_with_transforms
    ])
    monkeypatch.setattr("core.remix.prerender.get_transform_cache_dir", lambda: Path("/tmp"))
    result = tab.generate_and_apply("shuffle", seed=4, transform_options={"hflip": True, "reverse": True})
    assert result["success"], result
    sequence = project.sequence
    recipe = sequence.readable_recipe
    assert recipe is not None and recipe.seed == 4 and result["recipe_id"] == recipe.id
    assert recipe.parameters["hflip"] and recipe.parameters["reverse"] and not recipe.parameters["vflip"]
    placed = sequence.get_all_clips()
    assert [(e.hflip, e.vflip, e.reverse) for e in placed] == [(e.hflip, e.vflip, e.reverse) for e in recipe.realized]
    assert [e.source_clip_id for e in placed] == [e.clip_id for e in recipe.realized]
    again = tab.generate_and_apply("color", direction="warm_to_cool", no_color_handling="exclude")
    assert again["success"] and project.sequence.readable_recipe.parameters == {
        "direction": "warm_to_cool", "no_color_handling": "exclude",
    }
    assert project.sequence.readable_recipe.seed is None
    tab.close()


def test_dice_roll_worker_records_recipe_and_prerenders_realized_transforms(qapp, monkeypatch):
    from ui.dialogs.dice_roll_dialog import DiceRollWorker

    project = _project(6)
    rendered = []

    def fake_prerender(clips_with_transforms, output_dir, progress_cb=None, cancel_event=None):
        rendered.extend(clips_with_transforms)
        return [(clip, source, None if not any(t.values()) else Path("/tmp/x.mp4")) for clip, source, t in clips_with_transforms]

    monkeypatch.setattr("ui.dialogs.dice_roll_dialog.prerender_batch", fake_prerender)
    monkeypatch.setattr("ui.dialogs.dice_roll_dialog.get_transform_cache_dir", lambda: Path("/tmp"))
    worker = DiceRollWorker(_pairs(project), hflip=True, vflip=False, reverse=False)
    results = []
    worker.finished_sequence.connect(results.append)
    worker.run()
    assert results and worker.recipe is not None
    assert worker.recipe.parameters["hflip"] is True and worker.recipe.seed is not None
    assert [(t["hflip"], t["vflip"], t["reverse"]) for _, _, t in results[0]] == [
        (e.hflip, e.vflip, e.reverse) for e in worker.recipe.realized
    ]
    assert [t["hflip"] for _, _, t in rendered] == [e.hflip for e in worker.recipe.realized]


# --- Review follow-ups (run 20260909-012153) ----------------------------------


def test_reused_empty_sequence_does_not_inherit_a_previous_recipe():
    from core.spine.sequences import SequenceDraft, generate_sequence as spine_generate, get_sequence_recipe

    project = _project(4)
    assert spine_generate(project, "shuffle", seed=1)["success"]
    project.clear_sequence()
    assert project.sequence.readable_recipe is not None  # clearing keeps provenance
    draft = SequenceDraft.prepare(project, "storyteller", "Story")
    assert draft.sequence.recipe is None
    from models.sequence import SequenceClip

    draft.sequence.tracks[0].clips.append(SequenceClip(source_id="s0", source_clip_id="c0", out_point=30))
    draft.commit(project)
    assert project.sequence.algorithm == "storyteller" and project.sequence.recipe is None
    assert "no recipe" in get_sequence_recipe(project)["error"]


def test_generation_and_reconstruction_keep_the_source_frame_rate():
    from core.spine.sequences import generate_sequence as spine_generate, reconstruct_sequence

    project = Project()
    project.add_source(Source(id="s24", file_path=Path("/tmp/v24.mp4"), fps=24.0, duration_seconds=60))
    project.add_clips([
        Clip(id=f"k{i}", source_id="s24", start_frame=i * 25, end_frame=(i + 1) * 25, dominant_colors=[(i * 60, 10, 10)])
        for i in range(4)
    ])
    project.add_to_sequence(["k0"])  # non-empty origin so generation creates a new sequence
    result = spine_generate(project, "color")
    assert result["success"]
    generated = project.sequence
    assert generated.fps == 24.0
    starts = [(e.start_frame, e.in_point, e.out_point) for e in generated.get_all_clips()]
    assert starts[1][0] == 25
    rebuilt = reconstruct_sequence(project, generated.id)
    assert rebuilt["success"]
    assert project.sequence.fps == 24.0
    assert [(e.start_frame, e.in_point, e.out_point) for e in project.sequence.get_all_clips()] == starts


def test_repeated_timeline_clips_collapse_before_registry_generation():
    project = _project(3)
    pairs = _pairs(project)
    run = run_registry_algorithm("color", pairs + pairs[:1], direction="rainbow")
    assert [i.clip_id for i in run.recipe.inputs] == ["c0", "c1", "c2"]
    assert len(generate_sequence("shuffle", pairs + pairs, len(pairs) * 2, seed=2)) == 3


def test_empty_registry_output_is_reported_without_publishing():
    from core.spine.sequences import generate_sequence as spine_generate

    project = _project(3)
    for clip in project.clips:
        clip.dominant_colors = None
    project.mark_clean()
    result = spine_generate(project, "color", parameters={"no_color_handling": "exclude"})
    assert not result["success"] and "empty" in result["error"] and result["notes"]
    assert not project.is_dirty and len(project.sequences) == 1


def test_legacy_hook_and_explicit_parameters_layer_per_definition():
    project = _project(4)
    pairs = _pairs(project)
    run = run_registry_algorithm(
        "shuffle", pairs, transform_options={"hflip": True}, seed=3,
        parameters={"max_consecutive_same_source": 2, "hflip": False},
    )
    assert run.recipe.parameters == {"hflip": False, "max_consecutive_same_source": 2, "reverse": False, "vflip": False}
    color = run_registry_algorithm("color", pairs, direction="complementary", transform_options={"hflip": True})
    assert color.recipe.parameters["direction"] == "complementary"
    assert not any(e.hflip for e in color.recipe.realized)  # transforms mean nothing to Chromatics
    with pytest.raises(ValueError, match="seeded random source"):
        registry.require("shuffle").generate(pairs, {"max_consecutive_same_source": 1, "hflip": False, "vflip": False, "reverse": False}, None, None)


def test_recipe_reports_whether_the_timeline_still_matches():
    from core.spine.sequences import generate_sequence as spine_generate, get_sequence_recipe

    project = _project(4)
    assert spine_generate(project, "shuffle", seed=8)["success"]
    assert get_sequence_recipe(project)["matches_timeline"] is True
    from core.spine.timeline import reorder_clips

    entries = list(project.sequence.tracks[0].clips)
    reorder_clips(project, project.sequence.id, [c.id for c in reversed(entries)])
    assert [c.id for c in project.sequence.tracks[0].clips] == [c.id for c in reversed(entries)]
    inspected = get_sequence_recipe(project)
    assert inspected["matches_timeline"] is False and inspected["reconstructable"] is True


def test_unreadable_recipe_with_nan_keeps_the_sequence_loadable():
    from models.sequence import Sequence

    document = {"schema_version": 1, "algorithm": "shuffle", "algorithm_version": 1, "parameters": {"x": float("nan")},
                "inputs": [], "realized": [], "id": "r", "created_at": "t"}
    sequence = Sequence.from_dict({"name": "nan", "recipe": document})
    assert sequence.readable_recipe is None
    assert sequence.to_dict()["recipe"]["algorithm"] == "shuffle"


def test_chat_generate_remix_reaches_registry_parameters_and_recipe_tools(qapp, monkeypatch):
    from core.chat_tools import generate_remix, get_sequence_recipe, reconstruct_sequence
    from ui.tabs.sequence_tab import SequenceTab

    project = _project(6)
    tab = SequenceTab()
    tab.set_project(project)
    tab.set_available_clips(_pairs(project))
    tab.set_gui_state(SimpleNamespace(analyze_selected_ids=[c.id for c in project.clips], cut_selected_ids=[]))
    main_window = SimpleNamespace(sequence_tab=tab)
    result = generate_remix(project, main_window, "shuffle", clip_count=6, seed=0, max_consecutive_same_source=3)
    assert result["success"], result
    assert result["seed"] is not None  # legacy 0 drew and recorded a seed
    assert project.sequence.readable_recipe.parameters["max_consecutive_same_source"] == 3
    assert not generate_remix(project, main_window, "color", clip_count=6, max_consecutive_same_source=2)["success"]
    colored = generate_remix(project, main_window, "color", clip_count=6, show_chromatic_color_bar=True)
    assert colored["success"] and project.sequence.show_chromatic_color_bar
    inspected = get_sequence_recipe(project)
    assert inspected["success"] and inspected["recipe"]["algorithm"] == "color"
    rebuilt = reconstruct_sequence(project, name="Replay")
    assert rebuilt["success"] and project.sequence.name == "Replay"
    assert rebuilt["clip_ids"] == [c["id"] for c in colored["clips"]]
    tab.close()


def test_chat_color_with_random_transforms_still_draws_transforms(qapp, monkeypatch):
    from ui.tabs.sequence_tab import SequenceTab

    project = _project(8)
    tab = SequenceTab()
    tab.set_project(project)
    tab.set_available_clips(_pairs(project))
    tab.set_gui_state(SimpleNamespace(analyze_selected_ids=[c.id for c in project.clips], cut_selected_ids=[]))
    monkeypatch.setattr("core.remix.prerender.prerender_batch", lambda clips_with_transforms, output_dir, **kw: [
        (clip, source, None) for clip, source, _ in clips_with_transforms
    ])
    monkeypatch.setattr("core.remix.prerender.get_transform_cache_dir", lambda: Path("/tmp"))
    monkeypatch.setattr("random.Random.random", lambda self: 0.1)
    result = tab.generate_and_apply("color", transform_options={"hflip": True})
    assert result["success"], result
    assert all(e.hflip for e in project.sequence.get_all_clips())
    tab.close()


def test_dice_roll_dialog_binds_recipe_to_the_worker_that_finished(qapp):
    from ui.dialogs.dice_roll_dialog import DiceRollDialog

    project = _project(3)
    dialog = DiceRollDialog(_pairs(project))
    delivered = []
    dialog.sequence_ready.connect(delivered.append)
    current = SimpleNamespace(recipe="current")
    stale = SimpleNamespace(recipe="stale")
    dialog._worker = current
    dialog._on_finished([("a", "b", {})], stale)
    assert not delivered and dialog.recipe is None
    dialog._on_finished([("a", "b", {})], current)
    assert delivered and dialog.recipe == "current"
