"""Recipe model: validation, round-trips, fingerprints, and sequence persistence."""

from pathlib import Path

import pytest

from models.clip import Clip, Source
from models.recipe import (
    RECIPE_SCHEMA_VERSION,
    RealizedEntry,
    RecipeInput,
    SequenceRecipe,
    UnreadableRecipe,
    load_recipe,
)
from models.sequence import Sequence


def _inputs(count: int = 3) -> tuple[RecipeInput, ...]:
    return tuple(
        RecipeInput(f"c{i}", "s1", i * 30, (i + 1) * 30, 30.0, {"colors": "a" * 64})
        for i in range(count)
    )


def _recipe(**overrides) -> SequenceRecipe:
    inputs = _inputs()
    fields = dict(
        algorithm="shuffle",
        algorithm_version=1,
        parameters={"hflip": True, "max_consecutive_same_source": 1},
        inputs=inputs,
        realized=(
            RealizedEntry("c2", "s1", 0, 30, hflip=True),
            RealizedEntry("c0", "s1", 5, 20, rationale="opens wide", provider_output={"score": 0.5}),
            RealizedEntry("c1", "s1", 0, 30),
        ),
        seed=0,
    )
    fields.update(overrides)
    return SequenceRecipe(**fields)


class TestRoundTrip:
    def test_round_trip_retains_inputs_analysis_parameters_and_realized_transforms(self):
        recipe = _recipe()
        restored = SequenceRecipe.from_dict(recipe.to_dict())
        assert restored == recipe
        assert restored.inputs[0].analysis == {"colors": "a" * 64}
        assert restored.realized[0].hflip and not restored.realized[0].vflip
        assert restored.realized[1].relative_range == (5, 20)
        assert restored.realized[1].rationale == "opens wide"
        assert restored.realized[1].provider_output == {"score": 0.5}
        assert restored.parameters == {"hflip": True, "max_consecutive_same_source": 1}
        assert restored.seed == 0

    def test_seed_zero_is_an_explicit_seed(self):
        recipe = _recipe(seed=0)
        assert recipe.seed == 0
        assert SequenceRecipe.from_dict(recipe.to_dict()).seed == 0
        assert _recipe(seed=None).seed is None

    def test_fingerprints_ignore_identity_but_track_generation_inputs(self):
        base = _recipe()
        same = _recipe(id="other-id", created_at="2026-01-01T00:00:00+00:00")
        assert base.generation_fingerprint == same.generation_fingerprint
        assert base.input_fingerprint == same.input_fingerprint
        assert _recipe(seed=1).generation_fingerprint != base.generation_fingerprint
        assert _recipe(parameters={"hflip": False, "max_consecutive_same_source": 1}).generation_fingerprint != base.generation_fingerprint
        assert _recipe(algorithm_version=2).generation_fingerprint != base.generation_fingerprint
        reordered = _recipe(inputs=tuple(reversed(_inputs())))
        assert reordered.input_fingerprint != base.input_fingerprint

    def test_derive_creates_a_child_with_new_identity(self):
        parent = _recipe()
        child = parent.derive(seed=9)
        assert child.parent_id == parent.id
        assert child.id != parent.id
        assert child.seed == 9
        assert child.realized == parent.realized

    def test_uses_provider_reflects_stored_outputs(self):
        assert _recipe().uses_provider  # entry-level provider output
        plain = _recipe(realized=(RealizedEntry("c0", "s1", 0, 30),))
        assert not plain.uses_provider
        assert _recipe(realized=(RealizedEntry("c0", "s1", 0, 30),), provider_outputs={"poem": "x"}).uses_provider


class TestValidation:
    def test_realized_entries_must_come_from_inputs(self):
        with pytest.raises(ValueError, match="was not a recipe input"):
            _recipe(realized=(RealizedEntry("zz", "s1", 0, 30),))
        with pytest.raises(ValueError, match="different source"):
            _recipe(realized=(RealizedEntry("c0", "s2", 0, 30),))
        with pytest.raises(ValueError, match="extends past"):
            _recipe(realized=(RealizedEntry("c0", "s1", 0, 31),))

    def test_inputs_cannot_repeat_and_ranges_must_be_ordered(self):
        with pytest.raises(ValueError, match="repeat"):
            _recipe(inputs=_inputs() + (_inputs()[0],))
        with pytest.raises(ValueError, match="exceed"):
            RecipeInput("c", "s", 10, 10, 30.0)
        with pytest.raises(ValueError, match="exceed"):
            RealizedEntry("c", "s", 5, 5)

    def test_parameters_and_provider_outputs_must_be_json(self):
        with pytest.raises(ValueError, match="JSON"):
            _recipe(parameters={"weird": object()})
        with pytest.raises(ValueError, match="JSON"):
            _recipe(parameters={"nan": float("nan")})
        with pytest.raises(ValueError, match="JSON"):
            RealizedEntry("c", "s", 0, 1, provider_output=object())

    def test_analysis_identities_must_be_sha256_or_none(self):
        RecipeInput("c", "s", 0, 1, 30.0, {"colors": None})
        with pytest.raises(ValueError, match="SHA-256"):
            RecipeInput("c", "s", 0, 1, 30.0, {"colors": "abc"})
        with pytest.raises(ValueError, match="lowercase"):
            RecipeInput("c", "s", 0, 1, 30.0, {"Colors": None})

    def test_future_schema_is_rejected_by_the_model_but_preserved_by_the_loader(self):
        document = _recipe().to_dict()
        document["schema_version"] = RECIPE_SCHEMA_VERSION + 1
        with pytest.raises(ValueError, match="schema version"):
            SequenceRecipe.from_dict(document)
        stored = load_recipe(document)
        assert isinstance(stored, UnreadableRecipe)
        assert stored.to_dict() == document

    def test_loader_returns_none_for_absent_and_preserves_garbage(self):
        assert load_recipe(None) is None
        stored = load_recipe({"nonsense": True})
        assert isinstance(stored, UnreadableRecipe)
        assert stored.to_dict() == {"nonsense": True}


class TestSequencePersistence:
    def test_sequence_serializes_and_restores_its_recipe(self):
        recipe = _recipe()
        sequence = Sequence(name="Shuffle", algorithm="shuffle", recipe=recipe)
        data = sequence.to_dict()
        assert data["recipe"]["algorithm"] == "shuffle"
        restored = Sequence.from_dict(data)
        assert restored.readable_recipe == recipe
        assert restored.recipe == recipe

    def test_sequence_without_recipe_omits_the_key_and_loads_none(self):
        sequence = Sequence(name="Manual")
        assert "recipe" not in sequence.to_dict()
        assert Sequence.from_dict(sequence.to_dict()).recipe is None
        assert Sequence.from_dict({"name": "old"}).readable_recipe is None

    def test_unreadable_recipe_round_trips_verbatim_and_is_hidden_from_readers(self):
        document = _recipe().to_dict()
        document["schema_version"] = 99
        sequence = Sequence.from_dict({"name": "future", "recipe": document})
        assert sequence.readable_recipe is None
        assert sequence.to_dict()["recipe"] == document

    def test_project_save_and_load_keeps_recipes(self, tmp_path):
        from core.project import Project

        source = Source(id="s1", file_path=tmp_path / "offline.mp4", fps=30.0, duration_seconds=3)
        clips = [Clip(id=f"c{i}", source_id="s1", start_frame=i * 30, end_frame=(i + 1) * 30) for i in range(3)]
        project = Project(sources=[source], clips=clips, sequences=[Sequence(name="Recipe", recipe=_recipe())])
        path = tmp_path / "p.sceneripper"
        assert project.save(path)
        loaded = Project.load(path)
        assert loaded.sequences[0].readable_recipe == _recipe(
            id=loaded.sequences[0].readable_recipe.id,
            created_at=loaded.sequences[0].readable_recipe.created_at,
        )
        assert Path(path).exists()
