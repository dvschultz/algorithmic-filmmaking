"""U12: every algorithm runs through the registry and reconstructs without providers.

Provider calls (LLM, VLM, face models, music analysis) are replaced by fakes
that count invocations, so the tests prove that reconstruction never calls
them and regeneration records a new run.
"""

from copy import deepcopy
from pathlib import Path
import pytest

from core.project import Project
from core.remix.registry import registry, run_algorithm
from core.spine.sequences import (
    generate_sequence, get_sequence_recipe, reconstruct_sequence, regenerate_sequence,
)
from core.transcription_models import TranscriptSegment, WordTimestamp
from models.cinematography import CinematographyAnalysis
from models.clip import Clip, Source
from models.recipe import SequenceRecipe

EXPECTED_ALGORITHMS = {
    "color", "duration", "brightness", "volume", "shuffle", "sequential", "shot_type", "proximity",
    "similarity_chain", "match_cut", "exquisite_corpus", "storyteller", "free_association",
    "cassette_tape", "reference_guided", "signature_style", "rose_hobart", "staccato",
    "gaze_sort", "gaze_consistency", "eyes_without_a_face", "word_sequencer", "word_llm_composer",
}


def _embedding(i: int, dims: int = 8) -> list[float]:
    vector = [0.0] * dims
    vector[i % dims] = 1.0
    vector[(i + 1) % dims] = 0.5
    norm = sum(v * v for v in vector) ** 0.5
    return [v / norm for v in vector]


def _rich_project(tmp_path: Path) -> Project:
    """Six clips over two sources carrying every analysis the algorithms read."""
    project = Project()
    for s in range(2):
        path = tmp_path / f"v{s}.mp4"
        path.write_bytes(b"0")
        project.add_source(Source(id=f"s{s}", file_path=path, fps=24.0, duration_seconds=60))
    shots = ["wide shot", "medium shot", "close-up", "extreme close-up", "full shot", "wide shot"]
    gaze = ["looking_left", "looking_right", "at_camera", "looking_up", "looking_down", "at_camera"]
    clips = []
    for i in range(6):
        clips.append(Clip(
            id=f"c{i}", source_id=f"s{i % 2}", start_frame=i * 48, end_frame=(i + 1) * 48,
            dominant_colors=[(255 - i * 40, i * 40, 30)], shot_type=shots[i],
            average_brightness=0.1 * (i + 1), rms_volume=-40.0 + i * 5,
            embedding=_embedding(i), first_frame_embedding=_embedding(i), last_frame_embedding=_embedding(i + 1),
            embedding_model="dinov2-vit-b-14",
            gaze_yaw=-30.0 + i * 12, gaze_pitch=-10.0 + i * 4, gaze_category=gaze[i],
            description=f"Clip {i} shows scene {i}",
            extracted_texts=None,
            cinematography=CinematographyAnalysis(shot_size="MS") if i % 2 else None,
            transcript=[TranscriptSegment(
                start_time=0.0, end_time=2.0, text=f"hello world {i}",
                words=[WordTimestamp(0.0, 0.5, "hello"), WordTimestamp(0.6, 1.0, "world"), WordTimestamp(1.1, 1.6, f"n{i}")],
            )],
        ))
    project.add_clips(clips)
    project.mark_clean()
    return project


class Fakes:
    """Provider stand-ins that count calls."""

    def __init__(self):
        self.calls = 0

    def poem(self, clips_with_text, mood, model=None, length="medium", form="free_verse"):
        from core.remix.exquisite_corpus import PoemLine

        self.calls += 1
        return [PoemLine(text=text, clip_id=clip.id, line_number=i + 1) for i, (clip, text) in enumerate(reversed(clips_with_text))]

    def narrative(self, clips_with_descriptions, target_duration_minutes, narrative_structure, theme=None, model=None):
        from core.remix.storyteller import NarrativeLine

        self.calls += 1
        return [NarrativeLine(clip.id, desc, "beat", i + 1) for i, (clip, desc) in enumerate(clips_with_descriptions[::-1])]

    def propose(self, current, candidates, recent, rejected, model=None, temperature=None):
        self.calls += 1
        return candidates[0][0], f"because {candidates[0][1][:10]}"

    def compose(self, inventory, **kwargs):
        from core.spine.words import WordInstance

        self.calls += 1
        instances = []
        for key in sorted(inventory.by_word)[: kwargs.get("target_length", 3)]:
            instances.append(inventory.by_word[key][0])
        return [w if isinstance(w, WordInstance) else w for w in instances]

    def match_person(self, reference_paths, clips, **kwargs):
        from core.remix.rose_hobart import PersonMatch, PersonMatches

        self.calls += 1
        return PersonMatches(
            tuple(PersonMatch(clip.id, 0.9 - 0.1 * i) for i, (clip, _) in enumerate(clips) if i % 2 == 0),
            {"runtime": {"packages": {}}}, {p: [1, 2] for p in reference_paths}, None, (),
        )

    def audio(self):
        from core.analysis.audio import AudioAnalysis

        return AudioAnalysis(
            tempo_bpm=120.0, beat_times=[0.0, 1.0, 2.0, 3.0], onset_times=[0.0, 0.5, 1.5, 2.5],
            onset_strengths=[1.0, 0.5, 0.8, 0.2], downbeat_times=[0.0, 2.0], duration_seconds=4.0,
        )


@pytest.fixture
def fakes(monkeypatch, tmp_path):
    fake = Fakes()
    monkeypatch.setattr("core.remix.exquisite_corpus.generate_poem", fake.poem)
    monkeypatch.setattr("core.remix.storyteller.generate_narrative", fake.narrative)
    monkeypatch.setattr("core.remix.free_association.propose_next_clip", fake.propose)
    monkeypatch.setattr("core.spine.words.compose_with_llm", fake.compose)
    monkeypatch.setattr("core.remix.rose_hobart.match_person", fake.match_person)
    monkeypatch.setattr("core.analysis.audio.analyze_music_file", lambda path: fake.audio())
    for name in ("_auto_compute_brightness", "_auto_compute_volume", "_auto_compute_embeddings", "_auto_compute_boundary_embeddings"):
        monkeypatch.setattr(f"core.remix.{name}", lambda clips, **kwargs: deepcopy(list(clips)))
    return fake


def _assets(tmp_path: Path) -> dict:
    from PIL import Image

    drawing = tmp_path / "drawing.png"
    image = Image.new("RGB", (120, 40), "white")
    for x in range(10, 60):
        for y in range(5, 25):
            image.putpixel((x, y), (220, 40, 40))
    for x in range(70, 110):
        for y in range(15, 35):
            image.putpixel((x, y), (40, 40, 220))
    image.save(drawing)
    music = tmp_path / "music.wav"
    music.write_bytes(b"RIFF")
    face = tmp_path / "face.jpg"
    face.write_bytes(b"\xff\xd8")
    return {"drawing": str(drawing), "music": str(music), "face": str(face)}


def parameters_for(key: str, project: Project, assets: dict) -> dict:
    return {
        "reference_guided": {"reference_source_id": "s1", "weights": {"color": 1.0, "duration": 0.5}, "allow_repeats": False},
        "rose_hobart": {"reference_image_paths": [assets["face"]], "ordering": "confidence"},
        "staccato": {"music_path": assets["music"], "strategy": "beats"},
        "cassette_tape": {"phrases": [{"phrase": "hello world", "count": 2}]},
        "word_sequencer": {"mode": "by_frequency"},
        "word_llm_composer": {"prompt": "sing", "target_length": 3},
        "exquisite_corpus": {"mood": "calm"},
        "storyteller": {"structure": "three_act"},
        "free_association": {"max_clips": 3},
        "signature_style": {"drawing_path": assets["drawing"], "mode": "parametric", "total_duration_seconds": 8.0, "fps": 24.0},
        "eyes_without_a_face": {"mode": "gaze_rotation", "axis": "yaw", "range_start": -40, "range_end": 40},
    }.get(key, {})


def test_every_matrix_algorithm_is_registered():
    from ui.algorithm_config import ALGORITHM_CONFIG

    assert set(registry.keys()) == EXPECTED_ALGORITHMS == set(ALGORITHM_CONFIG)


@pytest.mark.parametrize("key", sorted(EXPECTED_ALGORITHMS))
def test_every_algorithm_executes_headlessly_and_persists_a_reconstructable_recipe(key, tmp_path, fakes, monkeypatch):
    """Scenario 1 + 2: executable without its dialog; reconstruction is provider-free."""
    project = _rich_project(tmp_path)
    for clip in project.clips:
        clip.extracted_texts = None
        clip.combined_text_override = None
    monkeypatch.setattr(type(project.clips[0]), "combined_text", property(lambda self: f"text of {self.id}"), raising=False)
    params = parameters_for(key, project, _assets(tmp_path))
    seed = 5 if registry.require(key).seeded else None
    result = generate_sequence(project, key, parameters=params, seed=seed, name=f"{key} run")
    assert result["success"], (key, result)
    assert result["clip_count"] >= 1, key
    generated = project.sequence
    recipe = generated.readable_recipe
    assert recipe is not None and recipe.algorithm == key
    assert SequenceRecipe.from_dict(recipe.to_dict()) == recipe
    assert get_sequence_recipe(project, generated.id)["reconstructable"]
    saved = tmp_path / f"{key}.sceneripper"
    assert project.save(saved)
    loaded = Project.load(saved)
    stored = next(s for s in loaded.sequences if s.id == generated.id).readable_recipe
    assert stored == recipe
    placed = [(e.source_clip_id, e.in_point, e.out_point, e.hflip, e.vflip, e.reverse) for e in generated.get_all_clips()]
    calls_before = fakes.calls
    rebuilt = reconstruct_sequence(loaded, generated.id)
    assert rebuilt["success"], rebuilt
    assert fakes.calls == calls_before  # no provider call
    rebuilt_entries = [(e.source_clip_id, e.in_point, e.out_point, e.hflip, e.vflip, e.reverse) for e in loaded.sequence.get_all_clips()]
    assert rebuilt_entries == placed
    assert loaded.sequence.fps == generated.fps == 24.0
    if registry.require(key).provider:
        assert recipe.uses_provider or recipe.provider_outputs


@pytest.mark.parametrize("key", ["storyteller", "exquisite_corpus", "free_association", "word_llm_composer"])
def test_regenerating_an_llm_edit_calls_the_provider_again_and_records_a_new_run(key, tmp_path, fakes, monkeypatch):
    project = _rich_project(tmp_path)
    monkeypatch.setattr(type(project.clips[0]), "combined_text", property(lambda self: f"text of {self.id}"), raising=False)
    params = parameters_for(key, project, _assets(tmp_path))
    first = generate_sequence(project, key, parameters=params, seed=1 if registry.require(key).seeded else None)
    assert first["success"], first
    original = project.sequence
    before = original.to_dict()
    calls = fakes.calls
    variation = regenerate_sequence(project, original.id)
    assert variation["success"], variation
    assert fakes.calls > calls  # provider consulted again (free association asks per step)
    assert project.sequence is not original and original.to_dict() == before
    assert project.sequence.readable_recipe.parent_id == original.readable_recipe.id
    assert project.sequence.readable_recipe.id != original.readable_recipe.id


def test_dialog_free_recipes_match_the_engine_run_directly(tmp_path, fakes):
    """The spine and a direct engine run agree for a pure algorithm."""
    project = _rich_project(tmp_path)
    pairs = [(c, project.sources_by_id[c.source_id]) for c in project.clips]
    direct = run_algorithm(registry.require("color"), deepcopy(pairs), {"direction": "warm_to_cool"})
    via_spine = generate_sequence(project, "color", parameters={"direction": "warm_to_cool"})
    assert via_spine["clip_ids"] == [e.clip_id for e in direct.recipe.realized]
    assert via_spine["parameters"] == direct.recipe.parameters
