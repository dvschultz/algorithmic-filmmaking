"""Tests for SequenceClip.rationale persistence (Free Association sequencer)."""

import tempfile
from pathlib import Path

from models.clip import Clip, Source
from core.project import Project
from models.sequence import Sequence, SequenceClip, Track
from ui.algorithm_config import get_algorithm_config


def test_rationale_defaults_to_none():
    """New SequenceClips have no rationale by default."""
    clip = SequenceClip(source_clip_id="c1", source_id="s1")
    assert clip.rationale is None


def test_rationale_to_dict_omitted_when_none():
    """rationale is not included in to_dict when None (backward-compatible)."""
    clip = SequenceClip(source_clip_id="c1", source_id="s1")
    data = clip.to_dict()
    assert "rationale" not in data


def test_rationale_to_dict_included_when_set():
    """rationale is included in to_dict when set."""
    clip = SequenceClip(
        source_clip_id="c1",
        source_id="s1",
        rationale="Both clips share close-up framing with warm tones",
    )
    data = clip.to_dict()
    assert data["rationale"] == "Both clips share close-up framing with warm tones"


def test_rationale_roundtrip():
    """rationale survives to_dict/from_dict roundtrip."""
    clip = SequenceClip(
        source_clip_id="c1",
        source_id="s1",
        rationale="The transition preserves visual rhythm",
    )
    data = clip.to_dict()
    restored = SequenceClip.from_dict(data)
    assert restored.rationale == "The transition preserves visual rhythm"


def test_rationale_backward_compatible_load():
    """Old project data without rationale loads with rationale=None."""
    data = {
        "id": "sc-1",
        "source_clip_id": "c1",
        "source_id": "s1",
        "track_index": 0,
        "start_frame": 0,
        "in_point": 0,
        "out_point": 30,
    }
    clip = SequenceClip.from_dict(data)
    assert clip.rationale is None


def test_rationale_survives_full_project_roundtrip():
    """rationale persists through a full project save/load cycle."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        video = tmp / "video.mp4"
        video.touch()
        project_path = tmp / "project.json"

        # Build project with a sequence containing a rationale-bearing clip
        project = Project.new(name="Rationale Roundtrip")
        source = Source(
            id="src-1",
            file_path=video,
            duration_seconds=60.0,
            fps=30.0,
            width=1920,
            height=1080,
        )
        project.add_source(source)
        clip = Clip(id="clip-1", source_id="src-1", start_frame=0, end_frame=30)
        project.add_clips([clip])

        sequence = Sequence(name="Test", fps=30.0, algorithm="free_association")
        sequence.tracks[0].add_clip(
            SequenceClip(
                source_clip_id="clip-1",
                source_id="src-1",
                start_frame=0,
                in_point=0,
                out_point=30,
                rationale="Opening shot, user-selected",
            )
        )
        project.sequence = sequence
        assert project.save(project_path) is True

        # Load back and verify rationale survived
        loaded = Project.load(project_path)
        assert loaded.sequence is not None
        assert loaded.sequence.algorithm == "free_association"
        loaded_clips = loaded.sequence.get_all_clips()
        assert len(loaded_clips) == 1
        assert loaded_clips[0].rationale == "Opening shot, user-selected"


def test_free_association_registered_in_algorithm_config():
    """Free Association algorithm appears in ALGORITHM_CONFIG with is_dialog=True."""
    config = get_algorithm_config("free_association")
    assert config["label"] == "Free Association"
    assert config["is_dialog"] is True
    assert config["required_analysis"] == ["describe"]
    # embeddings is NOT required — the core module falls back to random
    # sampling when embeddings are missing, so gating on them would contradict
    # graceful degradation
    assert "embeddings" not in config["required_analysis"]
