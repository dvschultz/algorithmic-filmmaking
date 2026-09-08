"""In-memory projections cannot establish completion after artifact loss."""

import pytest

from core.analysis_availability import operation_is_complete_for_clip
from core.artifacts import ArtifactStore
from core.cost_estimates import estimate_sequence_cost
from core.project import Project
from models.clip import Clip
from tests.analysis_fixtures import verify_clip_analysis


@pytest.mark.parametrize(
    "operation,options",
    [
        ("embeddings", {"embeddings": True}),
        ("boundary_embeddings", {"boundary": True}),
    ],
)
@pytest.mark.parametrize("damage", ["missing", "changed"])
def test_embedding_completion_requires_current_artifact(
    tmp_path, monkeypatch, operation, options, damage
):
    monkeypatch.setattr(
        "core.paths.get_artifact_store_dir", lambda: tmp_path / "artifacts"
    )
    clip = Clip(source_id="source", start_frame=0, end_frame=30)
    source = verify_clip_analysis(clip, tmp_path, **options)
    project = Project(sources=[source], clips=[clip])
    path = tmp_path / "project.sceneripper"
    project.save(path)
    restored = Project.load(path)
    clip, source = restored.clips[0], restored.sources[0]
    record = clip.analysis_records[operation]
    assert record.artifact is not None
    assert operation_is_complete_for_clip(operation, clip, source=source)
    payload_path = ArtifactStore().path_for(record.artifact)
    if damage == "missing":
        payload_path.unlink()
    else:
        payload_path.write_bytes(b"x" * payload_path.stat().st_size)

    def blocked(*args, **kwargs):
        raise AssertionError("Completion must not hash artifact payloads")

    monkeypatch.setattr(ArtifactStore, "read_bytes", blocked)
    monkeypatch.setattr(ArtifactStore, "path_for", blocked)
    assert not operation_is_complete_for_clip(operation, clip, source=source)
    estimates = estimate_sequence_cost(
        "test", [clip], override_required=[operation], sources_by_id={source.id: source}
    )
    assert estimates[0].clips_needing == 1
