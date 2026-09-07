from threading import Event
from unittest.mock import Mock
import pytest
from tests.test_description_operations import project_with_thumbnails
from core.operations.boundary_embeddings import (
    BoundaryEmbeddingApplication,
    BoundaryEmbeddingTask,
    BoundaryEmbeddingOutcome,
)
from core.jobs.boundary_embeddings import run_boundary_embedding_job
from core.jobs.store import JobStore


@pytest.mark.parametrize("surface", ["gui", "headless"])
@pytest.mark.parametrize("old_model", ["clip-vit-b-32", None])
def test_preserve_thumbnail_model(tmp_path, monkeypatch, surface, old_model):
    p = project_with_thumbnails(tmp_path, 1)
    c, s = p.clips[0], p.sources[0]
    c.embedding = [0.3] * 512
    c.embedding_model = old_model
    p.save(tmp_path / "project.json")
    monkeypatch.setattr(
        "core.analysis.embeddings.extract_boundary_embeddings",
        Mock(return_value=([0.1] * 768, [0.2] * 768)),
    )
    monkeypatch.setattr("core.analysis.embeddings.unload_model", Mock())
    if surface == "gui":
        app = BoundaryEmbeddingApplication(
            p,
            (
                BoundaryEmbeddingTask(
                    c.id, s.file_path, c.start_frame, c.end_frame, s.fps
                ),
            ),
        )
        with pytest.raises(ValueError, match="thumbnail"):
            app.apply(
                p,
                BoundaryEmbeddingOutcome(
                    c.id, "succeeded", (0.1,) * 768, (0.2,) * 768, "dinov2-vit-b-14"
                ),
            )
        assert c.embedding_model == old_model
    else:
        store = JobStore(tmp_path / "jobs.db")
        try:
            with pytest.raises(ValueError, match="thumbnail"):
                run_boundary_embedding_job(
                    store, p.path, None, lambda *_: None, Event()
                )
        finally:
            store.close()
        from core.project import Project

        saved = Project.load(p.path)
        assert saved.clips[0].embedding_model == old_model
        assert saved.clips[0].first_frame_embedding is None
