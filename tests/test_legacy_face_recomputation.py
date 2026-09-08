"""Legacy face vectors require verified recomputation, not model relabelling."""

import json
from types import SimpleNamespace

import pytest

from core.project import Project
from core.settings import Settings
from models.analysis_record import AnalysisRecord
from tests.test_face_records import setup as face_setup  # noqa: F401


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["spine", "cli", "gui", "mcp"])
@pytest.mark.parametrize("empty", [False, True])
async def test_legacy_faces_recompute_across_surfaces(request, tmp_path, monkeypatch, surface, empty):
    from core.analysis_availability import operation_is_complete_for_clip

    project, provider, _ = request.getfixturevalue("face_setup")
    settings = Settings(cache_dir=tmp_path, model_cache_dir=tmp_path)
    monkeypatch.setattr("core.settings.load_settings", lambda: settings)
    clip = project.clips[0]
    legacy = [] if empty else [{"bbox": [1, 2, 3, 4], "embedding": [0.5] * 512, "confidence": 0.9, "frame_number": clip.start_frame}]
    clip.face_embeddings = legacy
    clip.analysis_records["face_embeddings"] = AnalysisRecord.legacy({"face_embeddings": legacy})
    path = tmp_path / "legacy.sceneripper"
    assert project.save(path)
    assert not operation_is_complete_for_clip("face_embeddings", clip, source=project.sources[0])
    if surface == "spine":
        from core.spine.analyze import face_embeddings
        result = face_embeddings(project)
        assert result["success"], result
    elif surface == "cli":
        from click.testing import CliRunner
        from cli.commands.analyze import analyze
        result = CliRunner().invoke(analyze, ["faces", str(path)], obj={})
        assert result.exit_code == 0, result.output
        project = Project.load(path)
    elif surface == "gui":
        from tests.test_gui_face_records import run
        _, outcomes = run(project, apply=True)
        assert outcomes[0].status == "succeeded"
    else:
        from core.jobs import JobRuntime, JobStore
        from scene_ripper_mcp.tools.jobs import start_detect_faces
        from scene_ripper_mcp.tests.test_jobs_tools import _wait_for_status
        store = JobStore(tmp_path / "mcp-jobs.db")
        runtime = JobRuntime(store, max_workers=1)
        ctx = SimpleNamespace(request_context=SimpleNamespace(lifespan_context={"job_store": store, "job_runtime": runtime}))
        try:
            result = json.loads(await start_detect_faces(str(path), ctx=ctx))
            assert result["success"], result
            _wait_for_status(store, result["task_id"], "completed")
            project = Project.load(path)
        finally:
            runtime.shutdown(wait=True)
            store.close()
    assert provider.call_count == 1
    record = project.clips[0].analysis_records["face_embeddings"]
    assert record.provenance == "verified" and not record.legacy_reuse
    assert record.identity.to_dict()["model"]["components"]
    assert operation_is_complete_for_clip("face_embeddings", project.clips[0], source=project.sources[0])
    project.close_writer()


def test_failed_face_recomputation_preserves_legacy_vectors(request):
    from core.spine.analyze import face_embeddings

    project, provider, _ = request.getfixturevalue("face_setup")
    legacy = [{"bbox": [1, 2, 3, 4], "embedding": [0.5] * 512, "confidence": 0.9, "frame_number": project.clips[0].start_frame}]
    project.clips[0].face_embeddings = legacy
    provider.side_effect = RuntimeError("model unavailable")
    face_embeddings(project)
    assert project.clips[0].face_embeddings == legacy
    assert project.clips[0].analysis_records["face_embeddings"].state == "failed"
