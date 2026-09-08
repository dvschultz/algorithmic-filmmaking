"""Explicit reuse retains unknown provenance and binds current media inputs."""

from unittest.mock import patch

import pytest

from core.analysis_availability import operation_is_complete_for_clip
from tests.test_description_operations import project_with_thumbnails


@pytest.mark.parametrize("operation", ["colors", "embeddings"])
def test_explicit_reuse_skips_inference_and_retains_unknown_provenance(tmp_path, operation):
    from core.spine.analysis_reuse import accept_legacy_analysis
    from core.spine.analyze import analyze_colors, embeddings
    from core.analysis_model_identity import DINOV2_TAG

    project = project_with_thumbnails(tmp_path, 1)
    clip = project.clips[0]
    clip.dominant_colors = [(1, 2, 3)]
    clip.embedding = [0.1] * 768
    clip.embedding_model = DINOV2_TAG
    assert not operation_is_complete_for_clip(operation, clip, source=project.sources[0])
    result = accept_legacy_analysis(project, operation, [clip.id])
    assert result["accepted"] == [clip.id]
    record = clip.analysis_records[operation]
    assert record.provenance == "unknown" and record.legacy_reuse
    assert operation_is_complete_for_clip(operation, clip, source=project.sources[0])
    with patch("core.analysis.color.extract_dominant_colors", side_effect=AssertionError("no inference")), patch("core.analysis.embeddings.extract_clip_embeddings_batch", side_effect=AssertionError("no inference")):
        reused = analyze_colors(project) if operation == "colors" else embeddings(project)
    assert len(reused["result"]["skipped"]) == 1
    assert clip.analysis_records[operation].provenance == "unknown"
    clip.start_frame += 1
    assert not operation_is_complete_for_clip(operation, clip, source=project.sources[0])


def test_verified_record_cannot_be_relabelled_as_legacy(tmp_path):
    from core.spine.analysis_reuse import accept_legacy_analysis
    from core.spine.analyze import analyze_colors

    project = project_with_thumbnails(tmp_path, 1)
    with patch("core.analysis.color.extract_dominant_colors", return_value=[(1, 2, 3)]):
        analyze_colors(project)
    previous = project.clips[0].analysis_records["colors"]
    result = accept_legacy_analysis(project, "colors")
    assert not result["accepted"] and len(result["failed"]) == 1
    assert project.clips[0].analysis_records["colors"] == previous


def test_cli_reuse_requires_an_explicit_command_and_saves_unknown_provenance(tmp_path):
    from click.testing import CliRunner
    from cli.commands.analyze import analyze
    from core.project import Project

    project = project_with_thumbnails(tmp_path, 1)
    project.clips[0].dominant_colors = [(1, 2, 3)]
    path = tmp_path / "project.json"
    assert project.save(path)
    result = CliRunner().invoke(analyze, ["accept-legacy", str(path), "--operation", "colors"])
    assert result.exit_code == 0, result.output
    loaded = Project.load(path)
    record = loaded.clips[0].analysis_records["colors"]
    assert record.provenance == "unknown" and record.legacy_reuse
    loaded.close_writer()


@pytest.mark.parametrize("operation", ["colors", "embeddings"])
def test_late_reuse_decision_cannot_overwrite_edited_inputs(tmp_path, operation):
    from core.analysis_model_identity import DINOV2_TAG
    from core.operations.colors import ColorApplication, color_request
    from core.operations.embeddings import EmbeddingApplication, embedding_task
    from core.operations.legacy_reuse import accept_legacy_colors, accept_legacy_embeddings

    project = project_with_thumbnails(tmp_path, 1)
    clip = project.clips[0]
    clip.dominant_colors = [(1, 2, 3)]
    clip.embedding, clip.embedding_model = [0.1] * 768, DINOV2_TAG
    if operation == "colors":
        request = color_request(project)
        application = ColorApplication(project, request)
        result = accept_legacy_colors(request)
        clip.start_frame += 1
        assert application.apply(result).outcomes[0].code == "stale_input"
    else:
        tasks = (embedding_task(clip, project.sources[0]),)
        publication = EmbeddingApplication(project, tasks)
        outcome = accept_legacy_embeddings(tasks)[0]
        clip.start_frame += 1
        assert not publication.apply(project, outcome)
    assert operation not in clip.analysis_records


def test_unknown_future_record_is_preserved(tmp_path):
    from core.spine.analysis_reuse import accept_legacy_analysis
    from models.analysis_record import UnreadableAnalysisRecord

    project = project_with_thumbnails(tmp_path, 1)
    clip = project.clips[0]
    clip.dominant_colors = [(1, 2, 3)]
    record = UnreadableAnalysisRecord('{"version":999}')
    clip.analysis_records["colors"] = record
    assert not accept_legacy_analysis(project, "colors")["accepted"]
    assert clip.analysis_records["colors"] is record


def test_unknown_embedding_model_requires_recomputation(tmp_path):
    from core.spine.analysis_reuse import accept_legacy_analysis

    project = project_with_thumbnails(tmp_path, 1)
    project.clips[0].embedding = [0.1] * 768
    project.clips[0].embedding_model = None
    assert not accept_legacy_analysis(project, "embeddings")["accepted"]
    assert "embeddings" not in project.clips[0].analysis_records
