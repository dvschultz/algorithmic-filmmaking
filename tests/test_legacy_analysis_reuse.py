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


@pytest.mark.parametrize("operation", ["colors", "brightness", "volume", "classify", "detect_objects", "boundary_embeddings", "gaze"])
def test_cli_reuse_requires_an_explicit_command_and_saves_unknown_provenance(tmp_path, operation):
    from click.testing import CliRunner
    from cli.commands.analyze import analyze
    from core.project import Project
    from core.analysis_model_identity import DINOV2_TAG

    project = project_with_thumbnails(tmp_path, 1)
    project.clips[0].dominant_colors = [(1, 2, 3)]
    project.clips[0].average_brightness = project.clips[0].rms_volume = 0.0
    project.clips[0].object_labels = project.clips[0].detected_objects = []
    project.clips[0].person_count = 0
    project.clips[0].gaze_yaw = project.clips[0].gaze_pitch = 0.0
    project.clips[0].gaze_category = "at_camera"
    project.clips[0].first_frame_embedding = [0.2] * 768
    project.clips[0].last_frame_embedding = [0.3] * 768
    project.clips[0].embedding_model = DINOV2_TAG
    path = tmp_path / "project.json"
    assert project.save(path)
    result = CliRunner().invoke(analyze, ["accept-legacy", str(path), "--operation", operation])
    assert result.exit_code == 0, result.output
    loaded = Project.load(path)
    record = loaded.clips[0].analysis_records[operation]
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


def test_cancelled_acceptance_leaves_legacy_values_unbound(tmp_path):
    from threading import Event
    from core.spine.analysis_reuse import accept_legacy_analysis

    project = project_with_thumbnails(tmp_path, 1)
    project.clips[0].dominant_colors = [(1, 2, 3)]
    cancel = Event()
    cancel.set()
    result = accept_legacy_analysis(project, "colors", cancel_event=cancel)
    assert result["unprocessed"] == [project.clips[0].id]
    assert not result["accepted"]
    assert "colors" not in project.clips[0].analysis_records


@pytest.mark.parametrize("operation,field", [("brightness", "average_brightness"), ("volume", "rms_volume")])
def test_legacy_zero_scalar_reuses_and_invalidates(tmp_path, operation, field):
    from core.spine.analysis_reuse import accept_legacy_analysis
    from core.spine.analyze import analyze_scalars

    project = project_with_thumbnails(tmp_path, 1)
    clip = project.clips[0]
    setattr(clip, field, 0.0)
    assert not operation_is_complete_for_clip(operation, clip, source=project.sources[0])
    assert accept_legacy_analysis(project, operation)["accepted"] == [clip.id]
    record = clip.analysis_records[operation]
    assert record.provenance == "unknown" and record.legacy_reuse
    assert operation_is_complete_for_clip(operation, clip, source=project.sources[0])
    with patch("core.analysis.color.get_average_brightness", side_effect=AssertionError("no inference")), patch("core.analysis.audio.extract_clip_volume", side_effect=AssertionError("no inference")):
        result = analyze_scalars(project, operation)["result"]
    assert len(result["skipped"]) == 1
    clip.start_frame += 1
    assert not operation_is_complete_for_clip(operation, clip, source=project.sources[0])


@pytest.mark.parametrize("operation", ["brightness", "volume"])
def test_missing_scalar_is_not_explicitly_accepted(tmp_path, operation):
    from core.spine.analysis_reuse import accept_legacy_analysis

    project = project_with_thumbnails(tmp_path, 1)
    result = accept_legacy_analysis(project, operation)
    assert result["accepted"] == [] and len(result["failed"]) == 1
    assert operation not in project.clips[0].analysis_records


@pytest.mark.parametrize("operation", ["brightness", "volume"])
def test_scalar_reuse_requires_source_even_when_binaries_exist(tmp_path, operation):
    from core.operations.legacy_reuse import accept_legacy_scalars
    from core.operations.scalars import scalar_task

    project = project_with_thumbnails(tmp_path, 1)
    clip = project.clips[0]
    clip.average_brightness = clip.rms_volume = 0.0
    outcome = accept_legacy_scalars((scalar_task(clip, None, operation),))[0]
    assert outcome.status == "failed" and outcome.record_json is None


@pytest.mark.parametrize("operation", ["classify", "detect_objects"])
@pytest.mark.parametrize("empty", [False, True])
def test_legacy_visual_results_include_valid_empty_outputs(tmp_path, operation, empty):
    from core.spine.analysis_reuse import accept_legacy_analysis
    from core.operations.classification import ClassificationOptions, classification_task, run_classification
    from core.operations.object_detection import ObjectDetectionOptions, object_detection_task, run_object_detection

    project = project_with_thumbnails(tmp_path, 1)
    clip = project.clips[0]
    clip.object_labels = [] if empty else ["cat"]
    clip.detected_objects = [] if empty else [{"label": "person", "confidence": 0.9, "bbox": [0, 0, 1, 1]}]
    clip.person_count = 0 if empty else 1
    result = accept_legacy_analysis(project, operation)
    assert result["accepted"] == [clip.id], result
    assert clip.analysis_records[operation].provenance == "unknown"
    assert operation_is_complete_for_clip(operation, clip, source=project.sources[0])
    with patch("core.analysis.classification.classify_frame", side_effect=AssertionError("no inference")), patch("core.analysis.detection.detect_objects", side_effect=AssertionError("no inference")):
        if operation == "classify":
            outcomes = run_classification((classification_task(clip, project.sources[0]),), ClassificationOptions())
        else:
            outcomes = run_object_detection((object_detection_task(clip, project.sources[0]),), ObjectDetectionOptions())
    assert outcomes[0].status == "skipped"
    clip.thumbnail_path.write_bytes(b"changed thumbnail")
    assert not operation_is_complete_for_clip(operation, clip, source=project.sources[0])


@pytest.mark.parametrize("count,confidence", [(0, 0.9), (True, 0.9), (1, True)])
def test_inconsistent_legacy_detections_require_recomputation(tmp_path, count, confidence):
    from core.spine.analysis_reuse import accept_legacy_analysis

    project = project_with_thumbnails(tmp_path, 1)
    clip = project.clips[0]
    clip.detected_objects = [{"label": "person", "confidence": confidence, "bbox": [0, 0, 1, 1]}]
    clip.person_count = count
    result = accept_legacy_analysis(project, "detect_objects")
    assert not result["accepted"] and len(result["failed"]) == 1
    assert "detect_objects" not in clip.analysis_records


@pytest.mark.parametrize("invalid", [None, "first", "last", "model"])
def test_boundary_pair_acceptance_and_reuse(tmp_path, invalid):
    from core.analysis_model_identity import DINOV2_TAG
    from core.spine.analysis_reuse import accept_legacy_analysis
    from core.operations.boundary_embeddings import boundary_embedding_task, run_boundary_embeddings

    project = project_with_thumbnails(tmp_path, 1)
    clip = project.clips[0]
    clip.first_frame_embedding, clip.last_frame_embedding = [0.2] * 768, [0.3] * 768
    clip.embedding_model = DINOV2_TAG
    if invalid == "first":
        clip.first_frame_embedding = None
    elif invalid == "last":
        clip.last_frame_embedding = [0.0] * 768
    elif invalid == "model":
        clip.embedding_model = "unknown"
    result = accept_legacy_analysis(project, "boundary_embeddings")
    if invalid is not None:
        assert not result["accepted"] and len(result["failed"]) == 1
        assert "boundary_embeddings" not in clip.analysis_records
        return
    assert result["accepted"] == [clip.id]
    assert clip.analysis_records["boundary_embeddings"].provenance == "unknown"
    assert operation_is_complete_for_clip("boundary_embeddings", clip, source=project.sources[0])
    # Reuse must return before acquiring the model for endpoint inference.
    with patch("core.operations.embeddings._EmbeddingModelSession.acquire", side_effect=AssertionError("no inference")):
        reused = run_boundary_embeddings((boundary_embedding_task(clip, project.sources[0]),))
    assert reused[0].status == "skipped"
    assert clip.first_frame_embedding == [0.2] * 768
    assert clip.last_frame_embedding == [0.3] * 768
    clip.end_frame -= 1
    assert not operation_is_complete_for_clip("boundary_embeddings", clip, source=project.sources[0])


@pytest.mark.parametrize("missing", [False, True])
def test_gaze_requires_complete_observation_and_reuses_without_inference(tmp_path, missing):
    from core.spine.analysis_reuse import accept_legacy_analysis
    from core.operations.gaze import GazeOptions, gaze_task, run_gaze

    project = project_with_thumbnails(tmp_path, 1)
    clip = project.clips[0]
    if not missing:
        clip.gaze_yaw = clip.gaze_pitch = 0.0
        clip.gaze_category = "at_camera"
    result = accept_legacy_analysis(project, "gaze")
    if missing:
        assert not result["accepted"] and len(result["failed"]) == 1
        assert "gaze" not in clip.analysis_records
        return
    assert result["accepted"] == [clip.id]
    assert clip.analysis_records["gaze"].provenance == "unknown"
    assert operation_is_complete_for_clip("gaze", clip, source=project.sources[0])
    with patch("core.analysis.gaze.extract_gaze_from_clip", side_effect=AssertionError("no inference")):
        reused = run_gaze((gaze_task(clip, project.sources[0]),), GazeOptions())
    assert reused[0].status == "skipped"
    clip.end_frame -= 1
    assert not operation_is_complete_for_clip("gaze", clip, source=project.sources[0])
