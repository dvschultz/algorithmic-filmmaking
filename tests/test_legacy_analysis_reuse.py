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


@pytest.mark.parametrize("operation", ["colors", "brightness", "volume", "classify", "detect_objects", "boundary_embeddings", "gaze", "shots", "extract_text", "describe", "cinematography", "transcribe", "align_words", "custom_query"])
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
    project.clips[0].extracted_texts = []
    project.clips[0].custom_queries = [{"query": "Person?", "match": False, "confidence": 0.0, "model": "legacy"}]
    project.clips[0].transcript = []
    project.clips[0].description = "A person walking"
    project.clips[0].shot_type = "wide shot"
    if operation == "cinematography":
        from models.cinematography import CinematographyAnalysis
        project.clips[0].cinematography = CinematographyAnalysis()
        project.clips[0].shot_type = "medium"
    project.clips[0].gaze_yaw = project.clips[0].gaze_pitch = 0.0
    project.clips[0].gaze_category = "at_camera"
    project.clips[0].first_frame_embedding = [0.2] * 768
    project.clips[0].last_frame_embedding = [0.3] * 768
    project.clips[0].embedding_model = DINOV2_TAG
    path = tmp_path / "project.json"
    assert project.save(path)
    result = CliRunner().invoke(analyze, ["accept-legacy", str(path), "--operation", operation, "--query", "Person?"])
    assert result.exit_code == 0, result.output
    loaded = Project.load(path)
    from core.operations.custom_query import custom_query_record_key
    record = loaded.clips[0].analysis_records[custom_query_record_key("Person?") if operation == "custom_query" else operation]
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


def test_shot_reuse_captures_settings_before_worker_execution(tmp_path):
    from core.settings import Settings
    from core.operations.shots import ShotTypeOptions, run_shot_types, shot_task
    from ui.workers.legacy_reuse_worker import LegacyReuseWorker

    project = project_with_thumbnails(tmp_path, 1)
    clip = project.clips[0]
    clip.shot_type = "wide shot"
    settings = Settings(shot_classifier_tier="cloud", shot_classifier_cloud_model="original-model")
    worker = LegacyReuseWorker(project, "shots", [clip.id], settings=settings)
    settings.shot_classifier_cloud_model = "changed-model"
    results = []
    worker.result_ready.connect(results.append)
    worker.run()
    outcome = results[0][0]
    assert worker.application.apply(project, outcome)
    record = clip.analysis_records["shots"]
    assert record.provenance == "unknown" and record.legacy_reuse
    assert record.identity.to_dict()["parameters"]["cloud_model"] == "original-model"
    assert not operation_is_complete_for_clip("shots", clip, source=project.sources[0], settings=settings)
    settings.shot_classifier_cloud_model = "original-model"
    assert operation_is_complete_for_clip("shots", clip, source=project.sources[0], settings=settings)
    with patch("core.analysis.shots.classify_shot_type_tiered", side_effect=AssertionError("no inference")), patch("core.analysis.shots.classify_shot_type", side_effect=AssertionError("no inference")):
        reused = run_shot_types((shot_task(clip, project.sources[0]),), ShotTypeOptions.from_settings(settings=settings))
    assert reused[0].status == "skipped"


@pytest.mark.parametrize("label", [None, "unknown", "unrecognized"])
def test_missing_or_unknown_shot_label_requires_recomputation(tmp_path, label):
    from core.spine.analysis_reuse import accept_legacy_analysis

    project = project_with_thumbnails(tmp_path, 1)
    project.clips[0].shot_type = label
    result = accept_legacy_analysis(project, "shots")
    assert not result["accepted"] and len(result["failed"]) == 1
    assert "shots" not in project.clips[0].analysis_records


@pytest.mark.parametrize("value", ["text", "empty", "missing", "outside"])
def test_ocr_legacy_reuse_validates_observations(tmp_path, value):
    from core.settings import Settings
    from core.spine.analysis_reuse import accept_legacy_analysis
    from core.operations.ocr import OcrOptions, ocr_task, resolve_ocr_options, run_ocr
    from models.clip import ExtractedText

    project = project_with_thumbnails(tmp_path, 1)
    clip = project.clips[0]
    if value == "empty":
        clip.extracted_texts = []
    elif value in ("text", "outside"):
        clip.extracted_texts = [ExtractedText(clip.start_frame if value == "text" else clip.end_frame, "SIGN", 0.9, "paddleocr")]
    settings = Settings(description_model_cloud="original-model")
    options = resolve_ocr_options(OcrOptions(), settings=settings)
    result = accept_legacy_analysis(project, "extract_text", ocr_options=options)
    if value in ("missing", "outside"):
        assert not result["accepted"] and len(result["failed"]) == 1
        assert "extract_text" not in clip.analysis_records
        return
    assert result["accepted"] == [clip.id]
    assert clip.analysis_records["extract_text"].provenance == "unknown"
    assert operation_is_complete_for_clip("extract_text", clip, source=project.sources[0], settings=settings)
    with patch("core.operations.ocr._inference_lock") as lock:
        lock.acquire.side_effect = AssertionError("no inference")
        reused = run_ocr((ocr_task(clip, project.sources[0]),), options)
    assert reused[0].status == "skipped"
    settings.description_model_cloud = "changed-model"
    assert not operation_is_complete_for_clip("extract_text", clip, source=project.sources[0], settings=settings)


def test_ocr_worker_captures_unsaved_model_settings(tmp_path):
    from core.settings import Settings
    from ui.workers.legacy_reuse_worker import LegacyReuseWorker

    project = project_with_thumbnails(tmp_path, 1)
    clip = project.clips[0]
    clip.extracted_texts = []
    settings = Settings(description_model_cloud="original-model")
    worker = LegacyReuseWorker(project, "extract_text", [clip.id], settings=settings)
    settings.description_model_cloud = "changed-model"
    results = []
    worker.result_ready.connect(results.append)
    worker.run()
    assert worker.application.apply(project, results[0][0])
    assert clip.analysis_records["extract_text"].identity.to_dict()["parameters"]["vlm_model"] == "original-model"


@pytest.mark.parametrize("frame_count", [None, 3])
def test_description_reuse_captures_settings_and_preserves_metadata(tmp_path, frame_count):
    from core.settings import Settings
    from core.operations.description import description_task, resolve_options, run_description
    from ui.workers.legacy_reuse_worker import LegacyReuseWorker

    project = project_with_thumbnails(tmp_path, 1)
    clip = project.clips[0]
    clip.description = "A person walking"
    clip.description_model = "legacy-model"
    clip.description_frames = frame_count
    settings = Settings(description_model_tier="cloud", description_model_cloud="original-model", description_input_mode="frame")
    worker = LegacyReuseWorker(project, "describe", [clip.id], settings=settings)
    settings.description_model_cloud = "changed-model"
    results = []
    worker.result_ready.connect(results.append)
    worker.run()
    assert worker.application.apply(project, results[0][0])
    record = clip.analysis_records["describe"]
    assert record.provenance == "unknown" and record.legacy_reuse
    assert record.identity.to_dict()["parameters"]["model"] == "original-model"
    assert clip.description_model == "legacy-model"
    assert clip.description_frames == frame_count
    assert not operation_is_complete_for_clip("describe", clip, source=project.sources[0], settings=settings)
    settings.description_model_cloud = "original-model"
    assert operation_is_complete_for_clip("describe", clip, source=project.sources[0], settings=settings)
    with patch("core.analysis.description.describe_frame", side_effect=AssertionError("no inference")), patch("core.analysis.description._load_local_model", side_effect=AssertionError("no weights")):
        reused = run_description((description_task(clip, project.sources[0]),), resolve_options(settings=settings))
    assert reused[0].status == "skipped"
    clip.end_frame -= 1
    assert not operation_is_complete_for_clip("describe", clip, source=project.sources[0], settings=settings)


@pytest.mark.parametrize("description,frames", [(None, None), ("", None), (" ", None), ("Error: failed", 1), ("A person", 0), ("A person", True)])
def test_invalid_legacy_description_requires_recomputation(tmp_path, description, frames):
    from core.spine.analysis_reuse import accept_legacy_analysis
    from core.operations.description import DescriptionOptions

    project = project_with_thumbnails(tmp_path, 1)
    clip = project.clips[0]
    clip.description = description
    clip.description_frames = frames
    result = accept_legacy_analysis(project, "describe", description_options=DescriptionOptions("cloud", model="test", input_mode="frame"))
    assert not result["accepted"] and len(result["failed"]) == 1
    assert "describe" not in clip.analysis_records


def test_description_reuse_rejects_changed_legacy_metadata(tmp_path):
    from core.operations.description import DescriptionOptions
    from core.spine.analysis_reuse import accept_legacy_analysis
    from models.analysis_record import AnalysisRecord

    project = project_with_thumbnails(tmp_path, 1)
    clip = project.clips[0]
    clip.description = "A person walking"
    clip.description_model = "changed-model"
    clip.analysis_records["describe"] = AnalysisRecord.legacy({"description": clip.description, "description_model": "old-model"})
    previous = clip.analysis_records["describe"]
    result = accept_legacy_analysis(project, "describe", description_options=DescriptionOptions("cloud", model="test", input_mode="frame"))
    assert not result["accepted"]
    assert clip.analysis_records["describe"] is previous


def test_cinematography_reuse_captures_settings_and_preserves_metadata(tmp_path):
    from core.settings import Settings
    from core.operations.cinematography import cinematography_task, resolve_options, run_cinematography
    from models.cinematography import CinematographyAnalysis
    from ui.workers.legacy_reuse_worker import LegacyReuseWorker

    project = project_with_thumbnails(tmp_path, 1)
    clip = project.clips[0]
    analysis = CinematographyAnalysis(shot_size_confidence=0.0, analysis_model="legacy-model", analysis_mode="video")
    clip.cinematography = analysis
    clip.shot_type = analysis.get_simple_shot_type()
    settings = Settings(cinematography_tier="cloud", cinematography_model="original-model", cinematography_input_mode="frame")
    worker = LegacyReuseWorker(project, "cinematography", [clip.id], settings=settings)
    settings.cinematography_model = "changed-model"
    results = []
    worker.result_ready.connect(results.append)
    worker.run()
    assert worker.application.apply(project, results[0][0])
    record = clip.analysis_records["cinematography"]
    assert record.provenance == "unknown" and record.legacy_reuse
    assert record.identity.to_dict()["parameters"]["model"] == "original-model"
    assert clip.cinematography is analysis
    assert analysis.analysis_mode == "video" and analysis.analysis_model == "legacy-model"
    assert not operation_is_complete_for_clip("cinematography", clip, source=project.sources[0], settings=settings)
    settings.cinematography_model = "original-model"
    assert operation_is_complete_for_clip("cinematography", clip, source=project.sources[0], settings=settings)
    with patch("core.analysis.cinematography.analyze_cinematography", side_effect=AssertionError("no inference")):
        reused = run_cinematography((cinematography_task(clip, project.sources[0]),), resolve_options(settings=settings))
    assert reused[0].status == "skipped"
    clip.end_frame -= 1
    assert not operation_is_complete_for_clip("cinematography", clip, source=project.sources[0], settings=settings)


@pytest.mark.parametrize("invalid", ["missing", "shot", "enum", "confidence", "boolean", "mode"])
def test_invalid_legacy_cinematography_requires_recomputation(tmp_path, invalid):
    from core.spine.analysis_reuse import accept_legacy_analysis
    from core.operations.cinematography import CinematographyOptions
    from models.cinematography import CinematographyAnalysis

    project = project_with_thumbnails(tmp_path, 1)
    clip = project.clips[0]
    clip.cinematography = CinematographyAnalysis()
    clip.shot_type = "medium"
    if invalid == "missing":
        clip.cinematography = None
    elif invalid == "shot":
        clip.shot_type = "wide"
    elif invalid == "enum":
        clip.cinematography.camera_angle = "invalid"
    elif invalid == "confidence":
        clip.cinematography.shot_size_confidence = 1.1
    elif invalid == "boolean":
        clip.cinematography.shot_size_confidence = True
    else:
        clip.cinematography.analysis_mode = "invalid"
    result = accept_legacy_analysis(project, "cinematography", cinematography_options=CinematographyOptions("cloud", "frame", "test", "local"))
    assert not result["accepted"] and len(result["failed"]) == 1
    assert "cinematography" not in clip.analysis_records


@pytest.mark.parametrize("empty", [False, True])
def test_transcription_reuse_preserves_words_and_captures_settings(tmp_path, empty):
    from core.settings import Settings
    from core.operations.legacy_reuse import legacy_transcription_options
    from core.operations.transcription import run_transcription
    from core.operations.transcription_records import transcription_task
    from core.transcription_models import TranscriptSegment, WordTimestamp
    from ui.workers.legacy_reuse_worker import LegacyReuseWorker
    from models.analysis_record import AnalysisRecord

    project = project_with_thumbnails(tmp_path, 1)
    clip = project.clips[0]
    transcript = [] if empty else [TranscriptSegment(0.0, 0.1, "Hello", -0.2, words=[WordTimestamp(0.0, 0.1, "Hello", 0.8)])]
    clip.transcript = transcript
    alignment = AnalysisRecord.legacy({"transcript": [s.to_dict() for s in transcript]})
    clip.analysis_records["align_words"] = alignment
    settings = Settings(transcription_backend="groq", transcription_cloud_model="original-model")
    worker = LegacyReuseWorker(project, "transcribe", [clip.id], settings=settings)
    settings.transcription_cloud_model = "changed-model"
    results = []
    worker.result_ready.connect(results.append)
    worker.run()
    assert worker.application.apply(project, results[0][0])
    record = clip.analysis_records["transcribe"]
    assert record.provenance == "unknown" and record.legacy_reuse
    assert clip.transcript is transcript
    assert clip.analysis_records["align_words"] is alignment
    assert not operation_is_complete_for_clip("transcribe", clip, source=project.sources[0], settings=settings)
    settings.transcription_cloud_model = "original-model"
    assert operation_is_complete_for_clip("transcribe", clip, source=project.sources[0], settings=settings)
    with patch("core.operations.transcription._compute_task", side_effect=AssertionError("no inference")):
        reused = run_transcription((transcription_task(clip, project.sources[0]),), legacy_transcription_options(settings))
    assert reused[0].status == "skipped"
    clip.end_frame -= 1
    assert not operation_is_complete_for_clip("transcribe", clip, source=project.sources[0], settings=settings)


@pytest.mark.parametrize("invalid", ["missing", "outside", "word", "negative", "confidence"])
def test_invalid_legacy_transcript_requires_recomputation(tmp_path, invalid):
    from core.operations.transcription import TranscriptionOptions
    from core.spine.analysis_reuse import accept_legacy_analysis
    from core.transcription_models import TranscriptSegment, WordTimestamp

    project = project_with_thumbnails(tmp_path, 1)
    clip = project.clips[0]
    segment = TranscriptSegment(0.0, 0.1, "Hello")
    clip.transcript = [segment]
    if invalid == "missing":
        clip.transcript = None
    elif invalid == "outside":
        segment.end_time = 10000
    elif invalid == "word":
        segment.words = [WordTimestamp(0.0, 0.2, "Hello")]
    elif invalid == "negative":
        segment.start_time = -1
    else:
        segment.confidence = True
    result = accept_legacy_analysis(project, "transcribe", transcription_options=TranscriptionOptions(backend="groq", cloud_model="test"))
    assert not result["accepted"] and len(result["failed"]) == 1
    assert "transcribe" not in clip.analysis_records


def test_alignment_acceptance_preserves_transcript_and_skips_inference(tmp_path):
    from core.spine.analysis_reuse import accept_legacy_analysis
    from core.operations.alignment import run_alignment, snapshot_alignment_tasks
    from core.transcription_models import TranscriptSegment, WordTimestamp
    from models.analysis_record import AnalysisRecord

    project = project_with_thumbnails(tmp_path, 1)
    clip = project.clips[0]
    transcript = [TranscriptSegment(0.0, 0.1, "Hello", words=[WordTimestamp(0.0, 0.1, "Hello")])]
    clip.transcript = transcript
    previous = AnalysisRecord.legacy({"transcript": [s.to_dict() for s in transcript]})
    clip.analysis_records["transcribe"] = previous
    assert not operation_is_complete_for_clip("align_words", clip, source=project.sources[0])
    assert accept_legacy_analysis(project, "align_words")["accepted"] == [clip.id]
    record = clip.analysis_records["align_words"]
    assert record.provenance == "unknown" and record.legacy_reuse
    assert record.identity.to_dict()["model"]["execution"] == []
    assert clip.transcript is transcript
    assert clip.analysis_records["transcribe"] is previous
    assert operation_is_complete_for_clip("align_words", clip, source=project.sources[0])
    with patch("core.operations.alignment._compute_raw", side_effect=AssertionError("no inference")):
        reused = run_alignment(snapshot_alignment_tasks([clip], project.sources_by_id, verified=True))
    assert reused[0].status == "skipped"
    clip.transcript[0].text = "Changed"
    assert not operation_is_complete_for_clip("align_words", clip, source=project.sources[0])


@pytest.mark.parametrize("invalid", ["missing", "words", "empty_words", "outside"])
def test_alignment_acceptance_rejects_missing_or_invalid_timings(tmp_path, invalid):
    from core.spine.analysis_reuse import accept_legacy_analysis
    from core.transcription_models import TranscriptSegment, WordTimestamp

    project = project_with_thumbnails(tmp_path, 1)
    clip = project.clips[0]
    segment = TranscriptSegment(0.0, 0.1, "Hello")
    if invalid != "missing":
        clip.transcript = [segment]
    if invalid == "empty_words":
        segment.words = []
    elif invalid == "outside":
        segment.words = [WordTimestamp(0.0, 0.2, "Hello")]
    result = accept_legacy_analysis(project, "align_words")
    assert not result["accepted"] and len(result["failed"]) == 1
    assert "align_words" not in clip.analysis_records


def test_verified_alignment_cannot_reuse_unknown_execution(tmp_path):
    from dataclasses import replace
    from core.spine.analysis_reuse import accept_legacy_analysis
    from core.transcription_models import TranscriptSegment, WordTimestamp

    project = project_with_thumbnails(tmp_path, 1)
    clip = project.clips[0]
    clip.transcript = [TranscriptSegment(0.0, 0.1, "Hello", words=[WordTimestamp(0.0, 0.1, "Hello")])]
    assert accept_legacy_analysis(project, "align_words")["accepted"] == [clip.id]
    clip.analysis_records["align_words"] = replace(clip.analysis_records["align_words"], provenance="verified", legacy_reuse=False)
    assert not operation_is_complete_for_clip("align_words", clip, source=project.sources[0])


def test_query_acceptance_preserves_negative_answer_and_history(tmp_path):
    from copy import deepcopy
    from core.settings import Settings
    from core.analysis_availability import custom_query_is_complete
    from core.operations.custom_query import custom_query_record_key, custom_query_task, resolve_options, run_custom_query
    from ui.workers.legacy_reuse_worker import LegacyReuseWorker

    project = project_with_thumbnails(tmp_path, 1)
    clip = project.clips[0]
    clip.custom_queries = [
        {"query": "Person?", "match": True, "confidence": 0.8, "model": "older"},
        {"query": "Car?", "match": True, "confidence": 0.8, "model": "other"},
        {"query": "Person?", "match": False, "confidence": 0.0, "model": "legacy"},
    ]
    history = deepcopy(clip.custom_queries)
    settings = Settings(description_model_tier="cloud", description_model_cloud="original-model")
    worker = LegacyReuseWorker(project, "custom_query", [clip.id], settings=settings, query=" Person? ")
    settings.description_model_cloud = "changed-model"
    results = []
    worker.result_ready.connect(results.append)
    worker.run()
    assert worker.application.apply(project, results[0][0])
    record = clip.analysis_records[custom_query_record_key("Person?")]
    assert record.provenance == "unknown" and record.legacy_reuse
    assert record.value["result"]["match"] is False
    assert clip.custom_queries == history
    assert not custom_query_is_complete(clip, project.sources[0], "Person?", settings=settings)
    settings.description_model_cloud = "original-model"
    assert custom_query_is_complete(clip, project.sources[0], "Person?", settings=settings)
    assert not custom_query_is_complete(clip, project.sources[0], "Car?", settings=settings)
    with patch("core.analysis.custom_query.evaluate_custom_query", side_effect=AssertionError("no inference")):
        reused = run_custom_query((custom_query_task(clip, project.sources[0], "Person?", skip_existing=True),), resolve_options(settings=settings))
    assert reused[0].status == "skipped"
    clip.custom_queries[-1]["match"] = True
    assert not custom_query_is_complete(clip, project.sources[0], "Person?", settings=settings)


@pytest.mark.parametrize("invalid", ["missing", "confidence", "match", "future"])
def test_query_acceptance_rejects_invalid_or_future_answers(tmp_path, invalid):
    from core.spine.analysis_reuse import accept_legacy_analysis
    from core.operations.custom_query import CustomQueryOptions, custom_query_record_key
    from models.analysis_record import UnreadableAnalysisRecord

    project = project_with_thumbnails(tmp_path, 1)
    clip = project.clips[0]
    answer = {"query": "Person?", "match": False, "confidence": 0.0, "model": "legacy"}
    clip.custom_queries = [] if invalid == "missing" else [answer]
    if invalid == "confidence":
        answer["confidence"] = None
    elif invalid == "match":
        answer["match"] = "false"
    key = custom_query_record_key("Person?")
    future = UnreadableAnalysisRecord('{"version":999}')
    if invalid == "future":
        clip.analysis_records[key] = future
    result = accept_legacy_analysis(project, "custom_query", query="Person?", query_options=CustomQueryOptions("cloud", "test"))
    assert not result["accepted"] and len(result["failed"]) == 1
    assert clip.analysis_records.get(key) is (future if invalid == "future" else None)
