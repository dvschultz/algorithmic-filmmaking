"""Verified analysis fixtures produced through the shared operation boundary."""

from pathlib import Path
from unittest.mock import patch

from core.project import Project
from models.clip import Clip, Source


def mock_face_execution(monkeypatch, directory: Path, provider) -> Path:
    """Report real fixture weight fingerprints around a configurable provider."""
    from types import SimpleNamespace
    from core.analysis.face_weights import FaceWeights
    from core.analysis.faces import face_model_execution
    from core.operations.face_records import face_packages

    pack = directory / "insightface" / "models" / "buffalo_l"
    pack.mkdir(parents=True, exist_ok=True)
    for name in ("detection", "recognition"):
        (pack / f"{name}.onnx").write_bytes(name.encode())
    monkeypatch.setattr("core.analysis.faces._get_model_cache_dir", lambda: directory)
    monkeypatch.setattr("core.operations.faces.face_environment", lambda: {
        "packages": face_packages(), "available_providers": ["CPUExecutionProvider"]})

    def compute(**kwargs):
        model = SimpleNamespace(_scene_ripper_weights=FaceWeights.capture(pack), models={
            name: SimpleNamespace(model_file=str(pack / f"{name}.onnx"),
                session=SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"]))
            for name in ("detection", "recognition")})
        kwargs["on_execution"](face_model_execution(model))
        return [{"frame_number": kwargs["start_frame"], **face} for face in provider(**kwargs)]

    monkeypatch.setattr("core.analysis.faces.extract_faces_from_clip", compute)
    return pack


def verify_word_alignment(clip: Clip, source: Source, directory: Path) -> None:
    """Give dialog fixtures real media and a record for their supplied word data."""
    from core.analysis.alignment import ALIGNMENT_MODEL
    from core.operations.alignment import AlignmentApplication, run_alignment, snapshot_alignment_tasks

    source.file_path = directory / f"{source.id}.mp4"
    if not source.file_path.exists():
        source.file_path.write_bytes(b"word fixture media")
    words = [word for segment in (clip.transcript or []) for word in (segment.words or [])]
    project = Project(sources=[source], clips=[clip])
    tasks = snapshot_alignment_tasks([clip], project.sources_by_id, verified=True, skip_existing=False)
    wav = directory / "alignment-fixture.wav"
    wav.write_bytes(b"audio")

    def align(*a, **kwargs):
        kwargs["on_execution"]({"backend": "ctc", "model": ALIGNMENT_MODEL, "revision": "fixture-r1"})
        return words

    with patch("core.operations.alignment_records.alignment_model_revision", return_value="fixture-r1"), patch("core.analysis.alignment.extract_audio_to_wav", return_value=wav), patch("core.analysis.alignment.align_words", side_effect=align):
        outcome = run_alignment(tasks)[0]
        assert AlignmentApplication(project, tasks).apply(project, outcome), outcome


def verify_clip_analysis(clip: Clip, directory: Path, *, embeddings: bool = False, objects: bool = False, ocr: bool = False, classify: bool = False, shots: bool = False, gaze: bool = False, boundary: bool = False, descriptions: bool = False, cinematography: bool = False, transcriptions: bool = False) -> Source:
    from core.spine.analyze import analyze_colors, embeddings as analyze_embeddings, detect_objects, extract_text, classify_content, analyze_shots, gaze as analyze_gaze, boundary_embeddings

    media = directory / f"{clip.id}.mp4"
    media.write_bytes(b"source fixture")
    thumbnail = directory / f"{clip.id}.jpg"
    thumbnail.write_bytes(b"thumbnail fixture")
    clip.thumbnail_path = thumbnail
    source = Source(id=clip.source_id, file_path=media, fps=30)
    project = Project(sources=[source], clips=[clip])
    with patch("core.analysis.color.extract_dominant_colors", return_value=[(10, 20, 30)]):
        analyze_colors(project)
    if embeddings:
        with patch("core.analysis.embeddings.extract_clip_embeddings_batch", return_value=[[0.1] * 768]), patch("core.analysis.embeddings.unload_model"):
            analyze_embeddings(project)
    if objects:
        with patch("core.analysis.detection.detect_objects", return_value=[]):
            detect_objects(project)
    if ocr:
        with patch("core.analysis.ocr.extract_text_from_clip", return_value=[]):
            extract_text(project)
    if classify:
        with patch("core.analysis.classification.classify_frame", return_value=[]):
            classify_content(project)
    if shots:
        with patch("core.analysis.shots.classify_shot_type", return_value=("wide" if cinematography else "wide shot", 0.9)):
            analyze_shots(project)
    if gaze:
        with patch("core.analysis.gaze.extract_gaze_from_clip", return_value=None), patch("core.analysis.gaze.load_face_mesh"), patch("core.analysis.gaze.unload_model"):
            analyze_gaze(project)
    if boundary:
        with patch("core.analysis.embeddings.extract_boundary_embeddings", return_value=([0.1] * 768, [0.2] * 768)), patch("core.analysis.embeddings.unload_model"):
            boundary_embeddings(project)
    if descriptions:
        from core.spine.analyze import describe

        with patch("core.analysis.description.describe_frame", return_value=("Fixture description", "gpt-test")):
            describe(project)
    if cinematography:
        from core.spine.analyze import cinematography as analyze_cinematography
        from models.cinematography import CinematographyAnalysis

        with patch("core.analysis.cinematography.analyze_cinematography", return_value=CinematographyAnalysis(shot_size="ELS", analysis_model="gpt-test")):
            analyze_cinematography(project)
    if transcriptions:
        from core.settings import load_settings
        from core.operations.transcription import TranscriptionOptions, TranscriptionApplication, run_transcription
        from core.operations.transcription_records import transcription_task

        settings = load_settings()
        options = TranscriptionOptions(model=settings.transcription_model, backend=settings.transcription_backend, language=settings.transcription_language, segmentation_mode=settings.transcription_segmentation_mode, segment_max_seconds=settings.transcription_segment_max_seconds)
        task = transcription_task(clip, source, skip_existing=False)
        with patch("core.transcription._has_audio_stream", return_value=True), patch("core.transcription.transcribe_clip", return_value=[]):
            outcome = run_transcription((task,), options)[0]
            assert TranscriptionApplication(project, (task,), options).apply(project, outcome)
    return source
