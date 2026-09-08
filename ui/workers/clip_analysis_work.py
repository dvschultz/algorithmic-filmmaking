"""Construct existing clip workers and their shared publication adapters."""

from typing import Any

from core.operations.clip_analysis import ClipAnalysisOptions


def create_clip_analysis_worker(
    project: Any,
    settings: Any,
    operation: str,
    clips: list,
    *,
    force_rerun: bool = False,
    query: str | None = None,
    options: ClipAnalysisOptions | None = None,
) -> tuple[Any, Any]:
    """Build one batch; transcription batches must belong to one source."""
    common = dict(clips=clips, project=project, skip_existing=not force_rerun)
    options = options or ClipAnalysisOptions()
    sources = project.sources_by_id
    worker: Any
    if operation == "colors":
        from ui.workers.color_worker import ColorAnalysisWorker

        worker = ColorAnalysisWorker(
            **common,
            sources_by_id=sources,
            parallelism=settings.color_analysis_parallelism,
        )
        return worker, worker.application
    if operation == "shots":
        from ui.workers.shot_type_worker import ShotTypeWorker
        from core.operations.shots import ShotTypeApplication, ShotTypeOptions

        worker = ShotTypeWorker(
            **common,
            sources_by_id=sources,
            parallelism=settings.local_model_parallelism,
            options=ShotTypeOptions(
                "cloud" if settings.shot_classifier_tier == "cloud" else "local",
                settings.shot_classifier_cloud_model,
            ),
        )
        return worker, ShotTypeApplication(project, worker.tasks)
    if operation == "classify":
        from ui.workers.classification_worker import ClassificationWorker
        from core.operations.classification import ClassificationApplication

        worker = ClassificationWorker(
            **common, top_k=options.top_k, parallelism=settings.local_model_parallelism
        )
        return worker, ClassificationApplication(project, worker.tasks)
    if operation == "detect_objects":
        from ui.workers.object_detection_worker import ObjectDetectionWorker
        from core.operations.object_detection import ObjectDetectionApplication

        worker = ObjectDetectionWorker(
            **common,
            confidence=options.confidence,
            detect_all=options.detect_all,
            parallelism=settings.local_model_parallelism,
        )
        return worker, ObjectDetectionApplication(project, worker.tasks, worker.options)
    if operation == "extract_text":
        from ui.workers.text_extraction_worker import TextExtractionWorker
        from core.operations.ocr import OcrApplication

        common.pop("skip_existing")
        method = settings.text_extraction_method
        worker = TextExtractionWorker(
            **common,
            sources_by_id=sources,
            use_vlm_fallback=method in ("vlm", "hybrid"),
            vlm_only=method == "vlm",
            vlm_model=(
                settings.text_extraction_vlm_model
                or settings.description_model_cloud
                or "gemini-3-flash-preview"
            )
            if method in ("vlm", "hybrid")
            else None,
        )
        return worker, OcrApplication(project, worker.tasks)
    if operation == "transcribe":
        from ui.workers.transcription_worker import TranscriptionWorker
        from core.operations.transcription import TranscriptionApplication

        source_ids = {clip.source_id for clip in clips}
        if len(source_ids) != 1 or next(iter(source_ids)) not in sources:
            raise ValueError("Transcription requires one existing source per batch")
        worker = TranscriptionWorker(
            **common,
            source=sources[clips[0].source_id],
            model_name=settings.transcription_model,
            language=settings.transcription_language,
            parallelism=settings.transcription_parallelism,
            backend=settings.transcription_backend,
            model_cache_dir=settings.model_cache_dir,
            min_free_disk_gb=settings.transcription_min_free_disk_gb,
            segmentation_mode=settings.transcription_segmentation_mode,
            segment_max_seconds=settings.transcription_segment_max_seconds,
        )
        return worker, TranscriptionApplication(project, worker.tasks)
    if operation == "face_embeddings":
        from ui.workers.face_detection_worker import FaceDetectionWorker
        from core.operations.faces import FaceApplication

        worker = FaceDetectionWorker(**common, sources_by_id=sources)
        return worker, FaceApplication(project, worker.tasks)
    if operation == "gaze":
        from ui.workers.gaze_worker import GazeAnalysisWorker
        from core.operations.gaze import GazeApplication

        worker = GazeAnalysisWorker(**common, sources_by_id=sources)
        return worker, GazeApplication(project, worker.tasks)
    if operation == "embeddings":
        from ui.workers.embedding_worker import EmbeddingAnalysisWorker
        from core.operations.embeddings import EmbeddingApplication

        worker = EmbeddingAnalysisWorker(**common)
        return worker, EmbeddingApplication(project, worker.tasks)
    if operation == "boundary_embeddings":
        from ui.workers.boundary_embedding_worker import BoundaryEmbeddingWorker
        from core.operations.boundary_embeddings import BoundaryEmbeddingApplication

        worker = BoundaryEmbeddingWorker(**common)
        return worker, BoundaryEmbeddingApplication(project, worker.tasks)
    if operation == "describe":
        from ui.workers.description_worker import DescriptionWorker
        from core.operations.description import (
            DescriptionApplication,
            DescriptionOptions,
            DEFAULT_PROMPT,
            resolve_tier,
        )

        tier = resolve_tier(options.tier or settings.description_model_tier)
        worker = DescriptionWorker(
            **common,
            sources=sources,
            tier=options.tier or settings.description_model_tier,
            parallelism=settings.description_parallelism,
            options=DescriptionOptions(
                tier=tier,
                prompt=options.prompt or DEFAULT_PROMPT,
                parallelism=settings.description_parallelism,
                model=settings.description_model_local
                if tier == "local"
                else settings.description_model_cloud,
                input_mode=settings.description_input_mode,
            ),
        )
        return worker, DescriptionApplication(project, worker.tasks)
    if operation == "cinematography":
        from ui.workers.cinematography_worker import CinematographyWorker
        from core.operations.cinematography import (
            CinematographyApplication,
            CinematographyOptions,
        )

        worker = CinematographyWorker(
            **common,
            sources_by_id=sources,
            mode=settings.cinematography_input_mode,
            model=settings.cinematography_model,
            parallelism=settings.cinematography_batch_parallelism,
            options=CinematographyOptions(
                settings.cinematography_tier,
                settings.cinematography_input_mode,
                settings.cinematography_model,
                settings.cinematography_local_model,
                settings.cinematography_batch_parallelism,
            ),
        )
        return worker, CinematographyApplication(project, worker.tasks)
    if operation == "custom_query":
        from ui.workers.custom_query_worker import CustomQueryWorker
        from core.operations.custom_query import (
            CustomQueryApplication,
            CustomQueryOptions,
        )

        if not query or not query.strip():
            raise ValueError("Custom query text is required")
        common["skip_existing"] = False
        tier = {"cpu": "local", "gpu": "cloud"}.get(
            settings.description_model_tier, settings.description_model_tier
        )
        worker = CustomQueryWorker(
            **common,
            sources_by_id=sources,
            query=query,
            tier=tier,
            parallelism=3 if tier == "cloud" else 1,
            options=CustomQueryOptions(
                tier,
                settings.description_model_local
                if tier == "local"
                else settings.description_model_cloud,
                3 if tier == "cloud" else 1,
            ),
        )
        return worker, CustomQueryApplication(project, worker.tasks)
    raise ValueError(f"Unsupported clip analysis operation: {operation}")
