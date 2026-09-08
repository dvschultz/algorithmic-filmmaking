"""Acknowledge GUI computation receipts from an atomically saved snapshot."""

from hashlib import sha256
import json
from pathlib import Path

from core.jobs.commits import StaleJobResult
from core.jobs.store import JobStore
from core.transcription_models import TranscriptSegment, WordTimestamp


def checkpoint_saved_gui_results(path: Path, snapshot: dict) -> int:
    """Call only after writing this exact snapshot, while holding its writer.

    Historical results that no longer match the saved transcript stay unchanged.
    Missing cache entries and other operation kinds cannot be acknowledged here.
    """
    receipts = snapshot.get("job_results", {})
    if not receipts:
        return 0
    from core.settings import load_settings

    database = load_settings().cache_dir / "jobs.db"
    if not database.is_file():
        return 0
    store = JobStore(database)
    canonical = str(path.expanduser().resolve())
    clips = {clip["id"]: clip for clip in snapshot.get("clips", [])}
    frames = {frame["id"]: frame for frame in snapshot.get("frames", [])}
    sources = {source["id"]: source for source in snapshot.get("sources", [])}
    audio_sources = {audio["id"]: audio for audio in snapshot.get("audio_sources", [])}
    pending = []
    for row in store.get_pending_results(list(receipts)):
        result_id = row["result_id"]
        receipt_digest = receipts[result_id]
        identity = json.loads(row["spec_json"])
        if (
            identity["kind"]
            not in (
                "gui_transcribe",
                "gui_audio_transcribe",
                "gui_audio_import",
                "gui_import_images",
                "gui_extract_frames",
                "gui_align_words",
                "gui_describe",
                "gui_custom_query",
                "gui_cinematography",
                "gui_classification",
                "gui_shots_clip",
                "gui_shots_frame",
                "gui_object_detection",
                "gui_faces",
                "gui_gaze",
                "gui_embeddings",
                "gui_boundary_embeddings",
                "gui_ocr_clip",
                "gui_ocr_frame",
            )
            or identity["version"] != 1
        ):
            continue
        if (
            sha256(row["spec_json"].encode()).hexdigest() != result_id
            or sha256(row["payload_json"].encode()).hexdigest() != receipt_digest
            or row["payload_digest"] != receipt_digest
        ):
            raise StaleJobResult("Saved GUI result identity or payload is corrupt")
        if identity["project_path"] != canonical or identity["inputs"][
            "project_id"
        ] != snapshot.get("id"):
            continue
        if identity["kind"] in ("gui_shots_clip", "gui_shots_frame"):
            from core.jobs.gui_shots import saved_shot_matches

            targets = frames if identity["kind"] == "gui_shots_frame" else clips
            if saved_shot_matches(identity, json.loads(row["payload_json"]), targets, sources, path):
                pending.append((result_id, receipt_digest))
            continue
        if identity["kind"] == "gui_import_images":
            from core.jobs.image_import import ImageImportRecord
            from core.operations.image_import import ImageImportOutcome
            from models.frame import Frame

            recorded = ImageImportRecord.from_dict(json.loads(row["payload_json"]))
            if recorded.batch_id != identity["target_id"] or recorded.batch_id != identity["inputs"]["source_id"]:
                raise StaleJobResult("Saved image import does not match its batch")
            outcome = ImageImportOutcome.from_dict(recorded.outcome)
            matches = True
            for imported in outcome.frames:
                saved = frames.get(imported.id)
                if saved is None:
                    matches = False
                    break
                actual, expected = Frame.from_dict(saved, path.parent).to_dict(), imported.to_model().to_dict()
                if any(actual.get(key) != expected.get(key) for key in ("id", "file_path", "width", "height", "source_id", "clip_id", "frame_number")):
                    matches = False
                    break
            if matches:
                pending.append((result_id, receipt_digest))
            continue
        if identity["kind"] == "gui_audio_import":
            from core.jobs.audio_import import AudioImportRecord
            from core.operations.audio_import import AudioImportTask, AudioImportOutcome
            from models.audio_source import AudioSource

            recorded = AudioImportRecord.from_dict(json.loads(row["payload_json"]))
            if recorded.path != identity["target_id"] or recorded.path != identity["inputs"]["source_id"]:
                raise StaleJobResult("Saved audio import does not match its target")
            task = AudioImportTask.from_dict(recorded.task)
            outcome = AudioImportOutcome.from_dict(recorded.outcome)
            saved = audio_sources.get(outcome.audio_source_id)
            if saved is not None:
                actual = AudioSource.from_dict(saved, path.parent).to_dict()
                expected = outcome.to_model(task).to_dict()
                if all(actual.get(key) == expected.get(key) for key in ("id", "file_path", "duration_seconds", "sample_rate", "channels")):
                    pending.append((result_id, receipt_digest))
            continue
        if identity["kind"] == "gui_audio_transcribe":
            from core.operations.audio_transcription import AudioTranscriptionOutcome

            audio = audio_sources.get(identity["target_id"])
            if audio is None or audio["id"] != identity["inputs"]["source_id"]:
                continue
            outcome = AudioTranscriptionOutcome.from_dict(json.loads(row["payload_json"]))
            if outcome.audio_source_id != audio["id"] or outcome.status != "succeeded":
                raise StaleJobResult("Saved audio transcription does not match its target")
            if audio.get("transcript") == [segment.to_dict() for segment in outcome.segments]:
                pending.append((result_id, receipt_digest))
            continue
        if identity["kind"] == "gui_extract_frames":
            from core.jobs.gui_frame_extraction import FrameExtractionRecord
            from core.operations.frame_extraction import FrameExtractionTask, FrameExtractionOutcome
            from models.frame import Frame

            recorded = FrameExtractionRecord.from_dict(json.loads(row["payload_json"]))
            if recorded.source_id != identity["target_id"] or recorded.source_id != identity["inputs"]["source_id"]:
                raise StaleJobResult("Saved extraction does not match its source")
            task = FrameExtractionTask.from_dict(recorded.task)
            outcome = FrameExtractionOutcome.from_dict(recorded.outcome)
            if not any(source["id"] == task.source_id for source in snapshot.get("sources", [])):
                continue
            matches = True
            for extracted in outcome.frames:
                saved = frames.get(extracted.id)
                if saved is None:
                    matches = False
                    break
                actual = Frame.from_dict(saved, path.parent).to_dict()
                expected = extracted.to_model(task).to_dict()
                fields = ("id", "file_path", "source_id", "clip_id", "frame_number", "width", "height", "thumbnail_path")
                if any(actual.get(key) != expected.get(key) for key in fields):
                    matches = False
                    break
            if matches:
                pending.append((result_id, receipt_digest))
            continue
        is_frame = identity["kind"] == "gui_ocr_frame" or (
            identity["kind"]
            in ("gui_describe", "gui_cinematography", "gui_classification", "gui_object_detection")
            and identity["inputs"]["task"]["target_type"] == "frame"
        )
        clip = (frames if is_frame else clips).get(identity["target_id"])
        if (
            clip is None
            or (clip.get("source_id") or "") != identity["inputs"]["source_id"]
        ):
            continue
        payload = json.loads(row["payload_json"])
        if payload["clip_id"] != clip["id"] or payload["status"] != "succeeded":
            raise StaleJobResult("Saved GUI result does not match its target")
        if identity["kind"] in ("gui_ocr_clip", "gui_ocr_frame"):
            from core.operations.ocr import OcrOutcome

            outcome = OcrOutcome.from_dict(payload)
            if outcome.target_type != ("frame" if is_frame else "clip"):
                raise StaleJobResult("Saved OCR target type changed")
            if outcome.record_json is not None and clip.get("analysis_records", {}).get("extract_text") != json.loads(outcome.record_json):
                continue
            if clip.get("extracted_texts") == [text.to_dict() for text in outcome.to_models()]:
                pending.append((result_id, receipt_digest))
            continue
        if identity["kind"] == "gui_embeddings":
            from core.operations.embeddings import EmbeddingOutcome

            embedding_outcome = EmbeddingOutcome.from_dict(payload)
            matches = clip.get("embedding") == list(embedding_outcome.vector) and clip.get("embedding_model") == embedding_outcome.model
            if embedding_outcome.record_json is not None and "embeddings" in clip.get("analysis_records", {}):
                from models.analysis_record import AnalysisRecord
                from core.artifacts import ArtifactStore

                saved_record = AnalysisRecord.from_dict(clip["analysis_records"]["embeddings"])
                expected_record = AnalysisRecord.from_dict(json.loads(embedding_outcome.record_json))
                value = json.loads(ArtifactStore().read_bytes(saved_record.artifact)) if saved_record.artifact is not None else saved_record.value
                matches = (
                    saved_record.identity == expected_record.identity and saved_record.state == expected_record.state == "succeeded"
                    and saved_record.provenance == expected_record.provenance and saved_record.input_json == expected_record.input_json
                    and value == {"embedding": list(embedding_outcome.vector), "embedding_model": embedding_outcome.model}
                )
            if matches:
                pending.append((result_id, receipt_digest))
            continue
        if identity["kind"] == "gui_boundary_embeddings":
            from core.operations.boundary_embeddings import BoundaryEmbeddingOutcome

            outcome = BoundaryEmbeddingOutcome.from_dict(payload)
            if (clip.get("first_frame_embedding") == list(outcome.first)
                and clip.get("last_frame_embedding") == list(outcome.last)
                and clip.get("embedding_model") == outcome.model):
                pending.append((result_id, receipt_digest))
            continue
        if identity["kind"] == "gui_gaze":
            from core.jobs.gaze import _saved_gaze
            from core.operations.gaze import GazeOutcome

            outcome = GazeOutcome.from_dict(payload)
            if outcome.record_json is not None and clip.get("analysis_records", {}).get("gaze") != json.loads(outcome.record_json):
                continue
            expected = _saved_gaze(
                {
                    "gaze_yaw": outcome.yaw,
                    "gaze_pitch": outcome.pitch,
                    "gaze_category": outcome.category,
                }
            )
            if {key: clip.get(key) for key in expected} == expected:
                pending.append((result_id, receipt_digest))
            continue
        if identity["kind"] == "gui_faces":
            from core.jobs.faces import _saved_faces
            from core.operations.faces import FaceOutcome

            expected = _saved_faces(FaceOutcome.from_dict(payload).face_dicts())
            if clip.get("face_embeddings") == expected:
                pending.append((result_id, receipt_digest))
            continue
        if identity["kind"] == "gui_object_detection":
            if payload.get("record_json") is not None and clip.get("analysis_records", {}).get("detect_objects") != json.loads(payload["record_json"]):
                continue
            if clip.get("person_count") == payload["person_count"] and (
                not identity["arguments"]["detect_all"]
                or clip.get("detected_objects") == payload["detections"]
            ):
                pending.append((result_id, receipt_digest))
            continue
        if identity["kind"] == "gui_classification":
            if payload.get("record_json") is not None and clip.get("analysis_records", {}).get("classify") != json.loads(payload["record_json"]):
                continue
            if clip.get("object_labels") == [label for label, _ in payload["labels"]]:
                pending.append((result_id, receipt_digest))
            continue
        if identity["kind"] == "gui_cinematography":
            from core.operations.cinematography import CinematographyOutcome

            analysis = CinematographyOutcome(**payload).analysis
            if analysis is not None and (
                clip.get("cinematography") == analysis.to_dict()
                and clip.get("shot_type") == analysis.get_simple_shot_type()
            ):
                pending.append((result_id, receipt_digest))
            continue
        if identity["kind"] == "gui_custom_query":
            expected = [
                *identity["inputs"]["task"]["previous_queries"],
                {
                    "query": payload["query"],
                    "match": payload["match"],
                    "confidence": round(payload["confidence"] or 0.0, 4),
                    "model": payload["model"],
                },
            ]
            # Later appends may follow this result before the explicit save.
            if (clip.get("custom_queries") or [])[: len(expected)] == expected:
                pending.append((result_id, receipt_digest))
            continue
        if identity["kind"] == "gui_describe":
            if (
                clip.get("description") == payload["description"]
                and clip.get("description_model") == payload["model"]
                and (is_frame or clip.get("description_frames") == 1)
            ):
                pending.append((result_id, receipt_digest))
            continue
        if identity["kind"] == "gui_transcribe":
            expected = [
                TranscriptSegment.from_dict(s).to_dict() for s in payload["segments"]
            ]
        else:
            from core.analysis.alignment import distribute_words_to_segments

            segments = [
                TranscriptSegment.from_dict(s)
                for s in json.loads(identity["inputs"]["task"]["transcript_json"])
            ]
            distribute_words_to_segments(
                segments, [WordTimestamp.from_dict(w) for w in payload["words"]]
            )
            expected = [segment.to_dict() for segment in segments]
        if clip.get("transcript") == expected:
            pending.append((result_id, receipt_digest))
    if pending:
        store.checkpoint_results(pending)
    return len(pending)
