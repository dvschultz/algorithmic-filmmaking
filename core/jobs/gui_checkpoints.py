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
    pending = []
    for row in store.get_pending_results(list(receipts)):
        result_id = row["result_id"]
        receipt_digest = receipts[result_id]
        identity = json.loads(row["spec_json"])
        if (
            identity["kind"]
            not in (
                "gui_transcribe",
                "gui_align_words",
                "gui_describe",
                "gui_custom_query",
                "gui_cinematography",
                "gui_classification",
                "gui_object_detection",
                "gui_faces",
                "gui_gaze",
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
        is_frame = (
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
        if identity["kind"] == "gui_gaze":
            from core.jobs.gaze import _saved_gaze
            from core.operations.gaze import GazeOutcome

            outcome = GazeOutcome.from_dict(payload)
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
            if clip.get("person_count") == payload["person_count"] and (
                not identity["arguments"]["detect_all"]
                or clip.get("detected_objects") == payload["detections"]
            ):
                pending.append((result_id, receipt_digest))
            continue
        if identity["kind"] == "gui_classification":
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
