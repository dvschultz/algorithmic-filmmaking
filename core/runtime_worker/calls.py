"""Allowlisted engine calls a worker may run on behalf of the host (plan U14).

Each entry maps a stable call name to ``(module, function)`` inside the
engine package. The host's ``core.runtime_families.isolated`` decorator
validates against this table at import time, so a call that is not listed
here can never reach a worker. Payloads are plain JSON: paths travel as
strings and callables are dropped.
"""

from __future__ import annotations

# Calls whose return value is a live object (a model) and must not be shipped back.
DISCARD_RESULT: frozenset[str] = frozenset({"faces.load", "gaze.load", "vlm.load"})

# Diagnostics only; refused unless the host enabled test tasks in its hello.
TEST_CALLS: frozenset[str] = frozenset({
    "selftest.identity", "selftest.execution", "selftest.missing", "selftest.model_error",
    "selftest.big_value", "selftest.execution_then_fail", "selftest.wait_for_cancel",
})

ISOLATED_CALLS: dict[str, tuple[str, str]] = {
    # diagnostics
    "selftest.identity": ("runtime_worker._selftest", "identity"),
    "selftest.execution": ("runtime_worker._selftest", "with_execution"),
    "selftest.missing": ("runtime_worker._selftest", "missing_dependency"),
    "selftest.model_error": ("runtime_worker._selftest", "model_error"),
    "selftest.big_value": ("runtime_worker._selftest", "big_value"),
    "selftest.execution_then_fail": ("runtime_worker._selftest", "execution_then_fail"),
    "selftest.wait_for_cancel": ("runtime_worker._selftest", "wait_for_cancel"),
    # vision (torch / transformers / ultralytics / insightface / mediapipe)
    "embeddings.thumbnails": ("core.analysis.embeddings", "extract_clip_embeddings_batch"),
    "embeddings.boundary": ("core.analysis.embeddings", "extract_boundary_embeddings"),
    "objects.detect": ("core.analysis.detection", "detect_objects"),
    "objects.detect_open_vocab": ("core.analysis.detection", "detect_objects_open_vocab"),
    "shots.classify": ("core.analysis.shots", "classify_shot_type"),
    "faces.from_image": ("core.analysis.faces", "extract_faces_from_image"),
    "faces.from_clip": ("core.analysis.faces", "extract_faces_from_clip"),
    "faces.environment": ("core.operations.face_records", "face_environment"),
    "gaze.from_clip": ("core.analysis.gaze", "extract_gaze_from_clip"),
    "classification.frame": ("core.analysis.classification", "classify_frame"),
    # ocr (paddle; the VLM fallback stays in the host with its credentials)
    "ocr.paddle": ("core.analysis.ocr", "paddle_extract_text"),
    # local vlm (mlx / transformers)
    "vlm.describe_local": ("core.analysis.description", "describe_frame_local"),
    "vlm.describe_cpu": ("core.analysis.description", "describe_frame_cpu"),
    "vlm.custom_query_local": ("core.analysis.custom_query", "evaluate_custom_query_local"),
    "vlm.cinematography_local": ("core.analysis.cinematography", "analyze_cinematography_local"),
    # model warm-ups used by batch operations (result discarded; model stays in the worker)
    "faces.load": ("core.analysis.faces", "_load_insightface"),
    "gaze.load": ("core.analysis.gaze", "load_face_mesh"),
    "vlm.load": ("core.analysis.description", "_load_local_model"),
    # model release (no-op in the host when the family is isolated: the worker holds the model)
    "embeddings.unload": ("core.analysis.embeddings", "unload_model"),
    "shots.unload": ("core.analysis.shots", "unload_model"),
    "objects.unload": ("core.analysis.detection", "unload_model"),
    "faces.unload": ("core.analysis.faces", "unload_model"),
    "gaze.unload": ("core.analysis.gaze", "unload_model"),
    "classification.unload": ("core.analysis.classification", "unload_model"),
    "vlm.unload": ("core.analysis.description", "unload_model"),
    # audio (librosa / demucs)
    "audio.analyze": ("core.analysis.audio", "analyze_audio"),
    "audio.clip_volume": ("core.analysis.audio", "extract_clip_volume"),
    "audio.music_file": ("core.analysis.audio", "analyze_music_file"),
    "audio.separate_stems": ("core.analysis.stem_separation", "separate_stems"),
    # alignment (ctc forced aligner)
    "alignment.engine": ("core.analysis.alignment", "_run_alignment_engine"),
}
