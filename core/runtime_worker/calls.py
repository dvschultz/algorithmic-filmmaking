"""Allowlisted engine calls a worker may run on behalf of the host (plan U14).

Each entry maps a stable call name to ``(module, function)`` inside the
engine package. The host's ``core.runtime_families.isolated`` decorator
validates against this table at import time, so a call that is not listed
here can never reach a worker. Payloads are plain JSON: paths travel as
strings and callables are dropped.
"""

from __future__ import annotations

ISOLATED_CALLS: dict[str, tuple[str, str]] = {
    # diagnostics
    "selftest.identity": ("runtime_worker._selftest", "identity"),
    "selftest.execution": ("runtime_worker._selftest", "with_execution"),
    "selftest.missing": ("runtime_worker._selftest", "missing_dependency"),
    # vision (torch / transformers / ultralytics / insightface / mediapipe)
    "embeddings.thumbnails": ("core.analysis.embeddings", "extract_clip_embeddings_batch"),
    "embeddings.boundary": ("core.analysis.embeddings", "extract_boundary_embeddings"),
    "objects.detect": ("core.analysis.detection", "detect_objects"),
    "objects.detect_open_vocab": ("core.analysis.detection", "detect_objects_open_vocab"),
    "shots.classify": ("core.analysis.shots", "classify_shot_type"),
    "faces.from_image": ("core.analysis.faces", "extract_faces_from_image"),
    "faces.from_clip": ("core.analysis.faces", "extract_faces_from_clip"),
    "gaze.from_clip": ("core.analysis.gaze", "extract_gaze_from_clip"),
    "classification.frame": ("core.analysis.classification", "classify_frame"),
    # ocr (paddle; the VLM fallback stays in the host with its credentials)
    "ocr.paddle": ("core.analysis.ocr", "paddle_extract_text"),
    # local vlm (mlx / transformers)
    "vlm.describe_local": ("core.analysis.description", "describe_frame_local"),
    "vlm.describe_cpu": ("core.analysis.description", "describe_frame_cpu"),
    "vlm.custom_query_local": ("core.analysis.custom_query", "evaluate_custom_query_local"),
    "vlm.cinematography_local": ("core.analysis.cinematography", "analyze_cinematography_local"),
    # audio (librosa / demucs)
    "audio.analyze": ("core.analysis.audio", "analyze_audio"),
    "audio.clip_volume": ("core.analysis.audio", "extract_clip_volume"),
    "audio.music_file": ("core.analysis.audio", "analyze_music_file"),
    "audio.separate_stems": ("core.analysis.stem_separation", "separate_stems"),
    # alignment (ctc forced aligner)
    "alignment.engine": ("core.analysis.alignment", "_run_alignment_engine"),
}
