"""Detection metadata and result contracts for the shared runtime."""

import json
from threading import Event

from core.jobs.detection import detection_job_spec, run_detection_job
from core.operations.detection import DetectionGuard, DetectionRequest
from core.project import Project
from models.clip import Clip, Source


def test_detection_spec_captures_target_and_labels_session_persistence(tmp_path):
    source = Source(file_path=tmp_path / "video.mp4")
    source.file_path.write_bytes(b"media")
    project = Project.new(name="test")
    project.add_source(source)
    request = DetectionRequest.build(source.file_path)
    guard = DetectionGuard.capture(project, source.file_path)
    operation = detection_job_spec(request, guard)
    assert operation.session_id == project.session.session_id
    assert operation.persistence == "session_only"
    inputs = json.loads(operation.inputs_json)
    assert inputs["source_id"] == source.id
    assert inputs["media_stamp"] == list(request.media_stamp)
    assert inputs["target_digest"] == guard.target_digest
    assert operation.arguments == {"mode": "adaptive", "config": {}, "karaoke_config": {}}


def test_detection_job_serializes_complete_source_and_clip_payload(tmp_path, monkeypatch):
    source = Source(file_path=tmp_path / "video.mp4", fps=24, duration_seconds=10)
    clip = Clip(source_id=source.id, start_frame=12, end_frame=60)
    clip.notes = "detected text"
    clip.dominant_colors = [(1, 2, 3)]
    monkeypatch.setattr("core.jobs.detection.run_detection", lambda *a, **kw: (source, [clip]))
    payload = run_detection_job(DetectionRequest.build(source.file_path), lambda *a: None, Event())
    wire = json.loads(json.dumps(payload))
    assert wire["success"] is True
    restored_source = Source.from_dict(wire["source"])
    restored_clip = Clip.from_dict(wire["clips"][0])
    assert restored_source == source
    assert restored_clip.id == clip.id
    assert restored_clip.source_id == source.id
    assert (restored_clip.start_frame, restored_clip.end_frame) == (12, 60)
    assert restored_clip.notes == clip.notes
    assert restored_clip.dominant_colors == clip.dominant_colors
