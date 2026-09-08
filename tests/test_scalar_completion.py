"""Scalar completion and estimates require current provenance, without inference."""

from unittest.mock import Mock

import pytest

from core.analysis_availability import (
    clear_operation_result,
    compute_operation_need_counts,
    operation_is_complete_for_clip,
)
from core.cost_estimates import estimate_sequence_cost
from core.operations.scalars import FIELDS
from tests import test_scalar_records

analyzed_inputs = test_scalar_records.setup
run = test_scalar_records.run


@pytest.mark.parametrize(
    "change", ["none", "legacy", "media", "range", "fps", "path", "value", "source"]
)
def test_scalar_completion_tracks_current_inputs(analyzed_inputs, change, tmp_path, monkeypatch):
    project, operation, _ = analyzed_inputs
    run(analyzed_inputs)
    clip, source = project.clips[0], project.sources[0]
    if change == "legacy":
        clip.analysis_records.clear()
    elif change == "media":
        source.file_path.write_bytes(b"changed media")
    elif change == "range":
        clip.end_frame += 1
    elif change == "fps":
        source.fps = 24
    elif change == "path":
        source.file_path = tmp_path / "other.mp4"
        source.file_path.write_bytes(b"video")
    elif change == "value":
        setattr(clip, FIELDS[operation], 0.1)
    elif change == "source":
        source = None
    blocked = Mock(side_effect=AssertionError("completion must not hash or infer"))
    monkeypatch.setattr("core.jobs.media.MediaFingerprints.get", blocked)
    monkeypatch.setattr("core.analysis.color.get_average_brightness", blocked)
    monkeypatch.setattr("core.analysis.audio.extract_clip_volume", blocked)
    complete = change == "none"
    assert operation_is_complete_for_clip(operation, clip, source=source) == complete
    sources = {source.id: source} if source else {}
    counts = compute_operation_need_counts([clip], [operation], sources_by_id=sources)
    assert counts == {operation: int(not complete)}
    estimates = estimate_sequence_cost(operation, [clip], sources_by_id=sources)
    assert len(estimates) == int(not complete)
    if estimates:
        assert estimates[0].clips_needing == 1
    blocked.assert_not_called()


def test_scalar_completion_includes_valid_empty_and_clear(analyzed_inputs):
    project, operation, provider = analyzed_inputs
    provider.return_value = None if operation == "volume" else 0.0
    run(analyzed_inputs)
    clip, source = project.clips[0], project.sources[0]
    assert operation_is_complete_for_clip(operation, clip, source=source)
    assert estimate_sequence_cost(operation, [clip], sources_by_id=project.sources_by_id) == []
    assert clear_operation_result(clip, operation)
    assert operation not in clip.analysis_records
    assert getattr(clip, FIELDS[operation]) is None
    assert not operation_is_complete_for_clip(operation, clip, source=source)


def test_scalar_completion_rejects_changed_runtime(analyzed_inputs, tmp_path, monkeypatch):
    project, operation, _ = analyzed_inputs
    run(analyzed_inputs)
    if operation == "volume":
        (tmp_path / "ffmpeg").write_bytes(b"different binary")
    else:
        monkeypatch.setattr("core.operations.scalars.scalar_runtime", lambda _: {"name": "changed"})
    clip, source = project.clips[0], project.sources[0]
    assert not operation_is_complete_for_clip(operation, clip, source=source)
    assert estimate_sequence_cost(operation, [clip], sources_by_id=project.sources_by_id)


def test_scalar_completion_tracks_requested_sampling(analyzed_inputs):
    from core.analysis_availability import scalar_analysis_is_complete

    project, operation, _ = analyzed_inputs
    run(analyzed_inputs)
    clip, source = project.clips[0], project.sources[0]
    assert scalar_analysis_is_complete(clip, source, operation, num_samples=5)
    assert scalar_analysis_is_complete(clip, source, operation, num_samples=3) == (operation == "volume")
    assert not scalar_analysis_is_complete(clip, source, operation, num_samples=0)


def test_scalar_failed_record_needs_analysis(analyzed_inputs):
    project, operation, provider = analyzed_inputs
    run(analyzed_inputs)
    provider.side_effect = RuntimeError("decode failed")
    run(analyzed_inputs, skip=False)
    clip, source = project.clips[0], project.sources[0]
    assert not operation_is_complete_for_clip(operation, clip, source=source)
    assert estimate_sequence_cost(operation, [clip], sources_by_id=project.sources_by_id)
