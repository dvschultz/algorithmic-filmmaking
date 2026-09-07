"""Sequencer prerequisites must not publish into caller-owned clip models."""

from threading import Event
from unittest.mock import Mock

import pytest

from tests.test_description_operations import project_with_thumbnails


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 2)
    for clip in project.clips:
        clip.thumbnail_path = tmp_path / f"{clip.id}.jpg"
        clip.thumbnail_path.write_bytes(b"fake image")
    compute = Mock(side_effect=lambda paths: [[0.1] * 768 for _ in paths])
    monkeypatch.setattr("core.feature_registry.check_feature", lambda _: (True, []))
    monkeypatch.setattr(
        "core.analysis.embeddings.extract_clip_embeddings_batch", compute
    )
    monkeypatch.setattr("core.analysis.embeddings.unload_model", Mock())
    return [(c, project.sources_by_id[c.source_id]) for c in project.clips], compute


def test_similarity_computation_does_not_mutate_input_clips(inputs):
    from core.remix import generate_sequence

    clips, compute = inputs
    result = generate_sequence("similarity_chain", clips, len(clips))
    assert compute.call_count == 1
    assert all(c.embedding is None for c, _ in clips)
    assert all(c.embedding == [0.1] * 768 for c, _ in result)


@pytest.mark.parametrize("vector", [[0.0] * 768, [float("nan")] * 768, [1.0]])
def test_similarity_rejects_invalid_provider_vectors(inputs, vector):
    from core.remix import generate_sequence

    clips, compute = inputs
    compute.side_effect = lambda paths: [vector for _ in paths]
    result = generate_sequence("similarity_chain", clips, len(clips))
    assert all(c.embedding is None for c, _ in result)


def test_similarity_cancellation_rejects_late_batch(inputs):
    from core.remix import generate_sequence

    clips, compute = inputs
    cancel = Event()

    def infer(paths):
        cancel.set()
        return [[0.1] * 768 for _ in paths]

    compute.side_effect = infer
    assert (
        generate_sequence("similarity_chain", clips, len(clips), cancel_event=cancel)
        == []
    )
    assert all(c.embedding is None for c, _ in clips)


def test_staccato_worker_captures_private_inputs(inputs):
    from core.analysis.audio import AudioAnalysis
    from ui.dialogs.staccato_dialog import StaccatoGenerateWorker

    clips, _ = inputs
    worker = StaccatoGenerateWorker(clips, AudioAnalysis([], [], [], 1.0), "onsets")
    worker._auto_compute_embeddings()
    assert all(c.embedding is None for c, _ in clips)
    assert all(c.embedding == [0.1] * 768 for c, _ in worker._clips)


def test_short_batch_does_not_publish_partial_assignments(inputs):
    from core.remix import generate_sequence

    clips, compute = inputs
    compute.side_effect = lambda paths: [[0.1] * 768]
    result = generate_sequence("similarity_chain", clips, len(clips))
    assert all(c.embedding is None for c, _ in result)


def test_staccato_requires_thumbnails_even_when_no_inference_is_possible(inputs):
    from core.analysis.audio import AudioAnalysis
    from ui.dialogs.staccato_dialog import StaccatoGenerateWorker

    clips, compute = inputs
    for clip, _ in clips:
        clip.thumbnail_path = None
    worker = StaccatoGenerateWorker(clips, AudioAnalysis([], [], [], 1.0), "onsets")
    with pytest.raises(RuntimeError, match="Missing DINOv2 embeddings for 2 clips"):
        worker._auto_compute_embeddings()
    compute.assert_not_called()


def test_changed_thumbnail_is_not_used_in_sequence(inputs):
    from core.remix import generate_sequence

    clips, compute = inputs

    def infer(paths):
        paths[0].write_bytes(b"replacement image")
        return [[0.1] * 768 for _ in paths]

    compute.side_effect = infer
    result = generate_sequence("similarity_chain", clips, len(clips))
    by_id = {c.id: c for c, _ in result}
    assert by_id[clips[0][0].id].embedding is None
    assert by_id[clips[1][0].id].embedding == [0.1] * 768


def test_sequence_worker_snapshots_before_dispatch_and_cancels_batches(inputs):
    from ui.workers.sequence_worker import SequenceWorker

    clips, compute = inputs
    worker = SequenceWorker("similarity_chain", clips)
    assert worker._clips[0][0] is not clips[0][0]
    ready = Mock()
    worker.sequence_ready.connect(ready)
    worker.cancel()
    worker.run()
    compute.assert_not_called()
    ready.assert_not_called()
