"""Boundary prerequisites share model ownership and keep project inputs detached."""

from threading import Event, Thread
from unittest.mock import Mock

import pytest

from tests.test_description_operations import project_with_thumbnails


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 2)
    for source in project.sources:
        source.file_path.write_bytes(b"video")
    compute = Mock(return_value=([0.1] * 768, [0.2] * 768))
    unload = Mock()
    monkeypatch.setattr("core.feature_registry.check_feature", lambda _: (True, []))
    monkeypatch.setattr("core.analysis.embeddings.extract_boundary_embeddings", compute)
    monkeypatch.setattr("core.analysis.embeddings.unload_model", unload)
    return (
        [(c, project.sources_by_id[c.source_id]) for c in project.clips],
        compute,
        unload,
    )


def test_match_cut_does_not_mutate_project_clips(inputs):
    from tests.remix_compat import generate_sequence

    clips, _, unload = inputs
    result = generate_sequence("match_cut", clips, len(clips))
    assert all(c.first_frame_embedding is None for c, _ in clips)
    assert all(c.first_frame_embedding == [0.1] * 768 for c, _ in result)
    assert all(c.last_frame_embedding == [0.2] * 768 for c, _ in result)
    unload.assert_called_once()


def test_boundary_reuse_and_failed_refresh_are_recorded(inputs):
    from core.project import Project
    from core.spine.analyze import boundary_embeddings

    clips, compute, _ = inputs
    project = Project(
        sources=list({s.id: s for _, s in clips}.values()), clips=[c for c, _ in clips]
    )
    boundary_embeddings(project)
    boundary_embeddings(project)
    assert compute.call_count == 2
    assert all(
        c.analysis_records["boundary_embeddings"].state == "succeeded"
        for c in project.clips
    )
    compute.side_effect = RuntimeError("provider unavailable")
    boundary_embeddings(project, skip_existing=False)
    assert all(
        c.analysis_records["boundary_embeddings"].state == "failed"
        for c in project.clips
    )
    assert all(c.first_frame_embedding == [0.1] * 768 for c in project.clips)
    compute.side_effect = None
    boundary_embeddings(project)
    assert compute.call_count == 6


@pytest.mark.parametrize("change", ["range", "source", "fps", "runtime", "projection"])
def test_boundary_identity_invalidates_changed_inputs(inputs, monkeypatch, change):
    from core.project import Project
    from core.spine.analyze import boundary_embeddings

    clips, compute, _ = inputs
    project = Project(sources=list({s.id: s for _, s in clips}.values()), clips=[c for c, _ in clips])
    boundary_embeddings(project)
    if change == "range":
        project.clips[0].end_frame += 1
    elif change == "source":
        project.sources[0].file_path.write_bytes(b"changed media")
    elif change == "fps":
        project.sources[0].fps += 1
    elif change == "runtime":
        monkeypatch.setattr("core.analysis_model_identity.DINOV2_REVISION", "a" * 40)
    else:
        project.clips[0].last_frame_embedding = [0.9] * 768
    boundary_embeddings(project)
    assert compute.call_count == (3 if change in ("range", "projection") else 4)


@pytest.mark.parametrize("bad", [[0.0] * 768, [float("nan")] * 768, [1.0]])
def test_invalid_second_vector_does_not_publish_either_boundary(inputs, bad):
    from tests.remix_compat import generate_sequence

    clips, compute, _ = inputs
    compute.return_value = ([0.1] * 768, bad)
    result = generate_sequence("match_cut", clips, len(clips))
    assert all(
        c.first_frame_embedding is None and c.last_frame_embedding is None
        for c, _ in result
    )


def test_cancelled_boundary_call_stops_later_clips_and_publication(inputs):
    from tests.remix_compat import generate_sequence

    clips, compute, unload = inputs
    cancel = Event()

    def infer(**kwargs):
        cancel.set()
        return [0.1] * 768, [0.2] * 768

    compute.side_effect = infer
    assert generate_sequence("match_cut", clips, len(clips), cancel_event=cancel) == []
    assert compute.call_count == 1
    assert all(c.first_frame_embedding is None for c, _ in clips)
    unload.assert_called_once()


def test_source_replaced_during_boundary_call_is_rejected(inputs):
    from tests.remix_compat import generate_sequence

    clips, compute, _ = inputs

    def infer(**kwargs):
        kwargs["source_path"].write_bytes(b"different video")
        return [0.1] * 768, [0.2] * 768

    compute.side_effect = infer
    result = generate_sequence("match_cut", clips, len(clips))
    assert all(c.first_frame_embedding is None for c, _ in result)


def test_later_source_change_invalidates_earlier_detached_result(inputs):
    from tests.remix_compat import generate_sequence

    clips, compute, _ = inputs

    def infer(**kwargs):
        if compute.call_count == 2:
            kwargs["source_path"].write_bytes(b"source changed after first clip")
        return [0.1] * 768, [0.2] * 768

    compute.side_effect = infer
    result = generate_sequence("match_cut", clips, len(clips))
    assert all(c.first_frame_embedding is None for c, _ in result)


def test_model_download_failure_stops_remaining_items(inputs):
    from core.errors import ModelDownloadError
    from core.operations.boundary_embeddings import (
        BoundaryEmbeddingTask,
        run_boundary_embeddings,
    )

    clips, compute, unload = inputs
    compute.side_effect = ModelDownloadError("unavailable")
    tasks = tuple(
        BoundaryEmbeddingTask(c.id, s.file_path, c.start_frame, c.end_frame, s.fps)
        for c, s in clips
    )
    outcomes = run_boundary_embeddings(tasks)
    assert [o.status for o in outcomes] == ["failed", "unprocessed"]
    compute.assert_called_once()
    unload.assert_called_once()


@pytest.mark.parametrize(
    "fps,start,end",
    [(0, 0, 1), (float("nan"), 0, 1), (24, -1, 2), (24, 2, 2), (24, True, 2)],
)
def test_invalid_range_does_not_load_model(inputs, fps, start, end):
    from core.operations.boundary_embeddings import (
        BoundaryEmbeddingTask,
        run_boundary_embeddings,
    )

    clips, compute, unload = inputs
    outcomes = run_boundary_embeddings(
        (BoundaryEmbeddingTask("clip", clips[0][1].file_path, start, end, fps),)
    )
    assert outcomes[0].status == "failed"
    compute.assert_not_called()
    unload.assert_not_called()


def test_verified_boundaries_do_not_load_or_unload_model(inputs, monkeypatch):
    from tests.remix_compat import generate_sequence

    clips, compute, unload = inputs
    clips = generate_sequence("match_cut", clips, len(clips))
    compute.reset_mock()
    unload.reset_mock()
    monkeypatch.setattr(
        "core.feature_registry.check_feature",
        lambda _: (False, ["not needed for reuse"]),
    )
    result = generate_sequence("match_cut", clips, len(clips))
    assert len(result) == len(clips)
    compute.assert_not_called()
    unload.assert_not_called()


def test_failed_match_cut_refresh_does_not_use_stale_boundary_vectors(inputs):
    from tests.remix_compat import generate_sequence

    clips, compute, _ = inputs
    for clip, _ in clips:
        clip.first_frame_embedding = [0.7] * 768
        clip.last_frame_embedding = [0.8] * 768
    compute.side_effect = RuntimeError("provider unavailable")
    result = generate_sequence("match_cut", clips, len(clips))
    assert all(
        c.first_frame_embedding is None and c.last_frame_embedding is None
        for c, _ in result
    )
    assert all(c.first_frame_embedding == [0.7] * 768 for c, _ in clips)


def test_match_cut_cannot_relabel_foreign_thumbnail_vectors(inputs):
    from tests.remix_compat import generate_sequence

    clips, _, _ = inputs
    clips[0][0].embedding = [0.5] * 768
    clips[0][0].embedding_model = "foreign-model"
    with pytest.raises(ValueError, match="different or unknown model"):
        generate_sequence("match_cut", clips, len(clips))
    assert clips[0][0].embedding_model == "foreign-model"


def test_boundary_and_thumbnail_inference_share_model_ownership(inputs, monkeypatch):
    from core.operations.boundary_embeddings import (
        BoundaryEmbeddingTask,
        run_boundary_embeddings,
    )
    from core.operations.embeddings import (
        EmbeddingOptions,
        EmbeddingTask,
        run_embeddings,
    )

    clips, compute, unload = inputs
    entered, release, thumbnail_entered, thumbnail_started = (
        Event(),
        Event(),
        Event(),
        Event(),
    )
    failures = []

    def boundary(**kwargs):
        entered.set()
        assert release.wait(5)
        return [0.1] * 768, [0.2] * 768

    def thumbnail(paths):
        thumbnail_entered.set()
        return [[0.3] * 768 for _ in paths]

    def run(call):
        try:
            call()
        except BaseException as exc:
            failures.append(exc)

    compute.side_effect = boundary
    monkeypatch.setattr(
        "core.analysis.embeddings.extract_clip_embeddings_batch", thumbnail
    )
    clip, source = clips[0]
    task = BoundaryEmbeddingTask(
        clip.id, source.file_path, clip.start_frame, clip.end_frame, source.fps
    )
    first = Thread(target=lambda: run(lambda: run_boundary_embeddings((task,))))

    def run_thumbnail():
        thumbnail_started.set()
        run_embeddings(
            (EmbeddingTask(clip.id, clip.thumbnail_path),), EmbeddingOptions()
        )

    second = Thread(target=lambda: run(run_thumbnail))
    first.start()
    try:
        assert entered.wait(5)
        second.start()
        assert thumbnail_started.wait(5)
        assert not thumbnail_entered.wait(0.1)
        unload.assert_not_called()
    finally:
        release.set()
        first.join(5)
        if second.ident is not None:
            second.join(5)
    assert not first.is_alive() and not second.is_alive()
    assert failures == []
    assert thumbnail_entered.is_set()
    assert unload.call_count == 2


def test_native_provider_cancels_before_second_frame_and_closes_image(
    tmp_path, monkeypatch
):
    from PIL import Image
    from core.analysis.embeddings import extract_boundary_embeddings

    cancel = Event()
    image = Image.new("RGB", (2, 2))
    extract = Mock(return_value=image)
    monkeypatch.setattr("core.analysis.embeddings._extract_frame_image", extract)

    def infer(frame):
        cancel.set()
        return [0.1] * 768

    monkeypatch.setattr("core.analysis.embeddings._image_to_embedding", infer)
    with pytest.raises(InterruptedError):
        extract_boundary_embeddings(
            tmp_path / "video.mp4", 0, 2, 24, cancel_event=cancel
        )
    extract.assert_called_once()
    with pytest.raises(ValueError, match="closed image"):
        image.getpixel((0, 0))
