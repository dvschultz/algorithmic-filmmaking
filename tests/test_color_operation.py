"""Contract tests for immutable color computation and explicit application."""

import threading
from dataclasses import FrozenInstanceError
from unittest.mock import patch

import pytest

from core.operations.colors import ColorApplication, color_request, compute_colors
from tests.test_spine_analyze import _build_project


def test_compute_is_immutable_and_application_notifies_once(tmp_path):
    project = _build_project(tmp_path, 2)
    request = color_request(project)
    with pytest.raises(FrozenInstanceError):
        request.num_colors = 3
    with patch("core.analysis.color.extract_dominant_colors", return_value=[(1, 2, 3)]):
        result = compute_colors(request)
    assert all(c.dominant_colors is None for c in project.clips)
    assert [o.status for o in result.outcomes] == ["succeeded", "succeeded"]
    application = ColorApplication(project, request)
    with patch.object(project, "update_clips", wraps=project.update_clips) as notify:
        applied = application.apply(result)
        assert applied == result
        assert all(c.dominant_colors == [(1, 2, 3)] for c in project.clips)
        assert notify.call_count == 1
        with pytest.raises(ValueError, match="already applied"):
            application.apply(result)
        assert notify.call_count == 1


def test_failures_and_unknown_ids_are_explicit(tmp_path):
    project = _build_project(tmp_path, 3)
    request = color_request(project, ["c-0", "c-1", "c-2", "missing"])
    with patch(
        "core.analysis.color.extract_dominant_colors",
        side_effect=[[(1, 2, 3)], [], RuntimeError("bad frame")],
    ):
        result = compute_colors(request)
    assert [(o.target_id, o.status, o.code) for o in result.outcomes] == [
        ("c-0", "succeeded", None),
        ("c-1", "failed", "no_colors_extracted"),
        ("c-2", "failed", "extraction_failed"),
        ("missing", "failed", "target_not_found"),
    ]
    assert result.outcomes[2].message == "bad frame"


def test_missing_media_does_not_dispatch_extraction(tmp_path):
    project = _build_project(tmp_path, 1)
    request = color_request(project)
    project.sources[0].file_path.unlink()
    with patch("core.analysis.color.extract_dominant_colors") as extract:
        result = compute_colors(request)
    assert result.outcomes[0].code == "source_file_missing"
    extract.assert_not_called()


def test_cancel_preserves_completed_and_accounts_for_unprocessed(tmp_path):
    project = _build_project(tmp_path, 3)
    cancel = threading.Event()

    def extract(**kwargs):
        cancel.set()
        return [(1, 2, 3)]

    with patch(
        "core.analysis.color.extract_dominant_colors", side_effect=extract
    ) as mock:
        result = compute_colors(color_request(project), cancel_event=cancel)
    assert mock.call_count == 1
    assert [o.status for o in result.outcomes] == [
        "succeeded",
        "unprocessed",
        "unprocessed",
    ]


def test_skip_policy_is_explicit_for_empty_palettes(tmp_path):
    project = _build_project(tmp_path, 1)
    project.clips[0].dominant_colors = []
    with patch(
        "core.analysis.color.extract_dominant_colors", return_value=[(1, 2, 3)]
    ) as extract:
        assert (
            compute_colors(color_request(project, skip_empty=True)).outcomes[0].status
            == "skipped"
        )
        extract.assert_not_called()
        assert (
            compute_colors(color_request(project, skip_empty=False)).outcomes[0].status
            == "succeeded"
        )


def test_changed_inputs_reject_late_results(tmp_path):
    project = _build_project(tmp_path, 1)
    request = color_request(project)
    application = ColorApplication(project, request)
    with patch("core.analysis.color.extract_dominant_colors", return_value=[(1, 2, 3)]):
        result = compute_colors(request)
    project.clips[0].start_frame += 1
    applied = application.apply(result)
    assert applied.outcomes[0].code == "stale_input"
    assert project.clips[0].dominant_colors is None


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_invalid_color_count_rejected_before_dispatch(tmp_path, count):
    with pytest.raises(ValueError, match="positive integer"):
        color_request(_build_project(tmp_path, 1), num_colors=count)


def test_parallel_cancellation_stops_refilling_the_pool(tmp_path):
    project = _build_project(tmp_path, 12)
    cancel = threading.Event()
    callbacks = []

    def progress(done, total, outcome):
        callbacks.append(threading.get_ident())
        cancel.set()

    with patch(
        "core.analysis.color.extract_dominant_colors", return_value=[(1, 2, 3)]
    ) as extract:
        result = compute_colors(
            color_request(project),
            parallelism=2,
            cancel_event=cancel,
            progress_callback=progress,
        )
    assert 1 <= extract.call_count <= 2
    assert len(result.outcomes) == 12
    assert sum(o.status == "unprocessed" for o in result.outcomes) >= 10
    assert set(callbacks) == {threading.get_ident()}


def test_frame_result_is_applied_through_project_model(tmp_path):
    from core.analysis_target import AnalysisTarget
    from core.operations.colors import request_from_targets
    from models.frame import Frame

    project = _build_project(tmp_path, 0)
    path = tmp_path / "still.png"
    path.touch()
    frame = Frame(id="still", file_path=path)
    project.add_frames([frame])
    request = request_from_targets([AnalysisTarget.from_frame(frame)])
    with patch(
        "core.analysis.color.extract_dominant_colors", return_value=[(4, 5, 6)]
    ) as extract:
        result = compute_colors(request)
    assert frame.dominant_colors is None
    assert extract.call_args.kwargs["image_path"] == path
    with patch.object(project, "update_frame", wraps=project.update_frame) as update:
        ColorApplication(project, request).apply(result)
    update.assert_called_once_with("still", dominant_colors=[(4, 5, 6)])
    assert frame.dominant_colors == [(4, 5, 6)]


def test_application_rejects_a_different_request(tmp_path):
    project = _build_project(tmp_path, 0)
    application = ColorApplication(project, color_request(project))
    with pytest.raises(ValueError, match="different request"):
        application.apply(compute_colors(color_request(project)))


def test_media_probe_error_is_a_per_target_failure(tmp_path):
    project = _build_project(tmp_path, 2)
    request = color_request(project)
    with (
        patch("pathlib.Path.is_file", side_effect=[PermissionError("denied"), True]),
        patch("core.analysis.color.extract_dominant_colors", return_value=[(1, 2, 3)]),
    ):
        result = compute_colors(request)
    assert [o.status for o in result.outcomes] == ["failed", "succeeded"]
    assert result.outcomes[0].message == "denied"
