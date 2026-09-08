"""Scalar records prove reuse and protect owner publication."""

from threading import Event
from unittest.mock import Mock
import pytest

from core.operations.scalars import ScalarApplication, scalar_task, run_scalars, FIELDS
from core.project import Project
from models.clip import Clip, Source


@pytest.fixture(params=["brightness", "volume"])
def setup(tmp_path, monkeypatch, request):
    source = Source(file_path=tmp_path / "video.mp4", fps=30)
    source.file_path.write_bytes(b"video")
    clip = Clip(source_id=source.id, start_frame=30, end_frame=60)
    project = Project(sources=[source], clips=[clip])
    for name in ("ffmpeg", "ffprobe"):
        (tmp_path / name).write_bytes(name.encode())
    monkeypatch.setattr(
        "core.binary_resolver.find_binary", lambda name: tmp_path / name
    )
    operation = request.param
    provider = Mock(return_value=0.5 if operation == "brightness" else -20.0)
    monkeypatch.setattr(
        "core.analysis.color.get_average_brightness"
        if operation == "brightness"
        else "core.analysis.audio.extract_clip_volume",
        provider,
    )
    return project, operation, provider


def run(setup, *, skip=True, apply=True, cancel=None):
    project, operation, _ = setup
    task = scalar_task(
        project.clips[0], project.sources[0], operation, skip_existing=skip
    )
    application = ScalarApplication(project, task)
    outcome = run_scalars((task,), cancel_event=cancel)[0]
    if apply:
        assert application.apply(project, outcome), outcome
    return outcome, application


def test_verified_scalar_reuses_after_save(setup, tmp_path):
    project, operation, provider = setup
    first, _ = run(setup)
    assert first.status == "succeeded"
    path = tmp_path / "project.sceneripper"
    project.save(path)
    loaded = Project.load(path)
    reused, _ = run((loaded, operation, provider))
    assert reused.status == "skipped"
    assert provider.call_count == 1
    if operation == "volume":
        assert provider.call_args.kwargs["_ffmpeg_path"] == tmp_path / "ffmpeg"
        assert provider.call_args.kwargs["_ffprobe_path"] == tmp_path / "ffprobe"


def test_sequencing_verifies_legacy_values_on_detached_clips(setup):
    from core.remix import generate_sequence

    project, operation, provider = setup
    clip, source = project.clips[0], project.sources[0]
    setattr(clip, FIELDS[operation], 0.1)
    before = clip.to_dict()
    result = generate_sequence(operation, [(clip, source)], 1)
    assert provider.call_count == 1
    assert clip.to_dict() == before
    assert result[0][0] is not clip
    assert operation in result[0][0].analysis_records
    assert getattr(result[0][0], FIELDS[operation]) == provider.return_value
    generate_sequence(operation, result, 1)
    assert provider.call_count == 1
    source.file_path.write_bytes(b"changed media")
    generate_sequence(operation, result, 1)
    assert provider.call_count == 2


def test_sequencing_cancel_discards_scalar_results(setup):
    from core.remix import generate_sequence

    project, operation, provider = setup
    cancel = Event()
    provider.side_effect = lambda *args, **kwargs: (cancel.set(), 0.5)[1]
    clip, source = project.clips[0], project.sources[0]
    before = clip.to_dict()
    assert generate_sequence(operation, [(clip, source)], 1, cancel_event=cancel) == []
    assert clip.to_dict() == before


def test_sequencing_does_not_sort_stale_values_after_failure(setup):
    from core.remix import generate_sequence

    project, operation, provider = setup
    clip, source = project.clips[0], project.sources[0]
    setattr(clip, FIELDS[operation], 0.1)
    provider.side_effect = RuntimeError("decode failed")
    before = clip.to_dict()
    with pytest.raises(RuntimeError, match="analysis failed"):
        generate_sequence(operation, [(clip, source)], 1)
    assert clip.to_dict() == before


def test_sequencing_rechecks_earlier_media_after_batch(setup, tmp_path):
    from core.remix import generate_sequence

    project, operation, provider = setup
    first, source = project.clips[0], project.sources[0]
    second_source = Source(file_path=tmp_path / "second.mp4", fps=30)
    second_source.file_path.write_bytes(b"second video")
    second = Clip(source_id=second_source.id, start_frame=0, end_frame=30)

    def compute(*args, **kwargs):
        if provider.call_count == 2:
            source.file_path.write_bytes(b"changed during later task")
        return 0.5

    provider.side_effect = compute
    with pytest.raises(RuntimeError, match="inputs changed"):
        generate_sequence(operation, [(first, source), (second, second_source)], 2)
    assert not first.analysis_records and not second.analysis_records


def test_sequencing_verified_no_audio_reuses(setup):
    from core.remix import generate_sequence

    project, operation, provider = setup
    if operation != "volume":
        return
    provider.return_value = None
    inputs = [(project.clips[0], project.sources[0])]
    first = generate_sequence(operation, inputs, 1)
    assert len(first) == 1  # Preserve the existing all-silent sequence policy.
    assert first[0][0].rms_volume is None
    assert first[0][0].analysis_records[operation].state == "succeeded"
    generate_sequence(operation, first, 1)
    assert provider.call_count == 1


@pytest.mark.parametrize("change", ["media", "range", "fps", "value", "legacy"])
def test_changed_inputs_recompute(setup, change):
    project, operation, provider = setup
    run(setup)
    clip, source = project.clips[0], project.sources[0]
    if change == "media":
        source.file_path.write_bytes(b"changed")
    elif change == "range":
        clip.end_frame += 1
    elif change == "fps":
        source.fps = 24
    elif change == "value":
        setattr(clip, FIELDS[operation], 0.2)
    else:
        clip.analysis_records.clear()
    assert run(setup)[0].status == "succeeded"
    assert provider.call_count == 2


def test_failure_preserves_old_value_and_records_failure(setup):
    project, operation, provider = setup
    run(setup)
    old = getattr(project.clips[0], FIELDS[operation])
    provider.side_effect = RuntimeError("failed")
    outcome, _ = run(setup, skip=False)
    assert outcome.status == "failed"
    assert getattr(project.clips[0], FIELDS[operation]) == old
    assert project.clips[0].analysis_records[operation].state == "failed"


@pytest.mark.parametrize("change", ["range", "session", "path", "replacement", "value"])
def test_stale_delivery_is_rejected(setup, change, tmp_path):
    project, operation, _ = setup
    outcome, application = run(setup, apply=False)
    clip = project.clips[0]
    if change == "range":
        clip.end_frame += 1
    elif change == "session":
        project.session.session_id = "changed"
    elif change == "path":
        project.path = tmp_path / "other.sceneripper"
    elif change == "replacement":
        project.clips_by_id[clip.id] = Clip(
            id=clip.id, source_id=clip.source_id, start_frame=30, end_frame=60
        )
    else:
        setattr(clip, FIELDS[operation], 0.7)
    assert not application.apply(project, outcome)
    assert operation not in clip.analysis_records


def test_cancellation_drops_completed_measurement(setup):
    project, operation, provider = setup
    cancel = Event()

    def compute(*args, **kwargs):
        cancel.set()
        return 0.4

    provider.side_effect = compute
    outcome, application = run(setup, apply=False, cancel=cancel)
    assert outcome.status == "unprocessed"
    assert outcome.record_json is None
    assert not application.apply(project, outcome)


def test_valid_empty_scalar_reuses(setup):
    project, operation, provider = setup
    provider.return_value = 0.0 if operation == "brightness" else None
    run(setup)
    assert run(setup)[0].status == "skipped"
    assert provider.call_count == 1


def test_spine_publishes_verified_records(setup):
    from core.spine.analyze import analyze_scalars

    project, operation, provider = setup
    result = analyze_scalars(project, operation)
    assert len(result["result"]["succeeded"]) == 1
    assert project.clips[0].analysis_records[operation].state == "succeeded"
    result = analyze_scalars(project, operation)
    assert len(result["result"]["skipped"]) == 1
    assert provider.call_count == 1


def test_binary_or_package_change_invalidates_reuse(setup, tmp_path, monkeypatch):
    project, operation, provider = setup
    run(setup)
    if operation == "volume":
        (tmp_path / "ffmpeg").write_bytes(b"changed binary")
    else:
        monkeypatch.setattr(
            "core.operations.scalars.model_runtime",
            lambda *args: {"name": "changed", "packages": {}},
        )
    assert run(setup)[0].status == "succeeded"
    assert provider.call_count == 2


def test_scalar_module_does_not_import_inference_runtimes():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            'import sys; import core.operations.scalars; assert not set(sys.modules) & {"PySide6", "cv2", "librosa", "torch", "av"}',
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr


def test_volume_executes_captured_binaries(tmp_path, monkeypatch):
    from core.analysis.audio import extract_clip_volume
    from types import SimpleNamespace

    process = Mock(
        side_effect=[
            SimpleNamespace(returncode=0, stdout="audio", stderr=""),
            SimpleNamespace(returncode=0, stderr="mean_volume: -20 dB"),
        ]
    )
    monkeypatch.setattr("core.analysis.audio.subprocess.run", process)
    monkeypatch.setattr(
        "core.analysis.audio.find_binary",
        Mock(side_effect=AssertionError("must use captured paths")),
    )
    assert (
        extract_clip_volume(
            tmp_path / "video.mp4",
            1.0,
            2.0,
            _ffmpeg_path=tmp_path / "encoder",
            _ffprobe_path=tmp_path / "probe",
        )
        == -20.0
    )
    assert process.call_args_list[0].args[0][0] == str(tmp_path / "probe")
    assert process.call_args_list[1].args[0][0] == str(tmp_path / "encoder")
