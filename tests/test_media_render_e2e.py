"""Decode actual output to verify source-frame identity, not FFmpeg arguments."""

from pathlib import Path
import subprocess

import pytest

from core.binary_resolver import find_binary
from core.project import Project
from core.sequence_export import ExportConfig, SequenceExporter
from models.clip import Clip, Source
from models.sequence import Sequence


@pytest.fixture
def ffmpeg():
    binary = find_binary("ffmpeg")
    if binary is None:
        pytest.skip("FFmpeg is required for decoded media regression tests")
    return binary


def make_numbered_video(ffmpeg: str, path: Path, rate: float = 24, count: int = 24, *, id_offset: int = 0) -> Source:
    from PIL import Image, ImageDraw
    payload = bytearray()
    for index in range(count):
        image = Image.new("RGB", (64, 64), "black")
        draw = ImageDraw.Draw(image)
        for bit in range(12):
            draw.rectangle((bit * 5, 0, bit * 5 + 4, 20), fill="white" if (index + id_offset) & (1 << bit) else "black")
        draw.text((5, 40), str(index + id_offset), fill="white")
        payload.extend(image.tobytes())
    subprocess.run([
        ffmpeg, "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", "64x64", "-r", str(rate), "-i", "pipe:0", "-c:v", "libx264",
        "-crf", "0", "-pix_fmt", "yuv420p", str(path),
    ], input=payload, check=True, timeout=30)
    return Source(file_path=path, fps=rate, width=64, height=64, duration_seconds=count / rate)


def decoded_ids(ffmpeg: str, path: Path, *, hflip=False, vflip=False) -> list[int]:
    import numpy as np
    output = subprocess.run([
        ffmpeg, "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ], capture_output=True, check=True, timeout=30).stdout
    stride = 64 * 64 * 3
    assert len(output) % stride == 0
    frames = np.frombuffer(output, dtype=np.uint8).reshape(-1, 64, 64, 3)
    if hflip:
        frames = frames[:, :, ::-1]
    if vflip:
        frames = frames[:, ::-1]
    return [sum((int(frame[8, bit * 5 + 2, 0]) > 128) << bit for bit in range(12)) for frame in frames]


@pytest.mark.parametrize("reverse", [False, True])
def test_managed_prerender_decodes_the_trimmed_and_transformed_frames(ffmpeg, tmp_path, reverse):
    from core.remix.prerender import prerender_clip

    source = make_numbered_video(ffmpeg, tmp_path / "source.mp4", count=48)
    kwargs = dict(source_path=source.file_path, start_frame=12, end_frame=24,
                  fps=24.0, hflip=True, vflip=True, reverse=reverse,
                  output_dir=tmp_path / "cache", clip_id="clip")
    output = prerender_clip(**kwargs)
    assert output is not None
    expected = list(range(12, 24))
    if reverse:
        expected.reverse()
    assert decoded_ids(ffmpeg, output, hflip=True, vflip=True) == expected
    assert prerender_clip(**kwargs) == output


@pytest.mark.parametrize("reverse,hflip,vflip", [(False, False, False), (True, True, True)])
def test_nonzero_offset_and_transforms_agree_in_preview_and_export(ffmpeg, tmp_path, reverse, hflip, vflip):
    from core.sequence_preview import render_sequence_preview, SequencePreviewSettings

    source = make_numbered_video(ffmpeg, tmp_path / "source.mp4", count=264)
    project = Project.new()
    project.sequence = Sequence(fps=24)
    project.add_source(source)
    clip = Clip(source_id=source.id, start_frame=240, end_frame=264)
    project.add_clips([clip])
    project.add_to_sequence([clip.id])
    entry = project.sequence.get_all_clips()[0]
    project.update_sequence_clip(entry.id, reverse=reverse, hflip=hflip, vflip=vflip)
    lookup = {clip.id: (clip, source)}
    output = tmp_path / "export.mp4"
    assert SequenceExporter(ffmpeg).export(
        project.sequence, project.sources_by_id, lookup,
        ExportConfig(output_path=output, fps=24, width=64, height=64, crf=0),
    )
    preview = render_sequence_preview(
        project.sequence, project.sources_by_id, lookup, cache_root=tmp_path / "cache",
        settings=SequencePreviewSettings(width=64, height=64, crf=0),
    )
    expected = list(range(240, 264))
    if reverse:
        expected.reverse()
    assert decoded_ids(ffmpeg, output, hflip=hflip, vflip=vflip) == expected
    assert decoded_ids(ffmpeg, preview.path, hflip=hflip, vflip=vflip) == expected


def test_export_retains_first_and_last_selected_source_frames(ffmpeg, tmp_path):
    source = make_numbered_video(ffmpeg, tmp_path / "source.mp4")
    project = Project.new()
    project.sequence = Sequence(fps=24)
    project.add_source(source)
    clip = Clip(source_id=source.id, start_frame=12, end_frame=24)
    project.add_clips([clip])
    project.add_to_sequence([clip.id])
    output = tmp_path / "export.mp4"
    assert SequenceExporter(ffmpeg).export(
        project.sequence, project.sources_by_id, {clip.id: (clip, source)},
        ExportConfig(output_path=output, fps=24, width=64, height=64, crf=0),
    )
    assert decoded_ids(ffmpeg, output) == list(range(12, 24))


def test_silent_video_then_audio_video_retains_audio_at_its_cut(ffmpeg, tmp_path):
    import numpy as np

    silent = make_numbered_video(ffmpeg, tmp_path / "silent.mp4")
    audible_path = tmp_path / "audible.mp4"
    subprocess.run([
        ffmpeg, "-v", "error", "-i", str(silent.file_path),
        "-f", "lavfi", "-i", "sine=frequency=1000:sample_rate=48000:duration=1",
        "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", str(audible_path),
    ], check=True, timeout=30)
    audible = Source(file_path=audible_path, fps=24, width=64, height=64, duration_seconds=1)
    project = Project.new()
    project.sequence = Sequence(fps=24)
    lookup = {}
    for source in (silent, audible, silent):
        if source.id not in project.sources_by_id:
            project.add_source(source)
        clip = Clip(source_id=source.id, start_frame=0, end_frame=24)
        project.add_clips([clip])
        project.add_to_sequence([clip.id])
        lookup[clip.id] = (clip, source)
    output = tmp_path / "cuts.mp4"
    assert SequenceExporter(ffmpeg).export(
        project.sequence, project.sources_by_id, lookup,
        ExportConfig(output_path=output, fps=24, width=64, height=64, crf=0),
    )
    assert decoded_ids(ffmpeg, output) == list(range(24)) * 3
    pcm = subprocess.run([
        ffmpeg, "-v", "error", "-i", str(output), "-map", "0:a:0",
        "-ac", "1", "-ar", "48000", "-f", "f32le", "pipe:1",
    ], capture_output=True, check=True, timeout=30).stdout
    samples = np.frombuffer(pcm, dtype="<f4")
    assert 144000 <= len(samples) < 145024  # Final AAC packet padding only.
    assert np.max(np.abs(samples[:47000])) < 0.001
    assert np.sqrt(np.mean(samples[49000:95000] ** 2)) > 0.05
    assert np.max(np.abs(samples[97000:144000])) < 0.001


@pytest.mark.parametrize("failure", ["encoder", "cancel", "none"])
@pytest.mark.parametrize("existing", [False, True])
def test_export_publication_is_atomic(ffmpeg, tmp_path, monkeypatch, failure, existing):
    source = make_numbered_video(ffmpeg, tmp_path / "source.mp4")
    project = Project.new()
    project.sequence = Sequence(fps=24)
    project.add_source(source)
    clip = Clip(source_id=source.id, start_frame=0, end_frame=24)
    project.add_clips([clip])
    project.add_to_sequence([clip.id])
    output = tmp_path / "result.mp4"
    if existing:
        output.write_bytes(b"previous successful export")
    cancelled = False

    def progress(value, message):
        nonlocal cancelled
        if value < 1:
            assert output.exists() == existing
            if existing:
                assert output.read_bytes() == b"previous successful export"
        if value >= 0.8 and failure == "cancel":
            cancelled = True

    exporter = SequenceExporter(ffmpeg)
    if failure == "encoder":
        monkeypatch.setattr(exporter, "_concat_segments", lambda **kwargs: False)
    success = exporter.export(
        project.sequence, project.sources_by_id, {clip.id: (clip, source)},
        ExportConfig(output_path=output, fps=24, width=64, height=64, crf=0, cancel_check=lambda: cancelled),
        progress_callback=progress,
    )
    assert success == (failure == "none")
    assert not list(tmp_path.glob(".sequence_render_*"))
    if success:
        assert decoded_ids(ffmpeg, output) == list(range(24))
    else:
        assert output.exists() == existing
        if existing:
            assert output.read_bytes() == b"previous successful export"


@pytest.mark.parametrize("rate", [24, 30000 / 1001])
def test_still_gap_and_short_music_match_preview(ffmpeg, tmp_path, rate):
    import numpy as np
    from PIL import Image
    from models.frame import Frame
    from core.sequence_preview import render_sequence_preview, SequencePreviewSettings

    still = tmp_path / "still.png"
    Image.new("RGB", (64, 64), "white").save(still)
    project = Project.new()
    project.sequence = Sequence(fps=rate)
    hold = round(rate) // 2
    frame = Frame(file_path=still, width=64, height=64)
    project.add_frames([frame])
    project.add_frames_to_sequence([frame.id], hold_frames=hold)
    project.update_sequence_clip(project.sequence.get_all_clips()[0].id, start_frame=hold)
    music = tmp_path / "music.wav"
    subprocess.run([
        ffmpeg, "-v", "error", "-f", "lavfi", "-i", "sine=frequency=1000:sample_rate=48000:duration=0.25", str(music),
    ], check=True, timeout=30)
    project.sequence.music_path = str(music)
    output = tmp_path / "export.mp4"
    assert SequenceExporter(ffmpeg).export(
        project.sequence, {}, {}, ExportConfig(output_path=output, fps=rate, width=64, height=64, crf=0),
        frames=project.frames_by_id,
    )
    preview = render_sequence_preview(
        project.sequence, {}, {}, cache_root=tmp_path / "cache", frames=project.frames_by_id,
        settings=SequencePreviewSettings(width=64, height=64, crf=0),
    )
    for path in (output, preview.path):
        assert decoded_ids(ffmpeg, path) == [0] * hold + [4095] * hold
        pcm = subprocess.run([
            ffmpeg, "-v", "error", "-i", str(path), "-map", "0:a:0",
            "-ac", "1", "-ar", "48000", "-f", "f32le", "pipe:1",
        ], capture_output=True, check=True, timeout=30).stdout
        samples = np.frombuffer(pcm, dtype="<f4")
        sample_count = round(hold * 2 / rate * 48000)
        assert sample_count <= len(samples) < sample_count + 1024
        assert np.sqrt(np.mean(samples[1000:11000] ** 2)) > 0.05
        assert np.max(np.abs(samples[13000:48000])) < 0.001


def test_mixed_rates_match_compiled_frame_mapping(ffmpeg, tmp_path):
    from core.render_plan import compile_render_plan

    project = Project.new()
    project.sequence = Sequence(fps=30)
    lookup = {}
    for index, rate in enumerate((24, 25, 30, 30000 / 1001)):
        source = make_numbered_video(ffmpeg, tmp_path / f"source{index}.mp4", rate=rate, count=264)
        project.add_source(source)
        clip = Clip(source_id=source.id, start_frame=240, end_frame=263)
        project.add_clips([clip])
        project.add_to_sequence([clip.id])
        lookup[clip.id] = (clip, source)
    plan = compile_render_plan(project.sequence, project.sources_by_id, lookup)
    output = tmp_path / "mixed.mp4"
    assert SequenceExporter(ffmpeg).export(
        project.sequence, project.sources_by_id, lookup,
        ExportConfig(output_path=output, fps=30, width=64, height=64, crf=0),
    )
    expected = [plan.source_frame_at(frame) for frame in range(plan.frame_count)]
    assert decoded_ids(ffmpeg, output) == expected
    assert plan.frame_count == round(float(project.sequence.duration_time) * 30)


def test_mixed_rate_preview_and_export_match_independent_frame_oracle(ffmpeg, tmp_path):
    from core.sequence_preview import SequencePreviewSettings, render_sequence_preview

    project = Project.new()
    project.sequence = Sequence(fps=30)
    lookup = {}
    for index, rate in enumerate((24, 25, 30, 30000 / 1001)):
        source = make_numbered_video(
            ffmpeg, tmp_path / f"source-{index}.mp4", rate=rate,
            count=264, id_offset=index * 512,
        )
        project.add_source(source)
        clip = Clip(source_id=source.id, start_frame=240, end_frame=263)
        project.add_clips([clip])
        project.add_to_sequence([clip.id])
        lookup[clip.id] = (clip, source)

    # Each clip selects 23 frames. Exact cumulative cut positions at 30 fps
    # round to 0, 29, 56, 79, 102; source images change at the nearest global
    # output boundary (half ties advance). Distinct source IDs catch wrong
    # clip selection as well as wrong frame selection. No compiler calls
    # determine this oracle.
    expected = (
        [240, 241, 241, 242, 243, 244, 245, 245, 246, 247, 248, 249, 249,
         250, 251, 252, 253, 253, 254, 255, 256, 257, 257, 258, 259, 260,
         261, 261, 262]
        + [752, 753, 754, 755, 755, 756, 757, 758, 759, 760, 760, 761,
           762, 763, 764, 765, 765, 766, 767, 768, 769, 770, 770, 771,
           772, 773, 774]
        + list(range(1264, 1287)) + list(range(1776, 1799))
    )
    output = tmp_path / "mixed-export.mp4"
    assert SequenceExporter(ffmpeg).export(
        project.sequence, project.sources_by_id, lookup,
        ExportConfig(output_path=output, fps=30, width=64, height=64, crf=0),
    )
    preview = render_sequence_preview(
        project.sequence, project.sources_by_id, lookup,
        cache_root=tmp_path / "cache",
        settings=SequencePreviewSettings(width=64, height=64, crf=0),
    )
    assert decoded_ids(ffmpeg, output) == expected
    assert decoded_ids(ffmpeg, preview.path) == expected


def test_vfr_import_records_verified_mapping_and_export_uses_it(ffmpeg, tmp_path):
    from core.spine.sources import probe_source
    from core.render_plan import compile_render_plan

    original = make_numbered_video(ffmpeg, tmp_path / "original.mp4")
    path = tmp_path / "variable.mp4"
    subprocess.run([
        ffmpeg, "-v", "error", "-i", str(original.file_path),
        "-vf", "select='eq(n,0)+eq(n,1)+eq(n,3)+eq(n,4)+eq(n,7)+eq(n,8)'",
        "-fps_mode", "vfr", "-c:v", "libx264", "-crf", "0", str(path),
    ], check=True, timeout=30)
    source = probe_source(path)
    assert source.variable_frame_rate and len(source.frame_timestamps) == 7
    project = Project.new()
    project.sequence = Sequence(fps=24)
    project.add_source(source)
    clip = Clip(source_id=source.id, start_frame=1, end_frame=5)
    project.add_clips([clip])
    project.add_to_sequence([clip.id])
    lookup = {clip.id: (clip, source)}
    plan = compile_render_plan(project.sequence, project.sources_by_id, lookup)
    output = tmp_path / "vfr-export.mp4"
    assert SequenceExporter(ffmpeg).export(
        project.sequence, project.sources_by_id, lookup,
        ExportConfig(output_path=output, fps=24, width=64, height=64, crf=0),
    )
    original_ids = [0, 1, 3, 4, 7, 8]
    assert decoded_ids(ffmpeg, output) == [original_ids[plan.source_frame_at(frame)] for frame in range(plan.frame_count)]
    assert decoded_ids(ffmpeg, output) == [1, 1, 3, 4, 4, 4, 7]


def test_export_rejects_range_beyond_decoded_source_end(ffmpeg, tmp_path):
    source = make_numbered_video(ffmpeg, tmp_path / "source.mp4")
    project = Project.new()
    project.sequence = Sequence(fps=24)
    project.add_source(source)
    clip = Clip(source_id=source.id, start_frame=12, end_frame=30)
    project.add_clips([clip])
    project.add_to_sequence([clip.id])
    output = tmp_path / "beyond.mp4"
    messages = []
    assert not SequenceExporter(ffmpeg).export(
        project.sequence, project.sources_by_id, {clip.id: (clip, source)},
        ExportConfig(output_path=output, fps=24, width=64, height=64),
        progress_callback=lambda value, message: messages.append(message),
    )
    assert not output.exists()
    assert "last decoded source frame" in messages[-1]


def test_seek_retains_video_and_audio_with_nonzero_source_pts(ffmpeg, tmp_path):
    import numpy as np
    from core.spine.sources import probe_source

    original = make_numbered_video(ffmpeg, tmp_path / "original.mp4", count=264)
    shifted = tmp_path / "shifted.mp4"
    subprocess.run([
        ffmpeg, "-v", "error", "-i", str(original.file_path),
        "-f", "lavfi", "-i", "sine=frequency=1000:sample_rate=48000:duration=11",
        "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac",
        "-output_ts_offset", "7", str(shifted),
    ], check=True, timeout=30)
    source = probe_source(shifted)
    project = Project.new()
    project.sequence = Sequence(fps=24)
    project.add_source(source)
    clip = Clip(source_id=source.id, start_frame=240, end_frame=264)
    project.add_clips([clip])
    project.add_to_sequence([clip.id])
    output = tmp_path / "shift-export.mp4"
    assert SequenceExporter(ffmpeg).export(
        project.sequence, project.sources_by_id, {clip.id: (clip, source)},
        ExportConfig(output_path=output, fps=24, width=64, height=64, crf=0),
    )
    assert decoded_ids(ffmpeg, output) == list(range(240, 264))
    pcm = subprocess.run([
        ffmpeg, "-v", "error", "-i", str(output), "-map", "0:a:0",
        "-ac", "1", "-ar", "48000", "-f", "f32le", "pipe:1",
    ], capture_output=True, check=True, timeout=30).stdout
    samples = np.frombuffer(pcm, dtype="<f4")
    assert 48000 <= len(samples) < 49024
    assert np.sqrt(np.mean(samples[1000:47000] ** 2)) > 0.05


def test_cancel_terminates_an_active_encoder(ffmpeg, tmp_path, monkeypatch):
    import time

    processes = []
    start = time.monotonic()
    popen = subprocess.Popen

    def capture(*args, **kwargs):
        process = popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr("core.sequence_export.subprocess.Popen", capture)
    config = ExportConfig(tmp_path / "unused.mp4", cancel_check=lambda: time.monotonic() - start > 0.2)
    assert not SequenceExporter(ffmpeg)._run_ffmpeg([
        ffmpeg, "-v", "error", "-re", "-f", "lavfi", "-i", "color=size=64x64:rate=24", "-f", "null", "-",
    ], config, timeout=10)
    assert len(processes) == 1 and processes[0].poll() is not None


def test_one_source_frame_fills_its_planned_output_interval(ffmpeg, tmp_path):
    source = make_numbered_video(ffmpeg, tmp_path / "one.mp4", count=1)
    project = Project.new()
    project.sequence = Sequence(fps=24)
    project.add_source(source)
    clip = Clip(source_id=source.id, start_frame=0, end_frame=1)
    project.add_clips([clip])
    project.add_to_sequence([clip.id])
    output = tmp_path / "one-export.mp4"
    assert SequenceExporter(ffmpeg).export(
        project.sequence, project.sources_by_id, {clip.id: (clip, source)},
        ExportConfig(output_path=output, fps=120, width=64, height=64, crf=0),
    )
    assert decoded_ids(ffmpeg, output) == [0] * 5
