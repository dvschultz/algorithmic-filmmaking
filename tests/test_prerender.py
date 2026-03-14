"""Tests for pre-render functions.

Covers:
- FFmpeg command construction for all 7 transform combos
- Idempotency (skip if output exists)
- Reverse safety limit
- Batch function with cancellation
- Export skips transforms when prerendered_path exists
- Project save/load copies and resolves prerendered files
"""

import subprocess
from pathlib import Path
from threading import Event
from unittest.mock import MagicMock, patch

import pytest

from core.remix.prerender import (
    _REVERSE_MAX_DURATION,
    prerender_clip,
    prerender_batch,
)
from core.sequence_export import SequenceExporter, ExportConfig
from models.sequence import Sequence, SequenceClip


# -- prerender_clip -----------------------------------------------------------

class TestPrerenderClip:
    """prerender_clip constructs correct FFmpeg commands."""

    def test_no_transforms_returns_none(self, tmp_path):
        """No transforms → no pre-render needed."""
        result = prerender_clip(
            source_path=Path("/video.mp4"),
            start_frame=0, end_frame=150, fps=30.0,
            hflip=False, vflip=False, reverse=False,
            output_dir=tmp_path, clip_id="c1",
        )
        assert result is None

    @patch("core.remix.prerender.find_binary", return_value="/usr/bin/ffmpeg")
    @patch("core.remix.prerender.subprocess.run")
    def test_hflip_only(self, mock_run, mock_find, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        result = prerender_clip(
            source_path=Path("/video.mp4"),
            start_frame=0, end_frame=150, fps=30.0,
            hflip=True, vflip=False, reverse=False,
            output_dir=tmp_path, clip_id="c1",
        )
        assert result is not None
        assert result.name == "c1_1_0_0.mp4"
        cmd = mock_run.call_args[0][0]
        assert "-vf" in cmd
        vf_idx = cmd.index("-vf")
        assert cmd[vf_idx + 1] == "hflip"

    @patch("core.remix.prerender.find_binary", return_value="/usr/bin/ffmpeg")
    @patch("core.remix.prerender.subprocess.run")
    def test_vflip_only(self, mock_run, mock_find, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        result = prerender_clip(
            source_path=Path("/video.mp4"),
            start_frame=0, end_frame=150, fps=30.0,
            hflip=False, vflip=True, reverse=False,
            output_dir=tmp_path, clip_id="c1",
        )
        assert result.name == "c1_0_1_0.mp4"
        cmd = mock_run.call_args[0][0]
        vf_idx = cmd.index("-vf")
        assert cmd[vf_idx + 1] == "vflip"

    @patch("core.remix.prerender.find_binary", return_value="/usr/bin/ffmpeg")
    @patch("core.remix.prerender.subprocess.run")
    def test_reverse_only(self, mock_run, mock_find, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        result = prerender_clip(
            source_path=Path("/video.mp4"),
            start_frame=0, end_frame=150, fps=30.0,
            hflip=False, vflip=False, reverse=True,
            output_dir=tmp_path, clip_id="c1",
        )
        assert result.name == "c1_0_0_1.mp4"
        cmd = mock_run.call_args[0][0]
        vf_idx = cmd.index("-vf")
        assert cmd[vf_idx + 1] == "reverse"
        assert "-af" in cmd
        af_idx = cmd.index("-af")
        assert cmd[af_idx + 1] == "areverse"

    @patch("core.remix.prerender.find_binary", return_value="/usr/bin/ffmpeg")
    @patch("core.remix.prerender.subprocess.run")
    def test_all_transforms(self, mock_run, mock_find, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        result = prerender_clip(
            source_path=Path("/video.mp4"),
            start_frame=0, end_frame=150, fps=30.0,
            hflip=True, vflip=True, reverse=True,
            output_dir=tmp_path, clip_id="c1",
        )
        assert result.name == "c1_1_1_1.mp4"
        cmd = mock_run.call_args[0][0]
        vf_idx = cmd.index("-vf")
        assert cmd[vf_idx + 1] == "hflip,vflip,reverse"

    @patch("core.remix.prerender.find_binary", return_value="/usr/bin/ffmpeg")
    @patch("core.remix.prerender.subprocess.run")
    def test_hflip_vflip(self, mock_run, mock_find, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        result = prerender_clip(
            source_path=Path("/video.mp4"),
            start_frame=0, end_frame=150, fps=30.0,
            hflip=True, vflip=True, reverse=False,
            output_dir=tmp_path, clip_id="c1",
        )
        assert result.name == "c1_1_1_0.mp4"
        cmd = mock_run.call_args[0][0]
        vf_idx = cmd.index("-vf")
        assert cmd[vf_idx + 1] == "hflip,vflip"
        # No areverse
        assert "-af" not in cmd

    @patch("core.remix.prerender.find_binary", return_value="/usr/bin/ffmpeg")
    @patch("core.remix.prerender.subprocess.run")
    def test_hflip_reverse(self, mock_run, mock_find, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        prerender_clip(
            source_path=Path("/video.mp4"),
            start_frame=0, end_frame=150, fps=30.0,
            hflip=True, vflip=False, reverse=True,
            output_dir=tmp_path, clip_id="c1",
        )
        cmd = mock_run.call_args[0][0]
        vf_idx = cmd.index("-vf")
        assert cmd[vf_idx + 1] == "hflip,reverse"

    @patch("core.remix.prerender.find_binary", return_value="/usr/bin/ffmpeg")
    @patch("core.remix.prerender.subprocess.run")
    def test_vflip_reverse(self, mock_run, mock_find, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        prerender_clip(
            source_path=Path("/video.mp4"),
            start_frame=0, end_frame=150, fps=30.0,
            hflip=False, vflip=True, reverse=True,
            output_dir=tmp_path, clip_id="c1",
        )
        cmd = mock_run.call_args[0][0]
        vf_idx = cmd.index("-vf")
        assert cmd[vf_idx + 1] == "vflip,reverse"

    def test_idempotent_skips_existing(self, tmp_path):
        """If output file already exists, skip FFmpeg and return path."""
        output = tmp_path / "c1_1_0_0.mp4"
        output.write_text("fake")  # Create existing file

        result = prerender_clip(
            source_path=Path("/video.mp4"),
            start_frame=0, end_frame=150, fps=30.0,
            hflip=True, vflip=False, reverse=False,
            output_dir=tmp_path, clip_id="c1",
        )
        assert result == output

    def test_reverse_safety_limit(self, tmp_path):
        """Clips over 15s skip reverse transform."""
        # 20 seconds at 30fps = 600 frames
        with patch("core.remix.prerender.find_binary", return_value="/usr/bin/ffmpeg"), \
             patch("core.remix.prerender.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            # reverse-only on a long clip: no transforms to apply → returns None
            result = prerender_clip(
                source_path=Path("/video.mp4"),
                start_frame=0, end_frame=600, fps=30.0,
                hflip=False, vflip=False, reverse=True,
                output_dir=tmp_path, clip_id="long",
            )
            assert result is None

    def test_reverse_safety_limit_with_hflip(self, tmp_path):
        """Long clip with hflip + reverse: reverse skipped, hflip still applied."""
        with patch("core.remix.prerender.find_binary", return_value="/usr/bin/ffmpeg"), \
             patch("core.remix.prerender.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            result = prerender_clip(
                source_path=Path("/video.mp4"),
                start_frame=0, end_frame=600, fps=30.0,
                hflip=True, vflip=False, reverse=True,
                output_dir=tmp_path, clip_id="long2",
            )
            assert result is not None
            cmd = mock_run.call_args[0][0]
            vf_idx = cmd.index("-vf")
            assert cmd[vf_idx + 1] == "hflip"  # reverse skipped
            assert "-af" not in cmd  # areverse skipped too

    def test_safety_limit_value(self):
        assert _REVERSE_MAX_DURATION == 15.0


# -- prerender_batch ----------------------------------------------------------

class TestPrerenderBatch:
    """prerender_batch processes clips correctly."""

    @patch("core.remix.prerender.prerender_clip")
    def test_batch_no_transforms_passes_through(self, mock_prerender, tmp_path):
        """Clips without transforms pass through with None path."""
        clip = MagicMock(id="c1", start_frame=0, end_frame=150)
        source = MagicMock(file_path=Path("/v.mp4"), fps=30.0)
        transforms = {"hflip": False, "vflip": False, "reverse": False}

        results = prerender_batch(
            clips_with_transforms=[(clip, source, transforms)],
            output_dir=tmp_path,
        )
        assert len(results) == 1
        assert results[0][2] is None
        mock_prerender.assert_not_called()

    @patch("core.remix.prerender.prerender_clip")
    def test_batch_with_transforms_calls_prerender(self, mock_prerender, tmp_path):
        mock_prerender.return_value = tmp_path / "c1_1_0_0.mp4"
        clip = MagicMock(id="c1", start_frame=0, end_frame=150)
        source = MagicMock(file_path=Path("/v.mp4"), fps=30.0)
        transforms = {"hflip": True, "vflip": False, "reverse": False}

        results = prerender_batch(
            clips_with_transforms=[(clip, source, transforms)],
            output_dir=tmp_path,
        )
        assert len(results) == 1
        assert results[0][2] == tmp_path / "c1_1_0_0.mp4"
        mock_prerender.assert_called_once()

    @patch("core.remix.prerender.prerender_clip")
    def test_batch_cancel(self, mock_prerender, tmp_path):
        """Cancellation stops processing."""
        cancel = Event()
        cancel.set()  # Already cancelled

        clip = MagicMock(id="c1", start_frame=0, end_frame=150)
        source = MagicMock(file_path=Path("/v.mp4"), fps=30.0)
        transforms = {"hflip": True, "vflip": False, "reverse": False}

        results = prerender_batch(
            clips_with_transforms=[(clip, source, transforms)],
            output_dir=tmp_path,
            cancel_event=cancel,
        )
        assert len(results) == 0
        mock_prerender.assert_not_called()

    @patch("core.remix.prerender.prerender_clip")
    def test_batch_progress_callback(self, mock_prerender, tmp_path):
        mock_prerender.return_value = None
        clip = MagicMock(id="c1", start_frame=0, end_frame=150)
        source = MagicMock(file_path=Path("/v.mp4"), fps=30.0)
        transforms = {"hflip": False, "vflip": False, "reverse": False}

        progress_calls = []
        results = prerender_batch(
            clips_with_transforms=[(clip, source, transforms)],
            output_dir=tmp_path,
            progress_cb=lambda c, t: progress_calls.append((c, t)),
        )
        # Should get progress(0, 1) and progress(1, 1)
        assert (0, 1) in progress_calls
        assert (1, 1) in progress_calls


# -- Export with prerendered_path ---------------------------------------------

class TestExportPrerenderedPath:
    """Export skips transforms when prerendered_path exists."""

    def test_build_video_filter_ignores_transforms_for_prerendered(self):
        """When using prerendered clip, transforms are already baked in."""
        exporter = SequenceExporter.__new__(SequenceExporter)
        config = ExportConfig(output_path=Path("/out.mp4"))

        # A clip with transforms AND prerendered_path — the export code path
        # would use _export_prerendered_segment which doesn't apply hflip/vflip/reverse.
        # Here we just verify that _build_video_filter with no seq_clip produces no filters.
        result = exporter._build_video_filter(
            config=config, bar_color=None, seq_clip=None,
        )
        assert result is None

    def test_export_prerendered_segment_no_transform_filters(self):
        """_export_prerendered_segment only applies scale and chromatic bar, not transforms."""
        exporter = SequenceExporter.__new__(SequenceExporter)
        exporter.ffmpeg_path = "/usr/bin/ffmpeg"
        config = ExportConfig(output_path=Path("/out.mp4"), width=1920, height=1080)

        with patch("core.sequence_export.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            exporter._export_prerendered_segment(
                prerendered_path=Path("/cache/clip.mp4"),
                output_path=Path("/tmp/segment.mp4"),
                config=config,
                bar_color=None,
            )
            cmd = mock_run.call_args[0][0]
            # Should have scale but not hflip/vflip/reverse
            if "-vf" in cmd:
                vf_idx = cmd.index("-vf")
                vf_str = cmd[vf_idx + 1]
                assert "hflip" not in vf_str
                assert "vflip" not in vf_str
                assert "reverse" not in vf_str
                assert "scale=" in vf_str


# -- Project save/load with prerendered clips --------------------------------

class TestPrerenderedProjectPersistence:
    """Project save copies pre-rendered files; load resolves relative paths."""

    def test_save_copies_prerendered_to_project_folder(self, tmp_path):
        """Pre-rendered files from cache are copied into transformed_clips/."""
        from core.project import _prepare_prerendered_clips

        # Simulate a cache directory with a pre-rendered clip
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cached_file = cache_dir / "c1_1_0_0.mp4"
        cached_file.write_text("fake video")

        project_dir = tmp_path / "project"
        project_dir.mkdir()

        seq = Sequence(algorithm="shuffle")
        seq.tracks[0].clips.append(
            SequenceClip(
                id="c1", in_point=0, out_point=90,
                hflip=True,
                prerendered_path=str(cached_file),
            )
        )

        mapping = _prepare_prerendered_clips(seq, project_dir)

        # File should be copied/linked
        dest = project_dir / "transformed_clips" / "c1_1_0_0.mp4"
        assert dest.exists()
        assert dest.read_text() == "fake video"
        # Mapping should contain the original -> project-local entry
        assert mapping[str(cached_file)] == str(dest)
        # In-memory path must NOT be mutated (Issue 091)
        assert seq.tracks[0].clips[0].prerendered_path == str(cached_file)

    def test_to_dict_stores_relative_path(self, tmp_path):
        """to_dict with base_path makes prerendered_path relative."""
        project_dir = tmp_path / "project"
        tc_dir = project_dir / "transformed_clips"
        tc_dir.mkdir(parents=True)
        clip_file = tc_dir / "c1_1_0_0.mp4"
        clip_file.write_text("fake")

        clip = SequenceClip(
            id="c1", in_point=0, out_point=90,
            hflip=True,
            prerendered_path=str(clip_file),
        )
        data = clip.to_dict(base_path=project_dir)
        assert data["prerendered_path"] == "transformed_clips/c1_1_0_0.mp4"

    def test_from_dict_resolves_relative_path(self, tmp_path):
        """from_dict with base_path resolves relative prerendered_path to absolute."""
        project_dir = tmp_path / "project"
        tc_dir = project_dir / "transformed_clips"
        tc_dir.mkdir(parents=True)
        clip_file = tc_dir / "c1_1_0_0.mp4"
        clip_file.write_text("fake")

        data = {
            "id": "c1",
            "source_clip_id": "src",
            "source_id": "s1",
            "start_frame": 0,
            "in_point": 0,
            "out_point": 90,
            "hflip": True,
            "prerendered_path": "transformed_clips/c1_1_0_0.mp4",
        }
        clip = SequenceClip.from_dict(data, base_path=project_dir)
        assert Path(clip.prerendered_path).is_absolute()
        assert Path(clip.prerendered_path) == clip_file.resolve()

    def test_round_trip_through_sequence(self, tmp_path):
        """Full Sequence save/load round-trip preserves prerendered_path."""
        project_dir = tmp_path / "project"
        tc_dir = project_dir / "transformed_clips"
        tc_dir.mkdir(parents=True)
        clip_file = tc_dir / "c1_1_0_1.mp4"
        clip_file.write_text("fake")

        seq = Sequence(algorithm="shuffle")
        seq.tracks[0].clips.append(
            SequenceClip(
                id="c1", in_point=0, out_point=90,
                hflip=True, reverse=True,
                prerendered_path=str(clip_file),
            )
        )

        data = seq.to_dict(base_path=project_dir)
        # Stored as relative
        clip_data = data["tracks"][0]["clips"][0]
        assert not Path(clip_data["prerendered_path"]).is_absolute()

        # Restore
        restored = Sequence.from_dict(data, base_path=project_dir)
        rc = restored.tracks[0].clips[0]
        assert rc.hflip is True
        assert rc.reverse is True
        assert Path(rc.prerendered_path).is_absolute()
        assert Path(rc.prerendered_path) == clip_file.resolve()

    def test_no_prerendered_path_unaffected(self):
        """Clips without prerendered_path are unaffected by base_path."""
        clip = SequenceClip(id="c1", in_point=0, out_point=90)
        data = clip.to_dict(base_path=Path("/some/project"))
        assert "prerendered_path" not in data

        restored = SequenceClip.from_dict(data, base_path=Path("/some/project"))
        assert restored.prerendered_path is None

    def test_skip_copy_when_already_in_project(self, tmp_path):
        """Files already in transformed_clips/ are not re-copied."""
        project_dir = tmp_path / "project"
        tc_dir = project_dir / "transformed_clips"
        tc_dir.mkdir(parents=True)
        clip_file = tc_dir / "c1_1_0_0.mp4"
        clip_file.write_text("original content")

        from core.project import _prepare_prerendered_clips

        seq = Sequence(algorithm="shuffle")
        seq.tracks[0].clips.append(
            SequenceClip(
                id="c1", in_point=0, out_point=90,
                hflip=True,
                prerendered_path=str(clip_file.resolve()),
            )
        )

        mapping = _prepare_prerendered_clips(seq, project_dir)
        # Content unchanged (not overwritten)
        assert clip_file.read_text() == "original content"
        # Mapping still returned for the already-in-project file
        assert str(clip_file) in mapping or str(clip_file.resolve()) in mapping

    def test_filename_collision_uses_numeric_suffix(self, tmp_path):
        """When dest exists with different content, a numeric suffix is added."""
        from core.project import _prepare_prerendered_clips

        # Create a cache file
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cached_file = cache_dir / "c1_1_0_0.mp4"
        cached_file.write_text("new content")

        # Create a project dir with existing file of DIFFERENT content
        project_dir = tmp_path / "project"
        tc_dir = project_dir / "transformed_clips"
        tc_dir.mkdir(parents=True)
        existing = tc_dir / "c1_1_0_0.mp4"
        existing.write_text("old different content")

        seq = Sequence(algorithm="shuffle")
        seq.tracks[0].clips.append(
            SequenceClip(
                id="c1", in_point=0, out_point=90,
                hflip=True,
                prerendered_path=str(cached_file),
            )
        )

        mapping = _prepare_prerendered_clips(seq, project_dir)

        # Original file should be untouched
        assert existing.read_text() == "old different content"
        # New file should be at a suffixed path
        dest_path = Path(mapping[str(cached_file)])
        assert dest_path.name == "c1_1_0_0_2.mp4"
        assert dest_path.exists()

    def test_hard_link_fallback_to_copy(self, tmp_path):
        """When os.link fails (e.g. cross-device), falls back to shutil.copy2."""
        from core.project import _prepare_prerendered_clips
        from unittest.mock import patch as mock_patch

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cached_file = cache_dir / "c1_1_0_0.mp4"
        cached_file.write_text("fake video")

        project_dir = tmp_path / "project"
        project_dir.mkdir()

        seq = Sequence(algorithm="shuffle")
        seq.tracks[0].clips.append(
            SequenceClip(
                id="c1", in_point=0, out_point=90,
                hflip=True,
                prerendered_path=str(cached_file),
            )
        )

        with mock_patch("core.project.os.link", side_effect=OSError("cross-device")):
            mapping = _prepare_prerendered_clips(seq, project_dir)

        dest = project_dir / "transformed_clips" / "c1_1_0_0.mp4"
        assert dest.exists()
        assert dest.read_text() == "fake video"
        assert mapping[str(cached_file)] == str(dest)
