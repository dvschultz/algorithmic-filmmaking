"""Tests for random transform options on Dice Roll sequencer.

Covers:
- SequenceClip model round-trip (hflip, vflip, reverse)
- Backward compatibility (old project files load without errors)
- assign_random_transforms determinism and distribution
- FFmpeg filter chain for all transform combinations
- Reverse safety limit (>15s clips skip reverse)
"""

import random

import pytest

from models.sequence import SequenceClip, Sequence
from core.remix import assign_random_transforms
from core.sequence_export import SequenceExporter, ExportConfig
from pathlib import Path


# -- Model round-trip ---------------------------------------------------------

class TestSequenceClipTransformFields:
    """SequenceClip hflip/vflip/reverse serialize and deserialize correctly."""

    def test_transforms_default_to_false(self):
        clip = SequenceClip()
        assert clip.hflip is False
        assert clip.vflip is False
        assert clip.reverse is False

    def test_transforms_not_serialized_when_false(self):
        clip = SequenceClip()
        data = clip.to_dict()
        assert "hflip" not in data
        assert "vflip" not in data
        assert "reverse" not in data

    def test_hflip_round_trip(self):
        clip = SequenceClip(hflip=True)
        data = clip.to_dict()
        assert data["hflip"] is True
        restored = SequenceClip.from_dict(data)
        assert restored.hflip is True

    def test_vflip_round_trip(self):
        clip = SequenceClip(vflip=True)
        data = clip.to_dict()
        assert data["vflip"] is True
        restored = SequenceClip.from_dict(data)
        assert restored.vflip is True

    def test_reverse_round_trip(self):
        clip = SequenceClip(reverse=True)
        data = clip.to_dict()
        assert data["reverse"] is True
        restored = SequenceClip.from_dict(data)
        assert restored.reverse is True

    def test_all_transforms_round_trip(self):
        clip = SequenceClip(hflip=True, vflip=True, reverse=True)
        data = clip.to_dict()
        restored = SequenceClip.from_dict(data)
        assert restored.hflip is True
        assert restored.vflip is True
        assert restored.reverse is True

    def test_backward_compat_missing_fields(self):
        """Old project files without transform fields load with False defaults."""
        data = {
            "id": "old-clip",
            "source_clip_id": "src",
            "source_id": "s1",
            "start_frame": 0,
            "in_point": 0,
            "out_point": 90,
        }
        clip = SequenceClip.from_dict(data)
        assert clip.hflip is False
        assert clip.vflip is False
        assert clip.reverse is False

    def test_sequence_with_transforms_round_trip(self):
        """Full Sequence save/load preserves transform flags on clips."""
        seq = Sequence(algorithm="shuffle")
        seq.tracks[0].clips.append(
            SequenceClip(hflip=True, reverse=True, in_point=0, out_point=90)
        )
        data = seq.to_dict()
        restored = Sequence.from_dict(data)
        clip = restored.tracks[0].clips[0]
        assert clip.hflip is True
        assert clip.vflip is False
        assert clip.reverse is True

    def test_prerendered_path_default_none(self):
        clip = SequenceClip()
        assert clip.prerendered_path is None

    def test_prerendered_path_not_serialized_when_none(self):
        clip = SequenceClip()
        data = clip.to_dict()
        assert "prerendered_path" not in data

    def test_prerendered_path_round_trip(self):
        clip = SequenceClip(prerendered_path="/cache/clip_1_0_0.mp4")
        data = clip.to_dict()
        assert data["prerendered_path"] == "/cache/clip_1_0_0.mp4"
        restored = SequenceClip.from_dict(data)
        assert restored.prerendered_path == "/cache/clip_1_0_0.mp4"

    def test_prerendered_path_backward_compat(self):
        """Old project files without prerendered_path load with None."""
        data = {
            "id": "old-clip",
            "source_clip_id": "src",
            "source_id": "s1",
            "start_frame": 0,
            "in_point": 0,
            "out_point": 90,
        }
        clip = SequenceClip.from_dict(data)
        assert clip.prerendered_path is None

    def test_prerendered_path_with_transforms_round_trip(self):
        """prerendered_path round-trips alongside transform flags."""
        clip = SequenceClip(
            hflip=True, reverse=True,
            prerendered_path="/cache/clip_1_0_1.mp4",
            in_point=0, out_point=90,
        )
        data = clip.to_dict()
        restored = SequenceClip.from_dict(data)
        assert restored.hflip is True
        assert restored.reverse is True
        assert restored.prerendered_path == "/cache/clip_1_0_1.mp4"


# -- Transform assignment ----------------------------------------------------

class TestAssignRandomTransforms:
    """assign_random_transforms assigns flags correctly."""

    def _make_clips(self, n: int) -> list[SequenceClip]:
        return [SequenceClip(id=f"c{i}", in_point=0, out_point=90) for i in range(n)]

    def test_deterministic_with_seed(self):
        """Same seed produces same transform assignment."""
        clips_a = self._make_clips(20)
        clips_b = self._make_clips(20)
        opts = {"hflip": True, "vflip": True, "reverse": True}

        assign_random_transforms(clips_a, opts, seed=42)
        assign_random_transforms(clips_b, opts, seed=42)

        for a, b in zip(clips_a, clips_b):
            assert a.hflip == b.hflip
            assert a.vflip == b.vflip
            assert a.reverse == b.reverse

    def test_different_seeds_differ(self):
        """Different seeds produce different assignments (with high probability)."""
        clips_a = self._make_clips(50)
        clips_b = self._make_clips(50)
        opts = {"hflip": True, "vflip": True, "reverse": True}

        assign_random_transforms(clips_a, opts, seed=1)
        assign_random_transforms(clips_b, opts, seed=2)

        # At least one clip should differ
        any_different = any(
            a.hflip != b.hflip or a.vflip != b.vflip or a.reverse != b.reverse
            for a, b in zip(clips_a, clips_b)
        )
        assert any_different

    def test_roughly_50_percent_distribution(self):
        """Over many clips, each transform applies to roughly 50%."""
        clips = self._make_clips(1000)
        opts = {"hflip": True, "vflip": True, "reverse": True}

        assign_random_transforms(clips, opts, seed=123)

        hflip_count = sum(1 for c in clips if c.hflip)
        vflip_count = sum(1 for c in clips if c.vflip)
        reverse_count = sum(1 for c in clips if c.reverse)

        # Allow 40%-60% range for 1000 samples (very generous)
        for count in [hflip_count, vflip_count, reverse_count]:
            assert 400 <= count <= 600, f"Expected ~500, got {count}"

    def test_disabled_option_never_set(self):
        """Options set to False are never applied."""
        clips = self._make_clips(100)
        opts = {"hflip": True, "vflip": False, "reverse": False}

        assign_random_transforms(clips, opts, seed=99)

        assert all(not c.vflip for c in clips)
        assert all(not c.reverse for c in clips)
        # But hflip should have some True
        assert any(c.hflip for c in clips)

    def test_no_options_clears_all(self):
        """Empty transform options set all flags to False."""
        clips = self._make_clips(10)
        for c in clips:
            c.hflip = True
            c.vflip = True
        opts = {"hflip": False, "vflip": False, "reverse": False}

        assign_random_transforms(clips, opts)

        assert all(not c.hflip for c in clips)
        assert all(not c.vflip for c in clips)


# -- FFmpeg filter chain ------------------------------------------------------

class TestBuildVideoFilter:
    """_build_video_filter produces correct filter strings for all transform combos."""

    def setup_method(self):
        self.exporter = SequenceExporter.__new__(SequenceExporter)
        self.config = ExportConfig(output_path=Path("/out.mp4"))

    def test_no_transforms_no_filter(self):
        clip = SequenceClip()
        result = self.exporter._build_video_filter(
            config=self.config, bar_color=None, seq_clip=clip,
        )
        assert result is None

    def test_hflip_only(self):
        clip = SequenceClip(hflip=True)
        result = self.exporter._build_video_filter(
            config=self.config, bar_color=None, seq_clip=clip,
        )
        assert result == "hflip"

    def test_vflip_only(self):
        clip = SequenceClip(vflip=True)
        result = self.exporter._build_video_filter(
            config=self.config, bar_color=None, seq_clip=clip,
        )
        assert result == "vflip"

    def test_reverse_only(self):
        clip = SequenceClip(reverse=True)
        result = self.exporter._build_video_filter(
            config=self.config, bar_color=None, seq_clip=clip,
            apply_reverse=True,
        )
        assert result == "reverse"

    def test_hflip_vflip(self):
        clip = SequenceClip(hflip=True, vflip=True)
        result = self.exporter._build_video_filter(
            config=self.config, bar_color=None, seq_clip=clip,
        )
        assert result == "hflip,vflip"

    def test_hflip_reverse(self):
        clip = SequenceClip(hflip=True, reverse=True)
        result = self.exporter._build_video_filter(
            config=self.config, bar_color=None, seq_clip=clip,
            apply_reverse=True,
        )
        assert result == "hflip,reverse"

    def test_vflip_reverse(self):
        clip = SequenceClip(vflip=True, reverse=True)
        result = self.exporter._build_video_filter(
            config=self.config, bar_color=None, seq_clip=clip,
            apply_reverse=True,
        )
        assert result == "vflip,reverse"

    def test_all_transforms(self):
        clip = SequenceClip(hflip=True, vflip=True, reverse=True)
        result = self.exporter._build_video_filter(
            config=self.config, bar_color=None, seq_clip=clip,
            apply_reverse=True,
        )
        assert result == "hflip,vflip,reverse"

    def test_filter_order_scale_before_transforms(self):
        """scale comes before hflip/vflip/reverse."""
        config = ExportConfig(output_path=Path("/out.mp4"), width=1920, height=1080)
        clip = SequenceClip(hflip=True, reverse=True)
        result = self.exporter._build_video_filter(
            config=config, bar_color=None, seq_clip=clip,
            apply_reverse=True,
        )
        assert result == "scale=1920:1080,hflip,reverse"

    def test_chromatic_bar_after_transforms(self):
        """Chromatic bar filter comes after transforms."""
        config = ExportConfig(
            output_path=Path("/out.mp4"),
            show_chromatic_color_bar=True,
        )
        clip = SequenceClip(hflip=True)
        result = self.exporter._build_video_filter(
            config=config, bar_color=(255, 0, 0), seq_clip=clip,
        )
        assert result.startswith("hflip,drawbox")

    def test_reverse_flag_not_applied_without_apply_reverse(self):
        """reverse on SequenceClip is ignored if apply_reverse=False (safety limit)."""
        clip = SequenceClip(reverse=True)
        result = self.exporter._build_video_filter(
            config=self.config, bar_color=None, seq_clip=clip,
            apply_reverse=False,
        )
        assert result is None


# -- Reverse safety limit ----------------------------------------------------

class TestReverseSafetyLimit:
    """Clips over 15 seconds skip the reverse transform."""

    def test_safety_limit_value(self):
        assert SequenceExporter._REVERSE_MAX_DURATION == 15.0

    def test_short_clip_gets_reverse_in_filter(self):
        """Clip under 15s should include reverse in the filter chain."""
        # 5-second clip at 30fps = 150 frames
        clip = SequenceClip(reverse=True, in_point=0, out_point=150)
        exporter = SequenceExporter.__new__(SequenceExporter)
        config = ExportConfig(output_path=Path("/out.mp4"))

        duration = (clip.out_point - clip.in_point) / 30.0
        apply_reverse = clip.reverse and duration <= SequenceExporter._REVERSE_MAX_DURATION

        result = exporter._build_video_filter(
            config=config, bar_color=None, seq_clip=clip,
            apply_reverse=apply_reverse,
        )
        assert "reverse" in result

    def test_long_clip_skips_reverse_in_filter(self):
        """Clip over 15s should NOT include reverse in the filter chain."""
        # 20-second clip at 30fps = 600 frames
        clip = SequenceClip(reverse=True, in_point=0, out_point=600)
        exporter = SequenceExporter.__new__(SequenceExporter)
        config = ExportConfig(output_path=Path("/out.mp4"))

        duration = (clip.out_point - clip.in_point) / 30.0
        apply_reverse = clip.reverse and duration <= SequenceExporter._REVERSE_MAX_DURATION

        result = exporter._build_video_filter(
            config=config, bar_color=None, seq_clip=clip,
            apply_reverse=apply_reverse,
        )
        # Should be None since reverse is skipped and no other filters
        assert result is None
