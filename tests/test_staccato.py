"""Tests for the Staccato beat-driven sequencer algorithm."""

import pytest
import numpy as np
from unittest.mock import MagicMock

from core.analysis.audio import (
    AudioAnalysis,
    OnsetDetectionConfig,
    analyze_audio,
    make_onset_detection_config,
)
from core.remix.staccato import (
    StaccatoSlot,
    StaccatoResult,
    StaccatoDebugInfo,
    expand_staccato_slot_segments,
    generate_beat_slots,
    generate_staccato_sequence,
    _cosine_distance,
    _select_clip_for_slot,
)


# --- AudioAnalysis onset_strengths tests ---

class TestAudioAnalysisOnsetStrengths:
    """Tests for the onset_strengths field and onset_strength_at() method."""

    def test_onset_strengths_default_empty(self):
        analysis = AudioAnalysis()
        assert analysis.onset_strengths == []

    def test_onset_strengths_serialization(self):
        analysis = AudioAnalysis(
            onset_times=[0.5, 1.0, 1.5],
            onset_strengths=[0.3, 1.0, 0.7],
        )
        d = analysis.to_dict()
        assert d["onset_strengths"] == [0.3, 1.0, 0.7]

        restored = AudioAnalysis.from_dict(d)
        assert restored.onset_strengths == [0.3, 1.0, 0.7]

    def test_onset_strengths_from_dict_missing_key(self):
        """Old serialized data without onset_strengths should default to empty."""
        d = {"tempo_bpm": 120, "beat_times": [0.5], "onset_times": [0.5]}
        restored = AudioAnalysis.from_dict(d)
        assert restored.onset_strengths == []

    def test_onset_strength_at_returns_nearest(self):
        analysis = AudioAnalysis(
            onset_times=[1.0, 2.0, 3.0],
            onset_strengths=[0.2, 0.8, 0.5],
        )
        # Exactly at onset
        assert analysis.onset_strength_at(1.0) == 0.2
        assert analysis.onset_strength_at(2.0) == 0.8
        # Closer to second onset
        assert analysis.onset_strength_at(1.6) == 0.8
        # Before first onset
        assert analysis.onset_strength_at(0.0) == 0.2
        # After last onset
        assert analysis.onset_strength_at(5.0) == 0.5

    def test_onset_strength_at_empty_data(self):
        analysis = AudioAnalysis()
        assert analysis.onset_strength_at(1.0) == 0.5


class TestTunedOnsetDetection:
    """Tests for Staccato-oriented onset detector configuration."""

    def test_drums_profile_is_more_sensitive_than_balanced(self):
        balanced = make_onset_detection_config("balanced", cut_density=5)
        drums = make_onset_detection_config("drums", cut_density=7)

        assert drums.delta < balanced.delta
        assert drums.hop_length < balanced.hop_length
        assert drums.backtrack is True
        assert drums.superflux is True

    def test_analyze_audio_passes_tuned_peak_pick_parameters(self, tmp_path, monkeypatch):
        calls = {}

        class FakeBeat:
            def beat_track(self, y, sr):
                return 120.0, np.array([1, 2])

        class FakeOnset:
            def onset_strength(self, **kwargs):
                calls["onset_strength"] = kwargs
                return np.array([0.1, 1.0, 0.2, 0.8])

            def onset_detect(self, **kwargs):
                calls["onset_detect"] = kwargs
                return np.array([1, 3])

            def onset_backtrack(self, events, energy):
                calls["onset_backtrack"] = {"events": events, "energy": energy}
                return np.array([0, 2])

        class FakeLibrosa:
            beat = FakeBeat()
            onset = FakeOnset()

            def load(self, path, sr):
                calls["load"] = {"path": path, "sr": sr}
                return np.ones(1024), sr

            def frames_to_time(self, frames, sr, hop_length=512):
                return np.asarray(frames) * hop_length / sr

        fake_librosa = FakeLibrosa()
        monkeypatch.setattr("core.analysis.audio._get_librosa", lambda: fake_librosa)

        config = OnsetDetectionConfig(
            profile="drums",
            hop_length=256,
            delta=0.025,
            wait_seconds=0.06,
            backtrack=True,
            superflux=False,
        )
        result = analyze_audio(
            tmp_path / "drums.wav",
            include_onsets=True,
            onset_config=config,
        )

        assert calls["onset_strength"]["hop_length"] == 256
        assert calls["onset_detect"]["onset_envelope"].tolist() == [0.1, 1.0, 0.2, 0.8]
        assert calls["onset_detect"]["hop_length"] == 256
        assert calls["onset_detect"]["delta"] == 0.025
        assert calls["onset_detect"]["backtrack"] is False
        assert calls["onset_detect"]["wait"] == 5
        assert calls["onset_backtrack"]["events"].tolist() == [1, 3]
        assert result.onset_times == pytest.approx([0.0, 512 / 22050])
        assert result.onset_strengths == [1.0, 0.8]

    def test_analyze_audio_uses_superflux_envelope(self, tmp_path, monkeypatch):
        calls = {}

        class FakeBeat:
            def beat_track(self, y, sr):
                return 120.0, np.array([1, 2])

        class FakeFeature:
            def melspectrogram(self, **kwargs):
                calls["melspectrogram"] = kwargs
                return np.array([[1.0, 2.0], [3.0, 4.0]])

        class FakeOnset:
            def onset_strength(self, **kwargs):
                calls["onset_strength"] = kwargs
                return np.array([0.2, 0.9])

            def onset_detect(self, **kwargs):
                calls["onset_detect"] = kwargs
                return np.array([1])

        class FakeLibrosa:
            beat = FakeBeat()
            feature = FakeFeature()
            onset = FakeOnset()

            def load(self, path, sr):
                return np.ones(1024), sr

            def frames_to_time(self, frames, sr, hop_length=512):
                return np.asarray(frames) * hop_length / sr

            def power_to_db(self, value, ref):
                calls["power_to_db"] = {"value": value, "ref": ref}
                return value

        fake_librosa = FakeLibrosa()
        monkeypatch.setattr("core.analysis.audio._get_librosa", lambda: fake_librosa)

        config = OnsetDetectionConfig(
            profile="drums",
            hop_length=256,
            delta=0.025,
            wait_seconds=0.06,
            backtrack=False,
            superflux=True,
        )
        result = analyze_audio(
            tmp_path / "drums.wav",
            include_onsets=True,
            onset_config=config,
        )

        assert calls["melspectrogram"]["hop_length"] == 256
        assert calls["onset_strength"]["S"].tolist() == [[1.0, 2.0], [3.0, 4.0]]
        assert calls["onset_strength"]["lag"] == 2
        assert calls["onset_strength"]["max_size"] == 3
        assert result.onset_strengths == [1.0]


# --- Beat slot generation tests ---

class TestGenerateBeatSlots:

    def _make_analysis(self, **kwargs):
        defaults = dict(
            tempo_bpm=120,
            beat_times=[0.5, 1.0, 1.5, 2.0],
            onset_times=[0.3, 0.8, 1.2, 1.7, 2.1],
            onset_strengths=[0.4, 1.0, 0.6, 0.8, 0.3],
            downbeat_times=[0.5, 2.0],
            duration_seconds=2.5,
        )
        defaults.update(kwargs)
        return AudioAnalysis(**defaults)

    def test_onsets_strategy(self):
        analysis = self._make_analysis()
        slots = generate_beat_slots(analysis, strategy="onsets")
        assert len(slots) == 5
        assert slots[0].start_time == 0.3
        assert slots[0].onset_strength == 0.4

    def test_beats_strategy(self):
        analysis = self._make_analysis()
        slots = generate_beat_slots(analysis, strategy="beats")
        assert len(slots) == 4
        assert slots[0].start_time == 0.5

    def test_downbeats_strategy(self):
        analysis = self._make_analysis()
        slots = generate_beat_slots(analysis, strategy="downbeats")
        assert len(slots) == 2
        assert slots[0].start_time == 0.5
        assert slots[1].start_time == 2.0

    def test_empty_analysis(self):
        analysis = AudioAnalysis()
        slots = generate_beat_slots(analysis, strategy="onsets")
        assert slots == []

    def test_explicit_cut_times_override_strategy_markers(self):
        analysis = self._make_analysis()
        slots = generate_beat_slots(
            analysis,
            strategy="onsets",
            cut_times=[0.5, 1.5],
        )
        assert [slot.start_time for slot in slots] == [0.5, 1.5]

    def test_short_slots_filtered(self):
        """Slots shorter than 0.1s should be skipped."""
        analysis = AudioAnalysis(
            onset_times=[1.0, 1.05, 2.0],
            onset_strengths=[0.5, 0.5, 0.5],
            duration_seconds=3.0,
        )
        slots = generate_beat_slots(analysis, strategy="onsets")
        # slot at 1.0 has end 1.05 = 0.05s duration -> filtered
        # slot at 1.05 has end 2.0 = 0.95s -> kept
        # slot at 2.0 has end 3.0 = 1.0s -> kept
        assert len(slots) == 2
        assert slots[0].start_time == 1.05
        assert slots[1].start_time == 2.0

    def test_slot_duration_property(self):
        slot = StaccatoSlot(start_time=1.0, end_time=2.5, onset_strength=0.7)
        assert slot.duration == pytest.approx(1.5)


# --- Cosine distance tests ---

class TestCosineDistance:

    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert _cosine_distance(v, v) == pytest.approx(0.0, abs=1e-6)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert _cosine_distance(a, b) == pytest.approx(1.0, abs=1e-6)

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert _cosine_distance(a, b) == pytest.approx(2.0, abs=1e-6)


# --- Clip selection tests ---

class TestSelectClipForSlot:

    def test_strong_onset_prefers_distant_clip(self):
        slot = StaccatoSlot(start_time=0, end_time=1.0, onset_strength=1.0)
        prev_emb = [1.0, 0.0, 0.0]
        clips = [
            (0, [0.99, 0.1, 0.0]),   # very similar
            (1, [0.0, 1.0, 0.0]),     # orthogonal (distance ~1.0)
            (2, [0.5, 0.5, 0.707]),   # moderate distance
        ]
        durations = [2.0, 2.0, 2.0]
        idx, cos_dist, score = _select_clip_for_slot(slot, prev_emb, clips, durations)
        assert idx == 1  # Most distant
        assert cos_dist == pytest.approx(1.0, abs=0.01)
        assert score < 0.1  # Good match: distance ~1.0, target 1.0

    def test_weak_onset_prefers_similar_clip(self):
        slot = StaccatoSlot(start_time=0, end_time=1.0, onset_strength=0.0)
        prev_emb = [1.0, 0.0, 0.0]
        clips = [
            (0, [0.99, 0.1, 0.0]),   # very similar
            (1, [0.0, 1.0, 0.0]),     # orthogonal
        ]
        durations = [2.0, 2.0]
        idx, cos_dist, score = _select_clip_for_slot(slot, prev_emb, clips, durations)
        assert idx == 0  # Most similar
        assert cos_dist is not None
        assert cos_dist < 0.5

    def test_first_clip_no_prev_embedding(self):
        slot = StaccatoSlot(start_time=0, end_time=1.0, onset_strength=0.5)
        clips = [(0, [1.0, 0.0]), (1, [0.0, 1.0])]
        durations = [2.0, 2.0]
        idx, cos_dist, score = _select_clip_for_slot(slot, None, clips, durations)
        assert idx == 0  # Returns first clip
        assert cos_dist is None
        assert score == 0.0


# --- Full sequence generation tests ---

class TestGenerateStaccatoSequence:

    def _make_clip(self, clip_id, embedding=None, start_frame=0, end_frame=48):
        clip = MagicMock()
        clip.id = clip_id
        clip.embedding = embedding
        clip.start_frame = start_frame
        clip.end_frame = end_frame
        clip.thumbnail_path = None
        return clip

    def _make_source(self, source_id, fps=24.0):
        source = MagicMock()
        source.id = source_id
        source.fps = fps
        return source

    def test_basic_generation(self):
        clips = [
            (self._make_clip("c1", [1.0, 0.0]), self._make_source("s1")),
            (self._make_clip("c2", [0.0, 1.0]), self._make_source("s1")),
            (self._make_clip("c3", [0.5, 0.5]), self._make_source("s1")),
        ]
        analysis = AudioAnalysis(
            beat_times=[0.5, 1.0, 1.5, 2.0],
            onset_times=[0.5, 1.0, 1.5, 2.0],
            onset_strengths=[0.2, 1.0, 0.3, 0.8],
            duration_seconds=2.5,
        )
        result = generate_staccato_sequence(clips, analysis, strategy="onsets")
        assert len(result) == 4  # 4 onset slots
        # Each result is a (Clip, Source, slot_duration) tuple
        for entry in result:
            clip, source, slot_duration = entry
            assert hasattr(clip, 'id')
            assert hasattr(source, 'id')
            assert slot_duration > 0

    def test_empty_clips(self):
        analysis = AudioAnalysis(beat_times=[1.0], onset_times=[1.0])
        result = generate_staccato_sequence([], analysis)
        assert result == []

    def test_empty_analysis(self):
        clips = [(self._make_clip("c1"), self._make_source("s1"))]
        analysis = AudioAnalysis()
        result = generate_staccato_sequence(clips, analysis)
        assert result == []

    def test_clips_repeat_when_more_slots_than_clips(self):
        """With 1 clip and 4 slots, the clip should repeat."""
        clips = [
            (self._make_clip("c1", [1.0, 0.0]), self._make_source("s1")),
        ]
        analysis = AudioAnalysis(
            beat_times=[0.5, 1.0, 1.5, 2.0],
            onset_times=[0.5, 1.0, 1.5, 2.0],
            onset_strengths=[0.5, 0.5, 0.5, 0.5],
            duration_seconds=2.5,
        )
        result = generate_staccato_sequence(clips, analysis, strategy="onsets")
        assert len(result) == 4
        # All should be the same clip since there's only one
        assert all(entry[0].id == "c1" for entry in result)

    def test_progress_callback(self):
        clips = [
            (self._make_clip("c1", [1.0, 0.0]), self._make_source("s1")),
        ]
        analysis = AudioAnalysis(
            onset_times=[1.0, 2.0],
            onset_strengths=[0.5, 0.5],
            duration_seconds=3.0,
        )
        progress_calls = []
        result = generate_staccato_sequence(
            clips, analysis, strategy="onsets",
            progress_cb=lambda c, t: progress_calls.append((c, t)),
        )
        assert len(result) == 2
        # Should have progress calls for each slot + final
        assert len(progress_calls) >= 2

    def test_generation_uses_explicit_cut_times(self):
        clips = [
            (self._make_clip("c1", [1.0, 0.0]), self._make_source("s1")),
            (self._make_clip("c2", [0.0, 1.0]), self._make_source("s1")),
        ]
        analysis = AudioAnalysis(
            beat_times=[0.5, 1.0, 1.5, 2.0],
            onset_times=[0.5, 1.0, 1.5, 2.0],
            onset_strengths=[0.2, 0.9, 0.3, 0.8],
            duration_seconds=2.5,
        )
        result = generate_staccato_sequence(
            clips,
            analysis,
            strategy="onsets",
            cut_times=[0.5, 1.5],
        )
        assert len(result) == 2
        assert result.debug is not None
        assert result.debug.total_slots == 2


class TestExpandStaccatoSlotSegments:

    def test_expands_short_clip_into_looping_segments(self):
        clip = MagicMock()
        clip.start_frame = 10
        clip.end_frame = 58  # 48 frames = 2 seconds at 24fps
        source = MagicMock()
        source.fps = 24.0

        segments = expand_staccato_slot_segments(clip, source, slot_duration=5.0)

        assert segments == [
            (10, 58),
            (10, 58),
            (10, 34),
        ]


# --- StaccatoResult wrapper tests ---

class TestStaccatoResult:

    def test_acts_like_list_iter(self):
        items = [("a", 1), ("b", 2)]
        result = StaccatoResult(items)
        assert list(result) == items

    def test_acts_like_list_len(self):
        result = StaccatoResult([("a", 1), ("b", 2)])
        assert len(result) == 2

    def test_acts_like_list_getitem(self):
        items = [("a", 1), ("b", 2)]
        result = StaccatoResult(items)
        assert result[0] == ("a", 1)
        assert result[1] == ("b", 2)

    def test_acts_like_list_bool(self):
        assert bool(StaccatoResult([("a", 1)])) is True
        assert bool(StaccatoResult([])) is False

    def test_equality_with_list(self):
        items = [("a", 1), ("b", 2)]
        result = StaccatoResult(items)
        assert result == items

    def test_carries_debug_info(self):
        debug = StaccatoDebugInfo(
            strategy="onsets", total_slots=3, total_clips_available=5,
        )
        result = StaccatoResult([("a", 1)], debug=debug)
        assert result.debug is debug
        assert result.debug.strategy == "onsets"

    def test_debug_defaults_to_none(self):
        result = StaccatoResult([])
        assert result.debug is None


# --- Debug info from sequence generation ---

class TestStaccatoDebugInfo:

    def _make_clip(self, clip_id, embedding=None, start_frame=0, end_frame=48):
        clip = MagicMock()
        clip.id = clip_id
        clip.name = clip_id
        clip.embedding = embedding
        clip.start_frame = start_frame
        clip.end_frame = end_frame
        clip.thumbnail_path = None
        return clip

    def _make_source(self, source_id, fps=24.0, file_path="video.mp4"):
        source = MagicMock()
        source.id = source_id
        source.fps = fps
        source.file_path = MagicMock()
        source.file_path.name = file_path
        return source

    def test_debug_info_populated(self):
        clips = [
            (self._make_clip("c1", [1.0, 0.0]), self._make_source("s1")),
            (self._make_clip("c2", [0.0, 1.0]), self._make_source("s1")),
        ]
        analysis = AudioAnalysis(
            onset_times=[0.5, 1.0, 1.5],
            onset_strengths=[0.3, 0.9, 0.5],
            duration_seconds=2.0,
        )
        result = generate_staccato_sequence(clips, analysis, strategy="onsets")

        assert isinstance(result, StaccatoResult)
        assert result.debug is not None
        assert result.debug.strategy == "onsets"
        assert result.debug.total_slots == 3
        assert result.debug.total_clips_available == 2
        assert len(result.debug.slots) == 3

    def test_first_slot_has_no_cosine_distance(self):
        clips = [
            (self._make_clip("c1", [1.0, 0.0]), self._make_source("s1")),
        ]
        analysis = AudioAnalysis(
            onset_times=[1.0, 2.0],
            onset_strengths=[0.5, 0.5],
            duration_seconds=3.0,
        )
        result = generate_staccato_sequence(clips, analysis, strategy="onsets")
        assert result.debug.slots[0].cosine_distance is None

    def test_subsequent_slots_have_cosine_distance(self):
        clips = [
            (self._make_clip("c1", [1.0, 0.0]), self._make_source("s1")),
            (self._make_clip("c2", [0.0, 1.0]), self._make_source("s1")),
        ]
        analysis = AudioAnalysis(
            onset_times=[0.5, 1.5],
            onset_strengths=[0.3, 0.9],
            duration_seconds=2.0,
        )
        result = generate_staccato_sequence(clips, analysis, strategy="onsets")
        # Second slot should have a cosine distance value
        assert result.debug.slots[1].cosine_distance is not None
        assert result.debug.slots[1].cosine_distance >= 0.0

    def test_debug_slot_fields(self):
        clips = [
            (self._make_clip("c1", [1.0, 0.0]), self._make_source("s1", file_path="test.mp4")),
        ]
        analysis = AudioAnalysis(
            onset_times=[1.0],
            onset_strengths=[0.7],
            duration_seconds=2.0,
        )
        result = generate_staccato_sequence(clips, analysis, strategy="onsets")
        slot = result.debug.slots[0]
        assert slot.slot_index == 0
        assert slot.start_time == 1.0
        assert slot.onset_strength == 0.7
        assert slot.clip_id == "c1"
        assert slot.clip_name == "c1"
        assert slot.source_filename == "test.mp4"
        assert slot.target_distance == 0.7

    def test_empty_result_has_no_debug_slots(self):
        result = generate_staccato_sequence([], AudioAnalysis())
        assert isinstance(result, StaccatoResult)
        assert len(result) == 0
