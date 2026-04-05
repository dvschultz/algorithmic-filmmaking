"""Tests for ClipBrowser similarity mode (Find Similar / Clear Similarity)."""

import sys
import math
import pytest
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtWidgets import QApplication

from models.clip import Clip, Source
from ui.clip_browser import ClipBrowser, ClipThumbnail

# Ensure a QApplication exists for widget tests
app = QApplication.instance() or QApplication(sys.argv)


def _make_source(source_id: str = "src1", filename: str = "test.mp4") -> Source:
    return Source(
        id=source_id,
        file_path=Path(f"/tmp/{filename}"),
        fps=30.0,
        width=1920,
        height=1080,
    )


def _make_clip(
    clip_id: str = "clip1",
    source_id: str = "src1",
    start_frame: int = 0,
    end_frame: int = 90,
    embedding: Optional[list[float]] = None,
    gaze_category: Optional[str] = None,
    object_labels: Optional[list[str]] = None,
    description: Optional[str] = None,
    average_brightness: Optional[float] = None,
) -> Clip:
    return Clip(
        id=clip_id,
        source_id=source_id,
        start_frame=start_frame,
        end_frame=end_frame,
        embedding=embedding,
        gaze_category=gaze_category,
        object_labels=object_labels,
        description=description,
        average_brightness=average_brightness,
    )


def _normalized_vector(seed: int, dim: int = 768) -> list[float]:
    """Create a deterministic, L2-normalized embedding vector."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(float)
    vec /= np.linalg.norm(vec)
    return vec.tolist()


def _zero_vector(dim: int = 768) -> list[float]:
    """Create a zero embedding vector."""
    return [0.0] * dim


class TestActivateSimilarity:
    """Tests for _activate_similarity — computing similarity scores."""

    def setup_method(self):
        self.browser = ClipBrowser()
        self.source = _make_source()

    def test_similarity_scores_computed(self):
        """Find Similar on a clip with embeddings computes similarity scores."""
        emb_a = _normalized_vector(1)
        emb_b = _normalized_vector(2)
        emb_c = _normalized_vector(3)

        c1 = _make_clip("c1", embedding=emb_a)
        c2 = _make_clip("c2", embedding=emb_b)
        c3 = _make_clip("c3", embedding=emb_c)
        self.browser.add_clip(c1, self.source)
        self.browser.add_clip(c2, self.source)
        self.browser.add_clip(c3, self.source)

        self.browser._activate_similarity(c1)

        assert self.browser._similarity_anchor_id == "c1"
        assert len(self.browser._similarity_scores) == 3
        # Anchor should have score ~1.0 (self-similarity of a normalized vector)
        assert self.browser._similarity_scores["c1"] == pytest.approx(1.0, abs=0.01)
        # Other scores should be valid floats
        assert isinstance(self.browser._similarity_scores["c2"], float)
        assert isinstance(self.browser._similarity_scores["c3"], float)

    def test_clips_sorted_by_similarity(self):
        """Visible clips are sorted most-similar-first within each source group."""
        # Create an anchor and two clips with known similarity ordering
        anchor_emb = _normalized_vector(42)
        # A very similar vector (same + small perturbation)
        similar_vec = np.array(anchor_emb) * 0.99 + np.random.RandomState(100).randn(768) * 0.01
        similar_vec /= np.linalg.norm(similar_vec)
        similar_emb = similar_vec.tolist()
        # A very different vector
        different_emb = _normalized_vector(999)

        c_anchor = _make_clip("anchor", embedding=anchor_emb)
        c_similar = _make_clip("similar", embedding=similar_emb)
        c_different = _make_clip("different", embedding=different_emb)

        self.browser.add_clip(c_anchor, self.source)
        self.browser.add_clip(c_similar, self.source)
        self.browser.add_clip(c_different, self.source)

        self.browser._activate_similarity(c_anchor)

        # Anchor self-similarity should be highest
        assert self.browser._similarity_scores["anchor"] > self.browser._similarity_scores["similar"]
        # Similar clip should score higher than different clip
        assert self.browser._similarity_scores["similar"] > self.browser._similarity_scores["different"]

    def test_no_embedding_is_noop(self):
        """Find Similar on a clip without embedding is a no-op."""
        c1 = _make_clip("c1", embedding=None)
        self.browser.add_clip(c1, self.source)

        self.browser._activate_similarity(c1)

        assert self.browser._similarity_anchor_id is None
        assert self.browser._similarity_scores == {}

    def test_zero_vector_embedding_is_noop(self):
        """Find Similar on a clip with zero-vector embedding is a no-op."""
        c1 = _make_clip("c1", embedding=_zero_vector())
        self.browser.add_clip(c1, self.source)

        self.browser._activate_similarity(c1)

        assert self.browser._similarity_anchor_id is None
        assert self.browser._similarity_scores == {}

    def test_zero_vector_clips_excluded_from_scores(self):
        """Clips with zero-vector embeddings are excluded from similarity scores."""
        anchor_emb = _normalized_vector(1)
        c1 = _make_clip("c1", embedding=anchor_emb)
        c2 = _make_clip("c2", embedding=_zero_vector())
        c3 = _make_clip("c3", embedding=None)
        c4 = _make_clip("c4", embedding=_normalized_vector(2))
        self.browser.add_clip(c1, self.source)
        self.browser.add_clip(c2, self.source)
        self.browser.add_clip(c3, self.source)
        self.browser.add_clip(c4, self.source)

        self.browser._activate_similarity(c1)

        # Only c1 and c4 have valid embeddings
        assert "c1" in self.browser._similarity_scores
        assert "c4" in self.browser._similarity_scores
        assert "c2" not in self.browser._similarity_scores
        assert "c3" not in self.browser._similarity_scores

    def test_zero_vector_clips_hidden_in_similarity_mode(self):
        """Clips without valid embeddings are hidden when similarity mode is active."""
        anchor_emb = _normalized_vector(1)
        c1 = _make_clip("c1", embedding=anchor_emb)
        c2 = _make_clip("c2", embedding=_zero_vector())
        c3 = _make_clip("c3", embedding=None)
        c4 = _make_clip("c4", embedding=_normalized_vector(2))
        self.browser.add_clip(c1, self.source)
        self.browser.add_clip(c2, self.source)
        self.browser.add_clip(c3, self.source)
        self.browser.add_clip(c4, self.source)

        self.browser._activate_similarity(c1)

        visible_count = self.browser.get_visible_clip_count()
        # Only c1 and c4 should be visible
        assert visible_count == 2

    def test_clear_similarity_btn_shown(self):
        """Clear Similarity button becomes visible after activating similarity."""
        c1 = _make_clip("c1", embedding=_normalized_vector(1))
        self.browser.add_clip(c1, self.source)

        # Button should not be explicitly visible initially
        assert not self.browser._clear_similarity_btn.testAttribute(
            __import__("PySide6.QtCore", fromlist=["Qt"]).Qt.WA_WState_ExplicitShowHide
        ) or not self.browser._clear_similarity_btn.isVisibleTo(self.browser._clear_similarity_btn.parentWidget())

        self.browser._activate_similarity(c1)

        # After activation, the button's visibility flag should be set to true
        # (using isVisibleTo parent to avoid window-level visibility issues in tests)
        assert self.browser._clear_similarity_btn.isVisibleTo(
            self.browser._clear_similarity_btn.parentWidget()
        )


class TestClearSimilarity:
    """Tests for _clear_similarity — exiting similarity mode."""

    def setup_method(self):
        self.browser = ClipBrowser()
        self.source = _make_source()

    def test_clear_resets_state(self):
        """Clear Similarity resets anchor and scores."""
        c1 = _make_clip("c1", embedding=_normalized_vector(1))
        c2 = _make_clip("c2", embedding=_normalized_vector(2))
        self.browser.add_clip(c1, self.source)
        self.browser.add_clip(c2, self.source)

        self.browser._activate_similarity(c1)
        assert self.browser._similarity_anchor_id is not None

        self.browser._clear_similarity()

        assert self.browser._similarity_anchor_id is None
        assert self.browser._similarity_scores == {}
        assert not self.browser._clear_similarity_btn.isVisible()

    def test_clear_restores_all_clips(self):
        """All clips visible again after clearing similarity."""
        c1 = _make_clip("c1", embedding=_normalized_vector(1))
        c2 = _make_clip("c2", embedding=None)  # Would be hidden in similarity mode
        c3 = _make_clip("c3", embedding=_normalized_vector(2))
        self.browser.add_clip(c1, self.source)
        self.browser.add_clip(c2, self.source)
        self.browser.add_clip(c3, self.source)

        self.browser._activate_similarity(c1)
        assert self.browser.get_visible_clip_count() == 2  # c2 hidden

        self.browser._clear_similarity()
        assert self.browser.get_visible_clip_count() == 3  # all visible

    def test_clear_preserves_other_filters(self):
        """Clearing similarity preserves other active filters."""
        c1 = _make_clip("c1", embedding=_normalized_vector(1), gaze_category="at_camera")
        c2 = _make_clip("c2", embedding=_normalized_vector(2), gaze_category="looking_left")
        c3 = _make_clip("c3", embedding=_normalized_vector(3), gaze_category="at_camera")
        self.browser.add_clip(c1, self.source)
        self.browser.add_clip(c2, self.source)
        self.browser.add_clip(c3, self.source)

        # Activate gaze filter
        self.browser._gaze_filter = "At Camera"
        # Activate similarity
        self.browser._activate_similarity(c1)

        # Clear similarity but NOT gaze filter
        self.browser._clear_similarity()

        # Gaze filter should still be active
        assert self.browser._gaze_filter == "At Camera"
        # Only clips with at_camera gaze should be visible (c1, c3)
        visible_count = self.browser.get_visible_clip_count()
        assert visible_count == 2


class TestSimilarityWithFilters:
    """Tests for AND logic between similarity mode and other filters."""

    def setup_method(self):
        self.browser = ClipBrowser()
        self.source = _make_source()

    def test_similarity_and_gaze_filter(self):
        """Only clips matching BOTH similarity AND gaze filter shown."""
        c1 = _make_clip("c1", embedding=_normalized_vector(1), gaze_category="at_camera")
        c2 = _make_clip("c2", embedding=_normalized_vector(2), gaze_category="looking_left")
        c3 = _make_clip("c3", embedding=_normalized_vector(3), gaze_category="at_camera")
        c4 = _make_clip("c4", embedding=None, gaze_category="at_camera")
        self.browser.add_clip(c1, self.source)
        self.browser.add_clip(c2, self.source)
        self.browser.add_clip(c3, self.source)
        self.browser.add_clip(c4, self.source)

        # Activate gaze filter first
        self.browser._gaze_filter = "At Camera"
        # Activate similarity
        self.browser._activate_similarity(c1)

        # c1: has embedding + at_camera -> visible
        # c2: has embedding but looking_left -> hidden by gaze filter
        # c3: has embedding + at_camera -> visible
        # c4: no embedding -> hidden by similarity filter
        visible_count = self.browser.get_visible_clip_count()
        assert visible_count == 2

    def test_similarity_and_object_search(self):
        """Only clips matching BOTH similarity AND object search shown."""
        c1 = _make_clip("c1", embedding=_normalized_vector(1), object_labels=["person"])
        c2 = _make_clip("c2", embedding=_normalized_vector(2), object_labels=["car"])
        c3 = _make_clip("c3", embedding=_normalized_vector(3), object_labels=["person", "dog"])
        self.browser.add_clip(c1, self.source)
        self.browser.add_clip(c2, self.source)
        self.browser.add_clip(c3, self.source)

        self.browser._object_search = "person"
        self.browser._activate_similarity(c1)

        # c1: has embedding + "person" -> visible
        # c2: has embedding but no "person" -> hidden by object filter
        # c3: has embedding + "person" -> visible
        visible_count = self.browser.get_visible_clip_count()
        assert visible_count == 2


class TestClearAllFiltersIncludesSimilarity:
    """Tests that clear_all_filters also clears similarity mode."""

    def setup_method(self):
        self.browser = ClipBrowser()
        self.source = _make_source()

    def test_clear_all_resets_similarity(self):
        """clear_all_filters resets similarity mode."""
        c1 = _make_clip("c1", embedding=_normalized_vector(1))
        c2 = _make_clip("c2", embedding=_normalized_vector(2))
        self.browser.add_clip(c1, self.source)
        self.browser.add_clip(c2, self.source)

        self.browser._activate_similarity(c1)
        assert self.browser._similarity_anchor_id is not None

        self.browser.clear_all_filters()

        assert self.browser._similarity_anchor_id is None
        assert self.browser._similarity_scores == {}
        assert not self.browser._clear_similarity_btn.isVisible()


class TestGetActiveFilters:
    """Tests that get_active_filters includes similarity state."""

    def setup_method(self):
        self.browser = ClipBrowser()
        self.source = _make_source()

    def test_no_similarity_returns_none(self):
        """get_active_filters returns None for similarity_anchor when inactive."""
        filters = self.browser.get_active_filters()
        assert filters["similarity_anchor"] is None

    def test_active_similarity_returns_anchor_id(self):
        """get_active_filters returns anchor clip ID when similarity is active."""
        c1 = _make_clip("c1", embedding=_normalized_vector(1))
        self.browser.add_clip(c1, self.source)
        self.browser._activate_similarity(c1)

        filters = self.browser.get_active_filters()
        assert filters["similarity_anchor"] == "c1"

    def test_has_active_filters_includes_similarity(self):
        """has_active_filters returns True when similarity is active."""
        c1 = _make_clip("c1", embedding=_normalized_vector(1))
        self.browser.add_clip(c1, self.source)

        assert not self.browser.has_active_filters()

        self.browser._activate_similarity(c1)
        assert self.browser.has_active_filters()


class TestClipThumbnailFindSimilarSignal:
    """Tests for the find_similar_requested signal on ClipThumbnail."""

    def test_signal_exists(self):
        """ClipThumbnail has find_similar_requested signal."""
        source = _make_source()
        clip = _make_clip("c1")
        thumb = ClipThumbnail(clip, source)
        # Verify signal is connectable
        received = []
        thumb.find_similar_requested.connect(lambda c: received.append(c))
        thumb.find_similar_requested.emit(clip)
        assert len(received) == 1
        assert received[0] is clip

    def test_context_menu_has_find_similar(self):
        """Context menu includes 'Find Similar' action."""
        source = _make_source()
        clip = _make_clip("c1")
        thumb = ClipThumbnail(clip, source)
        menu = thumb._build_context_menu()
        action_texts = [a.text() for a in menu.actions()]
        assert "Find Similar" in action_texts


class TestMultiSourceSimilarity:
    """Tests for similarity mode with clips from multiple sources."""

    def setup_method(self):
        self.browser = ClipBrowser()
        self.source1 = _make_source("src1", "video1.mp4")
        self.source2 = _make_source("src2", "video2.mp4")

    def test_cross_source_similarity(self):
        """Similarity scores computed across multiple sources."""
        c1 = _make_clip("c1", source_id="src1", embedding=_normalized_vector(1))
        c2 = _make_clip("c2", source_id="src1", embedding=_normalized_vector(2))
        c3 = _make_clip("c3", source_id="src2", embedding=_normalized_vector(3))
        c4 = _make_clip("c4", source_id="src2", embedding=_normalized_vector(4))

        self.browser.add_clip(c1, self.source1)
        self.browser.add_clip(c2, self.source1)
        self.browser.add_clip(c3, self.source2)
        self.browser.add_clip(c4, self.source2)

        self.browser._activate_similarity(c1)

        # All 4 clips should have scores
        assert len(self.browser._similarity_scores) == 4
        assert "c1" in self.browser._similarity_scores
        assert "c2" in self.browser._similarity_scores
        assert "c3" in self.browser._similarity_scores
        assert "c4" in self.browser._similarity_scores
