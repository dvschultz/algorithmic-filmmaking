"""Tests for Analyze tab clip ID / browser sync.

Reproduces a bug where remove_orphaned_clips could leave stale IDs in
_clip_ids when the lookup dict (_clips_by_id) was stale, causing the
progress counter to show more clips than were visible in the browser.
"""

import pytest
from pathlib import Path
from unittest.mock import patch

from models.clip import Source, Clip
from tests.conftest import make_test_clip


@pytest.fixture
def qapp():
    """Minimal QApplication for widget tests."""
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def analyze_tab(qapp):
    """Create an AnalyzeTab with test data."""
    from ui.tabs.analyze_tab import AnalyzeTab
    tab = AnalyzeTab()
    return tab


@pytest.fixture
def source_a():
    return Source(
        id="src-a",
        file_path=Path("/test/video_a.mp4"),
        duration_seconds=60.0,
        fps=30.0,
        width=1920,
        height=1080,
    )


@pytest.fixture
def source_b():
    return Source(
        id="src-b",
        file_path=Path("/test/video_b.mp4"),
        duration_seconds=30.0,
        fps=24.0,
        width=1280,
        height=720,
    )


def _make_clips(prefix: str, source_id: str, count: int) -> list[Clip]:
    """Create a list of test clips."""
    return [
        make_test_clip(f"{prefix}-{i}", source_id=source_id,
                       start_frame=i * 30, end_frame=(i + 1) * 30)
        for i in range(count)
    ]


class TestRemoveOrphanedClipSync:
    """Verify _clip_ids stays in sync with browser after orphan removal."""

    def test_orphan_removal_discards_unresolvable_ids(
        self, analyze_tab, source_a, source_b
    ):
        """Bug: remove_orphaned_clips left stale IDs in _clip_ids when
        _clips_by_id was stale. This caused get_clips() to later resolve
        those IDs against an updated dict, returning more clips than
        were visible in the browser. The progress bar would then show
        e.g. '4/8 clips' when only 4 were visible.

        Scenario:
        1. Tab has clips from source A and source B
        2. Source A is re-detected (old clips replaced with new ones)
        3. remove_orphaned_clips is called with stale _clips_by_id
        4. Source A old clips are orphaned and removed
        5. Source B clips survive orphan check but can't resolve in stale dict
        6. BUG: source B IDs stayed in _clip_ids without being in browser
        """
        clips_a = _make_clips("a", "src-a", 4)
        clips_b = _make_clips("b", "src-b", 4)

        # Set up lookup dicts with all clips
        all_clips = {c.id: c for c in clips_a + clips_b}
        sources = {"src-a": source_a, "src-b": source_b}
        analyze_tab.set_lookups(all_clips, sources)

        # Add all 8 clips to the tab
        analyze_tab.add_clips([c.id for c in clips_a + clips_b])
        assert len(analyze_tab._clip_ids) == 8

        # Simulate source A re-detection: new clips replace old
        new_clips_a = _make_clips("new-a", "src-a", 4)
        # New valid set: new source A clips + existing source B clips
        valid_ids = {c.id for c in new_clips_a + clips_b}

        # Simulate stale _clips_by_id (hasn't been updated via set_lookups yet)
        # The tab's _clips_by_id still references old dict (has old-A + B)
        # But valid_ids includes new-A + B
        # Old-A clips are orphaned. B clips pass the orphan check.
        # However, in the rebuild, B clips resolve fine from the old dict.
        # This specific test covers the case where _clips_by_id is stale
        # and SOME surviving IDs can't be resolved.

        # Make _clips_by_id stale by only keeping source A (old) clips
        stale_dict = {c.id: c for c in clips_a}  # No source B clips!
        analyze_tab._clips_by_id = stale_dict

        # Remove orphans — source A old clips are orphaned
        removed = analyze_tab.remove_orphaned_clips(valid_ids)
        assert removed == 4  # 4 old source-A clips removed

        # BUG FIX: source B IDs should also be discarded because they
        # can't be resolved in the stale _clips_by_id dict
        browser_count = len(analyze_tab.clip_browser.thumbnails)
        clip_ids_count = len(analyze_tab._clip_ids)

        # Key assertion: _clip_ids must match browser count
        assert clip_ids_count == browser_count, (
            f"_clip_ids ({clip_ids_count}) and browser ({browser_count}) "
            f"are out of sync — stale IDs not discarded"
        )

    def test_get_clips_matches_clip_ids_after_orphan_removal(
        self, analyze_tab, source_a
    ):
        """get_clips() should return exactly len(_clip_ids) clips."""
        clips = _make_clips("c", "src-a", 4)
        clips_by_id = {c.id: c for c in clips}
        sources = {"src-a": source_a}
        analyze_tab.set_lookups(clips_by_id, sources)
        analyze_tab.add_clips([c.id for c in clips])

        result = analyze_tab.get_clips()
        assert len(result) == len(analyze_tab._clip_ids) == 4

    def test_normal_orphan_removal_works(self, analyze_tab, source_a):
        """Standard orphan removal (no stale dict) works correctly."""
        old_clips = _make_clips("old", "src-a", 4)
        new_clips = _make_clips("new", "src-a", 4)

        clips_by_id = {c.id: c for c in old_clips}
        sources = {"src-a": source_a}
        analyze_tab.set_lookups(clips_by_id, sources)
        analyze_tab.add_clips([c.id for c in old_clips])
        assert len(analyze_tab._clip_ids) == 4

        # After re-detection, only new clips are valid
        valid_ids = {c.id for c in new_clips}
        removed = analyze_tab.remove_orphaned_clips(valid_ids)
        assert removed == 4
        assert len(analyze_tab._clip_ids) == 0
        assert len(analyze_tab.clip_browser.thumbnails) == 0
