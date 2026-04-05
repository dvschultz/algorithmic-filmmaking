"""Tests for ClipBrowser new filter controls (gaze, object, description, brightness)."""

import sys
import pytest
from pathlib import Path
from typing import Optional

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
    gaze_category: Optional[str] = None,
    object_labels: Optional[list[str]] = None,
    detected_objects: Optional[list[dict]] = None,
    description: Optional[str] = None,
    average_brightness: Optional[float] = None,
) -> Clip:
    return Clip(
        id=clip_id,
        source_id=source_id,
        start_frame=start_frame,
        end_frame=end_frame,
        gaze_category=gaze_category,
        object_labels=object_labels,
        detected_objects=detected_objects,
        description=description,
        average_brightness=average_brightness,
    )


class TestGazeFilter:
    """Tests for the gaze direction filter."""

    def setup_method(self):
        self.browser = ClipBrowser()
        self.source = _make_source()

    def test_all_gaze_shows_all_clips(self):
        """All clips visible when gaze filter is 'All Gaze'."""
        c1 = _make_clip("c1", gaze_category="at_camera")
        c2 = _make_clip("c2", gaze_category="looking_left")
        c3 = _make_clip("c3", gaze_category=None)
        self.browser.add_clip(c1, self.source)
        self.browser.add_clip(c2, self.source)
        self.browser.add_clip(c3, self.source)

        assert self.browser._gaze_filter == "All Gaze"
        assert self.browser.get_visible_clip_count() == 3

    def test_filter_at_camera(self):
        """Only clips with gaze_category='at_camera' visible."""
        c1 = _make_clip("c1", gaze_category="at_camera")
        c2 = _make_clip("c2", gaze_category="looking_left")
        c3 = _make_clip("c3", gaze_category=None)
        self.browser.add_clip(c1, self.source)
        self.browser.add_clip(c2, self.source)
        self.browser.add_clip(c3, self.source)

        self.browser._gaze_filter = "At Camera"
        count = sum(1 for t in self.browser.thumbnails if self.browser._matches_filter(t))
        assert count == 1

    def test_no_gaze_category_excluded(self):
        """Clips without gaze_category are excluded by any gaze filter."""
        c1 = _make_clip("c1", gaze_category=None)
        self.browser.add_clip(c1, self.source)

        self.browser._gaze_filter = "Looking Left"
        count = sum(1 for t in self.browser.thumbnails if self.browser._matches_filter(t))
        assert count == 0

    def test_gaze_combo_populated(self):
        """Gaze combo box has correct options."""
        items = [self.browser.gaze_combo.itemText(i) for i in range(self.browser.gaze_combo.count())]
        assert items[0] == "All Gaze"
        assert "At Camera" in items
        assert "Looking Left" in items
        assert "Looking Right" in items
        assert "Looking Up" in items
        assert "Looking Down" in items


class TestObjectSearchFilter:
    """Tests for the object search filter."""

    def setup_method(self):
        self.browser = ClipBrowser()
        self.source = _make_source()

    def test_empty_search_shows_all(self):
        """Empty object search shows all clips."""
        c1 = _make_clip("c1", object_labels=["person", "car"])
        c2 = _make_clip("c2", object_labels=None)
        self.browser.add_clip(c1, self.source)
        self.browser.add_clip(c2, self.source)

        assert self.browser._object_search == ""
        assert self.browser.get_visible_clip_count() == 2

    def test_search_object_labels(self):
        """Search matches against object_labels."""
        c1 = _make_clip("c1", object_labels=["person", "car"])
        c2 = _make_clip("c2", object_labels=["dog", "tree"])
        self.browser.add_clip(c1, self.source)
        self.browser.add_clip(c2, self.source)

        self.browser._object_search = "person"
        count = sum(1 for t in self.browser.thumbnails if self.browser._matches_filter(t))
        assert count == 1

    def test_search_detected_objects(self):
        """Search matches against detected_objects labels."""
        c1 = _make_clip(
            "c1",
            detected_objects=[{"label": "person", "confidence": 0.9, "bbox": [0, 0, 1, 1]}],
        )
        c2 = _make_clip(
            "c2",
            detected_objects=[{"label": "cat", "confidence": 0.8, "bbox": [0, 0, 1, 1]}],
        )
        self.browser.add_clip(c1, self.source)
        self.browser.add_clip(c2, self.source)

        self.browser._object_search = "person"
        count = sum(1 for t in self.browser.thumbnails if self.browser._matches_filter(t))
        assert count == 1

    def test_search_case_insensitive(self):
        """Object search is case-insensitive."""
        c1 = _make_clip("c1", object_labels=["Person", "Car"])
        self.browser.add_clip(c1, self.source)

        self.browser._object_search = "person"
        count = sum(1 for t in self.browser.thumbnails if self.browser._matches_filter(t))
        assert count == 1

    def test_search_substring_match(self):
        """Object search matches substrings."""
        c1 = _make_clip("c1", object_labels=["racecar"])
        self.browser.add_clip(c1, self.source)

        self.browser._object_search = "car"
        count = sum(1 for t in self.browser.thumbnails if self.browser._matches_filter(t))
        assert count == 1

    def test_no_labels_excluded(self):
        """Clips with no labels are excluded when searching."""
        c1 = _make_clip("c1", object_labels=None, detected_objects=None)
        self.browser.add_clip(c1, self.source)

        self.browser._object_search = "person"
        count = sum(1 for t in self.browser.thumbnails if self.browser._matches_filter(t))
        assert count == 0


class TestDescriptionSearchFilter:
    """Tests for the description search filter."""

    def setup_method(self):
        self.browser = ClipBrowser()
        self.source = _make_source()

    def test_empty_search_shows_all(self):
        """Empty description search shows all clips."""
        c1 = _make_clip("c1", description="A sunset over the ocean")
        c2 = _make_clip("c2", description=None)
        self.browser.add_clip(c1, self.source)
        self.browser.add_clip(c2, self.source)

        assert self.browser._description_search == ""
        assert self.browser.get_visible_clip_count() == 2

    def test_search_description_text(self):
        """Search matches against description text."""
        c1 = _make_clip("c1", description="A sunset over the ocean")
        c2 = _make_clip("c2", description="A person walking in the city")
        self.browser.add_clip(c1, self.source)
        self.browser.add_clip(c2, self.source)

        self.browser._description_search = "sunset"
        count = sum(1 for t in self.browser.thumbnails if self.browser._matches_filter(t))
        assert count == 1

    def test_search_case_insensitive(self):
        """Description search is case-insensitive."""
        c1 = _make_clip("c1", description="A Beautiful Sunset")
        self.browser.add_clip(c1, self.source)

        self.browser._description_search = "beautiful sunset"
        count = sum(1 for t in self.browser.thumbnails if self.browser._matches_filter(t))
        assert count == 1

    def test_no_description_excluded(self):
        """Clips without description are excluded when searching."""
        c1 = _make_clip("c1", description=None)
        self.browser.add_clip(c1, self.source)

        self.browser._description_search = "sunset"
        count = sum(1 for t in self.browser.thumbnails if self.browser._matches_filter(t))
        assert count == 0


class TestBrightnessFilter:
    """Tests for the brightness range filter."""

    def setup_method(self):
        self.browser = ClipBrowser()
        self.source = _make_source()

    def test_no_filter_shows_all(self):
        """No brightness filter shows all clips."""
        c1 = _make_clip("c1", average_brightness=0.2)
        c2 = _make_clip("c2", average_brightness=0.8)
        c3 = _make_clip("c3", average_brightness=None)
        self.browser.add_clip(c1, self.source)
        self.browser.add_clip(c2, self.source)
        self.browser.add_clip(c3, self.source)

        assert self.browser._min_brightness is None
        assert self.browser._max_brightness is None
        assert self.browser.get_visible_clip_count() == 3

    def test_min_brightness_filter(self):
        """Min brightness filters out dark clips."""
        c1 = _make_clip("c1", average_brightness=0.2)
        c2 = _make_clip("c2", average_brightness=0.8)
        self.browser.add_clip(c1, self.source)
        self.browser.add_clip(c2, self.source)

        self.browser._min_brightness = 0.5
        count = sum(1 for t in self.browser.thumbnails if self.browser._matches_filter(t))
        assert count == 1

    def test_max_brightness_filter(self):
        """Max brightness filters out bright clips."""
        c1 = _make_clip("c1", average_brightness=0.2)
        c2 = _make_clip("c2", average_brightness=0.8)
        self.browser.add_clip(c1, self.source)
        self.browser.add_clip(c2, self.source)

        self.browser._max_brightness = 0.5
        count = sum(1 for t in self.browser.thumbnails if self.browser._matches_filter(t))
        assert count == 1

    def test_brightness_range_filter(self):
        """Both min and max brightness filter together."""
        c1 = _make_clip("c1", average_brightness=0.2)
        c2 = _make_clip("c2", average_brightness=0.5)
        c3 = _make_clip("c3", average_brightness=0.8)
        self.browser.add_clip(c1, self.source)
        self.browser.add_clip(c2, self.source)
        self.browser.add_clip(c3, self.source)

        self.browser._min_brightness = 0.3
        self.browser._max_brightness = 0.7
        count = sum(1 for t in self.browser.thumbnails if self.browser._matches_filter(t))
        assert count == 1

    def test_no_brightness_data_excluded(self):
        """Clips without brightness are excluded when brightness filter is active."""
        c1 = _make_clip("c1", average_brightness=None)
        self.browser.add_clip(c1, self.source)

        self.browser._min_brightness = 0.0
        count = sum(1 for t in self.browser.thumbnails if self.browser._matches_filter(t))
        assert count == 0


class TestCombinedFilters:
    """Tests for AND logic across multiple new filters."""

    def setup_method(self):
        self.browser = ClipBrowser()
        self.source = _make_source()

    def test_gaze_and_brightness(self):
        """Only clips matching BOTH gaze and brightness filters shown."""
        c1 = _make_clip("c1", gaze_category="at_camera", average_brightness=0.8)
        c2 = _make_clip("c2", gaze_category="at_camera", average_brightness=0.2)
        c3 = _make_clip("c3", gaze_category="looking_left", average_brightness=0.8)
        self.browser.add_clip(c1, self.source)
        self.browser.add_clip(c2, self.source)
        self.browser.add_clip(c3, self.source)

        self.browser._gaze_filter = "At Camera"
        self.browser._min_brightness = 0.5
        count = sum(1 for t in self.browser.thumbnails if self.browser._matches_filter(t))
        assert count == 1

    def test_object_and_description(self):
        """Only clips matching BOTH object and description filters shown."""
        c1 = _make_clip("c1", object_labels=["person"], description="walking in park")
        c2 = _make_clip("c2", object_labels=["person"], description="city at night")
        c3 = _make_clip("c3", object_labels=["dog"], description="walking in park")
        self.browser.add_clip(c1, self.source)
        self.browser.add_clip(c2, self.source)
        self.browser.add_clip(c3, self.source)

        self.browser._object_search = "person"
        self.browser._description_search = "walking"
        count = sum(1 for t in self.browser.thumbnails if self.browser._matches_filter(t))
        assert count == 1


class TestClearAllFilters:
    """Tests for clear_all_filters including new filters."""

    def setup_method(self):
        self.browser = ClipBrowser()
        self.source = _make_source()

    def test_clear_resets_gaze(self):
        """clear_all_filters resets gaze filter."""
        self.browser._gaze_filter = "At Camera"
        self.browser.clear_all_filters()
        assert self.browser._gaze_filter == "All Gaze"
        assert self.browser.gaze_combo.currentText() == "All Gaze"

    def test_clear_resets_object_search(self):
        """clear_all_filters resets object search."""
        self.browser._object_search = "person"
        self.browser.object_search_input.setText("person")
        self.browser.clear_all_filters()
        assert self.browser._object_search == ""
        assert self.browser.object_search_input.text() == ""

    def test_clear_resets_description_search(self):
        """clear_all_filters resets description search."""
        self.browser._description_search = "sunset"
        self.browser.description_search_input.setText("sunset")
        self.browser.clear_all_filters()
        assert self.browser._description_search == ""
        assert self.browser.description_search_input.text() == ""

    def test_clear_resets_brightness(self):
        """clear_all_filters resets brightness filter."""
        self.browser._min_brightness = 0.3
        self.browser._max_brightness = 0.7
        self.browser.clear_all_filters()
        assert self.browser._min_brightness is None
        assert self.browser._max_brightness is None

    def test_clear_shows_all_clips(self):
        """After clearing, all clips are visible."""
        c1 = _make_clip("c1", gaze_category="at_camera", average_brightness=0.8)
        c2 = _make_clip("c2", gaze_category="looking_left", average_brightness=0.2)
        self.browser.add_clip(c1, self.source)
        self.browser.add_clip(c2, self.source)

        self.browser._gaze_filter = "At Camera"
        self.browser._min_brightness = 0.5
        self.browser.clear_all_filters()
        assert self.browser.get_visible_clip_count() == 2


class TestGetActiveFilters:
    """Tests for get_active_filters including new filter keys."""

    def setup_method(self):
        self.browser = ClipBrowser()

    def test_no_active_filters(self):
        """Default state returns None for all new filter keys."""
        filters = self.browser.get_active_filters()
        assert filters["gaze"] is None
        assert filters["object_search"] is None
        assert filters["description_search"] is None
        assert filters["min_brightness"] is None
        assert filters["max_brightness"] is None

    def test_gaze_active(self):
        """Gaze filter reported when active."""
        self.browser._gaze_filter = "At Camera"
        filters = self.browser.get_active_filters()
        assert filters["gaze"] == "At Camera"

    def test_object_search_active(self):
        """Object search reported when active."""
        self.browser._object_search = "person"
        filters = self.browser.get_active_filters()
        assert filters["object_search"] == "person"

    def test_description_search_active(self):
        """Description search reported when active."""
        self.browser._description_search = "sunset"
        filters = self.browser.get_active_filters()
        assert filters["description_search"] == "sunset"

    def test_brightness_active(self):
        """Brightness filter reported when active."""
        self.browser._min_brightness = 0.3
        self.browser._max_brightness = 0.7
        filters = self.browser.get_active_filters()
        assert filters["min_brightness"] == 0.3
        assert filters["max_brightness"] == 0.7


class TestApplyFilters:
    """Tests for apply_filters with new filter keys."""

    def setup_method(self):
        self.browser = ClipBrowser()

    def test_apply_gaze(self):
        """apply_filters sets gaze filter."""
        self.browser.apply_filters({"gaze": "At Camera"})
        assert self.browser._gaze_filter == "At Camera"
        assert self.browser.gaze_combo.currentText() == "At Camera"

    def test_apply_object_search(self):
        """apply_filters sets object search."""
        self.browser.apply_filters({"object_search": "person"})
        assert self.browser._object_search == "person"
        assert self.browser.object_search_input.text() == "person"

    def test_apply_description_search(self):
        """apply_filters sets description search."""
        self.browser.apply_filters({"description_search": "sunset"})
        assert self.browser._description_search == "sunset"
        assert self.browser.description_search_input.text() == "sunset"

    def test_apply_brightness(self):
        """apply_filters sets brightness range."""
        self.browser.apply_filters({"min_brightness": 0.3, "max_brightness": 0.7})
        assert self.browser._min_brightness == 0.3
        assert self.browser._max_brightness == 0.7

    def test_apply_none_resets_gaze(self):
        """apply_filters with None resets gaze."""
        self.browser._gaze_filter = "At Camera"
        self.browser.apply_filters({"gaze": None})
        assert self.browser._gaze_filter == "All Gaze"

    def test_apply_none_resets_object_search(self):
        """apply_filters with None resets object search."""
        self.browser._object_search = "person"
        self.browser.apply_filters({"object_search": None})
        assert self.browser._object_search == ""

    def test_apply_none_resets_description_search(self):
        """apply_filters with None resets description search."""
        self.browser._description_search = "sunset"
        self.browser.apply_filters({"description_search": None})
        assert self.browser._description_search == ""


class TestHasActiveFilters:
    """Tests for has_active_filters including new filters."""

    def setup_method(self):
        self.browser = ClipBrowser()

    def test_default_no_active(self):
        """No filters active by default."""
        assert not self.browser.has_active_filters()

    def test_gaze_active(self):
        """has_active_filters detects gaze filter."""
        self.browser._gaze_filter = "At Camera"
        assert self.browser.has_active_filters()

    def test_object_search_active(self):
        """has_active_filters detects object search."""
        self.browser._object_search = "person"
        assert self.browser.has_active_filters()

    def test_description_search_active(self):
        """has_active_filters detects description search."""
        self.browser._description_search = "sunset"
        assert self.browser.has_active_filters()

    def test_brightness_active(self):
        """has_active_filters detects brightness filter."""
        self.browser._min_brightness = 0.3
        assert self.browser.has_active_filters()
