"""Tests for core.filter_state.FilterState."""

import os

import pytest


@pytest.fixture
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_defaults(qapp):
    from core.filter_state import FilterState
    fs = FilterState()
    assert fs.shot_type == "All"
    assert fs.color_palette == "All"
    assert fs.aspect_ratio == "All"
    assert fs.search_query == ""
    assert fs.selected_custom_queries == set()
    assert fs.min_duration is None
    assert fs.max_duration is None
    assert fs.gaze_filter is None
    assert fs.object_search == ""
    assert fs.description_search == ""
    assert fs.min_brightness is None
    assert fs.max_brightness is None
    assert fs.similarity_anchor_id is None
    assert fs.similarity_scores == {}
    assert fs.has_active() is False


def test_assignment_emits_once(qapp):
    from core.filter_state import FilterState
    fs = FilterState()
    emissions: list[int] = []
    fs.changed.connect(lambda: emissions.append(1))

    fs.shot_type = "Close-up"
    assert len(emissions) == 1


def test_assignment_same_value_no_emit(qapp):
    from core.filter_state import FilterState
    fs = FilterState()
    fs.shot_type = "Close-up"
    emissions: list[int] = []
    fs.changed.connect(lambda: emissions.append(1))

    fs.shot_type = "Close-up"
    assert emissions == []


def test_apply_dict_batches_emissions(qapp):
    from core.filter_state import FilterState
    fs = FilterState()
    emissions: list[int] = []
    fs.changed.connect(lambda: emissions.append(1))

    fs.apply_dict({
        "shot_type": "Close-up",
        "color_palette": "Warm",
        "min_duration": 2.0,
        "max_duration": 5.0,
    })
    # 4 fields changed, single emission
    assert len(emissions) == 1
    assert fs.shot_type == "Close-up"
    assert fs.color_palette == "Warm"
    assert fs.min_duration == 2.0
    assert fs.max_duration == 5.0


def test_apply_dict_empty_is_noop(qapp):
    from core.filter_state import FilterState
    fs = FilterState()
    emissions: list[int] = []
    fs.changed.connect(lambda: emissions.append(1))

    fs.apply_dict({})
    assert emissions == []


def test_apply_dict_none_clears_enum_fields(qapp):
    from core.filter_state import FilterState
    fs = FilterState()
    fs.shot_type = "Close-up"
    fs.apply_dict({"shot_type": None})
    assert fs.shot_type == "All"


def test_apply_dict_custom_query_coercions(qapp):
    from core.filter_state import FilterState
    fs = FilterState()

    fs.apply_dict({"custom_query": "blue car"})
    assert fs.selected_custom_queries == {"blue car"}

    fs.apply_dict({"custom_query": ["a", "b"]})
    assert fs.selected_custom_queries == {"a", "b"}

    fs.apply_dict({"custom_query": None})
    assert fs.selected_custom_queries == set()

    # Legacy sentinel strings from the pre-refactor API are treated as "clear"
    fs.apply_dict({"custom_query": "All"})
    assert fs.selected_custom_queries == set()

    fs.apply_dict({"custom_query": ("x", " y ", "")})
    assert fs.selected_custom_queries == {"x", "y"}


def test_apply_dict_search_query_lowercased_and_trimmed(qapp):
    from core.filter_state import FilterState
    fs = FilterState()
    fs.apply_dict({"search_query": "  Hello  "})
    assert fs.search_query == "hello"


def test_to_dict_round_trip(qapp):
    from core.filter_state import FilterState
    fs = FilterState()
    fs.apply_dict({
        "shot_type": "Close-up",
        "color_palette": "Warm",
        "min_duration": 2.0,
        "max_duration": 5.0,
        "aspect_ratio": "16:9",
        "gaze": "at_camera",
        "object_search": "dog",
        "description_search": "sunset",
        "min_brightness": 0.2,
        "max_brightness": 0.9,
        "custom_query": ["blue car", "person"],
    })
    out = fs.to_dict()
    assert out["shot_type"] == "Close-up"
    assert out["color_palette"] == "Warm"
    assert out["min_duration"] == 2.0
    assert out["aspect_ratio"] == "16:9"
    assert out["gaze"] == "at_camera"
    assert out["object_search"] == "dog"
    assert out["min_brightness"] == 0.2
    assert sorted(out["custom_query"]) == ["blue car", "person"]

    # Round-trip: applying to_dict to a fresh state yields identical output
    fs2 = FilterState()
    fs2.apply_dict(out)
    assert fs2.to_dict() == out


def test_to_dict_defaults_are_none_for_inactive(qapp):
    from core.filter_state import FilterState
    fs = FilterState()
    out = fs.to_dict()
    assert out["shot_type"] is None
    assert out["color_palette"] is None
    assert out["aspect_ratio"] is None
    assert out["search_query"] is None
    assert out["custom_query"] is None
    assert out["object_search"] is None
    assert out["description_search"] is None


def test_has_active_reflects_any_field(qapp):
    from core.filter_state import FilterState
    fs = FilterState()
    assert fs.has_active() is False
    fs.gaze_filter = "at_camera"
    assert fs.has_active() is True
    fs.gaze_filter = None
    assert fs.has_active() is False


def test_clear_all_resets_and_emits_once(qapp):
    from core.filter_state import FilterState
    fs = FilterState()
    fs.apply_dict({
        "shot_type": "Close-up",
        "min_brightness": 0.5,
        "custom_query": ["a"],
    })

    emissions: list[int] = []
    fs.changed.connect(lambda: emissions.append(1))

    fs.clear_all()
    assert emissions == [1]
    assert fs.has_active() is False


def test_clear_all_on_empty_is_noop(qapp):
    from core.filter_state import FilterState
    fs = FilterState()
    emissions: list[int] = []
    fs.changed.connect(lambda: emissions.append(1))

    fs.clear_all()
    assert emissions == []


def test_shared_instance_between_consumers(qapp):
    """Two consumers sharing one FilterState both see the same mutations."""
    from core.filter_state import FilterState
    fs = FilterState()
    consumer_a: list[str] = []
    consumer_b: list[str] = []
    fs.changed.connect(lambda: consumer_a.append(fs.shot_type))
    fs.changed.connect(lambda: consumer_b.append(fs.shot_type))

    fs.shot_type = "Close-up"
    assert consumer_a == ["Close-up"]
    assert consumer_b == ["Close-up"]
