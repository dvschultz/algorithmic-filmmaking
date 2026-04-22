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
    # Enum fields are empty sets (= no filter)
    assert fs.shot_type == set()
    assert fs.color_palette == set()
    assert fs.aspect_ratio == set()
    assert fs.gaze_filter == set()
    assert fs.search_query == ""
    assert fs.selected_custom_queries == set()
    assert fs.min_duration is None
    assert fs.max_duration is None
    assert fs.object_search == ""
    assert fs.description_search == ""
    assert fs.min_brightness is None
    assert fs.max_brightness is None
    assert fs.similarity_anchor_id is None
    assert fs.similarity_scores == {}
    assert fs.has_active() is False


def test_enum_assignment_coerces_string_to_set(qapp):
    from core.filter_state import FilterState
    fs = FilterState()
    fs.shot_type = "Close-up"
    assert fs.shot_type == {"Close-up"}

    fs.shot_type = None
    assert fs.shot_type == set()

    fs.shot_type = "All"  # Sentinel clears
    assert fs.shot_type == set()


def test_enum_assignment_accepts_list_and_set(qapp):
    from core.filter_state import FilterState
    fs = FilterState()
    fs.shot_type = ["Close-up", "Extreme CU"]
    assert fs.shot_type == {"Close-up", "Extreme CU"}

    fs.shot_type = {"Wide Shot"}
    assert fs.shot_type == {"Wide Shot"}


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

    fs.shot_type = {"Close-up"}
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
    assert len(emissions) == 1
    assert fs.shot_type == {"Close-up"}
    assert fs.color_palette == {"Warm"}
    assert fs.min_duration == 2.0
    assert fs.max_duration == 5.0


def test_apply_dict_multi_select_list(qapp):
    from core.filter_state import FilterState
    fs = FilterState()
    fs.apply_dict({"shot_type": ["Close-up", "Extreme CU"]})
    assert fs.shot_type == {"Close-up", "Extreme CU"}


def test_apply_dict_backward_compat_string(qapp):
    """String input to an enum field behaves like a single-item set."""
    from core.filter_state import FilterState
    fs_a = FilterState()
    fs_b = FilterState()
    fs_a.apply_dict({"shot_type": "Close-up"})
    fs_b.apply_dict({"shot_type": ["Close-up"]})
    assert fs_a.shot_type == fs_b.shot_type
    assert fs_a.to_dict() == fs_b.to_dict()


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
    assert fs.shot_type == set()


def test_apply_dict_custom_query_coercions(qapp):
    from core.filter_state import FilterState
    fs = FilterState()

    fs.apply_dict({"custom_query": "blue car"})
    assert fs.selected_custom_queries == {"blue car"}

    fs.apply_dict({"custom_query": ["a", "b"]})
    assert fs.selected_custom_queries == {"a", "b"}

    fs.apply_dict({"custom_query": None})
    assert fs.selected_custom_queries == set()

    fs.apply_dict({"custom_query": "All"})
    assert fs.selected_custom_queries == set()

    fs.apply_dict({"custom_query": ("x", " y ", "")})
    assert fs.selected_custom_queries == {"x", "y"}


def test_apply_dict_search_query_lowercased_and_trimmed(qapp):
    from core.filter_state import FilterState
    fs = FilterState()
    fs.apply_dict({"search_query": "  Hello  "})
    assert fs.search_query == "hello"


def test_to_dict_single_value_emitted_as_string(qapp):
    """Backward compat: single-value enums emit a plain string."""
    from core.filter_state import FilterState
    fs = FilterState()
    fs.shot_type = "Close-up"
    out = fs.to_dict()
    assert out["shot_type"] == "Close-up"


def test_to_dict_multi_value_emitted_as_sorted_list(qapp):
    from core.filter_state import FilterState
    fs = FilterState()
    fs.shot_type = {"Extreme CU", "Close-up"}
    out = fs.to_dict()
    assert out["shot_type"] == ["Close-up", "Extreme CU"]


def test_to_dict_empty_enum_emits_none(qapp):
    from core.filter_state import FilterState
    fs = FilterState()
    out = fs.to_dict()
    assert out["shot_type"] is None
    assert out["color_palette"] is None
    assert out["aspect_ratio"] is None
    assert out["gaze"] is None


def test_to_dict_round_trip(qapp):
    from core.filter_state import FilterState
    fs = FilterState()
    fs.apply_dict({
        "shot_type": ["Close-up", "Extreme CU"],
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
    assert out["shot_type"] == ["Close-up", "Extreme CU"]
    assert out["color_palette"] == "Warm"
    assert out["min_duration"] == 2.0
    assert out["aspect_ratio"] == "16:9"
    assert out["gaze"] == "at_camera"
    assert out["object_search"] == "dog"
    assert out["min_brightness"] == 0.2
    assert sorted(out["custom_query"]) == ["blue car", "person"]

    fs2 = FilterState()
    fs2.apply_dict(out)
    assert fs2.to_dict() == out


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
        "shot_type": ["Close-up", "Extreme CU"],
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
    from core.filter_state import FilterState
    fs = FilterState()
    consumer_a: list[set] = []
    consumer_b: list[set] = []
    fs.changed.connect(lambda: consumer_a.append(set(fs.shot_type)))
    fs.changed.connect(lambda: consumer_b.append(set(fs.shot_type)))

    fs.shot_type = "Close-up"
    assert consumer_a == [{"Close-up"}]
    assert consumer_b == [{"Close-up"}]
