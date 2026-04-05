"""Tests for ui/algorithm_config.py — category system."""

from ui.algorithm_config import ALGORITHM_CONFIG, CATEGORY_ORDER


def test_category_order_has_six_entries():
    assert CATEGORY_ORDER == ["All", "Arrange", "Find", "Connect", "Audio", "Text"]


def test_every_algorithm_has_categories():
    for key, config in ALGORITHM_CONFIG.items():
        assert "categories" in config, f"{key} missing 'categories' field"
        assert isinstance(config["categories"], list), f"{key} categories not a list"
        assert len(config["categories"]) > 0, f"{key} has empty categories list"


def test_all_category_values_are_valid():
    valid = {c.lower() for c in CATEGORY_ORDER if c != "All"}
    for key, config in ALGORITHM_CONFIG.items():
        for cat in config["categories"]:
            assert cat in valid, f"{key} has invalid category '{cat}'"


def test_multi_tagged_algorithms():
    expected = {
        "volume": ["arrange", "audio"],
        "gaze_sort": ["arrange", "find"],
        "eyes_without_a_face": ["find", "connect"],
        "reference_guided": ["connect", "audio"],
    }
    for key, cats in expected.items():
        assert ALGORITHM_CONFIG[key]["categories"] == cats, (
            f"{key} expected {cats}, got {ALGORITHM_CONFIG[key]['categories']}"
        )


def test_algorithm_count():
    assert len(ALGORITHM_CONFIG) == 19
