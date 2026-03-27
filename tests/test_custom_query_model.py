"""Tests for custom_queries field on the Clip model."""

from models.clip import Clip


class TestCustomQueriesSerialization:
    """Test custom_queries field serialization round-trip."""

    def test_to_dict_includes_custom_queries(self):
        """Clip.to_dict() includes custom_queries when present."""
        clip = Clip(
            id="test-1",
            source_id="src-1",
            custom_queries=[
                {"query": "blue flower", "match": True, "confidence": 0.92, "model": "gpt-4o"},
                {"query": "red car", "match": False, "confidence": 0.15, "model": "moondream-2b"},
            ],
        )
        data = clip.to_dict()
        assert "custom_queries" in data
        assert len(data["custom_queries"]) == 2
        assert data["custom_queries"][0]["query"] == "blue flower"
        assert data["custom_queries"][0]["match"] is True
        assert data["custom_queries"][0]["confidence"] == 0.92
        assert data["custom_queries"][1]["match"] is False

    def test_to_dict_omits_none_custom_queries(self):
        """Clip.to_dict() does not include custom_queries when None."""
        clip = Clip(id="test-2", source_id="src-1")
        data = clip.to_dict()
        assert "custom_queries" not in data

    def test_to_dict_omits_empty_custom_queries(self):
        """Clip.to_dict() does not include custom_queries when empty list."""
        clip = Clip(id="test-3", source_id="src-1", custom_queries=[])
        data = clip.to_dict()
        assert "custom_queries" not in data

    def test_from_dict_restores_custom_queries(self):
        """Clip.from_dict() restores custom_queries from saved data."""
        data = {
            "id": "test-4",
            "source_id": "src-1",
            "start_frame": 0,
            "end_frame": 100,
            "custom_queries": [
                {"query": "sunset", "match": True, "confidence": 0.88, "model": "gemini-2.0-flash"},
            ],
        }
        clip = Clip.from_dict(data)
        assert clip.custom_queries is not None
        assert len(clip.custom_queries) == 1
        assert clip.custom_queries[0]["query"] == "sunset"
        assert clip.custom_queries[0]["match"] is True
        assert clip.custom_queries[0]["confidence"] == 0.88
        assert clip.custom_queries[0]["model"] == "gemini-2.0-flash"

    def test_from_dict_backward_compat_no_key(self):
        """Loading a project saved before this feature produces None."""
        data = {
            "id": "test-5",
            "source_id": "src-1",
            "start_frame": 0,
            "end_frame": 100,
        }
        clip = Clip.from_dict(data)
        assert clip.custom_queries is None

    def test_round_trip_fidelity(self):
        """custom_queries survive a full to_dict → from_dict round trip."""
        queries = [
            {"query": "person wearing hat", "match": True, "confidence": 0.75, "model": "gpt-4o"},
            {"query": "dog", "match": False, "confidence": 0.1, "model": "moondream-2b"},
            {"query": "outdoor scene", "match": True, "confidence": 0.99, "model": "claude-sonnet-4-20250514"},
        ]
        original = Clip(
            id="test-6",
            source_id="src-1",
            start_frame=0,
            end_frame=200,
            custom_queries=queries,
        )
        data = original.to_dict()
        restored = Clip.from_dict(data)
        assert restored.custom_queries == queries

    def test_accumulation_semantics(self):
        """Multiple queries can be appended independently."""
        clip = Clip(id="test-7", source_id="src-1")
        assert clip.custom_queries is None

        # First query
        clip.custom_queries = [
            {"query": "blue flower", "match": True, "confidence": 0.9, "model": "gpt-4o"},
        ]
        # Second query appended
        clip.custom_queries.append(
            {"query": "red car", "match": False, "confidence": 0.2, "model": "gpt-4o"},
        )
        assert len(clip.custom_queries) == 2
        assert clip.custom_queries[0]["query"] == "blue flower"
        assert clip.custom_queries[1]["query"] == "red car"

    def test_from_dict_discards_malformed_custom_queries(self):
        """Malformed custom_queries (not a list of dicts) are discarded."""
        data = {
            "id": "test-8",
            "source_id": "src-1",
            "start_frame": 0,
            "end_frame": 100,
            "custom_queries": "not a list",
        }
        clip = Clip.from_dict(data)
        assert clip.custom_queries is None

    def test_from_dict_discards_non_dict_entries(self):
        """custom_queries with non-dict entries are discarded entirely."""
        data = {
            "id": "test-9",
            "source_id": "src-1",
            "start_frame": 0,
            "end_frame": 100,
            "custom_queries": ["not a dict", 42],
        }
        clip = Clip.from_dict(data)
        assert clip.custom_queries is None
