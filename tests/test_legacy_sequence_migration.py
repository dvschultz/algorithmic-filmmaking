"""Legacy source and clip-relative ranges must never be guessed when ambiguous."""

from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import json

import pytest

from core.legacy_sequence_time import convert_legacy_entry, migrate_sequence_time
from models.sequence import SequenceClip


@pytest.mark.parametrize(
    "lo,hi,expected,status",
    [(0, 24, (240, 264), "resolved"), (600, 624, (600, 624), "resolved"),
     (264, 288, (264, 288), "unresolved")],
)
def test_classify_only_unambiguous_coordinates(lo, hi, expected, status):
    clip = {"id": "clip", "source_id": "source", "start_frame": 240, "end_frame": 720}
    source = {"id": "source", "fps": 24}
    entry = {"source_clip_id": "clip", "source_id": "source", "in_point": lo, "out_point": hi}
    original = deepcopy(entry)
    result = convert_legacy_entry(entry, clip, source, 30)
    assert entry == original
    assert result["legacy_timing"]["original"] == original
    assert result["legacy_timing"]["status"] == status
    assert (result["in_point"], result["out_point"]) == expected
    restored = SequenceClip.from_dict(result)
    if status == "unresolved":
        with pytest.raises(ValueError, match="Resolve legacy"):
            _ = restored.source_range
    else:
        assert restored.source_range.duration == 1


def test_explicit_resolution_preserves_original_and_is_idempotent():
    clip = {"id": "clip", "source_id": "source", "start_frame": 240, "end_frame": 720}
    source = {"id": "source", "fps": 24}
    entry = {"source_clip_id": "clip", "source_id": "source", "in_point": 264, "out_point": 288}
    result = convert_legacy_entry(entry, clip, source, 30)
    resolved = convert_legacy_entry(result, clip, source, 30, "clip-relative")
    assert (resolved["in_point"], resolved["out_point"]) == (504, 528)
    assert resolved["legacy_timing"]["original"] == entry
    assert convert_legacy_entry(resolved, clip, source, 30, "clip-relative") == resolved


def test_migration_keeps_both_sequence_keys_and_unrecognized_fields():
    entry = {"in_point": 5, "out_point": 10, "future_extension": {"a": [1, 2]}}
    sequence = {"tracks": [{"clips": [entry]}]}
    data = {"sequence": sequence, "sequences": [deepcopy(sequence)]}
    migrate_sequence_time(data)
    for seq in (data["sequence"], data["sequences"][0]):
        migrated = seq["tracks"][0]["clips"][0]
        assert migrated["legacy_timing"]["status"] == "unresolved"
        assert migrated["legacy_timing"]["original"] == entry
        assert migrated["future_extension"] == {"a": [1, 2]}


@pytest.mark.parametrize("version", ["1.0", "1.1", "1.2", "1.3", "1.4"])
def test_all_legacy_fixtures_preserve_original_entries(version):
    from core.project_migrations import migrate_project_data
    data = json.loads((Path(__file__).parent / "fixtures" / "projects" / f"v{version}.json").read_text())
    original = deepcopy(data)
    migrated = migrate_project_data(data)
    assert data == original
    sequence = migrated["sequences"][0]
    old_sequence = original.get("sequence", original.get("sequences", [None])[0])
    for old_track, track in zip(old_sequence["tracks"], sequence["tracks"]):
        for old, entry in zip(old_track["clips"], track["clips"]):
            assert entry["legacy_timing"]["original"] == old
            assert entry["legacy_timing"]["status"] == "resolved"
            assert (entry["in_point"], entry["out_point"]) == (48, 96)
            restored = SequenceClip.from_dict(entry)
            assert restored.legacy_timing["original"] == old


def test_malformed_legacy_coordinates_remain_inspectable_without_losing_raw_values():
    original = {"start_frame": "bad", "in_point": None, "out_point": -1}
    entry = SequenceClip.from_dict(convert_legacy_entry(original, None, None, 30))
    assert entry.duration_frames == 0
    assert entry.start_frame == 0
    assert entry.legacy_timing["original"] == original
    with pytest.raises(ValueError, match="Resolve legacy"):
        _ = entry.source_range


@pytest.mark.parametrize("hold", ["invalid", None, -1, False])
def test_malformed_legacy_still_remains_inspectable(hold):
    original = {"frame_id": "still", "hold_frames": hold, "start_frame": 0}
    entry = SequenceClip.from_dict(convert_legacy_entry(original, None, None, 30))
    assert entry.end_frame() == 0
    assert entry.legacy_timing["status"] == "unresolved"
    assert entry.legacy_timing["original"] == original
    with pytest.raises(ValueError, match="Resolve legacy"):
        _ = entry.source_range


def test_repeated_resolution_preserves_exact_retimed_origin():
    clip = {"id": "clip", "source_id": "source", "start_frame": 240, "end_frame": 720}
    source = {"id": "source", "fps": 24}
    entry = {"source_clip_id": "clip", "source_id": "source", "start_frame": 1,
             "in_point": 0, "out_point": 24}
    resolved = convert_legacy_entry(entry, clip, source, 30)
    resolved["timeline_rate"] = "24"
    repeated = convert_legacy_entry(resolved, clip, source, 24, "clip-relative")
    assert Fraction(repeated["timeline_start"]) == Fraction(1, 30)
    assert repeated["legacy_timing"]["original"] == entry
