"""Tests for Rose Hobart face-filter sequencer.

Tests cover:
- Face comparison logic (compare_faces)
- Embedding averaging (average_embeddings)
- Clip model face_embeddings serialization
- Sensitivity threshold presets
"""

import math
import numpy as np
import pytest

from core.analysis.faces import average_embeddings, compare_faces, SENSITIVITY_PRESETS
from models.clip import Clip


# ── compare_faces tests ──


def _make_embedding(seed: int = 0, dim: int = 512) -> list[float]:
    """Create a deterministic unit-norm embedding."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()


def _make_face(seed: int = 0) -> dict:
    """Create a face dict with a deterministic embedding."""
    return {
        "bbox": [10, 20, 100, 120],
        "embedding": _make_embedding(seed),
        "confidence": 0.95,
    }


def test_compare_faces_matching_above_threshold():
    """Identical embeddings should match with similarity ~1.0."""
    ref = [_make_embedding(42)]
    clip_faces = [_make_face(42)]  # Same seed = same embedding

    is_match, similarity = compare_faces(ref, clip_faces, threshold=0.5)

    assert is_match is True
    assert similarity > 0.99


def test_compare_faces_non_matching_below_threshold():
    """Unrelated embeddings should not match at strict threshold."""
    ref = [_make_embedding(1)]
    clip_faces = [_make_face(999)]  # Different seed = different embedding

    is_match, similarity = compare_faces(ref, clip_faces, threshold=0.5)

    assert is_match is False
    assert similarity < 0.5


def test_compare_faces_empty_reference():
    """Empty reference should return no match."""
    is_match, similarity = compare_faces([], [_make_face(1)], threshold=0.1)
    assert is_match is False
    assert similarity == 0.0


def test_compare_faces_empty_clip_faces():
    """Empty clip faces should return no match."""
    is_match, similarity = compare_faces([_make_embedding(1)], [], threshold=0.1)
    assert is_match is False
    assert similarity == 0.0


def test_compare_faces_multiple_references():
    """Should match against any reference embedding."""
    ref = [_make_embedding(1), _make_embedding(42)]
    clip_faces = [_make_face(42)]  # Matches second reference

    is_match, similarity = compare_faces(ref, clip_faces, threshold=0.5)
    assert is_match is True
    assert similarity > 0.99


def test_compare_faces_multiple_clip_faces():
    """Should find match among multiple clip faces."""
    ref = [_make_embedding(42)]
    clip_faces = [_make_face(1), _make_face(42), _make_face(999)]

    is_match, similarity = compare_faces(ref, clip_faces, threshold=0.5)
    assert is_match is True
    assert similarity > 0.99


# ── average_embeddings tests ──


def test_average_embeddings_single():
    """Single embedding should return itself."""
    emb = _make_embedding(42)
    result = average_embeddings([emb])
    assert result == emb


def test_average_embeddings_multiple():
    """Multiple embeddings should produce a normalized mean."""
    emb1 = _make_embedding(1)
    emb2 = _make_embedding(2)
    result = average_embeddings([emb1, emb2])

    # Result should be L2-normalized
    norm = math.sqrt(sum(x * x for x in result))
    assert abs(norm - 1.0) < 1e-5

    # Result should be between the two inputs (dot product > 0 with both)
    dot1 = sum(a * b for a, b in zip(result, emb1))
    dot2 = sum(a * b for a, b in zip(result, emb2))
    assert dot1 > 0
    assert dot2 > 0


def test_average_embeddings_empty():
    """Empty list should return empty."""
    assert average_embeddings([]) == []


# ── Sensitivity threshold tests ──


def test_sensitivity_presets_exist():
    """All three presets should be defined."""
    assert "strict" in SENSITIVITY_PRESETS
    assert "balanced" in SENSITIVITY_PRESETS
    assert "loose" in SENSITIVITY_PRESETS


def test_sensitivity_ordering():
    """Strict > Balanced > Loose thresholds."""
    assert SENSITIVITY_PRESETS["strict"] > SENSITIVITY_PRESETS["balanced"]
    assert SENSITIVITY_PRESETS["balanced"] > SENSITIVITY_PRESETS["loose"]


def test_sensitivity_thresholds_filter_correctly():
    """Given known similarity scores, each preset filters as expected."""
    ref_emb = np.array(_make_embedding(42), dtype=np.float32)

    # Create an embedding with exact cosine similarity of 0.40
    # by mixing the reference with an orthogonal vector
    noise_emb = np.array(_make_embedding(999), dtype=np.float32)
    # Make noise orthogonal to reference via Gram-Schmidt
    noise_emb = noise_emb - np.dot(noise_emb, ref_emb) * ref_emb
    noise_emb = noise_emb / np.linalg.norm(noise_emb)
    # cos(theta) = 0.40 => sin(theta) = sqrt(1-0.16) ~= 0.9165
    target_sim = 0.40
    perturbed = target_sim * ref_emb + math.sqrt(1 - target_sim ** 2) * noise_emb
    perturbed = perturbed / np.linalg.norm(perturbed)

    clip_faces = [{
        "bbox": [0, 0, 50, 50],
        "embedding": perturbed.tolist(),
        "confidence": 0.9,
    }]

    _, sim = compare_faces([ref_emb.tolist()], clip_faces, threshold=0.0)
    assert abs(sim - target_sim) < 0.02, f"Expected ~{target_sim}, got {sim}"

    # Loose (0.25) should match
    is_match_loose, _ = compare_faces([ref_emb.tolist()], clip_faces, SENSITIVITY_PRESETS["loose"])
    assert is_match_loose is True

    # Balanced (0.35) should match
    is_match_balanced, _ = compare_faces([ref_emb.tolist()], clip_faces, SENSITIVITY_PRESETS["balanced"])
    assert is_match_balanced is True

    # Strict (0.50) should NOT match
    is_match_strict, _ = compare_faces([ref_emb.tolist()], clip_faces, SENSITIVITY_PRESETS["strict"])
    assert is_match_strict is False


# ── Clip model serialization tests ──


def test_clip_to_dict_with_face_embeddings():
    """face_embeddings should serialize when present."""
    clip = Clip(
        id="test-1",
        source_id="src-1",
        start_frame=0,
        end_frame=100,
        face_embeddings=[{
            "bbox": [10, 20, 50, 60],
            "embedding": [0.1] * 512,
            "confidence": 0.95,
            "frame_number": 15,
        }],
    )
    data = clip.to_dict()
    assert "face_embeddings" in data
    assert len(data["face_embeddings"]) == 1
    assert len(data["face_embeddings"][0]["embedding"]) == 512


def test_clip_to_dict_without_face_embeddings():
    """face_embeddings should not appear in dict when None."""
    clip = Clip(id="test-1", source_id="src-1", start_frame=0, end_frame=100)
    data = clip.to_dict()
    assert "face_embeddings" not in data


def test_clip_from_dict_roundtrip():
    """face_embeddings should survive to_dict/from_dict roundtrip."""
    original_faces = [{
        "bbox": [10, 20, 50, 60],
        "embedding": [0.1] * 512,
        "confidence": 0.95,
        "frame_number": 15,
    }]
    clip = Clip(
        id="test-1",
        source_id="src-1",
        start_frame=0,
        end_frame=100,
        face_embeddings=original_faces,
    )
    data = clip.to_dict()
    restored = Clip.from_dict(data)
    assert restored.face_embeddings is not None
    assert len(restored.face_embeddings) == 1
    assert len(restored.face_embeddings[0]["embedding"]) == 512
    assert restored.face_embeddings[0]["confidence"] == 0.95


def test_clip_from_dict_missing_face_embeddings():
    """Old project data without face_embeddings should load cleanly."""
    data = {
        "id": "test-1",
        "source_id": "src-1",
        "start_frame": 0,
        "end_frame": 100,
    }
    clip = Clip.from_dict(data)
    assert clip.face_embeddings is None


def test_clip_from_dict_malformed_face_embeddings():
    """Malformed entries should be discarded."""
    data = {
        "id": "test-1",
        "source_id": "src-1",
        "start_frame": 0,
        "end_frame": 100,
        "face_embeddings": [
            # Valid entry
            {
                "bbox": [10, 20, 50, 60],
                "embedding": [0.1] * 512,
                "confidence": 0.95,
            },
            # Invalid: wrong embedding length
            {
                "bbox": [10, 20, 50, 60],
                "embedding": [0.1] * 256,
                "confidence": 0.9,
            },
            # Invalid: missing bbox
            {
                "embedding": [0.1] * 512,
                "confidence": 0.8,
            },
            # Invalid: not a dict
            "bad_entry",
        ],
    }
    clip = Clip.from_dict(data)
    assert clip.face_embeddings is not None
    assert len(clip.face_embeddings) == 1  # Only the valid entry survives
