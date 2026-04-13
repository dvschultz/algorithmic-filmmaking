"""Unit tests for the Free Association core algorithm module."""

import json
from unittest.mock import Mock, patch

import numpy as np
import pytest

from core.remix.free_association import (
    DEFAULT_SHORTLIST_SIZE,
    _parse_proposal_response,
    build_id_mapping,
    format_clip_digest,
    format_clip_full_metadata,
    propose_next_clip,
    shortlist_candidates,
)
from models.clip import Clip, Source


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_UNSET = object()


def _make_clip(
    clip_id: str = "c-test",
    description=_UNSET,
    shot_type=_UNSET,
    dominant_colors=_UNSET,
    average_brightness=_UNSET,
    person_count=_UNSET,
    object_labels=_UNSET,
    embedding=None,
    **extra,
) -> Clip:
    kwargs = dict(
        id=clip_id,
        source_id="src-1",
        start_frame=0,
        end_frame=30,
        description="A person walking" if description is _UNSET else description,
        shot_type="close-up" if shot_type is _UNSET else shot_type,
        dominant_colors=[(220, 150, 100)] if dominant_colors is _UNSET else dominant_colors,
        average_brightness=0.7 if average_brightness is _UNSET else average_brightness,
        person_count=1 if person_count is _UNSET else person_count,
        object_labels=["person"] if object_labels is _UNSET else object_labels,
        embedding=embedding,
    )
    kwargs.update(extra)
    return Clip(**kwargs)


def _source(source_id: str = "src-1") -> Source:
    return Source(
        id=source_id,
        file_path="/test/video.mp4",
        duration_seconds=60.0,
        fps=30.0,
        width=1920,
        height=1080,
    )


# ---------------------------------------------------------------------------
# format_clip_digest
# ---------------------------------------------------------------------------


class TestFormatClipDigest:
    def test_fully_populated_clip(self):
        clip = _make_clip(
            shot_type="CU",
            dominant_colors=[(220, 150, 100)],  # warm
            average_brightness=0.8,
            person_count=2,
            object_labels=["person", "chair"],
        )
        digest = format_clip_digest(clip)
        assert "CU" in digest
        assert "warm" in digest
        assert "bright" in digest
        assert "2 people" in digest
        assert "person" in digest
        assert " | " in digest

    def test_missing_most_metadata_falls_back_to_description(self):
        clip = _make_clip(
            shot_type=None,
            dominant_colors=None,
            average_brightness=None,
            person_count=None,
            object_labels=None,
            description="A person walking down a sunlit street",
        )
        digest = format_clip_digest(clip)
        assert digest == "A person walking down a sunlit street"

    def test_no_metadata_at_all_returns_placeholder(self):
        clip = _make_clip(
            description=None,
            shot_type=None,
            dominant_colors=None,
            average_brightness=None,
            person_count=None,
            object_labels=None,
        )
        clip.name = ""
        digest = format_clip_digest(clip)
        assert digest == "no metadata available"

    def test_falls_back_to_name_when_no_description(self):
        clip = _make_clip(
            description=None,
            shot_type=None,
            dominant_colors=None,
            average_brightness=None,
            person_count=None,
            object_labels=None,
        )
        clip.name = "My clip"
        assert format_clip_digest(clip) == "My clip"

    def test_single_person_grammar(self):
        clip = _make_clip(person_count=1)
        assert "1 person" in format_clip_digest(clip)
        assert "1 people" not in format_clip_digest(clip)


# ---------------------------------------------------------------------------
# format_clip_full_metadata
# ---------------------------------------------------------------------------


class TestFormatClipFullMetadata:
    def test_includes_description_and_structured_fields(self):
        clip = _make_clip(description="A close-up of hands")
        text = format_clip_full_metadata(clip)
        assert "Description: A close-up of hands" in text
        assert "Shot type: close-up" in text
        assert "People:" in text

    def test_handles_empty_clip(self):
        clip = _make_clip(
            description=None,
            shot_type=None,
            dominant_colors=None,
            average_brightness=None,
            person_count=None,
            object_labels=None,
        )
        text = format_clip_full_metadata(clip)
        assert text == "(no metadata)"


# ---------------------------------------------------------------------------
# shortlist_candidates
# ---------------------------------------------------------------------------


class TestShortlistCandidates:
    def test_returns_top_k_by_similarity(self):
        current = _make_clip("current", embedding=_normalize([1.0, 0.0]))
        # Pool: first 3 are highly similar, last 2 are dissimilar
        pool_clips = [
            (_make_clip(f"c{i}", embedding=_normalize([1.0, 0.1 * i])), _source())
            for i in range(5)
        ]
        result = shortlist_candidates(current, pool_clips, k=2)
        assert len(result) == 2
        # Most similar first
        assert result[0][0].id == "c0"

    def test_missing_embeddings_falls_back_to_random_sample(self):
        current = _make_clip("current", embedding=None)
        pool_clips = [(_make_clip(f"c{i}", embedding=None), _source()) for i in range(20)]
        result = shortlist_candidates(current, pool_clips, k=5)
        assert len(result) == 5

    def test_fewer_clips_than_k_returns_all(self):
        current = _make_clip("current", embedding=_normalize([1.0, 0.0]))
        pool_clips = [
            (_make_clip(f"c{i}", embedding=_normalize([1.0, 0.1])), _source())
            for i in range(3)
        ]
        result = shortlist_candidates(current, pool_clips, k=10)
        assert len(result) == 3

    def test_mixed_embeddings_pads_with_non_embedding_clips(self):
        current = _make_clip("current", embedding=_normalize([1.0, 0.0]))
        with_emb = [
            (_make_clip(f"e{i}", embedding=_normalize([1.0, 0.0])), _source())
            for i in range(2)
        ]
        without_emb = [
            (_make_clip(f"n{i}", embedding=None), _source()) for i in range(5)
        ]
        result = shortlist_candidates(current, with_emb + without_emb, k=4)
        assert len(result) == 4
        # First two should be the embedding-equipped clips
        assert all(c.id.startswith("e") for c, _ in result[:2])


def _normalize(vec: list[float]) -> list[float]:
    arr = np.array(vec, dtype=np.float32)
    return (arr / np.linalg.norm(arr)).tolist()


# ---------------------------------------------------------------------------
# build_id_mapping
# ---------------------------------------------------------------------------


class TestBuildIdMapping:
    def test_bidirectional_mapping(self):
        candidates = [
            (_make_clip("uuid-1"), _source()),
            (_make_clip("uuid-2"), _source()),
            (_make_clip("uuid-3"), _source()),
        ]
        short_to_full, full_to_short = build_id_mapping(candidates)
        assert short_to_full["c1"] == "uuid-1"
        assert short_to_full["c2"] == "uuid-2"
        assert short_to_full["c3"] == "uuid-3"
        assert full_to_short["uuid-1"] == "c1"

    def test_round_trip(self):
        candidates = [(_make_clip(f"uuid-{i}"), _source()) for i in range(5)]
        short_to_full, full_to_short = build_id_mapping(candidates)
        for clip, _ in candidates:
            assert short_to_full[full_to_short[clip.id]] == clip.id


# ---------------------------------------------------------------------------
# propose_next_clip
# ---------------------------------------------------------------------------


def _mock_response(content: str | None) -> Mock:
    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message.content = content
    return response


def _mock_settings() -> Mock:
    settings = Mock()
    settings.exquisite_corpus_model = "gemini-2.5-flash"
    settings.exquisite_corpus_temperature = 0.7
    return settings


class TestProposeNextClip:
    def test_happy_path_returns_clip_id_and_rationale(self):
        candidates = [("c1", "CU | warm | bright"), ("c2", "wide | cool | dark")]
        response = _mock_response(
            json.dumps({"clip_id": "c2", "rationale": "Contrast makes the cut sharp."})
        )

        with patch("core.settings.load_settings", return_value=_mock_settings()), \
             patch("core.settings.get_llm_api_key", return_value="key"), \
             patch("litellm.completion", return_value=response):
            clip_id, rationale = propose_next_clip(
                current_clip_metadata="Description: a close-up",
                candidate_digests=candidates,
                recent_rationales=[],
                rejected_short_ids=[],
            )

        assert clip_id == "c2"
        assert rationale == "Contrast makes the cut sharp."

    def test_none_content_raises_valueerror(self):
        """Regression guard for the documented LLM None bug pattern."""
        candidates = [("c1", "CU"), ("c2", "wide")]
        response = _mock_response(None)

        with patch("core.settings.load_settings", return_value=_mock_settings()), \
             patch("core.settings.get_llm_api_key", return_value="key"), \
             patch("litellm.completion", return_value=response), \
             pytest.raises(ValueError, match="no content"):
            propose_next_clip(
                current_clip_metadata="x",
                candidate_digests=candidates,
                recent_rationales=[],
                rejected_short_ids=[],
            )

    def test_empty_string_content_raises_valueerror(self):
        candidates = [("c1", "CU")]
        response = _mock_response("   ")

        with patch("core.settings.load_settings", return_value=_mock_settings()), \
             patch("core.settings.get_llm_api_key", return_value="key"), \
             patch("litellm.completion", return_value=response), \
             pytest.raises(ValueError, match="no content"):
            propose_next_clip(
                current_clip_metadata="x",
                candidate_digests=candidates,
                recent_rationales=[],
                rejected_short_ids=[],
            )

    def test_malformed_json_raises_valueerror(self):
        candidates = [("c1", "CU")]
        response = _mock_response("This is not JSON at all")

        with patch("core.settings.load_settings", return_value=_mock_settings()), \
             patch("core.settings.get_llm_api_key", return_value="key"), \
             patch("litellm.completion", return_value=response), \
             pytest.raises(ValueError, match="No JSON object"):
            propose_next_clip(
                current_clip_metadata="x",
                candidate_digests=candidates,
                recent_rationales=[],
                rejected_short_ids=[],
            )

    def test_hallucinated_clip_id_raises_valueerror(self):
        candidates = [("c1", "CU"), ("c2", "wide")]
        response = _mock_response(
            json.dumps({"clip_id": "c99", "rationale": "Great match"})
        )

        with patch("core.settings.load_settings", return_value=_mock_settings()), \
             patch("core.settings.get_llm_api_key", return_value="key"), \
             patch("litellm.completion", return_value=response), \
             pytest.raises(ValueError, match="not in candidate set"):
            propose_next_clip(
                current_clip_metadata="x",
                candidate_digests=candidates,
                recent_rationales=[],
                rejected_short_ids=[],
            )

    def test_rejected_clip_id_raises_valueerror(self):
        candidates = [("c1", "CU"), ("c2", "wide")]
        response = _mock_response(
            json.dumps({"clip_id": "c1", "rationale": "Ignoring rejection"})
        )

        with patch("core.settings.load_settings", return_value=_mock_settings()), \
             patch("core.settings.get_llm_api_key", return_value="key"), \
             patch("litellm.completion", return_value=response), \
             pytest.raises(ValueError, match="rejected set"):
            propose_next_clip(
                current_clip_metadata="x",
                candidate_digests=candidates,
                recent_rationales=[],
                rejected_short_ids=["c1"],
            )

    def test_handles_markdown_fenced_json(self):
        candidates = [("c1", "CU")]
        response = _mock_response(
            "```json\n"
            + json.dumps({"clip_id": "c1", "rationale": "Good fit"})
            + "\n```"
        )

        with patch("core.settings.load_settings", return_value=_mock_settings()), \
             patch("core.settings.get_llm_api_key", return_value="key"), \
             patch("litellm.completion", return_value=response):
            clip_id, _ = propose_next_clip(
                current_clip_metadata="x",
                candidate_digests=candidates,
                recent_rationales=[],
                rejected_short_ids=[],
            )
        assert clip_id == "c1"


# ---------------------------------------------------------------------------
# _parse_proposal_response — unit tests for the parser helper
# ---------------------------------------------------------------------------


class TestParseProposalResponse:
    def test_plain_json(self):
        clip_id, rationale = _parse_proposal_response(
            '{"clip_id": "c2", "rationale": "Nice transition"}'
        )
        assert clip_id == "c2"
        assert rationale == "Nice transition"

    def test_missing_clip_id_raises(self):
        with pytest.raises(ValueError, match="'clip_id'"):
            _parse_proposal_response('{"rationale": "only rationale"}')

    def test_missing_rationale_raises(self):
        with pytest.raises(ValueError, match="'rationale'"):
            _parse_proposal_response('{"clip_id": "c1"}')

    def test_json_array_instead_of_object_raises(self):
        with pytest.raises(ValueError):
            _parse_proposal_response('["c1", "reason"]')
