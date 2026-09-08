"""Tests for custom visual query evaluation."""

import pytest
from core.analysis.custom_query import _parse_yes_no_response, _build_query_prompt


class TestParseYesNoResponse:
    """Test VLM response parsing for yes/no + confidence."""

    def test_yes_with_percentage_newline(self):
        match, conf = _parse_yes_no_response("YES\n85%")
        assert match is True
        assert conf == 0.85

    def test_no_with_percentage_newline(self):
        match, conf = _parse_yes_no_response("NO\n15%")
        assert match is False
        assert conf == 0.15

    def test_yes_comma_percentage(self):
        match, conf = _parse_yes_no_response("Yes, 92%")
        assert match is True
        assert conf == 0.92

    def test_no_parenthetical_confidence(self):
        match, conf = _parse_yes_no_response("No (confidence: 15%)")
        assert match is False
        assert conf == 0.15

    def test_bare_yes(self):
        match, conf = _parse_yes_no_response("yes")
        assert match is True
        assert conf == 0.9  # High but not absolute when no explicit percentage

    def test_bare_no(self):
        match, conf = _parse_yes_no_response("no")
        assert match is False
        assert conf == 0.1  # Low but not zero when no explicit percentage

    def test_yes_with_dash_confidence(self):
        match, conf = _parse_yes_no_response("YES - I am 90% confident")
        assert match is True
        assert conf == 0.90

    def test_no_with_explanation(self):
        match, conf = _parse_yes_no_response("No, I don't see a blue flower. 10%")
        assert match is False
        assert conf == 0.10

    def test_true_as_yes(self):
        match, conf = _parse_yes_no_response("True\n75%")
        assert match is True
        assert conf == 0.75

    def test_false_as_no(self):
        match, conf = _parse_yes_no_response("False\n5%")
        assert match is False
        assert conf == 0.05

    def test_invalid_confidence_is_not_clamped(self):
        with pytest.raises(ValueError):
            _parse_yes_no_response("YES\n150%")

    def test_zero_confidence(self):
        match, conf = _parse_yes_no_response("NO\n0%")
        assert match is False
        assert conf == 0.0

    def test_ambiguous_yes_before_no(self):
        """When both yes and no appear, use whichever comes first."""
        match, conf = _parse_yes_no_response("Yes, it contains a dog. It is not a cat.")
        assert match is True

    def test_ambiguous_no_before_yes(self):
        with pytest.raises(ValueError):
            _parse_yes_no_response(
                "I would say no, but yesterday I might have said yes"
            )

    def test_malformed_response_is_not_a_negative_match(self):
        with pytest.raises(ValueError):
            _parse_yes_no_response("I'm not sure about that")

    def test_truly_unparseable_response(self):
        with pytest.raises(ValueError):
            _parse_yes_no_response("42")

    @pytest.mark.parametrize(
        "response",
        [
            "",
            "unknown",
            "yesterday",
            "not visible",
            "YES\n-5%",
            "NO\nNaN%",
            "YES\n1e999%",
        ],
    )
    def test_invalid_responses_raise(self, response):
        with pytest.raises(ValueError):
            _parse_yes_no_response(response)

    def test_decimal_confidence(self):
        assert _parse_yes_no_response("YES\n82.5%") == (True, 0.825)

    @pytest.mark.parametrize("response", ["YES 1,000%", "YES 1/50%", "YES --5%"])
    def test_malformed_number_is_not_parsed_as_a_valid_suffix(self, response):
        with pytest.raises(ValueError):
            _parse_yes_no_response(response)

    def test_whitespace_handling(self):
        match, conf = _parse_yes_no_response("  YES  \n  88%  ")
        assert match is True
        assert conf == 0.88

    def test_multiline_explanation(self):
        response = "YES\n95%\nThe image clearly shows a blue flower in the foreground."
        match, conf = _parse_yes_no_response(response)
        assert match is True
        assert conf == 0.95


class TestBuildQueryPrompt:
    """Test prompt construction."""

    def test_prompt_includes_query(self):
        prompt = _build_query_prompt("blue flower")
        assert "blue flower" in prompt
        assert "YES" in prompt
        assert "NO" in prompt

    def test_prompt_asks_for_confidence(self):
        prompt = _build_query_prompt("person wearing a hat")
        assert "%" in prompt


def test_local_query_reports_actual_fallback_model(tmp_path, monkeypatch):
    from core.analysis.custom_query import evaluate_custom_query_local

    monkeypatch.setattr("core.analysis.description.is_mlx_vlm_available", lambda: False)
    monkeypatch.setattr(
        "core.analysis.description.describe_frame_local", lambda *a, **kw: "NO\n10%"
    )
    assert evaluate_custom_query_local(
        tmp_path / "image.jpg", "person", model_name="mlx-community/Qwen3-VL"
    ) == (False, 0.1, "vikhyatk/moondream2")
