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

    def test_confidence_clamped_to_100(self):
        match, conf = _parse_yes_no_response("YES\n150%")
        assert match is True
        assert conf == 1.0

    def test_zero_confidence(self):
        match, conf = _parse_yes_no_response("NO\n0%")
        assert match is False
        assert conf == 0.0

    def test_ambiguous_yes_before_no(self):
        """When both yes and no appear, use whichever comes first."""
        match, conf = _parse_yes_no_response("Yes, it contains a dog. It is not a cat.")
        assert match is True

    def test_ambiguous_no_before_yes(self):
        """When no appears before yes, treat as no match."""
        match, conf = _parse_yes_no_response("I would say no, but yesterday I might have said yes")
        assert match is False

    def test_malformed_response_returns_false(self):
        # "not" contains "no" so parser treats as negative with default confidence
        match, conf = _parse_yes_no_response("I'm not sure about that")
        assert match is False
        assert conf == 0.1

    def test_truly_unparseable_response(self):
        match, conf = _parse_yes_no_response("42")
        assert match is False
        assert conf == 0.0

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
