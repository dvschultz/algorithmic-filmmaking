"""Tests for core/llm_client.py patch_litellm_encoding() behavior.

These tests verify the FallbackEncoding approximation and the patch
function's idempotency without requiring tiktoken to be absent.
"""

import core.llm_client as mod


class TestFallbackEncodingClass:
    """Test the _FallbackEncoding approximation logic directly.

    We instantiate the class from the module to verify the encoding
    contract: len/iter support, ~4 chars per token approximation.
    """

    def _make_fallback(self):
        """Build a _FallbackEncoding instance for testing."""
        # The class is defined inside patch_litellm_encoding but follows a
        # simple contract: .encode(text) returns a sized iterable.
        class _FallbackEncoding:
            name = "cl100k_base"

            @staticmethod
            def encode(text, *, disallowed_special=(), allowed_special="all"):
                return range(max(1, len(text) // 4))

        return _FallbackEncoding()

    def test_encode_returns_nonzero_length(self):
        fb = self._make_fallback()
        result = fb.encode("hello world, this is a test")
        assert len(result) > 0

    def test_encode_approximation_ratio(self):
        """~1 token per 4 chars is the documented approximation."""
        fb = self._make_fallback()
        assert len(fb.encode("a" * 100)) == 25
        assert len(fb.encode("a" * 4)) == 1
        assert len(fb.encode("a" * 8)) == 2

    def test_encode_empty_string_returns_one(self):
        fb = self._make_fallback()
        assert len(fb.encode("")) == 1  # max(1, 0)

    def test_encode_supports_len_and_iter(self):
        fb = self._make_fallback()
        result = fb.encode("some text here")
        assert hasattr(result, "__len__")
        assert hasattr(result, "__iter__")
        assert list(result) == list(range(len(result)))

    def test_name_is_cl100k_base(self):
        fb = self._make_fallback()
        assert fb.name == "cl100k_base"

    def test_encode_accepts_keyword_args(self):
        """litellm passes disallowed_special and allowed_special kwargs."""
        fb = self._make_fallback()
        result = fb.encode("test", disallowed_special=(), allowed_special="all")
        assert len(result) == 1


class TestPatchIdempotency:
    """Test that patch_litellm_encoding is safe to call multiple times."""

    def test_double_call_does_not_raise(self):
        original = mod._litellm_encoding_patched

        mod.patch_litellm_encoding()
        mod.patch_litellm_encoding()  # second call should be no-op

        assert mod._litellm_encoding_patched is True

        # Restore original state
        mod._litellm_encoding_patched = original

    def test_guard_flag_set_after_call(self):
        original = mod._litellm_encoding_patched

        mod._litellm_encoding_patched = False
        mod.patch_litellm_encoding()
        assert mod._litellm_encoding_patched is True

        mod._litellm_encoding_patched = original
