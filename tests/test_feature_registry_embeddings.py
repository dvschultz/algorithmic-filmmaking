"""Tests for the embeddings feature in the feature registry."""

from core.feature_registry import FEATURE_DEPS, check_feature


class TestEmbeddingsFeatureDeps:
    """Tests for the embeddings entry in FEATURE_DEPS."""

    def test_embeddings_in_feature_deps(self):
        assert "embeddings" in FEATURE_DEPS

    def test_embeddings_requires_torch_and_transformers(self):
        deps = FEATURE_DEPS["embeddings"]
        assert "torch" in deps.packages
        assert "transformers" in deps.packages

    def test_embeddings_has_reasonable_size_estimate(self):
        deps = FEATURE_DEPS["embeddings"]
        assert deps.size_estimate_mb > 0

    def test_embeddings_uses_native_install(self):
        deps = FEATURE_DEPS["embeddings"]
        assert deps.native_install is True

    def test_embeddings_has_repair_packages(self):
        deps = FEATURE_DEPS["embeddings"]
        assert len(deps.repair_packages) > 0
        assert "torch" in deps.repair_packages
        assert "transformers" in deps.repair_packages
