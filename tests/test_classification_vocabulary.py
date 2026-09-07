"""Classification labels belong to the selected model weights."""

from types import SimpleNamespace
from unittest.mock import Mock
from pathlib import Path

import pytest

from core.analysis import classification


@pytest.mark.parametrize("cache_failure", [False, True])
def test_model_uses_weight_vocabulary_instead_of_unversioned_cache(
    tmp_path, monkeypatch, cache_failure
):
    cache = tmp_path / "imagenet_classes.txt"
    cache.write_text("wrong\nlabels\n")
    weights = SimpleNamespace(meta={"categories": ["person", "car"]})
    model = Mock()
    models = SimpleNamespace(
        MobileNet_V3_Small_Weights=SimpleNamespace(IMAGENET1K_V1=weights),
        mobilenet_v3_small=Mock(return_value=model),
    )
    transforms = Mock()
    monkeypatch.setattr(classification, "_model", None)
    monkeypatch.setattr(classification, "_labels", None)
    monkeypatch.setattr(classification, "_preprocess", None)
    monkeypatch.setattr(classification, "_get_model_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(
        classification,
        "ensure_image_classification_runtime_available",
        lambda: (models, transforms),
    )
    if cache_failure:
        monkeypatch.setattr(
            Path, "replace", Mock(side_effect=OSError("read-only cache"))
        )
        monkeypatch.setattr(
            Path, "unlink", Mock(side_effect=OSError("read-only cache"))
        )
    loaded, labels, preprocess = classification._load_model()
    assert labels == ["person", "car"]
    models.mobilenet_v3_small.assert_called_once_with(weights=weights)
    assert loaded is model
    assert preprocess is transforms.Compose.return_value
    if not cache_failure:
        assert classification.load_imagenet_class_list() == labels
