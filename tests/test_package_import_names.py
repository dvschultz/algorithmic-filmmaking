"""Tests for _PACKAGE_IMPORT_NAMES mapping and is_package_available lookup.

These tests target core.dependency_manager in isolation. Since core.__init__
eagerly imports heavy deps (cv2, scenedetect) and uses 3.10+ syntax in
transitive imports, we bypass it entirely with importlib to load only the
module under test.
"""

import importlib
import sys
from unittest.mock import MagicMock, patch


def _load_dependency_manager():
    """Import core.dependency_manager without triggering core.__init__."""
    # Ensure core package is registered but without running __init__
    if "core" not in sys.modules:
        import types
        core_pkg = types.ModuleType("core")
        core_pkg.__path__ = [str(__import__("pathlib").Path(__file__).resolve().parent.parent / "core")]
        core_pkg.__package__ = "core"
        sys.modules["core"] = core_pkg

    # Stub core.paths and core.binary_resolver (transitive deps of dependency_manager)
    stubs = [
        "core.paths", "core.binary_resolver",
    ]
    originals = {}
    for mod_name in stubs:
        if mod_name in sys.modules:
            originals[mod_name] = sys.modules[mod_name]
        else:
            sys.modules[mod_name] = MagicMock()

    try:
        if "core.dependency_manager" in sys.modules:
            return sys.modules["core.dependency_manager"]
        loader = importlib.util.find_spec("core.dependency_manager")
        if loader is None:
            raise ImportError("Cannot find core.dependency_manager")
        mod = importlib.util.module_from_spec(loader)
        sys.modules["core.dependency_manager"] = mod
        loader.loader.exec_module(mod)
        return mod
    finally:
        # Restore originals if any were replaced
        for mod_name, orig in originals.items():
            sys.modules[mod_name] = orig


dm = _load_dependency_manager()


def test_mapping_contains_protobuf():
    """protobuf must map to google.protobuf."""
    assert dm._PACKAGE_IMPORT_NAMES["protobuf"] == "google.protobuf"


def test_mapping_contains_common_mismatches():
    """Verify other well-known pip-name-to-import-name mappings exist."""
    assert dm._PACKAGE_IMPORT_NAMES["pillow"] == "PIL"
    assert dm._PACKAGE_IMPORT_NAMES["pyyaml"] == "yaml"
    assert dm._PACKAGE_IMPORT_NAMES["opencv-python"] == "cv2"
    assert dm._PACKAGE_IMPORT_NAMES["scikit-learn"] == "sklearn"


@patch.object(dm, "_ensure_managed_packages_importable")
@patch("importlib.util.find_spec")
def test_is_package_available_uses_mapped_name(mock_find_spec, _mock_ensure):
    """is_package_available('protobuf') should call find_spec('google.protobuf')."""
    mock_find_spec.return_value = MagicMock()  # non-None → available

    result = dm.is_package_available("protobuf")

    assert result is True
    mock_find_spec.assert_called_once_with("google.protobuf")


@patch.object(dm, "_ensure_managed_packages_importable")
@patch("importlib.util.find_spec")
def test_is_package_available_unmapped_passes_through(mock_find_spec, _mock_ensure):
    """A package not in the mapping should be passed to find_spec unchanged."""
    mock_find_spec.return_value = MagicMock()

    result = dm.is_package_available("torch")

    assert result is True
    mock_find_spec.assert_called_once_with("torch")


@patch.object(dm, "_ensure_managed_packages_importable")
@patch("importlib.util.find_spec")
def test_is_package_available_returns_false_when_not_found(mock_find_spec, _mock_ensure):
    """find_spec returning None means the package is unavailable."""
    mock_find_spec.return_value = None

    result = dm.is_package_available("nonexistent")

    assert result is False
