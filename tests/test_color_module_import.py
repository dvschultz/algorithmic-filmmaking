"""Regression tests for lazy color-analysis imports."""

import builtins
import importlib.util
import sys
from pathlib import Path


def _load_color_module(name: str):
    project_root = Path(__file__).resolve().parent.parent
    module_path = project_root / "core" / "analysis" / "color.py"
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_color_module_import_does_not_require_sklearn(monkeypatch):
    """UI startup should not fail just because sklearn isn't importable."""
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("sklearn"):
            raise ModuleNotFoundError("No module named 'sklearn'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    module = _load_color_module("scene_ripper_color_lazy_import_test")

    assert module.get_primary_hue([(255, 0, 0)]) == 0.0
