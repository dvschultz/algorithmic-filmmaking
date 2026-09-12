"""Function-local ``from <project module> import <name>`` statements must resolve.

Lazy imports inside function bodies are invisible to linters and only fail when
the enclosing function runs, which for GUI code often means only inside the
frozen app (the macOS release smoke caught ``ui/main_window.py`` importing
``_GITHUB_OWNER`` from ``core.update_checker`` after a ruff cleanup dropped that
re-export). This test resolves every such import against the real module.
"""

from __future__ import annotations

import ast
import importlib
import pkgutil
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCAN_ROOTS = ("ui", "core", "models", "cli", "scene_ripper_mcp")
PROJECT_PACKAGES = ("ui", "core", "models", "cli", "scene_ripper_mcp")

# Modules whose import needs optional native runtimes; only their *names* are
# checked when the module imports, otherwise the site is skipped, not failed.
_OPTIONAL_MODULE_PREFIXES = ("core.analysis.", "core.transcription", "core.remix.")


def _function_local_imports() -> list[tuple[str, int, str, str]]:
    """Return (file, line, module, name) for every function-local project import."""
    sites: list[tuple[str, int, str, str]] = []
    for root in SCAN_ROOTS:
        for path in sorted((PROJECT_ROOT / root).rglob("*.py")):
            if "tests" in path.parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for func in ast.walk(tree):
                if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                for node in ast.walk(func):
                    if not isinstance(node, ast.ImportFrom) or node.level or not node.module:
                        continue
                    if node.module.split(".")[0] not in PROJECT_PACKAGES:
                        continue
                    for alias in node.names:
                        if alias.name == "*":
                            continue
                        rel = path.relative_to(PROJECT_ROOT).as_posix()
                        sites.append((rel, node.lineno, node.module, alias.name))
    return sites


_SITES = _function_local_imports()
_BY_MODULE: dict[str, list[tuple[str, int, str]]] = {}
for _file, _line, _module, _name in _SITES:
    _BY_MODULE.setdefault(_module, []).append((_file, _line, _name))


def _is_submodule(module_name: str, attr: str) -> bool:
    module = importlib.import_module(module_name)
    if not hasattr(module, "__path__"):
        return False
    return any(info.name == attr for info in pkgutil.iter_modules(module.__path__))


def test_scan_found_lazy_imports():
    assert len(_SITES) > 50, "scan unexpectedly found almost no function-local imports"
    assert any(
        file == "ui/main_window.py" and module.startswith("core.update_")
        for file, _line, module, _name in _SITES
    )


@pytest.mark.parametrize("module_name", sorted(_BY_MODULE))
def test_function_local_import_names_resolve(module_name):
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        if module_name.startswith(_OPTIONAL_MODULE_PREFIXES):
            pytest.skip(f"{module_name} needs an optional runtime: {exc}")
        raise
    missing = [
        f"{file}:{line} imports {name!r}"
        for file, line, name in _BY_MODULE[module_name]
        if not hasattr(module, name) and not _is_submodule(module_name, name)
    ]
    assert not missing, f"names missing from {module_name}:\n  " + "\n  ".join(missing)
