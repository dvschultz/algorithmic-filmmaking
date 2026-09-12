"""U17: engine, desktop, MCP and ML dependency contracts stay aligned.

requirements-engine.txt is the Qt-free engine contract, requirements-core.txt
the frozen desktop contract (engine + GUI), requirements-optional.txt the
on-demand ML runtimes. The pyproject extras must resolve the same
definitions so `pip install .[mcp]` and the requirement files never drift.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GUI_ONLY = {"PySide6", "python-mpv"}
VENDORED_LITELLM = "./vendor/wheels/litellm-"


def _requirements(name: str) -> dict[str, str]:
    """Map normalized package name -> full requirement line (comments stripped)."""
    entries: dict[str, str] = {}
    for raw in (ROOT / name).read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith(VENDORED_LITELLM):
            version = re.search(r"litellm-([\d.]+)-", line).group(1)
            entries["litellm"] = f"litellm>={version}"
            continue
        entries[_name(line)] = line
    return entries


def _name(spec: str) -> str:
    return re.split(r"[\[<>=!;~ ]", spec, maxsplit=1)[0].strip().lower().replace("_", "-")


def _extras() -> dict[str, dict[str, str]]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    extras = data["project"]["optional-dependencies"]
    return {
        extra: {_name(spec): spec for spec in specs if not spec.startswith("scene-ripper[")}
        for extra, specs in extras.items()
    }


def _self_refs(extra: str) -> set[str]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    return {
        re.search(r"\[(.+)\]", spec).group(1)
        for spec in data["project"]["optional-dependencies"][extra] if spec.startswith("scene-ripper[")
    }


def test_engine_contract_is_the_desktop_contract_minus_the_gui():
    engine = _requirements("requirements-engine.txt")
    core = _requirements("requirements-core.txt")
    assert not (set(engine) & {n.lower() for n in GUI_ONLY})
    assert set(core) - set(engine) == {n.lower() for n in GUI_ONLY}
    for name, spec in engine.items():
        assert core[name] == spec, f"{name}: engine {spec!r} != core {core[name]!r}"


def test_pyproject_extras_resolve_the_requirement_files():
    extras = _extras()
    engine = _requirements("requirements-engine.txt")
    assert extras["engine"] == {name: engine[name] for name in extras["engine"]}
    assert set(extras["engine"]) == set(engine)
    assert _self_refs("desktop") == {"engine"} and _self_refs("mcp") == {"engine"}
    assert set(extras["desktop"]) == {n.lower() for n in GUI_ONLY}
    core = _requirements("requirements-core.txt")
    assert all(core[name] == spec for name, spec in extras["desktop"].items())
    assert set(extras["mcp"]) == {"mcp"}
    optional = _requirements("requirements-optional.txt")
    assert extras["ml"] == optional


def test_engine_has_no_gui_or_ml_runtime():
    engine = _requirements("requirements-engine.txt")
    optional = _requirements("requirements-optional.txt")
    assert not set(engine) & set(optional), set(engine) & set(optional)
    assert not set(engine) & {n.lower() for n in GUI_ONLY}


def _upper_bound(specifier_set):
    """Highest version a specifier set allows, or None when it is open-ended."""
    from packaging.version import Version

    bounds = [
        Version(spec.version)
        for spec in specifier_set
        if spec.operator in ("<", "<=")
    ]
    return max(bounds) if bounds else None


def test_tokenizers_pin_is_not_looser_than_the_installed_transformers_allows():
    """The manifest must not let pip land a tokenizers transformers rejects at import.

    transformers re-checks its tokenizers bound at import time. The manifest
    pinned tokenizers<0.24 while transformers 4.57 requires <=0.23.0, so a
    sequential profile install with --upgrade landed 0.23.2 and every vlm
    worker import then failed with "tokenizers>=0.22.0,<=0.23.0 is required
    ... but found tokenizers==0.23.2". The packaged native-analysis gate caught
    it; this keeps a future widening from reintroducing it.
    """
    import json
    from importlib.metadata import PackageNotFoundError, requires

    import pytest
    from packaging.requirements import Requirement
    from packaging.specifiers import SpecifierSet

    try:
        declared = requires("transformers") or []
    except PackageNotFoundError:
        pytest.skip("transformers is not installed in this environment")

    wanted = None
    for raw in declared:
        requirement = Requirement(raw)
        if requirement.name == "tokenizers" and requirement.marker is None:
            wanted = requirement.specifier
            break
    if wanted is None:
        pytest.skip("this transformers does not constrain tokenizers")

    manifest = json.loads((ROOT / "core" / "package_manifest.json").read_text())
    ours = SpecifierSet(
        Requirement(manifest["packages"]["tokenizers"]["pip_specifier"]).specifier.__str__()
    )

    ours_max, wanted_max = _upper_bound(ours), _upper_bound(wanted)
    assert wanted_max is not None
    assert ours_max is not None, f"tokenizers pin {ours} has no upper bound"
    assert ours_max <= wanted_max, (
        f"manifest allows tokenizers up to {ours_max} but the installed transformers "
        f"requires {wanted}; a --upgrade install can land a version it rejects at import"
    )
