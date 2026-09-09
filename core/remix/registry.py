"""The application's algorithm registry with built-in definitions registered.

Import this module to look algorithms up; import :mod:`core.remix.engine` to
define one. Keeping registration here avoids import cycles between definition
modules and the engine.
"""

from __future__ import annotations

from core.remix.engine import (
    AlgorithmDefinition,
    AlgorithmRegistry,
    ClipInput,
    GenerationRun,
    ParameterSpec,
    ProposedEntry,
    SequenceProposal,
    legacy_seed,
    normalize_parameters,
    parameters_document,
    realize,
    recipe_inputs,
    resolve_seed,
    run_algorithm,
)

__all__ = [
    "AlgorithmDefinition",
    "AlgorithmRegistry",
    "ClipInput",
    "GenerationRun",
    "ParameterSpec",
    "ProposedEntry",
    "SequenceProposal",
    "legacy_seed",
    "normalize_parameters",
    "parameters_document",
    "realize",
    "recipe_inputs",
    "registry",
    "resolve_seed",
    "run_algorithm",
]

registry = AlgorithmRegistry()


def _register_builtin() -> None:
    from core.remix.chromatics import ChromaticsDefinition
    from core.remix.shuffle import ShuffleDefinition

    registry.register(ShuffleDefinition())
    registry.register(ChromaticsDefinition())


_register_builtin()
