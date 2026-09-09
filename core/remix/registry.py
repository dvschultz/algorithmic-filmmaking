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
    Prepared,
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
    "Prepared",
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
    from core.remix.arrange import ARRANGE_DEFINITIONS
    from core.remix.chromatics import ChromaticsDefinition
    from core.remix.gaze import EyesWithoutAFaceDefinition
    from core.remix.match_cut import MatchCutDefinition
    from core.remix.reference_match import ReferenceGuidedDefinition
    from core.remix.rose_hobart import RoseHobartDefinition
    from core.remix.shuffle import ShuffleDefinition
    from core.remix.similarity_chain import SimilarityChainDefinition

    registry.register(ShuffleDefinition())
    registry.register(ChromaticsDefinition())
    for definition in ARRANGE_DEFINITIONS:
        registry.register(definition())
    registry.register(SimilarityChainDefinition())
    registry.register(MatchCutDefinition())
    registry.register(EyesWithoutAFaceDefinition())
    registry.register(ReferenceGuidedDefinition())
    registry.register(RoseHobartDefinition())


_register_builtin()
