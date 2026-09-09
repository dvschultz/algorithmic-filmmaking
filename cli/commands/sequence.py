"""Sequence generation and variation commands (registry-backed)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import click

from cli.utils.errors import ExitCode, exit_with
from cli.utils.output import output_result, output_table
from cli.utils.project_writer import own_project


@click.group()
def sequence() -> None:
    """Generate, inspect, and vary sequences through the algorithm registry.

    \b
    Commands:
        algorithms    List registry algorithms and their parameter schemas
        list          List a project's sequences with recipe ids
        generate      Run an algorithm on project clips and store its recipe
        recipe        Show a sequence's stored recipe
        reconstruct   Replay a recipe without running the algorithm
        regenerate    Run a recipe again as a new variation
        duplicate     Copy a sequence and its recipe
        activate      Make a sequence active
    """


def _json(ctx: click.Context) -> bool:
    return bool((ctx.obj or {}).get("json", False))


def _parse_parameters(raw: Optional[str]) -> Optional[dict[str, Any]]:
    if raw is None:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        exit_with(ExitCode.VALIDATION_ERROR, f"--parameters must be a JSON object: {exc}")
    if not isinstance(value, dict):
        exit_with(ExitCode.VALIDATION_ERROR, "--parameters must be a JSON object")
    return value


def _load(project_file: Path):
    from core.project import Project, ProjectLoadError

    try:
        return Project.load(project_file, missing_source_callback=lambda path, sid: None)
    except ProjectLoadError as exc:
        exit_with(ExitCode.GENERAL_ERROR, f"Failed to load project: {exc}")
    except FileNotFoundError:
        exit_with(ExitCode.FILE_NOT_FOUND, f"Project file not found: {project_file}")


def _mutate(ctx: click.Context, project_file: Path, operation) -> None:
    """Load under ownership, apply one spine operation, save when it succeeded."""
    path = own_project(ctx, project_file)
    project = _load(path)
    try:
        result = operation(project)
    except Exception as exc:  # noqa: BLE001 - provider/network errors become results
        result = {"success": False, "error": str(exc)}
    if not result.get("success"):
        if _json(ctx):
            output_result(result, as_json=True)
            exit_with(ExitCode.VALIDATION_ERROR)
        exit_with(ExitCode.VALIDATION_ERROR, str(result.get("error")))
    if not project.save():
        exit_with(ExitCode.GENERAL_ERROR, "Failed to save project")
    output_result(result, as_json=_json(ctx))


@sequence.command("algorithms")
@click.pass_context
def algorithms(ctx: click.Context) -> None:
    """List registry algorithms with versions and parameter schemas."""
    from core.spine.sequences import list_algorithms

    result = list_algorithms()
    if _json(ctx):
        output_result(result, as_json=True)
        return
    rows = []
    for entry in result["algorithms"]:
        params = ", ".join(
            f"{p['name']}={p['default']!r}" + (f" [{'/'.join(map(str, p['choices']))}]" if p.get("choices") else "")
            for p in entry["parameters"]
        )
        rows.append([entry["key"], entry["version"], "yes" if entry["seeded"] else "no", ", ".join(entry["prerequisites"]), params])
    output_table(["Key", "Version", "Seeded", "Prerequisites", "Parameters"], rows)


@sequence.command("list")
@click.argument("project_file", type=click.Path(exists=True, path_type=Path))
@click.pass_context
def list_cmd(ctx: click.Context, project_file: Path) -> None:
    """List sequences with ids, algorithm, clip count and recipe id."""
    from core.spine.sequences import list_sequences

    result = list_sequences(_load(project_file))
    if _json(ctx):
        output_result(result, as_json=True)
        return
    output_table(
        ["ID", "Name", "Active", "Algorithm", "Clips", "Recipe"],
        [[s["id"], s["name"], "*" if s["active"] else "", s["algorithm"] or "", s["clip_count"], s["recipe_id"] or ""]
         for s in result["sequences"]],
    )


@sequence.command("generate")
@click.argument("project_file", type=click.Path(exists=True, path_type=Path))
@click.argument("algorithm")
@click.option("--clip", "clip_ids", multiple=True, help="Clip ID to include, in order (repeatable); default: all enabled clips")
@click.option("--parameters", "raw_parameters", default=None, help='JSON object of algorithm parameters, e.g. \'{"direction": "warm_to_cool"}\'')
@click.option("--seed", type=int, default=None, help="Explicit seed for seeded algorithms (0 is a valid seed)")
@click.option("--name", default=None, help="Sequence name")
@click.pass_context
def generate(
    ctx: click.Context, project_file: Path, algorithm: str, clip_ids: tuple[str, ...],
    raw_parameters: Optional[str], seed: Optional[int], name: Optional[str],
) -> None:
    """Run a registry algorithm and publish a new sequence with its recipe.

    \b
    Examples:
        scene_ripper sequence generate my.sceneripper color --parameters '{"direction": "complementary"}'
        scene_ripper sequence generate my.sceneripper shuffle --seed 7 --clip c1 --clip c2
    """
    from core.spine.sequences import generate_sequence

    parameters = _parse_parameters(raw_parameters)
    if seed is not None and seed < 0:
        exit_with(ExitCode.VALIDATION_ERROR, "--seed must be a non-negative integer")

    _mutate(ctx, project_file, lambda project: generate_sequence(
        project, algorithm, clip_ids=list(clip_ids) or None, parameters=parameters, seed=seed, name=name,
    ))


@sequence.command("recipe")
@click.argument("project_file", type=click.Path(exists=True, path_type=Path))
@click.option("--sequence-id", default=None, help="Sequence ID; default: active sequence")
@click.pass_context
def recipe(ctx: click.Context, project_file: Path, sequence_id: Optional[str]) -> None:
    """Show a sequence's stored recipe and whether it can be reconstructed."""
    from core.spine.sequences import get_sequence_recipe

    _read(ctx, project_file, lambda project: get_sequence_recipe(project, sequence_id))


def _read(ctx: click.Context, project_file: Path, operation) -> None:
    """Run a read-only spine query and print its result (no save)."""
    try:
        result = operation(_load(project_file))
    except Exception as exc:  # noqa: BLE001
        result = {"success": False, "error": str(exc)}
    if not result.get("success"):
        if _json(ctx):
            output_result(result, as_json=True)
            exit_with(ExitCode.VALIDATION_ERROR)
        exit_with(ExitCode.VALIDATION_ERROR, str(result.get("error")))
    output_result(result, as_json=_json(ctx))


@sequence.command("compare")
@click.argument("project_file", type=click.Path(exists=True, path_type=Path))
@click.argument("sequence_a")
@click.argument("sequence_b")
@click.pass_context
def compare(ctx: click.Context, project_file: Path, sequence_a: str, sequence_b: str) -> None:
    """Compare two sequences: counts, durations, seeds and changed recipe parameters."""
    from core.spine.sequences import compare_sequences

    _read(ctx, project_file, lambda project: compare_sequences(project, sequence_a, sequence_b))


@sequence.command("reconstruct")
@click.argument("project_file", type=click.Path(exists=True, path_type=Path))
@click.option("--sequence-id", default=None, help="Sequence ID; default: active sequence")
@click.option("--name", default=None, help="Name for the rebuilt sequence")
@click.pass_context
def reconstruct(ctx: click.Context, project_file: Path, sequence_id: Optional[str], name: Optional[str]) -> None:
    """Rebuild a sequence from its recipe without running the algorithm or providers."""
    from core.spine.sequences import reconstruct_sequence

    _mutate(ctx, project_file, lambda project: reconstruct_sequence(project, sequence_id, name=name))


@sequence.command("regenerate")
@click.argument("project_file", type=click.Path(exists=True, path_type=Path))
@click.option("--sequence-id", default=None, help="Sequence ID; default: active sequence")
@click.option("--parameters", "raw_parameters", default=None, help="JSON object of parameter overrides")
@click.option("--seed", type=int, default=None, help="Explicit seed; default draws a fresh seed")
@click.option("--keep-seed", is_flag=True, help="Reuse the recipe's seed")
@click.option("--name", default=None, help="Name for the variation")
@click.pass_context
def regenerate(
    ctx: click.Context, project_file: Path, sequence_id: Optional[str], raw_parameters: Optional[str],
    seed: Optional[int], keep_seed: bool, name: Optional[str],
) -> None:
    """Run a recipe's algorithm again as a new variation; the original is untouched."""
    from core.spine.sequences import regenerate_sequence

    parameters = _parse_parameters(raw_parameters)
    _mutate(ctx, project_file, lambda project: regenerate_sequence(
        project, sequence_id, parameters=parameters, seed=seed, keep_seed=keep_seed, name=name,
    ))


@sequence.command("duplicate")
@click.argument("project_file", type=click.Path(exists=True, path_type=Path))
@click.option("--sequence-id", default=None, help="Sequence ID; default: active sequence")
@click.option("--name", default=None, help="Name for the copy")
@click.pass_context
def duplicate(ctx: click.Context, project_file: Path, sequence_id: Optional[str], name: Optional[str]) -> None:
    """Copy a sequence's timeline and recipe as a new active sequence."""
    from core.spine.sequences import duplicate_sequence

    _mutate(ctx, project_file, lambda project: duplicate_sequence(project, sequence_id, name=name))


@sequence.command("activate")
@click.argument("project_file", type=click.Path(exists=True, path_type=Path))
@click.argument("sequence_id")
@click.pass_context
def activate(ctx: click.Context, project_file: Path, sequence_id: str) -> None:
    """Make a sequence active."""
    from core.spine.sequences import activate_sequence

    def operation(project):
        result = activate_sequence(project, sequence_id)
        if result.get("success"):
            project.mark_dirty()
        return result

    _mutate(ctx, project_file, operation)
