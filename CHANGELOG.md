# Changelog

Notable changes to Scene Ripper. Versions correspond to git tags (`vX.Y.Z`);
the tag is the single source of truth and CI asserts it matches
`pyproject.toml` on release builds.

## Unreleased

### Added
- LICENSE (MIT) and this changelog.
- Informational CI: lint (ruff) + scoped type check (mypy) in `quality.yml`,
  macOS unit-test job in `macos-ci.yml`. Gates flip to required per the rule
  in `docs/releases.md`.
- Direct characterization tests for the core analysis math
  (`tests/test_analysis_math.py`).
- Runtime smoke targets: analyze-clip, sequence-build, render-short, mcp-stdio.
- Runtime-acquisition inventory (`docs/runtime-acquisition-2026-07-01.md`).

### Changed
- Repo-wide lint cleanup to a zero-error ruff baseline.
- `ctc-forced-aligner` pip spec pinned to a commit (was git HEAD).
- Release workflows assert the tag matches `pyproject.toml` on tag builds.
- README corrections: 23 sequencing modes, 6-tab workflow, libmpv listed as a
  from-source system dependency, on-demand component disclosure.

## 0.4.10 and earlier

See the [GitHub releases](https://github.com/dvschultz/algorithmic-filmmaking/releases)
for release history prior to this changelog.
