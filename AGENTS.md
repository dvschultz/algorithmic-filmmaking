# Repository Guidelines

## Project Structure & Module Organization
`main.py` launches the PySide6 GUI. Core logic lives in `core/`, UI code in `ui/`, models in `models/`, CLI commands in `cli/commands/`, and MCP server code in `scene_ripper_mcp/`.  
Tests are in `tests/` (plus `scene_ripper_mcp/tests/`), media fixtures in `test_clips/`, Linux packaging in `packaging/linux/`, and technical notes in `docs/`.

## Build, Test, and Development Commands
- `python -m pip install -r requirements.txt`: install runtime dependencies.
- `python -m pip install -r requirements-core.txt`: install the dependency set that frozen desktop builds must bundle.
- `python main.py`: launch the desktop application.
- `python -m cli.main --help`: inspect CLI commands.
- `python -m cli.main detect <video> --threshold 3.0`: run scene detection from CLI.
- `python -m pytest tests/ -v`: run the main test suite.
- `python -m pytest tests/test_project.py -v`: run a focused test module.
- `python -m mypy cli core models ui scene_ripper_mcp`: static type checks.
- `ruff check .` and `ruff format .`: lint/format (if Ruff is installed in your environment).

## Coding Style & Naming Conventions
Target Python 3.11+ with 4-space indentation. Use `snake_case` for modules/functions, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants. Add type hints for new/changed public APIs. Keep reusable logic in `core/` and UI-only behavior in `ui/`.

## Architecture & Safety Patterns
- Treat `core/project.py` as the source of truth: use methods (`add_source()`, `add_clips()`) instead of direct list edits.
- Run heavy operations in worker threads (`ui/workers/*`) to keep the UI responsive.
- For FFmpeg/subprocess calls, pass argument arrays, never shell-interpolated strings.
- Use paths from `core/settings.py` (`download_dir`, `project_dir`, `cache_dir`) instead of hardcoded directories.
- For new form controls, reuse sizing constants from `ui/theme.py` (`UISizes`) for consistent UI.
- Treat `requirements-core.txt` as the frozen-app contract. If a dependency is required for the packaged desktop app to launch or for a bundled feature to work, it must be represented in the PyInstaller packaging path as well, not left to incidental import discovery.
- For frozen bundle changes, keep `packaging/build_support.py` as the source of truth for audited package collection rules and avoid ad hoc hidden-import fixes in only one platform spec.
- Do not exclude stdlib or Qt modules from PyInstaller specs unless you have verified they are unused in the current runtime path. Direct imports in the UI and startup path take precedence over bundle-size optimization.
- When adding runtime dependencies with dynamic imports, package data, or transitive compiled dependencies, prefer PyInstaller collection helpers that gather the full package payload (`collect_all` or explicit data/binary collection) rather than only `collect_submodules`.

## Testing Guidelines
Pytest is configured in `pyproject.toml` with `tests/` as the default path and `test_*.py` naming. Add/update tests for behavior changes, especially project save/load, CLI flows, and analysis pipelines. For bug fixes, follow a prove-it loop: add a failing test, fix, then confirm it passes. Mock network/API dependencies where possible. No formal coverage threshold is enforced.
- For packaging fixes, add regression tests that cover import-time safety for the affected module and update `tests/test_build_support.py` when the frozen dependency contract changes.
- Treat frozen-app startup as a separately testable surface. Windows release changes should preserve the startup smoke test in `.github/workflows/build-windows.yml`, and equivalent packaging regressions should be verified before shipping a new tag.

## Release & Packaging
- Tag pushes (`v*`) drive the GitHub release builds in `.github/workflows/build-windows.yml` and `.github/workflows/build-macos.yml`.
- Windows packaging uses `packaging/windows/scene_ripper.spec` plus Inno Setup; macOS packaging uses `packaging/macos/scene_ripper.spec`, signing, and notarization.
- Before changing release packaging, check `docs/releases.md` and keep workflow verification steps aligned with the actual runtime risks. Existence checks alone are not sufficient for frozen desktop apps when import-time failures are possible.

## Commit & Pull Request Guidelines
Recent history follows conventional commit style with optional scopes, e.g. `feat(sequence): ...`, `fix(storyteller): ...`, `docs: ...`. Use imperative, focused commit messages.  
PRs should include a short problem/solution summary, linked issues when applicable, test commands run, and screenshots/GIFs for UI changes.

## Security & Configuration Tips
Do not commit secrets. Configure API keys via Settings (stored in system keyring) or environment variables. Ensure required system tools (notably `ffmpeg`) are installed before running analysis or export workflows.
