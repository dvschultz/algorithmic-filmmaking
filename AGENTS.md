<!-- BEGIN COMPOUND CODEX TOOL MAP -->
## Compound Codex Tool Mapping (Claude Compatibility)

This section maps Claude Code plugin tool references to Codex behavior.
Only this block is managed automatically.

Tool mapping:
- Read: use shell reads (cat/sed) or rg
- Write: create files via shell redirection or apply_patch
- Edit/MultiEdit: use apply_patch
- Bash: use shell_command
- Grep: use rg (fallback: grep)
- Glob: use rg --files or find
- LS: use ls via shell_command
- WebFetch/WebSearch: use curl or Context7 for library docs
- AskUserQuestion/Question: ask the user in chat
- Task/Subagent/Parallel: run sequentially in main thread; use multi_tool_use.parallel for tool calls
- TodoWrite/TodoRead: use file-based todos in todos/ with file-todos skill
- Skill: open the referenced SKILL.md and follow it
- ExitPlanMode: ignore
<!-- END COMPOUND CODEX TOOL MAP -->

--- project-doc ---

# Repository Guidelines

## Project Structure & Module Organization
`main.py` launches the PySide6 desktop app and contains important frozen-startup ordering. Core application logic lives in `core/`, UI code in `ui/`, and domain models in `models/`.

The primary wired UI flow is six tabs in `ui/main_window.py`: `Collect`, `Cut`, `Analyze`, `Frames`, `Sequence`, and `Render`. `ui/tabs/generate_tab.py` exists as a stub but is not currently mounted in the main window.

CLI code lives in `cli/` and is exposed as the `scene_ripper` console script. Registered Click command groups live in `cli/commands/` and currently cover project management, scene detection, analysis, transcription, export, and YouTube search/download.

MCP server code lives in `scene_ripper_mcp/` and is exposed as the `scene-ripper-mcp` console script. Tool registrations live under `scene_ripper_mcp/tools/`, schemas under `scene_ripper_mcp/schemas/`, and MCP integration tests under `scene_ripper_mcp/tests/`.

Tests live primarily in `tests/`. Packaging code is split across `packaging/linux/`, `packaging/macos/`, `packaging/windows/`, and `packaging/release/`. Runtime assets and packaging inputs live in `assets/`, `packaging/runtime/`, and `vendor/wheels/`. Helper scripts live in `scripts/`. Long-form technical context lives in `docs/` with active user guides, release notes, plans, brainstorms, research, and solved-issue writeups.

## Build, Test, and Development Commands
- `python -m pip install -r requirements.txt`: install the full source-development dependency set.
- `python -m pip install -r requirements-core.txt`: install the minimal frozen-app dependency contract.
- `python -m pip install -r requirements-optional.txt`: install optional on-demand ML and media features explicitly in a source environment.
- `python main.py`: launch the desktop application from source.
- `scene_ripper --help` or `python -m cli.main --help`: inspect the CLI.
- `scene_ripper detect <video>`: run scene detection from the CLI.
- `scene_ripper analyze describe <project.json>`: run analysis from the CLI.
- `scene_ripper export sequence <project.json> -o ./out.mp4`: export a sequence from the CLI.
- `scene-ripper-mcp --transport stdio`: run the MCP server over stdio.
- `scene-ripper-mcp --transport http --port 8765`: run the MCP server over HTTP for local integration testing.
- `python -m pytest tests/ -v`: run the main test suite.
- `python -m pytest scene_ripper_mcp/tests/ -v`: run MCP-specific tests.
- `python -m pytest tests/test_build_support.py tests/test_runtime_smoke.py -v`: run frozen-build regression coverage.
- `python -m mypy cli core models ui scene_ripper_mcp`: run static type checks.
- `ruff check .` and `ruff format .`: lint and format if Ruff is available.
- `./packaging/macos/build.sh --dmg`: build a local macOS app/DMG.
- `pwsh ./packaging/windows/build.ps1`: build a local Windows executable and installer.
- `./packaging/linux/build-appimage.sh <version>`: build a local Linux AppImage.

## Coding Style & Naming Conventions
Target Python 3.11+ with 4-space indentation. Use `snake_case` for modules/functions, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants. Add type hints for new or changed public APIs. Keep reusable logic in `core/` and UI-only behavior in `ui/`.

## Architecture & Safety Patterns
- Treat `core/project.py` as the source of truth for project mutations. Use model methods such as `add_source()` and `add_clips()` instead of mutating underlying lists directly.
- Keep heavy operations off the UI thread. Long-running detection, analysis, export, and sequencing work belongs in `ui/workers/` or equivalent worker-thread infrastructure.
- Preserve `main.py` startup ordering. Frozen environment setup must happen before heavy imports, `torch` should remain pre-imported before `PySide6`, and macOS MLX warmup must stay on the main thread.
- Use `core/paths.py` for app-support, managed binary, managed package, and log locations. Use `core/settings.py` for user-configurable paths, persisted settings, and environment overrides.
- Treat `requirements-core.txt` as the frozen base-app contract and `requirements-optional.txt` as the on-demand feature set. New heavyweight or dynamically imported dependencies should not be pulled into startup accidentally.
- Keep `core/dependency_manager.py` and `core/feature_registry.py` aligned when adding or changing optional capabilities. If a feature is install-on-demand, make its dependency gates, availability checks, and user-facing install path consistent.
- For FFmpeg or other subprocess calls, pass argument arrays and never shell-interpolated strings.
- For new UI controls, reuse shared sizing and spacing primitives from `ui/theme.py`, especially `UISizes` and `Spacing`.
- Treat multi-sequence state as model data, not widget state. Sequence edits should flow through the project/sequence model and the `SequenceTab` APIs rather than ad hoc timeline mutations.
- Treat `core/update_service.py` as the app-facing update coordinator. Platform adapters live in `core/macos_updater.py` and `core/windows_updater.py`; release-feed generation lives in `packaging/release/`.
- For frozen bundle changes, keep `packaging/build_support.py` as the source of truth for package/data/metadata collection and avoid one-off hidden-import tweaks in only a single spec file.
- Stage external runtimes through the existing packaging scripts and manifests, not ad hoc downloads. Windows runtime staging goes through `packaging/windows/stage-runtimes.ps1`; macOS and Linux staging logic lives under `packaging/macos/` and `packaging/linux/`.

## Testing Guidelines
Pytest is configured in `pyproject.toml` with `tests/` as the default path and `test_*.py` naming. `scene_ripper_mcp/tests/` exists outside that default path, so run it explicitly when changing MCP server behavior.

For bug fixes, follow a prove-it loop: add a failing test, fix the bug, and confirm the test passes. Prioritize coverage around project save/load, CLI flows, tab synchronization, sequencing logic, analysis pipelines, dependency gating, and updater/runtime safety.

- For optional dependency or feature-gating changes, update tests around `core/analysis_dependencies.py`, `core/dependency_manager.py`, `core/feature_registry.py`, or the affected UI availability paths.
- For packaging changes, update `tests/test_build_support.py` and add or adjust import-time safety coverage.
- For frozen startup or updater changes, preserve and update `tests/test_runtime_smoke.py`, `tests/test_update_service.py`, `tests/test_macos_updater.py`, and `tests/test_windows_updater.py` as applicable.
- For platform-specific path or binary-resolution changes, update the corresponding Linux and Windows compatibility tests.
- Mock network and API dependencies where practical. No formal coverage threshold is enforced, but packaging and startup regressions should be treated as release blockers.

## Release & Packaging
- Tag pushes matching `v*` drive the GitHub release workflows in `.github/workflows/build-macos.yml`, `.github/workflows/build-windows.yml`, and `.github/workflows/linux-build.yml`.
- `.github/workflows/windows-ci.yml` provides non-release Windows CI coverage on `main` pushes and pull requests.
- Windows packaging uses `packaging/windows/scene_ripper.spec`, `packaging/windows/build.ps1`, runtime staging, and Inno Setup.
- macOS packaging uses `packaging/macos/scene_ripper.spec`, `packaging/macos/build.sh`, staged Sparkle/FFmpeg runtimes, and optional signing/notarization.
- Linux packaging uses `packaging/linux/build-appimage.sh` and `packaging/linux/AppImageBuilder.yml`.
- Release feed generation and publishing live in `packaging/release/` and publish stable `update-feed` assets used by Sparkle and WinSparkle.
- Before changing packaging or updater behavior, check `docs/releases.md`, the relevant workflow file, and the runtime smoke/update tests. File-existence checks alone are not sufficient when startup, import order, updater metadata, or bundled runtimes can fail at runtime.

## Commit & Pull Request Guidelines
Use conventional commit style when possible, optionally with scopes, for example `feat(sequence): ...`, `fix(updater): ...`, or `docs: ...`. Keep commits focused and imperative.

PRs should include a short problem/solution summary, linked issues when applicable, exact test commands run, and screenshots or GIFs for UI changes. Packaging or release-surface changes should call out platform coverage and any manual validation performed.

## Security & Configuration Tips
Do not commit secrets. Store API keys in system keyring through the app when possible, or provide them through environment variables such as provider-specific `*_API_KEY` values and `SCENE_RIPPER_*` overrides.

Respect the split between bundled and managed runtime assets. Downloaded binaries, optional Python packages, logs, and app support files should resolve through the managed-path helpers instead of hardcoded directories. Ensure required system tools such as `ffmpeg` or platform media dependencies are available before exercising analysis, playback, export, or packaging flows.
