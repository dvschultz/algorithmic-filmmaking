---
title: "feat: Agent-Native Architecture Phases 2-4"
type: feat
date: 2026-01-25
status: Ready for Review
depends_on: Phase 1 (CLI Interface) - COMPLETED
---

# Agent-Native Architecture: Phases 2, 3, and 4

## Overview

This plan covers the remaining phases of the agent-native architecture transformation, enabling Scene Ripper to be fully accessible by AI agents without requiring the Qt GUI.

**Phase 1 (CLI Interface)**: ✅ COMPLETED - Merged in commit `dc11ae8`

**This plan covers:**
- **Phase 2**: State Management Extraction - Move application state from `MainWindow` to an independent `Project` class
- **Phase 3**: Environment Variable Support - Enable configuration via environment variables for agent/CI workflows
- **Phase 4**: JSON-Based Settings - Remove Qt dependency from settings module

## Problem Statement

Currently, after Phase 1 completion:
- CLI commands work but have to manually unpack tuples from `load_project()`
- Application state is duplicated between CLI (uses `CLIConfig`) and GUI (uses `Settings` with `QSettings`)
- `core/settings.py` requires Qt to be initialized, preventing true headless operation
- No unified `Project` class for managing state programmatically

## Proposed Solution

### Architecture Overview

```
                    ┌─────────────────────────────────────────────────┐
                    │                 Configuration                    │
                    │  Priority: ENV VARS > JSON > (QSettings) > Defaults │
                    └─────────────────────────────────────────────────┘
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    │                                           │
              ┌─────▼─────┐                              ┌─────▼─────┐
              │    CLI    │                              │    GUI    │
              │  (click)  │                              │ (PySide6) │
              └─────┬─────┘                              └─────┬─────┘
                    │                                           │
                    │         ┌───────────────────┐            │
                    └────────►│   Project Class   │◄───────────┘
                              │   (core/project)  │
                              │  - sources        │
                              │  - clips          │
                              │  - sequence       │
                              │  - observers[]    │
                              └─────────┬─────────┘
                                        │
              ┌─────────────────────────┼─────────────────────────┐
              │                         │                         │
        ┌─────▼─────┐            ┌─────▼─────┐            ┌─────▼─────┐
        │  Sources  │            │   Clips   │            │ Sequence  │
        │  (models) │            │  (models) │            │  (models) │
        └───────────┘            └───────────┘            └───────────┘
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Observer pattern | Python callbacks (not Qt signals) | Keeps `Project` class Qt-free for CLI |
| Settings priority | ENV > JSON > defaults | Standard for agent/CI workflows |
| API key storage | Keyring + env var override | Security (never in JSON) |
| Config file location | `~/.config/scene-ripper/config.json` | XDG-compliant |
| Concurrent access | Advisory file lock | Simple, cross-platform |
| QSettings handling | Read-only fallback during migration | Backward compatibility |

---

## Technical Approach

### Phase 2: State Management Extraction

#### 2.1 Create Unified Project Class

**File**: `core/project.py` (extend existing)

```python
@dataclass
class Project:
    """Application state independent of UI.

    This is the single source of truth for project data.
    Both CLI and GUI use this class to manage state.
    """

    # File path (None if unsaved)
    path: Optional[Path] = None

    # Metadata
    metadata: ProjectMetadata = field(default_factory=ProjectMetadata)

    # Core data
    sources: list[Source] = field(default_factory=list)
    clips: list[Clip] = field(default_factory=list)
    sequence: Optional[Sequence] = None

    # State tracking
    _dirty: bool = field(default=False, repr=False)
    _observers: list[Callable] = field(default_factory=list, repr=False)

    # Computed indexes (cached)
    @cached_property
    def sources_by_id(self) -> dict[str, Source]:
        return {s.id: s for s in self.sources}

    @cached_property
    def clips_by_id(self) -> dict[str, Clip]:
        return {c.id: c for c in self.clips}

    @cached_property
    def clips_by_source(self) -> dict[str, list[Clip]]:
        result: dict[str, list[Clip]] = {}
        for clip in self.clips:
            result.setdefault(clip.source_id, []).append(clip)
        return result

    def _invalidate_caches(self) -> None:
        """Clear cached properties when data changes."""
        for attr in ('sources_by_id', 'clips_by_id', 'clips_by_source'):
            self.__dict__.pop(attr, None)

    def _notify_observers(self, event: str, data: Any = None) -> None:
        """Notify all observers of state change."""
        for observer in self._observers:
            try:
                observer(event, data)
            except Exception as e:
                logger.warning(f"Observer error: {e}")

    def add_observer(self, callback: Callable[[str, Any], None]) -> None:
        """Register an observer for state changes."""
        self._observers.append(callback)

    def remove_observer(self, callback: Callable) -> None:
        """Unregister an observer."""
        self._observers.remove(callback)

    # Operations
    def add_source(self, source: Source) -> None:
        """Add a source video to the project."""
        self.sources.append(source)
        self._invalidate_caches()
        self._dirty = True
        self._notify_observers("source_added", source)

    def add_clips(self, clips: list[Clip]) -> None:
        """Add detected clips to the project."""
        self.clips.extend(clips)
        self._invalidate_caches()
        self._dirty = True
        self._notify_observers("clips_added", clips)

    def add_to_sequence(self, clip_ids: list[str]) -> None:
        """Add clips to the sequence by ID."""
        if self.sequence is None:
            self.sequence = Sequence()
        for clip_id in clip_ids:
            if clip := self.clips_by_id.get(clip_id):
                self.sequence.add_clip(clip)
        self._dirty = True
        self._notify_observers("sequence_changed", clip_ids)

    @property
    def is_dirty(self) -> bool:
        """Check if project has unsaved changes."""
        return self._dirty

    def mark_clean(self) -> None:
        """Mark project as saved."""
        self._dirty = False

    # Persistence
    def save(self, path: Optional[Path] = None) -> bool:
        """Save project to file."""
        save_path = path or self.path
        if save_path is None:
            raise ValueError("No path specified for save")

        success = save_project(
            filepath=save_path,
            sources=self.sources,
            clips=self.clips,
            sequence=self.sequence,
            metadata=self.metadata,
        )

        if success:
            self.path = save_path
            self.mark_clean()
            self._notify_observers("project_saved", save_path)

        return success

    @classmethod
    def load(
        cls,
        path: Path,
        missing_source_callback: Optional[Callable] = None,
    ) -> "Project":
        """Load project from file."""
        sources, clips, sequence, metadata, ui_state = load_project(
            filepath=path,
            missing_source_callback=missing_source_callback,
        )

        project = cls(
            path=path,
            metadata=metadata,
            sources=sources,
            clips=clips,
            sequence=sequence,
        )
        project._dirty = False
        return project

    @classmethod
    def new(cls, name: str = "Untitled Project") -> "Project":
        """Create a new empty project."""
        return cls(metadata=ProjectMetadata(name=name))
```

#### 2.2 Qt Signal Adapter for GUI

**File**: `ui/project_adapter.py` (new)

```python
from PySide6.QtCore import QObject, Signal

class ProjectSignalAdapter(QObject):
    """Bridges Project callbacks to Qt signals for UI updates."""

    source_added = Signal(object)  # Source
    clips_added = Signal(list)     # list[Clip]
    sequence_changed = Signal(list) # list[str] clip_ids
    project_saved = Signal(object)  # Path
    project_loaded = Signal()

    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self._project = project
        project.add_observer(self._on_project_event)

    def _on_project_event(self, event: str, data: Any) -> None:
        """Convert callback events to Qt signals."""
        signal = getattr(self, event, None)
        if signal is not None:
            signal.emit(data)
```

#### 2.3 Update MainWindow to Use Project

**File**: `ui/main_window.py` (modify)

Key changes:
1. Replace individual state variables with `self.project: Project`
2. Replace direct state manipulation with `Project` method calls
3. Use `ProjectSignalAdapter` to connect UI updates

```python
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Single source of truth for project state
        self.project = Project.new()
        self._project_adapter = ProjectSignalAdapter(self.project, self)

        # Connect signals to UI updates
        self._project_adapter.source_added.connect(self._on_source_added)
        self._project_adapter.clips_added.connect(self._on_clips_added)
        self._project_adapter.sequence_changed.connect(self._on_sequence_changed)

        # ... rest of init

    # Replace: self.sources.append(source)
    # With: self.project.add_source(source)

    # Replace: self.clips.extend(clips)
    # With: self.project.add_clips(clips)

    # Properties delegate to project
    @property
    def sources(self) -> list[Source]:
        return self.project.sources

    @property
    def clips(self) -> list[Clip]:
        return self.project.clips

    @property
    def clips_by_id(self) -> dict[str, Clip]:
        return self.project.clips_by_id
```

#### 2.4 Update CLI Commands to Use Project Class

**File**: `cli/commands/*.py` (modify all)

Before:
```python
sources, clips, sequence, metadata, ui_state = load_project(project_file)
# ... do work ...
save_project(project_file, sources, clips, sequence, ui_state, metadata)
```

After:
```python
project = Project.load(project_file)
# ... do work ...
project.save()
```

---

### Phase 3: Environment Variable Support

#### 3.1 Supported Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `YOUTUBE_API_KEY` | string | (keyring) | YouTube Data API key |
| `SCENE_RIPPER_CACHE_DIR` | path | `~/.cache/scene-ripper` | Thumbnail cache location |
| `SCENE_RIPPER_DOWNLOAD_DIR` | path | `~/Movies/Scene Ripper Downloads` | Default download location |
| `SCENE_RIPPER_EXPORT_DIR` | path | `~/Movies` | Default export location |
| `SCENE_RIPPER_CONFIG` | path | `~/.config/scene-ripper/config.json` | Config file path |
| `SCENE_RIPPER_SENSITIVITY` | float | `3.0` | Default detection sensitivity |
| `SCENE_RIPPER_WHISPER_MODEL` | string | `small.en` | Transcription model |

#### 3.2 Implementation

**File**: `core/settings.py` (modify `load_settings`)

The CLI already implements this pattern in `cli/utils/config.py`. Apply the same pattern to `core/settings.py`:

```python
def load_settings() -> Settings:
    """Load settings with priority: env vars > JSON > defaults."""
    settings = Settings()

    # 1. Load from JSON file
    config_path = _get_config_path()
    if config_path.exists():
        settings = _load_from_json(config_path, settings)

    # 2. Override with environment variables (highest priority)
    if api_key := os.environ.get("YOUTUBE_API_KEY"):
        settings.youtube_api_key = api_key
    elif not settings.youtube_api_key:
        # Only check keyring if no env var and no config value
        settings.youtube_api_key = _get_api_key_from_keyring()

    if cache_dir := os.environ.get("SCENE_RIPPER_CACHE_DIR"):
        settings.thumbnail_cache_dir = Path(cache_dir)

    if download_dir := os.environ.get("SCENE_RIPPER_DOWNLOAD_DIR"):
        settings.download_dir = Path(download_dir)

    if export_dir := os.environ.get("SCENE_RIPPER_EXPORT_DIR"):
        settings.export_dir = Path(export_dir)

    if sensitivity := os.environ.get("SCENE_RIPPER_SENSITIVITY"):
        try:
            settings.default_sensitivity = float(sensitivity)
        except ValueError:
            logger.warning(f"Invalid SCENE_RIPPER_SENSITIVITY: {sensitivity}")

    if model := os.environ.get("SCENE_RIPPER_WHISPER_MODEL"):
        settings.transcription_model = model

    return settings
```

---

### Phase 4: JSON-Based Settings

#### 4.1 JSON Settings Schema

**File**: `~/.config/scene-ripper/config.json`

```json
{
  "version": "1.0",
  "paths": {
    "thumbnail_cache_dir": "~/.cache/scene-ripper/thumbnails",
    "download_dir": "~/Movies/Scene Ripper Downloads",
    "export_dir": "~/Movies"
  },
  "detection": {
    "default_sensitivity": 3.0,
    "min_scene_length_seconds": 0.5,
    "auto_analyze_colors": true,
    "auto_classify_shots": true
  },
  "transcription": {
    "model": "small.en",
    "language": "en",
    "auto_transcribe": true
  },
  "export": {
    "quality": "medium",
    "resolution": "original",
    "fps": "original"
  },
  "appearance": {
    "theme_preference": "system"
  },
  "youtube": {
    "results_count": 25,
    "parallel_downloads": 2
  }
}
```

**Note**: YouTube API key is NOT stored in JSON. It uses keyring (secure) or environment variable.

#### 4.2 Remove Qt Dependency from Settings

**File**: `core/settings.py` (modify)

```python
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# NO Qt imports!

def _get_config_dir() -> Path:
    """Get platform-appropriate config directory (XDG-compliant)."""
    if os.name == "nt":  # Windows
        base = Path(os.environ.get("APPDATA", Path.home()))
        return base / "scene-ripper"
    else:  # macOS/Linux
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
        return base / "scene-ripper"

def _get_config_path() -> Path:
    """Get config file path, respecting SCENE_RIPPER_CONFIG env var."""
    if custom_path := os.environ.get("SCENE_RIPPER_CONFIG"):
        return Path(custom_path)
    return _get_config_dir() / "config.json"

def load_settings() -> Settings:
    """Load settings with priority: env vars > JSON > defaults.

    This function is Qt-free and works in headless environments.
    """
    settings = Settings()

    # Load from JSON
    config_path = _get_config_path()
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            settings = _apply_json_to_settings(data, settings)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Could not load config from {config_path}: {e}")

    # Apply environment variable overrides
    settings = _apply_env_overrides(settings)

    return settings

def save_settings(settings: Settings) -> bool:
    """Save settings to JSON file.

    This function is Qt-free and works in headless environments.
    """
    config_path = _get_config_path()

    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Set restrictive permissions on Unix
        if os.name != "nt":
            os.chmod(config_path.parent, 0o700)

        data = _settings_to_json(settings)

        # Atomic write
        temp_path = config_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        if os.name != "nt":
            os.chmod(temp_path, 0o600)

        os.replace(temp_path, config_path)

        # Also save API key to keyring (secure storage)
        if settings.youtube_api_key:
            _set_api_key_in_keyring(settings.youtube_api_key)

        logger.info(f"Settings saved to {config_path}")
        return True

    except (OSError, IOError) as e:
        logger.error(f"Failed to save settings: {e}")
        return False
```

#### 4.3 Migration from QSettings (One-Time)

**File**: `core/settings.py` (add migration function)

```python
def migrate_from_qsettings() -> bool:
    """Migrate settings from QSettings to JSON (one-time operation).

    Called by GUI on first launch if JSON doesn't exist but QSettings does.
    Returns True if migration was performed.
    """
    config_path = _get_config_path()

    # Skip if JSON already exists
    if config_path.exists():
        return False

    try:
        # Import Qt only for migration
        from PySide6.QtCore import QSettings

        qsettings = QSettings()
        if not qsettings.allKeys():
            return False  # No QSettings to migrate

        logger.info("Migrating settings from QSettings to JSON...")

        # Load from QSettings using old logic
        settings = _load_from_qsettings_legacy(qsettings)

        # Save to JSON
        save_settings(settings)

        logger.info("Migration complete. Settings saved to JSON.")
        return True

    except ImportError:
        # Qt not available (headless), skip migration
        return False
```

#### 4.4 Unify CLI and GUI Settings

**File**: `cli/utils/config.py` (deprecate, redirect to core)

```python
"""CLI configuration utilities.

DEPRECATED: Use core.settings instead. This module redirects for compatibility.
"""

import warnings
from core.settings import Settings, load_settings, save_settings

# Re-export for backward compatibility
CLIConfig = Settings

def get_config() -> Settings:
    warnings.warn(
        "cli.utils.config is deprecated, use core.settings.load_settings()",
        DeprecationWarning,
        stacklevel=2
    )
    return load_settings()
```

---

## Acceptance Criteria

### Phase 2: State Management

- [x] `Project` class exists with `add_source()`, `add_clips()`, `add_to_sequence()`, `save()`, `load()` methods
- [x] `Project` uses callback-based observers (Qt-free)
- [x] `ProjectSignalAdapter` bridges callbacks to Qt signals for GUI
- [x] MainWindow uses `self.project` instead of individual state variables
- [x] CLI commands use `Project.load()` and `project.save()`
- [x] Dirty state tracking works (`project.is_dirty`)
- [x] Computed properties (`clips_by_id`, etc.) are cached and invalidated correctly

### Phase 3: Environment Variables

- [ ] All 7 environment variables are supported
- [ ] Environment variables take precedence over JSON config
- [ ] Invalid environment variable values log warnings and use defaults
- [ ] GUI Settings dialog shows "(from environment)" for env-overridden values
- [ ] `SCENE_RIPPER_CONFIG` allows custom config file path

### Phase 4: JSON Settings

- [ ] `load_settings()` works without Qt initialized
- [ ] Settings saved to `~/.config/scene-ripper/config.json` with schema version
- [ ] File permissions are 0600 (Unix) for security
- [ ] Atomic writes prevent corruption
- [ ] YouTube API key stays in keyring, not JSON
- [ ] Migration from QSettings happens automatically on first GUI launch
- [ ] CLI and GUI use the same settings module (`core/settings`)

### Integration Tests

- [ ] CLI can create project, GUI can open it
- [ ] GUI can create project, CLI can modify it
- [ ] Settings changed in CLI are reflected in GUI (after restart)
- [ ] Environment variables work in both CLI and GUI

---

## Implementation Phases

### Phase 2a: Core Project Class (2-3 days)

1. Extend `core/project.py` with `Project` class
2. Add observer pattern (callbacks)
3. Add dirty state tracking
4. Add cached properties with invalidation
5. Write unit tests for `Project` class

### Phase 2b: GUI Integration (2-3 days)

1. Create `ui/project_adapter.py` with `ProjectSignalAdapter`
2. Update `MainWindow` to use `self.project`
3. Add property delegates for backward compatibility
4. Verify all existing functionality still works
5. Write integration tests

### Phase 2c: CLI Integration (1 day)

1. Update all CLI commands to use `Project.load()` / `project.save()`
2. Simplify command code (remove tuple unpacking)
3. Update tests

### Phase 3: Environment Variables (0.5 day)

1. Add env var checking to `core/settings.py`
2. Add GUI indication for env-overridden settings
3. Document environment variables in README
4. Write tests

### Phase 4a: JSON Settings Core (1 day)

1. Implement `_get_config_path()` with XDG support
2. Implement `_load_from_json()` and `_settings_to_json()`
3. Remove Qt imports from `load_settings()` / `save_settings()`
4. Write tests

### Phase 4b: Migration and Cleanup (0.5 day)

1. Implement `migrate_from_qsettings()`
2. Add migration call to GUI startup
3. Deprecate `cli/utils/config.py`
4. Update imports throughout codebase

---

## Dependencies & Risks

### Dependencies

| Dependency | Status | Risk |
|------------|--------|------|
| Phase 1 (CLI) | ✅ Complete | None |
| PySide6 | Installed | None |
| keyring | Installed | May fail in CI (handled) |

### Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Breaking existing projects | Low | High | No project file format changes |
| Breaking GUI functionality | Medium | High | Extensive testing, property delegates |
| Settings migration failure | Low | Medium | Keep QSettings as read-only fallback |
| Concurrent access conflicts | Medium | Medium | Document limitation, add file lock later |

---

## Future Considerations

### Phase 5: MCP Server (Not in this plan)

- Depends on Phases 1-4 being complete
- Will use `Project` class directly for tool implementations
- See `agent-native-architecture-plan.md` for details

### Potential Enhancements

1. **File watching**: Detect external project file changes and reload
2. **Undo/redo**: Transaction-based state changes with history
3. **Project locking**: File-level locks for concurrent access safety
4. **Settings sync**: Cloud sync for settings across machines

---

## References

### Internal

- `docs/plans/agent-native-architecture-plan.md` - Original architecture plan
- `core/project.py:74-179` - Existing save/load functions
- `cli/utils/config.py` - CLI config implementation (Phase 1)
- `ui/main_window.py:479-509` - Current state variables

### External

- [XDG Base Directory Specification](https://specifications.freedesktop.org/basedir-spec/latest/)
- [Click Documentation](https://click.palletsprojects.com/)

### Documented Learnings

- `docs/solutions/ui-bugs/timeline-widget-sequence-mismatch-20260124.md` - Single source of truth pattern
- `docs/solutions/runtime-errors/qthread-destroyed-duplicate-signal-delivery-20260124.md` - Signal guard patterns
- `docs/solutions/security-issues/ffmpeg-path-escaping-20260124.md` - Path validation patterns
