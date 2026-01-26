---
title: "feat: MCP Server Implementation (Phase 5)"
type: feat
date: 2026-01-26
status: ready
depends_on: Phase 1-4 (complete)
---

# MCP Server Implementation (Phase 5)

## Overview

Create an MCP (Model Context Protocol) server for Scene Ripper that exposes video processing tools to external AI agents like Claude, ChatGPT, and OpenAI Agents SDK. This enables any MCP-compatible client to search YouTube, detect scenes, analyze clips, and export video sequences programmatically.

## Problem Statement

While Scene Ripper has 57+ agent tools in `core/chat_tools.py`, these are only accessible through:
1. The embedded chat panel (requires GUI)
2. The CLI commands (requires shell access)

External AI applications cannot directly invoke Scene Ripper capabilities. MCP provides a standardized protocol that allows any compatible AI client to discover and use these tools.

## Proposed Solution

Implement an MCP server using FastMCP (Python SDK) that wraps existing tool implementations. The server will support stdio transport for local Claude Desktop integration and Streamable HTTP for remote/production use.

---

## Technical Approach

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     MCP Clients                              │
│  (Claude Desktop, OpenAI Agents, ChatGPT, Custom Clients)   │
└─────────────────────────┬───────────────────────────────────┘
                          │ MCP Protocol (JSON-RPC)
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                   MCP Server Layer                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  server.py   │  │   tools.py   │  │  schemas.py  │      │
│  │  (FastMCP)   │  │ (Tool defs)  │  │ (Pydantic)   │      │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘      │
│         │                 │                                  │
│  ┌──────▼─────────────────▼──────────────────────────┐     │
│  │              executor.py                           │     │
│  │         (Wraps core functions)                     │     │
│  └──────────────────────┬────────────────────────────┘     │
└─────────────────────────┼───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                   Core Layer (Existing)                      │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │  project   │  │  settings  │  │  youtube   │            │
│  ├────────────┤  ├────────────┤  ├────────────┤            │
│  │  ffmpeg    │  │  scene_det │  │ transcribe │            │
│  └────────────┘  └────────────┘  └────────────┘            │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| MCP Framework | FastMCP (mcp[cli]) | Official Python SDK with decorator syntax matching existing patterns |
| Schema Validation | Pydantic v2 | Already used in codebase, auto-generates JSON Schema |
| Transport (Local) | stdio | Required for Claude Desktop integration |
| Transport (Remote) | Streamable HTTP | Modern standard, supports stateless operation |
| Python Version | 3.10+ | Already required by project |

### Directory Structure

```
mcp/
├── __init__.py           # Package marker
├── server.py             # FastMCP server entry point
├── tools/
│   ├── __init__.py
│   ├── project.py        # Project management tools
│   ├── import_analyze.py # Import and analysis tools
│   ├── clips.py          # Clip query and manipulation
│   ├── sequence.py       # Sequence/timeline tools
│   ├── youtube.py        # YouTube search and download
│   ├── export.py         # Export operations
│   └── settings.py       # Settings tools
├── schemas/
│   ├── __init__.py
│   ├── inputs.py         # Pydantic input models
│   └── outputs.py        # Pydantic output models
├── executor.py           # Execution engine (wraps core/)
├── security.py           # Path validation, auth helpers
└── tests/
    ├── __init__.py
    ├── test_tools.py
    ├── test_executor.py
    └── test_integration.py
```

---

## Implementation Phases

### Phase 5.1: Foundation (Core Server)

**Goal**: Minimal working MCP server with 5 essential tools

#### Files to Create

**mcp/__init__.py**
```python
"""Scene Ripper MCP Server - Expose video processing tools via Model Context Protocol."""
__version__ = "0.1.0"
```

**mcp/server.py**
```python
"""MCP Server entry point using FastMCP."""
from mcp.server.fastmcp import FastMCP
from contextlib import asynccontextmanager
import logging
import sys

# Configure logging to stderr (critical for stdio transport)
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@asynccontextmanager
async def lifespan():
    """Initialize resources on server startup."""
    from core.settings import load_settings
    settings = load_settings()
    yield {"settings": settings}

mcp = FastMCP(
    "scene_ripper_mcp",
    version="0.1.0",
    lifespan=lifespan
)

# Import tool registrations
from mcp.tools import project, youtube, export

def main():
    """Run the MCP server."""
    import argparse
    parser = argparse.ArgumentParser(description="Scene Ripper MCP Server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    if args.transport == "http":
        mcp.run(transport="streamable-http", port=args.port)
    else:
        mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
```

**mcp/schemas/inputs.py**
```python
"""Pydantic input models for MCP tools."""
from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Optional, List
from pathlib import Path

class ProjectPathInput(BaseModel):
    """Input requiring a project path."""
    model_config = ConfigDict(str_strip_whitespace=True)

    project_path: str = Field(
        ...,
        description="Absolute path to .json project file",
        min_length=1
    )

    @field_validator('project_path')
    @classmethod
    def validate_path(cls, v: str) -> str:
        if not v.endswith('.json'):
            raise ValueError("Project path must end with .json")
        return v

class DetectScenesInput(BaseModel):
    """Input for scene detection."""
    model_config = ConfigDict(str_strip_whitespace=True)

    video_path: str = Field(..., description="Absolute path to video file")
    output_project: str = Field(..., description="Path for output project JSON")
    sensitivity: float = Field(
        default=3.0,
        description="Detection sensitivity (1.0=more scenes, 10.0=fewer)",
        ge=1.0,
        le=10.0
    )
    min_scene_length: float = Field(
        default=0.5,
        description="Minimum scene length in seconds",
        ge=0.1
    )

class YouTubeSearchInput(BaseModel):
    """Input for YouTube search."""
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(..., description="Search query", min_length=2, max_length=200)
    max_results: int = Field(default=25, ge=1, le=50)

class DownloadVideoInput(BaseModel):
    """Input for video download."""
    model_config = ConfigDict(str_strip_whitespace=True)

    url: str = Field(..., description="YouTube or video URL")
    output_dir: Optional[str] = Field(
        default=None,
        description="Output directory (defaults to settings.download_dir)"
    )

class FilterClipsInput(BaseModel):
    """Input for filtering clips."""
    model_config = ConfigDict(str_strip_whitespace=True)

    project_path: str = Field(..., description="Path to project file")
    shot_type: Optional[str] = Field(default=None, description="Filter by shot type")
    has_speech: Optional[bool] = Field(default=None, description="Filter by speech presence")
    min_duration: Optional[float] = Field(default=None, description="Minimum duration (seconds)")
    max_duration: Optional[float] = Field(default=None, description="Maximum duration (seconds)")
    tags: Optional[List[str]] = Field(default=None, description="Filter by tags (any match)")
```

**mcp/security.py**
```python
"""Security utilities for MCP server."""
from pathlib import Path
from typing import Tuple
import os

# Safe root directories (reuse pattern from chat_tools.py)
SAFE_ROOTS = [
    Path.home(),
    Path("/tmp"),
    Path("/var/folders"),  # macOS temp
]

# Add macOS Volumes for external drives
if Path("/Volumes").exists():
    SAFE_ROOTS.append(Path("/Volumes"))

def validate_path(
    path_str: str,
    must_exist: bool = False,
    must_be_file: bool = False,
    must_be_dir: bool = False
) -> Tuple[bool, str, Path]:
    """Validate a path for security and existence.

    Args:
        path_str: Path string to validate
        must_exist: Require path to exist
        must_be_file: Require path to be a file
        must_be_dir: Require path to be a directory

    Returns:
        Tuple of (is_valid, error_message, resolved_path)
    """
    try:
        path = Path(path_str).expanduser().resolve()
    except Exception as e:
        return False, f"Invalid path: {e}", Path()

    # Check for path traversal
    if ".." in str(path):
        return False, "Path traversal not allowed", path

    # Verify path is under safe root
    is_safe = any(
        str(path).startswith(str(root.resolve()))
        for root in SAFE_ROOTS
    )
    if not is_safe:
        return False, f"Path must be under home directory or temp: {path}", path

    # Existence checks
    if must_exist and not path.exists():
        return False, f"Path does not exist: {path}", path

    if must_be_file and path.exists() and not path.is_file():
        return False, f"Path is not a file: {path}", path

    if must_be_dir and path.exists() and not path.is_dir():
        return False, f"Path is not a directory: {path}", path

    return True, "", path
```

**mcp/tools/project.py**
```python
"""Project management MCP tools."""
from mcp.server.fastmcp import Context
from mcp.server import mcp
from mcp.schemas.inputs import ProjectPathInput, DetectScenesInput
from mcp.security import validate_path
import json

@mcp.tool(
    name="get_project_info",
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def get_project_info(params: ProjectPathInput, ctx: Context) -> str:
    """Get information about a Scene Ripper project.

    Returns project metadata including source count, clip count,
    sequence length, and analysis status.

    Args:
        params: Project path input

    Returns:
        JSON with project information
    """
    valid, error, path = validate_path(params.project_path, must_exist=True, must_be_file=True)
    if not valid:
        return json.dumps({"success": False, "error": error})

    try:
        from core.project import Project
        project = Project.load(path)

        return json.dumps({
            "success": True,
            "project": {
                "path": str(path),
                "name": project.name,
                "source_count": len(project.sources),
                "clip_count": len(project.clips),
                "sequence_length": len(project.sequence.clips) if project.sequence else 0,
                "has_colors": any(c.metadata.get("dominant_colors") for c in project.clips),
                "has_shots": any(c.metadata.get("shot_type") for c in project.clips),
                "has_transcripts": any(c.metadata.get("transcript") for c in project.clips),
            }
        })
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})

@mcp.tool(
    name="detect_scenes",
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False
    }
)
async def detect_scenes(params: DetectScenesInput, ctx: Context) -> str:
    """Detect scenes in a video file and create a new project.

    Uses adaptive scene detection to identify shot boundaries.
    Creates a project file with the detected clips.

    Args:
        params: Video path, output project path, detection settings

    Returns:
        JSON with detection results and clip count
    """
    await ctx.report_progress(0.0, "Validating paths...")

    valid, error, video_path = validate_path(params.video_path, must_exist=True, must_be_file=True)
    if not valid:
        return json.dumps({"success": False, "error": f"Video: {error}"})

    valid, error, output_path = validate_path(params.output_project)
    if not valid:
        return json.dumps({"success": False, "error": f"Output: {error}"})

    try:
        await ctx.report_progress(0.1, "Running scene detection...")

        from core.scene_detect import SceneDetector
        from core.project import Project
        from models.clip import Source

        # Run detection
        detector = SceneDetector(
            threshold=params.sensitivity,
            min_scene_len=params.min_scene_length
        )
        scenes = detector.detect(video_path)

        await ctx.report_progress(0.8, "Creating project...")

        # Create project
        source = Source(file_path=video_path)
        project = Project(name=video_path.stem)
        project.add_source(source)

        # Create clips from scenes
        from models.clip import Clip
        for i, (start, end) in enumerate(scenes):
            clip = Clip(
                source_id=source.id,
                start_time=start,
                end_time=end
            )
            project.add_clip(clip)

        # Save project
        project.save(output_path)

        await ctx.report_progress(1.0, "Complete")

        return json.dumps({
            "success": True,
            "project_path": str(output_path),
            "source": str(video_path),
            "clip_count": len(project.clips),
            "total_duration": sum(c.duration for c in project.clips)
        })
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})
```

**mcp/tools/youtube.py**
```python
"""YouTube search and download MCP tools."""
from mcp.server.fastmcp import Context
from mcp.server import mcp
from mcp.schemas.inputs import YouTubeSearchInput, DownloadVideoInput
from mcp.security import validate_path
import json

@mcp.tool(
    name="search_youtube",
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def search_youtube(params: YouTubeSearchInput, ctx: Context) -> str:
    """Search YouTube for videos matching a query.

    Requires YOUTUBE_API_KEY environment variable or keyring credential.

    Args:
        params: Search query and result limit

    Returns:
        JSON with video results (id, title, channel, duration, thumbnail)
    """
    try:
        from core.youtube_api import YouTubeSearchClient

        client = YouTubeSearchClient()
        results = await client.search(params.query, max_results=params.max_results)

        videos = [
            {
                "video_id": v.video_id,
                "title": v.title,
                "channel": v.channel_title,
                "duration": v.duration_str,
                "thumbnail": v.thumbnail_url,
                "url": f"https://www.youtube.com/watch?v={v.video_id}"
            }
            for v in results
        ]

        return json.dumps({
            "success": True,
            "query": params.query,
            "count": len(videos),
            "videos": videos
        })
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})

@mcp.tool(
    name="download_video",
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True
    }
)
async def download_video(params: DownloadVideoInput, ctx: Context) -> str:
    """Download a video from YouTube or other supported sites.

    Uses yt-dlp for downloading. Supports YouTube, Vimeo, and many other sites.

    Args:
        params: Video URL and optional output directory

    Returns:
        JSON with download result and file path
    """
    # Validate output directory if provided
    if params.output_dir:
        valid, error, output_path = validate_path(params.output_dir, must_be_dir=True)
        if not valid:
            return json.dumps({"success": False, "error": error})
    else:
        from core.settings import load_settings
        settings = load_settings()
        output_path = settings.download_dir

    try:
        await ctx.report_progress(0.0, "Starting download...")

        from core.downloader import VideoDownloader

        downloader = VideoDownloader(output_dir=output_path)
        result = await downloader.download(params.url)

        await ctx.report_progress(1.0, "Complete")

        if result.success:
            return json.dumps({
                "success": True,
                "file_path": str(result.file_path),
                "title": result.title,
                "duration": result.duration
            })
        else:
            return json.dumps({"success": False, "error": result.error})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})
```

#### Dependencies to Add

**pyproject.toml** (add to dependencies):
```toml
[project.optional-dependencies]
mcp = [
    "mcp[cli]>=1.2.0",
]

[project.scripts]
scene-ripper-mcp = "mcp.server:main"
```

---

### Phase 5.2: Core Tool Coverage

**Goal**: Expose all major operations (30+ tools)

#### Tools to Implement

**Analysis Tools** (mcp/tools/analyze.py):
- `analyze_colors(project_path)` - Extract color palettes
- `analyze_shots(project_path)` - Classify shot types
- `transcribe(project_path, model, language)` - Speech transcription
- `get_analysis_status(project_path)` - Check what's analyzed

**Clip Tools** (mcp/tools/clips.py):
- `list_clips(project_path)` - List all clips with metadata
- `filter_clips(project_path, filters)` - Query by criteria
- `get_clip_metadata(project_path, clip_id)` - Single clip details
- `add_clip_tags(project_path, clip_id, tags)` - Tag clips
- `remove_clip_tags(project_path, clip_id, tags)` - Remove tags
- `add_clip_note(project_path, clip_id, note)` - Add annotation

**Sequence Tools** (mcp/tools/sequence.py):
- `get_sequence(project_path)` - Get timeline state
- `add_to_sequence(project_path, clip_ids)` - Add clips
- `remove_from_sequence(project_path, clip_ids)` - Remove clips
- `reorder_sequence(project_path, clip_ids)` - Reorder
- `clear_sequence(project_path)` - Empty timeline

**Export Tools** (mcp/tools/export.py):
- `export_clips(project_path, output_dir, clip_ids)` - Export video clips
- `export_sequence(project_path, output_path)` - Render sequence video
- `export_edl(project_path, output_path)` - Export EDL
- `export_dataset(project_path, output_path)` - Export metadata JSON

**Project Tools** (expand mcp/tools/project.py):
- `list_projects(directory)` - Find .json projects
- `create_project(name, output_path)` - New empty project
- `import_video(project_path, video_path)` - Add video to project
- `list_sources(project_path)` - List imported videos
- `remove_source(project_path, source_id)` - Remove video

---

### Phase 5.3: HTTP Transport & Claude Desktop Integration

**Goal**: Production-ready server with Claude Desktop config

#### HTTP Transport Implementation

**mcp/http.py**:
```python
"""HTTP transport configuration for production deployment."""
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.middleware.cors import CORSMiddleware
from mcp.server import mcp

def create_app():
    """Create Starlette app with MCP mounted."""
    app = Starlette(
        routes=[
            Mount("/mcp", app=mcp.streamable_http_app(json_response=True))
        ]
    )

    # CORS for web clients
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app

app = create_app()
```

#### Claude Desktop Configuration

**README section for users**:
```json
// Add to ~/Library/Application Support/Claude/claude_desktop_config.json
{
  "mcpServers": {
    "scene-ripper": {
      "command": "python",
      "args": ["-m", "mcp.server"],
      "cwd": "/path/to/algorithmic-filmmaking",
      "env": {
        "YOUTUBE_API_KEY": "your-api-key"
      }
    }
  }
}
```

#### MCP Inspector Testing

```bash
# Test server with MCP Inspector
npx @modelcontextprotocol/inspector python -m mcp.server
```

---

## Acceptance Criteria

### Functional Requirements

- [x] MCP server starts without errors via `python -m scene_ripper_mcp.server`
- [x] Server exposes 30+ tools matching existing chat_tools capabilities (33 tools implemented)
- [x] Tools return structured JSON responses with success/error status
- [x] Path validation prevents directory traversal attacks
- [x] YouTube search works with API key from env or keyring
- [x] Scene detection creates valid project files
- [ ] Export operations produce expected file outputs (pending integration test)

### Non-Functional Requirements

- [x] No Qt dependency in MCP server (headless operation)
- [x] Logging uses stderr only (stdio transport compatibility)
- [ ] Tool timeouts configurable via environment
- [ ] Memory efficient for large video operations (pending validation)

### Quality Gates

- [x] All tools have comprehensive docstrings with Args/Returns
- [x] Pydantic models validate all inputs
- [ ] MCP Inspector shows all tools with correct schemas (pending test)
- [ ] Integration tests cover happy path for each tool category
- [ ] Claude Desktop integration documented with example config

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Tool coverage | 30+ tools exposed |
| Parity with chat_tools | 80%+ functionality |
| Claude Desktop integration | Working example |
| MCP Inspector validation | All tools pass |

---

## Dependencies & Prerequisites

### Completed Prerequisites
- [x] Phase 1: CLI complete (commands working)
- [x] Phase 2: Project class extracted from MainWindow
- [x] Phase 3: Environment variable support
- [x] Phase 4: JSON-based settings (no Qt dependency)

### New Dependencies
```
mcp[cli]>=1.2.0      # MCP Python SDK with FastMCP
starlette>=0.40.0    # HTTP transport (optional)
uvicorn>=0.30.0      # ASGI server (optional, for HTTP)
```

---

## Risk Analysis & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| MCP SDK v2 breaking changes | Medium | Pin to v1.x, test before upgrading |
| Long operations timeout | High | Use progress reporting, configurable timeouts |
| Path security vulnerabilities | Critical | Reuse validated path checking from chat_tools |
| Qt imports in core modules | Medium | Audit imports, create headless variants if needed |
| YouTube API rate limits | Medium | Document limits, add retry logic |

---

## Future Considerations

### Phase 5.4 (Future)
- OAuth 2.1 authentication for HTTP transport
- Multi-project session management
- Streaming progress for long operations
- Resource exposure (project files as MCP resources)
- Prompt templates for common workflows

### Integration Opportunities
- OpenAI Agents SDK compatibility
- VS Code extension with MCP client
- Web UI with MCP backend

---

## References

### Internal
- `docs/plans/agent-native-architecture-plan.md` - Original Phase 5 spec
- `core/chat_tools.py` - Existing tool implementations (57+ tools)
- `core/tool_executor.py` - Execution patterns
- `core/project.py` - Project class

### External
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [MCP Specification](https://modelcontextprotocol.io/specification/2025-11-25)
- [FastMCP Documentation](https://github.com/jlowin/fastmcp)
- [Claude Desktop MCP Guide](https://modelcontextprotocol.io/docs/quickstart/user)
