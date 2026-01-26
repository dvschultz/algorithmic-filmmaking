# Scene Ripper MCP Server - Claude Desktop Setup

## Prerequisites

- Python 3.10+
- Scene Ripper installed
- Claude Desktop app

## Installation

1. Install MCP dependencies:

   ```bash
   pip install -e ".[mcp]"
   # Or with pip:
   pip install mcp[cli]>=1.2.0
   ```

2. Configure Claude Desktop:

   Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

   ```json
   {
     "mcpServers": {
       "scene-ripper": {
         "command": "python",
         "args": ["-m", "scene_ripper_mcp.server"],
         "cwd": "/path/to/algorithmic-filmmaking",
         "env": {
           "YOUTUBE_API_KEY": "your-api-key-here"
         }
       }
     }
   }
   ```

   > **Note:** Replace `/path/to/algorithmic-filmmaking` with the actual path to your Scene Ripper installation.

3. Restart Claude Desktop

## Available Tools

Scene Ripper exposes 33 tools for video processing:

### Project Management

| Tool | Description |
|------|-------------|
| `create_project` | Create a new empty project |
| `get_project_info` | Get project metadata (sources, clips, status) |
| `list_projects` | Find Scene Ripper projects in a directory |
| `import_video` | Import a video file into a project |
| `list_sources` | List all video sources in a project |
| `remove_source` | Remove a source and its clips |
| `detect_scenes` | Run scene detection on a video |

### Analysis

| Tool | Description |
|------|-------------|
| `analyze_colors` | Extract dominant color palettes |
| `analyze_shots` | Classify shot types (wide, medium, close-up) |
| `transcribe` | Transcribe audio using Whisper |
| `get_analysis_status` | Check analysis completion status |
| `get_video_info` | Get video file metadata |

### Clip Operations

| Tool | Description |
|------|-------------|
| `list_clips` | List all clips with metadata |
| `filter_clips` | Query clips by shot type, duration, tags |
| `get_clip_metadata` | Get detailed info for one clip |
| `add_clip_tags` | Tag clips for organization |
| `remove_clip_tags` | Remove tags from clips |
| `add_clip_note` | Add annotation notes |
| `search_transcripts` | Search clip transcripts |

### Sequence/Timeline

| Tool | Description |
|------|-------------|
| `get_sequence` | Get current timeline state |
| `add_to_sequence` | Add clips to timeline |
| `remove_from_sequence` | Remove clips from timeline |
| `reorder_sequence` | Reorder timeline clips |
| `clear_sequence` | Clear all clips from timeline |
| `shuffle_sequence` | Randomly reorder or sort by shot type |

### Export

| Tool | Description |
|------|-------------|
| `export_clips` | Export individual video clips |
| `export_sequence` | Render final sequence video |
| `export_edl` | Export Edit Decision List |
| `export_dataset` | Export clip metadata as JSON |
| `export_full_dataset` | Export complete project as JSON |

### YouTube

| Tool | Description |
|------|-------------|
| `search_youtube` | Search for videos (requires API key) |
| `download_video` | Download a single video |
| `download_videos` | Batch download multiple videos |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `YOUTUBE_API_KEY` | For YouTube tools | YouTube Data API v3 key |
| `MCP_TOOL_TIMEOUT` | No | Timeout for long operations (default: 300s) |

### Getting a YouTube API Key

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create or select a project
3. Enable "YouTube Data API v3"
4. Create credentials (API key)
5. Add the key to your config

## Example Workflows

### Detect scenes and analyze

```
Claude: "Create a project from /Videos/interview.mp4 with scene detection"
→ detect_scenes(video_path="/Videos/interview.mp4", output_project="/Projects/interview.json")

Claude: "Analyze colors and shot types"
→ analyze_colors(project_path="/Projects/interview.json")
→ analyze_shots(project_path="/Projects/interview.json")
```

### Build a sequence from search

```
Claude: "Download 10 nature videos and create a random cut"
→ search_youtube(query="nature documentary", max_results=10)
→ download_videos(urls=[...], output_dir="/Downloads")
→ For each video: detect_scenes → import clips
→ filter_clips(shot_type="wide")
→ add_to_sequence(clip_ids=[...])
→ shuffle_sequence(method="random")
→ export_sequence(output_path="/Exports/nature_cut.mp4")
```

### Export for editing software

```
Claude: "Export the sequence as an EDL for DaVinci Resolve"
→ export_edl(project_path="/Projects/my_project.json", output_path="/Exports/timeline.edl")
```

## Troubleshooting

### Server doesn't start

**Check Python path:**
```bash
# Verify Python can import the module
python -c "from scene_ripper_mcp.server import mcp; print('OK')"
```

**Check cwd:**
Ensure `cwd` in your config points to the algorithmic-filmmaking directory.

**Check logs:**
Claude Desktop logs MCP server output. Look for Python errors.

### YouTube search fails

**API key issues:**
- Verify `YOUTUBE_API_KEY` is set correctly
- Check API quota in Google Cloud Console
- Ensure YouTube Data API v3 is enabled

**Rate limits:**
YouTube API has daily quotas. Heavy usage may exhaust limits.

### Scene detection is slow

Scene detection processes every frame. For long videos:
- Detection can take several minutes
- Progress is reported via MCP progress events
- Consider using higher `sensitivity` value (less scenes = faster)

### Missing video files

If source files were moved:
- Update paths in the project JSON
- Or re-import videos

### Permission errors

Ensure the configured user has:
- Read access to video files
- Write access to project/export directories

## Testing the Server

### With MCP Inspector

```bash
npm install -g @modelcontextprotocol/inspector
npx @modelcontextprotocol/inspector python -m scene_ripper_mcp.server
```

This opens a web UI to test tools interactively.

### Programmatically

```python
from scene_ripper_mcp.server import mcp

# List all tools
tools = mcp._tool_manager._tools
print(f"Tools: {len(tools)}")
for name in sorted(tools.keys()):
    print(f"  - {name}")
```

## Security Notes

- Path validation prevents directory traversal
- Only paths under home directory or temp are allowed
- Sensitive data (API keys) should use environment variables, not hardcoded
