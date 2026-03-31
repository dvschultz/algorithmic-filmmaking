# Agent Tools Reference

The Scene Ripper chat agent has access to 103 tools organized by category. You don't need to call these by name — just describe what you want in natural language and the agent will use the right tool. This reference is for understanding what's possible.

> Type `/help` in the chat panel to see a condensed version of this list.

---

## Project Management

| Tool | What it does | Example prompt |
|------|-------------|----------------|
| `new_project` | Create a new empty project | "Start a new project called Detroit Remix" |
| `load_project` | Open an existing .sceneripper file | "Open the project at ~/Videos/my-edit.sceneripper" |
| `save_project` | Save the current project | "Save the project" |
| `set_project_name` | Rename the project | "Rename this project to Summer Montage" |
| `get_project_state` | Get full project data | "What's in this project?" |
| `get_project_summary` | Get a formatted project overview | "Give me a project summary" |
| `get_settings` | Read current app settings | "What are my current settings?" |
| `update_settings` | Change a setting | "Set the transcription model to large-v3" |

---

## Import & Sources

| Tool | What it does | Example prompt |
|------|-------------|----------------|
| `import_video` | Import a local video file | "Import /Users/me/Videos/interview.mp4" |
| `download_video` | Download from YouTube/Vimeo URL | "Download this video: https://youtube.com/watch?v=..." |
| `download_videos` | Download multiple URLs | "Download all 3 of these videos" |
| `search_youtube` | Search YouTube for videos | "Search YouTube for 'experimental film 1960s'" |
| `search_internet_archive` | Search Internet Archive | "Search Internet Archive for public domain cartoons" |
| `list_sources` | List all imported source videos | "Show me all my sources" |
| `select_source` | Set the active source | "Switch to the interview source" |
| `update_source` | Update source metadata | "Mark this source as analyzed" |
| `remove_source` | Remove a source and its clips | "Remove the first source" |

---

## Scene Detection

| Tool | What it does | Example prompt |
|------|-------------|----------------|
| `detect_scenes` | Detect scenes (basic, offline) | "Detect scenes in this video" |
| `detect_scenes_live` | Detect scenes with live preview | "Detect scenes with high sensitivity" |
| `detect_all_unanalyzed` | Detect scenes in all new sources | "Detect scenes in everything" |
| `check_detection_status` | Check if detection is running | "Is scene detection still running?" |

---

## Clip Management

| Tool | What it does | Example prompt |
|------|-------------|----------------|
| `list_clips` | List clips with optional filters | "List all clips" |
| `filter_clips` | Filter clips by metadata | "Show me all close-up shots" |
| `select_clips` | Select specific clips | "Select clips 1 through 5" |
| `select_all_clips` | Select all visible clips | "Select all" |
| `deselect_all_clips` | Clear selection | "Deselect everything" |
| `update_clip` | Edit clip metadata | "Rename this clip to 'Opening Shot'" |
| `show_clip_details` | Open the clip detail sidebar | "Show me the details of this clip" |
| `add_tags` | Add tags to clips | "Tag these clips as 'hero shots'" |
| `remove_tags` | Remove tags from clips | "Remove the 'rough' tag" |
| `add_note` | Add a note to a clip | "Note: this clip has lens flare" |
| `update_clip_transcript` | Edit a transcript segment | "Fix the transcript — it should say 'hello' not 'halo'" |
| `toggle_clip_disabled` | Enable/disable a clip | "Disable clip 3" |
| `delete_clips` | Delete clips permanently | "Delete the selected clips" |

---

## Analysis

| Tool | What it does | Example prompt |
|------|-------------|----------------|
| `start_clip_analysis` | Run one or more analysis types | "Analyze colors and shot types on all clips" |
| `analyze_all_live` | Run analysis pipeline on all clips | "Run all analysis" |
| `describe_content_live` | Describe clip content with VLM | "Describe what's in the selected clips" |
| `custom_visual_query` | Ask a yes/no question about clips | "Which clips show someone dancing?" |
| `get_clip_cinematography` | Get full cinematography analysis | "Show me the cinematography for clip 5" |
| `update_clip_cinematography` | Manually edit cinematography data | "Change the shot size to ECU" |
| `clear_clip_cinematography` | Remove cinematography analysis | "Clear the cinematography data for these clips" |

---

## Search & Discovery

| Tool | What it does | Example prompt |
|------|-------------|----------------|
| `search_transcripts` | Search across all transcripts | "Find clips where someone says 'revolution'" |
| `find_similar_clips` | Find visually similar clips | "Find clips similar to clip 7" |
| `group_clips_by` | Group clips by a property | "Group clips by shot type" |
| `get_film_term_definition` | Look up a film term | "What does 'rack focus' mean?" |
| `search_glossary` | Search the film glossary | "Search the glossary for lighting terms" |

---

## Sequencing

### Algorithm Generation

| Tool | What it does | Example prompt |
|------|-------------|----------------|
| `generate_remix` | Generate a sequence with any algorithm | "Arrange clips by color, warm to cool" |
| `generate_staccato` | Beat-driven sequence from music | "Create a staccato sequence using this song" |
| `generate_rose_hobart` | Filter clips by a person's face | "Show only clips with this person" |
| `generate_reference_guided` | Match clips to a reference edit | "Match my clips to the reference video's structure" |
| `list_sorting_algorithms` | See available algorithms | "What sequencing algorithms can I use?" |
| `get_available_dimensions` | See matching dimensions | "What dimensions are available for reference matching?" |

**Available algorithms for `generate_remix`:**

| Algorithm | Name | Example |
|-----------|------|---------|
| `shuffle` | Hatchet Job | "Shuffle the clips randomly" |
| `color` | Chromatics | "Sort by color, rainbow order" |
| `duration` | Tempo Shift | "Arrange shortest to longest" |
| `brightness` | Into the Dark | "Sort from bright to dark" |
| `volume` | Crescendo | "Arrange by volume, quiet to loud" |
| `shot_type` | Focal Ladder | "Sort by shot type, wide to close" |
| `proximity` | Up Close and Personal | "Alternate between wide and close shots" |
| `similarity_chain` | Human Centipede | "Chain clips by visual similarity" |
| `match_cut` | Match Cut | "Create match cuts between clips" |
| `sequential` | Time Capsule | "Keep clips in original order" |
| `exquisite_corpus` | Exquisite Corpus | "Create a poem from the on-screen text" |
| `storyteller` | Storyteller | "Tell a story with these clips" |

### Sequence Editing

| Tool | What it does | Example prompt |
|------|-------------|----------------|
| `get_sequence_state` | See the current sequence | "What's on the timeline?" |
| `add_to_sequence` | Add clips to the timeline | "Add clips 1, 3, and 5 to the sequence" |
| `remove_from_sequence` | Remove clips from timeline | "Remove the last clip" |
| `clear_sequence` | Clear the entire timeline | "Clear the sequence" |
| `reorder_sequence` | Rearrange clip order | "Move clip 3 to the beginning" |
| `sort_sequence` | Sort the sequence | "Sort the timeline by duration" |
| `update_sequence_clip` | Edit a clip on the timeline | "Trim the first clip to 2 seconds" |
| `set_sequence_shot_filter` | Filter sequence by shot type | "Show only close-ups in the sequence" |
| `create_sequence` | Create a new empty sequence | "Create a new sequence at 24fps" |
| `update_sequence` | Update sequence settings | "Set the sequence FPS to 30" |

### Audio Analysis

| Tool | What it does | Example prompt |
|------|-------------|----------------|
| `detect_audio_beats` | Find beats in audio | "Detect beats in this music file" |
| `align_sequence_to_audio` | Align cuts to beats | "Align the sequence to the music" |
| `get_sequence_analysis` | Analyze pacing and flow | "Analyze the pacing of this sequence" |
| `check_continuity_issues` | Find editing problems | "Check for continuity issues" |
| `generate_analysis_report` | Generate a detailed report | "Create a full analysis report" |

---

## Playback

| Tool | What it does | Example prompt |
|------|-------------|----------------|
| `play_preview` | Start playback | "Play" |
| `pause_preview` | Pause playback | "Pause" |
| `stop_playback` | Stop playback | "Stop" |
| `seek_to_time` | Jump to a time | "Go to 1 minute 30 seconds" |
| `frame_step_forward` | Advance one frame | "Next frame" |
| `frame_step_backward` | Go back one frame | "Previous frame" |
| `set_playback_speed` | Change speed | "Play at half speed" |
| `set_ab_loop` | Loop a section | "Loop between 10s and 20s" |

---

## Navigation & UI

| Tool | What it does | Example prompt |
|------|-------------|----------------|
| `navigate_to_tab` | Switch tabs | "Go to the Analyze tab" |
| `send_to_analyze` | Send clips to Analyze tab | "Send these to the Analyze tab" |
| `clear_analyze_clips` | Clear the Analyze tab | "Clear the Analyze tab" |
| `apply_filters` | Set clip browser filters | "Filter to clips longer than 3 seconds" |
| `clear_filters` | Remove all filters | "Clear all filters" |
| `set_clip_sort_order` | Change sort order | "Sort by duration" |
| `navigate_to_frames_tab` | Go to Frames tab | "Go to frames" |

---

## Frames

| Tool | What it does | Example prompt |
|------|-------------|----------------|
| `extract_frames` | Extract frames from clips | "Extract 1 frame per second" |
| `import_frames` | Import image files as frames | "Import these images as frames" |
| `list_frames` | List all frames | "Show me my frames" |
| `select_frames` | Select specific frames | "Select frames 1 through 10" |
| `analyze_frames` | Run analysis on frames | "Analyze these frames" |
| `add_frames_to_sequence` | Add frames to timeline | "Add these frames to the sequence" |
| `update_frame` | Edit frame metadata | "Tag this frame as 'key moment'" |
| `delete_frames` | Delete frames | "Delete the selected frames" |

---

## Export

| Tool | What it does | Example prompt |
|------|-------------|----------------|
| `export_sequence` | Export sequence as MP4 | "Export the sequence as a video" |
| `export_clips` | Export individual clips | "Export all clips as separate files" |
| `export_edl` | Export as EDL for NLEs | "Export an EDL file" |
| `export_srt` | Export subtitles/metadata as SRT | "Export an SRT file" |
| `export_dataset` | Export clip data as JSON | "Export a dataset of all clip metadata" |
| `export_bundle` | Export a self-contained project bundle | "Export a portable project bundle" |

---

## Plans

The agent can create multi-step plans for complex tasks. Plans are proposed, confirmed by you, then executed step by step.

| Tool | What it does | Example prompt |
|------|-------------|----------------|
| `present_plan` | Propose a plan | "Make a plan to analyze and sequence these clips" |
| `start_plan_execution` | Begin executing a confirmed plan | (automatic after you confirm) |
| `complete_plan_step` | Mark a step complete | (automatic during execution) |
| `fail_plan_step` | Handle a step failure | (automatic on error) |
| `get_plan_status` | Check plan progress | "How's the plan going?" |
| `cancel_plan` | Cancel the current plan | "Cancel the plan" |

---

## Slash Commands

These shortcuts work in the chat input:

| Command | What it does |
|---------|-------------|
| `/help` | Show available capabilities |
| `/status` | Show project status summary |
| `/detect` | Detect scenes in all videos |
| `/analyze` | Run all analysis on all clips |
| `/export` | Export the current sequence |
