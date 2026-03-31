# Chat Agent

Scene Ripper includes an integrated chat agent that can control the app on your behalf. It can import videos, detect scenes, run analysis, build sequences, and export results -- all from natural language prompts.

Open the agent panel with **View > Show Agent Chat** (or Ctrl+Shift+C).

> The chat agent requires an LLM provider. Set up at least one API key in **Settings > API Keys** (Anthropic, OpenAI, Gemini, or OpenRouter), or use a local model via Ollama. See the [API Keys Guide](api-keys.md) for setup instructions.

---

## What the Agent Can Do

### Import

- Search YouTube and Internet Archive for videos
- Download videos from URLs
- List, inspect, and remove source videos in your project

**Example prompts:**
- "Search YouTube for time-lapse city videos"
- "Download this video: https://www.youtube.com/watch?v=..."
- "List all sources in the project"

### Detect Scenes

- Run scene detection on individual videos or all unanalyzed videos at once
- Check detection progress

**Example prompts:**
- "Detect scenes in all videos"
- "How many clips do we have?"

### Analyze

- Run any combination of analysis operations on your clips: colors, shot types, transcription, descriptions, object detection, face detection, text extraction, and cinematography
- Analyze all clips or just a selection

**Example prompts:**
- "Describe all the clips"
- "Run shot classification and color analysis on everything"
- "Transcribe the clips from the first video"
- "What objects appear in clip 3?"

### Sequence

- Add clips to the timeline
- Remove, reorder, and trim clips in the sequence
- Clear the sequence
- Apply sorting algorithms

**Example prompts:**
- "Add all clips to the sequence"
- "Reorder the sequence by brightness, darkest first"
- "Remove clips shorter than 1 second"
- "Clear the sequence"

### Export

- Export the sequence as MP4
- Export as EDL for external editors
- Export individual clips

**Example prompts:**
- "Export the sequence as MP4"
- "Export an EDL file"
- "Export all clips individually"

### Navigation and Playback

- Switch between tabs
- Play, pause, and seek through clips
- Step frame by frame
- Filter and sort clips in the browser

**Example prompts:**
- "Go to the Analyze tab"
- "Play clip 5"
- "Show me only close-up shots"

---

## Slash Commands

Type these in the chat input for quick actions:

| Command | What It Does |
|---------|-------------|
| `/help` | Show all available agent tools |
| `/status` | Show project status (sources, clips, analysis coverage, sequence) |
| `/detect` | Detect scenes in all unanalyzed videos |
| `/analyze` | Run all available analysis on all clips |
| `/export` | Export the current sequence as MP4 |

`/status` displays project info directly without calling the LLM. The other shortcut commands send a pre-written prompt to the agent.

---

## How Plans Work

For complex multi-step workflows, the agent proposes a plan before executing. This gives you a chance to review, edit, or cancel before any work begins.

1. You describe what you want ("Download this video, detect scenes, describe the clips, then export")
2. The agent responds with a numbered plan showing each step
3. You can **Confirm** to execute, **Cancel** to abort, or modify the plan
4. During execution, each step shows its status (running, complete, or failed)
5. If a step fails, you can **Retry** that step or **Stop** the plan

Plans are proposed automatically when the agent determines that a task requires multiple distinct operations.

---

## Local vs Cloud Features

Most agent capabilities work regardless of which provider you choose for the chat. However, the analysis operations the agent triggers follow the same local/cloud rules as the rest of the app:

| Feature | Runs Locally | Requires API Key |
|---------|-------------|-----------------|
| Chat agent itself | With Ollama | Yes (any LLM provider) |
| Color analysis | Yes | No |
| Shot classification | Yes (SigLIP 2) | Optional (cloud tier) |
| Object detection | Yes (YOLO) | No |
| Face detection | Yes (InsightFace) | No |
| Transcription | Yes (Whisper) | Optional (Groq cloud) |
| Content classification | Yes (MobileNet) | No |
| Describe clips | Yes (Qwen3-VL / Moondream) | Optional (cloud tier) |
| Text extraction (OCR) | Yes (PaddleOCR) | Optional (VLM fallback) |
| Cinematography analysis | Yes (Apple Silicon) | Optional (cloud tier) |
| Storyteller / Exquisite Corpus | No | Yes |
| Scene detection | Yes | No |
| Export | Yes | No |

See the [Local Models Guide](local-models.md) for details on which models run on your machine.

---

## Tips for Better Results

**Be specific about scope.** "Describe all clips" processes everything. "Describe the clips from sunset.mp4" processes only that source. The agent respects specificity.

**Name the analysis you want.** Instead of "analyze everything," try "run colors and shot classification." This avoids running expensive analyses you may not need.

**Check status first.** Use `/status` to see what has already been analyzed before requesting more work. The agent skips clips that already have the requested data.

**Use the right provider for the job.** Gemini and Claude tend to produce better plans and descriptions. Ollama works offline but may struggle with complex multi-step workflows.

**Let the agent finish.** The cancel button stops the current operation, but some background workers (like scene detection) take a moment to wind down. Wait for the "Cancelled" confirmation before sending a new message.
