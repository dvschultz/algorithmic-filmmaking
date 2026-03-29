# Analysis

Scene Ripper can analyze your clips to extract metadata that powers sequencer algorithms, search, filtering, and the chat agent. All analysis runs in the **Analyze** tab.

After detecting scenes in the **Cut** tab, switch to the **Analyze** tab. Select an analysis from the dropdown and click **Run**, or click **Analyze...** to run multiple analyses in one batch.

> Most analyses run locally on your machine. Some offer a cloud option that uses an API key for better results. See the [API Keys Guide](api-keys.md) for setup. For details on the AI models used, see the [Local Models Guide](local-models.md).

---

## Running Analyses

There are three ways to run analyses:

**Quick Run** — Select a single analysis from the dropdown at the top of the Analyze tab and click Run. This processes all clips (or only selected clips if you have a selection).

**Batch Run (Analyze...)** — Click the Analyze... button to open a picker where you can check multiple analyses to run together. The app automatically groups them into phases and runs them in the optimal order.

**Chat Agent** — Ask the chat agent to run analyses on your clips. For example: "Describe all the clips" or "Run shot classification on the selected clips."

Analyses that have already been completed for a clip are skipped automatically. To re-run an analysis, clear the existing data first.

---

## Color Analysis

Samples three frames from each clip (at 15%, 50%, and 85% through its duration) and extracts a palette of five dominant colors using k-means clustering. Also computes average brightness and color classification (warm, cool, neutral, or vibrant).

**Produces:**
- Dominant color palette (5 colors)
- Average brightness (0.0-1.0)
- Color purity score

**Required first:** Scene detection
**Runs:** Locally (no API key needed)
**Speed:** Fast (processes multiple clips in parallel)

**Used by:** Chromatics, Into the Dark, and other sorting algorithms that arrange clips by visual properties.

---

## Shot Classification

Classifies each clip into one of five standard shot types based on its thumbnail.

| Shot Type | Description |
|-----------|-------------|
| Wide Shot | Shows the full environment with subjects small in frame |
| Full Shot | Subjects visible head to toe |
| Medium Shot | Subjects framed from roughly the waist up |
| Close-Up | Head and shoulders, face fills most of the frame |
| Extreme Close-Up | A detail — eyes, hands, a single object |

**Produces:**
- Shot type label
- Confidence score (0.0-1.0)

**Required first:** Scene detection (thumbnails must exist)
**Runs:** Locally by default (SigLIP 2), or cloud (Gemini Flash Lite)
**Speed:** Fast (~50ms per clip locally)

**Used by:** Focal Ladder sequencer, filtering by shot type in the Sequence tab.

**Settings (Settings > Models > Shot Type Classification):**
- **Tier** — Local (SigLIP 2, free) or Cloud (Gemini, requires API key)

---

## Content Classification

Tags each clip's thumbnail with general content labels from ImageNet's 1,000 categories (e.g., "beach", "concert hall", "golden retriever").

**Produces:**
- Up to 5 content labels with confidence scores

**Required first:** Scene detection (thumbnails must exist)
**Runs:** Locally (MobileNetV3, no API key needed)
**Speed:** Fast

---

## Object Detection

Finds and labels specific objects in each clip's thumbnail, including bounding box positions.

Two detection modes are available:

- **Fixed vocabulary** (default) — Detects 80 common object categories (people, vehicles, animals, furniture, etc.) using YOLO26n. Fast and reliable.
- **Open vocabulary** — Detects custom object types you define (e.g., "microphone", "guitar", "protest sign") using YOLOE-26s. Slower but flexible.

**Produces:**
- List of detected objects with labels, confidence scores, and bounding boxes
- Person count

**Required first:** Scene detection (thumbnails must exist)
**Runs:** Locally (no API key needed)
**Speed:** Fast

**Settings (Settings > Models):**
- **Detection mode** — Fixed (80 classes) or Open Vocabulary (custom classes)
- **Custom classes** — Comma-separated list of object types for open vocabulary mode

---

## Text Extraction (OCR)

Reads on-screen text from each clip — titles, signs, subtitles, watermarks, lower thirds, and any other visible text. Samples multiple keyframes from each clip to catch text that appears at different times.

Three extraction methods are available:

- **Hybrid** (default) — Runs PaddleOCR first (fast, local). If confidence is low or no text is found, falls back to a cloud VLM for a second look.
- **PaddleOCR only** — Local only, no cloud fallback. Fastest option.
- **VLM only** — Uses a cloud vision model. Better at reading stylized or distorted text but requires an API key.

**Produces:**
- List of extracted text strings with confidence scores and frame numbers

**Required first:** Scene detection (source video must exist)
**Runs:** Locally (PaddleOCR) and/or cloud (Gemini VLM), depending on method
**Speed:** Moderate

**Settings (Settings > Models > Text Extraction):**
- **Method** — Hybrid, PaddleOCR, or VLM
- **VLM model** — Which cloud model to use for VLM fallback

---

## Transcription

Transcribes spoken dialogue and narration from your clips using Whisper. Produces timestamped transcript segments.

Scene Ripper offers multiple Whisper model sizes. Larger models are more accurate but slower:

| Model | Download Size | Best For |
|-------|--------------|----------|
| tiny.en | 39 MB | Quick previews, clear speech |
| small.en | 244 MB | Everyday use |
| medium.en | 769 MB | Default — good balance of speed and accuracy |
| large-v3 | 1.5 GB | Maximum accuracy, accented or noisy audio |

On Apple Silicon Macs, transcription runs on the GPU via MLX for significantly faster results. On other platforms, it runs on the CPU via faster-whisper.

**Produces:**
- Timestamped transcript segments (start time, end time, text)

**Required first:** Scene detection (source video must have audio)
**Runs:** Locally by default, or cloud via Groq (requires API key)
**Speed:** Varies by model size. medium.en typically processes faster than real-time on Apple Silicon.

**Settings (Settings > Models > Transcription):**
- **Model** — Whisper model size
- **Language** — English, Auto-detect, or a specific language code
- **Backend** — Auto (recommended), faster-whisper, mlx-whisper, or Groq (cloud)

**Used by:** Storyteller sequencer, transcript search, SRT export.

---

## Describe

Generates a natural language description of each clip — what subjects are visible, what action is happening, and what the setting looks like.

**Produces:**
- 2-3 sentence prose description
- Which model was used

**Required first:** Scene detection (thumbnails must exist)
**Runs:** Locally (Qwen3-VL on Apple Silicon, Moondream2 on other platforms) or cloud (Gemini, Claude, GPT-4o)
**Speed:** Local is slower (1-3 seconds per clip). Cloud is faster but requires an API key.

Two input modes are available for cloud descriptions:
- **Frame** (default) — Sends a single thumbnail. Fast and cheap.
- **Video** — Extracts the actual video clip and sends it to the API. Captures motion and temporal information that a single frame misses. Only supported with Gemini.

**Settings (Settings > Models > Vision Description):**
- **Tier** — Local (free) or Cloud (requires API key)
- **Local model** — Qwen3-VL-4B (Apple Silicon) or Moondream2 (CPU fallback)
- **Cloud model** — Gemini, Claude, or GPT-4o variant
- **Input mode** — Frame or Video (cloud only, Gemini only)

**Used by:** Chat agent search, Storyteller sequencer, clip filtering.

---

## Rich Analysis (Cinematography)

The most comprehensive analysis. Evaluates each clip across 20+ cinematography dimensions, producing structured data about camera work, composition, lighting, and emotional tone.

**Produces:**

| Category | Fields |
|----------|--------|
| **Camera** | Shot size (10 types from ELS to ECU), camera angle, camera movement, movement direction, dutch tilt, estimated lens type |
| **Composition** | Subject position (rule of thirds), headroom, lead room, balance, subject count and type |
| **Focus** | Depth of field (deep/shallow/rack focus), background type |
| **Lighting** | Lighting style (high-key/low-key/natural/dramatic), direction, quality, color temperature |
| **Mood** | Emotional intensity, suggested pacing |

**Required first:** Scene detection (thumbnails must exist)
**Runs:** Cloud (Gemini or other VLM) or locally (Qwen2.5-VL-7B on Apple Silicon)
**Speed:** Slowest analysis (1-3 seconds per clip cloud, 5-10 seconds local). Best run as a batch.

Two input modes are available:
- **Frame** — Single thumbnail. Cannot detect camera movement (static frames don't show motion).
- **Video** — Extracts the actual clip and sends it to the API. Full movement detection. Only supported with Gemini.

**Settings (Settings > Models):**
- **Tier** — Cloud or Local
- **Cloud model** — Gemini (recommended for video mode), Claude, or GPT-4o
- **Local model** — Qwen2.5-VL-7B (Apple Silicon only, ~4 GB)
- **Input mode** — Frame or Video

**Used by:** Focal Ladder (uses the detailed shot size), clip filtering by any cinematography field.

> **Tip:** Rich Analysis provides a more detailed shot size classification (10 types) than Shot Classification (5 types). If you run both, the Rich Analysis values take precedence for sequencer algorithms that use shot size.

---

## Face Detection

Detects faces in your clips and generates a unique embedding for each face. These embeddings allow the app to recognize the same person across different clips.

The detector samples one frame per second through each clip, so it catches faces that appear briefly or only partway through a clip.

**Produces:**
- Face bounding boxes and confidence scores
- 512-dimensional face embedding per detected face (for cross-clip person matching)

**Required first:** Scene detection (source video must exist)
**Runs:** Locally (InsightFace, no API key needed)
**Speed:** Moderate (processes sequentially to manage memory)

**Used by:** Rose Hobart sequencer (filters clips to only those containing a specific person).

> **Note:** Face detection requires a C/C++ compiler for initial installation of the InsightFace library (Xcode Command Line Tools on macOS).

---

## Custom Query

Ask a yes/no question about each clip and get a boolean answer with confidence. Useful for filtering clips by arbitrary criteria that other analyses don't cover.

Examples:
- "Is there a dog in this scene?"
- "Is this shot indoors?"
- "Does this clip contain text on screen?"

**Produces:**
- Boolean match (yes/no)
- Confidence score (0.0-1.0)

**Required first:** Scene detection (thumbnails must exist)
**Runs:** Uses the same tier as Describe (local VLM or cloud)
**Speed:** Same as Describe

---

## Recommended Order

If you're running all analyses, the app handles ordering automatically when using the batch picker. But if you're running them one at a time, here's a good order:

1. **Colors** — fastest, enables color-based sequencer algorithms immediately
2. **Shot Classification** — fast, useful for filtering and Focal Ladder
3. **Transcription** — if your clips have speech
4. **Describe** — adds searchable text descriptions
5. **Rich Analysis** — most expensive but richest metadata
6. **Object Detection, Face Detection, Text Extraction** — as needed for your project

You don't need to run every analysis. Start with what your project needs. Colors and Shot Classification cover most sequencer requirements. Add Describe if you want to search clips by content. Add Rich Analysis if you need detailed cinematography metadata.
