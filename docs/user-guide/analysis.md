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

<details>
<summary><strong>Prompts used</strong></summary>

**Local (SigLIP 2)** — SigLIP is a contrastive model, not a generative one, so it doesn't use a text prompt in the traditional sense. Instead, each thumbnail is scored against an ensemble of text descriptions per shot type:

| Shot Type | Text Labels |
|-----------|-------------|
| Wide Shot | "This is a photo of an establishing shot showing a vast landscape or cityscape." / "...a long shot where people appear very small in the environment." / "...a wide angle shot of a large space with tiny distant figures." / "...a panoramic view showing the entire location." |
| Full Shot | "...a shot showing one person's entire body from head to feet." / "...a single person standing with their full body visible in frame." / "...a full length portrait of someone from head to toe." / "...a shot framing one standing figure completely." |
| Medium Shot | "...a medium shot showing a person from the waist up to their head." / "...two or three people shown from the waist up in conversation." / "...a shot of people sitting at a table showing their upper bodies." / "...a cowboy shot showing someone from mid-thigh to head." |
| Close-Up | "...a close-up of a person's face filling most of the frame." / "...a head and shoulders shot focusing on facial expression." / "...a tight shot of someone's face showing emotion." / "...a portrait shot from the neck up." |
| Extreme Close-Up | "...an extreme close-up showing only eyes filling the screen." / "...a shot of just lips or mouth in extreme detail." / "...a macro shot of a single facial feature like an eye." / "...an intense close-up where only part of a face is visible." |

The model picks the shot type whose labels collectively score highest against the image.

**Cloud (Gemini)** — sends this prompt with the thumbnail:

```
Classify this film frame into exactly one shot type.
Valid types: "wide shot", "full shot", "medium shot", "close-up", "extreme close-up".

Return ONLY a JSON object: {"shot_type": "<type>", "confidence": <0.0-1.0>}
```

</details>

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

<details>
<summary><strong>Prompt used (VLM fallback)</strong></summary>

When PaddleOCR returns low-confidence results or no text, the VLM fallback sends this prompt with a frame from the clip:

```
Extract ALL visible text from this image. Include:
- Signs, labels, titles
- Subtitles or captions
- Text on documents or screens
- Any other readable text

Return ONLY the extracted text, one phrase per line. If no text is visible, return "NO_TEXT_FOUND".
Do not add any commentary or descriptions.
```

PaddleOCR itself uses no prompt — it's a dedicated OCR model that detects and reads text directly.

</details>

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

**Used by:** Cassette Tape, Reference Guide's Transcript dimension, transcript search, SRT export.

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

**Used by:** Chat agent search, Storyteller, Reference Guide's Description dimension, Free Association, clip filtering.

<details>
<summary><strong>Prompt used</strong></summary>

The description prompt sent with each thumbnail (or video clip):

```
Describe this video frame in 3 sentences or less.
Focus on the main subjects, action, and setting.
```

When using video input mode (Gemini only), a slightly different version is sent with the extracted video clip:

```
Describe this video clip in 3 sentences or less.
Focus on the main subjects, action, and setting.
```

</details>

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

**Used by:** Focal Ladder and Up Close and Personal (detailed shot size), Reference Guide's Shot Scale and Movement dimensions, clip filtering by any cinematography field.

> **Tip:** Rich Analysis provides a more detailed shot size classification (10 types) than Shot Classification (5 types). If you run both, the Rich Analysis values take precedence for sequencer algorithms that use shot size.

<details>
<summary><strong>Prompt used (frame mode)</strong></summary>

```
Analyze this film frame using professional cinematography terminology.

Evaluate each dimension carefully:

**Shot Size** (distance from subject):
- ELS (Extreme Long Shot): Vast environment, people tiny/absent
- VLS (Very Long Shot): Full environment with visible people
- LS (Long Shot): Head to toe, full body
- MLS (Medium Long Shot): 3/4 body (knees up)
- MS (Medium Shot): Waist to head
- MCU (Medium Close-Up): Head and shoulders
- CU (Close-Up): Face fills frame
- BCU (Big Close-Up): Face, partial features
- ECU (Extreme Close-Up): Single feature (eyes, lips)
- Insert: Object detail shot

**Camera Angle** (position relative to subject):
- low_angle: Below subject looking up (power, heroism)
- eye_level: Subject's eye height (neutral, equal)
- high_angle: Above subject looking down (vulnerability)
- dutch_angle: Tilted horizon (disorientation, unease)
- birds_eye: Directly above (omniscience, pattern)
- worms_eye: Directly below (extreme power)

**Composition**:
- Subject position: left_third, center, right_third, or distributed
- Headroom: tight, normal, excessive, or n/a
- Lead room: tight, normal, excessive, or n/a (space in direction of gaze)
- Balance: balanced, left_heavy, right_heavy, or symmetrical

**Subject Analysis**:
- Count: empty, single, two_shot (2 people), or group (3+)
- Type: person, object, landscape, text, or mixed

**Focus & Depth**:
- Focus type: deep (all sharp), shallow (subject isolated), or rack_focus
- Background: blurred, sharp, cluttered, or plain

**Lighting**:
- Style: high_key (bright, few shadows), low_key (dark, heavy shadows), natural, or dramatic
- Direction: front, three_quarter, side, back, or below
- Quality: hard (sharp shadow edges), soft (diffused, gradual shadows), mixed, or unknown
- Color temperature: warm (orange/yellow tones), neutral, cool (blue tones), or unknown

**Dutch Tilt** (horizon angle):
- none: Horizon is level
- slight: 5-15° tilt
- moderate: 15-30° tilt
- extreme: 30°+ tilt
- unknown: Cannot determine

**Camera Position** (relative to subject facing):
- frontal: Camera faces subject directly, face fully visible
- three_quarter: 45° angle to subject
- profile: 90° side view
- back: Behind subject
- unknown: No clear subject or ambiguous

**Lens Type** (estimate from visual characteristics):
- wide: Exaggerated perspective, barrel distortion, deep depth of field
- normal: Natural perspective matching human eye
- telephoto: Compressed perspective, shallow depth of field
- unknown: Cannot determine

**Derived Properties**:
- Emotional intensity: low, medium, or high (based on shot size + angle + lighting)
- Suggested pacing: fast (simple shots), medium, or slow (complex/wide shots)

Since this is a single frame, set camera_movement to "n/a".

Return your analysis as a JSON object with these exact field names.
```

</details>

<details>
<summary><strong>Prompt used (video mode)</strong></summary>

The video mode prompt is identical to the frame mode prompt except it replaces "film frame" with "video clip", removes the instruction to set `camera_movement` to "n/a", and adds a camera movement section:

```
**Camera Movement** (watch for motion throughout the clip):
- static: No camera movement
- pan: Horizontal rotation (left/right)
- tilt: Vertical rotation (up/down)
- track: Camera moves through space (dolly/traveling)
- handheld: Unstable, organic movement
- crane: Vertical spatial movement
- arc: Circling around subject

If movement is detected, also note the direction:
left, right, up, down, forward, backward, clockwise, counterclockwise
```

</details>

---

## Gaze Direction

Estimates where subjects are looking in each clip using MediaPipe Face Mesh iris tracking. Detects the iris position relative to eye corners and converts it to angular gaze direction.

The detector samples one frame per second through each clip and uses a dominant-category approach: each frame is categorized independently, then the most frequent category across all frames becomes the clip's gaze label. This prevents a brief glance from overriding the primary gaze direction.

**Produces:**
- Gaze yaw angle (horizontal, in degrees — positive = looking right)
- Gaze pitch angle (vertical, in degrees — positive = looking down)
- Categorical label: at camera, looking left, looking right, looking up, or looking down

**Required first:** Scene detection (source video must exist)
**Runs:** Locally (MediaPipe, no API key needed). Downloads a ~4 MB model on first use.
**Speed:** Moderate (processes sequentially)

**Used by:** Gaze Sort, Gaze Consistency, and Eyes Without a Face sequencer algorithms. Also available as a filter in the Sequence tab.

> **Note:** Gaze estimation works best when the subject's face is relatively frontal. Reliability decreases for strongly turned heads (beyond ~30° from center). When both horizontal and vertical gaze exceed their thresholds, horizontal gaze takes priority for categorization.

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

**Used by:** Rose Hobart sequencer and agent tool. The GUI can compute missing face embeddings inside the Rose Hobart dialog and cache them on clips.

> **Note:** Face detection requires a C/C++ compiler for initial installation of the InsightFace library (Xcode Command Line Tools on macOS).

---

## Generate Embeddings

Extracts a 768-dimensional DINOv2 visual feature vector from each clip's thumbnail. Embeddings capture what each clip "looks like" in a way that lets algorithms compare clips by visual similarity rather than by discrete metadata like shot type or color.

**Produces:**
- A 768-dim embedding vector on each clip (`embedding`)
- The model tag (`dinov2-vit-b-14`) so you can tell which version generated the vector

**Required first:** Scene detection with thumbnails (the default). Clips without thumbnails are skipped.
**Runs:** Locally (DINOv2 via transformers, no API key needed). Downloads ~450 MB of model files on first use.
**Speed:** Moderate (processes in batches on GPU if available, otherwise CPU).

**Used by:** Human Centipede (similarity chaining), Staccato, Reference Guide's Visual Match dimension, and Free Association candidate shortlisting. The same DINOv2 runtime also powers Match Cut's first/last-frame boundary embeddings.

> **Tip:** If you haven't run this explicitly and try to use a sequencer that needs embeddings, the app will auto-compute them as a side effect of running the sequencer. Running Generate Embeddings up front is faster when you plan to use multiple embedding-based sequencers on the same clips.

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
6. **Object Detection, Face Detection, Gaze Direction, Text Extraction** — as needed for your project

You don't need to run every analysis. Start with what your project needs. Colors and Shot Classification cover most sequencer requirements. Add Describe if you want to search clips by content. Add Rich Analysis if you need detailed cinematography metadata.
