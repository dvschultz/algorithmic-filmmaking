# Sequencer Algorithms

Scene Ripper includes 19 sequencer algorithms that arrange your clips into a sequence. Each algorithm uses a different creative logic to determine the order.

To use a sequencer, go to the **Sequence** tab, select an algorithm from the dropdown, and click **Generate**. Some algorithms require that you run specific analysis on your clips first (in the **Analyze** tab). Others open a dialog where you configure additional options before generating.

> Some algorithms (like Storyteller and Exquisite Corpus) require a cloud API key. See the [API Keys Guide](api-keys.md) for setup instructions.

---

## Sorting Algorithms

These algorithms arrange clips based on a measurable property. Most support a **direction** option that controls the sort order.

### Chromatics

Arrange clips along a color gradient or cycle through the spectrum.

**Required analysis:** Colors

**Direction options:**

| Direction | Description |
|-----------|-------------|
| Rainbow | Cycle through the full color spectrum (red, orange, yellow, green, blue, violet) |
| Warm to Cool | Start with warm tones (reds, oranges) and end with cool tones (blues, greens) |
| Cool to Warm | The reverse: cool tones first, warm tones last |
| Complementary | Alternate between complementary colors for maximum contrast |

Clips without color data can be appended to the end, excluded, or sorted inline depending on your settings.

### Tempo Shift

Order clips from shortest to longest (or reverse).

**Required analysis:** None

**Direction options:**

| Direction | Description |
|-----------|-------------|
| Shortest First | Start with the quickest cuts and build to longer takes |
| Longest First | Start with longer takes and accelerate toward shorter clips |

### Into the Dark

Arrange clips from light to shadow, or shadow to light.

**Required analysis:** Brightness

If clips haven't been analyzed for brightness yet, this algorithm will automatically compute it when you generate. This may take a moment for large clip sets.

**Direction options:**

| Direction | Description |
|-----------|-------------|
| Bright to Dark | Start with the brightest clips and descend into darkness |
| Dark to Bright | Emerge from shadow into light |

### Crescendo

Build from silence to thunder, or thunder to silence.

**Required analysis:** Volume

If clips haven't been analyzed for volume yet, this algorithm will automatically compute it when you generate. This uses FFmpeg to measure audio levels and may take a moment.

**Direction options:**

| Direction | Description |
|-----------|-------------|
| Quiet to Loud | Start with the quietest clips and build to the loudest |
| Loud to Quiet | Start loud and fade to quiet |

### Focal Ladder

Arrange clips by camera shot scale, from wide establishing shots to tight close-ups (or reverse).

**Required analysis:** Shots (shot type classification)

### Up Close and Personal

Glide from distant vistas to intimate close-ups.

**Required analysis:** Shots (shot type classification)

**Direction options:**

| Direction | Description |
|-----------|-------------|
| Far to Close | Start with wide/establishing shots and move to close-ups |
| Close to Far | Start with close-ups and pull back to wide shots |

### Gaze Sort

Arrange clips by where subjects are looking, from left to right or up to down.

**Required analysis:** Gaze direction

**Direction options:**

| Direction | Description |
|-----------|-------------|
| Left to Right | Start with subjects looking left and progress to subjects looking right |
| Right to Left | The reverse: right-looking clips first |
| Up to Down | Start with subjects looking up and progress to subjects looking down |
| Down to Up | The reverse: down-looking clips first |

Clips without gaze data (no face detected) are appended at the end of the sequence.

### Gaze Consistency

Group clips where subjects are looking in the same direction. All "looking left" clips are grouped together, then all "looking right", and so on — largest groups first.

**Required analysis:** Gaze direction

Within each group, clips are sorted by their actual gaze angle. Clips without gaze data are appended at the end.

---

## Relationship Algorithms

These algorithms find connections between clips based on visual similarity.

### Human Centipede

Chain clips together by visual similarity. Each clip is placed next to the one it most closely resembles, creating a continuous visual flow.

**Required analysis:** Embeddings (visual feature extraction via DINOv2)

If clips haven't been analyzed for embeddings yet, this algorithm will automatically compute them when you generate. Embedding extraction can take a while for large clip sets.

### Match Cut

Find hidden connections between clips at cut points. Analyzes the last frame of each clip and the first frame of the next to find the most visually similar transitions.

**Required analysis:** Boundary embeddings

If clips lack boundary embeddings, this algorithm will automatically compute them. This analyzes the first and last frames of each clip.

---

## Randomization

### Hatchet Job

Randomly shuffle clips into a new order. Opens a dialog where you can optionally apply random transforms to each clip.

**Required analysis:** None

**Dialog options:**
- **Random H-Flip:** Randomly mirror some clips horizontally
- **Random V-Flip:** Randomly flip some clips vertically
- **Random Reverse:** Randomly play some clips in reverse

When transforms are enabled, the dialog pre-renders each affected clip via FFmpeg before assembling the sequence. A progress bar shows rendering status.

### Time Capsule

Keep clips in their original order. This is the simplest algorithm: clips appear in the sequence exactly as they were detected, preserving the source video's timeline.

**Required analysis:** None

---

## Audio-Driven

### Staccato

Cut clips to the rhythm of a music track. Opens a dialog where you select an audio file, preview the waveform with beat markers, and generate a sequence where onset strength drives visual contrast — stronger beats trigger bigger visual jumps between consecutive clips.

**Required analysis:** Embeddings (DINOv2, auto-computed if missing)

**Dialog workflow:**
1. Click **Select Music File** to choose an MP3, WAV, FLAC, M4A, AAC, or OGG file
2. The audio is analyzed and a waveform is displayed with beat/onset markers overlaid
3. Adjust the **Sensitivity** slider to control the number of cut points ("Fewer Cuts" to "More Cuts")
4. Choose a **Beat Strategy** from the dropdown: Onsets (transients/hits), Beats (regular pulse), or Downbeats (strong beats only)
5. Click **Generate** to match clips to beat intervals. Each clip is trimmed to fit its slot; clips shorter than their slot are looped. Clips can repeat when there are more beat slots than clips.

The algorithm uses DINOv2 visual embeddings to measure similarity between clips. At each cut point, it measures the onset strength and selects a clip whose visual distance from the previous clip matches that strength — hard hits get jarring visual jumps, soft transitions get visually similar clips.

---

## AI-Powered Algorithms

These algorithms use language models or vision models to make creative decisions. They require a cloud API key — see the [API Keys Guide](api-keys.md).

### Exquisite Corpus

Generate a poem from on-screen text. Opens a multi-step dialog.

**Required analysis:** Text extraction (OCR/VLM)

**Dialog workflow:**
1. Enter a mood or vibe prompt (e.g., "melancholic and introspective") and select a poem length (Short, Medium, or Long)
2. The dialog extracts on-screen text from your clips using OCR or a vision-language model, showing a progress bar
3. An LLM arranges the extracted text into a poem, displayed as a drag-reorderable list where each line maps to a clip
4. You can **Regenerate** for a different poem or manually reorder lines before clicking **Create Sequence**

### Storyteller

Create a narrative from clip descriptions. Opens a multi-step dialog.

**Required analysis:** Describe (AI-generated clip descriptions)

All clips must have descriptions before using this algorithm. If some clips lack descriptions, the dialog will offer to exclude them or redirect you to the Analyze tab.

**Dialog workflow:**
1. Optionally enter a thematic focus and choose a narrative structure (Three-Act, Chronological, Thematic, or Auto)
2. Set a target duration (10 minutes to 90 minutes, or use all clips)
3. The LLM arranges clips into a narrative, assigning roles like "setup," "climax," and "resolution"
4. Review the narrative as a drag-reorderable list and adjust the order before confirming

### Reference Guide

Match your clips to a reference video's structure. Opens a dialog.

**Required analysis:** Varies by selected dimensions

**Dialog workflow:**
1. Select a source video as the "reference" — its clips become the template structure
2. All clips from other sources become the matching pool
3. Adjust seven dimension sliders to control matching weights: Color, Brightness, Shot Scale, Audio Energy, Visual Match (DINOv2), Movement, and Duration. Dimensions without analysis data are grayed out
4. Optionally allow repeated clips with the "Allow Repeats" checkbox
5. Click **Generate** to find the best-matching clip for each position in the reference

### Signature Style

Interpret a drawing as an editing guide. Opens a large canvas-based dialog.

**Required analysis:** Colors

**Dialog workflow:**
1. Draw on the canvas (or import an image). The Y-axis maps to pacing (spiky = fast cuts, smooth = slow) and color maps to color matching against your clips
2. Set a target duration and FPS
3. Choose between **Parametric** mode (pixel-level analysis with a granularity slider) or **VLM** mode (a vision model interprets the drawing's meaning)
4. Click **Generate** to match your drawing to clips and assemble a timed sequence

### Rose Hobart

Isolate clips featuring a specific person. Named after Joseph Cornell's 1936 found-footage film. Opens a dialog.

**Required analysis:** None (face detection runs in the dialog)

**Dialog workflow:**
1. Upload 1-3 reference photos of the person you want to find. Each image is analyzed for faces, and a green bounding box highlights the detected face
2. Configure matching sensitivity (Strict, Balanced, or Loose)
3. Choose how to order matched clips: Original, Duration, Color, Brightness, Confidence, or Random
4. Set a frame sample interval (how often to check for faces within each clip)
5. Click **Generate** to scan all clips for the matching face and build a sequence from the results

### Eyes Without a Face

Sequence clips based on where subjects are looking. Named after the Billy Idol song and the Georges Franju film. Opens a dialog with three modes.

**Required analysis:** Gaze direction

**Dialog modes:**

| Mode | What it does |
|------|-------------|
| **Eyeline Match** | Pair clips with complementary gaze directions for shot-reverse-shot patterns. If person A looks left, the next clip shows person B looking right. Tolerance slider controls how strict the matching is (5°-30°). |
| **Filter** | Keep only clips where subjects look in a specific direction (at camera, left, right, up, or down). Non-matching clips are appended at the end. |
| **Rotation** | Sweep through a range of gaze angles, creating a progressive rotation effect. Select the axis (horizontal or vertical), set a start and end angle, and choose ascending or descending direction. |

**Dialog workflow:**
1. Select a mode from the dropdown at the top
2. Configure the mode-specific parameters (tolerance, category, or angle range)
3. Click **Generate** to build the sequence
4. Results appear on the timeline

Clips without gaze data are always appended at the end of the sequence.
