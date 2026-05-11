# Sequencer Algorithms

Scene Ripper includes 23 sequencer algorithms that arrange your clips into a sequence. Each algorithm uses a different creative logic to determine the order, choose matches, or trim clips into timed slots.

To use a sequencer, go to the **Sequence** tab, select an algorithm from the dropdown, and click **Generate**. Some algorithms require that you run specific analysis on your clips first (in the **Analyze** tab). Others open a dialog where you configure additional options before generating.

> Some algorithms, modes, and agent workflows require an LLM or VLM API key. Storyteller, Exquisite Corpus, Free Association, and Signature Style's VLM mode are the main examples. See the [API Keys Guide](api-keys.md) for setup instructions.

Most sequencers use the clips currently selected in the Cut or Analyze tab. If no clips are selected, select the clips you want to sequence first. Dialog-based sequencers may add their own setup steps, such as choosing a music track, reference video, phrase list, drawing, or face reference.

## Timing and Trimming

Most sequencers place each matched clip at its full detected clip length. They change order and selection, but they do not retime the source material.

The main exceptions are:

| Sequencer | Timing behavior |
|-----------|-----------------|
| **Staccato** | Trims or loops clips so each beat slot is filled |
| **Signature Style** | Trims longer clips to the drawing segment duration; shorter clips play full length |
| **Cassette Tape** | Trims clips to the matched transcript segment |

Reference Guide uses duration as an optional matching dimension, but it does **not** trim a matched clip to the guide clip's length. If you enable Duration, it prefers clips with similar lengths; the output still uses the matched clip's own length.

---

## Algorithm Index

| Algorithm | Category | Requires first | Trims output? |
|-----------|----------|----------------|---------------|
| Chromatics | Arrange | Colors | No |
| Tempo Shift | Arrange | None | No |
| Into the Dark | Arrange | Brightness, auto-computed if missing | No |
| Crescendo | Arrange / Audio | Volume, auto-computed if missing | No |
| Focal Ladder | Arrange | Shot classification | No |
| Up Close and Personal | Arrange | Shot classification or Rich Analysis shot size | No |
| Gaze Sort | Arrange / Find | Gaze analysis | No |
| Gaze Consistency | Find | Gaze analysis | No |
| Hatchet Job | Arrange | None | No |
| Time Capsule | Arrange | None | No |
| Human Centipede | Connect | Visual embeddings, auto-computed if missing | No |
| Match Cut | Connect | Boundary embeddings, auto-computed if missing | No |
| Staccato | Audio | Audio source and visual embeddings | Yes |
| Exquisite Corpus | Text | Extract Text plus an LLM | No |
| Storyteller | Text | Describe plus an LLM | No |
| Reference Guide | Connect / Audio / Text | Depends on enabled dimensions | No |
| Signature Style | Connect | Colors; VLM mode benefits from shots/descriptions | Yes |
| Rose Hobart | Find | Reference face image; face dependencies install on demand | No |
| Eyes Without a Face | Find / Connect | Gaze analysis | No |
| Free Association | Connect / Text | Describe and embeddings plus an LLM | No |
| Cassette Tape | Text / Audio | Transcribe | Yes |
| Word Sequencer | Text | Transcribe and word-level alignment | Yes |
| LLM Word Composer | Text | Transcribe, word-level alignment, and a local LLM (Ollama) | Yes |

## Managing Multiple Sequences

A project holds any number of named sequences — run 23 different algorithms and keep every result side by side for comparison. Every algorithm run creates a new sequence instead of overwriting the previous one.

### Sequence dropdown

The leftmost dropdown in the Sequence tab header (and in the card view) shows all sequences in the project. Click a name to switch. The timeline, algorithm dropdown, and chromatic bar checkbox all update to reflect the selected sequence.

### New Sequence button

Click **New Sequence** to create a blank "Untitled Sequence" and switch to it. When the current sequence is empty, running an algorithm reuses it (rename + populate) instead of creating a second empty entry — avoiding orphan sequences.

### Unsaved-edits prompt

If you manually drag or remove clips on the timeline and then switch sequences, Scene Ripper shows a **Save / Discard / Cancel** dialog so you don't lose edits. Algorithm runs don't trigger this prompt — they're always clean by design.

### Re-run prompt (parameter tweaks)

Changing the algorithm or direction dropdown on a populated sequence shows a **Replace / Create New / Cancel** dialog:

| Choice | Effect |
|--------|--------|
| **Replace** | The current sequence is overwritten with the new algorithm result |
| **Create New** | The new algorithm run adds a new sequence alongside the current one |
| **Cancel** | No change |

### Rename and delete

Right-click the sequence dropdown to open a context menu with **Rename…** and **Delete** options. Deleting an empty sequence is immediate; deleting a populated one asks for confirmation. You can't delete the last sequence — Scene Ripper always keeps at least one (a fresh empty "Untitled Sequence" is auto-created if you try).

Duplicate names are allowed; auto-naming uses a monotonic counter per algorithm (Chromatics, Chromatics #2, Chromatics #3, …).

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

Arrange clips by camera shot scale, from wide establishing shots to tight close-ups.

**Required analysis:** Shots (shot type classification)

Focal Ladder has no direction dropdown in the current UI. Use **Up Close and Personal** when you want an explicit far-to-close or close-to-far direction.

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

The shuffle uses a constrained shuffle that avoids placing clips from the same source back-to-back when possible. When transforms are enabled, the dialog pre-renders each affected clip via FFmpeg before assembling the sequence. A progress bar shows rendering status. Reverse pre-rendering is capped for long clips to keep render time bounded.

### Time Capsule

Keep clips in their original order. This is the simplest algorithm: clips appear in the sequence exactly as they were detected, preserving the source video's timeline.

**Required analysis:** None

---

## Audio-Driven

### Staccato

Cut clips to the rhythm of a music track. Opens a dialog where you pick a project audio source, preview the waveform with beat markers, and generate a sequence where onset strength drives visual contrast — stronger beats trigger bigger visual jumps between consecutive clips.

**Required analysis:** Audio source, Embeddings (DINOv2, auto-computed in the dialog if dependencies and thumbnails are available)

**Dialog workflow:**
1. Pick an audio source from the **Audio source** dropdown. The dropdown is populated from your project's [audio library](audio-sources.md) — if you haven't imported audio yet, choose **Import new…** to import one without leaving the dialog.
2. The audio is analyzed and a waveform is displayed with beat/onset markers overlaid
3. Adjust the **Sensitivity** slider to control the number of cut points ("Fewer Cuts" to "More Cuts")
4. Choose a **Beat Strategy** from the dropdown: Onsets (transients/hits), Beats (regular pulse), or Downbeats (strong beats only)
5. Optional advanced onset controls let you choose an onset profile, minimum gap, timing mode, and analysis resolution when transient detection needs tuning
6. Click **Generate** to match clips to beat intervals. Each clip is trimmed to fit its slot; clips shorter than their slot are looped. Clips can repeat when there are more beat slots than clips.

The algorithm uses DINOv2 visual embeddings to measure similarity between clips. At each cut point, it measures the onset strength and selects a clip whose visual distance from the previous clip matches that strength — hard hits get jarring visual jumps, soft transitions get visually similar clips.

Staccato exhausts the available clip pool before repeating clips. The resulting timeline stores the music path on the sequence and should closely match the generated beat-slot timing.

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
3. Adjust dimension sliders to control matching weights: Color, Brightness, Shot Scale, Audio Energy, Visual Match (DINOv2), Description, Transcript, Movement, and Duration. Dimensions without analysis data are grayed out
4. Optionally allow repeated clips with the "Allow Repeats" checkbox
5. Click **Generate** to find the best-matching clip for each position in the reference

**Matching details:**

| Dimension | Data used | Match behavior |
|-----------|-----------|----------------|
| Color | Dominant color hue | Numeric distance after normalization |
| Brightness | Average luminance | Numeric distance after normalization |
| Shot Scale | Rich Analysis shot size, falling back to shot classification | Wide-to-close proximity distance |
| Audio Energy | RMS volume | Numeric distance after normalization |
| Visual Match | DINOv2 embedding | Cosine distance |
| Description | Describe text | Token-frequency cosine distance |
| Transcript | Transcribed speech text | Token-frequency cosine distance |
| Movement | Rich Analysis camera movement | Exact categorical match |
| Duration | Clip duration | Numeric distance after normalization |

Reference Guide matches greedily from the first reference clip to the last. With **Allow Repeats** off, each candidate clip can be used once; if the matching pool runs out, later reference positions stay unmatched. With **Allow Repeats** on, the same candidate can fill multiple reference positions.

Duration is only a matching signal. Matched clips are not trimmed to the guide clip duration.

### Signature Style

Interpret a drawing as an editing guide. Opens a large canvas-based dialog.

**Required analysis:** Colors

**Dialog workflow:**
1. Draw on the canvas (or import an image). In Parametric mode, the drawing is sampled left-to-right: the line's vertical changes map to pacing (spiky = fast cuts, smooth = slow), and ink color maps to color matching against your clips
2. Set a target duration and FPS
3. Choose between **Parametric** mode (pixel-level analysis with a granularity slider) or **VLM** mode (a vision model interprets the drawing's meaning)
4. Click **Generate** to match your drawing to clips and assemble a timed sequence

The matcher can reuse clips. Longer clips are center-trimmed to the target segment duration; clips shorter than a segment play at full length. Parametric mode mainly uses duration, color, brightness, and available shot metadata. VLM mode adds interpreted shot type, brightness, and energy targets when the vision model can infer them.

### Rose Hobart

Isolate clips featuring a specific person. Named after Joseph Cornell's 1936 found-footage film. Opens a dialog.

**Required analysis:** None required ahead of time in the GUI. Face detection runs in the dialog and caches results on clips. The agent tool uses already computed face embeddings.

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

### Free Association

Build a sequence one clip at a time with an LLM collaborator. The algorithm opens a dialog where you interactively accept, reject, or swap each proposed next clip, with a rationale from the LLM explaining each transition.

**Required analysis:** Describe, Embeddings, LLM provider

**Dialog workflow:**

1. Pick a starting clip (the LLM won't choose the first one)
2. The dialog proposes the next clip with a short rationale (why it pairs well with the current tail of the sequence)
3. Choose: **Accept** (add to sequence), **Reject** (ask for a different proposal), **Swap** (see alternatives), or **Stop** (finish)
4. Repeat until you're satisfied or the pool is exhausted

Embeddings power a local candidate shortlist so the LLM only sees the most similar clips at each step — this keeps prompts small and responses fast. Without embeddings the algorithm falls back to random sampling, which degrades proposal quality. Each accepted transition is saved as a rationale on the resulting SequenceClip, visible in exported SRTs.

### Cassette Tape

Find clips that say specific phrases — the transcript-driven mixtape. This is a quote-finder: you supply a list of phrases and Cassette Tape pulls the closest matches from your transcribed clips, returning a sequence of sub-clips trimmed to just the lines that matched.

**Required analysis:** Transcribe (speech transcription with per-segment timing)

**Dialog workflow:**

1. **Setup.** Enter one phrase per row. The slider next to each phrase sets how many matches to pull (1–5). Add or remove rows as needed. Click **Find Matches**.
2. **Review.** Cassette Tape scores every transcript segment in the project against each phrase and groups the top matches under their phrase. Each match shows the source clip, the matched transcript snippet (with the matching words highlighted), and a 0–100 confidence score (green ≥ 80, yellow 50–79, red < 50). Toggle any match off to exclude it.
3. **Generate.** The remaining matches are assembled into a sequence in phrase order, with each clip trimmed to just the matched transcript segment.

**Tips:**

- Short, distinct phrases (3–6 words) work best. Two-word phrases like "thank you" tend to match too broadly.
- The slider controls *how many* matches per phrase, not how strict the matching is. Quality control happens on the review screen.
- Clips without transcripts are silently excluded. If the project has no transcribed clips, the dialog will tell you to run **Analyze → Transcribe** first.
- Disabled clips are excluded automatically.

Cassette Tape uses local string-similarity matching (RapidFuzz `partial_ratio`). It does not require a cloud API key. It works at transcript-segment granularity, so it can trim the resulting sequence to the exact matched line instead of using the whole detected clip.

### Word Sequencer

Compose a film one *word* at a time. Where Cassette Tape works at transcript-segment granularity, Word Sequencer slices below the segment boundary: every emission is a single spoken word cut out of its parent clip. Opens a dialog with five ordering modes.

**Required analysis:** Transcribe → Word-Level Alignment (the dialog auto-runs alignment if any selected source is missing word data)

**Dialog workflow:**

1. **Pick sources.** A source list shows every checked source's alignment status as a badge: `✓ aligned`, `… needs alignment`, or `⚠ unsupported language`. Uncheck rows to exclude.
2. **Pick a mode.** The five modes are:

   | Mode | What it does |
   |------|--------------|
   | **Alphabetical** | Every word in the corpus in lexical order. |
   | **Chosen Words** | Subset filter — emit every instance of each word on your include list, grouped by include-list order. (E.g., "play every 'never', then every 'always', then every 'silence'.") |
   | **By Frequency** | Every word in the corpus ordered most-frequent → least-frequent, or reverse. |
   | **By Property** | Sort by word length (default), word duration, or log-frequency. Ascending or descending. |
   | **User-Curated Ordered List** | You supply an exact sequence — including repeats — that materializes literally. `["the", "the", "the", "sky"]` produces four slots. |
3. **Handle frames.** Optional padding (0–10 frames) on each side of the word boundary so consonants don't get clipped.
4. **Generate.** Output is a hard-cut sequence of word-sized SequenceClips with frame-accurate in/out points. No crossfades, no held frames.

**Tips:**
- Word boundaries are accurate to ~20–30ms. The handle-frame spinner is the escape valve when plosives clip mid-emission.
- **Chosen Words** vs **User-Curated** differ in two ways: chosen-words plays *every instance* of the listed words; user-curated plays *one slot per list entry*. Chosen-words groups by include-list word; user-curated is verbatim.
- The dialog auto-runs alignment when needed — no extra confirmation modal. Cancel mid-alignment is safe; already-aligned clips persist.
- Sources whose language isn't supported by the alignment model are shown with a `⚠ unsupported language` badge and excluded from the corpus (you can still proceed with the other checked sources).
- Disabled clips are excluded automatically.

### LLM Word Composer

Same word-level slicing as Word Sequencer, but the order comes from a local LLM. Type a prompt and an Ollama model composes a sentence using only the words in your corpus — vocabulary is enforced at decode time (JSON-schema-enum constraint), so the LLM cannot emit a word you don't have.

**Required analysis:** Transcribe → Word-Level Alignment. Also requires a healthy local Ollama runtime.

**Dialog workflow:**

1. **Pick sources.** Same source picker as Word Sequencer, with alignment badges.
2. **Enter a prompt.** e.g., "compose a sentence about silence" or "make a question someone would ask in a dream."
3. **Set target length.** Default 20 words; range 1–200.
4. **Choose a repeat policy.** When the LLM emits the same word twice, which corpus instance plays each time? Round-robin (default — cycle through instances), Random (with a seed for determinism), First (always pick the first instance), Longest, or Shortest.
5. **Generate.** The LLM drafts a sentence using only your corpus; each emitted word is mapped to a corpus instance and materialized as a hard-cut SequenceClip.

**Tips:**
- The dialog detects Ollama at open time. If Ollama isn't running, you'll see an installer prompt and the Accept button stays disabled until you start the runtime and re-check.
- Generation cancellation is cooperative — clicking Cancel during a long generation interrupts cleanly without leaving the sequence in a half-applied state.
- Latency scales with corpus size. ~1000 unique words takes around 30 seconds on `qwen3:8b`; much larger corpora can take several minutes. The dialog shows a progress indicator.
- The vocabulary constraint is strict — the LLM cannot emit out-of-vocabulary words even if your prompt suggests them. If "rainstorm" is in your prompt but no clip says it, the model will choose a different word from your corpus.
- Empty responses (LLM returned `None`) surface as an inline error so you can adjust the prompt and retry without dismissing the dialog.
