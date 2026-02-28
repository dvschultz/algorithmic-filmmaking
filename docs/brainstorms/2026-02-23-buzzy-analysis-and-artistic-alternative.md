# Buzzy.now Analysis and Artistic Alternative Approach

## What Buzzy Does

Buzzy is a viral video recreation engine. Its tagline: **"Hunts. Breaks down. Scales."**

The core loop is:

1. **Hunt** — Buzzy scrapes/indexes viral content from TikTok, Instagram Reels, YouTube, Pinterest. It tracks engagement metrics (impressions, likes, comments) and surfaces trending formats.

2. **Break Down** — When you pick a viral video (or drop in your own), Buzzy uses AI to decompose it into a structured "Viral Recipe": a scene-by-scene breakdown with shot descriptions, timing, hooks, and a narrative structure. Example breakdown:
   - Scene 1: "Face-level visual hook" (00:00-00:03)
   - Scene 2: "Eyewear detail focus" (00:03-00:06)
   - Scene 3: "Direct eye contact moment" (00:06-00:10)
   - Scene 4: "Full-look brand payoff" (00:10-00:14)

3. **Scale** — Buzzy generates multiple "ideation directions" from the recipe:
   - **Short version** (condensed edit)
   - **Background swap** (same structure, different setting)
   - **Content rewrite** (same format, different subject)
   - **Recreate My Version** (full generation with your avatar/content)

The output is ready-to-post video content for social media.

## How It Likely Works (Technical Speculation)

### Ingestion Pipeline
- Social media API scraping or partnership data for engagement metrics
- Video downloading + metadata extraction (similar to yt-dlp)
- Thumbnail/keyframe extraction for the gallery

### Analysis Engine (the "Break Down")
- **Scene detection** to segment the viral video into shots
- **VLM (Vision Language Model)** to describe each scene's content, framing, and purpose
- **Engagement pattern recognition** — LLM identifies why this particular format went viral (the "hook analysis")
- Outputs a structured "recipe" — essentially a shot list with timing, descriptions, and strategic annotations

### Generation Pipeline (the "Scale")
- **Avatar/face swap** — User uploads a face; Buzzy composites it into the template (likely using face-swap/reenactment models)
- **Background replacement** — Segment subject, swap environment (SAM + inpainting or video gen)
- **AI video generation** — For the "Recreate My Version" flow, likely uses a text-to-video or image-to-video model (Kling, Runway, Pika, or their own) conditioned on the scene descriptions
- **Template-based editing** — The recipe acts as an EDL; cuts, transitions, and text overlays are applied programmatically

### Input Options
- **Optional Avatar** — Upload your face for personalized recreation
- **Optional Input Video** — Drop in a reference video to analyze
- **Text prompt** — "Tell Buzzy what you want" (free-form creative direction)
- **Category filters** — Social Virals, Ads Virals, Viral Templates

### Content Categories
- **Social Media Virals** — TikTok/Reels recreations with engagement metrics shown
- **Brand TVC** — Television commercial style recreations
- **Meta UGC Ads** — User-generated content style ads for Meta platforms

## What Makes Buzzy Effective

- **Zero creative effort** — Users don't need to understand filmmaking; the recipe does the thinking
- **Proven formats** — Every template is a viral video with real engagement data
- **One-click recreation** — The "Recreate My Version" button is the entire UX
- **Speed** — From reference video to output in minutes, not hours

## What Buzzy Gets Wrong (from an artistic perspective)

- **Homogeneity** — Every output looks like a copy of a copy. The "viral recipe" flattens creative decisions into a formula.
- **No authorial voice** — The system optimizes for engagement metrics, not expression. There's no mechanism for a creator to inject their own visual sensibility.
- **Trend chasing as creative strategy** — "What went viral last week" is a race to the bottom. By the time you recreate it, the trend is over.
- **Template thinking** — Reducing filmmaking to slot-filling (swap face here, change background there) produces content, not work.
- **Metric worship** — Engagement numbers on every thumbnail train users to think in impressions, not ideas.

---

## An Artistic Alternative: Reference-Guided Remixing

### Target User

Experimental filmmakers and video artists — people who deconstruct and recombine existing video as a creative practice. Not content marketers. Not students learning basics. Artists who already have a visual sensibility and want a tool that extends it.

### Core Concept

A reference video is decomposed into its structural DNA using the analysis methods the artist selects. That same analysis runs on the artist's own clip library. The tool then matches clips to reference moments across weighted dimensions, generating a rough sequence the artist refines.

**Buzzy asks:** "What went viral? Let me clone it for you."
**This asks:** "What interests you? Let me help you remix your own footage through its structure."

### The Loop: Watch. Understand. Remix.

#### 1. Watch — Decompose a Reference

Import any video from your library, or search/download from YouTube, Vimeo, or Internet Archive. The tool doesn't care about engagement metrics. You choose what interests you.

**The artist selects which analysis dimensions to run on the reference:**
- **Rhythm** — Cut timing as a beat pattern, correlated with audio onsets
- **Color trajectory** — How the palette shifts through the piece (warm→cold, saturated→muted, etc.)
- **Movement vocabulary** — Camera motion vectors: static, pan, tilt, dolly, handheld, whip
- **Shot scale** — ECU, CU, MS, WS, EWS progression
- **Audio energy** — Loudness, beat density, spectral characteristics over time
- **Scene descriptions** — VLM-generated descriptions of content and framing
- **Compositional gravity** — Where the eye goes in each frame (saliency heat maps)
- **Tension mapping** — Composite curve combining all active dimensions

The artist chooses which dimensions matter for this particular study. A music video study might weight rhythm and audio energy heavily. A landscape film study might prioritize color trajectory and composition.

The output is a **score** — a multi-track visualization of what the filmmaker did, broken down per dimension.

#### 2. Understand — Compare and Visualize

**Side-by-side analysis** is the first concrete feature. The artist can:

- View the reference's rhythm strip, color trajectory, and other analysis tracks as overlaid visualizations
- Import a second reference and overlay their scores to see structural differences
- See where their own analyzed clips line up (or diverge) from the reference
- Annotate moments with notes for personal reference

This step builds intuition. Before the tool generates anything, the artist can *see* the structural DNA and verify it matches their read of the piece.

#### 3. Remix — Reference-Guided Sequence Generation

This is the core output. The reference's structural decomposition becomes a template, and the artist's clips are matched into it.

**How matching works:**

The reference video has been decomposed into N clips, each with an analysis vector across the selected dimensions (rhythm position, dominant color, motion class, shot scale, audio energy, etc.).

The artist's clip library has the same analysis run on it.

For each slot in the reference, the tool finds the best-matching clip from the artist's library using a weighted distance function. **The artist controls the weights via sliders:**

```
Rhythm weight:     ████████░░  80%
Color weight:      ████░░░░░░  40%
Motion weight:     ██████░░░░  60%
Shot scale weight: ██░░░░░░░░  20%
Audio weight:      ██████████  100%
```

Different slider positions produce radically different sequences from the same source material and reference. Cranking rhythm to 100% and everything else to 0% produces a sequence that mirrors the reference's pacing but with wild visual variation. Cranking color to 100% produces a sequence that follows the reference's palette journey but with completely different timing.

**The output is a rough sequence in the Sequence tab** — not a finished video, but a starting point the artist refines. They can:
- Reorder, swap, or remove clips
- Adjust in/out points
- Re-run with different slider weights to explore alternatives
- Keep what works and manually replace what doesn't

### Where This Lives in Scene Ripper

**The Sequence tab.** No new tab or mode needed. The workflow:

1. **Collect** — Import sources (your footage) + a reference video
2. **Cut** — Detect scenes in both your footage and the reference
3. **Analyze** — Run selected analysis on both (the artist picks which dimensions)
4. **Sequence** — Select the reference video as a "guide." Adjust dimension weights. Generate a matched sequence. Refine it.
5. **Render** — Export as usual

The Sequence tab gains a new capability: selecting a reference source and generating a weighted-match sequence from it. This sits alongside the existing sorting/filtering and manual drag-drop arrangement.

### Reference Sources

- **Bring your own** — Import from local files, as with any source
- **Search and download** — Use the existing YouTube, Vimeo, and Internet Archive search already built into Scene Ripper's Collect tab
- No curated gallery. No engagement metrics. The artist's taste is the curation.

### What This Reuses from Scene Ripper

| Capability | Existing Component | Reuse Level |
|---|---|---|
| Video import + scene detection | `core/scene_detect.py` | Direct reuse |
| Shot scale classification | `core/analysis/shots.py` | Direct reuse |
| Color palette per clip | `core/analysis/color.py` | Direct reuse |
| Scene descriptions (VLM) | `core/analysis/description.py` | Direct reuse |
| Audio analysis (librosa) | `core/analysis/` + librosa | Direct reuse |
| Object detection | `core/analysis/` + YOLO | Direct reuse |
| Clip data model + metadata | `models/clip.py` | Direct reuse |
| Sequence model + timeline | `models/sequence.py` + UI | Direct reuse |
| Shuffle/remix algorithms | `core/remix/shuffle.py` | Extend |
| Video download (yt-dlp) | `core/downloader.py` | Direct reuse |
| YouTube/Vimeo/Archive search | `core/youtube_api.py` + UI | Direct reuse |

### What's New

| Feature | Description | Complexity |
|---|---|---|
| **Reference-guided matching** | Score each user clip against each reference slot across weighted dimensions; assemble best-match sequence | Medium — core algorithm |
| **Dimension weight sliders** | UI for adjusting how much each analysis dimension influences matching | Small — UI widget |
| **Rhythm strip visualization** | Visualize cut timing as a beat pattern alongside audio waveform | Medium — new timeline track |
| **Color trajectory visualization** | Plot dominant color per clip over time as a gradient strip | Small — rendering existing data |
| **Side-by-side score comparison** | Overlay two videos' analysis tracks | Medium — new comparison view |
| **Movement analysis (optical flow)** | Classify camera motion per clip | Medium — new analysis module |
| **Compositional saliency** | Where the eye goes in each frame | Large — new model or heuristic |
| **Tension mapping** | Composite intensity curve from all dimensions | Small — weighted sum of existing |

### The Matching Algorithm

This is the most important piece. Pseudocode:

```python
def generate_reference_guided_sequence(
    reference_clips: list[Clip],      # analyzed reference video clips
    user_clips: list[Clip],           # analyzed user footage clips
    weights: dict[str, float],        # dimension name -> weight (0.0 to 1.0)
    analysis_dimensions: list[str],   # which dimensions are active
) -> list[SequenceClip]:
    """
    For each clip in the reference, find the best-matching clip
    from the user's library based on weighted multi-dimensional distance.
    """
    sequence = []
    used_clips = set()  # avoid repeats (optional, artist can toggle)

    for ref_clip in reference_clips:
        ref_vector = extract_feature_vector(ref_clip, analysis_dimensions)

        best_match = None
        best_distance = float('inf')

        for user_clip in user_clips:
            if user_clip.id in used_clips:
                continue
            user_vector = extract_feature_vector(user_clip, analysis_dimensions)
            distance = weighted_distance(ref_vector, user_vector, weights)
            if distance < best_distance:
                best_distance = distance
                best_match = user_clip

        if best_match:
            used_clips.add(best_match.id)
            sequence.append(make_sequence_clip(
                best_match,
                duration=ref_clip.duration,  # match reference timing
            ))

    return sequence
```

The `extract_feature_vector` function normalizes each dimension (duration, dominant hue, motion class, shot scale, audio energy, etc.) into a comparable numeric space. The `weighted_distance` function computes a weighted Euclidean distance (or cosine similarity, depending on the dimension type).

### MVP Scope

Build in this order:

1. **Color trajectory strip** — Plot dominant color per clip as a gradient in the timeline. Leverages existing color analysis. Proves the visualization concept.

2. **Rhythm strip** — Visualize cut timing as a beat pattern alongside audio waveform. Leverages existing scene detection + librosa.

3. **Side-by-side comparison view** — Import two videos, view their color + rhythm strips together. First taste of "seeing structural DNA."

4. **Reference-guided matching** — Select a reference source, set dimension weights, generate a matched sequence. The core algorithm. Initially using only the dimensions we already analyze (color, shot scale, duration, description similarity).

5. **Dimension weight sliders** — UI for adjusting matching weights. Makes the algorithm explorable.

6. **Movement analysis** — Add optical flow classification as a new analysis dimension. Enriches matching quality.

Steps 1-5 can be built primarily from existing infrastructure. Step 6 requires a new analysis module.

### What This Is Not

- Not a viral content cloner
- Not a face-swap tool
- Not an AI video generator
- Not a template marketplace
- Not optimizing for engagement metrics
- Not trying to replace the artist's judgment — it's trying to extend it

The tool assumes you have taste, footage, and something to say. It helps you say it in conversation with work that moves you.
