---
title: "feat: Add Exquisite Corpus sequencer for text-based poetry sequences"
type: feat
date: 2026-01-31
---

# Exquisite Corpus Sequencer

## Overview

Add a new sequencing algorithm called "Exquisite Corpus" that extracts on-screen text from video clips using OCR, then uses an LLM to compose a poem from the extracted phrases. The footage is sequenced to match the poem's line order, creating a visual poem where each clip corresponds to a line.

**Key Constraints:**
- Phrases must be used exactly as extracted (no word modifications)
- Each poem line = one clip's text = one clip in the sequence
- Clips without detectable text are excluded from the final sequence

**Inspiration:** Cut-up technique meets found footage poetry.

## Problem Statement / Motivation

Users working with archival footage, documentary material, or text-heavy video content (signs, titles, subtitles, documents) want to create poetic sequences based on the text visible in their clips. Currently, there's no automated way to:

1. Extract text from video frames
2. Use that text creatively for sequencing
3. Generate artistic arrangements based on textual content

This feature enables a new creative workflow combining OCR technology with generative AI to produce unique video poems.

## Proposed Solution

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      User Flow                                   │
├─────────────────────────────────────────────────────────────────┤
│  1. Select clips in Sequence Tab                                │
│  2. Click "Exquisite Corpus" algorithm card                     │
│  3. System extracts text from clips (OCR + VLM fallback)        │
│  4. User enters mood/vibe prompt                                │
│  5. LLM generates poem using exact extracted phrases            │
│  6. User previews/edits poem arrangement                        │
│  7. System sequences clips to match poem order                  │
│  8. Final sequence appears on timeline                          │
└─────────────────────────────────────────────────────────────────┘
```

### Component Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    New Components                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  core/analysis/ocr.py          # Text extraction module          │
│    ├── extract_text_from_frame()   # Single frame OCR           │
│    ├── extract_text_from_clip()    # Multi-keyframe extraction  │
│    └── _vlm_text_extraction()      # VLM fallback               │
│                                                                  │
│  core/remix/exquisite_corpus.py    # Poem generation + sequencing│
│    ├── generate_poem()             # LLM poem creation          │
│    ├── validate_poem_phrases()     # Ensure exact phrase usage  │
│    └── sequence_by_poem()          # Map poem lines to clips    │
│                                                                  │
│  ui/dialogs/exquisite_corpus_dialog.py  # Workflow dialog       │
│    ├── MoodPromptPage              # Mood input                 │
│    ├── ExtractionProgressPage      # OCR progress               │
│    └── PoemPreviewPage             # Review & reorder           │
│                                                                  │
│  ui/workers/text_extraction_worker.py   # Background OCR        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Technical Approach

### 1. Data Model Changes

**Add to `models/clip.py`:**

```python
@dataclass
class ExtractedText:
    """Text extracted from a video frame."""
    frame_number: int
    text: str
    confidence: float  # 0.0 - 1.0
    source: str  # "tesseract" or "vlm"
    bounding_boxes: Optional[list[dict]] = None  # [{x, y, w, h, text}]

@dataclass
class Clip:
    # ... existing fields ...

    # NEW: OCR extracted text
    extracted_texts: Optional[list[ExtractedText]] = None

    @property
    def combined_text(self) -> Optional[str]:
        """Returns deduplicated text from all extractions."""
        if not self.extracted_texts:
            return None
        unique_texts = []
        seen = set()
        for et in self.extracted_texts:
            normalized = et.text.strip().lower()
            if normalized and normalized not in seen:
                unique_texts.append(et.text.strip())
                seen.add(normalized)
        return " | ".join(unique_texts) if unique_texts else None
```

**Update `Clip.to_dict()` and `Clip.from_dict()`** to serialize/deserialize `extracted_texts`.

### 2. OCR Module (`core/analysis/ocr.py`)

```python
# core/analysis/ocr.py
"""On-screen text extraction using Tesseract OCR with VLM fallback."""

import threading
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

# Thread-safe model loading
_tesseract_available: Optional[bool] = None
_tesseract_lock = threading.Lock()

def _check_tesseract() -> bool:
    """Check if Tesseract is installed and available."""
    global _tesseract_available
    if _tesseract_available is not None:
        return _tesseract_available
    with _tesseract_lock:
        if _tesseract_available is None:
            try:
                import pytesseract
                pytesseract.get_tesseract_version()
                _tesseract_available = True
            except Exception:
                _tesseract_available = False
    return _tesseract_available

def extract_text_from_frame(
    frame_path: Path,
    use_vlm_fallback: bool = True,
    vlm_model: Optional[str] = None,
    confidence_threshold: float = 0.6
) -> tuple[str, float, str]:
    """
    Extract text from a single video frame.

    Args:
        frame_path: Path to the frame image
        use_vlm_fallback: Whether to use VLM if Tesseract fails/low confidence
        vlm_model: VLM model to use for fallback (default: from settings)
        confidence_threshold: Minimum confidence to accept Tesseract result

    Returns:
        Tuple of (text, confidence, source) where source is "tesseract" or "vlm"
    """
    text = ""
    confidence = 0.0
    source = "tesseract"

    # Try Tesseract first
    if _check_tesseract():
        import pytesseract
        from PIL import Image

        image = Image.open(frame_path)
        # Get detailed data including confidence
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

        # Filter by confidence and extract text
        words = []
        confidences = []
        for i, word in enumerate(data['text']):
            conf = int(data['conf'][i])
            if conf > 0 and word.strip():
                words.append(word)
                confidences.append(conf / 100.0)

        if words:
            text = " ".join(words)
            confidence = sum(confidences) / len(confidences)
            source = "tesseract"

    # VLM fallback if Tesseract unavailable or low confidence
    if use_vlm_fallback and (not text or confidence < confidence_threshold):
        vlm_text, vlm_conf = _vlm_text_extraction(frame_path, vlm_model)
        if vlm_text and (not text or vlm_conf > confidence):
            text = vlm_text
            confidence = vlm_conf
            source = "vlm"

    return text, confidence, source

def _vlm_text_extraction(
    frame_path: Path,
    model: Optional[str] = None
) -> tuple[str, float]:
    """
    Extract text using Vision Language Model.

    Returns:
        Tuple of (text, confidence)
    """
    from core.llm_client import LLMClient
    from core.settings import load_settings

    settings = load_settings()
    model = model or settings.vlm_model or "gpt-4o"

    client = LLMClient(model=model)

    prompt = """Extract ALL visible text from this image. Include:
- Signs, labels, titles
- Subtitles or captions
- Text on documents or screens
- Any other readable text

Return ONLY the extracted text, one phrase per line. If no text is visible, return "NO_TEXT_FOUND".
Do not add any commentary or descriptions."""

    try:
        response = client.vision_call(prompt, image_path=frame_path)
        text = response.strip()
        if text == "NO_TEXT_FOUND":
            return "", 0.0
        # VLM confidence is estimated based on response quality
        confidence = 0.85 if text else 0.0
        return text, confidence
    except Exception:
        return "", 0.0

def extract_text_from_clip(
    clip,
    source,
    num_keyframes: int = 3,
    use_vlm_fallback: bool = True,
    progress_callback: Optional[callable] = None
) -> list:
    """
    Extract text from multiple keyframes of a clip.

    Args:
        clip: Clip object
        source: Source object containing the video
        num_keyframes: Number of frames to sample (3-5 recommended)
        use_vlm_fallback: Whether to use VLM for low-confidence results
        progress_callback: Optional callback(frame_num, total) for progress

    Returns:
        List of ExtractedText objects
    """
    from core.ffmpeg import extract_frame
    from models.clip import ExtractedText
    import tempfile

    results = []

    # Calculate keyframe positions
    total_frames = clip.end_frame - clip.start_frame
    if total_frames <= 0:
        return results

    # Distribute keyframes evenly
    if num_keyframes >= total_frames:
        frame_positions = list(range(clip.start_frame, clip.end_frame + 1))
    else:
        step = total_frames / (num_keyframes - 1) if num_keyframes > 1 else 0
        frame_positions = [
            int(clip.start_frame + i * step)
            for i in range(num_keyframes)
        ]

    # Extract and OCR each keyframe
    for i, frame_num in enumerate(frame_positions):
        if progress_callback:
            progress_callback(i + 1, len(frame_positions))

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            frame_path = Path(tmp.name)

        try:
            # Extract frame from video
            extract_frame(source.file_path, frame_num, frame_path, source.fps)

            # Run OCR
            text, confidence, ocr_source = extract_text_from_frame(
                frame_path,
                use_vlm_fallback=use_vlm_fallback
            )

            if text:
                results.append(ExtractedText(
                    frame_number=frame_num,
                    text=text,
                    confidence=confidence,
                    source=ocr_source
                ))
        finally:
            # Clean up temp file
            if frame_path.exists():
                frame_path.unlink()

    return results
```

### 3. Poem Generation (`core/remix/exquisite_corpus.py`)

```python
# core/remix/exquisite_corpus.py
"""Exquisite Corpus: Generate poems from extracted video text."""

from typing import Optional
from dataclasses import dataclass

@dataclass
class PoemLine:
    """A single line of the generated poem."""
    text: str           # The exact phrase used
    clip_id: str        # Source clip ID
    line_number: int    # Position in poem (1-indexed)

@dataclass
class ExquisiteCorpusResult:
    """Result of Exquisite Corpus generation."""
    poem_lines: list[PoemLine]
    mood_prompt: str
    excluded_clip_ids: list[str]  # Clips with no text

    @property
    def poem_text(self) -> str:
        """Return the poem as formatted text."""
        return "\n".join(line.text for line in self.poem_lines)

def generate_poem(
    clips_with_text: list[tuple],  # [(clip, extracted_text_string), ...]
    mood_prompt: str,
    model: Optional[str] = None
) -> list[PoemLine]:
    """
    Generate a poem using LLM from extracted clip texts.

    Args:
        clips_with_text: List of (Clip, text) tuples
        mood_prompt: User's mood/vibe description
        model: LLM model to use (default: from settings)

    Returns:
        List of PoemLine objects in poem order
    """
    from core.llm_client import LLMClient
    from core.settings import load_settings
    import json

    settings = load_settings()
    model = model or settings.llm_model or "gpt-4o"

    # Build the phrase inventory
    phrase_inventory = {}
    for clip, text in clips_with_text:
        phrase_inventory[clip.id] = text

    # Create the LLM prompt
    system_prompt = """You are a poet creating visual poetry from found text.

CRITICAL RULES:
1. You MUST use phrases EXACTLY as provided - no modifications whatsoever
2. Each line of your poem must be one complete phrase from the inventory
3. You cannot split phrases, combine words from different phrases, or change any words
4. You may choose which phrases to use and in what order
5. Not all phrases need to be used
6. Create a cohesive poem that evokes the requested mood

OUTPUT FORMAT:
Return a JSON array where each element is the clip_id of the phrase to use, in poem order.
Example: ["clip_abc123", "clip_def456", "clip_ghi789"]"""

    user_prompt = f"""Create a poem with the mood: {mood_prompt}

Available phrases (clip_id: phrase):
{json.dumps(phrase_inventory, indent=2)}

Return ONLY the JSON array of clip_ids in the order they should appear in the poem."""

    client = LLMClient(model=model)
    response = client.chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    # Parse response
    try:
        # Extract JSON from response
        response_text = response.strip()
        if response_text.startswith("```"):
            # Remove markdown code blocks
            lines = response_text.split("\n")
            response_text = "\n".join(
                line for line in lines
                if not line.startswith("```")
            )

        clip_order = json.loads(response_text)

        # Build poem lines
        poem_lines = []
        for i, clip_id in enumerate(clip_order, 1):
            if clip_id in phrase_inventory:
                poem_lines.append(PoemLine(
                    text=phrase_inventory[clip_id],
                    clip_id=clip_id,
                    line_number=i
                ))

        return poem_lines

    except json.JSONDecodeError:
        # Fallback: try to match response text to phrases
        raise ValueError("LLM did not return valid JSON. Please try again.")

def validate_poem_phrases(
    poem_lines: list[PoemLine],
    original_phrases: dict[str, str]  # clip_id -> text
) -> list[tuple[int, str]]:
    """
    Validate that poem uses exact phrases.

    Returns:
        List of (line_number, error_message) for any violations
    """
    errors = []
    for line in poem_lines:
        if line.clip_id not in original_phrases:
            errors.append((line.line_number, f"Unknown clip: {line.clip_id}"))
        elif line.text != original_phrases[line.clip_id]:
            errors.append((
                line.line_number,
                f"Modified phrase detected. Expected: '{original_phrases[line.clip_id]}'"
            ))
    return errors

def sequence_by_poem(
    poem_lines: list[PoemLine],
    clips_by_id: dict,
    sources_by_id: dict
) -> list[tuple]:
    """
    Create a clip sequence matching the poem order.

    Returns:
        List of (Clip, Source) tuples in poem order
    """
    sequence = []
    for line in poem_lines:
        if line.clip_id in clips_by_id:
            clip = clips_by_id[line.clip_id]
            source = sources_by_id.get(clip.source_id)
            if source:
                sequence.append((clip, source))
    return sequence
```

### 4. Background Worker (`ui/workers/text_extraction_worker.py`)

```python
# ui/workers/text_extraction_worker.py
"""Background worker for OCR text extraction."""

from PySide6.QtCore import QThread, Signal
from typing import Optional

class TextExtractionWorker(QThread):
    """Extract text from multiple clips in background."""

    # Signals
    progress = Signal(int, int, str)  # current, total, clip_id
    clip_completed = Signal(str, list)  # clip_id, extracted_texts
    finished = Signal(dict)  # {clip_id: [ExtractedText, ...]}
    error = Signal(str)

    def __init__(
        self,
        clips: list,
        sources_by_id: dict,
        num_keyframes: int = 3,
        use_vlm_fallback: bool = True,
        parent=None
    ):
        super().__init__(parent)
        self.clips = clips
        self.sources_by_id = sources_by_id
        self.num_keyframes = num_keyframes
        self.use_vlm_fallback = use_vlm_fallback
        self._cancelled = False

    def cancel(self):
        """Request cancellation of the extraction."""
        self._cancelled = True

    def run(self):
        """Execute text extraction on all clips."""
        from core.analysis.ocr import extract_text_from_clip

        results = {}
        total = len(self.clips)

        for i, clip in enumerate(self.clips):
            if self._cancelled:
                break

            source = self.sources_by_id.get(clip.source_id)
            if not source:
                continue

            self.progress.emit(i + 1, total, clip.id)

            try:
                extracted = extract_text_from_clip(
                    clip=clip,
                    source=source,
                    num_keyframes=self.num_keyframes,
                    use_vlm_fallback=self.use_vlm_fallback
                )
                results[clip.id] = extracted
                self.clip_completed.emit(clip.id, extracted)
            except Exception as e:
                self.error.emit(f"Error extracting text from clip {clip.id}: {e}")

        if not self._cancelled:
            self.finished.emit(results)
```

### 5. Workflow Dialog (`ui/dialogs/exquisite_corpus_dialog.py`)

```python
# ui/dialogs/exquisite_corpus_dialog.py
"""Multi-step dialog for Exquisite Corpus workflow."""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QStackedWidget,
    QLabel, QTextEdit, QPushButton, QProgressBar,
    QListWidget, QListWidgetItem, QWidget, QFrame
)
from PySide6.QtCore import Qt, Signal
from ui.theme import ThemeColors, UISizes

class ExquisiteCorpusDialog(QDialog):
    """Multi-page dialog for Exquisite Corpus workflow."""

    sequence_ready = Signal(list)  # List of (Clip, Source) tuples

    def __init__(self, clips, sources_by_id, project, parent=None):
        super().__init__(parent)
        self.clips = clips
        self.sources_by_id = sources_by_id
        self.project = project
        self.extraction_results = {}
        self.poem_lines = []

        self.setWindowTitle("Exquisite Corpus")
        self.setMinimumSize(600, 500)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Stacked widget for pages
        self.stack = QStackedWidget()
        layout.addWidget(self.stack)

        # Page 1: Mood prompt
        self.mood_page = self._create_mood_page()
        self.stack.addWidget(self.mood_page)

        # Page 2: Extraction progress
        self.progress_page = self._create_progress_page()
        self.stack.addWidget(self.progress_page)

        # Page 3: Poem preview
        self.preview_page = self._create_preview_page()
        self.stack.addWidget(self.preview_page)

        # Navigation buttons
        nav_layout = QHBoxLayout()
        self.back_btn = QPushButton("Back")
        self.back_btn.clicked.connect(self._go_back)
        self.back_btn.setVisible(False)

        self.next_btn = QPushButton("Extract Text")
        self.next_btn.clicked.connect(self._go_next)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)

        nav_layout.addWidget(self.back_btn)
        nav_layout.addStretch()
        nav_layout.addWidget(self.cancel_btn)
        nav_layout.addWidget(self.next_btn)
        layout.addLayout(nav_layout)

    def _create_mood_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        # Header
        header = QLabel("Exquisite Corpus")
        header.setStyleSheet(f"font-size: 18px; font-weight: bold;")
        layout.addWidget(header)

        # Description
        desc = QLabel(
            f"Selected {len(self.clips)} clips. Text will be extracted from each clip "
            "and used to generate a poem. The clips will be sequenced to match the poem."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Mood prompt input
        layout.addWidget(QLabel("Enter the mood or vibe for your poem:"))
        self.mood_input = QTextEdit()
        self.mood_input.setPlaceholderText(
            "e.g., melancholic and introspective, chaotic urban energy, "
            "dreamlike and surreal, contemplative silence..."
        )
        self.mood_input.setMaximumHeight(100)
        layout.addWidget(self.mood_input)

        layout.addStretch()
        return page

    def _create_progress_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        layout.addWidget(QLabel("Extracting text from clips..."))

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("Starting...")
        layout.addWidget(self.progress_label)

        # Results summary
        self.results_label = QLabel("")
        layout.addWidget(self.results_label)

        layout.addStretch()
        return page

    def _create_preview_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        layout.addWidget(QLabel("Generated Poem (drag to reorder):"))

        # Poem line list (drag-drop enabled)
        self.poem_list = QListWidget()
        self.poem_list.setDragDropMode(QListWidget.InternalMove)
        self.poem_list.setDefaultDropAction(Qt.MoveAction)
        layout.addWidget(self.poem_list)

        # Regenerate button
        regen_btn = QPushButton("Regenerate Poem")
        regen_btn.clicked.connect(self._regenerate_poem)
        layout.addWidget(regen_btn)

        return page

    def _go_back(self):
        current = self.stack.currentIndex()
        if current > 0:
            self.stack.setCurrentIndex(current - 1)
            self._update_nav_buttons()

    def _go_next(self):
        current = self.stack.currentIndex()

        if current == 0:  # Mood page -> Progress page
            mood = self.mood_input.toPlainText().strip()
            if not mood:
                return  # Require mood input
            self.stack.setCurrentIndex(1)
            self._start_extraction()

        elif current == 2:  # Preview page -> Done
            self._finish()

        self._update_nav_buttons()

    def _update_nav_buttons(self):
        current = self.stack.currentIndex()
        self.back_btn.setVisible(current > 0 and current != 1)

        if current == 0:
            self.next_btn.setText("Extract Text")
            self.next_btn.setEnabled(True)
        elif current == 1:
            self.next_btn.setText("Please wait...")
            self.next_btn.setEnabled(False)
        elif current == 2:
            self.next_btn.setText("Create Sequence")
            self.next_btn.setEnabled(True)

    def _start_extraction(self):
        from ui.workers.text_extraction_worker import TextExtractionWorker

        self.worker = TextExtractionWorker(
            clips=self.clips,
            sources_by_id=self.sources_by_id,
            num_keyframes=3,
            use_vlm_fallback=True,
            parent=self
        )
        self.worker.progress.connect(self._on_extraction_progress)
        self.worker.finished.connect(self._on_extraction_finished)
        self.worker.error.connect(self._on_extraction_error)
        self.worker.start()

    def _on_extraction_progress(self, current, total, clip_id):
        self.progress_bar.setValue(int(current / total * 100))
        self.progress_label.setText(f"Processing clip {current}/{total}")

    def _on_extraction_finished(self, results):
        self.extraction_results = results

        # Count clips with text
        clips_with_text = sum(1 for texts in results.values() if texts)
        self.results_label.setText(
            f"Found text in {clips_with_text}/{len(self.clips)} clips"
        )

        if clips_with_text < 2:
            self.results_label.setText(
                f"Only {clips_with_text} clips have text. Need at least 2 for a poem."
            )
            return

        # Generate poem
        self._generate_poem()

    def _on_extraction_error(self, error_msg):
        self.progress_label.setText(f"Error: {error_msg}")

    def _generate_poem(self):
        from core.remix.exquisite_corpus import generate_poem

        # Build clips_with_text list
        clips_with_text = []
        clips_by_id = {c.id: c for c in self.clips}

        for clip_id, texts in self.extraction_results.items():
            if texts:
                clip = clips_by_id.get(clip_id)
                if clip:
                    combined = " | ".join(t.text for t in texts)
                    clips_with_text.append((clip, combined))

        mood = self.mood_input.toPlainText().strip()

        try:
            self.poem_lines = generate_poem(clips_with_text, mood)
            self._display_poem()
            self.stack.setCurrentIndex(2)
            self._update_nav_buttons()
        except Exception as e:
            self.progress_label.setText(f"Poem generation failed: {e}")

    def _display_poem(self):
        self.poem_list.clear()
        for line in self.poem_lines:
            item = QListWidgetItem(line.text)
            item.setData(Qt.UserRole, line.clip_id)
            self.poem_list.addItem(item)

    def _regenerate_poem(self):
        self._generate_poem()

    def _finish(self):
        from core.remix.exquisite_corpus import sequence_by_poem, PoemLine

        # Get reordered poem from list widget
        reordered_lines = []
        for i in range(self.poem_list.count()):
            item = self.poem_list.item(i)
            clip_id = item.data(Qt.UserRole)
            text = item.text()
            reordered_lines.append(PoemLine(
                text=text,
                clip_id=clip_id,
                line_number=i + 1
            ))

        # Create sequence
        clips_by_id = {c.id: c for c in self.clips}
        sequence = sequence_by_poem(
            reordered_lines,
            clips_by_id,
            self.sources_by_id
        )

        self.sequence_ready.emit(sequence)
        self.accept()
```

### 6. Integration Points

**Add to `ui/widgets/sorting_card_grid.py`:**

```python
ALGORITHMS = {
    "color": ("palette", "Color", "Sort clips by dominant color palette"),
    "duration": ("clock", "Duration", "Sort clips by length"),
    "shuffle": ("shuffle", "Shuffle", "Randomize clip order"),
    "sequential": ("list", "Sequential", "Keep clips in original order"),
    "exquisite_corpus": ("text", "Exquisite Corpus", "Create poem from on-screen text"),  # NEW
}
```

**Add to `ui/tabs/sequence_tab.py` `_on_card_clicked()`:**

```python
def _on_card_clicked(self, algorithm_key: str):
    if algorithm_key == "exquisite_corpus":
        self._show_exquisite_corpus_dialog()
    else:
        # Existing algorithm handling
        ...

def _show_exquisite_corpus_dialog(self):
    from ui.dialogs.exquisite_corpus_dialog import ExquisiteCorpusDialog

    # Get selected clips (or all clips if none selected)
    clips = self._get_clips_for_sequencing()
    if len(clips) < 2:
        # Show warning
        return

    dialog = ExquisiteCorpusDialog(
        clips=clips,
        sources_by_id=self.project.sources_by_id,
        project=self.project,
        parent=self
    )
    dialog.sequence_ready.connect(self._apply_exquisite_corpus_sequence)
    dialog.exec()

def _apply_exquisite_corpus_sequence(self, sequence: list):
    """Apply the poem-ordered sequence to the timeline."""
    # sequence is List[Tuple[Clip, Source]]
    self._apply_sequence(sequence)
    self._switch_to_timeline_state()
```

**Add to `core/remix/__init__.py`:**

```python
def generate_sequence(
    clips: list,
    sources_by_id: dict,
    algorithm: str = "shuffle",
    **kwargs
) -> list:
    # ... existing code ...

    elif algorithm == "exquisite_corpus":
        # This algorithm is handled via dialog, not direct generation
        # Return clips as-is; the dialog handles poem generation
        return clips_to_use
```

### 7. Agent Tool Integration

**Add to `core/chat_tools.py`:**

```python
@tools.register(
    description="Extract on-screen text from video clips using OCR",
    requires_project=True,
    modifies_project_state=True
)
def extract_text_from_clips(
    main_window,
    project,
    clip_ids: list[str],
    num_keyframes: int = 3,
    use_vlm_fallback: bool = True
) -> dict:
    """
    Extract on-screen text from selected clips.

    Args:
        clip_ids: List of clip IDs to process
        num_keyframes: Number of frames to sample per clip (1-5)
        use_vlm_fallback: Use VLM if Tesseract fails or has low confidence

    Returns:
        Dictionary with extraction results per clip
    """
    valid_clips = [
        project.clips_by_id[cid]
        for cid in clip_ids
        if cid in project.clips_by_id
    ]

    if not valid_clips:
        return {"success": False, "error": "No valid clips found"}

    return {
        "_wait_for_worker": "text_extraction",
        "clip_ids": [c.id for c in valid_clips],
        "num_keyframes": min(max(1, num_keyframes), 5),
        "use_vlm_fallback": use_vlm_fallback
    }

@tools.register(
    description="Generate a poem from extracted clip texts and sequence clips accordingly",
    requires_project=True,
    modifies_gui_state=True
)
def generate_exquisite_corpus(
    main_window,
    project,
    clip_ids: list[str],
    mood_prompt: str
) -> dict:
    """
    Generate an Exquisite Corpus poem and sequence.

    Args:
        clip_ids: List of clip IDs that have extracted text
        mood_prompt: The mood/vibe for the poem

    Returns:
        Generated poem and sequence information
    """
    from core.remix.exquisite_corpus import generate_poem, sequence_by_poem

    # Get clips with text
    clips_with_text = []
    for clip_id in clip_ids:
        clip = project.clips_by_id.get(clip_id)
        if clip and clip.combined_text:
            clips_with_text.append((clip, clip.combined_text))

    if len(clips_with_text) < 2:
        return {
            "success": False,
            "error": f"Need at least 2 clips with text, found {len(clips_with_text)}"
        }

    try:
        poem_lines = generate_poem(clips_with_text, mood_prompt)
        sequence = sequence_by_poem(
            poem_lines,
            project.clips_by_id,
            project.sources_by_id
        )

        # Apply sequence to timeline via GUI
        main_window.sequence_tab._apply_sequence(sequence)

        return {
            "success": True,
            "poem": "\n".join(line.text for line in poem_lines),
            "line_count": len(poem_lines),
            "sequence_clip_count": len(sequence)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
```

## Acceptance Criteria

### Functional Requirements

- [x] New "Exquisite Corpus" card appears in Sequence tab algorithm grid
- [x] Card is disabled if fewer than 2 clips are selected/available
- [x] Clicking card opens multi-step dialog
- [x] Text extraction uses Tesseract (if available) with VLM fallback
- [x] 3 keyframes sampled per clip by default
- [x] User can enter mood/vibe prompt
- [x] LLM generates poem using ONLY exact extracted phrases
- [x] Poem preview shows all lines with ability to drag-drop reorder
- [x] User can regenerate poem with same or different mood
- [x] Final sequence matches poem line order
- [x] Clips without text are excluded from sequence
- [x] Extracted text is stored in Clip model and persists with project

### Non-Functional Requirements

- [x] Text extraction runs in background thread (UI remains responsive)
- [x] Progress shown during extraction (X/Y clips)
- [x] Graceful handling if Tesseract not installed (VLM-only mode)
- [x] LLM errors display user-friendly messages
- [x] Cancel button works at any stage

### Agent Integration

- [ ] `extract_text_from_clips` tool available to agent
- [ ] `generate_exquisite_corpus` tool available to agent
- [ ] Agent can discover clips with/without extracted text

## Dependencies

### New Python Dependencies

```
pytesseract>=0.3.10  # Tesseract OCR wrapper (optional - graceful degradation)
```

### System Dependencies

```
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt install tesseract-ocr

# Windows
# Download from https://github.com/UB-Mannheim/tesseract/wiki
```

**Note:** Tesseract is optional. If not installed, the feature falls back to VLM-only text extraction.

## File Changes Summary

| File | Action | Status | Description |
|------|--------|--------|-------------|
| `models/clip.py` | Modify | ✅ Done | Add `ExtractedText` dataclass and `extracted_texts` field |
| `core/analysis/ocr.py` | Create | ✅ Done | OCR text extraction module |
| `core/remix/exquisite_corpus.py` | Create | ✅ Done | Poem generation and sequencing |
| `ui/workers/text_extraction_worker.py` | Create | ✅ Done | Background OCR worker |
| `ui/dialogs/exquisite_corpus_dialog.py` | Create | ✅ Done | Multi-step workflow dialog |
| `ui/widgets/sorting_card_grid.py` | Modify | ✅ Done | Add "exquisite_corpus" algorithm card |
| `ui/tabs/sequence_tab.py` | Modify | ✅ Done | Handle exquisite_corpus card click |
| `core/remix/__init__.py` | Modify | N/A | Not needed - handled via dialog |
| `core/chat_tools.py` | Modify | ⏳ Pending | Add agent tools |
| `requirements.txt` | Modify | ✅ Done | Add pytesseract |

## Test Plan

### Unit Tests

- [ ] `test_ocr.py`: Test text extraction from sample frames
- [ ] `test_exquisite_corpus.py`: Test poem generation with mock LLM
- [ ] `test_clip_model.py`: Test ExtractedText serialization

### Integration Tests

- [ ] Full workflow: Select clips → Extract → Generate → Sequence
- [ ] VLM fallback when Tesseract unavailable
- [ ] Cancellation at each stage
- [ ] Project save/load with extracted text

### Manual Testing

- [ ] Various video types (titles, signs, documents, subtitles)
- [ ] Non-English text handling
- [ ] Low-quality/blurry text
- [ ] Clips with no text
- [ ] Large number of clips (20+)

## Design Decisions

### Why Hybrid OCR (Tesseract + VLM)?

1. **Tesseract** is fast, free, and works offline - perfect for clear printed text
2. **VLM fallback** handles stylized text, handwriting, and unusual fonts
3. Hybrid approach balances speed, cost, and accuracy

### Why 3 Keyframes Default?

- First frame: Catches opening titles/text
- Middle frame: Catches mid-clip text
- Last frame: Catches ending text
- More frames = more accuracy but slower processing

### Why Exact Phrase Constraint?

This creates a true "cut-up" aesthetic where the original text is preserved intact. The LLM acts as curator/arranger, not author. This distinguishes the feature from generic AI poetry.

## Future Enhancements

1. **Text region selection**: Let user draw box around text area to focus OCR
2. **Multi-language support**: Add language selection for Tesseract
3. **Phrase filtering**: Let user exclude specific phrases before poem generation
4. **Multiple poems**: Generate several poem variations to choose from
5. **Export poem**: Save generated poem as text file alongside video

## References

- [PySceneDetect Patterns](./2026-01-28-feat-sequence-tab-ui-redesign-plan.md)
- [Agent Tool Patterns](./2026-01-25-feat-agent-native-phases-2-3-4-plan.md)
- [Tesseract Documentation](https://tesseract-ocr.github.io/)
- [pytesseract PyPI](https://pypi.org/project/pytesseract/)
