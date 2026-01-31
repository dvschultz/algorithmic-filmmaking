# Separate Exquisite Corpus Model Setting from Chat Agent

## Problem

Currently, the Exquisite Corpus poem generation uses `settings.llm_model` (the Chat Agent's model). This creates coupling issues:

1. **Different use cases**: Chat agent needs a capable reasoning model for tool use. Poem generation is a creative task that may benefit from different models or settings.

2. **Cost considerations**: User might want a cheaper/faster model for chat (e.g., local Ollama) but use a more creative cloud model for poetry generation.

3. **Temperature mismatch**: Chat agent typically uses lower temperature for tool reliability. Poetry benefits from higher temperature (currently hardcoded to 0.8).

## Current State

```
Settings → Models → Chat Agent:
  - Default Provider: [Ollama, OpenAI, Anthropic, Gemini, OpenRouter]
  - Per-provider model selections

Settings → Models → Text Extraction:
  - Method: [Tesseract, VLM, Hybrid]
  - VLM Model: [gpt-4o, claude-sonnet-4, etc.]  ← Already separate!
  - Pre-filter checkbox

Exquisite Corpus:
  - Uses settings.llm_model (Chat Agent's model)
  - Hardcoded temperature=0.8
```

## Proposed Solution

Add a dedicated "Exquisite Corpus" or "Poetry Generation" section to settings.

### Option A: Simple - Single Model Dropdown

Add one dropdown for Exquisite Corpus model selection, similar to how Text Extraction VLM model works.

```
Settings → Models → Exquisite Corpus:
  - Model: [gpt-4o, gpt-5.2, claude-sonnet-4, gemini-2.5-flash, ...]
```

**Pros**: Simple, consistent with Text Extraction section
**Cons**: Limited to predefined models, no provider-level control

### Option B: Full - Provider + Model (Recommended)

Mirror the Chat Agent pattern with provider selection and per-provider models.

```
Settings → Models → Exquisite Corpus:
  - Provider: [OpenAI, Anthropic, Gemini, OpenRouter]  (no local - needs creativity)
  - OpenAI Model: [gpt-4o, gpt-5.2, ...]
  - Anthropic Model: [claude-sonnet-4, ...]
  - Gemini Model: [gemini-2.5-flash, gemini-2.5-pro, ...]
  - OpenRouter Model: [...]
```

**Pros**: Full flexibility, consistent with Chat Agent section
**Cons**: More UI complexity, more settings to manage

### Recommendation: Option A (Simple)

For a creative poetry task, a single model dropdown is sufficient. Users who want more control can still set environment variables or modify the code. This keeps the UI clean.

## Implementation Plan

### 1. Update `core/settings.py`

Add new settings fields:

```python
# Exquisite Corpus Settings
exquisite_corpus_model: str = "gpt-4o"  # Default to capable creative model
exquisite_corpus_temperature: float = 0.8  # Expose the temperature setting
```

Update `_load_from_json()` and `_settings_to_json()` to handle new fields.

### 2. Update `core/remix/exquisite_corpus.py`

Change model selection in `generate_poem()`:

```python
# Before:
model = model or settings.llm_model or "gpt-4o"

# After:
model = model or settings.exquisite_corpus_model or "gpt-4o"
temperature = settings.exquisite_corpus_temperature
```

### 3. Update `ui/settings_dialog.py`

Add Exquisite Corpus section to the Models tab (after Text Extraction):

```python
# Exquisite Corpus group
corpus_group = QGroupBox("Exquisite Corpus (Poetry Generation)")
corpus_layout = QVBoxLayout(corpus_group)

# Model selection
model_layout = QHBoxLayout()
model_label = QLabel("Model:")
model_label.setFixedWidth(UISizes.FORM_LABEL_WIDTH)
model_layout.addWidget(model_label)

self.corpus_model_combo = QComboBox()
self.corpus_model_combo.setMinimumHeight(UISizes.COMBO_BOX_MIN_HEIGHT)
# Same models as VLM list, plus some text-focused models
self.corpus_model_combo.addItems([
    "gpt-4o",
    "gpt-5.2",
    "gpt-5",
    "claude-sonnet-4-5-20250929",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
])
self.corpus_model_combo.setToolTip(
    "Model for generating poems from extracted text.\n"
    "Creative models with good instruction-following work best."
)
model_layout.addWidget(self.corpus_model_combo)
model_layout.addStretch()
corpus_layout.addLayout(model_layout)

# Temperature slider (optional - could keep hardcoded)
temp_layout = QHBoxLayout()
temp_label = QLabel("Creativity:")
temp_label.setFixedWidth(UISizes.FORM_LABEL_WIDTH)
temp_layout.addWidget(temp_label)

self.corpus_temp_slider = StyledSlider(Qt.Horizontal)
self.corpus_temp_slider.setRange(0, 100)  # 0.0 to 1.0
self.corpus_temp_slider.setValue(80)  # Default 0.8
self.corpus_temp_slider.setToolTip(
    "Higher values = more creative/unpredictable\n"
    "Lower values = more focused/deterministic"
)
temp_layout.addWidget(self.corpus_temp_slider)

self.corpus_temp_value = QLabel("0.8")
self.corpus_temp_value.setFixedWidth(30)
temp_layout.addWidget(self.corpus_temp_value)
corpus_layout.addLayout(temp_layout)

layout.addWidget(corpus_group)
```

### 4. Wire up settings load/save

In `_load_settings()`:
```python
# Exquisite Corpus
idx = self.corpus_model_combo.findText(self.settings.exquisite_corpus_model)
if idx >= 0:
    self.corpus_model_combo.setCurrentIndex(idx)
self.corpus_temp_slider.setValue(int(self.settings.exquisite_corpus_temperature * 100))
```

In `_save_to_settings()`:
```python
# Exquisite Corpus
self.settings.exquisite_corpus_model = self.corpus_model_combo.currentText()
self.settings.exquisite_corpus_temperature = self.corpus_temp_slider.value() / 100.0
```

### 5. Update `original_settings` and `has_changes()`

Add the new fields to track changes properly.

### 6. Add API key validation

Similar to VLM warning label, show warning if selected model requires an API key that isn't configured.

## Files to Modify

| File | Changes |
|------|---------|
| `core/settings.py` | Add `exquisite_corpus_model`, `exquisite_corpus_temperature` |
| `core/remix/exquisite_corpus.py` | Use new settings instead of `llm_model` |
| `ui/settings_dialog.py` | Add Exquisite Corpus section with model dropdown and temperature |

## Testing

1. Open Settings → Models
2. Verify Exquisite Corpus section appears after Text Extraction
3. Change model and temperature settings
4. Run Exquisite Corpus workflow
5. Verify the selected model is used (check logs)
6. Verify settings persist after restart

## Future Considerations

- Could add "Use Chat Agent model" checkbox to fall back to current behavior
- Could add per-provider model selection if users request more flexibility
- Could expose max_tokens or other parameters for advanced users
