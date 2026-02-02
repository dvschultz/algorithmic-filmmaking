# VideoMAE Shot Type Classification Upgrade

## Goal

Replace the current CLIP-based shot type detection with a VideoMAE model fine-tuned for cinematographic shot classification. This provides temporal awareness (analyzes 16 frames vs 1) and better accuracy (~80% vs current inconsistent results).

## Problem

Current implementation (`core/analysis/shots.py`):
- Uses CLIP zero-shot classification on single frames
- Not trained for cinematographic shot types
- No temporal/video awareness
- Only 4 shot types: wide, medium, close-up, extreme close-up
- User reports: "very inaccurate"

## Solution: VideoMAE Cloud Deployment

The user's VideoMAE model requires GPU inference and processes 16-frame video clips. Since VideoMAE is not available on standard inference APIs, we need to deploy it ourselves.

### Recommended Approach: Replicate

**Why Replicate:**
- Simple deployment via Cog (Docker-based)
- Pay-per-use ($0.000225/sec on T4 GPU)
- No infrastructure management
- Easy API integration
- Supports custom models

**Alternative Options:**
| Platform | Pros | Cons |
|----------|------|------|
| **Replicate** | Simple, pay-per-use, good docs | ~$0.8/hr T4 |
| **Modal** | Python-native, fast cold starts | More complex setup |
| **HF Endpoints** | Native HF integration | $1.30/hr T4, overkill for this |
| **RunPod Serverless** | Cheap GPUs | More DIY |

### Model Details (from notebook)

- **Architecture**: VideoMAEForVideoClassification (transformers)
- **Input**: 16 frames sampled from video clip
- **Output**: 5 shot type classes
- **Model size**: ~345MB
- **GPU requirement**: T4 sufficient, V100/A100 for speed
- **Accuracy**: ~80% (per notebook)

### Shot Type Mapping

| Notebook Label | Current Label | Description |
|----------------|---------------|-------------|
| LS (Long Shot) | wide shot | Full scene, environment focus |
| FS (Full Shot) | - (new) | Full body visible |
| MS (Medium Shot) | medium shot | Waist up |
| CS (Close-up Shot) | close-up | Face/detail focus |
| ECS (Extreme Close-up) | extreme close-up | Tight detail |

## Implementation Plan

### Phase 1: Model Packaging for Replicate

- [x] Create Replicate account and install Cog CLI
- [x] Download model weights from Google Drive
- [x] Create `cog.yaml` with GPU environment (CUDA, PyTorch, transformers, av)
- [x] Write `predict.py` with Predictor class:
  - Input: video file (URL or base64)
  - Output: shot_type, confidence, all_scores
- [ ] Test locally with `cog predict`
- [ ] Push to Replicate: `cog push`

### Phase 2: Integration into Scene Ripper

- [x] Add Replicate API key to settings (`core/settings.py`)
- [x] Add Replicate key storage to keyring (`get_replicate_api_key()`)
- [x] Update `ui/settings_dialog.py` with Replicate API key field
- [x] Create new `core/analysis/shots_cloud.py`:
  - `classify_shot_replicate(clip_path: Path) -> tuple[str, float]`
  - Handle video extraction (reuse `extract_clip_segment` from description.py)
  - Call Replicate API
  - Return shot type and confidence
- [x] Update `core/analysis/shots.py`:
  - Add tier system like description.py (cpu/cloud)
  - CPU tier: keep current CLIP (free fallback)
  - Cloud tier: use Replicate VideoMAE
- [x] Update settings for shot classification tier choice
- [x] Update `ui/settings_dialog.py` with tier dropdown

### Phase 3: UI and Worker Updates

- [x] Update `ShotTypeWorker` in `ui/main_window.py` to use tiered classification
- [x] Add progress reporting for cloud inference
- [x] Handle API errors gracefully (timeout, rate limit) - falls back to CPU tier
- [x] Update shot type labels to 5-class system (wide, full, medium, close-up, extreme close-up)

### Phase 4: Testing and Polish

- [ ] Test with various video types
- [ ] Verify accuracy improvement over CLIP
- [x] Add logging for debugging
- [ ] Document API costs in README

## Files to Create/Modify

### New Files (Created)

1. **`replicate/shot-classifier/cog.yaml`** - Replicate model config ✅
2. **`replicate/shot-classifier/predict.py`** - Inference code for Replicate ✅
3. **`core/analysis/shots_cloud.py`** - Replicate API client ✅

### Modified Files (Completed)

1. **`core/settings.py`** - Add Replicate key, shot tier setting ✅
2. **`ui/settings_dialog.py`** - Add Replicate key field, tier dropdown ✅
3. **`core/analysis/shots.py`** - Add tier routing (`classify_shot_type_tiered`) ✅
4. **`ui/main_window.py`** - Update `ShotTypeWorker` to use tiered classification ✅

## Replicate Model Code

### cog.yaml

```yaml
build:
  gpu: true
  cuda: "11.8"
  python_version: "3.10"
  python_packages:
    - torch==2.0.1
    - torchvision==0.15.2
    - transformers==4.35.0
    - av==10.0.0
    - numpy==1.24.0

predict: "predict.py:Predictor"
```

### predict.py

```python
import tempfile
from pathlib import Path
from cog import BasePredictor, Input, Path as CogPath
import torch
import av
import numpy as np
from transformers import AutoImageProcessor, VideoMAEForVideoClassification

class Predictor(BasePredictor):
    def setup(self):
        """Load model into memory."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = "./model"  # Bundled with container
        self.processor = AutoImageProcessor.from_pretrained(model_path)
        self.model = VideoMAEForVideoClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    def read_video_pyav(self, container, indices):
        """Decode video frames at specified indices."""
        frames = []
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)
        return np.stack([x.to_ndarray(format="rgb24") for x in frames])

    def sample_frame_indices(self, clip_len, frame_sample_rate, seg_len):
        """Sample frame indices for VideoMAE."""
        converted_len = int(clip_len * frame_sample_rate)
        if seg_len < converted_len:
            # Not enough frames - sample what we have
            indices = np.linspace(0, seg_len - 1, num=clip_len)
        else:
            end_idx = np.random.randint(converted_len, seg_len)
            start_idx = end_idx - converted_len
            indices = np.linspace(start_idx, end_idx, num=clip_len)
        return np.clip(indices, 0, seg_len - 1).astype(np.int64)

    def predict(self, video: CogPath = Input(description="Video file")) -> dict:
        """Classify shot type from video."""
        container = av.open(str(video))
        stream = container.streams.video[0]
        total_frames = stream.frames or int(stream.duration * stream.average_rate)

        indices = self.sample_frame_indices(
            clip_len=16, frame_sample_rate=1, seg_len=total_frames
        )
        video_frames = self.read_video_pyav(container, indices)

        inputs = self.processor(list(video_frames), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        probs = torch.softmax(logits, dim=-1)[0]
        predicted_idx = logits.argmax(-1).item()
        predicted_label = self.model.config.id2label[predicted_idx]
        confidence = probs[predicted_idx].item()

        return {
            "shot_type": predicted_label,
            "confidence": confidence,
            "scores": {
                self.model.config.id2label[i]: probs[i].item()
                for i in range(len(probs))
            }
        }
```

## Integration Code

### core/analysis/shots_cloud.py

```python
"""Cloud-based shot type classification using Replicate."""

import logging
import tempfile
from pathlib import Path
from typing import Optional
import replicate

from core.settings import load_settings, get_replicate_api_key
from core.analysis.description import extract_clip_segment

logger = logging.getLogger(__name__)

# Map model labels to display labels
SHOT_TYPE_LABELS = {
    "LS": "Long Shot",
    "FS": "Full Shot",
    "MS": "Medium Shot",
    "CS": "Close-up",
    "ECS": "Extreme Close-up",
}

def classify_shot_replicate(
    source_path: Path,
    start_frame: int,
    end_frame: int,
    fps: float,
) -> tuple[str, float, dict]:
    """Classify shot type using VideoMAE on Replicate.

    Args:
        source_path: Path to source video
        start_frame: Clip start frame
        end_frame: Clip end frame
        fps: Video frame rate

    Returns:
        Tuple of (shot_type, confidence, all_scores)
    """
    api_key = get_replicate_api_key()
    if not api_key:
        raise ValueError("Replicate API key not configured")

    # Extract clip segment
    temp_video = extract_clip_segment(source_path, start_frame, end_frame, fps)

    try:
        client = replicate.Client(api_token=api_key)

        # Run prediction (replace with actual model version)
        output = client.run(
            "username/videomae-shot-classifier:version",
            input={"video": open(temp_video, "rb")}
        )

        shot_type = output["shot_type"]
        confidence = output["confidence"]
        scores = output["scores"]

        # Map to display label
        display_label = SHOT_TYPE_LABELS.get(shot_type, shot_type)

        return display_label, confidence, scores

    finally:
        if temp_video.exists():
            temp_video.unlink()
```

## Cost Estimate

| Usage | Clips/month | Inference time | Monthly cost |
|-------|-------------|----------------|--------------|
| Light | 100 | ~2 sec each | ~$0.05 |
| Medium | 1,000 | ~2 sec each | ~$0.45 |
| Heavy | 10,000 | ~2 sec each | ~$4.50 |

Replicate T4 GPU: $0.000225/second

## Verification

1. Deploy model to Replicate
2. Test with 10 video clips of known shot types
3. Compare accuracy to current CLIP method
4. Verify API integration works in Scene Ripper
5. Confirm error handling (API down, timeout, invalid video)

## Dependencies

Add to requirements.txt:
```
replicate>=0.20.0
```

## Notes

- Model weights need to be downloaded from user's Google Drive and bundled with Replicate container
- The model was fine-tuned by the user - ensure proper attribution
- Consider caching results to avoid re-classification
- Clips shorter than 16 frames will use interpolation (may be less accurate)
