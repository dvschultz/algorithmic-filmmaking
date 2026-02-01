# VideoMAE Shot Type Classifier - Replicate Deployment

Deploy your fine-tuned VideoMAE shot type classifier to Replicate for cloud inference.

## Prerequisites

1. **Replicate account**: Sign up at [replicate.com](https://replicate.com)
2. **Cog CLI**: Install with `brew install cog` or `pip install cog`
3. **Model weights**: Download from your Google Drive

## Setup

### 1. Download Model Weights

Download your fine-tuned model from Google Drive and place in this directory:

```
replicate/shot-classifier/
├── cog.yaml
├── predict.py
├── README.md
└── model/                    # Create this directory
    ├── config.json
    ├── preprocessor_config.json
    └── pytorch_model.bin     # ~345MB
```

The model directory should contain all files from your HuggingFace-format model:
- `config.json` - Model configuration
- `preprocessor_config.json` - Image processor config
- `pytorch_model.bin` - Model weights (~345MB)

### 2. Test Locally

```bash
cd replicate/shot-classifier

# Build the container
cog build

# Test with a sample video
cog predict -i video=@/path/to/test_clip.mp4
```

Expected output:
```json
{
  "shot_type": "MS",
  "shot_type_display": "Medium Shot",
  "confidence": 0.8734
}
```

### 3. Deploy to Replicate

```bash
# Login to Replicate
cog login

# Push to Replicate (creates model if needed)
cog push r8.im/YOUR_USERNAME/shot-type-classifier
```

After pushing, you'll get a model URL like:
`https://replicate.com/YOUR_USERNAME/shot-type-classifier`

### 4. Get Model Version

After deployment, note the version hash from Replicate dashboard. You'll need this for API calls:
`YOUR_USERNAME/shot-type-classifier:VERSION_HASH`

## Usage

### Python API

```python
import replicate

output = replicate.run(
    "YOUR_USERNAME/shot-type-classifier:VERSION",
    input={"video": open("clip.mp4", "rb")}
)

print(output)
# {"shot_type": "CS", "shot_type_display": "Close-up", "confidence": 0.91}
```

### With all scores

```python
output = replicate.run(
    "YOUR_USERNAME/shot-type-classifier:VERSION",
    input={
        "video": open("clip.mp4", "rb"),
        "return_all_scores": True
    }
)

print(output["all_scores"])
# {"LS": 0.02, "FS": 0.05, "MS": 0.12, "CS": 0.72, "ECS": 0.09}
```

## Shot Types

| Code | Name | Description |
|------|------|-------------|
| LS | Long Shot | Wide/establishing shot, environment dominates |
| FS | Full Shot | Full body visible, head to toe |
| MS | Medium Shot | Waist up, conversational distance |
| CS | Close-up | Head and shoulders |
| ECS | Extreme Close-up | Face detail, eyes or mouth |

## Costs

Replicate T4 GPU: ~$0.000225/second

Typical inference: ~2 seconds = **~$0.0005/clip**

## Troubleshooting

### "Model not found"
Ensure model files are in `model/` directory before building.

### Low confidence scores
- Clips should be at least 16 frames
- Very short clips may have interpolated frames

### CUDA errors
Check GPU memory. Model requires ~2GB VRAM.

## Integration

See `docs/plans/2026-02-01-feat-videomae-shot-type-classification-plan.md` for Scene Ripper integration details.
