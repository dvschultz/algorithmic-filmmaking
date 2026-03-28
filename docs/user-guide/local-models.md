# Local AI Models

Scene Ripper runs several AI models locally on your machine for analysis features. No data leaves your computer when using these models — everything runs offline.

Models are downloaded automatically the first time you use a feature that requires them. On macOS app builds, some models are bundled with the app and others are downloaded during first-run setup.

---

## Vision & Image Analysis

### SigLIP 2 — Shot Type Classification

Classifies each clip's thumbnail into one of five shot types: wide shot, full shot, medium shot, close-up, or extreme close-up.

SigLIP 2 is a contrastive vision-language model from Google. Instead of generating text, it scores how well an image matches a set of text descriptions. Scene Ripper compares each thumbnail against multiple descriptions per shot type and picks the best match. This approach is fast (~50ms per image) and reliable since it produces numerical confidence scores rather than text that needs parsing.

**Model:** [google/siglip2-base-patch16-224](https://huggingface.co/google/siglip2-base-patch16-224) (~400 MB)
**Used in:** Analyze tab > Classify Shots (local tier)
**Try it:** [SigLIP 2 blog post with examples](https://huggingface.co/blog/siglip2)

---

### DINOv2 — Visual Embeddings

Generates a 768-dimensional embedding vector for each clip's thumbnail. These embeddings capture the visual content of a frame — similar-looking frames produce similar vectors. Several sequencer algorithms use these embeddings to find visual relationships between clips (match cuts, visual similarity sorting, etc.).

DINOv2 is a self-supervised vision transformer from Meta. It was trained without text labels, learning visual features purely from images. This makes it especially good at capturing structural and compositional similarity rather than semantic categories.

**Model:** [facebook/dinov2-base](https://huggingface.co/facebook/dinov2-base) (~330 MB)
**Used in:** Analyze tab > Compute Embeddings; Sequence tab (Match Cut, Similarity Sort, Exquisite Corpus)
**Try it:** [DINOv2 Demo Lab](https://dinov2.metademolab.com/)

---

### Qwen3-VL — Scene Description & Cinematography (Apple Silicon)

Generates free-form text descriptions of what's happening in each clip, and performs structured cinematography analysis (shot size, camera angle, lighting, composition, etc.).

Qwen3-VL is a vision-language model from Alibaba's Qwen team. Scene Ripper uses a 4-bit quantized version optimized for Apple Silicon via MLX, so it runs efficiently on M1/M2/M3/M4 Macs without needing a GPU. It can answer arbitrary visual questions, which is why it's also used for custom queries and the cinematography analysis pipeline.

**Model:** [mlx-community/Qwen3-VL-4B-Instruct-4bit](https://huggingface.co/mlx-community/Qwen3-VL-4B-Instruct-4bit) (~2.9 GB)
**Used in:** Analyze tab > Describe (local tier), Cinematography Analysis (local tier), Custom Queries
**Try it:** [Qwen3-VL Demo](https://huggingface.co/spaces/Qwen/Qwen3-VL-Demo)

> **Note:** The cinematography analysis can optionally use the larger [Qwen2.5-VL-7B](https://huggingface.co/mlx-community/Qwen2.5-VL-7B-Instruct-4bit) model for richer results. You can change this in Settings.

---

### Moondream2 — Scene Description (CPU Fallback)

Serves the same role as Qwen3-VL but runs on any machine (Intel Macs, Windows, Linux) using standard PyTorch. It's smaller and less capable than Qwen3-VL, but works everywhere without Apple Silicon.

Moondream is a compact vision-language model designed for efficiency. Scene Ripper automatically falls back to Moondream when MLX (Apple Silicon acceleration) is not available.

**Model:** [vikhyatk/moondream2](https://huggingface.co/vikhyatk/moondream2) (~3.7 GB)
**Used in:** Analyze tab > Describe (local tier, non-Apple-Silicon only)
**Try it:** [Moondream Playground](https://moondream.ai/playground) | [HuggingFace Space](https://huggingface.co/spaces/vikhyatk/moondream2)

---

### YOLO — Object Detection

Detects and labels objects in each clip's thumbnail (people, cars, animals, furniture, etc.). Scene Ripper uses two YOLO variants:

- **YOLO26n** — fixed-vocabulary detection for 80 common object categories (COCO classes). Fast and lightweight.
- **YOLOE-26s** — open-vocabulary detection that can find objects matching custom text queries you provide.

YOLO (You Only Look Once) is an object detection architecture from Ultralytics. It processes the entire image in a single pass, making it much faster than older two-stage detectors.

**Models:** yolo26n.pt (~6 MB), yoloe-26s.pt (~30 MB)
**Used in:** Analyze tab > Detect Objects
**Try it:** [Ultralytics YOLOv8 Demo](https://huggingface.co/spaces/Ultralytics/YOLOv8)

---

### InsightFace — Face Detection & Embeddings

Detects faces in each clip's thumbnail and generates a 512-dimensional face embedding for each detected face. These embeddings allow the app to match the same person across different clips, enabling person-based filtering and sequencing.

InsightFace's buffalo_l model pack includes SCRFD for face detection and ArcFace for face recognition. Face embeddings are compared using cosine similarity — the same person in different clips will have similar embedding vectors.

**Model:** buffalo_l ONNX pack (~200 MB)
**Used in:** Analyze tab > Detect Faces
**Try it:** [InsightFace Face Localization Demo](http://demo.insightface.ai:7007/)

> **Note:** InsightFace requires a C/C++ compiler (Xcode Command Line Tools on macOS) for installation.

---

### MobileNetV3 — Image Classification

Classifies the general content of each clip's thumbnail using ImageNet's 1,000 categories (e.g., "beach", "mountain", "restaurant", "concert"). This provides broad scene-level tags.

MobileNetV3 is a lightweight classification model from Google designed for mobile and edge devices. It's small and fast while still being accurate enough for general-purpose tagging.

**Model:** MobileNetV3-Small with IMAGENET1K_V1 weights (via torchvision, ~15 MB)
**Used in:** Analyze tab > Classify Images

---

### PaddleOCR — Text Extraction

Detects and reads on-screen text in each clip's thumbnail (titles, signs, subtitles, watermarks, etc.). The extracted text is stored on each clip and can be used for search and filtering.

PaddleOCR is an open-source OCR system from Baidu. It uses three models working together: a text detector (finds text regions), a text recognizer (reads the characters), and an angle classifier (handles rotated text).

**Model:** PP-OCRv5 detection + recognition + classification models (~100 MB total)
**Used in:** Analyze tab > Extract Text (OCR)
**Try it:** [PP-OCRv5 Online Demo](https://huggingface.co/spaces/PaddlePaddle/PP-OCRv5_Online_Demo)

---

## Audio Analysis

### Whisper — Speech Transcription

Transcribes spoken dialogue in your video clips. Scene Ripper supports multiple Whisper model sizes — larger models are more accurate but slower.

Whisper is OpenAI's speech recognition model, trained on 680,000 hours of multilingual audio. Scene Ripper uses two backends depending on your hardware:

- **lightning-whisper-mlx** — Apple Silicon optimized, uses the GPU via MLX. This is the default on M1/M2/M3/M4 Macs.
- **faster-whisper** — CPU/CUDA optimized using CTranslate2. Used on Intel Macs, Windows, and Linux.

| Model Size | Download | Speed | Accuracy |
|------------|----------|-------|----------|
| tiny.en | 39 MB | Fastest | Basic |
| small.en | 244 MB | Fast | Good |
| medium.en | 769 MB | Moderate | Better |
| large-v3 | 1.5 GB | Slowest | Best |

**Used in:** Analyze tab > Transcribe
**Try it:** [Whisper Web (runs in browser)](https://huggingface.co/spaces/Xenova/whisper-web) | [Whisper large-v3 Space](https://huggingface.co/spaces/hf-audio/whisper-large-v3)

> **Note:** The macOS app bundles only medium.en for local transcription. You can also use cloud transcription (Groq) for faster results with any model size.

---

### Demucs — Stem Separation

Separates audio into four stems: vocals, drums, bass, and other instruments. This enables analysis features that work on isolated audio components (e.g., beat detection on drums only, or volume analysis on vocals only).

Demucs is Meta's music source separation model. The htdemucs variant uses a hybrid transformer architecture that works in both time and frequency domains for high-quality separation.

**Model:** htdemucs (~320 MB)
**Used in:** Analyze tab > Separate Stems
**Try it:** [Music Separation Demo](https://huggingface.co/spaces/abidlabs/music-separation)

---

### librosa — Beat & Onset Detection

Detects beats, onsets (note attacks), and rhythmic patterns in audio. Unlike the other tools on this page, librosa is not an AI model — it uses traditional digital signal processing algorithms. No model weights are downloaded.

**Used in:** Analyze tab > Analyze Audio; Sequence tab (Beat Sync algorithm)

---

## Storage & Memory

Models are cached after first download so subsequent launches are fast. Cache locations:

- **macOS:** `~/Library/Application Support/Scene Ripper/models/`
- **Windows:** `%LOCALAPPDATA%\Scene Ripper\models\`
- **Linux:** `~/.local/share/Scene Ripper/models/`

Only models for features you actually use are downloaded. Total disk usage depends on which features you enable:

| If you use... | Approximate download |
|---------------|---------------------|
| Shot classification only | ~400 MB (SigLIP 2) |
| Full analysis suite (Apple Silicon) | ~4-5 GB |
| Full analysis suite (other platforms) | ~6-7 GB |

Models are loaded into RAM only when needed and unloaded when analysis completes. Peak memory usage depends on which models are active simultaneously.
