"""Lightweight model versions shared by analysis identities and lazy loaders."""

from core.analysis_records import model_runtime

OCR_PROMPT = """Extract ALL visible text from this image. Include:
- Signs, labels, titles
- Subtitles or captions
- Text on documents or screens
- Any other readable text

Return ONLY the extracted text, one phrase per line. If no text is visible, return "NO_TEXT_FOUND".
Do not add any commentary or descriptions."""


def ocr_runtime() -> dict:
    from hashlib import sha256
    from pathlib import Path
    from core.binary_resolver import find_binary
    from core.jobs.media import media_stamp

    binary = find_binary("ffmpeg")
    return {
        "algorithm": "ocr-half-open-keyframes/v1",
        "paddle_model": "package-default/en",
        "use_angle_cls": True,
        "prompt_sha256": sha256(OCR_PROMPT.encode()).hexdigest(),
        "packages": model_runtime(
            "ocr", ("paddleocr", "paddlepaddle", "numpy", "Pillow", "litellm")
        )["packages"],
        "ffmpeg": str(binary) if binary else None,
        "ffmpeg_stamp": list(media_stamp(Path(binary)) or ()) if binary else None,
    }


def object_detection_runtime() -> dict:
    runtime = model_runtime("yolo26n", ("torch", "ultralytics"))
    return {
        "model": "yolo26n",
        "weights": "yolo26n.pt",
        "weights_release": "ultralytics/assets/v8.4.0",
        "vocabulary": "COCO-80",
        "packages": runtime["packages"],
    }


def classification_runtime() -> dict:
    return {
        "model": "mobilenet_v3_small",
        "weights": "IMAGENET1K_V1",
        "vocabulary": "weight_metadata",
        "packages": model_runtime("mobilenet", ("torch", "torchvision"))["packages"],
    }


DINOV2_NAME = "facebook/dinov2-base"
# https://huggingface.co/facebook/dinov2-base/commits/main
DINOV2_REVISION = "f9e44c814b77203eaa57a6bdbbd535f21ede1415"
DINOV2_DIMENSIONS = 768
DINOV2_TAG = "dinov2-vit-b-14"


def embedding_runtime() -> dict:
    return {
        "model": DINOV2_NAME,
        "revision": DINOV2_REVISION,
        "tag": DINOV2_TAG,
        "dimensions": DINOV2_DIMENSIONS,
        "algorithm": "cls-l2/v1",
        "packages": model_runtime("dinov2", ("torch", "transformers", "Pillow"))[
            "packages"
        ],
    }
