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


# Shot type categories for zero-shot classification
# Matches VideoMAE categories: LS, FS, MS, CS, ECS
SHOT_TYPES = [
    "wide shot",  # LS - Long Shot
    "full shot",  # FS - Full Shot (full body visible)
    "medium shot",  # MS - Medium Shot (waist up)
    "close-up",  # CS - Close-up (head and shoulders)
    "extreme close-up",  # ECS - Extreme Close-up (face detail)
]

# Detailed prompts for SigLIP 2 zero-shot classification
# SigLIP uses sigmoid per-label (not softmax), so prompts should be
# self-contained descriptions that work independently
SHOT_TYPE_PROMPTS = {
    "wide shot": [
        "This is a photo of an establishing shot showing a vast landscape or cityscape.",
        "This is a photo of a long shot where people appear very small in the environment.",
        "This is a photo of a wide angle shot of a large space with tiny distant figures.",
        "This is a photo of a panoramic view showing the entire location.",
    ],
    "full shot": [
        "This is a photo of a shot showing one person's entire body from head to feet.",
        "This is a photo of a single person standing with their full body visible in frame.",
        "This is a photo of a full length portrait of someone from head to toe.",
        "This is a photo of a shot framing one standing figure completely.",
    ],
    "medium shot": [
        "This is a photo of a medium shot showing a person from the waist up to their head.",
        "This is a photo of two or three people shown from the waist up in conversation.",
        "This is a photo of a shot of people sitting at a table showing their upper bodies.",
        "This is a photo of a cowboy shot showing someone from mid-thigh to head.",
    ],
    "close-up": [
        "This is a photo of a close-up of a person's face filling most of the frame.",
        "This is a photo of a head and shoulders shot focusing on facial expression.",
        "This is a photo of a tight shot of someone's face showing emotion.",
        "This is a photo of a portrait shot from the neck up.",
    ],
    "extreme close-up": [
        "This is a photo of an extreme close-up showing only eyes filling the screen.",
        "This is a photo of a shot of just lips or mouth in extreme detail.",
        "This is a photo of a macro shot of a single facial feature like an eye.",
        "This is a photo of an intense close-up where only part of a face is visible.",
    ],
}

SIGLIP_NAME = "google/siglip2-base-patch16-224"
# https://huggingface.co/google/siglip2-base-patch16-224/commit/75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2
SIGLIP_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
SHOT_DEFAULT_CLOUD_MODEL = "gemini-3.1-flash-lite-preview"
SHOT_CLOUD_PROMPT = (
    "Classify this film frame into exactly one shot type. "
    "Valid types: " + ", ".join(f'"{shot_type}"' for shot_type in SHOT_TYPES) + ".\n\n"
    "Return ONLY a JSON object: "
    '{"shot_type": "<type>", "confidence": <0.0-1.0>}'
)


def shot_runtime() -> dict:
    return {
        "model": SIGLIP_NAME,
        "revision": SIGLIP_REVISION,
        "vocabulary": list(SHOT_TYPES),
        "prompts": {key: list(prompts) for key, prompts in SHOT_TYPE_PROMPTS.items()},
        "cloud": {
            "default_model": SHOT_DEFAULT_CLOUD_MODEL,
            "prompt": SHOT_CLOUD_PROMPT,
            "temperature": 0.0,
            "max_tokens": 100,
            "fallback": "local",
        },
        "packages": model_runtime(
            "shots", ("torch", "transformers", "Pillow", "litellm")
        )["packages"],
    }
