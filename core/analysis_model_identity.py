"""Lightweight model versions shared by analysis identities and lazy loaders."""

from core.analysis_records import model_runtime


def object_detection_runtime() -> dict:
    runtime = model_runtime("yolo26n", ("torch", "ultralytics"))
    return {
        "model": "yolo26n", "weights": "yolo26n.pt",
        "weights_release": "ultralytics/assets/v8.4.0",
        "vocabulary": "COCO-80", "packages": runtime["packages"],
    }

DINOV2_NAME = "facebook/dinov2-base"
# https://huggingface.co/facebook/dinov2-base/commits/main
DINOV2_REVISION = "f9e44c814b77203eaa57a6bdbbd535f21ede1415"
DINOV2_DIMENSIONS = 768
DINOV2_TAG = "dinov2-vit-b-14"


def embedding_runtime() -> dict:
    return {
        "model": DINOV2_NAME, "revision": DINOV2_REVISION,
        "tag": DINOV2_TAG, "dimensions": DINOV2_DIMENSIONS,
        "algorithm": "cls-l2/v1",
        "packages": model_runtime("dinov2", ("torch", "transformers", "Pillow"))["packages"],
    }
