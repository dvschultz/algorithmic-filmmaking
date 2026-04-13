"""Central registry of analysis operations.

Qt-free module defining all available analysis operations with metadata
for UI rendering, phase-based execution ordering, and settings persistence.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AnalysisOperation:
    """Defines a single analysis operation."""

    key: str  # Unique identifier: "colors", "shots", etc.
    label: str  # UI display name: "Extract Colors"
    tooltip: str  # One-line description for tooltips
    phase: str  # Execution phase: "local" | "sequential" | "cloud"
    default_enabled: bool  # Pre-checked by default in picker
    cloud_capable: bool = False  # True if the operation can use a cloud API


# All operations in display order
ANALYSIS_OPERATIONS: list[AnalysisOperation] = [
    AnalysisOperation(
        "colors", "Extract Colors",
        "Extract dominant colors from clip thumbnails",
        "local", True,
    ),
    AnalysisOperation(
        "shots", "Classify Shots",
        "Classify shot types (close-up, medium, wide, etc.)",
        "local", True, cloud_capable=True,
    ),
    AnalysisOperation(
        "classify", "Classify Content",
        "Classify frame content using ImageNet labels (dog, car, tree, etc.)",
        "local", False,
    ),
    AnalysisOperation(
        "detect_objects", "Detect Objects",
        "Detect and locate objects using YOLO with bounding boxes and person count",
        "local", False,
    ),
    AnalysisOperation(
        "extract_text", "Extract Text",
        "Extract visible text from frames using OCR (titles, labels, captions)",
        "local", False, cloud_capable=True,
    ),
    AnalysisOperation(
        "transcribe", "Transcribe",
        "Transcribe speech in clips using Whisper",
        "sequential", True, cloud_capable=True,
    ),
    AnalysisOperation(
        "describe", "Describe",
        "Generate AI descriptions of frame content using a vision model",
        "cloud", False, cloud_capable=True,
    ),
    AnalysisOperation(
        "cinematography", "Rich Analysis",
        "Comprehensive film language analysis (shot size, camera angle, movement, lighting)",
        "cloud", False, cloud_capable=True,
    ),
    AnalysisOperation(
        "face_embeddings", "Detect Faces",
        "Extract face embeddings for person identification",
        "sequential", False,
    ),
    AnalysisOperation(
        "gaze", "Detect Gaze",
        "Estimate gaze direction using iris landmarks",
        "sequential", False,
    ),
    AnalysisOperation(
        "embeddings", "Generate Embeddings",
        "Extract DINOv2 visual feature vectors for similarity-based sequencing",
        "sequential", False,
    ),
    AnalysisOperation(
        "custom_query", "Custom Query",
        "Search clips for specific visual content using a VLM (e.g., 'blue flower')",
        "cloud", False, cloud_capable=True,
    ),
]

# Lookup by key
OPERATIONS_BY_KEY: dict[str, AnalysisOperation] = {
    op.key: op for op in ANALYSIS_OPERATIONS
}

# Phase groupings
LOCAL_OPS: list[str] = [op.key for op in ANALYSIS_OPERATIONS if op.phase == "local"]
SEQUENTIAL_OPS: list[str] = [op.key for op in ANALYSIS_OPERATIONS if op.phase == "sequential"]
CLOUD_OPS: list[str] = [op.key for op in ANALYSIS_OPERATIONS if op.phase == "cloud"]

# Default selection for new users / settings reset
DEFAULT_SELECTED: list[str] = [op.key for op in ANALYSIS_OPERATIONS if op.default_enabled]

# Ordered phase list for pipeline execution
PHASE_ORDER: list[str] = ["local", "sequential", "cloud"]
