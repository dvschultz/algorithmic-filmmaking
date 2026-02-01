#!/usr/bin/env python3
"""Compare different shot type classification methods.

Usage:
    python scripts/compare_shot_classifiers.py [--images DIR] [--limit N]

Methods compared:
    1. Current CLIP (baseline) - simple prompts
    2. Improved CLIP - cinematography-aware ensemble prompts
    3. VLM (GPT-5.2) - vision language model (requires API key)
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# Method 1: Current CLIP (baseline)
# ============================================================================

BASELINE_SHOT_TYPES = ["wide shot", "medium shot", "close-up", "extreme close-up"]


def classify_baseline_clip(image_path: Path) -> tuple[str, float]:
    """Current implementation - simple prompts."""
    import torch
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    image = Image.open(image_path).convert("RGB")
    text_prompts = [f"a {shot_type} of a scene" for shot_type in BASELINE_SHOT_TYPES]

    inputs = processor(text=text_prompts, images=image, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)

    best_idx = probs.argmax().item()
    return BASELINE_SHOT_TYPES[best_idx], probs[0, best_idx].item()


# ============================================================================
# Method 2: Improved CLIP (cinematography-aware ensemble prompts)
# ============================================================================

IMPROVED_SHOT_TYPES = {
    "extreme_closeup": [
        "an extreme close-up shot showing only eyes or mouth",
        "a tight close-up filling the frame with facial detail",
        "an extreme close-up of a small object detail",
        "a macro shot of a face or object",
    ],
    "closeup": [
        "a close-up shot of a person's face",
        "a close-up portrait showing head and shoulders",
        "a tight shot focusing on a single face",
        "a headshot in a film or movie",
    ],
    "medium_closeup": [
        "a medium close-up shot from the chest up",
        "a shot framing a person from mid-chest to head",
        "a news anchor style framing",
        "a bust shot showing upper torso and head",
    ],
    "medium": [
        "a medium shot showing a person from the waist up",
        "a shot framing a person from hip to head",
        "a conversational shot showing the upper body",
        "two people talking in a medium shot",
    ],
    "full": [
        "a full shot showing a person's entire body head to toe",
        "a shot showing a standing figure in full",
        "a person standing with their whole body visible",
        "a full body shot of a character",
    ],
    "long": [
        "a wide establishing shot showing the full scene",
        "a landscape shot with small figures in the distance",
        "an extreme wide shot of an environment",
        "a panoramic view with tiny people",
    ],
}

IMPROVED_DISPLAY_NAMES = {
    "extreme_closeup": "Extreme CU",
    "closeup": "Close-up",
    "medium_closeup": "Medium CU",
    "medium": "Medium",
    "full": "Full",
    "long": "Long/Wide",
}


def classify_improved_clip(image_path: Path) -> tuple[str, float, dict]:
    """Improved CLIP with ensemble prompts."""
    import torch
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    image = Image.open(image_path).convert("RGB")

    # Flatten all prompts
    all_prompts = []
    prompt_to_category = []
    for category, prompts in IMPROVED_SHOT_TYPES.items():
        for prompt in prompts:
            all_prompts.append(prompt)
            prompt_to_category.append(category)

    inputs = processor(text=all_prompts, images=image, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]

    # Average scores per category (ensemble)
    category_scores = {}
    for category in IMPROVED_SHOT_TYPES:
        indices = [i for i, c in enumerate(prompt_to_category) if c == category]
        category_scores[category] = sum(probs[i].item() for i in indices) / len(indices)

    best_category = max(category_scores, key=category_scores.get)
    return best_category, category_scores[best_category], category_scores


# ============================================================================
# Method 3: VLM (GPT-5.2 vision)
# ============================================================================

VLM_SHOT_TYPES = ["extreme_closeup", "closeup", "medium_closeup", "medium", "full", "long"]


def classify_vlm(image_path: Path) -> tuple[str, float, str]:
    """Use GPT-5.2 vision for classification."""
    import base64

    import litellm

    from core.settings import get_openai_api_key

    api_key = get_openai_api_key()
    if not api_key:
        return "error", 0.0, "No OpenAI API key configured"

    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")

    prompt = """Classify this video frame's shot type. Choose exactly ONE from:
- extreme_closeup: Shows only eyes, mouth, or tiny detail filling the frame
- closeup: Face fills most of frame, head and maybe shoulders visible
- medium_closeup: Chest up, like a news anchor framing
- medium: Waist up, conversational distance
- full: Entire body visible head to toe
- long: Wide/establishing shot, environment dominates, figures are small

Respond with ONLY the shot type label (e.g., "medium") and confidence 0-100.
Format: shot_type,confidence
Example: closeup,85"""

    try:
        response = litellm.completion(
            model="gpt-5.2",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ],
            api_key=api_key,
            max_tokens=50,
        )

        result = response.choices[0].message.content.strip().lower()
        parts = result.split(",")
        shot_type = parts[0].strip()
        confidence = float(parts[1].strip()) / 100 if len(parts) > 1 else 0.8

        # Validate shot type
        if shot_type not in VLM_SHOT_TYPES:
            # Try to match partial
            for st in VLM_SHOT_TYPES:
                if st in shot_type or shot_type in st:
                    shot_type = st
                    break
            else:
                return shot_type, confidence, f"Unknown type: {result}"

        return shot_type, confidence, result

    except Exception as e:
        return "error", 0.0, str(e)


# ============================================================================
# Main comparison
# ============================================================================


def compare_methods(image_paths: list[Path], include_vlm: bool = True):
    """Run all methods and compare results."""
    print("\n" + "=" * 100)
    print("SHOT TYPE CLASSIFICATION COMPARISON")
    print("=" * 100)

    results = []

    for i, image_path in enumerate(image_paths):
        print(f"\n[{i+1}/{len(image_paths)}] {image_path.name}")
        print("-" * 80)

        row = {"image": image_path.name}

        # Method 1: Baseline CLIP
        start = time.time()
        baseline_type, baseline_conf = classify_baseline_clip(image_path)
        baseline_time = time.time() - start
        row["baseline"] = f"{baseline_type} ({baseline_conf:.0%})"
        print(f"  Baseline CLIP:  {baseline_type:20} conf={baseline_conf:.2f}  ({baseline_time:.2f}s)")

        # Method 2: Improved CLIP
        start = time.time()
        improved_type, improved_conf, improved_scores = classify_improved_clip(image_path)
        improved_time = time.time() - start
        display_name = IMPROVED_DISPLAY_NAMES.get(improved_type, improved_type)
        row["improved"] = f"{display_name} ({improved_conf:.0%})"
        print(f"  Improved CLIP:  {display_name:20} conf={improved_conf:.2f}  ({improved_time:.2f}s)")

        # Show all improved scores
        sorted_scores = sorted(improved_scores.items(), key=lambda x: x[1], reverse=True)
        score_str = "  ".join(f"{IMPROVED_DISPLAY_NAMES.get(k,k)}:{v:.0%}" for k, v in sorted_scores[:3])
        print(f"                  Top 3: {score_str}")

        # Method 3: VLM (optional)
        if include_vlm:
            start = time.time()
            vlm_type, vlm_conf, vlm_raw = classify_vlm(image_path)
            vlm_time = time.time() - start
            if vlm_type != "error":
                display_name = IMPROVED_DISPLAY_NAMES.get(vlm_type, vlm_type)
                row["vlm"] = f"{display_name} ({vlm_conf:.0%})"
                print(f"  VLM (GPT-5.2):   {display_name:20} conf={vlm_conf:.2f}  ({vlm_time:.2f}s)")
            else:
                row["vlm"] = f"Error: {vlm_raw}"
                print(f"  VLM (GPT-5.2):   ERROR - {vlm_raw}")

        results.append(row)

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"{'Image':<30} {'Baseline':<20} {'Improved':<20} {'VLM (GPT-5.2)':<20}")
    print("-" * 100)
    for row in results:
        print(
            f"{row['image']:<30} {row.get('baseline',''):<20} {row.get('improved',''):<20} {row.get('vlm',''):<20}"
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare shot type classifiers")
    parser.add_argument(
        "--images",
        type=Path,
        default=Path.home() / ".cache/scene-ripper/thumbnails",
        help="Directory containing test images",
    )
    parser.add_argument("--limit", type=int, default=10, help="Max images to test")
    parser.add_argument("--no-vlm", action="store_true", help="Skip VLM (GPT-5.2) test")
    args = parser.parse_args()

    # Find test images
    if args.images.is_dir():
        image_paths = sorted(args.images.glob("*.jpg"))[: args.limit]
        if not image_paths:
            image_paths = sorted(args.images.glob("thumb_*.jpg"))[: args.limit]
    else:
        image_paths = [args.images]

    if not image_paths:
        print(f"No images found in {args.images}")
        return 1

    print(f"Testing {len(image_paths)} images from {args.images}")

    compare_methods(image_paths, include_vlm=not args.no_vlm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
