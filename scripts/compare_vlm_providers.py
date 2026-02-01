#!/usr/bin/env python3
"""Compare VLM providers for shot type classification.

Tests GPT-5.2, Claude, and Gemini on labeled test clips.
"""

import base64
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import litellm
from core.settings import get_openai_api_key, get_anthropic_api_key, get_gemini_api_key

# Labeled test clips (ground truth)
TEST_CLIPS_DIR = Path(__file__).parent.parent / "test_clips" / "labeled" / "clips"
TEST_CLIPS = [
    ("01_CS_royal_tenenbaums.mp4", "CS"),
    ("02_CS_the_shining.mp4", "CS"),
    ("03_LS_blade_runner.mp4", "LS"),
    ("04_LS_phantom_thread.mp4", "LS"),
    ("05_MS_interstellar.mp4", "MS"),
    ("06_MS_no_country.mp4", "MS"),
    ("07_FS_grand_budapest.mp4", "FS"),
    ("08_FS_django.mp4", "FS"),
    ("09_ECS_rocky_horror.mp4", "ECS"),
    ("10_ECS_kill_bill.mp4", "ECS"),
]

# VLM providers to test
PROVIDERS = [
    ("gpt-5.2", get_openai_api_key, None),
    ("claude-sonnet-4-5-20250929", get_anthropic_api_key, "anthropic/"),
    ("gemini-2.5-flash", get_gemini_api_key, "gemini/"),
]

# Chain-of-thought prompt that achieved 90% on GPT-4o
COT_PROMPT = """Analyze this film frame step by step:

1. What body parts are visible? (full body / waist up / face only / partial face)
2. If a face: Is the COMPLETE face visible (forehead to chin)? Or only PART (just eyes or just lips)?
3. How much of frame does the main subject fill?

Classify as:
- LS: Environment dominates, tiny/distant figures
- FS: Full body head-to-toe, one person prominent
- MS: Waist/hip up to head
- CS: Complete face visible (forehead through chin)
- ECS: Only PART of face (just eyes OR just lips, rest cut off)

Reason briefly, then answer with ONLY the abbreviation on the last line."""

# Simpler prompt for Gemini (chain-of-thought causes truncation)
SIMPLE_PROMPT = """Look at this film frame and classify the shot type.

Choose ONE:
- LS (Long Shot): Wide view, environment dominates, tiny figures
- FS (Full Shot): One person's full body visible head to toe
- MS (Medium Shot): Waist up to head
- CS (Close-up): Complete face fills frame
- ECS (Extreme Close-up): Only part of face (just eyes or lips)

Answer with just the two-letter code (LS, FS, MS, CS, or ECS)."""

VALID_TYPES = {"LS", "FS", "MS", "CS", "ECS"}


def extract_frame(clip_path: Path) -> bytes:
    """Extract a frame from video at 1.5 seconds."""
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        frame_path = tmp.name

    subprocess.run(
        ["ffmpeg", "-y", "-ss", "1.5", "-i", str(clip_path), "-vframes", "1", "-q:v", "2", frame_path],
        capture_output=True
    )

    with open(frame_path, "rb") as f:
        data = f.read()

    Path(frame_path).unlink(missing_ok=True)
    return data


def classify_with_vlm(image_data: bytes, model: str, api_key: str, prefix: str = None) -> tuple[str, float]:
    """Classify shot type using a VLM."""
    import re
    base64_image = base64.b64encode(image_data).decode("utf-8")

    # Apply model prefix if needed
    full_model = f"{prefix}{model}" if prefix else model

    # Use simpler prompt for Gemini (chain-of-thought causes truncation)
    prompt = SIMPLE_PROMPT if "gemini" in model.lower() else COT_PROMPT
    max_tok = 100 if "gemini" in model.lower() else 400

    try:
        response = litellm.completion(
            model=full_model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }],
            api_key=api_key,
            max_tokens=max_tok,
        )

        full_response = response.choices[0].message.content.strip()

        # Strategy 1: Check last line for standalone abbreviation
        last_line = full_response.split('\n')[-1].strip().upper()
        for valid in VALID_TYPES:
            if valid == last_line or valid in last_line.split():
                return valid, full_response

        # Strategy 2: Look for "Classification: XX" or "**XX**" patterns
        upper_response = full_response.upper()

        # Check for bold markdown pattern like **CS** or **LS**
        bold_match = re.search(r'\*\*(' + '|'.join(VALID_TYPES) + r')\*\*', upper_response)
        if bold_match:
            return bold_match.group(1), full_response

        # Check for "Classification:" or "Answer:" followed by type
        for pattern in [r'(?:CLASSIFICATION|ANSWER|RESULT|TYPE)[:\s]+(' + '|'.join(VALID_TYPES) + r')\b',
                        r'\b(' + '|'.join(VALID_TYPES) + r')\s*$']:
            match = re.search(pattern, upper_response)
            if match:
                return match.group(1), full_response

        # Strategy 3: Find the last occurrence of any valid type
        last_pos = -1
        predicted = "unknown"
        for valid in VALID_TYPES:
            # Look for word boundary matches
            for m in re.finditer(r'\b' + valid + r'\b', upper_response):
                if m.end() > last_pos:
                    last_pos = m.end()
                    predicted = valid

        return predicted, full_response

    except Exception as e:
        return "error", str(e)


def main():
    print("=" * 80)
    print("VLM PROVIDER COMPARISON - Shot Type Classification")
    print("=" * 80)
    print()

    # Check available providers
    available_providers = []
    for model, get_key_fn, prefix in PROVIDERS:
        api_key = get_key_fn()
        if api_key:
            available_providers.append((model, api_key, prefix))
            print(f"✓ {model}: API key found")
        else:
            print(f"✗ {model}: No API key")

    if not available_providers:
        print("\nNo VLM providers configured. Set API keys in settings.")
        return

    print()
    print("-" * 80)

    # Results storage
    results = {model: {"correct": 0, "total": 0, "times": []} for model, _, _ in available_providers}

    # Test each clip
    for clip_name, ground_truth in TEST_CLIPS:
        clip_path = TEST_CLIPS_DIR / clip_name
        if not clip_path.exists():
            print(f"⚠ Skipping {clip_name}: file not found")
            continue

        print(f"\n{clip_name} [Ground Truth: {ground_truth}]")

        # Extract frame once
        image_data = extract_frame(clip_path)

        # Test each provider
        for model, api_key, prefix in available_providers:
            start = time.time()
            predicted, _ = classify_with_vlm(image_data, model, api_key, prefix)
            elapsed = time.time() - start

            match = "✓" if predicted == ground_truth else "✗"
            results[model]["total"] += 1
            results[model]["times"].append(elapsed)
            if predicted == ground_truth:
                results[model]["correct"] += 1

            # Shorten model name for display
            display_name = model.split("-")[0] if "claude" in model else model.split("-")[0]
            if "gemini" in model:
                display_name = "gemini"

            print(f"  {match} {display_name:12} → {predicted:5} ({elapsed:.1f}s)")

            # Debug: show response if unknown
            if predicted == "unknown":
                # Show first and last 100 chars
                response_preview = _[:100] + "..." + _[-100:] if len(_) > 200 else _
                print(f"      Response: {response_preview}")

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Provider':<25} {'Accuracy':<15} {'Avg Time':<15}")
    print("-" * 55)

    for model, api_key, prefix in available_providers:
        r = results[model]
        if r["total"] > 0:
            accuracy = r["correct"] / r["total"]
            avg_time = sum(r["times"]) / len(r["times"]) if r["times"] else 0

            display_name = model
            if len(display_name) > 24:
                display_name = display_name[:21] + "..."

            print(f"{display_name:<25} {r['correct']}/{r['total']} ({accuracy:.0%}){'':<5} {avg_time:.2f}s")

    print()


if __name__ == "__main__":
    main()
