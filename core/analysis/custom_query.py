"""Custom visual query evaluation using VLMs.

Evaluates whether a clip's thumbnail matches a natural language visual query
(e.g., "blue flower", "person wearing red hat") using cloud or local VLMs.
Returns a boolean match with confidence score.
"""

import logging
import os
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# One-time SSL cert fix for Windows (certifi CA bundle)
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except ImportError:
    pass


def _build_query_prompt(query: str) -> str:
    """Build a structured yes/no prompt for the VLM."""
    return (
        f"Does this image contain: {query}?\n\n"
        "Answer with exactly YES or NO on the first line, "
        "followed by a confidence percentage (0-100%) on the second line.\n"
        "Example:\nYES\n85%"
    )


def _parse_yes_no_response(response: str) -> tuple[bool, float]:
    """Parse a VLM yes/no response into (match, confidence).

    Handles common formats:
    - "YES\\n85%"
    - "Yes, 92%"
    - "NO (confidence: 15%)"
    - "yes"
    - "No"
    - "YES - I am 90% confident"

    Returns:
        Tuple of (match: bool, confidence: float 0.0-1.0)
    """
    text = response.strip().lower()

    # Extract yes/no from first word or line
    first_line = text.split("\n")[0].strip()
    first_word = re.split(r"[\s,.:;!\-]", first_line)[0]

    if first_word in ("yes", "true"):
        match = True
    elif first_word in ("no", "false"):
        match = False
    else:
        # Check if the response contains yes or no anywhere
        has_yes = "yes" in text
        has_no = "no" in text

        if has_yes and not has_no:
            match = True
        elif has_no and not has_yes:
            match = False
        elif has_yes and has_no:
            # Ambiguous: use whichever keyword appears first
            match = text.find("yes") < text.find("no")
            logger.info(f"Ambiguous response, using first keyword: match={match}")
        else:
            logger.warning(f"Could not parse yes/no from response: {response[:100]}")
            return False, 0.0

    # Extract confidence percentage
    pct_match = re.search(r"(\d{1,3})\s*%", text)
    if pct_match:
        confidence = min(int(pct_match.group(1)), 100) / 100.0
    else:
        # No explicit percentage — use 0.9 for definitive answers (high but not absolute)
        confidence = 0.9 if match else 0.1

    return match, confidence


def _normalize_cloud_model(model: str) -> str:
    """Normalize model name for LiteLLM routing."""
    if "gemini" in model.lower() and not any(model.startswith(p) for p in ["gemini/", "vertex_ai/"]):
        return f"gemini/{model}"
    if "claude" in model.lower() and not any(model.startswith(p) for p in ["anthropic/", "bedrock/"]):
        return f"anthropic/{model}"
    return model


def _resolve_cloud_api_key(model: str) -> Optional[str]:
    """Resolve the API key for a cloud model."""
    from core.settings import (
        get_openai_api_key,
        get_anthropic_api_key,
        get_gemini_api_key,
    )

    lowered = model.lower()
    if "gpt" in lowered or "openai" in lowered:
        return get_openai_api_key()
    if "claude" in lowered or "anthropic" in lowered:
        return get_anthropic_api_key()
    if "gemini" in lowered:
        return get_gemini_api_key()
    return None


def evaluate_custom_query_cloud(
    image_path: Path,
    query: str,
) -> tuple[bool, float, str]:
    """Evaluate a custom visual query using cloud VLM via LiteLLM.

    Args:
        image_path: Path to thumbnail image
        query: Natural language visual query

    Returns:
        Tuple of (match, confidence, model_name)
    """
    from core.analysis.description import encode_image_base64, _format_cloud_api_error
    from core.settings import load_settings

    settings = load_settings()
    original_model = settings.description_model_cloud
    model = _normalize_cloud_model(original_model)
    api_key = _resolve_cloud_api_key(model)

    if not api_key:
        raise ValueError(
            f"No API key found for cloud model {original_model}. "
            "Please configure the API key in Settings."
        )

    base64_image = encode_image_base64(image_path)
    prompt = _build_query_prompt(query)

    messages = [
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
    ]

    try:
        import litellm

        response = litellm.completion(
            model=model,
            messages=messages,
            api_key=api_key,
            max_tokens=50,
        )
        content = response.choices[0].message.content

        if content is None or not content.strip():
            raise RuntimeError(f"API returned empty response from {original_model}")

        match, confidence = _parse_yes_no_response(content)
        return match, confidence, original_model

    except Exception as e:
        if isinstance(e, (ValueError, RuntimeError)):
            raise
        raise RuntimeError(_format_cloud_api_error(e, original_model, "query")) from e


def evaluate_custom_query_local(
    image_path: Path,
    query: str,
) -> tuple[bool, float, str]:
    """Evaluate a custom visual query using local VLM (Moondream/Qwen3-VL).

    Args:
        image_path: Path to thumbnail image
        query: Natural language visual query

    Returns:
        Tuple of (match, confidence, model_name)
    """
    from core.analysis.description import (
        is_mlx_vlm_available,
        describe_frame_local,
    )

    prompt = _build_query_prompt(query)

    # Use the existing local VLM infrastructure
    response = describe_frame_local(image_path, prompt)
    match, confidence = _parse_yes_no_response(response)

    # Determine model name from what's loaded
    if is_mlx_vlm_available():
        model_name = "qwen3-vl-4b"
    else:
        model_name = "moondream-2b"

    return match, confidence, model_name


def evaluate_custom_query(
    image_path: Path,
    query: str,
    tier: Optional[str] = None,
) -> tuple[bool, float, str]:
    """Evaluate a custom visual query using the configured VLM tier.

    Routes to either cloud (LiteLLM) or local (Moondream/Qwen3-VL)
    based on the description_model_tier setting.

    Args:
        image_path: Path to thumbnail image
        query: Natural language visual query (e.g., "blue flower")
        tier: 'local' or 'cloud'. If None, uses settings default.

    Returns:
        Tuple of (match: bool, confidence: float, model_name: str)
    """
    if tier is None:
        from core.settings import load_settings
        tier = load_settings().description_model_tier

    # Normalize legacy tier names
    if tier in ("cpu", "gpu"):
        tier = "local" if tier == "cpu" else "cloud"

    logger.info(f"Evaluating custom query '{query}' with tier={tier}")

    if tier == "cloud":
        return evaluate_custom_query_cloud(image_path, query)
    else:
        return evaluate_custom_query_local(image_path, query)
