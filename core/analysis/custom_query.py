"""Custom visual query evaluation using VLMs.

Evaluates whether a clip's thumbnail matches a natural language visual query
(e.g., "blue flower", "person wearing red hat") using cloud or local VLMs.
Returns a boolean match with confidence score.
"""

import logging
from math import isfinite
import os
import re
from pathlib import Path
from typing import Optional

from core.runtime_families import isolated  # noqa: E402

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
    from core.analysis_model_identity import custom_query_prompt

    return custom_query_prompt(query)


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

    Raises:
        ValueError: If the answer or explicit percentage is not valid.
    """
    if not isinstance(response, str):
        raise ValueError("Custom query response must be text")
    text = response.strip().lower()
    answer = re.match(r"^(?:\*\*|__|`)?(yes|no|true|false)\b", text)
    if answer is None:
        raise ValueError("Custom query response must begin with YES or NO")
    match = answer.group(1) in ("yes", "true")

    # Extract confidence percentage
    pct_match = re.search(
        r"(?<![\w.,/+\-])([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)\s*%", text
    )
    if pct_match:
        percentage = float(pct_match.group(1))
        if not isfinite(percentage) or not 0 <= percentage <= 100:
            raise ValueError(
                "Custom query confidence must be between 0 and 100 percent"
            )
        confidence = percentage / 100.0
    else:
        if "%" in text:
            raise ValueError("Custom query confidence percentage is invalid")
        # No explicit percentage — use 0.9 for definitive answers (high but not absolute)
        confidence = 0.9 if match else 0.1

    return match, confidence


def _normalize_cloud_model(model: str) -> str:
    """Normalize model name for LiteLLM routing."""
    if "gemini" in model.lower() and not any(
        model.startswith(p) for p in ["gemini/", "vertex_ai/"]
    ):
        return f"gemini/{model}"
    if "claude" in model.lower() and not any(
        model.startswith(p) for p in ["anthropic/", "bedrock/"]
    ):
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
    *,
    model_name: Optional[str] = None,
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

    original_model = model_name or load_settings().description_model_cloud
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
        from core.llm_client import complete_routed

        response = complete_routed(
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


@isolated("vlm", "vlm.custom_query_local", decode=lambda v: (bool(v[0]), float(v[1]), str(v[2])))
def evaluate_custom_query_local(
    image_path: Path,
    query: str,
    *,
    model_name: Optional[str] = None,
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
    from core.analysis_model_identity import local_description_runtime
    from core.settings import load_settings

    model_name = model_name or load_settings().description_model_local
    runtime = local_description_runtime(model_name, mlx=is_mlx_vlm_available())

    prompt = _build_query_prompt(query)

    # Use the existing local VLM infrastructure
    response = describe_frame_local(image_path, prompt, model_name=model_name)
    match, confidence = _parse_yes_no_response(response)

    return match, confidence, runtime["model"]


def evaluate_custom_query(
    image_path: Path,
    query: str,
    tier: Optional[str] = None,
    *,
    model_name: Optional[str] = None,
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
        return evaluate_custom_query_cloud(image_path, query, model_name=model_name)
    else:
        return evaluate_custom_query_local(image_path, query, model_name=model_name)
