from __future__ import annotations

import re

from .models import ISSUE_LABELS


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    return slug or "item"


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def looks_like_chatter(text: str) -> bool:
    lowered = text.lower()
    blocked = ("thank you", "can you hear", "screen share", "class", "syllabus", "breakout")
    return any(item in lowered for item in blocked)


def title_from_sentence(sentence: str) -> str:
    words = re.sub(r"\s+", " ", sentence).strip()[:90].rstrip(".,;:")
    return words[0].upper() + words[1:] if words else "Review recording idea"


def normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def title_similarity(left: str, right: str) -> float:
    left_words = set(left.split())
    right_words = set(right.split())
    if not left_words or not right_words:
        return 0.0
    return len(left_words & right_words) / len(left_words | right_words)


def optional_float(value, default: float | None) -> float | None:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def string_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def labels_from(value) -> list[str]:
    labels = ISSUE_LABELS.copy()
    for label in string_list(value):
        if label not in labels:
            labels.append(label)
    return labels


def timestamp_to_seconds(timestamp: str) -> float:
    hours, minutes, seconds = [int(part) for part in timestamp.split(":")]
    return float(hours * 3600 + minutes * 60 + seconds)


def format_timestamp(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, sec = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"
