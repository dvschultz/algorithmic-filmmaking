"""Film language glossary with terminology from professional cinematography.

This module provides a comprehensive glossary of film terms organized by category,
used for tooltips, the in-app glossary dialog, and agent tools.

Source: LearnAboutFilm.com Film Language Guide
"""

from typing import Optional


# Categories for organizing film terms
GLOSSARY_CATEGORIES = [
    "Shot Sizes",
    "Camera Angles",
    "Camera Movement",
    "Composition",
    "Lighting",
    "Focus",
    "Sound",
    "Editing",
]


# Film terminology glossary
# Key is the internal identifier (matches enum values in models/cinematography.py)
# Each entry has: name (display name), category, definition (1-2 sentences)
FILM_GLOSSARY: dict[str, dict[str, str]] = {
    # Shot Sizes
    "ELS": {
        "name": "Extreme Long Shot",
        "category": "Shot Sizes",
        "definition": "Setting dominates the frame with people appearing tiny. Creates feelings of isolation, insignificance, or establishes scope.",
    },
    "VLS": {
        "name": "Very Long Shot",
        "category": "Shot Sizes",
        "definition": "Shows the full environment with subjects visible but small. Establishes location and context.",
    },
    "LS": {
        "name": "Long Shot",
        "category": "Shot Sizes",
        "definition": "Shows subjects from head to toe within their environment. Provides context for action and group dynamics.",
    },
    "MLS": {
        "name": "Medium Long Shot",
        "category": "Shot Sizes",
        "definition": "Shows subjects from roughly the knees up. Balances character detail with environmental context.",
    },
    "MS": {
        "name": "Medium Shot",
        "category": "Shot Sizes",
        "definition": "Shows subjects from the waist up. The standard conversational shot for casual observation.",
    },
    "MCU": {
        "name": "Medium Close-Up",
        "category": "Shot Sizes",
        "definition": "Shows head and shoulders. Common for interviews and direct address to camera.",
    },
    "CU": {
        "name": "Close-Up",
        "category": "Shot Sizes",
        "definition": "Shows the head with a bit of shoulders. Emphasizes emotion, reaction, and creates intimacy.",
    },
    "BCU": {
        "name": "Big Close-Up",
        "category": "Shot Sizes",
        "definition": "Face fills the entire frame. Conveys intense emotion or suggests threat.",
    },
    "ECU": {
        "name": "Extreme Close-Up",
        "category": "Shot Sizes",
        "definition": "Shows only part of the face like eyes or mouth. Creates maximum intensity and dramatic impact.",
    },
    "Insert": {
        "name": "Insert Shot",
        "category": "Shot Sizes",
        "definition": "Close shot of an important detail or object. Used for clarification or emphasis.",
    },
    # Camera Angles
    "low_angle": {
        "name": "Low Angle",
        "category": "Camera Angles",
        "definition": "Camera positioned below subject, pointing up. Conveys power, heroism, or threat.",
    },
    "eye_level": {
        "name": "Eye Level",
        "category": "Camera Angles",
        "definition": "Camera at subject's eye height. Creates neutral, documentary-style objectivity.",
    },
    "high_angle": {
        "name": "High Angle",
        "category": "Camera Angles",
        "definition": "Camera positioned above subject, pointing down. Suggests weakness or vulnerability.",
    },
    "dutch_angle": {
        "name": "Dutch Angle",
        "category": "Camera Angles",
        "definition": "Camera tilted to create a slanted horizon. Creates unease or disorientation.",
    },
    "birds_eye": {
        "name": "Bird's Eye View",
        "category": "Camera Angles",
        "definition": "Camera positioned directly overhead. Provides omniscient perspective and spatial overview.",
    },
    "worms_eye": {
        "name": "Worm's Eye View",
        "category": "Camera Angles",
        "definition": "Camera at ground level pointing up. Creates extreme power or awe.",
    },
    # Camera Movement
    "static": {
        "name": "Static Shot",
        "category": "Camera Movement",
        "definition": "Camera remains stationary. Creates stability and observational distance.",
    },
    "pan": {
        "name": "Pan",
        "category": "Camera Movement",
        "definition": "Camera rotates left or right on a fixed axis. Used to scan, follow, or reveal.",
    },
    "tilt": {
        "name": "Tilt",
        "category": "Camera Movement",
        "definition": "Camera pivots up or down on a fixed axis. Reveals height or follows vertical action.",
    },
    "track": {
        "name": "Track/Dolly",
        "category": "Camera Movement",
        "definition": "Camera physically moves through space. Builds intensity and creates involvement.",
    },
    "handheld": {
        "name": "Handheld",
        "category": "Camera Movement",
        "definition": "Camera held by operator without stabilization. Creates urgency and documentary realism.",
    },
    "crane": {
        "name": "Crane Shot",
        "category": "Camera Movement",
        "definition": "Camera moves vertically through space on a crane. Creates grandeur and dramatic reveals.",
    },
    "arc": {
        "name": "Arc Shot",
        "category": "Camera Movement",
        "definition": "Camera circles around the subject. Provides dramatic emphasis and 360-degree view.",
    },
    "steadicam": {
        "name": "Steadicam",
        "category": "Camera Movement",
        "definition": "Stabilized handheld movement. Allows smooth following shots through complex spaces.",
    },
    # Composition
    "left_third": {
        "name": "Left Third",
        "category": "Composition",
        "definition": "Subject positioned on the left third of frame. Follows rule of thirds for natural composition.",
    },
    "center": {
        "name": "Center Frame",
        "category": "Composition",
        "definition": "Subject positioned in the center of frame. Creates formal, confrontational, or unusual emphasis.",
    },
    "right_third": {
        "name": "Right Third",
        "category": "Composition",
        "definition": "Subject positioned on the right third of frame. Follows rule of thirds for natural composition.",
    },
    "distributed": {
        "name": "Distributed Composition",
        "category": "Composition",
        "definition": "Multiple subjects spread across the frame. Balances visual attention across the image.",
    },
    "balanced": {
        "name": "Balanced Frame",
        "category": "Composition",
        "definition": "Visual weight distributed evenly across the frame. Creates harmony and stability.",
    },
    "left_heavy": {
        "name": "Left-Heavy Frame",
        "category": "Composition",
        "definition": "Visual weight concentrated on the left side. Creates intentional imbalance or tension.",
    },
    "right_heavy": {
        "name": "Right-Heavy Frame",
        "category": "Composition",
        "definition": "Visual weight concentrated on the right side. Creates intentional imbalance or tension.",
    },
    "symmetrical": {
        "name": "Symmetrical Composition",
        "category": "Composition",
        "definition": "Frame balanced with mirror-like symmetry. Creates formality and visual satisfaction.",
    },
    "tight": {
        "name": "Tight Framing",
        "category": "Composition",
        "definition": "Minimal space around the subject. Creates claustrophobia or intensity.",
    },
    "normal": {
        "name": "Normal Framing",
        "category": "Composition",
        "definition": "Standard amount of space around subject. Comfortable, neutral composition.",
    },
    "excessive": {
        "name": "Excessive Space",
        "category": "Composition",
        "definition": "Unusual amount of empty space in frame. Can feel uncomfortable or emphasize isolation.",
    },
    # Lighting
    "high_key": {
        "name": "High-Key Lighting",
        "category": "Lighting",
        "definition": "Bright, even illumination with minimal shadows. Associated with comedy and happiness.",
    },
    "low_key": {
        "name": "Low-Key Lighting",
        "category": "Lighting",
        "definition": "Dramatic lighting with strong shadows and contrast. Creates mystery and tension.",
    },
    "natural": {
        "name": "Natural Lighting",
        "category": "Lighting",
        "definition": "Light from natural sources like windows or sun. Creates realism and documentary feel.",
    },
    "dramatic": {
        "name": "Dramatic Lighting",
        "category": "Lighting",
        "definition": "Stylized lighting for emotional effect. Strong contrast and intentional shadows.",
    },
    "front": {
        "name": "Front Lighting",
        "category": "Lighting",
        "definition": "Light source in front of subject. Creates flat, characterless illumination.",
    },
    "three_quarter": {
        "name": "Three-Quarter Lighting",
        "category": "Lighting",
        "definition": "Light at 45-degree angle to subject. Standard setup providing depth and modeling.",
    },
    "side": {
        "name": "Side Lighting",
        "category": "Lighting",
        "definition": "Light at 90 degrees to subject. Creates atmospheric, dramatic effect.",
    },
    "back": {
        "name": "Back Lighting",
        "category": "Lighting",
        "definition": "Light behind subject creating rim light or silhouette. Provides separation from background.",
    },
    "below": {
        "name": "Under Lighting",
        "category": "Lighting",
        "definition": "Light from below the subject. Creates unsettling, horror-like effect.",
    },
    # Focus
    "deep": {
        "name": "Deep Focus",
        "category": "Focus",
        "definition": "Everything from foreground to background appears sharp. Gives equal attention across the scene.",
    },
    "shallow": {
        "name": "Shallow Focus",
        "category": "Focus",
        "definition": "Subject sharp with blurred background. Isolates subject and directs attention.",
    },
    "rack_focus": {
        "name": "Rack Focus",
        "category": "Focus",
        "definition": "Focus shifts between subjects during shot. Redirects viewer attention.",
    },
    "blurred": {
        "name": "Blurred Background",
        "category": "Focus",
        "definition": "Background rendered out of focus. Emphasizes subject separation.",
    },
    "sharp": {
        "name": "Sharp Background",
        "category": "Focus",
        "definition": "Background rendered in focus. Provides environmental context.",
    },
    # Subject
    "empty": {
        "name": "Empty Frame",
        "category": "Composition",
        "definition": "No clear subject in frame. Creates anticipation or emphasizes environment.",
    },
    "single": {
        "name": "Single Subject",
        "category": "Composition",
        "definition": "One primary subject in frame. Focuses attention on individual.",
    },
    "two_shot": {
        "name": "Two Shot",
        "category": "Composition",
        "definition": "Two subjects in frame together. Shows relationship and interaction.",
    },
    "group": {
        "name": "Group Shot",
        "category": "Composition",
        "definition": "Three or more subjects in frame. Establishes group dynamics.",
    },
    # Sound (for future phases)
    "diegetic": {
        "name": "Diegetic Sound",
        "category": "Sound",
        "definition": "Sound that exists within the scene world like dialogue or footsteps.",
    },
    "non_diegetic": {
        "name": "Non-Diegetic Sound",
        "category": "Sound",
        "definition": "Sound added from outside the scene like musical score or narration.",
    },
    # Editing (for future phases)
    "cut": {
        "name": "Cut",
        "category": "Editing",
        "definition": "Instant transition between shots. The most common edit for continuous action.",
    },
    "jump_cut": {
        "name": "Jump Cut",
        "category": "Editing",
        "definition": "Cut between similar shots creating jarring effect. Stylistic disorientation.",
    },
    "dissolve": {
        "name": "Cross Dissolve",
        "category": "Editing",
        "definition": "Images blend together during transition. Suggests time passage or journey.",
    },
    "fade": {
        "name": "Fade",
        "category": "Editing",
        "definition": "Transition to or from black. Indicates major time passage or scene ending.",
    },
    # Angle effects (derived properties)
    "power": {
        "name": "Power Effect",
        "category": "Camera Angles",
        "definition": "Psychological effect of low angle shots. Makes subject appear dominant or threatening.",
    },
    "neutral": {
        "name": "Neutral Effect",
        "category": "Camera Angles",
        "definition": "Psychological effect of eye-level shots. Creates objectivity and equal footing.",
    },
    "vulnerability": {
        "name": "Vulnerability Effect",
        "category": "Camera Angles",
        "definition": "Psychological effect of high angle shots. Makes subject appear weak or diminished.",
    },
    "disorientation": {
        "name": "Disorientation Effect",
        "category": "Camera Angles",
        "definition": "Psychological effect of dutch angle shots. Creates unease and instability.",
    },
    "omniscience": {
        "name": "Omniscience Effect",
        "category": "Camera Angles",
        "definition": "Psychological effect of bird's eye view. Suggests all-seeing perspective.",
    },
    "extreme_power": {
        "name": "Extreme Power Effect",
        "category": "Camera Angles",
        "definition": "Psychological effect of worm's eye view. Maximum subject dominance.",
    },
}


def get_term_definition(term: str) -> Optional[dict[str, str]]:
    """Look up a film term definition.

    Args:
        term: The term to look up (case-insensitive, matches key or display name)

    Returns:
        Dict with name, category, definition if found, None otherwise
    """
    # Try exact key match first
    term_lower = term.lower().strip()

    # Check keys (case-insensitive)
    for key, data in FILM_GLOSSARY.items():
        if key.lower() == term_lower:
            return {"key": key, **data}

    # Check display names (case-insensitive)
    for key, data in FILM_GLOSSARY.items():
        if data["name"].lower() == term_lower:
            return {"key": key, **data}

    # Check partial matches in name
    for key, data in FILM_GLOSSARY.items():
        if term_lower in data["name"].lower():
            return {"key": key, **data}

    return None


def search_glossary(
    query: str,
    category: Optional[str] = None
) -> list[dict[str, str]]:
    """Search the glossary for terms matching a query.

    Args:
        query: Search string (matches name or definition)
        category: Optional category to filter by

    Returns:
        List of matching terms with key, name, category, definition
    """
    query_lower = query.lower().strip()
    results = []

    for key, data in FILM_GLOSSARY.items():
        # Filter by category if specified
        if category and category != "All" and data["category"] != category:
            continue

        # Check if query matches name or definition
        if (
            query_lower in data["name"].lower()
            or query_lower in data["definition"].lower()
            or query_lower in key.lower()
        ):
            results.append({"key": key, **data})

    # Sort by name
    results.sort(key=lambda x: x["name"])
    return results


def get_terms_by_category(category: str) -> list[dict[str, str]]:
    """Get all terms in a specific category.

    Args:
        category: Category name to filter by

    Returns:
        List of terms in that category
    """
    results = []
    for key, data in FILM_GLOSSARY.items():
        if data["category"] == category:
            results.append({"key": key, **data})

    results.sort(key=lambda x: x["name"])
    return results


def get_tooltip_html(term_key: str) -> str:
    """Generate HTML tooltip content for a term.

    Args:
        term_key: The glossary key for the term

    Returns:
        HTML string suitable for Qt tooltip, or empty string if not found
    """
    data = FILM_GLOSSARY.get(term_key)
    if not data:
        return ""

    return f"""<b>{data['name']}</b><br/>
<i>{data['category']}</i><br/><br/>
{data['definition']}"""


def get_badge_tooltip(badge_text: str) -> str:
    """Get tooltip HTML for a cinematography badge.

    Maps badge display text back to glossary entries.

    Args:
        badge_text: The badge text as displayed (e.g., "low", "shallow focus")

    Returns:
        HTML tooltip string, or empty string if no match
    """
    # Direct key lookup
    if badge_text in FILM_GLOSSARY:
        return get_tooltip_html(badge_text)

    # Handle formatted badge text
    # "low" -> "low_angle"
    # "shallow focus" -> "shallow"
    # "low key" -> "low_key"
    normalized = badge_text.lower().strip()

    # Try adding _angle suffix
    if normalized + "_angle" in FILM_GLOSSARY:
        return get_tooltip_html(normalized + "_angle")

    # Try removing " focus" suffix
    if normalized.endswith(" focus"):
        base = normalized.replace(" focus", "")
        if base in FILM_GLOSSARY:
            return get_tooltip_html(base)

    # Try replacing space with underscore
    underscored = normalized.replace(" ", "_")
    if underscored in FILM_GLOSSARY:
        return get_tooltip_html(underscored)

    # Try removing "s eye" suffix for bird's eye, worm's eye
    if "s eye" in normalized:
        base = normalized.replace("'s eye", "s_eye").replace(" ", "_")
        if base in FILM_GLOSSARY:
            return get_tooltip_html(base)

    return ""
