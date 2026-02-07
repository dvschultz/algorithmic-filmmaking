"""Algorithm configuration — single source of truth for all sequencer algorithms.

Used by sequence_tab.py (labels, descriptions, allow_duplicates) and
sorting_card_grid.py (icons, labels, descriptions).
"""

ALGORITHM_CONFIG = {
    "color": {
        "icon": "\U0001f3a8",
        "label": "Chromatic Flow",
        "description": "Arrange clips along a color gradient",
        "allow_duplicates": False,
    },
    "color_cycle": {
        "icon": "\U0001f308",
        "label": "Color Cycle",
        "description": "Curate clips with strong color identity and cycle through the spectrum",
        "allow_duplicates": False,
    },
    "duration": {
        "icon": "\u23f1\ufe0f",
        "label": "Tempo Shift",
        "description": "Order clips from shortest to longest (or reverse)",
        "allow_duplicates": False,
    },
    "brightness": {
        "icon": "\U0001f317",
        "label": "Into the Dark",
        "description": "Arrange clips from light to shadow, or shadow to light",
        "allow_duplicates": False,
    },
    "volume": {
        "icon": "\U0001f50a",
        "label": "Crescendo",
        "description": "Build from silence to thunder, or thunder to silence",
        "allow_duplicates": False,
    },
    "shuffle": {
        "icon": "\U0001f3b2",
        "label": "Dice Roll",
        "description": "Randomly shuffle clips into a new order",
        "allow_duplicates": False,
    },
    "sequential": {
        "icon": "\U0001f4cb",
        "label": "Time Capsule",
        "description": "Keep clips in their original order",
        "allow_duplicates": False,
    },
    "shot_type": {
        "icon": "\U0001f3ac",
        "label": "Focal Ladder",
        "description": "Arrange clips by camera shot scale",
        "allow_duplicates": False,
    },
    "proximity": {
        "icon": "\U0001f52d",
        "label": "Up Close and Personal",
        "description": "Glide from distant vistas to intimate close-ups",
        "allow_duplicates": False,
    },
    "similarity_chain": {
        "icon": "\U0001f517",
        "label": "Human Centipede",
        "description": "Chain clips together by visual similarity",
        "allow_duplicates": False,
    },
    "match_cut": {
        "icon": "\u2702\ufe0f",
        "label": "Match Cut",
        "description": "Find hidden connections between clips at cut points",
        "allow_duplicates": False,
    },
    "exquisite_corpus": {
        "icon": "\U0001f4dd",
        "label": "Exquisite Corpus",
        "description": "Generate a poem from on-screen text",
        "allow_duplicates": True,
    },
    "storyteller": {
        "icon": "\U0001f4d6",
        "label": "Storyteller",
        "description": "Create a narrative from clip descriptions",
        "allow_duplicates": False,
    },
}


def get_algorithm_config(algorithm: str) -> dict:
    """Get configuration for an algorithm.

    Args:
        algorithm: Algorithm name (lowercase)

    Returns:
        Configuration dict with 'icon', 'label', 'description', 'allow_duplicates'
    """
    return ALGORITHM_CONFIG.get(algorithm.lower(), {
        "icon": "",
        "label": algorithm.replace("_", " ").title(),
        "description": "",
        "allow_duplicates": False,
    })


def get_algorithm_label(algorithm: str) -> str:
    """Get display label for an algorithm.

    Args:
        algorithm: Algorithm key (e.g., 'color', 'shot_type')

    Returns:
        Display label (e.g., 'Focal Ladder')
    """
    config = ALGORITHM_CONFIG.get(algorithm.lower())
    if config:
        return config["label"]
    return algorithm.replace("_", " ").title()
