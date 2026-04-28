"""Algorithm configuration — single source of truth for all sequencer algorithms.

Used by sequence_tab.py (labels, descriptions, allow_duplicates) and
sorting_card_grid.py (icons, labels, descriptions, categories).

Icons are currently empty strings. Rendering emoji glyphs via QLabel on
macOS can hit a CoreText sbix→ImageIO crash (EXC_BAD_ACCESS at 0xbad4007
inside CopyEmojiImage); SVG icons will replace this field in a later pass.
"""

CATEGORY_ORDER = ["All", "Arrange", "Find", "Connect", "Audio", "Text"]

ALGORITHM_CONFIG = {
    "color": {
        "icon": "",
        "label": "Chromatics",
        "description": "Arrange clips along a color gradient or cycle through the spectrum",
        "allow_duplicates": False,
        "required_analysis": ["colors"],
        "categories": ["arrange"],
    },
    "duration": {
        "icon": "",
        "label": "Tempo Shift",
        "description": "Order clips from shortest to longest (or reverse)",
        "allow_duplicates": False,
        "required_analysis": [],
        "categories": ["arrange"],
    },
    "brightness": {
        "icon": "",
        "label": "Into the Dark",
        "description": "Arrange clips from light to shadow, or shadow to light",
        "allow_duplicates": False,
        "required_analysis": ["brightness"],
        "categories": ["arrange"],
    },
    "volume": {
        "icon": "",
        "label": "Crescendo",
        "description": "Build from silence to thunder, or thunder to silence",
        "allow_duplicates": False,
        "required_analysis": ["volume"],
        "categories": ["arrange", "audio"],
    },
    "shuffle": {
        "icon": "",
        "label": "Hatchet Job",
        "description": "Randomly shuffle clips into a new order",
        "allow_duplicates": False,
        "required_analysis": [],
        "is_dialog": True,
        "categories": ["arrange"],
    },
    "sequential": {
        "icon": "",
        "label": "Time Capsule",
        "description": "Keep clips in their original order",
        "allow_duplicates": False,
        "required_analysis": [],
        "categories": ["arrange"],
    },
    "shot_type": {
        "icon": "",
        "label": "Focal Ladder",
        "description": "Arrange clips by camera shot scale",
        "allow_duplicates": False,
        "required_analysis": ["shots"],
        "categories": ["arrange"],
    },
    "proximity": {
        "icon": "",
        "label": "Up Close and Personal",
        "description": "Glide from distant vistas to intimate close-ups",
        "allow_duplicates": False,
        "required_analysis": ["shots"],
        "categories": ["arrange"],
    },
    "similarity_chain": {
        "icon": "",
        "label": "Human Centipede",
        "description": "Chain clips together by visual similarity",
        "allow_duplicates": False,
        "required_analysis": ["embeddings"],
        "categories": ["connect"],
    },
    "match_cut": {
        "icon": "",
        "label": "Match Cut",
        "description": "Find hidden connections between clips at cut points",
        "allow_duplicates": False,
        "required_analysis": ["boundary_embeddings"],
        "categories": ["connect"],
    },
    "exquisite_corpus": {
        "icon": "",
        "label": "Exquisite Corpus",
        "description": "Generate a poem from on-screen text",
        "allow_duplicates": True,
        "required_analysis": ["extract_text"],
        "is_dialog": True,
        "categories": ["text"],
    },
    "storyteller": {
        "icon": "",
        "label": "Storyteller",
        "description": "Create a narrative from clip descriptions",
        "allow_duplicates": False,
        "required_analysis": ["describe"],
        "is_dialog": True,
        "categories": ["text"],
    },
    "free_association": {
        "icon": "",
        "label": "Free Association",
        "description": "Build a sequence one clip at a time with an LLM collaborator",
        "allow_duplicates": False,
        # Embeddings power the local candidate shortlist — without them the
        # sequencer falls back to random sampling, which degrades proposal
        # quality. Declaring them here prompts the user via the cost gate.
        "required_analysis": ["describe", "embeddings"],
        "is_dialog": True,
        "categories": ["connect", "text"],
    },
    "cassette_tape": {
        "icon": "",
        "label": "Cassette Tape",
        "description": "Find clips that say specific phrases — the transcript-driven mixtape",
        "allow_duplicates": False,
        "required_analysis": ["transcribe"],
        "is_dialog": True,
        "categories": ["arrange", "audio"],
    },
    "reference_guided": {
        "icon": "",
        "label": "Reference Guide",
        "description": "Match your clips to a reference video's structure",
        "allow_duplicates": True,
        "required_analysis": [],  # Dynamic — depends on selected dimensions
        "is_dialog": True,
        "categories": ["connect", "audio"],
    },
    "signature_style": {
        "icon": "",
        "label": "Signature Style",
        "description": "Interpret a drawing as an editing guide",
        "allow_duplicates": True,
        "required_analysis": ["colors"],
        "is_dialog": True,
        "categories": ["connect"],
    },
    "rose_hobart": {
        "icon": "",
        "label": "Rose Hobart",
        "description": "Isolate clips featuring a specific person",
        "allow_duplicates": False,
        "required_analysis": [],
        "is_dialog": True,
        "categories": ["find"],
    },
    "staccato": {
        "icon": "",
        "label": "Staccato",
        "description": "Cut clips to the rhythm of a music track",
        "allow_duplicates": True,
        "required_analysis": ["embeddings"],
        "is_dialog": True,
        "categories": ["audio"],
    },
    "gaze_sort": {
        "icon": "",
        "label": "Gaze Sort",
        "description": "Arrange clips by gaze direction",
        "allow_duplicates": False,
        "required_analysis": ["gaze"],
        "categories": ["arrange", "find"],
    },
    "gaze_consistency": {
        "icon": "",
        "label": "Gaze Consistency",
        "description": "Group clips by matching gaze direction",
        "allow_duplicates": False,
        "required_analysis": ["gaze"],
        "categories": ["find"],
    },
    "eyes_without_a_face": {
        "icon": "",
        "label": "Eyes Without a Face",
        "description": "Eyeline matching, gaze filtering, and rotation sequencing",
        "allow_duplicates": False,
        "required_analysis": ["gaze"],
        "is_dialog": True,
        "categories": ["find", "connect"],
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
        "required_analysis": [],
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
