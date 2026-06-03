"""Built-in presets for common LLM application types."""

import copy
from typing import Dict, List


PRESETS: Dict[str, dict] = {
    "coding_assistant": {
        "preserve": [
            "coding_language",
            "framework",
            "cloud_provider",
            "architecture",
            "task",
            "constraint",
            "decision",
        ],
        "threshold": 0.35,
        "token_budget": 8000,
        "description": "For code generation and programming assistants",
    },
    "customer_support": {
        "preserve": [
            "user_name",
            "user_preference",
            "task",
            "constraint",
            "decision",
            "organization",
        ],
        "threshold": 0.40,
        "token_budget": 6000,
        "description": "For support bots and helpdesk agents",
    },
    "ai_tutor": {
        "preserve": [
            "user_name",
            "user_preference",
            "task",
            "constraint",
            "temporal_reference",
        ],
        "threshold": 0.30,
        "token_budget": 10000,
        "description": "For educational assistants and tutoring apps",
    },
    "agent_system": {
        "preserve": [
            "task",
            "decision",
            "constraint",
            "action_intent",
            "organization",
            "location",
            "product",
        ],
        "threshold": 0.45,
        "token_budget": 12000,
        "description": "For autonomous agents and multi-step task runners",
    },
    "general": {
        "preserve": [
            "user_name",
            "user_preference",
            "task",
            "decision",
            "constraint",
        ],
        "threshold": 0.40,
        "token_budget": 8000,
        "description": "Balanced default for general chat applications",
    },
}


def load_preset(name: str) -> dict:
    """Load a preset configuration by name.

    Args:
        name: The preset name (e.g. "coding_assistant", "general").

    Returns:
        A deep copy of the preset configuration dict.

    Raises:
        ValueError: If the preset name is not recognized.
    """
    if name not in PRESETS:
        raise ValueError(
            f"Unknown preset '{name}'. Available: {list(PRESETS.keys())}"
        )
    return copy.deepcopy(PRESETS[name])


def list_presets() -> Dict[str, str]:
    """List all available presets with their descriptions.

    Returns:
        A dict mapping preset names to their description strings.
    """
    return {name: cfg["description"] for name, cfg in PRESETS.items()}
