"""
Prompt loader for LLM system prompts.

All prompts are defined in config/prompts.yml.
Missing or empty prompts raise PromptNotFoundError immediately — no fallbacks.
"""

from pathlib import Path
from typing import Dict
import yaml

_PROMPTS_FILE = Path(__file__).parent / "prompts.yml"

_REQUIRED_KEYS = {"tool_calling", "rag_response", "query_generation"}


class PromptNotFoundError(RuntimeError):
    """Raised when a required prompt is missing or empty in prompts.yml."""
    pass


def _load_prompts() -> Dict[str, str]:
    """Load and validate all prompts from prompts.yml."""
    if not _PROMPTS_FILE.exists():
        raise PromptNotFoundError(
            f"Prompts file not found: {_PROMPTS_FILE}. "
            "Ensure config/prompts.yml exists."
        )

    with open(_PROMPTS_FILE, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise PromptNotFoundError(
            f"config/prompts.yml must be a YAML mapping, got {type(data).__name__}"
        )

    prompts: Dict[str, str] = {}
    missing = []

    for key in _REQUIRED_KEYS:
        value = data.get(key, "")
        if not value or not str(value).strip():
            missing.append(key)
        else:
            prompts[key] = str(value).strip()

    if missing:
        raise PromptNotFoundError(
            f"The following prompts are missing or empty in config/prompts.yml: "
            f"{', '.join(sorted(missing))}. "
            "Add them before starting the application."
        )

    return prompts


# Load once at import time — fails fast if anything is wrong
_prompts: Dict[str, str] = _load_prompts()


def get_tool_calling_prompt() -> str:
    """Return the initial tool-calling system prompt."""
    return _prompts["tool_calling"]


def get_rag_response_prompt() -> str:
    """Return the RAG response system prompt (used after retrieval)."""
    return _prompts["rag_response"]


def get_query_generation_prompt() -> str:
    """Return the query generation system prompt for LLMQueryGenerator."""
    return _prompts["query_generation"]
