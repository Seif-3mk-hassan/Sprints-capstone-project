"""Compatibility shim.

Canonical implementation lives in `NLP_Capstone_project/src/prompts.py`.
"""

from src.prompts import (  # noqa: F401
    PromptStyle,
    PromptTemplates,
    format_context,
    get_templates,
    render_prompt,
)

__all__ = [
    "PromptStyle",
    "PromptTemplates",
    "get_templates",
    "render_prompt",
    "format_context",
]
