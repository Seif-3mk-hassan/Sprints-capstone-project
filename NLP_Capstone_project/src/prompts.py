from __future__ import annotations

from dataclasses import dataclass


PromptStyle = str


@dataclass(frozen=True)
class PromptTemplates:
    zero_shot: str
    few_shot: str
    reasoned: str


def get_templates() -> PromptTemplates:
    # Keep templates compact and deterministic.
    zero_shot = (
        "You are a helpful IT service desk agent.\n"
        "Write a professional, concise reply to the customer issue below.\n"
        "If you need more info, ask 1-3 specific questions.\n\n"
        "Customer issue:\n{issue}\n"
    )

    few_shot = (
        "You are a helpful IT service desk agent.\n"
        "Write a professional, concise reply to the customer issue.\n\n"
        "Example\n"
        "Customer issue: Password reset not working; I don\u2019t receive the email.\n"
        "Reply: Thanks for reporting this. Please confirm the email on your account, check spam/junk, "
        "and let us know the exact time of your last attempt. We\u2019ll verify mail delivery logs and assist further.\n\n"
        "Customer issue:\n{issue}\n"
    )

    # "Reasoned" here means: structured + cautious, without exposing chain-of-thought.
    reasoned = (
        "You are a helpful IT service desk agent.\n"
        "Write a professional reply using this structure:\n"
        "1) Acknowledge the issue\n"
        "2) Likely cause(s) (1-2 bullets)\n"
        "3) Next steps (3-6 bullets)\n"
        "4) If needed, request missing details (1-3 questions)\n\n"
        "Customer issue:\n{issue}\n"
    )

    return PromptTemplates(zero_shot=zero_shot, few_shot=few_shot, reasoned=reasoned)


def render_prompt(*, style: PromptStyle, issue: str, context_block: str | None = None) -> str:
    templates = get_templates()

    style_norm = (style or "").strip().lower()
    if style_norm in {"zero", "zero-shot", "zeroshot"}:
        base = templates.zero_shot
    elif style_norm in {"few", "few-shot", "fewshot"}:
        base = templates.few_shot
    elif style_norm in {"reasoned", "cot", "structured"}:
        base = templates.reasoned
    else:
        raise ValueError(f"Unknown prompt_style: {style}")

    prompt = base.format(issue=issue.strip())

    if context_block:
        prompt += "\n\nReference context (use only if helpful):\n" + context_block.strip() + "\n"

    return prompt


def format_context(chunks: list[dict], *, max_chars: int = 4000) -> str:
    """Format retrieved documents into a compact context block.

    Expected chunk shape:
      {"document": str, "reference_reply": str | None}
    """

    parts: list[str] = []
    total = 0
    for idx, chunk in enumerate(chunks, start=1):
        doc = (chunk.get("document") or "").strip()
        ref = (chunk.get("reference_reply") or "").strip()
        text = f"[{idx}] Similar ticket:\n{doc}"
        if ref:
            text += f"\nSuggested reply (historical):\n{ref}"
        text += "\n"

        if total + len(text) > max_chars:
            break
        parts.append(text)
        total += len(text)

    return "\n".join(parts).strip()
