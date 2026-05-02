from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .rag_chain import RAGChain


def load_test_subset(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("test_subset.json must be a list")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate baseline outputs for prompt styles")
    parser.add_argument(
        "--test-path",
        default=str(Path(__file__).resolve().parents[1] / "data" / "test_subset.json"),
        help="Path to test_subset.json",
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "baseline_outputs.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--styles",
        default="zero-shot,few-shot,reasoned",
        help="Comma-separated list of prompt styles",
    )
    parser.add_argument(
        "--use-rag",
        action="store_true",
        help="Enable retrieval (requires chroma_db)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Top-k retrieved docs (when --use-rag)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Max number of test items to run",
    )

    args = parser.parse_args()

    test_path = Path(args.test_path)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    styles = [s.strip() for s in str(args.styles).split(",") if s.strip()]

    test_items = load_test_subset(test_path)[: max(0, int(args.limit))]

    chain = RAGChain()

    results: list[dict] = []
    for idx, item in enumerate(test_items, start=1):
        issue = str(item.get("customer_issue", "")).strip()
        reference = str(item.get("reference_reply", "")).strip()
        if not issue:
            continue

        for style in styles:
            payload = chain.generate(
                issue,
                use_rag=bool(args.use_rag),
                prompt_style=style,
                k=int(args.k),
                include_retrieval=False,
            )
            results.append(
                {
                    "id": idx,
                    "prompt_style": style,
                    "use_rag": bool(args.use_rag),
                    "k": int(args.k),
                    "customer_issue": issue,
                    "reference_reply": reference,
                    "model_reply": payload["reply"],
                }
            )

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(results),
        "items": results,
    }

    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(results)} items to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
