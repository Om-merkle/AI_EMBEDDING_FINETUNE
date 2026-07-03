"""OPTIONAL - LLM-based synthetic triplet generation.

This is an ALTERNATIVE to auto hard-negative mining, useful when you have raw
documents but no labelled (query, positive) pairs. For each document it asks an LLM
(`gpt-5.4-nano` via the OpenAI API) to invent:

  * a realistic user query the document answers   -> the POSITIVE pair, and
  * a plausible-but-wrong passage                  -> a HARD NEGATIVE.

It is plain Python (no LangGraph) and completely optional: if OPENAI_API_KEY is not
set, `generate()` is a no-op that tells you so. Output is written to the same
data/triplets.jsonl consumed by training, so the rest of the pipeline is unchanged.
"""

from __future__ import annotations

import json
from typing import Any

from core.config import settings

_SYSTEM = (
    "You create training data for a text-embedding retrieval model. "
    "Given a DOCUMENT, return a JSON object with exactly these keys: "
    '"query" (a realistic question a user would type that this document answers) and '
    '"hard_negative" (a short passage on a related topic that looks relevant but does '
    "NOT actually answer the query). Return ONLY the JSON object."
)


def _sample_docs(max_docs: int) -> list[str]:
    """Use the positives from stage 1 as raw documents if none are provided."""
    if not settings.pairs_path.exists():
        return []
    docs = []
    for line in settings.pairs_path.read_text(encoding="utf-8").splitlines():
        docs.append(json.loads(line)["positive"])
        if len(docs) >= max_docs:
            break
    return docs


def generate(docs: list[str] | None = None, max_docs: int = 100) -> dict[str, Any]:
    """Generate synthetic triplets. Requires OPENAI_API_KEY; otherwise a no-op."""
    if not settings.openai_api_key:
        return {"skipped": True, "reason": "OPENAI_API_KEY not set", "num_triplets": 0}

    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    docs = docs or _sample_docs(max_docs)
    if not docs:
        return {"skipped": True, "reason": "no documents available", "num_triplets": 0}

    triplets: list[dict[str, str]] = []
    for doc in docs[:max_docs]:
        try:
            resp = client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": f"DOCUMENT:\n{doc[:2000]}"},
                ],
                response_format={"type": "json_object"},
                temperature=0.7,
            )
            payload = json.loads(resp.choices[0].message.content)
            query, neg = payload.get("query"), payload.get("hard_negative")
            if query and neg:
                triplets.append({"anchor": query, "positive": doc, "negative": neg})
        except Exception:
            continue  # skip any doc the model/API fails on

    with settings.triplets_path.open("w", encoding="utf-8") as f:
        for row in triplets:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "skipped": False,
        "model": settings.openai_model,
        "num_triplets": len(triplets),
        "triplets_path": str(settings.triplets_path),
    }


if __name__ == "__main__":
    print(json.dumps(generate(), indent=2))
