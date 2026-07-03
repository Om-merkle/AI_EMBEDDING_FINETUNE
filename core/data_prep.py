"""Stage 1 - Data preparation.

Loads a public retrieval dataset (default: FiQA, financial QA) from the Hugging Face
Hub and builds two things:

  1. (anchor, positive) training pairs from the qrels  -> data/pairs.jsonl
  2. a small evaluation set (queries, corpus, relevant_docs) for a fast, consistent
     Information-Retrieval metric that we can run on BOTH the base and fine-tuned
     model                                             -> data/eval.json

FiQA is chosen because it is small, ungated, and is also an official MTEB retrieval
task (`FiQA2018`), so the baseline and the post-training evaluation measure the same
thing on the same domain -> an honest before/after comparison.
"""

from __future__ import annotations

import json
import random
from typing import Any

from datasets import Dataset, load_dataset

from core.config import settings

# Map a friendly domain key -> the BeIR dataset id used to build training data.
# (BeIR ships corpus / queries / qrels which is exactly what we need.)
DATASETS: dict[str, str] = {
    "fiqa": "BeIR/fiqa",
}

_RNG = random.Random(42)


def _load(hf_id: str, config: str | None = None, split: str | None = None):
    """load_dataset with a graceful fallback for datasets that need remote code."""
    try:
        return load_dataset(hf_id, config, split=split)
    except Exception:
        return load_dataset(hf_id, config, split=split, trust_remote_code=True)


def _corpus_text(row: dict[str, Any]) -> str:
    title = (row.get("title") or "").strip()
    text = (row.get("text") or "").strip()
    return f"{title} {text}".strip()


def _build_lookup(hf_id: str) -> tuple[dict[str, str], dict[str, str]]:
    """Return (corpus_by_id, queries_by_id) as {id -> text} dicts."""
    corpus_ds = _load(hf_id, "corpus")
    queries_ds = _load(hf_id, "queries")
    # These datasets expose a single split named "corpus" / "queries".
    corpus_ds = corpus_ds["corpus"] if hasattr(corpus_ds, "keys") else corpus_ds
    queries_ds = queries_ds["queries"] if hasattr(queries_ds, "keys") else queries_ds

    corpus = {str(r["_id"]): _corpus_text(r) for r in corpus_ds}
    queries = {str(r["_id"]): (r["text"] or "").strip() for r in queries_ds}
    return corpus, queries


def _load_qrels(hf_id: str, split: str):
    qrels = _load(f"{hf_id}-qrels")
    if split not in qrels:
        # Fall back to whatever splits exist (some mirrors only ship "test").
        split = "test" if "test" in qrels else next(iter(qrels.keys()))
    return qrels[split]


def build_pairs() -> dict[str, Any]:
    """Build training pairs + an evaluation set. Writes pairs.jsonl and eval.json."""
    hf_id = DATASETS[settings.domain]
    corpus, queries = _build_lookup(hf_id)

    # ---- Training pairs from the TRAIN qrels (positives only: score > 0) ----
    train_qrels = _load_qrels(hf_id, "train")
    pairs: list[dict[str, str]] = []
    for r in train_qrels:
        if int(r["score"]) <= 0:
            continue
        q = queries.get(str(r["query-id"]))
        d = corpus.get(str(r["corpus-id"]))
        if q and d:
            pairs.append({"anchor": q, "positive": d})

    _RNG.shuffle(pairs)
    if settings.sample_size:
        pairs = pairs[: settings.sample_size]

    with settings.pairs_path.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # ---- Evaluation set from the TEST qrels ----
    eval_blob = _build_eval(hf_id, corpus, queries)

    summary = {
        "domain": settings.domain,
        "num_pairs": len(pairs),
        "num_eval_queries": len(eval_blob["queries"]),
        "eval_corpus_size": len(eval_blob["corpus"]),
        "pairs_path": str(settings.pairs_path),
        "eval_path": str(settings.eval_path),
    }
    return summary


def _build_eval(hf_id: str, corpus: dict[str, str], queries: dict[str, str]) -> dict[str, Any]:
    """Create a compact IR eval set: N queries, their positives + distractors."""
    test_qrels = _load_qrels(hf_id, "test")

    relevant: dict[str, set[str]] = {}
    for r in test_qrels:
        if int(r["score"]) <= 0:
            continue
        qid, cid = str(r["query-id"]), str(r["corpus-id"])
        if qid in queries and cid in corpus:
            relevant.setdefault(qid, set()).add(cid)

    # Keep only a manageable number of eval queries.
    qids = list(relevant.keys())
    _RNG.shuffle(qids)
    qids = qids[: settings.eval_queries]

    eval_queries = {qid: queries[qid] for qid in qids}
    relevant = {qid: relevant[qid] for qid in qids}

    # Corpus = every relevant doc + random distractors, capped at eval_corpus_size.
    needed = set().union(*relevant.values()) if relevant else set()
    distractor_pool = [cid for cid in corpus.keys() if cid not in needed]
    _RNG.shuffle(distractor_pool)
    room = max(0, settings.eval_corpus_size - len(needed))
    corpus_ids = list(needed) + distractor_pool[:room]
    eval_corpus = {cid: corpus[cid] for cid in corpus_ids}

    blob = {
        "queries": eval_queries,
        "corpus": eval_corpus,
        "relevant_docs": {qid: sorted(cids) for qid, cids in relevant.items()},
    }
    with settings.eval_path.open("w", encoding="utf-8") as f:
        json.dump(blob, f, ensure_ascii=False)
    return blob


# ---- Loaders used by later stages ----------------------------------------------------

def load_pairs_dataset() -> Dataset:
    """Return the (anchor, positive) pairs as a Hugging Face Dataset."""
    rows = [json.loads(line) for line in settings.pairs_path.read_text(encoding="utf-8").splitlines()]
    return Dataset.from_list(rows)


def load_eval() -> dict[str, Any]:
    """Return {queries, corpus, relevant_docs} for the Information-Retrieval metric."""
    return json.loads(settings.eval_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    print(json.dumps(build_pairs(), indent=2))
