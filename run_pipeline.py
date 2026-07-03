"""Headless end-to-end runner.

Runs every stage in order and prints the before/after result. This is what the Kaggle
notebook calls (Path A) and what you use locally for a quick CPU demo.

Examples
--------
# Full run on a Kaggle GPU:
    python run_pipeline.py --domain fiqa --base-model BAAI/bge-small-en-v1.5 --epochs 1

# Tiny, fast CPU demo (skip the heavy official MTEB task):
    python run_pipeline.py --sample-size 50 --eval-queries 30 --no-mteb
"""

from __future__ import annotations

import argparse
import json

from core.config import settings


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="End-to-end embedding fine-tuning pipeline")
    p.add_argument("--domain", default=settings.domain)
    p.add_argument("--base-model", default=settings.base_model)
    p.add_argument("--epochs", type=int, default=settings.epochs)
    p.add_argument("--batch-size", type=int, default=settings.batch_size)
    p.add_argument("--sample-size", type=int, default=settings.sample_size,
                   help="limit number of training pairs (default: all)")
    p.add_argument("--eval-queries", type=int, default=settings.eval_queries)
    p.add_argument("--num-negatives", type=int, default=settings.num_negatives)
    p.add_argument("--no-mteb", action="store_true", help="skip the official MTEB task (faster)")
    p.add_argument("--llm-triplets", action="store_true",
                   help="use gpt-5.4-nano synthetic triplets instead of hard-negative mining")
    p.add_argument("--run-label", default="",
                   help="short name for this run, shown in the leaderboard")
    return p.parse_args()


def apply_args(args: argparse.Namespace) -> None:
    settings.domain = args.domain
    settings.base_model = args.base_model
    settings.epochs = args.epochs
    settings.batch_size = args.batch_size
    settings.sample_size = args.sample_size
    settings.eval_queries = args.eval_queries
    settings.num_negatives = args.num_negatives
    settings.run_mteb = not args.no_mteb


def main() -> None:
    args = parse_args()
    apply_args(args)

    # Imported after settings are applied so each stage sees the final config.
    from core import (
        baseline, compare, data_prep, evaluate, leaderboard, llm_triplet_gen, train, triplet_mining,
    )

    print(f"[device] {settings.device}  |  base_model={settings.base_model}  domain={settings.domain}")

    print("\n[1/6] Preparing data ...")
    print(json.dumps(data_prep.build_pairs(), indent=2))

    print("\n[2/6] Collecting triplets ...")
    triplet_info = llm_triplet_gen.generate() if args.llm_triplets else triplet_mining.mine()
    print(json.dumps(triplet_info, indent=2))

    print("\n[3/6] MTEB / IR baseline (base model) ...")
    print(json.dumps(baseline.run(), indent=2))

    print("\n[4/6] Fine-tuning ...")
    print(json.dumps(train.finetune(), indent=2))

    print("\n[5/6] Evaluating fine-tuned model ...")
    print(json.dumps(evaluate.run(), indent=2))

    print("\n[6/6] Comparison (before vs after) ...")
    result = compare.diff()
    print(json.dumps(result, indent=2))

    delta = result.get("headline_ir_ndcg@10_delta")
    verdict = "IMPROVED" if result.get("improved") else "NO IMPROVEMENT"
    print(f"\n=== DONE: IR nDCG@10 delta = {delta}  ->  {verdict} ===")

    # Log this run and print the ranked leaderboard of all runs so far.
    leaderboard.record(run_label=args.run_label, num_triplets=triplet_info.get("num_triplets"))
    print("\n=== LEADERBOARD (best first) ===")
    print(leaderboard.show())


if __name__ == "__main__":
    main()
