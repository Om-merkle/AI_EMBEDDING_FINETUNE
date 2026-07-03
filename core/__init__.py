"""Core embedding fine-tuning package.

A single shared package that holds ALL logic (data prep -> triplet mining -> MTEB
baseline -> train -> evaluate -> compare). It is imported unchanged by:

  * the FastAPI backend (api/main.py),
  * the Streamlit UI (app/streamlit_app.py),
  * the headless runner (run_pipeline.py), and
  * the Kaggle notebook (notebooks/run_on_kaggle.ipynb).

"Write once, run anywhere."

Note: submodules import `from core.config import settings` directly. This __init__ is
kept import-light on purpose (it does NOT pull in torch), so lightweight modules like
`core.jobs` can be used without the heavy ML stack installed.
"""

import os

# Force single-GPU by default. sentence-transformers' in-batch-negative losses (e.g.
# MultipleNegativesRankingLoss) don't work with PyTorch DataParallel on multi-GPU boxes
# (like Kaggle's 2x T4): every replica's embeddings get gathered onto GPU 0, which then
# OOMs. One T4 is plenty for bge-small. This runs before any submodule imports torch, so
# torch only ever sees a single device. Multi-GPU users can override by exporting
# CUDA_VISIBLE_DEVICES themselves (setdefault respects an existing value).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
