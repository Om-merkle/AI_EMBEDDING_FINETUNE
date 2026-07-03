"""Central configuration for the whole pipeline.

Every tunable lives here so the app, the CLI and the Kaggle notebook all behave
identically. Values can be overridden with environment variables (or a .env file)
using the exact field name in UPPER_CASE, e.g. `SAMPLE_SIZE=200`.
"""

from __future__ import annotations

from pathlib import Path

import torch
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = the folder that contains this `core/` package.
ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ---- Model / domain ----
    base_model: str = "BAAI/bge-small-en-v1.5"
    domain: str = "fiqa"                    # dataset key (see data_prep.DATASETS)
    mteb_task: str = "FiQA2018"             # MTEB retrieval task matching the domain

    # ---- Data sizing (keep small for CPU demos, raise on Kaggle GPU) ----
    sample_size: int | None = None         # of training pairs; None = use all
    eval_queries: int = 100                # how many eval queries for the quick IR metric
    eval_corpus_size: int = 5000           # cap corpus size for the quick IR metric
    num_negatives: int = 3                 # hard negatives mined per (query, positive)

    # ---- Training hyper-parameters ----
    epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1

    # ---- Optional MTEB run (the "real" domain baseline; slower) ----
    run_mteb: bool = True                  # also run the official MTEB task, not just the quick IR metric

    # ---- Optional LLM triplet generation ----
    openai_api_key: str = ""
    openai_model: str = "gpt-5.4-nano"

    # ---- Optional Hub push ----
    hf_token: str = ""

    # ---- Paths (created on demand) ----
    data_dir: Path = ROOT / "data"
    models_dir: Path = ROOT / "models"
    results_dir: Path = ROOT / "results"
    mteb_dir: Path = ROOT / "mteb_results"

    @property
    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def use_fp16(self) -> bool:
        # fp16 only helps on CUDA; bge models train fine with it.
        return self.device == "cuda"

    @property
    def finetuned_model_dir(self) -> Path:
        return self.models_dir / f"{self.domain}-{Path(self.base_model).name}-ft"

    @property
    def pairs_path(self) -> Path:
        return self.data_dir / "pairs.jsonl"

    @property
    def triplets_path(self) -> Path:
        return self.data_dir / "triplets.jsonl"

    @property
    def eval_path(self) -> Path:
        return self.data_dir / "eval.json"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.models_dir, self.results_dir, self.mteb_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
