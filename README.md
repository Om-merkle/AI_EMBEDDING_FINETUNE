# Embedding Fine-Tuning — end to end

A simple, understandable application that fine-tunes a text-embedding model and proves it got
better on your domain. It covers all three milestones:

1. **MTEB domain baseline run** — measure the base model *before* training.
2. **(query, positive, negative) triplet collection** — auto hard-negative mining (or optional LLM generation).
3. **FT framework set up (sentence-transformers)** — fine-tune, evaluate, and compare vs baseline.

**Stack:** Python · sentence-transformers · MTEB · FastAPI · Streamlit
**Default model:** `BAAI/bge-small-en-v1.5` · **Default domain:** FiQA (financial QA, MTEB task `FiQA2018`)

---

## How it works

One shared package (`core/`) holds all logic; the API, the UI, the CLI and the Kaggle notebook all
call the same functions.

```
prepare data → collect triplets → MTEB/IR baseline → fine-tune → evaluate → compare
   core/data_prep   core/triplet_mining   core/baseline   core/train   core/evaluate   core/compare
```

| File | Role |
|---|---|
| `core/config.py` | all settings (model, domain, sizes, hyper-params) |
| `core/data_prep.py` | load FiQA → `(anchor, positive)` pairs + eval set |
| `core/triplet_mining.py` | `util.mine_hard_negatives` → `(anchor, positive, negative)` |
| `core/llm_triplet_gen.py` | *optional* synthetic triplets via `gpt-5.4-nano` |
| `core/baseline.py` | evaluate the base model (IR metric + optional MTEB) |
| `core/train.py` | `SentenceTransformerTrainer` + `MultipleNegativesRankingLoss` |
| `core/evaluate.py` | evaluate the fine-tuned model |
| `core/compare.py` | before/after nDCG@10 table |
| `api/main.py` | FastAPI endpoints (long steps run as background jobs) |
| `app/streamlit_app.py` | click-through UI + before/after chart |
| `run_pipeline.py` | one-shot headless runner (CLI) |
| `notebooks/run_on_kaggle.ipynb` | run everything on a free Kaggle GPU |

---

## ▶️ Run on Kaggle (recommended — free T4 GPU)

**One-time:** sign in at kaggle.com → verify your phone (unlocks GPU + Internet).

1. Push this project to GitHub *(or upload it as a Kaggle Dataset).*
2. Kaggle → **Create → Notebook**. In **Settings**: **Accelerator = GPU T4 ×2**, **Internet = On**.
3. Open `notebooks/run_on_kaggle.ipynb` (or paste its cells), set your repo URL, and run:
   - **Cell 1** — `git clone` + `pip install -r requirements.txt`
   - **Path A** — `!python run_pipeline.py --domain fiqa --base-model BAAI/bge-small-en-v1.5 --epochs 1 --batch-size 32`
   - print `results/comparison.json` → **fine-tuned nDCG@10 should beat the baseline**.
4. The model (`models/`) and metrics (`results/`) show up in the notebook's **Output** tab to download.
5. *(Optional)* **Path B** runs the FastAPI + Streamlit UI behind a public `cloudflared` URL.

---

## 💻 Run locally (Windows, CPU — tiny demo or UI)

```bash
pip install -r requirements.txt

# Tiny end-to-end demo on CPU (fast; skips the heavy official MTEB task):
python run_pipeline.py --sample-size 50 --eval-queries 30 --no-mteb

# Or the full app (two terminals):
uvicorn api.main:app --reload            # terminal 1 → http://localhost:8000/docs
streamlit run app/streamlit_app.py       # terminal 2 → http://localhost:8501
```

> On CPU, keep `--sample-size` small. Full-scale training should run on Kaggle.

---

## Optional: LLM triplet generation

Copy `.env.example` → `.env` and set `OPENAI_API_KEY`. Then instead of mining:
```bash
python run_pipeline.py --llm-triplets --sample-size 50
```
If no key is set, this path is skipped automatically — mining remains the default.

---

## What "success" looks like

`results/comparison.json` shows the base vs fine-tuned model side by side. Success = the fine-tuned
**IR nDCG@10** (and the official **MTEB FiQA2018 nDCG@10**) is measurably higher than the baseline.

## Per-run leaderboard

Every run appends its metrics to `results/leaderboard.csv` and prints a ranked table (best first),
so you can compare experiments — different base models, epochs, batch sizes, or domains. Tag a run
with `--run-label`:
```bash
python run_pipeline.py --base-model BAAI/bge-small-en-v1.5 --run-label bge-1ep
python run_pipeline.py --base-model sentence-transformers/all-MiniLM-L6-v2 --run-label minilm-1ep
```
View it any time (ranked by MTEB nDCG@10, then IR nDCG@10):
```python
from core import leaderboard
print(leaderboard.show())          # text table
leaderboard.to_dataframe()         # pandas DataFrame (nice in a notebook)
```
Or via the API: `GET /leaderboard`. The CSV persists across runs and downloads from Kaggle's Output tab.
