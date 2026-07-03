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
