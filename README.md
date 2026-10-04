# Aero Predict

Predictive maintenance for turbofan engines: estimating RUL (Remaining Useful Life) on the NASA C-MAPSS dataset, from notebook to MLOps platform.

## Context

The NASA **C-MAPSS** dataset simulates the degradation of aircraft turbofan engines under various operating conditions and failure modes, as multivariate time series from 21 sensors. The goal is to estimate each engine's Remaining Useful Life (RUL) from its sensor readings, so that failures can be anticipated rather than suffered. It is an asymmetric-cost problem: underestimating the RUL costs a premature maintenance, overestimating it risks an in-flight failure.

Full documentation (written in French):
- [`docs/cahier_des_charges.md`](docs/cahier_des_charges.md): the project's reference specification
- [`docs/00_brief_original.md`](docs/00_brief_original.md): original teaching brief (Digityser / P. Lemaistre)
- [`AGENTS.md`](AGENTS.md): project working process and conventions (`CLAUDE.md` is a pointer to this file, for Claude Code)

## Project status

| Phase | Content | Status |
|---|---|---|
| 0 | State of the art | ✅ Done: [docs/00_etat_de_l_art.md](docs/00_etat_de_l_art.md) |
| 1 | EDA and data preparation | ✅ Done: [docs/01_eda_preparation.md](docs/01_eda_preparation.md) |
| 2 | Target engineering (Piecewise RUL) | ✅ Done: [docs/02_target_engineering.md](docs/02_target_engineering.md) |
| 3 | Modeling (classical ML + Deep Learning) | ✅ Done (XGBoost baseline): [docs/03_modelisation.md](docs/03_modelisation.md) |
| 4 | Evaluation (RMSE + NASA Score) | To do |
| 5 | Explainability (SHAP) + dashboard | To do |
| 6 | MLOps platform (bonus) | To do |

## Reference benchmark

Before the process above was put in place, an exploratory notebook established a first reference point on FD001 (Random Forest, per-engine cross-validation, rolling statistical features over 5 cycles):

| Evaluation | RMSE | NASA Score |
|---|---|---|
| Cross-validation: mean | 18.22 | 56,672.12 |
| Official test: raw | 18.97 | 1,053.90 |
| Official test: conservative (bias of -11 cycles) | 20.27 | 721.91 |

This notebook did not follow the normalization, smoothing or per-phase documentation required by the specification: it serves as a **numerical reference to beat**, and the official pipeline restarts from Phase 0. The file itself has since been deleted (see [ADR-0006](docs/decisions/0006-suppression-baseline-exploratoire.md)); only these figures are kept.

**This notebook no longer exists and its code cannot be re-executed.** The 6 figures in the table above are the only ones kept, as they are: **no other statistic may be derived from them** (e.g. a count such as "X engines out of 100 in a given case") without re-coding and re-verifying the calculation from scratch, with a current notebook and a cited source. A figure derived from this benchmark without reproducible code almost got used unverified in a presentation (2026-09-27): this table must never again influence a decision or a claim that is not directly one of these 6 numbers.

## Tech stack

Python · scikit-learn, XGBoost, LightGBM · PyTorch · SHAP · FastAPI · SQLite/PostgreSQL · Streamlit/Dash · Docker Compose · MLflow (optional) · Matplotlib, Seaborn

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pre-commit install
```

The last command enables the local hook that automatically cleans notebook outputs before each commit (details: [`CONTRIBUTING.md`](CONTRIBUTING.md)).

## Organization

- **GitHub** (this repository) = technical source of truth and portfolio: code, `docs/`, README.
- **Notion** = team coordination only (board of the 5 steps per phase, plain-language summaries).
