# Aero Predict

Maintenance prédictive de turboréacteurs : estimation de la RUL (Remaining Useful Life) sur le dataset NASA C-MAPSS, du notebook à une plateforme MLOps.

## Contexte

Le dataset **C-MAPSS** (NASA) simule la dégradation de turboréacteurs d'avion sous différentes conditions opératoires et modes de défaillance, via des séries temporelles multivariées issues de 21 capteurs. L'objectif est d'estimer la Durée de Vie Utile Restante (RUL) de chaque moteur à partir de ses lectures capteurs, pour anticiper la panne plutôt que la subir — un problème à coût asymétrique : sous-estimer la RUL coûte une maintenance prématurée, la surestimer risque une panne en vol.

Documentation complète :
- [`docs/cahier_des_charges.md`](docs/cahier_des_charges.md) — cahier des charges de référence du projet
- [`docs/00_brief_original.md`](docs/00_brief_original.md) — brief pédagogique source (Digityser / P. Lemaistre)
- [`AGENTS.md`](AGENTS.md) — process et conventions de travail du projet (`CLAUDE.md` est un pointeur vers ce fichier, pour Claude Code)

## Statut du projet

| Phase | Contenu | Statut |
|---|---|---|
| 0 | État de l'art | À faire |
| 1 | EDA et préparation des données | À faire |
| 2 | Target engineering (Piecewise RUL) | À faire |
| 3 | Modélisation (ML classique + Deep Learning) | À faire |
| 4 | Évaluation (RMSE + Score NASA) | À faire |
| 5 | Explicabilité (SHAP) + dashboard | À faire |
| 6 | Plateforme MLOps (bonus) | À faire |

## Benchmark de référence

Avant la mise en place du process ci-dessus, un notebook exploratoire (`notebooks/exploration/baseline_fd001_v0.ipynb`) a établi un premier point de repère sur FD001 (Random Forest, validation croisée par moteur, features statistiques glissantes sur 5 cycles) :

| Évaluation | RMSE | Score NASA |
|---|---|---|
| Validation croisée — moyenne | 18.22 | 56 672.12 |
| Test officiel — brut | 18.97 | 1 053.90 |
| Test officiel — prudent (biais -11 cycles) | 20.27 | 721.91 |

Ce notebook n'a pas suivi la normalisation, le lissage ni la documentation par phase prévus dans le cahier des charges — il sert de **référence chiffrée à battre**, le pipeline officiel repart de la Phase 0.

## Stack technique

Python · scikit-learn, XGBoost, LightGBM · PyTorch · SHAP · FastAPI · SQLite/PostgreSQL · Streamlit/Dash · Docker Compose · MLflow (optionnel) · Matplotlib, Seaborn

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pre-commit install
```

La dernière commande active le hook local qui nettoie automatiquement les outputs des notebooks avant chaque commit (détail : [`CONTRIBUTING.md`](CONTRIBUTING.md)).

## Organisation

- **GitHub** (ce dépôt) = vérité technique et portfolio : code, `docs/`, README.
- **Notion** = coordination d'équipe uniquement (board des 5 étapes par phase, résumés vulgarisés).
