# Résultats du notebook 03b (réseaux à réglages fixés à l'avance), exécution du 2026-11-04

Copie **versionnée** des fichiers écrits par `scripts/entrainer_deep_learning.py` (options par défaut et `--cv-sous-jeux`) lors de l'exécution du 2026-11-04 au soir, sur la machine de Benjamin. Les originaux sont dans `sorties/resultats/` (non versionné) du dossier v1 (`aero-predict`).

**Pourquoi cette copie** (réorganisation des notebooks, revue du 2026-11-07) : `notebooks/02_choix_du_modele.ipynb` et `notebooks/evolution.ipynb` lisent ces résultats pour montrer ce qui a fondé les décisions sur les réseaux (fenêtre, `time_cycles`, réseaux contre arbres, écart à l'état initial). Sans copie versionnée, ces deux notebooks ne se réexécutaient pas depuis un clone neuf, et ils lisaient deux exécutions différentes du 03b (02 celle-ci, `evolution` celle de `docs/resultats/04h/`), d'où deux chiffres différents pour le même CNN 1D.

**Deux exécutions du 03b existent, et elles ne donnent pas exactement les mêmes chiffres** (les réseaux ne sont pas reproductibles au bit près d'une machine à l'autre) :

| | Cette copie (2026-11-04, Benjamin) | `docs/resultats/04h/03b_*` (réentraînement du 2026-11-05, Matisse) |
|---|---|---|
| LSTM FD001, validation croisée / test (graine 42) | 8,94 / 12,29 | 8,94 / 12,29 |
| CNN 1D FD001, validation croisée / test (graine 42) | 10,05 / 12,74 | 10,15 / 13,09 |
| Sous-jeux | FD001 à FD004 | FD001 seulement |

Les chiffres de `docs/STATUS.md` (entrée du 2026-11-05) et le texte du notebook 03b sont ceux de **cette** copie. Les fichiers de `docs/resultats/04h/` restent ceux que relit le notebook 04h (comparaison avant / après Optuna) ; ils ne sont pas modifiés. Le choix de l'exécution qui fait foi pour la v2 reste à confirmer par l'équipe.

| Fichier | Contenu |
|---|---|
| `03b_choix_entrees_fd001.csv` / `.json` | Faut-il donner `time_cycles` aux réseaux ? (validation croisée FD001, avec et sans) |
| `03b_cv_fd001.csv` à `03b_cv_fd004.csv` | Validation croisée des réseaux, 5 plis de graine 42 (ceux des arbres), par sous-jeu |
| `03b_oof_fd001.csv.gz` | Prédictions hors-pli des réseaux sur FD001 (tests appariés contre les arbres) |
| `03b_historiques_fd001.json` | Courbes d'apprentissage époque par époque, pli par pli (FD001) |
| `03b_test_fd001.csv` | Test officiel FD001, 3 graines par réseau |
| `03b_predictions_test_fd001.csv` | Prédiction au dernier cycle de chaque moteur test FD001, par réseau et graine |
| `03b_candidats_fd001.csv` / `.json` | Candidats d'amélioration des réseaux (écart à l'état initial, fenêtre de 50 cycles), règle de décision |
| `03b_deep_learning.csv` | Version retenue des réseaux, FD001 à FD004 : validation et test officiel |
