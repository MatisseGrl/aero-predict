# Résultats de l'optimisation Optuna des réseaux (notebook 04h)

Copie **versionnée** des résultats du calcul de nuit du 2026-11-07 (`bash scripts/nuit_04h.sh`, de 0 h 36 à 12 h 34, aucune erreur). Les originaux sont dans `sorties/resultats/`, qui n'est pas versionné. Le notebook `notebooks/04h_optuna_reseaux.ipynb` relit **ces** fichiers : il ne relance ni Optuna, ni aucun entraînement.

Refaire ces calculs prend environ 11 heures sur tout le processeur. Les réglages retenus sont aussi recopiés dans le code : `CONFIGS_RESEAUX_RETENUES` et `ENSEMBLES_RETENUS` dans `src/optimisation_reseaux.py`. Les poids des 10 LSTM retenus sont dans `models/04h/` (`scripts/sauvegarder_poids_04h.py`).

| Fichier | Contenu |
|---|---|
| `04h_optuna.db` | Base SQLite d'Optuna : les 4 études (`04h_lstm`, `04h_cnn1d` pour FD001 ; `04h_fd002_lstm`, `04h_fd002_cnn1d`), tous les essais |
| `04h_optuna_<réseau>.csv` / `.json` | FD001 : tableau des essais, meilleur essai et sa configuration |
| `04h_confirmation_<réseau>.json` | FD001 : meilleur essai contre la référence du 03b, sur les plis 42 et 3 découpages neufs (graines 1000 à 1002), règle de décision |
| `04h_ablation_lstm.json` | FD001 : variantes d'entrées (pente glissante, sans capteurs) sur le LSTM retenu |
| `04h_ensemble.json` | FD001 : ensembles (5 graines LSTM, 5 LSTM + 5 CNN) contre le LSTM retenu seul |
| `04h_test.json`, `04h_test_dernier_cycle.csv` | FD001 : test officiel (lu une seule fois), métriques et prédictions au dernier cycle de chaque moteur |
| `04h_fd002_*` | Les mêmes fichiers pour FD002 |
| `04h_verification.json` | Contrôle : la référence du 03b redonne son RMSE de validation (8,9421) |
| `04h_nuit.log` | Journal complet de la nuit (non versionné : les fichiers `.log` sont ignorés par git ; original dans `sorties/resultats/`) |
| `03b_deep_learning.csv`, `03b_candidats_fd001.json`, `03b_test_fd001.csv` | **Anciens résultats** (notebook 03b, réglages fixés à l'avance), gardés pour la comparaison avant / après |
