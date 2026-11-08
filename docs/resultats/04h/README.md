# Résultats de l'optimisation Optuna des réseaux (notebook 04h)

Copie **versionnée** des résultats de deux calculs de nuit :

- **FD001 et FD002** : nuit du 2026-11-07 sur la machine de Matisse (`bash scripts/nuit_04h.sh`, de 0 h 36 à 12 h 34, aucune erreur) ;
- **FD003 et FD004** : du 2026-11-07 à 17 h 57 au 2026-11-08 à 10 h 09 sur le Mac de Benjamin (`bash scripts/nuit_04h_fd003_fd004_mac.sh`, 12 processus, 16 h 12 au total, aucune erreur). Mêmes étapes et mêmes nombres d'essais que pour FD001 et FD002. Seule différence : à 7 h 16, l'étude LSTM de FD004 étant finie, 9 travailleurs CNN 1D de plus (numéros 3 à 11) ont été lancés à la main sur l'étude `04h_fd004_cnn1d` pour occuper les cœurs libres. Ils ont tiré leurs essais dans la même étude, avec le même échantillonneur ; aucun essai n'a fini après le début de l'étape « confirmer » (dernier meilleur essai CNN de FD004, n° 81, fini à 8 h 11 ; « confirmer » a commencé à 8 h 34). Cet essai n° 81 vient du travailleur de renfort n° 10 ; il n'a pas passé la confirmation, et les réglages du 03b sont gardés.

Les originaux sont dans `sorties/resultats/`, qui n'est pas versionné. Le notebook `notebooks/04h_optuna_reseaux.ipynb` relit **ces** fichiers : il ne relance ni Optuna, ni aucun entraînement.

Refaire ces calculs prend environ 11 heures sur tout le processeur pour FD001 et FD002, environ 16 heures pour FD003 et FD004. Les réglages retenus sont aussi recopiés dans le code : `CONFIGS_RESEAUX_RETENUES` et `ENSEMBLES_RETENUS` dans `src/optimisation_reseaux.py`. Les poids des LSTM retenus (5 graines par sous-jeu), et des CNN 1D (non retenus dans l'ensemble, gardés pour comparer), sont dans `models/04h/` (`scripts/sauvegarder_poids_04h.py`).

| Fichier | Contenu |
|---|---|
| `04h_optuna.db` | Base SQLite d'Optuna : les 8 études (`04h_lstm`, `04h_cnn1d` pour FD001 ; `04h_fd002_lstm`, `04h_fd002_cnn1d` ; `04h_fd003_lstm`, `04h_fd003_cnn1d` ; `04h_fd004_lstm`, `04h_fd004_cnn1d`), tous les essais. Remplacée le 2026-11-08 par la base de la nuit FD003 / FD004, après vérification que les 4 études de FD001 et FD002 y sont identiques à la version précédente (mêmes essais, mêmes états, mêmes paramètres, mêmes valeurs) |
| `04h_optuna_<réseau>.csv` / `.json` | FD001 : tableau des essais, meilleur essai et sa configuration |
| `04h_confirmation_<réseau>.json` | FD001 : meilleur essai contre la référence du 03b, sur les plis 42 et 3 découpages neufs (graines 1000 à 1002), règle de décision |
| `04h_ablation_lstm.json` | FD001 : variantes d'entrées (pente glissante, sans capteurs) sur le LSTM retenu |
| `04h_ensemble.json` | FD001 : ensembles (5 graines LSTM, 5 LSTM + 5 CNN) contre le LSTM retenu seul |
| `04h_test.json`, `04h_test_dernier_cycle.csv` | FD001 : test officiel (lu une seule fois), métriques et prédictions au dernier cycle de chaque moteur |
| `04h_fd002_*` | Les mêmes fichiers pour FD002 |
| `04h_fd003_*` | Les mêmes fichiers pour FD003 (calcul du 2026-11-07 / 08, Mac de Benjamin) |
| `04h_fd004_*` | Les mêmes fichiers pour FD004 (même calcul ; étude CNN 1D avec les 9 travailleurs de renfort décrits plus haut) |
| `04h_verification.json` | Contrôle : la référence du 03b redonne son RMSE de validation (8,9421) |
| `04h_nuit.log` | Journal complet de la nuit FD001 / FD002 (non versionné : les fichiers `.log` sont ignorés par git ; original dans `sorties/resultats/`) |
| `04h_nuit_fd003_fd004.log` | Journal de la nuit FD003 / FD004 (non versionné, même raison ; original dans `sorties/resultats/`, avec les journaux de chaque travailleur, dont `04h_fd004_journal_cnn1d_renfort_*.log`) |
| `03b_deep_learning.csv`, `03b_candidats_fd001.json`, `03b_test_fd001.csv` | **Anciens résultats** (notebook 03b, réglages fixés à l'avance), gardés pour la comparaison avant / après |
