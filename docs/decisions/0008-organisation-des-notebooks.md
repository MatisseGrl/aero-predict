# ADR-0008 : Quatre notebooks écrits à la main et quatre notebooks générés remplacent les 14 notebooks par phase

**Statut** : Acceptée
**Date** : 2026-11-08
**Décideurs** : Matisse (accord pour le retrait des anciens notebooks, 2026-11-08), Benjamin (exécution des étapes B à D du plan)

## Contexte

Au 2026-11-06, `notebooks/` contenait 13 notebooks (01, 01b, 02, 03, 03b, 04a à 04g, 05), auxquels s'est ajouté le 04h (Optuna des réseaux). Ils suivaient l'ordre dans lequel l'équipe a travaillé, pas l'ordre des questions d'un lecteur : les résultats de FD002 étaient répartis entre 01b, 03b et 05, et le protocole d'évaluation ([`docs/protocole_evaluation.md`](../protocole_evaluation.md)) n'était déroulé en entier que sur FD001. Le hook `nbstripout` effaçait les sorties de tous les notebooks, donc aucun graphique n'était visible sur GitHub.

Le plan [`docs/plan_refonte_notebooks.md`](../plan_refonte_notebooks.md) a fixé une nouvelle organisation. Benjamin a exécuté les étapes B à D (nouveaux notebooks, `src/evaluation_sous_jeu.py`, `src/figures_resultats.py`, tests) et livré [`docs/comparaison_v1_v2.md`](../comparaison_v1_v2.md), qui met en regard les chiffres, les cellules et les figures de la v1 et de la v2.

## Décision

`notebooks/` ne contient plus que :

- `01_donnees.ipynb` : préparation des 4 sous-jeux ;
- `02_choix_du_modele.ipynb` : tout ce qui est décidé, sur FD001 ;
- `resultats/_modele.ipynb`, seul notebook de résultats écrit à la main, et `resultats/FD001.ipynb` à `FD004.ipynb`, générés depuis lui par `scripts/generer_notebooks_resultats.py` et jamais modifiés à la main ;
- `evolution.ipynb` : le parcours du projet, avant / après.

Les 14 anciens notebooks (01_eda_preparation, 01b, 02_target_engineering, 03_modelisation, 03b, 04a à 04h, 05) sont retirés de la branche. Ils restent consultables au tag git `v1-avant-reorganisation` (commit `efe9434`, poussé sur GitHub). Les notebooks de la v2 gardent leurs sorties dans git, par une exclusion `nbstripout` dans `.pre-commit-config.yaml`.

## Alternatives envisagées

- **Garder les anciens notebooks à côté des nouveaux** : écartée, deux versions du même calcul coexisteraient dans `notebooks/` et finiraient par diverger ; le lecteur ne saurait pas laquelle fait foi.
- **Déplacer les anciens notebooks dans `notebooks/exploration/`** : écartée, même problème de duplication, et le tag git donne déjà un accès en lecture sans les maintenir.
- **Garder les sorties par la métadonnée `keep_output` de chaque notebook** plutôt que par une exclusion dans `.pre-commit-config.yaml` : écartée par Benjamin (plan, journal du 2026-11-07), les `FD00X.ipynb` étant réécrits à chaque génération.

## Conséquences

**Positives** :
- Une question du jury sur un sous-jeu renvoie à un seul notebook, `resultats/FD00X.ipynb`, avec le même plan pour les 4 sous-jeux.
- La section 7 « Vérifier un moteur » recalcule une prédiction de la démo depuis `data/test_FD00X.txt` ; `tests/test_evaluation_sous_jeu.py` vérifie qu'elle redonne les fichiers de la démo (écart max 1,5e-5 sur 7 moteurs, d'après le journal du plan).
- Les figures des notebooks de la v2 sont enregistrées dans les fichiers (76 images : 4 + 17 + 19 + 4 × 9) : chaque `plt.show()` a son image (vérifié le 2026-11-08), visibles sur GitHub sans exécution.

**Négatives / coûts acceptés** :
- L'historique cellule par cellule des anciens notebooks n'est plus consultable que via le tag `v1-avant-reorganisation` ou `git log`, pas dans l'arborescence courante.
- Les entrées passées de `docs/STATUS.md` et plusieurs docstrings de `src/` (`config_modele.py`, `features.py`, `preprocessing.py`) et de `scripts/` citent les anciens notebooks par leur chemin ; ces chemins n'existent plus sur la branche. Ils ne sont pas réécrits (règle 10 pour STATUS) : ils se lisent au tag.
- 3 figures de la v1 n'ont pas d'équivalent exact en v2 (04b §7 en validation à N = 15, 30 et 50 ; 03b F.7 sur deux moteurs ; premier panneau du 04d, cellule 12), listées dans `docs/comparaison_v1_v2.md`. Le retrait des anciens notebooks les laisse disponibles seulement au tag.
- Environ 14 Mo de notebooks avec images dans git, et des diffs plus longs à chaque régénération.
- Écarts encore ouverts au moment du retrait, non tranchés par cette ADR : XGBoost FD001 au test à 12,78 (machine de génération) contre 12,73 (référence) ; réseaux Optuna sur FD003 et FD004, contraires à la règle du plan du 2026-11-07 ; repli k = 20 pour l'écart à l'état initial sur FD004 (moteur test le plus court : 19 cycles).
