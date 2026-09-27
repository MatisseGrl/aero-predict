# Phase 3 — Modélisation baseline (XGBoost, FD001)

**Statut** : validée.

## Contexte

Avant cette phase, aucun chiffre de performance n'était reproductible dans le repo : l'ancien notebook
exploratoire produisant RMSE/score NASA a été supprimé ([ADR-0006](decisions/0006-suppression-baseline-exploratoire.md)).
Cette phase entraîne un premier modèle réel, en réutilisant les pipelines validés des Phases 1 et 2
(regroupés dans [`src/preprocessing.py`](../src/preprocessing.py)), et l'évalue sur le jeu de test officiel.
Notebook complet : [`notebooks/03_modelisation.ipynb`](../notebooks/03_modelisation.ipynb).

## Méthode et résultats

### 1. Features par fenêtre glissante

Pour chaque capteur, deux colonnes sont ajoutées : la moyenne et l'écart-type calculés sur les 5 derniers
cycles (moteur par moteur, causal). Elles capturent la tendance et la dispersion récentes du signal —
en particulier la hausse de dispersion en fin de vie déjà démontrée en Phase 2. 45 features au total
(15 capteurs lissés + 15 moyennes glissantes + 15 écarts-types glissants).

Au tout premier cycle de chaque moteur, l'écart-type glissant n'existe pas mathématiquement (une seule
valeur disponible) : les 1500 `NaN` résultants (100 moteurs × 15 capteurs) sont remplacés par 0.

### 2. Score NASA (Saxena et al., 2008)

Implémenté et vérifié par un cas de calcul simple avant utilisation : une surestimation de 20 cycles coûte
`exp(20/10)-1 ≈ 6.39`, une sous-estimation de 20 cycles coûte `exp(20/13)-1 ≈ 3.66` —
l'asymétrie voulue (surestimer, donc risquer une panne en vol, coûte plus cher) est confirmée dans le code.

### 3. Validation croisée par moteur (GroupKFold)

Un même moteur ne peut jamais apparaître à la fois dans le sous-ensemble d'entraînement et de validation
d'un pli : sinon le modèle risquerait de reconnaître partiellement ce moteur plutôt que d'apprendre un
patron général de dégradation, ce qui fausserait l'estimation de performance. Vérifié explicitement à
chaque pli (intersection vide entre les deux ensembles de moteurs).

| Pli | RMSE | Score NASA |
|---|---|---|
| 1 | 19.03 | 55 698.9 |
| 2 | 19.22 | 45 158.5 |
| 3 | 16.20 | 46 919.7 |
| 4 | 20.19 | 82 516.5 |
| 5 | 16.20 | 49 454.1 |
| **Moyenne** | **18.17** | **55 949.6** |

### 4. Évaluation sur le jeu de test officiel

Un modèle final (mêmes hyperparamètres) est entraîné sur les 100 moteurs d'entraînement, puis évalué sur
`test_FD001.txt` (trajectoires tronquées avant la panne) comparé à `RUL_FD001.txt` (RUL vraie fournie
séparément pour le dernier cycle de chaque moteur test). Le même `scaler` que le train est appliqué en
`transform` seul (jamais réajusté sur le test), pour éviter toute fuite d'information.

- **RMSE test = 18.91**
- **Score NASA test = 1 082.6**
- **Moteurs surestimés de plus de 20 cycles : 14/100**

![Prédictions vs RUL vraie, jeu de test officiel](figures/03_predictions_vs_vraies_rul_test.png)

Le nuage de points suit globalement la diagonale idéale, avec un plafond visible autour de 125 : le
modèle ne peut pas prédire au-delà, puisqu'il a été entraîné sur une cible plafonnée à 125 (Phase 2).

## Trade-offs et limites

- Un seul jeu d'hyperparamètres XGBoost a été testé (`n_estimators=200, max_depth=5, learning_rate=0.05`),
  choisi raisonnablement mais sans recherche systématique — une optimisation (grid/random search) pourrait
  améliorer ces chiffres, à envisager en Phase 4 si le temps le permet.
- Ces résultats sont proches de ceux de l'ancien notebook supprimé (RMSE 18.97, score NASA 1053.9,
  cf. [ADR-0002](decisions/0002-archivage-baseline-fd001.md)) : cohérent, la recette est similaire
  (XGBoost, features glissantes, RUL plafonnée, validation par moteur), mais ce sont des chiffres
  différents, obtenus indépendamment avec le code actuel de ce repo — pas une reconstruction de l'ancien.
- Seul XGBoost a été testé (le cahier des charges section 5 en liste trois : XGBoost, LightGBM, SVR) ;
  une comparaison entre les trois n'a pas été faite à ce stade.
- Le comptage "surestimés de plus de 20 cycles" (14/100) utilise un seuil de 20 choisi arbitrairement
  pour l'interprétation, pas une valeur du cahier des charges.
