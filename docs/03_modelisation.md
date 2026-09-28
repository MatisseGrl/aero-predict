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

`GroupKFold(n_splits=5, shuffle=True, random_state=42)` : par défaut, `GroupKFold` répartit les moteurs
entre les plis selon leur ordre d'apparition dans le fichier, pas au hasard — un choix qui pourrait
biaiser chaque pli si cet ordre correspondait à autre chose qu'un simple ordre de simulation. Vérifié :
avec `shuffle=True`, le RMSE moyen ne change presque pas (18.61 sans mélange vs 18.64 avec) — donc pas de
biais caché ici — mais le mélange (avec une graine fixe pour rester reproductible) est conservé pour ne
plus jamais avoir à supposer que l'ordre du fichier n'a pas d'importance.

| Pli | RMSE | Score NASA |
|---|---|---|
| 1 | 16.58 | 34 222.3 |
| 2 | 20.22 | 32 603.8 |
| 3 | 18.09 | 74 407.3 |
| 4 | 18.34 | 64 938.7 |
| 5 | 19.97 | 105 223.5 |
| **Moyenne** | **18.64** | **62 279.1** |

**Note de reproductibilité** : ces chiffres de validation croisée peuvent varier légèrement d'une
exécution à l'autre (observé : 18.17 puis 18.61 sur deux runs successifs, avant l'ajout du `shuffle`
ci-dessus), malgré `random_state=42` fixé sur `XGBRegressor`. XGBoost parallélise la construction des
arbres sur plusieurs cœurs, et l'ordre des opérations flottantes en parallèle n'est pas garanti identique
d'un environnement à l'autre — un phénomène connu, pas un bug du code. Le modèle final et son évaluation
sur le test officiel (section 4 ci-dessous), en revanche, sont restés strictement identiques sur les
trois runs.

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
