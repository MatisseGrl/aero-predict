# ADR-0007 : Choisir le modèle sur validation croisée par une règle fixée à l'avance, avec une pénalité des surestimations dans l'apprentissage

**Statut** : Proposée
**Date** : 2026-11-04
**Décideurs** : Benjamin (proposition, avec Claude Code) — à valider par Matisse et l'équipe lors de la revue de la PR

## Contexte

Jusqu'à la Phase 3, un seul modèle avait été entraîné (XGBoost, 200 arbres, profondeur 5, taux 0,05, perte quadratique symétrique) : test FD001 RMSE 19,70, score NASA 852,9, 9 moteurs sur 100 surestimés de plus de 20 cycles. Lors de la réunion du 29/10, le coach a demandé de justifier ce choix face à d'autres modèles et a proposé deux pistes : faire varier avec Optuna les constantes 10 et 13 du score NASA, et pondérer davantage les surestimations que les sous-estimations.

Mesures faites sur FD001 (`notebooks/04_comparaison_evaluation.ipynb`, mêmes 5 plis `GroupKFold` par moteur pour tous les modèles) :

- À réglages de base, XGBoost, LightGBM et Random Forest ont le même RMSE de validation croisée (16,87 / 16,85 / 16,96 ; tests appariés sur 100 moteurs, p ≥ 0,58). XGBoost s'entraîne en 3,9 s pour 5 plis contre 28 s pour Random Forest.
- Changer les constantes 13/10 du score d'évaluation ne change aucune prédiction : le score passe de 852,9 à 0,13 pour un facteur 1 000. Optimiser ces constantes dans la note serait dégénéré.
- Placée dans l'apprentissage, une pénalité des surestimations réduit les surestimations en validation croisée, au prix d'un peu de RMSE et de plus de fausses alertes.

## Décision

1. Le score NASA d'évaluation garde ses constantes officielles (13 en sous-estimation, 10 en surestimation). Ce qui varie, c'est la perte d'apprentissage.
2. Le modèle est choisi uniquement sur la validation croisée, par une règle écrite avant tout résultat : parmi les essais Pareto-optimaux (RMSE CV, score NASA CV) des 3 études Optuna (160 essais, TPE graine 42), garder ceux dont le RMSE CV est au plus 1,05 × le meilleur RMSE CV du front, puis prendre celui qui a le plus petit score NASA CV. Le jeu de test n'intervient pas dans le choix.
3. Modèle à arbres retenu : XGBoost avec perte quadratique asymétrique, alpha = 2,26 (essai 17 de l'étude `xgb_quadratique`), configuration complète dans [`docs/resultats/04_config_retenue.json`](../resultats/04_config_retenue.json).

Portée : cette ADR ne tranche que le choix parmi les modèles à arbres de FD001. Elle a été proposée avant les résultats du deep learning ([`notebooks/03b_deep_learning_lstm_cnn.ipynb`](../../notebooks/03b_deep_learning_lstm_cnn.ipynb)) : sur FD001, le LSTM obtient un RMSE de validation croisée de 13,70 (mêmes plis) et un RMSE test de 13,86 (score NASA 293,2, 0 alerte manquée), significativement meilleur que les deux XGBoost (p = 0,0009 sur l'erreur absolue par moteur). Le choix de la famille de modèle pour la suite (arbres ou LSTM) reste à faire par l'équipe, dans une nouvelle ADR, avec une règle écrite avant de regarder le test.

## Alternatives envisagées

- **Garder XGBoost Phase 3 (perte symétrique)** : écartée parce qu'en validation croisée, le modèle retenu fait 36 % de surestimations de plus de 20 cycles en moins (p < 0,0001) et un score NASA 29 % plus bas (p = 0,007), sans différence de RMSE par moteur détectable (p = 0,67).
- **Prendre le meilleur RMSE d'Optuna** (étude `xgb_nasa`, essai 42, RMSE CV 16,48) : écartée par la règle, parce que son score NASA CV (24 586) est moins bon que celui du retenu (21 075). Il obtient un meilleur résultat sur le test (RMSE 18,69, score NASA 646,9), mais choisir sur le test serait une fuite.
- **Optimiser les constantes 13/10 du score avec Optuna** : écartée, ça change la note sans changer aucune prédiction.
- **Perte d'apprentissage de forme NASA** (dénominateurs `h_sous`/`h_sur`, version « apprentissage » de la piste du coach) : écartée comme perte par défaut. Une fois convergée (1 000 arbres), elle est moins bonne que la Phase 3 en validation croisée, en RMSE comme en score NASA ; son bon résultat à 200 arbres tient à l'arrêt précoce.
- **LightGBM ou Random Forest** : même précision que XGBoost, entraînement plus lent ; Random Forest n'accepte pas de perte sur mesure.

## Conséquences

**Positives** :
- Le choix du modèle est reproductible et défendable : règle écrite avant les résultats, test jamais utilisé pour choisir, essais Optuna versionnés dans `docs/resultats/`.
- La piste « pénaliser les surestimations » est mesurée, pas supposée.

**Négatives / coûts acceptés** :
- Sur le test officiel, aucune différence significative avec la Phase 3 : RMSE 19,73 contre 19,70, 9 surestimés dans les deux cas, 7 alertes manquées dans les deux cas (p = 0,94 sur les erreurs absolues par moteur). L'écart de score NASA (675,4 contre 852,9) vient presque entièrement du moteur 37 (sans lui : 646,1 contre 629,3).
- RMSE de validation croisée plus élevé (17,28 contre 16,87) et plus de fausses alertes en validation croisée (381 contre 259, seuil de 30 cycles qui reste une hypothèse de travail).
- La configuration a été choisie sur FD001 seulement, et elle ne se transfère pas aux autres sous-jeux ([`notebooks/04b_sous_jeux_fd002_fd004.ipynb`](../../notebooks/04b_sous_jeux_fd002_fd004.ipynb)). Sur le test de FD002 et FD004, le modèle retenu est significativement moins bon que la Phase 3 : erreur absolue +1,4 et +1,9 cycle par moteur (p ≤ 0,0002), RMSE 29,60 contre 27,23 (FD002) et 30,95 contre 28,24 (FD004). En validation croisée, il réduit toujours les surestimations de plus de 20 cycles sur les 4 sous-jeux (p < 0,0001). La décision ne vaut donc que pour FD001 ; pour FD002–FD004, une réoptimisation par sous-jeu est à faire avant tout choix.
- La perte sur mesure est une fonction Python interne : le modèle doit être sérialisé avec cloudpickle (voir `src/demo.py`) et dépend de `src/modeles.py` pour être rechargé.
- Les p-valeurs de validation croisée sont optimistes, parce que le candidat a été choisi parmi 160 essais évalués sur les mêmes plis.
