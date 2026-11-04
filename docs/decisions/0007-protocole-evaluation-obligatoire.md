# ADR-0007 : Un protocole d'évaluation unique et obligatoire pour tout modèle

**Statut** : Acceptée
**Date** : 2026-11-04
**Décideurs** : Matisse

## Contexte

En relisant le cours *Introduction to ML* (« Code cours 2 — Aspects pratiques du ML » : découpage, diagnostic biais/variance, métriques de régression et de classification, grid search et random search, feature engineering), Matisse a constaté que la Phase 3 (XGBoost, FD001) était présentée presque uniquement avec le RMSE et le score NASA. Aucune métrique de classification n'avait été calculée, aucun hyperparamètre n'avait été optimisé, une seule famille de features avait été essayée, et un seul modèle avait été entraîné. Le diagnostic biais/variance et la comparaison à une baseline moyenne existaient (ajoutés le 2026-10-21), mais seulement parce que Matisse avait posé la question, et seulement sur le RMSE. Le détail est dans la section 0 de [`docs/protocole_evaluation.md`](../protocole_evaluation.md).

Le cahier des charges (Phase 4) ne prévoyait que le RMSE, le score NASA, un test de significativité et une optimisation des hyperparamètres. Rien n'imposait le reste, ni n'obligeait les futurs modèles (LSTM, CNN 1D) à être évalués de la même façon que le XGBoost.

## Décision

Tout modèle du projet, présent ou futur, est évalué selon [`docs/protocole_evaluation.md`](../protocole_evaluation.md). Ce document fixe le cadre de validation, les baselines, les métriques de régression et de classification, les diagnostics sur-apprentissage/sous-apprentissage (dont les courbes d'apprentissage), l'optimisation des hyperparamètres (grid search, random search, Optuna), la justification des features par ablation, la comparaison statistique entre modèles et le contenu minimal d'un rapport. Une règle 11 d'[`AGENTS.md`](../../AGENTS.md) rend ce protocole non négociable, et demande aux assistants IA de signaler d'eux-mêmes toute étape manquante.

## Alternatives envisagées

- **Compléter seulement le rapport de la Phase 3** : écarté. Les mêmes manques se reproduiraient au premier LSTM, puisque rien ne décrit ce qu'une évaluation complète contient.
- **Ajouter les métriques dans `docs/cahier_des_charges.md`** : écarté. Le cahier des charges décrit le projet par phase, alors que le protocole s'applique à travers toutes les phases et à tous les modèles. Un document à part, cité par `AGENTS.md`, est plus simple à retrouver et à appliquer.
- **Ne garder que les métriques standard de la littérature C-MAPSS (RMSE, score NASA)** : écarté. Ces deux chiffres suffisent pour se comparer aux articles, mais pas pour comprendre le modèle (où il se trompe, s'il sur-apprend, s'il détecte les moteurs proches de la panne). Ils restent dans le protocole.

## Conséquences

**Positives** :
- Chaque résultat sera comparable d'un modèle à l'autre (mêmes plis, mêmes métriques, mêmes baselines), y compris entre le XGBoost et les futurs modèles de deep learning.
- Les choix (métriques gardées, seuil d'alerte, hyperparamètres, features) seront justifiés par des chiffres plutôt qu'affirmés.

**Négatives / coûts acceptés** :
- Évaluer un modèle prend nettement plus de temps de calcul et de rédaction (baselines, plusieurs valeurs de N, recherche d'hyperparamètres, courbes d'apprentissage). Pour un LSTM, chaque entraînement supplémentaire coûte cher, et il faudra peut-être réduire le budget de recherche en le documentant.
- Avec beaucoup de métriques, on risque de ne retenir que celles qui flattent le modèle. Le protocole impose de fixer la règle de décision avant de regarder les résultats, et de justifier chaque métrique écartée. Ce risque est réduit, pas supprimé.
- Les métriques de classification dépendent d'un seuil N (« panne dans les N prochains cycles ») que l'équipe n'a pas encore fixé. En attendant, il faut calculer plusieurs valeurs, ce qui alourdit les rapports.
