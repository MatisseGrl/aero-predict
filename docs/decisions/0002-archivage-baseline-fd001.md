# ADR-0002 : Archiver le notebook baseline FD001 plutôt que le reprendre

**Statut** : Acceptée
**Date** : 2026-09-21
**Décideurs** : Matisse

## Contexte

Un notebook `baseline_fd001.ipynb` existait déjà dans le dépôt avant la mise en place du process de phases : chargement des données, RUL plafonnée à 125, suppression des capteurs constants, features temporelles glissantes, Random Forest validée par `GroupKFold`. Résultats : RMSE test 18.97, score NASA test 1053.9. Il ne suivait pas la méthode prévue par le cahier des charges (Phase 1) : pas de normalisation par capteur, pas de lissage du signal, EDA et target engineering non séparés, aucune documentation par phase, aucun docstring.

## Décision

Déplacer le notebook vers `notebooks/exploration/baseline_fd001_v0.ipynb` (avec `git mv`, historique conservé) et le traiter comme un **benchmark chiffré de référence**, pas comme du code à reprendre. Le pipeline officiel repart de la Phase 0.

## Alternatives envisagées

- **Garder le notebook en place et le rétro-documenter** — écartée : corriger a posteriori un pipeline qui n'a pas suivi la méthode (ajouter normalisation/lissage dans du code déjà écrit) coûte probablement plus cher que refaire proprement en suivant le process phase par phase, et mélangerait du code "process" et "hors process" dans le même historique.
- **Supprimer le notebook** — écartée : il contient un résultat chiffré valide et un signal utile (une baseline RF sans normalisation atteint déjà RMSE ≈ 19 sur FD001) — le perdre aurait fait perdre un point de comparaison gratuit.

## Conséquences

**Positives** :
- Un chiffre de référence existe dès le départ pour juger si le pipeline officiel (avec normalisation, lissage, documentation) fait réellement mieux — pas seulement différemment.
- Le dossier `notebooks/exploration/` distingue clairement le travail exploratoire du travail qui suit le process, sans avoir à supprimer d'historique.

**Négatives / coûts acceptés** :
- Le travail déjà investi dans ce notebook (chargement, feature engineering, validation croisée) est refait depuis zéro dans le pipeline officiel plutôt que réutilisé tel quel — perte de temps mesurable, acceptée pour la rigueur méthodologique.
