# ADR-0006 : Suppression du notebook baseline exploratoire (remplace ADR-0002)

**Statut** : Acceptée
**Date** : 2026-09-23
**Décideurs** : Matisse

## Contexte

[ADR-0002](0002-archivage-baseline-fd001.md) avait décidé d'archiver `notebooks/exploration/baseline_fd001_v0.ipynb` plutôt que de le supprimer, avec pour argument principal qu'il contenait le seul enregistrement du résultat chiffré (RMSE test 18.97, score NASA test 1053.9). Depuis, ce chiffre a été dupliqué et documenté indépendamment dans `README.md` (table "Benchmark de référence") et `docs/00_etat_de_l_art.md` (section 6) — le notebook n'est donc plus la seule source de ce résultat. Le fichier lui-même a été écrit par un contributeur du projet (Gabriel Nestour, commit `bd13f17`) avant la mise en place des conventions actuelles du projet (commentaires systématiques, code explicite plutôt que condensé) et n'est de toute façon jamais réutilisé comme code, seulement comme référence chiffrée (confirmé par ADR-0002 et section 6 de `00_etat_de_l_art.md`).

## Décision

Supprimer `notebooks/exploration/baseline_fd001_v0.ipynb`. Le chiffre de référence (RMSE 18.97 / score NASA 1053.9) reste documenté dans `README.md` et `docs/00_etat_de_l_art.md`, qui deviennent la seule trace de ce benchmark.

## Alternatives envisagées

- **Garder le fichier tel quel (statu quo d'ADR-0002)** — écartée : le fichier n'apporte plus rien que les chiffres déjà dupliqués dans la documentation n'apportent pas déjà, et sa présence dans le repo peut laisser croire à tort qu'il fait partie du pipeline officiel ou qu'il doit être maintenu.
- **Réécrire le notebook pour le mettre aux conventions de style actuelles** (commentaires, code explicite) — écartée : retoucher a posteriori un artefact explicitement gelé mélangerait du travail "process" et "hors process" dans le même historique, exactement la raison déjà invoquée dans ADR-0002 pour repartir de zéro plutôt que de corriger l'existant.

## Conséquences

**Positives** :
- Un fichier de moins dans le repo qui pourrait laisser penser, à tort, qu'il fait partie du pipeline actif ou qu'il doit rester à jour.
- `notebooks/exploration/` redevient vide, cohérent avec le fait qu'aucun travail exploratoire actif n'y est stocké actuellement.

**Négatives / coûts acceptés** :
- Le code source qui a produit le chiffre de référence disparaît : si ce résultat est un jour remis en doute, il ne sera plus possible de le ré-exécuter ou de l'auditer directement dans le repo — seul le résultat texte reste, non vérifiable a posteriori sans aller chercher l'historique git.
- Le fichier reste techniquement récupérable via l'historique git (`git show bd13f17:baseline_fd001.ipynb` ou équivalent après le renommage), mais n'est plus visible ni accessible dans le travail courant.
