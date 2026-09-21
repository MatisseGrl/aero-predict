# État du projet — Aero Predict

Mis à jour après chaque changement significatif. Sert de point d'entrée pour savoir où en est le projet collectivement et ce qui vient de se passer — lu automatiquement en début de session par les assistants IA (voir [`AGENTS.md`](../AGENTS.md), section "Suivi de l'avancement"). Pour le détail de qui a poussé quel commit, `git log` / l'historique GitHub font déjà foi ; ce fichier donne le contexte que le git log ne donne pas (pourquoi, quel impact).

## Phase actuelle

**Phase 0 — État de l'art.** Pas encore commencée.

## Dernière modification

- **Qui** : Matisse (avec Claude Code)
- **Date** : 2026-09-21
- **Quoi** : mise en place du système de suivi d'avancement — ce fichier et les journaux personnels dans `docs/journal/`.
- **Pourquoi** : pour que n'importe quel membre de l'équipe reprenne une session Claude Code sur ce projet sans perdre le fil (ni le sien, ni celui du projet), et pour éviter les pushs hors-sujet en gardant une trace explicite de pourquoi chaque changement a été fait.
- **Impact** : aucun sur le pipeline ML/données ; c'est un outil de process.

## Historique

<!-- Nouvelle entrée en haut, même format que ci-dessus (Qui / Date / Quoi / Pourquoi / Impact). -->

### 2026-09-21 — Fondations du projet
- **Qui** : Matisse (avec Claude Code)
- **Quoi** : arborescence initiale (`CLAUDE.md`, `CONTRIBUTING.md`, `CONTRIBUTORS.md`, `docs/cahier_des_charges.md`, `docs/00_brief_original.md`), workflow git d'équipe (branches par phase, PR obligatoire, revue), archivage de `baseline_fd001.ipynb` en `notebooks/exploration/` comme benchmark de référence (RMSE test 18.97, score NASA test 1053.9).
- **Pourquoi** : poser un process d'ingénierie rigoureux avant de commencer la Phase 0, avec une équipe de 6 dont 3 poussent du code.
- **Impact** : aucun changement sur le pipeline ML ; fondation du process de travail.
