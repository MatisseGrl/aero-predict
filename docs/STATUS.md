# État du projet — Aero Predict

Mis à jour après chaque changement significatif. Sert de point d'entrée pour savoir où en est le projet collectivement et ce qui vient de se passer — lu automatiquement en début de session par les assistants IA (voir [`AGENTS.md`](../AGENTS.md), section "Suivi de l'avancement et des décisions"). Pour le détail de qui a poussé quel commit, `git log` / l'historique GitHub font déjà foi ; ce fichier donne le contexte que le git log ne donne pas (pourquoi, quel impact). Les décisions structurantes (pas le travail courant) vivent dans [`docs/decisions/`](decisions/), jamais réécrites une fois actées.

Écriture factuelle uniquement — pas d'opinion non étayée, pas de réécriture silencieuse d'une entrée déjà publiée (voir [`AGENTS.md`](../AGENTS.md), section "Ton d'écriture").

## Phase actuelle

**Phase 0 — État de l'art.** Pas encore commencée.

## Dernière modification

- **Qui** : Matisse (avec Claude Code)
- **Date** : 2026-09-22
- **Quoi** : correction de `AGENTS.md`/`CONTRIBUTING.md`, qui affirmaient à tort que `main` était protégée techniquement. En tentant de configurer un ruleset GitHub, découverte que GitHub Free n'applique pas ces règles sur un dépôt privé — le ruleset créé est inactif. Décision : rester privé, sans protection technique, discipline d'équipe à la place. Détail : [ADR-0004](decisions/0004-pas-de-protection-technique-main.md).
- **Pourquoi** : ne pas laisser une doc affirmer une protection qui n'existe pas — exactement le genre de fausse confiance qu'on avait identifié comme dangereux avec les journaux individuels (ADR-0003), ici appliqué à une garantie technique plutôt qu'à un historique.
- **Impact** : aucune barrière technique contre un push direct sur `main` — repose entièrement sur le fait que les 3 personnes qui codent suivent la convention documentée dans `CONTRIBUTING.md`.

## Historique

<!-- Nouvelle entrée en haut, même format que ci-dessus (Qui / Date / Quoi / Pourquoi / Impact). -->

### 2026-09-22 — Hook pre-commit (nbstripout)
- **Qui** : Matisse (avec Claude Code)
- **Quoi** : ajout d'un hook git local (`pre-commit` + `nbstripout`, voir `.pre-commit-config.yaml`) qui nettoie automatiquement les outputs des cellules de notebook avant chaque commit. Testé en local (notebook avec output → outputs vidés au commit, confirmé).
- **Pourquoi** : sans ça, un notebook Jupyter exécuté par 3 personnes différentes produit des diffs énormes et des conflits git sans rapport avec le vrai changement de code — dette technique identifiée avant même la Phase 0.
- **Impact** : chaque personne doit lancer `pre-commit install` une fois après avoir cloné/pull (documenté dans `CONTRIBUTING.md`/`README.md`) ; aucun changement sur le pipeline ML/données.

### 2026-09-22 — Abandon des journaux individuels
- **Qui** : Matisse (avec Claude Code)
- **Quoi** : suppression de `docs/journal/` (5 fichiers vides + celui de Matisse) et introduction de `docs/decisions/` (ADR). Détail et raisons complètes : [ADR-0003](decisions/0003-abandon-journaux-individuels.md).
- **Pourquoi** : les journaux individuels contredisaient une règle déjà actée (ne jamais déduire le niveau de quelqu'un d'un historique écrit) et risquaient de dériver vers la complaisance, rédigés par un assistant dont la consigne est d'être encourageant — problématique pour un dépôt public en portfolio.
- **Impact** : aucun sur le pipeline ML/données ; réduction du nombre de fichiers de process à maintenir.

### 2026-09-21 — Mise en place du suivi d'avancement (journaux individuels, depuis retirés — voir ADR-0003)
- **Qui** : Matisse (avec Claude Code)
- **Quoi** : ce fichier et un journal personnel par membre de l'équipe dans `docs/journal/`.
- **Pourquoi** : pour que n'importe quel membre de l'équipe reprenne une session Claude Code sans perdre le fil, et pour garder une trace de pourquoi chaque changement a été fait.
- **Impact** : aucun sur le pipeline ML/données. Le volet "journaux individuels" a été retiré le lendemain — voir l'entrée du 2026-09-22 ci-dessus.

### 2026-09-21 — Fondations du projet
- **Qui** : Matisse (avec Claude Code)
- **Quoi** : arborescence initiale (`CLAUDE.md`, `CONTRIBUTING.md`, `CONTRIBUTORS.md`, `docs/cahier_des_charges.md`, `docs/00_brief_original.md`), workflow git d'équipe (branches par phase, PR obligatoire, revue), archivage de `baseline_fd001.ipynb` en `notebooks/exploration/` comme benchmark de référence (RMSE test 18.97, score NASA test 1053.9).
- **Pourquoi** : poser un process d'ingénierie rigoureux avant de commencer la Phase 0, avec une équipe de 6 dont 3 poussent du code.
- **Impact** : aucun changement sur le pipeline ML ; fondation du process de travail.
