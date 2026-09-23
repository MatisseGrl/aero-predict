# État du projet — Aero Predict

Mis à jour après chaque changement significatif. Sert de point d'entrée pour savoir où en est le projet collectivement et ce qui vient de se passer — lu automatiquement en début de session par les assistants IA (voir [`AGENTS.md`](../AGENTS.md), section "Suivi de l'avancement et des décisions"). Pour le détail de qui a poussé quel commit, `git log` / l'historique GitHub font déjà foi ; ce fichier donne le contexte que le git log ne donne pas (pourquoi, quel impact). Les décisions structurantes (pas le travail courant) vivent dans [`docs/decisions/`](decisions/), jamais réécrites une fois actées.

Écriture factuelle uniquement — pas d'opinion non étayée, pas de réécriture silencieuse d'une entrée déjà publiée (voir [`AGENTS.md`](../AGENTS.md), section "Ton d'écriture").

## Phase actuelle

**Phase 1 — EDA et préparation (FD001) : en cours** (branche `phase-01/eda-preparation`, pas encore mergée). Étape 1/3 faite (capteurs à variance nulle). Prochaine étape : suppression effective des capteurs, puis normalisation par capteur.

## Dernière modification

- **Qui** : Matisse (avec Claude Code)
- **Date** : 2026-09-23
- **Quoi** : suppression du notebook `notebooks/exploration/baseline_fd001_v0.ipynb`, qui remplace la décision d'archivage prise dans ADR-0002. Références corrigées dans `README.md`, `AGENTS.md` et `docs/00_etat_de_l_art.md`. Détail : [ADR-0006](decisions/0006-suppression-baseline-exploratoire.md).
- **Pourquoi** : le chiffre de référence qu'il contenait (RMSE 18.97, score NASA 1053.9) est déjà dupliqué dans le README et `00_etat_de_l_art.md` depuis la Phase 0 — le fichier n'était donc plus la seule source de ce résultat, et sa présence pouvait laisser croire à tort qu'il fait partie du pipeline actif.
- **Impact** : aucun sur le pipeline ML/données (le fichier n'a jamais été repris comme code, seulement comme référence chiffrée). Coût accepté : le code source de ce résultat n'est plus auditable/ré-exécutable directement dans le repo (reste récupérable via l'historique git).

### 2026-09-23 — Assouplissement temporaire du workflow git
- **Qui** : Matisse (avec Claude Code)
- **Quoi** : revue obligatoire et PR suspendues tant que Matisse est seul contributeur actif au code, merge/push direct sur `main` autorisé. Détail : [ADR-0005](decisions/0005-assouplissement-workflow-git-periode-solo.md).
- **Pourquoi** : le workflow d'équipe (branche + PR + revue) suppose un second relecteur disponible ; en l'absence des deux autres contributeurs code, exiger une revue n'apportait aucune relecture réelle, juste de la friction.
- **Impact** : aucun sur le pipeline ML/données. Le workflow complet redevient obligatoire dès qu'un autre contributeur recode sur le projet (condition posée dans l'ADR elle-même).

## Historique

<!-- Nouvelle entrée en haut, même format que ci-dessus (Qui / Date / Quoi / Pourquoi / Impact). -->

### 2026-09-22 — Phase 0 validée
- **Qui** : Matisse (avec Claude Code)
- **Quoi** : [`docs/00_etat_de_l_art.md`](00_etat_de_l_art.md) rédigé à partir d'une lecture complète de Saxena et al. (2008) et de la veille communautaire déjà présente dans le cahier des charges. Une équation du papier (indice de santé `h(t)`) s'est révélée corrompue par l'extraction automatique du PDF — vérifié en la recalculant numériquement (elle diverge hors de [0,1]) — donc non reproduite dans le doc, remplacée par une description qualitative.
- **Pourquoi** : Phase 0 = pré-requis avant tout code (cahier des charges, Section 5). La fonction de score NASA, elle, a été vérifiée numériquement et confirmée cohérente (retard pénalisé ~2x plus qu'avance à écart égal) — c'est celle-là qui compte pour la Phase 4.
- **Impact** : aucun sur le code ; Phase 1 a pu démarrer.

### 2026-09-22 — Correction : main n'est pas protégée techniquement
- **Qui** : Matisse (avec Claude Code)
- **Quoi** : correction de `AGENTS.md`/`CONTRIBUTING.md`, qui affirmaient à tort que `main` était protégée techniquement. En tentant de configurer un ruleset GitHub, découverte que GitHub Free n'applique pas ces règles sur un dépôt privé — le ruleset créé est inactif. Décision : rester privé, sans protection technique, discipline d'équipe à la place. Détail : [ADR-0004](decisions/0004-pas-de-protection-technique-main.md).
- **Pourquoi** : ne pas laisser une doc affirmer une protection qui n'existe pas — exactement le genre de fausse confiance qu'on avait identifié comme dangereux avec les journaux individuels (ADR-0003), ici appliqué à une garantie technique plutôt qu'à un historique.
- **Impact** : aucune barrière technique contre un push direct sur `main` — repose entièrement sur le fait que les 3 personnes qui codent suivent la convention documentée dans `CONTRIBUTING.md`.

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
