# Workflow git — Aero Predict

Équipe de 6, dont 3 personnes poussent du code sur ce dépôt. `main` est protégée : personne ne pousse dessus directement, tout passe par une Pull Request (PR). Voir [`AGENTS.md`](AGENTS.md) pour le process de travail par phase (explication → implémentation → validation → doc → résumé équipe).

## Configuration locale (une fois, après avoir cloné le repo)

```bash
pip install -r requirements.txt
pre-commit install
```

La deuxième commande active un hook git local (`nbstripout`) qui nettoie automatiquement les outputs des cellules de notebook avant chaque commit — sans ça, un notebook exécuté produit des diffs énormes et des conflits inutiles à chaque fois que quelqu'un le rouvre. `pre-commit install` s'exécute une fois par personne et par clone : ce n'est pas synchronisé par git (seul `.pre-commit-config.yaml`, qui définit le hook, est versionné).

## Branches

- `main` — toujours stable, toujours fonctionnelle. Protégée : push direct interdit, merge uniquement via PR.
- Une branche par tâche, créée depuis `main` à jour :

```
git checkout main
git pull
git checkout -b phase-0X/description-courte
```

Convention de nom : `phase-0X/description-courte` (ex. `phase-01/normalisation-fd001`, `phase-03/lstm-baseline`). Pour du travail hors phase (config, CI, fix) : `chore/description` ou `fix/description`.

## Commits

Convention [Conventional Commits](https://www.conventionalcommits.org/), en français dans la description :

```
<type>: <description courte>

[corps optionnel : pourquoi, pas quoi]
```

Types utilisés sur ce projet : `feat` (nouvelle fonctionnalité/analyse), `fix` (correction), `docs` (documentation), `data` (changement lié aux données/dataset), `refactor`, `test`.

Exemple : `feat: ajoute la normalisation par capteur pour FD001`

## Pull Requests

1. Ouvrir la PR dès que la branche est prête, en ciblant `main`.
2. Remplir le template de PR (rempli automatiquement à l'ouverture), **y compris le champ de justification** (voir ci-dessous).
3. **Au moins une revue d'un autre contributeur code avant merge** (les 3 personnes qui poussent du code se relisent mutuellement — même une relecture rapide vaut mieux qu'aucune : c'est ce qui donne un historique défendable en entretien).
4. Merge en **squash** (une PR = un commit propre sur `main`) sauf si l'historique détaillé de la branche a un intérêt particulier.
5. Supprimer la branche après merge.
6. Mettre à jour `docs/STATUS.md` dans la même PR (voir "Traçabilité" ci-dessous). Si la PR change une décision déjà actée dans une ADR, ajouter une nouvelle ADR qui la remplace plutôt que d'éditer l'ancienne (voir "Décisions d'architecture" ci-dessous).

## Pas de changement hors-sujet, ni de régression déguisée en "autre idée"

Une PR doit se rattacher explicitement à une phase ou un objectif du [cahier des charges](docs/cahier_des_charges.md) — pas de changement qui n'a rien à voir avec ce sur quoi l'équipe travaille actuellement.

Si ta PR **remplace** une solution déjà validée (par exemple une autre approche de feature engineering, un autre modèle, un autre choix d'architecture), ce n'est pas suffisant qu'elle soit "différente" ou "une autre idée qui te semble bien" : il faut que ce soit un **progrès net et démontré** — un chiffre qui s'améliore (RMSE, score NASA), une simplification réelle, ou un rapprochement du cahier des charges. Sinon la revue doit la refuser et demander la justification manquante. C'est le champ "Justification" du template de PR — ne pas le laisser vide ou vague ("j'ai trouvé ça mieux").

## Traçabilité : qui a fait quoi

Pour le détail technique (quel commit, par qui, quand), `git log` et l'historique des PR sur GitHub font foi — pas besoin de le retracer ailleurs.

Ce que git ne capture pas — le contexte collectif : pourquoi ce changement, quel impact — vit dans [`docs/STATUS.md`](docs/STATUS.md). Une session Claude Code le lit en début de session pour te resituer sans tout réexpliquer, et le met à jour en fin de session si le travail était significatif.

On a testé en plus un journal individuel par personne (ce que chacun avait fait/compris) et on l'a retiré : ça contredisait notre propre règle de ne jamais déduire le niveau de quelqu'un d'un historique écrit, et un historique de compréhension rédigé par un assistant encourageant dérive vers la complaisance avec le temps. Détail dans [`docs/decisions/0003-abandon-journaux-individuels.md`](docs/decisions/0003-abandon-journaux-individuels.md).

## Décisions d'architecture (ADR)

Pour les décisions coûteuses à défaire, ou dont quelqu'un se demandera "pourquoi on a fait ça ?" dans trois mois — pas pour du travail courant. Format volontairement léger, à l'échelle d'une équipe de 6, pas le processus lourd d'une grande entreprise.

**Quand en écrire une** : un choix qui engage plusieurs phases (ex. architecture de modèle retenue, choix de base de données en Phase 6), un choix technique difficile à défaire, ou tout changement qui remplace une décision déjà actée (voir Règle 8 dans [`AGENTS.md`](AGENTS.md)).

**Où** : [`docs/decisions/NNNN-titre-court.md`](docs/decisions/), numérotées dans l'ordre à partir de 0001, en partant du gabarit [`docs/decisions/template.md`](docs/decisions/template.md).

**Règle d'or : une ADR ne se réécrit jamais après coup.** Si une décision change, on écrit une nouvelle ADR qui explique le changement, et on passe le statut de l'ancienne à "Remplacée par ADR-00XX" — jamais une édition silencieuse. C'est ce qui rend l'historique du projet crédible sur la durée : personne ne peut réécrire discrètement le passé pour qu'il ait l'air plus malin qu'il ne l'était, ce qui compte particulièrement ici puisque ce dépôt sera public en portfolio.

**Ton** : factuel, vérifiable, jamais complaisant — voir [`AGENTS.md`](AGENTS.md), section "Ton d'écriture". Toute ADR liste au moins une conséquence négative ou un coût accepté.

## Definition of Done avant d'ouvrir une PR

Reprend la Definition of Done par phase de [`docs/cahier_des_charges.md`](docs/cahier_des_charges.md#9-definition-of-done-par-phase) :

- Le concept a été expliqué avant le code (pas de notion introduite sans contexte).
- Le code est commenté et docstringé.
- La validation (visualisation ou métrique) est concluante.
- Si la PR clôt une phase : `docs/0X_phase.md` est à jour et le résumé Notion est rédigé.

## Configuration de la protection de `main` (à faire une fois, côté admin du repo)

Sur GitHub : **Settings → Branches → Add branch protection rule** sur `main` :
- "Require a pull request before merging" activé.
- "Require approvals" = 1.
- "Do not allow bypassing the above settings" activé (sinon les admins peuvent quand même pousser directement).
