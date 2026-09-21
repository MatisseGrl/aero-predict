# Workflow git — Aero Predict

Équipe de 6, dont 3 personnes poussent du code sur ce dépôt. `main` est protégée : personne ne pousse dessus directement, tout passe par une Pull Request (PR). Voir [`CLAUDE.md`](CLAUDE.md) pour le process de travail par phase (explication → implémentation → validation → doc → résumé équipe).

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
2. Remplir le template de PR (rempli automatiquement à l'ouverture).
3. **Au moins une revue d'un autre contributeur code avant merge** (les 3 personnes qui poussent du code se relisent mutuellement — même une relecture rapide vaut mieux qu'aucune : c'est ce qui donne un historique défendable en entretien).
4. Merge en **squash** (une PR = un commit propre sur `main`) sauf si l'historique détaillé de la branche a un intérêt particulier.
5. Supprimer la branche après merge.

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
