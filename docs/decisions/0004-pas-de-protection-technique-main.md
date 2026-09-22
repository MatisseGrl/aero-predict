# ADR-0004 : Pas de protection technique de `main`, discipline d'équipe à la place

**Statut** : Acceptée
**Date** : 2026-09-22
**Décideurs** : Matisse

## Contexte

Avec 3 personnes qui poussent du code, une protection technique de `main` (PR obligatoire, pas de push direct) avait été documentée dans `CONTRIBUTING.md` comme un acquis. En tentant de la configurer réellement sur GitHub (Settings → Rules → Rulesets), le ruleset a bien été créé, mais GitHub a affiché : *"Your rulesets won't be enforced on this private repository until you move to GitHub Team organization account."* — **GitHub Free n'applique pas les règles de protection de branche sur un dépôt privé**, seulement sur les dépôts publics ou sur un compte payant (Pro pour un compte personnel, Team pour une organisation). Le dépôt Aero Predict est privé sur un compte gratuit : le ruleset existe mais n'a aucun effet réel.

## Décision

Rester sur un dépôt **privé**, **sans protection technique de `main`**. Le workflow (branche par tâche, PR, revue avant merge) reste documenté et attendu, mais repose sur la discipline de chaque personne qui code, pas sur un blocage GitHub.

## Alternatives envisagées

- **Passer le dépôt en public** — aurait activé le ruleset gratuitement, et aurait été cohérent avec l'usage final du dépôt (portfolio montré en entretien). Écartée pour l'instant : `CONTRIBUTORS.md` contient les emails réels des 6 membres de l'équipe ; les rendre publics et indexables par les moteurs de recherche demande un nettoyage préalable (retirer les emails ou les remplacer par un renvoi vers un canal interne) et, idéalement, l'accord des personnes concernées — pas fait dans l'immédiat.
- **Passer sur GitHub Pro** (~4$/mois) — aurait gardé le dépôt privé avec une vraie protection. Écartée : coût récurrent pour un projet étudiant, pour un problème qui a une solution gratuite (voir ci-dessus) si le dépôt passe public plus tard.

## Conséquences

**Positives** :
- Aucun coût, aucune information personnelle exposée dans l'immédiat.
- Le ruleset reste configuré sur GitHub : si le dépôt passe public ou sur un compte payant plus tard, il s'active sans reconfiguration.

**Négatives / coûts acceptés** :
- **Aucune barrière technique réelle contre un push direct sur `main`.** Toute la protection repose sur le fait que les 3 personnes qui codent respectent la convention documentée dans `CONTRIBUTING.md` — un push direct, volontaire ou accidentel, passera sans que rien ne l'arrête.
- Le risque identifié plus tôt dans le projet (un push massif poussé sans compréhension réelle du code, notamment évoqué à propos de Benjamin) n'a donc pas de filet de sécurité technique — seule la revue humaine et la règle "exiger la compréhension avant de committer" (AGENTS.md) restent des garde-fous, et ce sont des garde-fous qu'on peut oublier de suivre.
