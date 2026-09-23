# ADR-0005 : Assouplissement temporaire du workflow git pendant la période solo

**Statut** : Acceptée
**Date** : 2026-09-23
**Décideurs** : Matisse

## Contexte

Le workflow git d'équipe (`CONTRIBUTING.md`) impose une branche par tâche, une Pull Request, et au moins une revue par un autre contributeur avant tout merge sur `main`. Ce process est pensé pour une équipe de 6 personnes dont 3 poussent du code. Dans les faits, à ce stade du projet, Matisse est le seul contributeur actif sur le code — les deux autres personnes censées coder n'ont pas encore repris le travail. Exiger une revue dans ce contexte n'apporte aucune relecture réelle (personne d'autre n'est disponible pour la faire) et bloque ou ralentit le travail solo sans bénéfice de qualité correspondant.

## Décision

Le temps que Matisse reste seul contributeur actif au code, la revue obligatoire par un tiers avant merge sur `main` est suspendue, et un merge/push direct sur `main` est autorisé sans PR. Les branches par tâche (`phase-0X/...`, `chore/...`) restent utilisées pour garder un historique de commits lisible par sujet, mais ne bloquent plus sur une revue externe. **Dès qu'un autre contributeur reprend du code sur le projet, le workflow complet de `CONTRIBUTING.md` (branche + PR + revue obligatoire) est réactivé automatiquement** — cette ADR fait elle-même office de règle de retour, sans qu'une nouvelle ADR soit nécessaire pour repasser au workflow complet.

## Alternatives envisagées

- **Garder la revue obligatoire malgré l'absence de second relecteur** — écartée : n'apporte aucune revue réelle, juste de la friction ; un merge resterait bloqué indéfiniment en attendant un reviewer qui ne viendra pas.
- **Retirer complètement branches et PR même en solo, tout pousser directement sur `main` sans distinction par tâche** — écartée : les branches par tâche gardent un historique de commits lisible (un sujet = une branche), utile même seul, et permettent de reprendre le workflow complet instantanément dès qu'un contributeur revient, sans avoir à réapprendre une nouvelle convention à ce moment-là.

## Conséquences

**Positives** :
- Workflow adapté à la réalité actuelle (1 seul contributeur actif) plutôt qu'un process pensé pour 3 personnes.
- Moins de friction et d'attente pour rien pendant la période solo.

**Négatives / coûts acceptés** :
- Aucune relecture externe du code produit pendant cette période — des bugs ou choix discutables risquent de rester non détectés jusqu'à ce qu'un tiers relise rétroactivement, plus tard.
- L'historique git montrera une période de merges directs sur `main` sans revue associée — à comprendre (par un jury ou un recruteur qui lirait l'historique) comme un choix de contexte documenté ici, pas un oubli de process.
