# ADR-0003 : Abandon des journaux individuels au profit de STATUS.md + ADR

**Statut** : Acceptée
**Date** : 2026-09-22
**Décideurs** : Matisse

## Contexte

Un système de suivi avait été mis en place avec deux volets : `docs/STATUS.md` (état collectif du projet) et `docs/journal/<prenom>.md` (un fichier par personne, décrivant ce qu'elle avait fait/compris/où elle en était), l'idée étant qu'une session Claude Code lise le journal de la personne en face pour recalibrer son niveau d'explication sans tout redemander.

En le repassant au crible, deux défauts structurels sont apparus :

1. **Contradiction avec une règle déjà actée** : le protocole d'identification interdit de déduire le niveau technique de quelqu'un de sa filière, et impose de redemander à chaque nouvelle notion (rien ne périme). Le journal faisait l'inverse : il poussait à traiter un état de compréhension vieux de plusieurs semaines comme actuel — un biais de figement (la personne a pu progresser ailleurs, sans que ça passe par Claude Code), qui de plus reconstruit exactement le type de stéréotype que la règle d'identification voulait éviter, sous une autre forme.
2. **Risque de complaisance** : le journal est rédigé par un assistant dont la consigne explicite est d'être pédagogue et encourageant. Appliqué à un historique projet (censé être un révélateur cru de l'évolution réelle), cette consigne dérive vers des formulations flatteuses plutôt que factuelles, ce qui est en particulier problématique pour un dépôt qui sera public en portfolio et lu dans la durée.

Aucune entreprise consultée dans la veille (ni ce qu'on sait des pratiques réelles en ingénierie logicielle) ne maintient de journal individuel de compréhension par personne dans un dépôt de code : ce qui existe réellement, ce sont des messages de commit/PR clairs, des ADR pour les décisions structurantes, et des notes de suivi managérial qui restent privées et hors du dépôt.

## Décision

Supprimer `docs/journal/` entièrement. Ne garder que `docs/STATUS.md` (état courant du projet, factuel) et introduire `docs/decisions/` (ADR, ce présent mécanisme) pour les décisions structurantes. Le protocole d'identification (redemander le niveau à chaque nouvelle notion) reste le seul mécanisme de calibration par personne.

## Alternatives envisagées

- **Garder les journaux mais les rendre plus factuels** — écartée : ne résout pas le problème de fond (un historique de compréhension reste, par construction, un jugement figé sur une personne, quel que soit le ton).
- **Ne garder qu'un journal par personne mais sans le lire pour calibrer, juste comme trace** — écartée : sans usage clair, c'est un fichier de plus à maintenir pour rien (déjà observé avec les 5 journaux vides créés par anticipation).

## Conséquences

**Positives** :
- Élimine un mécanisme qui contredisait une règle déjà actée sur ce projet.
- Réduit le nombre de fichiers à maintenir à chaque session (moins de discipline requise pour un système qui, de toute façon, décote vite si personne ne l'alimente).
- Le vocabulaire ADR est reconnu en entretien/soutenance ; un journal de compréhension personnel ne l'est pas et aurait pu être lu comme un gadget plutôt qu'un vrai artefact d'ingénierie.

**Négatives / coûts accepté** :
- On perd la reprise instantanée "voilà où en était cette personne précisément" sans avoir à le redemander — chaque session redemande le niveau sur la notion abordée, ce qui coûte quelques échanges en début de conversation.
- Le travail déjà fait pour créer et documenter le système de journaux (plusieurs itérations) est perdu — accepté parce que la correction rapide coûte moins cher que de laisser vivre un mécanisme biaisé sur un dépôt qui sera lu par d'autres.
