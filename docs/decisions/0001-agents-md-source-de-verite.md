# ADR-0001 : AGENTS.md comme source de vérité, CLAUDE.md en pointeur

**Statut** : Acceptée
**Date** : 2026-09-21
**Décideurs** : Matisse

## Contexte

L'équipe compte 6 personnes, dont 3 poussent du code, potentiellement avec des outils IA différents (Claude Code, Codex). Claude Code lit `CLAUDE.md` ; Codex et la plupart des autres outils lisent `AGENTS.md` (standard [agents.md](https://agents.md)). Il fallait une seule source d'instructions projet, pas une par outil, sinon les deux fichiers divergent avec le temps sans que personne ne s'en aperçoive.

## Décision

`AGENTS.md` contient l'intégralité des instructions projet. `CLAUDE.md` est réduit à un pointeur de deux lignes utilisant la syntaxe d'import native de Claude Code (`@AGENTS.md`), documentée officiellement (code.claude.com/docs/en/memory).

## Alternatives envisagées

- **Dupliquer le contenu dans les deux fichiers** — écartée : garantit une divergence à terme (l'un des deux finit par ne plus être mis à jour).
- **Symlink `CLAUDE.md` → `AGENTS.md`** — écartée : nécessite les droits Administrateur ou le mode développeur sur Windows pour être créé, et un clone Windows sans `core.symlinks` récupère un fichier texte inerte à la place du lien ; de plus les outils Edit/Write de Claude Code refusent d'écrire à travers un symlink.
- **Ne garder que `CLAUDE.md`, sans AGENTS.md** — écartée : les outils autres que Claude Code (Codex notamment) ne le liraient jamais.

## Conséquences

**Positives** :
- Une seule source de vérité à maintenir, quel que soit l'outil IA utilisé par la personne.
- Fonctionne aussi sur les sessions où le support natif d'AGENTS.md par Claude Code est indisponible (version antérieure à 2.1.277, Bedrock, télémétrie désactivée) puisque l'import explicite ne dépend pas de ce support natif.

**Négatives / coûts acceptés** :
- Un humain qui ouvre `CLAUDE.md` sur GitHub sans connaître la convention `@import` peut être surpris de ne voir que deux lignes — nécessite la phrase explicative gardée en tête du fichier.
- Dépend d'un mécanisme (`@import`) propre à Claude Code ; si ce mécanisme change de comportement dans une future version, cette ADR devra être révisée.
