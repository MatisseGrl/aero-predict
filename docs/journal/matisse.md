# Journal — Matisse GARLOT

Historique personnel de contribution au projet. Mis à jour après chaque session de travail significative (code, décision, notion expliquée et validée) — sert à reprendre exactement où on en était, sans tout réexpliquer. Lu automatiquement par Claude Code après le protocole d'identification (voir [`CLAUDE.md`](../../CLAUDE.md)).

## Format d'une entrée

```
## AAAA-MM-JJ
- Contexte : phase / tâche concernée
- Ce que j'ai fait :
- Ce que j'ai compris / validé :
- Où j'en suis :
```

---

## 2026-09-21
- Contexte : mise en place du projet (avant Phase 0).
- Ce que j'ai fait : cloné le dépôt existant, passé en revue le notebook `baseline_fd001.ipynb` déjà présent (baseline Random Forest sur FD001), décidé de l'archiver comme benchmark plutôt que de le reprendre tel quel. Mis en place l'arborescence du projet avec Claude Code : `CLAUDE.md`, `CONTRIBUTING.md`, `CONTRIBUTORS.md`, `docs/cahier_des_charges.md`, `docs/00_brief_original.md`, workflow git d'équipe (branches par phase, PR obligatoire), et ce système de suivi (`STATUS.md` + journaux personnels).
- Ce que j'ai compris / validé : pourquoi le notebook existant manque de normalisation/lissage/documentation par phase ; pourquoi une validation croisée par moteur (`GroupKFold`) est nécessaire plutôt qu'une validation aléatoire (éviter la fuite de données entre cycles d'un même moteur) ; pourquoi séparer vérité technique (GitHub) et coordination (Notion).
- Où j'en suis : Phase 0 (état de l'art) pas encore commencée. Prochaine étape : lire Saxena et al. (2008) et rédiger `docs/00_etat_de_l_art.md`.
