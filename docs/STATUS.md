# État du projet — Aero Predict

Mis à jour après chaque changement significatif. Sert de point d'entrée pour savoir où en est le projet collectivement et ce qui vient de se passer — lu automatiquement en début de session par les assistants IA (voir [`AGENTS.md`](../AGENTS.md), section "Suivi de l'avancement et des décisions"). Pour le détail de qui a poussé quel commit, `git log` / l'historique GitHub font déjà foi ; ce fichier donne le contexte que le git log ne donne pas (pourquoi, quel impact). Les décisions structurantes (pas le travail courant) vivent dans [`docs/decisions/`](decisions/), jamais réécrites une fois actées.

Écriture factuelle uniquement — pas d'opinion non étayée, pas de réécriture silencieuse d'une entrée déjà publiée (voir [`AGENTS.md`](../AGENTS.md), section "Ton d'écriture").

## Phase actuelle

**Phase 3 — Modélisation baseline XGBoost (FD001) : terminée et validée par Matisse.** Deep learning (LSTM/CNN, prévu au cahier des charges section 5) pas encore attaqué. Prochaine étape : Phase 4 (évaluation formalisée : comparaison statistique de modèles, optimisation des hyperparamètres) ou extension deep learning de la Phase 3, à trancher avec Matisse.

## Dernière modification

- **Qui** : Matisse (avec Claude Code)
- **Date** : 2026-09-28
- **Quoi** : ré-exécution complète de `notebooks/03_modelisation.ipynb` dans un environnement différent (venv du projet, Python 3.14, après résolution d'un problème d'installation de xgboost). Les chiffres de validation croisée (`GroupKFold`) ont légèrement changé : RMSE moyen 18.61 (contre 18.17 lors du premier run), malgré `random_state=42` fixé — XGBoost parallélise la construction des arbres, et l'ordre des calculs flottants en parallèle n'est pas garanti identique d'un environnement à l'autre. Le modèle final et son évaluation sur le test officiel sont restés strictement identiques (RMSE 18.91, score NASA 1082.6, 14/100). `docs/03_modelisation.md` mis à jour avec les nouveaux chiffres de validation croisée et une note de reproductibilité expliquant l'écart.
- **Pourquoi** : le notebook a été réexécuté pour vérifier qu'il tournait bien dans l'environnement réellement utilisé (le premier run avait eu lieu dans un environnement différent) — règle du projet : ne jamais committer un notebook dont l'exécution n'a pas été revérifiée.
- **Impact** : slide 4 de la Soutenance 1 (qui cite "RMSE moyen 18,17") doit être mise à jour avec 18,61. Les chiffres du test officiel (18,91 / 1082,6 / 14 sur 100), utilisés partout ailleurs dans le deck, restent inchangés et n'ont pas besoin d'être corrigés.

### 2026-09-27 — Baseline Phase 3 : premier run (XGBoost, score NASA)
- **Qui** : Matisse (avec Claude Code)
- **Quoi** : création de [`src/preprocessing.py`](../src/preprocessing.py) (fonctions réutilisables des Phases 1-2), puis [`notebooks/03_modelisation.ipynb`](../notebooks/03_modelisation.ipynb) : features par fenêtre glissante (moyenne/écart-type sur 5 cycles), score NASA implémenté et vérifié par calcul, XGBoost validé par `GroupKFold` par moteur (RMSE moyen 18.17, score NASA moyen 55 949.6 — voir l'entrée du 2026-09-28 ci-dessus pour la mise à jour de ce chiffre), puis évalué sur le jeu de test officiel `test_FD001.txt`+`RUL_FD001.txt` : **RMSE 18.91, score NASA 1082.6, 14/100 moteurs surestimés de plus de 20 cycles**. Documenté dans [`docs/03_modelisation.md`](03_modelisation.md), figure de validation `docs/figures/03_predictions_vs_vraies_rul_test.png`. Tableau des phases (`README.md`) et "État actuel du projet" (`AGENTS.md`) mis à jour.
- **Pourquoi** : remplacer les chiffres invérifiables utilisés (par erreur) dans le support de Soutenance 1 — voir l'entrée du 2026-09-27 sur la mise en garde baseline ci-dessous — par un résultat obtenu avec du code présent dans le repo aujourd'hui, reproductible par quiconque exécute le notebook.
- **Impact** : aucun sur le pipeline des Phases 1-2 (réutilisé tel quel via `src/preprocessing.py`, vérifié identique par exécution). La colonne "aujourd'hui" de la slide 1 et le chiffre de la slide 4 de la Soutenance 1 peuvent maintenant être corrigés avec ces valeurs sourcées.

### 2026-09-27 — Rédaction du rapport de Phase 2
- **Qui** : Matisse (avec Claude Code)
- **Quoi** : rédaction de [`docs/02_target_engineering.md`](02_target_engineering.md) (rapport de Phase 2), mise à jour du tableau des phases dans `README.md` et de l'"État actuel du projet" dans `AGENTS.md` (Phase 2 : ✅ Terminée).
- **Pourquoi** : dernière étape du déroulé obligatoire de phase (`AGENTS.md`) — documentation technique écrite seulement après validation explicite de la phase, jamais avant.
- **Impact** : aucun sur le code ; Phase 3 a pu démarrer ensuite.

### 2026-09-27 — Phase 2 validée : calcul et plafonnement de la RUL
- **Qui** : Matisse (avec Claude Code)
- **Quoi** : [`notebooks/02_target_engineering.ipynb`](../notebooks/02_target_engineering.ipynb) — calcul de la vraie RUL par cycle sur `train_FD001.txt` (`dernier_cycle_du_moteur − cycle_actuel`, vérifié sur le moteur 1 : RUL = 0 exactement au dernier cycle), puis plafonnement à 125 cycles (Piecewise Linear RUL). Argument du plafond vérifié empiriquement plutôt qu'affirmé : sur 3 moteurs (1, 50, 100), `sensor_11` est nettement plus stable en début de vie (RUL > 125, écart-type ≈ 0,09–0,12) qu'en fin de vie (RUL ≤ 125, écart-type ≈ 0,22–0,26, soit 2 à 2,7× plus dispersé), avec une dérive de moyenne de +0,29 à +0,33. Validation visuelle : RUL brute vs plafonnée sur le moteur 1 (`docs/figures/02_rul_brute_vs_plafonnee_moteur1.png`) et signal des 3 moteurs (`docs/figures/02_signal_3_moteurs_sensor11.png`).
- **Pourquoi** : cahier des charges, section Phase 2 — capping à 125 cycles avec validation attendue (visu RUL brute vs cappée). L'argument "signal plat en début de vie" est vérifié sur des données réelles plutôt que repris tel quel de la littérature communautaire.
- **Impact** : aucun sur `main` à l'époque (travail sur branche `phase-02/target-engineering`, mergée depuis). La slide de soutenance "Zoom 1, notre plus-value" sera reconstruite à partir de ce résultat réel, à la place de l'ancien benchmark exploratoire non reproductible.

### 2026-09-27 — Mise en garde sur le benchmark baseline (README, état de l'art)
- **Qui** : Matisse (avec Claude Code)
- **Quoi** : ajout d'une mise en garde explicite dans le `README.md` (section "Benchmark de référence") et `docs/00_etat_de_l_art.md` : le notebook baseline exploratoire (supprimé, ADR-0006) ne doit plus jamais servir à dériver une nouvelle statistique ou justifier une décision — seuls les 6 chiffres déjà publiés (RMSE/score NASA) restent utilisables tels quels.
- **Pourquoi** : en préparant la Soutenance 1, un chiffre dérivé de ce benchmark ("16/100 moteurs surestimés de plus de 20 vols") a été mis sur une slide sans code reproductible ni source vérifiable — impossible à défendre devant un jury. Plutôt que de reconstruire ce vieux notebook pour vérifier un chiffre ponctuel, décision de repartir sur un vrai résultat de Phase 2 (voir entrée ci-dessus) pour la slide concernée.
- **Impact** : aucun sur le pipeline ML/données.

### 2026-09-23 — Utiliser scikit-learn pour la normalisation (Phase 1)
- **Qui** : Matisse (avec Codex ; push et documentation par Claude Code après un crash de Codex avant le push)
- **Date** : 2026-09-23
- **Quoi** : dans [`notebooks/01_eda_preparation.ipynb`](../notebooks/01_eda_preparation.ipynb), l'étape de normalisation (min-max et standardisation) repasse d'une boucle manuelle (calcul direct de min/max ou moyenne/écart-type) à `MinMaxScaler`/`StandardScaler` de scikit-learn. Résultats numériques identiques (mêmes formules) — vérifié par ré-exécution complète du notebook à froid (0 erreur, min=0/max=1 confirmés sur les 15 capteurs). Documentation technique mise à jour en conséquence : [`docs/01_eda_preparation.md`](01_eda_preparation.md).
- **Pourquoi** : l'objet `scaler` retourné par `fit` conserve les paramètres appris sur `train` — nécessaire pour appliquer plus tard la même transformation à `test_FD001.txt` via `.transform()` sans réapprendre de paramètres sur le test (fuite de données sinon). `scikit-learn` était déjà une dépendance déclarée du projet (`requirements.txt`, prévue pour la Phase 3).
- **Impact** : aucun sur la décision déjà actée (min-max retenu) ni sur les valeurs produites ; changement d'implémentation uniquement.

### 2026-09-23 — Phase 1 validée : rapport et figures
- **Qui** : Matisse (avec Claude Code)
- **Quoi** : Phase 1 validée — [`docs/01_eda_preparation.md`](01_eda_preparation.md) rédigé, avec les deux figures de vérification (`docs/figures/01_normalisation_comparaison_minmax_standard.png`, `docs/figures/01_lissage_moteur1_sensor11.png`) sauvegardées séparément puisque `nbstripout` nettoie les outputs du notebook à chaque commit. Tableau des phases du README et `AGENTS.md` mis à jour (Phase 1 : ✅ Terminée).
- **Pourquoi** : dernière étape du déroulé obligatoire de phase (`AGENTS.md`) — documentation technique écrite seulement après validation explicite de la phase, jamais avant.
- **Impact** : aucun sur le code ; Phase 2 peut démarrer.

### 2026-09-23 — Fin des étapes techniques de la Phase 1 (normalisation, lissage)
- **Qui** : Matisse (avec Claude Code)
- **Quoi** : dans [`notebooks/01_eda_preparation.ipynb`](../notebooks/01_eda_preparation.ipynb), fin des étapes 2/3 et 3/3 de la Phase 1. Normalisation : comparaison empirique min-max vs `StandardScaler` sur les 15 capteurs (histogrammes de `sensor_9`/`sensor_11`) — les deux sont des transformations linéaires, aucune ne change la forme d'une distribution ; `sensor_9`/`14`/`8` montrent des valeurs extrêmes réelles (z-score jusqu'à +8.12 en standardisé). Min-max retenu : coût nul pour les modèles à arbres de la Phase 3, entrées bornées adaptées au LSTM prévu ensuite, hypothèse gaussienne de la standardisation de toute façon mise à mal par nos données. Lissage : moyenne mobile (fenêtre 5 cycles), appliquée moteur par moteur (masque booléen sur `unit_number`) pour ne jamais mélanger deux moteurs, causale par construction (`pandas.rolling()` ne regarde que le passé) pour rester valide sur le jeu de test.
- **Pourquoi** : cahier des charges, section Phase 1 (normalisation par capteur, lissage du signal) — méthodes choisies et justifiées plutôt qu'appliquées par défaut, avec vérification empirique sur les vraies données avant de trancher entre min-max et standardisation.
- **Impact** : aucun sur `main` (travail sur branche de phase à l'époque). Coût accepté : le choix min-max n'est pas prouvé optimal, seulement argumenté — à revalider en Phase 4 avec un vrai chiffre (RMSE / score NASA) si le LSTM performe mal sur les capteurs asymétriques identifiés.

## Historique

<!-- Nouvelle entrée en haut, même format que ci-dessus (Qui / Date / Quoi / Pourquoi / Impact). -->

### 2026-09-23 — Suppression du notebook baseline exploratoire
- **Qui** : Matisse (avec Claude Code)
- **Quoi** : suppression du notebook `notebooks/exploration/baseline_fd001_v0.ipynb`, qui remplace la décision d'archivage prise dans ADR-0002. Références corrigées dans `README.md`, `AGENTS.md` et `docs/00_etat_de_l_art.md`. Détail : [ADR-0006](decisions/0006-suppression-baseline-exploratoire.md).
- **Pourquoi** : le chiffre de référence qu'il contenait (RMSE 18.97, score NASA 1053.9) est déjà dupliqué dans le README et `00_etat_de_l_art.md` depuis la Phase 0 — le fichier n'était donc plus la seule source de ce résultat, et sa présence pouvait laisser croire à tort qu'il fait partie du pipeline actif.
- **Impact** : aucun sur le pipeline ML/données (le fichier n'a jamais été repris comme code, seulement comme référence chiffrée). Coût accepté : le code source de ce résultat n'est plus auditable/ré-exécutable directement dans le repo (reste récupérable via l'historique git).

### 2026-09-23 — Assouplissement temporaire du workflow git
- **Qui** : Matisse (avec Claude Code)
- **Quoi** : revue obligatoire et PR suspendues tant que Matisse est seul contributeur actif au code, merge/push direct sur `main` autorisé. Détail : [ADR-0005](decisions/0005-assouplissement-workflow-git-periode-solo.md).
- **Pourquoi** : le workflow d'équipe (branche + PR + revue) suppose un second relecteur disponible ; en l'absence des deux autres contributeurs code, exiger une revue n'apportait aucune relecture réelle, juste de la friction.
- **Impact** : aucun sur le pipeline ML/données. Le workflow complet redevient obligatoire dès qu'un autre contributeur recode sur le projet (condition posée dans l'ADR elle-même).

### 2026-09-22 — Démarrage Phase 1 : détection des capteurs constants
- **Qui** : Matisse (avec Claude Code)
- **Quoi** : [`notebooks/01_eda_preparation.ipynb`](../notebooks/01_eda_preparation.ipynb) (branche `phase-01/eda-preparation`, pas encore mergée) : chargement de `train_FD001.txt` et détection des capteurs à variance nulle. Un premier critère naïf (`variance == 0`) s'est révélé insuffisant — vérifié en croisant avec `nunique()` : `sensor_5` et `sensor_16` sont réellement constants mais leur variance calculée tombe à ≈1e-29/1e-35 (erreur d'arrondi flottant, pas 0 exact) — remplacé par le critère `nunique() == 1`, fiable. Résultat : 6 capteurs constants (`sensor_1, 5, 10, 16, 18, 19`), 15 capteurs informatifs conservés sur 21.
- **Pourquoi** : première étape de la Phase 1 (cahier des charges, section Phase 1) — retirer les capteurs non informatifs avant normalisation et lissage.
- **Impact** : aucun sur le pipeline de modélisation (pas encore commencé) ; travail encore sur une branche de phase, pas sur `main`.

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
