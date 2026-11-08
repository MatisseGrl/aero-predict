# Comparaison v1 / v2 des notebooks

> **Statut : en attente de relecture par Matisse.** Ce fichier est le livrable du plan de réorganisation, section 9.4, point 3 (fin de l'étape D). L'étape E (nettoyage) **n'est pas commencée** : aucun des 14 anciens notebooks n'est retiré, et rien ne doit l'être avant que Matisse ait relu ce fichier et validé la v2 (plan 9.4, point 4).
>
> **Mise à jour du 2026-11-08** : Matisse a donné son accord pour le retrait des anciens notebooks (étape E faite, [ADR-0008](decisions/0008-organisation-des-notebooks.md)). Le statut ci-dessus est celui du 2026-11-07, laissé tel quel. Les points listés plus bas comme « à faire valider » restent ouverts sauf mention contraire dans `docs/STATUS.md`.
>
> Rédigé le 2026-11-07 par une session Claude Code (réorganisation), dans `aero-predict-refonte`, branche `refonte/notebooks`. Au moment de l'écriture, la v2 n'est **pas committée** : les numéros de cellule cités ci-dessous sont ceux des fichiers de ce dossier le 2026-11-07, et ils changeront si un notebook est modifié.

> **Mise à jour du 2026-11-08** : FD003 et FD004 ont maintenant des réseaux en v2 (calcul de nuit du 2026-11-07 au 08). Ce qui change est décrit en section 6, ajoutée ce jour. Dans les sections 0 à 5, le texte du 2026-11-07 est gardé ; chaque endroit qui ne tient plus est suivi d'une note « *Mise à jour du 2026-11-08* », et les lignes de tableau modifiées le disent dans la cellule.

Plan : [`docs/plan_refonte_notebooks.md`](plan_refonte_notebooks.md). Les sections de ce fichier suivent la demande du plan 9.4 : (1) les chiffres, (2) les cellules, (3) les figures, puis (4) les limites et (5) les questions ouvertes.

---

## 0. Définitions et sources

### 0.1 Ce que désignent « v1 » et « v2 »

- **v1** : le tag local `v1-avant-reorganisation`. Sur cette machine, il pointe sur le commit `efe9434` (« feat(demo): la demo utilise les reseaux du 04h (FD001 et FD002) ») ; le journal du plan (ligne du 2026-11-07) cite `2905e54`, hachage qui n'existe pas dans ce dépôt local (probablement celui de la machine de Matisse pour le même commit, non vérifié). Entre ce tag et `HEAD` (`84d2620`), seul `docs/plan_refonte_notebooks.md` diffère : les 14 anciens notebooks de `notebooks/` sont donc identiques à ceux du tag. Ces 14 notebooks : `01_eda_preparation`, `01b_sous_jeux_fd002_fd004`, `02_target_engineering`, `03_modelisation`, `03b_deep_learning_lstm_cnn`, `04a` à `04h`, `05_comparaison_evaluation` (519 cellules en tout).
- **v2** : `notebooks/01_donnees.ipynb` (61 cellules), `notebooks/02_choix_du_modele.ipynb` (232), `notebooks/evolution.ipynb` (115), `notebooks/resultats/_modele.ipynb` (69), et les 4 notebooks générés `notebooks/resultats/FD001.ipynb` à `FD004.ipynb` (69 cellules chacun, jamais modifiés à la main). Les 4 notebooks écrits à la main comptent 477 cellules.

### 0.2 D'où vient chaque chiffre v1

Les notebooks du tag n'ont **aucune sortie enregistrée** (le hook `nbstripout` les efface au commit), et les fichiers `sorties/` de la v1 (prédictions et résultats écrits sur la machine de Matisse) **ne sont pas sur cette machine**. Chaque chiffre v1 vient donc de l'une de ces sources, citée dans les tableaux par son code :

| Code | Source | Ce qu'elle contient |
|---|---|---|
| **S1** | `docs/STATUS.md` au tag (entrées du 2026-11-04, 2026-11-05 et 2026-11-07) | Chiffres de test publiés : RMSE, score NASA, parfois les surestimés |
| **S2** | `src/config_modele.py` au tag (`CONFIG_*_FINAL["metriques_test"]`), identique à `ATTENDU_FD001_FINAL` de `scripts/generer_predictions_arbres.py` | Test FD001 des 3 arbres finaux : RMSE, score NASA, surestimés. Pas de MAE ni d'alertes manquées |
| **S3** | `docs/resultats/04h/04h_test.json` et `04h_fd002_test.json`, versionnés au tag | Test des réseaux du 04h (FD001, FD002) : toutes les métriques de ce fichier |
| **S4** | `../aero-predict/sorties/predictions/*.json` (lus, non modifiés), écrits le 2026-11-04 et le 2026-11-05 sur la machine de Benjamin (branche `phase-04/demo-ben`, commit `001af8c`, **pas le tag**) | Test des versions retenues du notebook 05 (arbres à réglages de base avec `ecartinit_20`, réseaux du 03b), FD001 à FD004, toutes les métriques. Recoupés : chaque RMSE et score NASA cité dans S1 (entrée du 2026-11-05 de Benjamin) et dans le markdown du 05 (F.8) y est identique |

Conséquence : pour les arbres finaux, la MAE et les alertes manquées de FD001, et **tous** les chiffres de FD002 à FD004, n'ont jamais été écrits dans un fichier versionné en v1 (S1 dit le 2026-11-05 : « FD002–FD004 régénérés avec les réglages finaux, mais pas encore relus »). Ils sont notés « — » ci-dessous, et leur écart n'est pas calculable (« n. c. »).

### 0.3 D'où vient chaque chiffre v2

- `sorties/predictions/*.json` (`metriques_test`), écrits le 2026-11-07 sur cette machine par `scripts/generer_predictions_arbres.py` (arbres, réglages `CONFIG_*_FINAL`, 31 variables sur FD001) et `scripts/publier_reseaux_04h.py` (réseaux du 04h relus dans `models/04h/`, aucun entraînement).
- Contrôle fait pour ce fichier : les 5 métriques des 20 fichiers sont recalculées à partir des `.csv.gz` (dernier cycle de chaque moteur test) et sont identiques aux JSON.
- Les mêmes chiffres sont affichés par les notebooks générés, section 4 (`notebooks/resultats/FD00X.ipynb`, cellule 26, contrôle « 42 métriques identiques au recalcul » sur FD001 et FD002, 28 sur FD003 et FD004). *Mise à jour du 2026-11-08 : 42 sur les 4 sous-jeux (6 fichiers de prédictions chacun), après la publication des réseaux de FD003 et FD004 par `scripts/publier_reseaux_04h.py`.*

### 0.4 Définition des métriques (identique en v1 et en v2)

Une prédiction par moteur test, au **dernier cycle observé**, ramenée à 0 si elle est négative, comparée à la RUL **brute** de `RUL_FD00X.txt`. Score NASA : somme sur les moteurs. Surestimés : prédiction − RUL vraie > 20 cycles. Alertes manquées : RUL vraie ≤ 30 et prédiction > 30 (seuil fixe de 30 cycles, celui de la démo ; ce n'est pas le seuil choisi en validation de la section 5 des notebooks de résultats).

---

## 1. Les chiffres

### 1.1 Mêmes modèles en v1 et en v2 (ceux de la démo au tag)

Au tag, la démo utilise les arbres finaux (`CONFIG_*_FINAL`) sur les 4 sous-jeux et les réseaux du 04h sur FD001 et FD002 (S1, entrée du 2026-11-07). La v2 garde exactement ces modèles (journal du plan, 2026-11-07 : « uniquement les nouveaux modèles »). Format : v1 → v2 (écart v2 − v1, à la précision publiée en v1).

**FD001** (100 moteurs test)

| Modèle | RMSE | MAE | Score NASA | Surestimés > 20 | Alertes manquées | Source v1 |
|---|---|---|---|---|---|---|
| Référence naïve | 43,07 → 43,07 (0) | 35,90 → 35,90 (0) | 33 629,2 → 33 629,2 (0) | 41 → 41 (0) | 25 → 25 (0) | S1 (RMSE), S4 |
| Random Forest | 13,14 → 13,14 (0) | — → 9,45 (n. c.) | 243,3 → 243,3 (0) | 3 → 3 (0) | — → 4 (n. c.) | S1, S2 |
| XGBoost | **12,73 → 12,78 (+0,05)** | — → 9,37 (n. c.) | **223,5 → 232,5 (+9,0)** | 2 → 2 (0) | — → 5 (n. c.) | S1, S2 |
| LightGBM | 12,76 → 12,76 (0) | — → 9,29 (n. c.) | 234,6 → 234,6 (0) | 3 → 3 (0) | — → 4 (n. c.) | S1, S2 |
| LSTM (ensemble de 5, 04h) | 11,16 → 11,16 (0) | 7,81 → 7,81 (0) | 227,2 → 227,2 (0) | 4 → 4 (0) | 0 → 0 (0) | S3 |
| CNN 1D (graine 42, 04h) | 11,68 → 11,68 (0) | 8,56 → 8,56 (0) | 189,3 → 189,3 (0) | 4 → 4 (0) | 0 → 0 (0) | S3 |

**FD002** (259 moteurs test)

| Modèle | RMSE | MAE | Score NASA | Surestimés > 20 | Alertes manquées | Source v1 |
|---|---|---|---|---|---|---|
| Référence naïve | 54,08 → 54,08 (0) | 45,64 → 45,64 (0) | 153 229,3 → 153 229,3 (0) | 106 → 106 (0) | 61 → 61 (0) | S4 |
| Random Forest | — → 23,90 (n. c.) | — → 15,65 (n. c.) | — → 4 326,3 (n. c.) | — → 13 (n. c.) | — → 0 (n. c.) | aucune |
| XGBoost | — → 23,60 (n. c.) | — → 15,12 (n. c.) | — → 4 394,7 (n. c.) | — → 15 (n. c.) | — → 1 (n. c.) | aucune |
| LightGBM | — → 23,73 (n. c.) | — → 15,27 (n. c.) | — → 4 600,3 (n. c.) | — → 15 (n. c.) | — → 0 (n. c.) | aucune |
| LSTM (ensemble de 5, 04h) | 22,38 → 22,38 (0) | 14,13 → 14,13 (0) | 3 432,5 → 3 432,5 (0) | 17 → 17 (0) | 0 → 0 (0) | S3 |
| CNN 1D (graine 42, réglages du 03b) | 23,89 → 23,89 (0) | 15,39 → 15,39 (0) | 4 436,5 → 4 436,5 (0) | 12 → 12 (0) | 0 → 0 (0) | S3 |

**FD003** (100 moteurs test) et **FD004** (248 moteurs test) : aucun réseau en v1 (S1, 2026-11-07 : « FD003 et FD004 n'ont toujours pas de réseaux dans la démo ») ni en v2 (journal du plan, 2026-11-07). *Mise à jour du 2026-11-08 : la v2 a maintenant un LSTM (ensemble de 5) et un CNN 1D sur FD003 et FD004. Ils n'existaient pas dans la démo au tag : ce ne sont pas des « mêmes modèles », ils n'entrent donc pas dans ce tableau ni dans le compte de non-régression de la section 4 ; leurs chiffres sont en 1.2 et en section 6.*

| Sous-jeu | Modèle | RMSE | MAE | Score NASA | Surestimés > 20 | Alertes manquées | Source v1 |
|---|---|---|---|---|---|---|---|
| FD003 | Référence naïve | 45,07 → 45,07 (0) | 38,00 → 38,00 (0) | 57 949,6 → 57 949,6 (0) | 49 → 49 (0) | 20 → 20 (0) | S4 |
| FD003 | Random Forest | — → 12,34 | — → 8,97 | — → 256,1 | — → 6 | — → 2 | aucune |
| FD003 | XGBoost | — → 11,78 | — → 8,63 | — → 246,0 | — → 6 | — → 3 | aucune |
| FD003 | LightGBM | — → 12,14 | — → 9,06 | — → 256,5 | — → 4 | — → 3 | aucune |
| FD004 | Référence naïve | 54,90 → 54,90 (0) | 46,75 → 46,75 (0) | 189 682,8 → 189 682,8 (0) | 101 → 101 (0) | 53 → 53 (0) | S4 |
| FD004 | Random Forest | — → 25,18 | — → 17,42 | — → 3 856,9 | — → 18 | — → 6 | aucune |
| FD004 | XGBoost | — → 24,67 | — → 16,79 | — → 3 715,2 | — → 21 | — → 3 | aucune |
| FD004 | LightGBM | — → 24,82 | — → 17,05 | — → 3 763,0 | — → 18 | — → 6 | aucune |

**Explication de chaque écart non nul**

- **XGBoost FD001, RMSE +0,05 et score NASA +9,0 (surestimés inchangés, 2).** Même code, mêmes réglages (`CONFIG_XGBOOST_FINAL`), même 31 variables. Avec `subsample` < 1, XGBoost ne construit pas exactement les mêmes arbres d'un environnement à l'autre. S1 (entrée du 2026-11-05) le mesurait déjà sur la machine de Matisse (16 cœurs) : 12,70 avec tous les cœurs, 12,73 avec 1 cœur, **12,78 avec 4 cœurs**, et en concluait que « sur 1 cœur, le résultat ne dépend plus de la machine ». Sur cette machine (Apple M5 Pro, arm64, xgboost 3.4.1), le résultat vaut 12,78 avec 1, 2, 4 et 8 cœurs (rapport de l'étape A1 de la réorganisation, et `evolution.ipynb`, section 8.1) : la conclusion de S1 ne tient pas ici. Cause probable (calcul en virgule flottante propre à la plateforme, combiné à `subsample` < 1), **non démontrée**. La version d'xgboost de la machine de Matisse n'est pas connue ici. XGBoost à réglages de base (sans `subsample`) redonne exactement 19,70 et 13,10. Conséquence pratique : `scripts/generer_predictions_arbres.py` s'arrête sur son contrôle `ATTENDU_FD001_FINAL` (12,73) sur cette machine ; les fichiers v2 ont été écrits en retirant ce seul contrôle par un script de lancement hors dépôt (les autres contrôles du script sont passés). Les notebooks v2 affichent 12,78 et le disent (`_modele`, section 6.2 ; `evolution`, section 5.4).
- Aucun autre écart non nul parmi les chiffres comparables.

**Ce que 1.1 ne prouve pas.** La non-régression (plan 9.2) n'est vérifiable que sur les chiffres publiés en v1 : RMSE, score NASA et surestimés des 3 arbres FD001, toutes les métriques des réseaux FD001 et FD002 et de la référence naïve. Pour les arbres de FD002 à FD004 (12 lignes) et pour la MAE et les alertes manquées des arbres de FD001, la v2 donne un chiffre que rien en v1 ne permet de contrôler.

### 1.2 Modèles v1 remplacés : versions retenues du notebook 05 contre modèles v2

Le notebook 05 de la v1 (Benjamin, 2026-11-05) présentait, pour les 4 sous-jeux, des modèles **antérieurs** à ceux de la démo du tag : arbres à réglages de base avec `ecartinit_20`, réseaux du 03b (réglages fixés à l'avance) avec `ecartinit_20`. Ces modèles ne sont plus dans les notebooks de résultats v2 ; ils restent visibles dans `evolution.ipynb`, section 4.7 (qui redonne les mêmes chiffres, recalculés pour les arbres et relus dans `docs/resultats/03b/` pour les réseaux). Ici, un écart n'est **pas une régression** : c'est un changement de modèle. Source v1 : S4 ; v2 : `sorties/predictions/`. Aucun test apparié v1 contre v2 n'a été fait pour ces écarts.

| Sous-jeu | Modèle | RMSE v1 → v2 (écart) | MAE | Score NASA | Surestimés > 20 | Alertes manquées |
|---|---|---|---|---|---|---|
| FD001 | Référence naïve | 43,07 → 43,07 (0) | 35,90 → 35,90 | 33 629,2 → 33 629,2 | 41 → 41 | 25 → 25 |
| FD001 | Random Forest | 14,27 → 13,14 (−1,13) | 10,23 → 9,45 | 301,9 → 243,3 | 3 → 3 | 4 → 4 |
| FD001 | XGBoost | 13,10 → 12,78 (−0,32) | 9,58 → 9,37 | 237,0 → 232,5 | 2 → 2 | 3 → 5 |
| FD001 | LightGBM | 13,40 → 12,76 (−0,64) | 9,53 → 9,29 | 283,6 → 234,6 | 3 → 3 | 3 → 4 |
| FD001 | LSTM | 12,29 → 11,16 (−1,13) | 8,33 → 7,81 | 228,7 → 227,2 | 5 → 4 | 1 → 0 |
| FD001 | CNN 1D | 12,74 → 11,68 (−1,06) | 9,21 → 8,56 | 346,4 → 189,3 | 11 → 4 | 0 → 0 |
| FD002 | Référence naïve | 54,08 → 54,08 (0) | 45,64 → 45,64 | 153 229,3 → 153 229,3 | 106 → 106 | 61 → 61 |
| FD002 | Random Forest | 24,11 → 23,90 (−0,21) | 15,81 → 15,65 | 4 573,1 → 4 326,3 | 13 → 13 | 1 → 0 |
| FD002 | XGBoost | 23,90 → 23,60 (−0,30) | 15,41 → 15,12 | 4 876,5 → 4 394,7 | 14 → 15 | 1 → 1 |
| FD002 | LightGBM | 23,77 → 23,73 (−0,04) | 15,36 → 15,27 | 4 573,8 → 4 600,3 | 15 → 15 | 0 → 0 |
| FD002 | LSTM | 22,57 → 22,38 (−0,19) | 14,45 → 14,13 | 3 505,1 → 3 432,5 | 18 → 17 | 0 → 0 |
| FD002 | CNN 1D | 23,80 → 23,89 (+0,09) | 15,28 → 15,39 | 4 386,6 → 4 436,5 | 12 → 12 | 0 → 0 |
| FD003 | Référence naïve | 45,07 → 45,07 (0) | 38,00 → 38,00 | 57 949,6 → 57 949,6 | 49 → 49 | 20 → 20 |
| FD003 | Random Forest | 12,72 → 12,34 (−0,38) | 9,39 → 8,97 | 255,7 → 256,1 | 4 → 6 | 1 → 2 |
| FD003 | XGBoost | 12,24 → 11,78 (−0,46) | 9,19 → 8,63 | 257,1 → 246,0 | 6 → 6 | 3 → 3 |
| FD003 | LightGBM | 12,17 → 12,14 (−0,03) | 9,16 → 9,06 | 237,5 → 256,5 | 6 → 4 | 3 → 3 |
| FD003 | LSTM | 13,40 → 13,30 (−0,10) *(mis à jour le 2026-11-08 ; avant : absent)* | 9,44 → 9,40 | 313,5 → 353,9 | 10 → 12 | 2 → 1 |
| FD003 | CNN 1D | 17,00 → 11,92 (−5,08) *(mis à jour le 2026-11-08 ; avant : absent)* | 12,90 → 8,28 | 382,3 → 265,8 | 2 → 7 | 0 → 0 |
| FD004 | Référence naïve | 54,90 → 54,90 (0) | 46,75 → 46,75 | 189 682,8 → 189 682,8 | 101 → 101 | 53 → 53 |
| FD004 | Random Forest | 25,47 → 25,18 (−0,29) | 17,58 → 17,42 | 4 029,4 → 3 856,9 | 20 → 18 | 5 → 6 |
| FD004 | XGBoost | 24,76 → 24,67 (−0,09) | 16,84 → 16,79 | 3 721,3 → 3 715,2 | 21 → 21 | 4 → 3 |
| FD004 | LightGBM | 24,95 → 24,82 (−0,13) | 17,07 → 17,05 | 3 803,3 → 3 763,0 | 21 → 18 | 6 → 6 |
| FD004 | LSTM | 23,55 → 23,75 (+0,20) *(mis à jour le 2026-11-08 ; avant : absent)* | 16,12 → 15,64 | 3 309,7 → 3 529,2 | 12 → 10 | 6 → 5 |
| FD004 | CNN 1D | 29,25 → 26,83 (−2,42) *(mis à jour le 2026-11-08 ; avant : absent)* | 21,61 → 19,27 | 446 373,7 → 5 288,3 | 55 → 11 | 23 → 2 |

**Explication des écarts**

- **Référence naïve** : même modèle (moyenne de la RUL plafonnée du train), écart nul partout.
- **Arbres (12 lignes)** : changement de réglages, de ceux « de base » du notebook 05 à `CONFIG_*_FINAL` (Optuna, notebooks 04f et 04g). Ces réglages ont été cherchés sur FD001 seulement, puis appliqués tels quels à FD002–FD004. Le RMSE baisse sur les 12 lignes (de 0,03 à 1,13 cycle) ; les autres métriques ne bougent pas toutes dans le même sens : score NASA en hausse pour LightGBM FD002 (+26,5), Random Forest FD003 (+0,4) et LightGBM FD003 (+19,0) ; surestimés en hausse pour XGBoost FD002 (+1) et Random Forest FD003 (+2) ; alertes manquées en hausse pour XGBoost FD001 (+2), LightGBM FD001 (+1), Random Forest FD003 (+1) et FD004 (+1). Ces hausses portent sur un à deux moteurs et ne sont pas testées.
- **LSTM FD001 et FD002** : réglages cherchés par Optuna sur chaque sous-jeu, confirmés sur 3 découpages neufs, et moyenne de 5 graines (notebook 04h). Le LSTM v1 était un seul réseau du 03b, graine 42.
- **CNN 1D FD001** : réglages Optuna du 04h (graine 42), contre ceux du 03b en v1.
- **CNN 1D FD002, +0,09** : mêmes réglages (ceux du 03b, le candidat Optuna n'ayant pas passé la confirmation), mais un autre entraînement : v1 = exécution du 03b du 2026-11-04 (machine de Benjamin), v2 = réseau réentraîné pendant le calcul du 04h (`models/04h/`). Un réseau n'est pas reproductible au bit près d'une machine à l'autre (`docs/resultats/03b/README.md`).
- **LSTM et CNN 1D de FD003 et FD004 : absents en v2**, par la règle « uniquement les nouveaux modèles » (aucune recherche Optuna sur ces sous-jeux). **Coût de cette règle, constaté sur le test** : sur FD004, le LSTM du 03b (23,55 brut, 10,95 contre la RUL plafonnée, score NASA 3 309,7) a un RMSE plus bas que tous les modèles de la v2 (meilleur : XGBoost, 24,67 / 12,35 / 3 715,2). Sur FD003, ce n'est pas le cas (LSTM du 03b 13,40 contre XGBoost v2 11,78). Constat fait sur le test, sans test apparié ; il ne change pas un choix déjà fait, mais la démo v2 ne propose plus ce modèle sur FD004.
- *Mise à jour du 2026-11-08 :* **LSTM et CNN 1D de FD003 et FD004 : présents en v2**, cherchés par Optuna et choisis par la règle de confirmation du 04h (section 6). Sur FD004, le LSTM du 03b de la v1 (un seul réseau, graine 42, 23,55) **reste un peu meilleur au test contre la RUL brute** que l'ensemble de 5 LSTM de la v2 (23,75, +0,20), mais pas contre la RUL plafonnée (10,95 contre 10,63). Sur FD003, le LSTM v2 (13,30) est presque au niveau du LSTM du 03b (13,40) et **les deux réseaux v2 restent moins bons au test que le XGBoost v2** (11,78). Écarts lus sur le test, sans test apparié v1 contre v2 ; ils ne changent aucun choix.

### 1.3 Anciens modèles v1 encore présents en v2, dans `evolution.ipynb` seulement

Le journal du plan (2026-11-07) interdit les modèles anciens dans la v2 ; le plan (5.4, 7.1, 8.2) envoie pourtant leur parcours dans `evolution`. Les deux sont appliqués : les anciens modèles n'apparaissent qu'en `evolution` (et, comme « ce qui a fondé la décision », en `02_choix_du_modele`). Leurs chiffres, v1 → v2 :

| Modèle | Chiffre v1 (source) | Chiffre v2 (où) | Écart |
|---|---|---|---|
| XGBoost Phase 3, FD001 | CV 16,87 ; test 19,70 / NASA 852,9 / 9 surestimés / 7 alertes manquées (S1, 2026-11-04) | `evolution` 1.3, 1.4, 2.2 : 16,87 ; 19,70 / 852,9 / 9 / 7 | 0 |
| XGBoost Phase 3, FD002 / FD003 / FD004 | test 27,23 / 18,09 / 28,24 (S1, 2026-11-04) | `evolution` 2.2 : 27,23 / 18,09 / 28,24 | 0 |
| Normalisation par régime, FD002 | CV 19,48 → 16,41 (S1) | `evolution` 2.1 : 19,48 → 16,41 | 0 |
| LSTM du 03b, sans écart initial, FD001 | CV 13,70 ; test 13,86 / 293,2 / 8 / 0 (S1) | `evolution` 3.1, recalculé : 13,70 ; 13,86 / 293,2 / 8 / 0 | 0 |
| Arbres de base + `ecartinit_20`, FD001 à FD004 | voir 1.2 (S4) | `evolution` 4.5 et 4.7, recalculés | 0 |
| Réseaux du 03b + `ecartinit_20`, FD001 à FD004 | voir 1.2 (S4) | `evolution` 4.5 et 4.7, relus dans `docs/resultats/03b/` | 0 |
| CNN 1D du 03b + `ecartinit_20`, FD001, réentraînement du 2026-11-05 | test 13,09 (`docs/resultats/04h/03b_deep_learning.csv`, S3) | `evolution` 6.2 : 13,09 | 0 |
| LSTM 04h, une graine ; ensemble 5 LSTM + 5 CNN (non retenu), FD001 | 11,46 ; 10,97 (S1, S3) | `evolution` 6.2 : 11,46 ; 10,97 | 0 |
| Benchmark exploratoire (Random Forest) | 18,97 (README) | `evolution` 1.1 : 18,97, **lu dans le README, non recalculé** (notebook supprimé, ADR-0006) | non vérifiable |

Deux exécutions du 03b coexistent pour le CNN 1D FD001 : 12,74 au test (2026-11-04, Benjamin, chiffre de S1 et du texte du 03b, copié dans `docs/resultats/03b/`) et 13,09 (réentraînement du 2026-11-05, `docs/resultats/04h/`). Les deux sont montrées en v2, chacune avec sa source ; le choix de celle qui fait foi est une question ouverte (section 5).

### 1.4 Validation croisée (complément)

- **Arbres finaux** : la v1 n'a **aucune** validation croisée avec ces réglages (journal du plan, 2026-11-07). La v2 la calcule (`scripts/valider_arbres_finaux.py`, mêmes 5 plis `GroupKFold`, graine 42), RMSE moyen ± écart-type entre plis : FD001 Random Forest 9,95 ± 0,58, XGBoost 9,12 ± 0,43, LightGBM 9,06 ± 0,39 ; FD002 11,45 / 10,97 / 11,00 ; FD003 9,49 / 8,87 / 9,07 ; FD004 11,85 / 11,54 / 11,69. Pas de chiffre v1 à comparer. Sur FD001, ces chiffres sont **optimistes** : les réglages ont été cherchés par Optuna sur FD001 (04f, 04g), sans doute sur les mêmes plis (non vérifié dans le 04f et le 04g).
- **Réseaux** : même source en v1 et en v2 (`docs/resultats/04h/`), mêmes chiffres : FD001 LSTM 7,38, CNN 1D 8,30 ; FD002 LSTM 9,50, CNN 1D 11,05 (moyennes sur 3 découpages neufs, graines 1000 à 1002). *Mise à jour du 2026-11-08 : FD003 LSTM 7,59, CNN 1D 7,77 ; FD004 LSTM 10,74, CNN 1D 11,96 (pas de chiffre v1 : aucun réseau du 04h sur ces sous-jeux au tag).*
- Les arbres et les réseaux ne sont pas validés sur les mêmes plis. La v2 ajoute la validation des arbres sur les 3 découpages neufs du 04h (FD001 : Random Forest 9,71, XGBoost 9,00, LightGBM 8,95 ; *mise à jour du 2026-11-08 : FD003 9,08 / 8,54 / 8,69, FD004 11,87 / 11,58 / 11,74, calculés le 2026-11-08 par `scripts/valider_arbres_finaux.py --complements`*), ce qui rend les moyennes comparables, mais pas appariées : aucune prédiction hors-pli par ligne n'a été enregistrée pour les réseaux.

---

## 2. Les cellules : où est passée chaque section

Pour chaque ancien notebook : la section v1 (numéros de cellule au tag entre crochets), son devenir, et l'emplacement v2 ou la raison du retrait. « 8.2 » renvoie au tri détaillé du plan. Les numéros de cellule v2 sont donnés pour `01_donnees` et `02_choix_du_modele` ; pour `evolution` et `resultats/`, on cite la section, plus stable. Les cellules de `resultats/` sont celles de `_modele.ipynb`, identiques dans les 4 notebooks générés. Inventaire vérifié sur les fichiers le 2026-11-07, après les correctifs de la revue.

Partout, les fonctions recopiées (`load_fd001`, `COLUMNS`, `ajouter_features_glissantes`, `score_nasa`, `calculer_metriques`, préparation des features) sont remplacées par l'appel à `src/` (8.2, « Partout : le code recopié »). Ce n'est pas répété ligne par ligne.

### 2.1 `01_eda_preparation` (34 cellules) → `01_donnees`

| Section v1 | Devenir | Emplacement v2 ou raison |
|---|---|---|
| §1 imports et noms des colonnes [1–2] | gardée | en-tête [2–3] ; `COLUMNS` vient de `src/preprocessing` |
| §2 fonction de chargement [3–4] | explication gardée, code remplacé | 2.1 ; appel à `load_fd001` |
| §3–4 chargement, contrôle visuel [5–8] | gardées | 2.2, 2.3 |
| §5–8 capteurs constants [9–17] | gardées | 2.4 à 2.8 ; boucles remplacées par `get_sensor_columns`, `find_constant_sensors` |
| §9–13 min-max contre standardisation, figure, décision [18–28] | gardées | §3 ; min-max par `normalize_minmax`, `StandardScaler` inchangé |
| §14–15 lissage et vérification [29–33] | gardées | §4 ; `lisser_capteurs` |

### 2.2 `01b_sous_jeux_fd002_fd004` (52 cellules) → `01_donnees`, `evolution`, `resultats/`

| Section v1 | Devenir | Emplacement v2 ou raison |
|---|---|---|
| Introduction [0] | remplacée | en-tête de `01_donnees` |
| Règles méthodologiques [1–2] | règle 1 gardée | `01_donnees` [1] ; règles 2 à 4 non reprises : elles portaient sur D.4 et F, déplacés |
| Résultats du script [3–4] | non reprise | prérequis de D.4 et F seulement |
| A, pourquoi FD001 d'abord, contrôle bit à bit [5–6] | gardée | `01_donnees` 1.1 et §8 (contrôle étendu à `ecart_initial = 20`) |
| A, le script retrouve 19,70 / 852,9 [7–8] | non reprise telle quelle | `evolution` étape 1 recalcule 19,70 / 852,9 |
| B, ce qui change [9–11] | gardée | `01_donnees` 1.2 |
| C.1 à C.4, régimes, K-Means, figure [12–22] | gardée | `01_donnees` §5 |
| D.1 à D.3, normalisation par régime, figure [23–30] | gardée | `01_donnees` §6 |
| D.4, gain en validation croisée [31–35] | déplacée (8.2) | `evolution` 2.1 |
| E, capteurs retirés par sous-jeu [36–40] | gardée | `01_donnees` §7 |
| F.1, F.3, tableau et figure XGBoost Phase 3 [41–42, 45–46] | déplacées (8.2) | `evolution` 2.2 |
| F.2, RUL vraie au-dessus de 125 [43–44] | déplacée (8.2) | `resultats/` 1.1 ; aussi en `evolution` 2.2 |
| F.4, temps de calcul [47–48] | **retirée** | 8.2 : non réutilisée |
| Lecture de F [49] | déplacée | `evolution` 2.3 |
| G, conclusion et limites [50] | en partie | préparation → `01_donnees` §9 ; points 4 et 5 et limites sur les résultats (hyperparamètres, plafond, graine unique) non repris (question ouverte) |
| G, fichiers pour l'application [51] | **retirée** | 8.2 : non réutilisée |

### 2.3 `02_target_engineering` (16 cellules) → `02_choix_du_modele` §2

| Section v1 | Devenir | Emplacement v2 |
|---|---|---|
| §1 imports et chargement [1–4] | gardée, chargement par `src` | 2.1 [3–5] |
| §2 calcul de la RUL [5–6] | gardée | 2.2 [6–7] |
| §3 vérification [7–8] | gardée | 2.3 [8–9] |
| §4 plafonnement et preuve chiffrée, figure [9–13] | gardée | 2.4 [10–14] |
| §5 RUL brute contre plafonnée, figure [14–15] | gardée | 2.5 [15–16] |

### 2.4 `03_modelisation` (36 cellules) → `02_choix_du_modele`, `evolution`

| Section v1 | Devenir | Emplacement v2 ou raison |
|---|---|---|
| §1–2 imports et pipeline [1–4] | gardées | `02` 3.1 [19–21] |
| §3–5 features glissantes, premier cycle, construction [5–10] | gardées, calcul par `src/features` | `02` 3.2 à 3.4 |
| §6 score NASA [11–12] | gardée, `score_nasa` de `src` | `02` 3.5 |
| §7 GroupKFold [13–14] | gardée | `02` 3.6 et `evolution` 1.3 |
| §8–10 modèle final, test officiel, figure [15–23] | déplacées | `evolution` 1.4, 1.5 |
| §11 ablation à 7 jeux [24–25] | **retirée** | 8.2 : remplacée par l'ablation du 04e ; une ligne en `evolution` 7.4 |
| §12 train contre validation par pli, figure, lecture [26–27, 32] | gardée | `02` §11 [191–195] |
| §12 courbes `max_depth`, `n_estimators` [28–31] | déplacées (8.2) | `evolution` 5.1 |
| §13 « quatre références », figure [33–35] | déplacée **avec toute la cellule** | `evolution` 1.6, 1.7. 8.2 retirait la section sauf sa figure ; la règle du 2026-11-07 (« figures déplacées avec leur cellule ») l'a fait garder entière |

### 2.5 `03b_deep_learning_lstm_cnn` (103 cellules) → `02_choix_du_modele`, `resultats/`, `evolution`

| Section v1 | Devenir | Emplacement v2 ou raison |
|---|---|---|
| A.1 règles [1–2] | gardée | `02` 4.1 [34] |
| A.2 imports [3–4] | gardée | `02` §1 [2] et [35–36] |
| A.3 résultats de l'entraînement [5–6] | gardée | `02` 4.2 |
| A.4 données et plis [7–8] | gardée | `02` 4.3 |
| A.5 référence XGBoost [9–10] | gardée | `02` 5.1 |
| B.1 à B.5, fenêtre, remplissage, contrôle, tenseur, 30 cycles [11–22] | gardées | `02` 4.4 [41–53] (figure B.1 en [44]) |
| B.6, B.7, B.9, neurone, perte, époque [23–26, 29–30] | condensées en texte (8.2) | `02` [54] |
| B.8 normalisation de la cible [27–28] | gardée | `02` [55–56] |
| B.10 à B.12, LSTM, convolution, CNN [31–36] | gardées | `02` [57–62] (figure B.11 en [60]) |
| B.13 dropout [37–38] | condensée en texte (8.2) | `02` [63] |
| B.14 arrêt précoce [39] | gardée | `02` [64] |
| B.15 démonstration de 3 époques [39–40] | **retirée** | 8.2 |
| B.16 reproductibilité CPU / MPS [41–43] | **retirée** | 8.2 |
| C, `time_cycles` [44–46] | gardée | `02` 4.5 |
| D.1 à D.4 et figure de synthèse [47–58, 62] | gardées | `02` 5.2 à 5.5 (figure en [85]) |
| D.5 courbes d'apprentissage [59–61] | gardée | `02` 12.9 [214–215] |
| D.6 conclusion [63] | gardée | `02` 5.6 |
| E, perte asymétrique α = 2,26 [64–70] | **retirée** | 8.2 : non retenue ; une ligne en `evolution` 7.4 |
| F.1 à F.3, F.5, test avec les anciens réglages [71–76, 79–80] | **retirées** | 8.2 : modèles remplacés par ceux de `resultats/` |
| F.4 robustesse à la graine [77–78] | gardée | `02` 10.6 [182–183] |
| F.6 figure du test [81–82] | retirée par 8.2, **déplacée** par la règle du 2026-11-07 | `evolution` 4.6 |
| F.7 trajectoires de deux moteurs [83–84] | remplacée | `resultats/` §7 : même idée (RUL prédite à chaque cycle), code nouveau (`figure_trajectoire_moteur`), moteur 42, modèles v2. La figure à deux moteurs du 03b n'est pas reprise telle quelle |
| F.8 conclusion [85] | **retirée** | 8.2 |
| G.1 à G.4, G.6, FD002 à FD004 [86–97, 100] | **retirées** | 8.2 : anciens réglages ; les chiffres par sous-jeu restent en `evolution` 4.7 |
| G.5 figure par sous-jeu [98–99] | retirée par 8.2, **déplacée** | `evolution` 4.7 |
| H, synthèse [101–102] | **retirée** | 8.2 |
| (ajout) LSTM du 03b sans écart initial, recalculé | code repris de `src/optimisation_reseaux.py` | `evolution` étape 3 (figure nouvelle, section 3.2) |

### 2.6 `04a_metriques_regression` (17 cellules) → `resultats/`

| Section v1 | Devenir | Emplacement v2 ou raison |
|---|---|---|
| §1 reconstruction de la Phase 3 [0–1] | **retirée** | 8.2 : 109 lignes recopiées |
| §2 les 5 baselines [2–3] | gardée, calcul dans `src/evaluation_sous_jeu.py` | `resultats/` §3 et §4 |
| §3 métriques calculées à la main [4–5] | remplacée | `metriques_regression` de `src` ; tableau complet en `resultats/` 3.1 et 4 |
| §4 tableaux train et validation [6–7] | gardée | `resultats/` 3.1 : baselines et arbres finaux (erreur train pli par pli, `scripts/valider_arbres_finaux.py --complements`). Erreur train des réseaux non calculée |
| §5 erreur par tranche de RUL [8–9] | gardée | `resultats/` 3.2 et 4.2 |
| §6 test officiel [10–11] | gardée | `resultats/` §4 |
| §7 figures prédit contre vrai et résidus [12–13] | gardée | `resultats/` 3.3 (validation) et 4.3 (test) |
| §8 XGBoost contre Ridge pli par pli [14–15] | **retirée** | 8.2 : remplacée par Wilcoxon (`resultats/` §6) |
| §9 gardée / écartée [16] | gardée | `resultats/` 4.1 : verdicts gardés ; chiffres de l'ancien XGBoost retirés, sauf un (+10,30 cycles), étiqueté « ancien modèle » |
| §10 ce que montrent les tableaux [16] | remplacée | cellules « Ce qu'on lit », construites à partir des chiffres calculés |
| §11 limites [16] | gardée | `resultats/` 6.2 |

### 2.7 `04b_metriques_classification` (32 cellules) → `resultats/`, `02_choix_du_modele`

| Section v1 | Devenir | Emplacement v2 ou raison |
|---|---|---|
| Chargement et features [1–4] | **retirées** | 8.2 : code recopié (`src`) |
| §1 étiquettes « RUL ≤ N » [5–6] | gardée | `resultats/` §5 (texte) ; `02` 13.2 [219] |
| §2 plis et prédictions hors-pli [7–9] | gardée | `02` 13.2 [220–221] |
| §3 approche 1, alerte dérivée de la régression [10–11] | gardée | `resultats/` §5 |
| §4 approche 2, classifieurs dédiés [12–14] | résumée (8.2) | `02` 13.1 [217] |
| §5 choix du seuil [15–18] | gardée, **deux règles** | `resultats/` §5 : rappel minimal 0,95 (`RAPPEL_MINIMUM` de `src`) ; `02` [222–224] : rappel 0,99 du 04b, avec sa fonction `choisir_seuil` gardée. L'écart est signalé dans les deux notebooks |
| §6 l'accuracy trompe [19–20] | gardée | `resultats/` 5.1 |
| §7 figures en validation, N = 15, 30, 50 [21–22] | remplacée | `resultats/` 5.2 et 5.3 : au test, N = 30, une figure par sous-jeu (plan 5.3 et 7.1). Les versions en validation et à N = 15 et 50 ne sont pas reprises |
| §8 cycles d'avance avant la panne [23–24] | gardée en attente | `02` 13.3 [225–226] (décision de Matisse attendue, plan 8.2) |
| §9 test officiel [25–27] | gardée | `resultats/` §5 (colonnes `test_*`) |
| §10 synthèse, gardée / écartée [28] | gardée | `resultats/` 5.4 |
| §11 test apparié des approches [29–31] | résumée (8.2) | `02` 13.1 [217] |

### 2.8 `04c_diagnostic_apprentissage` (17 cellules) → `02_choix_du_modele` §12

Tout est gardé (8.2) : §0 → 12.1 ; §1 chargement → 12.2 (par `src`) ; §2 outils de mesure → 12.3 (`calculer_metriques` appelle désormais `metriques_regression`) ; §3 plis → 12.4 ; §4 courbes et figure → 12.5 [209] ; §5 pente → 12.6 ; §6 conclusion → 12.7 ; §7 limites → 12.8.

### 2.9 `04d_hyperparametres` (18 cellules) → `02_choix_du_modele`, `evolution`

| Section v1 | Devenir | Emplacement v2 ou raison |
|---|---|---|
| Imports, chargement, `score_nasa` recopié [1–3] | remplacés | `src` (`02` [138–140]) |
| `evaluer_reglage` et référence [4] | gardée | `02` [141] |
| Règle de décision, espace de recherche [5–6] | gardées | `02` [142–143] |
| Grid search, random search [7–9] | **retirées de `02`** (8.2) | recalculées en `evolution` 5.2 (avec cache) pour la figure RMSE contre score NASA ; une ligne en `evolution` 7.4 |
| Optuna et meilleur essai [10–11] | gardées | `02` [144–147] |
| Figures [12] | gardée en partie | `02` [148] : historique (Optuna seul) et importance ; RMSE contre score NASA → `evolution` 5.2 |
| Confirmation sur plis neufs, figure [13–14] | gardée | `02` 9.1 [149–151] |
| Test des réglages inchangés [15] | **retirée** | 8.2 |
| Tableau des méthodes [16] | **retirée** | 8.2 ; une ligne en `evolution` 7.4 |
| Conclusion [17] | gardée | `02` [152] |

### 2.10 `04e_feature_engineering` (31 cellules) → `02_choix_du_modele` §6, `evolution`

| Section v1 | Devenir | Emplacement v2 ou raison |
|---|---|---|
| Point de départ, contraintes [0] | gardé | `02` 6.0 [89] |
| §1 chargement [1–3] | gardée, par `src` | `02` 6.1 |
| §2 features candidates, tests de causalité et de reproduction [4–9] | gardée | `02` 6.2 |
| §3 cadre et règle de décision [10–11] | gardée | `02` 6.3 |
| §4 hypothèses [12–15] | gardée | `02` 6.4 |
| §5 tableau et figure d'ablation [16–18] | gardée | `02` 6.5 (figure [107]) |
| §6 sélection [19–21] | gardée | `02` 6.6 |
| §7 importances gain et permutation [22–24] | gardée | `02` 6.7 (figure [113]) |
| Synthèse chiffrée | gardée, sans la ligne du test | `02` 6.8 |
| §8 test officiel du jeu retenu [25–27] | déplacée (8.2) | `evolution` 4.5 (Wilcoxon 31 contre 46 variables) |
| §9 variantes testées après lecture des résultats [28–29] | **retirée** | 8.2 ; une ligne en `02` [116] et en `evolution` 7.4 |
| §10 conclusions [30] | gardée | `02` 6.9 [117] |

### 2.11 `04f_modele_final` (5 cellules) → `02_choix_du_modele`, `evolution`

| Cellule v1 | Devenir | Emplacement v2 |
|---|---|---|
| [0] présentation | gardée | `02` 9.2 [153] |
| [1] `features_31`, `features_46` recopiées | remplacée | `src` (`preparer_train`) |
| [2] Optuna sur 31 variables, confirmation sur 10 découpages | gardée | `02` 9.2 [155], plus un contrôle ajouté [156–157] : la recherche relancée ici ne retrouve pas `CONFIG_XGBOOST_FINAL`, mais `CONFIG_XGBOOST_FINAL` gagne +0,229 sur les mêmes 10 découpages |
| [3] évaluation 46 contre 31, seuil d'alerte, métriques | déplacée (8.2) | `evolution` 5.3 |
| [4] figures (matrices par N, test 100 moteurs) | déplacée (8.2) | `evolution` 5.3 |

### 2.12 `04g_comparaison_modeles` (4 cellules) → `02_choix_du_modele`, `resultats/`, `evolution`

| Cellule v1 | Devenir | Emplacement v2 |
|---|---|---|
| [0] présentation | gardée | `02` 9.3 [158] |
| [1] features recopiées | remplacée | `src` |
| [2] validation et test des 3 arbres | gardée, séparée | validation → `02` 9.3 [159] ; test → `resultats/FD001` §4 (fichiers de la démo) et `evolution` 5.3 |
| [3] RUL prédite cycle par cycle, 4 moteurs | déplacée | `evolution` 5.3 bis, avec les 3 arbres finaux (ceux de la démo), et non les modèles bruts du 04g |

### 2.13 `04h_optuna_reseaux` (35 cellules) → `02_choix_du_modele`, `evolution`, `resultats/`

| Section v1 | Devenir | Emplacement v2 |
|---|---|---|
| Introduction, imports [0–1] | gardés | `02` [161–163] |
| §1 à §5 protocole, recherche, importance, confirmation, ablation [2–19] | gardées | `02` 10.1 à 10.5 (figures [169], [172], [177]) |
| §6 ensembles [20–22] | gardée | `02` [184–186] |
| §7 test officiel, tableau, remarques, figure [23–27] | déplacée | `evolution` 6.2 ; les chiffres de test des modèles retenus sont aussi en `resultats/` §4 (fichiers de la démo) |
| §8 évolution étape par étape [28–30] | déplacée | `evolution` étape 6 |
| §9 littérature [31] | déplacée | `resultats/` 4.4 (`ARTICLES` de `src/evaluation_sous_jeu.py`) |
| §10 où sont les réglages, §11 limites [32–34] | gardées | `02` 10.7, 10.8 |

### 2.14 `05_comparaison_evaluation` (119 cellules) → `02_choix_du_modele`, `resultats/`, `evolution`

| Section v1 | Devenir | Emplacement v2 ou raison |
|---|---|---|
| A.1 règles [2] | gardée | en-tête de `resultats/` (règle 4 non reprise : elle concerne la section C ; règle 2 complétée par l'exception des réseaux) |
| A.2 à A.5 imports, données, plis, colonnes [3–10] | remplacées | `src/evaluation_sous_jeu.py` ; `resumer_cv` et `resumer_test` recopiées dans `evolution` (helpers) |
| B, B.1, B.2 réglages de base en validation [11–16] | déplacées (8.2) | `evolution` 4.1 |
| B.3 figure avec temps d'entraînement [17–18] | déplacée | `evolution` 7.2 |
| B.4 conclusion [19] | déplacée | `evolution` 4.2 (lecture) et 7.2 |
| C, pénalité de la surestimation, figure [20–31] | déplacée (8.2) | `evolution` 7.1 |
| D, D.1, D.2 test à réglages de base [32–37] | déplacées (8.2) | `evolution` 4.2 |
| D.3 figure [38–39] | déplacée | `evolution` 4.2 |
| D.4 conclusion [40] | déplacée | `evolution` 4.2 (lecture) |
| E, moteurs surestimés, figure [41–47] | déplacée (8.2) | `evolution` 7.3 |
| F.1 à F.6, F.8 (partie validation) [48–62, 65] | gardées | `02` §7 [119–135] ; figure F.6 aussi en `evolution` 4.4 ; F.5 résumée en une ligne avec sa cellule de décision (`02` 7.5) |
| F.7 test des versions retenues, FD001 à FD004 [63–64] | déplacée | `evolution` 4.5 (FD001) et 4.7 (FD002 à FD004) |
| Contrôle F [66–67] | gardé | `02` [135] |
| G.1 à G.6, protocole sur FD001 [68–84] | généralisées aux 4 sous-jeux | `resultats/` §3 à §6, par `src/evaluation_sous_jeu.py` (seul code nouveau autorisé, principe 0) |
| G.7 et contrôle G [85–87] | remplacés | cellules « Ce qu'on lit » ; contrôle par `tests/test_evaluation_sous_jeu.py` |
| H.1 à H.3 et contrôle H, littérature [88–95] | déplacées | `resultats/` 4.4 (plancher de H.1 aussi en 1.1) |
| I, synthèse [96–105] | **retirée** | 8.2 : remplacée par `resultats/` |
| J.1, J.2, J.5, J.6, réglages, chargement, lancement de Streamlit [106–110, 115–118] | **retirées** | 8.2 |
| J.3, J.4, moteur mystère, révélation [111–114] | généralisées | `resultats/` §7 (`verifier_moteur`, `figure_trajectoire_moteur`, `comparer_aux_fichiers_demo`) |

---

## 3. Les figures

### 3.1 Les 27 figures de la section 7.1 du plan

Notebook v1 et cellule au tag ; emplacement v2 et cellule. Les 27 ont une image enregistrée dans leur notebook v2 (vérifié le 2026-11-07 : 4 images dans `01_donnees`, 17 dans `02_choix_du_modele`, 19 dans `evolution`, 9 dans chacun des 4 `resultats/FD00X`). Les cellules de `resultats/` sont celles de `FD00X.ipynb` (`_modele.ipynb` n'a pas de sortie).

| # | Figure | v1 | v2 | Remarque |
|---|---|---|---|---|
| 1 | Lissage avant / après | `01` [33] | `01_donnees` 4.2 [32] | |
| 2 | Régimes FD002 / FD004 | `01b` [21] | `01_donnees` 5.4 [42] | |
| 3 | Brut → méthode FD001 → normalisation par régime | `01b` [30] | `01_donnees` 6.3 [51] | |
| 4 | RUL brute contre plafonnée | `02_target` [15] | `02` 2.5 [16] | la preuve chiffrée ([13]) est en `02` [14] |
| 5 | Fenêtre glissante | `03b` [14] | `02` 4.4 [44] | |
| 6 | Ablation, gain RMSE | `04e` [18] | `02` 6.5 [107] | |
| 7 | Écart à l'état initial, 5 modèles | `05` [62] | `02` 7.6 [133] et `evolution` 4.4 [52] | |
| 8 | Optuna : historique, importance | `04d` [12] | `02` §8 [148] (2 images) | historique réduit à Optuna ; la version à 3 méthodes (Grid, Random, Optuna) n'est pas reprise (le PNG `docs/figures/04d_historique_recherche.png` reste) |
| 9 | Confirmation sur plis neufs | `04d` [14] | `02` 9.1 [151] | |
| 10 | Train contre validation par pli | `03` [27] | `02` §11 [193] | |
| 11 | Courbe d'apprentissage | `04c` [12] ; `03b` [60] | `02` 12.5 [209] ; 12.9 [215] | |
| 12 | RUL prédite contre vraie | `04a` [13] | `resultats/` 3.3 [22] (validation) et 4.3 [32] (test) | v1 : XGBoost Phase 3 seul ; v2 : une case par modèle v2, code de `src/figures_resultats.py` |
| 13 | Résidus selon la RUL vraie | `04a` [13] | `resultats/` 3.3 [23] et 4.3 [34] | même remarque |
| 14 | Barres RMSE / score NASA contre baselines | `01b` [46] ; `05` [18] | `resultats/` 4.3 [36] | figure nouvelle (`figure_barres_baselines`) ; les figures d'origine sont en `evolution` 2.2 [27] et 7.2 [101] |
| 15 | ROC et précision-rappel | `04b` [22] | `resultats/` 5.2 [46] | v1 : validation, N = 15, 30, 50 ; v2 : test, N = 30 |
| 16 | Importances gain et permutation | `04e` [24] | `02` 6.7 [113] | |
| 17 | Matrices de confusion | `04b` [22] | `resultats/` 5.3 [49] | v1 : validation, 3 valeurs de N ; v2 : test, N = 30 |
| 18 | Erreur par moteur, comparaisons appariées | `05` (B.2, G.6 : tableaux, sans figure) | `resultats/` 6.1 [55] | figure nouvelle, construite à partir des tableaux |
| 19 | RUL prédite cycle par cycle | `03b` [84] | `resultats/` §7 [64] | moteur 42, modèles v2, code nouveau ; aussi `05` J.3 [112] |
| 20 | Courbes `max_depth` / `n_estimators` | `03` [29], [31] | `evolution` 5.1 [64], [66] | |
| 21 | Erreur comparée à la baseline et au benchmark | `03` [34] | `evolution` 1.6 [16] | |
| 22 | XGBoost Phase 3 sur les 4 sous-jeux | `01b` [46] | `evolution` 2.2 [27] | |
| 23 | RMSE contre score NASA par essai | `04d` [12] | `evolution` 5.2 [72] | |
| 24 | Test, 100 moteurs | `04f` [4] | `evolution` 5.3 [77] | |
| 25 | Pénalisation de la surestimation | `05` [30] | `evolution` 7.1 [97] | |
| 26 | Temps d'entraînement | `05` [18] | `evolution` 7.2 [101] | figure B.3 entière |
| 27 | Écart contre RUL vraie / longueur de trajectoire | `05` [46] | `evolution` 7.3 [106] | |

### 3.2 Figures v1 absentes de la section 7.1

| Figure v1 | v2 |
|---|---|
| `01` [26] histogrammes min-max / standard | `01_donnees` 3.4 [26] |
| `02_target` [13] preuve chiffrée (dispersion en début et fin de vie) | `02` [14] |
| `03` [23] test officiel, prédit contre vrai | `evolution` 1.5 [14] |
| `03b` [34] convolution 1D | `02` B.11 [60] |
| `03b` [62] synthèse arbres contre réseaux en validation | `02` 5.5 [85] |
| `03b` [82] test FD001, arbres et réseaux (F.6) | `evolution` 4.6 [56] |
| `03b` [99] quatre sous-jeux (G.5) | `evolution` 4.7 [60] |
| `04g` [3] 4 moteurs cycle par cycle | `evolution` 5.3 bis [81] |
| `04h` [7], [10], [15] historique, importance, confirmation | `02` 10.2 [169], 10.3 [172], 10.4 [177] |
| `04h` [26] « Où l'ensemble retenu se trompe-t-il au test ? » | `evolution` 6.2 [91] |
| `04h` [29] évolution du RMSE du LSTM | `evolution` étape 6 [85] |
| `05` [39] test à réglages de base (D.3) | `evolution` 4.2 [45] |
| `05` [112] moteur mystère (J.3) | `resultats/` §7 [64] (même idée, code nouveau) |

**Figures v2 nouvelles**, sans origine v1 directe : `evolution` 3.2 [36] (LSTM du 03b contre XGBoost) et 8 [111] (escalier) ; `resultats/` 6.1 (erreur par moteur) et 4.3 [36] (barres contre baselines).

**Figures v1 sans équivalent exact en v2** (contraire à la lettre de la règle « aucun graphe supprimé » du 2026-11-07, à faire valider) :
1. `04b` §7 : matrices, ROC et précision-rappel **en validation**, pour N = 15, 30 et 50. La v2 les trace au test, à N = 30 (plan 5.3 et 7.1).
2. `03b` F.7 : trajectoires de **deux** moteurs test avec les réseaux du 03b. Remplacée par la figure du moteur 42 avec les modèles v2.
3. `04d` [12], premier panneau : meilleur RMSE selon le budget pour les **trois** méthodes. La v2 ne trace que l'historique d'Optuna.

Les PNG de `docs/figures/` (dont ceux de ces trois figures) ne sont ni déplacés ni écrasés (plan 6.4).

---

## 4. Limites et coûts

1. **Non-régression partielle.** 49 chiffres publiés en v1 (S1 à S4) peuvent être comparés aux modèles de la démo v2 (section 1.1) ; 47 sont identiques, 2 diffèrent (RMSE et score NASA de XGBoost FD001). 51 chiffres v2 de la section 1.1 n'ont pas de référence v1 : les 45 des arbres finaux de FD002 à FD004 (9 modèles × 5 métriques) et les 6 de la MAE et des alertes manquées des arbres de FD001. Ils existaient dans `sorties/predictions/` sur la machine de Matisse, pas ici.
2. **XGBoost FD001 dépend de la machine** (12,73 contre 12,78). Tant que ce n'est pas réglé, `scripts/generer_predictions_arbres.py`, donc `scripts/preparer_demo.py`, s'arrête sur cette machine ; la recette « clone neuf → démo » n'a pas été testée de bout en bout.
3. **La règle « uniquement les nouveaux modèles » retire de la démo le modèle au RMSE test le plus bas de FD004** (LSTM du 03b, 23,55, contre 24,67 pour le meilleur modèle v2). Constat fait sur le test, sans test apparié. *Mise à jour du 2026-11-08 : la v2 a maintenant un LSTM sur FD004 (ensemble de 5, réglages du 03b, 23,75). Le LSTM du 03b de la v1 reste un peu plus bas au test contre la RUL brute (23,55, écart 0,20 cycle), plus haut contre la RUL plafonnée (10,95 contre 10,63) ; section 6.*
4. **Les anciens modèles restent présents** dans `02_choix_du_modele` (« ce qui a fondé la décision ») et `evolution`. C'est une exception à la lettre du journal du 2026-11-07, qui n'est pas encore actée.
5. **Validation arbres contre réseaux** : pas sur les mêmes plis à l'origine ; la v2 ajoute les 3 découpages neufs pour les arbres de FD001 et FD002 (*et de FD003 et FD004 depuis le 2026-11-08*), mais aucune comparaison appariée n'est possible sans réentraîner les réseaux. La validation des arbres sur FD001 est optimiste (réglages cherchés sur FD001).
6. **Deux exécutions du 03b** (CNN 1D FD001 12,74 ou 13,09) : la v2 en montre deux, chacune avec sa source ; aucune n'est déclarée de référence.
7. **Taille** : les 4 notebooks générés pèsent 2,3 à 2,8 Mo chacun ; avec `01_donnees`, `02` et `evolution`, environ 15 Mo de notebooks dont les sorties seront versionnées (plan 6.3).
8. **Nombre de cellules** : 519 cellules v1 contre 477 dans les 4 notebooks v2 écrits à la main (plus 4 × 69 générées). Le constat 1 du plan (« trop de cellules ») n'est presque pas réduit ; le plan a retiré tout objectif de nombre de cellules (principe 8).
9. **Recherches non reproduites** : la recherche Optuna du 04f relancée sur cette machine ne retrouve pas `CONFIG_XGBOOST_FINAL` (la décision tient, voir 2.11) ; les recherches de 40 essais de LightGBM et de Random Forest (04g) n'ont pas de code dans le dépôt.
10. **Les numéros de cellule** de ce fichier sont ceux du 2026-11-07 ; toute modification d'un notebook les décale.

---

## 5. Questions ouvertes (pour Matisse)

1. XGBoost FD001 : 12,73 (machine de Matisse) ou 12,78 (cette machine) comme chiffre v2 ? Corriger `ATTENDU_FD001_FINAL` et `CONFIG_XGBOOST_FINAL["metriques_test"]` par une entrée de journal et de `STATUS.md`, ou régénérer les fichiers XGBoost sur la machine de Matisse ? Fixer les versions dans `requirements.txt` ?
2. Les fichiers `sorties/predictions/*.json` du tag existent-ils encore sur la machine de Matisse ? Ils permettraient de compléter la section 1.1 pour les arbres de FD002 à FD004.
3. Accepter le retrait des réseaux de FD003 et FD004 de la démo, sachant le constat de 1.2 sur FD004 ? *Mise à jour du 2026-11-08 : question résolue par le calcul de nuit du 2026-11-07 au 08 ; FD003 et FD004 ont maintenant leurs réseaux dans la démo v2 (section 6).*
4. Acter comme exception à la règle « uniquement les nouveaux modèles » la présence des anciens modèles dans `02_choix_du_modele` et `evolution` ?
5. Quelle exécution du 03b fait foi (2026-11-04, `docs/resultats/03b/`, ou 2026-11-05, `docs/resultats/04h/03b_*`) ?
6. Les trois figures v1 sans équivalent exact (section 3.2) : les reprendre dans `evolution`, ou accepter leur remplacement ?
7. Rappel minimal pour le seuil d'alerte : 0,95 (`resultats/`, `src`) ou 0,99 (04b, `02` §13) ?
8. N des matrices de confusion : 30 (proposition du plan, section 12) à confirmer.
9. 04b section 8 (« combien de cycles d'avance ») : gardée en attente en `02` 13.3.
10. Points 4 et 5 de la conclusion du 01b (section G) et ses limites sur les résultats : non repris par aucun notebook v2. Les reprendre en `evolution` ?

---

## 6. Mise à jour du 2026-11-08 : les réseaux de FD003 et FD004

### 6.1 Ce qui a été calculé

- **Calcul** : `bash scripts/nuit_04h_fd003_fd004_mac.sh`, du 2026-11-07 à 17 h 57 au 2026-11-08 à 10 h 09, sur le Mac de Benjamin, 12 processus, 16 h 12 au total, aucune erreur (`docs/resultats/04h/README.md`). Mêmes étapes, mêmes nombres d'essais et mêmes règles écrites avant le calcul que pour FD001 et FD002.
- **Seul écart au déroulé prévu** : à 7 h 16, l'étude LSTM de FD004 étant finie, 9 travailleurs CNN 1D de plus (numéros 3 à 11) ont été lancés sur l'étude `04h_fd004_cnn1d`. Aucun essai n'a fini après le début de l'étape « confirmer » (meilleur essai CNN de FD004, n° 81, fini à 8 h 11 ; « confirmer » du CNN commencé à 8 h 34). Cet essai n° 81 vient du travailleur de renfort n° 10 (`sorties/resultats/04h_fd004_journal_cnn1d_renfort_10.log`) ; il n'a pas passé la confirmation, et les réglages du 03b sont gardés.
- **Publication** : résultats copiés dans `docs/resultats/04h/` (fichiers `04h_fd003_*` et `04h_fd004_*`, base Optuna à 8 études), réglages dans `CONFIGS_RESEAUX_RETENUES` et `ENSEMBLES_RETENUS` (`src/optimisation_reseaux.py`), poids dans `models/04h/`, prédictions de la démo dans `sorties/predictions/`.

### 6.2 Ce que les règles ont décidé (validation seulement, 3 découpages neufs)

| Sous-jeu | Étape | Gain moyen (cycles) | Découpages gagnés | p Wilcoxon | Décision |
|---|---|---|---|---|---|
| FD003 | LSTM, essai Optuna n° 82 contre le 03b | 0,448 | 3 / 3 | 0,042 | retenu |
| FD003 | CNN 1D, essai Optuna n° 79 contre le 03b | 1,257 | 3 / 3 | 3,5e-7 | retenu |
| FD004 | LSTM, essai Optuna n° 36 contre le 03b | 0,142 | 1 / 3 | 0,53 | non retenu : réglages du 03b gardés |
| FD004 | CNN 1D, essai Optuna n° 81 contre le 03b | 0,266 | 2 / 3 | 0,65 | non retenu : réglages du 03b gardés |
| FD003 | 5 LSTM contre un seul | 0,498 | 3 / 3 | 6,4e-5 | retenu |
| FD003 | 5 LSTM + 5 CNN contre un seul LSTM | 0,772 | 3 / 3 | 1,2e-6 | passe la règle |
| FD004 | 5 LSTM contre un seul | 0,364 | 3 / 3 | 7,3e-7 | retenu |
| FD004 | 5 LSTM + 5 CNN contre un seul LSTM | 0,415 | 3 / 3 | 2,1e-7 | passe la règle |

Ablations des entrées du LSTM : aucune variante retenue, sur les deux sous-jeux.

**Choix de l'ensemble**, même règle que pour FD001 et FD002 (commentaire au-dessus de `ENSEMBLES_RETENUS`, et 04h, section 6) : la version à 10 réseaux n'est gardée que si elle gagne au moins 0,3 cycle sur les 5 LSTM en validation. FD003 : 7,589 → 7,315, soit 0,273 cycle ; FD004 : 10,744 → 10,693, soit 0,050 cycle. Les deux sont sous le seuil : **5 LSTM sur les deux sous-jeux**. Ce choix est fait sur la validation, avant de lire le test. Chiffres recalculés à partir de `rmse_cv_par_decoupage` des fichiers `04h_fd00X_ensemble.json` ; ils sont aussi recalculés, avec la même règle, par la cellule 10 des notebooks `resultats/FD003` et `resultats/FD004`, qui vérifie qu'ils redonnent `ENSEMBLES_RETENUS`.

### 6.3 Le test officiel, lu une fois (pour information)

RMSE contre la RUL brute / contre la RUL plafonnée à 125 / score NASA. v1 = réseaux du 03b, un seul réseau (graine 42), exécution du 2026-11-04 (`docs/resultats/03b/03b_deep_learning.csv`) ; v2 = `sorties/predictions/` (fichiers de la démo).

| Sous-jeu | Modèle | v1 | v2 |
|---|---|---|---|
| FD003 | LSTM (v2 : ensemble de 5, réglages Optuna) | 13,40 / 11,93 / 313,5 | 13,30 / 12,05 / 353,9 |
| FD003 | CNN 1D (v2 : graine 42, réglages Optuna) | 17,00 / 14,71 / 382,3 | 11,92 / 10,35 / 265,8 |
| FD003 | XGBoost final (pour comparer) | — | 11,78 / 9,96 / 246,0 |
| FD004 | LSTM (v2 : ensemble de 5, réglages du 03b) | 23,55 / 10,95 / 3 309,7 | 23,75 / 10,63 / 3 529,2 |
| FD004 | CNN 1D (v2 : graine 42, réglages du 03b) | 29,25 / 22,51 / 446 373,7 | 26,83 / 15,80 / 5 288,3 |
| FD004 | XGBoost final (pour comparer) | — | 24,67 / 12,35 / 3 715,2 |

**Ce qu'on peut dire, honnêtement** :

- **FD003 : les réseaux ne battent pas les arbres au test.** Le XGBoost final fait 11,78 (9,96 plafonné), le meilleur réseau 11,92 (CNN 1D) et l'ensemble de LSTM 13,30, alors qu'en validation, sur les mêmes 3 découpages neufs, les réseaux étaient devant (LSTM 7,59, CNN 1D 7,77, XGBoost 8,54). Au test, aucun écart entre modèles qui apprennent n'est significatif après correction de Bonferroni (`resultats/FD003`, section 6). Les réseaux restent publiés parce que les règles qui les ont retenus portent sur la validation, et que le test ne sert jamais à choisir (règle 1 des notebooks de résultats, protocole d'évaluation) : les retirer parce qu'ils perdent au test reviendrait à choisir sur le test. Même raison pour l'ensemble à 10 réseaux, meilleur au test (12,24) mais non retenu en validation (0,273 < 0,3).
- **FD004 : le LSTM v1 du 03b reste un peu meilleur au test contre la RUL brute** (23,55 contre 23,75 pour l'ensemble v2), pas contre la RUL plafonnée (10,95 contre 10,63). Les deux utilisent les réglages du 03b ; ils diffèrent par l'entraînement (autre exécution, 29 époques pour le modèle final en v1 contre 14 en v2, médiane des meilleures époques de leurs plis respectifs) et par la moyenne de 5 graines en v2. Aucun test apparié v1 contre v2 n'a été fait. Sur FD004, l'ensemble v2 fait mieux que le XGBoost final au test (23,75 contre 24,67 ; 10,63 contre 12,35), et le test apparié de `resultats/FD004` (section 6) ne déclare pas cet écart significatif après correction (p corrigée 0,12).
- **Comparaison v1 / v2 du CNN 1D** : sur les deux sous-jeux, le CNN v2 a un RMSE test plus bas que celui du 03b (FD003 : réglages Optuna ; FD004 : mêmes réglages du 03b, autre entraînement). Un CNN dépend fortement de sa graine (FD004 : 26,67 ± 1,55 sur 5 graines en v2, commentaire de `CONFIGS_RESEAUX_RETENUES`) : l'écart de FD004 ne se lit pas comme un progrès de méthode.

### 6.4 Point ouvert (signalé, non corrigé)

- **k de l'écart à l'état initial sur FD004.** Le plus court moteur test de FD004 a 19 cycles ; aucune valeur de `VALEURS_ECART_INITIAL` (20, 25, 30) n'est plus petite, et `valeurs_ecart_initial("FD004")` retombe sur k = 20 (`src/optimisation_reseaux.py`). Les deux réseaux de FD004 utilisent donc k = 20. Pour les moteurs test 141 et 204 (19 cycles chacun), la référence de l'écart n'est jamais figée : `src/features.calculer_ecart_etat_initial` prend, tant que t < k, la moyenne des t cycles déjà connus, donc au dernier cycle la moyenne de leurs 19 cycles, alors que les autres moteurs ont une référence figée sur leurs 20 premiers cycles. Pour ces deux moteurs, l'état initial est donc calculé sur moins de cycles que pour les autres (le cas t < k existe aussi au début de vie de chaque moteur, à l'entraînement comme dans la démo). Même choix que le 03b ; son effet n'est pas mesuré. À trancher par l'équipe.
