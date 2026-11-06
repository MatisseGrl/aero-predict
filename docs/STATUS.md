# État du projet — Aero Predict

Mis à jour après chaque changement significatif. Sert de point d'entrée pour savoir où en est le projet collectivement et ce qui vient de se passer — lu automatiquement en début de session par les assistants IA (voir [`AGENTS.md`](../AGENTS.md), section "Suivi de l'avancement et des décisions"). Pour le détail de qui a poussé quel commit, `git log` / l'historique GitHub font déjà foi ; ce fichier donne le contexte que le git log ne donne pas (pourquoi, quel impact). Les décisions structurantes (pas le travail courant) vivent dans [`docs/decisions/`](decisions/), jamais réécrites une fois actées.

Écriture factuelle uniquement — pas d'opinion non étayée, pas de réécriture silencieuse d'une entrée déjà publiée (voir [`AGENTS.md`](../AGENTS.md), section "Ton d'écriture").

## Phase actuelle

**Phase 4 — Évaluation : commencée le 2026-11-04.** Point de départ : le protocole d'évaluation obligatoire ([`docs/protocole_evaluation.md`](protocole_evaluation.md), [ADR-0007](decisions/0007-protocole-evaluation-obligatoire.md)), dont la section 0 liste ce qui manque au XGBoost de la Phase 3. La Phase 3 (XGBoost, FD001) reste terminée et validée.

**En revue (Benjamin), en deux Pull Requests vers `main`** : `phase-04/modeles-ben` (comparaison de modèles sur FD001, FD002–FD004, LSTM et CNN 1D, variable « écart à l'état initial » de Matisse appliquée aux 6 modèles, protocole d'évaluation sur FD001, comparaison à la littérature), puis `phase-04/demo-ben`, empilée sur la première (démo Streamlit avec prédiction en direct d'un moteur de test). Elles remplacent la PR de la branche `phase-04/prototype-ben`, fermée. Rien n'est encore validé par l'équipe ; les étapes du protocole encore manquantes sont listées dans la dernière entrée. Le workflow complet de `CONTRIBUTING.md` (PR + revue) s'applique : un second contributeur code (fin de la période solo, ADR-0005).

## Dernière modification

- **Qui** : Matisse (avec Claude Code)
- **Date** : 2026-11-07
- **Quoi** : fenêtres des réseaux construites à la demande, lot par lot, pour l'optimisation Optuna (suite des deux entrées suivantes).
  - `src/deep_learning.py` : `FenetresParesseuses` et `construire_fenetres_paresseuses`. Elles gardent le tableau 2D des entrées et, pour chaque fenêtre, les numéros de ses lignes ; une fenêtre est construite seulement quand un lot la demande. `entrainer_modele` et `predire_echelle` acceptent les deux formes. `construire_fenetres` est inchangée : la démo et le 03b ne changent pas.
  - `src/optimisation_reseaux.py` les utilise, et sépare les moteurs d'un pli sans recopier de fenêtres (`selection`).
  - `tests/test_fenetres_paresseuses.py` : fenêtres identiques (`np.array_equal`) sur FD001 et FD002, train et test, 20, 30 et 50 cycles ; un LSTM entraîné 2 époques donne des prédictions identiques au bit près avec les deux méthodes. Les essais Optuna déjà faits sur FD001 restent donc comparables. Suite complète : 29 tests réussis, 6 sautés (modèles de la démo non générés sur cette machine).
  - Mémoire mesurée (un processus, FD002, fenêtres de 50 cycles, données préparées plus les fenêtres d'un pli) : 1 371 Mo avant, 468 Mo après. La part des fenêtres passe de 948 Mo à 45 Mo.
- **Pourquoi** : le 2026-11-06, 15 processus sur FD002 saturaient les 31 Go de la machine (entrée suivante).
  - Nombre de processus : mesure du débit d'apprentissage (FD002, LSTM de référence, 3 époques, fenêtres paresseuses) : 8 processus 14 600 fenêtres/s au total, 12 processus 18 900, 15 processus 19 400 ; environ 0,63 Go par processus dans les trois cas. `scripts/nuit_04h.sh` utilise donc 12 processus par défaut (97 % du débit de 15), réglable avec `AERO_PROCESSUS` (ex. 8 pour moins de chaleur et de bruit, environ 25 % plus lent).
  - Essai rapide terminé (`AERO_ESSAI_RAPIDE=1`, 2 époques par réseau, chiffres sans valeur) : FD001 et FD002, toutes les étapes vont au bout, sans saturer la mémoire. La reprise marche : relancé, le script saute les études Optuna complètes et les étapes dont le fichier existe.
- **Impact** : avec 12 processus, environ 8 Go au lieu d'environ 20 Go pour 15 avant. Le vrai calcul de nuit (`bash scripts/nuit_04h.sh`) est prêt à être lancé : fin de FD001 environ 4 h, FD002 environ 11 h, donc sur deux nuits.

### 2026-11-06 — Optuna des réseaux étendu à FD002

- **Qui** : Matisse (avec Claude Code)
- **Date** : 2026-11-06
- **Quoi** : l'optimisation Optuna des réseaux (entrée suivante) est étendue à FD002. Rien n'est encore calculé sur FD002.
  - Changement de règle, écrit avant tout calcul : jusqu'ici, les réglages des réseaux choisis sur FD001 étaient appliqués tels quels à FD002–FD004 (entrée du 2026-11-05, et règle des fichiers `03b_*`). Désormais, chaque sous-jeu a sa propre étude Optuna sur ses propres plis par moteur, avec les mêmes étapes et les mêmes règles de décision que FD001 (confirmation sur 3 découpages neufs, gain d'au moins 0,3 cycle, Wilcoxon p < 0,05). La référence de chaque sous-jeu reste la configuration du 03b retenue sur FD001. La règle est recopiée dans le code (`TEXTE_REGLE_SOUS_JEUX`) et dans l'étude Optuna.
  - Seule différence avec FD001 : k (écart à l'état initial) reste strictement plus petit que le plus court moteur test du sous-jeu. FD002 : 21 cycles, donc k = 20 seulement (FD001 : 20, 25 ou 30).
  - `src/optimisation_reseaux.py` et `scripts/optimiser_reseaux.py` : option `--sous-jeu` (FD001 par défaut). FD002 à FD004 sont préparés par `src/sous_jeux.py`, comme au 03b. Fichiers et études séparés : `04h_*` et étude `04h_lstm` pour FD001 (noms inchangés, l'étude en cours reprend), `04h_fd002_*` et `04h_fd002_lstm` pour FD002. Un seul jeu de fenêtres en mémoire par processus (FD002 avec 50 cycles : environ 0,5 Go par jeu).
  - L'essai rapide (`AERO_ESSAI_RAPIDE=1`) a maintenant sa propre base Optuna dans `sorties/essai_rapide/`. Avant, il écrivait ses essais dans la vraie base.
  - `scripts/nuit_04h.sh` : FD001 puis FD002. Une étape dont le fichier de résultats existe est sautée, et une étude Optuna complète ne relance aucun essai : la même commande reprend la nuit suivante là où elle s'était arrêtée. Quand le CNN a fini, ses 4 cœurs passent au LSTM. `scripts/arreter_04h.sh` arrête seulement les processus de l'optimisation.
- **Pourquoi** : demande de Matisse. Appliquer les réglages de FD001 aux autres sous-jeux sans les y chercher n'est pas la méthode du projet (règle 11 d'`AGENTS.md`). FD003 et FD004 restent pour l'instant avec les réglages de FD001, et sont présentés comme tels.
- **Impact** :
  - Durées estimées à partir des essais déjà faits : fin de FD001 environ 4 h, FD002 environ 10 h 30. FD002 se termine donc la deuxième nuit (2026-11-07).
  - `sorties/predictions/` et la démo ne changent pas tant que les résultats `final` n'ont pas été relus.
  - Essai rapide du 2026-11-06 au soir (`AERO_ESSAI_RAPIDE=1 bash scripts/nuit_04h.sh`, 2 époques par réseau, chiffres sans valeur) : toutes les étapes de FD001 vont au bout (optuna, confirmer, ablation, ensemble, final). Sur FD002 : optuna, confirmer LSTM et confirmer CNN vont au bout. **L'essai a été arrêté pendant l'ablation de FD002** : la mémoire vive de la machine (31 Go, une partie déjà prise par d'autres programmes) ne suivait pas avec 15 processus sur FD002. Ablation, ensemble et final de FD002 ne sont donc **pas testés**. Avant de lancer FD002 en vrai, réduire le nombre de processus de ses étapes, ou fermer les autres programmes. Le vrai calcul de nuit n'a pas été lancé.

### 2026-11-06 — Optuna sur le LSTM et le CNN 1D (FD001, en cours)

- **Qui** : Matisse (avec Claude Code)
- **Date** : 2026-11-06
- **Quoi** : optimisation des hyperparamètres du LSTM et du CNN 1D avec Optuna, **en cours** (branche `phase-04/optuna-reseaux`). Aucun modèle n'est retenu et le test n'a pas été lu.
  - `src/optimisation_reseaux.py` : validation croisée d'une configuration de réseau pli par pli (mêmes plis et même séparation d'arrêt précoce que le 03b), pente glissante (famille F2 du 04e, vérifiée contre `np.polyfit`), deux architectures supplémentaires (CNN puis LSTM, LSTM avec attention), espaces de recherche Optuna.
  - `scripts/optimiser_reseaux.py` : étapes `verifier`, `nettoyer`, `optuna`, `confirmer`, `ablation`, `ensemble`, `final`, avec les règles de décision écrites dans le code avant les calculs. `scripts/nuit_04h.sh` enchaîne toutes les étapes (7 à 9 h estimées, tout le processeur).
  - Contrôle : la configuration de référence du 03b redonne le même RMSE de validation (8,9421) et les mêmes prédictions hors-pli à 5e-5 près.
  - État des études au 2026-11-06, 9 h 36 (RMSE de validation sur les plis 42, choisi sur ces mêmes plis donc optimiste, non confirmé) : LSTM 28 essais utiles sur 100, meilleur 7,60 (référence 8,94) ; CNN 1D 64 sur 80, meilleur 8,21 (référence 10,15).
  - Correction en cours de route : chaque processus fixe maintenant `OMP_NUM_THREADS`, `MKL_NUM_THREADS` et `OPENBLAS_NUM_THREADS` à 1. Avant, des travailleurs utilisaient jusqu'à 1,6 cœur et les 15 saturaient la machine.
- **Pourquoi** : les réglages des réseaux étaient fixés à l'avance, jamais optimisés (manque listé dans l'entrée du 2026-11-05, règle 11 d'`AGENTS.md`), et la méthode de choix des features du 04e n'avait pas été appliquée aux réseaux ; demande de Matisse.
- **Impact** :
  - Rien ne change dans `sorties/predictions/` ni dans la démo tant que l'étape `final` n'a pas tourné et que ses résultats n'ont pas été relus.
  - Le calcul occupe tout le processeur : il est lancé la nuit (`bash scripts/nuit_04h.sh`, journal `sorties/resultats/04h_nuit.log`).
  - Notebook 04h et résultats finaux pas encore écrits.

### 2026-11-05 — Démo : modèles à arbres finaux

- **Qui** : Matisse (avec Claude Code)
- **Date** : 2026-11-05
- **Quoi** : la démo utilise les modèles à arbres finaux de la Phase 4 (branche `phase-04/modele-final-demo`).
  - `src/config_modele.py` : 3 configurations finales (`CONFIG_XGBOOST_FINAL`, `CONFIG_LIGHTGBM_FINAL`, `CONFIG_RANDOM_FOREST_FINAL`), réglages Optuna recopiés du notebook 04g.
  - `scripts/generer_predictions_arbres.py` : la règle de choix des features de Benjamin est inchangée (réglages de base). Ensuite, si `ecart_initial_20` est retenu, le modèle final est réentraîné avec les réglages finaux (validation croisée refaite sur les mêmes plis). C'est lui qui écrit les fichiers de `sorties/predictions/`. Les résultats `05_*` (notebook 05) restent ceux des réglages de base.
  - `src/inference.py` : la démo lit la même configuration finale (`configuration_version_retenue`).
  - XGBoost final sur 1 cœur : avec `subsample` < 1, XGBoost ne construit pas les mêmes arbres selon le nombre de cœurs (FD001, machine à 16 cœurs : RMSE test 12,70 avec tous les cœurs, comme au 04g ; 12,73 avec 1 ; 12,78 avec 4). Sur 1 cœur, le résultat ne dépend plus de la machine.
  - Test FD001, contrôlé par le script (RMSE / score NASA / surestimés de plus de 20 cycles) : XGBoost 12,73 / 223,5 / 2 (avant : 13,10 / 237,0 / 2) ; LightGBM 12,76 / 234,6 / 3 (avant : 13,40 / 283,6) ; Random Forest 13,14 / 243,3 / 3 (avant : 14,27 / 301,9). Les chiffres à réglages de base (19,70 et 13,10) sont toujours retrouvés.
  - Tests anti-triche sur FD001 : 7 réussis, 2 sautés (LSTM et CNN 1D non entraînés sur cette machine au moment du test).
  - `app/vues/prediction_en_direct.py` : pendant la lecture, la courbe n'apparaissait qu'à la fin quand le calcul d'une image dépassait l'intervalle de 0,5 s (constaté avec XGBoost, LightGBM et Random Forest : environ 1 s pour 6 cycles). Streamlit relançait l'image avant que la courbe, dessinée en dernier, soit envoyée. Désormais, chaque image dessine d'abord (cartes et courbe), puis calcule les cycles suivants, dans un budget de 0,3 s (`BUDGET_CALCUL_S`). Vérifié avec ces 3 modèles et le processeur à 100 % : courbe en direct, au même cycle que les cartes. Sur une machine lente, la lecture ralentit au lieu de perdre la courbe.
- **Pourquoi** : la démo affichait des modèles à arbres avec les réglages de base, alors que le modèle final de la Phase 4 (04f) utilise les réglages Optuna.
- **Impact** :
  - FD002–FD004 régénérés avec les réglages finaux, mais pas encore relus. Ces réglages ont été cherchés sur FD001 seulement.
  - Le notebook 05 et l'entrée suivante citent encore les modèles à arbres de la démo à réglages de base (13,10 pour XGBoost).
  - Sur 100 moteurs test, l'écart avec le LSTM (12,29) n'est pas significatif (protocole du notebook 05, section G).

### 2026-11-05 — Démo Streamlit du prototype

- **Qui** : Benjamin (avec Claude Code)
- **Date** : 2026-11-05
- **Quoi** : démo Streamlit du prototype (branche `phase-04/demo-ben`, empilée sur `phase-04/modeles-ben`). Lancement : `streamlit run app/tableau_de_bord.py`.
  - Page « Tester un moteur » : un moteur de `test_FD00X.txt` est tiré au hasard (ou choisi) ; à chaque cycle reçu, XGBoost, LSTM et CNN 1D (au choix parmi les 6 modèles) prédisent la RUL **en direct** (`src/inference.py`) à partir des seuls cycles 1..t ; la vraie RUL (`RUL_FD00X.txt`) reste cachée jusqu'à la révélation. Règle d'alerte dans `src/alertes.py`.
  - Page « Comparer les modèles » : RMSE, score NASA, surestimés de plus de 20 cycles et alertes manquées au test, pour les 6 modèles et les 4 sous-jeux (fichiers de `sorties/predictions/`).
  - Les modèles de la démo sont les versions retenues de la PR des modèles (avec `ecartinit_20`) ; le nom affiché « XGBoost Phase 3 » devient « XGBoost » (version à 31 variables).
  - Section J du notebook 05 : la même démo dans le notebook (moteur mystère, prédiction en direct, révélation), puis lancement et arrêt de l'application.
  - Tests anti-triche (a) et (b) : la prédiction en direct au cycle t est égale aux prédictions précalculées pour les 24 couples sous-jeu × modèle (écart maximal 1,4e-14), et modifier les cycles après t ne la change pas. 28 tests pytest au total.
- **Pourquoi** : soutenance 2 (12/11) : montrer un prototype qui prédit la RUL de moteurs que les modèles n'ont jamais vus.
- **Impact** : `scripts/preparer_demo.py` régénère les prédictions et les modèles (non versionnés). Sur FD004, le CNN 1D reste proposé dans la démo malgré sa dégradation au test (voir l'entrée suivante).

### 2026-11-05 — Variable `ecartinit_20` sur les 6 modèles, protocole d'évaluation et littérature

- **Qui** : Benjamin (avec Claude Code)
- **Date** : 2026-11-05
- **Quoi** : la variable `ecartinit_20` de Matisse (notebook 04e : écart de chaque capteur lissé à sa moyenne sur les 20 premiers cycles du moteur) est ajoutée à `src/features.py` et `src/sous_jeux.py` (option `ecart_initial`, causale, calculée moteur par moteur ; un test le vérifie), puis testée sur nos 6 modèles. Le notebook 05 est réorganisé : partie 1 inchangée (sections A à E, 46 variables), section F (améliorations testées), G (protocole d'évaluation, FD001), H (littérature), I (synthèse).
  - Reproduction du XGBoost de Matisse : validation croisée 9,27 (moyenne de 3 graines de plis), test FD001 13,10 / score NASA 237,0, identiques au notebook 04e.
  - Règle de décision écrite avant les résultats : une amélioration est gardée si le RMSE de validation croisée FD001 baisse d'au moins 0,3 cycle et si le test de Wilcoxon apparié par moteur donne p < 0,05 ; le choix fait sur FD001 est appliqué tel quel à FD002–FD004.
  - Retenue pour les 5 modèles qui apprennent (la référence naïve n'utilise pas les variables). RMSE de validation croisée FD001, avant → après : Random Forest 16,96 → 10,19 ; XGBoost 16,87 → 9,36 ; LightGBM 16,85 → 9,35 ; LSTM 13,70 → 8,94 ; CNN 1D 16,96 → 10,05 (p ≤ 2,2e-13 pour chacun).
  - Fenêtre de 50 cycles pour les réseaux (idée tirée de DAST et STAGNN) : non retenue (CNN 1D : p = 0,41 ; LSTM avec la variable : 9,32 contre 8,94).
  - Test officiel FD001 (RMSE contre la RUL brute / contre la RUL plafonnée à 125, score NASA) : LSTM 12,29 / 11,37 / 228,7 (3 graines : 12,28 ± 0,25 / 11,26 ± 0,43) ; CNN 1D 12,74 / 12,38 / 346,4 ; XGBoost 13,10 / 11,88 / 237,0 ; LightGBM 13,40 / 12,13 / 283,6 ; Random Forest 14,27 / 13,03 / 301,9 ; référence naïve 43,07 / 41,94.
  - Test FD002 / FD003 / FD004, RMSE contre la RUL brute (contre la RUL plafonnée) : LSTM 22,57 (11,05) / 13,40 (11,93) / 23,55 (10,95) ; XGBoost 23,90 (11,47) / 12,24 (10,52) / 24,76 (12,57).
  - Protocole (FD001) : 5 baselines (moyenne, médiane, constante 125, aléatoire, Ridge), métriques de régression en validation et au test, erreur par tranche de RUL, classification pour N = 15, 30 et 50, comparaisons appariées des 21 paires de modèles. Les 5 modèles qui apprennent battent les baselines en RMSE, MAE, R² et score NASA ; sur les 100 moteurs test, aucune paire de modèles appris n'est significativement différente.
  - Littérature (H) : DCNN (Li et al., 2018) 12,61 sur FD001 avec les mêmes réglages (fenêtre 30, plafond 125) ; DAST (2022) 11,43 ; STAGNN (2023) 11,50. Les articles ne disent pas s'ils plafonnent la RUL vraie du test ; leurs chiffres FD002 et FD004, sous le plancher calculé pour une RUL brute, le suggèrent. Notre comparaison utilise donc la RMSE plafonnée, comme hypothèse.
  - `scripts/entrainer_deep_learning.py` : la mesure CPU contre MPS, qui faisait planter le script (code 139), ne se lance plus qu'avec l'option `--mesurer-appareils`.
  - Correction de l'entrée précédente : la branche comptait 27 tests pytest, et non 219 (nombre d'avant l'allègement) ; elle en compte maintenant 28, dont 3 dans la PR des modèles.
- **Pourquoi** : reprendre l'amélioration de Matisse sur tous nos modèles et appliquer le protocole d'évaluation (règle 11 d'`AGENTS.md`) avant la soutenance 2 ; demande de Benjamin.
- **Impact** :
  - Les versions retenues remplacent les versions de base dans `sorties/predictions/` ; la clé de fichier `xgboost_phase3` désigne maintenant la version à 31 variables (nom gardé pour le format des fichiers).
  - Limites : choix faits sur FD001 seulement ; hyperparamètres non réoptimisés pour les 31 variables ; un seul entraînement par modèle (3 graines pour les réseaux sur FD001) contre 10 dans les articles ; scalers min-max appris sur tout le train avant la validation croisée (écart à la règle 1.3 du protocole, sans information de cible).
  - Sur FD004, le CNN 1D retenu se dégrade au test (score NASA 446 374, 55 moteurs sur 248 surestimés de plus de 20 cycles) ; la validation croisée (11,60) ne le laissait pas prévoir. Constat fait sur le test, qui ne change pas le choix.
  - Encore manquant au protocole : optimisation des hyperparamètres des réseaux, courbes de validation et d'apprentissage complètes, protocole complet sur FD002–FD004.

### 2026-11-04 — Intégration de `main` et allègement de la branche

- **Qui** : Benjamin (avec Claude Code)
- **Date** : 2026-11-04
- **Quoi** : intégration de `main` (Phase 4 de Matisse) dans la branche `phase-04/prototype-ben`, puis allègement de la branche. Aucun fichier de Matisse n'est modifié.
  - Retirés, pour ne pas doubler ni contredire la Phase 4 de Matisse (notebook 04d : réglages XGBoost conservés) : la recherche Optuna (`src/optimisation.py`, `scripts/lancer_optuna.py`, résultats `04_optuna_*`), le modèle « XGBoost retenu » (alpha = 2,26) et l'ADR qui le proposait (`docs/decisions/0007-modele-retenu-penalite-surestimations.md`, jamais acceptée, dont le numéro était déjà pris par l'ADR-0007 de Matisse). L'expérience de pénalité des surestimations reste dans le notebook 05, sans modèle retenu.
  - Démo : application Streamlit réduite à 2 pages. « Tester un moteur » tire un moteur de test au hasard ; XGBoost Phase 3, LSTM et CNN 1D prédisent la RUL en direct à partir des seuls cycles reçus ; la vraie RUL reste cachée jusqu'à la révélation. « Comparer les modèles » montre un tableau et un graphique. La démo en ligne de commande (`src/demo.py`) est supprimée ; la règle d'alerte est déplacée dans `src/alertes.py`. Un test vérifie que l'inférence en direct est égale aux prédictions précalculées pour les 12 couples sous-jeu × modèle (écart maximal 1,4e-14).
  - Renommages, pour ne pas confondre nos notebooks avec les notebooks 04a–04e de Matisse : `notebooks/04b_sous_jeux_fd002_fd004.ipynb` devient [`notebooks/01b_sous_jeux_fd002_fd004.ipynb`](../notebooks/01b_sous_jeux_fd002_fd004.ipynb) (figures et résultats `04b_*` → `01b_*`), et `notebooks/04_comparaison_evaluation.ipynb` devient [`notebooks/05_comparaison_evaluation.ipynb`](../notebooks/05_comparaison_evaluation.ipynb) (`04_*` → `05_*`). Les liens des deux entrées suivantes pointent vers les anciens noms.
  - Chiffres inchangés après réexécution des notebooks 01b, 03b et 05 : XGBoost Phase 3 (CV 16,87 ; test 19,70 / score NASA 852,9), LSTM (CV 13,70 ; test 13,86 / 293,2), CNN 1D (CV 16,96 ; test 21,27 / 762,9). 219 tests pytest.
- **Pourquoi** : éviter les doublons et les contradictions avec la Phase 4 de Matisse fusionnée sur `main` ; une démo plus simple pour la soutenance (demande de Benjamin).
- **Impact** :
  - Nos comparaisons utilisent le XGBoost de la Phase 3 (test 19,70). Le XGBoost de Matisse avec la variable « écart à l'état initial » (notebook 04e : RMSE de validation 9,27, test 13,10) n'a pas été comparé au LSTM (test 13,86) moteur par moteur.
  - Le protocole d'évaluation (règle 11) n'est pas appliqué en entier au LSTM, au CNN 1D ni aux résultats par sous-jeu. Il manque : les baselines médiane, constante 125, aléatoire et Ridge ; les métriques de classification pour plusieurs N ; les courbes d'apprentissage ; l'optimisation des hyperparamètres des réseaux ; l'ablation de leurs entrées (dont la variable de Matisse).

### 2026-11-04 — Extension du prototype (FD002–FD004, deep learning, démo)
- **Qui** : Benjamin (avec Claude Code)
- **Quoi** : extension du prototype, sur la même branche.
  - Sous-jeux FD002–FD004 ([`notebooks/04b_sous_jeux_fd002_fd004.ipynb`](../notebooks/04b_sous_jeux_fd002_fd004.ipynb), `src/sous_jeux.py`) : 6 régimes opératoires pour FD002/FD004 (K-Means appris sur le train, silhouette 0,9997), un MinMaxScaler par régime, puis lissage et features par moteur ; FD001 reste identique bit à bit. Normalisation par régime en validation croisée : RMSE 16,41 contre 19,48 sans (FD002, XGBoost Phase 3). Test, XGBoost Phase 3 : FD002 27,23, FD003 18,09, FD004 28,24 ; 22 % (FD002) et 27 % (FD004) des moteurs test ont une RUL vraie supérieure au plafond de 125. Le XGBoost retenu (réglé sur FD001) est significativement moins bon que la Phase 3 sur le test de FD002 et FD004 (p ≤ 0,0002).
  - Deep learning ([`notebooks/03b_deep_learning_lstm_cnn.ipynb`](../notebooks/03b_deep_learning_lstm_cnn.ipynb), `src/deep_learning.py`) : fenêtres de 30 cycles, LSTM 2×64 et CNN 1D, réglages fixés à l'avance (non optimisés). FD001, mêmes 5 plis : RMSE CV LSTM 13,70, CNN 16,96, XGBoost Phase 3 16,87. Test FD001 : LSTM 13,86 / score NASA 293,2 / 8 surestimés / 0 alerte manquée, contre 19,70 / 852,9 / 9 / 7 pour XGBoost Phase 3 (p = 0,0009 sur l'erreur absolue par moteur) ; 3 graines : 14,22 ± 0,30. CNN 1D non différent de XGBoost (p ≥ 0,14) et instable selon la graine (17,3 à 21,3). Test LSTM sur FD002 / FD003 / FD004 : 25,45 / 16,80 / 26,14 (meilleur que XGBoost Phase 3 sur FD002 et FD004, non significatif sur FD003). Recalcul indépendant du test LSTM FD001 : mêmes chiffres ; fenêtres mélangées entre moteurs : RMSE 56,4.
  - Démo interactive Streamlit (`app/`, lancement : `streamlit run app/tableau_de_bord.py`), alimentée par un format commun de fichiers de prédictions (`src/contrat_predictions.py`, fichiers régénérés par `scripts/preparer_demo.py`, non versionnés).
  - Blocage et erreur de segmentation quand torch et xgboost tournent dans le même processus (trois copies d'OpenMP) : corrigé par torch sur 1 cœur et l'import de xgboost avant torch (`src/deep_learning.py`).
- **Pourquoi** : cahier des charges, section 5 (LSTM obligatoire, CNN 1D recommandé, FD001 puis FD002–FD004) ; soutenance 2 : démontrer un prototype qui prédit sur des moteurs jamais vus.
- **Impact** : l'ADR-0007 (proposée) ne couvre que le choix parmi les modèles à arbres ; le choix entre arbres et LSTM reste à décider par l'équipe. `docs/0X_*.md` et résumés Notion non rédigés (phases non validées).

### 2026-11-04 — Premier prototype de Phase 4 (comparaison de modèles, Optuna)
- **Qui** : Benjamin (avec Claude Code)
- **Quoi** : premier prototype de Phase 4, sur la branche `phase-04/prototype-ben` (non mergée).
  - Socle réutilisable : `src/features.py`, `src/evaluation.py`, `src/modeles.py`, `src/config_modele.py`, `src/optimisation.py` ; dans `src/preprocessing.py`, `DATA_DIR` devient un chemin absolu (seule modification). 125 tests pytest.
  - [`notebooks/04_comparaison_evaluation.ipynb`](../notebooks/04_comparaison_evaluation.ipynb), mêmes 5 plis `GroupKFold` pour tous les modèles : RMSE CV XGBoost 16,87, LightGBM 16,85, Random Forest 16,96, référence naïve 41,69. Écarts entre les trois modèles non significatifs (tests appariés sur le RMSE par moteur, 100 moteurs, p ≥ 0,58). Entraînement 5 plis : 3,9 s (XGBoost) contre 28 s (Random Forest).
  - Pénalité des surestimations placée dans l'apprentissage (perte quadratique asymétrique `alpha`, perte de forme NASA `h_sous`/`h_sur`), puis Optuna multi-objectif (RMSE CV, score NASA CV) : 160 essais, 3 études, TPE graine 42, règle de choix fixée avant le test (RMSE ≤ 1,05 × meilleur RMSE du front de Pareto, puis plus petit score NASA). Retenu : XGBoost, perte quadratique, `alpha` = 2,26 ([`docs/resultats/04_config_retenue.json`](resultats/04_config_retenue.json)). En CV, par rapport à la Phase 3 : surestimations > 20 cycles −36 % (p < 0,0001), score NASA −29 % (p = 0,007), RMSE 17,28 contre 16,87 (différence par moteur non significative, p = 0,67).
  - Test officiel, évalué une fois : retenu RMSE 19,73, score NASA 675,4, 9/100 surestimés de plus de 20 cycles, 7 alertes manquées (seuil 30 cycles, hypothèse de travail) ; Phase 3 : 19,70 / 852,9 / 9 / 7. Différence non significative sur 100 moteurs (p = 0,94) ; l'écart de score NASA vient presque entièrement du moteur 37 (sans lui : 646,1 contre 629,3). Résultat non concluant sur le test.
  - Perte de forme NASA une fois convergée (1 000 arbres) : moins bonne que la Phase 3 en CV, en RMSE comme en score NASA ; son bon résultat à 200 arbres tient à l'arrêt précoce, pas à la perte seule.
  - Changer les constantes 13/10 du score d'évaluation ne change aucune prédiction (score 852,9 → 0,13 pour un facteur 1 000) : ces constantes restent fixes pour évaluer.
  - Démo `python -m src.demo` (`entrainer`, `moteur`, `flotte`, `benchmark`) : préparation + prédiction d'un moteur en 35,6 ms au 95e centile (cible 200 ms), mesuré sous charge.
- **Pourquoi** : soutenance 2 du 12/11 : justifier XGBoost face à des alternatives mesurées (demande du coach, réunion du 29/10), tester la piste « pénaliser davantage les surestimations » et Optuna, disposer d'un prototype démontrable.
- **Impact** : dans le venv du projet (Python 3.13), la validation croisée de la Phase 3 donne 16,87 / 29 782,6 (doc de Phase 3 : 16,86 / 29 729,3, obtenus dans un autre environnement ; notebook 03 réexécuté identique au code refactoré). Chiffres de test de la Phase 3 inchangés. Branche à comparer avec le travail de Matisse avant toute intégration. `docs/04_evaluation.md` non rédigé (phase non validée).

### 2026-11-04 — Protocole d'évaluation obligatoire pour tout modèle
- **Qui** : Matisse (avec Claude Code)
- **Date** : 2026-11-04
- **Quoi** : création de [`docs/protocole_evaluation.md`](protocole_evaluation.md), un protocole d'évaluation obligatoire pour tout modèle. Il couvre le cadre de validation par moteur, 5 baselines (moyenne, médiane, constante 125, aléatoire, Ridge), 10 métriques de régression plus l'erreur par tranche de RUL, les métriques de classification sur la question « panne dans les N prochains cycles ? » (matrice de confusion, accuracy, précision, rappel, F1, ROC/AUC, précision-rappel), les diagnostics sur-apprentissage/sous-apprentissage (dont les courbes d'apprentissage), l'optimisation des hyperparamètres (grid search, random search, Optuna), les features justifiées par ablation, la comparaison statistique et le contenu minimal d'un rapport. Ajout de la règle 11 dans `AGENTS.md`, de l'[ADR-0007](decisions/0007-protocole-evaluation-obligatoire.md), et mise à jour de la ligne Phase 4 du tableau des phases. Aucun code ni aucun chiffre modifié.
- **Pourquoi** : revue de Matisse à partir du cours *Introduction to ML* (« Code cours 2 »). La Phase 3 n'avait été présentée qu'avec le RMSE et le score NASA, sans métrique de classification, sans optimisation des hyperparamètres, avec une seule famille de features et un seul modèle. Le diagnostic biais/variance et la baseline moyenne n'existaient que parce que la question avait été posée le 2026-10-21. Détail vérifié dans la section 0 du protocole.
- **Impact** : le XGBoost de la Phase 3 n'est pas considéré comme complètement évalué tant que le protocole ne lui a pas été appliqué. Les chiffres déjà publiés (RMSE test 19.70, score NASA 852.9) restent valables. Le seuil N de la lecture classification, le même que le seuil d'alerte resté ouvert depuis la Soutenance 1, reste à décider en équipe.

### 2026-10-21 — Situer l'erreur sur une échelle (Phase 3)
- **Qui** : Matisse (avec Claude Code)
- **Quoi** : nouvelle section 13 de `notebooks/03_modelisation.ipynb` qui situe l'erreur par rapport à quatre références, sans réentraîner le modèle : échelle de la cible (RUL 0-125), baseline qui prédit toujours la RUL moyenne du train, R² (formule vérifiée contre `sklearn.metrics.r2_score` sur un pli) et besoin du sujet. Résultats : baseline 41.69 en validation croisée et 43.07 sur le test officiel, contre 16.86 et 19.70 pour XGBoost (2.5 et 2.2 fois mieux) ; R² train 0.931, validation 0.835, test 0.775 ; RMSE validation = 13.5 % de la plage de la cible. Le cahier des charges ne fixe aucune erreur acceptable chiffrée ; la seule référence chiffrée sourcée, le benchmark exploratoire du README (RMSE test 18.97), **n'est pas battue** (19.70, écart non testé statistiquement). Figure `docs/figures/03_echelle_erreur_baseline.png`, nouvelle section 7 et limite ajoutée dans `docs/03_modelisation.md`.
- **Pourquoi** : question de Matisse après le diagnostic biais/variance : « si les deux erreurs sont grandes, comment on sait qu'elles sont grandes pour notre sujet ? » — sans référence, les mots « haute » et « basse » du diagnostic n'avaient pas de sens chiffré. Des RMSE publiés en littérature pour FD001 ont été cherchés mais n'ont pu être confirmés que par des sources secondaires : non utilisés, conformément à la règle sur les chiffres non sourcés.
- **Impact** : aucun chiffre déjà publié ne change. Si la Soutenance 1 présente le XGBoost comme meilleur que le benchmark exploratoire, c'est faux sur le RMSE test (19.70 contre 18.97) ; seul le score NASA est meilleur (852.9 contre 1 053.9, non testé statistiquement). Pour la Phase 4 : fixer une erreur acceptable sourcée et mesurer l'erreur près de la panne.

### 2026-10-21 — Diagnostic biais/variance (Phase 3)
- **Qui** : Matisse (avec Claude Code)
- **Quoi** : ajout d'un diagnostic biais/variance à `notebooks/03_modelisation.ipynb`. La boucle de validation croisée (section 7) calcule maintenant aussi le RMSE sur les moteurs appris (RMSE train), sans changer les chiffres de validation ; nouvelle section 12 avec trois mesures, toutes en validation croisée (jeu de test non utilisé) : écart train/validation par pli, courbe de validation sur `max_depth` (1 à 10) et sur `n_estimators` (1 à 1000 arbres, via `eval_set`). Résultats : RMSE train moyen **10.94** contre **16.86** en validation (écart +5.91, présent sur les 5 plis) ; le RMSE de validation le plus bas est obtenu avec `max_depth=3` (16.55) ou 90 arbres (16.72), soit au plus 0.31 de mieux que les réglages actuels, contre un écart-type d'environ 1.2 entre plis. Lecture : sur-apprentissage modéré mais peu coûteux ; la validation ne descend sous 16.5 avec aucun réglage, ce plancher vient des features ou du bruit des données, pas d'un modèle trop complexe. Réglages inchangés ; test officiel inchangé (19.70 / 852.9 / 9 sur 100). Trois figures ajoutées (`docs/figures/03_biais_variance_*.png`), `docs/03_modelisation.md` complété (nouvelle section 6, colonne RMSE train dans le tableau des plis, limites mises à jour), ligne Phase 3 du `README.md` mise à jour.
- **Pourquoi** : question de Matisse (« est-ce qu'on a comparé l'erreur train et l'erreur test pour voir si on a sur-appris ou sous-appris ? ») : vérification faite, aucune erreur train n'avait jamais été calculée en Phase 3, donc aucun diagnostic biais/variance n'était possible.
- **Impact** : aucun chiffre déjà publié ne change. Piste ajoutée pour la Phase 4 : tester un modèle plus simple (`max_depth=3` ou moins d'arbres) avec un test statistique. La Soutenance 1 peut citer l'écart train/validation (10.94 vs 16.86) si la question du sur-apprentissage est posée.

### 2026-09-30 — Ajout de time_cycles aux features (test d'ablation)
- **Qui** : Matisse (avec Claude Code)
- **Quoi** : ajout de `time_cycles` (numéro du cycle en cours) aux features de `notebooks/03_modelisation.ipynb` : 46 features au lieu de 45. Décision prise sur un test d'ablation (nouvelle section 11 du notebook, validation croisée uniquement, jeu de test non utilisé) comparant 7 jeux de features : les colonnes glissantes (moyennes, écarts-types) n'améliorent pas le RMSE de façon mesurable (18.54 à 18.73 pour 15, 30 et 45 features, contre un bruit d'environ 1.3 entre plis), alors que `time_cycles` fait passer le RMSE moyen de 18.64 à 16.86 et le score NASA moyen de 62 279 à 29 729. « 15 capteurs + `time_cycles` » (16 features) fait aussi bien (16.95) — simplification possible en Phase 4. Nouveaux résultats sur le test officiel : **RMSE 19.70, score NASA 852.9, 9/100 moteurs surestimés de plus de 20 cycles** (contre 18.91 / 1082.6 / 14/100 à 45 features, commit `a97a82d`) — résultat mixte, non concluant avec 100 points. `docs/03_modelisation.md` réécrit en conséquence, figure de validation regénérée ; deux compréhensions de liste du notebook remplacées par des boucles explicites (règle du projet).
- **Pourquoi** : question soulevée par Matisse pendant la revue du code (« pourquoi 45 colonnes ? ») — vérifiée par un test plutôt que supposée. L'hypothèse initiale (« l'âge du moteur dit peu de chose sur la RUL ») s'est révélée fausse en validation croisée.
- **Impact** : les chiffres de la Soutenance 1 doivent être mis à jour (slides 1, 4, 5 et 10 : RMSE 19,70 ; score NASA 852,9 ; 9/100 surestimés ; RMSE moyen de validation croisée 16,86 ; 46 features). Les chiffres des Phases 1 et 2 sont inchangés.

### 2026-09-28 — Ajout de shuffle=True à GroupKFold
- **Qui** : Matisse (avec Claude Code)
- **Quoi** : ajout de `shuffle=True, random_state=42` à `GroupKFold` dans `notebooks/03_modelisation.ipynb` — par défaut, `GroupKFold` répartit les moteurs entre les plis selon leur ordre d'apparition dans le fichier, pas au hasard, ce qui aurait pu biaiser la validation croisée si cet ordre correspondait à autre chose qu'un simple ordre de simulation (jamais vérifié jusqu'ici). Testé : avec le mélange, le RMSE moyen passe de 18.61 à 18.64 — écart négligeable, donc pas de biais cachés détecté, mais le mélange (graine fixe) est conservé pour ne plus avoir à le supposer. `docs/03_modelisation.md` mis à jour avec les nouveaux chiffres de plis (16.58/20.22/18.09/18.34/19.97, moyenne 18.64).
- **Pourquoi** : question soulevée par Matisse en apprenant le fonctionnement de `GroupKFold` (répartition déterministe par défaut, pas aléatoire) — vérifiée plutôt que supposée sans risque, conformément à la règle du projet.
- **Impact** : slide 4 de la Soutenance 1 (qui cite "RMSE moyen 18,17", puis "18,61") doit être mise à jour avec **18,64**. Les chiffres du test officiel (18,91 / 1082,6 / 14 sur 100) restent inchangés.

### 2026-09-28 — Ré-exécution dans le venv du projet (RMSE CV 18.17 → 18.61)
- **Qui** : Matisse (avec Claude Code)
- **Quoi** : ré-exécution complète de `notebooks/03_modelisation.ipynb` dans un environnement différent (venv du projet, Python 3.14, après résolution d'un problème d'installation de xgboost). Les chiffres de validation croisée (`GroupKFold`) ont légèrement changé : RMSE moyen 18.61 (contre 18.17 lors du premier run), malgré `random_state=42` fixé — XGBoost parallélise la construction des arbres, et l'ordre des calculs flottants en parallèle n'est pas garanti identique d'un environnement à l'autre. Le modèle final et son évaluation sur le test officiel sont restés strictement identiques (RMSE 18.91, score NASA 1082.6, 14/100).
- **Pourquoi** : le notebook a été réexécuté pour vérifier qu'il tournait bien dans l'environnement réellement utilisé (le premier run avait eu lieu dans un environnement différent) — règle du projet : ne jamais committer un notebook dont l'exécution n'a pas été revérifiée.
- **Impact** : chiffre remplacé par celui de l'entrée du dessus (18.64) après l'ajout du `shuffle`.

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
