# Plan de réorganisation des notebooks

- **Rédigé le** : 2026-11-06, par Matisse (avec Claude Code), à partir de la discussion du même jour.
- **Statut** : plan validé dans la discussion, **aucune étape exécutée**. Ce fichier sert de feuille de route pendant toute la réorganisation : on coche les cases au fur et à mesure, on note les écarts dans le journal (section 13). On ne réécrit pas une décision déjà actée : si on change d'avis, on ajoute une ligne au journal qui le dit.
- **C'est une réorganisation, pas une refonte complète** (voir section 3, principe 0). Le mot « refonte » reste dans le nom du fichier et de la branche (`refonte/notebooks`), choisis avant cette précision.
- 3 des 13 notebooks sont de Benjamin (01b, 03b, 05) ; il en garde ses propres copies. Rien n'est poussé sur GitHub sans demande explicite de Matisse.

---

## 1. Le problème

Au 2026-11-06, `notebooks/` contient 13 notebooks (01, 01b, 02, 03, 03b, 04a à 04g, 05), et un 04h est prévu pour les résultats Optuna des réseaux. Chacun répond à une question qu'on s'est posée **dans l'ordre où on a travaillé**, pas dans l'ordre où un lecteur la pose.

Constats :

1. **Trop de notebooks, trop de cellules.** Estimation : environ la moitié des cellules ne contribue pas aux résultats finaux (pistes abandonnées, essais intermédiaires, vérifications de passage).
2. **Aucun chemin clair pour le jury.** À la question « montrez-moi comment vous avez testé XGBoost ou le LSTM sur FD002 », il n'existe pas de notebook unique qui y répond : les résultats FD002 sont dispersés entre 01b, 03b et 05.
3. **Le protocole d'évaluation ([`protocole_evaluation.md`](protocole_evaluation.md)) n'a été déroulé en entier que sur FD001.** FD002 à FD004 ont des prédictions et des métriques de test, mais ni baselines complètes, ni classification, ni comparaisons statistiques, et tous les réglages ont été choisis sur FD001 seulement (voir `STATUS.md`, entrées du 2026-11-05).
4. **Sur GitHub, aucun graphique n'est visible** : le hook `nbstripout` (`.pre-commit-config.yaml`) efface les sorties de tous les notebooks à chaque commit.

## 2. Ce qu'on veut

L'objectif se lit à travers la démo Streamlit (`app/`) :

- **La démo est la vitrine.** Elle montre n'importe quel sous-jeu (FD001 à FD004), un moteur de test choisi ou tiré au hasard, une prédiction de RUL calculée en direct cycle par cycle par 6 modèles, et la vraie RUL révélée à la fin. C'est ce qu'on présente au jury.
- **Les notebooks sont la preuve.** Le jury dira tôt ou tard : « prenez ce moteur, prenez FD002, prouvez-moi ce résultat ». Il faut alors ouvrir **un** notebook, celui du sous-jeu, et retrouver le même chiffre et la même courbe, **recalculés à partir des données brutes** devant lui. C'est ce qui montre que la démo n'est pas un panneau visuel qui « fait du bon visuel ».
- **Des graphes à chaque étape où ils aident.** Un graphe dit en deux secondes où on en est, au jury comme à nous quand on reprend le code. Chaque figure a pour titre **la question à laquelle elle répond** (ex. « Où le modèle se trompe-t-il ? »), pas seulement le nom du graphe.
- **Le parcours du projet, à part.** Les avant/après (ex. RMSE test FD001 de XGBoost : 19,70 → 13,10 → 12,73) montrent comment on a progressé. On les garde dans un notebook dédié, séparé de la présentation des résultats.

Les 6 modèles (liste de `src/donnees_demo.py`, `MODELES_CONNUS`) : 5 qui apprennent (Random Forest, XGBoost, LightGBM, LSTM, CNN 1D) et 1 baseline (`reference_naive`, qui prédit la moyenne).

## 3. Principes

0. **Réorganiser, pas refaire.** On ne réécrit pas le travail : les cellules gardées sont déplacées avec leur code, leurs commentaires et leurs explications, dans la même forme (markdown explicatif, code détaillé et commenté). On fait seulement trois choses :
   - **retirer ce qui n'est jamais réutilisé** : ce qui est calculé dans un notebook et nulle part ailleurs ensuite, ni par la démo, ni par les preuves de la démo, ni par une décision qu'on doit défendre (règle 11, protocole), ni par une phase à venir (ex. Phase 5, explicabilité) ;
   - **remplacer le code recopié** par l'appel à la fonction de `src/` qui fait déjà la même chose, en gardant l'explication une seule fois, là où la notion apparaît pour la première fois ;
   - **déplacer** les cellules gardées dans la nouvelle arborescence (section 4).

   Le seul code nouveau : ce qui étend aux 4 sous-jeux ce qui n'existe que pour FD001 (`evaluer_sous_jeu`, section G du notebook 05 généralisée) et la section « Vérifier un moteur » (à partir de la section J.3 du notebook 05). En cas de doute sur une cellule, **on la garde**.
1. **Le calcul vit dans `src/`, une seule fois.** Les notebooks appellent des fonctions, ils ne contiennent pas de logique dupliquée. Le traitement des données se fait à la source (`src/`), paramétré par sous-jeu, pas recopié dans chaque dossier : 4 copies du même code finiraient par diverger sans que personne s'en aperçoive.
2. **Les 4 notebooks de résultats sont générés depuis un seul modèle**, jamais modifiés à la main. Même plan, mêmes cellules, mêmes graphes ; seul le paramètre `SOUS_JEU` change.
3. **La preuve n'est jamais circulaire.** La section « Vérifier un moteur » repart de `data/test_FD00X.txt` et ne lit jamais les fichiers de `sorties/predictions/` qu'affiche la démo. Elle compare ensuite son résultat à ces fichiers, pour montrer qu'ils sont identiques.
4. **Démo et notebooks lisent les mêmes résultats.** Les chiffres affichés dans Streamlit sont ceux que les notebooks de résultats calculent et vérifient.
5. **Rien n'est supprimé avant d'être remplacé et vérifié** (règle 9 d'`AGENTS.md`) : un ancien notebook ne sort de `main` que quand les nouveaux redonnent ses chiffres.
6. **Rien n'est perdu** : avant la suppression, un tag git (`archive-phases-1-4`) fige l'état actuel. Tout reste consultable sur GitHub à ce tag.
7. **Densité de commentaires et style** : ceux d'`AGENTS.md` (règles 2 et 3 : docstrings, commentaires en français bloc par bloc, pas de code condensé).
8. **Pas de nombre de cellules visé.** La taille d'un notebook résulte du tri (section 8.2), on ne compresse pas les explications pour atteindre un chiffre. (Une première version de ce plan visait 15 à 25 cellules ; retiré, voir le journal.)

## 4. Structure cible

```
notebooks/
├── 01_donnees.ipynb                 ← commun aux 4 sous-jeux : préparation des données
├── 02_choix_du_modele.ipynb         ← commun : tout ce qui est DÉCIDÉ, sur FD001
├── resultats/
│   ├── _modele.ipynb                ← le seul notebook de résultats écrit à la main
│   ├── FD001.ipynb  ┐
│   ├── FD002.ipynb  │  générés depuis _modele.ipynb (SOUS_JEU = "FD00X"),
│   ├── FD003.ipynb  │  jamais modifiés à la main, sorties gardées
│   └── FD004.ipynb  ┘
└── evolution.ipynb                  ← le parcours du projet, avant / après

src/
├── (existant) preprocessing.py, sous_jeux.py, features.py, modeles.py, config_modele.py,
│   deep_learning.py, inference.py, evaluation.py, contrat_predictions.py, ...
└── evaluation_sous_jeu.py           ← nouveau : evaluer_sous_jeu(), verifier_moteur()

scripts/
└── generer_notebooks_resultats.py   ← nouveau : exécute _modele.ipynb pour les 4 sous-jeux

app/                                 ← la démo, inchangée dans un premier temps
```

On passe de 13 notebooks à 4 écrits à la main (`01`, `02`, `_modele`, `evolution`), plus 4 générés.

## 5. Contenu détaillé de chaque notebook

### 5.1 `01_donnees.ipynb` : préparation des 4 sous-jeux

Question du jury : « Vos données, vous les avez préparées comment ? »

| Section | Contenu | Figure | Origine |
|---|---|---|---|
| Les 4 sous-jeux | Tableau : régimes de vol, modes de panne, nombre de moteurs train / test | — | 00_etat_de_l_art, 01 |
| Capteurs retirés | Capteurs constants sur FD001, même vérification sur les 3 autres | — | 01 |
| Régimes de vol | FD002 et FD004 : 6 groupes de réglages (K-Means appris sur le train) | Régimes opératoires FD002 / FD004 | 01b |
| Normalisation | Pourquoi par régime : brut → méthode FD001 → normalisation par régime | 3 panneaux (1) (2) (3) | 01b |
| Lissage | Avant / après sur un capteur d'un moteur | Lissage moteur 1, sensor_11 | 01 |
| Contrôle | FD001 identique bit à bit à l'ancienne préparation | — | 01b |

### 5.2 `02_choix_du_modele.ipynb` : ce qui est décidé, sur FD001

Question du jury : « Pourquoi ce modèle, ces variables et ces réglages ? » Chaque section se termine par **la décision** et le chiffre qui la justifie, conformément à la règle 11 (features justifiées par ablation, hyperparamètres optimisés).

| Section | Contenu | Figure | Origine |
|---|---|---|---|
| La cible | RUL plafonnée à 125, pourquoi | RUL brute vs plafonnée, moteur 1 | 02 |
| Fenêtre des réseaux | Ce que voit le LSTM / CNN 1D (30 cycles) | Schéma de la fenêtre glissante | 03b |
| Features | Ablation : gain de chaque jeu ajouté aux 16 features de référence | Ablation, gain RMSE | 04e |
| Écart à l'état initial | Gain sur les 5 modèles qui apprennent | Avant / après par modèle | 05 (F) |
| Hyperparamètres des arbres | Optuna : meilleur RMSE selon le budget, importance des réglages | Historique + importance | 04d |
| Hyperparamètres des réseaux | Optuna LSTM et CNN 1D (nuit du 2026-11-06) | Même format que les arbres | 04h (à écrire) |
| Confirmation | Candidat contre réglages actuels, sur plis neufs | Confirmation plis neufs | 04d |
| Biais / variance | Train vs validation par pli | Train vs val par pli | 03 |
| Plus de données ? | Courbe d'apprentissage | Courbe d'apprentissage | 04c, 03b |
| Décision finale | Tableau récapitulatif des réglages retenus pour les 6 modèles | — | `src/config_modele.py` |

### 5.3 `resultats/FD00X.ipynb` : le protocole complet, par sous-jeu

Question du jury : « Montrez-moi comment vous avez testé XGBoost ou le LSTM sur FD00X, et prouvez-moi ce résultat. » Les 4 notebooks ont **exactement** ce plan.

| Section | Contenu | Figures |
|---|---|---|
| 1. Données du sous-jeu | Régimes, nombre de moteurs, part des moteurs test dont la RUL vraie dépasse 125 | aucune |
| 2. Les 6 modèles | Réglages utilisés (lus dans `src/config_modele.py`), entraînement ou chargement (voir 6.2) | aucune |
| 3. Validation croisée | Mêmes plis `GroupKFold` pour tous les modèles, RMSE moyen ± écart-type | aucune |
| 4. Test officiel | RMSE, MAE, R², score NASA, contre les 5 baselines (moyenne, médiane, constante 125, aléatoire, Ridge) ; comparaison à la littérature quand elle existe | RUL prédite vs vraie (une case par modèle) ; résidus selon la RUL vraie ; barres RMSE et score NASA contre baselines |
| 5. Classification | Alertes pour N = 15, 30 et 50 cycles | Une figure à deux panneaux, ROC et précision-rappel, 6 modèles superposés (AUC et précision moyenne en légende) ; matrices de confusion, une par modèle, à un seul N |
| 6. Comparaison des modèles | Erreur par moteur, appariée, test de Wilcoxon entre paires de modèles | Erreur par moteur avec p-valeurs |
| 7. Vérifier un moteur | `MOTEUR = 42` : prédiction recalculée cycle par cycle depuis `test_FD00X.txt`, puis comparée aux fichiers de la démo | RUL prédite cycle par cycle (la même courbe que la démo) |

Note sur les ROC : au test, peu de moteurs sont proches de la panne (de l'ordre de 10 sur 100 pour N = 15), donc la courbe est en escalier et l'AUC paraît souvent très bonne. Le notebook 04b l'a mesuré en validation : l'AUC vaut 0,977 à 0,996 pour les 3 approches et ne les départage pas, alors que la courbe précision-rappel (ligne du hasard = proportion de positifs) montre le coût en précision de chaque niveau de rappel. D'où les deux panneaux côte à côte, et une phrase sous la figure qui le dit.

### 5.4 `evolution.ipynb` : le parcours du projet

Question du jury : « Comment êtes-vous arrivés là ? » Pas utilisé comme preuve, mais comme récit.

**Vue d'ensemble** : l'« escalier » du RMSE FD001, étape par étape, **validation croisée et test côte à côte** (nos décisions ont été prises sur la validation croisée ; montrer le test seul ferait croire qu'on a choisi en le regardant).

**Une section par étape**, toujours au même format : ce qu'on a changé (**une seule chose**), pourquoi, la figure avant/après, le lien vers le notebook d'origine au tag d'archive.

| Étape | Changement | Chiffres connus (à recalculer, voir 9.2) |
|---|---|---|
| 1 | Benchmark exploratoire (Random Forest) → XGBoost Phase 3 | Test FD001 : 18,97 → 19,70 |
| 2 | Normalisation par régime (FD002) | CV FD002 : 19,48 → 16,41 |
| 3 | Arbres → LSTM (réglages fixés à l'avance) | Test FD001 : 19,70 → 13,86 |
| 4 | Écart à l'état initial (`ecartinit_20`), 5 modèles | CV XGBoost : 16,87 → 9,36 ; test 19,70 → 13,10 |
| 5 | Optuna sur les arbres | Test XGBoost : 13,10 → 12,73 (1 cœur) |
| 6 | Optuna sur les réseaux | À venir (nuit du 2026-11-06) |

Source des chiffres : `STATUS.md`, entrées du 2026-11-04 et du 2026-11-05.

## 6. Points techniques

### 6.1 Génération des 4 notebooks de résultats

Deux options, décision à prendre au moment de l'étape C (section 10) :

- **papermill** : exécute un notebook avec des paramètres (cellule marquée `parameters`). Standard, simple. Coût : une dépendance de plus dans `requirements.txt`.
- **`jupyter nbconvert --execute`** avec une variable d'environnement `SOUS_JEU`. Aucune dépendance de plus, mais le paramètre est moins visible dans le notebook.

Dans les deux cas, `scripts/generer_notebooks_resultats.py` lance les 4 exécutions et écrit `FD001.ipynb` à `FD004.ipynb`.

### 6.2 Réentraîner ou charger les modèles

Les arbres se réentraînent en quelques secondes, le LSTM et le CNN 1D en plusieurs minutes par sous-jeu. Proposition : un drapeau en tête de `_modele.ipynb`.

- `ENTRAINER = False` (par défaut) : les modèles sauvegardés sont chargés et leurs réglages affichés. « Prouve-moi ce moteur » prend quelques secondes.
- `ENTRAINER = True` : tout est réentraîné et le notebook vérifie qu'il retrouve les mêmes chiffres. « Prouve-moi que ce modèle vient bien de ce code » prend quelques minutes.

### 6.3 Graphiques visibles sur GitHub

`nbstripout` efface les sorties de tous les notebooks. Pour les notebooks de la refonte, on garde leurs sorties : soit via la métadonnée `"keep_output": true` de chaque notebook (gérée par nbstripout), soit en excluant ces chemins dans `.pre-commit-config.yaml`. Les autres notebooks restent nettoyés. Coût accepté : des fichiers `.ipynb` plus lourds dans git (images encodées).

### 6.4 Figures de `docs/`

Les docs de phase (`docs/01_eda_preparation.md`, `03_modelisation.md`, etc.) pointent vers `docs/figures/*.png`. **Ces fichiers ne doivent pas bouger** : les nouveaux notebooks peuvent écrire leurs propres figures, mais pas écraser ni déplacer celles-ci sans mettre à jour les liens.

### 6.5 Fichiers de prédictions incomplets en local

Au 2026-11-06, `sorties/predictions/` ne contient le LSTM et le CNN 1D que pour FD001 (FD002 à FD004 : arbres et baseline seulement). Il faudra régénérer (`scripts/preparer_demo.py`) avant de générer les notebooks de résultats.

## 7. Inventaire des figures

### 7.1 Gardées

| Figure | Notebook d'origine | Destination |
|---|---|---|
| Lissage avant / après | 01 | `01_donnees` |
| Régimes FD002 / FD004 | 01b | `01_donnees` |
| Brut → méthode FD001 → normalisation par régime | 01b | `01_donnees` |
| RUL brute vs plafonnée | 02 | `02_choix_du_modele` |
| Fenêtre glissante | 03b | `02_choix_du_modele` |
| Ablation, gain RMSE | 04e | `02_choix_du_modele` |
| Écart à l'état initial, gain sur 5 modèles | 05 | `02_choix_du_modele` et `evolution` |
| Optuna : historique, importance des réglages | 04d | `02_choix_du_modele` |
| Confirmation sur plis neufs | 04d | `02_choix_du_modele` |
| Train vs validation par pli | 03 | `02_choix_du_modele` |
| Courbe d'apprentissage | 04c, 03b | `02_choix_du_modele` |
| RUL prédite vs vraie | 04a | `resultats/` section 4 |
| Résidus selon la RUL vraie | 04a | `resultats/` section 4 |
| Barres RMSE / score NASA contre baselines | 01b, 05 | `resultats/` section 4 |
| Courbes ROC et précision-rappel (deux panneaux) | 04b | `resultats/` section 5 (une figure par sous-jeu) |
| Importances (gain, permutation) du jeu retenu | 04e | `02_choix_du_modele` (réutilisées en Phase 5, explicabilité) |
| Matrices de confusion | 04b | `resultats/` section 5 (une par sous-jeu) |
| Erreur par moteur, comparaisons appariées | 05 | `resultats/` section 6 |
| RUL prédite cycle par cycle | 03b | `resultats/` section 7 |

### 7.2 Retirées (consultables au tag `archive-phases-1-4`)

| Figure | Notebook | Raison |
|---|---|---|
| Courbes de validation `max_depth` / `n_estimators` | 03 | Remplacées par Optuna, qui explore plus |
| Erreur comparée à la baseline et au benchmark | 03 | Remplacée par les barres du test (section 4) |
| XGBoost Phase 3 sur les 4 sous-jeux | 01b | Ancien modèle, remplacé par `resultats/` |
| RMSE vs score NASA par essai | 04d | Vérification intermédiaire, pas une décision |
| Pénalisation de la surestimation | 05 | Piste non retenue, déjà dans `docs/04_pistes_non_retenues.md` |
| Temps d'entraînement | 05 | Diagnostic sans rôle dans les résultats finaux |
| Écart vs longueur de la trajectoire observée | 05 | Idem |
| Test, 100 moteurs | 04f | Doublon de la section 4 |

## 8. Correspondance ancien → nouveau

### 8.1 Vue d'ensemble

| Ancien notebook | Auteur | Va dans |
|---|---|---|
| `01_eda_preparation` | Matisse | `01_donnees` |
| `01b_sous_jeux_fd002_fd004` | Benjamin | `01_donnees` (préparation) ; `evolution` (étape 2) |
| `02_target_engineering` | Matisse | `02_choix_du_modele` (la cible) |
| `03_modelisation` | Matisse | `02_choix_du_modele` (biais / variance) ; `evolution` (étape 1) |
| `03b_deep_learning_lstm_cnn` | Benjamin | `02_choix_du_modele` (fenêtre, courbes) ; `resultats/` (section 7) ; `evolution` (étape 3) |
| `04a_metriques_regression` | Matisse | `resultats/` section 4 |
| `04b_metriques_classification` | Matisse | `resultats/` section 5 |
| `04c_diagnostic_apprentissage` | Matisse | `02_choix_du_modele` |
| `04d_hyperparametres` | Matisse | `02_choix_du_modele` ; `evolution` (étape 5) |
| `04e_feature_engineering` | Matisse | `02_choix_du_modele` ; `evolution` (étape 4) |
| `04f_modele_final` | Matisse | `resultats/` |
| `04g_comparaison_modeles` | Matisse | `resultats/` section 6 |
| `04h` (à écrire) | Matisse | `02_choix_du_modele` ; `evolution` (étape 6) |
| `05_comparaison_evaluation` | Benjamin | `02_choix_du_modele` (F) ; `resultats/` (G, H, J.3) ; `evolution` (B, D) |

### 8.2 Tri détaillé, notebook par notebook

Critère (principe 0) : on retire ce qui est calculé dans un notebook et **jamais réutilisé ensuite** (ni démo, ni preuve de la démo, ni décision à défendre, ni phase à venir). Ce tri a été fait à partir des titres, des fonctions définies et des fichiers écrits par chaque cellule, et des conclusions lues en entier du 04a (section 9) et du 04b (sections 10 et 11). **Avant de retirer une cellule, on la lit en entier** pour vérifier qu'aucun chiffre ou objet qu'elle crée n'est utilisé plus bas.

#### Partout : le code recopié

`load_fd001`, `ajouter_features_glissantes`, `score_nasa`, `calculer_metriques` (et équivalents) sont redéfinis dans 02, 03, 04a, 04b, 04c, 04d, 04e, 04f et 04g alors qu'ils existent dans `src/` (`preprocessing.py`, `features.py`, `evaluation.py`). L'explication se garde **une fois**, là où la notion apparaît pour la première fois dans la nouvelle arborescence ; ailleurs, appel à `src/`.

#### Trois règles qui corrigent une lecture trop rapide du critère

1. **Métriques non réutilisées (MedAE, erreur max, MAPE, MSE…)** : rien ne les réutilise après le 04a, mais le protocole (§3) exige de les calculer une fois et d'écrire pourquoi on les garde ou les écarte. `src/evaluation.py` (`metriques_regression`) les calcule déjà toutes en un appel. On remplace donc le calcul à la main du 04a (`calculer_metriques`, ~60 lignes) par **cet appel et le tableau « gardée / écartée »** du 04a section 9, puis on n'affiche partout ailleurs que les métriques retenues.
2. **Courbes précision-rappel** : gardées (voir 5.3 et le journal), alors qu'une première version de ce plan les retirait.
3. **Baselines** : la démo n'affiche que la référence naïve (moyenne), mais le protocole (§2) en exige 5 ; elles restent dans les notebooks de preuve.

#### `01_eda_preparation` → `01_donnees`

- **Garder** : capteurs constants (statistiques, liste, suppression, vérification) ; comparaison min-max / standard et sa figure (justifie le min-max) ; lissage et sa vérification visuelle.
- **Retirer** : `load_fd001` recopié (→ `src/preprocessing`).

#### `01b_sous_jeux_fd002_fd004` → `01_donnees` + `evolution` + `resultats/`

- **Garder** : A (FD001 identique bit à bit : preuve) ; B (ce qui change dans FD002-FD004) ; C (régimes, K-Means, inertie et silhouette : justifie 6 groupes ; figure des six régimes) ; D.1 à D.3 (pourquoi normaliser par régime, avant de lisser ; figure en 3 panneaux) ; D.4 (gain mesuré en validation croisée → `evolution`, étape 2) ; E (capteurs retirés par sous-jeu) ; F.2 (pourquoi le RMSE test de FD002 et FD004 est élevé : part des RUL vraies au-dessus de 125 → section 1 de `resultats/FD00X`).
- **Retirer** : F.1 et F.3 (résultats de l'ancien XGBoost Phase 3 par sous-jeu, remplacés par `resultats/`) ; F.4 (temps de calcul) ; G (fichiers disponibles pour l'application).

#### `02_target_engineering` → `02_choix_du_modele`

- **Garder** : tout (calcul de la RUL, vérification, plafonnement, preuve chiffrée de la stabilité en début de vie, figure brute vs plafonnée).
- **Retirer** : `load_fd001` et les noms de colonnes recopiés.

#### `03_modelisation` → `02_choix_du_modele` + `evolution`

- **Garder** : explication des features glissantes et du cas du premier cycle (ce que `src/features.py` implémente, qui cite ce notebook) ; explication du score NASA et de son contrôle ; explication de GroupKFold ; diagnostic biais / variance train vs validation par pli ; test officiel Phase 3 (→ `evolution`, étape 1 : 19,70).
- **Retirer** : courbes de validation `max_depth` et `n_estimators` (remplacées par Optuna) ; section 13 « quatre références » (remplacée par les 5 baselines) ; ablation à 7 jeux de la section 11 (remplacée par celle du 04e ; une ligne dans `evolution`).
- Les PNG de `docs/figures/03_*` restent en place : `docs/03_modelisation.md` les cite.

#### `03b_deep_learning_lstm_cnn` → `02_choix_du_modele` + `resultats/`

- **Garder** : A.1 (règles fixées avant les résultats) ; notions liées à nos réseaux : fenêtre glissante (B.1), remplissage en début de vie (B.2), contrôle « jamais un autre moteur, jamais le futur » (B.3), tenseur 3D (B.4), pourquoi 30 cycles (B.5), normalisation de la cible (B.8), LSTM (B.10), convolution 1D et CNN (B.11, B.12), arrêt précoce (B.14) ; C (faut-il donner `time_cycles` aux réseaux : une décision) ; D (validation croisée et tests appariés arbres contre réseaux, courbes d'apprentissage) ; F.4 (robustesse à la graine) ; F.7 (trajectoires de moteurs test → section 7 de `resultats/`).
- **Condenser en texte, sans code de démonstration** : neurone (B.6), perte et gradient (B.7), époque et lot (B.9), dropout (B.13).
- **Retirer** : B.15 (démonstration de 3 époques) ; B.16 (reproductibilité CPU / MPS) ; E (perte asymétrique α = 2,26, non retenue) ; F.1 à F.3, F.5, F.6 et G (tests avec les anciens réglages, remplacés par `resultats/` régénérés) ; H (synthèse).

#### `04a_metriques_regression` → `resultats/` section 4 + `02_choix_du_modele`

- **Garder** : les 5 baselines ; figure prédit vs vrai ; figure des résidus selon la RUL vraie ; erreur par tranche de RUL (dans `src/evaluation.py` ; c'est elle qui montre le biais de +10,30 cycles entre 31 et 60 en validation) ; tableaux train vs validation résumés (diagnostic de sur-apprentissage, règle 11) ; tableau « gardée / écartée » (section 9) et limites (section 11).
- **Retirer** : section 1 (reconstruction de la Phase 3, 109 lignes recopiées) ; calcul à la main des métriques (→ `metriques_regression`) ; détail pli par pli de Ridge ; section 8 (XGBoost contre Ridge « description, pas un test », remplacée par Wilcoxon).

#### `04b_metriques_classification` → `resultats/` section 5 + `02_choix_du_modele`

- **Garder** : étiquettes « RUL ≤ N » et proportion de positifs ; approche 1 (alerte dérivée de la régression, celle de `src/alertes.py` et de la démo) ; choix du seuil sur la validation (`choisir_seuil_alerte` dans `src/`) ; « pourquoi l'accuracy trompe » (court) ; matrices de confusion, ROC, précision-rappel ; tableau « gardée / écartée » (section 10).
- **Garder sous forme de résumé dans `02_choix_du_modele`** : approche 2 (classifieurs dédiés, baseline aléatoire) et section 11 (test apparié). C'est ce qui justifie le choix de la régression, et le test montre que la `LogisticRegression` fait peut-être mieux à N = 30 (AP 0,961 contre 0,951, p = 0,003) : question possible du jury.
- **À décider (Matisse)** : section 8, « combien de cycles d'avance avant la panne ». Rien ne la réutilise, mais elle est parlante pour un jury.
- **Retirer** : chargement et features recopiés.

#### `04c_diagnostic_apprentissage` → `02_choix_du_modele`

- **Garder** : tout (courbe d'apprentissage, quantification de la pente, conclusion, limites : règle 11).
- **Retirer** : chargement, features et outils de mesure recopiés (~110 lignes).

#### `04d_hyperparametres` → `02_choix_du_modele` + `evolution`

- **Garder** : règle de décision écrite avant les recherches ; espace de recherche ; explication et lancement d'Optuna ; figure de l'importance des hyperparamètres ; confirmation sur plis neufs et sa figure.
- **Retirer** : la comparaison Grid / Random / Optuna (elle choisit seulement la méthode de recherche ; sa conclusion « on ne change pas les réglages » a été remplacée par le 04f ; une ligne dans `evolution`) ; figure RMSE vs score NASA par essai ; test officiel des réglages inchangés.

#### `04e_feature_engineering` → `02_choix_du_modele` + `evolution`

- **Garder** : features candidates et leur explication ; tests de causalité et de reproduction (`tests/test_anti_triche.py` renvoie à ce notebook) ; cadre d'évaluation et règle de décision ; hypothèses écrites avant ; tableau et figure d'ablation ; sélection ; importances gain et permutation (Phase 5) ; test officiel du jeu retenu (→ `evolution`, étape 4 : 13,10) ; conclusions.
- **Retirer** : section 9 (variantes testées après lecture des résultats, non retenues, jamais évaluées sur le test ; une ligne pour leur résultat).

#### `04f_modele_final` et `04g_comparaison_modeles` → `02_choix_du_modele` + `resultats/`

- **Garder** : tout. Ce sont les sources des `CONFIG_*_FINAL` utilisées par la démo (Optuna sur 31 features, confirmation sur 10 découpages, seuil d'alerte, comparaison des 3 modèles à arbres).
- **Déplacer** : la comparaison 46 contre 31 features → `evolution`.
- **Retirer** : chargement et features recopiés ; figure « Test, 100 moteurs » du 04f (doublon de la section 4 de `resultats/`).

#### `05_comparaison_evaluation` → `resultats/` + `02_choix_du_modele` + `evolution`

- **Garder** : A (règles fixées avant les résultats) ; B et D (réglages de base, validation croisée et test → `evolution`, « avant ») ; F (écart à l'état initial sur les 6 modèles → `02` et `evolution` ; fenêtre de 50 cycles rejetée, en une ligne avec sa p-valeur) ; G (protocole complet sur FD001 → **modèle de la structure de tous les `resultats/FD00X`**) ; H (littérature → section 4 de `resultats/`) ; J.3 et J.4 (prédiction en direct d'un moteur mystère, révélation → **base de la section 7 « Vérifier un moteur »**).
- **À confirmer (Matisse)** : E (hypothèses sur les moteurs surestimés de plus de 20 cycles, figure « écart vs RUL vraie / vs longueur de trajectoire »). Rien ne la réutilise.
- **Retirer** : C (pénalité de la surestimation, piste non retenue) ; I (synthèse, remplacée par `resultats/`) ; J.1, J.2, J.5, J.6 (réglages, chargement et lancement de Streamlit depuis le notebook).

## 9. Contraintes et précautions

### 9.1 Calendrier

- **Soutenance 2 : 12/11**, six jours après la rédaction de ce plan. Tant que la nouvelle structure n'est pas vérifiée, l'ancienne reste intacte : si le temps manque, on présente avec la démo et les anciens notebooks, et la refonte se termine après.
- **Nuit du 2026-11-06** : `scripts/nuit_04h.sh` occupe tout le processeur. Aucune génération de notebooks pendant ce calcul.

### 9.2 Non-régression

Avant toute suppression, les nouveaux notebooks doivent redonner les chiffres des anciens, pour chaque modèle et chaque sous-jeu. Repères (test FD001, `STATUS.md`) : XGBoost 12,73 (réglages finaux, 1 cœur), LightGBM 12,76, Random Forest 13,14, LSTM 12,29, CNN 1D 12,74, référence naïve 43,07. Les chiffres des réseaux changeront si Optuna fait retenir de nouveaux réglages : on compare alors au résultat de l'étape `final` de `scripts/optimiser_reseaux.py`.

Dans `evolution.ipynb`, chaque chiffre est **recalculé ou lu dans `sorties/resultats/`**, jamais recopié à la main.

### 9.3 Git

- Le travail Optuna non committé (`src/optimisation_reseaux.py`, `scripts/optimiser_reseaux.py`, `scripts/nuit_04h.sh`, `notebooks/04g_comparaison_modeles.ipynb`, `docs/STATUS.md`) est committé d'abord sur `phase-04/optuna-reseaux`.
- La refonte part sur une nouvelle branche `refonte/notebooks`, depuis `main`.
- Rien n'est poussé sur GitHub sans demande explicite de Matisse. Le passage par une PR (ADR-0005) se décide au moment de pousser.

### 9.4 Documentation

- `docs/0X_*.md` ne sont pas réécrits par ce plan (règle 4).
- `STATUS.md` reçoit une entrée à chaque étape significative, dans le même commit.
- La suppression des anciens notebooks mérite une ADR (elle remplace l'organisation actuelle), avec ses coûts : historique des notebooks consultable seulement au tag, liens existants vers les anciens notebooks à mettre à jour (`STATUS.md`, docs de phase, README).

## 10. Plan d'exécution

### Étape A : préparer le terrain (maintenant)

- [x] Committer le travail Optuna en cours sur `phase-04/optuna-reseaux`
- [x] Créer la branche `refonte/notebooks` depuis `main`
- [x] Tri détaillé notebook par notebook (section 8.2), validé par Matisse

### Étape B : le socle `src/` (ne dépend pas d'Optuna)

- [ ] `src/evaluation_sous_jeu.py` : `evaluer_sous_jeu(sous_jeu)` (sections 3 à 6 du protocole) et `verifier_moteur(sous_jeu, moteur)` (section 7, depuis les données brutes)
- [ ] Fonctions de figures (matplotlib), une par graphe de la section 7.1, titrées par leur question
- [ ] Tests pytest : `verifier_moteur` égal aux fichiers de la démo ; `evaluer_sous_jeu("FD001")` redonne les chiffres de 9.2

### Étape C : les notebooks qui ne dépendent pas d'Optuna

- [ ] Choisir papermill ou nbconvert (6.1)
- [ ] `notebooks/resultats/_modele.ipynb`, sections 1 à 7
- [ ] `scripts/generer_notebooks_resultats.py`
- [ ] Régénérer `sorties/predictions/` pour les 4 sous-jeux (6.5), en journée
- [ ] Générer FD001 à FD004 avec les modèles actuels, relire les 4
- [ ] `notebooks/01_donnees.ipynb`
- [ ] Garder les sorties de ces notebooks dans git (6.3)

### Étape D : après la relecture des résultats Optuna

- [ ] Relire `sorties/resultats/04h_nuit.log` et les résultats de l'étape `final`
- [ ] `notebooks/02_choix_du_modele.ipynb`
- [ ] Regénérer FD001 à FD004 avec les réglages finaux, contrôle de non-régression (9.2)
- [ ] `notebooks/evolution.ipynb`

### Étape E : nettoyage

- [ ] Tag `archive-phases-1-4` sur `main` avant suppression
- [ ] ADR de la nouvelle organisation des notebooks
- [ ] Retirer les 13 anciens notebooks de `notebooks/` (leur contenu gardé est déjà déplacé, section 8.2)
- [ ] Mettre à jour les liens (README, `STATUS.md` par une nouvelle entrée, docs de phase, `AGENTS.md` section « Structure du repo »)
- [ ] Pousser et ouvrir une PR vers `main` : seulement sur demande de Matisse

### Étape F : optionnel, après la soutenance

- [ ] Page Streamlit « Comment on a évalué » (baselines et protocole par sous-jeu, lus dans `sorties/`)
- [ ] Lien depuis la démo vers la section 7 du notebook du sous-jeu affiché

## 11. Critères de fin

La refonte est terminée quand :

1. `notebooks/` ne contient que `01_donnees`, `02_choix_du_modele`, `resultats/` (modèle + 4 générés) et `evolution`.
2. Pour n'importe quel moteur de n'importe quel sous-jeu affiché dans la démo, la section 7 du notebook correspondant redonne la même prédiction depuis les données brutes, et un test pytest le vérifie.
3. Le protocole d'évaluation est déroulé en entier sur les 4 sous-jeux (règle 11).
4. Les graphiques sont visibles sur GitHub sans exécuter les notebooks.
5. Les chiffres des nouveaux notebooks égalent ceux des anciens (9.2), ou l'écart est expliqué dans `STATUS.md`.
6. Les anciens notebooks sont consultables au tag `archive-phases-1-4`, et l'ADR est acceptée.

## 12. Questions ouvertes

| Question | État |
|---|---|
| Le chiffre 12,85 cité dans la discussion : d'où vient-il ? `STATUS.md` donne 13,10 (écart initial, réglages de base) et 12,73 (réglages Optuna, 1 cœur) pour XGBoost au test FD001 | À vérifier |
| papermill ou nbconvert | À décider en étape C |
| Valeur de N pour les matrices de confusion (15, 30 ou 50) | Proposition : 30 |
| Où placer la comparaison à la littérature (section 4 de `resultats/` ou `02_choix_du_modele`) | Proposition : section 4, FD001 uniquement si les articles ne donnent pas les autres |
| Importances de features (04e) : Phase 5 ou ce plan ? | Réglé : gardées dans `02_choix_du_modele`, réutilisées en Phase 5 |
| Accord de Benjamin | Réglé : il garde ses propres copies de 01b, 03b et 05 ; rien n'est poussé sans demande de Matisse |
| 04b section 8, « combien de cycles d'avance avant la panne » : garder ? | À décider (Matisse) |
| 05 section E, hypothèses sur les moteurs surestimés : garder ? | À confirmer (Matisse) |

## 13. Journal

Une ligne par étape faite ou par écart au plan, la plus récente en bas. On n'efface jamais une ligne.

| Date | Qui | Quoi |
|---|---|---|
| 2026-11-06 | Matisse | Plan rédigé et validé dans la discussion ; aucune étape exécutée |
| 2026-11-06 | Matisse | Étape A : travail Optuna committé sur `phase-04/optuna-reseaux` (`d6b6b10`). Branche `refonte/notebooks` créée depuis `main` (`8dcbedc`), dans un dossier séparé (`../aero-predict-refonte`, worktree git) : Optuna tournait dans le dossier principal, et changer de branche y aurait retiré du disque les fichiers qu'il utilise |
| 2026-11-06 | Matisse | Écart constaté : `main` n'a pas les commits `f79b0e2` et `65b0227` de `phase-04/modele-final-demo` (configurations finales des arbres, `CONFIG_*_FINAL`, et correctif de la démo), dont la refonte a besoin. À régler avant l'étape B : fusionner `phase-04/modele-final-demo` dans `main`, ou intégrer cette branche dans `refonte/notebooks` |
| 2026-11-06 | Matisse | Écart réglé : `phase-04/modele-final-demo` fusionnée dans `refonte/notebooks` (`7d3003c`), sans conflit. `main` n'a toujours pas ces commits : ils y arriveront avec la PR de la refonte. Suite de tests pytest non relancée (Optuna occupait le processeur) |
| 2026-11-06 | Matisse | Précision : c'est une **réorganisation, pas une refonte complète** (principe 0 ajouté en section 3) ; titre du plan changé en conséquence, nom du fichier et de la branche gardés. Tri détaillé ajouté (section 8.2). Corrections de la première version : courbes précision-rappel gardées à côté des ROC (le 04b montre que l'AUC ne départage pas les approches) ; importances de features gardées (Phase 5) ; métriques non réutilisées (MedAE, erreur max, MAPE…) calculées une fois par `metriques_regression` avec leur tableau « gardée / écartée », au lieu d'être retirées (protocole §3) ; cible de 15 à 25 cellules par notebook retirée. Benjamin garde ses propres copies ; rien n'est poussé sans demande de Matisse |
| 2026-11-06 | Matisse | Correction des lignes 2 et 4 de ce journal : Optuna **ne tournait pas**. Les processus Python vus à ce moment étaient des noyaux Jupyter ouverts depuis VS Code (lancés à 9 h 36) ; la base `04h_optuna.db` n'a plus été écrite après 9 h 36. Le dossier séparé (worktree) reste utile : la branche Optuna et la branche de réorganisation restent chacune dans leur dossier. Décision : lancer `scripts/nuit_04h.sh` la nuit du 2026-11-06 ; étapes B à D après relecture de ses résultats |
