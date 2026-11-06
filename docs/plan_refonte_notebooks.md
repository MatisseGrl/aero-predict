# Plan de refonte des notebooks

- **Rédigé le** : 2026-11-06, par Matisse (avec Claude Code), à partir de la discussion du même jour.
- **Statut** : plan validé dans la discussion, **aucune étape exécutée**. Ce fichier sert de feuille de route pendant toute la refonte : on coche les cases au fur et à mesure, on note les écarts dans le journal (section 12). On ne réécrit pas une décision déjà actée : si on change d'avis, on ajoute une ligne au journal qui le dit.
- **À valider avec Benjamin** avant toute suppression : 3 des 13 notebooks sont les siens (01b, 03b, 05).

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

1. **Le calcul vit dans `src/`, une seule fois.** Les notebooks appellent des fonctions, ils ne contiennent pas de logique dupliquée. Le traitement des données se fait à la source (`src/`), paramétré par sous-jeu, pas recopié dans chaque dossier : 4 copies du même code finiraient par diverger sans que personne s'en aperçoive.
2. **Les 4 notebooks de résultats sont générés depuis un seul modèle**, jamais modifiés à la main. Même plan, mêmes cellules, mêmes graphes ; seul le paramètre `SOUS_JEU` change.
3. **La preuve n'est jamais circulaire.** La section « Vérifier un moteur » repart de `data/test_FD00X.txt` et ne lit jamais les fichiers de `sorties/predictions/` qu'affiche la démo. Elle compare ensuite son résultat à ces fichiers, pour montrer qu'ils sont identiques.
4. **Démo et notebooks lisent les mêmes résultats.** Les chiffres affichés dans Streamlit sont ceux que les notebooks de résultats calculent et vérifient.
5. **Rien n'est supprimé avant d'être remplacé et vérifié** (règle 9 d'`AGENTS.md`) : un ancien notebook ne sort de `main` que quand les nouveaux redonnent ses chiffres.
6. **Rien n'est perdu** : avant la suppression, un tag git (`archive-phases-1-4`) fige l'état actuel. Tout reste consultable sur GitHub à ce tag.
7. **Densité de commentaires et style** : ceux d'`AGENTS.md` (règles 2 et 3 : docstrings, commentaires en français bloc par bloc, pas de code condensé).
8. **Cible : 15 à 25 cellules par notebook.**

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
| 5. Classification | Alertes pour N = 15, 30 et 50 cycles | Courbes ROC, 6 modèles superposés, AUC en légende ; matrices de confusion, une par modèle, à un seul N |
| 6. Comparaison des modèles | Erreur par moteur, appariée, test de Wilcoxon entre paires de modèles | Erreur par moteur avec p-valeurs |
| 7. Vérifier un moteur | `MOTEUR = 42` : prédiction recalculée cycle par cycle depuis `test_FD00X.txt`, puis comparée aux fichiers de la démo | RUL prédite cycle par cycle (la même courbe que la démo) |

Note sur les ROC : au test, peu de moteurs sont proches de la panne (de l'ordre de 10 sur 100 pour N = 15), donc la courbe est en escalier et l'AUC paraît souvent très bonne. Une phrase sous le graphe le dit.

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
| Courbes ROC | 04b | `resultats/` section 5 (une par sous-jeu) |
| Matrices de confusion | 04b | `resultats/` section 5 (une par sous-jeu) |
| Erreur par moteur, comparaisons appariées | 05 | `resultats/` section 6 |
| RUL prédite cycle par cycle | 03b | `resultats/` section 7 |

### 7.2 Retirées (consultables au tag `archive-phases-1-4`)

| Figure | Notebook | Raison |
|---|---|---|
| Courbes précision-rappel | 04b | Double emploi avec la ROC |
| Courbes de validation `max_depth` / `n_estimators` | 03 | Remplacées par Optuna, qui explore plus |
| Erreur comparée à la baseline et au benchmark | 03 | Remplacée par les barres du test (section 4) |
| XGBoost Phase 3 sur les 4 sous-jeux | 01b | Ancien modèle, remplacé par `resultats/` |
| RMSE vs score NASA par essai | 04d | Vérification intermédiaire, pas une décision |
| Pénalisation de la surestimation | 05 | Piste non retenue, déjà dans `docs/04_pistes_non_retenues.md` |
| Temps d'entraînement | 05 | Diagnostic sans rôle dans les résultats finaux |
| Écart vs longueur de la trajectoire observée | 05 | Idem |
| Test, 100 moteurs | 04f | Doublon de la section 4 |
| Importances (gain, permutation) du jeu retenu | 04e | À rediscuter : utile pour la Phase 5 (explicabilité), pas pour ce plan |

## 8. Correspondance ancien → nouveau

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
| `05_comparaison_evaluation` | Benjamin | `02_choix_du_modele` (F) ; `resultats/` (G, H) ; section J (démo dans le notebook) retirée, la démo est Streamlit |

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
- Passage par une PR relue par Benjamin (fin de la période solo, ADR-0005).

### 9.4 Documentation

- `docs/0X_*.md` ne sont pas réécrits par ce plan (règle 4).
- `STATUS.md` reçoit une entrée à chaque étape significative, dans le même commit.
- La suppression des anciens notebooks mérite une ADR (elle remplace l'organisation actuelle), avec ses coûts : historique des notebooks consultable seulement au tag, liens existants vers les anciens notebooks à mettre à jour (`STATUS.md`, docs de phase, README).

## 10. Plan d'exécution

### Étape A : préparer le terrain (maintenant)

- [ ] Committer le travail Optuna en cours sur `phase-04/optuna-reseaux`
- [ ] Créer la branche `refonte/notebooks` depuis `main`
- [ ] Présenter ce plan à Benjamin, obtenir son accord sur le principe et sur le sort de 01b, 03b, 05

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
- [ ] Supprimer les 13 anciens notebooks (accord de Benjamin)
- [ ] Mettre à jour les liens (README, `STATUS.md` par une nouvelle entrée, docs de phase, `AGENTS.md` section « Structure du repo »)
- [ ] PR vers `main`, relue par Benjamin

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
| Importances de features (04e) : Phase 5 ou ce plan ? | À rediscuter |
| Accord de Benjamin | À obtenir (étape A) |

## 13. Journal

Une ligne par étape faite ou par écart au plan, la plus récente en bas. On n'efface jamais une ligne.

| Date | Qui | Quoi |
|---|---|---|
| 2026-11-06 | Matisse | Plan rédigé et validé dans la discussion ; aucune étape exécutée |
| 2026-11-06 | Matisse | Étape A : travail Optuna committé sur `phase-04/optuna-reseaux` (`d6b6b10`). Branche `refonte/notebooks` créée depuis `main` (`8dcbedc`), dans un dossier séparé (`../aero-predict-refonte`, worktree git) : Optuna tournait dans le dossier principal, et changer de branche y aurait retiré du disque les fichiers qu'il utilise |
| 2026-11-06 | Matisse | Écart constaté : `main` n'a pas les commits `f79b0e2` et `65b0227` de `phase-04/modele-final-demo` (configurations finales des arbres, `CONFIG_*_FINAL`, et correctif de la démo), dont la refonte a besoin. À régler avant l'étape B : fusionner `phase-04/modele-final-demo` dans `main`, ou intégrer cette branche dans `refonte/notebooks` |
