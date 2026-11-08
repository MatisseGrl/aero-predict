# Protocole d'évaluation des modèles — obligatoire

**S'applique à** : tout modèle entraîné dans ce projet, présent ou futur (XGBoost actuel, Random Forest, LightGBM, SVR, LSTM, CNN 1D, Transformer…), sur tous les sous-jeux (FD001 à FD004).
**Statut** : en vigueur depuis le 2026-11-04 ([ADR-0007](decisions/0007-protocole-evaluation-obligatoire.md)).
**Lu par** : toute personne de l'équipe et tout assistant IA (référencé dans [`AGENTS.md`](../AGENTS.md), règle 11).

Un modèle n'est pas « évalué » tant que ce protocole n'a pas été appliqué en entier. Un chiffre isolé (« RMSE 19,70 ») ne dit rien : il prend son sens seulement comparé à d'autres métriques, à des modèles de référence volontairement mauvais, et à l'erreur sur les données d'entraînement.

---

## 0. Pourquoi ce document existe

Revue du 2026-11-04 (Matisse, à partir du cours *Introduction to ML*, « Code cours 2 — Aspects pratiques du ML ») : la Phase 3 a présenté le modèle XGBoost presque uniquement avec deux chiffres, le RMSE et le score NASA. Constats, vérifiés dans le repo :

| Point | État au 2026-11-04 | Où |
|---|---|---|
| Plusieurs métriques de régression comparées | ❌ RMSE et score NASA seulement ; R² ajouté après coup | `docs/03_modelisation.md` §4, §7 |
| Comparaison à un modèle naïf | ⚠️ une seule baseline (prédire la RUL moyenne), ajoutée le 2026-10-21 après une question de Matisse | §7 |
| Diagnostic sur-apprentissage / sous-apprentissage | ⚠️ fait le 2026-10-21 (RMSE train 10,94 vs validation 16,86, courbes de validation `max_depth` et `n_estimators`), mais seulement sur le RMSE et seulement parce que la question a été posée | §6 |
| Courbe d'apprentissage (erreur selon la quantité de données) | ❌ jamais faite | — |
| Métriques de classification (matrice de confusion, précision, rappel, ROC/AUC…) | ❌ jamais faites | — |
| Optimisation des hyperparamètres (grid search, random search, Optuna) | ❌ un seul jeu de réglages, jamais cherché | §3 |
| Feature engineering justifié | ⚠️ une seule famille (moyenne et écart-type sur 5 cycles) + `time_cycles` ; l'ablation montre que les colonnes glissantes n'apportent rien de mesurable ; l'asymétrie et le kurtosis demandés par le cahier des charges (Phase 3) n'ont pas été testés | §1, §5 |
| Autres modèles classiques du cahier des charges (LightGBM, SVR, Random Forest) | ❌ non entraînés | — |

Ces manques n'ont pas été signalés à temps par l'assistant IA du projet alors que son rôle (`AGENTS.md`, « senior staff engineer ») l'exigeait. Ce protocole sert à ce que ça ne dépende plus de la mémoire de quelqu'un.

**Règle de lecture** : *tester* une métrique n'oblige pas à la *garder* dans le rapport final. Mais une métrique ne peut être écartée qu'après avoir été calculée, avec une phrase qui explique pourquoi elle n'apporte rien (ou trompe) sur ce problème.

---

## 1. Cadre de validation (le même pour tous les modèles)

1. **Validation croisée par moteur** : `GroupKFold(n_splits=5, shuffle=True, random_state=42)` avec `groups = unit_number`. Un moteur n'est jamais à la fois en entraînement et en validation (vérifié à chaque pli par un `assert`).
2. **Mêmes plis pour tous les modèles comparés** : les plis sont calculés une fois (`plis = list(gkf.split(...))`) et réutilisés. Sinon une différence de score peut venir du découpage et pas du modèle.
3. **Tout ce qui apprend quelque chose se calcule sur l'entraînement seulement** (min/max du scaler, médiane d'un imputer, clusters de régimes…), puis s'applique en `transform` sur validation et test. Dans une validation croisée, ça veut dire **refaire le `fit` à l'intérieur de chaque pli**.
4. **Le jeu de test officiel (`test_FD00X.txt` + `RUL_FD00X.txt`) ne sert jamais à choisir quoi que ce soit** : ni features, ni hyperparamètres, ni seuil. Il est utilisé une seule fois par modèle final, pour le rapport.
5. **Ce qu'on mesure n'est pas la même chose en validation et en test** (à écrire dans chaque rapport) :
   - en validation croisée, chaque ligne (chaque cycle de chaque moteur) est une prédiction, cible plafonnée à 125 ;
   - sur le test officiel, il y a **une prédiction par moteur** (son dernier cycle observé), comparée à la vraie RUL non plafonnée.
   Les deux chiffres ne sont donc pas directement comparables. Un score NASA est une **somme** : 29 729 en validation (≈ 4 000 lignes par pli) et 852,9 sur le test (100 moteurs) ne veulent pas dire que le test est « meilleur ». Toujours rapporter aussi le **score NASA moyen par prédiction**.
6. **Graines fixées** partout (`random_state=42`, `seed` d'Optuna, `torch.manual_seed`) et versions des bibliothèques notées dans le rapport.

---

## 2. Modèles de référence (baselines) obligatoires

Chaque modèle est comparé, avec exactement les mêmes plis et les mêmes métriques, à :

| Référence | Ce qu'elle prédit | Ce qu'elle permet de dire |
|---|---|---|
| `DummyRegressor(strategy="mean")` | toujours la RUL moyenne du train | plancher : un modèle qui ne fait pas nettement mieux n'a rien appris (R² ≈ 0 par construction) |
| `DummyRegressor(strategy="median")` | toujours la RUL médiane | même idée, plus robuste ; référence pour la MAE |
| Constante = plafond (125) | « le moteur est toujours en pleine santé » | montre ce que coûte un modèle optimiste, surtout sur le score NASA et les surestimations |
| Tirage aléatoire uniforme entre 0 et 125 (graine fixe) | n'importe quoi | « modèle complètement raté » : borne basse de toutes les métriques, y compris les métriques de classification (AUC ≈ 0,5) |
| Régression linéaire (`Ridge`) sur les mêmes features | relation linéaire | sépare ce qui vient des features de ce qui vient de la complexité du modèle |

Un modèle qui ne bat pas clairement `Ridge` n'apporte pas grand-chose par rapport à un modèle simple et explicable.

---

## 3. Métriques de régression : toutes calculées, sur train, validation et test

Pour chaque modèle : **train** (moyenne des plis), **validation** (valeur par pli + moyenne ± écart-type entre plis), **test officiel**.

| Métrique | `sklearn.metrics` / formule | Ce qu'elle dit ici |
|---|---|---|
| MAE | `mean_absolute_error` | erreur moyenne en cycles, directement lisible (« on se trompe de N vols en moyenne ») |
| MSE | `mean_squared_error` | punit fortement les grosses erreurs ; sert surtout au calcul du RMSE et comme fonction de perte |
| RMSE | `np.sqrt(mean_squared_error(...))` | standard de la littérature C-MAPSS : indispensable pour se comparer aux articles |
| R² | `r2_score` | part de la variation de la RUL expliquée ; 0 = aussi bon que prédire la moyenne |
| Erreur absolue médiane | `median_absolute_error` | erreur « typique », insensible aux quelques moteurs ratés ; comparée à la MAE, elle montre si quelques grosses erreurs tirent la moyenne |
| Erreur maximale | `max_error` | pire cas : en maintenance, c'est le moteur qu'on aurait laissé voler |
| Biais moyen | `np.mean(y_pred - y_true)` | positif = le modèle surestime en moyenne (dangereux), négatif = prudent |
| Score NASA (somme **et** moyenne par prédiction) | `score_nasa` de `src/evaluation.py` (définie à l'origine dans `notebooks/03_modelisation.ipynb` §6, archivé au tag `v1-avant-reorganisation`) | métrique asymétrique du domaine (Saxena et al., 2008) |
| Surestimations de plus de 20 cycles | `np.sum(y_pred - y_true > 20)` (et en %) | nombre de cas dangereux, lisible par un non-spécialiste |
| MAPE | `mean_absolute_percentage_error` | **à tester puis très probablement écarter** : la RUL tend vers 0 en fin de vie, donc la division par la vraie valeur explose (1 cycle d'erreur sur une RUL de 1 = 100 %). Documenter le chiffre et la raison du rejet |

**Erreur par tranche de RUL** (obligatoire) : MAE, RMSE et biais séparément pour RUL vraie ≤ 30, 31-60, 61-125. Le RMSE global mélange le début de vie (où le plafond 125 rend la tâche facile) et la fin de vie, qui est la seule zone où une décision de maintenance se prend.

**Moyenne des erreurs** : un tableau de synthèse donne, pour chaque modèle, la moyenne ± écart-type entre plis de chaque métrique. Un écart entre deux modèles plus petit que cet écart-type n'est pas une différence (voir §8).

---

## 4. Métriques de classification : « panne dans les N prochains cycles ? »

L'utilisateur final (contrôle maintenance) ne décide pas sur une RUL exacte : il décide **de retirer ou non un moteur**. Le problème a donc aussi une lecture de classification binaire :

- **classe positive (1)** : RUL vraie ≤ N cycles (« à retirer ») ; **classe négative (0)** : RUL > N.
- **N n'est pas encore choisi.** C'est le même chiffre que le seuil d'alerte resté ouvert depuis la Soutenance 1. Il se décide avec l'équipe. En attendant, calculer pour plusieurs valeurs (15, 30, 50) et montrer comment les résultats en dépendent.

Deux façons de produire la décision, à comparer :
1. **À partir du modèle de régression, sans nouveau modèle** : alerte si RUL prédite ≤ N. Pour la courbe ROC, le score est `-RUL prédite` (plus la RUL prédite est basse, plus le moteur est « positif »).
2. **Classifieur dédié** (`LogisticRegression`, `XGBClassifier`…) entraîné directement sur l'étiquette 0/1, avec `predict_proba` pour la ROC. Validation : `StratifiedGroupKFold` (garde la proportion de positifs dans chaque pli et sépare toujours les moteurs).

| Métrique | `sklearn.metrics` | Ce qu'elle dit ici |
|---|---|---|
| Matrice de confusion | `confusion_matrix` (lignes = vrai, colonnes = prédit) | les 4 cas : moteur retiré à raison, retiré trop tôt (faux positif = coût de maintenance), **laissé voler alors qu'il allait tomber en panne (faux négatif = le cas grave)**, laissé voler à raison |
| Accuracy | `accuracy_score` | **trompeuse ici** : les positifs sont minoritaires. Toujours afficher la proportion de positifs à côté |
| Précision | `precision_score` | parmi les moteurs retirés, combien devaient l'être |
| **Rappel** | `recall_score` | parmi les moteurs proches de la panne, combien on a détectés : **métrique prioritaire pour la sécurité** |
| F1 | `f1_score` | compromis précision / rappel, pour comparer des modèles à seuil fixé |
| Courbe ROC + AUC | `roc_curve`, `roc_auc_score` | capacité à classer les moteurs du plus au moins urgent, tous seuils confondus (0,5 = hasard, cf. baseline aléatoire) |
| Courbe précision-rappel + précision moyenne | `precision_recall_curve`, `average_precision_score` | plus informative que la ROC quand les positifs sont rares |

**Choix du seuil de décision** : jamais 0,5 par défaut. Le seuil se choisit sur la validation (jamais sur le test), en fixant d'abord un rappel minimal acceptable (par exemple ≥ 0,95), puis en prenant le seuil qui maximise la précision sous cette contrainte. La règle retenue et sa justification sont écrites dans le rapport.

---

## 5. Sur-apprentissage, sous-apprentissage et quantité de données

Trois diagnostics, toujours en validation croisée par moteur :

1. **Erreur train vs validation, pli par pli**, sur au moins MAE, RMSE et R² (pas seulement le RMSE). Lecture (cours, §2) :

   | Erreur train | Erreur validation | Diagnostic |
   |---|---|---|
   | haute | haute, proche du train | biais (sous-apprentissage) |
   | basse | beaucoup plus haute | variance (sur-apprentissage) |
   | basse | basse, proche | bon modèle |

2. **Courbes de validation** (erreur selon un hyperparamètre de complexité : `max_depth`, nombre d'arbres, taille du réseau…) : montrent à partir de quand le modèle sur-apprend. Déjà faites pour XGBoost (`docs/03_modelisation.md` §6).

3. **Courbes d'apprentissage** (erreur selon la **quantité de données d'entraînement**) : nombre de moteurs d'entraînement croissant (10, 20, 40, 60, 80), validation toujours sur les mêmes moteurs tenus à part, plusieurs tirages par taille.
   Pourquoi c'est utile **ici précisément** : FD001 n'a que 100 moteurs, et la question « faut-il plus de données ? » a une réponse concrète, puisque FD003 (même régime de vol, 100 moteurs de plus) et N-CMAPSS existent.
   - Si l'erreur de validation baisse encore à 80 moteurs : plus de moteurs aideraient → argument pour ajouter FD003.
   - Si elle est déjà plate : ajouter des données ne servira à rien → la limite vient des features ou du modèle, et c'est là qu'il faut travailler (§7).
   - Si l'écart train/validation reste grand quelle que soit la taille : sur-apprentissage → régulariser.
   Pour un LSTM ou un CNN, la courbe d'apprentissage dit aussi si 100 moteurs suffisent à l'entraîner.

---

## 6. Optimisation des hyperparamètres

- **Toujours à l'intérieur de la validation croisée par moteur**, avec les plis du §1. Jamais sur le test.
- Ordre de travail : comprendre d'abord avec des courbes de validation (§5) → chercher ensuite.
  - **Grid search** (`GridSearchCV`, avec `cv=plis` pour réutiliser les plis par moteur) : seulement pour 1 ou 2 hyperparamètres et une petite grille.
  - **Random search** (`RandomizedSearchCV`) : à préférer dès 3 hyperparamètres ou plus (cours, §4).
  - **Optuna** (source donnée par le mentor) : recherche guidée (TPE), qui concentre les essais là où les premiers résultats sont bons. `optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))`, la fonction objectif renvoie la moyenne des plis.
  - Comparer les trois méthodes sur un même budget (même nombre d'entraînements) fait partie du travail : c'est une preuve, pas une préférence.
- **Objectif** : RMSE moyen de validation, avec le score NASA moyen et les surestimations surveillés à chaque essai. Le cahier des charges (Phase 4) demande un équilibre RMSE / score NASA, pas le RMSE seul : si le meilleur essai en RMSE dégrade nettement le score NASA, le dire et justifier le choix.
- **Biais d'optimisme** : un score obtenu en choisissant le meilleur de 100 essais sur la même validation est trop optimiste. Le chiffre final honnête est celui du test officiel (touché une fois). Une validation croisée imbriquée (*nested CV*) est le niveau au-dessus, si le temps le permet.
- **À rapporter** : espace de recherche, budget (nombre d'essais), meilleurs réglages, gain par rapport aux réglages par défaut sur les mêmes plis (moyenne ± écart-type), test de significativité (§8).

---

## 7. Feature engineering

Aucune famille de features n'entre dans un modèle sans : (1) une hypothèse écrite (« la pente du capteur sur 10 cycles devrait capter l'accélération de la dégradation »), (2) un test d'ablation sur les plis du §1, (3) une conclusion chiffrée, y compris quand la feature n'apporte rien.

Contraintes : features **causales** (le cycle t n'utilise que les cycles ≤ t), calculées **moteur par moteur**, et tout ce qui apprend se calcule sur le train seulement.

Familles à tester, de la plus simple à la plus coûteuse :
- statistiques glissantes demandées par le cahier des charges et pas encore testées : variance, **asymétrie (skew), kurtosis** ;
- **pente** sur la fenêtre (régression linéaire de chaque capteur sur les k derniers cycles) : tendance de dégradation ;
- **plusieurs tailles de fenêtre** (5, 10, 20, 30) : la fenêtre de 5 n'a jamais été comparée à d'autres ;
- **moyenne mobile exponentielle** (EWMA) : alternative au lissage actuel, déjà citée dans le cahier des charges ;
- **écart à l'état initial du moteur** (valeur actuelle moins moyenne de ses k premiers cycles) : neutralise les différences de fabrication entre moteurs ;
- pour FD002/FD004 : **normalisation par régime de vol** (K-Means sur les réglages opérationnels, cahier des charges Phase 1).

Après sélection : importance des features (gain XGBoost **et** importance par permutation, plus fiable), puis SHAP en Phase 5. Rappel du constat Phase 3 : 16 features (15 capteurs + `time_cycles`) font aussi bien que 46. Une version plus simple à performance égale est préférable (règle 9 d'`AGENTS.md`).

---

## 8. Comparer deux modèles

- Mêmes plis, mêmes métriques, même tableau.
- On ne conclut « A est meilleur que B » que si l'écart est **significatif** : test t apparié sur les scores des 5 plis (`scipy.stats.ttest_rel`), ou test de Wilcoxon (`scipy.stats.wilcoxon`) si on ne veut pas supposer une loi normale. Avec 5 plis le test a peu de puissance : on peut répéter la validation croisée avec plusieurs graines (`RepeatedGroupKFold` n'existe pas dans scikit-learn, donc boucle sur plusieurs `random_state`), en le documentant.
- Sur le test officiel (100 moteurs), même logique : comparer les erreurs moteur par moteur (test apparié), pas deux chiffres globaux.
- Règle de décision écrite **avant** de regarder les résultats : par exemple « on retient le modèle au meilleur RMSE de validation, sauf s'il augmente les surestimations de plus de 20 cycles ».

---

## 9. Ce que doit contenir chaque rapport de modèle

Tableaux :
1. Régression : modèle (et baselines) × métriques du §3 × {train, validation moyenne ± écart-type, test}.
2. Erreur par tranche de RUL (§3).
3. Classification pour chaque N testé : matrice de confusion, accuracy (avec la proportion de positifs), précision, rappel, F1, AUC ROC, précision moyenne, seuil retenu et justification.
4. Hyperparamètres : espace, méthode, budget, meilleurs réglages, gain significatif ou non.

Figures (titre et axes en français, enregistrées dans `docs/figures/`) :
- RUL prédite vs vraie (validation et test), avec la diagonale idéale ;
- résidus (prédit − vrai) selon la RUL vraie : montre si le modèle se trompe surtout près de la panne ;
- erreur train vs validation par pli ; courbes de validation ; courbe d'apprentissage ;
- matrice de confusion ; courbes ROC et précision-rappel, avec la baseline aléatoire sur le même graphique.

Pour chaque métrique : **gardée** (ce qu'elle apporte) ou **écartée** (chiffre obtenu + pourquoi elle trompe ou n'apporte rien).

---

## 10. Checklist (à copier dans la PR et dans le rapport de phase)

- [ ] Plis par moteur fixés une fois, identiques pour tous les modèles comparés
- [ ] Baselines du §2 calculées sur les mêmes plis
- [ ] Toutes les métriques de régression du §3, sur train / validation / test, avec moyenne ± écart-type
- [ ] Erreur par tranche de RUL
- [ ] Métriques de classification du §4 pour plusieurs N, seuil choisi sur la validation
- [ ] Diagnostic train vs validation sur plusieurs métriques + courbe de validation + courbe d'apprentissage
- [ ] Hyperparamètres optimisés dans la validation croisée (méthode, budget, gain, significativité)
- [ ] Chaque famille de features justifiée par une ablation
- [ ] Comparaison entre modèles avec test de significativité
- [ ] Test officiel utilisé une seule fois, pour le rapport final
- [ ] Chaque métrique gardée ou écartée avec une phrase de justification
