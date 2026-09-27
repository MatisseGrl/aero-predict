# Phase 0 — État de l'art

**Statut** : brouillon en attente de validation.

## Problématique

Estimer la RUL (Remaining Useful Life) d'un turboréacteur à partir de ses seuls capteurs, alors qu'aucune donnée réelle de panne n'existe (voir Section 1 ci-dessous) — et le faire en tenant compte du fait qu'une surestimation (panne non anticipée) coûte bien plus cher qu'une sous-estimation (maintenance un peu précoce).

## Sources

1. Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation.* PHM08. — **source normative** : définit le dataset C-MAPSS et la fonction de score NASA. Fichier local : `data/Damage Propagation Modeling.pdf`.
2. Synthèse de 9 dépôts GitHub publics traitant du même problème — déjà rédigée dans [`docs/cahier_des_charges.md`](cahier_des_charges.md), Section 3. Pas dupliquée ici, ce document y renvoie.

## 1. Pourquoi ce dataset est simulé, pas réel

En pronostic industriel, les données "run-to-failure" (un historique complet jusqu'à la panne réelle) sont rares pour deux raisons :
- Un moteur d'avion n'est presque jamais laissé aller jusqu'à la panne réelle — il est réparé/remplacé avant, pour des raisons de sécurité.
- Les données de dégradation que les compagnies ont quand même sont confidentielles (avantage concurrentiel).

D'où l'approche du papier : simuler un moteur réaliste et y injecter une dégradation artificielle pour générer les données qu'on ne pourrait jamais observer en vrai.

## 2. Le simulateur : C-MAPSS

C-MAPSS (Commercial Modular Aero-Propulsion System Simulation, NASA) simule un vrai turboréacteur commercial (classe 90 000 lb de poussée), avec son système de contrôle (régulateur de vitesse, limiteurs), sur 5 composants tournants : Fan, LPC, HPC, HPT, LPT. Sur 58 sorties possibles du simulateur, **21 ont été retenues** comme capteurs pour le challenge — ce sont les colonnes `sensor_1` à `sensor_21` de nos fichiers de données.

## 3. Comment la panne est injectée

Le papier définit un **indice de santé** `h(t)` qui part de 1 (moteur neuf) et décroît vers 0 (panne) selon une forme "plate longtemps, puis chute rapide en fin de vie", en faisant varier le rendement et le débit du module HPC (High-Pressure Compressor). La panne survient quand cet indice atteint 0 — c'est le critère d'arrêt de chaque simulation, pas une durée fixe : chaque moteur a donc une durée de vie totale différente et aléatoire.

*Note* : l'équation exacte de `h(t)` (papier, Section IV.D, Eq. 4-8, p.4-5) n'est pas reproduite ici — l'extraction automatique du PDF l'a corrompue (vérifié : recalculée numériquement, elle sort de l'intervalle [0,1] et diverge, donc pas fiable telle quelle) et de toute façon inutile pour le projet : on part directement des données déjà générées par la NASA, pas besoin de refaire tourner leur simulateur.

Conséquence directe pour la Phase 2 : la dégradation étant exponentielle (quasi plate longtemps, puis chute rapide), les capteurs ne montrent presque aucun signe de dégradation en début de vie — vérifié empiriquement sur le moteur 1 de `train_FD001.txt` (`sensor_2` : 641.7-643.0 en début de vie, identique autour de RUL=125, dérive seulement visible en toute fin de vie, RUL<20). C'est l'argument technique qui justifie le plafonnement de la RUL (Phase 2, cap à 125 cycles) : demander à un modèle de distinguer RUL=300 de RUL=280 à partir d'un signal plat est une tâche impossible, pas un manque d'algorithme.

## 4. Structure train / test / RUL

- **Train** : chaque moteur est suivi **jusqu'à la panne réelle** (dernier cycle = panne).
- **Test** : les trajectoires sont **coupées avant la panne**, volontairement — c'est l'exercice : estimer combien de cycles il reste à partir d'un historique partiel.
- **RUL_FD001.txt** : la vraie réponse (un chiffre par moteur test), jamais visible pendant l'entraînement.

FD001 est le sous-jeu le plus simple (1 régime de vol, 1 mode de panne) — point de départ recommandé avant FD002-004 (plusieurs régimes, plusieurs modes de panne).

## 5. La fonction de score NASA

Le score pénalise de façon **asymétrique et exponentielle** : une prédiction en retard (le modèle croit que le moteur tiendra plus longtemps qu'en réalité — risque de panne non anticipée) est punie plus sévèrement qu'une prédiction en avance (juste un coût de maintenance précoce). Formule (papier, Eq. 11), avec `d = RUL_estimée - RUL_vraie` :

```
s = exp(-d / 13) - 1   si d < 0  (prédiction en avance)
s = exp(d / 10) - 1    si d >= 0 (prédiction en retard)
```

Le dénominateur plus petit (10 < 13) sur la branche "retard" fait grimper la pénalité plus vite pour un même écart — c'est ce qui traduit l'asymétrie en pratique. Cette version est celle déjà implémentée dans `notebooks/exploration/baseline_fd001_v0.ipynb` et systématiquement citée dans la littérature communautaire.

*Note de rigueur* : l'extraction automatique du PDF source a légèrement mélangé l'affectation exacte des constantes `a1`/`a2` du papier aux deux branches (artefact classique sur les formules à accolades dans un texte scanné). La direction de l'asymétrie et les deux valeurs (10 et 13) sont en revanche confirmées sans ambiguïté par le texte du papier et par la convergence avec l'implémentation communautaire — c'est cette version qu'on retient.

## 6. Ce qu'on reprend tel quel, ce qu'on adapte, ce qui différencie Aero Predict

| | Détail |
|---|---|
| **Repris tel quel** | Score NASA (formule ci-dessus), RUL plafonnée à 125 (convention communautaire, Section 3 du cahier des charges), FD001 comme point de départ |
| **Adapté** | Le baseline exploratoire n'avait pas suivi la normalisation ni le lissage prévus en Phase 1 — gardé un temps comme benchmark chiffré (ADR-0002), puis le fichier lui-même supprimé (ADR-0006) ; seuls les 6 chiffres du tableau (README, section "Benchmark de référence") sont conservés, tels quels. **Ce notebook n'existe plus et ne doit plus jamais servir de base à une autre affirmation ou décision** — voir la mise en garde ajoutée au README. |
| **Différenciant** | La Phase 6 (plateforme MLOps — API, base relationnelle, simulateur de flux, Docker Compose) : aucun des 9 dépôts communautaires observés ne va aussi loin (Section 3 du cahier des charges) |

## Prochaine étape

Phase 1 — EDA et préparation des données sur FD001 : suppression des capteurs constants, normalisation, lissage.
