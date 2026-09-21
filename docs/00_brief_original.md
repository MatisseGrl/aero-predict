# Brief pédagogique original — Aero Predict

> Document source, non modifié dans le fond. Conservé pour traçabilité : c'est l'un des trois documents croisés pour produire [`docs/cahier_des_charges.md`](cahier_des_charges.md) (voir sa Section 1).

**Déposant** : Paul Lemaistre — plemaistre-tw3@omneseducation.com
**Source** : Digityser, 2026 © Digityser
**Accessibilité / difficulté** : Avancé
**Pré-requis** : Python / scikit-learn / PyTorch · Séries temporelles · Feature engineering · Matplotlib
**Suggestions de problématiques** : #IA / Machine Learning · #Entreprise · #Technique

## Contexte

Dans l'industrie 4.0 et l'aéronautique, les pannes inattendues d'équipements critiques entraînent des coûts d'arrêt colossaux et des risques de sécurité majeurs. La maintenance préventive classique (remplacement à intervalles fixes) est souvent sous-optimale, entraînant le remplacement de pièces encore fonctionnelles. La maintenance prédictive vise à anticiper la panne en analysant en continu les données des capteurs (vibrations, températures, pressions) pour estimer précisément la Durée de Vie Utile Restante (Remaining Useful Life - RUL). Le dataset C-MAPSS (Commercial Modular Aero-Propulsion System Simulation), généré par la NASA, est le jeu de données de référence mondial pour ce problème. Il simule la dégradation de turboréacteurs d'avion sous différentes conditions opératoires et modes de défaillance, via des séries temporelles multivariées issues de 21 capteurs.

## Description courte

Maintenance Prédictive Industrielle et Estimation de la Durée de Vie Utile Restante (RUL) sur Turboréacteurs

## Objectif principal / Résumé

Développer, évaluer et comparer un pipeline complet d'apprentissage automatique pour la prédiction de RUL sur le dataset NASA C-MAPSS. Le projet implique une phase critique d'ingénierie des caractéristiques et de traitement des signaux temporels, suivie de la modélisation comparative allant des algorithmes traditionnels (Random Forest, XGBoost) aux architectures d'apprentissage profond spécialisées dans les séquences (LSTM, CNN 1D, Transformers). L'évaluation se fera sur l'optimisation du RMSE ainsi que sur la fonction de score asymétrique spécifique à ce domaine.

## Cahier des charges succinct

### Phase 1 — Analyse Exploratoire et Préparation des Données (EDA)

- Chargement des 4 sous-jeux de données (FD001 à FD004) qui varient selon le nombre de conditions opératoires (1 ou 6) et le nombre de modes de défaillance (1 ou 2).
- Identification et suppression des capteurs constants ou non informatifs (variance nulle).
- Normalisation des données par capteur. Pour les sous-jeux multi-régimes (FD002, FD004), implémentation d'une normalisation conditionnelle (regroupement par K-Means sur les Operating Settings pour normaliser les capteurs intra-régime et gommer l'effet des changements de régime de vol).
- Lissage des signaux bruités (moyennes mobiles, filtres de Savitzky-Golay, ou lissage exponentiel).

### Phase 2 — Définition de la Cible (Target Engineering)

- Calcul de la RUL réelle pour chaque cycle d'un moteur d'entraînement.
- Implémentation de la technique de la Piecewise Linear RUL (RUL linéaire par morceaux) : plafonnement de la RUL maximale en début de vie du moteur (généralement capée à 125 ou 130 cycles), car un moteur neuf ne présente pas de signaux de dégradation mesurables avant un certain stade d'usure.

### Phase 3 — Modélisation et Architectures Temporelles

- **Baseline Machine Learning** : extraction de caractéristiques statistiques glissantes (moyenne, variance, asymétrie, kurtosis sur une fenêtre temporelle) et entraînement de modèles de régression (XGBoost, LightGBM, Support Vector Regression).
- **Deep Learning séquentiel** :
  - Formatage des données en tenseurs 3D (échantillons, fenêtres temporelles, capteurs) avec une technique de fenêtrage glissant (sliding window).
  - Implémentation d'un réseau récurrent : LSTM (Long Short-Term Memory) ou GRU.
  - Implémentation d'un réseau convolutif temporel : CNN 1D ou TCN (Temporal Convolutional Network), souvent plus rapides à entraîner.
- **Optionnel (Avancé)** : architecture basée sur le mécanisme d'attention (Time Series Transformer) pour capturer les dépendances à très long terme.

### Phase 4 — Évaluation et Optimisation des Métriques

- Évaluation des prédictions sur le jeu de test fourni par C-MAPSS (qui s'arrête avant la panne réelle).
- Métrique standard : minimisation du RMSE (Root Mean Squared Error).
- Métrique métier : implémentation de la Fonction de Score Asymétrique de la NASA. Cette fonction pénalise de manière exponentielle les surestimations de RUL (prédire une panne plus tard qu'elle n'arrive = risque de crash) tout en pénalisant plus doucement les sous-estimations (prédire une panne trop tôt = simple coût de maintenance prématurée).
- Optimisation des hyperparamètres (taille de la fenêtre temporelle, nombre de couches, taux d'apprentissage) pour équilibrer ces deux métriques.

### Phase 5 — Explicabilité et Mise en Production Simulée

- Interprétabilité des modèles : utilisation des valeurs de Shapley (SHAP) pour identifier quels capteurs (ex : température à la sortie de la turbine, vitesse du compresseur) contribuent le plus à la prédiction d'une panne imminente.
- Création d'un tableau de bord de supervision (Dash ou Streamlit) permettant de visualiser la courbe de dégradation du moteur en temps réel simulé, l'intervalle de confiance de la RUL, et les alertes d'intervention.
