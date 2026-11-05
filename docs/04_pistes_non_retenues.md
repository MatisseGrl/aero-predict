# Phase 4 : pistes testées et non retenues dans le notebook 04f

Règle (décidée le 2026-11-05) : le notebook `04f_modele_final.ipynb` ne contient que ce qui améliore le modèle. Un test sans gain est consigné ici, pas dans le notebook.

Protocole commun : 10 découpages par moteur neufs (graines 1000 à 1009, 5 plis), XGBoost aux réglages de la Phase 3, RMSE de validation poolé, évalué contre la RUL plafonnée à 125 (prédictions écrêtées à 125). Test jamais regardé.

| Variante | RMSE validation | Écart vs base (k = 20, plafond 125) | Plus mauvais que la base | p (t apparié) | Décision |
|---|---|---|---|---|---|
| Base (k = 20, plafond 125) | 9,291 | / | / | / | gardée |
| Plafond 130 | 9,308 | +0,017 | 8/10 | 0,31 | non retenu |
| Plafond 140 | 9,489 | +0,198 | 10/10 | < 0,001 | non retenu (pire) |
| Plafond 150 | 9,646 | +0,355 | 10/10 | < 0,001 | non retenu (pire) |
| Fenêtre d'état initial k = 25 | 9,092 | −0,199 | 0/10 | < 0,001 | non retenu : gain < 0,3 cycle (règle écrite avant les tests) |

À retenir :
- Monter le plafond de la RUL n'aide pas : plus le plafond est haut, plus l'erreur augmente.
- k = 25 est meilleur de façon régulière (10 découpages sur 10), mais le gain (0,2 cycle) reste sous le seuil fixé à 0,3 cycle. Reste une option à adopter si l'équipe le décide.
- k ne peut pas dépasser 30 : le moteur de test le plus court n'a que 31 cycles observés.
