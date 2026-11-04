# Fichiers de mesures d'exemple

Un moteur du jeu de **test** par sous-jeu, pour montrer l'import pendant la démo (page « Tester un moteur »,
section « Importer un fichier de mesures »).

- **Origine** : chaque fichier recopie, sans aucune modification, les lignes du moteur indiqué dans
  `data/test_FD00X.txt` (format NASA C-MAPSS brut : 26 nombres par ligne, séparés par des espaces).
  Vérification : `grep "^34 " data/test_FD001.txt` donne exactement `mesures_FD001.txt`.
- **Pas de RUL dans les fichiers** : la RUL vraie au dernier cycle observé est dans `data/RUL_FD00X.txt`
  (ligne n° = numéro du moteur) ; elle est recopiée ci-dessous pour vérifier après coup.
- **Règle de choix** (`choisir_moteur_exemple` dans `src/inference.py`) : parmi les moteurs test d'au moins
  150 cycles observés, celui dont la RUL vraie au dernier cycle est la plus petite (le plus proche de la panne).
- Ces moteurs n'ont jamais servi à l'entraînement : les modèles ont appris sur `data/train_FD00X.txt`.
- Contrôle automatique : `tests/test_inference.py::test_fichiers_exemples_extraits_exacts_du_jeu_de_test`.

| Sous-jeu | Fichier | Moteur test | Cycles observés | RUL vraie au dernier cycle |
|---|---|---|---|---|
| FD001 | `mesures_FD001.txt` | 34 | 203 | 7 |
| FD002 | `mesures_FD002.txt` | 7 | 184 | 6 |
| FD003 | `mesures_FD003.txt` | 82 | 194 | 6 |
| FD004 | `mesures_FD004.txt` | 158 | 171 | 6 |

Pour révéler la vérité dans l'application après l'import, saisir la RUL vraie de la dernière colonne.
