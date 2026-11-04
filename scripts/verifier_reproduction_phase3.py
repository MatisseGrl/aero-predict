"""
Verifie que les nouveaux modules src/ reproduisent les chiffres de la Phase 3 (notebook 03).

Relance tout le pipeline Phase 3 avec src/features.py, src/evaluation.py et src/modeles.py :
XGBoost avec les reglages de la Phase 3, validation croisee GroupKFold par moteur (5 plis melanges,
graine 42), puis modele final entraine sur les 100 moteurs et evalue sur le test officiel.

Lancement, depuis la racine du depot (ou n'importe quel autre dossier) :

    python scripts/verifier_reproduction_phase3.py

Rappel (docs/03_modelisation.md, note de reproductibilite) : XGBoost n'est pas garanti identique
bit a bit d'un environnement a l'autre. Les chiffres de validation croisee peuvent donc differer
legerement des valeurs du document si l'environnement (versions, machine) n'est pas le meme. Le
test decisif pour le refactor est de comparer avec le notebook 03 execute dans le MEME environnement.
"""

import os
import sys

# on ajoute la racine du depot (dossier parent de scripts/) au chemin d'import, pour "import src..."
DOSSIER_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
RACINE_DEPOT = os.path.dirname(DOSSIER_SCRIPTS)
sys.path.insert(0, RACINE_DEPOT)

from src.evaluation import calculer_plis, evaluer_en_validation_croisee, evaluer_sur_test  # noqa: E402
from src.features import preparer_test, preparer_train  # noqa: E402
from src.modeles import fabrique_xgboost  # noqa: E402

# Valeurs de reference publiees dans docs/03_modelisation.md (notebook 03, 46 features)
ATTENDU_CV_RMSE_MOYEN = 16.86
ATTENDU_CV_NASA_MOYEN = 29729.3
ATTENDU_TEST_RMSE = 19.70
ATTENDU_TEST_NASA = 852.9
ATTENDU_TEST_SURESTIMES = 9


def afficher_comparaison(nom, obtenu, attendu, decimales):
    """
    Affiche une ligne "obtenu vs attendu", arrondie comme dans la documentation.

    Parameters
    ----------
    nom : str
        Le nom de la metrique.
    obtenu : float
        La valeur calculee par ce script.
    attendu : float
        La valeur publiee dans docs/03_modelisation.md.
    decimales : int
        Nombre de decimales utilisees par la documentation pour cette metrique.

    Returns
    -------
    bool
        Vrai si la valeur obtenue, arrondie, est egale a la valeur attendue.
    """
    obtenu_arrondi = round(obtenu, decimales)  # meme arrondi que la documentation
    identique = obtenu_arrondi == round(attendu, decimales)
    if identique:
        verdict = "identique"
    else:
        verdict = "ECART"
    print(f"  {nom:38s} obtenu = {obtenu:12.{decimales}f}   attendu = {attendu:12.{decimales}f}   -> {verdict}")
    return identique


def main():
    """
    Enchaine preparation des donnees, validation croisee, modele final et test officiel, puis compare.

    Returns
    -------
    None
    """
    # --- 1. Donnees : meme pipeline que le notebook 03 ---
    contexte_train = preparer_train(fenetre=5, plafond=125)
    contexte_test = preparer_test(contexte_train)
    X = contexte_train["X"]
    y = contexte_train["y"]
    groups = contexte_train["groups"]
    print(f"Train : X {X.shape}, y {y.shape} | test : {contexte_test['X_test_dernier'].shape[0]} moteurs")

    # --- 2. Validation croisee par moteur, XGBoost reglages Phase 3 ---
    plis = calculer_plis(X, y, groups, n_splits=5, random_state=42)
    resultats_cv = evaluer_en_validation_croisee(fabrique_xgboost, X, y, groups, plis)

    print("\nValidation croisee (GroupKFold, 5 plis, shuffle=True, random_state=42) :")
    for numero_pli in range(len(plis)):  # un pli a la fois, comme l'affichage du notebook 03
        rmse_pli = resultats_cv["rmse_par_pli"][numero_pli]
        nasa_pli = resultats_cv["nasa_par_pli"][numero_pli]
        print(f"  Pli {numero_pli + 1} : RMSE={rmse_pli:.2f}, score NASA={nasa_pli:.1f}")
    print(f"  Moyenne : RMSE={resultats_cv['rmse_moyen']:.2f} (ecart-type {resultats_cv['rmse_ecart_type']:.2f}), "
          f"score NASA={resultats_cv['nasa_moyen']:.1f}")

    # --- 3. Modele final sur les 100 moteurs, puis test officiel ---
    modele_final = fabrique_xgboost()
    modele_final.fit(X, y)
    resultats_test = evaluer_sur_test(modele_final, contexte_test["X_test_dernier"], contexte_test["y_test_vraie"])

    print("\nTest officiel (dernier cycle de chaque moteur, vs RUL_FD001.txt brute) :")
    print(f"  RMSE={resultats_test['rmse']:.2f}, score NASA={resultats_test['score_nasa']:.1f}, "
          f"surestimes > 20 cycles : {resultats_test['nb_surestimes']}/100")

    # --- 4. Comparaison avec les valeurs publiees ---
    print("\nComparaison avec docs/03_modelisation.md :")
    verdicts = []
    verdicts.append(afficher_comparaison("CV RMSE moyen", resultats_cv["rmse_moyen"], ATTENDU_CV_RMSE_MOYEN, 2))
    verdicts.append(afficher_comparaison("CV score NASA moyen", resultats_cv["nasa_moyen"], ATTENDU_CV_NASA_MOYEN, 1))
    verdicts.append(afficher_comparaison("Test RMSE", resultats_test["rmse"], ATTENDU_TEST_RMSE, 2))
    verdicts.append(afficher_comparaison("Test score NASA", resultats_test["score_nasa"], ATTENDU_TEST_NASA, 1))
    verdicts.append(afficher_comparaison("Test surestimes > 20 cycles", resultats_test["nb_surestimes"],
                                         ATTENDU_TEST_SURESTIMES, 0))

    if all(verdicts):
        print("\nTous les chiffres publies sont reproduits.")
    else:
        print("\nAu moins un ecart avec la documentation. Si seuls les chiffres de validation croisee different")
        print("legerement, c'est probablement l'environnement (XGBoost, voir docs/03_modelisation.md) : pour")
        print("trancher, executer notebooks/03_modelisation.ipynb dans ce meme environnement et comparer.")


if __name__ == "__main__":
    main()
