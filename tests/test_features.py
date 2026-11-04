"""
Tests du pipeline de features (src/features.py) : ordre des colonnes, absence de fuite, causalite,
RUL vraie par cycle sur le test.
"""

import numpy as np
import pandas as pd

from src.features import construire_features_glissantes
from src.preprocessing import find_constant_sensors, get_sensor_columns, lisser_capteurs, load_fd001


def test_features_46_colonnes_dans_l_ordre_du_notebook(contexte_train, contexte_test):
    """X contient les 46 features dans l'ordre du notebook 03 : capteurs, moyennes, ecarts-types, time_cycles."""
    capteurs = contexte_train["sensor_cols_restants"]
    assert len(capteurs) == 15

    # ordre attendu, reconstruit bloc par bloc
    attendu = []
    for capteur in capteurs:
        attendu.append(capteur)
    for capteur in capteurs:
        attendu.append(capteur + "_glissante_moy")
    for capteur in capteurs:
        attendu.append(capteur + "_glissante_std")
    attendu.append("time_cycles")

    assert list(contexte_train["X"].columns) == attendu
    assert list(contexte_test["X_test_dernier"].columns) == attendu

    # aucune valeur manquante (les NaN du 1er cycle ont ete remplaces par 0)
    assert contexte_train["X"].isna().sum().sum() == 0
    assert contexte_test["X_test_dernier"].isna().sum().sum() == 0
    assert contexte_train["X"].shape == (20631, 46)
    assert contexte_test["X_test_dernier"].shape == (100, 46)


def test_absence_de_fuite_scaler_appris_sur_le_train_brut(contexte_train, contexte_test):
    """
    Apres preparer_test, le scaler contient exactement les min/max des capteurs du TRAIN brut.

    On recalcule ces min/max a la main sur le fichier train brut. Si preparer_test avait reajuste
    le scaler sur le test, ses parametres seraient ceux du test (qui different : verifie aussi).
    """
    scaler = contexte_train["scaler"]
    capteurs = contexte_train["sensor_cols_restants"]

    train_brut = load_fd001("train")  # fichier brut, sans aucune transformation
    test_brut = load_fd001("test")

    au_moins_une_difference_avec_le_test = False
    for position in range(len(capteurs)):  # un capteur a la fois
        capteur = capteurs[position]
        minimum_train = train_brut[capteur].min()  # minimum recalcule a la main sur le train brut
        maximum_train = train_brut[capteur].max()
        assert scaler.data_min_[position] == minimum_train, f"{capteur} : min du scaler different du min du train"
        assert scaler.data_max_[position] == maximum_train, f"{capteur} : max du scaler different du max du train"

        # le test brut a d'autres min/max : le test ci-dessus aurait donc detecte un reajustement
        if test_brut[capteur].min() != minimum_train or test_brut[capteur].max() != maximum_train:
            au_moins_une_difference_avec_le_test = True

    assert au_moins_une_difference_avec_le_test
    assert scaler.n_samples_seen_ == len(train_brut)  # le scaler n'a vu que les lignes du train

    # les capteurs retires sont ceux trouves sur le train
    capteurs_bruts = get_sensor_columns(train_brut.columns)
    assert contexte_train["constant_sensors"] == find_constant_sensors(train_brut, capteurs_bruts)


def construire_tableau_synthetique():
    """
    Construit un petit tableau synthetique : 2 moteurs (12 et 8 cycles), 2 capteurs aleatoires.

    Returns
    -------
    pandas.DataFrame
        Colonnes "unit_number", "time_cycles", "sensor_a", "sensor_b".
    """
    generateur = np.random.default_rng(0)  # graine fixe : le test est reproductible
    unites = []
    cycles = []
    for moteur_id, nb_cycles in [(1, 12), (2, 8)]:  # moteur 1 : 12 cycles, moteur 2 : 8 cycles
        for cycle in range(1, nb_cycles + 1):
            unites.append(moteur_id)
            cycles.append(cycle)
    tableau = pd.DataFrame({"unit_number": unites, "time_cycles": cycles})
    tableau["sensor_a"] = generateur.normal(size=len(tableau))
    tableau["sensor_b"] = generateur.normal(size=len(tableau))
    return tableau


def calculer_features_synthetiques(tableau, capteurs):
    """
    Applique au tableau synthetique les memes etapes que le pipeline : lissage puis features glissantes.

    Parameters
    ----------
    tableau : pandas.DataFrame
        Le tableau synthetique.
    capteurs : list de str
        Les colonnes capteurs.

    Returns
    -------
    (pandas.DataFrame, list de str)
        Le tableau enrichi et la liste des colonnes de features a comparer.
    """
    tableau_lisse = lisser_capteurs(tableau, capteurs, fenetre=5)
    enrichi, colonnes_moyenne, colonnes_ecart_type = construire_features_glissantes(tableau_lisse, capteurs, 5)
    colonnes_features = capteurs + colonnes_moyenne + colonnes_ecart_type
    return enrichi, colonnes_features


def test_causalite_lissage_et_features_glissantes():
    """
    Modifier un capteur au cycle t d'un moteur ne change aucune feature des cycles < t de ce moteur,
    ni aucune feature d'un autre moteur (lissage et features glissantes causaux et par moteur).
    """
    capteurs = ["sensor_a", "sensor_b"]
    tableau = construire_tableau_synthetique()
    reference, colonnes_features = calculer_features_synthetiques(tableau, capteurs)

    for cycle_modifie in [1, 4, 7, 12]:  # plusieurs positions, y compris le premier et le dernier cycle
        tableau_modifie = tableau.copy()
        ligne_modifiee = (tableau_modifie["unit_number"] == 1) & (tableau_modifie["time_cycles"] == cycle_modifie)
        tableau_modifie.loc[ligne_modifiee, "sensor_a"] = tableau_modifie.loc[ligne_modifiee, "sensor_a"] + 10.0
        modifie, _ = calculer_features_synthetiques(tableau_modifie, capteurs)

        # 1) cycles strictement anterieurs du moteur 1 : features identiques bit a bit
        passe_moteur_1 = (reference["unit_number"] == 1) & (reference["time_cycles"] < cycle_modifie)
        assert reference.loc[passe_moteur_1, colonnes_features].equals(modifie.loc[passe_moteur_1, colonnes_features])

        # 2) l'autre moteur : features identiques bit a bit
        moteur_2 = reference["unit_number"] == 2
        assert reference.loc[moteur_2, colonnes_features].equals(modifie.loc[moteur_2, colonnes_features])

        # 3) controle : la modification se voit bien au cycle modifie (sinon le test ne prouverait rien)
        valeur_avant = reference.loc[ligne_modifiee, "sensor_a"].iloc[0]
        valeur_apres = modifie.loc[ligne_modifiee, "sensor_a"].iloc[0]
        assert valeur_avant != valeur_apres


def test_rul_vraie_a_chaque_cycle_du_test(contexte_test):
    """
    Sur le test : RUL_vraie = RUL_FD001 au dernier cycle, et baisse d'exactement 1 a chaque cycle.
    """
    test = contexte_test["test"]
    y_test_vraie = contexte_test["y_test_vraie"]
    moteurs = contexte_test["moteurs_test"]

    assert len(y_test_vraie) == 100
    assert list(moteurs) == list(range(1, 101))  # moteurs tries, alignes sur RUL_FD001.txt

    for position in range(len(moteurs)):  # un moteur a la fois
        lignes_moteur = test[test["unit_number"] == moteurs[position]].sort_values("time_cycles")
        rul_par_cycle = lignes_moteur["RUL_vraie"].values
        assert rul_par_cycle[-1] == y_test_vraie[position]  # dernier cycle : valeur de RUL_FD001.txt
        ecarts_successifs = np.diff(rul_par_cycle)  # difference entre deux cycles consecutifs
        assert np.all(ecarts_successifs == -1)  # la RUL baisse de 1 a chaque cycle

    # version plafonnee : jamais au-dessus de 125, egale a la brute en dessous
    assert test["RUL_vraie_cappee"].max() <= 125
    sous_le_plafond = test["RUL_vraie"] <= 125
    assert (test.loc[sous_le_plafond, "RUL_vraie_cappee"] == test.loc[sous_le_plafond, "RUL_vraie"]).all()
