"""
Tests des fonctions validees des Phases 1-2 (src/preprocessing.py) sur de petits exemples calcules a la main.
"""

import os

import pandas as pd

from src.preprocessing import DATA_DIR, calculer_rul, plafonner_rul


def construire_petit_tableau():
    """
    Construit un petit tableau synthetique : moteur 1 suivi sur 4 cycles, moteur 2 sur 3 cycles.

    Returns
    -------
    pandas.DataFrame
        Colonnes "unit_number" et "time_cycles", 7 lignes.
    """
    unites = [1, 1, 1, 1, 2, 2, 2]  # le moteur 1 a 4 cycles, le moteur 2 en a 3
    cycles = [1, 2, 3, 4, 1, 2, 3]  # numero du cycle, qui recommence a 1 pour chaque moteur
    return pd.DataFrame({"unit_number": unites, "time_cycles": cycles})


def test_data_dir_absolu_et_existant():
    """DATA_DIR doit etre un chemin absolu (str) qui pointe vers le dossier data/ contenant les fichiers FD001."""
    assert isinstance(DATA_DIR, str)  # reste une chaine de caracteres (f"{DATA_DIR}/..." du notebook 03)
    assert os.path.isabs(DATA_DIR)  # absolu : ne depend plus du dossier de lancement
    assert os.path.isfile(os.path.join(DATA_DIR, "train_FD001.txt"))
    assert os.path.isfile(f"{DATA_DIR}/RUL_FD001.txt")  # meme construction que dans le notebook 03


def test_calculer_rul_dernier_et_premier_cycle():
    """La RUL vaut 0 au dernier cycle de chaque moteur et n - 1 au premier (n = nombre de cycles du moteur)."""
    tableau = construire_petit_tableau()
    resultat = calculer_rul(tableau)

    # valeurs attendues, calculees a la main : moteur 1 (4 cycles) -> 3, 2, 1, 0 ; moteur 2 (3 cycles) -> 2, 1, 0
    attendu = [3, 2, 1, 0, 2, 1, 0]
    assert list(resultat["RUL"]) == attendu

    # verification explicite des deux extremites, moteur par moteur
    for moteur_id in [1, 2]:
        lignes_moteur = resultat[resultat["unit_number"] == moteur_id]
        nb_cycles = len(lignes_moteur)
        assert lignes_moteur["RUL"].iloc[-1] == 0  # dernier cycle : panne, RUL nulle
        assert lignes_moteur["RUL"].iloc[0] == nb_cycles - 1  # premier cycle : n - 1 cycles restants

    # le tableau d'origine n'est pas modifie (une copie est renvoyee)
    assert "RUL" not in tableau.columns


def test_plafonner_rul():
    """Une RUL de 180 est plafonnee a 125 ; une RUL de 80 reste 80 ; 125 reste 125."""
    tableau = pd.DataFrame({"RUL": [180, 80, 125]})
    resultat = plafonner_rul(tableau, plafond=125)
    assert list(resultat["RUL_cappee"]) == [125, 80, 125]
