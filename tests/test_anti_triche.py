"""
Preuves essentielles que les modeles ne trichent pas. Lancement, depuis la racine du depot : python -m pytest -q

(c) `test_fenetres_sans_autre_moteur_ni_futur` : chaque fenetre du LSTM / CNN 1D ne contient que des cycles <= t
    du moteur de la ligne ;
(d) `test_preparation_apprise_sur_le_train` : scalers min-max et K-Means des regimes appris sur le train seulement ;
(e) `test_ecart_etat_initial_causal_et_par_moteur` : la variable "ecart a l'etat initial" (option ecart_initial,
    notebook 04e) au cycle t ne depend ni des cycles > t ni des autres moteurs.

Les preuves (a) et (b), qui portent sur la prediction en direct de la demo, arrivent avec la demo (src/inference.py).
"""

import os
import sys

import numpy as np

# racine du depot (dossier parent de tests/) importable, pour "import src..." quel que soit le dossier de lancement
RACINE_DEPOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RACINE_DEPOT not in sys.path:
    sys.path.insert(0, RACINE_DEPOT)

from src.deep_learning import construire_fenetres  # noqa: E402
from src.features import calculer_ecart_etat_initial, preparer_train  # noqa: E402
from src.preprocessing import COLUMNS  # noqa: E402
from src.sous_jeux import (  # noqa: E402
    COLONNES_REGLAGES,
    appliquer_preparation,
    attribuer_regimes,
    charger_sous_jeu,
    creer_kmeans,
    preparer_sous_jeu,
)


def test_fenetres_sans_autre_moteur_ni_futur():
    """(c) Sur tout le train FD001 : chaque position de chaque fenetre vient du meme moteur et d'un cycle <= t."""
    train = preparer_train(fenetre=5, plafond=125)["train"]

    # astuce : on donne comme "entrees" le numero de moteur et le cycle, pour lire d'ou vient chaque position
    fenetres, _, numeros, cycles = construire_fenetres(train, ["unit_number", "time_cycles"], longueur=30)
    moteurs_vus = fenetres[:, :, 0]  # (nb_lignes, 30) : le moteur de chaque position
    cycles_vus = fenetres[:, :, 1]   # (nb_lignes, 30) : le cycle de chaque position

    assert fenetres.shape[0] == len(train)  # une fenetre par ligne
    assert np.array_equal(numeros, train["unit_number"].to_numpy())  # dans l'ordre des lignes
    assert np.all(moteurs_vus == numeros[:, np.newaxis])  # jamais un autre moteur
    assert np.all(cycles_vus <= cycles[:, np.newaxis])  # jamais le futur
    assert np.array_equal(cycles_vus[:, -1], cycles)  # la derniere position est le cycle present


def test_preparation_apprise_sur_le_train():
    """(d) FD001 : min-max du train brut. FD002 : K-Means et un min-max par regime appris sur le train seul."""
    # FD001 (pipeline des Phases 1 a 3) : le scaler a vu exactement les lignes du train, et leurs min / max
    contexte_fd001 = preparer_train(fenetre=5, plafond=125)
    train_fd001 = charger_sous_jeu("FD001", "train")
    scaler = contexte_fd001["scaler"]
    capteurs = contexte_fd001["sensor_cols_restants"]
    assert scaler.n_samples_seen_ == len(train_fd001)
    assert np.array_equal(scaler.data_min_, train_fd001[capteurs].min().to_numpy())
    assert np.array_equal(scaler.data_max_, train_fd001[capteurs].max().to_numpy())

    # FD002 (6 regimes) : un K-Means appris sur le train seul donne les memes centres (a 1e-9 pres : K-Means
    # additionne en parallele, l'ordre des additions peut changer au dernier chiffre) et les memes regimes
    contexte = preparer_sous_jeu("FD002")
    train_brut = charger_sous_jeu("FD002", "train")
    test_brut = charger_sous_jeu("FD002", "test")
    kmeans_train_seul = creer_kmeans(6)
    kmeans_train_seul.fit(train_brut[COLONNES_REGLAGES])
    assert np.allclose(contexte["kmeans"].cluster_centers_, kmeans_train_seul.cluster_centers_, rtol=0, atol=1e-9)
    assert np.array_equal(contexte["test"]["regime"].to_numpy(), kmeans_train_seul.predict(test_brut[COLONNES_REGLAGES]))

    # chaque scaler de regime a vu exactement les lignes TRAIN de ce regime (aucune ligne test)
    regimes_train = attribuer_regimes(train_brut, contexte["kmeans"])
    for regime in range(6):
        lignes_regime = train_brut.loc[regimes_train == regime, contexte["capteurs"]]
        scaler_regime = contexte["scalers"][regime]
        assert scaler_regime.n_samples_seen_ == len(lignes_regime)
        assert np.array_equal(scaler_regime.data_min_, lignes_regime.min().to_numpy())
        assert np.array_equal(scaler_regime.data_max_, lignes_regime.max().to_numpy())


def test_ecart_etat_initial_causal_et_par_moteur():
    """(e) Ecart a l'etat initial (k = 20) : definition, causalite (cycles > t sans effet) et calcul moteur par moteur."""
    # definition sur un exemple calcule a la main (k = 3) : references [10, 11, 12, 12] -> ecarts [0, 1, 2, 8]
    ecarts_exemple = calculer_ecart_etat_initial(np.array([10.0, 12.0, 14.0, 20.0]), 3)
    assert np.allclose(ecarts_exemple, [0.0, 1.0, 2.0, 8.0], rtol=0, atol=1e-12)

    # pipeline complet FD001 avec l'option (regles apprises sur le train), applique a des moteurs test bruts
    contexte = preparer_sous_jeu("FD001", ecart_initial=20)
    colonnes_ecart = contexte["colonnes_ecart_initial"]
    assert len(colonnes_ecart) == len(contexte["capteurs"])
    test_brut = charger_sous_jeu("FD001", "test")
    moteur_1 = test_brut[test_brut["unit_number"] == 1].reset_index(drop=True)
    moteur_2 = test_brut[test_brut["unit_number"] == 2].reset_index(drop=True)
    capteurs_bruts = []  # les 21 capteurs du fichier brut
    for colonne in COLUMNS:
        if colonne.startswith("sensor_"):
            capteurs_bruts.append(colonne)

    # causalite : cycles > t multiplies par 1000 -> valeurs aux cycles <= t identiques (t < k et t > k)
    reference = appliquer_preparation(contexte, moteur_1)
    for cycle in [10, 50]:
        modifie = moteur_1.copy()
        masque_futur = modifie["time_cycles"] > cycle
        modifie.loc[masque_futur, capteurs_bruts] = modifie.loc[masque_futur, capteurs_bruts] * 1000.0
        avec_futur_modifie = appliquer_preparation(contexte, modifie)
        masque_passe = reference["time_cycles"] <= cycle
        assert np.array_equal(avec_futur_modifie.loc[masque_passe, colonnes_ecart].to_numpy(),
                              reference.loc[masque_passe, colonnes_ecart].to_numpy())

    # moteur par moteur : le moteur 2 prepare seul ou a la suite du moteur 1 donne les memes valeurs,
    # et chaque moteur part de 0 a son premier cycle (sa propre reference, pas celle du moteur precedent)
    deux_moteurs = test_brut[test_brut["unit_number"] <= 2].reset_index(drop=True)
    ensemble = appliquer_preparation(contexte, deux_moteurs)
    seul = appliquer_preparation(contexte, moteur_2)
    lignes_moteur_2 = ensemble["unit_number"] == 2
    assert np.array_equal(ensemble.loc[lignes_moteur_2, colonnes_ecart].to_numpy(), seul[colonnes_ecart].to_numpy())
    premiers_cycles = ensemble.groupby("unit_number").head(1)
    assert np.all(premiers_cycles[colonnes_ecart].to_numpy() == 0.0)
