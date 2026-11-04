"""
Preuves essentielles que le prototype ne triche pas. Lancement, depuis la racine du depot : python -m pytest -q

(a) `test_direct_egal_precalcule` : au cycle t, la prediction calculee EN DIRECT a partir des seules mesures 1..t
    est egale (a 1e-4 cycle pres) a la prediction precalculee de sorties/predictions/<SOUS_JEU>__<modele>.csv.gz ;
(b) `test_cycles_futurs_sans_effet` : remplacer les mesures des cycles > t par n'importe quoi ne change pas la
    prediction au cycle t (le modele ne lit jamais le futur) ;
(c) `test_fenetres_sans_autre_moteur_ni_futur` : chaque fenetre du LSTM / CNN 1D ne contient que des cycles <= t
    du moteur de la ligne ;
(d) `test_preparation_apprise_sur_le_train` : scalers min-max et K-Means des regimes appris sur le train seulement ;
(e) `test_ecart_etat_initial_causal_et_par_moteur` : la variable "ecart a l'etat initial" (option ecart_initial,
    notebook 04e) au cycle t ne depend ni des cycles > t ni des autres moteurs.

(a) et (b) portent sur les six modeles de la demo (reference naive, Random Forest, XGBoost, LightGBM, LSTM,
CNN 1D) et utilisent les vrais modeles de models/ : un modele dont les fichiers n'ont pas encore ete generes
(python scripts/preparer_demo.py) est saute, avec la raison. Rien n'est ecrit sur le disque. Pour tenir en moins
d'une minute, (a) compare un echantillon fixe : deux moteurs test par sous-jeu et cinq cycles par moteur.
"""

import os
import sys

import numpy as np
import pytest

# racine du depot (dossier parent de tests/) importable, pour "import src..." quel que soit le dossier de lancement
RACINE_DEPOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RACINE_DEPOT not in sys.path:
    sys.path.insert(0, RACINE_DEPOT)

# src.inference importe xgboost AVANT torch (ordre voulu, voir src/deep_learning.py)
from src import inference as inf  # noqa: E402
from src.contrat_predictions import SOUS_JEUX, chemin_metadonnees, chemin_predictions  # noqa: E402
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

TOLERANCE_EGALITE = 1e-4  # cycles : ecart maximal admis entre inference en direct et fichier precalcule
COMMANDE_PREPARATION = "python scripts/preparer_demo.py"

# l'objet d'inference du DERNIER sous-jeu charge : un seul a la fois en memoire (une Random Forest pese lourd)
_INFERENCE_COURANTE = {}

# les cas de (a), sous-jeu par sous-jeu : les six modeles d'un sous-jeu sont testes a la suite (un chargement)
CAS_DIRECT = []
for _sous_jeu in SOUS_JEUX:
    for _modele in inf.MODELES_INFERENCE:
        CAS_DIRECT.append((_sous_jeu, _modele))


def inference_reelle(sous_jeu):
    """
    Charge les vrais modeles de models/ pour un sous-jeu (garde en memoire tant qu'on reste sur ce sous-jeu).

    Rien n'est ecrit sur le disque : un modele a arbres absent est entraine en memoire, un reseau absent est
    indisponible.

    Parameters
    ----------
    sous_jeu : str
        "FD001" a "FD004".

    Returns
    -------
    dict
        Le resultat de `src.inference.charger_modeles`.
    """
    if _INFERENCE_COURANTE.get("sous_jeu") != sous_jeu:
        _INFERENCE_COURANTE.clear()  # libere les modeles du sous-jeu precedent
        _INFERENCE_COURANTE["sous_jeu"] = sous_jeu
        _INFERENCE_COURANTE["inference"] = inf.charger_modeles(sous_jeu, entrainer_si_absent=True, sauvegarder=False)
    return _INFERENCE_COURANTE["inference"]


def test_cycles_futurs_sans_effet():
    """(b) Mesures des cycles > t multipliees par 1000 ou effacees (NaN) : predictions au cycle t identiques."""
    inference = inference_reelle("FD001")
    moteur = inf.mesures_moteur_test("FD001", 1)
    mesures = moteur["mesures"]

    # les colonnes de mesure : 3 reglages et 21 capteurs (unit_number et time_cycles restent intacts)
    colonnes_mesures = []
    for colonne in COLUMNS:
        if colonne.startswith("sensor_") or colonne.startswith("op_setting_"):
            colonnes_mesures.append(colonne)

    for cycle in [1, 15, 31, moteur["nb_cycles"] - 1]:
        reference = inf.predire_au_cycle(inference, mesures, cycle)

        # futur multiplie par 1000 : un modele qui lirait le futur changerait completement d'avis
        absurdes = mesures.copy()
        masque_futur = absurdes["time_cycles"] > cycle
        absurdes.loc[masque_futur, colonnes_mesures] = absurdes.loc[masque_futur, colonnes_mesures] * 1000.0
        avec_absurdes = inf.predire_au_cycle(inference, absurdes, cycle)

        # futur efface (NaN) : la preparation refuserait des NaN si elle les recevait
        effacees = mesures.copy()
        effacees.loc[effacees["time_cycles"] > cycle, colonnes_mesures] = np.nan
        avec_effacees = inf.predire_au_cycle(inference, effacees, cycle)

        assert reference["nb_cycles_recus"] == cycle  # seuls les cycles 1..t ont ete transmis
        assert len(reference["predictions"]) > 0
        for modele in reference["predictions"]:
            assert avec_absurdes["predictions"][modele] == reference["predictions"][modele]
            assert avec_effacees["predictions"][modele] == reference["predictions"][modele]


@pytest.mark.parametrize("sous_jeu, modele", CAS_DIRECT)
def test_direct_egal_precalcule(sous_jeu, modele):
    """(a) Prediction en direct au cycle t (mesures 1..t seulement) = prediction precalculee, a 1e-4 pres."""
    # fichiers precalcules absents (sorties/ n'est pas versionne) : test saute, avec la commande a lancer
    if not os.path.exists(chemin_predictions(sous_jeu, modele)) or not os.path.exists(chemin_metadonnees(sous_jeu, modele)):
        pytest.skip(f"{sous_jeu}/{modele} : predictions precalculees absentes (lancer {COMMANDE_PREPARATION})")
    inference = inference_reelle(sous_jeu)
    if modele not in inference["modeles"]:
        pytest.skip(f"{sous_jeu}/{modele} : {inference['indisponibles'].get(modele)}")

    # premier et dernier moteur test ; cycles 1, 30, 31 (bord de la fenetre des reseaux), milieu et dernier
    moteurs = inf.lister_moteurs_test(sous_jeu)
    for moteur in [moteurs[0], moteurs[-1]]:
        nb_cycles = inf.mesures_moteur_test(sous_jeu, moteur)["nb_cycles"]
        cycles = []
        for cycle in [1, 30, 31, nb_cycles // 2, nb_cycles]:
            if 1 <= cycle <= nb_cycles and cycle not in cycles:
                cycles.append(cycle)
        tableau, absents = inf.ecarts_aux_predictions_precalculees(inference, moteur, cycles, modeles=[modele])
        assert len(absents) == 0, absents
        assert len(tableau) == len(cycles)  # une comparaison par cycle demande
        ecart_max = float(tableau["ecart_absolu"].max())
        assert ecart_max <= TOLERANCE_EGALITE, f"{sous_jeu}/{modele}, moteur {moteur} :\n{tableau.to_string()}"


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
