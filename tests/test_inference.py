"""
Tests de l'inference en direct (src/inference.py) : lecture des fichiers importes, absence de triche, blocage.

Trois preuves pour le jury, chacune dans un test nomme (noms affiches par l'application) :

1. `test_inference_en_direct_egale_predictions_precalculees` : pour chaque sous-jeu et chaque modele dont les
   fichiers existent, la prediction calculee EN DIRECT au cycle t (mesures 1..t seulement) est egale, a 1e-4
   cycle pres, a la prediction du fichier precalcule sorties/predictions/<SJ>__<modele>.csv.gz au meme cycle ;
2. `test_cycles_futurs_sans_effet` : remplacer les mesures des cycles > t par n'importe quoi ne change pas la
   prediction au cycle t (le modele ne lit jamais le futur) ;
3. `test_xgboost_et_reseaux_dans_le_meme_processus_sans_blocage` : un sous-processus enchaine XGBoost, LSTM,
   CNN puis XGBoost (multi-coeurs compris) et doit finir en moins de 60 s.

Les tests qui n'ont pas besoin des vrais modeles utilisent des modeles LEGERS ecrits dans un dossier
temporaire : le XGBoost entraine sur train_FD001 (recette du script des arbres, moins d'une seconde) et deux
reseaux NON entraines (poids tires au hasard, graine fixe). Le dossier models/ n'est jamais modifie.
"""

import json
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from src import inference as inf
from src.contrat_predictions import SOUS_JEUX, chemin_predictions, chemin_metadonnees
from src.deep_learning import NOM_COLONNE_TIME_CYCLES_ECHELLE, creer_modele, sauvegarder_modele
from src.preprocessing import COLUMNS
from src.sous_jeux import charger_rul_sous_jeu, charger_sous_jeu

TOLERANCE_EGALITE = 1e-4  # cycles : ecart maximal admis entre inference en direct et fichier precalcule
DELAI_MAX_SOUS_PROCESSUS_S = 60  # au-dela, le test anti-blocage echoue (un blocage OpenMP ne finit jamais)
DOSSIER_EXEMPLES = os.path.join(inf.RACINE_DEPOT, "app", "exemples")


# ---------------------------------------------------------------------------------------------------
# Aides et fixtures
# ---------------------------------------------------------------------------------------------------

def ecrire_reseaux_non_entraines(dossier, sous_jeu, capteurs):
    """
    Ecrit dans `dossier` un LSTM et un CNN 1D non entraines (poids au hasard, graine 0), au format de src/deep_learning.py.

    Parameters
    ----------
    dossier : str
        Le dossier des modeles (temporaire).
    sous_jeu : str
        Le sous-jeu ecrit dans les informations du fichier.
    capteurs : list de str
        Les capteurs prepares du sous-jeu (entrees du reseau, puis time_cycles mis a l'echelle).

    Returns
    -------
    None
    """
    colonnes = []
    for capteur in capteurs:
        colonnes.append(capteur)
    colonnes.append(NOM_COLONNE_TIME_CYCLES_ECHELLE)
    for type_modele in inf.MODELES_RESEAUX:
        reseau = creer_modele(type_modele, len(colonnes), graine=0)
        informations = {"sous_jeu": sous_jeu, "colonnes_entree": colonnes, "longueur_fenetre": 30}
        sauvegarder_modele(reseau, inf.chemin_poids_reseau(sous_jeu, type_modele, dossier), type_modele,
                           len(colonnes), informations=informations)


@pytest.fixture(scope="module")
def dossier_modeles_legers(tmp_path_factory):
    """
    Dossier temporaire avec les trois modeles FD001 : XGBoost entraine ici, reseaux non entraines.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Fabrique de dossiers temporaires de pytest.

    Returns
    -------
    str
        Le dossier.
    """
    dossier = str(tmp_path_factory.mktemp("modeles_legers"))
    contexte = inf.preparer_contexte("FD001")
    ecrire_reseaux_non_entraines(dossier, "FD001", contexte["capteurs"])
    for modele in inf.MODELES_ARBRES:  # entraines puis sauvegardes dans le dossier temporaire
        inf.obtenir_modele_arbres(contexte, modele, dossier=dossier)
    return dossier


@pytest.fixture(scope="module")
def inference_legere(dossier_modeles_legers):
    """
    L'objet d'inference FD001 construit sur les modeles legers (rien n'est entraine : tout est relu).

    Parameters
    ----------
    dossier_modeles_legers : str
        La fixture ci-dessus.

    Returns
    -------
    dict
        Le resultat de `charger_modeles("FD001", dossier=...)`.
    """
    return inf.charger_modeles("FD001", dossier=dossier_modeles_legers, entrainer_si_absent=False)


# ---------------------------------------------------------------------------------------------------
# Preuve 1 : inference en direct = predictions precalculees
# ---------------------------------------------------------------------------------------------------

# un objet d'inference par sous-jeu, construit une seule fois pour tout le module
_INFERENCES_REELLES = {}


def inference_reelle(sous_jeu):
    """
    Charge (une fois par sous-jeu) les VRAIS modeles de models/ ; un XGBoost absent ou perime est entraine en
    memoire seulement (rien n'est ecrit), avec la recette du script des arbres.

    Parameters
    ----------
    sous_jeu : str
        Le sous-jeu.

    Returns
    -------
    dict
        Le resultat de `charger_modeles`.
    """
    if sous_jeu not in _INFERENCES_REELLES:
        _INFERENCES_REELLES[sous_jeu] = inf.charger_modeles(
            sous_jeu, dossier=inf.DOSSIER_MODELES_DEFAUT, entrainer_si_absent=True, sauvegarder=False
        )
    return _INFERENCES_REELLES[sous_jeu]


def moteurs_et_cycles_a_comparer(sous_jeu):
    """
    Choisit les moteurs (premier, milieu, dernier du jeu de test) et les cycles (debut, milieu, fin) compares.

    Parameters
    ----------
    sous_jeu : str
        Le sous-jeu.

    Returns
    -------
    list de (int, list de int)
        (moteur, cycles) ; les cycles 1, 2, 30, 31 encadrent le remplissage de la fenetre des reseaux.
    """
    test = charger_sous_jeu(sous_jeu, "test")
    moteurs = np.sort(test["unit_number"].unique())
    nb_cycles = test.groupby("unit_number").size()
    choix = []
    for moteur in [int(moteurs[0]), int(moteurs[len(moteurs) // 2]), int(moteurs[-1])]:
        dernier = int(nb_cycles.loc[moteur])
        cycles = []
        for cycle in [1, 2, 30, 31, dernier // 2, dernier]:
            if 1 <= cycle <= dernier and cycle not in cycles:
                cycles.append(cycle)
        choix.append((moteur, cycles))
    return choix


@pytest.mark.parametrize("modele", inf.MODELES_INFERENCE)
@pytest.mark.parametrize("sous_jeu", SOUS_JEUX)
def test_inference_en_direct_egale_predictions_precalculees(sous_jeu, modele):
    """
    Au cycle t, inference en direct (mesures 1..t) = fichier precalcule, a 1e-4 cycle pres.

    Prouve a la fois que les courbes precalculees sont exactement ce que donne l'inference en direct et que le
    modele n'a besoin d'aucune donnee future (le fichier, lui, a ete calcule sur toute la trajectoire).
    """
    # fichiers precalcules absents (ignores par git, produits par scripts/preparer_demo.py) : test saute
    if not os.path.exists(chemin_predictions(sous_jeu, modele)) or not os.path.exists(chemin_metadonnees(sous_jeu, modele)):
        pytest.skip(f"{sous_jeu}/{modele} : fichier precalcule absent (python scripts/preparer_demo.py)")
    inference = inference_reelle(sous_jeu)
    if modele not in inference["modeles"]:
        pytest.skip(f"{sous_jeu}/{modele} : modele indisponible ({inference['indisponibles'].get(modele)})")

    for moteur, cycles in moteurs_et_cycles_a_comparer(sous_jeu):
        tableau, absents = inf.ecarts_aux_predictions_precalculees(inference, moteur, cycles, modeles=[modele])
        assert len(absents) == 0, absents
        assert len(tableau) == len(cycles)  # une comparaison par cycle demande
        ecart_max = float(tableau["ecart_absolu"].max())
        assert ecart_max <= TOLERANCE_EGALITE, (
            f"{sous_jeu}/{modele}, moteur {moteur} : ecart max {ecart_max:.3g} cycle\n{tableau.to_string()}"
        )


# ---------------------------------------------------------------------------------------------------
# Preuve 2 : les cycles futurs ne changent rien
# ---------------------------------------------------------------------------------------------------

def test_cycles_futurs_sans_effet(inference_legere):
    """Mesures des cycles > t remplacees par des valeurs absurdes (ou NaN) : predictions au cycle t identiques."""
    moteur = inf.mesures_moteur_test("FD001", 1)
    mesures = moteur["mesures"]
    capteurs = []
    for colonne in COLUMNS:
        if colonne.startswith("sensor_") or colonne.startswith("op_setting_"):
            capteurs.append(colonne)

    for cycle in [1, 15, 31, moteur["nb_cycles"] - 1]:
        reference = inf.predire_au_cycle(inference_legere, mesures, cycle)

        # futur multiplie par 1000 : un modele qui lirait le futur changerait completement d'avis
        absurdes = mesures.copy()
        masque_futur = absurdes["time_cycles"] > cycle
        absurdes.loc[masque_futur, capteurs] = absurdes.loc[masque_futur, capteurs] * 1000.0
        avec_absurdes = inf.predire_au_cycle(inference_legere, absurdes, cycle)

        # futur efface (NaN) : la preparation refuserait des NaN si elle les recevait
        manquantes = mesures.copy()
        manquantes.loc[manquantes["time_cycles"] > cycle, capteurs] = np.nan
        avec_manquantes = inf.predire_au_cycle(inference_legere, manquantes, cycle)

        for modele in reference["predictions"]:
            assert avec_absurdes["predictions"][modele] == reference["predictions"][modele]
            assert avec_manquantes["predictions"][modele] == reference["predictions"][modele]
        assert reference["nb_cycles_recus"] == cycle


def test_predire_au_cycle_et_trajectoire(inference_legere):
    """Resultat complet (3 modeles, temps en ms), RUL >= 0, et trajectoire = predire_au_cycle cycle par cycle."""
    mesures = inf.mesures_moteur_test("FD001", 2)["mesures"]
    resultat = inf.predire_au_cycle(inference_legere, mesures, 40)
    assert sorted(resultat["predictions"].keys()) == sorted(inf.MODELES_INFERENCE)
    for modele in resultat["predictions"]:
        assert resultat["predictions"][modele] >= 0.0
        assert resultat["temps_ms"][modele] > 0.0
    assert resultat["temps_preparation_ms"] > 0.0
    assert resultat["temps_total_ms"] >= resultat["temps_preparation_ms"]

    trajectoire = inf.predire_trajectoire(inference_legere, mesures, cycles=[10, 40], modeles=["xgboost_phase3", "lstm"])
    assert trajectoire["time_cycles"].tolist() == [10, 40]
    assert trajectoire["xgboost_phase3"].iloc[1] == resultat["predictions"]["xgboost_phase3"]
    assert trajectoire["lstm"].iloc[1] == resultat["predictions"]["lstm"]

    # cycle absent des mesures et modele inconnu : erreurs claires
    with pytest.raises(ValueError, match="absent"):
        inf.predire_au_cycle(inference_legere, mesures, int(mesures["time_cycles"].max()) + 5)
    with pytest.raises(ValueError, match="indisponible"):
        inf.predire_au_cycle(inference_legere, mesures, 10, modeles=["transformer"])


def test_reseau_lot_de_copies_egal_au_calcul_en_lot(inference_legere):
    """
    La fenetre du cycle t, calculee dans un lot de copies d'elle-meme, donne le resultat du calcul en lot
    de toutes les fenetres du moteur (celui des fichiers precalcules), a 1e-9 cycle pres.
    """
    from src.deep_learning import ajouter_time_cycles_echelle, construire_fenetres, predire
    from src.sous_jeux import appliquer_preparation

    mesures = inf.mesures_moteur_test("FD001", 3)["mesures"]
    preparees = ajouter_time_cycles_echelle(appliquer_preparation(inference_legere["contexte"], mesures))
    for modele in inf.MODELES_RESEAUX:
        entree = inference_legere["modeles"][modele]
        fenetres, _, _, cycles = construire_fenetres(preparees, entree["colonnes_entree"], longueur=entree["longueur_fenetre"])
        en_lot = predire(entree["objet"], fenetres)  # toutes les fenetres du moteur d'un coup
        for position in [0, 29, len(cycles) - 1]:
            en_direct = inf.predire_au_cycle(inference_legere, mesures, int(cycles[position]), modeles=[modele])
            assert abs(en_direct["predictions"][modele] - float(en_lot[position])) <= 1e-9


def test_xgboost_un_coeur_memes_predictions(dossier_modeles_legers):
    """n_jobs=1 (inference en direct) et n_jobs=4 (entrainement du script) donnent les memes predictions."""
    contexte = inf.preparer_contexte("FD001")
    artefact, _ = inf.obtenir_modele_arbres(contexte, "xgboost_phase3", dossier=dossier_modeles_legers,
                                            entrainer_si_absent=False)
    modele = artefact["modele"]
    X_test = contexte["test"][contexte["feature_cols"]]
    sur_un_coeur = modele.predict(X_test)  # regle a 1 coeur par obtenir_modele_arbres
    assert modele.get_params()["n_jobs"] == 1
    modele.set_params(n_jobs=4)
    sur_quatre_coeurs = modele.predict(X_test)
    modele.set_params(n_jobs=1)
    assert np.array_equal(sur_un_coeur, sur_quatre_coeurs)


# ---------------------------------------------------------------------------------------------------
# Chargement des modeles
# ---------------------------------------------------------------------------------------------------

def test_charger_modeles_reseau_absent_et_artefact_perime(tmp_path, dossier_modeles_legers):
    """Poids absents : modele indisponible avec un message clair ; artefact perime : reentraine, pas reutilise."""
    dossier_vide = str(tmp_path / "vide")
    inference = inf.charger_modeles("FD001", modeles=["lstm", "xgboost_phase3"], dossier=dossier_vide,
                                    entrainer_si_absent=True, sauvegarder=False)
    assert "lstm" in inference["indisponibles"]
    assert "absent" in inference["indisponibles"]["lstm"]
    assert "scripts/entrainer_deep_learning.py" in inference["indisponibles"]["lstm"]
    assert "xgboost_phase3" in inference["modeles"]  # entraine en memoire
    assert len(inference["messages"]) == 1
    assert not os.path.exists(inf.chemin_artefact_arbres("FD001", "xgboost_phase3", dossier_vide))  # rien d'ecrit

    # sans entrainement autorise : l'absence est signalee, pas cachee
    sans = inf.charger_modeles("FD001", modeles=["xgboost_phase3"], dossier=dossier_vide, entrainer_si_absent=False)
    assert "xgboost_phase3" in sans["indisponibles"]

    # artefact relu a l'identique, puis declare perime si sa configuration ne correspond plus
    contexte = inf.preparer_contexte("FD001")
    chemin = inf.chemin_artefact_arbres("FD001", "xgboost_phase3", dossier_modeles_legers)
    artefact = inf.charger_artefact_arbres(chemin)
    assert inf.raison_artefact_perime(artefact, contexte, "xgboost_phase3") == ""
    artefact["config"]["hyperparametres"]["max_depth"] = 99
    assert "configuration" in inf.raison_artefact_perime(artefact, contexte, "xgboost_phase3")


def test_contexte_appris_sur_le_train_seulement(inference_legere):
    """Les scalers ont vu exactement les lignes du train (ni plus, ni moins) : aucun reajustement sur le test."""
    contexte = inference_legere["contexte"]
    train = charger_sous_jeu("FD001", "train")
    total = 0
    for regime in contexte["nb_lignes_vues_par_scaler"]:
        total = total + contexte["nb_lignes_vues_par_scaler"][regime]
    assert total == len(train)
    assert contexte["nb_lignes_train"] == len(train)


# ---------------------------------------------------------------------------------------------------
# Moteurs du jeu de test, verite
# ---------------------------------------------------------------------------------------------------

def test_tirage_reproductible_et_verite():
    """Meme graine = meme moteur ; RUL vraie = RUL_FD00X.txt au dernier cycle, puis +1 par cycle en remontant."""
    premier = inf.tirer_moteur_test("FD002", graine=7)
    second = inf.tirer_moteur_test("FD002", graine=7)
    assert premier["moteur"] == second["moteur"]
    assert premier["moteur"] in inf.lister_moteurs_test("FD002")
    assert premier["rul_finale"] == float(charger_rul_sous_jeu("FD002")[premier["moteur"] - 1])

    verite = inf.verite_moteur(premier["mesures"], premier["rul_finale"])
    assert len(verite) == premier["nb_cycles"]
    assert float(verite["rul_vraie"].iloc[-1]) == premier["rul_finale"]
    assert np.all(np.diff(verite["rul_vraie"].to_numpy()) == -1.0)
    assert float(verite["rul_vraie_cappee"].max()) <= 125.0

    with pytest.raises(ValueError, match="n'existe pas"):
        inf.mesures_moteur_test("FD001", 10_000)


# ---------------------------------------------------------------------------------------------------
# Fichiers importes
# ---------------------------------------------------------------------------------------------------

def test_lire_fichier_mesures_brut_et_csv():
    """Format NASA brut et CSV (colonnes melangees, points-virgules) : memes mesures que test_FD001.txt."""
    attendu = inf.mesures_moteur_test("FD001", 5)["mesures"]
    brut = "\n".join(inf.lignes_brutes_moteur("FD001", 5)) + "\n\n"
    lu_brut = inf.lire_fichier_mesures(brut.encode("utf-8"))
    pd.testing.assert_frame_equal(lu_brut, attendu, check_dtype=False)

    colonnes_melangees = list(reversed(COLUMNS))
    csv_virgules = attendu[colonnes_melangees].to_csv(index=False)
    pd.testing.assert_frame_equal(inf.lire_fichier_mesures(csv_virgules), attendu, check_dtype=False)
    csv_points_virgules = attendu.to_csv(index=False, sep=";")
    pd.testing.assert_frame_equal(inf.lire_fichier_mesures(csv_points_virgules), attendu, check_dtype=False)


def fabriquer_ligne(moteur, cycle, nb_valeurs=26):
    """
    Fabrique une ligne au format NASA brut : moteur, cycle, puis des valeurs 1.0.

    Parameters
    ----------
    moteur, cycle : int
        Les deux premieres valeurs.
    nb_valeurs : int, optionnel
        Nombre total de valeurs (26 par defaut).

    Returns
    -------
    str
        La ligne.
    """
    valeurs = [str(moteur), str(cycle)]
    for _ in range(nb_valeurs - 2):
        valeurs.append("1.0")
    return " ".join(valeurs)


@pytest.mark.parametrize(
    "contenu, morceau_du_message",
    [
        (b"", "vide"),
        (b"\xff\xfe\x00\x81binaire", "texte"),
        ((fabriquer_ligne(1, 1) + "\n" + fabriquer_ligne(1, 2, nb_valeurs=25)).encode(), "Ligne 2 : 25 valeurs au lieu de 26"),
        (fabriquer_ligne(1, 1).replace("1.0", "abc", 1).encode(), "n'est pas un nombre"),
        (fabriquer_ligne(1, 1).replace("1.0", "nan", 1).encode(), "non finie"),
        ((fabriquer_ligne(1, 1) + "\n" + fabriquer_ligne(2, 1)).encode(), "2 moteurs"),
        (fabriquer_ligne(1, 2).encode(), "cycle 2 au lieu de 1"),
        ((fabriquer_ligne(1, 1) + "\n" + fabriquer_ligne(1, 3)).encode(), "Ligne 2 : cycle 3 au lieu de 2"),
        (("unit_number,time_cycles\n1,1\n").encode(), "Colonnes manquantes"),
        ((",".join(COLUMNS) + ",RUL\n").encode(), "Colonnes inconnues dans l'en-tête : RUL"),
        ((",".join(COLUMNS) + "\n").encode(), "Aucune ligne de mesures"),
    ],
)
def test_lire_fichier_mesures_erreurs(contenu, morceau_du_message):
    """Chaque fichier invalide donne une ErreurFichierMesures dont le message dit quoi corriger."""
    with pytest.raises(inf.ErreurFichierMesures) as erreur:
        inf.lire_fichier_mesures(contenu)
    assert morceau_du_message in str(erreur.value)


def test_lire_fichier_mesures_trop_long():
    """Plus de NB_MAX_CYCLES_FICHIER lignes : refuse (un moteur C-MAPSS a au plus quelques centaines de cycles)."""
    lignes = []
    for cycle in range(1, inf.NB_MAX_CYCLES_FICHIER + 2):
        lignes.append(fabriquer_ligne(1, cycle))
    with pytest.raises(inf.ErreurFichierMesures, match="au plus"):
        inf.lire_fichier_mesures("\n".join(lignes))


def test_fichiers_exemples_extraits_exacts_du_jeu_de_test():
    """
    app/exemples/ : un fichier par sous-jeu, lisible par `lire_fichier_mesures`, identique aux lignes du moteur
    annonce dans LISEZMOI.md, et RUL vraie annoncee = RUL_FD00X.txt (le fichier lui-meme ne contient pas la RUL).
    """
    with open(os.path.join(DOSSIER_EXEMPLES, "LISEZMOI.md"), "r", encoding="utf-8") as fichier:
        lisezmoi = fichier.read()
    # lignes du tableau : | FD001 | `mesures_FD001.txt` | moteur | cycles | RUL vraie |
    lignes_tableau = re.findall(r"\|\s*(FD00\d)\s*\|\s*`([^`]+)`\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|", lisezmoi)
    assert len(lignes_tableau) == len(SOUS_JEUX)
    for sous_jeu, nom_fichier, moteur, nb_cycles, rul in lignes_tableau:
        with open(os.path.join(DOSSIER_EXEMPLES, nom_fichier), "rb") as fichier:
            contenu = fichier.read()
        mesures = inf.lire_fichier_mesures(contenu)
        attendu = inf.mesures_moteur_test(sous_jeu, int(moteur))
        pd.testing.assert_frame_equal(mesures, attendu["mesures"], check_dtype=False)
        assert contenu.decode("utf-8").splitlines() == inf.lignes_brutes_moteur(sous_jeu, int(moteur))
        assert attendu["nb_cycles"] == int(nb_cycles)
        assert attendu["rul_finale"] == float(rul)
        assert int(moteur) == inf.choisir_moteur_exemple(sous_jeu)  # regle de choix documentee


# ---------------------------------------------------------------------------------------------------
# Preuve 3 : torch et xgboost dans le meme processus, sans blocage
# ---------------------------------------------------------------------------------------------------

# Script lance dans un sous-processus NEUF : il enchaine XGBoost (entrainement et prediction sur 4 coeurs,
# la situation qui precedait le blocage constate), inference en direct des 3 modeles cycle par cycle,
# LSTM et CNN sur tout un moteur, puis de nouveau XGBoost sur 4 coeurs. Il ecrit ses temps en JSON.
SCRIPT_ANTI_BLOCAGE = r"""
import json, os, sys, tempfile, time
sys.path.insert(0, os.getcwd())
debut = time.perf_counter()
from src import inference as inf
from src.deep_learning import NOM_COLONNE_TIME_CYCLES_ECHELLE, creer_modele, sauvegarder_modele, predire
from src.deep_learning import ajouter_time_cycles_echelle, construire_fenetres
from src.config_modele import construire_fabrique_depuis_config
from src.sous_jeux import appliquer_preparation

dossier = tempfile.mkdtemp()
contexte = inf.preparer_contexte("FD001")
colonnes = list(contexte["capteurs"]) + [NOM_COLONNE_TIME_CYCLES_ECHELLE]
for type_modele in inf.MODELES_RESEAUX:
    reseau = creer_modele(type_modele, len(colonnes), graine=0)
    sauvegarder_modele(reseau, inf.chemin_poids_reseau("FD001", type_modele, dossier), type_modele, len(colonnes),
                       informations={"sous_jeu": "FD001", "colonnes_entree": colonnes, "longueur_fenetre": 30})

# 1. XGBoost sur 4 coeurs (entrainement + prediction de tout le test) AVANT tout calcul PyTorch
xgb = construire_fabrique_depuis_config(inf.configuration_arbres("xgboost_phase3"))()
xgb.fit(contexte["X"], contexte["y"])
xgb.predict(contexte["test"][contexte["feature_cols"]])
temps = {"xgboost_4_coeurs_avant_s": time.perf_counter() - debut}

# 2. inference en direct des 3 modeles (XGBoost -> LSTM -> CNN a chaque cycle)
inference = inf.charger_modeles("FD001", dossier=dossier)
mesures = inf.mesures_moteur_test("FD001", 2)["mesures"]  # moteur test 2 : 49 cycles observes
durees = []
for cycle in range(1, 41):
    resultat = inf.predire_au_cycle(inference, mesures, cycle)
    durees.append(resultat["temps_total_ms"])
temps["pas_median_ms"] = sorted(durees)[len(durees) // 2]

# 3. LSTM puis CNN sur toutes les fenetres du moteur (un seul lot)
preparees = ajouter_time_cycles_echelle(appliquer_preparation(inference["contexte"], mesures))
for type_modele in inf.MODELES_RESEAUX:
    fenetres, _, _, _ = construire_fenetres(preparees, colonnes, longueur=30)
    predire(inference["modeles"][type_modele]["objet"], fenetres)

# 4. de nouveau XGBoost sur 4 coeurs, APRES PyTorch (entrainement + prediction)
xgb2 = construire_fabrique_depuis_config(inf.configuration_arbres("xgboost_phase3"))()
xgb2.fit(contexte["X"], contexte["y"])
xgb2.predict(contexte["test"][contexte["feature_cols"]])
temps["total_s"] = time.perf_counter() - debut
temps["reglages"] = inf.configurer_calcul_un_coeur()
print("RESULTAT " + json.dumps(temps))
"""


@pytest.mark.parametrize("omp_un_coeur", [True, False], ids=["OMP_NUM_THREADS=1 (application)", "sans variable"])
def test_xgboost_et_reseaux_dans_le_meme_processus_sans_blocage(omp_un_coeur):
    """
    Sous-processus XGBoost (4 coeurs) -> inference en direct XGBoost / LSTM / CNN -> XGBoost (4 coeurs) : fini en
    moins de 60 s, avec ou sans OMP_NUM_THREADS=1 (l'application le fixe ; la suite de tests ne le fixe pas).
    """
    environnement = dict(os.environ)
    environnement.pop("OMP_NUM_THREADS", None)
    if omp_un_coeur:
        environnement["OMP_NUM_THREADS"] = "1"
    try:
        execution = subprocess.run(
            [sys.executable, "-c", SCRIPT_ANTI_BLOCAGE], cwd=inf.RACINE_DEPOT, env=environnement,
            capture_output=True, text=True, timeout=DELAI_MAX_SOUS_PROCESSUS_S,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"Blocage : le sous-processus n'a pas fini en {DELAI_MAX_SOUS_PROCESSUS_S} s.")
    assert execution.returncode == 0, execution.stderr[-3000:]
    resultat = None
    for ligne in execution.stdout.splitlines():
        if ligne.startswith("RESULTAT "):
            resultat = json.loads(ligne[len("RESULTAT "):])
    assert resultat is not None, execution.stdout[-2000:]
    assert resultat["total_s"] < DELAI_MAX_SOUS_PROCESSUS_S
    assert resultat["reglages"]["torch_coeurs"] == 1
    assert resultat["reglages"]["torch_fils_inter_operations"] == 1
