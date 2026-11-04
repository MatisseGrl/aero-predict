"""
Tests de l'application Streamlit (app/tableau_de_bord.py, deux pages) avec streamlit.testing.v1.AppTest.

AppTest execute vraiment le script de chaque page (sans navigateur) : on verifie qu'aucune page ne leve
d'exception, que le tirage (graine fixee), la lecture, l'import d'un fichier d'exemple et la revelation de la
verite font ce qu'ils annoncent, et que la page de comparaison affiche les modeles de MODELES_AFFICHES.

Donnees, toujours dans des dossiers temporaires (sorties/ et models/ ne sont jamais lus ni modifies) :
- fichiers de predictions SYNTHETIQUES (page "Comparer les modeles"), designes par AERO_PREDICT_DOSSIER_PREDICTIONS ;
- modeles LEGERS pour FD001 (page "Tester un moteur"), designes par AERO_PREDICT_DOSSIER_MODELES : le XGBoost
  entraine avec la recette du script des arbres (moins d'une seconde) et deux reseaux NON entraines (poids au
  hasard, graine fixe) : on teste l'application, pas la qualite des modeles.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src import donnees_demo as dd
from src import inference as inf
from src.contrat_predictions import MODELES_CONNUS, calculer_rul_vraie, ecrire_predictions, metriques_dernier_cycle
from src.deep_learning import NOM_COLONNE_TIME_CYCLES_ECHELLE, creer_modele, sauvegarder_modele
from src.preprocessing import DATA_DIR
from src.sous_jeux import charger_sous_jeu

# point d'entree de l'application et chemins des pages (relatifs au point d'entree, comme st.Page)
CHEMIN_APP = os.path.join(inf.RACINE_DEPOT, "app", "tableau_de_bord.py")
PAGE_TESTER = "vues/prediction_en_direct.py"
PAGE_COMPARER = "vues/comparaison.py"
DOSSIER_EXEMPLES = os.path.join(inf.RACINE_DEPOT, "app", "exemples")

DELAI_MAX_S = 180  # duree maximale d'une execution de page dans AppTest (le premier import de torch est le plus long)

# cles de session de la page "Tester un moteur" (memes noms que app/vues/prediction_en_direct.py ; on ne peut pas
# importer la page : l'importer executerait tout son script hors de Streamlit)
CLE_GRAINE = "live_graine_tirage"
CLE_PREDICTIONS = "live_predictions"


# ---------------------------------------------------------------------------------------------------
# Donnees et modeles de test
# ---------------------------------------------------------------------------------------------------

def ecrire_predictions_synthetiques(dossier, sous_jeu, modele, nb_moteurs, decalage, graine):
    """
    Ecrit des predictions synthetiques pour les premiers moteurs test d'un sous-jeu (cycles reels).

    RUL predite = min(RUL vraie, 125) + decalage + bruit gaussien (ecart-type 5), clippee a 0.

    Parameters
    ----------
    dossier : str
        Le dossier temporaire des predictions.
    sous_jeu, modele : str
        Le couple a ecrire.
    nb_moteurs : int
        Nombre de moteurs test gardes (les premiers numeros).
    decalage : float
        Biais ajoute aux predictions (cycles).
    graine : int
        Graine du bruit (resultat reproductible).

    Returns
    -------
    None
    """
    brut = charger_sous_jeu(sous_jeu, "test")
    brut = brut[brut["unit_number"] <= nb_moteurs]
    chemin_rul = os.path.join(DATA_DIR, f"RUL_{sous_jeu}.txt")
    rul_finales = pd.read_csv(chemin_rul, sep=r"\s+", header=None)[0].to_numpy()[:nb_moteurs]

    df = brut[["unit_number", "time_cycles"]].copy()
    df["regime"] = 0
    df = calculer_rul_vraie(df, rul_finales)
    generateur = np.random.default_rng(graine)
    bruit = generateur.normal(0.0, 5.0, len(df))
    df["rul_predite"] = np.clip(df["rul_vraie_cappee"].to_numpy() + decalage + bruit, 0, None)

    meta = {}
    meta["sous_jeu"] = sous_jeu
    meta["modele"] = modele
    meta["nom_affiche"] = MODELES_CONNUS[modele]
    meta["famille"] = "arbres"
    if modele in inf.MODELES_RESEAUX:
        meta["famille"] = "sequentiel"
    meta["metriques_test"] = metriques_dernier_cycle(df)
    meta["metriques_validation"] = {"rmse_cv_moyen": 17.0}
    meta["protocole_validation"] = "Synthetique (tests/test_app.py)"
    meta["duree_entrainement_s"] = 0.5
    meta["date"] = "2026-11-04"
    meta["source"] = "tests/test_app.py"
    ecrire_predictions(df, meta, dossier=dossier)


@pytest.fixture(scope="module")
def dossier_synthetique(tmp_path_factory):
    """
    Dossier temporaire de predictions synthetiques FD001 : un fichier par modele affiche (MODELES_AFFICHES).

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Fabrique de dossiers temporaires de pytest.

    Returns
    -------
    str
        Le dossier.
    """
    dossier = str(tmp_path_factory.mktemp("predictions"))
    for position in range(len(dd.MODELES_AFFICHES)):  # biais differents : des barres differentes
        ecrire_predictions_synthetiques(dossier, "FD001", dd.MODELES_AFFICHES[position], 20, 3.0 * position, position)
    return dossier


@pytest.fixture(scope="module")
def dossier_modeles_legers(tmp_path_factory):
    """
    Dossier temporaire des modeles FD001 : XGBoost entraine ici, LSTM et CNN 1D non entraines.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Fabrique de dossiers temporaires de pytest.

    Returns
    -------
    str
        Le dossier.
    """
    dossier = str(tmp_path_factory.mktemp("modeles"))
    contexte = inf.preparer_contexte("FD001")
    colonnes = []
    for capteur in contexte["capteurs"]:
        colonnes.append(capteur)
    colonnes.append(NOM_COLONNE_TIME_CYCLES_ECHELLE)
    for type_modele in inf.MODELES_RESEAUX:
        reseau = creer_modele(type_modele, len(colonnes), graine=0)
        informations = {"sous_jeu": "FD001", "colonnes_entree": colonnes, "longueur_fenetre": 30}
        sauvegarder_modele(reseau, inf.chemin_poids_reseau("FD001", type_modele, dossier), type_modele,
                           len(colonnes), informations=informations)
    for modele in inf.MODELES_ARBRES:
        inf.obtenir_modele_arbres(contexte, modele, dossier=dossier)
    return dossier


@pytest.fixture
def environnement(monkeypatch, dossier_synthetique, dossier_modeles_legers):
    """
    Oriente l'application vers les predictions synthetiques et les modeles legers (variables d'environnement).

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Outil pytest de modification temporaire de l'environnement.
    dossier_synthetique, dossier_modeles_legers : str
        Les fixtures ci-dessus.

    Returns
    -------
    dict
        "predictions" et "modeles" : les deux dossiers.
    """
    monkeypatch.setenv("AERO_PREDICT_DOSSIER_PREDICTIONS", dossier_synthetique)
    monkeypatch.setenv(inf.VARIABLE_ENV_DOSSIER_MODELES, dossier_modeles_legers)
    dossiers = {}
    dossiers["predictions"] = dossier_synthetique
    dossiers["modeles"] = dossier_modeles_legers
    return dossiers


def ouvrir_page(page, etat=None):
    """
    Lance l'application dans AppTest et affiche la page demandee.

    Parameters
    ----------
    page : str
        Le chemin de la page, relatif a app/ (ex. "vues/comparaison.py").
    etat : dict ou None, optionnel
        Valeurs a placer dans st.session_state avant la premiere execution (ex. la graine du tirage).

    Returns
    -------
    streamlit.testing.v1.AppTest
        L'application apres execution de la page.
    """
    application = AppTest.from_file(CHEMIN_APP, default_timeout=DELAI_MAX_S)
    if etat is not None:
        for cle in etat:
            application.session_state[cle] = etat[cle]
    application.run()
    if page != PAGE_TESTER:  # la page "Tester un moteur" est la page par defaut
        application.switch_page(page)
        application.run()
    return application


def textes(application, type_element):
    """
    Rassemble le texte de tous les elements d'un type (warning, error, info, caption, markdown, subheader...).

    Parameters
    ----------
    application : streamlit.testing.v1.AppTest
        L'application executee.
    type_element : str
        Le nom de l'attribut AppTest (ex. "warning").

    Returns
    -------
    str
        Les textes, separes par des retours a la ligne.
    """
    morceaux = []
    for element in getattr(application, type_element):
        morceaux.append(str(element.value))
    return "\n".join(morceaux)


def valeur_metrique(application, libelle):
    """
    Renvoie la valeur affichee d'un st.metric a partir de son libelle.

    Parameters
    ----------
    application : streamlit.testing.v1.AppTest
        L'application executee.
    libelle : str
        Le libelle exact du st.metric.

    Returns
    -------
    str
        La valeur affichee.
    """
    for metrique in application.metric:
        if metrique.label == libelle:
            return metrique.value
    raise AssertionError(f"metrique {libelle!r} introuvable")


def cycle_recu(application):
    """
    Lit le cycle courant dans la carte "Cycle recu" ("12 / 203" -> 12).

    Parameters
    ----------
    application : streamlit.testing.v1.AppTest
        L'application executee.

    Returns
    -------
    int
        Le dernier cycle recu.
    """
    texte = valeur_metrique(application, "Cycle reçu")
    return int(texte.split("/")[0].strip())


def contenu_graphique(application):
    """
    Renvoie la description JSON (texte) de la courbe de la page "Tester un moteur", telle qu'envoyee au navigateur.

    Parameters
    ----------
    application : streamlit.testing.v1.AppTest
        L'application executee.

    Returns
    -------
    str
        La figure Plotly serialisee.
    """
    graphiques = application.get("plotly_chart")
    assert len(graphiques) == 1
    return graphiques[0].proto.spec


# ---------------------------------------------------------------------------------------------------
# Les deux pages s'executent sans exception
# ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("page", [PAGE_TESTER, PAGE_COMPARER])
def test_chaque_page_sans_exception(environnement, page):
    """Chaque page : aucune exception, un seul titre."""
    application = ouvrir_page(page)
    assert len(application.exception) == 0, application.exception
    assert len(application.title) == 1


# ---------------------------------------------------------------------------------------------------
# Page "Comparer les modeles"
# ---------------------------------------------------------------------------------------------------

def test_comparaison_trois_modeles_affiches(environnement):
    """Un tableau des modeles affiches (ordre de MODELES_AFFICHES) et un graphique ; le filtre est teste dans test_donnees_demo."""
    application = ouvrir_page(PAGE_COMPARER)
    assert len(application.exception) == 0, application.exception
    assert len(application.dataframe) == 1
    tableau = application.dataframe[0].value
    attendus = []
    for modele in dd.MODELES_AFFICHES:
        attendus.append(MODELES_CONNUS[modele])
    assert tableau["Modèle"].tolist() == attendus
    assert len(application.get("plotly_chart")) == 1

    # RMSE du tableau = metadonnees du fichier
    with open(os.path.join(environnement["predictions"], "FD001__lstm.json"), encoding="utf-8") as fichier:
        meta = json.load(fichier)
    rmse_affiche = float(tableau[tableau["Modèle"] == MODELES_CONNUS["lstm"]]["RMSE test (cycles)"].iloc[0])
    assert rmse_affiche == round(meta["metriques_test"]["rmse"], 2)


def test_comparaison_sans_fichier(monkeypatch, tmp_path, dossier_modeles_legers):
    """Aucun fichier de predictions : un message avec la commande a lancer, pas d'exception."""
    monkeypatch.setenv("AERO_PREDICT_DOSSIER_PREDICTIONS", str(tmp_path / "vide"))
    monkeypatch.setenv(inf.VARIABLE_ENV_DOSSIER_MODELES, dossier_modeles_legers)
    application = ouvrir_page(PAGE_COMPARER)
    assert len(application.exception) == 0, application.exception
    assert "scripts/preparer_demo.py" in textes(application, "warning")


# ---------------------------------------------------------------------------------------------------
# Page "Tester un moteur"
# ---------------------------------------------------------------------------------------------------

def test_tirage_lecture_puis_revelation(environnement):
    """
    Tirage avec graine : "moteur mystere", verite absente de la page ET de la courbe ; Lecture : les predictions
    affichees sont exactement celles de `predire_au_cycle` ; revelation : numero, RUL vraie et statuts.
    """
    graine = 3
    application = ouvrir_page(PAGE_TESTER, etat={CLE_GRAINE: graine})
    assert "Tirer un moteur au hasard" in textes(application, "info")
    application.button(key="live_tirer").click().run()
    assert len(application.exception) == 0, application.exception

    attendu = inf.tirer_moteur_test("FD001", graine=graine)  # meme graine : meme moteur
    assert "Moteur mystère" in textes(application, "subheader")
    assert f"n° {attendu['moteur']}" not in textes(application, "subheader")
    assert valeur_metrique(application, "Cycle reçu") == f"0 / {attendu['nb_cycles']}"

    # Lecture : AppTest execute la zone animee une fois, donc un seul pas de quelques cycles
    application.button(key="live_bouton_lecture").click().run()
    assert len(application.exception) == 0, application.exception
    pas = cycle_recu(application)
    assert 1 <= pas < attendu["nb_cycles"]
    assert application.button(key="live_bouton_lecture").label == "Pause"
    application.button(key="live_bouton_lecture").click().run()  # Pause
    assert application.button(key="live_bouton_lecture").label == "Lecture"

    # les valeurs de la page = l'inference en direct, cycle par cycle
    inference = inf.charger_modeles("FD001", dossier=environnement["modeles"], entrainer_si_absent=False)
    predictions_page = application.session_state[CLE_PREDICTIONS]
    for cycle in range(1, pas + 1):
        directe = inf.predire_au_cycle(inference, attendu["mesures"], cycle)
        for modele in inf.MODELES_INFERENCE:
            assert predictions_page[modele][cycle - 1] == directe["predictions"][modele]

    # verite cachee : ni texte, ni courbe, ni info-bulle
    assert "RUL vraie" not in textes(application, "markdown")
    assert "RUL vraie" not in contenu_graphique(application)

    # revelation
    application.button(key="live_bouton_verite").click().run()
    assert len(application.exception) == 0, application.exception
    assert f"Moteur test n° {attendu['moteur']}" in textes(application, "subheader")
    verite = inf.verite_moteur(attendu["mesures"], attendu["rul_finale"])
    rul_au_cycle = int(verite["rul_vraie"].iloc[pas - 1])
    assert f"RUL vraie au cycle {pas} : {rul_au_cycle} cycles" in textes(application, "markdown")
    assert "RUL vraie" in contenu_graphique(application)
    assert application.button(key="live_bouton_verite").label == "Masquer la vraie RUL"

    # Debut : on repart de zero
    application.button(key="live_bouton_debut").click().run()
    assert valeur_metrique(application, "Cycle reçu") == f"0 / {attendu['nb_cycles']}"


def test_choix_par_numero(environnement):
    """Choisir un numero charge ce moteur test (identite visible, verite toujours cachee)."""
    application = ouvrir_page(PAGE_TESTER)
    application.selectbox(key="live_numero").set_value(5).run()
    assert len(application.exception) == 0, application.exception
    assert "Moteur test n° 5" in textes(application, "subheader")
    nb_cycles = inf.mesures_moteur_test("FD001", 5)["nb_cycles"]
    assert valeur_metrique(application, "Cycle reçu") == f"0 / {nb_cycles}"
    assert application.button(key="live_bouton_verite").label == "Révéler la vraie RUL"


def test_import_fichier_exemple_et_revelation(environnement):
    """Import de app/exemples/mesures_FD001.txt avec sa RUL vraie : lecture, puis revelation de la verite."""
    with open(os.path.join(DOSSIER_EXEMPLES, "mesures_FD001.txt"), "rb") as fichier:
        contenu = fichier.read()
    mesures = inf.lire_fichier_mesures(contenu)
    moteur = int(mesures["unit_number"].iloc[0])
    rul_finale = inf.mesures_moteur_test("FD001", moteur)["rul_finale"]  # celle annoncee dans LISEZMOI.md

    application = ouvrir_page(PAGE_TESTER)
    application.file_uploader(key="live_fichier").set_value(("mesures_FD001.txt", contenu, "text/plain"))
    application.number_input(key="live_rul_import").set_value(int(rul_finale))
    application.button(key="live_importer").click().run()
    assert len(application.exception) == 0, application.exception
    assert "Fichier importé : mesures_FD001.txt" in textes(application, "subheader")

    application.button(key="live_bouton_lecture").click().run()
    assert cycle_recu(application) > 0
    application.button(key="live_bouton_verite").click().run()
    assert len(application.exception) == 0, application.exception
    assert f"au dernier cycle ({len(mesures)}) : {int(rul_finale)}" in textes(application, "markdown")


def test_import_fichier_invalide_et_sans_verite(environnement):
    """Fichier invalide : message clair ; fichier valide sans RUL : bouton de revelation desactive."""
    application = ouvrir_page(PAGE_TESTER)
    application.file_uploader(key="live_fichier").set_value(("faux.txt", b"1 1 2 3\n", "text/plain"))
    application.button(key="live_importer").click().run()
    assert len(application.exception) == 0, application.exception
    assert "4 valeurs au lieu de 26" in textes(application, "error")

    with open(os.path.join(DOSSIER_EXEMPLES, "mesures_FD001.txt"), "rb") as fichier:
        contenu = fichier.read()
    application.file_uploader(key="live_fichier").set_value(("mesures_FD001.txt", contenu, "text/plain"))
    application.button(key="live_importer").click().run()
    assert len(application.exception) == 0, application.exception
    assert application.button(key="live_bouton_verite").disabled


def test_changer_de_sous_jeu_et_modele_indisponible(environnement):
    """Un moteur test est oublie si on change de sous-jeu ; poids absents : avertissement clair, page utilisable."""
    application = ouvrir_page(PAGE_TESTER)
    application.button(key="live_tirer").click().run()
    application.selectbox(key="live_sous_jeu").set_value("FD003").run()  # pas de reseau FD003 dans le dossier
    assert len(application.exception) == 0, application.exception
    assert "Tirer un moteur au hasard" in textes(application, "info")
    avertissements = textes(application, "warning")
    assert "LSTM indisponible" in avertissements and "absent" in avertissements
    application.button(key="live_tirer").click().run()  # le XGBoost seul suffit pour tester
    application.button(key="live_bouton_lecture").click().run()
    assert len(application.exception) == 0, application.exception
