"""
Tests de l'application Streamlit (app/tableau_de_bord.py) avec streamlit.testing.v1.AppTest.

AppTest execute vraiment le script de chaque page (sans navigateur) : on verifie qu'aucune page ne
leve d'exception, que les messages attendus s'affichent et que les commandes du rejeu (Lecture, Pause,
Debut, Fin) font avancer le cycle comme prevu.

Donnees : des fichiers de predictions SYNTHETIQUES ecrits dans un dossier temporaire, designe par la
variable d'environnement AERO_PREDICT_DOSSIER_PREDICTIONS (les vraies predictions de
sorties/predictions/ ne sont jamais lues) ; un artefact XGBoost leger (30 arbres) ecrit dans un dossier
temporaire et designe par AERO_PREDICT_ARTEFACT (models/modele_retenu.joblib n'est jamais touche).
Les predictions synthetiques suivent les vrais moteurs test (memes numeros, memes cycles, vraie RUL de
RUL_FD00X.txt) pour que le panneau des capteurs bruts ait des donnees.
"""

import os

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src import donnees_demo as dd
from src.contrat_predictions import MODELES_CONNUS, calculer_rul_vraie, ecrire_predictions, metriques_dernier_cycle
from src.demo import entrainer_modele_retenu, sauvegarder_artefact

# point d'entree de l'application et chemins des pages (relatifs au point d'entree, comme st.Page)
CHEMIN_APP = os.path.join(dd.RACINE_DEPOT, "app", "tableau_de_bord.py")
PAGE_ACCUEIL = "vues/accueil.py"
PAGE_MOTEUR = "vues/moteur_en_direct.py"
PAGE_FLOTTE = "vues/flotte.py"
PAGE_COMPARAISON = "vues/comparaison.py"
PAGE_SIMULATION = "vues/simulation.py"
PAGE_FIABILITE = "vues/fiabilite.py"
TOUTES_LES_PAGES = [PAGE_ACCUEIL, PAGE_MOTEUR, PAGE_FLOTTE, PAGE_COMPARAISON, PAGE_SIMULATION, PAGE_FIABILITE]

DELAI_MAX_S = 120  # duree maximale d'une execution de page dans AppTest (le premier import est le plus long)

# configuration legere : on teste l'application, pas la qualite du modele
CONFIG_RAPIDE = {
    "famille": "xgboost",
    "perte": "native",
    "parametres_perte": {},
    "hyperparametres": {"n_estimators": 30, "max_depth": 4, "learning_rate": 0.1, "random_state": 42},
    "nom": "XGBoost test rapide",
    "regle_de_choix": "tests de l'application",
}


# ---------------------------------------------------------------------------------------------------
# Donnees synthetiques
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
    brut = dd.charger_donnees_brutes_test(sous_jeu)
    brut = brut[brut["unit_number"] <= nb_moteurs]
    chemin_rul = os.path.join(dd.DATA_DIR, f"RUL_{sous_jeu}.txt")
    rul_finales = pd.read_csv(chemin_rul, sep=r"\s+", header=None)[0].to_numpy()[:nb_moteurs]

    df = brut[["unit_number", "time_cycles"]].copy()
    if sous_jeu in ["FD002", "FD004"]:
        df["regime"] = df["time_cycles"] % 6  # regimes fictifs : exerce l'affichage multi-regimes
    else:
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
    meta["metriques_test"] = metriques_dernier_cycle(df)
    meta["metriques_validation"] = {"rmse_cv_moyen": 17.0, "nasa_cv_moyen": 22000.0}
    meta["protocole_validation"] = "Synthetique (tests/test_app.py)"
    meta["duree_entrainement_s"] = 0.5
    meta["date"] = "2026-11-04"
    meta["source"] = "tests/test_app.py"
    ecrire_predictions(df, meta, dossier=dossier)


@pytest.fixture(scope="module")
def chemin_artefact_leger(contexte_train, tmp_path_factory):
    """
    Entraine un artefact leger et l'ecrit dans un dossier temporaire (une fois pour ce module).

    Parameters
    ----------
    contexte_train : dict
        La fixture de session de tests/conftest.py (train FD001 prepare).
    tmp_path_factory : pytest.TempPathFactory
        Fabrique de dossiers temporaires de pytest.

    Returns
    -------
    str
        Le chemin du fichier .joblib.
    """
    artefact = entrainer_modele_retenu(CONFIG_RAPIDE, contexte_train=contexte_train)
    chemin = str(tmp_path_factory.mktemp("artefact") / "modele_test.joblib")
    sauvegarder_artefact(artefact, chemin)
    return chemin


@pytest.fixture(scope="module")
def dossier_synthetique(tmp_path_factory):
    """
    Dossier temporaire avec trois fichiers de predictions synthetiques (FD001 x 2 modeles, FD002 x 1).

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
    ecrire_predictions_synthetiques(dossier, "FD001", "xgboost_phase3", 20, 3.0, 1)
    ecrire_predictions_synthetiques(dossier, "FD001", "xgboost_retenu", 20, -3.0, 2)
    ecrire_predictions_synthetiques(dossier, "FD002", "xgboost_retenu", 8, 0.0, 3)
    return dossier


@pytest.fixture
def avec_fichiers(monkeypatch, dossier_synthetique, chemin_artefact_leger):
    """
    Oriente l'application vers les fichiers synthetiques et l'artefact leger (variables d'environnement).

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Outil pytest de modification temporaire de l'environnement.
    dossier_synthetique, chemin_artefact_leger : str
        Les fixtures ci-dessus.

    Returns
    -------
    str
        Le dossier des predictions.
    """
    monkeypatch.setenv(dd.VARIABLE_ENV_ARTEFACT, chemin_artefact_leger)
    monkeypatch.setenv("AERO_PREDICT_DOSSIER_PREDICTIONS", dossier_synthetique)
    return dossier_synthetique


@pytest.fixture
def sans_fichiers(monkeypatch, tmp_path, chemin_artefact_leger):
    """
    Oriente l'application vers un dossier de predictions VIDE (l'artefact leger reste disponible).

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Outil pytest de modification temporaire de l'environnement.
    tmp_path : pathlib.Path
        Dossier temporaire propre a ce test.
    chemin_artefact_leger : str
        La fixture de l'artefact leger.

    Returns
    -------
    str
        Le dossier vide.
    """
    dossier = str(tmp_path / "vide")
    monkeypatch.setenv(dd.VARIABLE_ENV_ARTEFACT, chemin_artefact_leger)
    monkeypatch.setenv("AERO_PREDICT_DOSSIER_PREDICTIONS", dossier)
    return dossier


def ouvrir_page(page):
    """
    Lance l'application dans AppTest et affiche la page demandee.

    Parameters
    ----------
    page : str
        Le chemin de la page, relatif a app/ (ex. "vues/flotte.py").

    Returns
    -------
    streamlit.testing.v1.AppTest
        L'application apres execution de la page.
    """
    application = AppTest.from_file(CHEMIN_APP, default_timeout=DELAI_MAX_S)
    application.run()
    if page != PAGE_ACCUEIL:
        application.switch_page(page)
        application.run()
    return application


def textes_affiches(application, type_element):
    """
    Rassemble le texte de tous les elements d'un type (warning, success, error, info, caption...).

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


# ---------------------------------------------------------------------------------------------------
# Chaque page s'execute sans exception
# ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("page", TOUTES_LES_PAGES)
def test_chaque_page_avec_fichiers_synthetiques(avec_fichiers, page):
    """Avec des fichiers synthetiques : aucune exception, un titre, aucun message d'absence de fichiers."""
    application = ouvrir_page(page)
    assert len(application.exception) == 0, application.exception
    assert len(application.title) == 1
    assert "Aucun fichier de prédictions" not in textes_affiches(application, "warning")


@pytest.mark.parametrize("page", [PAGE_MOTEUR, PAGE_FLOTTE, PAGE_COMPARAISON])
def test_pages_sans_aucun_fichier(sans_fichiers, page):
    """Sans fichier de predictions : message clair avec la commande a lancer, pas d'exception."""
    application = ouvrir_page(page)
    assert len(application.exception) == 0, application.exception
    message = textes_affiches(application, "warning")
    assert "Aucun fichier de prédictions" in message
    assert "scripts/preparer_demo.py" in message


@pytest.mark.parametrize("page", [PAGE_ACCUEIL, PAGE_SIMULATION, PAGE_FIABILITE])
def test_pages_independantes_des_predictions(sans_fichiers, page):
    """Accueil, "Et si..." et Fiabilite fonctionnent sans fichier de predictions (artefact seulement)."""
    application = ouvrir_page(page)
    assert len(application.exception) == 0, application.exception
    if page == PAGE_ACCUEIL:
        assert valeur_metrique(application, "Fichiers de prédictions disponibles") == "0"


# ---------------------------------------------------------------------------------------------------
# Page "Moteur en direct" : rejeu
# ---------------------------------------------------------------------------------------------------

def test_moteur_en_direct_rejeu(avec_fichiers):
    """
    Par defaut : moteur le plus proche de la panne, au dernier cycle. Lecture : retour au premier cycle
    puis avance d'un pas ; Pause : le cycle ne bouge plus ; Debut / Fin : bornes ; fin du rejeu : arret.
    """
    application = ouvrir_page(PAGE_MOTEUR)
    df, meta = dd.charger_predictions("FD001", "xgboost_retenu", avec_fichiers)
    moteur_attendu = dd.moteur_le_plus_proche_de_la_panne(df)
    assert application.selectbox(key="direct_moteur").value == moteur_attendu

    trajectoire = dd.extraire_trajectoire(df, moteur_attendu)
    premier = int(trajectoire["time_cycles"].min())
    dernier = int(trajectoire["time_cycles"].max())
    curseur = application.slider(key="direct_cycle")
    assert curseur.value == dernier
    pas = application.segmented_control(key="direct_vitesse").value

    # Lecture depuis la fin : on repart du premier cycle, puis la zone animee avance d'un pas
    application.button(key="bouton_lecture").click().run()
    assert len(application.exception) == 0, application.exception
    assert application.slider(key="direct_cycle").value == premier + pas
    assert application.button(key="bouton_lecture").label == "Pause"

    # Pause : plus d'avance
    application.button(key="bouton_lecture").click().run()
    assert application.button(key="bouton_lecture").label == "Lecture"
    assert application.slider(key="direct_cycle").value == premier + pas

    # Fin puis Debut
    application.button(key="bouton_fin").click().run()
    assert application.slider(key="direct_cycle").value == dernier
    application.button(key="bouton_debut").click().run()
    assert application.slider(key="direct_cycle").value == premier

    # a un pas de la fin, Lecture atteint le dernier cycle et s'arrete toute seule
    application.slider(key="direct_cycle").set_value(dernier - 1).run()
    application.button(key="bouton_lecture").click().run()
    assert application.slider(key="direct_cycle").value == dernier
    assert application.button(key="bouton_lecture").label == "Lecture"

    # verite masquee : ni RUL vraie, ni statut
    application.toggle(key="direct_verite").set_value(False).run()
    assert valeur_metrique(application, "RUL vraie") == "masquée"
    assert len(application.exception) == 0, application.exception


def test_moteur_en_direct_sous_jeu_multi_regimes(avec_fichiers):
    """Sur FD002 (regimes synthetiques), la page s'affiche et le panneau capteurs signale les regimes."""
    application = ouvrir_page(PAGE_MOTEUR)
    application.selectbox(key="direct_sous_jeu").set_value("FD002").run()
    assert len(application.exception) == 0, application.exception
    assert "plusieurs régimes de vol" in textes_affiches(application, "caption")


# ---------------------------------------------------------------------------------------------------
# Pages "Flotte", "Et si...", "Fiabilite"
# ---------------------------------------------------------------------------------------------------

def test_flotte_coherence_et_seuil(avec_fichiers):
    """Le controle de coherence est affiche ; changer le seuil change les alertes, pas le RMSE."""
    application = ouvrir_page(PAGE_FLOTTE)
    assert "Contrôle de cohérence" in textes_affiches(application, "success")
    rmse_avant = valeur_metrique(application, "RMSE (cycles)")
    alertes_avant = valeur_metrique(application, "Alertes émises")

    application.slider(key="seuil_alerte").set_value(100).run()
    assert len(application.exception) == 0, application.exception
    assert valeur_metrique(application, "RMSE (cycles)") == rmse_avant
    assert int(valeur_metrique(application, "Alertes émises")) > int(alertes_avant)


def test_simulation_derive(avec_fichiers):
    """Passer en derive additive et changer le capteur ne leve pas d'exception ; 5 cartes indicateurs."""
    application = ouvrir_page(PAGE_SIMULATION)
    assert len(application.metric) == 5
    application.radio(key="sim_type").set_value("decalage").run()
    assert len(application.exception) == 0, application.exception
    application.selectbox(key="sim_capteur").set_value("sensor_4").run()
    application.toggle(key="sim_progressive").set_value(True).run()
    assert len(application.exception) == 0, application.exception


def test_fiabilite_mesure_des_temps(avec_fichiers):
    """Le bouton de mesure lance mesurer_benchmark (10 moteurs, 1 repetition) et affiche p95 contre la cible."""
    application = ouvrir_page(PAGE_FIABILITE)
    application.selectbox(key="fiabilite_nb_moteurs").set_value(10).run()
    application.button(key="fiabilite_mesurer").click().run()
    assert len(application.exception) == 0, application.exception
    messages = textes_affiches(application, "success") + textes_affiches(application, "error")
    assert "p95 =" in messages
    assert "10 moteurs" in messages
    # l'artefact leger n'a pas la configuration retenue : l'ecart est signale, pas cache
    assert valeur_metrique(application, "Config. vs 04_config_retenue.json") == "DIFFÉRENTE"
