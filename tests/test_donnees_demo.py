"""
Tests de la couche d'acces aux donnees de l'application (src/donnees_demo.py) et de scripts/preparer_demo.py.

Les fichiers de predictions sont SYNTHETIQUES et ecrits dans un dossier temporaire (tmp_path de pytest) :
les vraies predictions de sorties/predictions/ ne sont jamais lues ni modifiees ici. L'ensemble tourne en
moins d'une seconde.
"""

import json
import os
import sys

import pandas as pd
import pytest

from src import donnees_demo as dd
from src.contrat_predictions import MODELES_CONNUS, calculer_rul_vraie, ecrire_predictions, metriques_dernier_cycle
from src.sous_jeux import DESCRIPTION_SOUS_JEUX


# ---------------------------------------------------------------------------------------------------
# Fabrication de petits fichiers de predictions synthetiques
# ---------------------------------------------------------------------------------------------------

def fabriquer_flotte():
    """
    Quatre moteurs test de 3 cycles, predictions constantes (seuil d'alerte 30, seuil de surestimation 20).

    - moteur 1 : vraie 10, predite 15 -> alerte correcte ;
    - moteur 2 : vraie 20, predite 60 -> alerte manquee (et surestimation de 40 > 20) ;
    - moteur 3 : vraie 80, predite 25 -> fausse alerte ;
    - moteur 4 : vraie 140, predite 120 -> rien a signaler.

    Returns
    -------
    pandas.DataFrame
        Les colonnes obligatoires du contrat.
    """
    rul_finales = [10, 20, 80, 140]
    predites_par_moteur = [15.0, 60.0, 25.0, 120.0]
    numeros = []
    cycles = []
    predites = []
    for position in range(len(rul_finales)):
        for cycle in [1, 2, 3]:
            numeros.append(position + 1)
            cycles.append(cycle)
            predites.append(predites_par_moteur[position])
    df = pd.DataFrame()
    df["unit_number"] = numeros
    df["time_cycles"] = cycles
    df["rul_predite"] = predites
    df["regime"] = 0
    return calculer_rul_vraie(df, rul_finales)


def ecrire_fichier(df, dossier, sous_jeu, modele):
    """
    Ecrit un fichier de predictions synthetique et des metadonnees coherentes dans un dossier.

    Parameters
    ----------
    df : pandas.DataFrame
        Les predictions.
    dossier : str
        Le dossier de sortie (temporaire).
    sous_jeu, modele : str
        Le couple a ecrire.

    Returns
    -------
    None
    """
    meta = {}
    meta["sous_jeu"] = sous_jeu
    meta["modele"] = modele
    meta["nom_affiche"] = MODELES_CONNUS[modele]
    meta["famille"] = "arbres"
    if modele in ["lstm", "cnn1d"]:
        meta["famille"] = "sequentiel"
    meta["metriques_test"] = metriques_dernier_cycle(df)
    meta["metriques_validation"] = {"rmse_cv_moyen": 16.5}
    meta["protocole_validation"] = "GroupKFold 5 plis par moteur (synthetique)"
    meta["duree_entrainement_s"] = 1.0
    meta["date"] = "2026-11-04"
    meta["source"] = "tests/test_donnees_demo.py"
    ecrire_predictions(df, meta, dossier=dossier)


# ---------------------------------------------------------------------------------------------------
# Inventaire, filtre des modeles affiches, comparaison, coherence
# ---------------------------------------------------------------------------------------------------

def test_modeles_affiches_filtrent_l_inventaire():
    """Un modele du contrat absent de MODELES_AFFICHES est ignore ; l'ordre est celui de MODELES_AFFICHES."""
    lignes = []
    for modele in ["cnn1d", "modele_non_affiche", "xgboost_phase3"]:
        lignes.append({"sous_jeu": "FD002", "modele": modele})
    lignes.append({"sous_jeu": "FD004", "modele": "modele_non_affiche"})
    inventaire = pd.DataFrame(lignes)
    attendus = []
    for modele in dd.MODELES_AFFICHES:
        if modele in ["cnn1d", "xgboost_phase3"]:
            attendus.append(modele)
    assert dd.modeles_disponibles(inventaire, "FD002") == attendus
    assert dd.modeles_disponibles(inventaire, "FD004") == []
    assert dd.sous_jeux_disponibles(inventaire) == ["FD002"]  # FD004 n'a aucun modele affiche


def test_inventaire_comparaison_et_coherence(tmp_path):
    """Comparaison d'un sous-jeu lue dans les metadonnees, metriques = contrat ; incoherence detectee."""
    dossier = str(tmp_path)
    df = fabriquer_flotte()
    for modele in dd.MODELES_AFFICHES:
        ecrire_fichier(df, dossier, "FD001", modele)
    ecrire_fichier(df, dossier, "FD003", dd.MODELES_AFFICHES[0])

    liste = dd.inventaire_predictions(dossier)
    assert dd.sous_jeux_disponibles(liste) == ["FD001", "FD003"]
    assert dd.modeles_disponibles(liste, "FD001") == dd.MODELES_AFFICHES

    tableau = dd.tableau_comparaison_sous_jeu(liste, "FD001", dossier)
    assert tableau["modele"].tolist() == dd.MODELES_AFFICHES
    contrat = metriques_dernier_cycle(df)
    premiere = tableau.iloc[0]
    assert premiere["nb_moteurs"] == 4
    assert premiere["rmse"] == pytest.approx(contrat["rmse"])
    assert premiere["score_nasa"] == pytest.approx(contrat["score_nasa"])
    assert premiere["nb_surestimes"] == 1  # moteur 2 : +40 cycles
    assert premiere["alertes_manquees"] == 1  # moteur 2
    assert bool(tableau["controle_coherent"].all())

    # metrique recopiee de travers dans un JSON : le controle le detecte
    chemin_json = os.path.join(dossier, f"FD001__{dd.MODELES_AFFICHES[0]}.json")
    with open(chemin_json, "r", encoding="utf-8") as fichier:
        meta = json.load(fichier)
    meta["metriques_test"]["rmse"] = meta["metriques_test"]["rmse"] + 1.0
    with open(chemin_json, "w", encoding="utf-8") as fichier:
        json.dump(meta, fichier)
    tableau = dd.tableau_comparaison_sous_jeu(dd.inventaire_predictions(dossier), "FD001", dossier)
    assert tableau["controle_coherent"].tolist()[0] is False

    # signature de dossier : change quand un fichier change (cle des caches de l'application)
    signature_avant = dd.signature_dossier(dossier)
    os.remove(chemin_json)
    assert dd.signature_dossier(dossier) != signature_avant
    assert dd.signature_dossier(os.path.join(dossier, "absent")) == tuple()
    assert dd.signature_fichier(chemin_json)[1] is None


def test_descriptions_seuils_et_libelles():
    """Une ligne par sous-jeu lue dans src/sous_jeux.py ; seuils lus dans le contrat ; un libelle par statut."""
    for sous_jeu in DESCRIPTION_SOUS_JEUX:
        texte = dd.description_sous_jeu(sous_jeu)
        assert texte.startswith(sous_jeu)
        assert str(DESCRIPTION_SOUS_JEUX[sous_jeu]["nb_moteurs_test"]) in texte
        assert "\n" not in texte
    assert "6 régimes de vol" in dd.description_sous_jeu("FD002")
    assert "2 modes de panne" in dd.description_sous_jeu("FD003")

    seuils = dd.seuils_metriques()
    assert seuils["seuil_surestimation"] > 0 and seuils["seuil_alerte"] > 0
    assert sorted(dd.LIBELLES_STATUTS.keys()) == sorted(dd.EXPLICATIONS_STATUTS.keys())
    assert dd.nom_modele("lstm") == MODELES_CONNUS["lstm"]
    assert dd.nom_modele("inconnu") == "inconnu"


# ---------------------------------------------------------------------------------------------------
# scripts/preparer_demo.py (sans lancer les vrais entrainements)
# ---------------------------------------------------------------------------------------------------

def test_preparer_demo_scripts_absents(tmp_path, monkeypatch, capsys):
    """
    Scripts de production absents : etapes sautees avec un message clair ; les etapes longues (artefacts) sont
    remplacees par un simulacre qui note les commandes ; etat final lu dans le dossier des predictions.
    """
    racine = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # dossier parent de tests/
    sys.path.insert(0, os.path.join(racine, "scripts"))  # pour "import preparer_demo"
    try:
        import preparer_demo
    finally:
        sys.path.remove(os.path.join(racine, "scripts"))

    ecrire_fichier(fabriquer_flotte(), str(tmp_path), "FD001", dd.MODELES_AFFICHES[0])
    monkeypatch.setenv("AERO_PREDICT_DOSSIER_PREDICTIONS", str(tmp_path))
    monkeypatch.setattr(preparer_demo, "SCRIPT_ARBRES", os.path.join(str(tmp_path), "absent_arbres.py"))
    monkeypatch.setattr(preparer_demo, "SCRIPT_DEEP_LEARNING", os.path.join(str(tmp_path), "absent_dl.py"))

    commandes_lancees = []

    def lancer_etape_simulee(nom, commande):
        """Simulacre de lancer_etape : note la commande au lieu de l'executer."""
        commandes_lancees.append(commande)
        return {"etape": nom, "statut": "reussie", "duree_s": 0.0}

    monkeypatch.setattr(preparer_demo, "lancer_etape", lancer_etape_simulee)

    code = preparer_demo.main(["--sous-jeux", "FD001"])
    sortie = capsys.readouterr().out
    assert code == 0
    assert sortie.count("n'existe pas encore : etape sautee") == 2
    assert commandes_lancees == [
        [sys.executable, "-m", "src.inference", "--forcer", "--sous-jeux", "FD001"],  # artefacts de l'inference
        [sys.executable, "-m", "src.demo", "entrainer"],
    ]
    assert "FD001" in sortie and "[coherent]" in sortie

    # --seulement-arbres : l'etape deep learning n'est pas lancee du tout
    preparer_demo.main(["--seulement-arbres"])
    sortie = capsys.readouterr().out
    assert sortie.count("n'existe pas encore : etape sautee") == 1
    assert "--seulement-arbres" in sortie
