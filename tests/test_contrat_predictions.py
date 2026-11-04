"""
Tests du contrat des fichiers de predictions (src/contrat_predictions.py), sur de petits exemples.

Tous les fichiers sont ecrits dans un dossier temporaire (tmp_path de pytest) : les vraies predictions
de sorties/predictions/ ne sont jamais touchees.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

from src.contrat_predictions import (
    COLONNES_OBLIGATOIRES,
    DOSSIER_PREDICTIONS,
    MODELES_CONNUS,
    SOUS_JEUX,
    VARIABLE_ENV_DOSSIER,
    calculer_rul_vraie,
    chemin_metadonnees,
    chemin_predictions,
    dossier_par_defaut,
    ecrire_predictions,
    extraire_dernier_cycle,
    lire_predictions,
    lister_predictions,
    metriques_dernier_cycle,
    valider_metadonnees,
    valider_predictions,
    verifier_coherence,
)
from src.evaluation import score_nasa


def fabriquer_predictions():
    """
    Fabrique un petit fichier de predictions a 2 moteurs, calculable a la main.

    Moteur 1 : 3 cycles, RUL finale 10 -> RUL vraie 12, 11, 10 ; predictions 20, 18, 15 (ecart final +5).
    Moteur 2 : 2 cycles, RUL finale 130 -> RUL vraie 131, 130 (plafonnee a 125) ; predictions 100, 100
    (ecart final -30).

    Returns
    -------
    pandas.DataFrame
        Les predictions, colonnes obligatoires du contrat.
    """
    df = pd.DataFrame()
    df["unit_number"] = [1, 1, 1, 2, 2]
    df["time_cycles"] = [1, 2, 3, 1, 2]
    df["rul_predite"] = [20.0, 18.0, 15.0, 100.0, 100.0]
    df["regime"] = [0, 0, 0, 0, 0]
    df = calculer_rul_vraie(df, [10, 130])  # ajoute rul_vraie et rul_vraie_cappee
    return df


def fabriquer_metadonnees(df, sous_jeu="FD001", modele="xgboost_phase3"):
    """
    Fabrique des metadonnees valides pour les predictions donnees.

    Parameters
    ----------
    df : pandas.DataFrame
        Les predictions (pour remplir les metriques de test).
    sous_jeu : str, optionnel
        Le sous-jeu.
    modele : str, optionnel
        La cle du modele.

    Returns
    -------
    dict
        Des metadonnees conformes au contrat.
    """
    meta = {}
    meta["sous_jeu"] = sous_jeu
    meta["modele"] = modele
    meta["nom_affiche"] = MODELES_CONNUS[modele]
    meta["famille"] = "arbres"
    meta["metriques_test"] = metriques_dernier_cycle(df)
    meta["metriques_validation"] = {"rmse_moyen": np.float64(16.2)}  # type numpy : doit etre converti
    meta["protocole_validation"] = "GroupKFold 5 plis par moteur"
    meta["duree_entrainement_s"] = np.float32(1.5)
    meta["date"] = "2026-11-04"
    meta["source"] = "tests/test_contrat_predictions.py"
    return meta


def test_constantes_du_contrat():
    """Les constantes ont exactement les valeurs du contrat partage entre les agents."""
    assert SOUS_JEUX == ["FD001", "FD002", "FD003", "FD004"]
    assert list(MODELES_CONNUS.keys()) == ["xgboost_phase3", "lstm", "cnn1d"]
    assert MODELES_CONNUS["xgboost_phase3"] == "XGBoost Phase 3"
    assert COLONNES_OBLIGATOIRES == ["unit_number", "time_cycles", "rul_predite", "rul_vraie", "rul_vraie_cappee", "regime"]
    assert os.path.isabs(DOSSIER_PREDICTIONS)
    assert DOSSIER_PREDICTIONS.endswith(os.path.join("sorties", "predictions"))


def test_chemins(tmp_path):
    """Les noms de fichiers suivent <SOUS_JEU>__<modele>.csv.gz / .json ; un nom inconnu est refuse."""
    assert chemin_predictions("FD002", "lstm", tmp_path) == os.path.join(str(tmp_path), "FD002__lstm.csv.gz")
    assert chemin_metadonnees("FD002", "lstm", tmp_path) == os.path.join(str(tmp_path), "FD002__lstm.json")
    with pytest.raises(ValueError, match="Sous-jeu inconnu"):
        chemin_predictions("FD005", "lstm", tmp_path)
    with pytest.raises(ValueError, match="Modele inconnu"):
        chemin_predictions("FD001", "transformer", tmp_path)


def test_variable_environnement(monkeypatch, tmp_path):
    """La variable d'environnement remplace le dossier par defaut ; sans elle, on revient au defaut."""
    monkeypatch.setenv(VARIABLE_ENV_DOSSIER, str(tmp_path))
    assert dossier_par_defaut() == str(tmp_path)
    assert chemin_predictions("FD001", "lstm") == os.path.join(str(tmp_path), "FD001__lstm.csv.gz")
    monkeypatch.delenv(VARIABLE_ENV_DOSSIER)
    assert dossier_par_defaut() == DOSSIER_PREDICTIONS


def test_calculer_rul_vraie():
    """rul_vraie = RUL finale + (dernier cycle - cycle), plafonnee a 125 ; formats liste et dict equivalents."""
    df = fabriquer_predictions()
    assert df["rul_vraie"].tolist() == [12.0, 11.0, 10.0, 131.0, 130.0]
    assert df["rul_vraie_cappee"].tolist() == [12.0, 11.0, 10.0, 125.0, 125.0]
    # meme resultat avec un dictionnaire moteur -> RUL finale
    df_dict = calculer_rul_vraie(df[["unit_number", "time_cycles"]], {1: 10, 2: 130})
    assert df_dict["rul_vraie"].tolist() == df["rul_vraie"].tolist()
    # mauvais nombre de valeurs : erreur explicite
    with pytest.raises(ValueError, match="valeurs de RUL finale"):
        calculer_rul_vraie(df, [10])


def test_aller_retour_ecrire_lire(tmp_path):
    """Ecrire puis relire redonne les memes predictions (triees) et des metadonnees en types JSON simples."""
    df = fabriquer_predictions()
    df["colonne_libre"] = [1, 2, 3, 4, 5]  # colonne supplementaire : autorisee et conservee
    melange = df.sample(frac=1.0, random_state=0)  # ordre des lignes melange : l'ecriture doit retrier
    meta = fabriquer_metadonnees(df)

    chemin_csv, chemin_json = ecrire_predictions(melange, meta, dossier=tmp_path / "sous_dossier")
    assert os.path.isfile(chemin_csv) and chemin_csv.endswith("FD001__xgboost_phase3.csv.gz")
    assert os.path.isfile(chemin_json)

    relu, meta_relue = lire_predictions("FD001", "xgboost_phase3", dossier=tmp_path / "sous_dossier")
    # memes valeurs, dans l'ordre (moteur, cycle), colonnes obligatoires en premier
    assert relu.columns.tolist()[:6] == COLONNES_OBLIGATOIRES
    assert relu["colonne_libre"].tolist() == [1, 2, 3, 4, 5]
    pd.testing.assert_frame_equal(relu[COLONNES_OBLIGATOIRES], df[COLONNES_OBLIGATOIRES], check_dtype=False)
    # types numpy convertis, informations ajoutees automatiquement
    assert meta_relue["metriques_validation"]["rmse_moyen"] == pytest.approx(16.2)
    assert meta_relue["duree_entrainement_s"] == pytest.approx(1.5)
    assert meta_relue["nb_lignes"] == 5 and meta_relue["nb_moteurs"] == 2
    assert meta_relue["version_contrat"] == 1
    # le JSON est du JSON strict (pas de NaN, lisible par n'importe quel outil)
    with open(chemin_json, "r", encoding="utf-8") as fichier:
        assert json.load(fichier)["sous_jeu"] == "FD001"
    # le dictionnaire de l'appelant n'a pas ete modifie
    assert "nb_lignes" not in meta


@pytest.mark.parametrize(
    "modification, message",
    [
        ("colonne_manquante", "Colonnes obligatoires manquantes : regime"),
        ("nan", "valeur\\(s\\) manquante"),
        ("negative", "rul_predite sont negatives"),
        ("doublon", "doublon"),
        ("cappee_fausse", "rul_vraie_cappee doit valoir"),
        ("regime_reel", "regime"),
    ],
)
def test_valider_predictions_erreurs(modification, message):
    """Chaque violation du contrat leve une ValueError avec un message en francais explicite."""
    df = fabriquer_predictions()
    if modification == "colonne_manquante":
        df = df.drop(columns=["regime"])
    elif modification == "nan":
        df.loc[1, "rul_predite"] = np.nan
    elif modification == "negative":
        df.loc[0, "rul_predite"] = -1.0
    elif modification == "doublon":
        df.loc[1, "time_cycles"] = 1  # le moteur 1 a deux lignes au cycle 1
    elif modification == "cappee_fausse":
        df.loc[3, "rul_vraie_cappee"] = 131.0  # devrait valoir 125
    elif modification == "regime_reel":
        df["regime"] = 0.5
    with pytest.raises(ValueError, match=message):
        valider_predictions(df)


def test_valider_predictions_accepte_un_fichier_correct():
    """Un fichier conforme ne leve aucune erreur."""
    valider_predictions(fabriquer_predictions())


@pytest.mark.parametrize(
    "cle, valeur, message",
    [
        ("sous_jeu", "FD009", "Sous-jeu inconnu"),
        ("modele", "random_forest", "Modele inconnu"),
        ("famille", "lineaire", "Famille inconnue"),
        ("date", "04/11/2026", "AAAA-MM-JJ"),
        ("metriques_validation", None, "metriques_validation"),
        ("duree_entrainement_s", -3, "positive"),
    ],
)
def test_valider_metadonnees_erreurs(cle, valeur, message):
    """Sous-jeu ou modele inconnu, famille invalide, date mal formee... : ValueError en francais."""
    meta = fabriquer_metadonnees(fabriquer_predictions())
    meta[cle] = valeur
    with pytest.raises(ValueError, match=message):
        valider_metadonnees(meta)


def test_valider_metadonnees_cles_manquantes():
    """Une cle obligatoire absente (ou une metrique de test absente) est signalee par son nom."""
    meta = fabriquer_metadonnees(fabriquer_predictions())
    del meta["source"]
    with pytest.raises(ValueError, match="Cles de metadonnees manquantes : source"):
        valider_metadonnees(meta)

    meta = fabriquer_metadonnees(fabriquer_predictions())
    del meta["metriques_test"]["score_nasa"]
    with pytest.raises(ValueError, match="Metriques de test manquantes : score_nasa"):
        valider_metadonnees(meta)


def test_ecrire_refuse_des_predictions_invalides(tmp_path):
    """L'ecriture valide avant d'ecrire : rien n'est cree si le contrat n'est pas respecte."""
    df = fabriquer_predictions()
    meta = fabriquer_metadonnees(df)
    df.loc[0, "rul_predite"] = -5.0
    with pytest.raises(ValueError):
        ecrire_predictions(df, meta, dossier=tmp_path)
    assert not os.path.exists(chemin_predictions("FD001", "xgboost_phase3", tmp_path))


def test_lister_predictions(tmp_path):
    """La liste suit l'ordre SOUS_JEUX puis MODELES_CONNUS ; fichiers incomplets ou etrangers ignores."""
    df = fabriquer_predictions()
    # ecrits volontairement dans le desordre
    ecrire_predictions(df, fabriquer_metadonnees(df, "FD002", "cnn1d"), dossier=tmp_path)
    ecrire_predictions(df, fabriquer_metadonnees(df, "FD001", "lstm"), dossier=tmp_path)
    ecrire_predictions(df, fabriquer_metadonnees(df, "FD001", "xgboost_phase3"), dossier=tmp_path)
    # un CSV sans JSON et un fichier au nom hors contrat : ignores
    df.to_csv(os.path.join(str(tmp_path), "FD003__lstm.csv.gz"), index=False, compression="gzip")
    with open(os.path.join(str(tmp_path), "notes.txt"), "w", encoding="utf-8") as fichier:
        fichier.write("rien a voir")

    liste = lister_predictions(tmp_path)
    couples = []
    for _, ligne in liste.iterrows():
        couples.append((ligne["sous_jeu"], ligne["modele"]))
    assert couples == [("FD001", "xgboost_phase3"), ("FD001", "lstm"), ("FD002", "cnn1d")]
    assert liste.loc[0, "nom_affiche"] == "XGBoost Phase 3"
    assert liste.columns.tolist() == [
        "sous_jeu", "modele", "nom_affiche", "famille", "chemin_predictions", "chemin_metadonnees",
    ]


def test_lister_predictions_dossier_absent(tmp_path):
    """Dossier inexistant : DataFrame vide, avec les bonnes colonnes (pas d'exception)."""
    liste = lister_predictions(tmp_path / "inexistant")
    assert len(liste) == 0
    assert "sous_jeu" in liste.columns


def test_metriques_dernier_cycle_calcul_a_la_main():
    """
    Au dernier cycle : moteur 1 (vrai 10, predit 15, ecart +5) ; moteur 2 (vrai 130, predit 100, ecart -30).

    RMSE = sqrt((25 + 900) / 2) ; contre la RUL plafonnee : vrai 125 pour le moteur 2 -> ecart -25.
    Alerte (seuil 30) : moteur 1 vrai 10 <= 30 et predit 15 <= 30 -> bonne alerte, rien de manque.
    """
    df = fabriquer_predictions()
    metriques = metriques_dernier_cycle(df)
    assert metriques["nb_moteurs"] == 2
    assert metriques["rmse"] == pytest.approx(np.sqrt((25 + 900) / 2))
    assert metriques["rmse_contre_rul_plafonnee"] == pytest.approx(np.sqrt((25 + 625) / 2))
    assert metriques["mae"] == pytest.approx((5 + 30) / 2)
    assert metriques["score_nasa"] == pytest.approx((np.exp(5 / 10) - 1) + (np.exp(30 / 13) - 1))
    assert metriques["score_nasa"] == pytest.approx(score_nasa([10, 130], [15, 100]))
    assert metriques["nb_surestimes"] == 0
    assert metriques["alertes_manquees"] == 0
    assert metriques["fausses_alertes"] == 0

    # avec un seuil d'alerte de 12 : moteur 1 vrai 10 <= 12 mais predit 15 > 12 -> alerte manquee
    assert metriques_dernier_cycle(df, seuil_alerte=12)["alertes_manquees"] == 1
    # avec un seuil de surestimation de 4 : l'ecart +5 compte
    assert metriques_dernier_cycle(df, seuil_surestimation=4)["nb_surestimes"] == 1


def test_extraire_dernier_cycle_et_coherence():
    """Le dernier cycle est bien la ligne de cycle maximal ; la coherence detecte une metrique fausse."""
    df = fabriquer_predictions()
    derniers = extraire_dernier_cycle(df.sample(frac=1.0, random_state=1))
    assert derniers["unit_number"].tolist() == [1, 2]
    assert derniers["time_cycles"].tolist() == [3, 2]

    meta = fabriquer_metadonnees(df)
    controle = verifier_coherence(df, meta)
    assert controle["coherent"].all()

    meta["metriques_test"]["rmse"] = 99.0  # metrique recopiee de travers
    controle = verifier_coherence(df, meta)
    lignes_rmse = controle[controle["metrique"] == "rmse"]
    assert not bool(lignes_rmse["coherent"].iloc[0])
