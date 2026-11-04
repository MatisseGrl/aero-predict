"""
Tests de la couche d'acces aux donnees du tableau de bord (src/donnees_demo.py) et de scripts/preparer_demo.py.

Les fichiers de predictions sont SYNTHETIQUES et ecrits dans un dossier temporaire (tmp_path de pytest) :
les vraies predictions de sorties/predictions/ ne sont jamais lues ni modifiees ici. Les seuls fichiers
reels lus sont ceux du depot qui ne changent pas pendant les tests (data/, docs/resultats/ en lecture).
L'ensemble tourne en quelques secondes.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

from src import donnees_demo as dd
from src.contrat_predictions import (
    MODELES_CONNUS,
    calculer_rul_vraie,
    ecrire_predictions,
    metriques_dernier_cycle,
)
from src.demo import (
    STATUT_ALERTE_CORRECTE,
    STATUT_ALERTE_MANQUEE,
    STATUT_FAUSSE_ALERTE,
    STATUT_RIEN_A_SIGNALER,
    classer_alerte,
    entrainer_modele_retenu,
    predire_moteur,
)

# configuration volontairement legere (30 arbres), comme tests/test_demo.py : on teste la mecanique
CONFIG_RAPIDE = {
    "famille": "xgboost",
    "perte": "native",
    "parametres_perte": {},
    "hyperparametres": {"n_estimators": 30, "max_depth": 4, "learning_rate": 0.1, "random_state": 42},
    "nom": "XGBoost test rapide",
    "regle_de_choix": "tests du tableau de bord",
}


# ---------------------------------------------------------------------------------------------------
# Fabrication de petits fichiers de predictions synthetiques
# ---------------------------------------------------------------------------------------------------

def fabriquer_predictions(rul_finales, predictions_par_moteur, regime=0):
    """
    Fabrique un DataFrame de predictions au format du contrat, calculable a la main.

    Parameters
    ----------
    rul_finales : list de float
        RUL vraie au dernier cycle de chaque moteur (moteurs numerotes 1, 2, ...).
    predictions_par_moteur : list de list de float
        Pour chaque moteur, la RUL predite a chacun de ses cycles (cycles 1, 2, ...).
    regime : int, optionnel
        Numero de regime ecrit sur toutes les lignes.

    Returns
    -------
    pandas.DataFrame
        Les colonnes obligatoires du contrat.
    """
    numeros = []
    cycles = []
    predites = []
    for position in range(len(predictions_par_moteur)):
        for indice_cycle in range(len(predictions_par_moteur[position])):
            numeros.append(position + 1)
            cycles.append(indice_cycle + 1)
            predites.append(float(predictions_par_moteur[position][indice_cycle]))
    df = pd.DataFrame()
    df["unit_number"] = numeros
    df["time_cycles"] = cycles
    df["rul_predite"] = predites
    df["regime"] = regime
    return calculer_rul_vraie(df, rul_finales)


def ecrire_fichier(df, dossier, sous_jeu, modele, validation=None):
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
    validation : dict ou None, optionnel
        Les metriques de validation (par defaut, un exemple a deux cles et une liste).

    Returns
    -------
    None
    """
    if validation is None:
        validation = {"rmse_cv_moyen": 16.5, "nasa_cv_moyen": 25000.0, "rmse_par_pli": [16.0, 17.0]}
    meta = {}
    meta["sous_jeu"] = sous_jeu
    meta["modele"] = modele
    meta["nom_affiche"] = MODELES_CONNUS[modele]
    meta["famille"] = "arbres"
    meta["metriques_test"] = metriques_dernier_cycle(df)
    meta["metriques_validation"] = validation
    meta["protocole_validation"] = "GroupKFold 5 plis par moteur (synthetique)"
    meta["duree_entrainement_s"] = 1.0
    meta["date"] = "2026-11-04"
    meta["source"] = "tests/test_donnees_demo.py"
    meta["graine"] = 7
    ecrire_predictions(df, meta, dossier=dossier)


# ---------------------------------------------------------------------------------------------------
# Un moteur : premiere alerte et etat a un cycle
# ---------------------------------------------------------------------------------------------------

def test_premier_cycle_alerte_et_etat_au_cycle():
    """
    Moteur a 4 cycles, RUL finale 25 -> RUL vraie 28, 27, 26, 25 ; predictions 50, 31, 30, 40.

    Seuil 30 : la premiere alerte est au cycle 3 (30 <= 30), RUL vraie 26 a ce cycle. Au cycle 2, pas
    encore d'alerte ; au cycle 4, la prediction remonte a 40 (pas d'alerte) alors que la vraie RUL vaut
    25 <= 30 : ALERTE MANQUEE, mais la premiere alerte (cycle 3) reste memorisee.
    """
    df = fabriquer_predictions([25], [[50, 31, 30, 40]])
    trajectoire = dd.extraire_trajectoire(df, 1)

    assert dd.premier_cycle_alerte(trajectoire, 30) == (3, 26.0)
    assert dd.premier_cycle_alerte(trajectoire, 30, cycle_max=2) is None  # le "futur" n'est pas regarde
    assert dd.premier_cycle_alerte(trajectoire, 10) is None

    etat = dd.etat_au_cycle(trajectoire, 2, seuil_alerte=30)
    assert etat["rul_predite"] == 31.0 and etat["rul_vraie"] == 27.0 and etat["ecart"] == pytest.approx(4.0)
    assert etat["statut"] == STATUT_ALERTE_MANQUEE  # vraie 27 <= 30 mais predite 31 > 30
    assert etat["premiere_alerte"] is None

    etat = dd.etat_au_cycle(trajectoire, 4, seuil_alerte=30)
    assert etat["statut"] == STATUT_ALERTE_MANQUEE
    assert etat["premiere_alerte"] == (3, 26.0)

    # meme regle que classer_alerte de src/demo.py
    etat = dd.etat_au_cycle(trajectoire, 3, seuil_alerte=30)
    assert (etat["alerte"], etat["statut"]) == classer_alerte(30.0, 26.0, 30)

    # un cycle au-dela du dernier : on reste sur le dernier cycle observe
    assert dd.etat_au_cycle(trajectoire, 99, seuil_alerte=30)["cycle"] == 4

    with pytest.raises(ValueError, match="n'existe pas"):
        dd.extraire_trajectoire(df, 5)


# ---------------------------------------------------------------------------------------------------
# La flotte
# ---------------------------------------------------------------------------------------------------

def fabriquer_flotte():
    """
    Quatre moteurs, un par statut d'alerte (seuil 30), predictions constantes sur 3 cycles.

    - moteur 1 : vraie 10, predite 15 -> alerte correcte ;
    - moteur 2 : vraie 20, predite 60 -> ALERTE MANQUEE (et surestimation de 40 > 20) ;
    - moteur 3 : vraie 80, predite 25 -> fausse alerte ;
    - moteur 4 : vraie 140, predite 120 -> rien a signaler (RUL vraie au-dessus du plafond 125).

    Returns
    -------
    pandas.DataFrame
        Les predictions.
    """
    return fabriquer_predictions([10, 20, 80, 140], [[15] * 3, [60] * 3, [25] * 3, [120] * 3])


def test_tableau_flotte_statuts_tri_et_metriques():
    """Chaque moteur a le bon statut, le tableau est trie par RUL predite, les metriques = le contrat."""
    df = fabriquer_flotte()
    tableau = dd.tableau_flotte(df, seuil_alerte=30)

    # tri par urgence : RUL predite croissante
    assert tableau["moteur"].tolist() == [1, 3, 2, 4]
    statuts = {}
    for position in range(len(tableau)):
        statuts[int(tableau["moteur"].iloc[position])] = tableau["statut"].iloc[position]
    assert statuts == {
        1: STATUT_ALERTE_CORRECTE, 2: STATUT_ALERTE_MANQUEE, 3: STATUT_FAUSSE_ALERTE, 4: STATUT_RIEN_A_SIGNALER,
    }
    assert tableau["nb_cycles"].tolist() == [3, 3, 3, 3]

    # le resume (src/demo.py) donne les memes nombres que metriques_dernier_cycle (le contrat)
    resume = dd.resumer_tableau_flotte(tableau, seuil_alerte=30)
    contrat = metriques_dernier_cycle(df)
    assert resume["rmse"] == pytest.approx(contrat["rmse"])
    assert resume["score_nasa"] == pytest.approx(contrat["score_nasa"])
    assert resume["nb_surestimes"] == contrat["nb_surestimes"] == 1
    assert resume["alertes_manquees"] == contrat["alertes_manquees"] == 1
    assert resume["fausses_alertes"] == contrat["fausses_alertes"] == 1
    assert resume["alertes_emises"] == 2

    comptes = dd.compter_statuts(tableau)
    assert list(comptes.keys()) == dd.ORDRE_STATUTS
    assert list(comptes.values()) == [1, 1, 1, 1]

    # premiere alerte des moteurs alertes des le cycle 1 (predictions constantes)
    ligne_moteur_1 = tableau[tableau["moteur"] == 1].iloc[0]
    assert ligne_moteur_1["cycle_premiere_alerte"] == 1
    assert np.isnan(tableau[tableau["moteur"] == 2].iloc[0]["cycle_premiere_alerte"])

    assert dd.moteur_le_plus_proche_de_la_panne(df) == 1


# ---------------------------------------------------------------------------------------------------
# Inventaire, comparaison, coherence, tests apparies (fichiers synthetiques dans tmp_path)
# ---------------------------------------------------------------------------------------------------

def test_inventaire_comparaison_et_coherence(tmp_path):
    """Inventaire dans l'ordre du contrat ; comparaison lue dans les metadonnees ; incoherence detectee."""
    dossier = str(tmp_path)
    df = fabriquer_flotte()
    ecrire_fichier(df, dossier, "FD003", "xgboost_retenu")
    ecrire_fichier(df, dossier, "FD001", "xgboost_retenu")
    ecrire_fichier(df, dossier, "FD001", "xgboost_phase3")

    liste = dd.inventaire_predictions(dossier)
    assert dd.sous_jeux_disponibles(liste) == ["FD001", "FD003"]
    assert dd.modeles_disponibles(liste, "FD001") == ["xgboost_phase3", "xgboost_retenu"]
    assert dd.nom_modele("xgboost_retenu") == MODELES_CONNUS["xgboost_retenu"]

    comparaison = dd.tableau_comparaison(liste, dossier)
    assert len(comparaison) == 3
    premiere = comparaison.iloc[0]
    assert premiere["nb_moteurs"] == 4
    assert premiere["nb_au_dela_plafond"] == 1  # le moteur 4 (RUL vraie 140 > 125)
    assert premiere["part_au_dela_plafond"] == pytest.approx(25.0)
    assert premiere["rmse_validation"] == pytest.approx(16.5)
    assert "rmse_par_pli" not in premiere["validation"]  # les listes ne sont pas des metriques scalaires
    assert premiere["score_nasa_par_moteur"] == pytest.approx(premiere["score_nasa"] / 4)

    controles = dd.controler_tous_les_fichiers(liste, dossier)
    assert controles["coherent"].all()

    # metrique recopiee de travers dans un JSON : le controle le detecte
    chemin_json = os.path.join(dossier, "FD003__xgboost_retenu.json")
    with open(chemin_json, "r", encoding="utf-8") as fichier:
        meta = json.load(fichier)
    meta["metriques_test"]["rmse"] = meta["metriques_test"]["rmse"] + 1.0
    with open(chemin_json, "w", encoding="utf-8") as fichier:
        json.dump(meta, fichier)
    controles = dd.controler_tous_les_fichiers(dd.inventaire_predictions(dossier), dossier)
    assert controles["coherent"].tolist() == [True, True, False]

    # signature de dossier : change quand un fichier change (cle des caches de l'application)
    signature_avant = dd.signature_dossier(dossier)
    os.remove(chemin_json)
    assert dd.signature_dossier(dossier) != signature_avant
    assert dd.signature_dossier(os.path.join(dossier, "absent")) == tuple()

    # graines lues dans les metadonnees
    graines = dd.lister_graines(dd.inventaire_predictions(dossier), dossier)
    assert "FD001__xgboost_retenu.json" in graines["source"].tolist()


def test_comparer_deux_modeles_apparie(tmp_path):
    """
    12 moteurs : A se trompe toujours de 10 cycles de plus que B -> A significativement moins bon.

    Un sous-jeu ou un seul des deux modeles existe n'apparait pas dans le resultat.
    """
    dossier = str(tmp_path)
    rul_finales = []
    predictions_a = []
    predictions_b = []
    for position in range(12):
        rul_finales.append(40.0 + position)
        predictions_b.append([40.0 + position + 1.0 + 0.1 * position])  # B : erreur 1 a 2,1 cycles
        predictions_a.append([40.0 + position + 11.0 + 0.2 * position])  # A : erreur 11 a 13,2 cycles
    ecrire_fichier(fabriquer_predictions(rul_finales, predictions_a), dossier, "FD002", "xgboost_retenu")
    ecrire_fichier(fabriquer_predictions(rul_finales, predictions_b), dossier, "FD002", "xgboost_phase3")
    ecrire_fichier(fabriquer_predictions(rul_finales, predictions_a), dossier, "FD003", "xgboost_retenu")

    liste = dd.inventaire_predictions(dossier)
    resultat = dd.comparer_deux_modeles(liste, "xgboost_retenu", "xgboost_phase3", dossier)
    assert resultat["sous_jeu"].tolist() == ["FD002"]
    ligne = resultat.iloc[0]
    assert ligne["nb_moteurs"] == 12
    assert ligne["difference_moyenne"] > 9.0
    assert ligne["p_valeur_wilcoxon"] < 0.05 and ligne["p_valeur_t"] < 0.05
    assert "significativement moins bon" in ligne["conclusion"]

    # sens inverse : B contre A
    inverse = dd.comparer_deux_modeles(liste, "xgboost_phase3", "xgboost_retenu", dossier)
    assert "significativement meilleur" in inverse["conclusion"].iloc[0]

    # aucun sous-jeu commun : tableau vide
    assert len(dd.comparer_deux_modeles(liste, "lstm", "xgboost_retenu", dossier)) == 0

    # regle de conclusion : les deux tests doivent etre d'accord
    assert dd.conclure_test_apparie(0.5, 0.6, 1.0, "A", "B") == "différence non significative"
    assert "désaccord" in dd.conclure_test_apparie(0.01, 0.2, 1.0, "A", "B")


# ---------------------------------------------------------------------------------------------------
# Description des sous-jeux, donnees brutes
# ---------------------------------------------------------------------------------------------------

def test_readme_et_ecarts():
    """Le readme NASA est lu tel quel (FD004 : 248 train / 249 test) ; un ecart avec les fichiers est signale."""
    readme = dd.lire_readme_donnees()
    assert readme["sous_jeu"].tolist() == ["FD001", "FD002", "FD003", "FD004"]
    ligne_fd004 = readme[readme["sous_jeu"] == "FD004"].iloc[0]
    assert (ligne_fd004["moteurs_train"], ligne_fd004["moteurs_test"]) == (248, 249)
    assert "HPC" in readme.iloc[0]["modes_de_panne"]

    description = pd.DataFrame({"sous_jeu": ["FD001", "FD004"], "nb_moteurs_train": [100, 249], "nb_moteurs_test": [100, 248]})
    ecarts = dd.ecarts_readme(description, readme)
    assert len(ecarts) == 2
    assert "FD004 train" in ecarts[0] and "248" in ecarts[0] and "249" in ecarts[0]

    tableau, source = dd.decrire_sous_jeux()
    assert len(tableau) == 4 and source != ""


def test_donnees_brutes_et_capteurs():
    """26 colonnes de src/preprocessing.py ; les capteurs constants de FD001 ne sont pas proposes."""
    brut = dd.charger_donnees_brutes_test("FD001")
    assert brut.shape[1] == 26
    capteurs = dd.capteurs_variables(brut)
    assert "sensor_1" not in capteurs and "sensor_11" in capteurs
    assert dd.choisir_capteurs_par_defaut(capteurs) == ["sensor_11", "sensor_4", "sensor_7"]
    assert dd.choisir_capteurs_par_defaut(["sensor_2", "sensor_3"]) == ["sensor_2", "sensor_3"]
    assert dd.nombre_capteurs_bruts() == 21
    assert len(dd.tableau_grandeurs_saxena()) == 21
    with pytest.raises(ValueError):
        dd.charger_donnees_brutes_test("FD009")


# ---------------------------------------------------------------------------------------------------
# Derive injectee et inference en direct (page "Et si...")
# ---------------------------------------------------------------------------------------------------

def test_appliquer_derive():
    """Pourcentage, decalage, mode progressif ; rien ne change avant le cycle de debut ; erreurs claires."""
    donnees = pd.DataFrame({"unit_number": [1] * 5, "time_cycles": [1, 2, 3, 4, 5], "sensor_11": [100.0] * 5})

    pourcentage = dd.appliquer_derive(donnees, "sensor_11", "pourcentage", 2.0, cycle_debut=3)
    assert pourcentage["sensor_11"].tolist() == pytest.approx([100.0, 100.0, 102.0, 102.0, 102.0])
    assert donnees["sensor_11"].tolist() == [100.0] * 5  # l'original n'est pas modifie

    decalage = dd.appliquer_derive(donnees, "sensor_11", "decalage", -4.0, cycle_debut=4)
    assert decalage["sensor_11"].tolist() == pytest.approx([100.0, 100.0, 100.0, 96.0, 96.0])

    # progressive : 0 au cycle 3, la moitie au cycle 4, tout au cycle 5
    progressive = dd.appliquer_derive(donnees, "sensor_11", "decalage", 10.0, cycle_debut=3, progressive=True)
    assert progressive["sensor_11"].tolist() == pytest.approx([100.0, 100.0, 100.0, 105.0, 110.0])

    with pytest.raises(ValueError, match="Type de derive"):
        dd.appliquer_derive(donnees, "sensor_11", "rampe", 1.0, 1)
    with pytest.raises(ValueError, match="Capteur absent"):
        dd.appliquer_derive(donnees, "sensor_99", "decalage", 1.0, 1)


@pytest.fixture(scope="module")
def artefact_rapide(contexte_train):
    """
    Artefact en memoire entraine avec CONFIG_RAPIDE (contexte train partage de tests/conftest.py).

    Parameters
    ----------
    contexte_train : dict
        La fixture de session de tests/conftest.py.

    Returns
    -------
    dict
        L'artefact.
    """
    return entrainer_modele_retenu(CONFIG_RAPIDE, contexte_train=contexte_train)


def test_simuler_derive(artefact_rapide):
    """
    Amplitude nulle : predictions identiques ; sans derive : memes predictions que predire_moteur (demo CLI) ;
    derive forte : les predictions changent apres le cycle de debut seulement ; RUL vraie inchangee.
    """
    test_brut, rul_test = dd.donnees_test_fd001()
    moteur = 1
    donnees_moteur = test_brut[test_brut["unit_number"] == moteur]
    rul_finale = float(rul_test[0])

    nul = dd.simuler_derive(artefact_rapide, donnees_moteur, rul_finale, "sensor_11", "pourcentage", 0.0, 10)
    assert nul["difference_max"] == 0.0

    # sans derive, la page "Et si..." predit exactement comme la commande `moteur` de src/demo.py
    reference = predire_moteur(artefact_rapide, test_brut, rul_test, moteur)
    assert np.allclose(nul["tableau"]["rul_predite_sans"].to_numpy(), reference["trajectoire"]["rul_predite"].to_numpy())
    assert np.allclose(nul["tableau"]["rul_vraie"].to_numpy(), reference["trajectoire"]["RUL_vraie"].to_numpy())

    etendue = dd.etendue_capteur_train(artefact_rapide, "sensor_11")
    assert etendue["etendue"] > 0
    debut = 20
    fort = dd.simuler_derive(artefact_rapide, donnees_moteur, rul_finale, "sensor_11", "decalage",
                             etendue["etendue"], debut)
    tableau = fort["tableau"]
    avant = tableau[tableau["time_cycles"] < debut]
    assert np.allclose(avant["rul_predite_sans"].to_numpy(), avant["rul_predite_avec"].to_numpy())
    assert fort["difference_max"] > 0
    assert fort["dernier"]["rul_vraie"] == rul_finale
    assert fort["temps_ms_sans"] > 0 and fort["temps_ms_avec"] > 0

    with pytest.raises(ValueError, match="pas utilise par le modele"):
        dd.etendue_capteur_train(artefact_rapide, "sensor_1")  # capteur constant retire

    resume = dd.resume_pipeline(artefact_rapide)
    assert resume["nb_capteurs_bruts"] == 21
    assert resume["nb_capteurs_gardes"] + resume["nb_capteurs_retires"] == 21


# ---------------------------------------------------------------------------------------------------
# Resultats de la Phase 4 (lecture seule de docs/resultats/)
# ---------------------------------------------------------------------------------------------------

def test_analyser_optuna_rejoue_la_regle(tmp_path):
    """La regle rejouee sur les CSV Optuna redonne la configuration de 04_config_retenue.json."""
    analyse = dd.analyser_optuna()
    config = dd.lire_config_retenue()
    if analyse is None or config is None:
        pytest.skip("resultats Optuna de la Phase 4 absents de docs/resultats/")
    assert analyse["config_coherente"] is True
    assert int(analyse["essais"]["est_retenu"].sum()) == 1
    assert analyse["seuil_rmse"] == pytest.approx(1.05 * analyse["rmse_min_front"])
    assert analyse["nb_admissibles"] <= analyse["nb_front"] <= analyse["nb_essais"]
    assert "parametres_texte" in analyse["retenu"].index

    # dossier sans CSV : None (l'application affiche alors un message)
    assert dd.analyser_optuna(dossier=str(tmp_path)) is None
    assert dd.lire_config_retenue(os.path.join(str(tmp_path), "absent.json")) is None
    assert dd.lire_csv_resultat("absent.csv", dossier=str(tmp_path)) is None


def test_valeurs_egales_et_graines():
    """Comparaison tolerante ; graines trouvees par nom exact (pas "rmse_moyenne_5_graines")."""
    assert dd.valeurs_egales({"a": [1, 2.0], "b": "x"}, {"a": [1.0, 2.0 + 1e-12], "b": "x"})
    assert not dd.valeurs_egales({"a": 1}, {"a": 1, "b": 2})
    assert not dd.valeurs_egales([1, 2], [1, 3])

    config = {"hyperparametres": {"random_state": 42}, "rmse_moyenne_5_graines": 19.6, "graine_kmeans": 3}
    trouvees = dd.chercher_graines(config)
    assert ("hyperparametres.random_state", 42) in trouvees
    assert ("graine_kmeans", 3) in trouvees
    assert len(trouvees) == 2

    graines = dd.lister_graines(None)
    assert "random_state" in graines["parametre"].tolist()  # valeur par defaut de calculer_plis


# ---------------------------------------------------------------------------------------------------
# Reproductibilite et limites connues
# ---------------------------------------------------------------------------------------------------

def test_reproductibilite(tmp_path):
    """Lecture du nombre de tests dans la sortie de pytest ; aller-retour du fichier JSON."""
    sortie = "tests/test_a.py::test_x\n\n149 tests collected in 0.59s\n"
    assert dd.analyser_sortie_collecte(sortie) == 149
    assert dd.analyser_sortie_collecte("1 test collected in 0.1s") == 1
    assert dd.analyser_sortie_collecte("erreur") is None

    chemin = os.path.join(str(tmp_path), "demo", "infos.json")
    dd.ecrire_infos_reproductibilite({"nb_tests": np.int64(12), "date": "2026-11-04T10:00:00"}, chemin)
    assert dd.lire_infos_reproductibilite(chemin) == {"nb_tests": 12, "date": "2026-11-04T10:00:00"}
    assert dd.lire_infos_reproductibilite(os.path.join(str(tmp_path), "absent.json")) is None

    git = dd.informations_git()
    assert set(git.keys()) == {"commit", "branche", "modifications_locales"}


def test_extraire_limites_status(tmp_path):
    """Seules des phrases du fichier sont reprises (mise en forme retiree), avec un mot-cle de limite."""
    texte = (
        "# Etat\n\n## Phase actuelle\n\n**Phase 4** : prototype, pas encore validé par l'équipe.\n\n"
        "## Dernière modification\n\n- **Date** : 2026-11-04\n"
        "- **Quoi** : RMSE 19,73 contre 19.70. Différence non significative (p = 0,94). Score en baisse.\n"
        "- Voir [le notebook](notebooks/04.ipynb) : résultat non concluant sur le test.\n\n"
        "### 2026-09-30 — Ancienne entrée\n- Rien de non significatif ici.\n\n## Historique\n- pas encore lu\n"
    )
    chemin = os.path.join(str(tmp_path), "STATUS.md")
    with open(chemin, "w", encoding="utf-8") as fichier:
        fichier.write(texte)

    limites = dd.extraire_limites_status(chemin)
    assert limites["present"] and limites["date"] == "2026-11-04"
    assert limites["phrases"] == [
        "Phase 4 : prototype, pas encore validé par l'équipe.",
        "Différence non significative (p = 0,94).",
        "Voir le notebook : résultat non concluant sur le test.",
    ]
    # chaque phrase existe dans le fichier, une fois la mise en forme retiree (rien d'invente)
    texte_nettoye = dd.nettoyer_markdown(texte)
    for phrase in limites["phrases"]:
        assert phrase in texte_nettoye

    # un nombre decimal "19.70" ne coupe pas une phrase
    assert dd.decouper_phrases("RMSE 19.70 obtenu. Deuxième phrase.") == ["RMSE 19.70 obtenu.", "Deuxième phrase."]
    assert dd.extraire_limites_status(os.path.join(str(tmp_path), "absent.md"))["present"] is False


# ---------------------------------------------------------------------------------------------------
# scripts/preparer_demo.py (sans lancer les vrais entrainements)
# ---------------------------------------------------------------------------------------------------

def test_preparer_demo_scripts_absents(tmp_path, monkeypatch, capsys):
    """
    Scripts de production absents : etapes sautees avec un message clair ; etat final lu dans le dossier
    des predictions (variable d'environnement) ; les etapes longues sont remplacees par des simulacres.
    """
    sys.path.insert(0, os.path.join(dd.RACINE_DEPOT, "scripts"))
    try:
        import preparer_demo
    finally:
        sys.path.remove(os.path.join(dd.RACINE_DEPOT, "scripts"))

    ecrire_fichier(fabriquer_flotte(), str(tmp_path), "FD001", "xgboost_retenu")
    monkeypatch.setenv("AERO_PREDICT_DOSSIER_PREDICTIONS", str(tmp_path))
    monkeypatch.setattr(preparer_demo, "SCRIPT_ARBRES", os.path.join(str(tmp_path), "absent_arbres.py"))
    monkeypatch.setattr(preparer_demo, "SCRIPT_DEEP_LEARNING", os.path.join(str(tmp_path), "absent_dl.py"))

    commandes_lancees = []

    def lancer_etape_simulee(nom, commande):
        """Simulacre de lancer_etape : note la commande au lieu de l'executer."""
        commandes_lancees.append(commande)
        return {"etape": nom, "statut": "reussie", "duree_s": 0.0}

    def reproductibilite_simulee():
        """Simulacre de etape_reproductibilite : pas de pytest dans pytest."""
        return {"etape": "Reproductibilite", "statut": "reussie", "duree_s": 0.0}

    monkeypatch.setattr(preparer_demo, "lancer_etape", lancer_etape_simulee)
    monkeypatch.setattr(preparer_demo, "etape_reproductibilite", reproductibilite_simulee)

    code = preparer_demo.main(["--sous-jeux", "FD001"])
    sortie = capsys.readouterr().out
    assert code == 0
    assert sortie.count("n'existe pas encore : etape sautee") == 2
    assert commandes_lancees == [[sys.executable, "-m", "src.demo", "entrainer"]]
    assert "FD001" in sortie and "[coherent]" in sortie

    # --seulement-arbres : l'etape deep learning n'est pas lancee du tout
    preparer_demo.main(["--seulement-arbres"])
    sortie = capsys.readouterr().out
    assert sortie.count("n'existe pas encore : etape sautee") == 1
    assert "--seulement-arbres" in sortie
