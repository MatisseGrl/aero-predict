"""
Tests de la demo en ligne de commande (src/demo.py) : classement des alertes calcule a la main,
commandes de bout en bout dans un dossier temporaire, coherence avec src/evaluation.py, refus des
identifiants hors bornes, artefact (sauvegarde / relecture) et benchmark.

Les modeles entraines ici sont volontairement petits (quelques dizaines d'arbres) : on teste la
mecanique de la demo, pas la qualite du modele. Le tout tourne en une vingtaine de secondes.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import pytest

from src.config_modele import sauvegarder_config
from src.demo import (
    CLES_ARTEFACT,
    STATUT_ALERTE_CORRECTE,
    STATUT_ALERTE_MANQUEE,
    STATUT_FAUSSE_ALERTE,
    STATUT_RIEN_A_SIGNALER,
    ErreurDemo,
    charger_artefact,
    charger_donnees_test,
    classer_alerte,
    controler_coherence_evaluation,
    entrainer_modele_retenu,
    main,
    mesurer_benchmark,
    predire_flotte,
    predire_moteur,
    premier_cycle_en_alerte,
    preparer_donnees_brutes,
    resumer_flotte,
    sauvegarder_artefact,
    verifier_fraicheur_artefact,
    verifier_id_moteur,
)
from src.evaluation import alertes_manquees, evaluer_sur_test, fausses_alertes

# configuration volontairement legere (30 arbres) : entrainement en une fraction de seconde
CONFIG_RAPIDE = {
    "famille": "xgboost",
    "perte": "native",
    "parametres_perte": {},
    "hyperparametres": {"n_estimators": 30, "max_depth": 4, "learning_rate": 0.1, "random_state": 42},
    "nom": "XGBoost test rapide",
    "regle_de_choix": "tests de la demo",
}


@pytest.fixture(scope="module")
def artefact_rapide(contexte_train):
    """
    Artefact en memoire entraine avec CONFIG_RAPIDE, en reutilisant le contexte train de la session.

    Parameters
    ----------
    contexte_train : dict
        La fixture de tests/conftest.py.

    Returns
    -------
    dict
        L'artefact (modele + preparation apprise sur le train).
    """
    return entrainer_modele_retenu(CONFIG_RAPIDE, contexte_train=contexte_train)


@pytest.fixture(scope="module")
def donnees_test():
    """
    Donnees test brutes et RUL vraie au dernier cycle (lues une seule fois pour ce module).

    Returns
    -------
    (pandas.DataFrame, numpy.ndarray)
        test_FD001.txt brut et RUL_FD001.txt.
    """
    return charger_donnees_test()


# ---------------------------------------------------------------------------------------------------
# Classement des alertes (cas calcules a la main)
# ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "rul_predite, rul_vraie, seuil, alerte_attendue, statut_attendu",
    [
        (10.0, 5, 30, True, STATUT_ALERTE_CORRECTE),  # proche de la panne, alerte : bonne decision
        (30.0, 30, 30, True, STATUT_ALERTE_CORRECTE),  # cas limite : les deux EGAUX au seuil -> alerte correcte
        (0.0, 0, 30, True, STATUT_ALERTE_CORRECTE),  # panne imminente, RUL predite nulle
        (30.01, 30, 30, False, STATUT_ALERTE_MANQUEE),  # vraie RUL = seuil, prediction juste au-dessus
        (45.0, 12, 30, False, STATUT_ALERTE_MANQUEE),  # surestimation qui fait rater l'alerte
        (30.0, 31, 30, True, STATUT_FAUSSE_ALERTE),  # prediction = seuil (alerte) mais vraie RUL au-dessus
        (5.0, 100, 30, True, STATUT_FAUSSE_ALERTE),  # alerte alors qu'il reste beaucoup de marge
        (31.0, 31, 30, False, STATUT_RIEN_A_SIGNALER),  # les deux juste au-dessus du seuil
        (90.0, 112, 30, False, STATUT_RIEN_A_SIGNALER),  # moteur sain
        (20.0, 25, 15, False, STATUT_RIEN_A_SIGNALER),  # meme moteur, seuil plus bas : plus d'alerte
        (15.0, 25, 15, True, STATUT_FAUSSE_ALERTE),  # cas limite egal au seuil (15) cote prediction
    ],
)
def test_classer_alerte_cas_a_la_main(rul_predite, rul_vraie, seuil, alerte_attendue, statut_attendu):
    """Chaque cas (y compris les egalites au seuil) donne la decision et le statut attendus."""
    alerte, statut = classer_alerte(rul_predite, rul_vraie, seuil)
    assert alerte is alerte_attendue
    assert statut == statut_attendu


def test_classer_alerte_coherent_avec_evaluation():
    """
    Sur 500 cas aleatoires (dont beaucoup d'egalites au seuil), les comptes de statuts redonnent
    exactement `alertes_manquees` et `fausses_alertes` de src/evaluation.py.
    """
    generateur = np.random.default_rng(1)
    vrais = generateur.integers(0, 60, size=500).astype(float)  # entiers : beaucoup de valeurs = 30
    predits = generateur.integers(0, 60, size=500).astype(float)

    nb_manquees = 0
    nb_fausses = 0
    for position in range(len(vrais)):
        alerte, statut = classer_alerte(predits[position], vrais[position], 30)
        assert alerte == (predits[position] <= 30)
        if statut == STATUT_ALERTE_MANQUEE:
            nb_manquees = nb_manquees + 1
        if statut == STATUT_FAUSSE_ALERTE:
            nb_fausses = nb_fausses + 1

    assert nb_manquees == alertes_manquees(vrais, predits, seuil_alerte=30)
    assert nb_fausses == fausses_alertes(vrais, predits, seuil_alerte=30)


def test_premier_cycle_en_alerte():
    """Le premier cycle ou la RUL predite passe sous le seuil (egalite comprise) est bien retrouve."""
    trajectoire = pd.DataFrame(
        {"time_cycles": [1, 2, 3, 4], "rul_predite": [50.0, 31.0, 30.0, 10.0], "RUL_vraie": [40, 39, 38, 37]}
    )
    assert premier_cycle_en_alerte(trajectoire, 30) == (3, 38)  # 30.0 <= 30 au cycle 3
    assert premier_cycle_en_alerte(trajectoire, 5) is None  # jamais sous 5 cycles


# ---------------------------------------------------------------------------------------------------
# Identifiants hors bornes
# ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("moteur_id", [0, -3, 101, 1000])
def test_id_moteur_hors_bornes_refuse(moteur_id):
    """Un numero hors de [1, 100] est refuse avec un message clair."""
    with pytest.raises(ErreurDemo, match="1 a 100"):
        verifier_id_moteur(moteur_id)


@pytest.mark.parametrize("moteur_id", [1, 34, 100])
def test_id_moteur_dans_les_bornes_accepte(moteur_id):
    """Les numeros 1 (borne basse), 34 et 100 (borne haute) sont acceptes."""
    verifier_id_moteur(moteur_id)


def test_cli_moteur_hors_bornes(tmp_path, capsys):
    """La commande refuse --id 101 : code de sortie 1, message en francais, et AUCUN entrainement lance."""
    chemin_artefact = tmp_path / "modele.joblib"
    code = main(["moteur", "--id", "101", "--artefact", str(chemin_artefact), "--sortie", str(tmp_path)])
    sortie = capsys.readouterr()
    assert code == 1
    assert "ERREUR" in sortie.err
    assert "inexistant" in sortie.err
    assert not chemin_artefact.exists()  # verification faite AVANT tout entrainement


def test_cli_moteur_id_non_entier(tmp_path, capsys):
    """--id abc est refuse par argparse (code 2) avec un message en francais."""
    with pytest.raises(SystemExit) as sortie_systeme:
        main(["moteur", "--id", "abc", "--artefact", str(tmp_path / "m.joblib")])
    assert sortie_systeme.value.code == 2
    assert "entier attendu" in capsys.readouterr().err


# ---------------------------------------------------------------------------------------------------
# Commandes de bout en bout dans un dossier temporaire
# ---------------------------------------------------------------------------------------------------

def test_cli_entrainer_moteur_flotte(tmp_path, capsys):
    """entrainer, puis moteur --id 1, puis flotte : tout tourne sans erreur et produit ses fichiers."""
    chemin_config = str(tmp_path / "config.json")
    chemin_artefact = str(tmp_path / "models" / "modele.joblib")
    dossier_figures = str(tmp_path / "figures")
    sauvegarder_config(CONFIG_RAPIDE, chemin_config)

    # 1. entrainer
    code = main(["entrainer", "--config", chemin_config, "--artefact", chemin_artefact])
    sortie = capsys.readouterr().out
    assert code == 0
    assert os.path.exists(chemin_artefact)
    assert "XGBoost test rapide" in sortie
    assert "100 moteurs, 20 631 cycles, 46 features" in sortie

    # 2. moteur --id 1 : resume affiche et figure ecrite
    arguments_moteur = ["moteur", "--id", "1", "--artefact", chemin_artefact, "--config", chemin_config]
    code = main(arguments_moteur + ["--sortie", dossier_figures])
    sortie = capsys.readouterr().out
    assert code == 0
    chemin_png = os.path.join(dossier_figures, "moteur_1.png")
    assert os.path.exists(chemin_png)
    assert os.path.getsize(chemin_png) > 10_000  # une vraie image, pas un fichier vide
    assert "Moteur test 1" in sortie
    assert "31 cycles" in sortie  # le moteur test 1 a 31 cycles observes
    assert "RUL vraie (RUL_FD001.txt)      : 112 cycles" in sortie  # 1re ligne de RUL_FD001.txt
    assert "[attention]" not in sortie  # l'artefact est a jour par rapport a sa configuration

    # 3. flotte : tableau, resume et controle de coherence
    code = main(["flotte", "--artefact", chemin_artefact, "--config", chemin_config, "--top", "5"])
    sortie = capsys.readouterr().out
    assert code == 0
    assert "RMSE" in sortie
    assert "Score NASA" in sortie
    assert "OK (memes chiffres que evaluer_sur_test)" in sortie
    assert "95 autres moteurs non affiches" in sortie


def test_cli_moteur_entraine_automatiquement(tmp_path, capsys):
    """Sans artefact, la commande moteur entraine d'abord le modele, puis predit."""
    chemin_config = str(tmp_path / "config.json")
    chemin_artefact = str(tmp_path / "absent" / "modele.joblib")
    config_legere = dict(CONFIG_RAPIDE)
    config_legere["hyperparametres"] = {"n_estimators": 10}
    sauvegarder_config(config_legere, chemin_config)

    arguments_moteur = ["moteur", "--id", "100", "--artefact", chemin_artefact, "--config", chemin_config]
    code = main(arguments_moteur + ["--sortie", str(tmp_path)])
    sortie = capsys.readouterr().out
    assert code == 0
    assert "entrainement automatique" in sortie
    assert os.path.exists(chemin_artefact)
    assert os.path.exists(tmp_path / "moteur_100.png")


# ---------------------------------------------------------------------------------------------------
# Coherence avec le pipeline et l'evaluation officiels
# ---------------------------------------------------------------------------------------------------

def test_flotte_coherente_avec_evaluer_sur_test(artefact_rapide, donnees_test, contexte_test):
    """
    La demo (donnees brutes -> preparation -> prediction) redonne exactement les predictions et les
    metriques de `evaluer_sur_test` applique aux features de `preparer_test` (src/features.py).
    """
    test_brut, rul_test = donnees_test
    resultat = predire_flotte(artefact_rapide, test_brut, rul_test, seuil_alerte=30)
    tableau = resultat["tableau"]
    resume = resumer_flotte(tableau, seuil_alerte=30)

    officiel = evaluer_sur_test(
        artefact_rapide["modele"], contexte_test["X_test_dernier"], contexte_test["y_test_vraie"], seuil_alerte=30
    )

    assert tableau["moteur"].tolist() == contexte_test["moteurs_test"].tolist()  # meme ordre de moteurs
    assert np.allclose(tableau["rul_predite"].values, officiel["predictions"], rtol=0, atol=1e-9)
    assert np.array_equal(tableau["rul_vraie"].values, contexte_test["y_test_vraie"])
    assert resume["rmse"] == pytest.approx(officiel["rmse"], abs=1e-9)
    assert resume["score_nasa"] == pytest.approx(officiel["score_nasa"], abs=1e-9)
    assert resume["nb_surestimes"] == officiel["nb_surestimes"]
    assert resume["alertes_manquees"] == officiel["alertes_manquees"]
    assert resume["fausses_alertes"] == officiel["fausses_alertes"]
    assert resume["alertes_emises"] == int(np.sum(officiel["predictions"] <= 30))

    # le controle integre a la commande flotte arrive a la meme conclusion
    coherent, ecarts = controler_coherence_evaluation(
        artefact_rapide, resultat["X_dernier"], rul_test, tableau, resume, 30
    )
    assert coherent, ecarts


@pytest.mark.parametrize("moteur_id", [1, 34, 100])
def test_preparation_d_un_moteur_identique_au_pipeline_complet(artefact_rapide, donnees_test, contexte_test, moteur_id):
    """
    Preparer UN moteur seul donne exactement les memes features que `preparer_test` sur les 100
    moteurs, et la trajectoire de RUL vraie est la colonne RUL_vraie du test enrichi.
    """
    test_brut, rul_test = donnees_test
    feature_cols = artefact_rapide["feature_cols"]

    lignes_brutes = test_brut[test_brut["unit_number"] == moteur_id]
    prepare_seul = preparer_donnees_brutes(lignes_brutes, artefact_rapide)
    reference = contexte_test["test"]
    reference_moteur = reference[reference["unit_number"] == moteur_id]

    assert np.allclose(prepare_seul[feature_cols].values, reference_moteur[feature_cols].values, rtol=0, atol=1e-12)

    resultat = predire_moteur(artefact_rapide, test_brut, rul_test, moteur_id)
    trajectoire = resultat["trajectoire"]
    assert resultat["nb_cycles"] == len(reference_moteur)
    assert np.array_equal(trajectoire["RUL_vraie"].values, reference_moteur["RUL_vraie"].values)
    assert np.array_equal(trajectoire["RUL_vraie_cappee"].values, reference_moteur["RUL_vraie_cappee"].values)
    assert resultat["rul_vraie_dernier"] == int(rul_test[moteur_id - 1])
    assert np.all(trajectoire["rul_predite"].values >= 0)  # predictions ramenees a 0 si negatives
    assert resultat["temps_ms"] > 0

    # la prediction au dernier cycle est celle de la commande flotte pour ce moteur
    flotte = predire_flotte(artefact_rapide, test_brut, rul_test)["tableau"]
    prediction_flotte = flotte.loc[flotte["moteur"] == moteur_id, "rul_predite"].iloc[0]
    assert resultat["rul_predite_dernier"] == pytest.approx(prediction_flotte, abs=1e-9)


def test_scaler_jamais_reajuste_sur_le_test(artefact_rapide, donnees_test):
    """Predire le test ne modifie pas le scaler appris sur le train (transform seul, jamais fit)."""
    test_brut, rul_test = donnees_test
    scaler = artefact_rapide["scaler"]
    minimums_avant = scaler.data_min_.copy()
    maximums_avant = scaler.data_max_.copy()
    nb_vus_avant = scaler.n_samples_seen_

    predire_flotte(artefact_rapide, test_brut, rul_test)
    predire_moteur(artefact_rapide, test_brut, rul_test, 34)

    assert np.array_equal(scaler.data_min_, minimums_avant)
    assert np.array_equal(scaler.data_max_, maximums_avant)
    assert scaler.n_samples_seen_ == nb_vus_avant  # 20 631 lignes du train, pas une de plus
    assert nb_vus_avant == 20631


# ---------------------------------------------------------------------------------------------------
# Artefact
# ---------------------------------------------------------------------------------------------------

def test_artefact_aller_retour_avec_perte_sur_mesure(contexte_train, contexte_test, tmp_path):
    """
    Un modele a perte sur mesure (closure, impossible a enregistrer avec pickle seul) se sauvegarde
    et se relit, avec exactement les memes predictions ; l'artefact contient tout le contexte.
    """
    config = {
        "famille": "lightgbm",
        "perte": "quadratique_asymetrique",
        "parametres_perte": {"alpha": 2.0},
        "hyperparametres": {"n_estimators": 20},
        "nom": "LightGBM asymetrique test",
    }
    artefact = entrainer_modele_retenu(config, contexte_train=contexte_train)
    chemin = str(tmp_path / "modele.joblib")
    sauvegarder_artefact(artefact, chemin)
    relu = charger_artefact(chemin)

    for cle in CLES_ARTEFACT:  # toutes les cles attendues sont presentes
        assert cle in relu
    assert relu["config"] == config
    assert relu["feature_cols"] == contexte_train["feature_cols"]
    assert relu["constant_sensors"] == contexte_train["constant_sensors"]
    assert relu["fenetre"] == 5
    assert relu["plafond"] == 125
    assert np.array_equal(relu["scaler"].data_min_, contexte_train["scaler"].data_min_)

    X = contexte_test["X_test_dernier"]
    assert np.array_equal(relu["modele"].predict(X), artefact["modele"].predict(X))

    # le fichier reste un fichier joblib ordinaire (le modele y est range sous forme d'octets)
    brut = joblib.load(chemin)
    assert brut["format_modele"] == "cloudpickle"
    assert isinstance(brut["modele"], bytes)


def test_charger_artefact_absent_ou_incomplet(tmp_path):
    """Artefact absent ou incomplet : erreur claire qui dit de relancer l'entrainement."""
    with pytest.raises(ErreurDemo, match="entrainer"):
        charger_artefact(str(tmp_path / "absent.joblib"))

    chemin_incomplet = str(tmp_path / "incomplet.joblib")
    joblib.dump({"version_format": 1, "modele": b""}, chemin_incomplet)
    with pytest.raises(ErreurDemo, match="incomplet"):
        charger_artefact(chemin_incomplet)


def test_avertissement_si_config_modifiee(contexte_train, tmp_path, capsys):
    """Si la configuration change apres l'entrainement, la demo previent (sans rien bloquer)."""
    chemin_config = str(tmp_path / "config.json")
    sauvegarder_config(CONFIG_RAPIDE, chemin_config)
    artefact = entrainer_modele_retenu(CONFIG_RAPIDE, contexte_train=contexte_train, chemin_config=chemin_config)
    assert verifier_fraicheur_artefact(artefact) is True  # rien n'a change

    config_modifiee = dict(CONFIG_RAPIDE)
    config_modifiee["hyperparametres"] = {"n_estimators": 99}
    with open(chemin_config, "w", encoding="utf-8") as fichier:
        json.dump(config_modifiee, fichier)
    assert verifier_fraicheur_artefact(artefact) is False
    assert "[attention]" in capsys.readouterr().out


# ---------------------------------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------------------------------

def test_benchmark_renvoie_des_nombres_positifs(artefact_rapide, donnees_test, tmp_path):
    """Le benchmark (reduit a 3 moteurs, 2 repetitions) renvoie des temps et une taille positifs, ordonnes."""
    test_brut, rul_test = donnees_test
    chemin = str(tmp_path / "modele.joblib")
    sauvegarder_artefact(artefact_rapide, chemin)

    resultats = mesurer_benchmark(
        artefact_rapide, chemin, test_brut, rul_test, repetitions=2, cible_ms=200.0, moteurs=[1, 2, 3]
    )

    assert resultats["nb_mesures"] == 6  # 3 moteurs x 2 repetitions
    assert len(resultats["temps_moteur_ms"]) == 6
    assert resultats["p50_ms"] > 0
    assert resultats["p50_ms"] <= resultats["p95_ms"] <= resultats["max_ms"]
    assert resultats["moyenne_ms"] > 0
    assert resultats["temps_flotte_s"] > 0
    assert resultats["temps_flotte_max_s"] >= resultats["temps_flotte_s"]
    assert resultats["taille_ko"] > 0
    assert resultats["temps_chargement_ms"] > 0
    assert resultats["cible_atteinte"] == (resultats["p95_ms"] <= 200.0)

    # une cible impossible a tenir donne bien un verdict negatif
    impossible = mesurer_benchmark(
        artefact_rapide, chemin, test_brut, rul_test, repetitions=1, cible_ms=1e-6, moteurs=[1]
    )
    assert impossible["cible_atteinte"] is False
