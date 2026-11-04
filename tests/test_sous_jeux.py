"""
Tests de src/sous_jeux.py : chargement des quatre sous-jeux, regimes operatoires, normalisation par
regime, egalite avec le pipeline FD001 valide, et application a de nouvelles donnees sans fuite.

Les contextes FD001, FD002 et FD004 sont prepares une seule fois par module (fixtures) : environ une
seconde chacun.
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import COLUMNS
from src.sous_jeux import (
    COLONNES_REGLAGES,
    DESCRIPTION_SOUS_JEUX,
    SOUS_JEUX,
    appliquer_preparation,
    attribuer_regimes,
    charger_rul_sous_jeu,
    charger_sous_jeu,
    choisir_nombre_regimes,
    creer_kmeans,
    decrire_regimes,
    preparer_sous_jeu,
    tableau_valeurs_distinctes_par_regime,
    trouver_capteurs_constants_par_regime,
)


# ---------------------------------------------------------------------------------------------------
# Fixtures : contextes prepares une fois pour tout le module
# ---------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def contexte_fd001():
    """
    Contexte FD001 prepare par src/sous_jeux.py (a comparer au pipeline valide de src/features.py).

    Returns
    -------
    dict
        Le resultat de `preparer_sous_jeu("FD001")`.
    """
    return preparer_sous_jeu("FD001")


@pytest.fixture(scope="module")
def contexte_fd002():
    """
    Contexte FD002 (6 regimes, normalisation par regime).

    Returns
    -------
    dict
        Le resultat de `preparer_sous_jeu("FD002")`.
    """
    return preparer_sous_jeu("FD002")


@pytest.fixture(scope="module")
def contexte_fd004():
    """
    Contexte FD004 (6 regimes, 2 modes de panne, normalisation par regime).

    Returns
    -------
    dict
        Le resultat de `preparer_sous_jeu("FD004")`.
    """
    return preparer_sous_jeu("FD004")


# ---------------------------------------------------------------------------------------------------
# Chargement et comptes
# ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("sous_jeu", SOUS_JEUX)
def test_comptes_moteurs_et_lignes(sous_jeu):
    """Les comptes de DESCRIPTION_SOUS_JEUX sont ceux des fichiers (moteurs, lignes, RUL_FD00X)."""
    description = DESCRIPTION_SOUS_JEUX[sous_jeu]
    train = charger_sous_jeu(sous_jeu, "train")
    test = charger_sous_jeu(sous_jeu, "test")
    rul = charger_rul_sous_jeu(sous_jeu)

    # memes 26 colonnes que FD001, dans le meme ordre
    assert list(train.columns) == COLUMNS
    assert list(test.columns) == COLUMNS

    # nombres de moteurs et de lignes : ceux annonces dans la description
    assert train["unit_number"].nunique() == description["nb_moteurs_train"]
    assert test["unit_number"].nunique() == description["nb_moteurs_test"]
    assert len(train) == description["nb_lignes_train"]
    assert len(test) == description["nb_lignes_test"]

    # une RUL vraie par moteur test, et des moteurs numerotes 1, 2, ..., n
    assert len(rul) == description["nb_moteurs_test"]
    assert train["unit_number"].min() == 1 and train["unit_number"].max() == description["nb_moteurs_train"]
    assert test["unit_number"].min() == 1 and test["unit_number"].max() == description["nb_moteurs_test"]


def test_erreurs_de_parametres():
    """Un sous-jeu, un split ou une normalisation inconnus levent une erreur claire."""
    with pytest.raises(ValueError):
        charger_sous_jeu("FD005", "train")
    with pytest.raises(ValueError):
        charger_sous_jeu("FD001", "validation")
    with pytest.raises(ValueError):
        preparer_sous_jeu("FD002", normalisation="standard")
    with pytest.raises(ValueError):
        preparer_sous_jeu("FD001", regime_en_feature=True)  # un seul regime : one-hot inutile


# ---------------------------------------------------------------------------------------------------
# FD001 : egalite stricte avec le pipeline valide (src/features.py)
# ---------------------------------------------------------------------------------------------------

def test_fd001_identique_au_pipeline_valide(contexte_fd001, contexte_train, contexte_test):
    """
    Pour FD001, preparer_sous_jeu redonne exactement preparer_train / preparer_test (au bit pres).

    contexte_train et contexte_test sont les fixtures de tests/conftest.py (pipeline de src/features.py).
    """
    nouveau = contexte_fd001

    # memes features, dans le meme ordre, memes valeurs exactes
    assert nouveau["feature_cols"] == contexte_train["feature_cols"]
    pd.testing.assert_frame_equal(nouveau["X"], contexte_train["X"], check_exact=True)
    pd.testing.assert_series_equal(nouveau["y"], contexte_train["y"], check_exact=True)
    pd.testing.assert_series_equal(nouveau["groups"], contexte_train["groups"], check_exact=True)

    # memes capteurs retires, memes capteurs gardes
    assert nouveau["capteurs_retires"] == contexte_train["constant_sensors"]
    assert nouveau["capteurs"] == contexte_train["sensor_cols_restants"]

    # test : memes features au dernier cycle, meme RUL vraie, memes moteurs
    pd.testing.assert_frame_equal(nouveau["X_test_dernier"], contexte_test["X_test_dernier"], check_exact=True)
    assert np.array_equal(nouveau["y_test_vraie"], contexte_test["y_test_vraie"])
    assert np.array_equal(nouveau["moteurs_test"], contexte_test["moteurs_test"])

    # les tableaux complets aussi (colonnes du pipeline valide ; la seule colonne en plus est "regime")
    colonnes_train = list(contexte_train["train"].columns)
    pd.testing.assert_frame_equal(nouveau["train"][colonnes_train], contexte_train["train"], check_exact=True)
    colonnes_test = list(contexte_test["test"].columns)
    pd.testing.assert_frame_equal(nouveau["test"][colonnes_test], contexte_test["test"], check_exact=True)

    # meme normalisation apprise (minimums et maximums du train)
    scaler_nouveau = nouveau["scalers"][0]
    assert np.array_equal(scaler_nouveau.data_min_, contexte_train["scaler"].data_min_)
    assert np.array_equal(scaler_nouveau.data_max_, contexte_train["scaler"].data_max_)


@pytest.mark.parametrize("sous_jeu", ["FD001", "FD003"])
def test_un_seul_regime(sous_jeu):
    """FD001 et FD003 : pas de K-Means, regime 0 partout, normalisation globale."""
    contexte = preparer_sous_jeu(sous_jeu)
    assert contexte["kmeans"] is None
    assert contexte["nb_regimes"] == 1
    assert contexte["normalisation"] == "globale"
    assert set(contexte["train"]["regime"].unique()) == {0}
    assert set(contexte["test"]["regime"].unique()) == {0}


# ---------------------------------------------------------------------------------------------------
# Regimes operatoires et normalisation par regime (FD002, FD004)
# ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("sous_jeu", ["FD002", "FD004"])
def test_six_regimes_justifies_par_la_silhouette(sous_jeu):
    """Parmi k = 4 a 8, la silhouette du K-Means sur les reglages du train est maximale pour k = 6."""
    train = charger_sous_jeu(sous_jeu, "train")
    tableau = choisir_nombre_regimes(train[COLONNES_REGLAGES], valeurs_k=[4, 5, 6, 7, 8])
    meilleure_ligne = tableau.loc[tableau["silhouette"].idxmax()]
    assert int(meilleure_ligne["k"]) == 6
    assert DESCRIPTION_SOUS_JEUX[sous_jeu]["nb_regimes"] == 6


def test_kmeans_appris_sur_le_train_seulement(contexte_fd002):
    """
    Le K-Means du contexte est celui qu'on obtient en l'entrainant sur le TRAIN seul, et le regime des
    lignes test vient de predict (aucun reapprentissage sur le test).
    """
    train_brut = charger_sous_jeu("FD002", "train")
    test_brut = charger_sous_jeu("FD002", "test")

    # un K-Means appris sur le train seul, avec les memes reglages, a les memes centres. Pas au bit pres :
    # K-Means calcule ses moyennes en parallele (plusieurs coeurs), et l'ordre des additions flottantes
    # peut changer d'une execution a l'autre (ecarts observes de l'ordre de 1e-15). Les REGIMES attribues,
    # eux, sont identiques (verifie juste apres) : seuls ces numeros servent a la suite de la preparation.
    kmeans_train_seul = creer_kmeans(6)
    kmeans_train_seul.fit(train_brut[COLONNES_REGLAGES])
    assert np.allclose(contexte_fd002["kmeans"].cluster_centers_, kmeans_train_seul.cluster_centers_, rtol=0, atol=1e-9)
    assert contexte_fd002["kmeans"].n_features_in_ == 3  # les 3 reglages operatoires, rien d'autre

    # les regimes du train et du test sont ceux de predict
    assert np.array_equal(contexte_fd002["train"]["regime"].to_numpy(), kmeans_train_seul.predict(train_brut[COLONNES_REGLAGES]))
    assert np.array_equal(contexte_fd002["test"]["regime"].to_numpy(), kmeans_train_seul.predict(test_brut[COLONNES_REGLAGES]))


def test_scalers_appris_sur_le_train_seulement(contexte_fd002):
    """Chaque scaler de regime a vu exactement les lignes TRAIN de ce regime : min et max du train."""
    train_brut = charger_sous_jeu("FD002", "train")
    regimes_train = attribuer_regimes(train_brut, contexte_fd002["kmeans"])
    capteurs = contexte_fd002["capteurs"]

    assert len(contexte_fd002["scalers"]) == 6  # un scaler par regime
    for regime in range(6):
        masque = regimes_train == regime
        lignes_regime = train_brut.loc[masque, capteurs]
        scaler = contexte_fd002["scalers"][regime]
        assert scaler.n_samples_seen_ == int(masque.sum())  # aucune ligne test vue par le scaler
        assert np.array_equal(scaler.data_min_, lignes_regime.min().to_numpy())
        assert np.array_equal(scaler.data_max_, lignes_regime.max().to_numpy())


def test_capteurs_normalises_entre_0_et_1_sur_le_train(contexte_fd002):
    """Normalises par regime puis lisses, les capteurs du train restent dans [0, 1]."""
    valeurs = contexte_fd002["train"][contexte_fd002["capteurs"]].to_numpy()
    assert valeurs.min() >= -1e-12
    assert valeurs.max() <= 1 + 1e-12


@pytest.mark.parametrize("nom_fixture", ["contexte_fd002", "contexte_fd004"])
def test_capteurs_retires_constants_dans_chaque_regime(nom_fixture, request):
    """
    Critere documente : un capteur est retire s'il est constant dans TOUS les regimes du train ; un
    capteur garde varie dans au moins un regime.
    """
    contexte = request.getfixturevalue(nom_fixture)  # la fixture dont le nom est donne
    assert contexte["capteurs_retires"] == ["sensor_1", "sensor_5", "sensor_18", "sensor_19"]

    train_brut = charger_sous_jeu(contexte["sous_jeu"], "train")
    train_brut["regime"] = attribuer_regimes(train_brut, contexte["kmeans"])
    tous_les_capteurs = []
    for numero in range(1, 22):
        tous_les_capteurs.append(f"sensor_{numero}")
    valeurs_distinctes = tableau_valeurs_distinctes_par_regime(train_brut, tous_les_capteurs)

    for capteur in contexte["capteurs_retires"]:
        assert valeurs_distinctes.loc[capteur].max() == 1  # une seule valeur dans chaque regime
    for capteur in contexte["capteurs"]:
        assert valeurs_distinctes.loc[capteur].max() > 1  # varie dans au moins un regime

    # la fonction du module redonne la meme liste
    assert trouver_capteurs_constants_par_regime(train_brut, tous_les_capteurs) == contexte["capteurs_retires"]


def test_normalisation_globale_ne_retire_aucun_capteur_sur_fd002():
    """Avec la methode FD001 telle quelle (nunique() == 1 sur tout le train), FD002 ne perd aucun capteur."""
    contexte = preparer_sous_jeu("FD002", normalisation="globale")
    assert contexte["capteurs_retires"] == []
    assert len(contexte["capteurs"]) == 21


def test_regime_en_one_hot():
    """Avec regime_en_feature=True, 6 colonnes 0/1 sont ajoutees a la fin des features, une seule a 1 par ligne."""
    contexte = preparer_sous_jeu("FD002", regime_en_feature=True)
    colonnes = contexte["colonnes_regime"]
    assert colonnes == ["regime_0", "regime_1", "regime_2", "regime_3", "regime_4", "regime_5"]
    assert contexte["feature_cols"][-6:] == colonnes
    sommes = contexte["X"][colonnes].sum(axis=1).to_numpy()
    assert np.all(sommes == 1.0)


def test_decrire_regimes(contexte_fd002):
    """Le resume des regimes compte toutes les lignes train et test."""
    tableau = decrire_regimes(contexte_fd002)
    assert len(tableau) == 6
    assert tableau["nb_lignes_train"].sum() == DESCRIPTION_SOUS_JEUX["FD002"]["nb_lignes_train"]
    assert tableau["nb_lignes_test"].sum() == DESCRIPTION_SOUS_JEUX["FD002"]["nb_lignes_test"]


# ---------------------------------------------------------------------------------------------------
# Cible, absence de NaN, alignement du test
# ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("nom_fixture", ["contexte_fd001", "contexte_fd002", "contexte_fd004"])
def test_pas_de_nan(nom_fixture, request):
    """Aucune valeur manquante dans les features (train et test, tous les cycles) ni dans la cible."""
    contexte = request.getfixturevalue(nom_fixture)
    assert int(contexte["X"].isna().sum().sum()) == 0
    assert int(contexte["y"].isna().sum()) == 0
    assert int(contexte["test"][contexte["feature_cols"]].isna().sum().sum()) == 0
    assert int(contexte["test"][["RUL_vraie", "RUL_vraie_cappee"]].isna().sum().sum()) == 0


@pytest.mark.parametrize("nom_fixture", ["contexte_fd001", "contexte_fd002", "contexte_fd004"])
def test_rul_vraie_au_dernier_cycle(nom_fixture, request):
    """Au dernier cycle de chaque moteur test, RUL_vraie = la valeur de RUL_FD00X.txt."""
    contexte = request.getfixturevalue(nom_fixture)
    rul_fichier = charger_rul_sous_jeu(contexte["sous_jeu"])
    dernier = contexte["dernier_cycle_par_moteur"]
    assert np.array_equal(dernier["RUL_vraie"].to_numpy(), rul_fichier)
    assert np.array_equal(dernier["unit_number"].to_numpy(), np.arange(1, len(rul_fichier) + 1))
    # plafond : RUL_vraie_cappee = min(RUL_vraie, 125) a chaque cycle
    test = contexte["test"]
    assert np.array_equal(test["RUL_vraie_cappee"].to_numpy(), np.minimum(test["RUL_vraie"].to_numpy(), 125))


def test_cible_du_train_plafonnee(contexte_fd002):
    """Cible du train : RUL = 0 au dernier cycle de chaque moteur, RUL_cappee = min(RUL, 125)."""
    train = contexte_fd002["train"]
    derniers = train.groupby("unit_number").tail(1)  # derniere ligne de chaque moteur = panne
    assert np.all(derniers["RUL"].to_numpy() == 0)
    assert np.array_equal(train["RUL_cappee"].to_numpy(), np.minimum(train["RUL"].to_numpy(), 125))


# ---------------------------------------------------------------------------------------------------
# appliquer_preparation : memes features que preparer_sous_jeu, un moteur a la fois, sans le futur
# ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("nom_fixture", ["contexte_fd001", "contexte_fd002"])
def test_appliquer_preparation_sur_un_moteur_test(nom_fixture, request):
    """Preparer un seul moteur test redonne exactement ses features calculees avec toute la flotte."""
    contexte = request.getfixturevalue(nom_fixture)
    test_brut = charger_sous_jeu(contexte["sous_jeu"], "test")
    feature_cols = contexte["feature_cols"]

    for moteur_id in [1, 37, 100]:  # trois moteurs test pris au hasard
        lignes_brutes = test_brut[test_brut["unit_number"] == moteur_id]
        prepare_seul = appliquer_preparation(contexte, lignes_brutes)
        attendu = contexte["test"].loc[lignes_brutes.index]  # memes lignes, preparees avec toute la flotte
        pd.testing.assert_frame_equal(prepare_seul[feature_cols], attendu[feature_cols], check_exact=True)
        assert np.array_equal(prepare_seul["regime"].to_numpy(), attendu["regime"].to_numpy())


def test_appliquer_preparation_est_causale(contexte_fd002):
    """Couper la trajectoire au cycle t ne change aucune feature des cycles <= t (aucun regard vers le futur)."""
    test_brut = charger_sous_jeu("FD002", "test")
    feature_cols = contexte_fd002["feature_cols"]
    lignes_moteur = test_brut[test_brut["unit_number"] == 5]

    complet = appliquer_preparation(contexte_fd002, lignes_moteur)
    coupe = appliquer_preparation(contexte_fd002, lignes_moteur.iloc[:20])  # seulement les 20 premiers cycles
    pd.testing.assert_frame_equal(coupe[feature_cols], complet[feature_cols].iloc[:20], check_exact=True)


def test_appliquer_preparation_refuse_les_entrees_invalides(contexte_fd002):
    """Index en double ou colonne capteur manquante : erreur claire plutot qu'un resultat faux."""
    test_brut = charger_sous_jeu("FD002", "test")
    lignes = test_brut[test_brut["unit_number"] == 1]

    with pytest.raises(ValueError):
        appliquer_preparation(contexte_fd002, pd.concat([lignes, lignes]))  # index en double
    with pytest.raises(ValueError):
        appliquer_preparation(contexte_fd002, lignes.drop(columns=["sensor_2"]))  # capteur garde manquant


# ---------------------------------------------------------------------------------------------------
# Aides du script scripts/generer_predictions_arbres.py (elles ecrivent les CSV relus par le notebook 01b)
# ---------------------------------------------------------------------------------------------------

def charger_module_script():
    """
    Importe scripts/generer_predictions_arbres.py comme un module (scripts/ n'est pas un paquet Python).

    Returns
    -------
    module
        Le module du script (seules les definitions sont executees, pas main()).
    """
    import importlib.util
    import os

    chemin = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts",
                          "generer_predictions_arbres.py")
    specification = importlib.util.spec_from_file_location("generer_predictions_arbres", chemin)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_metriques_par_moteur_calcul_a_la_main():
    """RMSE, penalite NASA et cycles surestimes de plus de 20 : exemple a 2 moteurs calculable a la main."""
    script = charger_module_script()
    y_vrai = [10, 10, 50, 50]
    y_predit = [40, 10, 50, 37]  # moteur 1 : ecarts +30 et 0 ; moteur 2 : ecarts 0 et -13
    groupes = [1, 1, 2, 2]
    tableau = script.metriques_par_moteur(y_vrai, y_predit, groupes)

    assert list(tableau["unit_number"]) == [1, 2]
    assert tableau["surestimes"].tolist() == [1, 0]  # seul l'ecart +30 depasse 20 cycles
    assert tableau["rmse"].iloc[0] == pytest.approx(np.sqrt((30 ** 2 + 0) / 2))
    assert tableau["nasa"].iloc[0] == pytest.approx(np.exp(30 / 10) - 1)  # surestimation : exp(d/10) - 1
    assert tableau["nasa"].iloc[1] == pytest.approx(np.exp(13 / 13) - 1)  # sous-estimation : exp(-d/13) - 1


def test_fusionner_resultats_remplace_seulement_les_lignes_recalculees(tmp_path):
    """Relancer le script sur un sous-jeu remplace ses lignes et garde celles des autres sous-jeux."""
    script = charger_module_script()
    chemin = str(tmp_path / "resume.csv")

    premier = pd.DataFrame({"sous_jeu": ["FD001", "FD002"], "modele": ["xgboost_phase3", "xgboost_phase3"],
                            "rmse": [1.0, 2.0]})
    script.fusionner_resultats(chemin, premier, ["sous_jeu", "modele"])

    relance = pd.DataFrame({"sous_jeu": ["FD002"], "modele": ["xgboost_phase3"], "rmse": [9.0]})
    tableau = script.fusionner_resultats(chemin, relance, ["sous_jeu", "modele"])

    assert list(tableau["sous_jeu"]) == ["FD001", "FD002"]  # une ligne par cle, dans l'ordre des sous-jeux
    assert list(tableau["rmse"]) == [1.0, 9.0]  # FD001 garde, FD002 remplace
    assert pd.read_csv(chemin)["rmse"].tolist() == [1.0, 9.0]  # et c'est bien ce qui est ecrit sur le disque
