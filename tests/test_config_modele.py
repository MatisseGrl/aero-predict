"""
Tests de src/config_modele.py : validation du schema, conversion en types Python (JSON), racine du depot et
construction des modeles depuis une configuration.
"""

import copy
import json
import os

import numpy as np
import pytest

from src.config_modele import (
    CONFIG_PHASE3,
    RACINE_DEPOT,
    construire_fabrique_depuis_config,
    convertir_en_types_python,
    decrire_config,
    valider_config,
)
from src.modeles import RegresseurCentre


def construire_config(famille, perte, parametres_perte=None, hyperparametres=None):
    """
    Construit une configuration minimale pour les tests.

    Parameters
    ----------
    famille : str
        La famille de modele.
    perte : str
        La perte d'apprentissage.
    parametres_perte : dict, optionnel
        Les parametres de la perte ({} par defaut).
    hyperparametres : dict, optionnel
        Les hyperparametres ({} par defaut).

    Returns
    -------
    dict
        La configuration.
    """
    config = {}
    config["famille"] = famille
    config["perte"] = perte
    if parametres_perte is None:
        parametres_perte = {}
    if hyperparametres is None:
        hyperparametres = {}
    config["parametres_perte"] = parametres_perte
    config["hyperparametres"] = hyperparametres
    config["nom"] = f"test {famille} {perte}"
    return config


# Les 8 combinaisons valides, avec des parametres de perte corrects et des modeles petits (rapides a entrainer)
COMBINAISONS_VALIDES = [
    ("xgboost", "native", {}, {"n_estimators": 10}),
    ("xgboost", "quadratique_asymetrique", {"alpha": 2.0}, {"n_estimators": 10}),
    ("xgboost", "nasa", {"h_sous": 13.0, "h_sur": 10.0}, {"n_estimators": 10}),
    ("lightgbm", "native", {}, {"n_estimators": 10}),
    ("lightgbm", "quadratique_asymetrique", {"alpha": 3.0}, {"n_estimators": 10}),
    ("lightgbm", "nasa", {"h_sous": 13.0, "h_sur": 8.0, "d_max": 40.0}, {"n_estimators": 10}),
    ("random_forest", "native", {}, {"n_estimators": 10}),
    ("reference_naive", "native", {}, {}),
]


def jeu_synthetique():
    """
    Petit jeu de donnees synthetique : 200 lignes, 5 features, cible entre 0 et 125.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        X (200 x 5) et y (200,).
    """
    generateur = np.random.default_rng(0)
    X = generateur.uniform(0, 1, size=(200, 5))
    y = 125.0 * X[:, 0] + generateur.normal(0, 5, size=200)  # la cible depend surtout de la 1re feature
    return X, y


# ---------------------------------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------------------------------

def test_racine_depot_absolue():
    """RACINE_DEPOT est un chemin absolu : le dossier parent de tests/ (la racine du depot)."""
    assert os.path.isabs(RACINE_DEPOT)
    racine = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # tests/.. = racine du depot
    assert RACINE_DEPOT == racine


def test_config_phase3_valide_et_conforme():
    """CONFIG_PHASE3 est valide et reprend exactement les reglages XGBoost de la Phase 3."""
    valider_config(CONFIG_PHASE3)
    assert CONFIG_PHASE3["famille"] == "xgboost"
    assert CONFIG_PHASE3["perte"] == "native"
    assert CONFIG_PHASE3["hyperparametres"] == {
        "n_estimators": 200, "max_depth": 5, "learning_rate": 0.05, "random_state": 42,
    }


# ---------------------------------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("famille, perte, parametres_perte, hyperparametres", COMBINAISONS_VALIDES)
def test_validation_combinaisons_valides(famille, perte, parametres_perte, hyperparametres):
    """Les 8 combinaisons famille + perte autorisees passent la validation sans erreur."""
    valider_config(construire_config(famille, perte, parametres_perte, hyperparametres))


def test_validation_champs_optionnels_absents():
    """parametres_perte et hyperparametres peuvent etre absents (valent {} par defaut)."""
    valider_config({"famille": "xgboost", "perte": "native"})


@pytest.mark.parametrize(
    "config, morceau_du_message",
    [
        ("pas un dict", "dictionnaire"),
        ({"perte": "native"}, "famille"),
        ({"famille": "xgboost"}, "perte"),
        ({"famille": "catboost", "perte": "native"}, "Famille de modele inconnue"),
        ({"famille": "xgboost", "perte": "huber"}, "Perte inconnue"),
        (
            {"famille": "random_forest", "perte": "nasa", "parametres_perte": {"h_sous": 13, "h_sur": 10}},
            "Combinaison invalide",
        ),
        (
            {"famille": "reference_naive", "perte": "quadratique_asymetrique", "parametres_perte": {"alpha": 2}},
            "Combinaison invalide",
        ),
        ({"famille": "xgboost", "perte": "quadratique_asymetrique", "parametres_perte": {}}, "manquant"),
        ({"famille": "lightgbm", "perte": "nasa", "parametres_perte": {"h_sous": 13}}, "manquant"),
        ({"famille": "xgboost", "perte": "native", "parametres_perte": {"alpha": 2}}, "aucun parametre"),
        (
            {"famille": "xgboost", "perte": "nasa", "parametres_perte": {"h_sous": 13, "h_sur": 10, "hsur": 5}},
            "inconnu",
        ),
        ({"famille": "xgboost", "perte": "quadratique_asymetrique", "parametres_perte": {"alpha": -1.0}}, "> 0"),
        ({"famille": "xgboost", "perte": "quadratique_asymetrique", "parametres_perte": {"alpha": "2"}}, "nombre"),
        ({"famille": "xgboost", "perte": "quadratique_asymetrique", "parametres_perte": {"alpha": True}}, "nombre"),
        ({"famille": "xgboost", "perte": "native", "parametres_perte": []}, "parametres_perte"),
        ({"famille": "xgboost", "perte": "native", "hyperparametres": [1, 2]}, "hyperparametres"),
        (
            {"famille": "reference_naive", "perte": "native", "hyperparametres": {"n_estimators": 5}},
            "aucun hyperparametre",
        ),
    ],
)
def test_validation_cas_invalides(config, morceau_du_message):
    """Chaque configuration incoherente leve une ValueError dont le message dit ce qui ne va pas."""
    with pytest.raises(ValueError, match=morceau_du_message):
        valider_config(config)


def test_validation_accepte_types_numpy():
    """Un alpha numpy (numpy.float64, numpy.int64), typique d'un resultat de calcul numpy, est accepte."""
    valider_config(construire_config("xgboost", "quadratique_asymetrique", {"alpha": np.float64(2.5)}))
    valider_config(construire_config("lightgbm", "quadratique_asymetrique", {"alpha": np.int64(2)}))


# ---------------------------------------------------------------------------------------------------
# Conversion en types Python (ecriture JSON)
# ---------------------------------------------------------------------------------------------------

def test_convertir_en_types_python_ecrivable_en_json():
    """Les types numpy (entiers, reels, booleens, tableaux) deviennent des types Python, ecrivables en JSON."""
    config = construire_config(
        "lightgbm",
        "nasa",
        {"h_sous": np.float64(13.0), "h_sur": np.float32(8.5), "d_max": 45},
        {"n_estimators": np.int64(300), "num_leaves": 15, "learning_rate": np.float64(0.03)},
    )
    config["metriques_cv"] = {"rmse": np.float64(15.2), "rmse_par_pli": np.array([15.0, 15.4]), "ok": np.bool_(True)}
    config["metriques_test"] = {"rmse": 16.1, "nb": np.int32(100), "plis": (1, 2)}

    convertie = convertir_en_types_python(config)
    assert type(convertie["hyperparametres"]["n_estimators"]) is int  # numpy.int64 -> int
    assert type(convertie["parametres_perte"]["h_sur"]) is float  # numpy.float32 -> float
    assert convertie["metriques_cv"]["rmse_par_pli"] == [15.0, 15.4]  # tableau numpy -> liste
    assert convertie["metriques_cv"]["ok"] is True  # numpy.bool_ -> bool
    assert convertie["metriques_test"]["plis"] == [1, 2]  # tuple -> liste (JSON n'a pas de tuple)
    valider_config(convertie)  # la configuration convertie reste valide

    # aller-retour JSON : le texte ecrit puis relu redonne exactement la valeur convertie
    assert json.loads(json.dumps(convertie)) == convertie

    # l'original n'est pas modifie (on renvoie une NOUVELLE valeur)
    assert type(config["hyperparametres"]["n_estimators"]) is np.int64


# ---------------------------------------------------------------------------------------------------
# Construction des modeles
# ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("famille, perte, parametres_perte, hyperparametres", COMBINAISONS_VALIDES)
def test_fabrique_entraine_et_predit(famille, perte, parametres_perte, hyperparametres):
    """Chaque combinaison valide construit un modele qui s'entraine et predit des valeurs finies."""
    X, y = jeu_synthetique()
    fabrique = construire_fabrique_depuis_config(construire_config(famille, perte, parametres_perte, hyperparametres))

    modele = fabrique()
    modele.fit(X, y)
    predictions = np.asarray(modele.predict(X))

    assert predictions.shape == (200,)
    assert np.all(np.isfinite(predictions))
    if perte != "native":  # les pertes sur mesure passent par l'enveloppe qui part de la moyenne de y
        assert isinstance(modele, RegresseurCentre)
    if famille != "reference_naive":  # un vrai modele doit faire mieux que predire la moyenne
        assert np.std(predictions) > 1.0


@pytest.mark.parametrize("famille, perte, parametres_perte, hyperparametres", COMBINAISONS_VALIDES)
def test_fabrique_renvoie_des_modeles_neufs_distincts(famille, perte, parametres_perte, hyperparametres):
    """Deux appels de la fabrique renvoient deux objets distincts : entrainer l'un ne touche pas l'autre."""
    X, y = jeu_synthetique()
    fabrique = construire_fabrique_depuis_config(construire_config(famille, perte, parametres_perte, hyperparametres))

    modele_1 = fabrique()
    modele_2 = fabrique()
    assert modele_1 is not modele_2

    modele_1.fit(X, y)  # on n'entraine que le premier
    with pytest.raises(Exception):  # le second est toujours vierge : predict doit echouer
        modele_2.predict(X)


def test_fabrique_phase3_redonne_les_reglages_phase3():
    """La fabrique construite depuis CONFIG_PHASE3 donne exactement le XGBoost de la Phase 3."""
    modele = construire_fabrique_depuis_config(CONFIG_PHASE3)()
    parametres = modele.get_params()
    assert type(modele).__name__ == "XGBRegressor"
    assert parametres["n_estimators"] == 200
    assert parametres["max_depth"] == 5
    assert parametres["learning_rate"] == 0.05
    assert parametres["random_state"] == 42


def test_fabrique_independante_de_la_config_modifiee_ensuite():
    """Modifier la configuration apres avoir construit la fabrique ne change pas les modeles produits."""
    config = copy.deepcopy(CONFIG_PHASE3)
    fabrique = construire_fabrique_depuis_config(config)
    config["hyperparametres"]["max_depth"] = 9  # modification APRES la construction
    assert fabrique().get_params()["max_depth"] == 5


def test_fabrique_hyperparametres_transmis():
    """Les hyperparametres de la configuration arrivent bien jusqu'au modele (enveloppe comprise)."""
    config = construire_config(
        "xgboost", "quadratique_asymetrique", {"alpha": 2.0}, {"max_depth": 3, "n_estimators": 7}
    )
    modele = construire_fabrique_depuis_config(config)()
    parametres_internes = modele.modele.get_params()  # le XGBRegressor enveloppe par RegresseurCentre
    assert parametres_internes["max_depth"] == 3
    assert parametres_internes["n_estimators"] == 7


def test_fabrique_config_invalide_refusee():
    """Une configuration invalide est refusee des la construction de la fabrique."""
    with pytest.raises(ValueError):
        construire_fabrique_depuis_config({"famille": "random_forest", "perte": "quadratique_asymetrique"})


def test_decrire_config():
    """Le resume lisible contient le nom, la famille, la perte et les parametres de perte."""
    assert decrire_config(CONFIG_PHASE3) == "XGBoost Phase 3 (xgboost, perte native)"
    description = decrire_config(construire_config("lightgbm", "nasa", {"h_sous": 13.0, "h_sur": 8.0}))
    assert "lightgbm" in description
    assert "h_sous=13.0" in description
    assert "h_sur=8.0" in description
