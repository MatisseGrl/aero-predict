"""
Tests des fabriques et des pertes sur mesure (src/modeles.py) : derivees calculees a la main ou par
differences finies, stabilite numerique, point de depart a la moyenne de y.
"""

import numpy as np
import pytest

from src.modeles import (
    D_MAX_DEFAUT,
    construire_objectif_asymetrique,
    construire_objectif_nasa,
    fabrique_lightgbm_asymetrique,
    fabrique_lightgbm_nasa,
    fabrique_xgboost,
    fabrique_xgboost_asymetrique,
    fabrique_xgboost_nasa,
    gradient_hessienne_quadratique_asymetrique,
    perte_et_derivees_nasa,
    perte_quadratique_asymetrique,
)


def test_fabrique_xgboost_reglages_phase3():
    """Par defaut, fabrique_xgboost redonne exactement les reglages de la Phase 3."""
    parametres = fabrique_xgboost().get_params()
    assert parametres["n_estimators"] == 200
    assert parametres["max_depth"] == 5
    assert parametres["learning_rate"] == 0.05
    assert parametres["random_state"] == 42
    assert fabrique_xgboost(max_depth=6).get_params()["max_depth"] == 6  # un reglage peut etre change


def test_perte_asymetrique_calcul_a_la_main():
    """
    y = 10 partout, p = 13, 10, 6, alpha = 3 :
    residus = +3, 0, -4 ; poids = 3, 1, 1 ; perte = 27, 0, 16 ; gradient = 18, 0, -8 ; hessienne = 6, 2, 2.
    """
    vrais = np.array([10.0, 10.0, 10.0])
    predits = np.array([13.0, 10.0, 6.0])
    gradient, hessienne = gradient_hessienne_quadratique_asymetrique(vrais, predits, alpha=3.0)
    assert gradient.tolist() == [18.0, 0.0, -8.0]
    assert hessienne.tolist() == [6.0, 2.0, 2.0]
    assert perte_quadratique_asymetrique(vrais, predits, alpha=3.0).tolist() == [27.0, 0.0, 16.0]

    # l'objectif passe a XGBoost / LightGBM renvoie exactement les memes valeurs
    objectif = construire_objectif_asymetrique(3.0)
    gradient_objectif, hessienne_objectif = objectif(vrais, predits)
    assert np.array_equal(gradient_objectif, gradient)
    assert np.array_equal(hessienne_objectif, hessienne)


def test_perte_asymetrique_alpha_1_redonne_le_quadratique():
    """Avec alpha = 1 : gradient = 2 * (p - y) et hessienne = 2 partout (perte quadratique classique)."""
    generateur = np.random.default_rng(0)
    vrais = generateur.uniform(0, 125, size=50)
    predits = generateur.uniform(0, 125, size=50)
    gradient, hessienne = gradient_hessienne_quadratique_asymetrique(vrais, predits, alpha=1.0)
    assert np.allclose(gradient, 2.0 * (predits - vrais))
    assert np.allclose(hessienne, 2.0)


@pytest.mark.parametrize("h_sous, h_sur", [(13.0, 10.0), (13.0, 5.0), (20.0, 8.0)])
def test_perte_nasa_derivees_par_differences_finies(h_sous, h_sur):
    """
    Le gradient et la hessienne analytiques de la perte NASA correspondent aux differences finies
    centrees, dans la zone exponentielle ET dans la zone prolongee (|d| > d_max).
    """
    ecarts = np.array([-120.0, -60.0, -49.0, -20.0, -3.0, -0.4, 0.6, 4.0, 25.0, 49.0, 51.0, 80.0, 150.0])
    vrais = np.full(len(ecarts), 60.0)
    predits = vrais + ecarts  # p = y + d
    pas = 1e-4  # petit pas pour les differences finies (loin du point anguleux d = 0)

    perte, gradient, hessienne = perte_et_derivees_nasa(vrais, predits, h_sous, h_sur)
    perte_plus, gradient_plus, _ = perte_et_derivees_nasa(vrais, predits + pas, h_sous, h_sur)
    perte_moins, gradient_moins, _ = perte_et_derivees_nasa(vrais, predits - pas, h_sous, h_sur)

    gradient_numerique = (perte_plus - perte_moins) / (2 * pas)  # derivee de la perte
    hessienne_numerique = (gradient_plus - gradient_moins) / (2 * pas)  # derivee du gradient
    assert np.allclose(gradient, gradient_numerique, rtol=1e-5, atol=1e-8)
    assert np.allclose(hessienne, hessienne_numerique, rtol=1e-5, atol=1e-8)

    # dans la zone exacte (|d| <= d_max), la perte est exactement la forme NASA
    zone_exacte = np.abs(ecarts) <= D_MAX_DEFAUT
    attendu = np.where(ecarts < 0, np.exp(-ecarts / h_sous) - 1, np.exp(ecarts / h_sur) - 1)
    assert np.allclose(perte[zone_exacte], attendu[zone_exacte])


def test_perte_nasa_calcul_a_la_main_et_continuite():
    """
    d = +10 avec h_sur = 10 : L = e - 1, L' = e / 10, L'' = e / 100 ;
    et la perte, le gradient et la hessienne sont continus au raccord d = d_max.
    """
    perte, gradient, hessienne = perte_et_derivees_nasa(np.array([0.0]), np.array([10.0]), 13.0, 10.0)
    assert perte[0] == pytest.approx(np.e - 1)
    assert gradient[0] == pytest.approx(np.e / 10)
    assert hessienne[0] == pytest.approx(np.e / 100)

    # de part et d'autre du raccord, les trois quantites varient tres peu
    juste_avant = np.array([D_MAX_DEFAUT - 1e-7])
    juste_apres = np.array([D_MAX_DEFAUT + 1e-7])
    resultat_avant = perte_et_derivees_nasa(np.array([0.0]), juste_avant, 13.0, 10.0)
    resultat_apres = perte_et_derivees_nasa(np.array([0.0]), juste_apres, 13.0, 10.0)
    for valeur_avant, valeur_apres in zip(resultat_avant, resultat_apres):
        assert valeur_apres[0] == pytest.approx(valeur_avant[0], rel=1e-6)


def test_perte_nasa_stable_pour_des_ecarts_enormes():
    """Meme avec des ecarts de +-10 000 cycles, aucun depassement de capacite (overflow) ni NaN."""
    vrais = np.array([0.0, 0.0])
    predits = np.array([10000.0, -10000.0])
    with np.errstate(over="raise", invalid="raise"):  # transforme un overflow eventuel en erreur
        objectif = construire_objectif_nasa(13.0, 5.0)
        gradient, hessienne = objectif(vrais, predits)
    assert np.all(np.isfinite(gradient))
    assert np.all(np.isfinite(hessienne))
    assert np.all(hessienne > 0)  # hessienne strictement positive : perte convexe


def construire_donnees_synthetiques():
    """
    Construit un petit jeu de regression synthetique dont la cible a une moyenne loin de 0 (~80).

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        X (400 x 3) et y (400 valeurs).
    """
    generateur = np.random.default_rng(3)
    X = generateur.normal(size=(400, 3))
    y = 80.0 + 10.0 * X[:, 0] + generateur.normal(size=400)
    return X, y


def petit_xgboost_asymetrique():
    """Renvoie un XGBoost a perte asymetrique (alpha = 1) avec seulement 20 arbres (test rapide)."""
    return fabrique_xgboost_asymetrique(1.0, n_estimators=20)


def petit_lightgbm_asymetrique():
    """Renvoie un LightGBM a perte asymetrique (alpha = 1) avec seulement 20 arbres (test rapide)."""
    return fabrique_lightgbm_asymetrique(1.0, n_estimators=20)


def petit_xgboost_nasa():
    """Renvoie un XGBoost a perte NASA (13, 10) avec seulement 20 arbres (test rapide)."""
    return fabrique_xgboost_nasa(13.0, 10.0, n_estimators=20)


def petit_lightgbm_nasa():
    """Renvoie un LightGBM a perte NASA (13, 10) avec seulement 20 arbres (test rapide)."""
    return fabrique_lightgbm_nasa(13.0, 10.0, n_estimators=20)


@pytest.mark.parametrize("fabrique", [
    petit_xgboost_asymetrique,
    petit_lightgbm_asymetrique,
    petit_xgboost_nasa,
    petit_lightgbm_nasa,
])
def test_perte_sur_mesure_part_de_la_moyenne(fabrique):
    """
    Avec une perte sur mesure, le modele enveloppe part bien de la moyenne de y : apres seulement
    20 petits arbres (learning_rate 0.05), ses predictions sont deja centrees sur ~80. Sans
    l'enveloppe, XGBoost partirait de 0,5 et LightGBM de 0, et la moyenne predite serait vers 50.
    """
    X, y = construire_donnees_synthetiques()
    modele = fabrique()
    modele.fit(X, y)
    predictions = modele.predict(X)
    assert np.all(np.isfinite(predictions))
    assert np.mean(predictions) == pytest.approx(np.mean(y), abs=3.0)
