"""
Tests des metriques et protocoles d'evaluation (src/evaluation.py), sur des exemples calcules a la main.
"""

import numpy as np
import pytest
from sklearn.dummy import DummyRegressor

from src.evaluation import (
    alertes_manquees,
    calculer_plis,
    comparer_apparie,
    demontrer_constantes_score_degenerees,
    erreurs_par_moteur,
    evaluer_en_validation_croisee,
    evaluer_sur_test,
    fausses_alertes,
    mae,
    nb_surestimes,
    rmse,
    score_nasa,
)


def test_score_nasa_cas_calcules_a_la_main():
    """+20 cycles -> e^2 - 1 ; -20 cycles -> e^(20/13) - 1 ; prediction exacte -> 0."""
    assert score_nasa([100], [120]) == pytest.approx(np.exp(2) - 1)  # surestimation de 20 : exp(20/10) - 1
    assert score_nasa([100], [80]) == pytest.approx(np.exp(20 / 13) - 1)  # sous-estimation de 20
    assert score_nasa([100], [100]) == 0.0  # prediction exacte : aucune penalite
    # la surestimation coute plus cher que la sous-estimation de meme taille
    assert score_nasa([100], [120]) > score_nasa([100], [80])


def test_rmse_mae_et_surestimations():
    """RMSE, MAE et nombre de surestimations > 20 sur un exemple calcule a la main."""
    vrais = [10, 20, 30, 40]
    predits = [13, 16, 30, 65]  # ecarts : +3, -4, 0, +25
    assert rmse(vrais, predits) == pytest.approx(np.sqrt((9 + 16 + 0 + 625) / 4))
    assert mae(vrais, predits) == pytest.approx((3 + 4 + 0 + 25) / 4)
    assert nb_surestimes(vrais, predits, seuil=20) == 1  # seul l'ecart de +25 depasse 20


def test_alertes_manquees_et_fausses_alertes():
    """
    Exemple calcule a la main avec un seuil d'alerte de 30 cycles :

    vraie RUL | prediction | lecture
    10        | 35         | alerte manquee (proche de la panne, mais pas d'alerte)
    25        | 20         | bonne alerte
    40        | 25         | fausse alerte (encore de la marge, mais alerte)
    50        | 60         | pas d'alerte, a raison
    30        | 31         | alerte manquee (cas limite : 30 <= 30 mais 31 > 30)
    31        | 30         | fausse alerte (cas limite : 31 > 30 mais 30 <= 30)
    """
    vrais = [10, 25, 40, 50, 30, 31]
    predits = [35, 20, 25, 60, 31, 30]
    assert alertes_manquees(vrais, predits, seuil_alerte=30) == 2
    assert fausses_alertes(vrais, predits, seuil_alerte=30) == 2


def construire_groupes_synthetiques(nb_moteurs, nb_lignes_par_moteur):
    """
    Construit un vecteur de numeros de moteur : 1,1,...,2,2,...,nb_moteurs.

    Parameters
    ----------
    nb_moteurs : int
        Nombre de moteurs.
    nb_lignes_par_moteur : int
        Nombre de lignes (cycles) par moteur.

    Returns
    -------
    numpy.ndarray
        Le numero de moteur de chaque ligne.
    """
    groupes = []
    for moteur_id in range(1, nb_moteurs + 1):
        for _ in range(nb_lignes_par_moteur):
            groupes.append(moteur_id)
    return np.array(groupes)


def verifier_plis_independamment(plis, groupes):
    """
    Verifie, sans utiliser calculer_plis, qu'aucun moteur n'est des deux cotes d'un pli et que
    chaque moteur est en validation exactement une fois.

    Parameters
    ----------
    plis : list de (numpy.ndarray, numpy.ndarray)
        Les plis a verifier.
    groupes : numpy.ndarray
        Le numero de moteur de chaque ligne.

    Returns
    -------
    None
    """
    compte_validation = {}
    for idx_train, idx_val in plis:
        moteurs_train = set(groupes[idx_train])
        moteurs_val = set(groupes[idx_val])
        assert moteurs_train.isdisjoint(moteurs_val)  # aucun moteur des deux cotes
        for moteur_id in moteurs_val:
            compte_validation[moteur_id] = compte_validation.get(moteur_id, 0) + 1
    for moteur_id in np.unique(groupes):
        assert compte_validation[moteur_id] == 1  # en validation exactement une fois


def test_plis_synthetiques():
    """Sur 20 moteurs synthetiques : plis disjoints par moteur, chaque moteur une fois en validation."""
    groupes = construire_groupes_synthetiques(nb_moteurs=20, nb_lignes_par_moteur=7)
    X = np.zeros((len(groupes), 2))  # le contenu de X n'a pas d'importance pour le decoupage
    y = np.zeros(len(groupes))
    plis = calculer_plis(X, y, groupes, n_splits=5, random_state=42)
    assert len(plis) == 5
    verifier_plis_independamment(plis, groupes)


def test_plis_reels_fd001(contexte_train):
    """Sur les 100 moteurs reels de FD001 : memes verifications, 20 moteurs en validation par pli."""
    groupes = np.asarray(contexte_train["groups"])
    plis = calculer_plis(contexte_train["X"], contexte_train["y"], groupes)
    verifier_plis_independamment(plis, groupes)
    for idx_train, idx_val in plis:
        assert len(np.unique(groupes[idx_val])) == 20


def test_validation_croisee_predictions_hors_pli_alignees():
    """
    Avec un modele qui predit la moyenne, la prediction hors-pli de chaque ligne doit valoir la moyenne
    de y sur les lignes d'ENTRAINEMENT de son pli : verifie que les OOF sont rangees aux bonnes lignes.
    """
    groupes = construire_groupes_synthetiques(nb_moteurs=10, nb_lignes_par_moteur=5)
    generateur = np.random.default_rng(1)
    X = generateur.normal(size=(len(groupes), 3))
    y = generateur.uniform(0, 125, size=len(groupes))
    plis = calculer_plis(X, y, groupes)

    resultats = evaluer_en_validation_croisee(DummyRegressor, X, y, groupes, plis)

    for idx_train, idx_val in plis:
        moyenne_entrainement = np.mean(y[idx_train])
        assert np.allclose(resultats["predictions_oof"][idx_val], moyenne_entrainement)
    assert len(resultats["rmse_par_pli"]) == 5
    assert resultats["rmse_moyen"] == pytest.approx(np.mean(resultats["rmse_par_pli"]))
    # les cles demandees sont presentes
    for cle in ["mae_oof", "alertes_manquees_oof", "fausses_alertes_oof", "biais_oof", "temps_entrainement_par_pli"]:
        assert cle in resultats


def test_evaluer_sur_test_predictions_positives():
    """evaluer_sur_test clippe a 0 : meme un modele qui predit -10 partout donne des predictions >= 0."""
    modele = DummyRegressor(strategy="constant", constant=-10.0)  # predit toujours une RUL negative
    modele.fit(np.zeros((3, 1)), [0, 0, 0])
    X_test = np.zeros((5, 1))
    y_test = np.array([5, 50, 100, 150, 200])

    resultats = evaluer_sur_test(modele, X_test, y_test)

    assert np.all(resultats["predictions"] >= 0)  # aucune RUL negative
    assert resultats["rmse"] == pytest.approx(rmse(y_test, [0, 0, 0, 0, 0]))
    # RMSE contre la RUL plafonnee a 125 : 150 et 200 deviennent 125
    assert resultats["rmse_contre_rul_plafonnee"] == pytest.approx(rmse([5, 50, 100, 125, 125], [0, 0, 0, 0, 0]))


def test_erreurs_par_moteur():
    """Une ligne par moteur, avec RMSE, biais et surestimation maximale calcules a la main."""
    vrais = [10, 20, 30, 40]
    predits = [15, 10, 30, 30]  # moteur 1 : +5, -10 ; moteur 2 : 0, -10
    groupes = [1, 1, 2, 2]
    tableau = erreurs_par_moteur(vrais, predits, groupes)
    assert list(tableau["unit_number"]) == [1, 2]
    assert tableau["biais"].tolist() == pytest.approx([-2.5, -5.0])
    assert tableau["surestimation_max"].tolist() == pytest.approx([5.0, 0.0])  # moteur 2 jamais surestime
    assert tableau["rmse"].tolist() == pytest.approx([np.sqrt((25 + 100) / 2), np.sqrt(100 / 2)])
    assert list(tableau["nb_lignes"]) == [2, 2]


def test_comparer_apparie_detecte_un_decalage():
    """Si A vaut B + 1 (plus un petit bruit) sur 100 moteurs, la difference est detectee par les deux tests."""
    generateur = np.random.default_rng(2)
    b = generateur.uniform(10, 30, size=100)
    a = b + 1.0 + generateur.normal(scale=0.5, size=100)
    resultats = comparer_apparie(a, b)
    assert resultats["n"] == 100
    assert resultats["difference_moyenne"] == pytest.approx(1.0, abs=0.2)
    assert resultats["p_valeur_t"] < 0.001
    assert resultats["p_valeur_wilcoxon"] < 0.001


def test_constantes_du_score_degenerees():
    """
    Agrandir les constantes 13/10 du SCORE ne change aucune prediction et fait tendre le score vers 0 :
    "optimiser" ces constantes de la note serait donc degenere.
    """
    vrais = np.array([10.0, 50.0, 90.0, 120.0])
    predits = np.array([30.0, 40.0, 100.0, 90.0])  # un modele quelconque (ecarts +20, -10, +10, -30)
    tableau = demontrer_constantes_score_degenerees(vrais, predits, facteurs=(1, 10, 100, 1000))

    assert tableau["predictions_inchangees"].all()  # le score ne touche jamais aux predictions
    assert tableau["score_nasa"].iloc[0] == pytest.approx(score_nasa(vrais, predits))  # k = 1 : score officiel
    scores = tableau["score_nasa"].values
    assert np.all(np.diff(scores) < 0)  # le score diminue a chaque agrandissement des constantes
    assert scores[-1] < 0.001 * scores[0]  # et devient negligeable : il ne mesure plus rien
