"""
Tests de la decision d'alerte (src/alertes.py) : cas calcules a la main, egalites au seuil comprises, et
coherence avec les comptes `alertes_manquees` / `fausses_alertes` de src/evaluation.py.
"""

import numpy as np
import pytest

from src.alertes import (
    SEUIL_ALERTE_DEFAUT,
    STATUT_ALERTE_CORRECTE,
    STATUT_ALERTE_MANQUEE,
    STATUT_FAUSSE_ALERTE,
    STATUT_RIEN_A_SIGNALER,
    classer_alerte,
)
from src.evaluation import alertes_manquees, fausses_alertes


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


def test_classer_alerte_seuil_par_defaut():
    """Sans seuil explicite, c'est SEUIL_ALERTE_DEFAUT (30 cycles) qui s'applique."""
    assert SEUIL_ALERTE_DEFAUT == 30.0
    assert classer_alerte(30.0, 30.0) == (True, STATUT_ALERTE_CORRECTE)  # egal au seuil par defaut : alerte
    assert classer_alerte(31.0, 29.0) == (False, STATUT_ALERTE_MANQUEE)  # juste au-dessus : pas d'alerte


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
