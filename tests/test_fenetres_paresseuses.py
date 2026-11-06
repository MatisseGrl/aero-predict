"""
Les fenetres paresseuses (src.deep_learning.FenetresParesseuses) sont EXACTEMENT les fenetres de
`construire_fenetres`, et un apprentissage donne les memes predictions avec l'une ou l'autre.

Pourquoi ce test ? Depuis le 2026-11-07, l'optimisation des reseaux (src/optimisation_reseaux.py) construit
les fenetres lot par lot pour economiser la memoire. Les essais Optuna deja faits sur FD001 l'ont ete avec le
tenseur 3D complet : ils ne restent comparables aux nouveaux que si les deux methodes donnent les memes
nombres au bit pres. Lancement, depuis la racine du depot : python -m pytest -q tests/test_fenetres_paresseuses.py

(a) `test_memes_fenetres` : FD001 et FD002 (un et six regimes), train et test, fenetres de 20, 30 et 50 cycles ;
    toutes les fenetres, puis un sous-ensemble (`selection`) et une tranche, sont identiques (np.array_equal) ;
(b) `test_meme_apprentissage` : LSTM entraine 2 epoques sur 10 moteurs de FD001, une fois avec le tenseur 3D,
    une fois avec les fenetres paresseuses (meme graine) : predictions identiques au bit pres.
"""

import os
import sys

import numpy as np
import pytest

# racine du depot (dossier parent de tests/) importable, pour "import src..." quel que soit le dossier de lancement
RACINE_DEPOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RACINE_DEPOT not in sys.path:
    sys.path.insert(0, RACINE_DEPOT)

# src.optimisation_reseaux importe les modules du projet AVANT torch (ordre voulu, voir src/deep_learning.py)
from src import optimisation_reseaux as opt  # noqa: E402
from src.deep_learning import (  # noqa: E402
    construire_fenetres,
    construire_fenetres_paresseuses,
    creer_modele,
    entrainer_modele,
    predire,
)


@pytest.mark.parametrize("sous_jeu", ["FD001", "FD002"])
@pytest.mark.parametrize("longueur", [20, 30, 50])
def test_memes_fenetres(sous_jeu, longueur):
    """Fenetres paresseuses = tenseur 3D de construire_fenetres, valeur par valeur."""
    tableaux = opt.preparer_tableaux(20, sous_jeu=sous_jeu)
    colonnes = opt.colonnes_des_entrees(tableaux, {"capteurs": True})
    for nom in ["train", "test"]:  # le train et le test (moteurs test courts : remplissage en debut de vie)
        df = tableaux[nom]
        completes, _, _, _ = construire_fenetres(df, colonnes, longueur=longueur)
        paresseuses, _, _, _ = construire_fenetres_paresseuses(df, colonnes, longueur=longueur)
        assert paresseuses.shape == completes.shape
        assert np.array_equal(paresseuses[np.arange(len(paresseuses))], completes)

        # sous-ensemble paresseux (comme un pli de validation croisee), puis une tranche (comme un lot)
        generateur = np.random.default_rng(0)
        indices = np.sort(generateur.choice(len(df), size=len(df) // 3, replace=False))
        sous_ensemble = paresseuses.selection(indices)
        assert np.array_equal(sous_ensemble[np.arange(len(sous_ensemble))], completes[indices])
        assert np.array_equal(sous_ensemble[10:500], completes[indices][10:500])


def test_meme_apprentissage():
    """Meme graine, memes donnees : les deux methodes donnent les memes poids, donc les memes predictions."""
    tableaux = opt.preparer_tableaux(20)
    colonnes = opt.colonnes_des_entrees(tableaux, {"capteurs": True})
    df = tableaux["train"]
    lignes = np.flatnonzero(df["unit_number"].to_numpy() <= 10)  # 10 moteurs : quelques secondes
    apprentissage = lignes[df["unit_number"].to_numpy()[lignes] <= 8]  # moteurs 1 a 8
    arret = lignes[df["unit_number"].to_numpy()[lignes] > 8]  # moteurs 9 et 10 : arret precoce
    y = tableaux["y"]

    completes, _, _, _ = construire_fenetres(df, colonnes, longueur=30)
    paresseuses, _, _, _ = construire_fenetres_paresseuses(df, colonnes, longueur=30)

    predictions = []
    for fenetres_apprentissage, fenetres_arret in [
        (completes[apprentissage], completes[arret]),
        (paresseuses.selection(apprentissage), paresseuses.selection(arret)),
    ]:
        modele = creer_modele("lstm", len(colonnes), graine=42)
        entrainer_modele(modele, fenetres_apprentissage, y[apprentissage], fenetres_validation=fenetres_arret,
                         y_validation=y[arret], max_epoques=2, patience=1, graine=42)
        predictions.append(predire(modele, fenetres_arret))
    assert np.array_equal(predictions[0], predictions[1])
