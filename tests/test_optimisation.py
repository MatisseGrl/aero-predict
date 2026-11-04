"""
Tests de l'optimisation Optuna (src/optimisation.py) : front de Pareto et regle de choix sur de petits
tableaux calcules a la main, puis mini-etudes de 2 essais sur un sous-ensemble de moteurs (rapides).
"""

import numpy as np
import optuna
import pandas as pd
import pytest

from src.config_modele import valider_config
from src.evaluation import calculer_plis
from src.optimisation import (
    NOMS_ETUDES,
    appliquer_regle_de_choix,
    calculer_front_pareto,
    config_depuis_ligne,
    lancer_etude,
    proposer_configuration,
)


def construire_tableau_synthetique():
    """
    Construit un petit tableau d'essais (2 etudes) dont le front et le choix se calculent a la main.

    essai | etude | RMSE  | NASA   | surestimes_oof | lecture
    A     | e1    | 16.00 | 30 000 | 500            | front (meilleur RMSE)
    B     | e1    | 16.50 | 27 000 | 450            | front, sous le seuil 1,05 x 16 = 16.8
    C     | e2    | 16.50 | 27 000 | 400            | front (exactement les memes objectifs que B : aucun
          |       |       |        |                | des deux ne domine l'autre), moins de surestimes
    D     | e2    | 17.00 | 25 000 | 300            | front, mais au-dessus du seuil (17.0 > 16.8)
    E     | e1    | 16.60 | 31 000 | 100            | domine par A (moins bon RMSE et moins bon NASA)
    F     | e2    | 18.00 | 26 000 | 200            | domine par D
    G     | e1    | 16.70 | 27 000 | 50             | domine par B (meme NASA, moins bon RMSE)

    Attendu : front = {A, B, C, D} ; admissibles = {A, B, C} ; B et C ont le meme NASA (27 000),
    C a moins de surestimations (400 < 450) : C est retenu ; A est le meilleur RMSE. G montre qu'un
    essai a NASA egal mais RMSE moins bon n'est PAS sur le front, meme avec tres peu de surestimes.

    Returns
    -------
    pandas.DataFrame
        Le tableau, avec une colonne "nom" pour lire les resultats.
    """
    lignes = [
        {"nom": "A", "etude": "e1", "numero": 0, "rmse_cv_moyen": 16.00, "nasa_cv_moyen": 30000.0, "surestimes_oof": 500},
        {"nom": "B", "etude": "e1", "numero": 1, "rmse_cv_moyen": 16.50, "nasa_cv_moyen": 27000.0, "surestimes_oof": 450},
        {"nom": "C", "etude": "e2", "numero": 0, "rmse_cv_moyen": 16.50, "nasa_cv_moyen": 27000.0, "surestimes_oof": 400},
        {"nom": "D", "etude": "e2", "numero": 1, "rmse_cv_moyen": 17.00, "nasa_cv_moyen": 25000.0, "surestimes_oof": 300},
        {"nom": "E", "etude": "e1", "numero": 2, "rmse_cv_moyen": 16.60, "nasa_cv_moyen": 31000.0, "surestimes_oof": 100},
        {"nom": "F", "etude": "e2", "numero": 2, "rmse_cv_moyen": 18.00, "nasa_cv_moyen": 26000.0, "surestimes_oof": 200},
        {"nom": "G", "etude": "e1", "numero": 3, "rmse_cv_moyen": 16.70, "nasa_cv_moyen": 27000.0, "surestimes_oof": 50},
    ]
    return pd.DataFrame(lignes)


def test_front_pareto_calcule_a_la_main():
    """Front attendu : A, B, C, D (E est domine par A, F par D, G par B)."""
    tableau = construire_tableau_synthetique()
    masque = calculer_front_pareto(tableau)
    noms_sur_le_front = tableau.loc[masque, "nom"].tolist()
    assert noms_sur_le_front == ["A", "B", "C", "D"]


def test_regle_de_choix_calculee_a_la_main():
    """Seuil 16.8 : A, B, C admissibles ; egalite NASA entre B et C, departagee par les surestimations : C."""
    tableau = construire_tableau_synthetique()
    resultat = appliquer_regle_de_choix(tableau, tolerance=1.05)

    assert resultat["rmse_min_front"] == pytest.approx(16.0)
    assert resultat["seuil_rmse"] == pytest.approx(16.8)
    assert resultat["admissibles"]["nom"].tolist() == ["A", "B", "C"]  # D (17.0) depasse le seuil
    assert resultat["retenu"]["nom"] == "C"
    assert resultat["meilleur_rmse"]["nom"] == "A"
    assert resultat["meme_essai"] is False


def test_regle_de_choix_tolerance_1_redonne_le_meilleur_rmse():
    """Avec une tolerance de 1 (aucune perte de precision acceptee), le retenu est le meilleur RMSE."""
    tableau = construire_tableau_synthetique()
    resultat = appliquer_regle_de_choix(tableau, tolerance=1.0)
    assert resultat["retenu"]["nom"] == "A"
    assert resultat["meme_essai"] is True


def sous_ensemble_moteurs(contexte_train, nb_moteurs):
    """
    Garde seulement les `nb_moteurs` premiers moteurs du train, pour des mini-etudes rapides.

    Parameters
    ----------
    contexte_train : dict
        La fixture `contexte_train` (pipeline complet du train FD001).
    nb_moteurs : int
        Nombre de moteurs a garder (moteurs 1 a nb_moteurs).

    Returns
    -------
    (pandas.DataFrame, pandas.Series, pandas.Series)
        X, y et groups restreints a ces moteurs.
    """
    masque = contexte_train["groups"] <= nb_moteurs
    X = contexte_train["X"][masque].reset_index(drop=True)
    y = contexte_train["y"][masque].reset_index(drop=True)
    groups = contexte_train["groups"][masque].reset_index(drop=True)
    return X, y, groups


@pytest.mark.parametrize("nom_etude", NOMS_ETUDES)
def test_mini_etude_tourne_et_est_reproductible(contexte_train, nom_etude):
    """
    Mini-etude de 2 essais sur 12 moteurs (3 plis, 10 arbres) : elle tourne sans erreur, produit un
    tableau complet et des configurations valides, et redonne exactement les memes reglages et les
    memes metriques quand on la relance avec la meme graine.
    """
    X, y, groups = sous_ensemble_moteurs(contexte_train, nb_moteurs=12)
    plis = calculer_plis(X, y, groups, n_splits=3, random_state=42)
    imposes = {"n_estimators": 10}  # 10 arbres seulement : le test reste rapide

    etude, tableau = lancer_etude(nom_etude, X, y, groups, plis, n_essais=2,
                                  hyperparametres_imposes=imposes, afficher_progression=False)
    assert len(tableau) == 2
    assert set(tableau["etude"]) == {nom_etude}
    assert len(etude.directions) == 2  # bien une etude multi-objectif

    # colonnes indispensables pour la suite (notebook, regle de choix)
    for colonne in ["rmse_cv_moyen", "nasa_cv_moyen", "surestimes_oof", "alertes_manquees_oof",
                    "fausses_alertes_oof", "biais_oof", "mae_oof", "duree_essai_s", "pareto_etude",
                    "parametres_perte_json", "hyperparametres_json"]:
        assert colonne in tableau.columns, f"colonne manquante : {colonne}"
    assert np.all(np.isfinite(tableau["rmse_cv_moyen"].to_numpy(dtype=float)))
    assert tableau["pareto_etude"].sum() >= 1  # au moins un essai sur le front de son etude

    # chaque ligne redonne une configuration valide (meme schema que la demo)
    for position in range(len(tableau)):
        config = config_depuis_ligne(tableau.iloc[position])
        valider_config(config)
        assert config["hyperparametres"]["n_estimators"] == 10

    # reproductibilite : meme graine -> memes reglages et memes metriques (seuls les temps changent)
    _, tableau_bis = lancer_etude(nom_etude, X, y, groups, plis, n_essais=2,
                                  hyperparametres_imposes=imposes, afficher_progression=False)
    for colonne in tableau.columns:
        if colonne.startswith("param_") or colonne in ["rmse_cv_moyen", "nasa_cv_moyen", "surestimes_oof"]:
            assert tableau[colonne].tolist() == tableau_bis[colonne].tolist(), f"colonne {colonne} differente"


def test_espace_de_recherche_respecte_les_bornes():
    """Les reglages tires par Optuna restent dans les bornes de l'etude xgb_nasa (20 tirages)."""
    etude = optuna.create_study(directions=["minimize", "minimize"],
                                sampler=optuna.samplers.RandomSampler(seed=0))
    for _ in range(20):
        essai = etude.ask()  # un essai "a la main", sans fonction objectif
        config = proposer_configuration(essai, "xgb_nasa")
        assert 8.0 <= config["parametres_perte"]["h_sous"] <= 30.0
        assert 3.0 <= config["parametres_perte"]["h_sur"] <= 15.0
        assert 100 <= config["hyperparametres"]["n_estimators"] <= 1200
        assert 0.01 <= config["hyperparametres"]["learning_rate"] <= 0.3
        assert 3 <= config["hyperparametres"]["max_depth"] <= 8
        etude.tell(essai, [1.0, 1.0])  # on "termine" l'essai avec des valeurs factices


def test_etude_inconnue_leve_une_erreur():
    """Un nom d'etude inconnu donne une erreur claire."""
    etude = optuna.create_study(directions=["minimize", "minimize"])
    essai = etude.ask()
    with pytest.raises(ValueError):
        proposer_configuration(essai, "svm_magique")
