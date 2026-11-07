"""
Entraine et sauvegarde les poids des modeles retenus au notebook 04h, SANS relancer Optuna.

Les modeles retenus sont des ensembles de 5 LSTM (graines 42 a 46) sur FD001 et sur FD002, avec les
reglages figes dans src/optimisation_reseaux.py (CONFIGS_RESEAUX_RETENUES, ENSEMBLES_RETENUS). Chaque LSTM
est entraine sur tous les moteurs du train, pendant le nombre d'epoques fixe par la validation, exactement
comme l'etape "final" du calcul de nuit (scripts/optimiser_reseaux.py) : memes fonctions, memes graines.

Controle : les predictions au dernier cycle de chaque moteur test sont comparees a celles du calcul de
nuit (docs/resultats/04h/04h*_test_dernier_cycle.csv). Elles doivent etre identiques : sinon le script
s'arrete avec une erreur et rien n'est presente comme valide.

Fichiers ecrits : models/04h/<SOUS_JEU>__lstm_g<graine>.pt (10 fichiers, src.deep_learning.sauvegarder_modele :
poids + type + hyperparametres + informations pour predire : colonnes d'entree, longueur de fenetre, k,
bornes min-max de l'ecart initial). Les anciens poids de la demo (models/FD001__lstm.pt, 03b) ne sont pas
touches.

Lancement, depuis la racine du depot (environ 5 min, 10 processus d'un coeur) :

    python scripts/sauvegarder_poids_04h.py
"""

import os
import sys

# 1 fil de calcul par processus, fixe avant numpy et torch (voir scripts/optimiser_reseaux.py)
for variable in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[variable] = "1"

DOSSIER_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
RACINE_DEPOT = os.path.dirname(DOSSIER_SCRIPTS)
sys.path.insert(0, RACINE_DEPOT)

from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# modules du projet AVANT torch (conflit OpenMP, voir src/deep_learning.py)
from src import optimisation_reseaux as opt  # noqa: E402
from src.deep_learning import sauvegarder_modele  # noqa: E402

DOSSIER_POIDS = os.path.join(RACINE_DEPOT, "models", "04h")
DOSSIER_RESULTATS = os.path.join(RACINE_DEPOT, "docs", "resultats", "04h")
PREFIXES = {"FD001": "04h_", "FD002": "04h_fd002_"}


def entrainer_et_sauvegarder(tache):
    """
    Tache d'un processus : un LSTM final (un sous-jeu, une graine), sauvegarde, predictions au dernier cycle.

    Parameters
    ----------
    tache : dict
        "sous_jeu", "reseau", "graine".

    Returns
    -------
    dict
        La tache, plus "chemin" (fichier ecrit) et "predictions" (une RUL par moteur test).
    """
    sous_jeu = tache["sous_jeu"]
    config = opt.CONFIGS_RESEAUX_RETENUES[sous_jeu][tache["reseau"]]
    modele = opt.entrainer_modele_final(config, config["nb_epoques_final"], graine=tache["graine"], sous_jeu=sous_jeu)
    predictions = opt.predictions_test_dernier_cycle(modele, config, sous_jeu=sous_jeu)

    # tout ce qu'il faut pour refaire une prediction : memes entrees, memes bornes de mise a l'echelle
    donnees = opt.preparer_fenetres(config["entrees"], sous_jeu=sous_jeu)
    tableaux = opt.preparer_tableaux(config["entrees"]["ecart_initial"], config["entrees"].get("pente"), sous_jeu=sous_jeu)
    informations = {
        "sous_jeu": sous_jeu,
        "graine": tache["graine"],
        "nb_epoques": config["nb_epoques_final"],
        "colonnes_entree": donnees["colonnes_entree"],
        "longueur": config["entrees"]["longueur"],
        "ecart_initial": config["entrees"]["ecart_initial"],
        "min_max_ecart_initial": tableaux["min_max_ecart_initial"],
        "configuration": opt.copie_config(config),
        "source": "notebooks/04h_optuna_reseaux.ipynb (reglages Optuna confirmes, calcul de nuit du 2026-11-07)",
    }
    nom = f"{sous_jeu}__{tache['reseau']}_g{tache['graine']}.pt"
    chemin = sauvegarder_modele(modele, os.path.join(DOSSIER_POIDS, nom), config["type"],
                                donnees["fenetres"].shape[2], hyperparametres=config["hyperparametres"],
                                informations=informations)
    resultat = dict(tache)
    resultat["chemin"] = chemin
    resultat["predictions"] = predictions
    return resultat


def main():
    """Entraine les 10 LSTM en parallele, les sauvegarde, puis verifie leurs predictions."""
    taches = []
    for sous_jeu in opt.ENSEMBLES_RETENUS:
        ensemble = opt.ENSEMBLES_RETENUS[sous_jeu]
        for graine in ensemble["graines"]:
            taches.append({"sous_jeu": sous_jeu, "reseau": ensemble["reseau"], "graine": graine})

    resultats = []
    with ProcessPoolExecutor(max_workers=len(taches)) as executeur:
        futurs = []
        for tache in taches:
            futurs.append(executeur.submit(entrainer_et_sauvegarder, tache))
        for futur in as_completed(futurs):
            resultat = futur.result()
            resultats.append(resultat)
            print(f"ecrit : {resultat['chemin']}", flush=True)

    # controle : memes predictions que le calcul de nuit, puis RMSE de l'ensemble
    for sous_jeu in opt.ENSEMBLES_RETENUS:
        nuit = pd.read_csv(os.path.join(DOSSIER_RESULTATS, PREFIXES[sous_jeu] + "test_dernier_cycle.csv"))
        lignes = []
        for resultat in resultats:
            if resultat["sous_jeu"] != sous_jeu:
                continue
            colonne = f"{resultat['reseau']}_g{resultat['graine']}"
            ecart = float(np.max(np.abs(resultat["predictions"] - nuit[colonne].to_numpy())))
            assert ecart < 1e-4, f"{sous_jeu} {colonne} : predictions differentes du calcul de nuit (ecart {ecart})"
            lignes.append(resultat["predictions"])
        moyenne = np.mean(np.vstack(lignes), axis=0)  # l'ensemble retenu
        metriques = opt.evaluer_sur_test_dernier_cycle(moyenne, nuit["rul_vraie"].to_numpy())
        print(f"{sous_jeu} : {len(lignes)} LSTM identiques au calcul de nuit ; ensemble au test : RMSE "
              f"{metriques['rmse']:.2f} / plafonnee {metriques['rmse_contre_rul_plafonnee']:.2f}", flush=True)


if __name__ == "__main__":
    main()
