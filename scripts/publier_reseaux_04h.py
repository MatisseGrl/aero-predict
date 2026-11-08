"""
Publie les reseaux du notebook 04h dans la demo : predictions precalculees de sorties/predictions/, a partir
des poids deja sauvegardes dans models/04h/ (scripts/sauvegarder_poids_04h.py). Rien n'est entraine.

Pour chaque sous-jeu (FD001 a FD004) et chaque reseau :

- LSTM : moyenne simple des 5 LSTM retenus (graines 42 a 46, ENSEMBLES_RETENUS) ; reglages Optuna sur FD001,
  FD002 et FD003, reglages du 03b sur FD004 (le meilleur essai Optuna n'y a pas ete confirme) ;
- CNN 1D : configuration retenue, graine 42 (FD001 et FD003 : reglages Optuna ; FD002 et FD004 : reglages du
  03b, le meilleur essai Optuna n'y a pas ete confirme).

Les memes graines que la demo en direct (src.inference.graines_reseau) : precalcule et direct concordent
(test anti-triche (a), tests/test_anti_triche.py).

Fichiers ecrits, au format du contrat (src/contrat_predictions.py) : sorties/predictions/<SOUS_JEU>__lstm.csv.gz
et __cnn1d.csv.gz, avec leurs .json. Controle : le RMSE au dernier cycle doit etre celui du calcul de nuit
(docs/resultats/04h/04h*_test.json) ; sinon le script s'arrete. Les anciens fichiers du 03b ont ete deplaces
dans sorties/predictions_archive_03b/.

Lancement, depuis la racine du depot (quelques secondes par reseau) :

    python scripts/publier_reseaux_04h.py                          # les quatre sous-jeux
    python scripts/publier_reseaux_04h.py --sous-jeux FD003 FD004  # certains sous-jeux seulement
"""

import argparse
import datetime
import json
import os
import sys

for variable in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[variable] = "1"

DOSSIER_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
RACINE_DEPOT = os.path.dirname(DOSSIER_SCRIPTS)
sys.path.insert(0, RACINE_DEPOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# modules du projet AVANT torch (conflit OpenMP, voir src/deep_learning.py)
from src import inference as inf  # noqa: E402
from src import optimisation_reseaux as opt  # noqa: E402
from src.contrat_predictions import (  # noqa: E402
    MODELES_CONNUS,
    calculer_rul_vraie,
    ecrire_predictions,
    metriques_dernier_cycle,
)
from src.deep_learning import charger_modele, predire  # noqa: E402

DOSSIER_RESULTATS = os.path.join(RACINE_DEPOT, "docs", "resultats", "04h")
# prefixe des fichiers de resultats de chaque sous-jeu dans docs/resultats/04h/ (FD001 : sans numero)
PREFIXES = {"FD001": "04h_", "FD002": "04h_fd002_", "FD003": "04h_fd003_", "FD004": "04h_fd004_"}
COLONNE_NUIT = {"lstm": "lstm_5_graines", "cnn1d": "cnn1d_g42"}  # colonne du calcul de nuit a retrouver


def lire_json(sous_jeu, nom):
    """Relit un fichier de resultats du 04h (copie versionnee)."""
    with open(os.path.join(DOSSIER_RESULTATS, PREFIXES[sous_jeu] + nom + ".json"), encoding="utf-8") as fichier:
        return json.load(fichier)


def validation_et_protocole(sous_jeu, modele):
    """
    Metriques de validation (3 decoupages neufs) et phrase de protocole du reseau publie, lues dans le 04h.

    Returns
    -------
    (dict, str)
        Les metriques de validation et le texte du protocole.
    """
    confirmation = lire_json(sous_jeu, "confirmation_" + modele)
    cle = "cv_candidat" if confirmation["comparaison"]["retenu"] else "cv_reference"
    rmse_neufs = []
    for graine in ["1000", "1001", "1002"]:
        rmse_neufs.append(confirmation[cle][graine]["rmse_cv_moyen"])
    metriques = {"rmse_cv_decoupages_neufs": float(np.mean(rmse_neufs)),
                 "rmse_cv_par_decoupage": rmse_neufs}
    protocole = ("Validation croisee 5 plis par moteur sur 3 decoupages neufs (graines 1000 a 1002), "
                 "notebook 04h. ")
    if modele == "lstm":
        ensemble = lire_json(sous_jeu, "ensemble")["ensembles"]["lstm_5_graines"]
        metriques["rmse_cv_decoupages_neufs"] = float(np.mean(ensemble["rmse_cv_par_decoupage"]))
        metriques["rmse_cv_par_decoupage"] = ensemble["rmse_cv_par_decoupage"]
        # reglages du LSTM : ceux d'Optuna s'ils ont passe la confirmation (FD001 a FD003), sinon ceux du 03b (FD004)
        if confirmation["comparaison"]["retenu"]:
            reglages = "reglages Optuna confirmes"
        else:
            reglages = "reglages du 03b (le meilleur essai Optuna n'a pas ete confirme sur ce sous-jeu)"
        protocole = protocole + ("Moyenne simple de 5 LSTM (graines 42 a 46), " + reglages + " ; "
                                 "retenue contre un seul LSTM (gain >= 0,3 cycle, 3 decoupages sur 3, Wilcoxon p < 0,05).")
    elif confirmation["comparaison"]["retenu"]:
        protocole = protocole + "CNN 1D, graine 42, reglages Optuna confirmes (non retenu comme modele final)."
    else:
        protocole = protocole + ("CNN 1D, graine 42, reglages du 03b : le meilleur essai Optuna n'a pas ete "
                                 "confirme sur ce sous-jeu.")
    return metriques, protocole


def publier(sous_jeu, modele):
    """Predictions de tous les cycles des moteurs test, moyenne des graines, ecriture au format du contrat."""
    config = opt.CONFIGS_RESEAUX_RETENUES[sous_jeu][modele]
    donnees = opt.preparer_fenetres(config["entrees"], sous_jeu=sous_jeu)  # meme preparation qu'a l'entrainement
    tableaux = opt.preparer_tableaux(config["entrees"]["ecart_initial"], config["entrees"].get("pente"), sous_jeu=sous_jeu)
    test = tableaux["test"]

    chemins = inf.chemins_poids_reseau(sous_jeu, modele)
    somme = np.zeros(len(test))
    for chemin in chemins:  # un reseau par graine
        reseau, contenu = charger_modele(chemin)
        assert contenu["informations"]["colonnes_entree"] == donnees["colonnes_entree"], chemin
        somme = somme + predire(reseau, donnees["fenetres_test"])
    predictions = somme / len(chemins)  # moyenne simple

    df = pd.DataFrame()
    df["unit_number"] = test["unit_number"].to_numpy()
    df["time_cycles"] = test["time_cycles"].to_numpy()
    df["rul_predite"] = np.clip(predictions, a_min=0.0, a_max=None)
    if "regime" in test.columns:
        df["regime"] = test["regime"].to_numpy().astype(int)
    else:
        df["regime"] = 0  # FD001 et FD003 : un seul regime
    df = calculer_rul_vraie(df, tableaux["y_test_vraie"])
    assert np.array_equal(df["rul_vraie"].to_numpy(dtype=float), test["RUL_vraie"].to_numpy(dtype=float))

    metriques_test = metriques_dernier_cycle(df)
    nuit = lire_json(sous_jeu, "test")["metriques_test"][COLONNE_NUIT[modele]]
    assert abs(metriques_test["rmse"] - nuit["rmse"]) < 1e-4, (sous_jeu, modele, metriques_test["rmse"], nuit["rmse"])

    metriques_validation, protocole = validation_et_protocole(sous_jeu, modele)
    meta = {}
    meta["sous_jeu"] = sous_jeu
    meta["modele"] = modele
    meta["nom_affiche"] = MODELES_CONNUS[modele]
    meta["famille"] = "sequentiel"
    meta["date"] = datetime.date.today().isoformat()
    meta["source"] = "scripts/publier_reseaux_04h.py (reseaux du notebook 04h, poids de models/04h/)"
    meta["hyperparametres"] = config["hyperparametres"]
    meta["reglages_entrainement"] = config["reglages"]
    meta["colonnes_entree"] = donnees["colonnes_entree"]
    meta["longueur_fenetre"] = config["entrees"]["longueur"]
    meta["ecart_initial"] = config["entrees"]["ecart_initial"]
    meta["nb_epoques_modele_final"] = config["nb_epoques_final"]
    meta["graines"] = inf.graines_reseau(sous_jeu, modele)
    meta["chemins_poids"] = []
    for chemin in chemins:
        meta["chemins_poids"].append(os.path.relpath(chemin, RACINE_DEPOT).replace("\\", "/"))
    meta["metriques_test"] = metriques_test
    meta["metriques_validation"] = metriques_validation
    meta["protocole_validation"] = protocole
    meta["duree_entrainement_s"] = 0.0  # non mesuree (poids entraines par scripts/sauvegarder_poids_04h.py)
    meta["note_duree_entrainement"] = "duree non mesuree : les poids viennent de scripts/sauvegarder_poids_04h.py"
    chemin_csv, _ = ecrire_predictions(df, meta)
    print(f"{sous_jeu} {modele} ({len(chemins)} reseau(x)) : test RMSE {metriques_test['rmse']:.2f} "
          f"(calcul de nuit {nuit['rmse']:.2f}) -> {chemin_csv}", flush=True)


def main(argv=None):
    """
    Publie le LSTM et le CNN 1D des sous-jeux demandes (par defaut, les quatre de ENSEMBLES_RETENUS).

    Parameters
    ----------
    argv : list de str, optionnel
        Les arguments (sans le nom du programme) ; par defaut ceux de la ligne de commande.
    """
    analyseur = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # sous-jeux a traiter : par defaut tous ceux qui ont un ensemble retenu
    analyseur.add_argument("--sous-jeux", dest="sous_jeux", nargs="+", default=list(opt.ENSEMBLES_RETENUS),
                           choices=list(opt.ENSEMBLES_RETENUS))
    arguments = analyseur.parse_args(argv)
    for sous_jeu in arguments.sous_jeux:
        for modele in inf.MODELES_RESEAUX:
            publier(sous_jeu, modele)


if __name__ == "__main__":
    main()
