"""
Lance les etudes Optuna de la Phase 4 (calcul lourd) et ecrit un CSV d'essais par etude dans docs/resultats/.

Trois etudes multi-objectif (minimiser le RMSE CV moyen ET le score NASA CV moyen), toutes evaluees
sur les MEMES 5 plis GroupKFold par moteur que la Phase 3 (calcules une seule fois) :

- xgb_quadratique  : XGBoost + perte quadratique asymetrique (alpha dans [1, 4]) ;
- xgb_nasa         : XGBoost + perte de forme NASA (h_sur dans [3, 15], h_sous dans [8, 30]) ;
- lgbm_quadratique : LightGBM + perte quadratique asymetrique (alpha dans [1, 4]).

Les espaces de recherche et la logique sont dans src/optimisation.py ; ce script ne fait que les
enchainer, afficher la progression et ecrire les resultats. Le notebook 04 relit ensuite les CSV
(sans relancer Optuna).

Exemples, depuis la racine du depot :

    python scripts/lancer_optuna.py                                  # les 3 etudes, l'une apres l'autre
    python scripts/lancer_optuna.py --parallele                      # les 3 etudes en meme temps (3 processus)
    python scripts/lancer_optuna.py --etudes xgb_nasa --n-essais 5   # une seule etude, 5 essais
    python scripts/lancer_optuna.py --dossier-sortie /tmp/essai      # ecrire ailleurs (ex. pour tester)

Reproductibilite : a graine egale (--graine, 42 par defaut), memes donnees et meme environnement, on
obtient les memes essais et les memes metriques (seuls les temps de calcul changent). Le mode
--parallele ne change rien aux resultats : chaque etude reste sequentielle dans son propre processus.
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

# on ajoute la racine du depot (dossier parent de scripts/) au chemin d'import, pour "import src..."
DOSSIER_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
RACINE_DEPOT = os.path.dirname(DOSSIER_SCRIPTS)
sys.path.insert(0, RACINE_DEPOT)

import lightgbm  # noqa: E402
import xgboost  # noqa: E402

from src.evaluation import calculer_plis  # noqa: E402
from src.features import preparer_train  # noqa: E402
from src.optimisation import (  # noqa: E402
    DOSSIER_RESULTATS,
    GRAINE_OPTUNA,
    N_ESSAIS_PAR_DEFAUT,
    NOMS_ETUDES,
    chemin_csv_etude,
    lancer_etude,
)


def lire_arguments():
    """
    Lit les options de la ligne de commande.

    Returns
    -------
    argparse.Namespace
        etudes (list de str), n_essais (int ou None), graine (int), dossier_sortie (str), parallele (bool).
    """
    analyseur = argparse.ArgumentParser(description="Etudes Optuna de la Phase 4 (FD001).")
    analyseur.add_argument("--etudes", nargs="+", choices=NOMS_ETUDES, default=NOMS_ETUDES,
                           help="etudes a lancer (toutes par defaut)")
    analyseur.add_argument("--n-essais", type=int, default=None,
                           help=f"nombre d'essais par etude (defaut : {N_ESSAIS_PAR_DEFAUT})")
    analyseur.add_argument("--graine", type=int, default=GRAINE_OPTUNA,
                           help="graine du TPESampler (42 par defaut)")
    analyseur.add_argument("--dossier-sortie", default=DOSSIER_RESULTATS,
                           help="dossier ou ecrire les CSV (docs/resultats/ par defaut)")
    analyseur.add_argument("--parallele", action="store_true",
                           help="lance les etudes en meme temps, une par processus (resultats identiques)")
    return analyseur.parse_args()


def executer_une_etude(nom_etude, n_essais, graine, dossier_sortie):
    """
    Prepare les donnees, lance UNE etude Optuna et ecrit son CSV.

    Chaque etude prepare ses donnees elle-meme (quelques secondes) : la fonction peut ainsi tourner
    seule dans un processus separe (mode --parallele) sans rien partager avec les autres.

    Parameters
    ----------
    nom_etude : str
        Le nom de l'etude.
    n_essais : int
        Nombre d'essais.
    graine : int
        Graine du TPESampler.
    dossier_sortie : str
        Dossier ou ecrire le CSV.

    Returns
    -------
    (str, int, float)
        Le chemin du CSV ecrit, le nombre d'essais termines et la duree de l'etude en secondes.
    """
    print(f"[{nom_etude}] preparation des donnees et des plis...", flush=True)
    contexte_train = preparer_train(fenetre=5, plafond=125)  # meme pipeline que la Phase 3
    X = contexte_train["X"]
    y = contexte_train["y"]
    groups = contexte_train["groups"]

    # les MEMES plis que la Phase 3 et que le notebook 04 : GroupKFold(5, shuffle=True, random_state=42)
    plis = calculer_plis(X, y, groups, n_splits=5, random_state=42)

    print(f"[{nom_etude}] debut : {n_essais} essais, graine {graine}", flush=True)
    debut = time.perf_counter()
    etude, tableau = lancer_etude(nom_etude, X, y, groups, plis, n_essais=n_essais, graine=graine)
    duree = time.perf_counter() - debut

    # versions des bibliotheques de modeles, pour la tracabilite (meme valeur sur chaque ligne)
    tableau["version_xgboost"] = xgboost.__version__
    tableau["version_lightgbm"] = lightgbm.__version__

    os.makedirs(dossier_sortie, exist_ok=True)  # cree docs/resultats/ si besoin
    chemin = chemin_csv_etude(nom_etude, dossier_sortie)
    tableau.to_csv(chemin, index=False)

    nb_pareto = int(tableau["pareto_etude"].sum())
    print(f"[{nom_etude}] termine en {duree / 60:.1f} min : {len(tableau)} essais, {nb_pareto} sur le front "
          f"de Pareto de l'etude -> {chemin}", flush=True)
    return chemin, len(tableau), duree


def main():
    """
    Lance les etudes demandees (l'une apres l'autre, ou en parallele avec --parallele) et resume.

    Returns
    -------
    None
    """
    arguments = lire_arguments()

    # nombre d'essais de chaque etude : celui de la ligne de commande, sinon la valeur par defaut
    essais_par_etude = {}
    for nom_etude in arguments.etudes:
        if arguments.n_essais is not None:
            essais_par_etude[nom_etude] = arguments.n_essais
        else:
            essais_par_etude[nom_etude] = N_ESSAIS_PAR_DEFAUT[nom_etude]

    print(f"Etudes : {essais_par_etude} | graine {arguments.graine} | sortie : {arguments.dossier_sortie} | "
          f"parallele : {arguments.parallele}", flush=True)

    debut = time.perf_counter()
    bilans = []  # un tuple (nom, chemin, nb_essais, duree) par etude

    if arguments.parallele:
        # une etude par processus ; chaque modele utilise N_JOBS_MODELE = 4 coeurs (3 x 4 = 12 coeurs)
        with ProcessPoolExecutor(max_workers=len(arguments.etudes)) as executeur:
            taches = {}
            for nom_etude in arguments.etudes:
                tache = executeur.submit(executer_une_etude, nom_etude, essais_par_etude[nom_etude],
                                         arguments.graine, arguments.dossier_sortie)
                taches[nom_etude] = tache
            for nom_etude in arguments.etudes:  # on attend chaque etude, dans l'ordre demande
                chemin, nb_essais, duree = taches[nom_etude].result()
                bilans.append((nom_etude, chemin, nb_essais, duree))
    else:
        for nom_etude in arguments.etudes:  # l'une apres l'autre
            chemin, nb_essais, duree = executer_une_etude(nom_etude, essais_par_etude[nom_etude],
                                                          arguments.graine, arguments.dossier_sortie)
            bilans.append((nom_etude, chemin, nb_essais, duree))

    duree_totale = time.perf_counter() - debut
    print("\nBilan :")
    for nom_etude, chemin, nb_essais, duree in bilans:
        print(f"  {nom_etude:18s} {nb_essais:3d} essais en {duree / 60:5.1f} min -> {chemin}")
    print(f"Duree totale du script : {duree_totale / 60:.1f} min")


if __name__ == "__main__":
    main()
