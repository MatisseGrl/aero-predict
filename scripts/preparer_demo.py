"""
Prepare tout ce que le tableau de bord de demo (app/tableau_de_bord.py) lit, en une commande.

Utilisation, depuis la racine du depot :

    python scripts/preparer_demo.py                              # tout : arbres, reseaux du 04h, artefacts
    python scripts/preparer_demo.py --seulement-arbres           # sans les reseaux
    python scripts/preparer_demo.py --sous-jeux FD001 FD003      # seulement certains sous-jeux

Etapes, chacune dans un SOUS-PROCESSUS separe, avec le meme interpreteur Python que ce script :
1. scripts/generer_predictions_arbres.py  -> predictions de la reference naive, de la Random Forest, de XGBoost
                                             Phase 3 et de LightGBM (sorties/predictions/) ;
2. scripts/publier_reseaux_04h.py         -> predictions LSTM et CNN 1D des sous-jeux demandes (les quatre ont des
                                             reseaux depuis le calcul de nuit FD003 / FD004 du 2026-11-07 au 08 ;
                                             avant, FD001 et FD002 seulement, decision du 2026-11-07), a partir des
                                             poids du notebook 04h deja sauvegardes (models/04h/) ; rien n'est
                                             entraine (etape sautee avec --seulement-arbres). Avant la reorganisation, cette
                                             etape lancait scripts/entrainer_deep_learning.py (reseaux du 03b),
                                             qui ecrasait les fichiers du 04h que la demo utilise depuis le
                                             commit efe9434 ; ce script reste disponible a part ;
3. python -m src.inference --forcer       -> artefacts de ces quatre modeles pour l'inference en direct
                                             (models/<SOUS_JEU>__<modele>.joblib), reentraines pour correspondre aux
                                             predictions ecrites, puis verification : direct = fichiers precalcules.

Pourquoi des sous-processus ? Charger PyTorch et XGBoost / LightGBM dans le meme processus peut le bloquer
(conflit entre leurs bibliotheques de calcul parallele OpenMP, voir src/deep_learning.py) : chaque etape a
donc son propre processus. Le code de sortie vaut 1 si une etape a echoue, 0 sinon.
"""

import argparse
import os
import subprocess
import sys
import time

# racine du depot = dossier parent de scripts/ (les commandes sont lancees depuis ce dossier)
DOSSIER_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
RACINE_DEPOT = os.path.dirname(DOSSIER_SCRIPTS)

SOUS_JEUX = ["FD001", "FD002", "FD003", "FD004"]

# sous-jeux qui ont des reseaux du 04h : les cles de ENSEMBLES_RETENUS (src/optimisation_reseaux.py), recopiees
# ici pour ne pas charger PyTorch dans ce script (il ne fait que lancer des sous-processus). Les quatre depuis le
# calcul de nuit FD003 / FD004 du 2026-11-07 au 08 (avant : FD001 et FD002). Si cette liste contenait un sous-jeu
# sans ensemble retenu, scripts/publier_reseaux_04h.py le refuserait (choices de --sous-jeux) et l'etape echouerait.
SOUS_JEUX_AVEC_RESEAUX = ["FD001", "FD002", "FD003", "FD004"]


def lancer_etape(nom, commande):
    """
    Lance une commande dans un sous-processus (sortie affichee en direct) et chronometre sa duree.

    Parameters
    ----------
    nom : str
        Le nom de l'etape (pour les messages).
    commande : list de str
        La commande complete (interpreteur, script, options).

    Returns
    -------
    bool
        Vrai si la commande a reussi (code de sortie 0).
    """
    print("\n" + "=" * 86)
    print(f"  {nom} : {' '.join(commande[1:])}")
    print("=" * 86, flush=True)
    debut = time.perf_counter()
    execution = subprocess.run(commande, cwd=RACINE_DEPOT)  # meme environnement, sortie en direct
    reussie = execution.returncode == 0
    etat = "reussie"
    if not reussie:
        etat = f"ECHEC (code {execution.returncode})"
    print(f"  -> {nom} : {etat} en {time.perf_counter() - debut:.1f} s", flush=True)
    return reussie


def main(argv=None):
    """
    Enchaine les trois etapes de preparation.

    Parameters
    ----------
    argv : list de str, optionnel
        Les arguments (sans le nom du programme) ; par defaut ceux de la ligne de commande.

    Returns
    -------
    int
        0 si toutes les etapes lancees ont reussi, 1 sinon.
    """
    parseur = argparse.ArgumentParser(description="Prepare les fichiers lus par l'application de demo.")
    parseur.add_argument("--seulement-arbres", action="store_true",
                         help="ne publie pas les reseaux LSTM et CNN 1D du notebook 04h")
    parseur.add_argument("--sous-jeux", nargs="+", choices=SOUS_JEUX, default=SOUS_JEUX,
                         help="sous-jeux a traiter (defaut : les quatre)")
    arguments = parseur.parse_args(argv)

    # option --sous-jeux transmise telle quelle aux trois etapes
    option_sous_jeux = ["--sous-jeux"]
    for sous_jeu in arguments.sous_jeux:
        option_sous_jeux.append(sous_jeu)

    # les sous-jeux demandes qui ont des reseaux du 04h (option --sous-jeux de publier_reseaux_04h.py)
    option_reseaux = ["--sous-jeux"]
    for sous_jeu in arguments.sous_jeux:
        if sous_jeu in SOUS_JEUX_AVEC_RESEAUX:
            option_reseaux.append(sous_jeu)
    avec_reseaux = len(option_reseaux) > 1  # au moins un sous-jeu avec reseaux

    # les etapes, dans l'ordre : (nom, commande, l'option --sous-jeux est-elle transmise ?)
    etapes = []
    etapes.append(("Modeles a arbres", [sys.executable, os.path.join("scripts", "generer_predictions_arbres.py")], True))
    if not arguments.seulement_arbres and avec_reseaux:
        # publier_reseaux_04h.py recoit seulement les sous-jeux qui ont des reseaux (option deja dans la commande)
        etapes.append(("Reseaux LSTM et CNN 1D du 04h",
                       [sys.executable, os.path.join("scripts", "publier_reseaux_04h.py")] + option_reseaux, False))
    etapes.append(("Artefacts de l'inference en direct", [sys.executable, "-m", "src.inference", "--forcer"], True))

    code_sortie = 0
    for nom, commande, avec_option in etapes:
        commande_complete = commande
        if avec_option:
            commande_complete = commande + option_sous_jeux
        if not lancer_etape(nom, commande_complete):
            code_sortie = 1
    print("\nLancer l'application : streamlit run app/tableau_de_bord.py")
    return code_sortie


if __name__ == "__main__":
    sys.exit(main())
