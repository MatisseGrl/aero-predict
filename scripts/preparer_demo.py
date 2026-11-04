"""
Prepare tout ce que le tableau de bord de demo (app/tableau_de_bord.py) lit, en une commande.

Utilisation, depuis la racine du depot :

    python scripts/preparer_demo.py                              # tout : arbres, deep learning, artefacts
    python scripts/preparer_demo.py --seulement-arbres           # sans l'entrainement deep learning (long)
    python scripts/preparer_demo.py --sous-jeux FD001 FD003      # seulement certains sous-jeux

Etapes, chacune dans un SOUS-PROCESSUS separe, avec le meme interpreteur Python que ce script :
1. scripts/generer_predictions_arbres.py  -> predictions XGBoost Phase 3 et retenu (sorties/predictions/) ;
2. scripts/entrainer_deep_learning.py     -> predictions LSTM et CNN 1D (sauf --seulement-arbres) ;
3. python -m src.inference --forcer       -> artefacts des deux XGBoost de l'inference en direct, par sous-jeu
                                             (models/<SOUS_JEU>__<modele>.joblib, page "Tester un moteur"),
                                             reentraines pour correspondre aux predictions qui viennent d'etre
                                             ecrites, puis verification : inference en direct = fichiers precalcules ;
4. python -m src.demo entrainer           -> artefact du modele retenu (models/modele_retenu.joblib),
                                             utilise par la demo en ligne de commande (python -m src.demo) ;
5. etat final : fichiers de predictions disponibles et controle de coherence de chacun.

Pourquoi des sous-processus ? Les scripts 1 et 2 sont ecrits et maintenus a part ; les lancer comme
en ligne de commande garantit le meme comportement. Et charger PyTorch (deep learning) et
XGBoost / LightGBM dans le meme processus peut le bloquer (conflit entre leurs bibliotheques de calcul
parallele OpenMP) : chaque script a donc son propre processus. Ce script-ci n'importe jamais torch
(src/inference.py, qui le charge, est lui aussi lance dans un sous-processus).

Si un script n'existe pas encore, l'etape est sautee avec un message clair. Le code de sortie vaut 1
si une etape a echoue, 0 sinon.
"""

import argparse
import os
import subprocess
import sys
import time

# on ajoute la racine du depot (dossier parent de scripts/) au chemin d'import, pour "import src..."
DOSSIER_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
RACINE_DEPOT = os.path.dirname(DOSSIER_SCRIPTS)
if RACINE_DEPOT not in sys.path:
    sys.path.insert(0, RACINE_DEPOT)

from src import donnees_demo as dd  # noqa: E402
from src.contrat_predictions import SOUS_JEUX, lire_predictions  # noqa: E402

# les deux scripts de production des predictions (ecrits par ailleurs)
SCRIPT_ARBRES = os.path.join(RACINE_DEPOT, "scripts", "generer_predictions_arbres.py")
SCRIPT_DEEP_LEARNING = os.path.join(RACINE_DEPOT, "scripts", "entrainer_deep_learning.py")

LARGEUR = 86  # largeur des bandeaux affiches


def afficher_bandeau(texte):
    """
    Affiche un titre d'etape encadre de lignes de "=".

    Parameters
    ----------
    texte : str
        Le titre.

    Returns
    -------
    None
    """
    print()
    print("=" * LARGEUR)
    print("  " + texte)
    print("=" * LARGEUR, flush=True)


def accepte_option(script, option):
    """
    Dit si un script accepte une option, en lisant son aide (`python script --help`).

    Parameters
    ----------
    script : str
        Le chemin du script.
    option : str
        L'option cherchee (ex. "--sous-jeux").

    Returns
    -------
    bool
        Vrai si l'option apparait dans l'aide du script.
    """
    aide = subprocess.run([sys.executable, script, "--help"], cwd=RACINE_DEPOT, capture_output=True, text=True)
    return option in (aide.stdout + aide.stderr)


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
    dict
        "etape", "statut" ("reussie" ou "ECHEC (code N)"), "duree_s".
    """
    afficher_bandeau(f"{nom} : {' '.join(commande[1:])}")
    debut = time.perf_counter()
    execution = subprocess.run(commande, cwd=RACINE_DEPOT)  # meme environnement, sortie en direct
    duree = time.perf_counter() - debut
    resultat = {}
    resultat["etape"] = nom
    resultat["duree_s"] = duree
    if execution.returncode == 0:
        resultat["statut"] = "reussie"
    else:
        resultat["statut"] = f"ECHEC (code {execution.returncode})"
    print(f"  -> {nom} : {resultat['statut']} en {duree:.1f} s", flush=True)
    return resultat


def etape_script(nom, script, sous_jeux):
    """
    Lance un script de production de predictions s'il existe (avec --sous-jeux s'il l'accepte).

    Parameters
    ----------
    nom : str
        Le nom de l'etape.
    script : str
        Le chemin du script.
    sous_jeux : list de str ou None
        Les sous-jeux demandes (None : le defaut du script).

    Returns
    -------
    dict
        Le resultat de `lancer_etape`, ou une etape "sautee" si le script n'existe pas.
    """
    if not os.path.isfile(script):
        afficher_bandeau(nom)
        print(f"  {os.path.relpath(script, RACINE_DEPOT)} n'existe pas encore : etape sautee.")
        print("  (ce script est ecrit par ailleurs ; l'application affiche les fichiers deja presents)")
        resultat = {}
        resultat["etape"] = nom
        resultat["statut"] = "sautee (script absent)"
        resultat["duree_s"] = 0.0
        return resultat

    commande = [sys.executable, script]
    if sous_jeux is not None:
        if accepte_option(script, "--sous-jeux"):
            commande.append("--sous-jeux")
            for sous_jeu in sous_jeux:
                commande.append(sous_jeu)
        else:
            print(f"  [attention] {os.path.basename(script)} n'accepte pas --sous-jeux : lance sur ses sous-jeux par defaut.")
    return lancer_etape(nom, commande)


def afficher_etat_final():
    """
    Affiche les fichiers de predictions disponibles et le controle de coherence de chacun.

    Returns
    -------
    bool
        Vrai si tous les fichiers sont coherents (metriques recalculees = metadonnees).
    """
    afficher_bandeau("Etat final : fichiers de predictions lus par l'application")
    dossier = dd.dossier_predictions()
    liste = dd.inventaire_predictions(dossier)
    print(f"  Dossier : {dossier}")
    if len(liste) == 0:
        print("  Aucun fichier de predictions disponible.")
        return True
    tout_coherent = True
    for position in range(len(liste)):
        sous_jeu = liste["sous_jeu"].iloc[position]
        modele = liste["modele"].iloc[position]
        df, meta = lire_predictions(sous_jeu, modele, dossier)
        controle = dd.controler_coherence(df, meta)
        if controle["coherent"]:
            etat = "coherent"
        else:
            etat = "INCOHERENT"
            tout_coherent = False
        metriques = meta["metriques_test"]
        print(
            f"  {sous_jeu}  {meta['nom_affiche']:<26} {df['unit_number'].nunique():>4} moteurs  "
            f"RMSE test {metriques['rmse']:6.2f}  NASA {metriques['score_nasa']:9.1f}  [{etat}]"
        )
    return tout_coherent


def construire_parseur():
    """
    Construit le parseur des options de la ligne de commande.

    Returns
    -------
    argparse.ArgumentParser
        Le parseur.
    """
    parseur = argparse.ArgumentParser(
        prog="python scripts/preparer_demo.py",
        description="Prepare les fichiers lus par l'application de demo (predictions, artefacts).",
    )
    parseur.add_argument(
        "--seulement-arbres", action="store_true",
        help="ne lance pas l'entrainement deep learning (LSTM, CNN 1D), qui est long",
    )
    parseur.add_argument(
        "--sous-jeux", nargs="+", choices=SOUS_JEUX, default=None, metavar="FD00X",
        help="sous-jeux a traiter (defaut : ceux de chaque script) ; ex. --sous-jeux FD001 FD003",
    )
    return parseur


def main(argv=None):
    """
    Enchaine les etapes de preparation et affiche un resume.

    Parameters
    ----------
    argv : list de str, optionnel
        Les arguments (sans le nom du programme) ; par defaut ceux de la ligne de commande.

    Returns
    -------
    int
        0 si toutes les etapes lancees ont reussi et si les fichiers sont coherents, 1 sinon.
    """
    arguments = construire_parseur().parse_args(argv)
    debut = time.perf_counter()
    resultats = []

    # 1. et 2. : predictions des modeles (scripts ecrits par ailleurs)
    resultats.append(etape_script("Predictions des modeles a arbres", SCRIPT_ARBRES, arguments.sous_jeux))
    if arguments.seulement_arbres:
        print("\n  --seulement-arbres : entrainement deep learning non lance.")
    else:
        resultats.append(etape_script("Predictions deep learning (LSTM, CNN 1D)", SCRIPT_DEEP_LEARNING, arguments.sous_jeux))

    # 3. artefacts XGBoost de l'inference en direct (un par sous-jeu et par modele), reentraines pour
    #    correspondre aux predictions qui viennent d'etre ecrites, puis comparaison directe / precalcule
    commande_inference = [sys.executable, "-m", "src.inference", "--forcer"]
    if arguments.sous_jeux is not None:
        commande_inference.append("--sous-jeux")
        for sous_jeu in arguments.sous_jeux:
            commande_inference.append(sous_jeu)
    resultats.append(lancer_etape("Artefacts de l'inference en direct", commande_inference))

    # 4. artefact du modele retenu (demo en ligne de commande)
    resultats.append(lancer_etape("Artefact du modele retenu", [sys.executable, "-m", "src.demo", "entrainer"]))

    # 5. etat final
    tout_coherent = afficher_etat_final()

    afficher_bandeau("Resume")
    code_sortie = 0
    for resultat in resultats:
        print(f"  {resultat['etape']:<42} {resultat['statut']:<28} {resultat['duree_s']:7.1f} s")
        if resultat["statut"].startswith("ECHEC"):
            code_sortie = 1
    if not tout_coherent:
        print("  Au moins un fichier de predictions est incoherent avec ses metadonnees.")
        code_sortie = 1
    print(f"  Duree totale : {time.perf_counter() - debut:.1f} s")
    print("  Lancer l'application : streamlit run app/tableau_de_bord.py")
    return code_sortie


if __name__ == "__main__":
    sys.exit(main())
