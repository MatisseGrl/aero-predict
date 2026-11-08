"""
Genere les 4 notebooks de resultats (notebooks/resultats/FD001.ipynb a FD004.ipynb) depuis un seul modele.

Pourquoi ce script ? (plan de reorganisation, docs/plan_refonte_notebooks.md, principe 2 et section 6.1)
Les 4 notebooks de resultats ont exactement le meme plan, les memes cellules et les memes graphes ; seul le
sous-jeu change. On n'ecrit donc a la main qu'un seul notebook, notebooks/resultats/_modele.ipynb, et ce script
l'execute une fois par sous-jeu. Les FD00X.ipynb ne sont jamais modifies a la main : on modifie le modele, puis on
relance ce script.

Comment le sous-jeu est transmis (choix de la section 6.1 du plan) : nbconvert (son executeur, ExecutePreprocessor),
sans papermill. Raison du choix : nbconvert est deja installe avec Jupyter (aucune dependance de plus dans
requirements.txt), alors que papermill en ajouterait une. Comme papermill, le script RECRIT la cellule marquee
« parameters » du modele avant de l'executer : dans FD002.ipynb, cette cellule contient donc `SOUS_JEU = "FD002"`.
Ainsi, un notebook genere qu'on rouvre et qu'on relance a la main (« Restart & Run All ») recalcule bien son propre
sous-jeu, et pas celui ecrit par defaut dans le modele (correction apres la revue de la reorganisation : la
premiere version ne passait le sous-jeu que par une variable d'environnement, et FD002.ipynb relance a la main
recalculait FD001). Les variables d'environnement SOUS_JEU, MOTEUR et ENTRAINER sont aussi fixees, aux memes
valeurs, pour le noyau.

Ce que fait le script, pour chaque sous-jeu demande :
1. lit _modele.ipynb, recrit sa cellule de parametres (SOUS_JEU, MOTEUR, ENTRAINER) ;
2. execute ce notebook (dossier de travail : notebooks/resultats/, comme a la main) ;
3. ecrit le resultat, sorties comprises, dans notebooks/resultats/<SOUS_JEU>.ipynb (ou un autre dossier) ;
4. relit le notebook ecrit et verifie : aucune cellule en erreur, toutes les cellules de code executees, autant de
   figures que le modele contient d'appels `plt.show()`, le sous-jeu affiche est celui demande, et la cellule de
   parametres contient bien ce sous-jeu.

Calcul sur une machine partagee : si la variable LOKY_MAX_CPU_COUNT n'est pas deja fixee, le script la fixe a 4
(la Random Forest des reglages finaux utilise n_jobs=-1, c'est-a-dire tous les coeurs ; cela ne change pas ses
predictions, seulement le nombre de coeurs utilises). Elle ne sert que si ENTRAINER est vrai ou si un modele manque.

Lancement, depuis la racine du depot :
    python scripts/generer_notebooks_resultats.py                      # les 4 sous-jeux, modeles charges
    python scripts/generer_notebooks_resultats.py --sous-jeux FD002    # un seul sous-jeu
    python scripts/generer_notebooks_resultats.py --entrainer --sous-jeux FD001 --dossier-sortie /tmp/verif
        # arbres reentraines et verifies, notebook ecrit ailleurs (pour ne pas remplacer celui du depot)

Prerequis (fichiers non versionnes, sorties/ et models/), dans cet ordre, depuis la racine du depot :
    python scripts/preparer_demo.py                                          # arbres, reseaux du 04h, inference
    LOKY_MAX_CPU_COUNT=4 python scripts/valider_arbres_finaux.py              # validation croisee des arbres
    LOKY_MAX_CPU_COUNT=4 python scripts/valider_arbres_finaux.py --complements  # erreur train, decoupages neufs
"""

import argparse
import os
import sys
import time

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

# racine du depot : dossier parent de scripts/
RACINE_DEPOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOSSIER_RESULTATS = os.path.join(RACINE_DEPOT, "notebooks", "resultats")
CHEMIN_MODELE = os.path.join(DOSSIER_RESULTATS, "_modele.ipynb")

SOUS_JEUX = ["FD001", "FD002", "FD003", "FD004"]
MOTEUR_DEFAUT = 42  # le moteur de la section 7, le meme que la valeur par defaut du modele
COEURS_MAX_DEFAUT = "4"  # LOKY_MAX_CPU_COUNT fixe si absent (machine partagee)


def injecter_parametres(notebook, sous_jeu, moteur, entrainer):
    """
    Recrit la cellule « parameters » du modele avec les valeurs demandees (comme le fait papermill).

    Seules les trois lignes d'affectation (SOUS_JEU = ..., MOTEUR = ..., ENTRAINER = ...) sont remplacees ; le reste
    de la cellule (commentaires, lecture des variables d'environnement) est garde tel quel.

    Parameters
    ----------
    notebook : nbformat.NotebookNode
        Le modele, lu par nbformat (modifie sur place).
    sous_jeu : str
        "FD001" a "FD004".
    moteur : int
        Le moteur test de la section 7.
    entrainer : bool
        Vrai : les arbres sont reentraines et verifies.

    Returns
    -------
    int
        La position de la cellule de parametres dans le notebook.

    Raises
    ------
    ValueError
        S'il n'y a pas exactement une cellule marquee « parameters », ou si une des trois lignes manque.
    """
    # 1. la cellule de parametres : celle qui porte le tag « parameters » (une seule)
    positions = []
    for position in range(len(notebook.cells)):
        if "parameters" in notebook.cells[position].get("metadata", {}).get("tags", []):
            positions.append(position)
    if len(positions) != 1:
        raise ValueError(f"_modele.ipynb doit avoir exactement une cellule « parameters » ({len(positions)} trouvee(s))")
    cellule = notebook.cells[positions[0]]

    # 2. les nouvelles lignes d'affectation, avec leur commentaire
    nouvelles = {}
    nouvelles["SOUS_JEU"] = f'SOUS_JEU = "{sous_jeu}"  # sous-jeu évalué (écrit par scripts/generer_notebooks_resultats.py)'
    nouvelles["MOTEUR"] = f"MOTEUR = {int(moteur)}  # moteur test recalculé en section 7"
    nouvelles["ENTRAINER"] = (f"ENTRAINER = {bool(entrainer)}  # False : modèles chargés depuis models/ ; "
                              "True : arbres réentraînés puis vérifiés (plan 6.2)")

    # 3. remplacement ligne par ligne : seules les lignes qui commencent par « NOM = » changent
    lignes = cellule.source.split("\n")
    trouvees = []
    for numero in range(len(lignes)):
        for nom in nouvelles:
            if lignes[numero].startswith(nom + " = "):
                lignes[numero] = nouvelles[nom]
                trouvees.append(nom)
    for nom in nouvelles:
        if trouvees.count(nom) != 1:
            raise ValueError(f"cellule de parametres : la ligne « {nom} = ... » doit apparaitre une fois")
    cellule.source = "\n".join(lignes)
    return positions[0]


def nb_figures_attendues(notebook):
    """
    Compte les figures que le modele doit produire : un appel `plt.show()` par figure.

    Parameters
    ----------
    notebook : nbformat.NotebookNode
        Le modele.

    Returns
    -------
    int
        Le nombre d'appels `plt.show()` dans ses cellules de code.
    """
    total = 0
    for cellule in notebook.cells:
        if cellule.cell_type == "code":
            total = total + cellule.source.count("plt.show()")
    return total


def environnement_execution(sous_jeu, moteur, entrainer):
    """
    Prepare les variables d'environnement transmises au noyau qui execute le modele.

    Parameters
    ----------
    sous_jeu : str
        "FD001" a "FD004".
    moteur : int
        Le moteur test de la section 7.
    entrainer : bool
        Vrai : les arbres sont reentraines et verifies (section 2.1 du notebook).

    Returns
    -------
    dict
        Une copie de os.environ avec SOUS_JEU, MOTEUR, ENTRAINER et LOKY_MAX_CPU_COUNT.
    """
    environnement = dict(os.environ)  # copie : on ne modifie pas l'environnement de ce script
    environnement["SOUS_JEU"] = sous_jeu
    environnement["MOTEUR"] = str(moteur)
    if entrainer:
        environnement["ENTRAINER"] = "1"
    else:
        environnement["ENTRAINER"] = "0"
    # machine partagee : la Random Forest (n_jobs=-1) ne prendra pas plus de 4 coeurs, sauf choix contraire
    if "LOKY_MAX_CPU_COUNT" not in environnement:
        environnement["LOKY_MAX_CPU_COUNT"] = COEURS_MAX_DEFAUT
    return environnement


def executer_modele(sous_jeu, dossier_sortie, moteur, entrainer):
    """
    Execute _modele.ipynb pour un sous-jeu et ecrit <dossier_sortie>/<sous_jeu>.ipynb, sorties comprises.

    La cellule de parametres est d'abord recrite avec le sous-jeu (`injecter_parametres`). Le noyau s'execute dans
    le dossier du modele (notebooks/resultats/), comme si on ouvrait le notebook a la main : ses chemins relatifs
    (../../src) restent valables, quel que soit le dossier de sortie.

    Parameters
    ----------
    sous_jeu : str
        "FD001" a "FD004".
    dossier_sortie : str
        Dossier ou ecrire le notebook genere.
    moteur : int
        Le moteur test de la section 7.
    entrainer : bool
        Vrai : les arbres sont reentraines et verifies.

    Returns
    -------
    (str, float, int)
        Le chemin du notebook ecrit, la duree d'execution en secondes et le nombre de figures attendues.

    Raises
    ------
    nbconvert.preprocessors.CellExecutionError
        Si une cellule leve une erreur (par exemple un controle non verifie) ; le message donne la cellule.
    """
    # 1. le modele, avec sa cellule de parametres recrite pour ce sous-jeu
    notebook = nbformat.read(CHEMIN_MODELE, as_version=4)
    injecter_parametres(notebook, sous_jeu, moteur, entrainer)
    attendues = nb_figures_attendues(notebook)

    # 2. les memes valeurs dans l'environnement du noyau (le noyau herite de l'environnement de ce processus)
    environnement_avant = dict(os.environ)
    os.environ.clear()
    os.environ.update(environnement_execution(sous_jeu, moteur, entrainer))

    # 3. execution, dossier de travail = notebooks/resultats/ (chemins relatifs du modele)
    executeur = ExecutePreprocessor(timeout=None, kernel_name="python3")  # pas de limite de duree par cellule
    debut = time.perf_counter()
    try:
        executeur.preprocess(notebook, {"metadata": {"path": DOSSIER_RESULTATS}})
    finally:
        os.environ.clear()
        os.environ.update(environnement_avant)  # on remet l'environnement de ce script tel qu'il etait
    duree = time.perf_counter() - debut

    # 4. trace de la generation dans les metadonnees du notebook, puis ecriture
    notebook.metadata["genere_par"] = {"script": "scripts/generer_notebooks_resultats.py",
                                       "modele": "notebooks/resultats/_modele.ipynb",
                                       "SOUS_JEU": sous_jeu, "MOTEUR": int(moteur), "ENTRAINER": bool(entrainer)}
    chemin = os.path.join(dossier_sortie, sous_jeu + ".ipynb")
    nbformat.write(notebook, chemin)
    return chemin, duree, attendues


def verifier_notebook_genere(chemin, sous_jeu, figures_attendues):
    """
    Relit un notebook genere et verifie qu'il est complet.

    Controles : aucune sortie d'erreur ; toutes les cellules de code ont ete executees ; exactement le nombre de
    figures (images PNG) attendu ; le titre affiche par la cellule des imports est celui du sous-jeu demande ; la
    cellule de parametres contient ce sous-jeu (un « Restart & Run All » recalcule le bon sous-jeu).

    Parameters
    ----------
    chemin : str
        Le notebook genere.
    sous_jeu : str
        Le sous-jeu demande.
    figures_attendues : int
        Le nombre de figures du modele (`nb_figures_attendues`).

    Returns
    -------
    dict
        "nb_cellules_code", "nb_figures", "nb_erreurs", "nb_non_executees", "titre_trouve" et
        "parametre_trouve" (bool).

    Raises
    ------
    AssertionError
        Si un controle echoue.
    """
    notebook = nbformat.read(chemin, as_version=4)
    nb_cellules_code = 0
    nb_figures = 0
    nb_erreurs = 0
    nb_non_executees = 0
    titre_trouve = False
    titre_attendu = f"# Sous-jeu {sous_jeu}"
    parametre_attendu = f'SOUS_JEU = "{sous_jeu}"'
    parametre_trouve = False
    for cellule in notebook.cells:
        if cellule.cell_type != "code":
            continue  # seules les cellules de code ont des sorties
        # la cellule de parametres doit contenir le sous-jeu de ce notebook
        if "parameters" in cellule.get("metadata", {}).get("tags", []) and parametre_attendu in cellule.source:
            parametre_trouve = True
        nb_cellules_code = nb_cellules_code + 1
        if cellule.get("execution_count") is None:
            nb_non_executees = nb_non_executees + 1
        for sortie in cellule.get("outputs", []):
            if sortie.get("output_type") == "error":
                nb_erreurs = nb_erreurs + 1
            donnees = sortie.get("data", {})
            if "image/png" in donnees:
                nb_figures = nb_figures + 1
            if "text/markdown" in donnees and donnees["text/markdown"].startswith(titre_attendu):
                titre_trouve = True

    bilan = {}
    bilan["nb_cellules_code"] = nb_cellules_code
    bilan["nb_figures"] = nb_figures
    bilan["nb_erreurs"] = nb_erreurs
    bilan["nb_non_executees"] = nb_non_executees
    bilan["titre_trouve"] = titre_trouve
    bilan["parametre_trouve"] = parametre_trouve
    # chaque controle, avec un message qui dit quoi regarder
    assert nb_erreurs == 0, f"{chemin} : {nb_erreurs} sortie(s) d'erreur"
    assert nb_non_executees == 0, f"{chemin} : {nb_non_executees} cellule(s) de code non executee(s)"
    assert nb_figures == figures_attendues, f"{chemin} : {nb_figures} figure(s), {figures_attendues} attendue(s)"
    assert parametre_trouve, f"{chemin} : la cellule de parametres ne contient pas « {parametre_attendu} »"
    assert titre_trouve, f"{chemin} : le titre « {titre_attendu} » est absent (mauvais sous-jeu ?)"
    return bilan


def lire_arguments(argv=None):
    """
    Lit les arguments de la ligne de commande.

    Parameters
    ----------
    argv : list de str ou None, optionnel
        Les arguments (None : ceux de la ligne de commande).

    Returns
    -------
    argparse.Namespace
        "sous_jeux" (list de str), "dossier_sortie" (str), "moteur" (int), "entrainer" (bool).
    """
    analyseur = argparse.ArgumentParser(description="Genere notebooks/resultats/FD00X.ipynb depuis _modele.ipynb.")
    analyseur.add_argument("--sous-jeux", nargs="+", default=SOUS_JEUX, choices=SOUS_JEUX,
                           help="sous-jeux a generer (defaut : les 4)")
    analyseur.add_argument("--dossier-sortie", default=DOSSIER_RESULTATS,
                           help="dossier des notebooks generes (defaut : notebooks/resultats/)")
    analyseur.add_argument("--moteur", type=int, default=MOTEUR_DEFAUT,
                           help="moteur test de la section 7 (defaut : 42)")
    analyseur.add_argument("--entrainer", action="store_true",
                           help="reentrainer les arbres et verifier qu'ils redonnent les predictions de la demo")
    return analyseur.parse_args(argv)


def main(argv=None):
    """
    Genere les notebooks demandes, puis affiche un bilan par sous-jeu.

    Parameters
    ----------
    argv : list de str ou None, optionnel
        Les arguments (None : ceux de la ligne de commande).

    Returns
    -------
    int
        0 si tous les notebooks ont ete generes et verifies.
    """
    # arguments, dossier de sortie, et le modele doit exister
    arguments = lire_arguments(argv)
    os.makedirs(arguments.dossier_sortie, exist_ok=True)
    if not os.path.exists(CHEMIN_MODELE):
        raise FileNotFoundError(f"Modele introuvable : {CHEMIN_MODELE}")

    debut_total = time.perf_counter()
    for sous_jeu in arguments.sous_jeux:
        print(f"{sous_jeu} : execution de _modele.ipynb (MOTEUR={arguments.moteur}, ENTRAINER={arguments.entrainer})...",
              flush=True)
        # execution du modele pour ce sous-jeu, puis relecture et controles du notebook ecrit
        chemin, duree, attendues = executer_modele(sous_jeu, arguments.dossier_sortie, arguments.moteur,
                                                   arguments.entrainer)
        bilan = verifier_notebook_genere(chemin, sous_jeu, attendues)
        print(f"{sous_jeu} : {os.path.relpath(chemin, RACINE_DEPOT)} ecrit en {duree:.0f} s ; "
              f"{bilan['nb_cellules_code']} cellules de code executees, {bilan['nb_figures']} figures "
              f"({attendues} attendues), {bilan['nb_erreurs']} erreur.", flush=True)
    print(f"Termine en {time.perf_counter() - debut_total:.0f} s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
