"""
Entraine et evalue les reseaux sequentiels (LSTM, CNN 1D) sur FD001 a FD004 : le calcul lourd du notebook 03b.

Le notebook notebooks/03b_deep_learning_lstm_cnn.ipynb relit les fichiers ecrits ici (docs/resultats/03b_*)
au lieu de tout recalculer : il s'execute ainsi en quelques minutes.

Ce que fait le script, dans l'ordre :

FD001 (protocole complet, memes plis que les arbres du notebook 05) :
1. "appareils"     : mesure de vitesse CPU / MPS et de reproductibilite (docs/resultats/03b_cpu_vs_mps.csv) ;
2. "choix_entrees" : faut-il donner time_cycles aux reseaux ? Decide sur UNE separation de validation par
                     moteurs du train (jamais sur le test), regle fixee avant de regarder les resultats
                     (docs/resultats/03b_choix_entrees_fd001.csv) ;
3. "cv"            : validation croisee 5 plis (GroupKFold par moteur, shuffle, graine 42 : les plis des
                     arbres), un reseau neuf par pli, arret precoce sur 20 % des moteurs d'ENTRAINEMENT du pli ;
4. "final"         : modele final sur les 100 moteurs, nombre d'epoques = mediane des meilleures epoques
                     des 5 plis (jamais choisi sur le test), 3 graines ; test officiel evalue une fois ;
                     fichiers de predictions (src/contrat_predictions.py) et poids (models/FD001__<modele>.pt) ;
5. "asymetrique"   : validation croisee du meilleur des deux reseaux avec la perte asymetrique
                     alpha = 2,26 (ALPHA_ASYMETRIQUE, valeur fixee avant l'experience, non optimisee
                     pour les reseaux).

FD002, FD003, FD004 (src/sous_jeux.py) : separation 80 / 20 des moteurs du train (graine fixe) pour
l'arret precoce et le nombre d'epoques, puis modele final sur tout le train, prediction a chaque cycle de
chaque moteur test, fichiers de predictions et poids. Option --cv-sous-jeux : ajoute une validation
croisee 5 plis (plus couteuse) sur ces sous-jeux.

Exemples, depuis la racine du depot :

    python scripts/entrainer_deep_learning.py                                   # tout (23 min mesurees, 6 processus)
    python scripts/entrainer_deep_learning.py --sous-jeux FD001                 # FD001 seulement
    python scripts/entrainer_deep_learning.py --sous-jeux FD002 FD003 FD004 --modeles lstm
    python scripts/entrainer_deep_learning.py --sous-jeux FD002 FD003 FD004 --seulement-cv-sous-jeux
                                                    # validation croisee 5 plis complementaire sur FD002-FD004
    python scripts/entrainer_deep_learning.py --seulement-resume                # reecrit le fichier des derniers cycles
    python scripts/entrainer_deep_learning.py --sous-jeux FD001 --max-epoques 1 --dossier-resultats /tmp/essai \\
        --dossier-predictions /tmp/essai --dossier-modeles /tmp/essai           # essai rapide, sans rien ecraser

Reproductibilite : CPU, 1 coeur par apprentissage, graines fixees ; un meme apprentissage donne les
memes poids au bit pres qu'il tourne seul ou en parallele avec d'autres (verifie par
tests/test_deep_learning.py et par l'etape "appareils"). Seules les durees changent d'une execution a
l'autre (elles dependent de la charge de la machine).
"""

import argparse
import datetime
import json
import os
import sys
import time

# on ajoute la racine du depot (dossier parent de scripts/) au chemin d'import, pour "import src..."
DOSSIER_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
RACINE_DEPOT = os.path.dirname(DOSSIER_SCRIPTS)
sys.path.insert(0, RACINE_DEPOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# les modules du projet AVANT torch : xgboost / lightgbm (charges par src.config_modele) doivent l'etre avant
# PyTorch, sinon conflit OpenMP (voir src/deep_learning.py)
from src.config_modele import convertir_en_types_python  # noqa: E402
from src.contrat_predictions import (  # noqa: E402
    MODELES_CONNUS,
    calculer_rul_vraie,
    ecrire_predictions,
    metriques_dernier_cycle,
)
from src.deep_learning import (  # noqa: E402
    DOSSIER_MODELES,
    assembler_validation_croisee,
    LONGUEUR_FENETRE_DEFAUT,
    REGLAGES_ENTRAINEMENT_DEFAUT,
    TYPES_MODELES,
    ajouter_time_cycles_echelle,
    calculer_metriques,
    chemin_modele,
    completer_reglages,
    construire_fenetres,
    creer_modele,
    entrainer_modele,
    evaluer_en_validation_croisee_sequentielle,
    executer_taches,
    hyperparametres_par_defaut,
    lister_colonnes_entree,
    predire,
    preparer_taches_validation_croisee,
    sauvegarder_modele,
    separer_moteurs,
)
from src.evaluation import calculer_plis, erreurs_par_moteur, nb_surestimes, score_nasa  # noqa: E402

import torch  # noqa: E402  (apres les modules du projet)

# dossier des resultats versionnes (lus par le notebook 03b)
DOSSIER_RESULTATS = os.path.join(RACINE_DEPOT, "docs", "resultats")

# coefficient de la perte quadratique asymetrique de l'etape "asymetrique" (erreur^2 x alpha si le reseau
# surestime, x 1 sinon). Valeur fixee AVANT l'experience : c'est celle d'un premier essai de reglage de cette
# perte sur XGBoost (FD001, validation croisee), recopiee ici en entier pour reproduire exactement les
# resultats deja ecrits dans docs/resultats/03b_cv_fd001.csv. Elle n'a pas ete optimisee pour les reseaux.
ALPHA_ASYMETRIQUE = 2.2563503313740805

SOUS_JEUX = ["FD001", "FD002", "FD003", "FD004"]
ETAPES_FD001 = ["appareils", "choix_entrees", "cv", "final", "asymetrique"]
SEUIL_SURESTIMATION = 20  # cycles (choix de la Phase 3)
SEUIL_ALERTE = 30  # cycles, hypothese de travail (pas de valeur officielle)
PLAFOND_RUL = 125
N_PLIS = 5
GRAINE_PLIS = 42  # memes plis que les arbres (Phase 3, notebook 05, scripts/generer_predictions_arbres.py)
FRACTION_VALIDATION = 0.2  # 20 % des moteurs pour l'arret precoce (et pour la decision des entrees)
NB_GRAINES_ROBUSTESSE = 3  # le modele final FD001 est entraine avec 3 graines
NB_GRAINES_CHOIX_ENTREES = 3  # chaque variante du choix des entrees est entrainee avec 3 graines

# Regle de decision des entrees, ECRITE AVANT de lancer l'experience :
TEXTE_REGLE_ENTREES = (
    "Separation unique des 100 moteurs du train FD001 (graine 42) : 20 moteurs de decision, 80 moteurs "
    "d'entrainement eux-memes separes en 64 d'apprentissage et 16 d'arret precoce. Chaque reseau (LSTM, CNN 1D) "
    "est entraine avec et sans time_cycles, 3 graines chacun ; on mesure le RMSE sur les 20 moteurs de decision "
    "(tous leurs cycles). time_cycles est garde si la moyenne, sur les deux reseaux et les 3 graines, de ce RMSE "
    "est plus basse avec time_cycles que sans ; sinon on garde les capteurs seuls (variante la plus simple). Le "
    "jeu de test n'intervient pas. Decision prise une fois sur FD001 et appliquee a FD002-FD004."
)


# ---------------------------------------------------------------------------------------------------
# Options et petits utilitaires
# ---------------------------------------------------------------------------------------------------

def lire_arguments():
    """
    Lit les options de la ligne de commande.

    Returns
    -------
    argparse.Namespace
        Les options (voir l'aide : python scripts/entrainer_deep_learning.py --help).
    """
    analyseur = argparse.ArgumentParser(description="Reseaux sequentiels (LSTM, CNN 1D) sur FD001 a FD004.")
    analyseur.add_argument("--sous-jeux", nargs="+", choices=SOUS_JEUX, default=SOUS_JEUX,
                           help="sous-jeux a traiter (tous par defaut)")
    analyseur.add_argument("--modeles", nargs="+", choices=TYPES_MODELES, default=TYPES_MODELES,
                           help="reseaux a entrainer (les deux par defaut)")
    analyseur.add_argument("--graine", type=int, default=42, help="graine principale (42 par defaut)")
    analyseur.add_argument("--etapes", nargs="+", choices=ETAPES_FD001, default=ETAPES_FD001,
                           help="etapes FD001 a executer (toutes par defaut)")
    analyseur.add_argument("--nb-processus", type=int, default=6,
                           help="apprentissages en parallele, 1 coeur chacun (6 par defaut)")
    analyseur.add_argument("--max-epoques", type=int, default=REGLAGES_ENTRAINEMENT_DEFAUT["max_epoques"],
                           help="nombre maximal d'epoques avec arret precoce")
    analyseur.add_argument("--patience", type=int, default=REGLAGES_ENTRAINEMENT_DEFAUT["patience"],
                           help="patience de l'arret precoce")
    analyseur.add_argument("--entrees", choices=["auto", "capteurs", "capteurs_time_cycles"], default="auto",
                           help="entrees des reseaux ; auto = decision de l'etape choix_entrees")
    analyseur.add_argument("--cv-sous-jeux", action="store_true",
                           help="ajoute une validation croisee 5 plis sur FD002-FD004 (couteux)")
    analyseur.add_argument("--dossier-resultats", default=DOSSIER_RESULTATS, help="ou ecrire les CSV / JSON")
    analyseur.add_argument("--dossier-predictions", default=None,
                           help="ou ecrire les fichiers de predictions (defaut : celui du contrat)")
    analyseur.add_argument("--dossier-modeles", default=DOSSIER_MODELES, help="ou ecrire les poids (.pt)")
    analyseur.add_argument("--seulement-cv-sous-jeux", action="store_true",
                           help="ne faire que la validation croisee 5 plis des reseaux sur FD002-FD004")
    analyseur.add_argument("--seulement-integrer-cv", action="store_true",
                           help="reporter les validations croisees FD002-FD004 deja faites (metadonnees, tableau agrege)")
    analyseur.add_argument("--seulement-resume", action="store_true",
                           help="ne rien entrainer : reecrire seulement 03b_predictions_test_dernier_cycle.csv")
    return analyseur.parse_args()


def afficher(texte):
    """
    Affiche une ligne de progression precedee de l'heure (le log du script se lit ainsi facilement).

    Parameters
    ----------
    texte : str
        Le message.
    """
    heure = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{heure}] {texte}", flush=True)


def ecrire_json(objet, chemin):
    """
    Ecrit un objet en JSON lisible (types numpy convertis en types Python).

    Parameters
    ----------
    objet : dict ou list
        L'objet a ecrire.
    chemin : str
        Le fichier.
    """
    with open(chemin, "w", encoding="utf-8") as fichier:
        json.dump(convertir_en_types_python(objet), fichier, ensure_ascii=False, indent=1)
        fichier.write("\n")


def lire_json(chemin, defaut):
    """
    Lit un fichier JSON s'il existe, sinon renvoie la valeur par defaut.

    Parameters
    ----------
    chemin : str
        Le fichier.
    defaut : objet
        Valeur renvoyee si le fichier n'existe pas.

    Returns
    -------
    objet
        Le contenu du fichier, ou `defaut`.
    """
    if not os.path.isfile(chemin):
        return defaut
    with open(chemin, "r", encoding="utf-8") as fichier:
        return json.load(fichier)


def mettre_a_jour_csv(chemin, nouvelles_lignes, colonnes_cles):
    """
    Ajoute des lignes a un CSV, en remplacant les lignes existantes qui ont les memes cles.

    Permet de relancer le script sur un seul sous-jeu ou un seul modele sans effacer les autres resultats.

    Parameters
    ----------
    chemin : str
        Le fichier CSV (cree s'il n'existe pas).
    nouvelles_lignes : list de dict
        Les lignes a ecrire.
    colonnes_cles : list de str
        Les colonnes qui identifient une ligne (ex. ["sous_jeu", "modele"]).

    Returns
    -------
    pandas.DataFrame
        Le tableau complet ecrit.
    """
    nouveau = pd.DataFrame(nouvelles_lignes)
    if os.path.isfile(chemin):
        ancien = pd.read_csv(chemin)
        # on garde les anciennes lignes dont les cles ne sont pas dans les nouvelles lignes
        garder = np.ones(len(ancien), dtype=bool)
        for position in range(len(ancien)):
            for _, ligne_nouvelle in nouveau.iterrows():
                identiques = True
                for cle in colonnes_cles:
                    if str(ancien.iloc[position][cle]) != str(ligne_nouvelle[cle]):
                        identiques = False
                if identiques:
                    garder[position] = False
        tableau = pd.concat([ancien[garder], nouveau], ignore_index=True)
    else:
        tableau = nouveau
    tableau.to_csv(chemin, index=False)
    return tableau


def reglages_communs(arguments, alpha=1.0):
    """
    Construit les reglages d'apprentissage communs a toutes les taches (avec les options du script).

    Parameters
    ----------
    arguments : argparse.Namespace
        Les options du script.
    alpha : float, optionnel
        Poids des surestimations dans la perte (1 = MSE).

    Returns
    -------
    dict
        Les reglages complets.
    """
    reglages = {}
    reglages["max_epoques"] = arguments.max_epoques
    reglages["patience"] = arguments.patience
    reglages["alpha"] = alpha
    return completer_reglages(reglages)


def creer_tache(type_modele, graine, fenetres_entrainement, y_entrainement, reglages, etiquette,
                fenetres_validation=None, y_validation=None, nb_epoques=None, a_predire=None,
                renvoyer_poids=False):
    """
    Construit une tache d'apprentissage pour `src.deep_learning.executer_entrainement`.

    Parameters
    ----------
    type_modele : str
        "lstm" ou "cnn1d".
    graine : int
        Graine de la tache.
    fenetres_entrainement, y_entrainement : numpy.ndarray
        Donnees d'apprentissage.
    reglages : dict
        Reglages d'apprentissage.
    etiquette : str
        Texte de progression.
    fenetres_validation, y_validation : numpy.ndarray ou None, optionnel
        Moteurs d'arret precoce.
    nb_epoques : int ou None, optionnel
        Nombre d'epoques fixe (modele final).
    a_predire : dict ou None, optionnel
        Fenetres a predire apres l'apprentissage.
    renvoyer_poids : bool, optionnel
        Vrai pour recuperer les poids appris.

    Returns
    -------
    dict
        La tache.
    """
    tache = {}
    tache["type_modele"] = type_modele
    tache["graine"] = graine
    tache["fenetres_entrainement"] = fenetres_entrainement
    tache["y_entrainement"] = y_entrainement
    tache["reglages"] = reglages
    tache["etiquette"] = etiquette
    tache["fenetres_validation"] = fenetres_validation
    tache["y_validation"] = y_validation
    tache["nb_epoques"] = nb_epoques
    tache["a_predire"] = {}
    if a_predire is not None:
        tache["a_predire"] = a_predire
    tache["renvoyer_poids"] = renvoyer_poids
    tache["nb_threads"] = 1  # 1 coeur par apprentissage (reproductible, et plusieurs en parallele)
    tache["appareil"] = "cpu"
    return tache


def resumer_cv(nom, type_modele, resultats, nb_moteurs, alpha, colonnes_entree):
    """
    Resume une validation croisee en une ligne de tableau (memes colonnes que docs/resultats/05_comparaison_cv.csv).

    Parameters
    ----------
    nom : str
        Nom de la ligne (ex. "lstm").
    type_modele : str
        "lstm" ou "cnn1d".
    resultats : dict
        Le resultat de `evaluer_en_validation_croisee_sequentielle`.
    nb_moteurs : int
        Nombre de moteurs predits au total (pour le temps de prediction par moteur).
    alpha : float
        Poids des surestimations de la perte.
    colonnes_entree : list de str
        Les entrees du reseau.

    Returns
    -------
    dict
        La ligne du tableau.
    """
    ligne = {}
    ligne["modele"] = nom
    ligne["type_modele"] = type_modele
    ligne["alpha"] = alpha
    ligne["rmse_cv_moyen"] = resultats["rmse_moyen"]
    ligne["rmse_cv_ecart_type"] = resultats["rmse_ecart_type"]
    ligne["mae_cv_moyen"] = resultats["mae_moyen"]
    ligne["mae_cv_ecart_type"] = resultats["mae_ecart_type"]
    ligne["nasa_cv_moyen"] = resultats["nasa_moyen"]
    ligne["nasa_cv_ecart_type"] = resultats["nasa_ecart_type"]
    ligne["surestimes_oof"] = resultats["surestimes_oof"]
    ligne["alertes_manquees_oof"] = resultats["alertes_manquees_oof"]
    ligne["fausses_alertes_oof"] = resultats["fausses_alertes_oof"]
    ligne["biais_oof"] = resultats["biais_oof"]
    ligne["rmse_oof"] = resultats["rmse_oof"]
    ligne["temps_entrainement_5_plis_s"] = resultats["temps_entrainement_total"]
    ligne["prediction_ms_par_moteur"] = 1000.0 * resultats["temps_prediction_total"] / nb_moteurs
    ligne["nb_parametres"] = resultats["nb_parametres"]
    ligne["meilleures_epoques"] = json.dumps(resultats["meilleures_epoques"])
    ligne["rmse_par_pli"] = json.dumps(resultats["rmse_par_pli"])
    ligne["nasa_par_pli"] = json.dumps(resultats["nasa_par_pli"])
    ligne["entrees"] = "+".join(colonnes_entree)
    ligne["nb_entrees"] = len(colonnes_entree)
    ligne["longueur_fenetre"] = LONGUEUR_FENETRE_DEFAUT
    ligne["date"] = datetime.date.today().isoformat()
    return ligne


def historiques_simplifies(historiques):
    """
    Garde, pour chaque historique d'apprentissage, les courbes et les informations utiles a la figure.

    Parameters
    ----------
    historiques : list de dict
        Les historiques de `entrainer_modele`.

    Returns
    -------
    list de dict
        Les memes informations, en types simples.
    """
    simplifies = []
    for historique in historiques:
        element = {}
        for cle in ["perte_entrainement", "perte_validation", "rmse_validation", "meilleure_epoque",
                    "nb_epoques_effectuees", "arret_precoce", "duree_s", "duree_par_epoque_s",
                    "nb_fenetres_entrainement", "nb_fenetres_validation"]:
            element[cle] = historique[cle]
        simplifies.append(element)
    return simplifies


# ---------------------------------------------------------------------------------------------------
# Preparation des donnees
# ---------------------------------------------------------------------------------------------------

def preparer_donnees(sous_jeu, avec_time_cycles):
    """
    Prepare le train et le test d'un sous-jeu et construit les fenetres (une par ligne).

    FD001 : src/features.py (pipeline valide des Phases 1 a 3). FD002 a FD004 : src/sous_jeux.py (meme
    pipeline, normalisation par regime pour FD002 / FD004). Les entrees des reseaux sont les capteurs
    normalises puis lisses du contexte, plus time_cycles / 100 si demande.

    Parameters
    ----------
    sous_jeu : str
        "FD001" a "FD004".
    avec_time_cycles : bool
        Vrai pour ajouter time_cycles mis a l'echelle aux entrees.

    Returns
    -------
    dict
        "train", "test" (DataFrames), "capteurs", "colonnes_entree", "fenetres", "y", "groups",
        "fenetres_test", "y_test_vraie", "moteurs_test", "X" (features des arbres, pour calculer_plis),
        "nb_regimes", "duree_preparation_s".
    """
    debut = time.perf_counter()
    if sous_jeu == "FD001":
        from src.features import preparer_test, preparer_train
        contexte_train = preparer_train(fenetre=5, plafond=PLAFOND_RUL)
        contexte_test = preparer_test(contexte_train)
        train = contexte_train["train"]
        test = contexte_test["test"]
        capteurs = contexte_train["sensor_cols_restants"]
        X = contexte_train["X"]
        y_test_vraie = contexte_test["y_test_vraie"]
        moteurs_test = contexte_test["moteurs_test"]
        nb_regimes = 1
    else:
        from src.sous_jeux import preparer_sous_jeu
        contexte = preparer_sous_jeu(sous_jeu, fenetre=5, plafond=PLAFOND_RUL)
        train = contexte["train"]
        test = contexte["test"]
        capteurs = contexte["capteurs"]
        X = contexte["X"]
        y_test_vraie = contexte["y_test_vraie"]
        moteurs_test = contexte["moteurs_test"]
        nb_regimes = contexte["nb_regimes"]

    train = ajouter_time_cycles_echelle(train)  # colonne ajoutee dans tous les cas, utilisee si demande
    test = ajouter_time_cycles_echelle(test)
    colonnes_entree = lister_colonnes_entree(capteurs, avec_time_cycles)

    fenetres, y, groups, _ = construire_fenetres(train, colonnes_entree, longueur=LONGUEUR_FENETRE_DEFAUT,
                                                 colonne_cible="RUL_cappee")
    fenetres_test, _, _, _ = construire_fenetres(test, colonnes_entree, longueur=LONGUEUR_FENETRE_DEFAUT)

    donnees = {}
    donnees["sous_jeu"] = sous_jeu
    donnees["train"] = train
    donnees["test"] = test
    donnees["capteurs"] = capteurs
    donnees["colonnes_entree"] = colonnes_entree
    donnees["fenetres"] = fenetres
    donnees["y"] = y.astype(float)
    donnees["groups"] = groups
    donnees["fenetres_test"] = fenetres_test
    donnees["y_test_vraie"] = y_test_vraie
    donnees["moteurs_test"] = moteurs_test
    donnees["X"] = X
    donnees["nb_regimes"] = nb_regimes
    donnees["duree_preparation_s"] = time.perf_counter() - debut
    return donnees


def tableau_predictions_test(donnees, predictions):
    """
    Construit le DataFrame de predictions au format du contrat (une ligne par moteur test et par cycle).

    Parameters
    ----------
    donnees : dict
        Le resultat de `preparer_donnees`.
    predictions : numpy.ndarray
        Une prediction par ligne du test (meme ordre que donnees["test"]).

    Returns
    -------
    pandas.DataFrame
        unit_number, time_cycles, rul_predite, rul_vraie, rul_vraie_cappee, regime.
    """
    test = donnees["test"]
    df = pd.DataFrame()
    df["unit_number"] = test["unit_number"].to_numpy()
    df["time_cycles"] = test["time_cycles"].to_numpy()
    df["rul_predite"] = np.clip(predictions, a_min=0.0, a_max=None)
    if "regime" in test.columns:
        df["regime"] = test["regime"].to_numpy().astype(int)
    else:
        df["regime"] = 0  # FD001 : un seul regime
    df = calculer_rul_vraie(df, donnees["y_test_vraie"])  # RUL vraie a chaque cycle (fonction du contrat)
    # controle : meme RUL vraie que celle calculee par la preparation
    assert np.array_equal(df["rul_vraie"].to_numpy(dtype=float), test["RUL_vraie"].to_numpy(dtype=float))
    return df


# ---------------------------------------------------------------------------------------------------
# Etape "appareils" : CPU contre MPS
# ---------------------------------------------------------------------------------------------------

def mesurer_debit(type_modele, fenetres, y, appareil, nb_threads, nb_lots=40, taille_lot=256):
    """
    Mesure le nombre de lots d'apprentissage traites par seconde (avant + arriere + mise a jour).

    Parameters
    ----------
    type_modele : str
        "lstm" ou "cnn1d".
    fenetres, y : numpy.ndarray
        Des fenetres de FD001 et leur cible.
    appareil : str
        "cpu" ou "mps".
    nb_threads : int
        Coeurs CPU utilises.
    nb_lots : int, optionnel
        Nombre de lots mesures (40 par defaut).
    taille_lot : int, optionnel
        Taille d'un lot (256 par defaut).

    Returns
    -------
    float
        Lots par seconde.
    """
    torch.set_num_threads(nb_threads)
    n = nb_lots * taille_lot
    modele = creer_modele(type_modele, fenetres.shape[2], graine=0)
    # une epoque de "chauffe" (allocation memoire, compilation des noyaux MPS), puis une epoque mesuree
    entrainer_modele(modele, fenetres[:2 * taille_lot], y[:2 * taille_lot], nb_epoques=1, appareil=appareil)
    historique = entrainer_modele(modele, fenetres[:n], y[:n], nb_epoques=1, appareil=appareil)
    return nb_lots / historique["duree_s"]


def ecart_entre_deux_apprentissages(type_modele, fenetres, y, appareil_1, appareil_2, nb_threads_1=1, nb_threads_2=1):
    """
    Entraine deux fois le meme reseau (meme graine, memes donnees, 2 epoques) et mesure l'ecart des predictions.

    Parameters
    ----------
    type_modele : str
        "lstm" ou "cnn1d".
    fenetres, y : numpy.ndarray
        Des fenetres de FD001 et leur cible.
    appareil_1, appareil_2 : str
        Appareil de chaque apprentissage.
    nb_threads_1, nb_threads_2 : int, optionnel
        Coeurs CPU de chaque apprentissage.

    Returns
    -------
    float
        Ecart maximal (en cycles) entre les predictions des deux reseaux sur 2 000 fenetres.
    """
    predictions = []
    for appareil, nb_threads in [(appareil_1, nb_threads_1), (appareil_2, nb_threads_2)]:
        torch.set_num_threads(nb_threads)
        modele = creer_modele(type_modele, fenetres.shape[2], graine=0)
        entrainer_modele(modele, fenetres[:5120], y[:5120], nb_epoques=2, graine=0, appareil=appareil)
        predictions.append(predire(modele, fenetres[10000:12000]))
    return float(np.max(np.abs(predictions[0] - predictions[1])))


def etape_appareils(donnees, arguments):
    """
    Mesure CPU contre MPS (vitesse et reproductibilite) et ecrit docs/resultats/03b_cpu_vs_mps.csv.

    Parameters
    ----------
    donnees : dict
        Donnees FD001 (`preparer_donnees`).
    arguments : argparse.Namespace
        Les options du script.
    """
    afficher("Etape appareils : mesure CPU / MPS")
    fenetres = donnees["fenetres"]
    y = donnees["y"]
    mps_disponible = torch.backends.mps.is_available()
    charge = os.getloadavg()[0]  # charge moyenne de la machine sur la derniere minute (autres calculs)

    lignes = []
    for type_modele in arguments.modeles:
        # La mesure a 6 coeurs est sans risque ICI : xgboost / lightgbm sont importes (via src.config_modele) mais
        # n'ont jamais calcule dans ce processus. Ailleurs (notebook, pytest), PyTorch reste sur 1 coeur pour eviter le
        # blocage OpenMP decrit dans src/deep_learning.py (NB_THREADS_DEFAUT).
        configurations = [("cpu", 1), ("cpu", 6)]
        if mps_disponible:
            configurations.append(("mps", 1))
        for appareil, nb_threads in configurations:
            debit = mesurer_debit(type_modele, fenetres, y, appareil, nb_threads)
            ligne = {}
            ligne["mesure"] = "debit_lots_par_s"
            ligne["type_modele"] = type_modele
            ligne["appareil"] = appareil
            ligne["nb_threads_cpu"] = nb_threads
            ligne["valeur"] = debit
            ligne["charge_machine_1min"] = charge
            lignes.append(ligne)
            afficher(f"  {type_modele} {appareil} ({nb_threads} coeur(s)) : {debit:.1f} lots de 256 par seconde")

        comparaisons = [("cpu", 1, "cpu", 1), ("cpu", 1, "cpu", 6)]
        if mps_disponible:
            comparaisons.append(("mps", 1, "mps", 1))
            comparaisons.append(("cpu", 1, "mps", 1))
        for appareil_1, threads_1, appareil_2, threads_2 in comparaisons:
            ecart = ecart_entre_deux_apprentissages(type_modele, fenetres, y, appareil_1, appareil_2, threads_1, threads_2)
            ligne = {}
            ligne["mesure"] = f"ecart_max_predictions_{appareil_1}{threads_1}_vs_{appareil_2}{threads_2}"
            ligne["type_modele"] = type_modele
            ligne["appareil"] = f"{appareil_1}/{appareil_2}"
            ligne["nb_threads_cpu"] = f"{threads_1}/{threads_2}"
            ligne["valeur"] = ecart
            ligne["charge_machine_1min"] = charge
            lignes.append(ligne)
            afficher(f"  {type_modele} : ecart max des predictions {appareil_1}({threads_1}) vs "
                     f"{appareil_2}({threads_2}) = {ecart:.6f} cycles")
    torch.set_num_threads(1)
    pd.DataFrame(lignes).to_csv(os.path.join(arguments.dossier_resultats, "03b_cpu_vs_mps.csv"), index=False)


# ---------------------------------------------------------------------------------------------------
# Etape "choix_entrees" : time_cycles ou pas
# ---------------------------------------------------------------------------------------------------

def etape_choix_entrees(arguments):
    """
    Decide si time_cycles fait partie des entrees, sur une separation de validation du train FD001.

    Ecrit docs/resultats/03b_choix_entrees_fd001.csv (une ligne par reseau, variante et graine) et
    docs/resultats/03b_choix_entrees_fd001.json (la regle, les moyennes et la decision).

    Parameters
    ----------
    arguments : argparse.Namespace
        Les options du script.

    Returns
    -------
    bool
        Vrai si time_cycles est garde.
    """
    afficher("Etape choix_entrees : time_cycles ou pas (FD001, separation de validation unique)")
    donnees_avec = preparer_donnees("FD001", True)
    groups = donnees_avec["groups"]
    y = donnees_avec["y"]
    nb_capteurs = len(donnees_avec["capteurs"])

    # separation unique : 20 moteurs de decision, puis 64 / 16 parmi les 80 autres
    moteurs_entrainement, moteurs_decision = separer_moteurs(np.unique(groups), FRACTION_VALIDATION, arguments.graine)
    moteurs_apprentissage, moteurs_arret = separer_moteurs(moteurs_entrainement, FRACTION_VALIDATION, arguments.graine)
    masque_apprentissage = np.isin(groups, moteurs_apprentissage)
    masque_arret = np.isin(groups, moteurs_arret)
    masque_decision = np.isin(groups, moteurs_decision)
    afficher(f"  {len(moteurs_apprentissage)} moteurs d'apprentissage, {len(moteurs_arret)} d'arret precoce, "
             f"{len(moteurs_decision)} de decision")

    reglages = reglages_communs(arguments)
    taches = []
    descriptions = []
    for type_modele in arguments.modeles:
        for avec_time_cycles in [False, True]:
            # sans time_cycles : on retire simplement la derniere entree des fenetres (time_cycles / 100)
            fenetres = donnees_avec["fenetres"]
            if not avec_time_cycles:
                fenetres = fenetres[:, :, :nb_capteurs]
            for decalage_graine in range(NB_GRAINES_CHOIX_ENTREES):
                graine = arguments.graine + decalage_graine
                etiquette = f"choix_entrees {type_modele} time_cycles={avec_time_cycles} graine={graine}"
                taches.append(creer_tache(
                    type_modele, graine, fenetres[masque_apprentissage], y[masque_apprentissage], reglages, etiquette,
                    fenetres_validation=fenetres[masque_arret], y_validation=y[masque_arret],
                    a_predire={"decision": fenetres[masque_decision]},
                ))
                description = {}
                description["type_modele"] = type_modele
                description["avec_time_cycles"] = avec_time_cycles
                description["graine"] = graine
                descriptions.append(description)

    resultats = executer_taches(taches, nb_processus=arguments.nb_processus)

    lignes = []
    for position in range(len(taches)):
        predictions = resultats[position]["predictions"]["decision"]
        metriques = calculer_metriques(y[masque_decision], predictions, SEUIL_SURESTIMATION, SEUIL_ALERTE)
        ligne = {}
        for cle in descriptions[position]:
            ligne[cle] = descriptions[position][cle]
        ligne["rmse_decision"] = metriques["rmse"]
        ligne["mae_decision"] = metriques["mae"]
        ligne["nasa_decision"] = metriques["score_nasa"]
        ligne["surestimes_decision"] = metriques["nb_surestimes"]
        ligne["meilleure_epoque"] = resultats[position]["historique"]["meilleure_epoque"]
        ligne["duree_s"] = resultats[position]["historique"]["duree_s"]
        ligne["moteurs_decision"] = json.dumps(convertir_en_types_python(moteurs_decision))
        lignes.append(ligne)
    tableau = pd.DataFrame(lignes)
    tableau.to_csv(os.path.join(arguments.dossier_resultats, "03b_choix_entrees_fd001.csv"), index=False)

    # application de la regle (ecrite avant l'experience, TEXTE_REGLE_ENTREES)
    rmse_moyen_avec = float(tableau[tableau["avec_time_cycles"]]["rmse_decision"].mean())
    rmse_moyen_sans = float(tableau[~tableau["avec_time_cycles"]]["rmse_decision"].mean())
    garder = rmse_moyen_avec < rmse_moyen_sans
    decision = {}
    decision["regle"] = TEXTE_REGLE_ENTREES
    decision["rmse_moyen_avec_time_cycles"] = rmse_moyen_avec
    decision["rmse_moyen_sans_time_cycles"] = rmse_moyen_sans
    decision["garder_time_cycles"] = bool(garder)
    decision["date"] = datetime.date.today().isoformat()
    ecrire_json(decision, os.path.join(arguments.dossier_resultats, "03b_choix_entrees_fd001.json"))
    afficher(f"  RMSE moyen de decision : avec time_cycles {rmse_moyen_avec:.2f}, sans {rmse_moyen_sans:.2f} "
             f"-> time_cycles garde : {garder}")
    return bool(garder)


def decider_entrees(arguments):
    """
    Renvoie la decision sur time_cycles : option --entrees, sinon fichier de decision, sinon experience.

    Parameters
    ----------
    arguments : argparse.Namespace
        Les options du script.

    Returns
    -------
    bool
        Vrai si time_cycles fait partie des entrees.
    """
    if arguments.entrees == "capteurs":
        return False
    if arguments.entrees == "capteurs_time_cycles":
        return True
    chemin = os.path.join(arguments.dossier_resultats, "03b_choix_entrees_fd001.json")
    if "choix_entrees" in arguments.etapes and "FD001" in arguments.sous_jeux:
        return etape_choix_entrees(arguments)
    decision = lire_json(chemin, None)
    if decision is None:  # pas encore de decision : on fait l'experience
        return etape_choix_entrees(arguments)
    afficher(f"Decision des entrees relue dans {chemin} : time_cycles garde = {decision['garder_time_cycles']}")
    return bool(decision["garder_time_cycles"])


# ---------------------------------------------------------------------------------------------------
# FD001 : validation croisee, modele final, variante asymetrique
# ---------------------------------------------------------------------------------------------------

def lancer_cv(donnees, plis, type_modele, arguments, alpha=1.0):
    """
    Lance la validation croisee d'un reseau sur les plis des arbres.

    Parameters
    ----------
    donnees : dict
        Le resultat de `preparer_donnees`.
    plis : list
        Les plis de `calculer_plis`.
    type_modele : str
        "lstm" ou "cnn1d".
    arguments : argparse.Namespace
        Les options du script.
    alpha : float, optionnel
        Poids des surestimations de la perte (1 = MSE).

    Returns
    -------
    dict
        Le resultat de `evaluer_en_validation_croisee_sequentielle`.
    """
    afficher(f"Validation croisee {donnees['sous_jeu']} {type_modele} (alpha = {alpha:g}), {len(plis)} plis")
    return evaluer_en_validation_croisee_sequentielle(
        donnees["fenetres"], donnees["y"], donnees["groups"], plis, type_modele,
        reglages_entrainement=reglages_communs(arguments, alpha=alpha),
        fraction_arret_precoce=FRACTION_VALIDATION, graine=arguments.graine,
        seuil_surestimation=SEUIL_SURESTIMATION, seuil_alerte=SEUIL_ALERTE,
        nb_processus=arguments.nb_processus, nb_threads=1, verbeux=True,
    )


def ecrire_resultats_cv(sous_jeu, donnees, resultats_cv, alphas, arguments, ecrire_oof=True):
    """
    Ecrit les resultats d'une validation croisee : tableau, predictions hors-pli et historiques.

    Fichiers (dans le dossier de resultats) : 03b_cv_<sj>.csv, 03b_oof_<sj>.csv.gz, 03b_historiques_<sj>.json
    (mis a jour sans effacer les autres modeles).

    Parameters
    ----------
    sous_jeu : str
        Le sous-jeu.
    donnees : dict
        Le resultat de `preparer_donnees`.
    resultats_cv : dict
        Nom de la ligne -> resultat de la validation croisee.
    alphas : dict
        Nom de la ligne -> alpha de la perte.
    arguments : argparse.Namespace
        Les options du script.
    ecrire_oof : bool, optionnel
        Vrai (defaut) pour ecrire les predictions hors-pli (03b_oof_<sj>.csv.gz). Pour FD002-FD004, on ecrit
        plutot les metriques par moteur (fichier plus leger, voir `ecrire_metriques_par_moteur`).
    """
    code = sous_jeu.lower()
    nb_moteurs = int(len(np.unique(donnees["groups"])))
    lignes = []
    for nom in resultats_cv:
        type_modele = nom.split("_")[0]
        lignes.append(resumer_cv(nom, type_modele, resultats_cv[nom], nb_moteurs, alphas[nom], donnees["colonnes_entree"]))
    mettre_a_jour_csv(os.path.join(arguments.dossier_resultats, f"03b_cv_{code}.csv"), lignes, ["modele"])

    # predictions hors-pli, alignees ligne a ligne sur le train (memes lignes que les arbres)
    if ecrire_oof:
        chemin_oof = os.path.join(arguments.dossier_resultats, f"03b_oof_{code}.csv.gz")
        if os.path.isfile(chemin_oof):
            tableau_oof = pd.read_csv(chemin_oof)
        else:
            tableau_oof = pd.DataFrame()
            tableau_oof["unit_number"] = donnees["train"]["unit_number"].to_numpy()
            tableau_oof["time_cycles"] = donnees["train"]["time_cycles"].to_numpy()
            tableau_oof["RUL_cappee"] = donnees["y"]
        for nom in resultats_cv:
            tableau_oof[f"oof_{nom}"] = np.round(resultats_cv[nom]["predictions_oof"], 4)
        tableau_oof.to_csv(chemin_oof, index=False, compression="gzip")

    # courbes d'apprentissage de chaque pli
    chemin_historiques = os.path.join(arguments.dossier_resultats, f"03b_historiques_{code}.json")
    historiques = lire_json(chemin_historiques, {})
    for nom in resultats_cv:
        element = {}
        element["plis"] = historiques_simplifies(resultats_cv[nom]["historiques"])
        element["moteurs_arret_precoce_par_pli"] = resultats_cv[nom]["moteurs_arret_precoce_par_pli"]
        historiques[nom] = element
    ecrire_json(historiques, chemin_historiques)


def nb_epoques_depuis_validation(meilleures_epoques):
    """
    Nombre d'epoques du modele final : mediane des meilleures epoques de validation, arrondie, au moins 1.

    Parameters
    ----------
    meilleures_epoques : list de int
        La meilleure epoque de chaque pli (ou de l'unique separation).

    Returns
    -------
    int
        Le nombre d'epoques du modele final.
    """
    return max(1, int(round(float(np.median(meilleures_epoques)))))


def metadonnees_communes(sous_jeu, type_modele, donnees, nb_epoques, graine, nb_parametres, reglages):
    """
    Construit les metadonnees communes a tous les fichiers de predictions des reseaux.

    Parameters
    ----------
    sous_jeu : str
        Le sous-jeu.
    type_modele : str
        "lstm" ou "cnn1d".
    donnees : dict
        Le resultat de `preparer_donnees`.
    nb_epoques : int
        Nombre d'epoques du modele final.
    graine : int
        Graine du modele final.
    nb_parametres : int
        Nombre de poids du reseau.
    reglages : dict
        Reglages d'apprentissage.

    Returns
    -------
    dict
        Les metadonnees (sans metriques ni protocole).
    """
    meta = {}
    meta["sous_jeu"] = sous_jeu
    meta["modele"] = type_modele
    meta["nom_affiche"] = MODELES_CONNUS[type_modele]
    meta["famille"] = "sequentiel"
    meta["date"] = datetime.date.today().isoformat()
    meta["source"] = "scripts/entrainer_deep_learning.py"
    meta["hyperparametres"] = hyperparametres_par_defaut(type_modele)
    meta["reglages_entrainement"] = reglages
    meta["colonnes_entree"] = donnees["colonnes_entree"]
    meta["longueur_fenetre"] = LONGUEUR_FENETRE_DEFAUT
    meta["nb_epoques_modele_final"] = nb_epoques
    meta["graine"] = graine
    meta["nb_parametres"] = nb_parametres
    meta["appareil"] = "cpu (1 coeur par apprentissage)"
    meta["version_torch"] = torch.__version__
    meta["nb_moteurs_train"] = int(len(np.unique(donnees["groups"])))
    meta["nb_lignes_train"] = int(len(donnees["y"]))
    meta["duree_preparation_s"] = donnees["duree_preparation_s"]
    return meta


def entrainer_finaux(donnees, type_modele, nb_epoques, graines, arguments):
    """
    Prepare les taches "modele final" d'un reseau : tout le train, nb_epoques fixe, une tache par graine.

    Parameters
    ----------
    donnees : dict
        Le resultat de `preparer_donnees`.
    type_modele : str
        "lstm" ou "cnn1d".
    nb_epoques : int
        Nombre d'epoques (fixe par la validation).
    graines : list de int
        Les graines.
    arguments : argparse.Namespace
        Les options du script.

    Returns
    -------
    list de dict
        Les taches.
    """
    taches = []
    for graine in graines:
        etiquette = f"final {donnees['sous_jeu']} {type_modele} graine={graine} ({nb_epoques} epoques)"
        taches.append(creer_tache(
            type_modele, graine, donnees["fenetres"], donnees["y"], reglages_communs(arguments), etiquette,
            nb_epoques=nb_epoques, a_predire={"test": donnees["fenetres_test"]}, renvoyer_poids=True,
        ))
    return taches


def publier_modele_final(donnees, type_modele, resultat, nb_epoques, graine, metriques_validation, protocole,
                         arguments, informations_supplementaires=None):
    """
    Ecrit le fichier de predictions (contrat) et les poids d'un modele final ; renvoie ses metriques de test.

    Parameters
    ----------
    donnees : dict
        Le resultat de `preparer_donnees`.
    type_modele : str
        "lstm" ou "cnn1d".
    resultat : dict
        Le resultat de la tache "final" (predictions test, historique, poids).
    nb_epoques : int
        Nombre d'epoques du modele final.
    graine : int
        Graine du modele final.
    metriques_validation : dict
        Metriques de validation (validation croisee ou separation 80/20).
    protocole : str
        Description honnete du protocole de validation.
    arguments : argparse.Namespace
        Les options du script.
    informations_supplementaires : dict ou None, optionnel
        Cles ajoutees telles quelles aux metadonnees.

    Returns
    -------
    dict
        Les metriques de test au dernier cycle (`metriques_dernier_cycle`).
    """
    sous_jeu = donnees["sous_jeu"]
    df = tableau_predictions_test(donnees, resultat["predictions"]["test"])
    metriques_test = metriques_dernier_cycle(df, plafond=PLAFOND_RUL, seuil_surestimation=SEUIL_SURESTIMATION,
                                             seuil_alerte=SEUIL_ALERTE)

    meta = metadonnees_communes(sous_jeu, type_modele, donnees, nb_epoques, graine, resultat["nb_parametres"],
                                reglages_communs(arguments))
    meta["metriques_test"] = metriques_test
    meta["metriques_validation"] = metriques_validation
    meta["protocole_validation"] = protocole
    meta["duree_entrainement_s"] = resultat["historique"]["duree_s"]
    nb_moteurs_test = len(np.unique(df["unit_number"]))
    meta["duree_prediction_test_s"] = resultat["temps_prediction_s"]["test"]
    meta["prediction_ms_par_moteur_test"] = 1000.0 * resultat["temps_prediction_s"]["test"] / nb_moteurs_test

    # poids du modele (models/<SOUS_JEU>__<modele>.pt, ignore par git)
    chemin_poids = chemin_modele(sous_jeu, type_modele, dossier=arguments.dossier_modeles)
    modele = creer_modele(type_modele, donnees["fenetres"].shape[2], graine=graine)
    modele.load_state_dict(resultat["etat_poids"])
    informations = {}
    informations["sous_jeu"] = sous_jeu
    informations["colonnes_entree"] = donnees["colonnes_entree"]
    informations["longueur_fenetre"] = LONGUEUR_FENETRE_DEFAUT
    informations["echelle_time_cycles"] = 100.0
    informations["nb_epoques"] = nb_epoques
    informations["graine"] = graine
    informations["metriques_test"] = metriques_test
    sauvegarder_modele(modele, chemin_poids, type_modele, donnees["fenetres"].shape[2],
                       hyperparametres=hyperparametres_par_defaut(type_modele), informations=informations)
    meta["chemin_poids"] = os.path.relpath(chemin_poids, RACINE_DEPOT)
    if informations_supplementaires is not None:
        for cle in informations_supplementaires:
            meta[cle] = informations_supplementaires[cle]

    chemin_csv, _ = ecrire_predictions(df, meta, dossier=arguments.dossier_predictions)
    afficher(f"  {sous_jeu} {type_modele} : test RMSE {metriques_test['rmse']:.2f}, NASA {metriques_test['score_nasa']:.1f}, "
             f"surestimes > 20 : {metriques_test['nb_surestimes']} -> {chemin_csv}")
    return metriques_test


def metriques_validation_depuis_cv(resultats):
    """
    Extrait d'une validation croisee les metriques de validation (memes cles que les arbres).

    Parameters
    ----------
    resultats : dict
        Le resultat de `evaluer_en_validation_croisee_sequentielle`.

    Returns
    -------
    dict
        rmse_cv_moyen, rmse_cv_ecart_type, mae_cv_moyen, ..., rmse_par_pli, meilleures_epoques, nb_plis.
    """
    metriques = {}
    metriques["rmse_cv_moyen"] = resultats["rmse_moyen"]
    metriques["rmse_cv_ecart_type"] = resultats["rmse_ecart_type"]
    metriques["mae_cv_moyen"] = resultats["mae_moyen"]
    metriques["mae_cv_ecart_type"] = resultats["mae_ecart_type"]
    metriques["nasa_cv_moyen"] = resultats["nasa_moyen"]
    metriques["nasa_cv_ecart_type"] = resultats["nasa_ecart_type"]
    metriques["surestimes_cv_moyen"] = resultats["surestimes_moyen"]
    metriques["surestimes_cv_ecart_type"] = resultats["surestimes_ecart_type"]
    metriques["rmse_oof"] = resultats["rmse_oof"]
    metriques["nasa_oof"] = resultats["nasa_oof"]
    metriques["surestimes_oof"] = resultats["surestimes_oof"]
    metriques["alertes_manquees_oof"] = resultats["alertes_manquees_oof"]
    metriques["fausses_alertes_oof"] = resultats["fausses_alertes_oof"]
    metriques["biais_oof"] = resultats["biais_oof"]
    metriques["rmse_par_pli"] = resultats["rmse_par_pli"]
    metriques["nasa_par_pli"] = resultats["nasa_par_pli"]
    metriques["meilleures_epoques"] = resultats["meilleures_epoques"]
    metriques["nb_plis"] = len(resultats["rmse_par_pli"])
    metriques["temps_entrainement_cv_s"] = resultats["temps_entrainement_total"]
    return metriques


def protocole_cv(sous_jeu, nb_epoques):
    """
    Decrit honnetement le protocole "validation croisee" des reseaux (recopie dans les metadonnees).

    Parameters
    ----------
    sous_jeu : str
        Le sous-jeu.
    nb_epoques : int
        Nombre d'epoques du modele final.

    Returns
    -------
    str
        La description.
    """
    return (
        f"Validation croisee par moteur sur {sous_jeu} : GroupKFold, 5 plis, shuffle=True, random_state=42 (les memes "
        "plis que les modeles a arbres) ; un reseau neuf par pli ; dans chaque pli, 20 % des moteurs d'ENTRAINEMENT "
        "(graine 42) servent a l'arret precoce (patience 10, 80 epoques au plus), les moteurs de validation du pli "
        "ne servent jamais a l'apprentissage ni a l'arret precoce ; predictions de validation clippees a 0 (celles "
        "des arbres ne le sont pas) ; metriques par pli puis moyenne et ecart-type (ddof=0). Preparation (scaler "
        "min-max) apprise sur tout le train avant la validation croisee, comme pour les arbres. Modele final : tous "
        f"les moteurs d'entrainement, {nb_epoques} epoques (mediane des meilleures epoques des 5 plis), sans arret "
        "precoce ; test : prediction a chaque cycle (fenetre des 30 derniers cycles du moteur, remplissage par "
        "repetition du 1er cycle si le moteur en a moins), clippee a 0, metriques au dernier cycle observe contre la "
        "RUL brute. Hyperparametres fixes a priori (non optimises)."
    )


def protocole_separation(sous_jeu, nb_epoques, nb_app, nb_val, avec_cv=False):
    """
    Decrit honnetement le protocole "separation 80 / 20" des reseaux sur FD002-FD004.

    Parameters
    ----------
    sous_jeu : str
        Le sous-jeu.
    nb_epoques : int
        Nombre d'epoques du modele final.
    nb_app, nb_val : int
        Nombre de moteurs d'apprentissage et de validation.
    avec_cv : bool, optionnel
        Vrai si la validation croisee 5 plis complementaire a ete faite (--seulement-cv-sous-jeux).

    Returns
    -------
    str
        La description.
    """
    phrase_cv = "Pas de validation croisee 5 plis pour les reseaux sur ce sous-jeu (cout de calcul). "
    if avec_cv:
        phrase_cv = (f"Une validation croisee 5 plis des memes reseaux, avec les plis des arbres du sous-jeu, a ete faite a "
                     f"part (docs/resultats/03b_cv_{sous_jeu.lower()}.csv, cle metriques_validation_croisee_5_plis) : elle n'a "
                     "servi a aucun choix pour ce modele publie. ")
    return (
        f"Separation unique des moteurs du train {sous_jeu} (graine 42) : {nb_app} moteurs d'apprentissage, {nb_val} "
        "moteurs de validation (20 %). Le reseau est entraine sur les premiers, l'arret precoce (patience 10, 80 epoques "
        "au plus) et les metriques de validation portent sur les seconds (tous leurs cycles) : ces metriques ont donc "
        "servi a choisir l'epoque et sont legerement optimistes ; elles ne sont PAS comparables a la validation "
        "croisee 5 plis des arbres (autre protocole). " + phrase_cv +
        f"Modele final : tous les moteurs d'entrainement, {nb_epoques} epoques (meilleure epoque de "
        "la separation), sans arret precoce ; test : prediction a chaque cycle (fenetre des 30 derniers cycles, "
        "remplissage par repetition du 1er cycle si le moteur en a moins), clippee a 0, metriques au dernier cycle "
        "observe contre la RUL brute. Preparation (regimes K-Means, scalers par regime) de src/sous_jeux.py, apprise "
        "sur tout le train. Entrees et hyperparametres decides sur FD001, non reoptimises."
    )


def traiter_fd001(arguments, avec_time_cycles):
    """
    Protocole complet FD001 : validation croisee, modele final (3 graines), variante asymetrique.

    Parameters
    ----------
    arguments : argparse.Namespace
        Les options du script.
    avec_time_cycles : bool
        Decision sur les entrees.
    """
    donnees = preparer_donnees("FD001", avec_time_cycles)
    afficher(f"FD001 : {donnees['fenetres'].shape[0]} fenetres d'entrainement {donnees['fenetres'].shape[1:]}, "
             f"{donnees['fenetres_test'].shape[0]} fenetres de test ; entrees : {donnees['colonnes_entree']}")
    plis = calculer_plis(donnees["X"], donnees["y"], donnees["groups"], n_splits=N_PLIS, random_state=GRAINE_PLIS)
    chemin_cv = os.path.join(arguments.dossier_resultats, "03b_cv_fd001.csv")
    chemin_historiques = os.path.join(arguments.dossier_resultats, "03b_historiques_fd001.json")

    # --- validation croisee (alpha = 1) ---
    resultats_cv = {}
    if "cv" in arguments.etapes:
        for type_modele in arguments.modeles:
            resultats_cv[type_modele] = lancer_cv(donnees, plis, type_modele, arguments)
            alphas = {type_modele: 1.0}
            un_resultat = {type_modele: resultats_cv[type_modele]}
            ecrire_resultats_cv("FD001", donnees, un_resultat, alphas, arguments)
            afficher(f"FD001 {type_modele} : RMSE CV {resultats_cv[type_modele]['rmse_moyen']:.2f} "
                     f"+/- {resultats_cv[type_modele]['rmse_ecart_type']:.2f}, meilleures epoques "
                     f"{resultats_cv[type_modele]['meilleures_epoques']}")

    tableau_cv = pd.read_csv(chemin_cv)  # resultats de cette execution OU d'une execution precedente

    # --- modele final, 3 graines ---
    if "final" in arguments.etapes:
        graines = []
        for decalage in range(NB_GRAINES_ROBUSTESSE):
            graines.append(arguments.graine + decalage)
        taches = []
        nb_epoques_par_modele = {}
        for type_modele in arguments.modeles:
            ligne_cv = tableau_cv[tableau_cv["modele"] == type_modele].iloc[0]
            nb_epoques_par_modele[type_modele] = nb_epoques_depuis_validation(json.loads(ligne_cv["meilleures_epoques"]))
            for tache in entrainer_finaux(donnees, type_modele, nb_epoques_par_modele[type_modele], graines, arguments):
                taches.append(tache)
        afficher(f"FD001 : {len(taches)} modeles finaux (graines {graines}), epoques {nb_epoques_par_modele}")
        resultats_finaux = executer_taches(taches, nb_processus=arguments.nb_processus)

        lignes_test = []
        predictions_dernier_cycle = None
        historiques = lire_json(chemin_historiques, {})
        position = 0
        for type_modele in arguments.modeles:
            metriques_par_graine = []
            for graine in graines:
                resultat = resultats_finaux[position]
                position = position + 1
                df = tableau_predictions_test(donnees, resultat["predictions"]["test"])
                metriques = metriques_dernier_cycle(df, plafond=PLAFOND_RUL, seuil_surestimation=SEUIL_SURESTIMATION,
                                                    seuil_alerte=SEUIL_ALERTE)
                metriques_par_graine.append(metriques)
                ligne = {}
                ligne["modele"] = type_modele
                ligne["graine"] = graine
                for cle in metriques:
                    ligne[cle] = metriques[cle]
                ligne["nb_epoques"] = nb_epoques_par_modele[type_modele]
                ligne["duree_entrainement_s"] = resultat["historique"]["duree_s"]
                ligne["prediction_ms_par_moteur_test"] = 1000.0 * resultat["temps_prediction_s"]["test"] / metriques["nb_moteurs"]
                lignes_test.append(ligne)

                # predictions au dernier cycle de chaque moteur test, une colonne par (modele, graine)
                derniers = df.sort_values(["unit_number", "time_cycles"]).groupby("unit_number").tail(1)
                if predictions_dernier_cycle is None:
                    predictions_dernier_cycle = pd.DataFrame()
                    predictions_dernier_cycle["unit_number"] = derniers["unit_number"].to_numpy()
                    predictions_dernier_cycle["rul_vraie"] = derniers["rul_vraie"].to_numpy()
                    predictions_dernier_cycle["nb_cycles_observes"] = derniers["time_cycles"].to_numpy()
                predictions_dernier_cycle[f"{type_modele}_graine{graine}"] = np.round(derniers["rul_predite"].to_numpy(), 4)

                historiques[f"final_{type_modele}_graine{graine}"] = historiques_simplifies([resultat["historique"]])[0]

            # robustesse : moyenne et ecart-type (ddof=1, comme pandas .std du notebook 05) sur les 3 graines
            robustesse = {}
            for cle in ["rmse", "rmse_contre_rul_plafonnee", "mae", "score_nasa", "nb_surestimes", "alertes_manquees",
                        "fausses_alertes"]:
                valeurs = []
                for metriques in metriques_par_graine:
                    valeurs.append(metriques[cle])
                robustesse[cle + "_moyenne"] = float(np.mean(valeurs))
                robustesse[cle + "_ecart_type"] = float(np.std(valeurs, ddof=1))
            robustesse["graines"] = graines

            # publication du modele de la graine principale (la premiere)
            ligne_cv = tableau_cv[tableau_cv["modele"] == type_modele].iloc[0]
            metriques_validation = {}
            for cle in ["rmse_cv_moyen", "rmse_cv_ecart_type", "mae_cv_moyen", "mae_cv_ecart_type", "nasa_cv_moyen",
                        "nasa_cv_ecart_type", "surestimes_oof", "alertes_manquees_oof", "fausses_alertes_oof",
                        "biais_oof", "rmse_oof", "temps_entrainement_5_plis_s"]:
                metriques_validation[cle] = float(ligne_cv[cle])
            metriques_validation["rmse_par_pli"] = json.loads(ligne_cv["rmse_par_pli"])
            metriques_validation["meilleures_epoques"] = json.loads(ligne_cv["meilleures_epoques"])
            metriques_validation["nb_plis"] = N_PLIS
            position_principale = position - NB_GRAINES_ROBUSTESSE  # premiere graine de ce modele
            supplement = {"robustesse_graines_test": robustesse}
            publier_modele_final(donnees, type_modele, resultats_finaux[position_principale],
                                 nb_epoques_par_modele[type_modele], graines[0], metriques_validation,
                                 protocole_cv("FD001", nb_epoques_par_modele[type_modele]), arguments, supplement)

        mettre_a_jour_csv(os.path.join(arguments.dossier_resultats, "03b_test_fd001.csv"), lignes_test, ["modele", "graine"])
        chemin_dernier = os.path.join(arguments.dossier_resultats, "03b_predictions_test_fd001.csv")
        if os.path.isfile(chemin_dernier):  # on garde les colonnes des modeles non relances
            ancien = pd.read_csv(chemin_dernier)
            for colonne in ancien.columns:
                if colonne not in predictions_dernier_cycle.columns:
                    predictions_dernier_cycle[colonne] = ancien[colonne].to_numpy()
        predictions_dernier_cycle.to_csv(chemin_dernier, index=False)
        ecrire_json(historiques, chemin_historiques)

        # ligne(s) du tableau agrege
        lignes_agregees = []
        tableau_test = pd.read_csv(os.path.join(arguments.dossier_resultats, "03b_test_fd001.csv"))
        for type_modele in arguments.modeles:
            ligne_cv = tableau_cv[tableau_cv["modele"] == type_modele].iloc[0]
            lignes_modele = tableau_test[tableau_test["modele"] == type_modele]
            ligne_principale = lignes_modele[lignes_modele["graine"] == graines[0]].iloc[0]
            lignes_agregees.append(ligne_agregee("FD001", type_modele, "validation croisee 5 plis", ligne_cv["rmse_cv_moyen"],
                                                 ligne_cv["mae_cv_moyen"], ligne_cv["nasa_cv_moyen"],
                                                 ligne_cv["surestimes_oof"], ligne_principale, lignes_modele,
                                                 donnees, ligne_cv["temps_entrainement_5_plis_s"], ligne_cv["rmse_cv_ecart_type"]))
        mettre_a_jour_csv(os.path.join(arguments.dossier_resultats, "03b_deep_learning.csv"), lignes_agregees,
                          ["sous_jeu", "modele"])

    # --- variante asymetrique du meilleur reseau ---
    if "asymetrique" in arguments.etapes:
        alpha = ALPHA_ASYMETRIQUE  # valeur fixee avant l'experience (voir la constante)
        lignes_alpha_1 = tableau_cv[tableau_cv["alpha"] == 1.0]
        meilleur = str(lignes_alpha_1.sort_values("rmse_cv_moyen").iloc[0]["type_modele"])
        afficher(f"Variante asymetrique : meilleur reseau en CV = {meilleur}, alpha = {alpha:.4f}")
        resultat_alpha = lancer_cv(donnees, plis, meilleur, arguments, alpha=alpha)
        nom = f"{meilleur}_alpha{alpha:.2f}"
        ecrire_resultats_cv("FD001", donnees, {nom: resultat_alpha}, {nom: alpha}, arguments)
        afficher(f"FD001 {nom} : RMSE CV {resultat_alpha['rmse_moyen']:.2f}, NASA CV {resultat_alpha['nasa_moyen']:.0f}, "
                 f"surestimes OOF {resultat_alpha['surestimes_oof']}")


def ligne_agregee(sous_jeu, type_modele, protocole_court, rmse_val, mae_val, nasa_val, surestimes_val, ligne_test,
                  lignes_graines, donnees, temps_validation, rmse_val_ecart_type):
    """
    Construit une ligne du tableau agrege docs/resultats/03b_deep_learning.csv.

    Parameters
    ----------
    sous_jeu : str
        Le sous-jeu.
    type_modele : str
        "lstm" ou "cnn1d".
    protocole_court : str
        "validation croisee 5 plis" ou "separation 80/20".
    rmse_val, mae_val, nasa_val, surestimes_val : float
        Metriques de validation.
    ligne_test : pandas.Series ou dict
        Metriques de test du modele publie.
    lignes_graines : pandas.DataFrame ou None
        Metriques de test de toutes les graines (robustesse), ou None.
    donnees : dict
        Le resultat de `preparer_donnees`.
    temps_validation : float
        Duree d'apprentissage de la validation (s).
    rmse_val_ecart_type : float
        Ecart-type du RMSE de validation entre plis (NaN pour une separation unique).

    Returns
    -------
    dict
        La ligne.
    """
    ligne = {}
    ligne["sous_jeu"] = sous_jeu
    ligne["modele"] = type_modele
    ligne["nom_affiche"] = MODELES_CONNUS[type_modele]
    ligne["protocole_validation"] = protocole_court
    ligne["rmse_validation"] = float(rmse_val)
    ligne["rmse_validation_ecart_type"] = float(rmse_val_ecart_type)
    ligne["mae_validation"] = float(mae_val)
    ligne["nasa_validation"] = float(nasa_val)
    ligne["surestimes_validation"] = int(surestimes_val)
    ligne["rmse_test"] = float(ligne_test["rmse"])
    ligne["rmse_test_rul_plafonnee"] = float(ligne_test["rmse_contre_rul_plafonnee"])
    ligne["mae_test"] = float(ligne_test["mae"])
    ligne["nasa_test"] = float(ligne_test["score_nasa"])
    ligne["surestimes_test"] = int(ligne_test["nb_surestimes"])
    ligne["alertes_manquees_test"] = int(ligne_test["alertes_manquees"])
    ligne["fausses_alertes_test"] = int(ligne_test["fausses_alertes"])
    ligne["nb_moteurs_test"] = int(ligne_test["nb_moteurs"])
    ligne["rmse_test_moyenne_graines"] = np.nan
    ligne["rmse_test_ecart_type_graines"] = np.nan
    ligne["nasa_test_moyenne_graines"] = np.nan
    ligne["nasa_test_ecart_type_graines"] = np.nan
    ligne["nb_graines"] = 1
    if lignes_graines is not None and len(lignes_graines) > 1:
        ligne["rmse_test_moyenne_graines"] = float(lignes_graines["rmse"].mean())
        ligne["rmse_test_ecart_type_graines"] = float(lignes_graines["rmse"].std())
        ligne["nasa_test_moyenne_graines"] = float(lignes_graines["score_nasa"].mean())
        ligne["nasa_test_ecart_type_graines"] = float(lignes_graines["score_nasa"].std())
        ligne["nb_graines"] = int(len(lignes_graines))
    ligne["nb_epoques_final"] = int(ligne_test["nb_epoques"])
    ligne["duree_entrainement_validation_s"] = float(temps_validation)
    ligne["duree_entrainement_final_s"] = float(ligne_test["duree_entrainement_s"])
    ligne["prediction_ms_par_moteur_test"] = float(ligne_test["prediction_ms_par_moteur_test"])
    ligne["entrees"] = "+".join(donnees["colonnes_entree"])
    ligne["nb_regimes"] = donnees["nb_regimes"]
    ligne["nb_moteurs_train"] = int(len(np.unique(donnees["groups"])))
    ligne["date"] = datetime.date.today().isoformat()
    return ligne


# ---------------------------------------------------------------------------------------------------
# FD002 a FD004 : separation 80 / 20, modele final
# ---------------------------------------------------------------------------------------------------

def traiter_autres_sous_jeux(sous_jeux, arguments, avec_time_cycles):
    """
    FD002-FD004 : separation 80/20 pour l'arret precoce, modele final, predictions et poids.

    Toutes les taches de tous les sous-jeux sont lancees ensemble (meilleure occupation des processus) :
    d'abord les apprentissages de validation, puis les modeles finaux.

    Parameters
    ----------
    sous_jeux : list de str
        Les sous-jeux a traiter (parmi FD002, FD003, FD004).
    arguments : argparse.Namespace
        Les options du script.
    avec_time_cycles : bool
        Decision sur les entrees (prise sur FD001).
    """
    if len(sous_jeux) == 0:
        return
    donnees_par_sous_jeu = {}
    separations = {}
    taches = []
    descriptions = []
    for sous_jeu in sous_jeux:
        donnees = preparer_donnees(sous_jeu, avec_time_cycles)
        donnees_par_sous_jeu[sous_jeu] = donnees
        groups = donnees["groups"]
        moteurs_apprentissage, moteurs_validation = separer_moteurs(np.unique(groups), FRACTION_VALIDATION, arguments.graine)
        masque_apprentissage = np.isin(groups, moteurs_apprentissage)
        separations[sous_jeu] = (moteurs_apprentissage, moteurs_validation, masque_apprentissage)
        afficher(f"{sous_jeu} : {donnees['fenetres'].shape[0]} fenetres, {len(moteurs_apprentissage)} moteurs "
                 f"d'apprentissage, {len(moteurs_validation)} de validation, entrees {len(donnees['colonnes_entree'])}")
        for type_modele in arguments.modeles:
            fenetres = donnees["fenetres"]
            y = donnees["y"]
            etiquette = f"separation {sous_jeu} {type_modele}"
            taches.append(creer_tache(
                type_modele, arguments.graine, fenetres[masque_apprentissage], y[masque_apprentissage],
                reglages_communs(arguments), etiquette,
                fenetres_validation=fenetres[~masque_apprentissage], y_validation=y[~masque_apprentissage],
                a_predire={"validation": fenetres[~masque_apprentissage]},
            ))
            descriptions.append((sous_jeu, type_modele))

    afficher(f"FD002-FD004 : {len(taches)} apprentissages de validation (separation 80/20)")
    resultats_validation = executer_taches(taches, nb_processus=arguments.nb_processus)

    # metriques de validation et nombre d'epoques de chaque modele final
    validations = {}
    chemin_historiques_autres = os.path.join(arguments.dossier_resultats, "03b_historiques_sous_jeux.json")
    historiques = lire_json(chemin_historiques_autres, {})
    taches_finales = []
    for position in range(len(taches)):
        sous_jeu, type_modele = descriptions[position]
        donnees = donnees_par_sous_jeu[sous_jeu]
        masque_apprentissage = separations[sous_jeu][2]
        resultat = resultats_validation[position]
        metriques = calculer_metriques(donnees["y"][~masque_apprentissage], resultat["predictions"]["validation"],
                                       SEUIL_SURESTIMATION, SEUIL_ALERTE)
        nb_epoques = nb_epoques_depuis_validation([resultat["historique"]["meilleure_epoque"]])
        validation = {}
        validation["rmse_validation"] = metriques["rmse"]
        validation["mae_validation"] = metriques["mae"]
        validation["nasa_validation"] = metriques["score_nasa"]
        validation["surestimes_validation"] = metriques["nb_surestimes"]
        validation["alertes_manquees_validation"] = metriques["alertes_manquees"]
        validation["fausses_alertes_validation"] = metriques["fausses_alertes"]
        validation["biais_validation"] = metriques["biais"]
        validation["nb_lignes_validation"] = metriques["nb_lignes"]
        validation["nb_moteurs_validation"] = int(len(separations[sous_jeu][1]))
        validation["meilleure_epoque"] = resultat["historique"]["meilleure_epoque"]
        validation["nb_epoques_effectuees"] = resultat["historique"]["nb_epoques_effectuees"]
        validation["duree_entrainement_validation_s"] = resultat["historique"]["duree_s"]
        validation["moteurs_validation"] = convertir_en_types_python(separations[sous_jeu][1])
        validations[(sous_jeu, type_modele)] = (validation, nb_epoques)
        historiques[f"{sous_jeu}_{type_modele}_separation"] = historiques_simplifies([resultat["historique"]])[0]
        for tache in entrainer_finaux(donnees, type_modele, nb_epoques, [arguments.graine], arguments):
            taches_finales.append(tache)
    ecrire_json(historiques, chemin_historiques_autres)

    afficher(f"FD002-FD004 : {len(taches_finales)} modeles finaux")
    resultats_finaux = executer_taches(taches_finales, nb_processus=arguments.nb_processus)

    lignes_agregees = []
    for position in range(len(descriptions)):
        sous_jeu, type_modele = descriptions[position]
        donnees = donnees_par_sous_jeu[sous_jeu]
        validation, nb_epoques = validations[(sous_jeu, type_modele)]
        nb_app = len(separations[sous_jeu][0])
        nb_val = len(separations[sous_jeu][1])
        resultat = resultats_finaux[position]
        metriques_test = publier_modele_final(donnees, type_modele, resultat, nb_epoques, arguments.graine, validation,
                                              protocole_separation(sous_jeu, nb_epoques, nb_app, nb_val), arguments)
        ligne_test = {}
        for cle in metriques_test:
            ligne_test[cle] = metriques_test[cle]
        ligne_test["nb_epoques"] = nb_epoques
        ligne_test["duree_entrainement_s"] = resultat["historique"]["duree_s"]
        ligne_test["prediction_ms_par_moteur_test"] = 1000.0 * resultat["temps_prediction_s"]["test"] / metriques_test["nb_moteurs"]
        lignes_agregees.append(ligne_agregee(sous_jeu, type_modele, "separation 80/20", validation["rmse_validation"],
                                             validation["mae_validation"], validation["nasa_validation"],
                                             validation["surestimes_validation"], ligne_test, None, donnees,
                                             validation["duree_entrainement_validation_s"], np.nan))
    mettre_a_jour_csv(os.path.join(arguments.dossier_resultats, "03b_deep_learning.csv"), lignes_agregees,
                      ["sous_jeu", "modele"])

    # validation croisee optionnelle (memes plis que les arbres du sous-jeu)
    if arguments.cv_sous_jeux:
        traiter_cv_sous_jeux(sous_jeux, arguments, avec_time_cycles)
        integrer_cv_sous_jeux(arguments)


# ---------------------------------------------------------------------------------------------------
# Validation croisee optionnelle sur FD002-FD004 (memes plis que les arbres du sous-jeu)
# ---------------------------------------------------------------------------------------------------

def metriques_par_moteur_cv(sous_jeu, type_modele, y, predictions_oof, groups):
    """
    Calcule les metriques hors-pli de chaque moteur (format de docs/resultats/01b_metriques_par_moteur_cv.csv).

    Parameters
    ----------
    sous_jeu : str
        Le sous-jeu.
    type_modele : str
        "lstm" ou "cnn1d".
    y : numpy.ndarray
        La cible de chaque ligne.
    predictions_oof : numpy.ndarray
        Les predictions hors-pli de chaque ligne.
    groups : numpy.ndarray
        Le moteur de chaque ligne.

    Returns
    -------
    pandas.DataFrame
        Une ligne par moteur : sous_jeu, modele, unit_number, rmse, biais, surestimation_max, nb_lignes, nasa,
        surestimes.
    """
    tableau = erreurs_par_moteur(y, predictions_oof, groups)  # unit_number, rmse, biais, surestimation_max, nb_lignes
    penalites = []
    surestimes = []
    for moteur_id in tableau["unit_number"]:  # penalite NASA et surestimations > 20, moteur par moteur
        masque = groups == moteur_id
        penalites.append(score_nasa(y[masque], predictions_oof[masque]))
        surestimes.append(nb_surestimes(y[masque], predictions_oof[masque], seuil=SEUIL_SURESTIMATION))
    tableau["nasa"] = penalites
    tableau["surestimes"] = surestimes
    tableau.insert(0, "modele", type_modele)
    tableau.insert(0, "sous_jeu", sous_jeu)
    return tableau


def remplacer_groupes_csv(chemin, nouveau, colonnes_groupe):
    """
    Ecrit un CSV en remplacant les lignes des groupes presents dans `nouveau` (ex. un (sous-jeu, modele)).

    Parameters
    ----------
    chemin : str
        Le fichier CSV (cree s'il n'existe pas).
    nouveau : pandas.DataFrame
        Les nouvelles lignes.
    colonnes_groupe : list de str
        Les colonnes qui definissent un groupe.
    """
    if os.path.isfile(chemin):
        ancien = pd.read_csv(chemin)
        cles_nouvelles = set()
        for _, ligne in nouveau[colonnes_groupe].drop_duplicates().iterrows():
            cle = []
            for colonne in colonnes_groupe:
                cle.append(str(ligne[colonne]))
            cles_nouvelles.add(tuple(cle))
        garder = []
        for _, ligne in ancien[colonnes_groupe].iterrows():
            cle = []
            for colonne in colonnes_groupe:
                cle.append(str(ligne[colonne]))
            garder.append(tuple(cle) not in cles_nouvelles)
        nouveau = pd.concat([ancien[np.array(garder, dtype=bool)], nouveau], ignore_index=True)
    nouveau.to_csv(chemin, index=False)


def traiter_cv_sous_jeux(sous_jeux, arguments, avec_time_cycles):
    """
    Validation croisee 5 plis des reseaux sur FD002-FD004, avec les plis des arbres de chaque sous-jeu.

    Optionnelle (--seulement-cv-sous-jeux) : elle ne change ni les modeles finaux ni les fichiers de
    predictions ; elle permet de comparer reseaux et arbres en validation, sur les memes moteurs. Pour
    chaque sous-jeu, les 10 apprentissages (2 reseaux x 5 plis) tournent ensemble dans le meme lot de
    processus. Fichiers : 03b_cv_<sj>.csv, 03b_historiques_<sj>.json, 03b_metriques_par_moteur_cv.csv.

    Parameters
    ----------
    sous_jeux : list de str
        Les sous-jeux (parmi FD002, FD003, FD004).
    arguments : argparse.Namespace
        Les options du script.
    avec_time_cycles : bool
        Decision sur les entrees (prise sur FD001).
    """
    for sous_jeu in sous_jeux:
        donnees = preparer_donnees(sous_jeu, avec_time_cycles)
        plis = calculer_plis(donnees["X"], donnees["y"], donnees["groups"], n_splits=N_PLIS, random_state=GRAINE_PLIS)
        toutes_les_taches = []
        blocs = []  # (type_modele, position de la premiere tache, moteurs d'arret precoce par pli)
        for type_modele in arguments.modeles:
            taches, moteurs_arret = preparer_taches_validation_croisee(
                donnees["fenetres"], donnees["y"], donnees["groups"], plis, type_modele,
                reglages_entrainement=reglages_communs(arguments), fraction_arret_precoce=FRACTION_VALIDATION,
                graine=arguments.graine, nb_threads=1,
            )
            blocs.append((type_modele, len(toutes_les_taches), moteurs_arret))
            for tache in taches:
                tache["etiquette"] = f"{sous_jeu} {tache['etiquette']}"
                toutes_les_taches.append(tache)
        afficher(f"Validation croisee {sous_jeu} : {len(toutes_les_taches)} apprentissages")
        resultats_taches = executer_taches(toutes_les_taches, nb_processus=arguments.nb_processus)

        for type_modele, position, moteurs_arret in blocs:
            resultats_modele = resultats_taches[position:position + len(plis)]
            resultat_cv = assembler_validation_croisee(donnees["y"], plis, resultats_modele, moteurs_arret,
                                                       seuil_surestimation=SEUIL_SURESTIMATION, seuil_alerte=SEUIL_ALERTE,
                                                       etiquette=f"{sous_jeu} {type_modele}", verbeux=True)
            ecrire_resultats_cv(sous_jeu, donnees, {type_modele: resultat_cv}, {type_modele: 1.0}, arguments,
                                ecrire_oof=False)
            par_moteur = metriques_par_moteur_cv(sous_jeu, type_modele, donnees["y"], resultat_cv["predictions_oof"],
                                                 donnees["groups"])
            remplacer_groupes_csv(os.path.join(arguments.dossier_resultats, "03b_metriques_par_moteur_cv.csv"),
                                  par_moteur, ["sous_jeu", "modele"])
            afficher(f"{sous_jeu} {type_modele} : RMSE CV {resultat_cv['rmse_moyen']:.2f} +/- "
                     f"{resultat_cv['rmse_ecart_type']:.2f}, meilleures epoques {resultat_cv['meilleures_epoques']}")


COLONNES_CV_AGREGEES = ["rmse_cv_moyen", "rmse_cv_ecart_type", "mae_cv_moyen", "nasa_cv_moyen", "surestimes_oof",
                        "alertes_manquees_oof", "fausses_alertes_oof", "biais_oof", "temps_entrainement_5_plis_s"]


def integrer_cv_sous_jeux(arguments):
    """
    Reporte les validations croisees 5 plis dans le tableau agrege et dans les metadonnees des modeles publies.

    Rien n'est reentraine : on relit 03b_cv_<sj>.csv et on ajoute (1) des colonnes "cv5_..." a
    03b_deep_learning.csv (tous les sous-jeux, FD001 compris), (2) pour FD002-FD004, la cle
    "metriques_validation_croisee_5_plis" et un texte de protocole exact dans le JSON du fichier de predictions.

    Parameters
    ----------
    arguments : argparse.Namespace
        Les options du script (dossiers).
    """
    from src.contrat_predictions import lire_predictions

    chemin_agrege = os.path.join(arguments.dossier_resultats, "03b_deep_learning.csv")
    agrege = pd.read_csv(chemin_agrege)
    for sous_jeu in SOUS_JEUX:
        chemin_cv = os.path.join(arguments.dossier_resultats, f"03b_cv_{sous_jeu.lower()}.csv")
        if not os.path.isfile(chemin_cv):
            continue  # pas de validation croisee pour ce sous-jeu
        tableau_cv = pd.read_csv(chemin_cv)
        for type_modele in TYPES_MODELES:
            lignes_modele = tableau_cv[tableau_cv["modele"] == type_modele]
            if len(lignes_modele) == 0:
                continue
            ligne_cv = lignes_modele.iloc[0]
            masque = (agrege["sous_jeu"] == sous_jeu) & (agrege["modele"] == type_modele)
            for cle in COLONNES_CV_AGREGEES:
                agrege.loc[masque, "cv5_" + cle] = float(ligne_cv[cle])
            if sous_jeu == "FD001":
                continue  # FD001 : la validation croisee est deja le protocole principal (metadonnees a jour)
            try:
                df, meta = lire_predictions(sous_jeu, type_modele, dossier=arguments.dossier_predictions)
            except (FileNotFoundError, ValueError):
                continue
            resume_cv = {}
            for cle in COLONNES_CV_AGREGEES:
                resume_cv[cle] = float(ligne_cv[cle])
            resume_cv["rmse_par_pli"] = json.loads(ligne_cv["rmse_par_pli"])
            resume_cv["meilleures_epoques"] = json.loads(ligne_cv["meilleures_epoques"])
            resume_cv["nb_plis"] = N_PLIS
            meta["metriques_validation_croisee_5_plis"] = resume_cv
            nb_val = int(meta["metriques_validation"]["nb_moteurs_validation"])
            nb_app = int(meta["nb_moteurs_train"]) - nb_val
            meta["protocole_validation"] = protocole_separation(sous_jeu, int(meta["nb_epoques_modele_final"]), nb_app,
                                                                nb_val, avec_cv=True)
            for cle in ["version_contrat", "nb_lignes", "nb_moteurs"]:  # recalcules par ecrire_predictions
                meta.pop(cle, None)
            ecrire_predictions(df, meta, dossier=arguments.dossier_predictions)
            afficher(f"Metadonnees {sous_jeu} {type_modele} completees (RMSE CV {resume_cv['rmse_cv_moyen']:.2f})")
    agrege.to_csv(chemin_agrege, index=False)
    afficher(f"Tableau agrege mis a jour : {chemin_agrege}")


# ---------------------------------------------------------------------------------------------------
# Resume : dernier cycle de chaque moteur test, tous sous-jeux (fichier versionne)
# ---------------------------------------------------------------------------------------------------

def ecrire_derniers_cycles(arguments):
    """
    Ecrit docs/resultats/03b_predictions_test_dernier_cycle.csv a partir des fichiers de predictions.

    Meme format que docs/resultats/01b_predictions_test_dernier_cycle.csv (modeles a arbres) : une ligne par
    (sous-jeu, modele, moteur test), au dernier cycle observe. Ce fichier est versionne, contrairement aux
    fichiers de predictions complets (sorties/, ignore par git) : le notebook 03b peut ainsi faire ses tests
    apparies sur le test sans relancer l'apprentissage.

    Parameters
    ----------
    arguments : argparse.Namespace
        Les options du script (dossiers).
    """
    from src.contrat_predictions import extraire_dernier_cycle, lire_predictions

    lignes = []
    for sous_jeu in SOUS_JEUX:
        for type_modele in TYPES_MODELES:
            try:
                df, _ = lire_predictions(sous_jeu, type_modele, dossier=arguments.dossier_predictions)
            except (FileNotFoundError, ValueError):
                continue  # ce couple n'a pas (encore) ete produit
            derniers = extraire_dernier_cycle(df)
            for _, ligne_df in derniers.iterrows():
                ligne = {}
                ligne["sous_jeu"] = sous_jeu
                ligne["modele"] = type_modele
                ligne["unit_number"] = int(ligne_df["unit_number"])
                ligne["rul_vraie"] = float(ligne_df["rul_vraie"])
                ligne["rul_predite"] = float(ligne_df["rul_predite"])
                ligne["nb_cycles_observes"] = int(ligne_df["time_cycles"])
                ligne["regime_dernier_cycle"] = int(ligne_df["regime"])
                lignes.append(ligne)
    chemin = os.path.join(arguments.dossier_resultats, "03b_predictions_test_dernier_cycle.csv")
    pd.DataFrame(lignes).to_csv(chemin, index=False)
    afficher(f"Dernier cycle des moteurs test ecrit : {chemin} ({len(lignes)} lignes)")


# ---------------------------------------------------------------------------------------------------
# Programme principal
# ---------------------------------------------------------------------------------------------------

def main():
    """
    Enchaine les etapes demandees par les options, affiche la progression et la duree totale.
    """
    arguments = lire_arguments()
    if arguments.seulement_resume:  # aucun apprentissage : on relit les fichiers de predictions existants
        ecrire_derniers_cycles(arguments)
        return
    if arguments.seulement_cv_sous_jeux:  # validation croisee FD002-FD004 seulement (entrees decidees sur FD001)
        torch.set_num_threads(1)
        debut = time.perf_counter()
        decision = lire_json(os.path.join(arguments.dossier_resultats, "03b_choix_entrees_fd001.json"), None)
        autres = []
        for sous_jeu in arguments.sous_jeux:
            if sous_jeu != "FD001":
                autres.append(sous_jeu)
        traiter_cv_sous_jeux(autres, arguments, bool(decision["garder_time_cycles"]))
        integrer_cv_sous_jeux(arguments)
        afficher(f"Validation croisee FD002-FD004 terminee en {(time.perf_counter() - debut) / 60:.1f} minutes.")
        return
    if arguments.seulement_integrer_cv:  # aucun apprentissage
        integrer_cv_sous_jeux(arguments)
        return
    os.makedirs(arguments.dossier_resultats, exist_ok=True)
    torch.set_num_threads(1)  # le processus principal ne fait que preparer les donnees et coordonner
    debut = time.perf_counter()
    afficher(f"Debut : sous-jeux {arguments.sous_jeux}, modeles {arguments.modeles}, graine {arguments.graine}, "
             f"{arguments.nb_processus} processus, torch {torch.__version__}")

    if "FD001" in arguments.sous_jeux and "appareils" in arguments.etapes:
        etape_appareils(preparer_donnees("FD001", True), arguments)

    avec_time_cycles = decider_entrees(arguments)
    texte_entrees = "capteurs seuls"
    if avec_time_cycles:
        texte_entrees = "capteurs + time_cycles / 100"
    afficher(f"Entrees des reseaux : {texte_entrees}")

    if "FD001" in arguments.sous_jeux:
        traiter_fd001(arguments, avec_time_cycles)

    autres = []
    for sous_jeu in arguments.sous_jeux:
        if sous_jeu != "FD001":
            autres.append(sous_jeu)
    traiter_autres_sous_jeux(autres, arguments, avec_time_cycles)
    ecrire_derniers_cycles(arguments)

    afficher(f"Termine en {(time.perf_counter() - debut) / 60:.1f} minutes.")


if __name__ == "__main__":
    main()
