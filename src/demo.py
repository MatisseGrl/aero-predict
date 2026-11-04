"""
Demo en ligne de commande du prototype Aero Predict (Phase 4) : prediction de la RUL de turboreacteurs.

But : prouver devant un jury que le prototype fonctionne de bout en bout, de la donnee capteur brute
jusqu'a une decision de maintenance, en quelques secondes et sans notebook. Quatre commandes :

    python -m src.demo entrainer [--config CHEMIN]
        entraine le modele retenu (docs/resultats/04_config_retenue.json, sinon XGBoost Phase 3)
        sur les 100 moteurs d'entrainement et sauvegarde l'artefact models/modele_retenu.joblib ;

    python -m src.demo moteur --id N [--seuil-alerte 30] [--sortie DOSSIER]
        predit la RUL a chaque cycle du moteur test N, affiche un resume et ecrit une figure PNG ;

    python -m src.demo flotte [--seuil-alerte 30] [--top K]
        predit les 100 moteurs test au dernier cycle observe, trie par urgence, resume les metriques ;

    python -m src.demo benchmark [--repetitions 5] [--cible-ms 200]
        mesure les temps de reponse (exigence non fonctionnelle) sur la machine courante.

L'artefact contient le modele entraine ET tout ce qui a ete appris sur le train pour preparer de
nouvelles donnees exactement de la meme facon (capteurs constants retires, scaler min-max, fenetre,
plafond, liste ordonnee des features). Les donnees test ne servent jamais a rien apprendre : le
scaler est seulement applique (transform), jamais reajuste (fit).
"""

import argparse
import copy
import datetime
import os
import pickle
import platform
import sys
import time

import cloudpickle
import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn
import xgboost

from src.config_modele import (
    CHEMIN_CONFIG_RETENUE,
    RACINE_DEPOT,
    charger_config,
    construire_fabrique_depuis_config,
    convertir_en_types_python,
    decrire_config,
)
from src.evaluation import (
    alertes_manquees,
    evaluer_sur_test,
    fausses_alertes,
    mae,
    nb_surestimes,
    rmse,
    score_nasa,
)
from src.features import ajouter_rul_vraie_test, charger_rul_test, construire_features_glissantes, preparer_train
from src.preprocessing import lisser_capteurs, load_fd001

# ---------------------------------------------------------------------------------------------------
# Constantes de la demo
# ---------------------------------------------------------------------------------------------------

# emplacements par defaut (chemins absolus, calcules depuis la racine du depot)
CHEMIN_ARTEFACT_DEFAUT = os.path.join(RACINE_DEPOT, "models", "modele_retenu.joblib")
DOSSIER_SORTIE_DEFAUT = os.path.join(RACINE_DEPOT, "sorties", "demo")

# reglages par defaut des commandes
SEUIL_ALERTE_DEFAUT = 30.0  # seuil d'intervention en cycles (hypothese de travail, voir src/evaluation.py)
SEUIL_SURESTIMATION = 20  # une surestimation de plus de 20 cycles est comptee comme dangereuse (Phase 3)
TOP_DEFAUT = 15  # nombre de moteurs affiches par la commande flotte
REPETITIONS_DEFAUT = 5  # nombre de passages sur les 100 moteurs pour le benchmark
CIBLE_MS_DEFAUT = 200.0  # temps de reponse cible pour UN moteur (exigence non fonctionnelle)
NB_MOTEURS_TEST = 100  # le jeu de test FD001 contient les moteurs 1 a 100

# format de l'artefact : a incrementer si son contenu change (un ancien artefact sera alors refuse)
VERSION_FORMAT_ARTEFACT = 1
CLES_ARTEFACT = [
    "version_format",
    "modele",
    "config",
    "description_modele",
    "chemin_config",
    "config_repli_phase3",
    "scaler",
    "feature_cols",
    "sensor_cols_restants",
    "constant_sensors",
    "fenetre",
    "plafond",
    "date_entrainement",
    "duree_preparation_s",
    "duree_entrainement_s",
    "nb_moteurs_train",
    "nb_lignes_train",
    "versions",
]

# les 4 statuts possibles d'un moteur au regard de l'alerte (meme definition que src/evaluation.py)
STATUT_ALERTE_CORRECTE = "alerte correcte"  # vraie RUL <= seuil ET RUL predite <= seuil
STATUT_ALERTE_MANQUEE = "ALERTE MANQUEE"  # vraie RUL <= seuil MAIS RUL predite > seuil (dangereux)
STATUT_FAUSSE_ALERTE = "fausse alerte"  # vraie RUL > seuil MAIS RUL predite <= seuil (cout, pas de risque)
STATUT_RIEN_A_SIGNALER = "rien a signaler"  # vraie RUL > seuil ET RUL predite > seuil

EXPLICATIONS_STATUT = {
    STATUT_ALERTE_CORRECTE: "le moteur est vraiment proche de la panne et le modele declenche l'alerte",
    STATUT_ALERTE_MANQUEE: "le moteur est proche de la panne mais le modele ne declenche PAS l'alerte (cas dangereux)",
    STATUT_FAUSSE_ALERTE: (
        "le modele alerte alors que le moteur a encore de la marge (maintenance inutile, pas de risque)"
    ),
    STATUT_RIEN_A_SIGNALER: "le moteur a encore de la marge et le modele ne declenche pas d'alerte",
}

LARGEUR_ECRAN = 86  # largeur des bandeaux de titre, pensee pour un terminal projete


class ErreurDemo(Exception):
    """
    Erreur "utilisateur" de la demo (identifiant de moteur invalide, configuration incorrecte, ...).

    `main` l'attrape et affiche son message en francais, sans trace Python : devant un jury, une
    faute de frappe doit donner une phrase claire, pas 30 lignes de traceback. Les autres erreurs
    (de vrais bugs) ne sont PAS attrapees, pour rester visibles et corrigeables.
    """


# ---------------------------------------------------------------------------------------------------
# Affichage : petits utilitaires pour une sortie lisible sur un ecran projete
# ---------------------------------------------------------------------------------------------------

def afficher_titre(texte):
    """
    Affiche un bandeau de titre encadre de lignes de "=".

    Parameters
    ----------
    texte : str
        Le titre a afficher.

    Returns
    -------
    None
    """
    print()
    print("=" * LARGEUR_ECRAN)
    print("  " + texte)
    print("=" * LARGEUR_ECRAN)


def afficher_section(texte):
    """
    Affiche un titre de section encadre de lignes de "-".

    Parameters
    ----------
    texte : str
        Le titre de la section.

    Returns
    -------
    None
    """
    print()
    print("-" * LARGEUR_ECRAN)
    print("  " + texte)
    print("-" * LARGEUR_ECRAN)


def afficher_ligne(libelle, valeur, largeur_libelle=30):
    """
    Affiche une ligne "libelle : valeur", avec les deux-points alignes d'une ligne a l'autre.

    Parameters
    ----------
    libelle : str
        Le nom de la grandeur affichee.
    valeur : objet
        La valeur (convertie en texte).
    largeur_libelle : int, optionnel
        Largeur reservee au libelle, pour aligner les deux-points (30 par defaut).

    Returns
    -------
    None
    """
    print(f"  {libelle:<{largeur_libelle}} : {valeur}")


def afficher_suite(texte, largeur_libelle=30):
    """
    Affiche une ligne de suite, alignee sur la colonne des valeurs de `afficher_ligne`.

    Parameters
    ----------
    texte : str
        Le texte a afficher (par exemple une explication de la ligne precedente).
    largeur_libelle : int, optionnel
        La meme largeur que celle utilisee par `afficher_ligne` (30 par defaut).

    Returns
    -------
    None
    """
    print("  " + " " * largeur_libelle + "   " + texte)  # 2 espaces + libelle vide + " : " remplace par 3 espaces


def formater_entier(nombre):
    """
    Ecrit un entier avec une espace comme separateur de milliers, a la francaise (20631 -> "20 631").

    Parameters
    ----------
    nombre : int
        L'entier a formater.

    Returns
    -------
    str
        Le nombre formate.
    """
    texte_anglais = f"{int(nombre):,}"  # separateur anglais : "20,631"
    return texte_anglais.replace(",", " ")  # separateur francais : "20 631"


def formater_ecart(ecart):
    """
    Ecrit un ecart (RUL predite - RUL vraie) avec son signe, a 1 decimale ("+2.8", "-12.4").

    Parameters
    ----------
    ecart : float
        L'ecart en cycles.

    Returns
    -------
    str
        L'ecart signe.
    """
    return f"{ecart:+.1f}"


def chemin_lisible(chemin):
    """
    Raccourcit un chemin pour l'affichage : relatif a la racine du depot s'il est dedans, absolu sinon.

    Parameters
    ----------
    chemin : str
        Le chemin a afficher.

    Returns
    -------
    str
        Par exemple "models/modele_retenu.joblib" au lieu du chemin absolu complet.
    """
    chemin_absolu = os.path.abspath(chemin)
    if chemin_absolu.startswith(RACINE_DEPOT + os.sep):  # le fichier est dans le depot
        return os.path.relpath(chemin_absolu, RACINE_DEPOT)
    return chemin_absolu  # en dehors du depot (ex. dossier temporaire des tests) : chemin complet


def nature_ecart(ecart):
    """
    Donne en mots le sens d'un ecart RUL predite - RUL vraie.

    Parameters
    ----------
    ecart : float
        L'ecart en cycles (positif = le modele croit qu'il reste plus de cycles qu'en realite).

    Returns
    -------
    str
        "surestimation (sens dangereux)", "sous-estimation (sens prudent)" ou "prediction exacte".
    """
    if ecart > 0:  # le modele croit qu'il reste plus de cycles qu'en realite
        return "surestimation (sens dangereux)"
    if ecart < 0:  # le modele est plus pessimiste que la realite
        return "sous-estimation (sens prudent)"
    return "prediction exacte"


# ---------------------------------------------------------------------------------------------------
# Classification d'alerte
# ---------------------------------------------------------------------------------------------------

def classer_alerte(rul_predite, rul_vraie, seuil_alerte=SEUIL_ALERTE_DEFAUT):
    """
    Dit si le modele declenche une alerte, et si cette decision etait la bonne.

    Regle (identique a `alertes_manquees` / `fausses_alertes` de src/evaluation.py) : le modele
    declenche une alerte des que la RUL predite est INFERIEURE OU EGALE au seuil ; le moteur
    "aurait du" etre arrete si sa vraie RUL est inferieure ou egale au seuil. Un cas limite
    egal au seuil (30 cycles predits, ou 30 cycles vrais) compte donc comme "en alerte".

    Parameters
    ----------
    rul_predite : float
        La RUL predite (en cycles).
    rul_vraie : float
        La vraie RUL (en cycles).
    seuil_alerte : float, optionnel
        Le seuil d'intervention (30 cycles par defaut, hypothese de travail de l'equipe).

    Returns
    -------
    (bool, str)
        - Vrai si une alerte est declenchee (RUL predite <= seuil) ;
        - le statut : STATUT_ALERTE_CORRECTE, STATUT_ALERTE_MANQUEE, STATUT_FAUSSE_ALERTE ou
          STATUT_RIEN_A_SIGNALER.
    """
    alerte_declenchee = float(rul_predite) <= seuil_alerte  # decision du modele
    aurait_du_alerter = float(rul_vraie) <= seuil_alerte  # ce qu'il aurait fallu faire

    if alerte_declenchee and aurait_du_alerter:  # bonne alerte
        statut = STATUT_ALERTE_CORRECTE
    elif aurait_du_alerter:  # il fallait alerter, le modele ne l'a pas fait : le cas dangereux
        statut = STATUT_ALERTE_MANQUEE
    elif alerte_declenchee:  # alerte alors que le moteur avait de la marge
        statut = STATUT_FAUSSE_ALERTE
    else:  # ni alerte, ni besoin d'alerte
        statut = STATUT_RIEN_A_SIGNALER

    return bool(alerte_declenchee), statut


# ---------------------------------------------------------------------------------------------------
# Artefact : entrainement, sauvegarde, chargement
# ---------------------------------------------------------------------------------------------------

def lire_versions_bibliotheques():
    """
    Releve les versions de Python et des bibliotheques utilisees, pour la tracabilite de l'artefact.

    Un modele sauvegarde avec une version de XGBoost peut se comporter differemment avec une autre :
    garder ces versions dans l'artefact permet de comprendre un ecart plus tard.

    Returns
    -------
    dict
        Nom de la bibliotheque -> numero de version (texte).
    """
    versions = {}
    versions["python"] = platform.python_version()
    versions["numpy"] = np.__version__
    versions["pandas"] = pd.__version__
    versions["scikit-learn"] = sklearn.__version__
    versions["xgboost"] = xgboost.__version__
    versions["lightgbm"] = lightgbm.__version__
    return versions


def entrainer_modele_retenu(config, contexte_train=None, chemin_config="", config_repli_phase3=False):
    """
    Entraine le modele decrit par `config` sur les 100 moteurs d'entrainement et construit l'artefact.

    Parameters
    ----------
    config : dict
        Une configuration valide (voir src/config_modele.py).
    contexte_train : dict, optionnel
        Le resultat de `preparer_train()`, s'il est deja calcule (les tests le reutilisent pour
        gagner du temps). Par defaut, le pipeline train est recalcule ici.
    chemin_config : str, optionnel
        Le fichier de configuration d'origine (garde dans l'artefact pour la tracabilite).
    config_repli_phase3 : bool, optionnel
        Vrai si `config` est le repli CONFIG_PHASE3 (fichier de configuration absent).

    Returns
    -------
    dict
        L'artefact en memoire, avec les cles de CLES_ARTEFACT : le modele entraine, la configuration,
        le contexte de preparation appris sur le train (scaler, capteurs constants, features,
        fenetre, plafond), la date et les durees d'entrainement, les versions des bibliotheques.
    """
    # 1. preparation du train : meme pipeline que la Phase 3 (src/features.py)
    debut = time.perf_counter()
    if contexte_train is None:
        contexte_train = preparer_train()
    duree_preparation = time.perf_counter() - debut

    # 2. un modele neuf construit depuis la configuration, entraine sur TOUT le train (100 moteurs)
    fabrique = construire_fabrique_depuis_config(config)
    modele = fabrique()
    debut = time.perf_counter()
    modele.fit(contexte_train["X"], contexte_train["y"])
    duree_entrainement = time.perf_counter() - debut

    # 3. l'artefact : le modele + tout ce qu'il faut pour preparer de nouvelles donnees a l'identique
    artefact = {}
    artefact["version_format"] = VERSION_FORMAT_ARTEFACT
    artefact["modele"] = modele
    artefact["config"] = convertir_en_types_python(copy.deepcopy(config))  # copie en types Python simples
    artefact["description_modele"] = decrire_config(config)
    if chemin_config:  # chemin absolu du fichier de configuration d'origine, s'il a ete donne
        artefact["chemin_config"] = os.path.abspath(chemin_config)
    else:  # configuration passee directement en memoire (cas des tests)
        artefact["chemin_config"] = ""
    artefact["config_repli_phase3"] = bool(config_repli_phase3)
    artefact["scaler"] = contexte_train["scaler"]  # appris sur le train, jamais reajuste ensuite
    artefact["feature_cols"] = list(contexte_train["feature_cols"])  # ordre exact attendu par le modele
    artefact["sensor_cols_restants"] = list(contexte_train["sensor_cols_restants"])
    artefact["constant_sensors"] = list(contexte_train["constant_sensors"])
    artefact["fenetre"] = contexte_train["fenetre"]
    artefact["plafond"] = contexte_train["plafond"]
    artefact["date_entrainement"] = datetime.datetime.now().isoformat(timespec="seconds")
    artefact["duree_preparation_s"] = float(duree_preparation)
    artefact["duree_entrainement_s"] = float(duree_entrainement)
    artefact["nb_moteurs_train"] = int(contexte_train["groups"].nunique())
    artefact["nb_lignes_train"] = int(len(contexte_train["X"]))
    artefact["versions"] = lire_versions_bibliotheques()
    return artefact


def sauvegarder_artefact(artefact, chemin):
    """
    Ecrit l'artefact sur le disque avec joblib (le dossier parent est cree si besoin).

    Detail technique : les pertes sur mesure (quadratique asymetrique, forme NASA) sont des fonctions
    definies a l'interieur d'autres fonctions (src/modeles.py). Le module pickle standard, utilise par
    joblib, ne sait pas les enregistrer ("Can't pickle local object"). On serialise donc le MODELE
    avec cloudpickle (installe avec joblib), qui enregistre ces fonctions par valeur, puis on range ces
    octets dans le dictionnaire ecrit par joblib. `charger_artefact` fait l'operation inverse.

    Parameters
    ----------
    artefact : dict
        L'artefact renvoye par `entrainer_modele_retenu` (n'est pas modifie).
    chemin : str
        Le fichier a ecrire (par exemple models/modele_retenu.joblib).

    Returns
    -------
    str
        Le chemin du fichier ecrit.
    """
    os.makedirs(os.path.dirname(os.path.abspath(chemin)), exist_ok=True)  # cree models/ si besoin

    # copie superficielle du dictionnaire : on remplace seulement le modele par ses octets cloudpickle
    a_ecrire = {}
    for cle in artefact:
        a_ecrire[cle] = artefact[cle]
    a_ecrire["modele"] = cloudpickle.dumps(artefact["modele"])
    a_ecrire["format_modele"] = "cloudpickle"

    joblib.dump(a_ecrire, chemin, compress=3)  # compression moderee : fichier plus petit, lecture rapide
    return chemin


def charger_artefact(chemin):
    """
    Relit un artefact ecrit par `sauvegarder_artefact` et verifie qu'il est complet.

    Parameters
    ----------
    chemin : str
        Le fichier .joblib a lire.

    Returns
    -------
    dict
        L'artefact, avec le modele entraine pret a predire sous la cle "modele".

    Raises
    ------
    ErreurDemo
        Si le fichier n'existe pas, ou s'il ne contient pas toutes les cles attendues (ancien format).
    """
    if not os.path.exists(chemin):
        raise ErreurDemo(f"Aucun modele entraine trouve : {chemin}. Lancez d'abord : python -m src.demo entrainer")

    contenu = joblib.load(chemin)  # dictionnaire ecrit par sauvegarder_artefact

    # verification : toutes les cles attendues sont presentes, et le format est le bon
    cles_manquantes = []
    for cle in CLES_ARTEFACT:
        if cle not in contenu:
            cles_manquantes.append(cle)
    if len(cles_manquantes) > 0 or contenu.get("version_format") != VERSION_FORMAT_ARTEFACT:
        raise ErreurDemo(
            f"Artefact incomplet ou d'un ancien format ({chemin}, cles manquantes : {cles_manquantes}). "
            "Relancez : python -m src.demo entrainer"
        )

    # le modele a ete serialise avec cloudpickle : pickle.loads sait le relire
    artefact = {}
    for cle in contenu:
        artefact[cle] = contenu[cle]
    artefact["modele"] = pickle.loads(contenu["modele"])
    return artefact


def lancer_entrainement(chemin_config, chemin_artefact):
    """
    Charge la configuration, entraine le modele, sauvegarde l'artefact et affiche un resume.

    C'est le coeur de la commande `entrainer` ; `moteur`, `flotte` et `benchmark` l'appellent aussi
    automatiquement quand l'artefact n'existe pas encore.

    Parameters
    ----------
    chemin_config : str
        Le fichier de configuration JSON (s'il n'existe pas : repli sur XGBoost Phase 3, avec message).
    chemin_artefact : str
        Le fichier .joblib a ecrire.

    Returns
    -------
    dict
        L'artefact entraine (en memoire).

    Raises
    ------
    ErreurDemo
        Si la configuration existe mais est invalide.
    """
    debut_total = time.perf_counter()
    afficher_titre("AERO PREDICT - Entrainement du modele retenu")

    # configuration : fichier retenu par l'optimisation, ou repli sur la Phase 3 (message affiche)
    config_repli_phase3 = not os.path.exists(chemin_config)
    try:
        config = charger_config(chemin_config)
    except ValueError as erreur:  # fichier present mais incorrect : message clair, pas de traceback
        raise ErreurDemo(f"Configuration invalide dans {chemin_config} : {erreur}") from erreur

    if config_repli_phase3:
        source_affichee = f"{chemin_lisible(chemin_config)} absent -> repli sur la configuration Phase 3"
    else:
        source_affichee = chemin_lisible(chemin_config)
    afficher_ligne("Configuration", source_affichee)
    afficher_ligne("Modele", decrire_config(config))
    afficher_ligne("Regle de choix", config.get("regle_de_choix", "(non precisee)"))
    afficher_ligne("Hyperparametres", config.get("hyperparametres", {}))
    print("  ... preparation des donnees et entrainement en cours ...")

    artefact = entrainer_modele_retenu(
        config, chemin_config=chemin_config, config_repli_phase3=config_repli_phase3
    )
    sauvegarder_artefact(artefact, chemin_artefact)
    taille_ko = os.path.getsize(chemin_artefact) / 1024.0
    duree_totale = time.perf_counter() - debut_total

    # resume lisible
    afficher_ligne(
        "Donnees d'entrainement",
        f"{artefact['nb_moteurs_train']} moteurs, {formater_entier(artefact['nb_lignes_train'])} cycles, "
        f"{len(artefact['feature_cols'])} features",
    )
    afficher_ligne("Preparation des donnees", f"{artefact['duree_preparation_s']:.1f} s")
    afficher_ligne("Entrainement du modele", f"{artefact['duree_entrainement_s']:.1f} s")
    afficher_ligne("Duree totale", f"{duree_totale:.1f} s")
    afficher_ligne("Artefact sauvegarde", f"{chemin_lisible(chemin_artefact)} ({taille_ko:.0f} Ko)")
    afficher_ligne("Date d'entrainement", artefact["date_entrainement"])

    # metriques annoncees dans la configuration (calculees par l'optimisation), si elles existent
    blocs_metriques = (("metriques_cv", "Metriques CV annoncees"), ("metriques_test", "Metriques test annoncees"))
    for cle_metriques, libelle in blocs_metriques:
        metriques = config.get(cle_metriques, {})
        morceaux = []
        if isinstance(metriques, dict):
            for nom_metrique in metriques:
                valeur = metriques[nom_metrique]
                if isinstance(valeur, (int, float)) and not isinstance(valeur, bool):  # on n'affiche que les nombres
                    morceaux.append(f"{nom_metrique}={valeur:.2f}")
        if len(morceaux) > 0:
            afficher_ligne(libelle, ", ".join(morceaux))
    return artefact


def verifier_fraicheur_artefact(artefact):
    """
    Previent si la configuration a change depuis l'entrainement de l'artefact (sans rien bloquer).

    Exemple typique : l'artefact a ete entraine avec le repli Phase 3, puis l'optimisation a ecrit
    docs/resultats/04_config_retenue.json. Sans ce message, la demo montrerait l'ancien modele.

    Parameters
    ----------
    artefact : dict
        L'artefact charge.

    Returns
    -------
    bool
        Vrai si l'artefact est a jour (ou si on ne peut pas le savoir), Faux si un avertissement a ete affiche.
    """
    chemin_config = artefact.get("chemin_config", "")
    if chemin_config == "" or not os.path.exists(chemin_config):  # rien a comparer
        return True

    message = ""
    if artefact.get("config_repli_phase3", False):  # entraine sans fichier... qui existe maintenant
        message = f"une configuration retenue existe maintenant ({chemin_lisible(chemin_config)})"
    else:
        try:
            config_actuelle = charger_config(chemin_config)
        except ValueError:
            return True  # fichier devenu invalide : la commande entrainer le signalera
        if convertir_en_types_python(config_actuelle) != artefact["config"]:
            message = f"la configuration {chemin_lisible(chemin_config)} a change"

    if message == "":
        return True
    print(f"  [attention] {message} depuis l'entrainement de cet artefact ({artefact['date_entrainement']}).")
    print("  [attention] Pour l'utiliser : python -m src.demo entrainer")
    return False


def charger_ou_entrainer(chemin_artefact, chemin_config=CHEMIN_CONFIG_RETENUE):
    """
    Charge l'artefact s'il existe ; sinon, entraine automatiquement le modele retenu puis le charge.

    Parameters
    ----------
    chemin_artefact : str
        Le fichier .joblib de l'artefact.
    chemin_config : str, optionnel
        La configuration a utiliser en cas d'entrainement automatique.

    Returns
    -------
    dict
        L'artefact, pret a predire.
    """
    if not os.path.exists(chemin_artefact):
        print(f"\n  Aucun modele entraine trouve ({chemin_lisible(chemin_artefact)}) : entrainement automatique.")
        lancer_entrainement(chemin_config, chemin_artefact)
    artefact = charger_artefact(chemin_artefact)  # on relit le fichier : on teste ce qui est vraiment sur le disque
    verifier_fraicheur_artefact(artefact)
    return artefact


# ---------------------------------------------------------------------------------------------------
# Preparation des donnees brutes et prediction
# ---------------------------------------------------------------------------------------------------

def charger_donnees_test():
    """
    Charge les donnees test brutes (capteurs) et la vraie RUL au dernier cycle de chaque moteur.

    Returns
    -------
    (pandas.DataFrame, numpy.ndarray)
        Le contenu brut de test_FD001.txt (26 colonnes, tous les cycles des 100 moteurs) et les
        100 valeurs de RUL_FD001.txt (moteur 1 en premier).
    """
    test_brut = load_fd001("test")
    rul_test = charger_rul_test()
    return test_brut, rul_test


def preparer_donnees_brutes(donnees_brutes, artefact):
    """
    Applique a des donnees capteurs BRUTES exactement la preparation apprise sur le train.

    Memes etapes, dans le meme ordre, que `preparer_test` (src/features.py) : retrait des capteurs
    constants trouves sur le train, normalisation avec le scaler du train (transform SEUL, jamais
    fit), lissage par moteur, features glissantes et NaN du 1er cycle remplaces par 0. Toutes ces
    etapes travaillent moteur par moteur : preparer un seul moteur ou les 100 d'un coup donne
    exactement les memes valeurs pour ce moteur (verifie dans tests/test_demo.py).

    Parameters
    ----------
    donnees_brutes : pandas.DataFrame
        Les lignes brutes (format test_FD001.txt) d'un ou de plusieurs moteurs. Non modifie.
    artefact : dict
        L'artefact (on y lit constant_sensors, sensor_cols_restants, scaler et fenetre).

    Returns
    -------
    pandas.DataFrame
        Les donnees preparees : une ligne par cycle, avec toutes les colonnes de features.
    """
    sensor_cols = artefact["sensor_cols_restants"]
    fenetre = artefact["fenetre"]

    donnees = donnees_brutes.drop(columns=artefact["constant_sensors"])  # memes capteurs retires que sur le train
    # transform seul : le scaler applique les minimums / maximums du TRAIN, il n'apprend rien ici
    donnees[sensor_cols] = artefact["scaler"].transform(donnees[sensor_cols])
    donnees = lisser_capteurs(donnees, sensor_cols, fenetre=fenetre)  # moyenne mobile causale, par moteur
    donnees, colonnes_moyenne, colonnes_ecart_type = construire_features_glissantes(donnees, sensor_cols, fenetre)
    return donnees


def predire_rul(modele, X):
    """
    Predit la RUL et la ramene a 0 si elle est negative (meme regle que `evaluer_sur_test`).

    Parameters
    ----------
    modele : objet avec .predict
        Le modele entraine.
    X : pandas.DataFrame
        Les features, dans l'ordre de l'artefact.

    Returns
    -------
    numpy.ndarray
        Les RUL predites, en cycles, toutes >= 0.
    """
    predictions = np.asarray(modele.predict(X), dtype=float)
    return np.clip(predictions, a_min=0, a_max=None)  # une RUL negative n'a pas de sens physique


def verifier_id_moteur(moteur_id, nb_moteurs=NB_MOTEURS_TEST):
    """
    Verifie qu'un numero de moteur test existe (1 a 100 pour FD001).

    Parameters
    ----------
    moteur_id : int
        Le numero demande.
    nb_moteurs : int, optionnel
        Le nombre de moteurs du jeu de test (100 pour FD001).

    Returns
    -------
    None

    Raises
    ------
    ErreurDemo
        Si le numero est hors de [1, nb_moteurs].
    """
    if moteur_id < 1 or moteur_id > nb_moteurs:
        raise ErreurDemo(
            f"Moteur {moteur_id} inexistant. Le jeu de test FD001 contient {nb_moteurs} moteurs, "
            f"numerotes de 1 a {nb_moteurs} (exemple : --id 34)."
        )


def predire_moteur(artefact, test_brut, rul_test, moteur_id):
    """
    Predit la RUL a CHAQUE cycle observe d'un moteur test, a partir de ses donnees capteurs brutes.

    Le chronometre couvre la preparation des donnees brutes de ce moteur + la prediction de tous
    ses cycles (ce que ferait le systeme a l'arrivee de nouvelles mesures), pas la lecture du fichier.

    Parameters
    ----------
    artefact : dict
        L'artefact charge.
    test_brut : pandas.DataFrame
        Les donnees test brutes (les 100 moteurs).
    rul_test : numpy.ndarray
        La RUL vraie au dernier cycle de chaque moteur (RUL_FD001.txt, moteur 1 en premier).
    moteur_id : int
        Le numero du moteur test (1 a 100).

    Returns
    -------
    dict
        - "moteur_id" : int ;
        - "trajectoire" : pandas.DataFrame avec, a chaque cycle, "time_cycles", "rul_predite",
          "RUL_vraie" et "RUL_vraie_cappee" (vraie RUL plafonnee, ce que le modele a appris a predire) ;
        - "nb_cycles" : nombre de cycles observes ;
        - "rul_predite_dernier", "rul_vraie_dernier", "ecart_dernier" : au dernier cycle observe ;
        - "premier_cycle_alerte" : None (fonction `ajouter_alertes_moteur` le remplit) ;
        - "temps_ms" : duree preparation + prediction, en millisecondes.

    Raises
    ------
    ErreurDemo
        Si le moteur n'existe pas.
    """
    verifier_id_moteur(moteur_id)

    # lignes brutes de ce moteur seulement (copie : on ne touche pas au tableau complet)
    lignes_brutes = test_brut[test_brut["unit_number"] == moteur_id].copy()
    if len(lignes_brutes) == 0:
        raise ErreurDemo(f"Aucune donnee pour le moteur {moteur_id} dans test_FD001.txt.")

    # chronometre : preparation des donnees brutes + prediction de tous les cycles
    debut = time.perf_counter()
    donnees = preparer_donnees_brutes(lignes_brutes, artefact)
    predictions = predire_rul(artefact["modele"], donnees[artefact["feature_cols"]])
    temps_ms = (time.perf_counter() - debut) * 1000.0

    # vraie RUL a chaque cycle : RUL au dernier cycle (RUL_FD001.txt) + cycles restant jusqu'au dernier
    rul_vraie_dernier = int(rul_test[moteur_id - 1])  # la ligne 1 du fichier correspond au moteur 1
    avec_rul = ajouter_rul_vraie_test(donnees, np.array([rul_vraie_dernier]), artefact["plafond"])

    # trajectoire : un tableau simple, une ligne par cycle
    trajectoire = pd.DataFrame()
    trajectoire["time_cycles"] = avec_rul["time_cycles"].values
    trajectoire["rul_predite"] = predictions
    trajectoire["RUL_vraie"] = avec_rul["RUL_vraie"].values
    trajectoire["RUL_vraie_cappee"] = avec_rul["RUL_vraie_cappee"].values

    # les lignes sont dans l'ordre des cycles : la derniere est la plus recente
    rul_predite_dernier = float(predictions[-1])

    resultat = {}
    resultat["moteur_id"] = int(moteur_id)
    resultat["trajectoire"] = trajectoire
    resultat["nb_cycles"] = int(len(trajectoire))
    resultat["rul_predite_dernier"] = rul_predite_dernier
    resultat["rul_vraie_dernier"] = rul_vraie_dernier
    resultat["ecart_dernier"] = rul_predite_dernier - rul_vraie_dernier
    resultat["premier_cycle_alerte"] = None
    resultat["temps_ms"] = float(temps_ms)
    return resultat


def premier_cycle_en_alerte(trajectoire, seuil_alerte):
    """
    Cherche le premier cycle ou la RUL predite passe sous le seuil d'alerte (le modele aurait alerte).

    Parameters
    ----------
    trajectoire : pandas.DataFrame
        La trajectoire renvoyee par `predire_moteur`.
    seuil_alerte : float
        Le seuil d'alerte en cycles.

    Returns
    -------
    (int, int) ou None
        (cycle, vraie RUL a ce cycle), ou None si le modele n'alerte jamais sur les cycles observes.
    """
    for position in range(len(trajectoire)):  # du premier cycle observe au dernier
        if trajectoire["rul_predite"].iloc[position] <= seuil_alerte:
            cycle = int(trajectoire["time_cycles"].iloc[position])
            rul_vraie = int(trajectoire["RUL_vraie"].iloc[position])
            return cycle, rul_vraie
    return None


def predire_flotte(artefact, test_brut, rul_test, seuil_alerte=SEUIL_ALERTE_DEFAUT):
    """
    Predit la RUL de tous les moteurs test au dernier cycle observe et classe chaque alerte.

    Parameters
    ----------
    artefact : dict
        L'artefact charge.
    test_brut : pandas.DataFrame
        Les donnees test brutes (les 100 moteurs).
    rul_test : numpy.ndarray
        La RUL vraie au dernier cycle de chaque moteur (moteur 1 en premier).
    seuil_alerte : float, optionnel
        Le seuil d'alerte en cycles (30 par defaut).

    Returns
    -------
    dict
        - "tableau" : pandas.DataFrame trie par numero de moteur, colonnes "moteur", "nb_cycles",
          "rul_predite", "rul_vraie", "ecart" (predite - vraie), "alerte" (bool), "statut" ;
        - "X_dernier" : pandas.DataFrame, les features du dernier cycle de chaque moteur (meme ordre) ;
        - "temps_s" : duree preparation + prediction des 100 moteurs, en secondes.
    """
    # chronometre : preparation des donnees brutes de toute la flotte + prediction
    debut = time.perf_counter()
    donnees = preparer_donnees_brutes(test_brut, artefact)
    # dernier cycle de chaque moteur, trie par numero de moteur (aligne avec RUL_FD001.txt)
    indices_derniers_cycles = donnees.groupby("unit_number")["time_cycles"].idxmax()
    derniers_cycles = donnees.loc[indices_derniers_cycles].sort_values("unit_number").reset_index(drop=True)
    X_dernier = derniers_cycles[artefact["feature_cols"]]
    predictions = predire_rul(artefact["modele"], X_dernier)
    temps_s = time.perf_counter() - debut

    assert len(derniers_cycles) == len(rul_test), "un dernier cycle par moteur attendu (aligne avec RUL_FD001.txt)"
    nb_cycles_par_moteur = donnees.groupby("unit_number").size()  # nombre de cycles observes par moteur

    # une ligne par moteur, construite explicitement
    lignes = []
    for position in range(len(derniers_cycles)):
        moteur_id = int(derniers_cycles["unit_number"].iloc[position])
        rul_predite = float(predictions[position])
        rul_vraie = int(rul_test[position])  # meme ordre : moteurs tries, comme RUL_FD001.txt
        alerte, statut = classer_alerte(rul_predite, rul_vraie, seuil_alerte)

        ligne = {}
        ligne["moteur"] = moteur_id
        ligne["nb_cycles"] = int(nb_cycles_par_moteur.loc[moteur_id])
        ligne["rul_predite"] = rul_predite
        ligne["rul_vraie"] = rul_vraie
        ligne["ecart"] = rul_predite - rul_vraie
        ligne["alerte"] = alerte
        ligne["statut"] = statut
        lignes.append(ligne)

    resultat = {}
    resultat["tableau"] = pd.DataFrame(lignes)
    resultat["X_dernier"] = X_dernier
    resultat["temps_s"] = float(temps_s)
    return resultat


def resumer_flotte(tableau, seuil_alerte=SEUIL_ALERTE_DEFAUT):
    """
    Calcule les metriques de la flotte avec les fonctions de src/evaluation.py.

    Parameters
    ----------
    tableau : pandas.DataFrame
        Le tableau renvoye par `predire_flotte` (colonnes rul_predite, rul_vraie, alerte, statut).
    seuil_alerte : float, optionnel
        Le seuil d'alerte en cycles.

    Returns
    -------
    dict
        "nb_moteurs", "rmse", "mae", "score_nasa", "nb_surestimes" (> 20 cycles), "biais",
        "nb_sous_seuil" (vraie RUL <= seuil), "alertes_emises", "alertes_correctes",
        "alertes_manquees", "fausses_alertes".
    """
    vrais = tableau["rul_vraie"].values
    predits = tableau["rul_predite"].values

    resume = {}
    resume["nb_moteurs"] = int(len(tableau))
    resume["rmse"] = rmse(vrais, predits)
    resume["mae"] = mae(vrais, predits)
    resume["score_nasa"] = score_nasa(vrais, predits)
    resume["nb_surestimes"] = nb_surestimes(vrais, predits, seuil=SEUIL_SURESTIMATION)
    resume["biais"] = float(np.mean(predits - vrais))  # negatif = le modele sous-estime en moyenne
    resume["nb_sous_seuil"] = int(np.sum(vrais <= seuil_alerte))  # moteurs qu'il aurait fallu arreter
    resume["alertes_emises"] = int(np.sum(tableau["alerte"].values))
    resume["alertes_correctes"] = int(np.sum(tableau["statut"].values == STATUT_ALERTE_CORRECTE))
    resume["alertes_manquees"] = alertes_manquees(vrais, predits, seuil_alerte=seuil_alerte)
    resume["fausses_alertes"] = fausses_alertes(vrais, predits, seuil_alerte=seuil_alerte)
    return resume


def controler_coherence_evaluation(artefact, X_dernier, rul_test, tableau, resume, seuil_alerte=SEUIL_ALERTE_DEFAUT):
    """
    Verifie que la demo donne EXACTEMENT les memes chiffres que `evaluer_sur_test` (src/evaluation.py).

    Deux chemins de calcul independants doivent tomber d'accord : la demo (donnees brutes -> preparation
    -> prediction -> classement) et la fonction d'evaluation officielle de la Phase 4.

    Parameters
    ----------
    artefact : dict
        L'artefact charge.
    X_dernier : pandas.DataFrame
        Les features du dernier cycle de chaque moteur (renvoyees par `predire_flotte`).
    rul_test : numpy.ndarray
        La RUL vraie au dernier cycle de chaque moteur.
    tableau : pandas.DataFrame
        Le tableau renvoye par `predire_flotte`, trie par numero de moteur.
    resume : dict
        Le resume renvoye par `resumer_flotte`.
    seuil_alerte : float, optionnel
        Le seuil d'alerte en cycles.

    Returns
    -------
    (bool, list de str)
        Vrai si tout concorde, et la liste des ecarts constates (vide si tout concorde).
    """
    officiel = evaluer_sur_test(
        artefact["modele"],
        X_dernier,
        rul_test,
        plafond=artefact["plafond"],
        seuil_surestimation=SEUIL_SURESTIMATION,
        seuil_alerte=seuil_alerte,
    )

    ecarts = []
    if not np.allclose(officiel["predictions"], tableau["rul_predite"].values, rtol=0, atol=1e-9):
        ecarts.append("predictions differentes")
    for cle_demo, cle_officielle in (("rmse", "rmse"), ("mae", "mae"), ("score_nasa", "score_nasa")):
        if abs(resume[cle_demo] - officiel[cle_officielle]) > 1e-9:
            ecarts.append(f"{cle_demo} : demo {resume[cle_demo]} / officiel {officiel[cle_officielle]}")
    for cle in ("nb_surestimes", "alertes_manquees", "fausses_alertes"):
        if resume[cle] != officiel[cle]:
            ecarts.append(f"{cle} : demo {resume[cle]} / officiel {officiel[cle]}")

    # le classement moteur par moteur doit redonner les memes comptes que les fonctions officielles
    nb_manquees_tableau = int(np.sum(tableau["statut"].values == STATUT_ALERTE_MANQUEE))
    nb_fausses_tableau = int(np.sum(tableau["statut"].values == STATUT_FAUSSE_ALERTE))
    if nb_manquees_tableau != officiel["alertes_manquees"]:
        ecarts.append("statuts 'alerte manquee' incoherents")
    if nb_fausses_tableau != officiel["fausses_alertes"]:
        ecarts.append("statuts 'fausse alerte' incoherents")
    return len(ecarts) == 0, ecarts


# ---------------------------------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------------------------------

def tracer_moteur(resultat, seuil_alerte, statut, description_modele, chemin_png):
    """
    Trace la RUL predite et la RUL vraie au fil des cycles d'un moteur test, et ecrit un PNG.

    Elements de la figure : vraie RUL (trait noir), vraie RUL plafonnee a 125 (pointilles verts :
    c'est ce que le modele a appris a predire), RUL predite (bleu), seuil d'alerte (tirets rouges) et
    zone d'alerte coloree en rouge pale, premier cycle ou le modele aurait alerte (tirets orange),
    point du dernier cycle observe et encadre de resume.

    Parameters
    ----------
    resultat : dict
        Le resultat de `predire_moteur` (avec "premier_cycle_alerte" rempli ou None).
    seuil_alerte : float
        Le seuil d'alerte en cycles.
    statut : str
        Le statut d'alerte au dernier cycle (voir `classer_alerte`).
    description_modele : str
        Le nom lisible du modele (pour le titre).
    chemin_png : str
        Le fichier PNG a ecrire (le dossier est cree si besoin).

    Returns
    -------
    str
        Le chemin du PNG ecrit.
    """
    # import ici (et pas en haut du fichier) : matplotlib n'est utile qu'a cette commande, et son
    # chargement prend quelques centaines de ms. On utilise directement la classe Figure (API objet)
    # plutot que pyplot : aucune fenetre, aucun etat global, donc aucun effet de bord si cette
    # fonction est appelee depuis un notebook qui a ses propres reglages d'affichage.
    from matplotlib.figure import Figure

    trajectoire = resultat["trajectoire"]
    cycles = trajectoire["time_cycles"].values
    dernier_cycle = int(cycles[-1])
    plafond = int(trajectoire["RUL_vraie_cappee"].max())

    fig = Figure(figsize=(12, 6.5))  # figure en memoire, independante de pyplot
    ax = fig.subplots()  # un seul graphique

    # zone d'alerte (RUL <= seuil) coloree, puis la ligne du seuil
    ax.axhspan(0, seuil_alerte, color="#d62728", alpha=0.10, label=f"zone d'alerte (RUL <= {seuil_alerte:g} cycles)")
    ax.axhline(
        seuil_alerte, color="#d62728", linestyle="--", linewidth=1.3, label=f"seuil d'alerte : {seuil_alerte:g} cycles"
    )

    # les trois courbes de RUL
    ax.plot(
        cycles, trajectoire["RUL_vraie"].values, color="black", linewidth=2.2,
        label="RUL vraie (RUL_FD001 + cycles restants)",
    )
    ax.plot(
        cycles, trajectoire["RUL_vraie_cappee"].values, color="#2ca02c", linestyle=":", linewidth=2.4,
        label=f"RUL vraie plafonnee a {plafond} (ce que le modele apprend a predire)",
    )
    ax.plot(
        cycles, trajectoire["rul_predite"].values, color="#1f77b4", linewidth=2.2, label="RUL predite par le modele"
    )

    # premier cycle ou le modele aurait declenche l'alerte
    if resultat["premier_cycle_alerte"] is not None:
        cycle_alerte, rul_vraie_alerte = resultat["premier_cycle_alerte"]
        ax.axvline(
            cycle_alerte, color="#ff7f0e", linestyle="-.", linewidth=1.5,
            label=f"1re alerte du modele : cycle {cycle_alerte} (RUL vraie {rul_vraie_alerte})",
        )

    # points du dernier cycle observe (la decision de maintenance se prend ici)
    ax.scatter([dernier_cycle], [resultat["rul_vraie_dernier"]], color="black", s=70, zorder=5)
    ax.scatter([dernier_cycle], [resultat["rul_predite_dernier"]], color="#1f77b4", s=70, zorder=5)

    # encadre de resume, en haut a droite
    texte_resume = (
        f"Dernier cycle observe : {dernier_cycle}\n"
        f"RUL predite : {resultat['rul_predite_dernier']:.1f} cycles\n"
        f"RUL vraie : {resultat['rul_vraie_dernier']} cycles\n"
        f"Ecart : {formater_ecart(resultat['ecart_dernier'])} cycles\n"
        f"Statut : {statut}"
    )
    ax.text(
        0.985, 0.97, texte_resume, transform=ax.transAxes, ha="right", va="top", multialignment="left", fontsize=11,
        family="monospace", bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "#888888", "alpha": 0.95},
    )

    # axes, titre, legende
    valeur_max = max(float(trajectoire["RUL_vraie"].max()), float(trajectoire["rul_predite"].max()), seuil_alerte)
    ax.set_ylim(0, valeur_max * 1.30)  # marge en haut pour l'encadre de resume
    ax.set_xlim(0, dernier_cycle + max(3, int(0.02 * dernier_cycle)))
    ax.set_xlabel("Cycle de fonctionnement observe", fontsize=12)
    ax.set_ylabel("RUL : cycles restants avant la panne", fontsize=12)
    ax.set_title(
        f"Moteur test {resultat['moteur_id']} (FD001) : RUL predite vs RUL vraie, cycle par cycle\n"
        f"Modele : {description_modele}",
        fontsize=13,
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, fontsize=10, frameon=False)

    os.makedirs(os.path.dirname(os.path.abspath(chemin_png)), exist_ok=True)
    fig.savefig(chemin_png, dpi=150, bbox_inches="tight")  # l'extension .png choisit le format
    return chemin_png


# ---------------------------------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------------------------------

def mesurer_benchmark(artefact, chemin_artefact, test_brut, rul_test, repetitions=REPETITIONS_DEFAUT,
                      cible_ms=CIBLE_MS_DEFAUT, moteurs=None, afficher_progression=False):
    """
    Mesure les temps de reponse du prototype (exigence non fonctionnelle), sur la machine courante.

    Protocole :
    1. un passage d'echauffement non compte (premier appel plus lent : caches, allocations) ;
    2. pour chaque repetition et chaque moteur : preparation des donnees brutes du moteur + prediction
       de tous ses cycles, chronometree avec time.perf_counter ;
    3. temps de la flotte entiere (preparation + prediction des 100 moteurs au dernier cycle) ;
    4. taille du fichier artefact et temps de chargement depuis le disque.
    Les temps dependent de la machine (processeur, charge) : ils ne valent que pour celle-ci.

    Parameters
    ----------
    artefact : dict
        L'artefact charge.
    chemin_artefact : str
        Le fichier de l'artefact (pour sa taille et son temps de chargement).
    test_brut : pandas.DataFrame
        Les donnees test brutes.
    rul_test : numpy.ndarray
        La RUL vraie au dernier cycle de chaque moteur.
    repetitions : int, optionnel
        Nombre de passages sur les moteurs (5 par defaut).
    cible_ms : float, optionnel
        Temps cible pour UN moteur, en ms (200 par defaut) ; la cible est atteinte si p95 <= cible.
    moteurs : list de int, optionnel
        Les moteurs a mesurer (par defaut : tous les moteurs du test ; les tests en prennent moins).
    afficher_progression : bool, optionnel
        Vrai pour afficher l'avancement (utile en demo : on voit que le calcul tourne).

    Returns
    -------
    dict
        "temps_moteur_ms" (liste), "nb_mesures", "p50_ms", "p95_ms", "max_ms", "moyenne_ms",
        "temps_flotte_s" (mediane), "temps_flotte_max_s", "nb_mesures_flotte", "taille_ko",
        "temps_chargement_ms", "cible_ms", "cible_atteinte" (bool).
    """
    if moteurs is None:  # par defaut : tous les moteurs presents dans le test
        moteurs = []
        for moteur_id in np.sort(test_brut["unit_number"].unique()):
            moteurs.append(int(moteur_id))

    # les lignes brutes de chaque moteur sont extraites AVANT le chronometre (simule l'arrivee des mesures)
    lignes_par_moteur = {}
    for moteur_id in moteurs:
        lignes_par_moteur[moteur_id] = test_brut[test_brut["unit_number"] == moteur_id].copy()

    modele = artefact["modele"]
    feature_cols = artefact["feature_cols"]

    # 1. echauffement (non compte)
    premieres_lignes = lignes_par_moteur[moteurs[0]]
    predire_rul(modele, preparer_donnees_brutes(premieres_lignes, artefact)[feature_cols])

    # 2. temps pour UN moteur : preparation + prediction, moteur par moteur, repetition par repetition
    temps_moteur_ms = []
    for numero_repetition in range(repetitions):
        debut_repetition = time.perf_counter()
        for moteur_id in moteurs:
            debut = time.perf_counter()
            donnees = preparer_donnees_brutes(lignes_par_moteur[moteur_id], artefact)
            predire_rul(modele, donnees[feature_cols])
            temps_moteur_ms.append((time.perf_counter() - debut) * 1000.0)
        if afficher_progression:
            duree_repetition = time.perf_counter() - debut_repetition
            print(
                f"    repetition {numero_repetition + 1}/{repetitions} : "
                f"{len(moteurs)} moteurs mesures en {duree_repetition:.1f} s"
            )

    # 3. temps pour la flotte entiere (au plus 3 mesures : chacune prend environ une seconde)
    nb_mesures_flotte = min(repetitions, 3)
    temps_flotte_s = []
    for numero_mesure in range(nb_mesures_flotte):
        resultat_flotte = predire_flotte(artefact, test_brut, rul_test)
        temps_flotte_s.append(resultat_flotte["temps_s"])

    # 4. taille du fichier et temps de chargement depuis le disque
    taille_ko = os.path.getsize(chemin_artefact) / 1024.0
    debut = time.perf_counter()
    charger_artefact(chemin_artefact)
    temps_chargement_ms = (time.perf_counter() - debut) * 1000.0

    tableau_temps = np.asarray(temps_moteur_ms)
    resultats = {}
    resultats["temps_moteur_ms"] = temps_moteur_ms
    resultats["nb_mesures"] = int(len(temps_moteur_ms))
    resultats["p50_ms"] = float(np.percentile(tableau_temps, 50))
    resultats["p95_ms"] = float(np.percentile(tableau_temps, 95))
    resultats["max_ms"] = float(np.max(tableau_temps))
    resultats["moyenne_ms"] = float(np.mean(tableau_temps))
    resultats["temps_flotte_s"] = float(np.median(temps_flotte_s))
    resultats["temps_flotte_max_s"] = float(np.max(temps_flotte_s))
    resultats["nb_mesures_flotte"] = int(nb_mesures_flotte)
    resultats["taille_ko"] = float(taille_ko)
    resultats["temps_chargement_ms"] = float(temps_chargement_ms)
    resultats["cible_ms"] = float(cible_ms)
    resultats["cible_atteinte"] = bool(resultats["p95_ms"] <= cible_ms)
    return resultats


# ---------------------------------------------------------------------------------------------------
# Commandes
# ---------------------------------------------------------------------------------------------------

def commande_entrainer(arguments):
    """
    Commande `entrainer` : entraine le modele retenu et sauvegarde l'artefact.

    Parameters
    ----------
    arguments : argparse.Namespace
        Les arguments lus (config, artefact).

    Returns
    -------
    int
        0 si tout s'est bien passe (code de sortie du programme).
    """
    lancer_entrainement(arguments.config, arguments.artefact)
    return 0


def commande_moteur(arguments):
    """
    Commande `moteur` : prediction cycle par cycle d'un moteur test, resume et figure PNG.

    Parameters
    ----------
    arguments : argparse.Namespace
        Les arguments lus (id, seuil_alerte, sortie, artefact, config).

    Returns
    -------
    int
        0 si tout s'est bien passe.
    """
    verifier_id_moteur(arguments.id)  # verification AVANT tout calcul (et avant un eventuel entrainement)
    seuil = arguments.seuil_alerte

    artefact = charger_ou_entrainer(arguments.artefact, arguments.config)
    test_brut, rul_test = charger_donnees_test()

    resultat = predire_moteur(artefact, test_brut, rul_test, arguments.id)
    resultat["premier_cycle_alerte"] = premier_cycle_en_alerte(resultat["trajectoire"], seuil)
    alerte, statut = classer_alerte(resultat["rul_predite_dernier"], resultat["rul_vraie_dernier"], seuil)

    chemin_png = os.path.join(arguments.sortie, f"moteur_{arguments.id}.png")
    tracer_moteur(resultat, seuil, statut, artefact["description_modele"], chemin_png)

    # --- affichage ---
    afficher_titre(f"AERO PREDICT - Moteur test {arguments.id} (jeu de test FD001)")
    afficher_ligne("Modele utilise", artefact["description_modele"])
    afficher_ligne("Entraine le", artefact["date_entrainement"])
    afficher_ligne("Cycles observes", f"{resultat['nb_cycles']} cycles (prediction faite a chaque cycle)")

    afficher_section("Au dernier cycle observe")
    afficher_ligne("RUL predite", f"{resultat['rul_predite_dernier']:.1f} cycles")
    afficher_ligne("RUL vraie (RUL_FD001.txt)", f"{resultat['rul_vraie_dernier']} cycles")
    afficher_ligne(
        "Ecart (predite - vraie)",
        f"{formater_ecart(resultat['ecart_dernier'])} cycles -> {nature_ecart(resultat['ecart_dernier'])}",
    )

    afficher_section(f"Decision de maintenance (seuil d'alerte : {seuil:g} cycles)")
    if alerte:
        afficher_ligne("Decision du modele", f"ALERTE (RUL predite <= {seuil:g}) -> maintenance a planifier")
    else:
        afficher_ligne("Decision du modele", f"pas d'alerte (RUL predite > {seuil:g})")
    afficher_ligne("Verification (RUL vraie)", statut)
    afficher_suite("-> " + EXPLICATIONS_STATUT[statut])
    if resultat["premier_cycle_alerte"] is not None:
        cycle_alerte, rul_vraie_alerte = resultat["premier_cycle_alerte"]
        afficher_ligne("1re alerte du modele", f"cycle {cycle_alerte} (il restait alors {rul_vraie_alerte} cycles)")
    else:
        afficher_ligne("1re alerte du modele", f"aucune sur les {resultat['nb_cycles']} cycles observes")

    afficher_section("Performance")
    afficher_ligne(
        "Temps de calcul", f"{resultat['temps_ms']:.1f} ms (preparation + {resultat['nb_cycles']} predictions)"
    )
    afficher_ligne("Figure", chemin_lisible(chemin_png))
    print()
    return 0


def formater_ligne_tableau(rang, ligne):
    """
    Met en forme une ligne du tableau de flotte, colonnes alignees.

    Parameters
    ----------
    rang : int
        Le rang d'urgence (1 = le plus urgent).
    ligne : pandas.Series
        Une ligne du tableau de `predire_flotte`.

    Returns
    -------
    str
        La ligne formatee.
    """
    if ligne["alerte"]:  # alerte declenchee : en majuscules pour qu'elle saute aux yeux
        texte_alerte = "OUI"
    else:
        texte_alerte = "non"
    return (
        f"  {rang:>4}  {int(ligne['moteur']):>6}  {int(ligne['nb_cycles']):>6}  {ligne['rul_predite']:>11.1f}"
        f"  {int(ligne['rul_vraie']):>9}  {formater_ecart(ligne['ecart']):>7}  {texte_alerte:>6}  {ligne['statut']}"
    )


def commande_flotte(arguments):
    """
    Commande `flotte` : les 100 moteurs test au dernier cycle, tries par urgence, puis les metriques.

    Parameters
    ----------
    arguments : argparse.Namespace
        Les arguments lus (seuil_alerte, top, artefact, config).

    Returns
    -------
    int
        0 si tout s'est bien passe.
    """
    seuil = arguments.seuil_alerte
    artefact = charger_ou_entrainer(arguments.artefact, arguments.config)
    test_brut, rul_test = charger_donnees_test()

    resultat = predire_flotte(artefact, test_brut, rul_test, seuil_alerte=seuil)
    tableau = resultat["tableau"]
    resume = resumer_flotte(tableau, seuil_alerte=seuil)
    coherent, ecarts = controler_coherence_evaluation(artefact, resultat["X_dernier"], rul_test, tableau, resume, seuil)

    # tri par urgence : RUL predite croissante (a egalite, le plus petit numero de moteur d'abord)
    tableau_trie = tableau.sort_values(["rul_predite", "moteur"]).reset_index(drop=True)
    nb_affiches = min(arguments.top, len(tableau_trie))

    afficher_titre(f"AERO PREDICT - Flotte de {resume['nb_moteurs']} moteurs test (FD001, dernier cycle observe)")
    afficher_ligne("Modele utilise", artefact["description_modele"], largeur_libelle=16)
    afficher_ligne("Seuil d'alerte", f"{seuil:g} cycles (ALERTE si RUL predite <= seuil)", largeur_libelle=16)
    if nb_affiches < len(tableau_trie):
        afficher_ligne(
            "Affichage",
            f"les {nb_affiches} moteurs les plus urgents (--top {len(tableau_trie)} pour tout voir)",
            largeur_libelle=16,
        )
    else:
        afficher_ligne("Affichage", f"les {nb_affiches} moteurs", largeur_libelle=16)
    print("  (RUL et ecarts en cycles ; ecart = RUL predite - RUL vraie, positif = surestimation)")
    print()

    # en-tete du tableau, puis une ligne par moteur affiche
    print(
        f"  {'Rang':>4}  {'Moteur':>6}  {'Cycles':>6}  {'RUL predite':>11}  {'RUL vraie':>9}"
        f"  {'Ecart':>7}  {'Alerte':>6}  Statut"
    )
    print(f"  {'-' * 4}  {'-' * 6}  {'-' * 6}  {'-' * 11}  {'-' * 9}  {'-' * 7}  {'-' * 6}  {'-' * 15}")
    for position in range(nb_affiches):
        print(formater_ligne_tableau(position + 1, tableau_trie.iloc[position]))
    if nb_affiches < len(tableau_trie):
        print(f"  ... {len(tableau_trie) - nb_affiches} autres moteurs non affiches (option --top {len(tableau_trie)})")

    afficher_section(f"Resume sur les {resume['nb_moteurs']} moteurs")
    afficher_ligne("RMSE", f"{resume['rmse']:.2f} cycles")
    afficher_ligne("MAE", f"{resume['mae']:.2f} cycles")
    afficher_ligne("Score NASA (plus bas = mieux)", f"{resume['score_nasa']:.1f}")
    afficher_ligne(
        f"Surestimes de > {SEUIL_SURESTIMATION} cycles", f"{resume['nb_surestimes']} moteurs (le cas dangereux)"
    )
    afficher_ligne("Moteurs vraiment sous le seuil", f"{resume['nb_sous_seuil']} (RUL vraie <= {seuil:g} cycles)")
    afficher_ligne(
        "Alertes emises",
        f"{resume['alertes_emises']} (correcte(s) : {resume['alertes_correctes']}, "
        f"fausse(s) : {resume['fausses_alertes']})",
    )
    afficher_ligne("Alertes manquees", f"{resume['alertes_manquees']} (moteur proche de la panne, sans alerte)")
    afficher_ligne("Fausses alertes", f"{resume['fausses_alertes']} (maintenance inutile, sans risque)")
    afficher_ligne(
        "Temps total",
        f"{resultat['temps_s'] * 1000.0:.0f} ms (preparation + prediction des {resume['nb_moteurs']} moteurs)",
    )
    if coherent:
        afficher_ligne("Controle src/evaluation.py", "OK (memes chiffres que evaluer_sur_test)")
    else:
        afficher_ligne("Controle src/evaluation.py", "ECART : " + " ; ".join(ecarts))
    print()
    if not coherent:  # un desaccord avec src/evaluation.py est un vrai probleme : code de sortie 1
        return 1
    return 0


def commande_benchmark(arguments):
    """
    Commande `benchmark` : temps de reponse, taille du modele et verdict par rapport a la cible.

    Parameters
    ----------
    arguments : argparse.Namespace
        Les arguments lus (repetitions, cible_ms, artefact, config).

    Returns
    -------
    int
        0 si tout s'est bien passe (le verdict n'influe pas sur le code de sortie).
    """
    artefact = charger_ou_entrainer(arguments.artefact, arguments.config)
    test_brut, rul_test = charger_donnees_test()
    nb_moteurs = int(test_brut["unit_number"].nunique())

    afficher_titre("AERO PREDICT - Benchmark des temps de reponse (exigence non fonctionnelle)")
    afficher_ligne(
        "Machine",
        f"{platform.system()} {platform.machine()}, {os.cpu_count()} coeurs, Python {platform.python_version()}",
    )
    afficher_ligne("Modele", artefact["description_modele"])
    afficher_ligne(
        "Protocole",
        f"{nb_moteurs} moteurs x {arguments.repetitions} repetitions = {nb_moteurs * arguments.repetitions} mesures",
    )
    print("  (chaque mesure = preparation des donnees brutes d'un moteur + prediction de tous ses cycles ;")
    print("   1 passage d'echauffement non compte ; temps valables pour CETTE machine uniquement)")
    print("  ... mesures en cours ...")

    resultats = mesurer_benchmark(
        artefact, arguments.artefact, test_brut, rul_test,
        repetitions=arguments.repetitions, cible_ms=arguments.cible_ms, afficher_progression=True,
    )

    afficher_section("Temps pour UN moteur (preparation + prediction)")
    afficher_ligne("p50 (mediane)", f"{resultats['p50_ms']:.1f} ms")
    afficher_ligne("p95", f"{resultats['p95_ms']:.1f} ms (95 % des mesures sont plus rapides)")
    afficher_ligne("max", f"{resultats['max_ms']:.1f} ms")
    afficher_ligne("moyenne", f"{resultats['moyenne_ms']:.1f} ms")

    afficher_section("Flotte entiere et modele")
    afficher_ligne(
        f"Flotte de {nb_moteurs} moteurs",
        f"{resultats['temps_flotte_s'] * 1000.0:.0f} ms (mediane de {resultats['nb_mesures_flotte']} mesures, "
        f"max {resultats['temps_flotte_max_s'] * 1000.0:.0f} ms)",
    )
    afficher_ligne(
        "Taille du fichier modele", f"{resultats['taille_ko']:.0f} Ko ({chemin_lisible(arguments.artefact)})"
    )
    afficher_ligne("Chargement du modele (disque)", f"{resultats['temps_chargement_ms']:.0f} ms")

    afficher_section(f"Verdict (cible : p95 <= {resultats['cible_ms']:g} ms pour un moteur)")
    if resultats["cible_atteinte"]:
        marge = resultats["cible_ms"] / max(resultats["p95_ms"], 1e-9)  # combien de fois plus rapide que la cible
        print(
            f"  CIBLE ATTEINTE : p95 = {resultats['p95_ms']:.1f} ms <= {resultats['cible_ms']:g} ms "
            f"(environ {marge:.1f} fois plus rapide)"
        )
    else:
        print(f"  CIBLE NON ATTEINTE : p95 = {resultats['p95_ms']:.1f} ms > {resultats['cible_ms']:g} ms")
    if resultats["max_ms"] > resultats["cible_ms"]:
        print(f"  (note : la pire mesure, {resultats['max_ms']:.1f} ms, depasse la cible)")
    print()
    return 0


# ---------------------------------------------------------------------------------------------------
# Lecture des arguments (argparse, aide en francais)
# ---------------------------------------------------------------------------------------------------

class FormateurAide(argparse.RawDescriptionHelpFormatter):
    """
    Formateur d'aide argparse : garde les retours a la ligne des descriptions et ecrit "utilisation :"
    au lieu de "usage:".
    """

    def add_usage(self, usage, actions, groups, prefix=None):
        """
        Ajoute la ligne d'utilisation, avec un prefixe en francais.

        Parameters
        ----------
        usage : str ou None
            Texte d'utilisation impose (None = calcule par argparse).
        actions : list
            Les arguments du parseur.
        groups : list
            Les groupes d'arguments mutuellement exclusifs.
        prefix : str, optionnel
            Le prefixe ("utilisation : " par defaut).

        Returns
        -------
        None
        """
        if prefix is None:
            prefix = "utilisation : "
        return super().add_usage(usage, actions, groups, prefix)


class ParseurFrancais(argparse.ArgumentParser):
    """
    Parseur argparse dont les erreurs commencent par "ERREUR" (en francais) au lieu de "error:".
    """

    def error(self, message):
        """
        Affiche la ligne d'utilisation et le message d'erreur, puis quitte avec le code 2.

        Parameters
        ----------
        message : str
            Le message d'erreur produit par argparse.

        Returns
        -------
        None
            Ne revient jamais : quitte le programme.
        """
        self.print_usage(sys.stderr)
        self.exit(2, f"ERREUR ({self.prog}) : {message}\n")


def type_entier(texte):
    """
    Convertit un argument en entier, avec un message d'erreur en francais.

    Parameters
    ----------
    texte : str
        La valeur tapee par l'utilisateur.

    Returns
    -------
    int
        La valeur convertie.

    Raises
    ------
    argparse.ArgumentTypeError
        Si le texte n'est pas un entier.
    """
    try:
        return int(texte)
    except ValueError:
        raise argparse.ArgumentTypeError(f"nombre entier attendu, recu : {texte!r}")


def type_entier_positif(texte):
    """
    Convertit un argument en entier >= 1, avec un message d'erreur en francais.

    Parameters
    ----------
    texte : str
        La valeur tapee par l'utilisateur.

    Returns
    -------
    int
        La valeur convertie.

    Raises
    ------
    argparse.ArgumentTypeError
        Si le texte n'est pas un entier >= 1.
    """
    valeur = type_entier(texte)
    if valeur < 1:
        raise argparse.ArgumentTypeError(f"entier >= 1 attendu, recu : {valeur}")
    return valeur


def type_reel_positif(texte):
    """
    Convertit un argument en nombre reel >= 0, avec un message d'erreur en francais.

    Parameters
    ----------
    texte : str
        La valeur tapee par l'utilisateur.

    Returns
    -------
    float
        La valeur convertie.

    Raises
    ------
    argparse.ArgumentTypeError
        Si le texte n'est pas un nombre >= 0.
    """
    try:
        valeur = float(texte)
    except ValueError:
        raise argparse.ArgumentTypeError(f"nombre attendu, recu : {texte!r}")
    if not np.isfinite(valeur) or valeur < 0:
        raise argparse.ArgumentTypeError(f"nombre >= 0 attendu, recu : {texte}")
    return valeur


def type_reel_strictement_positif(texte):
    """
    Convertit un argument en nombre reel > 0, avec un message d'erreur en francais.

    Parameters
    ----------
    texte : str
        La valeur tapee par l'utilisateur.

    Returns
    -------
    float
        La valeur convertie.

    Raises
    ------
    argparse.ArgumentTypeError
        Si le texte n'est pas un nombre > 0.
    """
    valeur = type_reel_positif(texte)
    if valeur == 0:
        raise argparse.ArgumentTypeError("nombre > 0 attendu, recu : 0")
    return valeur


def ajouter_options_communes(sous_parseur):
    """
    Ajoute a une sous-commande les options communes : aide, artefact et configuration.

    Parameters
    ----------
    sous_parseur : argparse.ArgumentParser
        Le parseur de la sous-commande.

    Returns
    -------
    None
    """
    sous_parseur.add_argument("-h", "--help", action="help", help="affiche cette aide et quitte")
    sous_parseur.add_argument(
        "--artefact", default=CHEMIN_ARTEFACT_DEFAUT, metavar="CHEMIN",
        help=f"fichier du modele entraine (defaut : {chemin_lisible(CHEMIN_ARTEFACT_DEFAUT)})",
    )
    sous_parseur.add_argument(
        "--config", default=CHEMIN_CONFIG_RETENUE, metavar="CHEMIN",
        help=(
            f"configuration JSON du modele retenu (defaut : {chemin_lisible(CHEMIN_CONFIG_RETENUE)} ; "
            "si le fichier est absent : XGBoost Phase 3)"
        ),
    )


def construire_parseur():
    """
    Construit le parseur d'arguments de la demo, avec ses 4 sous-commandes et une aide en francais.

    Returns
    -------
    ParseurFrancais
        Le parseur pret a lire la ligne de commande.
    """
    parseur = ParseurFrancais(
        prog="python -m src.demo",
        description=(
            "Aero Predict - demo du prototype de prediction de la RUL (duree de vie restante, en cycles)\n"
            "de turboreacteurs, jeu NASA C-MAPSS FD001."
        ),
        epilog=(
            "exemples :\n"
            "  python -m src.demo entrainer\n"
            "  python -m src.demo moteur --id 34\n"
            "  python -m src.demo flotte --top 20\n"
            "  python -m src.demo benchmark --repetitions 5 --cible-ms 200\n"
            "\n"
            "Aide d'une commande : python -m src.demo <commande> --help"
        ),
        formatter_class=FormateurAide,
        add_help=False,
    )
    parseur.add_argument("-h", "--help", action="help", help="affiche cette aide et quitte")
    sous_parseurs = parseur.add_subparsers(title="commandes", dest="commande", metavar="<commande>")

    # --- entrainer ---
    parseur_entrainer = sous_parseurs.add_parser(
        "entrainer",
        help="entraine le modele retenu sur les 100 moteurs d'entrainement et le sauvegarde",
        description=(
            "Entraine le modele decrit par la configuration retenue (Phase 4) sur les 100 moteurs\n"
            "d'entrainement, puis sauvegarde l'artefact (modele + preparation des donnees apprise sur le train)."
        ),
        formatter_class=FormateurAide,
        add_help=False,
    )
    ajouter_options_communes(parseur_entrainer)
    parseur_entrainer.set_defaults(fonction=commande_entrainer)

    # --- moteur ---
    parseur_moteur = sous_parseurs.add_parser(
        "moteur",
        help="predit la RUL d'un moteur test a chaque cycle, avec figure",
        description=(
            "Predit la RUL du moteur test N a chaque cycle observe, compare a la vraie RUL, classe la\n"
            "decision d'alerte et ecrit une figure PNG. Entraine automatiquement le modele s'il manque."
        ),
        formatter_class=FormateurAide,
        add_help=False,
    )
    parseur_moteur.add_argument(
        "--id", type=type_entier, required=True, metavar="N", help="numero du moteur test (1 a 100)"
    )
    parseur_moteur.add_argument(
        "--seuil-alerte", type=type_reel_positif, default=SEUIL_ALERTE_DEFAUT, metavar="CYCLES",
        help=f"ALERTE si RUL predite <= ce seuil (defaut : {SEUIL_ALERTE_DEFAUT:g} cycles)",
    )
    parseur_moteur.add_argument(
        "--sortie", default=DOSSIER_SORTIE_DEFAUT, metavar="DOSSIER",
        help=f"dossier de la figure moteur_<N>.png (defaut : {chemin_lisible(DOSSIER_SORTIE_DEFAUT)})",
    )
    ajouter_options_communes(parseur_moteur)
    parseur_moteur.set_defaults(fonction=commande_moteur)

    # --- flotte ---
    parseur_flotte = sous_parseurs.add_parser(
        "flotte",
        help="predit les 100 moteurs test et les trie par urgence",
        description=(
            "Predit la RUL des 100 moteurs test au dernier cycle observe, les trie par RUL predite\n"
            "croissante (les plus urgents en premier) et resume les metriques (RMSE, score NASA, alertes)."
        ),
        formatter_class=FormateurAide,
        add_help=False,
    )
    parseur_flotte.add_argument(
        "--seuil-alerte", type=type_reel_positif, default=SEUIL_ALERTE_DEFAUT, metavar="CYCLES",
        help=f"ALERTE si RUL predite <= ce seuil (defaut : {SEUIL_ALERTE_DEFAUT:g} cycles)",
    )
    parseur_flotte.add_argument(
        "--top", type=type_entier_positif, default=TOP_DEFAUT, metavar="K",
        help=f"nombre de moteurs affiches, les plus urgents d'abord (defaut : {TOP_DEFAUT} ; 100 = tous)",
    )
    ajouter_options_communes(parseur_flotte)
    parseur_flotte.set_defaults(fonction=commande_flotte)

    # --- benchmark ---
    parseur_benchmark = sous_parseurs.add_parser(
        "benchmark",
        help="mesure les temps de reponse et la taille du modele (exigence non fonctionnelle)",
        description=(
            "Mesure, SUR LA MACHINE COURANTE, le temps de preparation + prediction d'un moteur (p50, p95,\n"
            "max sur les 100 moteurs x repetitions), le temps de la flotte entiere, la taille du fichier\n"
            "modele et son temps de chargement, puis donne un verdict par rapport a la cible (sur le p95).\n"
            "Les temps dependent du processeur et de sa charge : ils ne valent que pour cette machine."
        ),
        formatter_class=FormateurAide,
        add_help=False,
    )
    parseur_benchmark.add_argument(
        "--repetitions", type=type_entier_positif, default=REPETITIONS_DEFAUT, metavar="R",
        help=f"nombre de passages sur les 100 moteurs (defaut : {REPETITIONS_DEFAUT})",
    )
    parseur_benchmark.add_argument(
        "--cible-ms", type=type_reel_strictement_positif, default=CIBLE_MS_DEFAUT, metavar="MS",
        help=f"temps cible pour un moteur, compare au p95 (defaut : {CIBLE_MS_DEFAUT:g} ms)",
    )
    ajouter_options_communes(parseur_benchmark)
    parseur_benchmark.set_defaults(fonction=commande_benchmark)

    return parseur


def main(argv=None):
    """
    Point d'entree de la demo : lit la ligne de commande et lance la commande demandee.

    Parameters
    ----------
    argv : list de str, optionnel
        Les arguments (sans le nom du programme). Par defaut : ceux de la ligne de commande. Les
        tests passent ici leur propre liste, par exemple ["moteur", "--id", "1", "--artefact", ...].

    Returns
    -------
    int
        Le code de sortie : 0 = succes, 1 = erreur (message en francais affiche), 2 = arguments invalides.
    """
    parseur = construire_parseur()
    arguments = parseur.parse_args(argv)

    if arguments.commande is None:  # aucune commande : on affiche l'aide
        parseur.print_help()
        return 2

    try:
        return arguments.fonction(arguments)
    except ErreurDemo as erreur:  # erreur "utilisateur" : une phrase claire, pas de traceback
        print(f"\nERREUR : {erreur}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
