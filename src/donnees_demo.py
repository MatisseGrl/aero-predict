"""
Couche d'acces aux donnees du tableau de bord de demo (app/tableau_de_bord.py).

Pourquoi ce module existe : l'application Streamlit ne doit contenir que de l'affichage. Tout ce qui
lit un fichier, recalcule une metrique ou fait une prediction est ici, dans des fonctions simples, sans
Streamlit ni Plotly, pour pouvoir les tester avec pytest (tests/test_donnees_demo.py) et les reutiliser
ailleurs. Rien n'est recode : on s'appuie sur le code deja valide du depot.

- predictions de tous les modeles : contrat d'echange `src/contrat_predictions.py` ;
- decision d'alerte : `classer_alerte` de `src/demo.py` (meme regle que la demo en ligne de commande) ;
- metriques de flotte : `resumer_flotte` de `src/demo.py` et `metriques_dernier_cycle` du contrat ;
- resultats de la Phase 4 : fichiers de `docs/resultats/` et regle de choix de `src/optimisation.py` ;
- inference en direct (page "Et si...") : `preparer_donnees_brutes` et `predire_rul` de `src/demo.py` ;
- donnees brutes des sous-jeux : `src/sous_jeux.charger_sous_jeu` s'il existe, sinon lecture directe.

Aucun chiffre affiche par l'application n'est ecrit en dur : ils sont tous lus dans ces fichiers ou
recalcules par ces fonctions.
"""

import datetime
import inspect
import json
import math
import os
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from src.config_modele import CHEMIN_CONFIG_RETENUE, RACINE_DEPOT, convertir_en_types_python
from src.contrat_predictions import (
    MODELES_CONNUS,
    PLAFOND_RUL,
    SOUS_JEUX,
    dossier_par_defaut,
    extraire_dernier_cycle,
    lire_metadonnees,
    lire_predictions,
    lister_predictions,
    verifier_coherence,
)
from src.demo import (
    CHEMIN_ARTEFACT_DEFAUT,
    CIBLE_MS_DEFAUT,
    SEUIL_ALERTE_DEFAUT,
    STATUT_ALERTE_CORRECTE,
    STATUT_ALERTE_MANQUEE,
    STATUT_FAUSSE_ALERTE,
    STATUT_RIEN_A_SIGNALER,
    charger_artefact,
    charger_donnees_test,
    charger_ou_entrainer,
    classer_alerte,
    mesurer_benchmark,
    predire_rul,
    preparer_donnees_brutes,
    resumer_flotte,
)
from src.preprocessing import COLUMNS, DATA_DIR, load_fd001

# ---------------------------------------------------------------------------------------------------
# Constantes : emplacements des fichiers lus par l'application
# ---------------------------------------------------------------------------------------------------

# dossier des resultats chiffres des phases (CSV, JSON versionnes avec le code)
DOSSIER_RESULTATS = os.path.join(RACINE_DEPOT, "docs", "resultats")

# fichier d'etat du projet : on y lit les limites connues (jamais inventees par l'application)
CHEMIN_STATUS = os.path.join(RACINE_DEPOT, "docs", "STATUS.md")

# description officielle des sous-jeux fournie par la NASA avec les donnees
CHEMIN_README_DONNEES = os.path.join(DATA_DIR, "readme.txt")

# informations de reproductibilite ecrites par scripts/preparer_demo.py (nombre de tests, date, commit)
CHEMIN_INFOS_REPRODUCTIBILITE = os.path.join(RACINE_DEPOT, "sorties", "demo", "infos_reproductibilite.json")

# variable d'environnement qui remplace l'artefact du modele retenu (utile pour les tests automatiques)
VARIABLE_ENV_ARTEFACT = "AERO_PREDICT_ARTEFACT"

# les deux tableaux optionnels produits par les autres scripts (affiches s'ils existent)
FICHIER_DEEP_LEARNING = "03b_deep_learning.csv"
FICHIER_SOUS_JEUX_ARBRES = "04b_sous_jeux_arbres.csv"

# seuil de significativite des tests apparies (convention usuelle, la meme que dans les notebooks)
SEUIL_SIGNIFICATIVITE = 0.05

# ordre d'affichage des statuts : du plus urgent (le cas dangereux) au plus calme
ORDRE_STATUTS = [STATUT_ALERTE_MANQUEE, STATUT_ALERTE_CORRECTE, STATUT_FAUSSE_ALERTE, STATUT_RIEN_A_SIGNALER]

# libelles affiches a l'ecran (avec accents) pour chaque statut de src/demo.py
LIBELLES_STATUTS = {
    STATUT_ALERTE_MANQUEE: "ALERTE MANQUÉE",
    STATUT_ALERTE_CORRECTE: "Alerte correcte",
    STATUT_FAUSSE_ALERTE: "Fausse alerte",
    STATUT_RIEN_A_SIGNALER: "Rien à signaler",
}

# explication de chaque statut, pour l'affichage (meme sens que EXPLICATIONS_STATUT de src/demo.py)
EXPLICATIONS_STATUTS = {
    STATUT_ALERTE_MANQUEE: "Le moteur est proche de la panne mais le modèle ne déclenche pas l'alerte : le cas dangereux.",
    STATUT_ALERTE_CORRECTE: "Le moteur est vraiment proche de la panne et le modèle déclenche l'alerte.",
    STATUT_FAUSSE_ALERTE: "Le modèle alerte alors que le moteur a encore de la marge : maintenance inutile, sans risque.",
    STATUT_RIEN_A_SIGNALER: "Le moteur a encore de la marge et le modèle ne déclenche pas d'alerte.",
}

# capteurs proposes par defaut dans les panneaux de capteurs bruts (sensor_11 est celui des figures
# des Phases 1 et 2) ; s'ils n'existent pas ou sont constants, on prend les premiers capteurs variables
CAPTEURS_PAR_DEFAUT = ["sensor_11", "sensor_4", "sensor_7"]

# les deux types de derive proposes par la page "Et si..." : cle -> libelle affiche
TYPES_DERIVE = {
    "pourcentage": "Pourcentage (valeur multipliée)",
    "decalage": "Décalage (valeur ajoutée, unités du capteur)",
}

# mots qui signalent une limite ou un resultat non concluant dans docs/STATUS.md (recherche sans
# tenir compte des majuscules) : l'application n'affiche que des phrases qui contiennent l'un d'eux
MOTS_CLES_LIMITES = [
    "non significati",
    "non concluant",
    "pas encore",
    "non valid",
    "moins bonne",
    "sous charge",
    "non rédigé",
    "à comparer",
    "autre environnement",
    "ne valent que",
]

# sections de docs/STATUS.md dans lesquelles on cherche ces limites
SECTIONS_STATUS = ["Phase actuelle", "Dernière modification"]

# Tableau 2 de Saxena et al. (2008), data/Damage Propagation Modeling.pdf, page 3 : les 21 grandeurs
# "available to participants as sensor data" (symbole, description originale en anglais, unite).
# ATTENTION : ni data/readme.txt ni l'article ne disent explicitement que la colonne sensor_N des
# fichiers correspond a la N-ieme ligne de ce tableau. L'application garde donc le nom sensor_N et
# n'affiche ce tableau qu'a titre d'information, avec cet avertissement.
GRANDEURS_TABLEAU_2_SAXENA = [
    ("T2", "Total temperature at fan inlet", "°R"),
    ("T24", "Total temperature at LPC outlet", "°R"),
    ("T30", "Total temperature at HPC outlet", "°R"),
    ("T50", "Total temperature at LPT outlet", "°R"),
    ("P2", "Pressure at fan inlet", "psia"),
    ("P15", "Total pressure in bypass-duct", "psia"),
    ("P30", "Total pressure at HPC outlet", "psia"),
    ("Nf", "Physical fan speed", "rpm"),
    ("Nc", "Physical core speed", "rpm"),
    ("epr", "Engine pressure ratio (P50/P2)", "--"),
    ("Ps30", "Static pressure at HPC outlet", "psia"),
    ("phi", "Ratio of fuel flow to Ps30", "pps/psi"),
    ("NRf", "Corrected fan speed", "rpm"),
    ("NRc", "Corrected core speed", "rpm"),
    ("BPR", "Bypass Ratio", "--"),
    ("farB", "Burner fuel-air ratio", "--"),
    ("htBleed", "Bleed Enthalpy", "--"),
    ("Nf_dmd", "Demanded fan speed", "rpm"),
    ("PCNfR_dmd", "Demanded corrected fan speed", "rpm"),
    ("W31", "HPT coolant bleed", "lbm/s"),
    ("W32", "LPT coolant bleed", "lbm/s"),
]


# ---------------------------------------------------------------------------------------------------
# Emplacements et signatures de fichiers (pour que le cache de l'application voie les nouveaux fichiers)
# ---------------------------------------------------------------------------------------------------

def dossier_predictions():
    """
    Renvoie le dossier des fichiers de predictions utilise par l'application.

    C'est celui du contrat (`dossier_par_defaut`) : <racine>/sorties/predictions, ou le dossier donne
    par la variable d'environnement AERO_PREDICT_DOSSIER_PREDICTIONS si elle est definie.

    Returns
    -------
    str
        Le chemin absolu du dossier.
    """
    return dossier_par_defaut()


def chemin_artefact_demo():
    """
    Renvoie le fichier de l'artefact du modele retenu (modele entraine + preparation apprise sur le train).

    Par defaut, celui de la demo en ligne de commande (models/modele_retenu.joblib) ; la variable
    d'environnement AERO_PREDICT_ARTEFACT le remplace si elle est definie (tests automatiques).

    Returns
    -------
    str
        Le chemin absolu de l'artefact.
    """
    valeur_env = os.environ.get(VARIABLE_ENV_ARTEFACT, "")  # chaine vide si la variable n'existe pas
    if valeur_env.strip() != "":  # variable definie et non vide : elle l'emporte
        return os.path.abspath(valeur_env)
    return CHEMIN_ARTEFACT_DEFAUT


def signature_fichier(chemin):
    """
    Resume l'etat d'un fichier en un petit tuple (chemin, date de modification, taille).

    Sert de cle de cache : si le fichier est reecrit (nouvelle date ou nouvelle taille), la cle change
    et l'application relit le fichier au lieu de garder l'ancienne version en memoire.

    Parameters
    ----------
    chemin : str
        Le fichier.

    Returns
    -------
    tuple
        (chemin, date de modification en nanosecondes, taille en octets), ou (chemin, None, None)
        si le fichier n'existe pas.
    """
    if not os.path.isfile(chemin):
        return (chemin, None, None)
    informations = os.stat(chemin)
    return (chemin, informations.st_mtime_ns, informations.st_size)


def signature_dossier(dossier):
    """
    Resume l'etat de tous les fichiers d'un dossier (nom, date, taille), trie par nom.

    Parameters
    ----------
    dossier : str
        Le dossier (par exemple celui des predictions).

    Returns
    -------
    tuple
        Un tuple de signatures de fichiers ; vide si le dossier n'existe pas.
    """
    if not os.path.isdir(dossier):
        return tuple()
    signatures = []
    for nom in sorted(os.listdir(dossier)):  # tri : meme ordre a chaque appel
        chemin = os.path.join(dossier, nom)
        if os.path.isfile(chemin):
            signatures.append(signature_fichier(chemin))
    return tuple(signatures)


# ---------------------------------------------------------------------------------------------------
# Inventaire des fichiers de predictions
# ---------------------------------------------------------------------------------------------------

def inventaire_predictions(dossier=None):
    """
    Liste les couples (sous-jeu, modele) dont les fichiers de predictions sont presents et valides.

    Parameters
    ----------
    dossier : str ou None, optionnel
        Dossier des predictions ; None pour le dossier par defaut.

    Returns
    -------
    pandas.DataFrame
        Le resultat de `lister_predictions` (colonnes sous_jeu, modele, nom_affiche, famille,
        chemin_predictions, chemin_metadonnees), vide si aucun fichier.
    """
    return lister_predictions(dossier)


def sous_jeux_disponibles(inventaire):
    """
    Renvoie les sous-jeux qui ont au moins un fichier de predictions, dans l'ordre de SOUS_JEUX.

    Parameters
    ----------
    inventaire : pandas.DataFrame
        Le resultat de `inventaire_predictions`.

    Returns
    -------
    list de str
        Par exemple ["FD001", "FD003"].
    """
    presents = []
    valeurs = inventaire["sous_jeu"].tolist()
    for sous_jeu in SOUS_JEUX:  # ordre officiel FD001 -> FD004
        if sous_jeu in valeurs:
            presents.append(sous_jeu)
    return presents


def modeles_disponibles(inventaire, sous_jeu):
    """
    Renvoie les modeles disponibles pour un sous-jeu, dans l'ordre de MODELES_CONNUS.

    Parameters
    ----------
    inventaire : pandas.DataFrame
        Le resultat de `inventaire_predictions`.
    sous_jeu : str
        Le sous-jeu (ex. "FD001").

    Returns
    -------
    list de str
        Les cles de modeles (ex. ["xgboost_phase3", "xgboost_retenu"]).
    """
    lignes = inventaire[inventaire["sous_jeu"] == sous_jeu]
    valeurs = lignes["modele"].tolist()
    presents = []
    for modele in MODELES_CONNUS:  # ordre d'affichage du contrat
        if modele in valeurs:
            presents.append(modele)
    return presents


def nom_modele(modele):
    """
    Donne le nom lisible d'un modele (ex. "xgboost_retenu" -> "XGBoost retenu (Optuna)").

    Parameters
    ----------
    modele : str
        La cle du modele.

    Returns
    -------
    str
        Le nom affiche du contrat, ou la cle elle-meme si elle est inconnue.
    """
    return MODELES_CONNUS.get(modele, modele)


def charger_predictions(sous_jeu, modele, dossier=None):
    """
    Lit (et valide) un fichier de predictions et ses metadonnees, via le contrat.

    Parameters
    ----------
    sous_jeu : str
        Le sous-jeu (ex. "FD001").
    modele : str
        La cle du modele.
    dossier : str ou None, optionnel
        Dossier des predictions ; None pour le dossier par defaut.

    Returns
    -------
    tuple
        (DataFrame des predictions trie par moteur puis cycle, dictionnaire des metadonnees).
    """
    return lire_predictions(sous_jeu, modele, dossier)


# ---------------------------------------------------------------------------------------------------
# Un moteur : trajectoire, etat a un cycle donne, premiere alerte
# ---------------------------------------------------------------------------------------------------

def nombre_capteurs_bruts():
    """
    Compte les colonnes capteurs (sensor_N) des fichiers bruts, d'apres COLUMNS (src/preprocessing.py).

    Returns
    -------
    int
        Le nombre de capteurs bruts par cycle.
    """
    nombre = 0
    for colonne in COLUMNS:
        if colonne.startswith("sensor_"):
            nombre = nombre + 1
    return nombre


def constantes_score_nasa():
    """
    Lit, dans le code de `score_nasa` (src/evaluation.py), les constantes de la formule officielle.

    Returns
    -------
    dict
        "a_sous" (sous-estimation) et "a_sur" (surestimation), valeurs par defaut de la fonction.
    """
    from src.evaluation import score_nasa

    parametres = inspect.signature(score_nasa).parameters  # introspection : rien n'est recopie a la main
    constantes = {}
    constantes["a_sous"] = parametres["a_sous"].default
    constantes["a_sur"] = parametres["a_sur"].default
    return constantes


def moteur_le_plus_proche_de_la_panne(df):
    """
    Trouve le moteur test dont la RUL vraie au dernier cycle observe est la plus petite.

    C'est le moteur propose par defaut dans la page "Moteur en direct" : son rejeu traverse la zone
    d'alerte, ce qui montre la decision du modele (a egalite, le plus petit numero).

    Parameters
    ----------
    df : pandas.DataFrame
        Un fichier de predictions.

    Returns
    -------
    int
        Le numero du moteur.
    """
    derniers = extraire_dernier_cycle(df)
    tries = derniers.sort_values(["rul_vraie", "unit_number"])
    return int(tries["unit_number"].iloc[0])


def lister_moteurs(df):
    """
    Renvoie les numeros de moteurs presents dans un fichier de predictions, tries.

    Parameters
    ----------
    df : pandas.DataFrame
        Un fichier de predictions.

    Returns
    -------
    list de int
        Les numeros de moteurs, du plus petit au plus grand.
    """
    moteurs = []
    for moteur_id in np.sort(df["unit_number"].unique()):
        moteurs.append(int(moteur_id))
    return moteurs


def extraire_trajectoire(df, moteur_id):
    """
    Garde les lignes d'un seul moteur, triees par cycle.

    Parameters
    ----------
    df : pandas.DataFrame
        Un fichier de predictions (tous les moteurs).
    moteur_id : int
        Le numero du moteur test.

    Returns
    -------
    pandas.DataFrame
        Les lignes de ce moteur, triees par time_cycles, index remis a 0..n-1.

    Raises
    ------
    ValueError
        Si le moteur n'existe pas dans le fichier.
    """
    lignes = df[df["unit_number"] == moteur_id]
    if len(lignes) == 0:
        raise ValueError(f"Le moteur {moteur_id} n'existe pas dans ce fichier de predictions.")
    return lignes.sort_values("time_cycles").reset_index(drop=True)


def premier_cycle_alerte(trajectoire, seuil_alerte, cycle_max=None):
    """
    Cherche le premier cycle ou la RUL predite passe sous le seuil (le modele aurait alerte).

    Meme regle que `premier_cycle_en_alerte` de src/demo.py (RUL predite <= seuil), mais sur les noms
    de colonnes du contrat, et en ne regardant que les cycles <= cycle_max (le "present" du rejeu).

    Parameters
    ----------
    trajectoire : pandas.DataFrame
        Les lignes d'un moteur (time_cycles, rul_predite, rul_vraie), triees par cycle.
    seuil_alerte : float
        Le seuil d'alerte en cycles.
    cycle_max : int ou None, optionnel
        Dernier cycle a examiner ; None pour toute la trajectoire.

    Returns
    -------
    (int, float) ou None
        (cycle de la premiere alerte, RUL vraie a ce cycle), ou None si aucune alerte.
    """
    cycles = trajectoire["time_cycles"].to_numpy()
    predites = trajectoire["rul_predite"].to_numpy(dtype=float)
    vraies = trajectoire["rul_vraie"].to_numpy(dtype=float)
    for position in range(len(cycles)):  # du premier cycle observe au dernier
        if cycle_max is not None and cycles[position] > cycle_max:
            break  # on ne regarde pas le "futur" du rejeu
        if predites[position] <= seuil_alerte:
            return int(cycles[position]), float(vraies[position])
    return None


def etat_au_cycle(trajectoire, cycle, seuil_alerte=SEUIL_ALERTE_DEFAUT):
    """
    Decrit l'etat d'un moteur a un cycle donne : RUL predite, RUL vraie, ecart, decision d'alerte.

    Si le cycle demande n'est pas observe (trou dans les donnees), on prend le dernier cycle observe
    avant lui.

    Parameters
    ----------
    trajectoire : pandas.DataFrame
        Les lignes d'un moteur, triees par cycle (colonnes du contrat).
    cycle : int
        Le cycle "present" du rejeu.
    seuil_alerte : float, optionnel
        Le seuil d'alerte (30 cycles par defaut, hypothese de travail).

    Returns
    -------
    dict
        "cycle", "rul_predite", "rul_vraie", "rul_vraie_cappee", "ecart" (predite - vraie),
        "alerte" (bool), "statut" (voir `classer_alerte` de src/demo.py), "premiere_alerte"
        ((cycle, RUL vraie) ou None, jusqu'a ce cycle inclus).
    """
    avant = trajectoire[trajectoire["time_cycles"] <= cycle]
    if len(avant) == 0:  # cycle avant le premier cycle observe : on prend le premier
        avant = trajectoire.iloc[[0]]
    ligne = avant.iloc[-1]  # le cycle observe le plus recent

    rul_predite = float(ligne["rul_predite"])
    rul_vraie = float(ligne["rul_vraie"])
    alerte, statut = classer_alerte(rul_predite, rul_vraie, seuil_alerte)  # meme regle que la demo CLI

    etat = {}
    etat["cycle"] = int(ligne["time_cycles"])
    etat["rul_predite"] = rul_predite
    etat["rul_vraie"] = rul_vraie
    etat["rul_vraie_cappee"] = float(ligne["rul_vraie_cappee"])
    etat["ecart"] = rul_predite - rul_vraie
    etat["alerte"] = alerte
    etat["statut"] = statut
    etat["premiere_alerte"] = premier_cycle_alerte(trajectoire, seuil_alerte, cycle_max=etat["cycle"])
    return etat


# ---------------------------------------------------------------------------------------------------
# La flotte : tous les moteurs au dernier cycle observe
# ---------------------------------------------------------------------------------------------------

def tableau_flotte(df, seuil_alerte=SEUIL_ALERTE_DEFAUT):
    """
    Construit le tableau de la flotte : une ligne par moteur test, a son dernier cycle observe.

    Le tableau est trie par urgence : RUL predite croissante (le moteur qu'il faudrait arreter en
    premier selon le modele est en haut), comme la commande `flotte` de src/demo.py.

    Parameters
    ----------
    df : pandas.DataFrame
        Un fichier de predictions (tous les cycles de tous les moteurs).
    seuil_alerte : float, optionnel
        Le seuil d'alerte en cycles.

    Returns
    -------
    pandas.DataFrame
        Colonnes : moteur, nb_cycles, dernier_cycle, rul_predite, rul_vraie, ecart, alerte, statut,
        cycle_premiere_alerte (NaN si aucune), rul_vraie_premiere_alerte (NaN si aucune).
    """
    derniers = extraire_dernier_cycle(df)  # une ligne par moteur, triee par numero de moteur
    nb_cycles_par_moteur = df.groupby("unit_number").size()

    lignes = []
    for position in range(len(derniers)):
        moteur_id = int(derniers["unit_number"].iloc[position])
        rul_predite = float(derniers["rul_predite"].iloc[position])
        rul_vraie = float(derniers["rul_vraie"].iloc[position])
        alerte, statut = classer_alerte(rul_predite, rul_vraie, seuil_alerte)

        # premiere alerte de ce moteur sur toute sa trajectoire observee
        trajectoire = extraire_trajectoire(df, moteur_id)
        premiere = premier_cycle_alerte(trajectoire, seuil_alerte)

        ligne = {}
        ligne["moteur"] = moteur_id
        ligne["nb_cycles"] = int(nb_cycles_par_moteur.loc[moteur_id])
        ligne["dernier_cycle"] = int(derniers["time_cycles"].iloc[position])
        ligne["rul_predite"] = rul_predite
        ligne["rul_vraie"] = rul_vraie
        ligne["ecart"] = rul_predite - rul_vraie
        ligne["alerte"] = bool(alerte)
        ligne["statut"] = statut
        if premiere is None:
            ligne["cycle_premiere_alerte"] = np.nan
            ligne["rul_vraie_premiere_alerte"] = np.nan
        else:
            ligne["cycle_premiere_alerte"] = premiere[0]
            ligne["rul_vraie_premiere_alerte"] = premiere[1]
        lignes.append(ligne)

    tableau = pd.DataFrame(lignes)
    # tri par urgence : RUL predite croissante, puis numero de moteur (resultat deterministe)
    tableau = tableau.sort_values(["rul_predite", "moteur"]).reset_index(drop=True)
    return tableau


def resumer_tableau_flotte(tableau, seuil_alerte=SEUIL_ALERTE_DEFAUT):
    """
    Calcule les indicateurs de la flotte avec `resumer_flotte` de src/demo.py (memes formules).

    Parameters
    ----------
    tableau : pandas.DataFrame
        Le resultat de `tableau_flotte` (colonnes rul_predite, rul_vraie, alerte, statut).
    seuil_alerte : float, optionnel
        Le seuil d'alerte en cycles.

    Returns
    -------
    dict
        nb_moteurs, rmse, mae, score_nasa, nb_surestimes (> 20 cycles), biais, nb_sous_seuil,
        alertes_emises, alertes_correctes, alertes_manquees, fausses_alertes.
    """
    return resumer_flotte(tableau, seuil_alerte)


def compter_statuts(tableau):
    """
    Compte les moteurs de chaque statut, dans l'ordre d'urgence ORDRE_STATUTS.

    Parameters
    ----------
    tableau : pandas.DataFrame
        Le resultat de `tableau_flotte`.

    Returns
    -------
    dict
        statut -> nombre de moteurs (les 4 statuts sont toujours presents, eventuellement a 0).
    """
    comptes = {}
    valeurs = tableau["statut"].tolist()
    for statut in ORDRE_STATUTS:
        nombre = 0
        for valeur in valeurs:
            if valeur == statut:
                nombre = nombre + 1
        comptes[statut] = nombre
    return comptes


def controler_coherence(df, metadonnees):
    """
    Recalcule les metriques de test depuis le fichier et les compare aux metadonnees (preuve affichee).

    Parameters
    ----------
    df : pandas.DataFrame
        Le fichier de predictions.
    metadonnees : dict
        Ses metadonnees.

    Returns
    -------
    dict
        - "tableau" : pandas.DataFrame de `verifier_coherence` (metrique, metadonnees, recalculee,
          ecart, coherent) ;
        - "coherent" : bool, Vrai si toutes les metriques concordent ;
        - "ecart_max" : float, le plus grand ecart absolu constate ;
        - "nb_metriques" : int.
    """
    tableau = verifier_coherence(df, metadonnees)
    ecart_max = 0.0
    for valeur in tableau["ecart"].tolist():
        if abs(float(valeur)) > ecart_max:
            ecart_max = abs(float(valeur))
    resultat = {}
    resultat["tableau"] = tableau
    resultat["coherent"] = bool(tableau["coherent"].all())
    resultat["ecart_max"] = ecart_max
    resultat["nb_metriques"] = int(len(tableau))
    return resultat


# ---------------------------------------------------------------------------------------------------
# Comparaison de tous les (sous-jeu x modele) disponibles
# ---------------------------------------------------------------------------------------------------

def est_un_nombre(valeur):
    """
    Dit si une valeur est un nombre utilisable (entier ou reel, pas un booleen, pas NaN).

    Parameters
    ----------
    valeur : objet
        La valeur a tester.

    Returns
    -------
    bool
        Vrai pour 3, 2.5, numpy.float64(1.0) ; Faux pour True, "3", None, NaN, une liste.
    """
    if isinstance(valeur, bool):
        return False
    if not isinstance(valeur, (int, float, np.integer, np.floating)):
        return False
    return not math.isnan(float(valeur))


def metriques_validation_scalaires(metadonnees):
    """
    Garde, dans les metriques de validation, celles qui sont un simple nombre (pas une liste par pli).

    Chaque script de production choisit ses propres cles de validation (rmse_cv_moyen, ...) : on ne
    suppose rien sur leurs noms, on garde tout ce qui est numerique, dans l'ordre du fichier.

    Parameters
    ----------
    metadonnees : dict
        Les metadonnees d'un fichier de predictions.

    Returns
    -------
    dict
        cle -> valeur (float), dans l'ordre du fichier.
    """
    scalaires = {}
    metriques = metadonnees.get("metriques_validation", {})
    for cle in metriques:
        valeur = metriques[cle]
        if est_un_nombre(valeur):
            scalaires[cle] = float(valeur)
    return scalaires


def formater_nombre(valeur):
    """
    Ecrit un nombre de facon lisible : entier sans decimale, reel avec 2 decimales (4 si petit).

    Parameters
    ----------
    valeur : float
        Le nombre.

    Returns
    -------
    str
        Par exemple "100", "17.28", "0.0544".
    """
    if float(valeur) == round(float(valeur)) and abs(float(valeur)) < 1e15:
        return str(int(round(float(valeur))))
    if abs(float(valeur)) < 1:
        return f"{float(valeur):.4f}"
    return f"{float(valeur):.2f}"


def tableau_comparaison(inventaire, dossier=None):
    """
    Rassemble les metriques de test et de validation de tous les fichiers de predictions disponibles.

    Les metriques de test sont lues dans les metadonnees (calculees au dernier cycle de chaque moteur
    test, voir le contrat). Les metriques de validation et le protocole sont recopies tels quels.

    Parameters
    ----------
    inventaire : pandas.DataFrame
        Le resultat de `inventaire_predictions`.
    dossier : str ou None, optionnel
        Dossier des predictions ; None pour le dossier par defaut.

    Returns
    -------
    pandas.DataFrame
        Une ligne par (sous-jeu, modele) : sous_jeu, modele, nom_affiche, famille, nb_moteurs,
        nb_au_dela_plafond (moteurs dont la RUL vraie au dernier cycle depasse le plafond de 125) et
        part_au_dela_plafond (la meme chose en % des moteurs test),
        rmse, rmse_contre_rul_plafonnee, mae, score_nasa, score_nasa_par_moteur, nb_surestimes,
        alertes_manquees, fausses_alertes, rmse_validation et nasa_validation (cles "rmse_cv_moyen" et
        "nasa_cv_moyen" des metriques de validation, NaN si absentes), validation (texte
        "cle = valeur ; ..."), protocole_validation, duree_entrainement_s, date, source.
        Vide si aucun fichier.
    """
    lignes = []
    for position in range(len(inventaire)):
        sous_jeu = inventaire["sous_jeu"].iloc[position]
        modele = inventaire["modele"].iloc[position]
        df, meta = lire_predictions(sous_jeu, modele, dossier)
        metriques = meta["metriques_test"]

        # moteurs dont la RUL vraie au dernier cycle depasse le plafond : le modele, qui a appris une
        # cible plafonnee, ne peut pas les predire exactement (ils pesent lourd dans le RMSE brut)
        derniers = extraire_dernier_cycle(df)
        nb_au_dela_plafond = int((derniers["rul_vraie"] > PLAFOND_RUL).sum())

        # nombre de moteurs test : ecrit par le contrat (nb_moteurs), sinon dans les metriques
        nb_moteurs = meta.get("nb_moteurs", metriques.get("nb_moteurs", np.nan))

        # metriques de validation numeriques, mises bout a bout en un texte lisible
        morceaux = []
        scalaires = metriques_validation_scalaires(meta)
        for cle in scalaires:
            morceaux.append(f"{cle} = {formater_nombre(scalaires[cle])}")

        ligne = {}
        ligne["sous_jeu"] = sous_jeu
        ligne["modele"] = modele
        ligne["nom_affiche"] = meta["nom_affiche"]
        ligne["famille"] = meta["famille"]
        ligne["nb_moteurs"] = nb_moteurs
        ligne["nb_au_dela_plafond"] = nb_au_dela_plafond
        ligne["part_au_dela_plafond"] = 100.0 * nb_au_dela_plafond / float(len(derniers))  # en %
        ligne["rmse"] = float(metriques["rmse"])
        ligne["rmse_contre_rul_plafonnee"] = float(metriques["rmse_contre_rul_plafonnee"])
        ligne["mae"] = float(metriques["mae"])
        ligne["score_nasa"] = float(metriques["score_nasa"])
        if est_un_nombre(nb_moteurs) and float(nb_moteurs) > 0:
            ligne["score_nasa_par_moteur"] = float(metriques["score_nasa"]) / float(nb_moteurs)
        else:
            ligne["score_nasa_par_moteur"] = np.nan
        ligne["nb_surestimes"] = int(metriques["nb_surestimes"])
        ligne["alertes_manquees"] = int(metriques["alertes_manquees"])
        ligne["fausses_alertes"] = int(metriques["fausses_alertes"])
        ligne["rmse_validation"] = scalaires.get("rmse_cv_moyen", np.nan)
        ligne["nasa_validation"] = scalaires.get("nasa_cv_moyen", np.nan)
        ligne["validation"] = " ; ".join(morceaux)
        ligne["protocole_validation"] = meta["protocole_validation"]
        ligne["duree_entrainement_s"] = float(meta["duree_entrainement_s"])
        ligne["date"] = meta["date"]
        ligne["source"] = meta["source"]
        lignes.append(ligne)

    colonnes = [
        "sous_jeu", "modele", "nom_affiche", "famille", "nb_moteurs", "nb_au_dela_plafond", "part_au_dela_plafond", "rmse",
        "rmse_contre_rul_plafonnee", "mae", "score_nasa", "score_nasa_par_moteur", "nb_surestimes",
        "alertes_manquees", "fausses_alertes", "rmse_validation", "nasa_validation", "validation",
        "protocole_validation", "duree_entrainement_s", "date", "source",
    ]
    return pd.DataFrame(lignes, columns=colonnes)


def conclure_test_apparie(p_valeur_t, p_valeur_wilcoxon, difference_moyenne, nom_a, nom_b,
                          seuil=SEUIL_SIGNIFICATIVITE):
    """
    Traduit le resultat des deux tests apparies en une phrase factuelle.

    Parameters
    ----------
    p_valeur_t, p_valeur_wilcoxon : float
        Les p-valeurs du test t apparie et du test de Wilcoxon signe.
    difference_moyenne : float
        Moyenne de (erreur de A - erreur de B) : positive = A se trompe davantage.
    nom_a, nom_b : str
        Les noms des deux modeles.
    seuil : float, optionnel
        Seuil de significativite (0,05 par defaut, convention usuelle).

    Returns
    -------
    str
        "A significativement moins bon que B", "A significativement meilleur que B",
        "différence non significative" ou "tests en désaccord".
    """
    significatif_t = p_valeur_t < seuil
    significatif_w = p_valeur_wilcoxon < seuil
    if significatif_t and significatif_w:
        if difference_moyenne > 0:
            return f"{nom_a} significativement moins bon que {nom_b}"
        return f"{nom_a} significativement meilleur que {nom_b}"
    if not significatif_t and not significatif_w:
        return "différence non significative"
    return "tests en désaccord (conclusion fragile)"


def comparer_deux_modeles(inventaire, modele_a, modele_b, dossier=None):
    """
    Compare deux modeles moteur par moteur sur le test de chaque sous-jeu ou les deux sont disponibles.

    Pour chaque moteur test, on prend l'erreur absolue au dernier cycle observe |RUL predite - RUL vraie|
    (c'est le "RMSE par moteur" de la Phase 4 : une seule valeur par moteur), puis on compare les deux
    modeles avec `comparer_apparie` de src/evaluation.py (test t apparie + test de Wilcoxon signe),
    exactement comme les tests apparies de la Phase 4.

    Parameters
    ----------
    inventaire : pandas.DataFrame
        Le resultat de `inventaire_predictions`.
    modele_a, modele_b : str
        Les deux cles de modeles a comparer.
    dossier : str ou None, optionnel
        Dossier des predictions.

    Returns
    -------
    pandas.DataFrame
        Une ligne par sous-jeu commun : sous_jeu, nb_moteurs, rmse_a, rmse_b, mae_a, mae_b,
        difference_moyenne (erreur absolue de A - de B, en cycles), p_valeur_t, p_valeur_wilcoxon,
        conclusion. Vide si aucun sous-jeu commun.

    Raises
    ------
    ValueError
        Si les deux fichiers d'un meme sous-jeu ne contiennent pas les memes moteurs.
    """
    from src.evaluation import comparer_apparie

    nom_a = nom_modele(modele_a)
    nom_b = nom_modele(modele_b)
    lignes = []
    for sous_jeu in sous_jeux_disponibles(inventaire):
        modeles = modeles_disponibles(inventaire, sous_jeu)
        if modele_a not in modeles or modele_b not in modeles:
            continue  # comparaison impossible sur ce sous-jeu
        df_a, meta_a = lire_predictions(sous_jeu, modele_a, dossier)
        df_b, meta_b = lire_predictions(sous_jeu, modele_b, dossier)
        derniers_a = extraire_dernier_cycle(df_a)  # une ligne par moteur, triee par numero
        derniers_b = extraire_dernier_cycle(df_b)
        if derniers_a["unit_number"].tolist() != derniers_b["unit_number"].tolist():
            raise ValueError(f"{sous_jeu} : les deux fichiers ne contiennent pas les memes moteurs test.")

        # erreur absolue au dernier cycle, moteur par moteur (predictions deja clippees a 0 par le contrat)
        erreurs_a = np.abs(derniers_a["rul_predite"].to_numpy(dtype=float) - derniers_a["rul_vraie"].to_numpy(dtype=float))
        erreurs_b = np.abs(derniers_b["rul_predite"].to_numpy(dtype=float) - derniers_b["rul_vraie"].to_numpy(dtype=float))
        test = comparer_apparie(erreurs_a, erreurs_b)

        ligne = {}
        ligne["sous_jeu"] = sous_jeu
        ligne["nb_moteurs"] = test["n"]
        ligne["rmse_a"] = float(np.sqrt(np.mean(erreurs_a ** 2)))
        ligne["rmse_b"] = float(np.sqrt(np.mean(erreurs_b ** 2)))
        ligne["mae_a"] = float(np.mean(erreurs_a))
        ligne["mae_b"] = float(np.mean(erreurs_b))
        ligne["difference_moyenne"] = test["difference_moyenne"]
        ligne["p_valeur_t"] = test["p_valeur_t"]
        ligne["p_valeur_wilcoxon"] = test["p_valeur_wilcoxon"]
        ligne["conclusion"] = conclure_test_apparie(
            test["p_valeur_t"], test["p_valeur_wilcoxon"], test["difference_moyenne"], nom_a, nom_b
        )
        lignes.append(ligne)
    colonnes = [
        "sous_jeu", "nb_moteurs", "rmse_a", "rmse_b", "mae_a", "mae_b", "difference_moyenne",
        "p_valeur_t", "p_valeur_wilcoxon", "conclusion",
    ]
    return pd.DataFrame(lignes, columns=colonnes)


def tableau_validation_detaille(inventaire, dossier=None):
    """
    Met toutes les metriques de validation numeriques de chaque fichier dans des colonnes separees.

    Parameters
    ----------
    inventaire : pandas.DataFrame
        Le resultat de `inventaire_predictions`.
    dossier : str ou None, optionnel
        Dossier des predictions ; None pour le dossier par defaut.

    Returns
    -------
    pandas.DataFrame
        Une ligne par (sous-jeu, modele) ; une colonne par cle de validation rencontree (NaN si un
        fichier n'a pas cette cle).
    """
    lignes = []
    for position in range(len(inventaire)):
        sous_jeu = inventaire["sous_jeu"].iloc[position]
        modele = inventaire["modele"].iloc[position]
        meta = lire_metadonnees(sous_jeu, modele, dossier)
        ligne = {}
        ligne["sous_jeu"] = sous_jeu
        ligne["modele"] = meta["nom_affiche"]
        scalaires = metriques_validation_scalaires(meta)
        for cle in scalaires:
            ligne[cle] = scalaires[cle]
        lignes.append(ligne)
    return pd.DataFrame(lignes)


def controler_tous_les_fichiers(inventaire, dossier=None):
    """
    Lance le controle de coherence (metriques recalculees = metadonnees) sur chaque fichier disponible.

    Parameters
    ----------
    inventaire : pandas.DataFrame
        Le resultat de `inventaire_predictions`.
    dossier : str ou None, optionnel
        Dossier des predictions ; None pour le dossier par defaut.

    Returns
    -------
    pandas.DataFrame
        Une ligne par fichier : sous_jeu, modele, nb_lignes, nb_moteurs, coherent (bool), ecart_max.
    """
    lignes = []
    for position in range(len(inventaire)):
        sous_jeu = inventaire["sous_jeu"].iloc[position]
        modele = inventaire["modele"].iloc[position]
        df, meta = lire_predictions(sous_jeu, modele, dossier)
        controle = controler_coherence(df, meta)
        ligne = {}
        ligne["sous_jeu"] = sous_jeu
        ligne["modele"] = meta["nom_affiche"]
        ligne["nb_lignes"] = int(len(df))
        ligne["nb_moteurs"] = int(df["unit_number"].nunique())
        ligne["coherent"] = controle["coherent"]
        ligne["ecart_max"] = controle["ecart_max"]
        lignes.append(ligne)
    colonnes = ["sous_jeu", "modele", "nb_lignes", "nb_moteurs", "coherent", "ecart_max"]
    return pd.DataFrame(lignes, columns=colonnes)


# ---------------------------------------------------------------------------------------------------
# Resultats de la Phase 4 (docs/resultats/)
# ---------------------------------------------------------------------------------------------------

def lire_csv_resultat(nom_fichier, dossier=None):
    """
    Lit un CSV de docs/resultats/ s'il existe.

    Parameters
    ----------
    nom_fichier : str
        Le nom du fichier (ex. "04_comparaison_cv.csv").
    dossier : str ou None, optionnel
        Le dossier des resultats (docs/resultats/ par defaut).

    Returns
    -------
    pandas.DataFrame ou None
        Le tableau, ou None si le fichier n'existe pas.
    """
    if dossier is None:
        dossier = DOSSIER_RESULTATS
    chemin = os.path.join(dossier, nom_fichier)
    if not os.path.isfile(chemin):
        return None
    return pd.read_csv(chemin)


def lire_config_retenue(chemin=None):
    """
    Lit le fichier JSON de la configuration retenue en Phase 4 (avec ses champs descriptifs).

    On lit le JSON tel quel (et non avec `charger_config`, qui renvoie la configuration Phase 3 si le
    fichier est absent) : l'application doit dire clairement quand le fichier manque.

    Parameters
    ----------
    chemin : str ou None, optionnel
        Le fichier (docs/resultats/04_config_retenue.json par defaut).

    Returns
    -------
    dict ou None
        La configuration, ou None si le fichier n'existe pas.
    """
    if chemin is None:
        chemin = CHEMIN_CONFIG_RETENUE
    if not os.path.isfile(chemin):
        return None
    with open(chemin, "r", encoding="utf-8") as fichier:
        return json.load(fichier)


def valeurs_egales(a, b, tolerance=1e-9):
    """
    Compare deux valeurs (nombres, textes, listes, dictionnaires imbriques) avec une tolerance sur les reels.

    Parameters
    ----------
    a, b : objet
        Les deux valeurs a comparer.
    tolerance : float, optionnel
        Ecart relatif tolere entre deux nombres (1e-9 par defaut).

    Returns
    -------
    bool
        Vrai si les deux valeurs sont egales (a la tolerance pres pour les nombres).
    """
    if isinstance(a, dict) and isinstance(b, dict):
        if sorted(a.keys()) != sorted(b.keys()):
            return False
        for cle in a:
            if not valeurs_egales(a[cle], b[cle], tolerance):
                return False
        return True
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        for position in range(len(a)):
            if not valeurs_egales(a[position], b[position], tolerance):
                return False
        return True
    if est_un_nombre(a) and est_un_nombre(b):
        limite = tolerance * max(1.0, abs(float(a)), abs(float(b)))
        return abs(float(a) - float(b)) <= limite
    return a == b


def texte_parametres_essai(ligne):
    """
    Ecrit les parametres d'un essai Optuna (colonnes "param_...") en un texte pour l'info-bulle.

    Parameters
    ----------
    ligne : pandas.Series
        Une ligne du tableau des essais.

    Returns
    -------
    str
        Par exemple "alpha = 2.26<br>n_estimators = 200<br>..." (balises <br> : retour a la ligne Plotly).
    """
    morceaux = []
    for colonne in ligne.index:
        if not colonne.startswith("param_"):
            continue
        valeur = ligne[colonne]
        if not est_un_nombre(valeur):
            continue  # parametre absent de cette etude (NaN)
        nom = colonne[len("param_"):]  # "param_alpha" -> "alpha"
        morceaux.append(f"{nom} = {formater_nombre(valeur)}")
    return "<br>".join(morceaux)


def analyser_optuna(dossier=None, chemin_config=None):
    """
    Reconstruit le front de Pareto des essais Optuna et rejoue la regle de choix de la Phase 4.

    On relit les CSV des trois etudes (`charger_essais`), on applique `appliquer_regle_de_choix` de
    src/optimisation.py (code valide, non modifie), puis on verifie que l'essai choisi par la regle
    redonne exactement la configuration ecrite dans docs/resultats/04_config_retenue.json.

    Parameters
    ----------
    dossier : str ou None, optionnel
        Le dossier des CSV (docs/resultats/ par defaut).
    chemin_config : str ou None, optionnel
        La configuration retenue (docs/resultats/04_config_retenue.json par defaut).

    Returns
    -------
    dict ou None
        None si un CSV manque ; sinon :
        - "essais" : pandas.DataFrame de tous les essais, avec en plus "sur_le_front", "est_retenu",
          "est_meilleur_rmse", "admissible" (bool) et "parametres_texte" ;
        - "seuil_rmse", "rmse_min_front" : float ;
        - "retenu", "meilleur_rmse" : pandas.Series ;
        - "nb_essais", "nb_front", "nb_admissibles" : int ;
        - "config_coherente" : bool ou None (None si la configuration retenue est absente).
    """
    # import ici : src/optimisation.py charge optuna, inutile pour les autres pages
    from src.optimisation import (
        NOMS_ETUDES,
        appliquer_regle_de_choix,
        calculer_front_pareto,
        charger_essais,
        config_depuis_ligne,
    )

    try:
        essais = charger_essais(NOMS_ETUDES, dossier)
    except FileNotFoundError:
        return None
    essais = essais.reset_index(drop=True)

    regle = appliquer_regle_de_choix(essais)  # front, seuil, retenu, meilleur RMSE
    retenu = regle["retenu"]
    meilleur_rmse = regle["meilleur_rmse"]

    # colonnes ajoutees ligne par ligne pour le graphique
    essais["sur_le_front"] = calculer_front_pareto(essais, "rmse_cv_moyen", "nasa_cv_moyen")
    est_retenu = []
    est_meilleur = []
    admissible = []
    textes = []
    for position in range(len(essais)):
        ligne = essais.iloc[position]
        meme_que_retenu = ligne["etude"] == retenu["etude"] and ligne["numero"] == retenu["numero"]
        meme_que_meilleur = ligne["etude"] == meilleur_rmse["etude"] and ligne["numero"] == meilleur_rmse["numero"]
        est_retenu.append(bool(meme_que_retenu))
        est_meilleur.append(bool(meme_que_meilleur))
        sous_le_seuil = float(ligne["rmse_cv_moyen"]) <= regle["seuil_rmse"]
        admissible.append(bool(essais["sur_le_front"].iloc[position] and sous_le_seuil))
        textes.append(texte_parametres_essai(ligne))
    essais["est_retenu"] = est_retenu
    essais["est_meilleur_rmse"] = est_meilleur
    essais["admissible"] = admissible
    essais["parametres_texte"] = textes

    # on relit les deux essais dans le tableau complete (avec les colonnes ajoutees ci-dessus)
    retenu = essais[essais["est_retenu"]].iloc[0]
    meilleur_rmse = essais[essais["est_meilleur_rmse"]].iloc[0]

    # la regle rejouee redonne-t-elle la configuration retenue ecrite dans le JSON ?
    config_retenue = lire_config_retenue(chemin_config)
    if config_retenue is None:
        config_coherente = None
    else:
        config_essai = config_depuis_ligne(retenu)
        config_coherente = True
        for cle in ["famille", "perte", "parametres_perte", "hyperparametres"]:
            if not valeurs_egales(config_essai[cle], config_retenue.get(cle)):
                config_coherente = False

    resultat = {}
    resultat["essais"] = essais
    resultat["seuil_rmse"] = float(regle["seuil_rmse"])
    resultat["rmse_min_front"] = float(regle["rmse_min_front"])
    resultat["retenu"] = retenu
    resultat["meilleur_rmse"] = meilleur_rmse
    resultat["nb_essais"] = int(len(essais))
    resultat["nb_front"] = int(essais["sur_le_front"].sum())
    resultat["nb_admissibles"] = int(len(regle["admissibles"]))
    resultat["config_coherente"] = config_coherente
    return resultat


# ---------------------------------------------------------------------------------------------------
# Description des sous-jeux et donnees brutes
# ---------------------------------------------------------------------------------------------------

def analyser_readme_donnees(texte):
    """
    Extrait du readme NASA (data/readme.txt) la description de chaque sous-jeu.

    Le fichier contient des blocs "Data Set: FD001 / Train trjectories: 100 / Test trajectories: 100 /
    Conditions: ONE (Sea Level) / Fault Modes: ONE (HPC Degradation)" (fautes de frappe d'origine).

    Parameters
    ----------
    texte : str
        Le contenu du readme.

    Returns
    -------
    pandas.DataFrame
        Une ligne par sous-jeu : sous_jeu, moteurs_train, moteurs_test, conditions, modes_de_panne.
    """
    lignes = []
    ligne_courante = None
    for brute in texte.splitlines():
        texte_ligne = brute.strip()
        if ":" not in texte_ligne:
            continue
        cle, valeur = texte_ligne.split(":", 1)  # on coupe au premier deux-points seulement
        cle = cle.strip().lower()
        valeur = valeur.strip()
        if cle == "data set":  # debut d'un nouveau bloc
            ligne_courante = {}
            ligne_courante["sous_jeu"] = valeur
            ligne_courante["moteurs_train"] = np.nan
            ligne_courante["moteurs_test"] = np.nan
            ligne_courante["conditions"] = ""
            ligne_courante["modes_de_panne"] = ""
            lignes.append(ligne_courante)
        elif ligne_courante is None:
            continue  # texte avant le premier bloc
        elif cle.startswith("train"):  # "Train trjectories" (faute d'origine)
            ligne_courante["moteurs_train"] = int(valeur)
        elif cle.startswith("test"):
            ligne_courante["moteurs_test"] = int(valeur)
        elif cle == "conditions":
            ligne_courante["conditions"] = valeur
        elif cle == "fault modes":
            ligne_courante["modes_de_panne"] = valeur
    colonnes = ["sous_jeu", "moteurs_train", "moteurs_test", "conditions", "modes_de_panne"]
    return pd.DataFrame(lignes, columns=colonnes)


def lire_description_module_sous_jeux():
    """
    Lit DESCRIPTION_SOUS_JEUX dans src/sous_jeux.py s'il existe (module ecrit en parallele).

    Le format exact de cette constante n'est pas fixe par le contrat : on accepte un dictionnaire
    sous-jeu -> dictionnaire de champs, et on le transforme en tableau sans rien supposer des champs.

    Returns
    -------
    pandas.DataFrame ou None
        Une ligne par sous-jeu (colonne "sous_jeu" + un champ par colonne), ou None si le module ou
        la constante n'existe pas, ou si son format n'est pas un dictionnaire de dictionnaires.
    """
    try:
        from src import sous_jeux
    except ImportError:
        return None
    description = getattr(sous_jeux, "DESCRIPTION_SOUS_JEUX", None)
    if not isinstance(description, dict):
        return None
    lignes = []
    for sous_jeu in description:
        champs = description[sous_jeu]
        if not isinstance(champs, dict):
            return None  # format inattendu : on se replie sur le readme
        ligne = {}
        ligne["sous_jeu"] = sous_jeu
        for cle in champs:
            valeur = champs[cle]
            if isinstance(valeur, (list, tuple)):  # liste (ex. modes de panne) -> texte lisible
                morceaux = []
                for element in valeur:
                    morceaux.append(str(element))
                valeur = ", ".join(morceaux)
            ligne[cle] = valeur
        lignes.append(ligne)
    return pd.DataFrame(lignes)


def lire_readme_donnees():
    """
    Lit et analyse data/readme.txt (description officielle des sous-jeux par la NASA).

    Returns
    -------
    pandas.DataFrame
        Le resultat de `analyser_readme_donnees`.
    """
    with open(CHEMIN_README_DONNEES, "r", encoding="utf-8", errors="replace") as fichier:
        texte = fichier.read()  # errors="replace" : le readme contient des caracteres mal encodes
    return analyser_readme_donnees(texte)


def decrire_sous_jeux():
    """
    Decrit les quatre sous-jeux (regimes, modes de panne, nombre de moteurs) pour la page d'accueil.

    Source privilegiee : src/sous_jeux.DESCRIPTION_SOUS_JEUX s'il existe (comptes verifies sur les
    fichiers par ses propres tests) ; sinon data/readme.txt.

    Returns
    -------
    (pandas.DataFrame, str)
        Le tableau et le nom du fichier source (affiche sous le tableau).
    """
    depuis_module = lire_description_module_sous_jeux()
    if depuis_module is not None and len(depuis_module) > 0:
        return depuis_module, "src/sous_jeux.py (DESCRIPTION_SOUS_JEUX)"
    return lire_readme_donnees(), "data/readme.txt"


def ecarts_readme(description, readme):
    """
    Compare les nombres de moteurs d'une description (ex. celle de src/sous_jeux.py) a ceux du readme.

    Exemple reel : data/readme.txt annonce 248 moteurs train et 249 test pour FD004, alors que les
    fichiers en contiennent l'inverse. L'application affiche ces ecarts plutot que de les cacher.

    Parameters
    ----------
    description : pandas.DataFrame
        Colonnes "sous_jeu", "nb_moteurs_train", "nb_moteurs_test".
    readme : pandas.DataFrame
        Le resultat de `analyser_readme_donnees` ("moteurs_train", "moteurs_test").

    Returns
    -------
    list de str
        Une phrase par ecart constate (vide si tout concorde ou si les colonnes manquent).
    """
    ecarts = []
    colonnes_attendues = ["sous_jeu", "nb_moteurs_train", "nb_moteurs_test"]
    for colonne in colonnes_attendues:
        if colonne not in description.columns:
            return ecarts  # description sans comptes de moteurs : rien a comparer
    for position in range(len(description)):
        sous_jeu = description["sous_jeu"].iloc[position]
        lignes_readme = readme[readme["sous_jeu"] == sous_jeu]
        if len(lignes_readme) == 0:
            continue
        for split, colonne_desc, colonne_readme in (("train", "nb_moteurs_train", "moteurs_train"),
                                                     ("test", "nb_moteurs_test", "moteurs_test")):
            valeur_desc = description[colonne_desc].iloc[position]
            valeur_readme = lignes_readme[colonne_readme].iloc[0]
            if est_un_nombre(valeur_desc) and est_un_nombre(valeur_readme) and int(valeur_desc) != int(valeur_readme):
                ecarts.append(
                    f"{sous_jeu} {split} : data/readme.txt annonce {int(valeur_readme)} moteurs, "
                    f"les fichiers en contiennent {int(valeur_desc)}."
                )
    return ecarts


def charger_donnees_brutes_test(sous_jeu):
    """
    Charge les donnees capteurs BRUTES du jeu de test d'un sous-jeu (26 colonnes de src/preprocessing.py).

    Ordre de preference :
    1. `src.sous_jeux.charger_sous_jeu(sous_jeu, "test")` s'il existe (module ecrit en parallele) ;
    2. pour FD001, `load_fd001("test")` (code deja valide) ;
    3. pour les autres, lecture directe de data/test_FD00X.txt avec exactement les memes reglages que
       `load_fd001` (separateur espaces, pas d'en-tete, noms de COLUMNS).

    Parameters
    ----------
    sous_jeu : str
        Le sous-jeu (ex. "FD002").

    Returns
    -------
    pandas.DataFrame
        Une ligne par (moteur, cycle), colonnes unit_number, time_cycles, op_setting_1..3, sensor_1..21.

    Raises
    ------
    ValueError
        Si le sous-jeu est inconnu.
    """
    if sous_jeu not in SOUS_JEUX:
        raise ValueError(f"Sous-jeu inconnu : {sous_jeu!r}.")
    try:
        from src.sous_jeux import charger_sous_jeu
    except ImportError:
        charger_sous_jeu = None
    if charger_sous_jeu is not None:
        return charger_sous_jeu(sous_jeu, "test")
    if sous_jeu == "FD001":
        return load_fd001("test")
    chemin = os.path.join(DATA_DIR, f"test_{sous_jeu}.txt")
    return pd.read_csv(chemin, sep=r"\s+", header=None, names=COLUMNS)


def capteurs_variables(donnees):
    """
    Liste les capteurs (sensor_N) qui ne sont pas constants dans un tableau de donnees brutes.

    Parameters
    ----------
    donnees : pandas.DataFrame
        Des donnees brutes (un ou plusieurs moteurs).

    Returns
    -------
    list de str
        Les capteurs a plus d'une valeur distincte, dans l'ordre sensor_1 -> sensor_21.
    """
    capteurs = []
    for colonne in donnees.columns:
        if not colonne.startswith("sensor_"):
            continue
        if donnees[colonne].nunique() > 1:  # meme critere que find_constant_sensors (Phase 1)
            capteurs.append(colonne)
    return capteurs


def choisir_capteurs_par_defaut(capteurs, nombre=3):
    """
    Choisit les capteurs affiches par defaut : ceux de CAPTEURS_PAR_DEFAUT s'ils existent, sinon les premiers.

    Parameters
    ----------
    capteurs : list de str
        Les capteurs disponibles.
    nombre : int, optionnel
        Nombre de capteurs a proposer (3 par defaut).

    Returns
    -------
    list de str
        Au plus `nombre` capteurs.
    """
    choisis = []
    for capteur in CAPTEURS_PAR_DEFAUT:  # d'abord les capteurs de reference
        if capteur in capteurs and len(choisis) < nombre:
            choisis.append(capteur)
    for capteur in capteurs:  # puis on complete avec les premiers disponibles
        if capteur not in choisis and len(choisis) < nombre:
            choisis.append(capteur)
    return choisis


def tableau_grandeurs_saxena():
    """
    Met le tableau 2 de Saxena et al. (2008) sous forme de DataFrame (information, voir l'avertissement).

    Returns
    -------
    pandas.DataFrame
        Colonnes : rang, symbole, description, unite (21 lignes, dans l'ordre de l'article).
    """
    lignes = []
    for position in range(len(GRANDEURS_TABLEAU_2_SAXENA)):
        symbole, description, unite = GRANDEURS_TABLEAU_2_SAXENA[position]
        ligne = {}
        ligne["rang"] = position + 1
        ligne["symbole"] = symbole
        ligne["description"] = description
        ligne["unite"] = unite
        lignes.append(ligne)
    return pd.DataFrame(lignes)


# ---------------------------------------------------------------------------------------------------
# Inference en direct (page "Et si...") avec l'artefact de la demo
# ---------------------------------------------------------------------------------------------------

def charger_artefact_si_present(chemin=None):
    """
    Charge l'artefact du modele retenu s'il existe, SANS lancer d'entrainement.

    Parameters
    ----------
    chemin : str ou None, optionnel
        Le fichier de l'artefact (`chemin_artefact_demo()` par defaut).

    Returns
    -------
    dict ou None
        L'artefact, ou None s'il n'existe pas.
    """
    if chemin is None:
        chemin = chemin_artefact_demo()
    if not os.path.isfile(chemin):
        return None
    return charger_artefact(chemin)


def charger_artefact_demo(chemin=None):
    """
    Charge l'artefact du modele retenu ; s'il n'existe pas, l'entraine d'abord (`charger_ou_entrainer`).

    Parameters
    ----------
    chemin : str ou None, optionnel
        Le fichier de l'artefact (`chemin_artefact_demo()` par defaut).

    Returns
    -------
    dict
        L'artefact pret a predire.
    """
    if chemin is None:
        chemin = chemin_artefact_demo()
    return charger_ou_entrainer(chemin, CHEMIN_CONFIG_RETENUE)


def comparer_artefact_et_config(artefact, chemin_config=None):
    """
    Dit si l'artefact a ete entraine avec la configuration actuellement ecrite dans 04_config_retenue.json.

    Parameters
    ----------
    artefact : dict
        L'artefact charge.
    chemin_config : str ou None, optionnel
        La configuration retenue (docs/resultats/04_config_retenue.json par defaut).

    Returns
    -------
    bool ou None
        Vrai si identiques, Faux sinon, None si le fichier de configuration est absent.
    """
    config = lire_config_retenue(chemin_config)
    if config is None:
        return None
    return valeurs_egales(convertir_en_types_python(config), artefact["config"])


def resume_pipeline(artefact):
    """
    Resume la preparation des donnees apprise sur le train, a partir de l'artefact (page d'accueil).

    Parameters
    ----------
    artefact : dict
        L'artefact charge.

    Returns
    -------
    dict
        nb_capteurs_bruts, nb_capteurs_retires, capteurs_retires, nb_capteurs_gardes, fenetre,
        nb_features, plafond, nb_moteurs_train, nb_lignes_train, description_modele, date_entrainement.
    """
    resume = {}
    resume["nb_capteurs_bruts"] = nombre_capteurs_bruts()
    resume["nb_capteurs_retires"] = len(artefact["constant_sensors"])
    resume["capteurs_retires"] = list(artefact["constant_sensors"])
    resume["nb_capteurs_gardes"] = len(artefact["sensor_cols_restants"])
    resume["fenetre"] = int(artefact["fenetre"])
    resume["nb_features"] = len(artefact["feature_cols"])
    resume["plafond"] = int(artefact["plafond"])
    resume["nb_moteurs_train"] = int(artefact["nb_moteurs_train"])
    resume["nb_lignes_train"] = int(artefact["nb_lignes_train"])
    resume["description_modele"] = artefact["description_modele"]
    resume["date_entrainement"] = artefact["date_entrainement"]
    return resume


def etendue_capteur_train(artefact, capteur):
    """
    Lit, dans le scaler min-max de l'artefact, le minimum et le maximum d'un capteur sur le train.

    Parameters
    ----------
    artefact : dict
        L'artefact charge.
    capteur : str
        Un capteur garde par la preparation (ex. "sensor_11").

    Returns
    -------
    dict
        "minimum", "maximum", "etendue" (maximum - minimum), en unites brutes du capteur.

    Raises
    ------
    ValueError
        Si le capteur n'est pas utilise par le modele (capteur constant retire en Phase 1).
    """
    capteurs = list(artefact["sensor_cols_restants"])
    if capteur not in capteurs:
        raise ValueError(f"Le capteur {capteur} n'est pas utilise par le modele (capteur constant retire).")
    position = capteurs.index(capteur)  # meme ordre que les colonnes vues par le scaler
    scaler = artefact["scaler"]
    resultat = {}
    resultat["minimum"] = float(scaler.data_min_[position])
    resultat["maximum"] = float(scaler.data_max_[position])
    resultat["etendue"] = float(scaler.data_max_[position] - scaler.data_min_[position])
    return resultat


def appliquer_derive(donnees_moteur, capteur, type_derive, amplitude, cycle_debut, progressive=False):
    """
    Injecte une derive artificielle sur un capteur brut, a partir d'un cycle donne.

    - "pourcentage" : valeur x (1 + amplitude / 100) ;
    - "decalage" : valeur + amplitude (en unites brutes du capteur).
    Mode progressif : la derive monte lineairement de 0 au cycle de debut jusqu'a l'amplitude complete
    au dernier cycle observe (une usure qui s'installe) ; sinon elle est appliquee d'un coup (un defaut).

    Parameters
    ----------
    donnees_moteur : pandas.DataFrame
        Les donnees brutes d'un moteur (non modifiees : une copie est renvoyee).
    capteur : str
        Le capteur a perturber (ex. "sensor_11").
    type_derive : str
        "pourcentage" ou "decalage".
    amplitude : float
        Le pourcentage (ex. 1.5 pour +1,5 %) ou le decalage (unites brutes).
    cycle_debut : int
        Premier cycle perturbe.
    progressive : bool, optionnel
        Vrai pour une montee lineaire, Faux (defaut) pour une derive appliquee d'un coup.

    Returns
    -------
    pandas.DataFrame
        Copie des donnees, avec le capteur perturbe a partir de cycle_debut.

    Raises
    ------
    ValueError
        Si le type de derive est inconnu ou si le capteur n'existe pas.
    """
    if type_derive not in TYPES_DERIVE:
        raise ValueError(f"Type de derive inconnu : {type_derive!r} (possibles : pourcentage, decalage).")
    if capteur not in donnees_moteur.columns:
        raise ValueError(f"Capteur absent des donnees : {capteur}.")

    perturbees = donnees_moteur.copy()  # on ne modifie jamais les donnees de l'appelant
    cycles = perturbees["time_cycles"].to_numpy()
    valeurs = perturbees[capteur].to_numpy(dtype=float).copy()
    dernier_cycle = int(np.max(cycles))

    for position in range(len(cycles)):
        cycle = int(cycles[position])
        if cycle < cycle_debut:
            continue  # avant le debut : signal d'origine
        # part de l'amplitude appliquee a ce cycle (1 = amplitude complete)
        if progressive and dernier_cycle > cycle_debut:
            fraction = (cycle - cycle_debut) / (dernier_cycle - cycle_debut)
        else:
            fraction = 1.0
        if type_derive == "pourcentage":
            valeurs[position] = valeurs[position] * (1.0 + fraction * amplitude / 100.0)
        else:  # "decalage"
            valeurs[position] = valeurs[position] + fraction * amplitude

    perturbees[capteur] = valeurs
    return perturbees


def predire_trajectoire_brute(artefact, donnees_moteur):
    """
    Prepare des donnees brutes avec l'artefact et predit la RUL a chaque cycle (inference en direct).

    Parameters
    ----------
    artefact : dict
        L'artefact charge.
    donnees_moteur : pandas.DataFrame
        Les donnees brutes d'un moteur (format test_FD001.txt).

    Returns
    -------
    (pandas.DataFrame, float)
        Le tableau (time_cycles, rul_predite) et la duree preparation + prediction en millisecondes.
    """
    debut = time.perf_counter()
    donnees = preparer_donnees_brutes(donnees_moteur, artefact)  # meme preparation que le train
    predictions = predire_rul(artefact["modele"], donnees[artefact["feature_cols"]])  # clippees a 0
    temps_ms = (time.perf_counter() - debut) * 1000.0

    resultat = pd.DataFrame()
    resultat["time_cycles"] = donnees["time_cycles"].to_numpy()
    resultat["rul_predite"] = predictions
    return resultat, float(temps_ms)


def simuler_derive(artefact, donnees_moteur, rul_finale, capteur, type_derive, amplitude, cycle_debut,
                   progressive=False, seuil_alerte=SEUIL_ALERTE_DEFAUT):
    """
    Predit la RUL d'un moteur avec et sans derive injectee sur un capteur, et compare les deux.

    La RUL vraie ne change pas : la derive est artificielle, elle teste la sensibilite du MODELE a une
    entree modifiee, pas l'evolution physique du moteur.

    Parameters
    ----------
    artefact : dict
        L'artefact charge.
    donnees_moteur : pandas.DataFrame
        Les donnees brutes d'un moteur test.
    rul_finale : float
        La RUL vraie au dernier cycle observe de ce moteur (RUL_FD001.txt).
    capteur, type_derive, amplitude, cycle_debut, progressive :
        Voir `appliquer_derive`.
    seuil_alerte : float, optionnel
        Le seuil d'alerte en cycles.

    Returns
    -------
    dict
        - "tableau" : pandas.DataFrame par cycle : time_cycles, rul_vraie, rul_vraie_cappee,
          rul_predite_sans, rul_predite_avec, difference (avec - sans), capteur_sans, capteur_avec ;
        - "temps_ms_sans", "temps_ms_avec" : durees des deux inferences ;
        - "dernier" : dict au dernier cycle (rul_vraie, sans, avec, difference, statut_sans, statut_avec) ;
        - "premiere_alerte_sans", "premiere_alerte_avec" : (cycle, RUL vraie) ou None ;
        - "difference_max" : plus grand ecart absolu entre les deux predictions.
    """
    sans, temps_sans = predire_trajectoire_brute(artefact, donnees_moteur)
    perturbees = appliquer_derive(donnees_moteur, capteur, type_derive, amplitude, cycle_debut, progressive)
    avec, temps_avec = predire_trajectoire_brute(artefact, perturbees)

    # RUL vraie a chaque cycle : RUL au dernier cycle + cycles restant jusqu'au dernier (comme le contrat)
    cycles = sans["time_cycles"].to_numpy()
    dernier_cycle = int(np.max(cycles))
    rul_vraie = float(rul_finale) + (dernier_cycle - cycles)

    tableau = pd.DataFrame()
    tableau["time_cycles"] = cycles
    tableau["rul_vraie"] = rul_vraie
    tableau["rul_vraie_cappee"] = np.minimum(rul_vraie, artefact["plafond"])
    tableau["rul_predite_sans"] = sans["rul_predite"].to_numpy()
    tableau["rul_predite_avec"] = avec["rul_predite"].to_numpy()
    tableau["difference"] = tableau["rul_predite_avec"] - tableau["rul_predite_sans"]
    tableau["capteur_sans"] = donnees_moteur.sort_values("time_cycles")[capteur].to_numpy(dtype=float)
    tableau["capteur_avec"] = perturbees.sort_values("time_cycles")[capteur].to_numpy(dtype=float)

    # trajectoires au format du contrat, pour reutiliser premier_cycle_alerte
    trajectoire_sans = pd.DataFrame()
    trajectoire_sans["time_cycles"] = cycles
    trajectoire_sans["rul_predite"] = tableau["rul_predite_sans"].to_numpy()
    trajectoire_sans["rul_vraie"] = rul_vraie
    trajectoire_avec = trajectoire_sans.copy()
    trajectoire_avec["rul_predite"] = tableau["rul_predite_avec"].to_numpy()

    derniere_ligne = tableau.iloc[-1]
    alerte_sans, statut_sans = classer_alerte(derniere_ligne["rul_predite_sans"], derniere_ligne["rul_vraie"], seuil_alerte)
    alerte_avec, statut_avec = classer_alerte(derniere_ligne["rul_predite_avec"], derniere_ligne["rul_vraie"], seuil_alerte)
    dernier = {}
    dernier["cycle"] = dernier_cycle
    dernier["rul_vraie"] = float(derniere_ligne["rul_vraie"])
    dernier["sans"] = float(derniere_ligne["rul_predite_sans"])
    dernier["avec"] = float(derniere_ligne["rul_predite_avec"])
    dernier["difference"] = float(derniere_ligne["difference"])
    dernier["statut_sans"] = statut_sans
    dernier["statut_avec"] = statut_avec

    resultat = {}
    resultat["tableau"] = tableau
    resultat["temps_ms_sans"] = temps_sans
    resultat["temps_ms_avec"] = temps_avec
    resultat["dernier"] = dernier
    resultat["premiere_alerte_sans"] = premier_cycle_alerte(trajectoire_sans, seuil_alerte)
    resultat["premiere_alerte_avec"] = premier_cycle_alerte(trajectoire_avec, seuil_alerte)
    resultat["difference_max"] = float(np.max(np.abs(tableau["difference"].to_numpy())))
    return resultat


def donnees_test_fd001():
    """
    Charge les donnees test brutes FD001 et la RUL vraie au dernier cycle (fonction de src/demo.py).

    Returns
    -------
    (pandas.DataFrame, numpy.ndarray)
        test_FD001.txt (26 colonnes) et les 100 valeurs de RUL_FD001.txt (moteur 1 en premier).
    """
    return charger_donnees_test()


def mesurer_temps_reponse(artefact, chemin_artefact, repetitions=1, nb_moteurs=None, cible_ms=CIBLE_MS_DEFAUT):
    """
    Lance la mesure des temps de reponse de src/demo.py (`mesurer_benchmark`) sur peu de repetitions.

    Parameters
    ----------
    artefact : dict
        L'artefact charge.
    chemin_artefact : str
        Son fichier (taille et temps de chargement).
    repetitions : int, optionnel
        Nombre de passages sur les moteurs (1 par defaut : la mesure doit rester courte en demo).
    nb_moteurs : int ou None, optionnel
        Nombre de moteurs mesures (les premiers numeros) ; None pour tous.
    cible_ms : float, optionnel
        Temps cible pour un moteur (200 ms par defaut, exigence non fonctionnelle de src/demo.py).

    Returns
    -------
    dict
        Le resultat de `mesurer_benchmark` (p50_ms, p95_ms, max_ms, temps_moteur_ms, cible_atteinte...),
        plus "date" (heure de la mesure), "nb_moteurs" et "repetitions".
    """
    test_brut, rul_test = charger_donnees_test()
    moteurs = []
    for moteur_id in np.sort(test_brut["unit_number"].unique()):
        if nb_moteurs is not None and len(moteurs) >= nb_moteurs:
            break
        moteurs.append(int(moteur_id))
    resultats = mesurer_benchmark(
        artefact, chemin_artefact, test_brut, rul_test, repetitions=repetitions, cible_ms=cible_ms, moteurs=moteurs
    )
    resultats["date"] = datetime.datetime.now().isoformat(timespec="seconds")
    resultats["nb_moteurs"] = len(moteurs)
    resultats["repetitions"] = int(repetitions)
    return resultats


# ---------------------------------------------------------------------------------------------------
# Reproductibilite : nombre de tests, graines, commit
# ---------------------------------------------------------------------------------------------------

def analyser_sortie_collecte(texte):
    """
    Lit le nombre de tests dans la sortie de `pytest --collect-only -q` ("149 tests collected in 0.59s").

    Parameters
    ----------
    texte : str
        La sortie complete de pytest.

    Returns
    -------
    int ou None
        Le nombre de tests collectes, ou None si la ligne n'est pas trouvee.
    """
    correspondance = re.search(r"(\d+) tests? collected", texte)
    if correspondance is None:
        return None
    return int(correspondance.group(1))


def compter_tests_pytest(python=None, racine=RACINE_DEPOT, delai_max_s=300):
    """
    Compte les tests automatises du depot en lancant `python -m pytest --collect-only -q`.

    La collecte ne lance aucun test : elle les liste seulement (quelques secondes).

    Parameters
    ----------
    python : str ou None, optionnel
        L'interpreteur a utiliser (celui qui tourne par defaut).
    racine : str, optionnel
        Le dossier depuis lequel lancer pytest (racine du depot).
    delai_max_s : float, optionnel
        Duree maximale avant abandon (300 s).

    Returns
    -------
    dict
        "nb_tests" (int ou None), "date", "commande", "code_retour", "duree_s".
    """
    if python is None:
        python = sys.executable
    commande = [python, "-m", "pytest", "--collect-only", "-q"]
    debut = time.perf_counter()
    execution = subprocess.run(commande, cwd=racine, capture_output=True, text=True, timeout=delai_max_s)
    duree_s = time.perf_counter() - debut
    resultat = {}
    resultat["nb_tests"] = analyser_sortie_collecte(execution.stdout + execution.stderr)
    resultat["date"] = datetime.datetime.now().isoformat(timespec="seconds")
    resultat["commande"] = "python -m pytest --collect-only -q"
    resultat["code_retour"] = int(execution.returncode)
    resultat["duree_s"] = float(duree_s)
    return resultat


def informations_git(racine=RACINE_DEPOT):
    """
    Releve le commit courant et la presence de modifications locales (commandes git en lecture seule).

    Parameters
    ----------
    racine : str, optionnel
        La racine du depot.

    Returns
    -------
    dict
        "commit" (hash court ou None), "branche" (ou None), "modifications_locales" (bool ou None).
    """
    resultat = {}
    resultat["commit"] = None
    resultat["branche"] = None
    resultat["modifications_locales"] = None
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=racine, capture_output=True, text=True)
        branche = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=racine, capture_output=True, text=True)
        etat = subprocess.run(["git", "status", "--porcelain"], cwd=racine, capture_output=True, text=True)
    except OSError:
        return resultat  # git absent de la machine
    if commit.returncode == 0:
        resultat["commit"] = commit.stdout.strip()
    if branche.returncode == 0:
        resultat["branche"] = branche.stdout.strip()
    if etat.returncode == 0:
        resultat["modifications_locales"] = etat.stdout.strip() != ""
    return resultat


def ecrire_infos_reproductibilite(infos, chemin=None):
    """
    Ecrit les informations de reproductibilite (nombre de tests, date, commit...) en JSON.

    Parameters
    ----------
    infos : dict
        Les informations a ecrire.
    chemin : str ou None, optionnel
        Le fichier (sorties/demo/infos_reproductibilite.json par defaut).

    Returns
    -------
    str
        Le chemin ecrit.
    """
    if chemin is None:
        chemin = CHEMIN_INFOS_REPRODUCTIBILITE
    os.makedirs(os.path.dirname(chemin), exist_ok=True)
    with open(chemin, "w", encoding="utf-8") as fichier:
        json.dump(convertir_en_types_python(infos), fichier, ensure_ascii=False, indent=2)
        fichier.write("\n")
    return chemin


def lire_infos_reproductibilite(chemin=None):
    """
    Relit les informations de reproductibilite ecrites par scripts/preparer_demo.py.

    Parameters
    ----------
    chemin : str ou None, optionnel
        Le fichier (sorties/demo/infos_reproductibilite.json par defaut).

    Returns
    -------
    dict ou None
        Les informations, ou None si le fichier n'existe pas ou est illisible.
    """
    if chemin is None:
        chemin = CHEMIN_INFOS_REPRODUCTIBILITE
    if not os.path.isfile(chemin):
        return None
    try:
        with open(chemin, "r", encoding="utf-8") as fichier:
            return json.load(fichier)
    except (OSError, ValueError):
        return None


def est_un_nom_de_graine(nom):
    """
    Dit si un nom de cle designe une graine aleatoire ("random_state", "graine_kmeans", "seed"...).

    Les noms qui contiennent seulement le mot (ex. "rmse_moyenne_5_graines", une moyenne sur plusieurs
    graines) ne sont PAS des graines : on exige le mot entier, en debut ou en fin de nom.

    Parameters
    ----------
    nom : str
        Le nom de la cle.

    Returns
    -------
    bool
        Vrai pour "random_state", "graine", "graine_optuna", "kmeans_seed", "seed".
    """
    nom = str(nom).lower()
    for mot in ["random_state", "graine", "seed"]:
        if nom == mot or nom.startswith(mot + "_") or nom.endswith("_" + mot):
            return True
    return False


def chercher_graines(valeur, prefixe=""):
    """
    Cherche recursivement, dans un dictionnaire, les cles qui designent une graine aleatoire.

    Parameters
    ----------
    valeur : objet
        Un dictionnaire (eventuellement imbrique), une liste ou une valeur simple.
    prefixe : str, optionnel
        Chemin des cles parentes (pour l'affichage, ex. "hyperparametres.random_state").

    Returns
    -------
    list de (str, objet)
        Les couples (chemin de la cle, valeur) dont le nom est une graine (voir `est_un_nom_de_graine`)
        et dont la valeur est un nombre.
    """
    trouvees = []
    if isinstance(valeur, dict):
        for cle in valeur:
            if prefixe == "":  # cle de premier niveau
                chemin = str(cle)
            else:  # cle imbriquee : on garde le chemin complet
                chemin = f"{prefixe}.{cle}"
            if est_un_nom_de_graine(cle) and est_un_nombre(valeur[cle]):
                trouvees.append((chemin, valeur[cle]))
            for element in chercher_graines(valeur[cle], chemin):
                trouvees.append(element)
    elif isinstance(valeur, list):
        for position in range(len(valeur)):
            for element in chercher_graines(valeur[position], f"{prefixe}[{position}]"):
                trouvees.append(element)
    return trouvees


def lister_graines(inventaire=None, dossier=None):
    """
    Rassemble les graines aleatoires fixees dans le code et les fichiers du projet (rien n'est recopie).

    Sources : valeurs par defaut de `calculer_plis` (src/evaluation.py, lues par introspection),
    GRAINE_OPTUNA et GRAINE_MODELE (src/optimisation.py), la configuration retenue (JSON), et les
    metadonnees de chaque fichier de predictions disponible.

    Parameters
    ----------
    inventaire : pandas.DataFrame ou None, optionnel
        Le resultat de `inventaire_predictions` (None : pas de fichiers de predictions).
    dossier : str ou None, optionnel
        Dossier des predictions.

    Returns
    -------
    pandas.DataFrame
        Colonnes : source, parametre, valeur.
    """
    from src.evaluation import calculer_plis
    from src.optimisation import GRAINE_MODELE, GRAINE_OPTUNA

    lignes = []

    # graine des plis de validation croisee : valeur par defaut lue dans le code lui-meme
    signature = inspect.signature(calculer_plis)
    lignes.append(ligne_graine("src/evaluation.py (calculer_plis, GroupKFold)", "random_state",
                               signature.parameters["random_state"].default))

    # graines de l'optimisation (Phase 4)
    lignes.append(ligne_graine("src/optimisation.py", "GRAINE_OPTUNA", GRAINE_OPTUNA))
    lignes.append(ligne_graine("src/optimisation.py", "GRAINE_MODELE", GRAINE_MODELE))

    # graines du modele retenu
    config = lire_config_retenue()
    if config is not None:
        for chemin, valeur in chercher_graines(config):
            lignes.append(ligne_graine("docs/resultats/04_config_retenue.json", chemin, valeur))

    # graines ecrites dans les metadonnees de chaque fichier de predictions
    if inventaire is not None:
        for position in range(len(inventaire)):
            sous_jeu = inventaire["sous_jeu"].iloc[position]
            modele = inventaire["modele"].iloc[position]
            meta = lire_metadonnees(sous_jeu, modele, dossier)
            for chemin, valeur in chercher_graines(meta):
                lignes.append(ligne_graine(f"{sous_jeu}__{modele}.json", chemin, valeur))
    return pd.DataFrame(lignes, columns=["source", "parametre", "valeur"])


def ligne_graine(source, parametre, valeur):
    """
    Construit une ligne du tableau des graines.

    Parameters
    ----------
    source : str
        Le fichier ou le module ou la graine est fixee.
    parametre : str
        Le nom du parametre (ex. "random_state").
    valeur : objet
        La valeur de la graine.

    Returns
    -------
    dict
        {"source": ..., "parametre": ..., "valeur": texte}.
    """
    ligne = {}
    ligne["source"] = source
    ligne["parametre"] = parametre
    ligne["valeur"] = str(valeur)
    return ligne


# ---------------------------------------------------------------------------------------------------
# Limites connues : lues dans docs/STATUS.md, jamais inventees
# ---------------------------------------------------------------------------------------------------

def lire_section_markdown(texte, titre):
    """
    Extrait d'un texte Markdown le contenu d'une section "## <titre>" (jusqu'a la section suivante).

    Parameters
    ----------
    texte : str
        Le contenu Markdown.
    titre : str
        Le titre de la section, sans les "##" (ex. "Dernière modification").

    Returns
    -------
    str
        Le contenu de la section (sans sa ligne de titre), ou "" si la section n'existe pas.
    """
    lignes_section = []
    dans_la_section = False
    for ligne in texte.splitlines():
        if ligne.startswith("## "):  # titre de niveau 2 : debut d'une section
            if dans_la_section:
                break  # on a atteint la section suivante
            dans_la_section = ligne[3:].strip() == titre
            continue
        if ligne.startswith("### ") and dans_la_section:
            break  # une entree d'historique commence : fin de la section courante
        if dans_la_section:
            lignes_section.append(ligne)
    return "\n".join(lignes_section).strip()


def nettoyer_markdown(texte):
    """
    Retire la mise en forme Markdown simple (gras, liens, code) pour afficher une phrase en citation.

    Parameters
    ----------
    texte : str
        Une ligne Markdown (ex. "**Quoi** : voir [le fichier](docs/x.md)").

    Returns
    -------
    str
        Le texte sans **, sans ` et avec les liens remplaces par leur libelle.
    """
    sans_liens = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", texte)  # [libelle](url) -> libelle
    sans_gras = sans_liens.replace("**", "")
    sans_code = sans_gras.replace("`", "")
    return sans_code.strip()


def decouper_phrases(texte):
    """
    Decoupe un paragraphe en phrases (point suivi d'un espace puis d'une majuscule).

    Les nombres decimaux ("19.70", "0,94") ne sont pas coupes : il faut un espace apres le point.

    Parameters
    ----------
    texte : str
        Le paragraphe.

    Returns
    -------
    list de str
        Les phrases, dans l'ordre.
    """
    morceaux = re.split(r"(?<=[.!?])\s+(?=[A-ZÀ-ÖØ-Ý«(`*])", texte)
    phrases = []
    for morceau in morceaux:
        if morceau.strip() != "":
            phrases.append(morceau.strip())
    return phrases


def extraire_limites_status(chemin=None):
    """
    Extrait de docs/STATUS.md les phrases qui decrivent une limite ou un resultat non concluant.

    On ne garde que des phrases recopiees telles quelles (mise en forme retiree), qui contiennent un
    des mots de MOTS_CLES_LIMITES, dans les sections "Phase actuelle" et "Dernière modification".

    Parameters
    ----------
    chemin : str ou None, optionnel
        Le fichier STATUS.md (docs/STATUS.md par defaut).

    Returns
    -------
    dict
        - "phrases" : list de str, les limites trouvees (ordre du fichier, sans doublon) ;
        - "date" : str, la date de la derniere modification ("" si absente) ;
        - "sections" : dict titre -> texte Markdown complet de la section (pour un affichage detaille) ;
        - "present" : bool, Faux si le fichier n'existe pas.
    """
    if chemin is None:
        chemin = CHEMIN_STATUS
    resultat = {}
    resultat["phrases"] = []
    resultat["date"] = ""
    resultat["sections"] = {}
    resultat["present"] = os.path.isfile(chemin)
    if not resultat["present"]:
        return resultat

    with open(chemin, "r", encoding="utf-8") as fichier:
        texte = fichier.read()

    for titre in SECTIONS_STATUS:
        section = lire_section_markdown(texte, titre)
        resultat["sections"][titre] = section
        for ligne in section.splitlines():
            ligne_propre = nettoyer_markdown(ligne.lstrip(" -"))
            if ligne_propre.lower().startswith("date :"):  # "- **Date** : 2026-11-04"
                resultat["date"] = ligne_propre.split(":", 1)[1].strip()
            for phrase in decouper_phrases(ligne_propre):
                phrase_minuscule = phrase.lower()
                contient_mot_cle = False
                for mot in MOTS_CLES_LIMITES:
                    if mot in phrase_minuscule:
                        contient_mot_cle = True
                if contient_mot_cle and phrase not in resultat["phrases"]:
                    resultat["phrases"].append(phrase)
    return resultat
