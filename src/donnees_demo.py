"""
Couche d'acces aux donnees de l'application de demo (app/tableau_de_bord.py), sans Streamlit ni Plotly.

Pourquoi ce module existe : l'application ne doit contenir que de l'affichage. Ce qui lit un fichier ou
recalcule une metrique est ici, dans des fonctions simples, testees par tests/test_donnees_demo.py :

- fichiers de predictions precalcules (page "Comparer les modeles") : contrat `src/contrat_predictions.py` ;
- decision d'alerte (page "Tester un moteur") : `classer_alerte` de `src/demo.py`, libelles affiches ici ;
- signatures de fichiers : cles de cache de l'application (un fichier reecrit est relu automatiquement).

L'inference en direct elle-meme est dans `src/inference.py`. Aucun chiffre affiche par l'application n'est
ecrit en dur : ils sont lus dans les fichiers ou recalcules par ces fonctions.
"""

import inspect
import os

import pandas as pd

from src.contrat_predictions import (
    MODELES_CONNUS,
    SOUS_JEUX,
    dossier_par_defaut,
    lire_predictions,
    lister_predictions,
    metriques_dernier_cycle,
    verifier_coherence,
)
from src.demo import STATUT_ALERTE_CORRECTE, STATUT_ALERTE_MANQUEE, STATUT_FAUSSE_ALERTE, STATUT_RIEN_A_SIGNALER
from src.sous_jeux import DESCRIPTION_SOUS_JEUX

# ---------------------------------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------------------------------

# LE SEUL ENDROIT A MODIFIER pour choisir les modeles affiches par l'application (ordre d'affichage).
# "xgboost_retenu" n'y est plus : la Phase 4 a conclu de garder les reglages de la Phase 3 ; ses fichiers
# eventuels dans sorties/predictions/ sont donc ignores par la page "Comparer les modeles".
MODELES_AFFICHES = ["xgboost_phase3", "lstm", "cnn1d"]

# libelles affiches a l'ecran (avec accents) pour chaque statut de src/demo.py
LIBELLES_STATUTS = {
    STATUT_ALERTE_MANQUEE: "ALERTE MANQUÉE",
    STATUT_ALERTE_CORRECTE: "Alerte correcte",
    STATUT_FAUSSE_ALERTE: "Fausse alerte",
    STATUT_RIEN_A_SIGNALER: "Rien à signaler",
}

# explication courte de chaque statut (info-bulle des badges)
EXPLICATIONS_STATUTS = {
    STATUT_ALERTE_MANQUEE: "Moteur proche de la panne, pas d'alerte : le cas dangereux.",
    STATUT_ALERTE_CORRECTE: "Moteur proche de la panne, alerte émise.",
    STATUT_FAUSSE_ALERTE: "Alerte alors que le moteur a de la marge : maintenance trop tôt, sans risque.",
    STATUT_RIEN_A_SIGNALER: "Moteur avec de la marge, pas d'alerte.",
}

# colonnes du tableau de comparaison, dans l'ordre d'affichage (cle de metriques_test -> cle du tableau)
METRIQUES_COMPAREES = ["rmse", "score_nasa", "nb_surestimes", "alertes_manquees"]


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
        chemin_predictions, chemin_metadonnees), vide si aucun fichier. TOUS les modeles du contrat.
    """
    return lister_predictions(dossier)


def sous_jeux_disponibles(inventaire):
    """
    Renvoie les sous-jeux qui ont au moins un fichier d'un modele affiche, dans l'ordre de SOUS_JEUX.

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
    for sous_jeu in SOUS_JEUX:  # ordre officiel FD001 -> FD004
        if len(modeles_disponibles(inventaire, sous_jeu)) > 0:
            presents.append(sous_jeu)
    return presents


def modeles_disponibles(inventaire, sous_jeu):
    """
    Renvoie les modeles AFFICHES (MODELES_AFFICHES) qui ont un fichier pour ce sous-jeu, dans cet ordre.

    Parameters
    ----------
    inventaire : pandas.DataFrame
        Le resultat de `inventaire_predictions`.
    sous_jeu : str
        Le sous-jeu (ex. "FD001").

    Returns
    -------
    list de str
        Les cles de modeles (ex. ["xgboost_phase3", "lstm"]) ; les autres modeles du contrat sont ignores.
    """
    lignes = inventaire[inventaire["sous_jeu"] == sous_jeu]
    valeurs = lignes["modele"].tolist()
    presents = []
    for modele in MODELES_AFFICHES:  # filtre explicite ET ordre d'affichage
        if modele in valeurs:
            presents.append(modele)
    return presents


def nom_modele(modele):
    """
    Donne le nom lisible d'un modele (ex. "xgboost_phase3" -> "XGBoost Phase 3").

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


def description_sous_jeu(sous_jeu):
    """
    Une ligne de description d'un sous-jeu : regimes de vol, modes de panne, moteurs (src/sous_jeux.py).

    Parameters
    ----------
    sous_jeu : str
        "FD001" a "FD004".

    Returns
    -------
    str
        Par exemple "FD001 — 1 régime de vol, 1 mode de panne (HPC), 100 moteurs test".
    """
    description = DESCRIPTION_SOUS_JEUX[sous_jeu]
    nb_regimes = int(description["nb_regimes"])
    modes = []
    for mode in description["modes_de_panne"]:
        modes.append(mode.replace(" Degradation", ""))  # "HPC Degradation" -> "HPC"
    texte_regimes = f"{nb_regimes} régime de vol"
    if nb_regimes > 1:
        texte_regimes = f"{nb_regimes} régimes de vol"
    texte_modes = f"{len(modes)} mode de panne"
    if len(modes) > 1:
        texte_modes = f"{len(modes)} modes de panne"
    return (f"{sous_jeu} — {texte_regimes}, {texte_modes} ({', '.join(modes)}), "
            f"{description['nb_moteurs_test']} moteurs test")


# ---------------------------------------------------------------------------------------------------
# Comparaison des modeles d'un sous-jeu (metriques de test au dernier cycle)
# ---------------------------------------------------------------------------------------------------

def seuils_metriques():
    """
    Lit les seuils utilises par les metriques de test (valeurs par defaut de `metriques_dernier_cycle`).

    Returns
    -------
    dict
        "seuil_surestimation" et "seuil_alerte" (cycles), lus dans le code du contrat (rien n'est recopie).
    """
    parametres = inspect.signature(metriques_dernier_cycle).parameters
    seuils = {}
    seuils["seuil_surestimation"] = parametres["seuil_surestimation"].default
    seuils["seuil_alerte"] = parametres["seuil_alerte"].default
    return seuils


def tableau_comparaison_sous_jeu(inventaire, sous_jeu, dossier=None):
    """
    Une ligne par modele affiche du sous-jeu : RMSE, score NASA, surestimes, alertes manquees (test).

    Les metriques sont celles des metadonnees (calculees au dernier cycle observe de chaque moteur test,
    voir le contrat) ; `controle_coherent` dit si elles se recalculent a l'identique depuis le fichier.

    Parameters
    ----------
    inventaire : pandas.DataFrame
        Le resultat de `inventaire_predictions`.
    sous_jeu : str
        Le sous-jeu.
    dossier : str ou None, optionnel
        Dossier des predictions ; None pour le dossier par defaut.

    Returns
    -------
    pandas.DataFrame
        Colonnes : modele, nom_affiche, nb_moteurs, rmse, score_nasa, nb_surestimes, alertes_manquees,
        controle_coherent. Vide si aucun fichier.
    """
    colonnes = ["modele", "nom_affiche", "nb_moteurs"]
    for metrique in METRIQUES_COMPAREES:
        colonnes.append(metrique)
    colonnes.append("controle_coherent")

    lignes = []
    for modele in modeles_disponibles(inventaire, sous_jeu):
        df, meta = lire_predictions(sous_jeu, modele, dossier)
        metriques = meta["metriques_test"]
        ligne = {}
        ligne["modele"] = modele
        ligne["nom_affiche"] = meta["nom_affiche"]
        ligne["nb_moteurs"] = int(df["unit_number"].nunique())
        for metrique in METRIQUES_COMPAREES:
            ligne[metrique] = float(metriques[metrique])
        ligne["controle_coherent"] = controler_coherence(df, meta)["coherent"]
        lignes.append(ligne)
    return pd.DataFrame(lignes, columns=colonnes)


def controler_coherence(df, metadonnees):
    """
    Recalcule les metriques de test depuis le fichier et les compare aux metadonnees.

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
