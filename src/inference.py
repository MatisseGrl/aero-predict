"""
Inference EN DIRECT : la RUL d'UN moteur, au cycle t, calculee a partir de ses seules mesures brutes 1..t.

Pourquoi ce module ? Les fichiers sorties/predictions/ contiennent des predictions calculees a l'avance pour
tous les cycles de tous les moteurs test. Pour une demonstration, on veut au contraire montrer le modele en
train de travailler : on lui donne un moteur qu'il n'a jamais vu (jeu de TEST), les mesures arrivent cycle
par cycle, et a chaque cycle t on prepare les mesures 1..t avec ce qui a ete appris sur le TRAIN, puis on
predit. Rien n'est lu dans un fichier de predictions, rien n'est reappris.

Interface publique (utilisee par l'application et par le notebook 05) :

- `charger_modeles(sous_jeu)` : prepare le sous-jeu (parametres appris sur train_FD00X : regimes, capteurs
  retires, min / max des capteurs), recharge les modeles a une ligne par cycle (reference naive, Random Forest,
  XGBoost, LightGBM : models/<SOUS_JEU>__<modele>.joblib, entraines et sauvegardes s'ils manquent,
  exactement comme les modeles finaux de scripts/generer_predictions_arbres.py) et les deux reseaux du
  notebook 04h (models/04h/<SOUS_JEU>__<lstm|cnn1d>_g<graine>.pt, reglages Optuna confirmes ; le LSTM est la
  moyenne de 5 reseaux, graines 42 a 46, le modele retenu ; jamais reentraines ici : un fichier absent rend
  seulement le modele indisponible, avec un message clair) ;
- `mesures_moteur_test(sous_jeu, moteur)` et `tirer_moteur_test(sous_jeu, graine)` : les mesures brutes d'un
  moteur du jeu de test, et sa RUL vraie au dernier cycle (RUL_FD00X.txt) ;
- `lire_fichier_mesures(contenu)` : lit et valide un fichier de mesures importe (format C-MAPSS brut ou CSV) ;
- `predire_au_cycle(inference, mesures, cycle)` : la RUL predite par chaque modele au cycle t, et le temps
  de calcul en millisecondes ;
- `predire_trajectoire(inference, mesures)` : `predire_au_cycle` repete a chaque cycle (une troncature par cycle) ;
- `verite_moteur(mesures, rul_finale)` : la RUL vraie a chaque cycle (connue apres coup seulement) ;
- `ecarts_aux_predictions_precalculees(...)` : compare l'inference en direct aux fichiers precalcules.

Calcul sur un seul coeur. Charger PyTorch et XGBoost / LightGBM dans le meme processus a deja bloque la suite
de tests (deux bibliotheques OpenMP de calcul parallele, attente infinie dans __kmp_join_barrier, voir
src/deep_learning.py). Ici : (1) xgboost est toujours importe AVANT torch (ordre des imports ci-dessous) ;
(2) PyTorch calcule sur 1 coeur, avec 1 fil "inter-operations" (`configurer_calcul_un_coeur`) ; (3) les modeles a
arbres predisent sur 1 coeur (`n_jobs=1` apres chargement : une prediction ligne a ligne ne change pas avec le nombre
de coeurs). Aucune region parallele OpenMP n'est donc ouverte par
l'inference en direct. L'application fixe en plus OMP_NUM_THREADS=1 au demarrage (app/tableau_de_bord.py).
"""

import datetime
import importlib.util
import os
import pickle
import platform
import random
import time

import cloudpickle
import joblib
import numpy as np
import pandas as pd
import lightgbm
import sklearn
import xgboost

# ORDRE DES IMPORTS VOULU (ne pas le "ranger") : les modules du projet chargent xgboost et lightgbm (donc leur
# bibliotheque OpenMP) AVANT PyTorch ; dans l'ordre inverse, un XGBoost multi-coeurs lance ensuite dans le
# meme processus peut planter (constate le 2026-11-04, voir src/deep_learning.py).
from src.config_modele import RACINE_DEPOT, construire_fabrique_depuis_config, convertir_en_types_python, decrire_config
from src.contrat_predictions import MODELES_CONNUS, SOUS_JEUX, calculer_rul_vraie, lire_predictions
from src.features import ajouter_ecart_etat_initial, construire_liste_features
from src.preprocessing import COLUMNS
from src.sous_jeux import appliquer_preparation, charger_rul_sous_jeu, charger_sous_jeu, preparer_sous_jeu

# src.deep_learning importe torch (apres xgboost, voir ci-dessus) et regle deja PyTorch sur 1 coeur
from src.deep_learning import (  # noqa: E402
    NOM_COLONNE_TIME_CYCLES_ECHELLE,
    ajouter_time_cycles_echelle,
    appliquer_min_max,
    charger_modele,
    construire_fenetres,
    predire,
)
from src.optimisation_reseaux import ENSEMBLES_RETENUS  # noqa: E402  (reseaux retenus au notebook 04h)

import torch  # noqa: E402  (apres les modules du projet, voir ci-dessus)

# ---------------------------------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------------------------------

# les six modeles de la demo, dans l'ordre d'affichage (memes cles et meme ordre que src/contrat_predictions.py)
# modeles a une ligne de features par cycle (scripts/generer_predictions_arbres.py) : reentraines ici s'ils manquent
MODELES_ARBRES = ["reference_naive", "random_forest", "xgboost_phase3", "lightgbm"]
MODELES_RESEAUX = ["lstm", "cnn1d"]  # jamais reentraines ici (poids produits par scripts/sauvegarder_poids_04h.py)

# Reseaux de la demo : ceux du notebook 04h (reglages Optuna confirmes), dans models/04h/. Le LSTM affiche est la
# moyenne simple des 5 graines retenues (ENSEMBLES_RETENUS, src/optimisation_reseaux.py) ; le CNN 1D est la
# configuration retenue avec la graine 42 (celle que la regle de confirmation a comparee). Les anciens reseaux du
# notebook 03b ne sont plus utilises (archives dans models/archive_03b/ et sorties/predictions_archive_03b/).
SOUS_DOSSIER_RESEAUX = "04h"
GRAINES_CNN_DEMO = [42]
MODELES_INFERENCE = []
for _cle in MODELES_ARBRES:
    MODELES_INFERENCE.append(_cle)
for _cle in MODELES_RESEAUX:
    MODELES_INFERENCE.append(_cle)

# dossier des modeles entraines (fichiers ignores par git) ; une variable d'environnement peut le remplacer
# (utile aux tests automatiques, qui ecrivent des modeles legers dans un dossier temporaire)
DOSSIER_MODELES_DEFAUT = os.path.join(RACINE_DEPOT, "models")
VARIABLE_ENV_DOSSIER_MODELES = "AERO_PREDICT_DOSSIER_MODELES"

# le script qui produit les predictions precalculees des arbres : on lui emprunte sa recette d'entrainement
CHEMIN_SCRIPT_ARBRES = os.path.join(RACINE_DEPOT, "scripts", "generer_predictions_arbres.py")

# format des artefacts des modeles a arbres : a incrementer si leur contenu change (un ancien sera reentraine)
VERSION_FORMAT_ARTEFACT = 1

# limites d'un fichier de mesures importe (protection contre un fichier enorme ou qui n'est pas un moteur)
TAILLE_MAX_FICHIER_OCTETS = 1_000_000  # 1 Mo : un moteur C-MAPSS fait au plus quelques dizaines de ko
NB_MAX_CYCLES_FICHIER = 1000  # le plus long moteur des quatre jeux d'entrainement a 543 cycles

# tolerance des comparaisons d'empreinte de preparation (reels recalcules : K-Means peut varier au dernier chiffre)
TOLERANCE_EMPREINTE = 1e-9

# Les reseaux calculent en float32. Le meme reseau, sur la meme fenetre, ne donne pas exactement le meme
# resultat au dernier chiffre selon le NOMBRE de fenetres calculees ensemble : pour un petit lot, la
# bibliotheque de calcul de PyTorch decoupe autrement les multiplications, donc additionne dans un autre
# ordre. Mesure du 2026-11-04 sur cette machine (500 fenetres par sous-jeu et par reseau) : une fenetre
# seule s'ecarte des fichiers precalcules (calcules par lots de 2048) de 9e-5 cycle au plus ; des lots de 32
# ou 48 fenetres, de 1,5e-5 au plus ; des lots de 64 fenetres ou plus donnent le meme resultat (ecart de
# 3e-14, l'arrondi d'ecriture du CSV). On calcule donc la fenetre du cycle t dans un lot de TAILLE_LOT_RESEAU
# COPIES D'ELLE-MEME : aucune autre donnee n'entre dans le calcul (ni autre moteur, ni futur) et le resultat
# est exactement celui du calcul en lot (verifie par tests/test_anti_triche.py). Cout mesure : 2,5 ms (LSTM)
# et 13 ms (CNN 1D) par cycle, au lieu de 0,4 et 0,2 ms pour une fenetre seule.
TAILLE_LOT_RESEAU = 64

# les tests automatiques qui prouvent l'absence de triche (noms affiches par l'application)
TEST_EGALITE_PRECALCULE = "tests/test_anti_triche.py::test_direct_egal_precalcule"
TEST_CYCLES_FUTURS = "tests/test_anti_triche.py::test_cycles_futurs_sans_effet"

# cache du module "script des arbres" (charge une seule fois)
_MODULE_SCRIPT_ARBRES = None


class ErreurFichierMesures(ValueError):
    """
    Fichier de mesures importe invalide : le message (en francais) dit quoi corriger.

    C'est une ValueError : le code qui attrape deja les ValueError l'attrape aussi.
    """


# ---------------------------------------------------------------------------------------------------
# Calcul sur un seul coeur (torch + xgboost dans le meme processus, sans blocage)
# ---------------------------------------------------------------------------------------------------

def configurer_calcul_un_coeur():
    """
    Regle PyTorch sur 1 coeur de calcul et 1 fil "inter-operations", et renvoie les reglages en vigueur.

    `torch.set_num_threads(1)` : aucune operation PyTorch n'ouvre de region parallele OpenMP (c'est ce qui
    bloquait quand xgboost avait deja calcule dans le meme processus). `torch.set_num_interop_threads(1)` :
    pas de reserve de fils pour executer plusieurs operations en meme temps (PyTorch refuse ce reglage s'il a
    deja ete fait ou si un calcul parallele a deja commence : on garde alors la valeur en place).

    Returns
    -------
    dict
        "torch_coeurs", "torch_fils_inter_operations" (entiers) et "omp_num_threads" (texte ou None).
    """
    torch.set_num_threads(1)  # calculs a l'interieur d'une operation : 1 coeur
    try:
        torch.set_num_interop_threads(1)  # calculs de plusieurs operations en parallele : 1 fil
    except RuntimeError:
        pass  # deja regle (ou deja utilise) dans ce processus : PyTorch interdit de le changer ensuite
    reglages = {}
    reglages["torch_coeurs"] = int(torch.get_num_threads())
    reglages["torch_fils_inter_operations"] = int(torch.get_num_interop_threads())
    reglages["omp_num_threads"] = os.environ.get("OMP_NUM_THREADS")
    return reglages


configurer_calcul_un_coeur()  # des l'import du module


def regler_un_coeur(modele):
    """
    Fait predire un modele a arbres (XGBoost, LightGBM, Random Forest, eventuellement enveloppe dans
    RegresseurCentre) sur un seul coeur ; sans effet sur un modele sans reglage n_jobs (reference naive).

    Pour XGBoost et LightGBM, la prediction d'une ligne est une somme sur les arbres, toujours dans le meme
    ordre : elle ne depend pas du nombre de coeurs (n_jobs ne fait que repartir les LIGNES entre coeurs). Pour
    la Random Forest, 1 coeur additionne les arbres dans un ordre fixe. Pour une seule ligne par cycle,
    plusieurs coeurs n'apportent rien et ouvriraient une region de calcul parallele.

    Parameters
    ----------
    modele : estimateur
        Le modele, ou RegresseurCentre (src/modeles.py) dont l'attribut `modele_` est le vrai modele.
        Modifie sur place.

    Returns
    -------
    estimateur
        Le meme objet (pour enchainer les appels).
    """
    interieur = modele
    if hasattr(modele, "modele_"):  # RegresseurCentre : le vrai modele est dans modele_
        interieur = modele.modele_
    if hasattr(interieur, "set_params") and "n_jobs" in interieur.get_params():
        interieur.set_params(n_jobs=1)  # pour XGBoost, met aussi a jour le "booster" interne
    return modele


# ---------------------------------------------------------------------------------------------------
# Emplacements des fichiers
# ---------------------------------------------------------------------------------------------------

def dossier_modeles(dossier=None):
    """
    Renvoie le dossier des modeles entraines : celui donne, sinon la variable d'environnement, sinon models/.

    Parameters
    ----------
    dossier : str ou None, optionnel
        Un dossier explicite (prioritaire).

    Returns
    -------
    str
        Le chemin du dossier.
    """
    if dossier is not None:
        return str(dossier)
    valeur = os.environ.get(VARIABLE_ENV_DOSSIER_MODELES, "")
    if valeur != "":
        return valeur
    return DOSSIER_MODELES_DEFAUT


def chemin_artefact_arbres(sous_jeu, modele, dossier=None):
    """
    Chemin de l'artefact d'un modele a arbres : <dossier des modeles>/<SOUS_JEU>__<modele>.joblib.

    Parameters
    ----------
    sous_jeu : str
        "FD001" a "FD004".
    modele : str
        Un modele de MODELES_ARBRES (ex. "xgboost_phase3").
    dossier : str ou None, optionnel
        Dossier des modeles (None : `dossier_modeles()`).

    Returns
    -------
    str
        Le chemin du fichier .joblib.
    """
    return os.path.join(dossier_modeles(dossier), f"{sous_jeu}__{modele}.joblib")


def graines_reseau(sous_jeu, modele):
    """
    Graines des reseaux moyennes par la demo : les 5 du LSTM retenu (ENSEMBLES_RETENUS), la graine 42 du CNN.

    Parameters
    ----------
    sous_jeu : str
        "FD001" a "FD004".
    modele : str
        "lstm" ou "cnn1d".

    Returns
    -------
    list de int
        Les graines.
    """
    if modele == "lstm" and sous_jeu in ENSEMBLES_RETENUS:
        return list(ENSEMBLES_RETENUS[sous_jeu]["graines"])
    return list(GRAINES_CNN_DEMO)


def chemins_poids_reseau(sous_jeu, modele, dossier=None):
    """
    Chemins des poids d'un reseau de la demo : <dossier des modeles>/04h/<SOUS_JEU>__<modele>_g<graine>.pt,
    un fichier par graine (src.deep_learning.sauvegarder_modele, scripts/sauvegarder_poids_04h.py).

    Parameters
    ----------
    sous_jeu : str
        "FD001" a "FD004".
    modele : str
        "lstm" ou "cnn1d".
    dossier : str ou None, optionnel
        Dossier des modeles (None : `dossier_modeles()`).

    Returns
    -------
    list de str
        Les chemins des fichiers .pt.
    """
    chemins = []
    for graine in graines_reseau(sous_jeu, modele):
        chemins.append(os.path.join(dossier_modeles(dossier), SOUS_DOSSIER_RESEAUX, f"{sous_jeu}__{modele}_g{graine}.pt"))
    return chemins


def chemin_poids_reseau(sous_jeu, modele, dossier=None):
    """
    Chemin du premier fichier de poids d'un reseau de la demo (voir `chemins_poids_reseau`).

    Parameters
    ----------
    sous_jeu, modele, dossier :
        Comme `chemins_poids_reseau`.

    Returns
    -------
    str
        Le chemin du fichier .pt de la premiere graine.
    """
    return chemins_poids_reseau(sous_jeu, modele, dossier)[0]


def chemin_lisible(chemin):
    """
    Raccourcit un chemin pour les messages : relatif a la racine du depot s'il est dedans.

    Parameters
    ----------
    chemin : str
        Le chemin.

    Returns
    -------
    str
        Par exemple "models/FD001__lstm.pt".
    """
    chemin_absolu = os.path.abspath(chemin)
    if chemin_absolu.startswith(RACINE_DEPOT + os.sep):
        return os.path.relpath(chemin_absolu, RACINE_DEPOT)
    return chemin_absolu


def verifier_sous_jeu(sous_jeu):
    """
    Verifie qu'un nom de sous-jeu est connu.

    Parameters
    ----------
    sous_jeu : str
        Le nom (ex. "FD002").

    Raises
    ------
    ValueError
        Si le sous-jeu est inconnu.
    """
    if sous_jeu not in SOUS_JEUX:
        raise ValueError(f"Sous-jeu inconnu : {sous_jeu!r} (possibles : {', '.join(SOUS_JEUX)}).")


# ---------------------------------------------------------------------------------------------------
# Preparation apprise sur le train
# ---------------------------------------------------------------------------------------------------

def module_script_arbres():
    """
    Charge (une fois) le module scripts/generer_predictions_arbres.py, pour reutiliser sa recette d'entrainement.

    Pourquoi ? Les predictions precalculees des arbres viennent de ce script (configuration de chaque modele,
    nombre de coeurs, fenetre, plafond). Lire la configuration DANS le script garantit que l'artefact de ce
    module est entraine exactement de la meme facon (le test d'egalite le verifie). Le chargement par chemin
    explicite evite toute confusion avec un autre paquet nomme "scripts".

    Returns
    -------
    module
        Le module Python du script (sa fonction main n'est pas lancee).
    """
    global _MODULE_SCRIPT_ARBRES
    if _MODULE_SCRIPT_ARBRES is None:
        specification = importlib.util.spec_from_file_location("generer_predictions_arbres", CHEMIN_SCRIPT_ARBRES)
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)  # execute seulement les imports et les constantes du script
        _MODULE_SCRIPT_ARBRES = module
    return _MODULE_SCRIPT_ARBRES


def configuration_arbres(modele):
    """
    Configuration du modele FINAL d'un modele de MODELES_ARBRES, telle que l'utilise scripts/generer_predictions_arbres.py.

    Reglages finaux des notebooks 04f et 04g si le jeu ecart_initial_20 est retenu, reglages de base sinon
    (`configuration_version_retenue` du script) : la demo entraine le meme modele que les fichiers du contrat.

    Parameters
    ----------
    modele : str
        Un modele de MODELES_ARBRES (configurations de src/config_modele.py).

    Returns
    -------
    dict
        La configuration (format de src/config_modele.py), nombre de coeurs du script compris.

    Raises
    ------
    ValueError
        Si le choix des features est introuvable (script des arbres pas encore lance).
    """
    script = module_script_arbres()
    try:
        choix = script.lire_choix_fd001()  # jeu de features retenu sur FD001, modele par modele
    except FileNotFoundError as erreur:
        raise ValueError(str(erreur)) from erreur
    return script.configuration_version_retenue(modele, choix[modele], n_jobs=script.N_JOBS_DEFAUT)


def preparer_contexte(sous_jeu):
    """
    Apprend la preparation sur train_FD00X (`preparer_sous_jeu`) avec les reglages du script des arbres.

    Les colonnes d'ecart a l'etat initial (jeu candidat du script) sont toujours calculees, en plus des
    features de la Phase 3 : elles s'ajoutent apres les autres colonnes et ne changent aucune d'elles.
    Chaque modele a arbres prend ensuite SES colonnes (`colonnes_features_arbres`), celles de la version
    retenue par le script ; les reseaux ne lisent que les capteurs lisses, inchanges.

    Parameters
    ----------
    sous_jeu : str
        "FD001" a "FD004".

    Returns
    -------
    dict
        Le contexte complet de `preparer_sous_jeu` (train prepare et y compris : utiles pour entrainer un XGBoost).
    """
    verifier_sous_jeu(sous_jeu)
    script = module_script_arbres()
    ecart_initial = script.JEUX_FEATURES[script.JEU_CANDIDAT]  # k du jeu candidat (20)
    return preparer_sous_jeu(sous_jeu, fenetre=script.FENETRE, plafond=script.PLAFOND_RUL, ecart_initial=ecart_initial)


def colonnes_features_arbres(contexte, modele):
    """
    Colonnes de features d'un modele a arbres : celles de la version retenue par le script des arbres.

    Le choix (jeu de base ou ecart a l'etat initial) est fait sur FD001 par le script et relu dans
    sorties/resultats/05_arbres_choix_features.csv (`lire_choix_fd001`), pour que l'inference en direct
    utilise exactement les colonnes des predictions precalculees.

    Parameters
    ----------
    contexte : dict
        Le contexte COMPLET de `preparer_contexte`.
    modele : str
        Un modele de MODELES_ARBRES.

    Returns
    -------
    list de str
        Les colonnes, dans l'ordre du train.

    Raises
    ------
    ValueError
        Si le choix des features est introuvable (script des arbres pas encore lance) ou incoherent.
    """
    script = module_script_arbres()
    try:
        choix = script.lire_choix_fd001()
    except FileNotFoundError as erreur:
        raise ValueError(str(erreur)) from erreur
    ecart_initial = script.JEUX_FEATURES[choix[modele]]

    if ecart_initial is None:
        # jeu de base : features de la Phase 3, dans l'ordre de preparer_sous_jeu (one-hot du regime a la fin)
        colonnes = construire_liste_features(contexte["capteurs"], contexte["colonnes_moyenne"],
                                             contexte["colonnes_ecart_type"])
        for colonne in contexte["colonnes_regime"]:
            colonnes.append(colonne)
        return colonnes

    # jeu candidat : c'est la liste du contexte, a condition qu'il ait ete prepare avec le meme k
    if contexte["ecart_initial"] != ecart_initial:
        raise ValueError(f"Contexte prepare avec ecart_initial={contexte['ecart_initial']}, attendu {ecart_initial}.")
    return list(contexte["feature_cols"])


def contexte_leger(contexte):
    """
    Garde du contexte complet seulement ce qui sert a preparer de NOUVELLES mesures, plus quelques faits.

    Les grands tableaux (train et test prepares) sont laisses de cote : l'inference en direct n'en a pas
    besoin, et l'application garde ce contexte en memoire pour toute la session.

    Parameters
    ----------
    contexte : dict
        Le resultat de `preparer_sous_jeu`.

    Returns
    -------
    dict
        Cles lues par `appliquer_preparation` (kmeans, scalers, capteurs, capteurs_retires, fenetre,
        nb_regimes, regime_en_feature, ecart_initial), plus sous_jeu, normalisation, plafond, feature_cols,
        nb_lignes_train, nb_moteurs_train, nb_lignes_vues_par_scaler (lignes du train vues par chaque scaler).
    """
    leger = {}
    for cle in ["sous_jeu", "nb_regimes", "normalisation", "regime_en_feature", "kmeans", "scalers",
                "capteurs_retires", "fenetre", "plafond"]:
        leger[cle] = contexte[cle]
    leger["ecart_initial"] = contexte.get("ecart_initial")  # colonnes d'ecart a l'etat initial (None : aucune)
    leger["capteurs"] = list(contexte["capteurs"])
    leger["feature_cols"] = list(contexte["feature_cols"])
    leger["nb_lignes_train"] = int(len(contexte["X"]))
    leger["nb_moteurs_train"] = int(contexte["groups"].nunique())

    # nombre de lignes du TRAIN vues par chaque scaler (preuve que les min / max viennent du train)
    vues = {}
    for regime in contexte["scalers"]:
        vues[int(regime)] = int(contexte["scalers"][regime].n_samples_seen_)
    leger["nb_lignes_vues_par_scaler"] = vues
    return leger


def empreinte_preparation(contexte, feature_cols=None):
    """
    Resume en types simples ce que la preparation a appris sur le train (pour reconnaitre un artefact perime).

    Parameters
    ----------
    contexte : dict
        Un contexte complet ou leger.
    feature_cols : list de str ou None, optionnel
        Les colonnes du modele (None : celles du contexte).

    Returns
    -------
    dict
        capteurs, capteurs_retires, feature_cols, normalisation, nb_regimes, regime_en_feature, fenetre,
        plafond, min et max de chaque scaler, centres des regimes (None si un seul regime).
    """
    empreinte = {}
    empreinte["capteurs"] = list(contexte["capteurs"])
    empreinte["capteurs_retires"] = list(contexte["capteurs_retires"])
    if feature_cols is None:
        feature_cols = contexte["feature_cols"]
    empreinte["feature_cols"] = list(feature_cols)  # les colonnes que le modele lit, dans l'ordre
    empreinte["normalisation"] = contexte["normalisation"]
    empreinte["nb_regimes"] = int(contexte["nb_regimes"])
    empreinte["regime_en_feature"] = bool(contexte["regime_en_feature"])
    empreinte["fenetre"] = int(contexte["fenetre"])
    empreinte["plafond"] = int(contexte["plafond"])
    minimums = {}
    maximums = {}
    for regime in contexte["scalers"]:
        minimums[str(int(regime))] = convertir_en_types_python(contexte["scalers"][regime].data_min_)
        maximums[str(int(regime))] = convertir_en_types_python(contexte["scalers"][regime].data_max_)
    empreinte["minimums"] = minimums
    empreinte["maximums"] = maximums
    if contexte["kmeans"] is None:
        empreinte["centres_regimes"] = None
    else:
        empreinte["centres_regimes"] = convertir_en_types_python(contexte["kmeans"].cluster_centers_)
    return empreinte


def valeurs_proches(a, b):
    """
    Compare deux valeurs en types simples (dict, list, nombre, texte), les reels a TOLERANCE_EMPREINTE pres.

    Parameters
    ----------
    a, b : object
        Les deux valeurs.

    Returns
    -------
    bool
        Vrai si elles sont egales (reels : ecart relatif ou absolu <= TOLERANCE_EMPREINTE).
    """
    if isinstance(a, dict) and isinstance(b, dict):
        if sorted(a.keys()) != sorted(b.keys()):
            return False
        for cle in a:
            if not valeurs_proches(a[cle], b[cle]):
                return False
        return True
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        for position in range(len(a)):
            if not valeurs_proches(a[position], b[position]):
                return False
        return True
    if isinstance(a, bool) or isinstance(b, bool):  # bool avant les nombres (True vaut 1 en Python)
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return bool(np.isclose(float(a), float(b), rtol=TOLERANCE_EMPREINTE, atol=TOLERANCE_EMPREINTE))
    return a == b


# ---------------------------------------------------------------------------------------------------
# Les modeles a arbres : entrainement, sauvegarde, chargement
# ---------------------------------------------------------------------------------------------------

def entrainer_modele_arbres(contexte, modele):
    """
    Entraine un modele a arbres sur TOUT le train du sous-jeu, comme le modele final du script des arbres.

    Meme recette que `evaluer_version_finale` (scripts/generer_predictions_arbres.py, etape 1) :
    configuration du script, modele neuf, fit sur les colonnes de la version retenue (`colonnes_features_arbres`)
    et y du train (RUL plafonnee).

    Parameters
    ----------
    contexte : dict
        Le contexte COMPLET de `preparer_contexte` (avec X et y).
    modele : str
        Un modele de MODELES_ARBRES (ex. "xgboost_phase3").

    Returns
    -------
    dict
        L'artefact : "version_format", "sous_jeu", "cle_modele", "modele" (l'estimateur entraine), "config",
        "description_modele", "empreinte_preparation", "fichier_entrainement", "nb_lignes_train",
        "nb_moteurs_train", "date_entrainement", "duree_entrainement_s", "versions".

    Raises
    ------
    ValueError
        Si le modele n'est pas dans MODELES_ARBRES.
    """
    if modele not in MODELES_ARBRES:
        raise ValueError(f"Modele a arbres inconnu : {modele!r} (possibles : {', '.join(MODELES_ARBRES)}).")
    config = configuration_arbres(modele)
    colonnes = colonnes_features_arbres(contexte, modele)  # jeu de features retenu par le script
    estimateur = construire_fabrique_depuis_config(config)()  # un modele neuf, non entraine
    debut = time.perf_counter()
    estimateur.fit(contexte["train"][colonnes], contexte["y"])  # tout le train du sous-jeu
    duree = time.perf_counter() - debut

    artefact = {}
    artefact["version_format"] = VERSION_FORMAT_ARTEFACT
    artefact["sous_jeu"] = contexte["sous_jeu"]
    artefact["cle_modele"] = modele
    artefact["modele"] = estimateur
    artefact["config"] = convertir_en_types_python(config)
    artefact["description_modele"] = decrire_config(config)
    artefact["empreinte_preparation"] = empreinte_preparation(contexte, colonnes)
    artefact["fichier_entrainement"] = f"data/train_{contexte['sous_jeu']}.txt"
    artefact["nb_lignes_train"] = int(len(contexte["X"]))
    artefact["nb_moteurs_train"] = int(contexte["groups"].nunique())
    artefact["date_entrainement"] = datetime.datetime.now().isoformat(timespec="seconds")
    artefact["duree_entrainement_s"] = float(duree)
    artefact["versions"] = lire_versions()
    return artefact


def lire_versions():
    """
    Releve les versions de Python et des bibliotheques de l'entrainement (tracabilite de l'artefact).

    Returns
    -------
    dict
        Nom -> numero de version (texte).
    """
    versions = {}
    versions["python"] = platform.python_version()
    versions["numpy"] = np.__version__
    versions["pandas"] = pd.__version__
    versions["scikit-learn"] = sklearn.__version__
    versions["xgboost"] = xgboost.__version__
    versions["lightgbm"] = lightgbm.__version__
    return versions


def sauvegarder_artefact_arbres(artefact, chemin):
    """
    Ecrit un artefact de modele a arbres avec joblib, le modele etant serialise par cloudpickle (dossier cree si besoin).

    Pourquoi cloudpickle ? Une perte sur mesure (src/modeles.py) est une fonction definie dans une autre
    fonction : le module pickle standard ne sait pas l'ecrire, cloudpickle si.

    Parameters
    ----------
    artefact : dict
        L'artefact de `entrainer_modele_arbres` (n'est pas modifie).
    chemin : str
        Le fichier .joblib a ecrire.

    Returns
    -------
    str
        Le chemin ecrit.
    """
    os.makedirs(os.path.dirname(os.path.abspath(chemin)), exist_ok=True)
    a_ecrire = {}
    for cle in artefact:  # copie superficielle : seul le modele est remplace par ses octets
        a_ecrire[cle] = artefact[cle]
    a_ecrire["modele"] = cloudpickle.dumps(artefact["modele"])
    a_ecrire["format_modele"] = "cloudpickle"
    joblib.dump(a_ecrire, chemin, compress=3)
    return chemin


def charger_artefact_arbres(chemin):
    """
    Relit un artefact de modele a arbres ecrit par `sauvegarder_artefact_arbres` (modele serialise par cloudpickle).

    Parameters
    ----------
    chemin : str
        Le fichier .joblib.

    Returns
    -------
    dict
        L'artefact, estimateur entraine sous la cle "modele".

    Raises
    ------
    ValueError
        Si le fichier n'a pas le format attendu.
    """
    contenu = joblib.load(chemin)
    if not isinstance(contenu, dict) or contenu.get("version_format") != VERSION_FORMAT_ARTEFACT:
        raise ValueError(f"Artefact d'un autre format : {chemin_lisible(chemin)}.")
    artefact = {}
    for cle in contenu:
        artefact[cle] = contenu[cle]
    artefact["modele"] = pickle.loads(contenu["modele"])  # octets cloudpickle (perte sur mesure comprise)
    return artefact


def raison_artefact_perime(artefact, contexte, modele):
    """
    Dit pourquoi un artefact ne correspond plus a la recette actuelle (texte vide s'il est a jour).

    Parameters
    ----------
    artefact : dict
        L'artefact relu.
    contexte : dict
        Le contexte de preparation actuel.
    modele : str
        La cle du modele attendu.

    Returns
    -------
    str
        "" si l'artefact est a jour ; sinon la raison (configuration ou preparation differente).
    """
    if artefact.get("cle_modele") != modele or artefact.get("sous_jeu") != contexte["sous_jeu"]:
        return "artefact d'un autre modèle ou d'un autre sous-jeu"
    if not valeurs_proches(artefact.get("config"), convertir_en_types_python(configuration_arbres(modele))):
        return "configuration modifiée depuis l'entraînement"
    colonnes = colonnes_features_arbres(contexte, modele)  # jeu de features retenu par le script
    if not valeurs_proches(artefact.get("empreinte_preparation"), empreinte_preparation(contexte, colonnes)):
        return "préparation des données modifiée depuis l'entraînement"
    return ""


def obtenir_modele_arbres(contexte, modele, dossier=None, entrainer_si_absent=True, sauvegarder=True, forcer=False):
    """
    Recharge l'artefact d'un modele a arbres ; l'entraine (et le sauvegarde) s'il manque ou s'il est perime.

    Parameters
    ----------
    contexte : dict
        Le contexte COMPLET de `preparer_contexte`.
    modele : str
        Un modele de MODELES_ARBRES (ex. "xgboost_phase3").
    dossier : str ou None, optionnel
        Dossier des modeles.
    entrainer_si_absent : bool, optionnel
        Faux : un artefact absent ou perime rend le modele indisponible (ValueError).
    sauvegarder : bool, optionnel
        Faux : un modele entraine ici reste en memoire (rien n'est ecrit sur le disque).
    forcer : bool, optionnel
        Vrai : reentraine meme si l'artefact existe et semble a jour (apres une regeneration des predictions).

    Returns
    -------
    (dict, str)
        L'artefact pret a predire (sur 1 coeur) et un message ("" si l'artefact a simplement ete relu).

    Raises
    ------
    ValueError
        Si l'artefact manque ou est perime et que `entrainer_si_absent` est Faux.
    """
    chemin = chemin_artefact_arbres(contexte["sous_jeu"], modele, dossier)
    raison = ""
    artefact = None
    if forcer:
        raison = "réentraînement demandé"
    elif os.path.exists(chemin):
        try:
            artefact = charger_artefact_arbres(chemin)
            raison = raison_artefact_perime(artefact, contexte, modele)
        except (ValueError, KeyError, EOFError, pickle.UnpicklingError) as erreur:
            raison = f"fichier illisible ({erreur})"
        if raison != "":
            artefact = None
    else:
        raison = "artefact absent"

    message = ""
    if artefact is None:
        if not entrainer_si_absent:
            raise ValueError(f"{chemin_lisible(chemin)} : {raison} (lancer python scripts/preparer_demo.py).")
        artefact = entrainer_modele_arbres(contexte, modele)
        if sauvegarder:
            sauvegarder_artefact_arbres(artefact, chemin)  # le modele est ecrit par cloudpickle
            message = (f"{nom_modele(modele)} : {raison}, entraîné sur train_{contexte['sous_jeu']} en "
                       f"{artefact['duree_entrainement_s']:.1f} s et sauvegardé ({chemin_lisible(chemin)}).")
        else:
            message = f"{nom_modele(modele)} : {raison}, entraîné en mémoire (non sauvegardé)."
    artefact["chemin"] = chemin
    regler_un_coeur(artefact["modele"])
    return artefact, message


# ---------------------------------------------------------------------------------------------------
# Les deux reseaux : chargement seulement
# ---------------------------------------------------------------------------------------------------

def obtenir_reseau(contexte, modele, dossier=None):
    """
    Recharge les poids d'un reseau (jamais reentraine ici) et verifie qu'il attend la preparation actuelle.

    Le reseau de la demo peut etre un ensemble : plusieurs fichiers de poids (un par graine, meme configuration),
    dont la prediction est la moyenne simple (`predire_un_modele`). Tous doivent attendre les memes entrees.

    Parameters
    ----------
    contexte : dict
        Le contexte de preparation (complet ou leger).
    modele : str
        "lstm" ou "cnn1d".
    dossier : str ou None, optionnel
        Dossier des modeles.

    Returns
    -------
    dict
        "modeles" (les torch.nn.Module en mode prediction, un par graine), "modele" (le premier),
        "chemins", "colonnes_entree", "longueur_fenetre", "chemin" (le premier fichier),
        "informations" (informations du fichier .pt), "date_fichier", et pour l'ecart a l'etat initial
        (scripts/entrainer_deep_learning.py, configuration "A") : "ecart_initial" (k ou None),
        "min_max_ecart_initial" (bornes apprises sur le train, ou None), "capteurs".

    Raises
    ------
    ValueError
        Fichier absent, ou entrees du reseau differentes des capteurs de la preparation actuelle.
    """
    sous_jeu = contexte["sous_jeu"]
    chemins = chemins_poids_reseau(sous_jeu, modele, dossier)
    for chemin in chemins:
        if not os.path.exists(chemin):
            raise ValueError(f"{chemin_lisible(chemin)} absent : lancer python scripts/sauvegarder_poids_04h.py.")
    reseaux = []
    informations_premier = None
    for chemin in chemins:  # un fichier par graine, meme configuration
        reseau, contenu = charger_modele(chemin)  # src/deep_learning.py : poids + hyperparametres, mode prediction
        informations_reseau = contenu.get("informations", {})
        if informations_premier is None:
            informations_premier = informations_reseau
        for cle in ["colonnes_entree", "longueur", "ecart_initial", "min_max_ecart_initial", "sous_jeu"]:
            if informations_reseau.get(cle) != informations_premier.get(cle):
                raise ValueError(f"{chemin_lisible(chemin)} : {cle} different des autres graines du meme reseau.")
        reseaux.append(reseau)
    chemin = chemins[0]
    informations = informations_premier
    colonnes_entree = list(informations.get("colonnes_entree", []))
    # longueur de fenetre : cle "longueur" (scripts/sauvegarder_poids_04h.py) ou "longueur_fenetre" (ancien 03b)
    longueur = int(informations.get("longueur", informations.get("longueur_fenetre", 0)))
    ecart_initial = informations.get("ecart_initial")  # None : le reseau n'utilise pas l'ecart a l'etat initial
    min_max_ecart = informations.get("min_max_ecart_initial")  # bornes min-max apprises sur le train

    # colonnes d'ecart a l'etat initial mises a l'echelle attendues : "<capteur>_ecartinit_<k>_mm", dans l'ordre
    # des capteurs (meme nommage que scripts/entrainer_deep_learning.py et src.deep_learning.appliquer_min_max)
    colonnes_ecart_attendues = []
    if ecart_initial is not None:
        for capteur in contexte["capteurs"]:
            colonnes_ecart_attendues.append(capteur + "_ecartinit_" + str(int(ecart_initial)) + "_mm")

    # les entrees attendues : les capteurs prepares de ce sous-jeu, dans le meme ordre (+ ecarts, + time_cycles)
    capteurs_reseau = []
    ecarts_reseau = []
    for colonne in colonnes_entree:
        if colonne == NOM_COLONNE_TIME_CYCLES_ECHELLE:
            continue  # time_cycles / 100 : recalcule a la prediction, constante fixe
        if colonne in colonnes_ecart_attendues:
            ecarts_reseau.append(colonne)
        else:
            capteurs_reseau.append(colonne)
    if informations.get("sous_jeu") not in [None, sous_jeu]:
        raise ValueError(f"{chemin_lisible(chemin)} a été entraîné sur {informations.get('sous_jeu')}, pas {sous_jeu}.")
    if capteurs_reseau != list(contexte["capteurs"]) or ecarts_reseau != colonnes_ecart_attendues or longueur < 1:
        raise ValueError(f"{chemin_lisible(chemin)} : entrées du réseau différentes de la préparation actuelle.")
    if ecart_initial is not None and min_max_ecart is None:
        raise ValueError(f"{chemin_lisible(chemin)} : bornes min-max de l'écart à l'état initial absentes.")

    entree = {}
    entree["modeles"] = reseaux
    entree["modele"] = reseaux[0]
    entree["chemins"] = chemins
    entree["colonnes_entree"] = colonnes_entree
    entree["longueur_fenetre"] = longueur
    entree["chemin"] = chemin
    entree["informations"] = informations
    entree["ecart_initial"] = ecart_initial
    entree["min_max_ecart_initial"] = min_max_ecart
    entree["capteurs"] = list(contexte["capteurs"])
    entree["date_fichier"] = datetime.datetime.fromtimestamp(os.path.getmtime(chemin)).isoformat(timespec="seconds")
    return entree


# ---------------------------------------------------------------------------------------------------
# Chargement de tous les modeles d'un sous-jeu
# ---------------------------------------------------------------------------------------------------

def nom_modele(modele):
    """
    Nom lisible d'un modele (ex. "lstm" -> "LSTM"), celui du contrat de predictions.

    Parameters
    ----------
    modele : str
        La cle du modele.

    Returns
    -------
    str
        Le nom affiche.
    """
    return MODELES_CONNUS.get(modele, modele)


def charger_modeles(sous_jeu, modeles=None, dossier=None, entrainer_si_absent=True, sauvegarder=True):
    """
    Prepare tout ce qu'il faut pour predire en direct sur un sous-jeu : preparation apprise sur le train + modeles.

    Parameters
    ----------
    sous_jeu : str
        "FD001" a "FD004".
    modeles : list de str ou None, optionnel
        Les modeles voulus (None : MODELES_INFERENCE, les trois).
    dossier : str ou None, optionnel
        Dossier des modeles (None : `dossier_modeles()`, models/ par defaut).
    entrainer_si_absent : bool, optionnel
        Vrai (defaut) : un modele a arbres absent ou perime est entraine.
    sauvegarder : bool, optionnel
        Vrai (defaut) : un modele a arbres entraine ici est sauvegarde dans le dossier des modeles.

    Returns
    -------
    dict
        L'objet "inference", a passer a `predire_au_cycle` :

        - "sous_jeu" : str ;
        - "contexte" : la preparation apprise sur train_FD00X (voir `contexte_leger`) ;
        - "modeles" : dict cle -> description du modele disponible ("famille" "arbres" ou "sequentiel",
          "objet" pret a predire, "chemin", "date", "description", et pour un reseau "colonnes_entree" et
          "longueur_fenetre") ;
        - "indisponibles" : dict cle -> raison (ex. fichier de poids absent) ;
        - "messages" : list de str (ex. un modele a arbres entraine a l'instant) ;
        - "duree_chargement_s" : float.
    """
    verifier_sous_jeu(sous_jeu)
    if modeles is None:
        modeles = list(MODELES_INFERENCE)
    debut = time.perf_counter()
    contexte = preparer_contexte(sous_jeu)  # appris sur train_FD00X uniquement

    disponibles = {}
    indisponibles = {}
    messages = []
    for modele in modeles:
        if modele not in MODELES_INFERENCE:
            indisponibles[modele] = "modèle inconnu de l'inférence en direct"
            continue
        try:
            if modele in MODELES_ARBRES:
                artefact, message = obtenir_modele_arbres(contexte, modele, dossier, entrainer_si_absent, sauvegarder)
                if message != "":
                    messages.append(message)
                entree = {}
                entree["famille"] = "arbres"
                entree["objet"] = artefact["modele"]
                entree["chemin"] = artefact["chemin"]
                entree["date"] = artefact["date_entrainement"]
                entree["description"] = artefact["description_modele"]
                entree["nb_lignes_train"] = artefact["nb_lignes_train"]
                entree["nb_moteurs_train"] = artefact["nb_moteurs_train"]
                # colonnes lues par ce modele (verifiees a jour par l'empreinte de son artefact)
                entree["feature_cols"] = list(artefact["empreinte_preparation"]["feature_cols"])
            else:
                reseau = obtenir_reseau(contexte, modele, dossier)
                entree = {}
                entree["famille"] = "sequentiel"
                entree["objet"] = reseau["modele"]
                entree["objets"] = reseau["modeles"]  # prediction = moyenne simple de ces reseaux
                entree["chemin"] = reseau["chemin"]
                entree["chemins"] = reseau["chemins"]
                entree["date"] = reseau["date_fichier"]
                entree["description"] = f"{nom_modele(modele)}, fenêtre de {reseau['longueur_fenetre']} cycles"
                if len(reseau["modeles"]) > 1:
                    graines = graines_reseau(sous_jeu, modele)
                    entree["description"] = (f"{nom_modele(modele)}, moyenne de {len(reseau['modeles'])} réseaux "
                                             f"(graines {graines[0]} à {graines[-1]}), fenêtre de "
                                             f"{reseau['longueur_fenetre']} cycles")
                entree["colonnes_entree"] = reseau["colonnes_entree"]
                entree["longueur_fenetre"] = reseau["longueur_fenetre"]
                entree["ecart_initial"] = reseau["ecart_initial"]
                entree["min_max_ecart_initial"] = reseau["min_max_ecart_initial"]
                entree["capteurs"] = reseau["capteurs"]
                if reseau["ecart_initial"] is not None:
                    entree["description"] = entree["description"] + f", + écart à l'état initial (k = {reseau['ecart_initial']})"
            disponibles[modele] = entree
        except ValueError as erreur:
            indisponibles[modele] = str(erreur)

    inference = {}
    inference["sous_jeu"] = sous_jeu
    inference["contexte"] = contexte_leger(contexte)  # les grands tableaux du train ne sont pas gardes
    inference["modeles"] = disponibles
    inference["indisponibles"] = indisponibles
    inference["messages"] = messages
    inference["duree_chargement_s"] = float(time.perf_counter() - debut)
    return inference


# ---------------------------------------------------------------------------------------------------
# Mesures d'un moteur : jeu de test, tirage au hasard, fichier importe
# ---------------------------------------------------------------------------------------------------

def lister_moteurs_test(sous_jeu):
    """
    Numeros des moteurs du jeu de test d'un sous-jeu (test_FD00X.txt), tries.

    Parameters
    ----------
    sous_jeu : str
        "FD001" a "FD004".

    Returns
    -------
    list de int
        Les numeros de moteurs.
    """
    verifier_sous_jeu(sous_jeu)
    test = charger_sous_jeu(sous_jeu, "test")
    moteurs = []
    for moteur in np.sort(test["unit_number"].unique()):
        moteurs.append(int(moteur))
    return moteurs


def mesures_moteur_test(sous_jeu, moteur):
    """
    Mesures brutes d'un moteur du jeu de test et sa RUL vraie au dernier cycle observe.

    Les moteurs de test_FD00X.txt ne servent jamais a l'entrainement (train_FD00X.txt) ; d'apres
    data/readme.txt, chaque serie temporelle vient d'un moteur different.

    Parameters
    ----------
    sous_jeu : str
        "FD001" a "FD004".
    moteur : int
        Le numero du moteur test.

    Returns
    -------
    dict
        "sous_jeu", "moteur" (int), "mesures" (DataFrame brut, 26 colonnes, cycles 1..n, index 0..n-1),
        "rul_finale" (float, ligne `moteur` de RUL_FD00X.txt), "nb_cycles" (int).

    Raises
    ------
    ValueError
        Si le moteur n'existe pas dans le jeu de test.
    """
    verifier_sous_jeu(sous_jeu)
    test = charger_sous_jeu(sous_jeu, "test")
    moteurs = np.sort(test["unit_number"].unique())
    if int(moteur) not in moteurs:
        raise ValueError(f"Le moteur {moteur} n'existe pas dans test_{sous_jeu}.txt (moteurs 1 à {int(moteurs.max())}).")
    mesures = test[test["unit_number"] == int(moteur)].sort_values("time_cycles").reset_index(drop=True)
    rul_finales = charger_rul_sous_jeu(sous_jeu)  # ligne i = i-eme plus petit numero de moteur
    position = int(np.flatnonzero(moteurs == int(moteur))[0])

    resultat = {}
    resultat["sous_jeu"] = sous_jeu
    resultat["moteur"] = int(moteur)
    resultat["mesures"] = mesures
    resultat["rul_finale"] = float(rul_finales[position])
    resultat["nb_cycles"] = int(len(mesures))
    return resultat


def tirer_moteur_test(sous_jeu, graine=None):
    """
    Tire un moteur du jeu de test au hasard (chaque moteur a la meme chance).

    Parameters
    ----------
    sous_jeu : str
        "FD001" a "FD004".
    graine : int ou None, optionnel
        Graine du tirage (tests reproductibles) ; None : tirage different a chaque appel.

    Returns
    -------
    dict
        Comme `mesures_moteur_test`.
    """
    moteurs = lister_moteurs_test(sous_jeu)
    generateur = random.Random(graine)  # graine None : le systeme fournit l'alea
    moteur = generateur.choice(moteurs)
    return mesures_moteur_test(sous_jeu, moteur)


def verite_moteur(mesures, rul_finale):
    """
    RUL vraie a chaque cycle d'un moteur test : RUL au dernier cycle + cycles restants (src/contrat_predictions.py).

    Parameters
    ----------
    mesures : pandas.DataFrame
        Les mesures du moteur (unit_number, time_cycles).
    rul_finale : float
        La RUL vraie au dernier cycle observe (RUL_FD00X.txt).

    Returns
    -------
    pandas.DataFrame
        "time_cycles", "rul_vraie", "rul_vraie_cappee" (plafonnee a 125), un cycle par ligne.
    """
    tableau = mesures[["unit_number", "time_cycles"]].copy()
    tableau = calculer_rul_vraie(tableau, [float(rul_finale)])  # un seul moteur : une seule RUL finale
    return tableau[["time_cycles", "rul_vraie", "rul_vraie_cappee"]].reset_index(drop=True)


def decoder_contenu(contenu):
    """
    Transforme le contenu d'un fichier importe (octets ou texte) en texte, avec des erreurs claires.

    Parameters
    ----------
    contenu : bytes ou str
        Le contenu du fichier.

    Returns
    -------
    str
        Le texte.

    Raises
    ------
    ErreurFichierMesures
        Fichier trop gros ou qui n'est pas du texte.
    """
    if isinstance(contenu, str):
        octets = contenu.encode("utf-8")
    else:
        octets = bytes(contenu)
    if len(octets) > TAILLE_MAX_FICHIER_OCTETS:
        raise ErreurFichierMesures(
            f"Fichier trop gros ({len(octets)} octets ; maximum {TAILLE_MAX_FICHIER_OCTETS}) : un seul moteur attendu."
        )
    try:
        return octets.decode("utf-8-sig")  # "-sig" : ignore la marque d'ordre des octets ajoutee par Excel
    except UnicodeDecodeError:
        raise ErreurFichierMesures("Le fichier n'est pas un fichier texte (encodage UTF-8 attendu).")


def convertir_en_reel(valeur, numero_ligne, colonne):
    """
    Convertit une valeur lue dans le fichier en nombre reel fini, avec un message d'erreur precis.

    Parameters
    ----------
    valeur : str
        Le texte de la case.
    numero_ligne : int
        Le numero de ligne dans le fichier (a partir de 1), pour le message.
    colonne : str
        Le nom de la colonne, pour le message.

    Returns
    -------
    float
        La valeur.

    Raises
    ------
    ErreurFichierMesures
        Si ce n'est pas un nombre fini.
    """
    try:
        nombre = float(valeur)
    except ValueError:
        raise ErreurFichierMesures(f"Ligne {numero_ligne}, colonne {colonne} : {valeur!r} n'est pas un nombre.")
    if not np.isfinite(nombre):
        raise ErreurFichierMesures(f"Ligne {numero_ligne}, colonne {colonne} : valeur non finie ({valeur!r}).")
    return nombre


def lire_fichier_mesures(contenu):
    """
    Lit et valide un fichier de mesures d'UN moteur, au format C-MAPSS brut ou CSV.

    Formats acceptes :
    - C-MAPSS brut (comme data/test_FD00X.txt) : une ligne par cycle, 26 nombres separes par des espaces
      (unit_number, time_cycles, 3 reglages, 21 capteurs), sans en-tete ;
    - CSV avec en-tete : les 26 noms de src/preprocessing.COLUMNS (dans n'importe quel ordre), separes
      par des virgules (ou des points-virgules).
    Regles : un seul moteur ; cycles consecutifs 1, 2, 3, ... dans l'ordre ; uniquement des nombres finis.

    Parameters
    ----------
    contenu : bytes ou str
        Le contenu du fichier.

    Returns
    -------
    pandas.DataFrame
        Les mesures, colonnes dans l'ordre de COLUMNS, unit_number et time_cycles entiers, index 0..n-1.

    Raises
    ------
    ErreurFichierMesures
        Avec un message en francais qui dit quoi corriger.
    """
    texte = decoder_contenu(contenu)

    # lignes non vides (les fichiers NASA finissent souvent par des espaces et une ligne vide)
    lignes = []
    numeros_lignes = []
    morceaux_texte = texte.splitlines()
    for position in range(len(morceaux_texte)):
        ligne = morceaux_texte[position].strip()
        if ligne != "":
            lignes.append(ligne)
            numeros_lignes.append(position + 1)  # numero de ligne dans le fichier, pour les messages
    if len(lignes) == 0:
        raise ErreurFichierMesures("Le fichier est vide.")

    # format : CSV avec en-tete si la premiere ligne contient le nom de la premiere colonne
    avec_en_tete = COLUMNS[0] in lignes[0]
    separateur = None  # None : espaces de largeur variable (format C-MAPSS brut)
    noms = list(COLUMNS)
    lignes_donnees = lignes
    numeros_donnees = numeros_lignes
    if avec_en_tete:
        separateur = ","
        if ";" in lignes[0] and "," not in lignes[0]:
            separateur = ";"
        noms = []
        for nom in lignes[0].split(separateur):
            noms.append(nom.strip().strip('"'))
        inconnus = []
        for nom in noms:
            if nom not in COLUMNS:
                inconnus.append(nom)
        manquants = []
        for nom in COLUMNS:
            if nom not in noms:
                manquants.append(nom)
        if len(inconnus) > 0:
            raise ErreurFichierMesures(f"Colonnes inconnues dans l'en-tête : {', '.join(inconnus)}.")
        if len(manquants) > 0:
            raise ErreurFichierMesures(f"Colonnes manquantes dans l'en-tête : {', '.join(manquants)}.")
        if len(set(noms)) != len(noms):
            raise ErreurFichierMesures("Une colonne apparaît deux fois dans l'en-tête.")
        lignes_donnees = lignes[1:]
        numeros_donnees = numeros_lignes[1:]
    if len(lignes_donnees) == 0:
        raise ErreurFichierMesures("Aucune ligne de mesures après l'en-tête.")
    if len(lignes_donnees) > NB_MAX_CYCLES_FICHIER:
        raise ErreurFichierMesures(
            f"{len(lignes_donnees)} lignes : au plus {NB_MAX_CYCLES_FICHIER} cycles attendus pour un moteur."
        )

    # conversion case par case (message precis a la premiere erreur)
    valeurs = []
    for position in range(len(lignes_donnees)):
        numero = numeros_donnees[position]
        if separateur is None:
            cases = lignes_donnees[position].split()  # espaces de largeur variable
        else:
            cases = lignes_donnees[position].split(separateur)
        if len(cases) != len(noms):
            raise ErreurFichierMesures(f"Ligne {numero} : {len(cases)} valeurs au lieu de {len(noms)}.")
        ligne_valeurs = []
        for indice in range(len(cases)):
            ligne_valeurs.append(convertir_en_reel(cases[indice].strip().strip('"'), numero, noms[indice]))
        valeurs.append(ligne_valeurs)
    mesures = pd.DataFrame(valeurs, columns=noms)
    mesures = mesures[list(COLUMNS)]  # ordre de src/preprocessing.COLUMNS

    # un seul moteur, numero entier
    moteurs = mesures["unit_number"].unique()
    if len(moteurs) != 1:
        apercu = []
        for moteur in moteurs[:5]:
            apercu.append(f"{moteur:g}")
        raise ErreurFichierMesures(
            f"Le fichier contient {len(moteurs)} moteurs (unit_number {', '.join(apercu)}...) : un seul attendu."
        )
    if float(moteurs[0]) != float(int(moteurs[0])):
        raise ErreurFichierMesures(f"unit_number doit être un entier (lu : {moteurs[0]:g}).")

    # cycles : 1, 2, 3, ... dans l'ordre, sans trou ni doublon
    cycles = mesures["time_cycles"].to_numpy(dtype=float)
    for position in range(len(cycles)):
        attendu = position + 1
        if cycles[position] != attendu:
            raise ErreurFichierMesures(
                f"Ligne {numeros_donnees[position]} : cycle {cycles[position]:g} au lieu de {attendu} "
                "(cycles consécutifs à partir de 1 attendus)."
            )

    mesures["unit_number"] = mesures["unit_number"].astype(int)
    mesures["time_cycles"] = mesures["time_cycles"].astype(int)
    return mesures.reset_index(drop=True)


# ---------------------------------------------------------------------------------------------------
# Prediction en direct
# ---------------------------------------------------------------------------------------------------

def verifier_mesures(mesures):
    """
    Verifie que des mesures decrivent UN moteur, avec des cycles consecutifs a partir de 1.

    Parameters
    ----------
    mesures : pandas.DataFrame
        Les mesures brutes (colonnes de src/preprocessing.COLUMNS).

    Raises
    ------
    ValueError
        Sinon.
    """
    if len(mesures) == 0:
        raise ValueError("Aucune mesure.")
    if mesures["unit_number"].nunique() != 1:
        raise ValueError("Les mesures doivent décrire un seul moteur.")
    cycles = mesures["time_cycles"].to_numpy()
    attendus = np.arange(1, len(cycles) + 1)
    if not np.array_equal(cycles, attendus):
        raise ValueError("Les cycles doivent être 1, 2, 3, ... dans l'ordre.")


def predire_un_modele(entree, preparees):
    """
    RUL predite par UN modele au DERNIER cycle des mesures preparees (>= 0, en cycles).

    Parameters
    ----------
    entree : dict
        La description du modele (`charger_modeles`, cle "modeles").
    preparees : pandas.DataFrame
        Les mesures 1..t preparees par `appliquer_preparation` (dernier cycle = t).

    Returns
    -------
    float
        La RUL predite au cycle t.
    """
    if entree["famille"] == "arbres":
        # une ligne : les features du cycle t (meme ordre que le train), RUL negative ramenee a 0
        ligne = preparees[entree["feature_cols"]].iloc[[-1]]
        prediction = np.asarray(entree["objet"].predict(ligne), dtype=float)
        return float(np.clip(prediction, a_min=0.0, a_max=None)[0])

    # reseau : fenetre des `longueur` derniers cycles (1..t seulement, remplissage par le cycle 1 au debut)
    avec_age = ajouter_time_cycles_echelle(preparees)  # time_cycles / 100, constante fixe
    if entree.get("ecart_initial") is not None:
        # ecart a l'etat initial : calcule sur les seuls cycles 1..t de ce moteur (src/features.py, causal), puis
        # mis a l'echelle avec les bornes min-max apprises sur le train (lues dans le fichier .pt, rien n'est reappris)
        k = int(entree["ecart_initial"])
        avec_age, colonnes_ecart = ajouter_ecart_etat_initial(avec_age, entree["capteurs"], k)
        avec_age, _ = appliquer_min_max(avec_age, colonnes_ecart, entree["min_max_ecart_initial"])
    fenetres, _, _, _ = construire_fenetres(avec_age, entree["colonnes_entree"], longueur=entree["longueur_fenetre"])
    fenetre_t = fenetres[-1:]  # la fenetre du cycle t (forme 1 x longueur x entrees)
    lot = np.repeat(fenetre_t, TAILLE_LOT_RESEAU, axis=0)  # copies de la MEME fenetre (voir TAILLE_LOT_RESEAU)
    reseaux = entree.get("objets", [entree["objet"]])
    somme = 0.0
    for reseau in reseaux:  # ensemble : moyenne simple des predictions des reseaux (poids egaux)
        somme = somme + float(predire(reseau, lot)[0])  # src/deep_learning.py : x 125 puis clip a 0
    return somme / len(reseaux)


def predire_au_cycle(inference, mesures, cycle, modeles=None):
    """
    RUL predite par chaque modele au cycle t, a partir des SEULES mesures brutes des cycles 1..t.

    Etapes : (1) on garde les lignes de cycle <= t (copie : rien d'autre n'est transmis) ; (2) preparation
    de ces lignes avec les parametres appris sur le train (`appliquer_preparation` : regime par K-Means,
    min / max du train, lissage et statistiques glissantes causales ; rien n'est reappris) ; (3) chaque modele
    predit au cycle t (arbres : la ligne du cycle t ; reseau : la fenetre des 30 derniers cycles).

    Parameters
    ----------
    inference : dict
        Le resultat de `charger_modeles`.
    mesures : pandas.DataFrame
        Les mesures brutes du moteur (cycles 1..n, n >= t) ; les cycles > t ne sont jamais lus.
    cycle : int
        Le cycle t ("present").
    modeles : list de str ou None, optionnel
        Les modeles a faire predire (None : tous les modeles disponibles).

    Returns
    -------
    dict
        "cycle" (int), "nb_cycles_recus" (int, = t), "predictions" (dict cle -> RUL predite), "temps_ms"
        (dict cle -> millisecondes du modele seul), "temps_preparation_ms" (preparation commune),
        "temps_total_ms" (preparation + tous les modeles).

    Raises
    ------
    ValueError
        Cycle hors des mesures, mesures invalides, ou modele indisponible.
    """
    debut_total = time.perf_counter()
    cycle = int(cycle)
    if modeles is None:
        modeles = list(inference["modeles"].keys())
    for modele in modeles:
        if modele not in inference["modeles"]:
            raise ValueError(f"Modèle indisponible : {modele!r}.")

    # 1. les seules mesures transmises : cycles 1..t (copie independante)
    recues = mesures[mesures["time_cycles"] <= cycle].reset_index(drop=True)
    verifier_mesures(recues)
    if int(recues["time_cycles"].iloc[-1]) != cycle:
        raise ValueError(f"Cycle {cycle} absent des mesures (dernier cycle : {int(mesures['time_cycles'].max())}).")

    # 2. preparation avec les parametres appris sur le train (transform seul)
    debut = time.perf_counter()
    preparees = appliquer_preparation(inference["contexte"], recues)
    temps_preparation = 1000.0 * (time.perf_counter() - debut)

    # 3. une prediction par modele, chronometree
    predictions = {}
    temps = {}
    for modele in modeles:
        entree = inference["modeles"][modele]  # un modele a arbres porte ses colonnes ("feature_cols")
        debut = time.perf_counter()
        predictions[modele] = predire_un_modele(entree, preparees)
        temps[modele] = 1000.0 * (time.perf_counter() - debut)

    resultat = {}
    resultat["cycle"] = cycle
    resultat["nb_cycles_recus"] = int(len(recues))
    resultat["predictions"] = predictions
    resultat["temps_ms"] = temps
    resultat["temps_preparation_ms"] = float(temps_preparation)
    resultat["temps_total_ms"] = float(1000.0 * (time.perf_counter() - debut_total))
    return resultat


def predire_trajectoire(inference, mesures, cycles=None, modeles=None):
    """
    Repete `predire_au_cycle` a plusieurs cycles : une troncature et une preparation par cycle.

    Parameters
    ----------
    inference : dict
        Le resultat de `charger_modeles`.
    mesures : pandas.DataFrame
        Les mesures brutes du moteur.
    cycles : list de int ou None, optionnel
        Les cycles voulus (None : tous, de 1 au dernier).
    modeles : list de str ou None, optionnel
        Les modeles (None : tous les modeles disponibles).

    Returns
    -------
    pandas.DataFrame
        Une ligne par cycle : "time_cycles", une colonne par modele (RUL predite), "temps_total_ms".
    """
    if cycles is None:
        cycles = []
        for cycle in mesures["time_cycles"].tolist():
            cycles.append(int(cycle))
    lignes = []
    for cycle in cycles:
        resultat = predire_au_cycle(inference, mesures, cycle, modeles)
        ligne = {}
        ligne["time_cycles"] = int(cycle)
        for modele in resultat["predictions"]:
            ligne[modele] = resultat["predictions"][modele]
        ligne["temps_total_ms"] = resultat["temps_total_ms"]
        lignes.append(ligne)
    return pd.DataFrame(lignes)


def ecarts_aux_predictions_precalculees(inference, moteur, cycles, mesures=None, modeles=None, dossier_predictions=None):
    """
    Compare l'inference en direct d'un moteur test aux fichiers precalcules sorties/predictions/<SJ>__<modele>.csv.gz.

    Parameters
    ----------
    inference : dict
        Le resultat de `charger_modeles`.
    moteur : int
        Le numero du moteur test.
    cycles : list de int
        Les cycles a comparer.
    mesures : pandas.DataFrame ou None, optionnel
        Les mesures du moteur (None : lues dans test_FD00X.txt).
    modeles : list de str ou None, optionnel
        Les modeles (None : tous les modeles disponibles).
    dossier_predictions : str ou None, optionnel
        Dossier des fichiers precalcules (None : celui du contrat).

    Returns
    -------
    (pandas.DataFrame, dict)
        Le tableau "modele", "time_cycles", "rul_directe", "rul_precalculee", "ecart_absolu", et les modeles
        sans fichier precalcule (cle -> raison).
    """
    sous_jeu = inference["sous_jeu"]
    if mesures is None:
        mesures = mesures_moteur_test(sous_jeu, moteur)["mesures"]
    if modeles is None:
        modeles = list(inference["modeles"].keys())

    # fichiers precalcules : une serie (moteur, cycle) -> RUL predite par modele
    references = {}
    absents = {}
    for modele in modeles:
        try:
            df, _ = lire_predictions(sous_jeu, modele, dossier_predictions)
        except (FileNotFoundError, ValueError) as erreur:
            absents[modele] = f"fichier précalculé absent ou invalide ({erreur})"
            continue
        lignes_moteur = df[df["unit_number"] == int(moteur)]
        references[modele] = lignes_moteur.set_index("time_cycles")["rul_predite"]

    lignes = []
    for cycle in cycles:
        modeles_compares = []
        for modele in modeles:
            if modele in references:
                modeles_compares.append(modele)
        if len(modeles_compares) == 0:
            break
        resultat = predire_au_cycle(inference, mesures, cycle, modeles_compares)
        for modele in modeles_compares:
            ligne = {}
            ligne["modele"] = modele
            ligne["time_cycles"] = int(cycle)
            ligne["rul_directe"] = resultat["predictions"][modele]
            ligne["rul_precalculee"] = float(references[modele].loc[int(cycle)])
            ligne["ecart_absolu"] = abs(ligne["rul_directe"] - ligne["rul_precalculee"])
            lignes.append(ligne)
    return pd.DataFrame(lignes, columns=["modele", "time_cycles", "rul_directe", "rul_precalculee", "ecart_absolu"]), absents


# ---------------------------------------------------------------------------------------------------
# Ligne de commande : construire les artefacts des modeles a arbres et verifier l'egalite avec les fichiers precalcules
# ---------------------------------------------------------------------------------------------------

def construire_artefacts(sous_jeux=None, forcer=False):
    """
    Construit (ou relit) les artefacts des modeles a arbres de chaque sous-jeu et verifie l'inference en direct.

    Appele par scripts/preparer_demo.py (dans un sous-processus : ce module charge PyTorch). Verification :
    pour le premier moteur test, aux cycles 1, milieu et dernier, ecart maximal entre l'inference en direct
    et chaque fichier precalcule present.

    Parameters
    ----------
    sous_jeux : list de str ou None, optionnel
        Les sous-jeux (None : les quatre).
    forcer : bool, optionnel
        Vrai : reentraine les modeles a arbres meme si leur artefact semble a jour.

    Returns
    -------
    pandas.DataFrame
        Une ligne par (sous-jeu, modele) : "sous_jeu", "modele", "etat" (disponible ou raison),
        "ecart_max_precalcule" (NaN si pas de fichier).
    """
    if sous_jeux is None:
        sous_jeux = list(SOUS_JEUX)
    lignes = []
    for sous_jeu in sous_jeux:
        contexte = preparer_contexte(sous_jeu)
        for modele in MODELES_ARBRES:  # entraine si absent, perime ou si forcer est demande
            artefact, message = obtenir_modele_arbres(contexte, modele, None, True, True, forcer)  # models/
            if message != "":
                print(f"  {sous_jeu} : {message}", flush=True)
        inference = charger_modeles(sous_jeu, entrainer_si_absent=False)
        moteur = lister_moteurs_test(sous_jeu)[0]
        mesures = mesures_moteur_test(sous_jeu, moteur)["mesures"]
        dernier = int(mesures["time_cycles"].max())
        ecarts, absents = ecarts_aux_predictions_precalculees(inference, moteur, [1, (dernier + 1) // 2, dernier],
                                                               mesures=mesures)
        for modele in MODELES_INFERENCE:
            ligne = {}
            ligne["sous_jeu"] = sous_jeu
            ligne["modele"] = modele
            ligne["ecart_max_precalcule"] = np.nan
            if modele in inference["modeles"]:
                ligne["etat"] = "disponible"
                if modele not in absents:
                    lignes_modele = ecarts[ecarts["modele"] == modele]
                    if len(lignes_modele) > 0:
                        ligne["ecart_max_precalcule"] = float(lignes_modele["ecart_absolu"].max())
            else:
                ligne["etat"] = inference["indisponibles"].get(modele, "indisponible")
            lignes.append(ligne)
            print(f"  {sous_jeu} {modele:<15} {ligne['etat']:<12} ecart max au precalcule : "
                  f"{ligne['ecart_max_precalcule']:.3g}", flush=True)
    return pd.DataFrame(lignes)


def main(argv=None):
    """
    Point d'entree : python -m src.inference [--sous-jeux FD001 ...] [--forcer].

    Parameters
    ----------
    argv : list de str ou None, optionnel
        Les arguments (None : ceux de la ligne de commande).

    Returns
    -------
    int
        0 si tous les ecarts mesures sont <= 1e-4 cycle, 1 sinon.
    """
    import argparse

    parseur = argparse.ArgumentParser(
        prog="python -m src.inference",
        description="Construit les artefacts des modeles a arbres et verifie l'egalite avec les fichiers precalcules.",
    )
    parseur.add_argument("--sous-jeux", nargs="+", choices=SOUS_JEUX, default=list(SOUS_JEUX))
    parseur.add_argument("--forcer", action="store_true", help="reentraine les modeles a arbres meme si leur artefact est a jour")
    arguments = parseur.parse_args(argv)
    tableau = construire_artefacts(arguments.sous_jeux, arguments.forcer)
    ecarts = tableau["ecart_max_precalcule"].dropna()
    if len(ecarts) > 0 and float(ecarts.max()) > 1e-4:
        print("  ECART > 1e-4 cycle entre inference en direct et fichier precalcule.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
