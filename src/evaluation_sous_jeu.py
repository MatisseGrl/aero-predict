"""
Protocole d'evaluation complet d'UN sous-jeu (FD001 a FD004) et verification d'un moteur depuis les donnees brutes.

Pourquoi ce module ? (plan de reorganisation des notebooks, docs/plan_refonte_notebooks.md, sections 3 et 5.3)
Le protocole (docs/protocole_evaluation.md, regle 11 d'AGENTS.md) n'avait ete deroule en entier que sur FD001,
dans la section G du notebook 05. Les 4 notebooks de resultats (notebooks/resultats/FD00X.ipynb, generes depuis
un seul modele) ont besoin du MEME calcul pour chaque sous-jeu : il vit donc ici, une seule fois.

Deux fonctions publiques :

- `evaluer_sous_jeu(sous_jeu)` : sections 1 a 6 du plan 5.3, pour les modeles disponibles du sous-jeu (6 sur
  chacun des quatre sous-jeux depuis le calcul de nuit FD003 / FD004 du 2026-11-07 au 08 ; avant, FD003 et FD004
  n'avaient pas de reseau, decision du 2026-11-07, et seulement 4 modeles) et les 5 baselines du
  protocole. Rien n'est reentraine, sauf les 5 baselines (quelques secondes) : la validation croisee des arbres
  est relue dans sorties/resultats/refonte/ (scripts/valider_arbres_finaux.py, reglages finaux, memes plis),
  celle des reseaux dans docs/resultats/04h/ (notebook 04h), et les predictions test dans les fichiers du contrat
  (sorties/predictions/, les memes que la demo : principe 4 du plan).
- `verifier_moteur(sous_jeu, moteur)` : section 7 du plan 5.3. Repart de data/test_FD00X.txt et recalcule la
  prediction cycle par cycle de chaque modele disponible, avec l'inference en direct de src/inference.py (les
  mesures 1..t seulement a chaque cycle t). Elle ne lit JAMAIS sorties/predictions/ (principe 3 : la preuve
  n'est pas circulaire). `comparer_aux_fichiers_demo(verification)` compare ENSUITE son resultat aux fichiers
  de la demo, pour montrer qu'ils sont identiques.

Le code vient des notebooks d'origine, deplace et non reecrit : notebook 05 sections G (protocole sur FD001),
H (litterature) et J.3 / J.4 (moteur mystere) ; notebook 04a (baselines, erreur par tranche) ; notebook 04b
(classification, seuil choisi sur la validation) ; notebook 04g (comparaison des modeles a arbres). Les calculs
eux-memes sont ceux de src/evaluation.py, src/inference.py et src/contrat_predictions.py, appeles tels quels.

Complements relus s'ils existent (scripts/valider_arbres_finaux.py --complements, ajoutes apres la revue de la
reorganisation) : l'erreur sur les moteurs d'entrainement des modeles a arbres, pli par pli (diagnostic de
sur-apprentissage, protocole §5.1), et leur validation croisee sur les 3 decoupages neufs du notebook 04h (les
sous-jeux qui ont des reseaux), ce qui rend les moyennes arbres / reseaux comparables (memes decoupages, pas un
test apparie).

Ce que ce module NE fait PAS (signale dans le resultat, cle "non_fait", regle 11) : l'erreur sur les moteurs
d'entrainement des reseaux ; les predictions hors-pli des reseaux (non enregistrees par le notebook 04h : pas de
test apparie arbres contre reseaux en validation, pas de seuil d'alerte choisi sur la validation pour les reseaux).
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ORDRE DES IMPORTS VOULU : src.inference importe xgboost et lightgbm AVANT PyTorch (voir src/deep_learning.py)
from src import inference as inf
from src.alertes import SEUIL_ALERTE_DEFAUT, classer_alerte
from src.config_modele import RACINE_DEPOT
from src.contrat_predictions import (
    MODELES_CONNUS,
    SOUS_JEUX,
    chemin_metadonnees,
    chemin_predictions,
    extraire_dernier_cycle,
    lire_metadonnees,
    lire_predictions,
    verifier_coherence,
)
from src.evaluation import (
    calculer_plis,
    choisir_seuil_alerte,
    comparer_modeles_par_paires,
    erreurs_par_moteur,
    erreurs_par_tranche_rul,
    metriques_classification_alerte,
    metriques_regression,
    metriques_regression_par_pli,
    resumer_plis,
    rmse,
)
from src.optimisation_reseaux import ENSEMBLES_RETENUS
from src.sous_jeux import DESCRIPTION_SOUS_JEUX, charger_rul_sous_jeu

# ---------------------------------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------------------------------

PLAFOND_RUL = 125  # plafond de la RUL (Phase 2) : la cible d'entrainement ne depasse jamais 125
SEUIL_ALERTE = SEUIL_ALERTE_DEFAUT  # 30 cycles, seuil de la demo (src/alertes.py), hypothese de travail

# classification "panne dans les N prochains cycles ?" (protocole section 4, notebook 05 G.5)
HORIZONS = [15, 30, 50]  # N etudies
HORIZON_MATRICES = 30  # N des matrices de confusion (proposition du plan, section 12)
RAPPEL_MINIMUM = 0.95  # rappel exige pour le seuil choisi sur la validation

# comparaisons appariees (protocole section 8, notebook 05 G.6)
SEUIL_P = 0.05  # seuil de significativite applique aux p-valeurs corrigees par Bonferroni

# baselines du protocole (section 2), dans l'ordre des tableaux du notebook 05 (G.2)
BASELINES = ["Moyenne", "Mediane", "Constante 125", "Aleatoire uniforme", "Ridge"]
GRAINE_BASELINES = 42  # graine du tirage de la baseline aleatoire

# validation croisee des arbres AVEC leurs reglages finaux (scripts/valider_arbres_finaux.py, journal du 2026-11-07)
DOSSIER_CV_ARBRES = os.path.join(RACINE_DEPOT, "sorties", "resultats", "refonte")
FICHIER_CV_RESUME = os.path.join(DOSSIER_CV_ARBRES, "cv_arbres_finaux_resume.csv")
FICHIER_CV_PAR_PLI = os.path.join(DOSSIER_CV_ARBRES, "cv_arbres_finaux_par_pli.csv")
FICHIER_CV_OOF = os.path.join(DOSSIER_CV_ARBRES, "cv_arbres_finaux_oof.csv.gz")
FICHIER_CV_PLIS = os.path.join(DOSSIER_CV_ARBRES, "cv_arbres_finaux_plis.csv")
COMMANDE_CV_ARBRES = "LOKY_MAX_CPU_COUNT=4 python scripts/valider_arbres_finaux.py"
# complements (meme script, option --complements) : erreur d'entrainement par pli, decoupages neufs du 04h
FICHIER_CV_TRAIN_PAR_PLI = os.path.join(DOSSIER_CV_ARBRES, "cv_arbres_finaux_train_par_pli.csv")
FICHIER_CV_DECOUPAGES_NEUFS = os.path.join(DOSSIER_CV_ARBRES, "cv_arbres_finaux_decoupages_neufs.csv")
COMMANDE_COMPLEMENTS = "LOKY_MAX_CPU_COUNT=4 python scripts/valider_arbres_finaux.py --complements"
GRAINES_DECOUPAGES_NEUFS = [1000, 1001, 1002]  # les 3 decoupages neufs du notebook 04h

# resultats du notebook 04h (copie versionnee du calcul de nuit) : validation des reseaux publies
DOSSIER_RESULTATS_04H = os.path.join(RACINE_DEPOT, "docs", "resultats", "04h")

# raison affichee quand un sous-jeu n'a pas de reseau, c'est-a-dire pas d'ensemble retenu au notebook 04h
# (ENSEMBLES_RETENUS). Depuis le calcul de nuit FD003 / FD004 du 2026-11-07 au 08, les quatre sous-jeux en ont
# un : ce cas ne se presente plus avec les fichiers du depot, il est garde pour un sous-jeu sans reseau.
# Avant, le texte citait FD001 et FD002 (decision du 2026-11-07, journal du plan).
RAISON_SANS_RESEAU = ("pas de réseau sur ce sous-jeu : la v2 ne garde que les réseaux optimisés du notebook 04h, "
                      "et aucun ensemble n'a été retenu pour ce sous-jeu (ENSEMBLES_RETENUS)")

# chiffres publies par les articles (RMSE test), recopies de leurs tableaux : ce ne sont pas des resultats du
# projet (notebook 05, section H.2, ou les references completes et la convention de chaque article sont donnees).
# np.nan : non publie, ou absent de la version relue de l'article (DCNN sur FD002 a FD004).
ARTICLES = []
ARTICLES.append({"source": "Babu, Zhao, Li 2016", "modele": "CNN", "fenetre": "15", "entrainements": "non precise",
                 "FD001": 18.45, "FD002": 30.29, "FD003": 19.82, "FD004": 29.16})
ARTICLES.append({"source": "Li, Ding, Sun 2018", "modele": "DCNN", "fenetre": "30", "entrainements": "moyenne de 10",
                 "FD001": 12.61, "FD002": np.nan, "FD003": np.nan, "FD004": np.nan})
ARTICLES.append({"source": "Zhang, Song, Li 2022", "modele": "DAST", "fenetre": "40 / 60", "entrainements": "moyenne de 10",
                 "FD001": 11.43, "FD002": 15.25, "FD003": 11.32, "FD004": 18.36})
ARTICLES.append({"source": "Huang, He, Sick 2023", "modele": "STAGNN* (norm. par regime)", "fenetre": "50",
                 "entrainements": "moyenne de 10", "FD001": 11.50, "FD002": 13.81, "FD003": 11.05, "FD004": 14.30})

# protocole de chaque source de validation, ecrit dans les tableaux (les deux ne sont PAS directement comparables)
PROTOCOLE_VALIDATION_ARBRES = ("GroupKFold par moteur, 5 plis, graine 42 (calculer_plis), reglages finaux "
                               "(CONFIG_*_FINAL) ; mêmes plis pour les arbres et les 5 baselines")
PROTOCOLE_VALIDATION_RESEAUX = ("notebook 04h : 5 plis par moteur sur 3 découpages neufs (graines 1000 à 1002) ; "
                                "plis différents de ceux des arbres, prédictions hors-pli non enregistrées")
PROTOCOLE_DECOUPAGES_NEUFS = ("5 plis par moteur sur les 3 découpages neufs du notebook 04h (graines 1000 à 1002), "
                              "les mêmes pour les arbres et les réseaux ; moyenne ± écart-type entre les 3 découpages")


# ---------------------------------------------------------------------------------------------------
# Petits outils d'affichage (deplaces du notebook 05, section A.5)
# ---------------------------------------------------------------------------------------------------

def formater_moyenne_ecart(moyenne, ecart_type, decimales=2):
    """
    Ecrit une moyenne et un ecart-type sous la forme "moyenne ± ecart-type".

    Parameters
    ----------
    moyenne : float
        La moyenne.
    ecart_type : float
        L'ecart-type.
    decimales : int, optionnel
        Nombre de decimales affichees (2 par defaut).

    Returns
    -------
    str
        Par exemple "9.36 ± 0.33".
    """
    return f"{moyenne:.{decimales}f} ± {ecart_type:.{decimales}f}"


def nom_affiche(cle):
    """
    Nom lisible d'un modele ou d'une baseline (ex. "xgboost_phase3" -> "XGBoost", "Ridge" -> "Ridge").

    Parameters
    ----------
    cle : str
        Une cle de MODELES_CONNUS (src/contrat_predictions.py) ou un nom de BASELINES.

    Returns
    -------
    str
        Le nom affiche dans les tableaux et les figures.
    """
    if cle in MODELES_CONNUS:
        return MODELES_CONNUS[cle]
    return cle  # les baselines portent deja leur nom lisible


def verifier_nom_sous_jeu(sous_jeu):
    """
    Verifie qu'un nom de sous-jeu est connu.

    Parameters
    ----------
    sous_jeu : str
        Le nom a verifier.

    Raises
    ------
    ValueError
        Si le nom n'est pas "FD001", "FD002", "FD003" ou "FD004".
    """
    if sous_jeu not in SOUS_JEUX:
        raise ValueError(f"Sous-jeu inconnu : {sous_jeu!r} (possibles : {', '.join(SOUS_JEUX)}).")


# ---------------------------------------------------------------------------------------------------
# Section 2 : quels modeles sont disponibles pour ce sous-jeu
# ---------------------------------------------------------------------------------------------------

def modeles_publies(sous_jeu):
    """
    Liste les modeles dont la demo a les fichiers de predictions pour ce sous-jeu, dans l'ordre d'affichage.

    Parameters
    ----------
    sous_jeu : str
        "FD001" a "FD004".

    Returns
    -------
    (list de str, dict)
        Les cles des modeles disponibles (ordre de MODELES_CONNUS), et les modeles absents (cle -> raison).
        Un sous-jeu sans ensemble retenu au notebook 04h (ENSEMBLES_RETENUS) n'a pas de reseau : les deux sont
        absents par decision (RAISON_SANS_RESEAU). Depuis le 2026-11-08, les quatre sous-jeux en ont un.
    """
    verifier_nom_sous_jeu(sous_jeu)
    disponibles = []
    absents = {}
    for cle in MODELES_CONNUS:  # ordre d'affichage du contrat : reference naive, 3 arbres, 2 reseaux
        fichier_present = os.path.exists(chemin_predictions(sous_jeu, cle))
        metadonnees_presentes = os.path.exists(chemin_metadonnees(sous_jeu, cle))
        if fichier_present and metadonnees_presentes:
            disponibles.append(cle)
        elif cle in inf.MODELES_RESEAUX and sous_jeu not in ENSEMBLES_RETENUS:
            absents[cle] = RAISON_SANS_RESEAU  # absence voulue, pas un oubli
        else:
            absents[cle] = "fichiers de prédictions absents (lancer python scripts/preparer_demo.py)"
    return disponibles, absents


def tableau_configurations(sous_jeu, modeles):
    """
    Section 2 : les reglages utilises par chaque modele, lus dans les metadonnees des fichiers du contrat.

    Les metadonnees recopient la configuration de src/config_modele.py (arbres) ou de
    src/optimisation_reseaux.py (reseaux) au moment ou les predictions ont ete produites : c'est donc ce
    qui a reellement servi.

    Parameters
    ----------
    sous_jeu : str
        "FD001" a "FD004".
    modeles : list de str
        Les modeles disponibles (`modeles_publies`).

    Returns
    -------
    pandas.DataFrame
        Une ligne par modele : "cle", "modele" (nom affiche), "famille", "description", "reglages" (texte JSON
        des hyperparametres), "source", "date".
    """
    lignes = []
    for cle in modeles:
        metadonnees = lire_metadonnees(sous_jeu, cle)
        # arbres : cle "config" (format de src/config_modele.py) ; reseaux : cle "hyperparametres"
        if "config" in metadonnees:
            reglages = metadonnees["config"].get("hyperparametres", {})
        else:
            reglages = metadonnees.get("hyperparametres", {})
        description = metadonnees.get("description_modele", metadonnees.get("protocole_validation", ""))
        ligne = {}
        ligne["cle"] = cle
        ligne["modele"] = nom_affiche(cle)
        ligne["famille"] = metadonnees.get("famille", "")
        if cle == "reference_naive":
            # le script des arbres la range avec les arbres (meme script) ; c'est une baseline, pas un arbre
            ligne["famille"] = "référence (baseline)"
        ligne["description"] = description
        ligne["reglages"] = json.dumps(reglages, ensure_ascii=False)
        ligne["source"] = metadonnees.get("source", "")
        ligne["date"] = metadonnees.get("date", "")
        lignes.append(ligne)
    return pd.DataFrame(lignes)


# ---------------------------------------------------------------------------------------------------
# Section 1 : les donnees du sous-jeu, et les plis de la validation croisee
# ---------------------------------------------------------------------------------------------------

def preparer_donnees(sous_jeu):
    """
    Prepare le sous-jeu comme le script des arbres et calcule les 5 plis de validation croisee, une seule fois.

    Preparation : `src.inference.preparer_contexte` (= `preparer_sous_jeu` avec la fenetre de lissage 5, le
    plafond 125 et l'ecart a l'etat initial k = 20 du script des arbres) : X contient les features des modeles
    a arbres retenus (31 sur FD001), celles que les baselines utilisent aussi (comme le notebook 05, G.2).
    Plis : `calculer_plis` (GroupKFold, 5 plis melanges, graine 42). On verifie qu'ils sont exactement ceux de
    scripts/valider_arbres_finaux.py (fichier cv_arbres_finaux_plis.csv) : la validation croisee relue plus bas
    a donc ete faite sur ces memes moteurs de validation.

    Parameters
    ----------
    sous_jeu : str
        "FD001" a "FD004".

    Returns
    -------
    dict
        "sous_jeu", "contexte" (le contexte complet de preparer_sous_jeu), "X", "y" (numpy, RUL plafonnee),
        "groupes" (numpy, moteur de chaque ligne), "plis" (liste de 5 couples idx_train / idx_val),
        "cles_lignes" (DataFrame unit_number / time_cycles du train, dans l'ordre de X), "moteurs_test" (numpy),
        "rul_vraie_test" (numpy, RUL_FD00X.txt), "rul_vraie_test_plafonnee" (numpy).

    Raises
    ------
    FileNotFoundError
        Si le fichier des plis de scripts/valider_arbres_finaux.py manque.
    AssertionError
        Si les plis different de ceux du script.
    """
    verifier_nom_sous_jeu(sous_jeu)
    contexte = inf.preparer_contexte(sous_jeu)  # appris sur train_FD00X seulement
    X = contexte["X"]  # features des arbres retenus, une ligne par cycle du train
    y = contexte["y"].to_numpy(dtype=float)  # RUL plafonnee a 125
    groupes = contexte["groups"].to_numpy()  # numero de moteur de chaque ligne
    plis = calculer_plis(X, y, groupes, n_splits=5, random_state=42)  # calcules UNE fois, reutilises partout

    # controle : memes moteurs de validation, pli par pli, que le script qui a produit la validation des arbres
    if not os.path.exists(FICHIER_CV_PLIS):
        raise FileNotFoundError(f"{FICHIER_CV_PLIS} absent : lancer {COMMANDE_CV_ARBRES}")
    plis_script = pd.read_csv(FICHIER_CV_PLIS)
    plis_script = plis_script[plis_script["sous_jeu"] == sous_jeu]
    for numero_pli in range(len(plis)):  # le fichier numerote les plis a partir de 0
        idx_val = plis[numero_pli][1]
        moteurs_val = np.sort(np.unique(groupes[idx_val]))  # moteurs de validation de ce pli
        moteurs_script = np.sort(plis_script.loc[plis_script["pli"] == numero_pli, "unit_number"].to_numpy())
        assert np.array_equal(moteurs_val, moteurs_script), f"{sous_jeu}, pli {numero_pli} : plis differents du script"

    # le test officiel : une ligne par moteur test, son dernier cycle observe
    rul_vraie_test = np.asarray(contexte["y_test_vraie"], dtype=float)  # RUL_FD00X.txt, non plafonnee
    cles_lignes = contexte["train"][["unit_number", "time_cycles"]].reset_index(drop=True)

    donnees = {}
    donnees["sous_jeu"] = sous_jeu
    donnees["contexte"] = contexte
    donnees["X"] = X
    donnees["y"] = y
    donnees["groupes"] = groupes
    donnees["plis"] = plis
    donnees["cles_lignes"] = cles_lignes
    donnees["moteurs_test"] = np.asarray(contexte["moteurs_test"])
    donnees["rul_vraie_test"] = rul_vraie_test
    donnees["rul_vraie_test_plafonnee"] = np.minimum(rul_vraie_test, PLAFOND_RUL)
    return donnees


def plancher_rmse_rul_brute(sous_jeu):
    """
    RMSE contre la RUL brute d'un predicteur PARFAIT sur la cible plafonnee (notebook 05, section H.1).

    Un modele entraine sur une cible plafonnee a 125 ne predit jamais beaucoup plus de 125. Meme s'il predit
    exactement min(vraie RUL, 125), il garde une erreur contre la RUL brute a cause des moteurs test dont la
    vraie RUL depasse 125 : c'est un plancher. Il explique pourquoi le RMSE test de FD002 et FD004 est eleve
    (notebook 01b, F.2).

    Parameters
    ----------
    sous_jeu : str
        "FD001" a "FD004".

    Returns
    -------
    float
        Le plancher, en cycles.
    """
    rul_vraie_test = np.asarray(charger_rul_sous_jeu(sous_jeu), dtype=float).ravel()  # RUL_FD00X.txt
    prediction_parfaite = np.minimum(rul_vraie_test, PLAFOND_RUL)  # le meilleur modele possible sur la cible plafonnee
    return rmse(rul_vraie_test, prediction_parfaite)


def decrire_donnees(donnees):
    """
    Section 1 : ce qu'il faut savoir du sous-jeu avant de lire ses resultats.

    Parameters
    ----------
    donnees : dict
        Le resultat de `preparer_donnees`.

    Returns
    -------
    dict
        "sous_jeu", "nb_regimes", "modes_de_panne", "normalisation", "nb_moteurs_train", "nb_lignes_train",
        "nb_moteurs_test", "capteurs_gardes", "capteurs_retires", "nb_features_arbres", "nb_rul_vraie_sup_plafond"
        (moteurs test de vraie RUL > 125), "part_rul_vraie_sup_plafond" (fraction), "rul_vraie_test_max",
        "plancher_rmse_rul_brute" (`plancher_rmse_rul_brute`).
    """
    contexte = donnees["contexte"]
    sous_jeu = donnees["sous_jeu"]
    rul_vraie_test = donnees["rul_vraie_test"]
    nb_sup = int(np.sum(rul_vraie_test > PLAFOND_RUL))  # moteurs que tout modele plafonne sous-estimera

    description = {}
    description["sous_jeu"] = sous_jeu
    description["nb_regimes"] = int(contexte["nb_regimes"])
    description["modes_de_panne"] = list(DESCRIPTION_SOUS_JEUX[sous_jeu]["modes_de_panne"])
    description["normalisation"] = contexte["normalisation"]
    description["nb_moteurs_train"] = int(len(np.unique(donnees["groupes"])))
    description["nb_lignes_train"] = int(len(donnees["y"]))
    description["nb_moteurs_test"] = int(len(rul_vraie_test))
    description["capteurs_gardes"] = list(contexte["capteurs"])
    description["capteurs_retires"] = list(contexte["capteurs_retires"])
    description["nb_features_arbres"] = int(donnees["X"].shape[1])
    description["nb_rul_vraie_sup_plafond"] = nb_sup
    description["part_rul_vraie_sup_plafond"] = float(nb_sup / len(rul_vraie_test))
    description["rul_vraie_test_max"] = float(np.max(rul_vraie_test))
    description["plancher_rmse_rul_brute"] = plancher_rmse_rul_brute(sous_jeu)
    return description


# ---------------------------------------------------------------------------------------------------
# Section 3 : validation croisee, relue (arbres : sorties/resultats/refonte/ ; reseaux : docs/resultats/04h/)
# ---------------------------------------------------------------------------------------------------

def lire_validation_arbres(donnees, modeles):
    """
    Relit les predictions hors-pli des modeles a arbres (reglages finaux) et les remet dans l'ordre du train.

    Meme methode que le notebook 05 (G.1) : fusion sur (moteur, cycle), puis controle que la cible du fichier
    est celle de la preparation et que le RMSE de chaque pli, recalcule ici, est celui ecrit par le script.

    Parameters
    ----------
    donnees : dict
        Le resultat de `preparer_donnees`.
    modeles : list de str
        Les modeles disponibles ; seuls ceux de src.inference.MODELES_ARBRES sont lus.

    Returns
    -------
    (dict, pandas.DataFrame)
        Les predictions hors-pli (cle -> numpy, une par ligne du train) et le resume du script (une ligne par
        modele : jeu de features, nombre de features, duree de la validation croisee...).

    Raises
    ------
    FileNotFoundError
        Si les fichiers de scripts/valider_arbres_finaux.py manquent.
    """
    sous_jeu = donnees["sous_jeu"]
    for chemin in [FICHIER_CV_OOF, FICHIER_CV_RESUME, FICHIER_CV_PAR_PLI]:
        if not os.path.exists(chemin):
            raise FileNotFoundError(f"{chemin} absent : lancer {COMMANDE_CV_ARBRES}")

    resume = pd.read_csv(FICHIER_CV_RESUME)
    resume = resume[resume["sous_jeu"] == sous_jeu].set_index("modele")
    par_pli_script = pd.read_csv(FICHIER_CV_PAR_PLI)
    par_pli_script = par_pli_script[par_pli_script["sous_jeu"] == sous_jeu]
    oof_fichier = pd.read_csv(FICHIER_CV_OOF)
    oof_fichier = oof_fichier[oof_fichier["sous_jeu"] == sous_jeu]

    oof = {}  # cle -> prediction hors-pli de chaque ligne du train
    lignes_resume = []
    for cle in modeles:
        if cle not in inf.MODELES_ARBRES:
            continue  # les reseaux n'ont pas de predictions hors-pli enregistrees
        jeu = resume.loc[cle, "jeu_features"]  # "base" (reference naive) ou "ecart_initial_20"
        selection = oof_fichier[(oof_fichier["modele"] == cle) & (oof_fichier["jeu_features"] == jeu)]
        # fusion sur (moteur, cycle) : les predictions sont remises dans l'ordre exact des lignes du train
        fusion = donnees["cles_lignes"].merge(selection, on=["unit_number", "time_cycles"], how="left",
                                             validate="one_to_one")
        assert fusion["rul_predite_oof"].notna().all(), f"{sous_jeu}/{cle} : une ligne du train sans prediction"
        assert np.allclose(fusion["rul_cappee"].to_numpy(dtype=float), donnees["y"]), f"{sous_jeu}/{cle} : cible differente"
        oof[cle] = fusion["rul_predite_oof"].to_numpy(dtype=float)

        # controle : le RMSE de chaque pli recalcule ici = celui du script (memes plis, memes predictions)
        recalcule = metriques_regression_par_pli(donnees["y"], oof[cle], donnees["plis"])["rmse"].to_numpy()
        ecrit = par_pli_script[par_pli_script["modele"] == cle].sort_values("pli")["rmse"].to_numpy()
        assert np.allclose(recalcule, ecrit, rtol=0, atol=1e-6), f"{sous_jeu}/{cle} : RMSE par pli different du script"

        ligne = {}
        ligne["cle"] = cle
        ligne["modele"] = nom_affiche(cle)
        ligne["jeu_features"] = jeu
        ligne["nb_features"] = int(resume.loc[cle, "nb_features"])
        ligne["description_modele"] = resume.loc[cle, "description_modele"]
        ligne["duree_cv_s"] = float(resume.loc[cle, "duree_cv_s"])
        ligne["date_cv"] = resume.loc[cle, "date"]
        lignes_resume.append(ligne)
    return oof, pd.DataFrame(lignes_resume)


def prefixe_fichiers_04h(sous_jeu):
    """
    Prefixe des fichiers du notebook 04h pour un sous-jeu ("04h_" pour FD001, "04h_fd002_" pour FD002, etc.).

    Parameters
    ----------
    sous_jeu : str
        "FD001" a "FD004".

    Returns
    -------
    str
        Le prefixe.
    """
    if sous_jeu == "FD001":
        return "04h_"
    return "04h_" + sous_jeu.lower() + "_"


def lire_validation_reseau(sous_jeu, modele):
    """
    Relit la validation croisee d'un reseau publie (notebook 04h) et verifie qu'elle est celle des metadonnees.

    Quel resultat du 04h correspond au reseau de la demo :
    - le LSTM de chaque sous-jeu est la moyenne de 5 LSTM (ENSEMBLES_RETENUS) : fichier 04h_..._ensemble.json,
      cle ensembles / lstm_5_graines (RMSE seulement, aucune autre metrique n'y est enregistree) ;
    - le CNN 1D est un seul reseau (graine 42) : fichier 04h_..._confirmation_cnn1d.json, le candidat Optuna s'il
      a ete retenu ("retenu" vrai, FD001 et FD003), sinon la reference du 03b (FD002 et FD004), sur les 3
      decoupages neufs.

    Parameters
    ----------
    sous_jeu : str
        "FD001" a "FD004".
    modele : str
        "lstm" ou "cnn1d".

    Returns
    -------
    dict
        "cle", "modele", "rmse_cv_moyen" (moyenne des 3 decoupages neufs, = metadonnees), "rmse_cv_ecart_type"
        (entre les 3 decoupages, ddof=0), "rmse_par_decoupage" (liste), "graines_decoupages" (liste),
        "rmse_par_pli" (liste des 15 plis, ou None pour l'ensemble), "mae_oof_moyen", "nasa_oof_moyen_par_ligne",
        "surestimes_oof_moyen", "surestimes_pct_oof_moyen" (en % des lignes), "biais_oof_moyen" (moyennes sur
        les 3 decoupages ; NaN pour l'ensemble), "metriques_enregistrees" (texte), "source" (fichier lu),
        "protocole".

    Raises
    ------
    AssertionError
        Si la moyenne relue differe de celle ecrite dans les metadonnees du fichier de predictions.
    """
    prefixe = prefixe_fichiers_04h(sous_jeu)
    graines_decoupages = [1000, 1001, 1002]  # les 3 decoupages neufs du notebook 04h (jamais vus par Optuna)
    rmse_par_decoupage = []
    rmse_par_pli = None
    metriques_oof_par_decoupage = []

    est_ensemble = sous_jeu in ENSEMBLES_RETENUS and ENSEMBLES_RETENUS[sous_jeu]["reseau"] == modele
    if est_ensemble:
        source = prefixe + "ensemble.json"
        with open(os.path.join(DOSSIER_RESULTATS_04H, source), encoding="utf-8") as fichier:
            resultats = json.load(fichier)
        for valeur in resultats["ensembles"]["lstm_5_graines"]["rmse_cv_par_decoupage"]:
            rmse_par_decoupage.append(float(valeur))
    else:
        source = prefixe + "confirmation_" + modele + ".json"
        with open(os.path.join(DOSSIER_RESULTATS_04H, source), encoding="utf-8") as fichier:
            resultats = json.load(fichier)
        # candidat Optuna s'il a passe la regle de confirmation, sinon la reference du 03b
        if resultats["comparaison"]["retenu"]:
            cle_cv = "cv_candidat"
        else:
            cle_cv = "cv_reference"
        source = source + " (" + cle_cv + ")"
        rmse_par_pli = []
        for graine in graines_decoupages:
            bloc = resultats[cle_cv][str(graine)]
            rmse_par_decoupage.append(float(bloc["rmse_cv_moyen"]))
            for valeur in bloc["rmse_par_pli"]:
                rmse_par_pli.append(float(valeur))
            metriques_oof_par_decoupage.append(bloc["metriques_oof"])

    # controle : la moyenne relue est celle que la demo publie dans les metadonnees du fichier de predictions
    metadonnees = lire_metadonnees(sous_jeu, modele)
    moyenne = float(np.mean(rmse_par_decoupage))
    publie = float(metadonnees["metriques_validation"]["rmse_cv_decoupages_neufs"])
    assert abs(moyenne - publie) < 1e-9, f"{sous_jeu}/{modele} : validation 04h ({moyenne}) != metadonnees ({publie})"

    resultat = {}
    resultat["cle"] = modele
    resultat["modele"] = nom_affiche(modele)
    resultat["rmse_cv_moyen"] = moyenne
    resultat["rmse_cv_ecart_type"] = float(np.std(rmse_par_decoupage))  # entre les 3 decoupages, ddof=0
    resultat["rmse_par_decoupage"] = rmse_par_decoupage
    resultat["graines_decoupages"] = graines_decoupages
    resultat["rmse_par_pli"] = rmse_par_pli
    # moyennes des metriques hors-pli des 3 decoupages (NaN pour l'ensemble : non enregistrees par le 04h)
    for nom_sortie, nom_04h in [("mae_oof_moyen", "mae"), ("surestimes_oof_moyen", "nb_surestimes"),
                                ("biais_oof_moyen", "biais")]:  # moyennes sur les 3 decoupages
        if len(metriques_oof_par_decoupage) == 0:
            resultat[nom_sortie] = np.nan
        else:
            valeurs = []
            for bloc in metriques_oof_par_decoupage:
                valeurs.append(float(bloc[nom_04h]))
            resultat[nom_sortie] = float(np.mean(valeurs))
    if len(metriques_oof_par_decoupage) == 0:
        # ensemble de LSTM : le 04h n'a enregistre que le RMSE de chaque decoupage
        resultat["nasa_oof_moyen_par_ligne"] = np.nan
        resultat["surestimes_pct_oof_moyen"] = np.nan
    else:
        valeurs_nasa = []
        valeurs_pct = []
        for bloc in metriques_oof_par_decoupage:
            valeurs_nasa.append(float(bloc["score_nasa"]) / float(bloc["nb_lignes"]))  # score NASA moyen par prediction
            valeurs_pct.append(100 * float(bloc["nb_surestimes"]) / float(bloc["nb_lignes"]))  # en % des lignes
        resultat["nasa_oof_moyen_par_ligne"] = float(np.mean(valeurs_nasa))
        resultat["surestimes_pct_oof_moyen"] = float(np.mean(valeurs_pct))
    # ce que le 04h a enregistre pour ce reseau (dit sous le tableau de validation)
    if est_ensemble:
        resultat["metriques_enregistrees"] = "RMSE par découpage seulement (ensemble de 5 LSTM)"
    else:
        resultat["metriques_enregistrees"] = "RMSE par pli et métriques hors-pli (MAE, NASA, surestimés, biais)"
    resultat["source"] = "docs/resultats/04h/" + source
    resultat["protocole"] = PROTOCOLE_VALIDATION_RESEAUX
    return resultat


# ---------------------------------------------------------------------------------------------------
# Section 4 : les 5 baselines du protocole (deplace du notebook 05, section G.2)
# ---------------------------------------------------------------------------------------------------

def fabrique_baseline(nom):
    """
    Renvoie une baseline neuve (non entrainee), sauf l'aleatoire qui est traitee a part.

    Parameters
    ----------
    nom : str
        "Moyenne", "Mediane", "Constante 125" ou "Ridge".

    Returns
    -------
    estimateur scikit-learn
        Le modele non entraine ; Ridge est precede d'un StandardScaler appris sur le train du pli.
    """
    if nom == "Moyenne":
        return DummyRegressor(strategy="mean")
    if nom == "Mediane":
        return DummyRegressor(strategy="median")
    if nom == "Constante 125":
        return DummyRegressor(strategy="constant", constant=PLAFOND_RUL)
    if nom == "Ridge":
        # centrer-reduire chaque variable (time_cycles n'est pas entre 0 et 1), puis regression lineaire, alpha non regle
        return Pipeline([("echelle", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    raise ValueError(f"baseline inconnue : {nom}")


def entrainer_baselines(donnees):
    """
    Entraine les 5 baselines sur les plis de la validation croisee, puis sur tout le train pour le test.

    Code du notebook 05 (G.2), applique au sous-jeu : chaque baseline est entrainee sur les moteurs
    d'entrainement de chaque pli, puis predit les moteurs de validation ; pour le test, elle est entrainee sur
    tous les moteurs d'entrainement. L'erreur sur les moteurs d'entrainement (moyenne des 5 plis) sert au
    diagnostic de sur-apprentissage (protocole section 5.1).

    Parameters
    ----------
    donnees : dict
        Le resultat de `preparer_donnees`.

    Returns
    -------
    dict
        "oof" (nom -> predictions hors-pli), "test" (nom -> prediction au dernier cycle des moteurs test,
        ramenee a 0 si negative), "train" (nom -> moyenne et ecart-type des metriques d'entrainement des 5 plis,
        format de `resumer_plis`).
    """
    X = donnees["X"]
    y = donnees["y"]
    plis = donnees["plis"]
    X_test_dernier = donnees["contexte"]["X_test_dernier"]
    nb_moteurs_test = len(donnees["rul_vraie_test"])

    oof = {}
    test = {}
    metriques_train = {}  # nom -> moyenne et ecart-type des metriques d'entrainement des 5 plis
    for nom in BASELINES:
        predictions_oof = np.full(len(y), np.nan)  # sera rempli pli par pli
        lignes_train = []  # metriques sur les moteurs d'entrainement, une ligne par pli
        generateur_val = np.random.default_rng(GRAINE_BASELINES)  # aleatoire : graine fixe, un tirage par pli
        generateur_train = np.random.default_rng(GRAINE_BASELINES + 1)
        for idx_train, idx_val in plis:
            if nom == "Aleatoire uniforme":
                predictions_oof[idx_val] = generateur_val.uniform(0, PLAFOND_RUL, size=len(idx_val))
                predictions_train = generateur_train.uniform(0, PLAFOND_RUL, size=len(idx_train))
            else:
                modele = fabrique_baseline(nom)
                modele.fit(X.iloc[idx_train], y[idx_train])  # appris sur les moteurs d'entrainement du pli
                predictions_oof[idx_val] = modele.predict(X.iloc[idx_val])
                predictions_train = modele.predict(X.iloc[idx_train])
            lignes_train.append(metriques_regression(y[idx_train], predictions_train))
        assert not np.isnan(predictions_oof).any()  # chaque ligne a recu sa prediction hors-pli
        oof[nom] = predictions_oof
        metriques_train[nom] = resumer_plis(pd.DataFrame(lignes_train), colonne_pli="aucune")

        # test : modele appris sur tous les moteurs, prediction au dernier cycle, ramenee a 0 si negative (comme les modeles)
        if nom == "Aleatoire uniforme":
            predictions_test = np.random.default_rng(GRAINE_BASELINES).uniform(0, PLAFOND_RUL, size=nb_moteurs_test)
        else:
            modele = fabrique_baseline(nom)
            modele.fit(X, y)
            predictions_test = modele.predict(X_test_dernier)
        test[nom] = np.clip(np.asarray(predictions_test, dtype=float), a_min=0, a_max=None)

    resultat = {}
    resultat["oof"] = oof
    resultat["test"] = test
    resultat["train"] = metriques_train
    return resultat


# ---------------------------------------------------------------------------------------------------
# Section 4 : predictions test des modeles, lues dans les fichiers du contrat (les memes que la demo)
# ---------------------------------------------------------------------------------------------------

def lire_predictions_test(donnees, modeles):
    """
    Relit la prediction au dernier cycle de chaque moteur test, pour chaque modele, dans les fichiers de la demo.

    Principe 4 du plan : la demo et les notebooks lisent les memes resultats. Controles : la RUL vraie du fichier
    est celle de RUL_FD00X.txt, et les metriques de test des metadonnees se recalculent a l'identique depuis le
    fichier (`verifier_coherence` de src/contrat_predictions.py, celle du tableau de bord).

    Parameters
    ----------
    donnees : dict
        Le resultat de `preparer_donnees`.
    modeles : list de str
        Les modeles disponibles.

    Returns
    -------
    (dict, pandas.DataFrame)
        Les predictions (cle -> numpy, dans l'ordre de donnees["moteurs_test"]) et le controle de coherence
        (une ligne par modele et par metrique : "cle", "metrique", "metadonnees", "recalculee", "ecart", "coherent").

    Raises
    ------
    AssertionError
        Si la RUL vraie d'un fichier differe de RUL_FD00X.txt, ou si une metrique publiee ne se recalcule pas.
    """
    sous_jeu = donnees["sous_jeu"]
    predictions = {}
    controles = []
    for cle in modeles:
        tableau, metadonnees = lire_predictions(sous_jeu, cle)  # tous les cycles de tous les moteurs test
        derniers = extraire_dernier_cycle(tableau).set_index("unit_number").loc[donnees["moteurs_test"]]
        assert np.allclose(derniers["rul_vraie"].to_numpy(dtype=float), donnees["rul_vraie_test"]), f"{cle} : RUL vraie"
        # prediction au dernier cycle, ramenee a 0 si negative (meme regle que metriques_dernier_cycle)
        predictions[cle] = np.clip(derniers["rul_predite"].to_numpy(dtype=float), a_min=0, a_max=None)

        coherence = verifier_coherence(tableau, metadonnees)  # metriques publiees = metriques recalculees ?
        coherence.insert(0, "cle", cle)
        controles.append(coherence)
        assert coherence["coherent"].all(), f"{sous_jeu}/{cle} : metriques publiees non recalculees a l'identique"
    return predictions, pd.concat(controles, ignore_index=True)


# ---------------------------------------------------------------------------------------------------
# Tableaux du protocole (sections 3 a 6 du plan 5.3)
# ---------------------------------------------------------------------------------------------------

def tableaux_regression(donnees, oof, test, train_baselines, ordre):
    """
    Toutes les metriques de regression (protocole section 3), en validation et sur le test brut et plafonne.

    Parameters
    ----------
    donnees : dict
        Le resultat de `preparer_donnees`.
    oof : dict
        Cle -> predictions hors-pli (modeles a arbres et baselines ; pas les reseaux).
    test : dict
        Cle -> prediction au dernier cycle de chaque moteur test (tous les modeles et baselines).
    train_baselines : dict
        Nom de baseline -> metriques d'entrainement resumees (`entrainer_baselines`, cle "train").
    ordre : list de str
        L'ordre des lignes (modeles disponibles puis baselines).

    Returns
    -------
    dict
        - "validation_par_pli" : DataFrame, une ligne par (cle, pli), toutes les cles de `metriques_regression` ;
        - "validation" : DataFrame, une ligne par cle ayant des predictions hors-pli, colonnes "<metrique>_moyenne"
          et "<metrique>_ecart_type" (entre les 5 plis, ddof=0) ;
        - "test_brute" / "test_plafonnee" : DataFrame, une ligne par cle, metriques au dernier cycle contre la
          RUL brute de RUL_FD00X.txt / contre min(RUL brute, 125) ;
        - "train_baselines" : DataFrame, une ligne par baseline, metriques d'entrainement (moyenne des plis).
        Chaque DataFrame a une colonne "cle" et une colonne "modele" (nom affiche).
    """
    lignes_par_pli = []
    lignes_validation = []
    lignes_brute = []
    lignes_plafonnee = []
    for cle in ordre:
        if cle in oof:
            par_pli = metriques_regression_par_pli(donnees["y"], oof[cle], donnees["plis"])
            par_pli.insert(0, "modele", nom_affiche(cle))
            par_pli.insert(0, "cle", cle)
            lignes_par_pli.append(par_pli)
            resume = resumer_plis(par_pli.drop(columns=["cle", "modele"]))
            ligne = {"cle": cle, "modele": nom_affiche(cle)}
            for colonne in resume:
                ligne[colonne] = resume[colonne]
            lignes_validation.append(ligne)
        # test officiel : une prediction par moteur, contre la RUL brute puis contre la RUL plafonnee
        ligne_brute = {"cle": cle, "modele": nom_affiche(cle)}
        metriques_brute = metriques_regression(donnees["rul_vraie_test"], test[cle])
        for colonne in metriques_brute:
            ligne_brute[colonne] = metriques_brute[colonne]
        lignes_brute.append(ligne_brute)
        ligne_plafonnee = {"cle": cle, "modele": nom_affiche(cle)}
        metriques_plafonnee = metriques_regression(donnees["rul_vraie_test_plafonnee"], test[cle])
        for colonne in metriques_plafonnee:
            ligne_plafonnee[colonne] = metriques_plafonnee[colonne]
        lignes_plafonnee.append(ligne_plafonnee)

    lignes_train = []
    for nom in BASELINES:
        ligne = {"cle": nom, "modele": nom}
        for colonne in train_baselines[nom]:
            ligne[colonne] = train_baselines[nom][colonne]
        lignes_train.append(ligne)

    tableaux = {}
    tableaux["validation_par_pli"] = pd.concat(lignes_par_pli, ignore_index=True)
    tableaux["validation"] = pd.DataFrame(lignes_validation)
    tableaux["test_brute"] = pd.DataFrame(lignes_brute)
    tableaux["test_plafonnee"] = pd.DataFrame(lignes_plafonnee)
    tableaux["train_baselines"] = pd.DataFrame(lignes_train)
    return tableaux


def tableau_tranches(donnees, oof, test, ordre):
    """
    Erreur par tranche de RUL vraie (protocole section 3, notebook 05 G.4), en validation et sur le test.

    Parameters
    ----------
    donnees : dict
        Le resultat de `preparer_donnees`.
    oof : dict
        Cle -> predictions hors-pli (pas de reseau).
    test : dict
        Cle -> prediction au dernier cycle des moteurs test.
    ordre : list de str
        Ordre des modeles.

    Returns
    -------
    pandas.DataFrame
        Une ligne par (cle, ensemble, tranche non vide) : "cle", "modele", "ensemble" ("validation" ou
        "test (RUL brute)"), "tranche", "n" (lignes ou moteurs), "mae", "rmse", "biais".
    """
    lignes = []
    for cle in ordre:
        ensembles = []
        if cle in oof:
            ensembles.append(("validation", donnees["y"], oof[cle]))
        ensembles.append(("test (RUL brute)", donnees["rul_vraie_test"], test[cle]))
        for ensemble, vrais, predits in ensembles:
            tranches = erreurs_par_tranche_rul(vrais, predits)  # une ligne par tranche
            for indice in tranches.index:
                source = tranches.loc[indice]
                if source["n"] == 0:
                    continue  # tranche vide (> 125 en validation : la cible est plafonnee)
                ligne = {}
                ligne["cle"] = cle
                ligne["modele"] = nom_affiche(cle)
                ligne["ensemble"] = ensemble
                ligne["tranche"] = source["tranche"]
                ligne["n"] = int(source["n"])
                ligne["mae"] = float(source["mae"])
                ligne["rmse"] = float(source["rmse"])
                ligne["biais"] = float(source["biais"])
                lignes.append(ligne)
    return pd.DataFrame(lignes)


def tableau_litterature(sous_jeu, test_brute, test_plafonnee, modeles):
    """
    Comparaison a la litterature (notebook 05, section H.2) pour ce sous-jeu.

    Les articles plafonnent leurs etiquettes d'entrainement mais ne disent pas si la RUL vraie du test l'est :
    on met donc nos deux RMSE (contre la RUL brute et contre la RUL plafonnee) a cote de leurs chiffres
    (recopies, jamais recalcules). Lecture et precautions : notebook 05, H.3 (un seul entrainement contre une
    moyenne de 10 ; convention du test non ecrite par les articles).

    Parameters
    ----------
    sous_jeu : str
        "FD001" a "FD004".
    test_brute : pandas.DataFrame
        Le tableau "test_brute" de `tableaux_regression`.
    test_plafonnee : pandas.DataFrame
        Le tableau "test_plafonnee" de `tableaux_regression`.
    modeles : list de str
        Les modeles disponibles (la reference naive n'est pas reprise : ce n'est pas un modele publie).

    Returns
    -------
    pandas.DataFrame
        Colonnes "source", "modele", "fenetre", "entrainements", "verite_test", "rmse_test" (NaN : non publie
        pour ce sous-jeu).
    """
    lignes = []
    for article in ARTICLES:
        ligne = {}
        ligne["source"] = article["source"]
        ligne["modele"] = article["modele"]
        ligne["fenetre"] = article["fenetre"]
        ligne["entrainements"] = article["entrainements"]
        ligne["verite_test"] = "non precisee"
        ligne["rmse_test"] = article[sous_jeu]
        lignes.append(ligne)

    brute = test_brute.set_index("cle")
    plafonnee = test_plafonnee.set_index("cle")
    for cle in modeles:
        if cle == "reference_naive":
            continue  # la reference naive n'a pas d'equivalent dans les articles
        for tableau, verite in [(plafonnee, "plafonnee a 125"), (brute, "brute")]:
            ligne = {}
            ligne["source"] = "ce projet"
            ligne["modele"] = nom_affiche(cle)
            if cle in inf.MODELES_RESEAUX:
                ligne["fenetre"] = "fenetre du reseau (notebook 04h)"
            else:
                ligne["fenetre"] = "1 cycle (capteurs lisses sur 5, ecart aux 20 premiers)"
            if cle == "lstm":
                ligne["entrainements"] = "moyenne de 5 reseaux (graines 42 a 46)"
            else:
                ligne["entrainements"] = "1"
            ligne["verite_test"] = verite
            ligne["rmse_test"] = float(tableau.loc[cle, "rmse"])
            lignes.append(ligne)
    return pd.DataFrame(lignes)


def tableau_classification(donnees, oof, test, ordre):
    """
    Lecture en classification "panne dans les N prochains cycles ?" (protocole section 4, notebook 05 G.5).

    Approche 1 du notebook 04b (celle de src/alertes.py et de la demo) : alerte si RUL predite <= seuil. Deux
    seuils : "defaut (= N)" et "choisi sur validation" (`choisir_seuil_alerte` sur les predictions hors-pli :
    rappel >= 0,95 puis meilleure precision), applique tel quel au test, qui ne sert jamais a le choisir. Les
    reseaux n'ont pas de predictions hors-pli enregistrees : seul leur seuil "defaut (= N)" est evalue, et leurs
    colonnes de validation valent NaN.

    Parameters
    ----------
    donnees : dict
        Le resultat de `preparer_donnees`.
    oof : dict
        Cle -> predictions hors-pli.
    test : dict
        Cle -> prediction au dernier cycle des moteurs test.
    ordre : list de str
        Ordre des modeles.

    Returns
    -------
    pandas.DataFrame
        Une ligne par (cle, N, regle de seuil) : "cle", "modele", "N", "regle_seuil", "seuil", puis
        "val_<mesure>" et "test_<mesure>" pour vn, fp, fn, vp, nb_positifs, proportion_positifs, accuracy,
        precision, rappel, f1, auc_roc, precision_moyenne ; "val_rappel_plis_moyenne" et
        "val_rappel_plis_ecart_type" (rappel de chaque pli de validation, ddof=0).
    """
    mesures = ["vn", "fp", "fn", "vp", "nb_positifs", "proportion_positifs", "accuracy", "precision", "rappel", "f1",
               "auc_roc", "precision_moyenne"]
    y = donnees["y"]
    lignes = []
    for cle in ordre:
        for horizon in HORIZONS:
            regles = [("defaut (= N)", float(horizon))]
            if cle in oof:  # seuil choisi sur la validation, jamais sur le test
                choix_seuil = choisir_seuil_alerte(y, oof[cle], horizon, rappel_min=RAPPEL_MINIMUM)
                regles.append(("choisi sur validation", choix_seuil["seuil_alerte"]))
            for regle_seuil, seuil in regles:
                resultat_test = metriques_classification_alerte(donnees["rul_vraie_test"], test[cle], horizon, seuil)
                ligne = {}
                ligne["cle"] = cle
                ligne["modele"] = nom_affiche(cle)
                ligne["N"] = horizon
                ligne["regle_seuil"] = regle_seuil
                ligne["seuil"] = seuil
                if cle in oof:
                    validation = metriques_classification_alerte(y, oof[cle], horizon, seuil)  # 5 plis reunis
                    rappels_plis = []  # rappel de chaque pli, pour mesurer sa variation
                    for idx_train, idx_val in donnees["plis"]:
                        rappel_pli = metriques_classification_alerte(y[idx_val], oof[cle][idx_val], horizon, seuil)["rappel"]
                        rappels_plis.append(rappel_pli)
                for nom_mesure in mesures:
                    if cle in oof:
                        ligne["val_" + nom_mesure] = validation[nom_mesure]
                    else:
                        ligne["val_" + nom_mesure] = np.nan  # reseau : pas de predictions hors-pli
                    ligne["test_" + nom_mesure] = resultat_test[nom_mesure]
                if cle in oof:
                    ligne["val_rappel_plis_moyenne"] = float(np.mean(rappels_plis))
                    ligne["val_rappel_plis_ecart_type"] = float(np.std(rappels_plis))  # ddof = 0, comme resumer_plis
                else:
                    ligne["val_rappel_plis_moyenne"] = np.nan
                    ligne["val_rappel_plis_ecart_type"] = np.nan
                lignes.append(ligne)
    return pd.DataFrame(lignes)


def seuils_matrices(classification, horizon=HORIZON_MATRICES):
    """
    Seuil utilise pour la matrice de confusion de chaque modele a un N donne (section 5 du plan 5.3).

    Le seuil choisi sur la validation quand il existe (arbres, baselines), sinon le seuil par defaut N (reseaux).

    Parameters
    ----------
    classification : pandas.DataFrame
        Le resultat de `tableau_classification`.
    horizon : int, optionnel
        N (30 par defaut).

    Returns
    -------
    dict
        Cle -> {"seuil": float, "regle_seuil": str}.
    """
    sous_tableau = classification[classification["N"] == horizon]
    seuils = {}
    for indice in sous_tableau.index:
        ligne = sous_tableau.loc[indice]
        cle = ligne["cle"]
        # la regle "choisi sur validation" remplace "defaut (= N)" quand elle existe (elle vient apres dans le tableau)
        if cle not in seuils or ligne["regle_seuil"] == "choisi sur validation":
            seuils[cle] = {"seuil": float(ligne["seuil"]), "regle_seuil": ligne["regle_seuil"]}
    return seuils


def erreurs_moteur_par_modele(donnees, oof, test, cles):
    """
    Valeurs appariees moteur par moteur, pour les comparaisons de la section 6 (notebook 05, G.6).

    Parameters
    ----------
    donnees : dict
        Le resultat de `preparer_donnees`.
    oof : dict
        Cle -> predictions hors-pli.
    test : dict
        Cle -> prediction au dernier cycle des moteurs test.
    cles : list de str
        Les modeles compares.

    Returns
    -------
    dict
        - "test" : DataFrame, une ligne par moteur test ("unit_number" puis une colonne par cle : erreur absolue
          au dernier cycle) ;
        - "test_signe" : meme forme, ecart signe (predit - vrai : positif = surestimation) ;
        - "validation" : DataFrame, une ligne par moteur d'entrainement ("unit_number" puis une colonne par cle
          ayant des predictions hors-pli : RMSE hors-pli du moteur) ;
        - "plis" : DataFrame, une ligne par pli (RMSE de chaque pli, cles ayant des predictions hors-pli).
    """
    test_absolu = pd.DataFrame({"unit_number": donnees["moteurs_test"]})
    test_signe = pd.DataFrame({"unit_number": donnees["moteurs_test"]})
    validation = None
    plis = pd.DataFrame({"pli": np.arange(1, len(donnees["plis"]) + 1)})
    for cle in cles:
        ecart = test[cle] - donnees["rul_vraie_test"]  # positif = surestimation
        test_absolu[cle] = np.abs(ecart)
        test_signe[cle] = ecart
        if cle in oof:
            par_moteur = erreurs_par_moteur(donnees["y"], oof[cle], donnees["groupes"])  # trie par moteur
            if validation is None:
                validation = pd.DataFrame({"unit_number": par_moteur["unit_number"].to_numpy()})
            validation[cle] = par_moteur["rmse"].to_numpy()
            plis[cle] = metriques_regression_par_pli(donnees["y"], oof[cle], donnees["plis"])["rmse"].to_numpy()
    resultat = {}
    resultat["test"] = test_absolu
    resultat["test_signe"] = test_signe
    resultat["validation"] = validation
    resultat["plis"] = plis
    return resultat


def comparaisons_appariees(erreurs_moteur, cles):
    """
    Compare toutes les paires de modeles, moteur par moteur (protocole section 8, notebook 05 G.6).

    Regle ecrite avant les resultats (celle de `comparer_modeles_par_paires`) : A est declare meilleur que B si la
    p-valeur de Wilcoxon bilaterale, multipliee par le nombre de paires (Bonferroni), est < 0,05 et que la
    difference moyenne A - B est negative ; sinon pas de difference significative.

    Parameters
    ----------
    erreurs_moteur : dict
        Le resultat de `erreurs_moteur_par_modele`.
    cles : list de str
        Les modeles compares (ordre des paires).

    Returns
    -------
    dict
        "test" (erreur absolue par moteur test, tous les modeles), "validation" (RMSE hors-pli par moteur
        d'entrainement, modeles ayant des predictions hors-pli), "plis" (RMSE des 5 plis, pour information : avec
        5 paires, Wilcoxon ne peut pas descendre sous 0,0625). Chacun : DataFrame de `comparer_modeles_par_paires`
        (colonnes modele_a, modele_b en noms affiches, n, difference_moyenne, ..., conclusion), ou None.
    """
    comparaisons = {}
    for jeu in ["test", "validation", "plis"]:
        tableau = erreurs_moteur[jeu]
        valeurs = {}  # nom affiche -> valeurs appariees
        if tableau is not None:
            for cle in cles:
                if cle in tableau.columns:
                    valeurs[nom_affiche(cle)] = tableau[cle].to_numpy(dtype=float)
        if len(valeurs) >= 2:
            comparaisons[jeu] = comparer_modeles_par_paires(valeurs, seuil_p=SEUIL_P)
        else:
            comparaisons[jeu] = None  # moins de deux modeles : rien a comparer
    return comparaisons


def tableau_validation(regression_validation, validations_reseaux, ordre, decoupages_neufs=None):
    """
    Section 3 : un seul tableau de validation croisee, RMSE moyen ± ecart-type, avec le protocole de chaque ligne.

    Parameters
    ----------
    regression_validation : pandas.DataFrame
        Le tableau "validation" de `tableaux_regression` (arbres et baselines, plis des arbres).
    validations_reseaux : dict
        Cle -> resultat de `lire_validation_reseau`.
    ordre : list de str
        Ordre des lignes.
    decoupages_neufs : pandas.DataFrame ou None, optionnel
        Le resume de `lire_complements_arbres` (cle "decoupages_resume") : RMSE des arbres sur les 3 decoupages
        neufs du 04h. None : colonne laissee vide pour les arbres.

    Returns
    -------
    pandas.DataFrame
        Une ligne par cle : "cle", "modele", "rmse_cv_moyen", "rmse_cv_ecart_type", "mae_cv_moyen",
        "nasa_moyen_par_ligne", "surestimes_pct", "ecart_type_entre" ("5 plis" ou "3 decoupages"),
        "rmse_decoupages_neufs_moyen", "rmse_decoupages_neufs_ecart_type" (les 3 decoupages du 04h : memes
        decoupages pour arbres et reseaux), "protocole", "note" (ce qui manque, et pourquoi).
    """
    validation = regression_validation.set_index("cle")
    # resume des decoupages neufs des arbres, indexe par cle (vide si les complements manquent)
    neufs = None
    if decoupages_neufs is not None and len(decoupages_neufs) > 0:
        neufs = decoupages_neufs.set_index("cle")
    lignes = []
    for cle in ordre:
        ligne = {"cle": cle, "modele": nom_affiche(cle)}
        if cle in validation.index:
            ligne["rmse_cv_moyen"] = float(validation.loc[cle, "rmse_moyenne"])
            ligne["rmse_cv_ecart_type"] = float(validation.loc[cle, "rmse_ecart_type"])
            ligne["mae_cv_moyen"] = float(validation.loc[cle, "mae_moyenne"])
            ligne["nasa_moyen_par_ligne"] = float(validation.loc[cle, "nasa_moyen_moyenne"])
            ligne["surestimes_pct"] = float(validation.loc[cle, "surestimes_pct_moyenne"])
            ligne["ecart_type_entre"] = "5 plis"
            # les memes modeles sur les decoupages neufs du 04h, s'ils ont ete calcules (arbres, sous-jeux avec reseaux)
            if neufs is not None and cle in neufs.index:
                ligne["rmse_decoupages_neufs_moyen"] = float(neufs.loc[cle, "rmse_moyen"])
                ligne["rmse_decoupages_neufs_ecart_type"] = float(neufs.loc[cle, "rmse_ecart_type"])
            else:
                ligne["rmse_decoupages_neufs_moyen"] = np.nan
                ligne["rmse_decoupages_neufs_ecart_type"] = np.nan
            ligne["protocole"] = PROTOCOLE_VALIDATION_ARBRES
            ligne["note"] = ""
        elif cle in validations_reseaux:
            reseau = validations_reseaux[cle]
            ligne["rmse_cv_moyen"] = reseau["rmse_cv_moyen"]
            ligne["rmse_cv_ecart_type"] = reseau["rmse_cv_ecart_type"]
            ligne["mae_cv_moyen"] = reseau["mae_oof_moyen"]
            ligne["nasa_moyen_par_ligne"] = reseau["nasa_oof_moyen_par_ligne"]
            ligne["surestimes_pct"] = reseau["surestimes_pct_oof_moyen"]  # NaN pour l'ensemble de LSTM
            ligne["ecart_type_entre"] = "3 decoupages"
            # les reseaux n'ont de validation que sur les decoupages neufs : c'est la meme valeur
            ligne["rmse_decoupages_neufs_moyen"] = reseau["rmse_cv_moyen"]
            ligne["rmse_decoupages_neufs_ecart_type"] = reseau["rmse_cv_ecart_type"]
            ligne["protocole"] = reseau["protocole"]
            ligne["note"] = "04h : " + reseau["metriques_enregistrees"]
        else:
            continue  # aucune validation connue pour ce modele
        lignes.append(ligne)
    return pd.DataFrame(lignes)


def lire_complements_arbres(sous_jeu, modeles, validation_par_pli):
    """
    Relit les complements de la validation des arbres : erreur d'entrainement par pli et decoupages neufs du 04h.

    Fichiers de `scripts/valider_arbres_finaux.py --complements` (sorties/resultats/refonte/). Controle : le RMSE
    de validation de chaque pli, ecrit a cote de l'erreur d'entrainement, doit etre celui que ce module recalcule
    a partir des predictions hors-pli (`tableaux_regression`, memes plis, memes modeles).

    Parameters
    ----------
    sous_jeu : str
        "FD001" a "FD004".
    modeles : list de str
        Les modeles disponibles ; seuls ceux de src.inference.MODELES_ARBRES sont lus.
    validation_par_pli : pandas.DataFrame
        Le tableau "validation_par_pli" de `tableaux_regression` (colonnes "cle", "pli", "rmse").

    Returns
    -------
    dict
        - "train_par_pli" : DataFrame (cle, modele, pli, rmse_train, rmse_val, mae_train, mae_val, r2_train,
          r2_val, biais_train, biais_val) ou None si le fichier manque ;
        - "train_resume" : DataFrame, une ligne par modele : moyenne ± ecart-type des 5 plis du RMSE train et
          validation, ecart validation - train, R² train et validation ; ou None ;
        - "decoupages" : DataFrame (cle, modele, graine_plis, rmse_cv_moyen) ou None (sous-jeu sans reseau, ou
          fichier absent, ou fichier sans ligne pour ce sous-jeu : signale dans "manquants") ;
        - "decoupages_resume" : DataFrame (cle, modele, rmse_moyen, rmse_ecart_type entre les 3 decoupages) ou None ;
        - "manquants" : list de str, les fichiers absents (avec la commande a lancer).
    """
    complements = {"train_par_pli": None, "train_resume": None, "decoupages": None, "decoupages_resume": None,
                   "manquants": []}

    # les modeles a arbres disponibles (la reference naive en fait partie : meme script)
    arbres = []
    for cle in modeles:
        if cle in inf.MODELES_ARBRES:
            arbres.append(cle)

    # (a) erreur d'entrainement et de validation, pli par pli
    if not os.path.exists(FICHIER_CV_TRAIN_PAR_PLI):
        complements["manquants"].append(f"{FICHIER_CV_TRAIN_PAR_PLI} (lancer {COMMANDE_COMPLEMENTS})")
    else:
        fichier = pd.read_csv(FICHIER_CV_TRAIN_PAR_PLI)
        fichier = fichier[(fichier["sous_jeu"] == sous_jeu) & (fichier["modele"].isin(arbres))].copy()
        fichier = fichier.rename(columns={"modele": "cle"})
        fichier.insert(1, "modele", fichier["cle"].map(nom_affiche))
        # controle : la validation du fichier = celle recalculee ici depuis les predictions hors-pli
        for indice in fichier.index:
            ligne = fichier.loc[indice]
            # le script numerote les plis a partir de 0, metriques_regression_par_pli a partir de 1
            selection = validation_par_pli[(validation_par_pli["cle"] == ligne["cle"])
                                           & (validation_par_pli["pli"] == ligne["pli"] + 1)]
            assert len(selection) == 1, f"{sous_jeu}/{ligne['cle']} pli {ligne['pli']} : pli introuvable"
            recalcule = float(selection["rmse"].iloc[0])
            assert abs(recalcule - float(ligne["rmse_val"])) < 1e-6, (
                f"{sous_jeu}/{ligne['cle']} pli {ligne['pli']} : RMSE de validation des complements different")
        complements["train_par_pli"] = fichier.reset_index(drop=True)

        # resume : moyenne ± ecart-type (ddof=0, comme les plis de validation) par modele
        lignes = []
        for cle in arbres:
            plis_modele = fichier[fichier["cle"] == cle]
            if len(plis_modele) == 0:
                continue
            ligne = {}
            ligne["cle"] = cle
            ligne["modele"] = nom_affiche(cle)
            ligne["rmse_train_moyen"] = float(np.mean(plis_modele["rmse_train"]))
            ligne["rmse_train_ecart_type"] = float(np.std(plis_modele["rmse_train"]))
            ligne["rmse_val_moyen"] = float(np.mean(plis_modele["rmse_val"]))
            ligne["rmse_val_ecart_type"] = float(np.std(plis_modele["rmse_val"]))
            ligne["ecart_val_moins_train"] = ligne["rmse_val_moyen"] - ligne["rmse_train_moyen"]
            ligne["nb_plis_val_superieure"] = int(np.sum(plis_modele["rmse_val"] > plis_modele["rmse_train"]))
            ligne["r2_train_moyen"] = float(np.mean(plis_modele["r2_train"]))
            ligne["r2_val_moyen"] = float(np.mean(plis_modele["r2_val"]))
            lignes.append(ligne)
        complements["train_resume"] = pd.DataFrame(lignes)

    # (b) decoupages neufs du 04h (seulement les sous-jeux qui ont des reseaux)
    if sous_jeu in ENSEMBLES_RETENUS:
        if not os.path.exists(FICHIER_CV_DECOUPAGES_NEUFS):
            complements["manquants"].append(f"{FICHIER_CV_DECOUPAGES_NEUFS} (lancer {COMMANDE_COMPLEMENTS})")
        else:
            fichier = pd.read_csv(FICHIER_CV_DECOUPAGES_NEUFS)
            fichier = fichier[(fichier["sous_jeu"] == sous_jeu) & (fichier["modele"].isin(arbres))].copy()
            if len(fichier) == 0:
                # fichier calcule avant que ce sous-jeu ait des reseaux : a refaire, et a dire (regle 11)
                complements["manquants"].append(f"{FICHIER_CV_DECOUPAGES_NEUFS} : aucune ligne pour {sous_jeu} "
                                                f"(relancer {COMMANDE_COMPLEMENTS})")
                return complements
            fichier = fichier.rename(columns={"modele": "cle"})
            fichier.insert(1, "modele", fichier["cle"].map(nom_affiche))
            complements["decoupages"] = fichier.reset_index(drop=True)
            lignes = []
            for cle in arbres:
                decoupages_modele = fichier[fichier["cle"] == cle]
                if len(decoupages_modele) == 0:
                    continue
                # les 3 graines du 04h, ni plus ni moins
                assert sorted(decoupages_modele["graine_plis"].tolist()) == GRAINES_DECOUPAGES_NEUFS, (
                    f"{sous_jeu}/{cle} : decoupages neufs incomplets")
                ligne = {}
                ligne["cle"] = cle
                ligne["modele"] = nom_affiche(cle)
                ligne["rmse_moyen"] = float(np.mean(decoupages_modele["rmse_cv_moyen"]))
                ligne["rmse_ecart_type"] = float(np.std(decoupages_modele["rmse_cv_moyen"]))  # entre les 3, ddof=0
                lignes.append(ligne)
            complements["decoupages_resume"] = pd.DataFrame(lignes)
    return complements


def liste_non_fait(sous_jeu, modeles, absents, complements=None):
    """
    Ce que le protocole demande et que `evaluer_sous_jeu` ne fait pas (regle 11 : a signaler, pas a cacher).

    Parameters
    ----------
    sous_jeu : str
        "FD001" a "FD004".
    modeles : list de str
        Les modeles disponibles.
    absents : dict
        Les modeles absents et leur raison.
    complements : dict ou None, optionnel
        Le resultat de `lire_complements_arbres` (None : traite comme absent).

    Returns
    -------
    list de str
        Une phrase par manque.
    """
    manques = []
    reseaux_presents = []
    for cle in modeles:
        if cle in inf.MODELES_RESEAUX:
            reseaux_presents.append(nom_affiche(cle))

    # erreur d'entrainement : faite pour les arbres si les complements existent, jamais pour les reseaux
    train_arbres_fait = complements is not None and complements["train_resume"] is not None
    if train_arbres_fait:
        if len(reseaux_presents) > 0:
            manques.append(f"Erreur sur les moteurs d'entraînement (protocole §3 et §5.1) : calculée pour les arbres "
                           f"et les baselines (section 3.1), pas pour {', '.join(reseaux_presents)} (il faudrait "
                           "réentraîner les réseaux pli par pli, ce qui est interdit ici : calcul de nuit du 04h).")
    else:
        manques.append("Erreur sur les moteurs d'entraînement des modèles appris (protocole §3 et §5.1) : non relue "
                       f"(lancer {COMMANDE_COMPLEMENTS}) ; elle n'est donnée que pour les 5 baselines.")

    if len(reseaux_presents) > 0:
        decoupages_fait = complements is not None and complements["decoupages_resume"] is not None
        if decoupages_fait:
            comparaison_moyennes = ("les moyennes arbres / réseaux sont aussi données sur les 3 mêmes découpages "
                                    "neufs (colonne « RMSE découpages neufs »), comparables mais pas appariées")
        else:
            comparaison_moyennes = ("les moyennes de validation arbres / réseaux ne viennent pas des mêmes plis "
                                    f"(lancer {COMMANDE_COMPLEMENTS} pour les arbres sur les découpages neufs)")
        manques.append(f"{', '.join(reseaux_presents)} : validation croisée du notebook 04h sur 3 découpages neufs "
                       "(graines 1000 à 1002), pas sur les plis des arbres, et sans prédictions hors-pli enregistrées. "
                       "Donc : pas de test apparié arbres contre réseaux en validation, pas de tranches de RUL ni de "
                       "seuil d'alerte choisi en validation pour les réseaux (seuil par défaut N) ; "
                       + comparaison_moyennes + ".")
    for cle in absents:
        manques.append(f"{nom_affiche(cle)} : {absents[cle]}.")
    manques.append("Classifieur dédié (protocole §4, approche 2) : non refait ici ; résumé dans 02_choix_du_modele "
                   "(notebook 04b, FD001).")
    manques.append("Courbes d'apprentissage et de validation, optimisation des hyperparamètres (§5, §6) : faites sur "
                   "FD001 seulement (02_choix_du_modele) ; réglages appliqués tels quels à ce sous-jeu.")
    if sous_jeu == "FD001":
        manques.append("Réglages finaux des arbres cherchés par Optuna sur FD001 (notebooks 04f et 04g) : leur "
                       "validation croisée sur FD001 est donc optimiste (biais d'optimisme, protocole §6).")
    return manques


# ---------------------------------------------------------------------------------------------------
# Fonction principale : sections 1 a 6
# ---------------------------------------------------------------------------------------------------

def evaluer_sous_jeu(sous_jeu):
    """
    Deroule le protocole d'evaluation complet sur un sous-jeu (sections 1 a 6 du plan 5.3).

    Rien n'est reentraine sauf les 5 baselines (quelques secondes) : la validation croisee des arbres est relue
    dans sorties/resultats/refonte/ (reglages finaux, memes plis, recalcules ici et verifies), celle des
    reseaux dans docs/resultats/04h/, et les predictions test dans sorties/predictions/ (les fichiers de la demo,
    dont les metriques publiees sont verifiees).

    Parameters
    ----------
    sous_jeu : str
        "FD001" a "FD004".

    Returns
    -------
    dict
        Voir le rapport d'API (cles "sous_jeu", "modeles", "absents", "ordre", "noms", "donnees", "configurations",
        "validation", "test", "regression", "tranches", "litterature", "classification", "seuils_matrices",
        "erreurs_par_moteur", "comparaisons", "non_fait").
    """
    verifier_nom_sous_jeu(sous_jeu)

    # sections 1 et 2 : donnees, plis, modeles disponibles et leurs reglages
    modeles, absents = modeles_publies(sous_jeu)
    donnees = preparer_donnees(sous_jeu)
    description = decrire_donnees(donnees)
    configurations = tableau_configurations(sous_jeu, modeles)

    # section 3 : validation croisee relue (arbres) et resultats du 04h (reseaux)
    oof, resume_cv_arbres = lire_validation_arbres(donnees, modeles)
    validations_reseaux = {}
    for cle in modeles:
        if cle in inf.MODELES_RESEAUX:
            validations_reseaux[cle] = lire_validation_reseau(sous_jeu, cle)

    # section 4 : baselines (entrainees ici) et predictions test des modeles (fichiers de la demo)
    baselines = entrainer_baselines(donnees)
    test, controle_contrat = lire_predictions_test(donnees, modeles)
    for nom in BASELINES:
        oof[nom] = baselines["oof"][nom]
        test[nom] = baselines["test"][nom]
    # la reference naive EST la baseline "Moyenne" : verification (notebook 05, G.2)
    if "reference_naive" in modeles:
        assert np.allclose(oof["Moyenne"], oof["reference_naive"])
        assert np.allclose(test["Moyenne"], test["reference_naive"])

    ordre = []  # les modeles disponibles, puis les 5 baselines
    for cle in modeles:
        ordre.append(cle)
    for nom in BASELINES:
        ordre.append(nom)
    noms = {}
    for cle in ordre:
        noms[cle] = nom_affiche(cle)

    regression = tableaux_regression(donnees, oof, test, baselines["train"], ordre)
    # complements des arbres (erreur d'entrainement, decoupages neufs du 04h), relus s'ils existent
    complements = lire_complements_arbres(sous_jeu, modeles, regression["validation_par_pli"])
    validation = tableau_validation(regression["validation"], validations_reseaux, ordre,
                                    decoupages_neufs=complements["decoupages_resume"])
    tranches = tableau_tranches(donnees, oof, test, ordre)
    litterature = tableau_litterature(sous_jeu, regression["test_brute"], regression["test_plafonnee"], modeles)

    # section 5 : classification
    classification = tableau_classification(donnees, oof, test, ordre)
    seuils = seuils_matrices(classification, HORIZON_MATRICES)

    # section 6 : comparaisons appariees entre les modeles disponibles et Ridge (comme le notebook 05, G.6)
    cles_comparees = []
    for cle in modeles:
        cles_comparees.append(cle)
    cles_comparees.append("Ridge")
    erreurs_moteur = erreurs_moteur_par_modele(donnees, oof, test, cles_comparees)
    comparaisons = comparaisons_appariees(erreurs_moteur, cles_comparees)

    # assemblage du resultat, section par section du plan 5.3 (les notebooks ne lisent que ce dictionnaire)
    resultat = {}
    # identite du sous-jeu, modeles disponibles et absents, ordre et noms d'affichage des lignes
    resultat["sous_jeu"] = sous_jeu
    resultat["modeles"] = modeles
    resultat["absents"] = absents
    resultat["ordre"] = ordre
    resultat["noms"] = noms
    # sections 1 et 2 : description du sous-jeu, reglages des modeles
    resultat["donnees"] = description
    resultat["configurations"] = configurations
    # section 3 : validation croisee (tableau commun, detail par pli, complements des arbres, reseaux du 04h)
    resultat["validation"] = {
        "tableau": validation,
        "par_pli": regression["validation_par_pli"],
        "complet": regression["validation"],
        "train_baselines": regression["train_baselines"],
        "train_arbres": complements["train_resume"],
        "train_arbres_par_pli": complements["train_par_pli"],
        "decoupages_neufs": complements["decoupages"],
        "complements_manquants": complements["manquants"],
        "protocole_decoupages_neufs": PROTOCOLE_DECOUPAGES_NEUFS,
        "arbres": resume_cv_arbres,
        "reseaux": validations_reseaux,
        "plis": donnees["plis"],
        "y": donnees["y"],
        "groupes": donnees["groupes"],
        "oof": oof,
        "protocole_arbres": PROTOCOLE_VALIDATION_ARBRES,
        "protocole_reseaux": PROTOCOLE_VALIDATION_RESEAUX,
    }
    # section 4 : test officiel (predictions au dernier cycle, controle des fichiers de la demo), tranches, litterature
    resultat["test"] = {
        "moteurs": donnees["moteurs_test"],
        "rul_vraie": donnees["rul_vraie_test"],
        "rul_vraie_plafonnee": donnees["rul_vraie_test_plafonnee"],
        "predictions": test,
        "controle_contrat": controle_contrat,
    }
    resultat["regression"] = {
        "test_brute": regression["test_brute"],
        "test_plafonnee": regression["test_plafonnee"],
    }
    resultat["tranches"] = tranches
    resultat["litterature"] = litterature
    # section 5 : classification (un tableau pour N = 15, 30, 50 ; seuils des matrices de confusion)
    resultat["classification"] = classification
    resultat["seuils_matrices"] = seuils
    # section 6 : erreurs par moteur et comparaisons appariees, puis ce qui manque au protocole (regle 11)
    resultat["erreurs_par_moteur"] = erreurs_moteur
    resultat["comparaisons"] = comparaisons
    resultat["non_fait"] = liste_non_fait(sous_jeu, modeles, absents, complements)
    return resultat


# ---------------------------------------------------------------------------------------------------
# Section 7 : verifier un moteur depuis les donnees brutes (notebook 05, J.3 et J.4)
# ---------------------------------------------------------------------------------------------------

def remplacer_arbres_par_reentraines(inference, sous_jeu):
    """
    Remplace dans un objet d'inference les modeles a arbres charges par des modeles reentraines en memoire.

    Sert a "prouve-moi que ce modele vient bien de ce code" (plan, section 6.2, ENTRAINER = True) : meme recette que
    le modele final du script des arbres (`obtenir_modele_arbres` avec forcer=True), rien n'est ecrit sur le disque.

    Parameters
    ----------
    inference : dict
        Le resultat de `src.inference.charger_modeles`.
    sous_jeu : str
        "FD001" a "FD004".

    Returns
    -------
    dict
        Cle -> duree d'entrainement en secondes, pour les modeles reentraines.
    """
    contexte = inf.preparer_contexte(sous_jeu)  # meme preparation, apprise sur le train
    durees = {}
    for cle in inf.MODELES_ARBRES:
        if cle not in inference["modeles"]:
            continue
        artefact, _ = inf.obtenir_modele_arbres(contexte, cle, None, True, False, True)  # forcer, sans sauvegarder
        inference["modeles"][cle]["objet"] = artefact["modele"]
        inference["modeles"][cle]["date"] = artefact["date_entrainement"]
        durees[cle] = artefact["duree_entrainement_s"]
    return durees


def verifier_moteur(sous_jeu, moteur, inference=None, modeles=None, cycles=None, entrainer_arbres=False,
                    seuil_alerte=SEUIL_ALERTE):
    """
    Recalcule, depuis data/test_FD00X.txt, la RUL predite cycle par cycle d'un moteur test par chaque modele.

    Principe 3 du plan : ne lit JAMAIS sorties/predictions/. A chaque cycle t, chaque modele ne recoit que les
    mesures brutes 1..t du moteur, preparees avec ce qui a ete appris sur le train (src/inference.py :
    `predire_trajectoire`, une troncature et une preparation par cycle). Les modeles sont ceux de la demo :
    models/<SOUS_JEU>__<modele>.joblib pour les arbres (reentraines en memoire s'ils manquent, ou si
    `entrainer_arbres` est vrai), models/04h/ pour les reseaux (jamais reentraines). La RUL vraie vient de
    RUL_FD00X.txt (`verite_moteur`). Comparaison aux fichiers de la demo : `comparer_aux_fichiers_demo`.

    Parameters
    ----------
    sous_jeu : str
        "FD001" a "FD004".
    moteur : int
        Le numero du moteur test (ex. 42).
    inference : dict ou None, optionnel
        Un objet de `src.inference.charger_modeles` deja charge (pour verifier plusieurs moteurs sans tout
        recharger) ; None : charge ici, sans rien ecrire sur le disque.
    modeles : list de str ou None, optionnel
        Les modeles voulus (None : tous les modeles disponibles).
    cycles : list de int ou None, optionnel
        Les cycles recalcules (None : tous, du premier au dernier observe).
    entrainer_arbres : bool, optionnel
        Vrai : les modeles a arbres sont reentraines ici en memoire avant de predire (quelques secondes a quelques
        minutes selon le sous-jeu).
    seuil_alerte : float, optionnel
        Seuil d'alerte (30 cycles, celui de la demo) pour le statut au dernier cycle.

    Returns
    -------
    dict
        - "sous_jeu", "moteur", "nb_cycles", "rul_finale" (RUL vraie au dernier cycle, RUL_FD00X.txt) ;
        - "modeles" : les cles des modeles recalcules ; "indisponibles" : cle -> raison ;
        - "trajectoire" : DataFrame, une ligne par cycle recalcule : "time_cycles", une colonne par modele
          (RUL predite), "rul_vraie", "rul_vraie_cappee", "temps_total_ms" ;
        - "dernier_cycle" : DataFrame, une ligne par modele : "cle", "modele", "rul_predite", "rul_vraie", "ecart"
          (predit - vrai), "alerte" (bool : RUL predite <= seuil), "statut_alerte" (`classer_alerte` de
          src/alertes.py : "alerte correcte", "ALERTE MANQUEE", "fausse alerte" ou "rien a signaler") ;
        - "durees_entrainement_s" : cle -> secondes (vide si `entrainer_arbres` est faux) ;
        - "source" : rappel de ce qui a ete lu.
    """
    verifier_nom_sous_jeu(sous_jeu)
    moteur_test = inf.mesures_moteur_test(sous_jeu, moteur)  # data/test_FD00X.txt et RUL_FD00X.txt
    mesures = moteur_test["mesures"]
    if inference is None:
        # entrainer_si_absent : un arbre absent est entraine en memoire ; sauvegarder=False : rien n'est ecrit
        inference = inf.charger_modeles(sous_jeu, entrainer_si_absent=True, sauvegarder=False)
    if inference["sous_jeu"] != sous_jeu:
        raise ValueError(f"L'objet d'inference est celui de {inference['sous_jeu']}, pas de {sous_jeu}.")
    durees = {}
    if entrainer_arbres:
        durees = remplacer_arbres_par_reentraines(inference, sous_jeu)
    if modeles is None:
        modeles = list(inference["modeles"].keys())

    # la prediction a chaque cycle t, avec les seules mesures 1..t
    trajectoire = inf.predire_trajectoire(inference, mesures, cycles=cycles, modeles=modeles)
    # la verite, connue apres coup seulement, ajoutee a cote (meme cycle)
    verite = inf.verite_moteur(mesures, moteur_test["rul_finale"])
    trajectoire = trajectoire.merge(verite, on="time_cycles", how="left", validate="one_to_one")

    # au dernier cycle observe : ecart et statut d'alerte de chaque modele (regle de la demo)
    lignes = []
    derniere_ligne = trajectoire[trajectoire["time_cycles"] == moteur_test["nb_cycles"]]
    if len(derniere_ligne) == 1:  # le dernier cycle fait partie des cycles recalcules
        for cle in modeles:
            rul_predite = float(derniere_ligne[cle].iloc[0])
            ligne = {}
            ligne["cle"] = cle
            ligne["modele"] = nom_affiche(cle)
            ligne["rul_predite"] = rul_predite
            ligne["rul_vraie"] = moteur_test["rul_finale"]
            ligne["ecart"] = rul_predite - moteur_test["rul_finale"]  # positif = surestimation
            alerte, statut = classer_alerte(rul_predite, moteur_test["rul_finale"], seuil_alerte)  # regle de la demo
            ligne["alerte"] = alerte
            ligne["statut_alerte"] = statut
            lignes.append(ligne)

    verification = {}
    verification["sous_jeu"] = sous_jeu
    verification["moteur"] = int(moteur)
    verification["nb_cycles"] = moteur_test["nb_cycles"]
    verification["rul_finale"] = moteur_test["rul_finale"]
    verification["modeles"] = list(modeles)
    verification["indisponibles"] = dict(inference["indisponibles"])
    verification["trajectoire"] = trajectoire
    verification["dernier_cycle"] = pd.DataFrame(lignes, columns=["cle", "modele", "rul_predite", "rul_vraie", "ecart",
                                                                  "alerte", "statut_alerte"])
    verification["durees_entrainement_s"] = durees
    verification["seuil_alerte"] = float(seuil_alerte)
    verification["source"] = (f"data/test_{sous_jeu}.txt (mesures brutes), data/RUL_{sous_jeu}.txt (vérité), "
                              "modèles de models/ ; aucun fichier de sorties/predictions/ lu")
    return verification


def comparer_aux_fichiers_demo(verification, dossier_predictions=None, tolerance=1e-4):
    """
    Compare la prediction recalculee par `verifier_moteur` aux fichiers de la demo (sorties/predictions/).

    Appelee APRES `verifier_moteur` et separement d'elle (principe 3 : le recalcul ne lit jamais ces fichiers,
    seule cette comparaison les ouvre). Pour chaque modele et chaque cycle recalcule : RUL recalculee, RUL du
    fichier, ecart absolu ; on verifie aussi que la RUL vraie du fichier est celle de RUL_FD00X.txt.

    Parameters
    ----------
    verification : dict
        Le resultat de `verifier_moteur`.
    dossier_predictions : str ou None, optionnel
        Dossier des fichiers de la demo (None : celui du contrat, sorties/predictions/).
    tolerance : float, optionnel
        Ecart maximal admis, en cycles (1e-4 par defaut, celui de tests/test_anti_triche.py).

    Returns
    -------
    (pandas.DataFrame, pandas.DataFrame)
        - le detail : une ligne par (modele, cycle) : "cle", "modele", "time_cycles", "rul_recalculee",
          "rul_demo", "ecart_absolu", "rul_vraie_recalculee", "rul_vraie_demo" ;
        - le resume : une ligne par modele : "cle", "modele", "nb_cycles_compares", "ecart_max", "identique"
          (ecart_max <= tolerance et memes RUL vraies), "fichier".
    """
    sous_jeu = verification["sous_jeu"]
    moteur = verification["moteur"]
    trajectoire = verification["trajectoire"]
    lignes_detail = []
    lignes_resume = []
    for cle in verification["modeles"]:
        tableau, _ = lire_predictions(sous_jeu, cle, dossier_predictions)  # fichier de la demo
        lignes_moteur = tableau[tableau["unit_number"] == moteur].set_index("time_cycles")
        ecarts = []
        rul_vraies_identiques = True
        for position in range(len(trajectoire)):
            cycle = int(trajectoire["time_cycles"].iloc[position])
            ligne = {}
            ligne["cle"] = cle
            ligne["modele"] = nom_affiche(cle)
            ligne["time_cycles"] = cycle
            ligne["rul_recalculee"] = float(trajectoire[cle].iloc[position])
            ligne["rul_demo"] = float(lignes_moteur.loc[cycle, "rul_predite"])
            ligne["ecart_absolu"] = abs(ligne["rul_recalculee"] - ligne["rul_demo"])
            ligne["rul_vraie_recalculee"] = float(trajectoire["rul_vraie"].iloc[position])
            ligne["rul_vraie_demo"] = float(lignes_moteur.loc[cycle, "rul_vraie"])
            if ligne["rul_vraie_recalculee"] != ligne["rul_vraie_demo"]:
                rul_vraies_identiques = False
            ecarts.append(ligne["ecart_absolu"])
            lignes_detail.append(ligne)
        resume = {}
        resume["cle"] = cle
        resume["modele"] = nom_affiche(cle)
        resume["nb_cycles_compares"] = len(ecarts)
        resume["ecart_max"] = float(np.max(ecarts))
        resume["identique"] = bool(resume["ecart_max"] <= tolerance and rul_vraies_identiques)
        resume["fichier"] = os.path.relpath(chemin_predictions(sous_jeu, cle, dossier_predictions), RACINE_DEPOT)
        lignes_resume.append(resume)
    return pd.DataFrame(lignes_detail), pd.DataFrame(lignes_resume)
