"""
Genere les fichiers de predictions des modeles a une ligne par cycle (arbres et reference naive) sur C-MAPSS.

Pour chaque sous-jeu (FD001 a FD004) et chaque modele de MODELES_ARBRES :

1. preparation des donnees (src/sous_jeux.py : regimes, normalisation, lissage, features) avec les deux jeux
   de features compares (JEUX_FEATURES) :
   - "base" : features de la Phase 3 (capteurs lisses, moyennes et ecarts-types glissants sur 5 cycles,
     time_cycles ; 46 colonnes sur FD001) ;
   - "ecart_initial_20" : jeu du notebook 04e (capteurs lisses, time_cycles, ecart de chaque capteur a la
     moyenne des 20 premiers cycles du moteur ; 31 colonnes sur FD001) ;
2. validation croisee par moteur des deux jeux : GroupKFold, 5 plis melanges (graine 42), calcules une seule
   fois par sous-jeu (`calculer_plis`) et identiques pour les deux jeux et tous les modeles : ce sont aussi
   les plis des reseaux (scripts/entrainer_deep_learning.py) ;
3. choix du jeu de features de chaque modele, sur FD001 seulement, par la REGLE_DE_DECISION ci-dessous
   (fixee avant les resultats, validation croisee uniquement) ; le choix de FD001 est applique tel quel a
   FD002, FD003 et FD004 (limite connue : il n'est pas refait par sous-jeu) ;
4. entrainement final sur TOUS les moteurs d'entrainement, prediction a CHAQUE cycle de CHAQUE moteur test
   (clippee a 0). Le test officiel n'est calcule que pour la version retenue et pour la version de base (le
   modele final deja publie avant cette comparaison) ; un candidat non retenu n'est jamais evalue sur le test ;
5. la version RETENUE est ecrite au format du contrat (src/contrat_predictions.py) dans
   sorties/predictions/<SOUS_JEU>__<modele>.csv.gz + .json (memes cles de modele qu'avant) ;
6. resultats relus par les notebooks dans sorties/resultats/ (non versionne) :
   - 01b_sous_jeux_arbres.csv, 01b_metriques_par_moteur_cv.csv, 01b_predictions_test_dernier_cycle.csv :
     version de BASE seulement, comme avant (notebooks 01b et 03b, chiffres de reference inchanges) ;
   - 05_arbres_choix_features.csv : une ligne par (sous-jeu, modele) : base contre candidat en validation
     croisee, p-valeurs, conditions (a) et (b) de la regle, jeu retenu ;
   - 05_arbres_resume.csv : une ligne par (sous-jeu, modele, jeu de features) : CV, test (vide si non
     evalue), durees ;
   - 05_arbres_par_moteur_cv.csv : erreurs hors-pli de chaque moteur d'entrainement, par jeu de features ;
   - 05_arbres_oof.csv.gz : predictions hors-pli de chaque ligne du train, par jeu de features ;
   - 05_arbres_test_dernier_cycle.csv : prediction au dernier cycle de chaque moteur test (versions evaluees).

Les modeles (MODELES_ARBRES) : les quatre modeles compares a "reglages de base" dans le notebook 05 (section B),
avec exactement ces reglages (src/config_modele.py) : reference naive (moyenne), Random Forest, XGBoost Phase 3
(CONFIG_PHASE3, reglages choisis sur FD001) et LightGBM. Ces reglages de base servent a choisir les features ;
le modele final (fichiers du contrat, demo) utilise les reglages finaux des notebooks 04f et 04g (CONFIGS_FINALES)
quand le jeu ecart_initial_20 est retenu. Aucun n'est reoptimise par sous-jeu : limite connue.

Option --ablations : compare, en validation croisee seulement (le test n'est jamais regarde), plusieurs
variantes de preparation sur FD002 et FD004 (normalisation globale vs par regime, regime ajoute ou non
comme feature one-hot), pour XGBoost Phase 3 seulement. Resultats dans
sorties/resultats/01b_ablations_preparation.csv et sorties/resultats/01b_ablations_par_moteur_cv.csv.

Exemples, depuis la racine du depot :

    python scripts/generer_predictions_arbres.py                          # 4 sous-jeux
    python scripts/generer_predictions_arbres.py --sous-jeux FD001        # FD001 seulement
    python scripts/generer_predictions_arbres.py --sous-jeux FD002 FD004  # relit le choix fait sur FD001
    python scripts/generer_predictions_arbres.py --ablations              # variantes de preparation (CV)
"""

import argparse
import copy
import datetime
import os
import sys
import time

# on ajoute la racine du depot (dossier parent de scripts/) au chemin d'import, pour "import src..."
DOSSIER_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
RACINE_DEPOT = os.path.dirname(DOSSIER_SCRIPTS)
sys.path.insert(0, RACINE_DEPOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from src.config_modele import (  # noqa: E402
    CONFIG_LIGHTGBM_BASE,
    CONFIG_LIGHTGBM_FINAL,
    CONFIG_PHASE3,
    CONFIG_RANDOM_FOREST_BASE,
    CONFIG_RANDOM_FOREST_FINAL,
    CONFIG_REFERENCE_NAIVE,
    CONFIG_XGBOOST_FINAL,
    construire_fabrique_depuis_config,
    decrire_config,
)
from src.contrat_predictions import (  # noqa: E402
    MODELES_CONNUS,
    ecrire_predictions,
    metriques_dernier_cycle,
    resoudre_dossier,
)
from src.evaluation import (  # noqa: E402
    calculer_plis,
    comparer_apparie,
    erreurs_par_moteur,
    evaluer_en_validation_croisee,
    evaluer_sur_test,
    nb_surestimes,
    score_nasa,
)
from src.sous_jeux import SOUS_JEUX, preparer_sous_jeu  # noqa: E402

# ---------------------------------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------------------------------

# les modeles de ce script, dans l'ordre d'execution et d'affichage (cles du contrat de predictions) : ceux de la
# section B du notebook 05, avec leurs reglages de base (cle -> configuration de src/config_modele.py)
CONFIGS_ARBRES = {
    "reference_naive": CONFIG_REFERENCE_NAIVE,
    "random_forest": CONFIG_RANDOM_FOREST_BASE,
    "xgboost_phase3": CONFIG_PHASE3,
    "lightgbm": CONFIG_LIGHTGBM_BASE,
}
MODELES_ARBRES = []
for _cle in CONFIGS_ARBRES:
    MODELES_ARBRES.append(_cle)
# reglages FINAUX (Optuna de Matisse, notebooks 04f et 04g), regles pour le jeu ecart_initial_20 : ils
# remplacent les reglages de base pour le modele final, si ce jeu est retenu. La reference naive n'en a pas.
CONFIGS_FINALES = {
    "random_forest": CONFIG_RANDOM_FOREST_FINAL,
    "xgboost_phase3": CONFIG_XGBOOST_FINAL,
    "lightgbm": CONFIG_LIGHTGBM_FINAL,
}
MODELES_ABLATION = ["xgboost_phase3"]  # --ablations : le modele de la regle de decision (notebook 01b, section D)

# les deux jeux de features compares : cle -> valeur de l'option `ecart_initial` de preparer_sous_jeu
JEU_BASE = "base"                   # features de la Phase 3 (None : option desactivee)
JEU_CANDIDAT = "ecart_initial_20"   # jeu du notebook 04e : ecart a la moyenne des 20 premiers cycles
JEUX_FEATURES = {
    JEU_BASE: None,
    JEU_CANDIDAT: 20,
}

# regle de decision, fixee AVANT de lancer la comparaison (recopiee telle quelle dans le notebook 05)
SOUS_JEU_DE_CHOIX = "FD001"   # le seul sous-jeu ou le choix est fait
SEUIL_BAISSE_RMSE_CV = 0.3    # cycles : condition (a), seuil de Matisse (notebook 04d)
SEUIL_P_VALEUR = 0.05         # condition (b)
REGLE_DE_DECISION = (
    "Pour chaque modele, une amelioration est retenue sur FD001 si, sur les memes 5 plis par moteur, (a) le "
    "RMSE moyen de validation baisse d'au moins 0,3 cycle (seuil de Matisse, notebook 04d) ET (b) le test de "
    "Wilcoxon apparie sur le RMSE par moteur (100 moteurs, prediction hors-pli) donne p < 0,05 en faveur du "
    "candidat. Sinon on garde la version de base. Le score NASA et les surestimations de plus de 20 cycles "
    "sont rapportes a cote ; si le candidat les degrade nettement, le dire. Les choix faits sur FD001 sont "
    "appliques tels quels a FD002-FD004 (le dire comme limite). Mise en oeuvre : p-valeur bilaterale de "
    "scipy.stats.wilcoxon (src/evaluation.py, comparer_apparie), et 'en faveur du candidat' = difference "
    "moyenne (RMSE candidat - RMSE base, moteur par moteur) negative ; la p-valeur unilaterale est rapportee a cote."
)

# dossier des resultats (petits CSV relus par les notebooks 01b, 03b et 05 ; sorties/ est ignore par git)
DOSSIER_RESULTATS = os.path.join(RACINE_DEPOT, "sorties", "resultats")
CHEMIN_RESUME = os.path.join(DOSSIER_RESULTATS, "01b_sous_jeux_arbres.csv")
CHEMIN_PAR_MOTEUR_CV = os.path.join(DOSSIER_RESULTATS, "01b_metriques_par_moteur_cv.csv")
CHEMIN_TEST_DERNIER_CYCLE = os.path.join(DOSSIER_RESULTATS, "01b_predictions_test_dernier_cycle.csv")
CHEMIN_ABLATIONS = os.path.join(DOSSIER_RESULTATS, "01b_ablations_preparation.csv")
CHEMIN_ABLATIONS_PAR_MOTEUR = os.path.join(DOSSIER_RESULTATS, "01b_ablations_par_moteur_cv.csv")
CHEMIN_CHOIX_05 = os.path.join(DOSSIER_RESULTATS, "05_arbres_choix_features.csv")
CHEMIN_RESUME_05 = os.path.join(DOSSIER_RESULTATS, "05_arbres_resume.csv")
CHEMIN_PAR_MOTEUR_05 = os.path.join(DOSSIER_RESULTATS, "05_arbres_par_moteur_cv.csv")
CHEMIN_OOF_05 = os.path.join(DOSSIER_RESULTATS, "05_arbres_oof.csv.gz")
CHEMIN_TEST_DERNIER_05 = os.path.join(DOSSIER_RESULTATS, "05_arbres_test_dernier_cycle.csv")

# reglages communs (memes valeurs que la Phase 3 et le notebook 05)
FENETRE = 5                # cycles : lissage ET features glissantes
PLAFOND_RUL = 125          # plafond de la RUL (Phase 2)
N_PLIS = 5                 # nombre de plis de la validation croisee par moteur
GRAINE_PLIS = 42           # graine du melange des moteurs entre plis
SEUIL_SURESTIMATION = 20   # cycles : surestimation comptee comme dangereuse (choix Phase 3)
SEUIL_ALERTE = 30          # cycles : seuil d'envoi en maintenance (HYPOTHESE DE TRAVAIL, notebook 05)
N_JOBS_DEFAUT = 4          # coeurs utilises par XGBoost Phase 3 (ne change pas les resultats, seulement la vitesse)

# chiffres de test FD001 deja publies : controle de reproduction, cle (modele, jeu de features)
# base : docs/03_modelisation.md et notebook 05 ; ecart_initial_20 : notebook 04e (Matisse, 31 variables)
ATTENDU_FD001 = {
    ("xgboost_phase3", JEU_BASE): {"rmse": 19.70, "score_nasa": 852.9, "nb_surestimes": 9},
    ("xgboost_phase3", JEU_CANDIDAT): {"rmse": 13.10, "score_nasa": 237.0, "nb_surestimes": 2},
}
# meme controle pour les modeles finaux (31 features + reglages finaux) : chiffres du notebook 04g (Matisse),
# sauf XGBoost, recalcule sur 1 coeur (le 04g, sur tous les coeurs d'une machine a 16 coeurs, donne 12,70 / 220,0)
ATTENDU_FD001_FINAL = {
    ("random_forest", JEU_CANDIDAT): {"rmse": 13.14, "score_nasa": 243.3, "nb_surestimes": 3},
    ("xgboost_phase3", JEU_CANDIDAT): {"rmse": 12.73, "score_nasa": 223.5, "nb_surestimes": 2},
    ("lightgbm", JEU_CANDIDAT): {"rmse": 12.76, "score_nasa": 234.6, "nb_surestimes": 3},
}

# description du protocole, recopiee dans chaque fichier de metadonnees
PROTOCOLE_VALIDATION = (
    "Validation croisee par moteur : GroupKFold, 5 plis, shuffle=True, random_state=42 (aucun moteur a la fois "
    "en entrainement et en validation ; chaque moteur en validation exactement une fois) ; plis calcules une "
    "seule fois par sous-jeu (calculer_plis), identiques pour les deux jeux de features ; un modele neuf par pli ; "
    "metriques par pli puis moyenne et ecart-type entre plis (ddof=0) ; predictions de validation non clippees ; "
    "metriques _oof sur toutes les predictions hors-pli. Preparation (regimes K-Means, scalers, capteurs "
    "retires) apprise sur tout le train du sous-jeu avant la validation croisee (meme protocole que la Phase 3 "
    "pour le scaler). Jeu de features choisi sur FD001 en validation croisee par la regle de decision du "
    "script (baisse du RMSE CV >= 0,3 cycle ET Wilcoxon apparie par moteur p < 0,05), applique tel quel a "
    "FD002-FD004. Modele final entraine sur tous les moteurs d'entrainement ; test : prediction a chaque "
    "cycle de chaque moteur test, clippee a 0, metriques au dernier cycle observe contre la RUL brute de "
    "RUL_FD00X.txt (et contre la RUL plafonnee a 125). Hyperparametres : pour choisir les features, ceux de "
    "la Phase 3 pour XGBoost et les reglages de base du notebook 05 pour Random Forest et LightGBM ; pour le "
    "modele final (fichiers du contrat), les reglages finaux d'Optuna (notebooks 04f et 04g, cherches sur FD001 "
    "avec le jeu ecart_initial_20), validation croisee refaite avec eux sur les memes plis ; non reoptimises "
    "par sous-jeu."
)

# variantes de preparation comparees par --ablations (en validation croisee uniquement)
# cle -> (normalisation, regime_en_feature) ; "par_regime" en premier : c'est la reference des tests apparies
VARIANTES_ABLATION = {
    "par_regime": ("par_regime", False),             # normalisation conditionnelle (cahier des charges)
    "globale": ("globale", False),                   # methode FD001 appliquee telle quelle
    "par_regime_onehot": ("par_regime", True),       # + regime courant en one-hot dans les features
}
SOUS_JEUX_ABLATION = ["FD002", "FD004"]  # les deux sous-jeux a plusieurs regimes


# ---------------------------------------------------------------------------------------------------
# Configurations des modeles
# ---------------------------------------------------------------------------------------------------

def configuration_modele(cle_modele, n_jobs=N_JOBS_DEFAUT):
    """
    Renvoie la configuration (format de src/config_modele.py) d'un modele de MODELES_ARBRES.

    Pour XGBoost Phase 3, le nombre de coeurs (n_jobs) est impose : il ne change pas les predictions (verifie
    sur FD001 par le controle de reproduction de ce script), seulement la vitesse. Les autres modeles gardent
    exactement les reglages du notebook 05.

    Parameters
    ----------
    cle_modele : str
        Une cle de MODELES_ARBRES (ex. "xgboost_phase3").
    n_jobs : int, optionnel
        Nombre de coeurs donnes a XGBoost Phase 3 (4 par defaut).

    Returns
    -------
    dict
        Une copie independante de la configuration.

    Raises
    ------
    ValueError
        Si la cle de modele est inconnue.
    """
    if cle_modele not in CONFIGS_ARBRES:
        raise ValueError(f"Modele inconnu pour ce script : {cle_modele!r} (attendus : {MODELES_ARBRES}).")
    config = copy.deepcopy(CONFIGS_ARBRES[cle_modele])  # copie : la configuration d'origine reste intacte
    if cle_modele == "xgboost_phase3":
        config["hyperparametres"]["n_jobs"] = int(n_jobs)  # nombre de coeurs impose (vitesse seulement)
    return config


def a_des_reglages_finaux(cle_modele, jeu_retenu):
    """
    Dit si le modele final utilise les reglages finaux (Optuna, notebooks 04f et 04g) plutot que ceux de base.

    Les reglages finaux ont ete cherches pour le jeu ecart_initial_20 seulement : ils ne s'appliquent que si
    ce jeu est retenu, et seulement aux modeles qui en ont (pas a la reference naive).

    Parameters
    ----------
    cle_modele : str
        Une cle de MODELES_ARBRES.
    jeu_retenu : str
        Le jeu de features retenu par la regle de decision sur FD001.

    Returns
    -------
    bool
        Vrai si le modele final doit utiliser CONFIGS_FINALES[cle_modele].
    """
    return jeu_retenu == JEU_CANDIDAT and cle_modele in CONFIGS_FINALES


def configuration_version_retenue(cle_modele, jeu_retenu, n_jobs=N_JOBS_DEFAUT):
    """
    Configuration du modele FINAL (celui des fichiers du contrat et de la demo) d'un modele de MODELES_ARBRES.

    Deux temps, comme les notebooks 04e puis 04f : les features sont choisies avec les reglages de base
    (`configuration_modele`), puis le modele final utilise les reglages finaux s'il en a pour ce jeu
    (`a_des_reglages_finaux`). src/inference.py appelle cette fonction : la demo entraine donc le meme modele.

    Parameters
    ----------
    cle_modele : str
        Une cle de MODELES_ARBRES (ex. "xgboost_phase3").
    jeu_retenu : str
        Le jeu de features retenu ("base" ou "ecart_initial_20").
    n_jobs : int, optionnel
        Nombre de coeurs donnes a XGBoost a reglages de base (4 par defaut ; vitesse seulement). Le XGBoost
        final tourne toujours sur 1 coeur (voir le commentaire dans le code).

    Returns
    -------
    dict
        Une copie independante de la configuration.
    """
    if not a_des_reglages_finaux(cle_modele, jeu_retenu):
        return configuration_modele(cle_modele, n_jobs=n_jobs)  # pas de reglages finaux : ceux de base
    config = copy.deepcopy(CONFIGS_FINALES[cle_modele])  # copie : la configuration d'origine reste intacte
    if config["famille"] == "xgboost":
        # 1 coeur IMPOSE (n_jobs ignore ici) : avec subsample < 1, XGBoost ne donne pas les memes arbres selon
        # le nombre de coeurs (mesure sur FD001, 16 coeurs : RMSE test 12,70 avec tous, 12,73 avec 1, 12,78
        # avec 4). Sur 1 coeur, le resultat ne depend plus de la machine ; cout : environ 2 s d'entrainement.
        config["hyperparametres"]["n_jobs"] = 1
    return config


# ---------------------------------------------------------------------------------------------------
# Petites aides
# ---------------------------------------------------------------------------------------------------

def metriques_par_moteur(y_vrai, y_predit, groupes, seuil_surestimation=SEUIL_SURESTIMATION):
    """
    Calcule, moteur par moteur, le RMSE, la penalite NASA et le nombre de cycles surestimes.

    Reprend `erreurs_par_moteur` (src/evaluation.py) et y ajoute deux colonnes utiles aux tests apparies
    (meme definition que la fonction `metriques_par_moteur` du notebook 05) : la somme des penalites NASA
    des lignes du moteur, et le nombre de ses lignes surestimees de plus de `seuil_surestimation` cycles.

    Parameters
    ----------
    y_vrai : array-like
        La RUL vraie (plafonnee en validation croisee), une valeur par ligne.
    y_predit : array-like
        Les predictions, une par ligne, dans le meme ordre.
    groupes : array-like
        Le numero de moteur de chaque ligne.
    seuil_surestimation : float, optionnel
        Seuil des surestimations dangereuses (20 cycles par defaut).

    Returns
    -------
    pandas.DataFrame
        Colonnes de `erreurs_par_moteur` (unit_number, rmse, biais, surestimation_max, nb_lignes),
        plus "nasa" et "surestimes", triees par numero de moteur.
    """
    tableau = erreurs_par_moteur(y_vrai, y_predit, groupes)  # rmse, biais, surestimation_max, nb_lignes

    vrais = np.asarray(y_vrai, dtype=float)
    predits = np.asarray(y_predit, dtype=float)
    numeros = np.asarray(groupes)

    penalites_nasa = []  # une valeur par moteur, dans l'ordre de `tableau` (moteurs tries)
    nb_lignes_surestimees = []
    for moteur_id in tableau["unit_number"]:
        masque = numeros == moteur_id  # Vrai sur les lignes de ce moteur
        penalites_nasa.append(score_nasa(vrais[masque], predits[masque]))
        nb_lignes_surestimees.append(nb_surestimes(vrais[masque], predits[masque], seuil=seuil_surestimation))

    tableau["nasa"] = penalites_nasa
    tableau["surestimes"] = nb_lignes_surestimees
    return tableau


def fusionner_resultats(chemin, nouvelles_lignes, colonnes_cle):
    """
    Met a jour un CSV de resultats : remplace les lignes recalculees, garde les autres, puis reecrit.

    Permet de relancer le script sur un seul sous-jeu ou un seul modele sans effacer les resultats des
    autres (ex. --sous-jeux FD001 ne supprime pas les lignes de FD002 a FD004).

    Parameters
    ----------
    chemin : str
        Le fichier CSV a mettre a jour (cree s'il n'existe pas ; compresse si le nom finit par .gz).
    nouvelles_lignes : pandas.DataFrame
        Les lignes recalculees par cette execution.
    colonnes_cle : list de str
        Les colonnes qui identifient un groupe de lignes (ex. ["sous_jeu", "modele"]) : toutes les
        anciennes lignes dont la cle apparait dans `nouvelles_lignes` sont remplacees.

    Returns
    -------
    pandas.DataFrame
        Le tableau complet ecrit sur le disque.
    """
    if os.path.exists(chemin):
        anciennes = pd.read_csv(chemin)

        # cles (tuples) recalculees par cette execution
        cles_recalculees = set()
        for position in range(len(nouvelles_lignes)):
            cle = []
            for colonne in colonnes_cle:
                cle.append(nouvelles_lignes[colonne].iloc[position])
            cles_recalculees.add(tuple(cle))

        # on garde les anciennes lignes dont la cle n'a pas ete recalculee
        a_garder = []
        for position in range(len(anciennes)):
            cle = []
            for colonne in colonnes_cle:
                cle.append(anciennes[colonne].iloc[position])
            a_garder.append(tuple(cle) not in cles_recalculees)
        anciennes = anciennes[np.array(a_garder, dtype=bool)]

        tableau = pd.concat([anciennes, nouvelles_lignes], ignore_index=True)
    else:
        tableau = nouvelles_lignes.copy()

    # tri stable : ordre des sous-jeux, puis ordre des modeles / variantes tel qu'ecrit
    if "sous_jeu" in tableau.columns:
        rang_sous_jeu = []
        for valeur in tableau["sous_jeu"]:
            rang_sous_jeu.append(SOUS_JEUX.index(valeur))
        tableau["_rang"] = rang_sous_jeu
        tableau = tableau.sort_values("_rang", kind="stable").drop(columns=["_rang"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(chemin), exist_ok=True)
    tableau.to_csv(chemin, index=False)
    return tableau


def lire_choix_fd001():
    """
    Relit le jeu de features retenu sur FD001 pour chaque modele (05_arbres_choix_features.csv).

    Sert quand le script est lance sans FD001 (ex. --sous-jeux FD002 FD004) : le choix n'est jamais refait
    sur un autre sous-jeu, il est relu.

    Returns
    -------
    dict
        Cle de modele -> jeu de features retenu ("base" ou "ecart_initial_20").

    Raises
    ------
    FileNotFoundError
        Si le fichier, ou la ligne FD001 d'un modele, manque : il faut d'abord lancer le script sur FD001.
    """
    message = (f"Choix des features sur {SOUS_JEU_DE_CHOIX} introuvable dans {CHEMIN_CHOIX_05} : lancer d'abord "
               f"python scripts/generer_predictions_arbres.py --sous-jeux {SOUS_JEU_DE_CHOIX}")
    if not os.path.exists(CHEMIN_CHOIX_05):
        raise FileNotFoundError(message)
    tableau = pd.read_csv(CHEMIN_CHOIX_05)
    tableau = tableau[tableau["sous_jeu"] == SOUS_JEU_DE_CHOIX]

    choix = {}
    for cle_modele in MODELES_ARBRES:
        lignes = tableau[tableau["modele"] == cle_modele]
        if len(lignes) != 1:
            raise FileNotFoundError(message + f" (modele {cle_modele})")
        choix[cle_modele] = str(lignes["jeu_retenu"].iloc[0])
    return choix


# ---------------------------------------------------------------------------------------------------
# Validation croisee et regle de decision
# ---------------------------------------------------------------------------------------------------

def valider_un_jeu(contexte, config, plis):
    """
    Validation croisee par moteur d'un modele sur un jeu de features (plis communs du sous-jeu).

    Parameters
    ----------
    contexte : dict
        Le contexte renvoye par `preparer_sous_jeu` (X, y, groups).
    config : dict
        La configuration du modele (voir `configuration_modele`).
    plis : list de (numpy.ndarray, numpy.ndarray)
        Les plis de `calculer_plis`, calcules une seule fois pour le sous-jeu.

    Returns
    -------
    (dict, pandas.DataFrame, float)
        Les resultats de `evaluer_en_validation_croisee`, les metriques hors-pli par moteur
        (`metriques_par_moteur`) et la duree de la validation croisee en secondes.
    """
    fabrique = construire_fabrique_depuis_config(config)  # renvoie un modele neuf a chaque appel
    debut = time.perf_counter()
    resultats_cv = evaluer_en_validation_croisee(
        fabrique, contexte["X"], contexte["y"], contexte["groups"], plis,
        seuil_surestimation=SEUIL_SURESTIMATION, seuil_alerte=SEUIL_ALERTE,
    )
    duree_cv = time.perf_counter() - debut
    par_moteur = metriques_par_moteur(contexte["y"], resultats_cv["predictions_oof"], contexte["groups"])
    return resultats_cv, par_moteur, duree_cv


def appliquer_regle_de_decision(resultats_base, resultats_candidat, par_moteur_base, par_moteur_candidat):
    """
    Compare le jeu candidat au jeu de base en validation croisee et applique REGLE_DE_DECISION.

    Condition (a) : le RMSE moyen des 5 plis baisse d'au moins SEUIL_BAISSE_RMSE_CV cycle.
    Condition (b) : test de Wilcoxon apparie sur le RMSE hors-pli de chaque moteur d'entrainement, p-valeur
    bilaterale < SEUIL_P_VALEUR, et difference moyenne (candidat - base) negative, donc en faveur du candidat.
    Si les predictions sont identiques moteur par moteur (reference naive : elle n'utilise pas les features),
    aucun test n'est possible et la base est gardee.

    Parameters
    ----------
    resultats_base, resultats_candidat : dict
        Les resultats de `evaluer_en_validation_croisee` des deux jeux (memes plis).
    par_moteur_base, par_moteur_candidat : pandas.DataFrame
        Les metriques hors-pli par moteur des deux jeux (`metriques_par_moteur`), memes moteurs.

    Returns
    -------
    dict
        Les chiffres de la comparaison (RMSE CV, baisse, p-valeurs, NASA, surestimations), les deux
        conditions et "jeu_selon_regle" ("base" ou "ecart_initial_20").
    """
    # garde-fou : memes moteurs, dans le meme ordre
    assert np.array_equal(par_moteur_base["unit_number"].to_numpy(), par_moteur_candidat["unit_number"].to_numpy())
    rmse_base = par_moteur_base["rmse"].to_numpy(dtype=float)
    rmse_candidat = par_moteur_candidat["rmse"].to_numpy(dtype=float)
    differences = rmse_candidat - rmse_base  # negatif = le candidat se trompe moins sur ce moteur

    comparaison = {}
    comparaison["rmse_cv_base"] = resultats_base["rmse_moyen"]
    comparaison["rmse_cv_ecart_type_base"] = resultats_base["rmse_ecart_type"]
    comparaison["rmse_cv_candidat"] = resultats_candidat["rmse_moyen"]
    comparaison["rmse_cv_ecart_type_candidat"] = resultats_candidat["rmse_ecart_type"]
    comparaison["baisse_rmse_cv"] = resultats_base["rmse_moyen"] - resultats_candidat["rmse_moyen"]
    comparaison["nb_plis_candidat_meilleur"] = int(
        np.sum(np.asarray(resultats_candidat["rmse_par_pli"]) < np.asarray(resultats_base["rmse_par_pli"]))
    )
    comparaison["nb_moteurs"] = int(len(differences))
    comparaison["diff_rmse_moteur_moyenne"] = float(np.mean(differences))
    comparaison["diff_rmse_moteur_mediane"] = float(np.median(differences))
    comparaison["nb_moteurs_candidat_meilleur"] = int(np.sum(differences < 0))

    # tests apparies (impossibles si toutes les differences sont nulles)
    comparaison["p_wilcoxon_bilaterale"] = np.nan
    comparaison["p_wilcoxon_unilaterale"] = np.nan
    comparaison["p_t_bilaterale"] = np.nan
    if np.any(differences != 0):
        apparie = comparer_apparie(rmse_candidat, rmse_base)  # tests bilateraux de src/evaluation.py
        comparaison["p_wilcoxon_bilaterale"] = apparie["p_valeur_wilcoxon"]
        comparaison["p_t_bilaterale"] = apparie["p_valeur_t"]
        # unilateral "le candidat a des RMSE plus petits", rapporte a cote (la regle utilise le bilateral)
        unilateral = stats.wilcoxon(rmse_candidat, rmse_base, alternative="less")
        comparaison["p_wilcoxon_unilaterale"] = float(unilateral.pvalue)

    # chiffres rapportes a cote de la regle (validation croisee, predictions hors-pli)
    nb_lignes = len(resultats_base["predictions_oof"])
    comparaison["nasa_cv_base"] = resultats_base["nasa_moyen"]
    comparaison["nasa_cv_candidat"] = resultats_candidat["nasa_moyen"]
    comparaison["nasa_oof_par_ligne_base"] = resultats_base["nasa_oof"] / nb_lignes
    comparaison["nasa_oof_par_ligne_candidat"] = resultats_candidat["nasa_oof"] / nb_lignes
    comparaison["surestimes_oof_base"] = resultats_base["surestimes_oof"]
    comparaison["surestimes_oof_candidat"] = resultats_candidat["surestimes_oof"]
    comparaison["alertes_manquees_oof_base"] = resultats_base["alertes_manquees_oof"]
    comparaison["alertes_manquees_oof_candidat"] = resultats_candidat["alertes_manquees_oof"]

    # la regle elle-meme (une comparaison avec NaN donne Faux : pas de test = condition non remplie)
    condition_a = bool(comparaison["baisse_rmse_cv"] >= SEUIL_BAISSE_RMSE_CV)
    condition_b = bool(comparaison["p_wilcoxon_bilaterale"] < SEUIL_P_VALEUR
                       and comparaison["diff_rmse_moteur_moyenne"] < 0)
    comparaison["condition_a_baisse_rmse"] = condition_a
    comparaison["condition_b_wilcoxon"] = condition_b
    if condition_a and condition_b:
        comparaison["jeu_selon_regle"] = JEU_CANDIDAT
    else:
        comparaison["jeu_selon_regle"] = JEU_BASE
    return comparaison


# ---------------------------------------------------------------------------------------------------
# Modele final et test officiel (version retenue et version de base seulement)
# ---------------------------------------------------------------------------------------------------

def evaluer_version_finale(contexte, cle_modele, jeu, config, ecrire_contrat, resultats_cv, duree_preparation,
                           duree_cv, attendus=None):
    """
    Entraine le modele final sur tout le train, predit chaque cycle du test et calcule les metriques de test.

    Si `ecrire_contrat` est vrai (version retenue), ecrit aussi le fichier de predictions et ses metadonnees
    au format du contrat (sorties/predictions/).

    Parameters
    ----------
    contexte : dict
        Le contexte renvoye par `preparer_sous_jeu` pour ce jeu de features.
    cle_modele : str
        Une cle de MODELES_ARBRES (cle du contrat de predictions).
    jeu : str
        Le jeu de features ("base" ou "ecart_initial_20").
    config : dict
        La configuration du modele (voir `configuration_modele`).
    ecrire_contrat : bool
        Vrai : ecrit sorties/predictions/<SOUS_JEU>__<modele>.csv.gz + .json.
    resultats_cv : dict
        Les resultats de validation croisee de ce jeu (recopies dans les metadonnees).
    duree_preparation, duree_cv : float
        Durees de la preparation et de la validation croisee, en secondes (metadonnees).
    attendus : dict, optionnel
        Chiffres de test FD001 a retrouver, cle (modele, jeu) ; ATTENDU_FD001 par defaut.

    Returns
    -------
    (dict, pandas.DataFrame, dict)
        Les metriques de test (`metriques_dernier_cycle`), la prediction au dernier cycle de chaque moteur
        test, et les durees ("entrainement_final_s", "prediction_test_s").
    """
    sous_jeu = contexte["sous_jeu"]
    test = contexte["test"]
    feature_cols = contexte["feature_cols"]
    fabrique = construire_fabrique_depuis_config(config)

    # 1. modele final : un modele neuf entraine sur TOUS les moteurs d'entrainement du sous-jeu
    modele_final = fabrique()
    debut = time.perf_counter()
    modele_final.fit(contexte["X"], contexte["y"])
    duree_entrainement = time.perf_counter() - debut

    # 2. prediction a chaque cycle de chaque moteur test (clippee a 0 : une RUL negative n'a pas de sens)
    debut = time.perf_counter()
    predictions_tous_cycles = np.asarray(modele_final.predict(test[feature_cols]), dtype=float)
    duree_prediction = time.perf_counter() - debut
    predictions_tous_cycles = np.clip(predictions_tous_cycles, a_min=0, a_max=None)

    # 3. tableau au format du contrat
    df_predictions = pd.DataFrame()
    df_predictions["unit_number"] = test["unit_number"].to_numpy()
    df_predictions["time_cycles"] = test["time_cycles"].to_numpy()
    df_predictions["rul_predite"] = predictions_tous_cycles
    df_predictions["rul_vraie"] = test["RUL_vraie"].to_numpy(dtype=float)
    df_predictions["rul_vraie_cappee"] = test["RUL_vraie_cappee"].to_numpy(dtype=float)
    df_predictions["regime"] = test["regime"].to_numpy(dtype=int)

    # 4. metriques de test au dernier cycle (fonction du contrat), recoupees avec evaluer_sur_test
    metriques_test = metriques_dernier_cycle(
        df_predictions, plafond=PLAFOND_RUL, seuil_surestimation=SEUIL_SURESTIMATION, seuil_alerte=SEUIL_ALERTE
    )
    controle = evaluer_sur_test(
        modele_final, contexte["X_test_dernier"], contexte["y_test_vraie"], plafond=PLAFOND_RUL,
        seuil_surestimation=SEUIL_SURESTIMATION, seuil_alerte=SEUIL_ALERTE,
    )
    for cle in ["rmse", "rmse_contre_rul_plafonnee", "score_nasa", "nb_surestimes", "alertes_manquees",
                "fausses_alertes"]:
        ecart = abs(float(metriques_test[cle]) - float(controle[cle]))
        assert ecart < 1e-9, f"{sous_jeu}/{cle_modele}/{jeu} : {cle} differe entre le contrat et evaluer_sur_test"
    print(f"  Test {jeu:17s} ({metriques_test['nb_moteurs']} moteurs) : RMSE brute {metriques_test['rmse']:.2f}, "
          f"plafonnee {metriques_test['rmse_contre_rul_plafonnee']:.2f}, NASA {metriques_test['score_nasa']:.1f}, "
          f"surestimes > 20 : {metriques_test['nb_surestimes']}, alertes manquees : "
          f"{metriques_test['alertes_manquees']} | entrainement final {duree_entrainement:.1f} s")

    # 5. controle de reproduction sur FD001 : les chiffres deja publies doivent etre retrouves
    if attendus is None:
        attendus = ATTENDU_FD001  # chiffres des versions a reglages de base
    if sous_jeu == "FD001" and (cle_modele, jeu) in attendus:
        attendu = attendus[(cle_modele, jeu)]
        assert round(metriques_test["rmse"], 2) == attendu["rmse"], (
            f"FD001/{cle_modele}/{jeu} : RMSE test {metriques_test['rmse']:.4f} au lieu de {attendu['rmse']}"
        )
        assert round(metriques_test["score_nasa"], 1) == attendu["score_nasa"], (
            f"FD001/{cle_modele}/{jeu} : score NASA {metriques_test['score_nasa']:.4f} au lieu de {attendu['score_nasa']}"
        )
        assert metriques_test["nb_surestimes"] == attendu["nb_surestimes"], (
            f"FD001/{cle_modele}/{jeu} : {metriques_test['nb_surestimes']} surestimes au lieu de {attendu['nb_surestimes']}"
        )
        print(f"  Controle FD001 ({jeu}) : chiffres publies retrouves ({attendu['rmse']} / {attendu['score_nasa']} / "
              f"{attendu['nb_surestimes']}).")

    # 6. fichier du contrat, pour la version retenue seulement
    if ecrire_contrat:
        metriques_validation = {}
        for cle in ["rmse", "mae", "nasa", "surestimes"]:
            metriques_validation[cle + "_cv_moyen"] = resultats_cv[cle + "_moyen"]
            metriques_validation[cle + "_cv_ecart_type"] = resultats_cv[cle + "_ecart_type"]
        for cle in ["rmse_oof", "nasa_oof", "surestimes_oof", "alertes_manquees_oof", "fausses_alertes_oof",
                    "biais_oof", "rmse_par_pli", "nasa_par_pli"]:
            metriques_validation[cle] = resultats_cv[cle]
        metriques_validation["nb_plis"] = len(resultats_cv["rmse_par_pli"])
        metriques_validation["nb_lignes_validation"] = int(len(contexte["y"]))

        metadonnees = {}
        metadonnees["sous_jeu"] = sous_jeu
        metadonnees["modele"] = cle_modele
        metadonnees["nom_affiche"] = MODELES_CONNUS[cle_modele]
        metadonnees["famille"] = "arbres"
        metadonnees["metriques_test"] = metriques_test
        metadonnees["metriques_validation"] = metriques_validation
        metadonnees["protocole_validation"] = PROTOCOLE_VALIDATION
        metadonnees["regle_de_decision_features"] = REGLE_DE_DECISION
        metadonnees["duree_entrainement_s"] = float(duree_entrainement)
        metadonnees["date"] = datetime.date.today().isoformat()
        metadonnees["source"] = "scripts/generer_predictions_arbres.py"
        metadonnees["config"] = config
        metadonnees["description_modele"] = decrire_config(config) + f" ; features : {jeu}"
        metadonnees["preparation"] = {
            "normalisation": contexte["normalisation"],
            "nb_regimes": contexte["nb_regimes"],
            "regime_en_feature": contexte["regime_en_feature"],
            "capteurs_retires": contexte["capteurs_retires"],
            "nb_capteurs": len(contexte["capteurs"]),
            "nb_features": len(feature_cols),
            "fenetre": contexte["fenetre"],
            "plafond": contexte["plafond"],
            "jeu_features": jeu,
            "ecart_initial": contexte["ecart_initial"],
            "feature_cols": list(feature_cols),
        }
        metadonnees["duree_validation_croisee_s"] = float(duree_cv)
        metadonnees["duree_preparation_s"] = float(duree_preparation)
        metadonnees["duree_prediction_test_s"] = float(duree_prediction)
        metadonnees["nb_moteurs_train"] = int(contexte["groups"].nunique())
        metadonnees["nb_lignes_train"] = int(len(contexte["X"]))
        chemin_csv, chemin_json = ecrire_predictions(df_predictions, metadonnees)  # dossier par defaut du contrat
        print(f"  Ecrit (version retenue : {jeu}) : {chemin_csv}")

    # 7. prediction au dernier cycle de chaque moteur test (pour les tests apparies sur le test)
    dernier = contexte["dernier_cycle_par_moteur"]
    nb_cycles_par_moteur = test.groupby("unit_number").size()  # longueur de la trajectoire observee
    test_dernier = pd.DataFrame()
    test_dernier["sous_jeu"] = [sous_jeu] * len(dernier)
    test_dernier["modele"] = [cle_modele] * len(dernier)
    test_dernier["unit_number"] = dernier["unit_number"].to_numpy()
    test_dernier["rul_vraie"] = contexte["y_test_vraie"]
    # en float64 : XGBoost predit en float32, et un CSV ecrit en float32 arrondit a 7 chiffres significatifs
    # (ecart jusqu'a 4e-6 cycle avec le fichier de predictions complet, ecrit en float64)
    test_dernier["rul_predite"] = np.asarray(controle["predictions"], dtype=float)
    test_dernier["nb_cycles_observes"] = nb_cycles_par_moteur.loc[dernier["unit_number"]].to_numpy()
    test_dernier["regime_dernier_cycle"] = dernier["regime"].to_numpy(dtype=int)

    durees = {"entrainement_final_s": float(duree_entrainement), "prediction_test_s": float(duree_prediction)}
    return metriques_test, test_dernier, durees


def ligne_de_resultats(contexte, cle_modele, jeu, retenu, resultats_cv, metriques_test, durees):
    """
    Construit la ligne de resume d'un (sous-jeu, modele, jeu de features) : CV, test (si evalue), durees.

    Parameters
    ----------
    contexte : dict
        Le contexte de `preparer_sous_jeu` pour ce jeu de features.
    cle_modele : str
        Une cle de MODELES_ARBRES.
    jeu : str
        Le jeu de features.
    retenu : bool
        Vrai si c'est la version retenue (celle des fichiers du contrat).
    resultats_cv : dict
        Les resultats de `evaluer_en_validation_croisee`.
    metriques_test : dict ou None
        Les metriques de test (`metriques_dernier_cycle`), ou None si le test n'a pas ete evalue.
    durees : dict
        Durees en secondes : "preparation_s", "cv_s", et si le test est evalue "entrainement_final_s",
        "prediction_test_s".

    Returns
    -------
    dict
        La ligne (colonnes de 01b_sous_jeux_arbres.csv, plus jeu_features, ecart_initial, retenu, test_evalue).
    """
    nb_lignes_train = int(len(contexte["y"]))
    ligne = {}
    ligne["sous_jeu"] = contexte["sous_jeu"]
    ligne["modele"] = cle_modele
    ligne["nom_affiche"] = MODELES_CONNUS[cle_modele]
    ligne["jeu_features"] = jeu
    ligne["ecart_initial"] = contexte["ecart_initial"]
    ligne["retenu"] = bool(retenu)
    ligne["test_evalue"] = metriques_test is not None
    ligne["nb_moteurs_train"] = int(contexte["groups"].nunique())
    ligne["nb_lignes_train"] = nb_lignes_train
    ligne["nb_moteurs_test"] = int(len(contexte["y_test_vraie"]))
    ligne["nb_regimes"] = contexte["nb_regimes"]
    ligne["normalisation"] = contexte["normalisation"]
    ligne["regime_en_feature"] = contexte["regime_en_feature"]
    ligne["nb_capteurs"] = len(contexte["capteurs"])
    ligne["nb_features"] = len(contexte["feature_cols"])
    ligne["rmse_cv_moyen"] = resultats_cv["rmse_moyen"]
    ligne["rmse_cv_ecart_type"] = resultats_cv["rmse_ecart_type"]
    ligne["mae_cv_moyen"] = resultats_cv["mae_moyen"]
    ligne["mae_cv_ecart_type"] = resultats_cv["mae_ecart_type"]
    ligne["nasa_cv_moyen"] = resultats_cv["nasa_moyen"]
    ligne["nasa_cv_ecart_type"] = resultats_cv["nasa_ecart_type"]
    ligne["nasa_oof_par_ligne"] = resultats_cv["nasa_oof"] / nb_lignes_train  # comparable entre sous-jeux
    ligne["surestimes_oof"] = resultats_cv["surestimes_oof"]
    ligne["part_surestimes_oof"] = resultats_cv["surestimes_oof"] / nb_lignes_train
    ligne["alertes_manquees_oof"] = resultats_cv["alertes_manquees_oof"]
    ligne["fausses_alertes_oof"] = resultats_cv["fausses_alertes_oof"]
    ligne["biais_oof"] = resultats_cv["biais_oof"]

    # test : vide (NaN) si cette version n'a pas ete evaluee (candidat non retenu)
    colonnes_test = ["rmse_test", "rmse_test_rul_plafonnee", "mae_test", "nasa_test", "nasa_test_par_moteur",
                     "surestimes_test", "part_surestimes_test", "alertes_manquees_test", "fausses_alertes_test",
                     "duree_entrainement_final_s", "duree_prediction_test_s"]
    for colonne in colonnes_test:
        ligne[colonne] = np.nan
    if metriques_test is not None:
        nb_moteurs_test = int(metriques_test["nb_moteurs"])
        ligne["rmse_test"] = metriques_test["rmse"]
        ligne["rmse_test_rul_plafonnee"] = metriques_test["rmse_contre_rul_plafonnee"]
        ligne["mae_test"] = metriques_test["mae"]
        ligne["nasa_test"] = metriques_test["score_nasa"]
        ligne["nasa_test_par_moteur"] = metriques_test["score_nasa"] / nb_moteurs_test  # comparable entre sous-jeux
        ligne["surestimes_test"] = metriques_test["nb_surestimes"]
        ligne["part_surestimes_test"] = metriques_test["nb_surestimes"] / nb_moteurs_test
        ligne["alertes_manquees_test"] = metriques_test["alertes_manquees"]
        ligne["fausses_alertes_test"] = metriques_test["fausses_alertes"]
        ligne["duree_entrainement_final_s"] = durees["entrainement_final_s"]
        ligne["duree_prediction_test_s"] = durees["prediction_test_s"]
    ligne["duree_preparation_s"] = durees["preparation_s"]
    ligne["duree_cv_s"] = durees["cv_s"]
    ligne["date"] = datetime.date.today().isoformat()
    return ligne


# ---------------------------------------------------------------------------------------------------
# Boucle principale
# ---------------------------------------------------------------------------------------------------

def generer_predictions(sous_jeux):
    """
    Compare les deux jeux de features, applique la regle sur FD001 et genere les predictions des versions retenues.

    Parameters
    ----------
    sous_jeux : list de str
        Les sous-jeux a traiter (ex. ["FD001", "FD002"]) ; FD001 est toujours traite en premier.

    Returns
    -------
    (pandas.DataFrame, pandas.DataFrame)
        Le tableau des choix (05_arbres_choix_features.csv) et le resume (05_arbres_resume.csv), complets.
    """
    # ordre des sous-jeux impose : FD001 d'abord, car son choix sert aux autres
    sous_jeux_ordonnes = []
    for sous_jeu in SOUS_JEUX:
        if sous_jeu in sous_jeux:
            sous_jeux_ordonnes.append(sous_jeu)

    choix_fd001 = {}  # modele -> jeu retenu sur FD001
    if SOUS_JEU_DE_CHOIX not in sous_jeux_ordonnes:
        choix_fd001 = lire_choix_fd001()  # le choix n'est jamais refait sur un autre sous-jeu

    lignes_choix = []
    lignes_05 = []
    par_moteur_05 = []
    oof_05 = []
    test_dernier_05 = []
    lignes_01b = []
    par_moteur_01b = []
    test_dernier_01b = []

    for sous_jeu in sous_jeux_ordonnes:  # un sous-jeu a la fois
        print(f"\n===================== {sous_jeu} =====================")

        # 1. preparation des deux jeux de features
        contextes = {}
        durees_preparation = {}
        for jeu in JEUX_FEATURES:
            debut = time.perf_counter()
            contextes[jeu] = preparer_sous_jeu(sous_jeu, fenetre=FENETRE, plafond=PLAFOND_RUL,
                                               ecart_initial=JEUX_FEATURES[jeu])
            durees_preparation[jeu] = time.perf_counter() - debut
            contexte = contextes[jeu]
            print(f"Preparation {jeu:17s} : {contexte['X'].shape[0]} lignes train, {contexte['X'].shape[1]} features, "
                  f"{contexte['groups'].nunique()} moteurs train, {len(contexte['moteurs_test'])} moteurs test, "
                  f"normalisation {contexte['normalisation']}, {contexte['nb_regimes']} regime(s) "
                  f"({durees_preparation[jeu]:.1f} s)")

        # garde-fou : memes lignes, memes moteurs, meme cible pour les deux jeux (donc memes plis)
        base = contextes[JEU_BASE]
        for jeu in JEUX_FEATURES:
            assert np.array_equal(contextes[jeu]["groups"].to_numpy(), base["groups"].to_numpy()), "moteurs differents"
            assert np.array_equal(contextes[jeu]["y"].to_numpy(), base["y"].to_numpy()), "cibles differentes"

        # les MEMES plis pour tous les modeles et les deux jeux de ce sous-jeu (calcules une seule fois)
        plis = calculer_plis(base["X"], base["y"], base["groups"], n_splits=N_PLIS, random_state=GRAINE_PLIS)

        for cle_modele in MODELES_ARBRES:  # un modele a la fois
            config = configuration_modele(cle_modele)
            print(f"\n--- {sous_jeu} / {cle_modele} : {decrire_config(config)} ---")

            # 2. validation croisee des deux jeux, sur les memes plis
            resultats_cv = {}
            par_moteur = {}
            durees_cv = {}
            for jeu in JEUX_FEATURES:
                resultats_cv[jeu], par_moteur[jeu], durees_cv[jeu] = valider_un_jeu(contextes[jeu], config, plis)
                print(f"  CV {jeu:17s} : RMSE {resultats_cv[jeu]['rmse_moyen']:.3f} +- "
                      f"{resultats_cv[jeu]['rmse_ecart_type']:.3f}, NASA {resultats_cv[jeu]['nasa_moyen']:.1f}, "
                      f"surestimes OOF > 20 : {resultats_cv[jeu]['surestimes_oof']} ({durees_cv[jeu]:.1f} s)")

            # 3. regle de decision (CV uniquement) ; sur FD001 elle decide, ailleurs elle est seulement rapportee
            comparaison = appliquer_regle_de_decision(resultats_cv[JEU_BASE], resultats_cv[JEU_CANDIDAT],
                                                      par_moteur[JEU_BASE], par_moteur[JEU_CANDIDAT])
            if sous_jeu == SOUS_JEU_DE_CHOIX:
                choix_fd001[cle_modele] = comparaison["jeu_selon_regle"]
                origine = f"regle appliquee sur {SOUS_JEU_DE_CHOIX}"
            else:
                origine = f"choix de {SOUS_JEU_DE_CHOIX} applique tel quel (la regle locale est seulement rapportee)"
            jeu_retenu = choix_fd001[cle_modele]
            print(f"  Regle : baisse RMSE CV {comparaison['baisse_rmse_cv']:.3f} (a : {comparaison['condition_a_baisse_rmse']}), "
                  f"Wilcoxon p = {comparaison['p_wilcoxon_bilaterale']:.3g} (b : {comparaison['condition_b_wilcoxon']}) "
                  f"-> selon la regle : {comparaison['jeu_selon_regle']} ; retenu : {jeu_retenu} ({origine})")

            ligne_choix = {}
            ligne_choix["sous_jeu"] = sous_jeu
            ligne_choix["modele"] = cle_modele
            ligne_choix["nb_features_base"] = len(contextes[JEU_BASE]["feature_cols"])
            ligne_choix["nb_features_candidat"] = len(contextes[JEU_CANDIDAT]["feature_cols"])
            for cle in comparaison:
                ligne_choix[cle] = comparaison[cle]
            ligne_choix["jeu_retenu"] = jeu_retenu
            ligne_choix["origine_du_choix"] = origine
            lignes_choix.append(ligne_choix)

            # avec des reglages finaux, le fichier du contrat vient de l'etape 4 bis, pas de la version a reglages de base
            reglages_finaux = a_des_reglages_finaux(cle_modele, jeu_retenu)

            # 4. modele final et test : version retenue (fichiers du contrat) et version de base seulement
            for jeu in JEUX_FEATURES:
                contexte = contextes[jeu]
                retenu = jeu == jeu_retenu
                durees = {"preparation_s": float(durees_preparation[jeu]), "cv_s": float(durees_cv[jeu])}
                metriques_test = None
                if jeu == JEU_BASE or retenu:
                    ecrire_contrat = retenu and not reglages_finaux
                    metriques_test, test_dernier, durees_finales = evaluer_version_finale(
                        contexte, cle_modele, jeu, config, ecrire_contrat, resultats_cv[jeu], durees_preparation[jeu],
                        durees_cv[jeu],
                    )
                    durees["entrainement_final_s"] = durees_finales["entrainement_final_s"]
                    durees["prediction_test_s"] = durees_finales["prediction_test_s"]
                    test_dernier_jeu = test_dernier.copy()
                    test_dernier_jeu.insert(2, "jeu_features", jeu)
                    test_dernier_05.append(test_dernier_jeu)
                    if jeu == JEU_BASE:
                        test_dernier_01b.append(test_dernier)
                else:
                    print(f"  Test {jeu:17s} : non evalue (candidat non retenu)")

                ligne = ligne_de_resultats(contexte, cle_modele, jeu, retenu, resultats_cv[jeu], metriques_test, durees)
                lignes_05.append(ligne)
                if jeu == JEU_BASE:
                    lignes_01b.append(ligne)

                # metriques hors-pli par moteur et predictions hors-pli de chaque ligne du train
                tableau = par_moteur[jeu].copy()
                tableau.insert(0, "modele", cle_modele)
                tableau.insert(0, "sous_jeu", sous_jeu)
                if jeu == JEU_BASE:
                    par_moteur_01b.append(tableau.copy())
                tableau.insert(2, "jeu_features", jeu)
                par_moteur_05.append(tableau)

                oof = pd.DataFrame()
                oof["sous_jeu"] = [sous_jeu] * len(contexte["y"])
                oof["modele"] = cle_modele
                oof["jeu_features"] = jeu
                oof["unit_number"] = contexte["groups"].to_numpy()
                oof["time_cycles"] = contexte["train"]["time_cycles"].to_numpy()
                oof["rul_cappee"] = contexte["y"].to_numpy(dtype=float)
                oof["rul_predite_oof"] = np.asarray(resultats_cv[jeu]["predictions_oof"], dtype=float)
                oof_05.append(oof)

            # 4 bis. modele FINAL avec les reglages finaux (notebooks 04f et 04g) : c'est lui qu'utilise la demo.
            # Les resultats 05_* ci-dessus gardent la version a reglages de base (comparaison du notebook 05).
            if reglages_finaux:
                config_finale = configuration_version_retenue(cle_modele, jeu_retenu)
                print(f"  Version finale : {decrire_config(config_finale)}")
                # validation croisee refaite avec ces reglages, sur les memes plis : les metadonnees du contrat
                # decrivent ainsi le modele qu'elles accompagnent
                resultats_cv_final, _, duree_cv_final = valider_un_jeu(contextes[jeu_retenu], config_finale, plis)
                print(f"  CV version finale      : RMSE {resultats_cv_final['rmse_moyen']:.3f} +- "
                      f"{resultats_cv_final['rmse_ecart_type']:.3f} ({duree_cv_final:.1f} s)")
                evaluer_version_finale(
                    contextes[jeu_retenu], cle_modele, jeu_retenu, config_finale, True, resultats_cv_final,
                    durees_preparation[jeu_retenu], duree_cv_final, attendus=ATTENDU_FD001_FINAL,
                )

    # 5. mise a jour des resultats (les lignes non recalculees sont conservees)
    cle_05 = ["sous_jeu", "modele", "jeu_features"]
    choix = fusionner_resultats(CHEMIN_CHOIX_05, pd.DataFrame(lignes_choix), ["sous_jeu", "modele"])
    resume = fusionner_resultats(CHEMIN_RESUME_05, pd.DataFrame(lignes_05), cle_05)
    fusionner_resultats(CHEMIN_PAR_MOTEUR_05, pd.concat(par_moteur_05, ignore_index=True), cle_05)
    fusionner_resultats(CHEMIN_OOF_05, pd.concat(oof_05, ignore_index=True), cle_05)
    fusionner_resultats(CHEMIN_TEST_DERNIER_05, pd.concat(test_dernier_05, ignore_index=True), cle_05)

    # fichiers 01b : version de base seulement, memes colonnes qu'avant (notebooks 01b et 03b)
    resume_01b = pd.DataFrame(lignes_01b).drop(columns=["jeu_features", "ecart_initial", "retenu", "test_evalue"])
    fusionner_resultats(CHEMIN_RESUME, resume_01b, ["sous_jeu", "modele"])
    fusionner_resultats(CHEMIN_PAR_MOTEUR_CV, pd.concat(par_moteur_01b, ignore_index=True), ["sous_jeu", "modele"])
    fusionner_resultats(CHEMIN_TEST_DERNIER_CYCLE, pd.concat(test_dernier_01b, ignore_index=True),
                        ["sous_jeu", "modele"])
    print(f"\nResultats ecrits : {CHEMIN_CHOIX_05}, {CHEMIN_RESUME_05}, {CHEMIN_PAR_MOTEUR_05}, {CHEMIN_OOF_05}, "
          f"{CHEMIN_TEST_DERNIER_05} (et fichiers 01b_* de la version de base)")
    return choix, resume


# ---------------------------------------------------------------------------------------------------
# Ablations de preparation (validation croisee uniquement)
# ---------------------------------------------------------------------------------------------------

def lancer_ablations():
    """
    Compare en validation croisee les variantes de preparation sur FD002 et FD004.

    Variantes (VARIANTES_ABLATION) : normalisation globale (methode FD001 telle quelle), normalisation
    par regime, normalisation par regime + regime courant en one-hot. Pour un sous-jeu donne, toutes les
    variantes ont les memes lignes et les memes moteurs : les plis sont calcules une seule fois et sont
    donc identiques pour toutes les variantes et tous les modeles. Le jeu de test n'est jamais utilise.

    Regle de decision pour le one-hot, fixee AVANT de lancer les ablations : on n'ajoute le regime comme
    feature que si, pour XGBoost Phase 3, le RMSE CV moyen baisse ET le test de Wilcoxon apparie sur le
    RMSE hors-pli par moteur (variante one-hot contre variante par_regime) donne p < 0,05. Sinon on garde
    le modele le plus simple (sans one-hot).

    Returns
    -------
    pandas.DataFrame
        Une ligne par (sous-jeu, variante, modele), ecrite dans sorties/resultats/01b_ablations_preparation.csv.
    """
    lignes = []
    tableaux_par_moteur = []

    for sous_jeu in SOUS_JEUX_ABLATION:
        print(f"\n===================== Ablations {sous_jeu} =====================")
        plis = None  # calcules sur la premiere variante, puis reutilises (memes lignes, memes moteurs)
        groupes_reference = None
        rmse_moteur_reference = {}  # modele -> RMSE hors-pli par moteur de la variante "par_regime"

        for nom_variante in VARIANTES_ABLATION:
            normalisation, regime_en_feature = VARIANTES_ABLATION[nom_variante]
            debut = time.perf_counter()
            contexte = preparer_sous_jeu(sous_jeu, fenetre=FENETRE, plafond=PLAFOND_RUL,
                                         normalisation=normalisation, regime_en_feature=regime_en_feature)
            duree_preparation = time.perf_counter() - debut

            if plis is None:
                plis = calculer_plis(contexte["X"], contexte["y"], contexte["groups"], n_splits=N_PLIS,
                                     random_state=GRAINE_PLIS)
                groupes_reference = contexte["groups"].to_numpy()
            # garde-fou : memes moteurs, dans le meme ordre, pour toutes les variantes
            assert np.array_equal(contexte["groups"].to_numpy(), groupes_reference), "lignes differentes entre variantes"

            for cle_modele in MODELES_ABLATION:
                config = configuration_modele(cle_modele)
                fabrique = construire_fabrique_depuis_config(config)
                debut = time.perf_counter()
                resultats_cv = evaluer_en_validation_croisee(
                    fabrique, contexte["X"], contexte["y"], contexte["groups"], plis,
                    seuil_surestimation=SEUIL_SURESTIMATION, seuil_alerte=SEUIL_ALERTE,
                )
                duree_cv = time.perf_counter() - debut
                par_moteur = metriques_par_moteur(contexte["y"], resultats_cv["predictions_oof"], contexte["groups"])

                ligne = {}
                ligne["sous_jeu"] = sous_jeu
                ligne["variante"] = nom_variante
                ligne["modele"] = cle_modele
                ligne["normalisation"] = normalisation
                ligne["regime_en_feature"] = regime_en_feature
                ligne["capteurs_retires"] = " ".join(contexte["capteurs_retires"])
                ligne["nb_features"] = len(contexte["feature_cols"])
                ligne["rmse_cv_moyen"] = resultats_cv["rmse_moyen"]
                ligne["rmse_cv_ecart_type"] = resultats_cv["rmse_ecart_type"]
                ligne["nasa_cv_moyen"] = resultats_cv["nasa_moyen"]
                ligne["surestimes_oof"] = resultats_cv["surestimes_oof"]
                ligne["biais_oof"] = resultats_cv["biais_oof"]
                ligne["duree_preparation_s"] = float(duree_preparation)
                ligne["duree_cv_s"] = float(duree_cv)

                # test apparie contre la variante de reference "par_regime" (meme modele, memes moteurs)
                if nom_variante == "par_regime":
                    rmse_moteur_reference[cle_modele] = par_moteur["rmse"].to_numpy()
                ligne["p_wilcoxon_vs_par_regime"] = np.nan
                ligne["p_t_vs_par_regime"] = np.nan
                ligne["diff_rmse_moteur_vs_par_regime"] = np.nan
                if nom_variante != "par_regime" and cle_modele in rmse_moteur_reference:
                    comparaison = comparer_apparie(par_moteur["rmse"].to_numpy(), rmse_moteur_reference[cle_modele])
                    ligne["p_wilcoxon_vs_par_regime"] = comparaison["p_valeur_wilcoxon"]
                    ligne["p_t_vs_par_regime"] = comparaison["p_valeur_t"]
                    ligne["diff_rmse_moteur_vs_par_regime"] = comparaison["difference_moyenne"]
                lignes.append(ligne)

                par_moteur.insert(0, "modele", cle_modele)
                par_moteur.insert(0, "variante", nom_variante)
                par_moteur.insert(0, "sous_jeu", sous_jeu)
                tableaux_par_moteur.append(par_moteur)

                print(f"  {nom_variante:18s} {cle_modele:15s} RMSE CV {resultats_cv['rmse_moyen']:6.2f} "
                      f"+- {resultats_cv['rmse_ecart_type']:.2f} | NASA CV {resultats_cv['nasa_moyen']:10.1f} | "
                      f"surestimes OOF {resultats_cv['surestimes_oof']:5d} | {len(contexte['feature_cols'])} features "
                      f"| {duree_cv:.1f} s")

    tableau = pd.DataFrame(lignes)
    os.makedirs(DOSSIER_RESULTATS, exist_ok=True)
    tableau.to_csv(CHEMIN_ABLATIONS, index=False)
    pd.concat(tableaux_par_moteur, ignore_index=True).to_csv(CHEMIN_ABLATIONS_PAR_MOTEUR, index=False)
    print(f"\nAblations ecrites : {CHEMIN_ABLATIONS}")
    return tableau


# ---------------------------------------------------------------------------------------------------
# Ligne de commande
# ---------------------------------------------------------------------------------------------------

def lire_arguments(argv=None):
    """
    Lit les options de la ligne de commande.

    Parameters
    ----------
    argv : list de str ou None, optionnel
        Les arguments (None : ceux de la ligne de commande).

    Returns
    -------
    argparse.Namespace
        sous_jeux (list de str) et ablations (bool).
    """
    parseur = argparse.ArgumentParser(
        description="Predictions des modeles a arbres et de la reference naive sur les sous-jeux C-MAPSS."
    )
    parseur.add_argument("--sous-jeux", nargs="+", default=list(SOUS_JEUX), choices=list(SOUS_JEUX),
                         help="sous-jeux a traiter (defaut : les quatre)")
    parseur.add_argument("--ablations", action="store_true",
                         help="compare les variantes de preparation sur FD002 et FD004 (CV uniquement)")
    return parseur.parse_args(argv)


def main(argv=None):
    """
    Point d'entree : lit les options puis lance les predictions ou les ablations.

    Parameters
    ----------
    argv : list de str ou None, optionnel
        Les arguments (None : ceux de la ligne de commande).

    Returns
    -------
    None
    """
    arguments = lire_arguments(argv)
    debut = time.perf_counter()

    if arguments.ablations:
        lancer_ablations()
    else:
        print(f"Dossier des predictions : {resoudre_dossier(None)}")
        choix, resume = generer_predictions(arguments.sous_jeux)
        # choix du jeu de features : base contre candidat en validation croisee
        colonnes_choix = ["sous_jeu", "modele", "rmse_cv_base", "rmse_cv_candidat", "baisse_rmse_cv",
                          "p_wilcoxon_bilaterale", "condition_a_baisse_rmse", "condition_b_wilcoxon", "jeu_retenu"]
        print(choix[colonnes_choix].round(4).to_string(index=False))
        # resume : CV et test (vide pour un candidat non retenu)
        colonnes = ["sous_jeu", "modele", "jeu_features", "retenu", "rmse_cv_moyen", "rmse_cv_ecart_type",
                    "nasa_cv_moyen", "rmse_test", "rmse_test_rul_plafonnee", "nasa_test", "surestimes_test",
                    "alertes_manquees_test", "nb_moteurs_test"]
        print(resume[colonnes].round(2).to_string(index=False))

    print(f"\nDuree totale : {time.perf_counter() - debut:.1f} s")


if __name__ == "__main__":
    main()
