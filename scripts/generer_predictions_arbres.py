"""
Genere les fichiers de predictions du modele a arbres (XGBoost Phase 3) sur les sous-jeux C-MAPSS.

Pour chaque sous-jeu (FD001 a FD004) et chaque modele demande :

1. preparation des donnees (src/sous_jeux.py : regimes, normalisation, lissage, features glissantes) ;
2. validation croisee par moteur : GroupKFold, 5 plis melanges (graine 42), calcules une seule fois par
   sous-jeu (`calculer_plis`) : ce sont aussi les plis des reseaux (scripts/entrainer_deep_learning.py) ;
3. entrainement final sur TOUS les moteurs d'entrainement du sous-jeu ;
4. prediction a CHAQUE cycle de CHAQUE moteur test (clippee a 0), ecrite au format du contrat
   (src/contrat_predictions.py) dans sorties/predictions/<SOUS_JEU>__<modele>.csv.gz + .json ;
5. ecriture (ou mise a jour) des resultats versionnes dans docs/resultats/ :
   - 01b_sous_jeux_arbres.csv : une ligne par (sous-jeu, modele) : CV moyenne +- ecart-type, test, durees ;
   - 01b_metriques_par_moteur_cv.csv : erreurs hors-pli de chaque moteur d'entrainement (tests apparies) ;
   - 01b_predictions_test_dernier_cycle.csv : prediction au dernier cycle de chaque moteur test.

Le modele : xgboost_phase3, c'est-a-dire CONFIG_PHASE3 de src/config_modele.py (XGBoost de la Phase 3,
perte native). Ses reglages ont ete choisis sur FD001 et ne sont PAS reoptimises par sous-jeu : c'est une
limite connue.

Option --ablations : compare, en validation croisee seulement (le test n'est jamais regarde), plusieurs
variantes de preparation sur FD002 et FD004 (normalisation globale vs par regime, regime ajoute ou non
comme feature one-hot). Resultats dans docs/resultats/01b_ablations_preparation.csv et
docs/resultats/01b_ablations_par_moteur_cv.csv.

Exemples, depuis la racine du depot :

    python scripts/generer_predictions_arbres.py                          # 4 sous-jeux
    python scripts/generer_predictions_arbres.py --sous-jeux FD001        # FD001 seulement
    python scripts/generer_predictions_arbres.py --sous-jeux FD002 FD004  # deux sous-jeux
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

from src.config_modele import CONFIG_PHASE3, construire_fabrique_depuis_config, decrire_config  # noqa: E402
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

# les modeles a arbres de ce script, dans l'ordre d'execution et d'affichage (cles du contrat de predictions)
MODELES_ARBRES = ["xgboost_phase3"]

# dossier des resultats versionnes (petits CSV relus par les notebooks 01b et 03b)
DOSSIER_RESULTATS = os.path.join(RACINE_DEPOT, "docs", "resultats")
CHEMIN_RESUME = os.path.join(DOSSIER_RESULTATS, "01b_sous_jeux_arbres.csv")
CHEMIN_PAR_MOTEUR_CV = os.path.join(DOSSIER_RESULTATS, "01b_metriques_par_moteur_cv.csv")
CHEMIN_TEST_DERNIER_CYCLE = os.path.join(DOSSIER_RESULTATS, "01b_predictions_test_dernier_cycle.csv")
CHEMIN_ABLATIONS = os.path.join(DOSSIER_RESULTATS, "01b_ablations_preparation.csv")
CHEMIN_ABLATIONS_PAR_MOTEUR = os.path.join(DOSSIER_RESULTATS, "01b_ablations_par_moteur_cv.csv")

# reglages communs (memes valeurs que la Phase 3 et le notebook 05)
FENETRE = 5                # cycles : lissage ET features glissantes
PLAFOND_RUL = 125          # plafond de la RUL (Phase 2)
N_PLIS = 5                 # nombre de plis de la validation croisee par moteur
GRAINE_PLIS = 42           # graine du melange des moteurs entre plis
SEUIL_SURESTIMATION = 20   # cycles : surestimation comptee comme dangereuse (choix Phase 3)
SEUIL_ALERTE = 30          # cycles : seuil d'envoi en maintenance (HYPOTHESE DE TRAVAIL, notebook 05)
N_JOBS_DEFAUT = 4          # coeurs utilises par XGBoost (ne change pas les resultats, seulement la vitesse)

# chiffres de test FD001 deja publies (docs/03_modelisation.md, notebook 05) : controle de reproduction
ATTENDU_FD001 = {
    "xgboost_phase3": {"rmse": 19.70, "score_nasa": 852.9, "nb_surestimes": 9},
}

# description du protocole, recopiee dans chaque fichier de metadonnees
PROTOCOLE_VALIDATION = (
    "Validation croisee par moteur : GroupKFold, 5 plis, shuffle=True, random_state=42 (aucun moteur a la fois "
    "en entrainement et en validation ; chaque moteur en validation exactement une fois) ; plis calcules une "
    "seule fois par sous-jeu (calculer_plis) ; un modele neuf par pli ; metriques par pli puis "
    "moyenne et ecart-type entre plis (ddof=0) ; predictions de validation non clippees ; metriques _oof sur "
    "toutes les predictions hors-pli. Preparation (regimes K-Means, scalers, capteurs retires) apprise sur tout "
    "le train du sous-jeu avant la validation croisee (meme protocole que la Phase 3 pour le scaler). Modele "
    "final entraine sur tous les moteurs d'entrainement ; test : prediction a chaque cycle de chaque moteur "
    "test, clippee a 0, metriques au dernier cycle observe contre la RUL brute de RUL_FD00X.txt. "
    "Hyperparametres choisis sur FD001 uniquement, non reoptimises par sous-jeu."
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
    Renvoie la configuration (format de src/config_modele.py) d'un modele a arbres de MODELES_ARBRES.

    Le nombre de coeurs (n_jobs) est impose : il ne change pas les predictions (verifie sur FD001 par le
    controle de reproduction de ce script), seulement la vitesse.

    Parameters
    ----------
    cle_modele : str
        "xgboost_phase3".
    n_jobs : int, optionnel
        Nombre de coeurs donnes a XGBoost (4 par defaut).

    Returns
    -------
    dict
        Une copie independante de la configuration, avec hyperparametres["n_jobs"] = n_jobs.

    Raises
    ------
    ValueError
        Si la cle de modele est inconnue.
    """
    if cle_modele == "xgboost_phase3":
        config = copy.deepcopy(CONFIG_PHASE3)  # copie : CONFIG_PHASE3 reste intact
    else:
        raise ValueError(f"Modele inconnu pour ce script : {cle_modele!r} (attendus : {MODELES_ARBRES}).")

    config["hyperparametres"]["n_jobs"] = int(n_jobs)  # nombre de coeurs impose (vitesse seulement)
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
        Le fichier CSV a mettre a jour (cree s'il n'existe pas).
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


# ---------------------------------------------------------------------------------------------------
# Un modele sur un sous-jeu : validation croisee, modele final, predictions test
# ---------------------------------------------------------------------------------------------------

def evaluer_modele_sur_sous_jeu(contexte, cle_modele, config, plis, dossier_predictions, duree_preparation):
    """
    Enchaine validation croisee, entrainement final et predictions test pour un (sous-jeu, modele).

    Ecrit le fichier de predictions et ses metadonnees au format du contrat, et renvoie les lignes de
    resultats a fusionner dans docs/resultats/.

    Parameters
    ----------
    contexte : dict
        Le contexte renvoye par `preparer_sous_jeu`.
    cle_modele : str
        "xgboost_phase3" (cle du contrat de predictions).
    config : dict
        La configuration du modele (voir `configuration_modele`).
    plis : list de (numpy.ndarray, numpy.ndarray)
        Les plis de `calculer_plis`, calcules une seule fois pour le sous-jeu.
    dossier_predictions : str ou None
        Dossier de sortie des predictions (None : dossier par defaut du contrat).
    duree_preparation : float
        Duree de la preparation des donnees du sous-jeu, en secondes (recopiee dans les resultats).

    Returns
    -------
    (dict, pandas.DataFrame, pandas.DataFrame)
        La ligne de resume (dictionnaire), les metriques hors-pli par moteur d'entrainement, et la
        prediction au dernier cycle de chaque moteur test.
    """
    sous_jeu = contexte["sous_jeu"]
    X = contexte["X"]
    y = contexte["y"]
    groups = contexte["groups"]
    feature_cols = contexte["feature_cols"]
    fabrique = construire_fabrique_depuis_config(config)  # renvoie un modele neuf a chaque appel

    print(f"\n--- {sous_jeu} / {cle_modele} : {decrire_config(config)} ---")

    # 1. validation croisee par moteur, sur les plis communs
    debut = time.perf_counter()
    resultats_cv = evaluer_en_validation_croisee(
        fabrique, X, y, groups, plis, seuil_surestimation=SEUIL_SURESTIMATION, seuil_alerte=SEUIL_ALERTE
    )
    duree_cv = time.perf_counter() - debut
    print(f"  CV : RMSE {resultats_cv['rmse_moyen']:.2f} +- {resultats_cv['rmse_ecart_type']:.2f}, "
          f"NASA {resultats_cv['nasa_moyen']:.1f}, surestimes OOF > 20 : {resultats_cv['surestimes_oof']} "
          f"({duree_cv:.1f} s)")

    # 2. modele final : un modele neuf entraine sur TOUS les moteurs d'entrainement du sous-jeu
    modele_final = fabrique()
    debut = time.perf_counter()
    modele_final.fit(X, y)
    duree_entrainement = time.perf_counter() - debut

    # 3. prediction a chaque cycle de chaque moteur test (clippee a 0 : une RUL negative n'a pas de sens)
    test = contexte["test"]
    debut = time.perf_counter()
    predictions_tous_cycles = np.asarray(modele_final.predict(test[feature_cols]), dtype=float)
    duree_prediction = time.perf_counter() - debut
    predictions_tous_cycles = np.clip(predictions_tous_cycles, a_min=0, a_max=None)

    # 4. fichier de predictions au format du contrat
    df_predictions = pd.DataFrame()
    df_predictions["unit_number"] = test["unit_number"].to_numpy()
    df_predictions["time_cycles"] = test["time_cycles"].to_numpy()
    df_predictions["rul_predite"] = predictions_tous_cycles
    df_predictions["rul_vraie"] = test["RUL_vraie"].to_numpy(dtype=float)
    df_predictions["rul_vraie_cappee"] = test["RUL_vraie_cappee"].to_numpy(dtype=float)
    df_predictions["regime"] = test["regime"].to_numpy(dtype=int)

    # 5. metriques de test au dernier cycle (fonction du contrat), recoupees avec evaluer_sur_test
    metriques_test = metriques_dernier_cycle(
        df_predictions, plafond=PLAFOND_RUL, seuil_surestimation=SEUIL_SURESTIMATION, seuil_alerte=SEUIL_ALERTE
    )
    controle = evaluer_sur_test(
        modele_final, contexte["X_test_dernier"], contexte["y_test_vraie"], plafond=PLAFOND_RUL,
        seuil_surestimation=SEUIL_SURESTIMATION, seuil_alerte=SEUIL_ALERTE,
    )
    for cle in ["rmse", "score_nasa", "nb_surestimes", "alertes_manquees", "fausses_alertes"]:
        ecart = abs(float(metriques_test[cle]) - float(controle[cle]))
        assert ecart < 1e-9, f"{sous_jeu}/{cle_modele} : {cle} differe entre le contrat et evaluer_sur_test"
    nb_moteurs_test = int(metriques_test["nb_moteurs"])
    print(f"  Test ({nb_moteurs_test} moteurs) : RMSE {metriques_test['rmse']:.2f}, "
          f"NASA {metriques_test['score_nasa']:.1f}, surestimes > 20 : {metriques_test['nb_surestimes']}"
          f" | entrainement final {duree_entrainement:.1f} s")

    # 6. controle de reproduction sur FD001 : les chiffres deja publies doivent etre retrouves
    if sous_jeu == "FD001" and cle_modele in ATTENDU_FD001:
        attendu = ATTENDU_FD001[cle_modele]
        assert round(metriques_test["rmse"], 2) == attendu["rmse"], (
            f"FD001/{cle_modele} : RMSE test {metriques_test['rmse']:.4f} au lieu de {attendu['rmse']}"
        )
        assert round(metriques_test["score_nasa"], 1) == attendu["score_nasa"], (
            f"FD001/{cle_modele} : score NASA {metriques_test['score_nasa']:.4f} au lieu de {attendu['score_nasa']}"
        )
        assert metriques_test["nb_surestimes"] == attendu["nb_surestimes"], (
            f"FD001/{cle_modele} : {metriques_test['nb_surestimes']} surestimes au lieu de {attendu['nb_surestimes']}"
        )
        print(f"  Controle FD001 : chiffres publies retrouves ({attendu['rmse']} / {attendu['score_nasa']} / "
              f"{attendu['nb_surestimes']}).")

    # 7. metriques de validation (moyennes CV) pour les metadonnees et le resume
    metriques_validation = {}
    metriques_validation["rmse_cv_moyen"] = resultats_cv["rmse_moyen"]
    metriques_validation["rmse_cv_ecart_type"] = resultats_cv["rmse_ecart_type"]
    metriques_validation["mae_cv_moyen"] = resultats_cv["mae_moyen"]
    metriques_validation["mae_cv_ecart_type"] = resultats_cv["mae_ecart_type"]
    metriques_validation["nasa_cv_moyen"] = resultats_cv["nasa_moyen"]
    metriques_validation["nasa_cv_ecart_type"] = resultats_cv["nasa_ecart_type"]
    metriques_validation["surestimes_cv_moyen"] = resultats_cv["surestimes_moyen"]
    metriques_validation["surestimes_cv_ecart_type"] = resultats_cv["surestimes_ecart_type"]
    metriques_validation["rmse_oof"] = resultats_cv["rmse_oof"]
    metriques_validation["nasa_oof"] = resultats_cv["nasa_oof"]
    metriques_validation["surestimes_oof"] = resultats_cv["surestimes_oof"]
    metriques_validation["alertes_manquees_oof"] = resultats_cv["alertes_manquees_oof"]
    metriques_validation["fausses_alertes_oof"] = resultats_cv["fausses_alertes_oof"]
    metriques_validation["biais_oof"] = resultats_cv["biais_oof"]
    metriques_validation["rmse_par_pli"] = resultats_cv["rmse_par_pli"]
    metriques_validation["nasa_par_pli"] = resultats_cv["nasa_par_pli"]
    metriques_validation["nb_plis"] = len(plis)
    metriques_validation["nb_lignes_validation"] = int(len(y))

    # 8. metadonnees du contrat (cles obligatoires + informations de preparation)
    metadonnees = {}
    metadonnees["sous_jeu"] = sous_jeu
    metadonnees["modele"] = cle_modele
    metadonnees["nom_affiche"] = MODELES_CONNUS[cle_modele]
    metadonnees["famille"] = "arbres"
    metadonnees["metriques_test"] = metriques_test
    metadonnees["metriques_validation"] = metriques_validation
    metadonnees["protocole_validation"] = PROTOCOLE_VALIDATION
    metadonnees["duree_entrainement_s"] = float(duree_entrainement)
    metadonnees["date"] = datetime.date.today().isoformat()
    metadonnees["source"] = "scripts/generer_predictions_arbres.py"
    metadonnees["config"] = config
    metadonnees["description_modele"] = decrire_config(config)
    metadonnees["preparation"] = {
        "normalisation": contexte["normalisation"],
        "nb_regimes": contexte["nb_regimes"],
        "regime_en_feature": contexte["regime_en_feature"],
        "capteurs_retires": contexte["capteurs_retires"],
        "nb_capteurs": len(contexte["capteurs"]),
        "nb_features": len(feature_cols),
        "fenetre": contexte["fenetre"],
        "plafond": contexte["plafond"],
    }
    metadonnees["duree_validation_croisee_s"] = float(duree_cv)
    metadonnees["duree_preparation_s"] = float(duree_preparation)
    metadonnees["duree_prediction_test_s"] = float(duree_prediction)
    metadonnees["nb_moteurs_train"] = int(groups.nunique())
    metadonnees["nb_lignes_train"] = int(len(X))

    chemin_csv, chemin_json = ecrire_predictions(df_predictions, metadonnees, dossier_predictions)
    print(f"  Ecrit : {chemin_csv}")

    # 9. ligne de resume (une par sous-jeu et par modele)
    nb_lignes_train = int(len(y))
    ligne = {}
    ligne["sous_jeu"] = sous_jeu
    ligne["modele"] = cle_modele
    ligne["nom_affiche"] = MODELES_CONNUS[cle_modele]
    ligne["nb_moteurs_train"] = int(groups.nunique())
    ligne["nb_lignes_train"] = nb_lignes_train
    ligne["nb_moteurs_test"] = nb_moteurs_test
    ligne["nb_regimes"] = contexte["nb_regimes"]
    ligne["normalisation"] = contexte["normalisation"]
    ligne["regime_en_feature"] = contexte["regime_en_feature"]
    ligne["nb_capteurs"] = len(contexte["capteurs"])
    ligne["nb_features"] = len(feature_cols)
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
    ligne["rmse_test"] = metriques_test["rmse"]
    ligne["rmse_test_rul_plafonnee"] = metriques_test["rmse_contre_rul_plafonnee"]
    ligne["mae_test"] = metriques_test["mae"]
    ligne["nasa_test"] = metriques_test["score_nasa"]
    ligne["nasa_test_par_moteur"] = metriques_test["score_nasa"] / nb_moteurs_test  # comparable entre sous-jeux
    ligne["surestimes_test"] = metriques_test["nb_surestimes"]
    ligne["part_surestimes_test"] = metriques_test["nb_surestimes"] / nb_moteurs_test
    ligne["alertes_manquees_test"] = metriques_test["alertes_manquees"]
    ligne["fausses_alertes_test"] = metriques_test["fausses_alertes"]
    ligne["duree_preparation_s"] = float(duree_preparation)
    ligne["duree_cv_s"] = float(duree_cv)
    ligne["duree_entrainement_final_s"] = float(duree_entrainement)
    ligne["duree_prediction_test_s"] = float(duree_prediction)
    ligne["date"] = metadonnees["date"]

    # 10. metriques hors-pli par moteur d'entrainement (tests apparies du notebook 03b : reseaux contre arbres)
    par_moteur_cv = metriques_par_moteur(y, resultats_cv["predictions_oof"], groups)
    par_moteur_cv.insert(0, "modele", cle_modele)
    par_moteur_cv.insert(0, "sous_jeu", sous_jeu)

    # 11. prediction au dernier cycle de chaque moteur test (pour les tests apparies sur le test)
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

    return ligne, par_moteur_cv, test_dernier


def generer_predictions(sous_jeux, cles_modeles, dossier_predictions=None, n_jobs=N_JOBS_DEFAUT):
    """
    Genere les predictions et les resultats pour plusieurs sous-jeux et modeles.

    Parameters
    ----------
    sous_jeux : list de str
        Les sous-jeux a traiter (ex. ["FD001", "FD002"]).
    cles_modeles : list de str
        Les modeles a evaluer (parmi MODELES_ARBRES).
    dossier_predictions : str ou None, optionnel
        Dossier de sortie des predictions (None : sorties/predictions/ ou la variable d'environnement
        du contrat).
    n_jobs : int, optionnel
        Coeurs donnes a XGBoost.

    Returns
    -------
    pandas.DataFrame
        Le resume complet (docs/resultats/01b_sous_jeux_arbres.csv apres mise a jour).
    """
    lignes_resume = []
    tableaux_par_moteur = []
    tableaux_test_dernier = []

    for sous_jeu in sous_jeux:  # un sous-jeu a la fois
        print(f"\n===================== {sous_jeu} =====================")
        debut = time.perf_counter()
        contexte = preparer_sous_jeu(sous_jeu, fenetre=FENETRE, plafond=PLAFOND_RUL)
        duree_preparation = time.perf_counter() - debut
        print(f"Preparation : {contexte['X'].shape[0]} lignes train, {contexte['X'].shape[1]} features, "
              f"{contexte['groups'].nunique()} moteurs train, {len(contexte['moteurs_test'])} moteurs test, "
              f"normalisation {contexte['normalisation']}, {contexte['nb_regimes']} regime(s), "
              f"capteurs retires {contexte['capteurs_retires']} ({duree_preparation:.1f} s)")

        # les MEMES plis pour tous les modeles de ce sous-jeu (calcules une seule fois)
        plis = calculer_plis(contexte["X"], contexte["y"], contexte["groups"], n_splits=N_PLIS,
                             random_state=GRAINE_PLIS)

        for cle_modele in cles_modeles:  # un modele a la fois
            config = configuration_modele(cle_modele, n_jobs=n_jobs)
            ligne, par_moteur_cv, test_dernier = evaluer_modele_sur_sous_jeu(
                contexte, cle_modele, config, plis, dossier_predictions, duree_preparation
            )
            lignes_resume.append(ligne)
            tableaux_par_moteur.append(par_moteur_cv)
            tableaux_test_dernier.append(test_dernier)

    # mise a jour des resultats versionnes (les lignes non recalculees sont conservees)
    resume = fusionner_resultats(CHEMIN_RESUME, pd.DataFrame(lignes_resume), ["sous_jeu", "modele"])
    fusionner_resultats(CHEMIN_PAR_MOTEUR_CV, pd.concat(tableaux_par_moteur, ignore_index=True),
                        ["sous_jeu", "modele"])
    fusionner_resultats(CHEMIN_TEST_DERNIER_CYCLE, pd.concat(tableaux_test_dernier, ignore_index=True),
                        ["sous_jeu", "modele"])
    print(f"\nResultats ecrits : {CHEMIN_RESUME}")
    return resume


# ---------------------------------------------------------------------------------------------------
# Ablations de preparation (validation croisee uniquement)
# ---------------------------------------------------------------------------------------------------

def lancer_ablations(cles_modeles, n_jobs=N_JOBS_DEFAUT):
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

    Parameters
    ----------
    cles_modeles : list de str
        Les modeles a evaluer (parmi MODELES_ARBRES).
    n_jobs : int, optionnel
        Coeurs donnes a XGBoost.

    Returns
    -------
    pandas.DataFrame
        Une ligne par (sous-jeu, variante, modele), ecrite dans docs/resultats/01b_ablations_preparation.csv.
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

            for cle_modele in cles_modeles:
                config = configuration_modele(cle_modele, n_jobs=n_jobs)
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
        sous_jeux (list de str), modeles (list de str), dossier (str ou None), n_jobs (int), ablations (bool).
    """
    parseur = argparse.ArgumentParser(
        description="Predictions du modele a arbres (XGBoost Phase 3) sur les sous-jeux C-MAPSS."
    )
    parseur.add_argument("--sous-jeux", nargs="+", default=list(SOUS_JEUX), choices=list(SOUS_JEUX),
                         help="sous-jeux a traiter (defaut : les quatre)")
    parseur.add_argument("--modeles", nargs="+", default=list(MODELES_ARBRES), choices=list(MODELES_ARBRES),
                         help="modeles a evaluer (defaut : tous)")
    parseur.add_argument("--dossier", default=None,
                         help="dossier des fichiers de predictions (defaut : sorties/predictions/)")
    parseur.add_argument("--n-jobs", type=int, default=N_JOBS_DEFAUT,
                         help=f"coeurs donnes a XGBoost (defaut : {N_JOBS_DEFAUT})")
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
        lancer_ablations(arguments.modeles, n_jobs=arguments.n_jobs)
    else:
        print(f"Dossier des predictions : {resoudre_dossier(arguments.dossier)}")
        resume = generer_predictions(arguments.sous_jeux, arguments.modeles,
                                     dossier_predictions=arguments.dossier, n_jobs=arguments.n_jobs)
        colonnes = ["sous_jeu", "modele", "rmse_cv_moyen", "rmse_cv_ecart_type", "nasa_cv_moyen", "rmse_test",
                    "nasa_test", "surestimes_test", "nb_moteurs_test"]
        print(resume[colonnes].round(2).to_string(index=False))

    print(f"\nDuree totale : {time.perf_counter() - debut:.1f} s")


if __name__ == "__main__":
    main()
