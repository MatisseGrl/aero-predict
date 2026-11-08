"""
Validation croisee des modeles sans reseau AVEC LEURS REGLAGES FINAUX, sur les 4 sous-jeux (reorganisation des
notebooks, journal du plan du 2026-11-07).

Pourquoi ce script ? Les resultats de validation croisee deja ecrits dans sorties/resultats/05_arbres_*.csv sont
ceux des reglages DE BASE (ils servent a choisir les features, notebook 05). Les modeles finaux de la demo
utilisent les reglages finaux (CONFIG_*_FINAL, notebooks 04f et 04g) : leur validation croisee n'etait gardee
qu'en resume dans les metadonnees des fichiers de predictions (sorties/predictions/<SOUS_JEU>__<modele>.json),
sans le detail par moteur ni les predictions hors-pli. Ce script la recalcule avec le code existant, sans rien
inventer :

- preparation : `preparer_sous_jeu` (src/sous_jeux.py), memes options que scripts/generer_predictions_arbres.py ;
- plis : `calculer_plis` (GroupKFold, 5 plis melanges, graine 42), calcules une seule fois par sous-jeu sur le
  jeu de base, exactement comme le script des arbres (donc les memes plis que les reseaux) ;
- modele : `configuration_version_retenue` du script des arbres (reglages finaux si le jeu ecart_initial_20 est
  retenu, reglages de base sinon : cas de la reference naive, qui n'a pas de reglages finaux) ;
- validation : `valider_un_jeu` du script des arbres (`evaluer_en_validation_croisee` + metriques par moteur).

Controle : le RMSE moyen de validation doit etre celui deja ecrit dans les metadonnees du fichier de
predictions du meme modele (etape 4 bis du script des arbres) ; l'ecart est ecrit dans le resume.

Fichiers ecrits, dans un dossier A PART qui ne touche pas la demo (sorties/resultats/refonte/, non versionne) :

- cv_arbres_finaux_resume.csv : une ligne par (sous-jeu, modele) ;
- cv_arbres_finaux_par_pli.csv : une ligne par (sous-jeu, modele, pli) ;
- cv_arbres_finaux_par_moteur.csv : une ligne par (sous-jeu, modele, moteur d'entrainement), erreurs hors-pli ;
- cv_arbres_finaux_oof.csv.gz : une ligne par (sous-jeu, modele, ligne du train), prediction hors-pli ;
- cv_arbres_finaux_plis.csv : le pli de validation de chaque moteur d'entrainement, par sous-jeu.

Complements (option --complements, ajoutee apres la revue de la reorganisation, regle 11 et protocole
§1.2, §3, §5.1) ; ils relisent les plis ci-dessus et n'ecrasent aucun des fichiers precedents :

- cv_arbres_finaux_train_par_pli.csv : pour chaque (sous-jeu, modele, pli), l'erreur sur les moteurs
  d'ENTRAINEMENT du pli et sur ses moteurs de validation (diagnostic de sur-apprentissage). Controle : le RMSE de
  validation recalcule ici doit etre celui de cv_arbres_finaux_par_pli.csv ;
- cv_arbres_finaux_decoupages_neufs.csv : les sous-jeux qui ont des reseaux (les quatre depuis le 2026-11-08 ;
  FD001 et FD002 seulement avant le calcul de nuit FD003 / FD004), la meme
  validation croisee sur les 3 decoupages neufs du notebook 04h (graines 1000, 1001, 1002). Les moyennes des
  arbres et des reseaux viennent alors des memes decoupages par moteur (pas d'un test apparie : les predictions
  hors-pli des reseaux n'ont pas ete enregistrees). Controle : les moteurs de validation de chaque pli sont ceux
  de `src.optimisation_reseaux.plis_sous_jeu` (les plis du 04h).

Lancement, depuis la racine du depot (quelques minutes ; Random Forest limite a 4 coeurs par
LOKY_MAX_CPU_COUNT=4, ce qui ne change pas ses predictions) :

    LOKY_MAX_CPU_COUNT=4 python scripts/valider_arbres_finaux.py                 # validation croisee (graine 42)
    LOKY_MAX_CPU_COUNT=4 python scripts/valider_arbres_finaux.py --complements   # erreur train, decoupages neufs
"""

import argparse
import datetime
import importlib.util
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

from src.config_modele import (  # noqa: E402
    construire_fabrique_depuis_config,
    convertir_en_types_python,
    decrire_config,
)
from src.contrat_predictions import MODELES_CONNUS, lire_metadonnees  # noqa: E402
from src.evaluation import calculer_plis, metriques_regression  # noqa: E402
from src.sous_jeux import SOUS_JEUX, preparer_sous_jeu  # noqa: E402

# le script des arbres, dont on reutilise les fonctions (charge comme un module, sans lancer son main)
CHEMIN_SCRIPT_ARBRES = os.path.join(DOSSIER_SCRIPTS, "generer_predictions_arbres.py")

# dossier a part : rien ici n'est lu par la demo (app/) ni par src/inference.py
DOSSIER_SORTIE = os.path.join(RACINE_DEPOT, "sorties", "resultats", "refonte")
CHEMIN_RESUME = os.path.join(DOSSIER_SORTIE, "cv_arbres_finaux_resume.csv")
CHEMIN_PAR_PLI = os.path.join(DOSSIER_SORTIE, "cv_arbres_finaux_par_pli.csv")
CHEMIN_PAR_MOTEUR = os.path.join(DOSSIER_SORTIE, "cv_arbres_finaux_par_moteur.csv")
CHEMIN_OOF = os.path.join(DOSSIER_SORTIE, "cv_arbres_finaux_oof.csv.gz")
CHEMIN_PLIS = os.path.join(DOSSIER_SORTIE, "cv_arbres_finaux_plis.csv")
CHEMIN_META = os.path.join(DOSSIER_SORTIE, "cv_arbres_finaux_meta.json")
CHEMIN_TRAIN_PAR_PLI = os.path.join(DOSSIER_SORTIE, "cv_arbres_finaux_train_par_pli.csv")
CHEMIN_DECOUPAGES_NEUFS = os.path.join(DOSSIER_SORTIE, "cv_arbres_finaux_decoupages_neufs.csv")

# les 3 decoupages neufs du notebook 04h (graines jamais utilisees pour choisir un reglage)
GRAINES_DECOUPAGES_NEUFS = [1000, 1001, 1002]
# les sous-jeux dont la demo publie des reseaux (validation des reseaux sur ces decoupages neufs) : doit rester
# egal aux cles de ENSEMBLES_RETENUS (src/optimisation_reseaux.py), recopiees ici parce que ce module n'importe
# PyTorch qu'apres xgboost, dans `complements`. Les quatre sous-jeux depuis le calcul de nuit FD003 / FD004 du
# 2026-11-07 au 08 (avant : FD001 et FD002).
SOUS_JEUX_AVEC_RESEAUX = ["FD001", "FD002", "FD003", "FD004"]


def charger_script_arbres():
    """
    Charge scripts/generer_predictions_arbres.py comme un module (imports et constantes seulement).

    Meme methode que src/inference.py (`module_script_arbres`), sans importer PyTorch.

    Returns
    -------
    module
        Le module du script des arbres.
    """
    specification = importlib.util.spec_from_file_location("generer_predictions_arbres", CHEMIN_SCRIPT_ARBRES)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)  # execute les imports et les constantes, pas le main
    return module


def pli_de_chaque_moteur(plis, groupes):
    """
    Donne, pour chaque moteur d'entrainement, le numero du pli ou il est en validation.

    Parameters
    ----------
    plis : list de (numpy.ndarray, numpy.ndarray)
        Les plis de `calculer_plis` (positions des lignes de train et de validation).
    groupes : array-like
        Le numero de moteur de chaque ligne du train.

    Returns
    -------
    (pandas.DataFrame, numpy.ndarray)
        Le tableau (unit_number, pli) trie par moteur, et le pli de validation de chaque ligne du train.
    """
    numeros = np.asarray(groupes)
    pli_par_ligne = np.full(len(numeros), -1, dtype=int)  # -1 : pas encore affecte
    for numero_pli in range(len(plis)):
        _, idx_val = plis[numero_pli]
        pli_par_ligne[idx_val] = numero_pli  # chaque ligne est en validation dans un seul pli
    assert np.all(pli_par_ligne >= 0), "une ligne n'est en validation dans aucun pli"

    # un moteur = un seul pli (garanti par GroupKFold et verifie par calculer_plis)
    tableau = pd.DataFrame()
    tableau["unit_number"] = numeros
    tableau["pli"] = pli_par_ligne
    tableau = tableau.drop_duplicates().sort_values("unit_number").reset_index(drop=True)
    assert tableau["unit_number"].is_unique, "un moteur apparait dans plusieurs plis"
    return tableau, pli_par_ligne


def rmse_cv_du_contrat(sous_jeu, modele):
    """
    Relit le RMSE moyen de validation ecrit dans les metadonnees du fichier de predictions (controle).

    Parameters
    ----------
    sous_jeu : str
        "FD001" a "FD004".
    modele : str
        Une cle de modele du contrat (ex. "xgboost_phase3").

    Returns
    -------
    float
        Le RMSE moyen de validation des metadonnees, ou NaN si le fichier est absent.
    """
    try:
        meta = lire_metadonnees(sous_jeu, modele)
    except (FileNotFoundError, ValueError):
        return float("nan")
    return float(meta["metriques_validation"]["rmse_cv_moyen"])


def erreur_entrainement_par_pli(contexte, config, plis):
    """
    Erreur sur les moteurs d'entrainement ET de validation de chaque pli (diagnostic de sur-apprentissage).

    Meme recette que `evaluer_en_validation_croisee` (src/evaluation.py) : un modele neuf par pli, entraine sur
    les moteurs d'entrainement du pli. On predit en plus ces memes moteurs d'entrainement : l'ecart entre
    l'erreur d'entrainement et l'erreur de validation dit si le modele apprend par coeur (protocole, section 5.1).

    Parameters
    ----------
    contexte : dict
        Le contexte renvoye par `preparer_sous_jeu` (X, y).
    config : dict
        La configuration du modele (celle de `configuration_version_retenue`).
    plis : list de (numpy.ndarray, numpy.ndarray)
        Les plis de `calculer_plis` (les memes que la validation croisee).

    Returns
    -------
    list de dict
        Une ligne par pli : "pli", puis "<metrique>_train" et "<metrique>_val" pour rmse, mae, r2, biais,
        nasa_moyen et surestimes_pct (calculees par `metriques_regression`).
    """
    fabrique = construire_fabrique_depuis_config(config)  # renvoie un modele neuf a chaque appel
    X = contexte["X"]
    y = np.asarray(contexte["y"], dtype=float)
    lignes = []
    for numero_pli in range(len(plis)):
        idx_train, idx_val = plis[numero_pli]

        # un modele neuf, entraine sur les seuls moteurs d'entrainement du pli
        modele = fabrique()
        modele.fit(X.iloc[idx_train], y[idx_train])

        # predictions sur les moteurs appris (train) et sur les moteurs jamais vus (validation), non clippees
        predites_train = np.asarray(modele.predict(X.iloc[idx_train]), dtype=float)
        predites_val = np.asarray(modele.predict(X.iloc[idx_val]), dtype=float)
        metriques_train = metriques_regression(y[idx_train], predites_train)
        metriques_val = metriques_regression(y[idx_val], predites_val)

        # une ligne par pli, les memes metriques des deux cotes
        ligne = {}
        ligne["pli"] = numero_pli
        for cle in ["rmse", "mae", "r2", "biais", "nasa_moyen", "surestimes_pct"]:
            ligne[cle + "_train"] = float(metriques_train[cle])
            ligne[cle + "_val"] = float(metriques_val[cle])
        lignes.append(ligne)
    return lignes


def moteurs_de_validation(plis, groupes):
    """
    Ensemble des moteurs de validation de chaque pli (pour comparer deux decoupages sans dependre de l'ordre des lignes).

    Parameters
    ----------
    plis : list de (numpy.ndarray, numpy.ndarray)
        Les plis (positions des lignes de train et de validation).
    groupes : array-like
        Le numero de moteur de chaque ligne.

    Returns
    -------
    list de frozenset
        Les numeros de moteurs de validation, pli par pli.
    """
    numeros = np.asarray(groupes)
    ensembles = []
    for numero_pli in range(len(plis)):
        _, idx_val = plis[numero_pli]
        ensembles.append(frozenset(np.unique(numeros[idx_val]).tolist()))
    return ensembles


def complements():
    """
    Calcule les deux complements de la validation des arbres finaux : erreur d'entrainement, decoupages neufs.

    Lit les plis deja ecrits par `main` (cv_arbres_finaux_plis.csv) pour verifier qu'on recalcule les memes, et le
    RMSE par pli de cv_arbres_finaux_par_pli.csv pour verifier que la validation se recalcule a l'identique.

    Returns
    -------
    (pandas.DataFrame, pandas.DataFrame)
        L'erreur train / validation par pli, et la validation sur les decoupages neufs (une ligne par sous-jeu,
        modele et graine), aussi ecrites dans CHEMIN_TRAIN_PAR_PLI et CHEMIN_DECOUPAGES_NEUFS.
    """
    # les fichiers de main() doivent exister : on s'appuie sur leurs plis et leur RMSE par pli
    for chemin in [CHEMIN_PAR_PLI, CHEMIN_PLIS]:
        if not os.path.exists(chemin):
            raise FileNotFoundError(f"{chemin} absent : lancer d'abord ce script sans --complements")
    par_pli_ecrit = pd.read_csv(CHEMIN_PAR_PLI)
    plis_ecrits = pd.read_csv(CHEMIN_PLIS)

    script = charger_script_arbres()
    choix_fd001 = script.lire_choix_fd001()  # le meme jeu de features que main()

    # PyTorch est importe ici, APRES xgboost (charge par le script des arbres) : voir src/deep_learning.py
    from src import optimisation_reseaux as opt

    # controle : la liste recopiee plus haut est bien celle des ensembles retenus du notebook 04h
    assert sorted(SOUS_JEUX_AVEC_RESEAUX) == sorted(opt.ENSEMBLES_RETENUS), "SOUS_JEUX_AVEC_RESEAUX a mettre a jour"

    lignes_train = []
    lignes_neufs = []
    for sous_jeu in SOUS_JEUX:
        print(f"\n===================== {sous_jeu} (complements) =====================", flush=True)

        # meme preparation que main() : un contexte par jeu de features
        contextes = {}
        for jeu in script.JEUX_FEATURES:
            contextes[jeu] = preparer_sous_jeu(sous_jeu, fenetre=script.FENETRE, plafond=script.PLAFOND_RUL,
                                               ecart_initial=script.JEUX_FEATURES[jeu])
        base = contextes[script.JEU_BASE]

        # controle : les plis de la graine 42 sont ceux ecrits par main()
        plis = calculer_plis(base["X"], base["y"], base["groups"], n_splits=script.N_PLIS,
                             random_state=script.GRAINE_PLIS)
        plis_moteurs, _ = pli_de_chaque_moteur(plis, base["groups"])
        ecrits = plis_ecrits[plis_ecrits["sous_jeu"] == sous_jeu].sort_values("unit_number")
        assert np.array_equal(ecrits["pli"].to_numpy(), plis_moteurs["pli"].to_numpy()), f"{sous_jeu} : plis differents"

        # les decoupages neufs (sous-jeux avec reseaux), controles contre ceux du notebook 04h
        decoupages = {}
        if sous_jeu in SOUS_JEUX_AVEC_RESEAUX:
            for graine in GRAINES_DECOUPAGES_NEUFS:
                plis_graine = calculer_plis(base["X"], base["y"], base["groups"], n_splits=script.N_PLIS,
                                            random_state=graine)
                plis_04h = opt.plis_sous_jeu(graine, sous_jeu=sous_jeu)
                tableaux_04h = opt.preparer_tableaux(20, sous_jeu=sous_jeu)
                identiques = (moteurs_de_validation(plis_graine, base["groups"])
                              == moteurs_de_validation(plis_04h, tableaux_04h["groups"]))
                assert identiques, f"{sous_jeu} : decoupage {graine} different de celui du 04h"
                decoupages[graine] = plis_graine
            print(f"  decoupages neufs {GRAINES_DECOUPAGES_NEUFS} : memes moteurs de validation que le 04h", flush=True)

        for modele in script.MODELES_ARBRES:
            jeu = choix_fd001[modele]
            contexte = contextes[jeu]
            config = script.configuration_version_retenue(modele, jeu)
            debut = time.perf_counter()

            # (a) erreur d'entrainement et de validation, pli par pli (graine 42)
            lignes = erreur_entrainement_par_pli(contexte, config, plis)
            ecrit = par_pli_ecrit[(par_pli_ecrit["sous_jeu"] == sous_jeu) & (par_pli_ecrit["modele"] == modele)]
            ecrit = ecrit.sort_values("pli")["rmse"].to_numpy()
            for ligne in lignes:
                # controle : la validation recalculee ici est celle de main() (meme modele, memes plis)
                ecart = abs(ligne["rmse_val"] - ecrit[ligne["pli"]])
                assert ecart < 1e-6, f"{sous_jeu}/{modele} pli {ligne['pli']} : RMSE de validation different ({ecart})"
                ligne["sous_jeu"] = sous_jeu
                ligne["modele"] = modele
                ligne["jeu_features"] = jeu
                lignes_train.append(ligne)
            rmse_train = []
            rmse_val = []
            for ligne in lignes:
                rmse_train.append(ligne["rmse_train"])
                rmse_val.append(ligne["rmse_val"])
            print(f"  {modele} ({jeu}) : RMSE train {np.mean(rmse_train):.3f}, validation {np.mean(rmse_val):.3f} "
                  f"({time.perf_counter() - debut:.1f} s)", flush=True)

            # (b) la meme validation croisee sur les decoupages neufs (FD001, FD002)
            for graine in decoupages:
                debut = time.perf_counter()
                resultats_cv, _, _ = script.valider_un_jeu(contexte, config, decoupages[graine])
                ligne = {}
                ligne["sous_jeu"] = sous_jeu
                ligne["modele"] = modele
                ligne["jeu_features"] = jeu
                ligne["graine_plis"] = graine
                ligne["rmse_cv_moyen"] = float(resultats_cv["rmse_moyen"])
                ligne["rmse_cv_ecart_type"] = float(resultats_cv["rmse_ecart_type"])
                ligne["rmse_par_pli"] = json.dumps(convertir_en_types_python(list(resultats_cv["rmse_par_pli"])))
                ligne["mae_oof"] = float(resultats_cv["mae_oof"])
                ligne["rmse_oof"] = float(resultats_cv["rmse_oof"])
                ligne["surestimes_oof"] = int(resultats_cv["surestimes_oof"])
                ligne["biais_oof"] = float(resultats_cv["biais_oof"])
                ligne["nb_lignes_train"] = int(len(contexte["y"]))
                lignes_neufs.append(ligne)
                print(f"    decoupage {graine} : RMSE CV {ligne['rmse_cv_moyen']:.3f} "
                      f"({time.perf_counter() - debut:.1f} s)", flush=True)

    # ecriture, dans le dossier a part (les fichiers de main() ne sont pas touches)
    os.makedirs(DOSSIER_SORTIE, exist_ok=True)
    train_par_pli = pd.DataFrame(lignes_train)
    premieres = ["sous_jeu", "modele", "jeu_features", "pli"]
    autres = []
    for colonne in train_par_pli.columns:
        if colonne not in premieres:
            autres.append(colonne)
    train_par_pli = train_par_pli[premieres + autres]
    train_par_pli.to_csv(CHEMIN_TRAIN_PAR_PLI, index=False)
    decoupages_neufs = pd.DataFrame(lignes_neufs)
    decoupages_neufs.to_csv(CHEMIN_DECOUPAGES_NEUFS, index=False)
    print(f"\nEcrit : {CHEMIN_TRAIN_PAR_PLI}\n        {CHEMIN_DECOUPAGES_NEUFS}")
    return train_par_pli, decoupages_neufs


def main():
    """
    Recalcule la validation croisee des 4 modeles sans reseau, reglages finaux, sur les 4 sous-jeux.

    Returns
    -------
    pandas.DataFrame
        Le resume (une ligne par sous-jeu et modele), aussi ecrit dans CHEMIN_RESUME.
    """
    script = charger_script_arbres()
    choix_fd001 = script.lire_choix_fd001()  # jeu de features retenu sur FD001, modele par modele
    print("Jeu de features retenu sur FD001 :", choix_fd001, flush=True)

    lignes_resume = []
    tableaux_par_pli = []
    tableaux_par_moteur = []
    tableaux_oof = []
    tableaux_plis = []
    configs_utilisees = {}

    for sous_jeu in SOUS_JEUX:  # un sous-jeu a la fois
        print(f"\n===================== {sous_jeu} =====================", flush=True)

        # 1. preparation des deux jeux de features (memes options que le script des arbres)
        contextes = {}
        for jeu in script.JEUX_FEATURES:
            contextes[jeu] = preparer_sous_jeu(sous_jeu, fenetre=script.FENETRE, plafond=script.PLAFOND_RUL,
                                               ecart_initial=script.JEUX_FEATURES[jeu])
        base = contextes[script.JEU_BASE]

        # 2. les plis : calcules une seule fois, sur le jeu de base, comme dans le script des arbres
        plis = calculer_plis(base["X"], base["y"], base["groups"], n_splits=script.N_PLIS,
                             random_state=script.GRAINE_PLIS)
        plis_moteurs, pli_par_ligne = pli_de_chaque_moteur(plis, base["groups"])
        plis_moteurs.insert(0, "sous_jeu", sous_jeu)
        tableaux_plis.append(plis_moteurs)

        for modele in script.MODELES_ARBRES:  # un modele a la fois
            jeu = choix_fd001[modele]
            contexte = contextes[jeu]
            # garde-fou : memes lignes, memes moteurs que le jeu de base (donc les plis s'appliquent)
            assert np.array_equal(contexte["groups"].to_numpy(), base["groups"].to_numpy()), "moteurs differents"

            # 3. configuration du modele final (celle des fichiers de predictions et de la demo)
            config = script.configuration_version_retenue(modele, jeu)
            configs_utilisees[sous_jeu + "__" + modele] = convertir_en_types_python(config)
            print(f"--- {sous_jeu} / {modele} ({jeu}) : {decrire_config(config)}", flush=True)

            # 4. validation croisee, avec la fonction du script des arbres
            resultats_cv, par_moteur, duree_cv = script.valider_un_jeu(contexte, config, plis)
            rmse_contrat = rmse_cv_du_contrat(sous_jeu, modele)
            ecart_contrat = abs(resultats_cv["rmse_moyen"] - rmse_contrat)
            print(f"  CV : RMSE {resultats_cv['rmse_moyen']:.4f} +- {resultats_cv['rmse_ecart_type']:.4f} "
                  f"(metadonnees du contrat : {rmse_contrat:.4f}, ecart {ecart_contrat:.2e}) en {duree_cv:.1f} s",
                  flush=True)

            # 5a. resume
            nb_lignes = int(len(contexte["y"]))
            ligne = {}
            ligne["sous_jeu"] = sous_jeu
            ligne["modele"] = modele
            ligne["nom_affiche"] = MODELES_CONNUS[modele]
            ligne["jeu_features"] = jeu
            ligne["nb_features"] = len(contexte["feature_cols"])
            ligne["description_modele"] = decrire_config(config)
            ligne["nb_plis"] = len(plis)
            ligne["graine_plis"] = script.GRAINE_PLIS
            ligne["nb_moteurs_train"] = int(contexte["groups"].nunique())
            ligne["nb_lignes_train"] = nb_lignes
            for cle in ["rmse", "mae", "nasa", "surestimes"]:
                ligne[cle + "_cv_moyen"] = resultats_cv[cle + "_moyen"]
                ligne[cle + "_cv_ecart_type"] = resultats_cv[cle + "_ecart_type"]
            ligne["rmse_oof"] = resultats_cv["rmse_oof"]
            ligne["mae_oof"] = resultats_cv["mae_oof"]
            ligne["nasa_oof"] = resultats_cv["nasa_oof"]
            ligne["nasa_oof_par_ligne"] = resultats_cv["nasa_oof"] / nb_lignes  # comparable entre sous-jeux
            ligne["surestimes_oof"] = resultats_cv["surestimes_oof"]
            ligne["alertes_manquees_oof"] = resultats_cv["alertes_manquees_oof"]
            ligne["fausses_alertes_oof"] = resultats_cv["fausses_alertes_oof"]
            ligne["biais_oof"] = resultats_cv["biais_oof"]
            ligne["rmse_cv_moyen_contrat"] = rmse_contrat
            ligne["ecart_au_contrat"] = ecart_contrat
            ligne["duree_cv_s"] = float(duree_cv)
            ligne["date"] = datetime.date.today().isoformat()
            lignes_resume.append(ligne)

            # 5b. detail par pli
            for numero_pli in range(len(plis)):
                _, idx_val = plis[numero_pli]
                ligne_pli = {}
                ligne_pli["sous_jeu"] = sous_jeu
                ligne_pli["modele"] = modele
                ligne_pli["jeu_features"] = jeu
                ligne_pli["pli"] = numero_pli
                ligne_pli["nb_moteurs_val"] = int(len(np.unique(np.asarray(base["groups"])[idx_val])))
                ligne_pli["nb_lignes_val"] = int(len(idx_val))
                ligne_pli["rmse"] = resultats_cv["rmse_par_pli"][numero_pli]
                ligne_pli["mae"] = resultats_cv["mae_par_pli"][numero_pli]
                ligne_pli["nasa"] = resultats_cv["nasa_par_pli"][numero_pli]
                ligne_pli["surestimes"] = resultats_cv["surestimes_par_pli"][numero_pli]
                tableaux_par_pli.append(ligne_pli)

            # 5c. erreurs hors-pli par moteur (pour les tests apparies en validation)
            tableau = par_moteur.copy()
            tableau = tableau.merge(plis_moteurs[["unit_number", "pli"]], on="unit_number", how="left")
            tableau.insert(0, "jeu_features", jeu)
            tableau.insert(0, "modele", modele)
            tableau.insert(0, "sous_jeu", sous_jeu)
            tableaux_par_moteur.append(tableau)

            # 5d. prediction hors-pli de chaque ligne du train
            oof = pd.DataFrame()
            oof["sous_jeu"] = [sous_jeu] * nb_lignes
            oof["modele"] = modele
            oof["jeu_features"] = jeu
            oof["unit_number"] = contexte["groups"].to_numpy()
            oof["time_cycles"] = contexte["train"]["time_cycles"].to_numpy()
            oof["pli"] = pli_par_ligne
            oof["rul_cappee"] = contexte["y"].to_numpy(dtype=float)
            oof["rul_predite_oof"] = np.asarray(resultats_cv["predictions_oof"], dtype=float)
            tableaux_oof.append(oof)

    # 6. ecriture, dans le dossier a part
    os.makedirs(DOSSIER_SORTIE, exist_ok=True)
    resume = pd.DataFrame(lignes_resume)
    resume.to_csv(CHEMIN_RESUME, index=False)
    pd.DataFrame(tableaux_par_pli).to_csv(CHEMIN_PAR_PLI, index=False)
    pd.concat(tableaux_par_moteur, ignore_index=True).to_csv(CHEMIN_PAR_MOTEUR, index=False)
    pd.concat(tableaux_oof, ignore_index=True).to_csv(CHEMIN_OOF, index=False, compression="gzip")
    pd.concat(tableaux_plis, ignore_index=True).to_csv(CHEMIN_PLIS, index=False)

    # metadonnees : protocole et reglages utilises
    meta = {}
    meta["date"] = datetime.date.today().isoformat()
    meta["source"] = "scripts/valider_arbres_finaux.py"
    meta["protocole_validation"] = script.PROTOCOLE_VALIDATION
    meta["choix_features_fd001"] = choix_fd001
    meta["configurations"] = configs_utilisees
    with open(CHEMIN_META, "w", encoding="utf-8") as fichier:
        json.dump(meta, fichier, ensure_ascii=False, indent=2)

    colonnes = ["sous_jeu", "modele", "jeu_features", "rmse_cv_moyen", "rmse_cv_ecart_type", "nasa_cv_moyen",
                "rmse_oof", "rmse_cv_moyen_contrat", "ecart_au_contrat"]
    print("\n" + resume[colonnes].round(4).to_string(index=False))
    print(f"\nEcrit dans {DOSSIER_SORTIE}")
    return resume


if __name__ == "__main__":
    # option --complements : erreur d'entrainement et decoupages neufs (apres un premier lancement sans option)
    parseur = argparse.ArgumentParser(description="Validation croisee des arbres finaux (reorganisation des notebooks).")
    parseur.add_argument("--complements", action="store_true",
                         help="erreur sur les moteurs d'entrainement par pli et decoupages neufs du 04h (FD001, FD002)")
    arguments = parseur.parse_args()
    debut = time.perf_counter()
    if arguments.complements:
        complements()
    else:
        main()
    print(f"Duree totale : {time.perf_counter() - debut:.1f} s")
