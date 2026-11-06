"""
Optimisation des reseaux sequentiels sur FD001 (Phase 4) : Optuna, confirmation, ablation, ensemble, test.

Le calcul lourd du notebook 04h. Chaque etape ecrit ses resultats dans sorties/resultats/04h_* (non
versionne) ; le notebook les relit. Le protocole de chaque etape et sa regle de decision sont ecrits
ci-dessous (TEXTE_REGLE_*) AVANT de lancer les calculs, et recopies dans les fichiers de resultats.

Etapes (une sous-commande chacune), depuis la racine du depot :

    python scripts/optimiser_reseaux.py verifier                 # reference 03b retrouvee ? (controle)
    python scripts/optimiser_reseaux.py optuna --reseau lstm --essais 100 --processus 11
    python scripts/optimiser_reseaux.py confirmer --reseau lstm  # meilleur essai vs reference, plis neufs
    python scripts/optimiser_reseaux.py ablation --reseau lstm   # familles d'entrees (methode du 04e)
    python scripts/optimiser_reseaux.py ensemble                 # moyenne de plusieurs graines / reseaux
    python scripts/optimiser_reseaux.py final                    # modele retenu, test officiel lu UNE fois

Calcul : CPU, 1 coeur par apprentissage, plusieurs apprentissages en parallele (un processus chacun).
Optuna : plusieurs processus "travailleurs" partagent la meme etude (base SQLite) ; chaque essai fait sa
validation croisee pli par pli et peut etre abandonne (elague) apres un pli s'il est nettement moins bon
que la mediane des essais precedents au meme pli.
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# 1 fil de calcul par processus pour TOUTES les bibliotheques de calcul (OpenMP, MKL, OpenBLAS), fixe AVANT
# d'importer numpy et torch, et herite par les processus enfants. torch.set_num_threads(1) ne bride que
# PyTorch : constate le 2026-11-06, des travailleurs Optuna utilisaient jusqu'a 1,6 coeur chacun, et les
# 15 travailleurs saturaient les 16 coeurs de la machine (ils se ralentissaient les uns les autres).
for variable in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[variable] = "1"

# racine du depot dans le chemin d'import, pour "import src..."
DOSSIER_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
RACINE_DEPOT = os.path.dirname(DOSSIER_SCRIPTS)
sys.path.insert(0, RACINE_DEPOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# modules du projet AVANT torch (conflit OpenMP, voir src/deep_learning.py)
from src.config_modele import convertir_en_types_python  # noqa: E402
from src.evaluation import comparer_apparie, erreurs_par_moteur, rmse  # noqa: E402
from src import optimisation_reseaux as opt  # noqa: E402

import optuna  # noqa: E402

DOSSIER_RESULTATS = os.path.join(RACINE_DEPOT, "sorties", "resultats")
CHEMIN_BASE_OPTUNA = os.path.join(DOSSIER_RESULTATS, "04h_optuna.db")
if opt.ESSAI_RAPIDE:
    # essai rapide (verification du script) : les etapes LISENT la vraie base Optuna mais ECRIVENT ailleurs,
    # pour ne jamais melanger des chiffres sans valeur avec les vrais resultats
    DOSSIER_RESULTATS = os.path.join(RACINE_DEPOT, "sorties", "essai_rapide")
CHEMIN_OOF_03B = os.path.join(DOSSIER_RESULTATS, "03b_oof_fd001.csv.gz")

GRAINES_PLIS_NEUFS = [1000, 1001, 1002]  # decoupages jamais utilises pour choisir (confirmation)
SEUIL_GAIN_RMSE = 0.3  # cycles, seuil de Matisse (notebook 04d), deja utilise au 03b et au 04f
SEUIL_P = 0.05

# --- Regles de decision, ECRITES AVANT les calculs ---
TEXTE_REGLE_OPTUNA = (
    "Recherche : Optuna, echantillonneur TPE (multivarie, graine 42, 12 essais aleatoires d'abord, "
    "'constant liar' pour les travailleurs paralleles), objectif = RMSE moyen de validation sur les 5 plis "
    "par moteur de FD001 (graine 42, les plis des arbres et du 03b), cible plafonnee a 125, toutes les lignes de "
    "validation. Elagage : MedianPruner (8 essais complets avant d'elaguer, a partir du 2e pli), compare le "
    "RMSE moyen des plis deja faits a la mediane des essais precedents au meme pli. Essai n° 0 = configuration "
    "de reference du 03b quand elle est dans l'espace (controle : on doit retrouver 8,94 pour le LSTM). Le "
    "meilleur essai n'est PAS retenu directement : choisi sur les memes plis que ceux qui le notent, son RMSE "
    "est optimiste ; il passe d'abord l'etape 'confirmer'. Le jeu de test n'intervient pas."
)
TEXTE_REGLE_CONFIRMATION = (
    "Le candidat (meilleur essai Optuna, ou meilleure variante d'une etape) remplace la reference si, sur 3 "
    "decoupages par moteur NEUFS (graines 1000, 1001, 1002, jamais utilises pour choisir), (a) le gain moyen de "
    "RMSE de validation (reference - candidat, moyenne des 3 decoupages) est >= 0,3 cycle, (b) le candidat est "
    "meilleur sur les 3 decoupages, et (c) le test de Wilcoxon apparie sur les 100 RMSE par moteur (hors-pli, "
    "moyennes sur les 3 decoupages) donne p < 0,05 en faveur du candidat. Score NASA et surestimations de plus "
    "de 20 cycles rapportes a cote. Le jeu de test n'intervient pas."
)
TEXTE_REGLE_ENSEMBLE = (
    "Ensemble = moyenne SIMPLE (poids egaux, rien n'est ajuste) des predictions de plusieurs reseaux : 5 "
    "graines de la meilleure configuration LSTM (5 initialisations differentes), et la meme chose en ajoutant "
    "5 graines du meilleur CNN si le CNN est disponible. Evalue sur les memes decoupages neufs que l'etape "
    "'confirmer', avec la meme regle (gain >= 0,3, 3 decoupages sur 3, Wilcoxon p < 0,05) contre la meilleure "
    "configuration seule (graine 42). Le jeu de test n'intervient pas."
)


# ---------------------------------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------------------------------

def afficher(texte):
    """Affiche une ligne horodatee, tout de suite (utile pour suivre un long calcul)."""
    print(f"[{datetime.datetime.now():%H:%M:%S}] {texte}", flush=True)


def ecrire_json(objet, nom_fichier):
    """
    Ecrit un objet en JSON lisible dans sorties/resultats/.

    Parameters
    ----------
    objet : dict
        L'objet a ecrire (converti en types Python simples).
    nom_fichier : str
        Nom du fichier (ex. "04h_confirmation_lstm.json").
    """
    os.makedirs(DOSSIER_RESULTATS, exist_ok=True)
    chemin = os.path.join(DOSSIER_RESULTATS, nom_fichier)
    with open(chemin, "w", encoding="utf-8") as fichier:
        json.dump(convertir_en_types_python(objet), fichier, indent=1, ensure_ascii=False)
    afficher(f"ecrit : {chemin}")


def lire_json(nom_fichier):
    """Relit un fichier JSON de sorties/resultats/ (None s'il n'existe pas)."""
    chemin = os.path.join(DOSSIER_RESULTATS, nom_fichier)
    if not os.path.exists(chemin):
        return None
    with open(chemin, encoding="utf-8") as fichier:
        return json.load(fichier)


def stockage_optuna():
    """
    Renvoie le stockage partage des etudes Optuna (base SQLite dans sorties/resultats/).

    Le delai d'attente de 60 s laisse plusieurs processus ecrire chacun a leur tour sans erreur de verrou.
    """
    url = "sqlite:///" + CHEMIN_BASE_OPTUNA.replace("\\", "/")
    return optuna.storages.RDBStorage(url=url, engine_kwargs={"connect_args": {"timeout": 60}})


def nom_etude(reseau):
    """Nom de l'etude Optuna d'un type de reseau (ex. "04h_lstm")."""
    return "04h_" + reseau


def lancer_cv_paralleles(configs, graines_plis, nb_processus, graine=opt.GRAINE):
    """
    Lance en parallele les validations croisees de plusieurs configurations sur plusieurs decoupages.

    Chaque (configuration, decoupage, pli) est une tache independante executee dans un processus (1 coeur).

    Parameters
    ----------
    configs : dict
        nom -> configuration.
    graines_plis : list de int
        Les graines des decoupages en plis.
    nb_processus : int
        Nombre de processus paralleles.
    graine : int ou dict, optionnel
        Graine des reseaux : un entier pour toutes les configurations, ou nom -> graine.

    Returns
    -------
    dict
        (nom, graine_plis) -> {"oof": predictions hors-pli, "rmse_par_pli": list, "meilleures_epoques": list}.
    """
    taches = []
    for nom in configs:
        graine_config = graine
        if isinstance(graine, dict):
            graine_config = graine[nom]
        for graine_plis in graines_plis:
            for numero_pli in range(opt.N_PLIS):
                tache = {}
                tache["nom"] = nom
                tache["config"] = configs[nom]
                tache["graine_plis"] = graine_plis
                tache["numero_pli"] = numero_pli
                tache["graine"] = graine_config
                taches.append(tache)

    afficher(f"{len(taches)} apprentissages, {nb_processus} processus")
    resultats_taches = []
    with ProcessPoolExecutor(max_workers=nb_processus) as executeur:
        futurs = []
        for tache in taches:
            futurs.append(executeur.submit(opt.executer_tache_pli, tache))
        nb_finis = 0
        for futur in as_completed(futurs):
            resultat = futur.result()
            resultats_taches.append(resultat)
            nb_finis = nb_finis + 1
            afficher(f"  {nb_finis}/{len(taches)} : {resultat['nom']} plis {resultat['graine_plis']} "
                     f"n° {resultat['numero_pli'] + 1}, epoque {resultat['meilleure_epoque']}, "
                     f"{resultat['duree_s']:.0f} s")

    # assemblage : predictions hors-pli de chaque (configuration, decoupage)
    y = opt.preparer_tableaux(20)["y"]
    assembles = {}
    for resultat in resultats_taches:
        cle = (resultat["nom"], resultat["graine_plis"])
        if cle not in assembles:
            assembles[cle] = {"oof": np.full(len(y), np.nan), "rmse_par_pli": [None] * opt.N_PLIS,
                              "meilleures_epoques": [None] * opt.N_PLIS}
        assembles[cle]["oof"][resultat["idx_val"]] = resultat["predictions"]
        assembles[cle]["rmse_par_pli"][resultat["numero_pli"]] = rmse(y[resultat["idx_val"]], resultat["predictions"])
        assembles[cle]["meilleures_epoques"][resultat["numero_pli"]] = resultat["meilleure_epoque"]
    return assembles


def comparer_sur_decoupages(oof_candidat, oof_reference, graines_plis):
    """
    Applique la regle de confirmation : gain moyen, nombre de decoupages gagnes, Wilcoxon par moteur.

    Parameters
    ----------
    oof_candidat, oof_reference : dict
        graine_plis -> {"oof", "rmse_par_pli"} (voir `lancer_cv_paralleles`).
    graines_plis : list de int
        Les decoupages compares.

    Returns
    -------
    dict
        Les chiffres de la comparaison et "retenu" (bool).
    """
    tableaux = opt.preparer_tableaux(20)
    y = tableaux["y"]
    groupes = tableaux["groups"]
    gains = []
    rmse_moteurs_candidat = []
    rmse_moteurs_reference = []
    for graine_plis in graines_plis:
        rmse_candidat = float(np.mean(oof_candidat[graine_plis]["rmse_par_pli"]))
        rmse_reference = float(np.mean(oof_reference[graine_plis]["rmse_par_pli"]))
        gains.append(rmse_reference - rmse_candidat)
        rmse_moteurs_candidat.append(erreurs_par_moteur(y, oof_candidat[graine_plis]["oof"], groupes)["rmse"].to_numpy())
        rmse_moteurs_reference.append(erreurs_par_moteur(y, oof_reference[graine_plis]["oof"], groupes)["rmse"].to_numpy())

    moyenne_candidat = np.mean(np.vstack(rmse_moteurs_candidat), axis=0)  # RMSE par moteur moyen sur les decoupages
    moyenne_reference = np.mean(np.vstack(rmse_moteurs_reference), axis=0)
    test = comparer_apparie(moyenne_candidat, moyenne_reference)

    nb_gagnes = 0
    for gain in gains:
        if gain > 0:
            nb_gagnes = nb_gagnes + 1

    comparaison = {}
    comparaison["gains_par_decoupage"] = gains
    comparaison["gain_moyen"] = float(np.mean(gains))
    comparaison["nb_decoupages_gagnes"] = nb_gagnes
    comparaison["difference_moyenne_rmse_par_moteur"] = test["difference_moyenne"]
    comparaison["p_wilcoxon"] = test["p_valeur_wilcoxon"]
    comparaison["p_t"] = test["p_valeur_t"]
    critere_a = comparaison["gain_moyen"] >= SEUIL_GAIN_RMSE
    critere_b = nb_gagnes == len(graines_plis)
    critere_c = test["p_valeur_wilcoxon"] < SEUIL_P and test["difference_moyenne"] < 0
    comparaison["critere_a"] = bool(critere_a)
    comparaison["critere_b"] = bool(critere_b)
    comparaison["critere_c"] = bool(critere_c)
    comparaison["retenu"] = bool(critere_a and critere_b and critere_c)
    return comparaison


def resume_cv(assembles, nom, graines_plis):
    """
    Resume les validations croisees d'une configuration : RMSE moyen par decoupage, score NASA, surestimes.

    Parameters
    ----------
    assembles : dict
        Le resultat de `lancer_cv_paralleles`.
    nom : str
        La configuration.
    graines_plis : list de int
        Les decoupages.

    Returns
    -------
    dict
        graine_plis -> {"rmse_cv_moyen", "rmse_par_pli", "meilleures_epoques", "metriques_oof"}.
    """
    y = opt.preparer_tableaux(20)["y"]
    resume = {}
    for graine_plis in graines_plis:
        element = assembles[(nom, graine_plis)]
        resume[str(graine_plis)] = {
            "rmse_cv_moyen": float(np.mean(element["rmse_par_pli"])),
            "rmse_par_pli": element["rmse_par_pli"],
            "meilleures_epoques": element["meilleures_epoques"],
            "metriques_oof": opt.calculer_metriques(y, element["oof"]),
        }
    return resume


def meilleure_config_optuna(reseau):
    """
    Reconstruit la configuration du meilleur essai termine d'une etude Optuna.

    Parameters
    ----------
    reseau : str
        Le type de reseau.

    Returns
    -------
    (dict, optuna.trial.FrozenTrial)
        La configuration et l'essai.
    """
    etude = optuna.load_study(study_name=nom_etude(reseau), storage=stockage_optuna())
    meilleur = etude.best_trial
    # on rejoue l'espace de recherche avec les valeurs du meilleur essai (FixedTrial) : meme construction
    config = opt.config_depuis_essai(optuna.trial.FixedTrial(meilleur.params), reseau)
    return config, meilleur


# ---------------------------------------------------------------------------------------------------
# Etapes
# ---------------------------------------------------------------------------------------------------

def etape_verifier(arguments):
    """
    Controle : la reference du 03b (LSTM, plis 42) redonne-t-elle le meme RMSE et les memes predictions ?

    Compare aux predictions hors-pli ecrites par le 03b (colonne oof_lstm de 03b_oof_fd001.csv.gz).
    """
    assembles = lancer_cv_paralleles({"reference": opt.CONFIG_LSTM_REFERENCE}, [opt.GRAINE_PLIS], arguments.processus)
    element = assembles[("reference", opt.GRAINE_PLIS)]
    rmse_moyen = float(np.mean(element["rmse_par_pli"]))
    oof_03b = pd.read_csv(CHEMIN_OOF_03B)["oof_lstm"].to_numpy()
    ecart_max = float(np.max(np.abs(oof_03b - element["oof"])))
    afficher(f"RMSE CV reference : {rmse_moyen:.4f} (03b : 8,9421) ; ecart maximal aux predictions du 03b : {ecart_max:.2e}")
    ecrire_json({"rmse_cv_moyen": rmse_moyen, "rmse_par_pli": element["rmse_par_pli"],
                 "ecart_max_oof_03b": ecart_max, "date": str(datetime.date.today())}, "04h_verification.json")


def objectif(essai, reseau):
    """
    Objectif Optuna : RMSE moyen de validation croisee (plis 42) d'une configuration tiree.

    Apres chaque pli, le RMSE moyen des plis deja faits est signale a Optuna ; s'il est nettement moins bon
    que la mediane des essais precedents au meme pli, l'essai est abandonne (TrialPruned).
    """
    config = opt.config_depuis_essai(essai, reseau)

    def apres_chaque_pli(numero_pli, rmse_moyen_partiel):
        """Signale le resultat intermediaire et abandonne l'essai si Optuna le juge sans espoir."""
        essai.report(rmse_moyen_partiel, numero_pli)
        if essai.should_prune():
            raise optuna.TrialPruned()

    resultat = opt.evaluer_config_cv(config, apres_chaque_pli=apres_chaque_pli)
    essai.set_user_attr("rmse_par_pli", resultat["rmse_par_pli"])
    essai.set_user_attr("meilleures_epoques", resultat["meilleures_epoques"])
    essai.set_user_attr("duree_s", resultat["duree_s"])
    return resultat["rmse_cv_moyen"]


def etape_travailleur(arguments):
    """
    Un processus travailleur Optuna : enchaine des essais jusqu'au nombre total demande pour l'etude.
    """
    import torch
    torch.set_num_threads(1)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    echantillonneur = optuna.samplers.TPESampler(seed=42 + arguments.numero, multivariate=True,
                                                 n_startup_trials=12, constant_liar=True)
    elagueur = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1)
    etude = optuna.load_study(study_name=nom_etude(arguments.reseau), storage=stockage_optuna(),
                              sampler=echantillonneur, pruner=elagueur)
    # on compte seulement les essais termines ou elagues : un essai interrompu (arret du calcul, marque
    # FAIL par l'etape "nettoyer") ne compte pas, la cible reste donc le nombre d'essais UTILES
    etats_comptes = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    fin = optuna.study.MaxTrialsCallback(arguments.essais, states=etats_comptes)

    def journal(etude_en_cours, essai):
        """Affiche une ligne par essai fini."""
        valeur = "elague" if essai.value is None else f"{essai.value:.3f}"
        afficher(f"travailleur {arguments.numero} : essai {essai.number} {essai.state.name} {valeur} {essai.params}")

    etude.optimize(lambda essai: objectif(essai, arguments.reseau), callbacks=[fin, journal],
                   n_trials=arguments.essais, gc_after_trial=True)


def etape_nettoyer(arguments):
    """
    Marque FAIL les essais restes "en cours" (RUNNING) apres un arret du calcul, dans toutes les etudes.

    Quand les processus sont arretes (fin de session, mise en veille), leurs essais en cours restent
    RUNNING dans la base pour toujours ; Optuna ne les reprend pas. Les marquer FAIL les exclut des
    statistiques (l'echantillonneur TPE ignore les essais FAIL) sans toucher aux essais termines.
    A lancer avant de relancer une etude, quand aucun travailleur ne tourne.
    """
    stockage = stockage_optuna()
    for resume in optuna.get_all_study_summaries(storage=stockage):
        etude = optuna.load_study(study_name=resume.study_name, storage=stockage)
        nb_marques = 0
        for essai in etude.trials:
            if essai.state == optuna.trial.TrialState.RUNNING:
                stockage.set_trial_state_values(essai._trial_id, optuna.trial.TrialState.FAIL)
                nb_marques = nb_marques + 1
        afficher(f"{resume.study_name} : {nb_marques} essai(s) interrompu(s) marque(s) FAIL")


def etape_optuna(arguments):
    """
    Cree (ou reprend) l'etude Optuna d'un reseau et lance les processus travailleurs, puis resume l'etude.
    """
    etude = optuna.create_study(study_name=nom_etude(arguments.reseau), storage=stockage_optuna(),
                                direction="minimize", load_if_exists=True)
    if len(etude.trials) == 0:
        parametres_reference = opt.parametres_optuna_reference(arguments.reseau)
        if parametres_reference is not None:
            etude.enqueue_trial(parametres_reference)  # essai n° 0 = reference du 03b (controle)
        etude.set_user_attr("regle", TEXTE_REGLE_OPTUNA)

    processus = []
    for numero in range(arguments.processus):  # un processus Python par travailleur
        commande = [sys.executable, os.path.abspath(__file__), "travailleur", "--reseau", arguments.reseau,
                    "--essais", str(arguments.essais), "--numero", str(numero)]
        journal = open(os.path.join(DOSSIER_RESULTATS, f"04h_journal_{arguments.reseau}_{numero}.log"), "w")
        processus.append(subprocess.Popen(commande, stdout=journal, stderr=subprocess.STDOUT, cwd=RACINE_DEPOT))
        time.sleep(2)  # departs decales : les premiers acces a la base ne se chevauchent pas
    for p in processus:
        p.wait()
    resumer_etude(arguments.reseau)


def resumer_etude(reseau):
    """Ecrit le tableau des essais (CSV) et le meilleur essai (JSON) d'une etude."""
    etude = optuna.load_study(study_name=nom_etude(reseau), storage=stockage_optuna())
    tableau = etude.trials_dataframe()
    tableau.to_csv(os.path.join(DOSSIER_RESULTATS, f"04h_optuna_{reseau}.csv"), index=False)
    config, meilleur = meilleure_config_optuna(reseau)
    nb_complets = 0
    nb_elagues = 0
    for essai in etude.trials:
        if essai.state == optuna.trial.TrialState.COMPLETE:
            nb_complets = nb_complets + 1
        if essai.state == optuna.trial.TrialState.PRUNED:
            nb_elagues = nb_elagues + 1
    ecrire_json({"regle": TEXTE_REGLE_OPTUNA, "nb_essais": len(etude.trials), "nb_complets": nb_complets,
                 "nb_elagues": nb_elagues, "meilleur_numero": meilleur.number, "meilleur_rmse_cv": meilleur.value,
                 "meilleurs_parametres": meilleur.params, "meilleur_detail": meilleur.user_attrs,
                 "configuration": opt.copie_config(config), "date": str(datetime.date.today())},
                f"04h_optuna_{reseau}.json")


def reference_du_reseau(reseau):
    """Configuration de reference (03b) contre laquelle comparer un reseau : la sienne, sinon le LSTM."""
    if reseau == "cnn1d":
        return opt.CONFIG_CNN_REFERENCE
    return opt.CONFIG_LSTM_REFERENCE


def etape_confirmer(arguments):
    """
    Meilleur essai Optuna contre la reference du 03b, sur les plis 42 et 3 decoupages neufs.
    """
    candidat, meilleur = meilleure_config_optuna(arguments.reseau)
    reference = reference_du_reseau(arguments.reseau)
    graines = [opt.GRAINE_PLIS] + GRAINES_PLIS_NEUFS
    assembles = lancer_cv_paralleles({"candidat": candidat, "reference": reference}, graines, arguments.processus)
    oof_candidat = {}
    oof_reference = {}
    for graine_plis in graines:
        oof_candidat[graine_plis] = assembles[("candidat", graine_plis)]
        oof_reference[graine_plis] = assembles[("reference", graine_plis)]
    comparaison = comparer_sur_decoupages(oof_candidat, oof_reference, GRAINES_PLIS_NEUFS)
    afficher(f"{arguments.reseau} : gain moyen {comparaison['gain_moyen']:.3f}, gains {comparaison['gains_par_decoupage']}, "
             f"p Wilcoxon {comparaison['p_wilcoxon']:.2e}, retenu = {comparaison['retenu']}")
    ecrire_json({"regle": TEXTE_REGLE_CONFIRMATION, "reseau": arguments.reseau, "essai_optuna": meilleur.number,
                 "candidat": opt.copie_config(candidat), "reference": opt.copie_config(reference),
                 "cv_candidat": resume_cv(assembles, "candidat", graines),
                 "cv_reference": resume_cv(assembles, "reference", graines),
                 "comparaison": comparaison, "date": str(datetime.date.today())},
                f"04h_confirmation_{arguments.reseau}.json")
    sauver_oof(assembles, f"04h_oof_confirmation_{arguments.reseau}.csv.gz")


def sauver_oof(assembles, nom_fichier):
    """Ecrit les predictions hors-pli de chaque (configuration, decoupage) dans un CSV compresse."""
    tableaux = opt.preparer_tableaux(20)
    tableau = pd.DataFrame({"unit_number": tableaux["groups"], "RUL_cappee": tableaux["y"]})
    for cle in assembles:
        tableau[f"oof_{cle[0]}_{cle[1]}"] = assembles[cle]["oof"]
    tableau.to_csv(os.path.join(DOSSIER_RESULTATS, nom_fichier), index=False)


def config_retenue(reseau):
    """
    Configuration retenue pour un reseau : le candidat confirme s'il l'a ete, sinon la reference du 03b.
    """
    confirmation = lire_json(f"04h_confirmation_{reseau}.json")
    if confirmation is not None and confirmation["comparaison"]["retenu"]:
        return confirmation["candidat"]
    return reference_du_reseau(reseau)


def variantes_ablation(base):
    """
    Variantes d'entrees testees sur la configuration retenue (familles du notebook 04e, une a la fois).

    - "pente_30" et "pente_50" : + pente glissante de chaque capteur (F2, deuxieme meilleure famille au 04e) ;
    - "sans_capteurs" : seulement les ecarts a l'etat initial et time_cycles (version plus simple, regle 9).
    """
    variantes = {}
    for k in [30, 50]:
        config = opt.copie_config(base)
        config["entrees"]["pente"] = k
        variantes[f"pente_{k}"] = config
    config = opt.copie_config(base)
    config["entrees"]["capteurs"] = False
    variantes["sans_capteurs"] = config
    return variantes


def etape_ablation(arguments):
    """
    Ablation des entrees (methode du 04e) sur la configuration retenue, decoupages neufs, meme regle.
    """
    base = config_retenue(arguments.reseau)
    configs = {"base": base}
    variantes = variantes_ablation(base)
    for nom in variantes:
        configs[nom] = variantes[nom]
    assembles = lancer_cv_paralleles(configs, GRAINES_PLIS_NEUFS, arguments.processus)
    resultats = {}
    for nom in variantes:
        oof_variante = {}
        oof_base = {}
        for graine_plis in GRAINES_PLIS_NEUFS:
            oof_variante[graine_plis] = assembles[(nom, graine_plis)]
            oof_base[graine_plis] = assembles[("base", graine_plis)]
        resultats[nom] = {"configuration": variantes[nom], "cv": resume_cv(assembles, nom, GRAINES_PLIS_NEUFS),
                          "comparaison": comparer_sur_decoupages(oof_variante, oof_base, GRAINES_PLIS_NEUFS)}
        afficher(f"ablation {nom} : gain {resultats[nom]['comparaison']['gain_moyen']:.3f}, "
                 f"retenu = {resultats[nom]['comparaison']['retenu']}")
    ecrire_json({"regle": TEXTE_REGLE_CONFIRMATION, "base": base, "cv_base": resume_cv(assembles, "base", GRAINES_PLIS_NEUFS),
                 "variantes": resultats, "date": str(datetime.date.today())}, f"04h_ablation_{arguments.reseau}.json")


def membres_ensemble():
    """
    Membres de l'ensemble : 5 graines de la meilleure configuration LSTM, et 5 du meilleur CNN.

    Returns
    -------
    (dict, dict)
        nom -> configuration, et nom -> graine.
    """
    configs = {}
    graines = {}
    for reseau in ["lstm", "cnn1d"]:
        config = config_retenue(reseau)
        for graine in [42, 43, 44, 45, 46]:
            nom = f"{reseau}_g{graine}"
            configs[nom] = config
            graines[nom] = graine
    return configs, graines


def etape_ensemble(arguments):
    """
    Ensembles par moyenne simple, compares a la meilleure configuration LSTM seule, decoupages neufs.
    """
    configs, graines = membres_ensemble()
    assembles = lancer_cv_paralleles(configs, GRAINES_PLIS_NEUFS, arguments.processus, graine=graines)
    y = opt.preparer_tableaux(20)["y"]

    membres_lstm = []  # les 5 graines du LSTM seulement
    for nom in configs:
        if nom.startswith("lstm_"):
            membres_lstm.append(nom)
    groupes_ensembles = {
        "lstm_5_graines": membres_lstm,
        "lstm_cnn_10": list(configs.keys()),  # LSTM et CNN, 10 reseaux
    }
    reference = {}
    for graine_plis in GRAINES_PLIS_NEUFS:
        reference[graine_plis] = assembles[("lstm_g42", graine_plis)]

    resultats = {}
    plis_par_graine = {}
    for graine_plis in GRAINES_PLIS_NEUFS:
        plis_par_graine[graine_plis] = opt.plis_fd001(graine_plis)
    for nom_ensemble in groupes_ensembles:
        membres = groupes_ensembles[nom_ensemble]
        oof_ensemble = {}
        for graine_plis in GRAINES_PLIS_NEUFS:
            somme = np.zeros(len(y))
            for membre in membres:  # moyenne simple, poids egaux
                somme = somme + assembles[(membre, graine_plis)]["oof"]
            moyenne = somme / len(membres)
            rmse_par_pli = []
            for _, idx_val in plis_par_graine[graine_plis]:
                rmse_par_pli.append(rmse(y[idx_val], moyenne[idx_val]))
            oof_ensemble[graine_plis] = {"oof": moyenne, "rmse_par_pli": rmse_par_pli}
        comparaison = comparer_sur_decoupages(oof_ensemble, reference, GRAINES_PLIS_NEUFS)
        rmse_moyens = []
        for graine_plis in GRAINES_PLIS_NEUFS:
            rmse_moyens.append(float(np.mean(oof_ensemble[graine_plis]["rmse_par_pli"])))
        resultats[nom_ensemble] = {"membres": membres, "rmse_cv_par_decoupage": rmse_moyens, "comparaison": comparaison}
        afficher(f"{nom_ensemble} : RMSE {rmse_moyens}, gain {comparaison['gain_moyen']:.3f}, retenu = {comparaison['retenu']}")

    membres_seuls = {}
    for nom in configs:
        membres_seuls[nom] = resume_cv(assembles, nom, GRAINES_PLIS_NEUFS)
    ecrire_json({"regle": TEXTE_REGLE_ENSEMBLE, "ensembles": resultats, "membres": membres_seuls,
                 "date": str(datetime.date.today())}, "04h_ensemble.json")


def etape_final(arguments):
    """
    Modeles finaux (100 moteurs, nombre d'epoques = mediane des meilleures epoques des plis 42) et test
    officiel FD001, lu UNE fois, pour les configurations retenues et l'ensemble.
    """
    resultats = {}
    y_test = opt.preparer_tableaux(20)["y_test_vraie"]
    predictions_membres = {}
    taches = []
    for reseau in ["lstm", "cnn1d"]:
        config = config_retenue(reseau)
        # nombre d'epoques : depuis la confirmation (plis 42) si elle existe, sinon le 03b
        confirmation = lire_json(f"04h_confirmation_{reseau}.json")
        epoques = None
        if confirmation is not None and confirmation["comparaison"]["retenu"]:
            epoques = int(np.median(confirmation["cv_candidat"][str(opt.GRAINE_PLIS)]["meilleures_epoques"]))
        elif confirmation is not None:
            epoques = int(np.median(confirmation["cv_reference"][str(opt.GRAINE_PLIS)]["meilleures_epoques"]))
        for graine in [42, 43, 44, 45, 46]:
            taches.append({"reseau": reseau, "config": config, "nb_epoques": epoques, "graine": graine})

    with ProcessPoolExecutor(max_workers=arguments.processus) as executeur:
        futurs = {}
        for tache in taches:
            futurs[executeur.submit(entrainer_et_predire_test, tache)] = tache
        for futur in as_completed(futurs):
            tache = futurs[futur]
            predictions_membres[f"{tache['reseau']}_g{tache['graine']}"] = futur.result()
            afficher(f"final {tache['reseau']} graine {tache['graine']} : termine")

    for nom in sorted(predictions_membres):
        resultats[nom] = opt.evaluer_sur_test_dernier_cycle(predictions_membres[nom], y_test)
    membres_lstm = []  # les 5 graines du LSTM seulement
    for graine in [42, 43, 44, 45, 46]:
        membres_lstm.append(f"lstm_g{graine}")
    ensembles = {"lstm_5_graines": membres_lstm, "lstm_cnn_10": sorted(predictions_membres.keys())}
    for nom in ensembles:
        lignes = []  # une ligne de predictions par membre
        for membre in ensembles[nom]:
            lignes.append(predictions_membres[membre])
        moyenne = np.mean(np.vstack(lignes), axis=0)  # moyenne simple, poids egaux
        resultats[nom] = opt.evaluer_sur_test_dernier_cycle(moyenne, y_test)
        predictions_membres[nom] = moyenne
    for nom in resultats:
        afficher(f"TEST {nom} : RMSE {resultats[nom]['rmse']:.2f} / plafonnee {resultats[nom]['rmse_contre_rul_plafonnee']:.2f} "
                 f"/ NASA {resultats[nom]['score_nasa']:.1f}")
    tableau = pd.DataFrame(predictions_membres)
    tableau.insert(0, "rul_vraie", y_test)
    tableau.insert(0, "unit_number", opt.preparer_tableaux(20)["moteurs_test"])
    tableau.to_csv(os.path.join(DOSSIER_RESULTATS, "04h_test_dernier_cycle.csv"), index=False)
    ecrire_json({"metriques_test": resultats, "configurations": {"lstm": config_retenue("lstm"), "cnn1d": config_retenue("cnn1d")},
                 "date": str(datetime.date.today())}, "04h_test.json")


def entrainer_et_predire_test(tache):
    """Tache d'un processus : modele final d'une configuration et d'une graine, predictions au dernier cycle."""
    modele = opt.entrainer_modele_final(tache["config"], tache["nb_epoques"], graine=tache["graine"])
    return opt.predictions_test_dernier_cycle(modele, tache["config"])


def lire_arguments():
    """Lit la ligne de commande."""
    analyseur = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    analyseur.add_argument("etape", choices=["verifier", "nettoyer", "optuna", "travailleur", "resumer", "confirmer",
                                             "ablation", "ensemble", "final"])
    analyseur.add_argument("--reseau", default="lstm", choices=opt.TYPES_RESEAUX)
    analyseur.add_argument("--essais", type=int, default=100, help="nombre total d'essais de l'etude Optuna")
    analyseur.add_argument("--processus", type=int, default=12, help="processus paralleles (1 coeur chacun)")
    analyseur.add_argument("--numero", type=int, default=0, help="numero du travailleur (usage interne)")
    return analyseur.parse_args()


def main():
    """Point d'entree."""
    arguments = lire_arguments()
    os.makedirs(DOSSIER_RESULTATS, exist_ok=True)
    debut = time.perf_counter()
    if arguments.etape == "verifier":
        etape_verifier(arguments)
    elif arguments.etape == "nettoyer":
        etape_nettoyer(arguments)
    elif arguments.etape == "optuna":
        etape_optuna(arguments)
    elif arguments.etape == "travailleur":
        etape_travailleur(arguments)
    elif arguments.etape == "resumer":
        resumer_etude(arguments.reseau)
    elif arguments.etape == "confirmer":
        etape_confirmer(arguments)
    elif arguments.etape == "ablation":
        etape_ablation(arguments)
    elif arguments.etape == "ensemble":
        etape_ensemble(arguments)
    elif arguments.etape == "final":
        etape_final(arguments)
    afficher(f"etape {arguments.etape} terminee en {time.perf_counter() - debut:.0f} s")


if __name__ == "__main__":
    main()
