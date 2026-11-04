"""
Optimisation des hyperparametres avec Optuna (Phase 4), en multi-objectif : RMSE ET score NASA.

Pourquoi deux objectifs ? Le RMSE mesure la precision moyenne ; le score NASA (13/10) punit surtout
les grosses surestimations (le cas dangereux). Les deux ne vont pas toujours ensemble : un modele plus
"prudent" peut avoir un meilleur score NASA mais un RMSE un peu moins bon. Plutot que de melanger les
deux en une seule note (avec un poids arbitraire), on demande a Optuna de minimiser les deux a la fois
et on garde le FRONT DE PARETO : l'ensemble des essais qu'aucun autre essai ne bat sur les deux
criteres en meme temps. Le choix final dans ce front se fait ensuite avec une regle ecrite a l'avance
(`appliquer_regle_de_choix`), jamais en regardant le jeu de test.

Contenu du module :
- les espaces de recherche des trois etudes (`proposer_configuration_*`) ;
- la construction de l'objectif Optuna (`construire_objectif_optuna`), qui evalue chaque essai en
  validation croisee sur les plis FIXES (les memes pour tous les essais et tous les modeles) et
  enregistre les autres metriques hors-pli dans les `user_attrs` de l'essai ;
- le lancement d'une etude reproductible (`lancer_etude`, TPESampler(seed=42)) ;
- l'extraction des essais en tableau et du front de Pareto (`essais_vers_dataframe`, `calculer_front_pareto`) ;
- la regle de choix (`appliquer_regle_de_choix`), fixee AVANT de voir le moindre resultat.

Rappel : le score NASA d'EVALUATION garde toujours ses constantes officielles 13/10. Ce que les etudes
font varier, c'est la perte d'APPRENTISSAGE (alpha de la perte quadratique asymetrique, ou h_sous / h_sur
de la perte de forme NASA, voir src/modeles.py).
"""

import json
import os
import time

import numpy as np
import optuna
import pandas as pd

from src.config_modele import construire_fabrique_depuis_config
from src.evaluation import evaluer_en_validation_croisee
from src.modeles import D_MAX_DEFAUT

# Dossier ou les scripts et le notebook ecrivent les resultats chiffres (CSV, JSON) de la Phase 4,
# calcule depuis l'emplacement de ce fichier (meme principe que DATA_DIR dans src/preprocessing.py)
DOSSIER_SRC = os.path.dirname(os.path.abspath(__file__))  # dossier qui contient ce fichier : src/
RACINE_DEPOT = os.path.normpath(os.path.join(DOSSIER_SRC, ".."))  # <racine du depot>
DOSSIER_RESULTATS = os.path.join(RACINE_DEPOT, "docs", "resultats")

# Les trois etudes Optuna de la Phase 4, et le nombre d'essais prevu pour chacune
NOMS_ETUDES = ["xgb_quadratique", "xgb_nasa", "lgbm_quadratique"]
N_ESSAIS_PAR_DEFAUT = {
    "xgb_quadratique": 60,
    "xgb_nasa": 60,
    "lgbm_quadratique": 40,
}

# Graine du TPESampler : meme graine = memes reglages proposes = memes resultats
GRAINE_OPTUNA = 42

# Graine des modeles eux-memes (sous-echantillonnage des lignes / colonnes), comme en Phase 3
GRAINE_MODELE = 42

# Nombre de coeurs utilises par CHAQUE modele. Mesure sur ce jeu (16 000 lignes environ par pli) :
# 4 coeurs est plus rapide que tous les coeurs (trop de coordination entre coeurs pour si peu de
# donnees), et le resultat est identique au bit pres avec 4, 15 ou tous les coeurs.
N_JOBS_MODELE = 4

# Regle de choix : on accepte de perdre au plus 5 % de precision (RMSE) pour gagner en score NASA
TOLERANCE_RMSE = 1.05

# Texte de la regle, recopie dans la configuration retenue (champ "regle_de_choix")
TEXTE_REGLE_DE_CHOIX = (
    "Parmi tous les essais Pareto-optimaux (RMSE CV moyen, score NASA CV moyen) de toutes les etudes "
    "Optuna, garder ceux dont le RMSE CV moyen est <= 1,05 x le plus petit RMSE CV moyen du front, puis "
    "prendre celui qui a le plus petit score NASA CV moyen ; en cas d'egalite, le moins de surestimations "
    "hors-pli > 20 cycles. Regle fixee avant tout resultat ; le jeu de test n'intervient pas."
)


# ---------------------------------------------------------------------------------------------------
# Espaces de recherche
# ---------------------------------------------------------------------------------------------------

def proposer_hyperparametres_xgboost(trial, n_estimators_max, learning_rate_max):
    """
    Fait proposer par Optuna les hyperparametres XGBoost communs aux deux etudes XGBoost.

    Bornes : n_estimators [100, n_estimators_max], learning_rate [0.01, learning_rate_max] (echelle log),
    max_depth [3, 8], min_child_weight [1, 20] (log), subsample [0.6, 1.0], colsample_bytree [0.5, 1.0],
    reg_lambda [0.1, 10] (log). "Echelle log" : Optuna tire aussi souvent entre 0,01 et 0,1 qu'entre 0,1
    et 1 ; c'est le bon choix pour un reglage dont l'effet depend de son ordre de grandeur.

    Parameters
    ----------
    trial : optuna.trial.Trial
        L'essai en cours (c'est lui qui "tire" chaque valeur).
    n_estimators_max : int
        Nombre maximal d'arbres (800 pour la perte quadratique, 1200 pour la perte NASA).
    learning_rate_max : float
        Pas d'apprentissage maximal (0.2 pour la perte quadratique, 0.3 pour la perte NASA).

    Returns
    -------
    dict
        Les hyperparametres XGBoost, prets a etre passes a la fabrique de src/modeles.py.
    """
    hyperparametres = {}
    hyperparametres["n_estimators"] = trial.suggest_int("n_estimators", 100, n_estimators_max)
    hyperparametres["learning_rate"] = trial.suggest_float("learning_rate", 0.01, learning_rate_max, log=True)
    hyperparametres["max_depth"] = trial.suggest_int("max_depth", 3, 8)
    hyperparametres["min_child_weight"] = trial.suggest_float("min_child_weight", 1.0, 20.0, log=True)
    hyperparametres["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
    hyperparametres["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.5, 1.0)
    hyperparametres["reg_lambda"] = trial.suggest_float("reg_lambda", 0.1, 10.0, log=True)
    # reglages fixes (pas cherches par Optuna) : graine du modele et nombre de coeurs
    hyperparametres["random_state"] = GRAINE_MODELE
    hyperparametres["n_jobs"] = N_JOBS_MODELE
    return hyperparametres


def proposer_configuration_xgb_quadratique(trial):
    """
    Espace de recherche de l'etude "xgb_quadratique" : XGBoost + perte quadratique asymetrique.

    alpha dans [1, 4] (1 = perte quadratique classique, 4 = une surestimation coute 4 fois plus cher
    qu'une sous-estimation de meme taille), puis les hyperparametres XGBoost (800 arbres et un pas de
    0.2 au plus).

    Parameters
    ----------
    trial : optuna.trial.Trial
        L'essai en cours.

    Returns
    -------
    dict
        Une configuration au schema de src/config_modele.py (famille, perte, parametres_perte,
        hyperparametres).
    """
    parametres_perte = {}
    parametres_perte["alpha"] = trial.suggest_float("alpha", 1.0, 4.0)

    config = {}
    config["famille"] = "xgboost"
    config["perte"] = "quadratique_asymetrique"
    config["parametres_perte"] = parametres_perte
    config["hyperparametres"] = proposer_hyperparametres_xgboost(trial, n_estimators_max=800, learning_rate_max=0.2)
    return config


def proposer_configuration_xgb_nasa(trial):
    """
    Espace de recherche de l'etude "xgb_nasa" : XGBoost + perte de forme NASA.

    C'est la version correcte de l'idee "faire varier 13 et 10 avec Optuna" : les denominateurs passent
    dans la perte d'APPRENTISSAGE (h_sur dans [3, 15], h_sous dans [8, 30]), la note d'evaluation
    restant le score NASA 13/10. Comme chaque arbre ne deplace une feuille que d'environ
    learning_rate * h cycles avec cette perte (voir src/modeles.py), on autorise plus d'arbres
    (jusqu'a 1200) et un pas plus grand (jusqu'a 0.3). d_max (raccord quadratique) reste fixe a 50.

    Parameters
    ----------
    trial : optuna.trial.Trial
        L'essai en cours.

    Returns
    -------
    dict
        Une configuration au schema de src/config_modele.py.
    """
    parametres_perte = {}
    parametres_perte["h_sous"] = trial.suggest_float("h_sous", 8.0, 30.0)
    parametres_perte["h_sur"] = trial.suggest_float("h_sur", 3.0, 15.0)
    parametres_perte["d_max"] = D_MAX_DEFAUT  # fixe, ecrit explicitement pour que la config se suffise

    config = {}
    config["famille"] = "xgboost"
    config["perte"] = "nasa"
    config["parametres_perte"] = parametres_perte
    config["hyperparametres"] = proposer_hyperparametres_xgboost(trial, n_estimators_max=1200, learning_rate_max=0.3)
    return config


def proposer_configuration_lgbm_quadratique(trial):
    """
    Espace de recherche de l'etude "lgbm_quadratique" : LightGBM + perte quadratique asymetrique.

    alpha [1, 4] ; n_estimators [100, 600] ; learning_rate [0.01, 0.2] (log) ; num_leaves [15, 127] (log) ;
    min_child_samples [5, 100] (log) ; subsample (= bagging_fraction) [0.6, 1.0] avec subsample_freq = 1
    (= bagging_freq : on re-tire les lignes a chaque arbre, sinon subsample n'aurait aucun effet) ;
    colsample_bytree [0.5, 1.0] ; reg_lambda [0.001, 10] (log). Les reglages de reproductibilite de
    `fabrique_lightgbm` (deterministic, force_row_wise) sont gardes.

    Parameters
    ----------
    trial : optuna.trial.Trial
        L'essai en cours.

    Returns
    -------
    dict
        Une configuration au schema de src/config_modele.py.
    """
    parametres_perte = {}
    parametres_perte["alpha"] = trial.suggest_float("alpha", 1.0, 4.0)

    hyperparametres = {}
    hyperparametres["n_estimators"] = trial.suggest_int("n_estimators", 100, 600)
    hyperparametres["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
    hyperparametres["num_leaves"] = trial.suggest_int("num_leaves", 15, 127, log=True)
    hyperparametres["min_child_samples"] = trial.suggest_int("min_child_samples", 5, 100, log=True)
    hyperparametres["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
    hyperparametres["subsample_freq"] = 1  # fixe : sous-echantillonnage refait a chaque arbre
    hyperparametres["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.5, 1.0)
    hyperparametres["reg_lambda"] = trial.suggest_float("reg_lambda", 0.001, 10.0, log=True)
    # reglages fixes : graine du modele et nombre de coeurs
    hyperparametres["random_state"] = GRAINE_MODELE
    hyperparametres["n_jobs"] = N_JOBS_MODELE

    config = {}
    config["famille"] = "lightgbm"
    config["perte"] = "quadratique_asymetrique"
    config["parametres_perte"] = parametres_perte
    config["hyperparametres"] = hyperparametres
    return config


# association "nom d'etude" -> fonction qui definit son espace de recherche
ESPACES_DE_RECHERCHE = {
    "xgb_quadratique": proposer_configuration_xgb_quadratique,
    "xgb_nasa": proposer_configuration_xgb_nasa,
    "lgbm_quadratique": proposer_configuration_lgbm_quadratique,
}


def proposer_configuration(trial, nom_etude, hyperparametres_imposes=None):
    """
    Fait proposer par Optuna une configuration de l'etude `nom_etude`.

    Parameters
    ----------
    trial : optuna.trial.Trial
        L'essai en cours.
    nom_etude : str
        "xgb_quadratique", "xgb_nasa" ou "lgbm_quadratique".
    hyperparametres_imposes : dict, optionnel
        Hyperparametres qui ECRASENT ceux proposes par Optuna (reserve aux tests rapides, par exemple
        {"n_estimators": 10} ; None en usage normal).

    Returns
    -------
    dict
        La configuration (schema de src/config_modele.py).

    Raises
    ------
    ValueError
        Si le nom d'etude est inconnu.
    """
    if nom_etude not in ESPACES_DE_RECHERCHE:
        raise ValueError(f"Etude inconnue : {nom_etude!r} (etudes possibles : {NOMS_ETUDES}).")

    fonction_espace = ESPACES_DE_RECHERCHE[nom_etude]  # la fonction qui definit l'espace de cette etude
    config = fonction_espace(trial)

    if hyperparametres_imposes is not None:  # usage test uniquement : on ecrase certains reglages
        for nom_hyperparametre in hyperparametres_imposes:
            config["hyperparametres"][nom_hyperparametre] = hyperparametres_imposes[nom_hyperparametre]
    return config


# ---------------------------------------------------------------------------------------------------
# Objectif Optuna
# ---------------------------------------------------------------------------------------------------

def enregistrer_metriques_essai(trial, config, resultats_cv, duree_essai):
    """
    Range dans les `user_attrs` de l'essai toutes les metriques qui ne sont pas les deux objectifs.

    Optuna ne minimise que deux nombres (RMSE CV moyen, score NASA CV moyen), mais on veut garder la
    trace du reste pour analyser les essais ensuite : ecarts-types entre plis, MAE, metriques hors-pli
    (surestimations > 20 cycles, alertes manquees, fausses alertes, biais), temps de calcul, et la
    configuration complete (en JSON) pour pouvoir reconstruire exactement le modele de chaque essai.

    Parameters
    ----------
    trial : optuna.trial.Trial
        L'essai en cours.
    config : dict
        La configuration evaluee.
    resultats_cv : dict
        Le dictionnaire renvoye par `evaluer_en_validation_croisee` (src/evaluation.py).
    duree_essai : float
        Duree totale de l'essai, en secondes.

    Returns
    -------
    None
    """
    # description du modele, et configuration complete en texte JSON (reconstruction exacte possible)
    trial.set_user_attr("famille", config["famille"])
    trial.set_user_attr("perte", config["perte"])
    trial.set_user_attr("parametres_perte_json", json.dumps(config["parametres_perte"], sort_keys=True))
    trial.set_user_attr("hyperparametres_json", json.dumps(config["hyperparametres"], sort_keys=True))

    # dispersion entre plis des deux objectifs, et MAE
    trial.set_user_attr("rmse_cv_ecart_type", resultats_cv["rmse_ecart_type"])
    trial.set_user_attr("nasa_cv_ecart_type", resultats_cv["nasa_ecart_type"])
    trial.set_user_attr("mae_cv_moyen", resultats_cv["mae_moyen"])
    trial.set_user_attr("surestimes_cv_moyen", resultats_cv["surestimes_moyen"])

    # metriques calculees sur TOUTES les predictions hors-pli d'un coup (20 631 lignes)
    trial.set_user_attr("rmse_oof", resultats_cv["rmse_oof"])
    trial.set_user_attr("mae_oof", resultats_cv["mae_oof"])
    trial.set_user_attr("nasa_oof", resultats_cv["nasa_oof"])
    trial.set_user_attr("surestimes_oof", resultats_cv["surestimes_oof"])
    trial.set_user_attr("alertes_manquees_oof", resultats_cv["alertes_manquees_oof"])
    trial.set_user_attr("fausses_alertes_oof", resultats_cv["fausses_alertes_oof"])
    trial.set_user_attr("biais_oof", resultats_cv["biais_oof"])

    # temps de calcul (secondes) : entrainement et prediction cumules sur les 5 plis, et essai complet
    trial.set_user_attr("temps_entrainement_total_s", resultats_cv["temps_entrainement_total"])
    trial.set_user_attr("temps_prediction_total_s", resultats_cv["temps_prediction_total"])
    trial.set_user_attr("duree_essai_s", duree_essai)


def construire_objectif_optuna(nom_etude, X, y, groups, plis, hyperparametres_imposes=None):
    """
    Construit la fonction objectif d'une etude Optuna multi-objectif (RMSE CV moyen, score NASA CV moyen).

    A chaque essai, la fonction : (1) fait proposer une configuration par Optuna ; (2) construit la
    fabrique de modeles correspondante (src/config_modele.py, la meme que la demo) ; (3) l'evalue en
    validation croisee sur les plis FIXES `plis` (modele neuf a chaque pli) ; (4) enregistre les autres
    metriques dans les `user_attrs` ; (5) renvoie les deux objectifs a minimiser.

    Parameters
    ----------
    nom_etude : str
        "xgb_quadratique", "xgb_nasa" ou "lgbm_quadratique".
    X : pandas.DataFrame
        Les features.
    y : pandas.Series
        La cible (RUL plafonnee).
    groups : pandas.Series
        Le numero de moteur de chaque ligne.
    plis : list de (numpy.ndarray, numpy.ndarray)
        Les plis de `calculer_plis`, calcules UNE fois et reutilises pour tous les essais.
    hyperparametres_imposes : dict, optionnel
        Reserve aux tests rapides (voir `proposer_configuration`).

    Returns
    -------
    callable
        objectif(trial) -> (rmse_cv_moyen, nasa_cv_moyen).
    """
    def objectif(trial):
        """
        Evalue un essai Optuna en validation croisee et renvoie les deux objectifs.

        Parameters
        ----------
        trial : optuna.trial.Trial
            L'essai en cours (fourni par Optuna).

        Returns
        -------
        (float, float)
            Le RMSE CV moyen et le score NASA CV moyen (tous deux a minimiser).
        """
        debut = time.perf_counter()  # chronometre de l'essai complet

        # 1) Optuna propose une configuration dans l'espace de recherche de l'etude
        config = proposer_configuration(trial, nom_etude, hyperparametres_imposes)

        # 2) fabrique de modeles neufs, construite exactement comme le fera la demo
        fabrique = construire_fabrique_depuis_config(config)

        # 3) validation croisee par moteur, sur les plis fixes
        resultats_cv = evaluer_en_validation_croisee(fabrique, X, y, groups, plis)

        # 4) on garde la trace de toutes les autres metriques
        duree_essai = time.perf_counter() - debut
        enregistrer_metriques_essai(trial, config, resultats_cv, duree_essai)

        # 5) les deux objectifs a minimiser, dans l'ordre des "directions" de l'etude
        return resultats_cv["rmse_moyen"], resultats_cv["nasa_moyen"]

    return objectif


# ---------------------------------------------------------------------------------------------------
# Lancement d'une etude
# ---------------------------------------------------------------------------------------------------

def construire_rappel_progression(nom_etude, n_essais):
    """
    Construit un "rappel" (callback) Optuna qui affiche une ligne de progression apres chaque essai.

    Parameters
    ----------
    nom_etude : str
        Le nom de l'etude (affiche en debut de ligne).
    n_essais : int
        Le nombre total d'essais prevus (pour afficher "essai 12/60").

    Returns
    -------
    callable
        rappel(study, trial), appele par Optuna a la fin de chaque essai.
    """
    def rappel(study, trial):
        """
        Affiche le resultat de l'essai qui vient de se terminer et le meilleur RMSE vu jusqu'ici.

        Parameters
        ----------
        study : optuna.study.Study
            L'etude en cours.
        trial : optuna.trial.FrozenTrial
            L'essai qui vient de se terminer.

        Returns
        -------
        None
        """
        if trial.values is None:  # essai echoue : rien a afficher d'autre que son etat
            print(f"[{nom_etude}] essai {trial.number + 1}/{n_essais} : {trial.state}", flush=True)
            return

        # meilleur RMSE CV parmi les essais termines jusqu'ici
        meilleur_rmse = None
        for essai in study.trials:
            if essai.values is None:
                continue
            if meilleur_rmse is None or essai.values[0] < meilleur_rmse:
                meilleur_rmse = essai.values[0]

        duree = trial.user_attrs.get("duree_essai_s", float("nan"))
        print(
            f"[{nom_etude}] essai {trial.number + 1}/{n_essais} : RMSE CV={trial.values[0]:.3f}, "
            f"NASA CV={trial.values[1]:.1f}, surestimes OOF>20={trial.user_attrs.get('surestimes_oof')}, "
            f"duree={duree:.1f} s | meilleur RMSE jusqu'ici={meilleur_rmse:.3f}",
            flush=True,  # affichage immediat (utile quand la sortie est redirigee dans un fichier log)
        )

    return rappel


def lancer_etude(nom_etude, X, y, groups, plis, n_essais, graine=GRAINE_OPTUNA, hyperparametres_imposes=None,
                 afficher_progression=True):
    """
    Lance une etude Optuna multi-objectif reproductible et renvoie l'etude et le tableau de ses essais.

    Reproductibilite : TPESampler(seed=graine), essais executes l'un apres l'autre (jamais en
    parallele : en parallele, l'ordre d'arrivee des resultats changerait les propositions suivantes),
    modeles a graine fixe. Meme graine + memes donnees + meme environnement = memes essais, memes
    metriques (seuls les temps de calcul changent).

    TPE (Tree-structured Parzen Estimator) : apres quelques essais tires au hasard (10 par defaut),
    Optuna separe les essais deja faits en "bons" et "moins bons" et propose de nouveaux reglages qui
    ressemblent davantage aux bons. En multi-objectif, "bon" veut dire proche du front de Pareto.

    Parameters
    ----------
    nom_etude : str
        "xgb_quadratique", "xgb_nasa" ou "lgbm_quadratique".
    X : pandas.DataFrame
        Les features.
    y : pandas.Series
        La cible.
    groups : pandas.Series
        Le numero de moteur de chaque ligne.
    plis : list de (numpy.ndarray, numpy.ndarray)
        Les plis fixes de `calculer_plis`.
    n_essais : int
        Nombre d'essais.
    graine : int, optionnel
        Graine du TPESampler (42 par defaut).
    hyperparametres_imposes : dict, optionnel
        Reserve aux tests rapides (voir `proposer_configuration`).
    afficher_progression : bool, optionnel
        Affiche une ligne par essai (Vrai par defaut).

    Returns
    -------
    (optuna.study.Study, pandas.DataFrame)
        L'etude et le tableau de ses essais (voir `essais_vers_dataframe`).
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)  # on remplace les logs d'Optuna par les notres

    echantillonneur = optuna.samplers.TPESampler(seed=graine)  # graine fixe : reproductible
    etude = optuna.create_study(
        study_name=nom_etude,
        directions=["minimize", "minimize"],  # objectif 1 : RMSE CV moyen ; objectif 2 : score NASA CV moyen
        sampler=echantillonneur,
    )

    objectif = construire_objectif_optuna(nom_etude, X, y, groups, plis, hyperparametres_imposes)

    rappels = []
    if afficher_progression:
        rappels.append(construire_rappel_progression(nom_etude, n_essais))

    debut = time.perf_counter()
    etude.optimize(objectif, n_trials=n_essais, n_jobs=1, callbacks=rappels)  # n_jobs=1 : essais l'un apres l'autre
    duree_etude = time.perf_counter() - debut

    tableau = essais_vers_dataframe(etude, nom_etude)
    tableau["duree_etude_totale_s"] = duree_etude  # meme valeur sur chaque ligne : le CSV se suffit a lui-meme
    tableau["graine_optuna"] = graine
    tableau["version_optuna"] = optuna.__version__
    return etude, tableau


# ---------------------------------------------------------------------------------------------------
# Extraction des essais et front de Pareto
# ---------------------------------------------------------------------------------------------------

def essais_vers_dataframe(etude, nom_etude):
    """
    Transforme les essais termines d'une etude en tableau : une ligne par essai.

    Colonnes : "etude", "numero", "famille", "perte", un "param_<nom>" par reglage tire par Optuna,
    les deux objectifs ("rmse_cv_moyen", "nasa_cv_moyen"), toutes les metriques des user_attrs, la
    configuration en JSON ("parametres_perte_json", "hyperparametres_json") et "pareto_etude" (Vrai si
    l'essai est sur le front de Pareto de SON etude, d'apres `etude.best_trials`).

    Parameters
    ----------
    etude : optuna.study.Study
        L'etude terminee.
    nom_etude : str
        Le nom de l'etude (recopie dans la colonne "etude").

    Returns
    -------
    pandas.DataFrame
        Le tableau des essais termines, trie par numero d'essai.
    """
    # numeros des essais du front de Pareto de l'etude (Optuna les calcule lui-meme)
    numeros_pareto = set()
    for essai in etude.best_trials:
        numeros_pareto.add(essai.number)

    lignes = []
    for essai in etude.trials:
        if essai.state != optuna.trial.TrialState.COMPLETE:  # on ignore les essais echoues ou interrompus
            continue

        ligne = {}
        ligne["etude"] = nom_etude
        ligne["numero"] = essai.number
        ligne["famille"] = essai.user_attrs["famille"]
        ligne["perte"] = essai.user_attrs["perte"]

        # les reglages tires par Optuna, un par colonne
        for nom_parametre in essai.params:
            ligne["param_" + nom_parametre] = essai.params[nom_parametre]

        # les deux objectifs
        ligne["rmse_cv_moyen"] = essai.values[0]
        ligne["nasa_cv_moyen"] = essai.values[1]

        # toutes les autres metriques (sauf famille / perte, deja ecrites)
        for nom_attribut in essai.user_attrs:
            if nom_attribut in ("famille", "perte"):
                continue
            ligne[nom_attribut] = essai.user_attrs[nom_attribut]

        ligne["pareto_etude"] = essai.number in numeros_pareto
        lignes.append(ligne)

    tableau = pd.DataFrame(lignes)
    if len(tableau) > 0:
        tableau = tableau.sort_values("numero").reset_index(drop=True)
    return tableau


def calculer_front_pareto(tableau, colonne_x="rmse_cv_moyen", colonne_y="nasa_cv_moyen"):
    """
    Indique, pour chaque ligne, si elle est sur le front de Pareto des deux colonnes (a minimiser).

    Une ligne A est "dominee" s'il existe une autre ligne B au moins aussi bonne sur les deux criteres
    (B.x <= A.x et B.y <= A.y) et strictement meilleure sur au moins un. Le front de Pareto, c'est
    l'ensemble des lignes non dominees : pour passer de l'une a l'autre, on gagne forcement sur un
    critere ce qu'on perd sur l'autre.

    Exemple : (RMSE, NASA) = A (16, 300), B (17, 200), C (17, 350). C est dominee par A (moins bon sur
    les deux) ; A et B ne se dominent pas : le front est {A, B}.

    Parameters
    ----------
    tableau : pandas.DataFrame
        Les essais (de une ou plusieurs etudes).
    colonne_x : str, optionnel
        Premier critere a minimiser ("rmse_cv_moyen" par defaut).
    colonne_y : str, optionnel
        Second critere a minimiser ("nasa_cv_moyen" par defaut).

    Returns
    -------
    numpy.ndarray de bool
        Vrai pour les lignes du front de Pareto, dans l'ordre des lignes du tableau.
    """
    valeurs_x = tableau[colonne_x].to_numpy(dtype=float)
    valeurs_y = tableau[colonne_y].to_numpy(dtype=float)
    nb_lignes = len(valeurs_x)

    sur_le_front = np.ones(nb_lignes, dtype=bool)  # au depart, on suppose chaque ligne non dominee

    for i in range(nb_lignes):  # la ligne candidate
        for j in range(nb_lignes):  # une concurrente
            if i == j:
                continue
            aussi_bon_partout = valeurs_x[j] <= valeurs_x[i] and valeurs_y[j] <= valeurs_y[i]
            meilleur_quelque_part = valeurs_x[j] < valeurs_x[i] or valeurs_y[j] < valeurs_y[i]
            if aussi_bon_partout and meilleur_quelque_part:  # j domine i
                sur_le_front[i] = False
                break  # inutile de chercher une autre concurrente

    return sur_le_front


def config_depuis_ligne(ligne):
    """
    Reconstruit la configuration d'un essai (schema de src/config_modele.py) a partir de sa ligne.

    Parameters
    ----------
    ligne : pandas.Series ou dict
        Une ligne du tableau des essais (avec "famille", "perte", "parametres_perte_json",
        "hyperparametres_json").

    Returns
    -------
    dict
        La configuration : famille, perte, parametres_perte, hyperparametres.
    """
    config = {}
    config["famille"] = ligne["famille"]
    config["perte"] = ligne["perte"]
    config["parametres_perte"] = json.loads(ligne["parametres_perte_json"])  # texte JSON -> dictionnaire
    config["hyperparametres"] = json.loads(ligne["hyperparametres_json"])
    return config


# ---------------------------------------------------------------------------------------------------
# Regle de choix (fixee avant tout resultat)
# ---------------------------------------------------------------------------------------------------

def choisir_meilleure_ligne(tableau, criteres):
    """
    Renvoie la position de la meilleure ligne selon une liste de criteres a minimiser, dans l'ordre.

    On compare d'abord le 1er critere ; en cas d'egalite exacte, le 2e ; etc. Si tout est egal, la
    premiere ligne rencontree gagne (resultat deterministe).

    Parameters
    ----------
    tableau : pandas.DataFrame
        Les lignes candidates (au moins une).
    criteres : list de str
        Les colonnes a minimiser, de la plus importante a la moins importante.

    Returns
    -------
    int
        La position (0 = premiere ligne) de la meilleure ligne dans `tableau`.
    """
    meilleure_position = 0
    for position in range(1, len(tableau)):  # on compare chaque ligne a la meilleure trouvee jusqu'ici
        for critere in criteres:
            valeur = tableau[critere].iloc[position]
            valeur_meilleure = tableau[critere].iloc[meilleure_position]
            if valeur < valeur_meilleure:  # strictement meilleure sur ce critere : nouvelle meilleure
                meilleure_position = position
                break
            if valeur > valeur_meilleure:  # strictement moins bonne : on passe a la ligne suivante
                break
            # egalite exacte sur ce critere : on regarde le critere suivant
    return meilleure_position


def appliquer_regle_de_choix(tableau_essais, tolerance=TOLERANCE_RMSE):
    """
    Applique la regle de choix de la Phase 4, fixee AVANT tout resultat (voir TEXTE_REGLE_DE_CHOIX).

    1. Front de Pareto (RMSE CV moyen, score NASA CV moyen) de TOUS les essais, toutes etudes confondues.
    2. On garde les essais du front dont le RMSE CV moyen est <= tolerance x le plus petit RMSE CV
       moyen du front (tolerance = 1,05 : on accepte de perdre au plus 5 % de precision).
    3. Parmi eux, le plus petit score NASA CV moyen ; en cas d'egalite, le moins de surestimations
       hors-pli > 20 cycles ("surestimes_oof").
    Second candidat garde a cote : "le meilleur RMSE CV" (plus petit RMSE CV moyen du front ; en cas
    d'egalite, le plus petit score NASA CV moyen).

    Exemple : front = A (16.0 ; 30 000), B (16.5 ; 27 000), C (17.0 ; 25 000). Seuil = 1,05 x 16.0 = 16.8 :
    A et B sont admissibles, C ne l'est pas (17.0 > 16.8). Le plus petit NASA parmi A et B : B est retenu ;
    A est le meilleur RMSE.

    Parameters
    ----------
    tableau_essais : pandas.DataFrame
        Les essais de toutes les etudes (colonnes "rmse_cv_moyen", "nasa_cv_moyen", "surestimes_oof").
    tolerance : float, optionnel
        Perte de precision acceptee (1.05 par defaut).

    Returns
    -------
    dict
        - "front" : pandas.DataFrame, les essais Pareto-optimaux, tries par RMSE CV moyen croissant ;
        - "rmse_min_front" : float, le plus petit RMSE CV moyen du front ;
        - "seuil_rmse" : float, tolerance x rmse_min_front ;
        - "admissibles" : pandas.DataFrame, les essais du front sous le seuil ;
        - "retenu" : pandas.Series, l'essai choisi par la regle ;
        - "meilleur_rmse" : pandas.Series, l'essai au plus petit RMSE CV moyen du front ;
        - "meme_essai" : bool, Vrai si les deux candidats sont le meme essai.
    """
    tableau = tableau_essais.reset_index(drop=True)  # positions 0, 1, 2, ... propres

    # 1) front de Pareto toutes etudes confondues
    masque_front = calculer_front_pareto(tableau, "rmse_cv_moyen", "nasa_cv_moyen")
    front = tableau[masque_front].sort_values("rmse_cv_moyen").reset_index(drop=True)

    # 2) seuil de precision : au plus 5 % de RMSE en plus que le meilleur RMSE du front
    rmse_min_front = float(front["rmse_cv_moyen"].min())
    seuil_rmse = tolerance * rmse_min_front
    admissibles = front[front["rmse_cv_moyen"] <= seuil_rmse].reset_index(drop=True)

    # 3) plus petit score NASA CV moyen, puis moins de surestimations hors-pli > 20 cycles
    position_retenu = choisir_meilleure_ligne(admissibles, ["nasa_cv_moyen", "surestimes_oof"])
    retenu = admissibles.iloc[position_retenu]

    # second candidat : le meilleur RMSE CV moyen du front (puis le plus petit NASA en cas d'egalite)
    position_meilleur_rmse = choisir_meilleure_ligne(front, ["rmse_cv_moyen", "nasa_cv_moyen"])
    meilleur_rmse = front.iloc[position_meilleur_rmse]

    meme_essai = (retenu["etude"] == meilleur_rmse["etude"]) and (retenu["numero"] == meilleur_rmse["numero"])

    resultat = {}
    resultat["front"] = front
    resultat["rmse_min_front"] = rmse_min_front
    resultat["seuil_rmse"] = seuil_rmse
    resultat["admissibles"] = admissibles
    resultat["retenu"] = retenu
    resultat["meilleur_rmse"] = meilleur_rmse
    resultat["meme_essai"] = bool(meme_essai)
    return resultat


# ---------------------------------------------------------------------------------------------------
# Lecture / ecriture des resultats
# ---------------------------------------------------------------------------------------------------

def chemin_csv_etude(nom_etude, dossier=None):
    """
    Renvoie le chemin du CSV des essais d'une etude : <dossier>/04_optuna_<etude>.csv.

    Parameters
    ----------
    nom_etude : str
        Le nom de l'etude.
    dossier : str, optionnel
        Le dossier des resultats (DOSSIER_RESULTATS, c'est-a-dire docs/resultats/, par defaut).

    Returns
    -------
    str
        Le chemin du fichier.
    """
    if dossier is None:
        dossier = DOSSIER_RESULTATS
    return os.path.join(dossier, f"04_optuna_{nom_etude}.csv")


def charger_essais(noms_etudes=None, dossier=None):
    """
    Relit les CSV des etudes Optuna et les rassemble en un seul tableau.

    Parameters
    ----------
    noms_etudes : list de str, optionnel
        Les etudes a relire (NOMS_ETUDES par defaut).
    dossier : str, optionnel
        Le dossier des resultats (docs/resultats/ par defaut).

    Returns
    -------
    pandas.DataFrame
        Tous les essais, etude par etude, dans l'ordre de `noms_etudes`.

    Raises
    ------
    FileNotFoundError
        Si le CSV d'une etude n'existe pas (il faut alors lancer scripts/lancer_optuna.py).
    """
    if noms_etudes is None:
        noms_etudes = NOMS_ETUDES

    tableaux = []
    for nom_etude in noms_etudes:
        chemin = chemin_csv_etude(nom_etude, dossier)
        if not os.path.exists(chemin):
            raise FileNotFoundError(
                f"{chemin} introuvable : lancer d'abord `python scripts/lancer_optuna.py --etudes {nom_etude}`."
            )
        tableaux.append(pd.read_csv(chemin))

    return pd.concat(tableaux, ignore_index=True)
