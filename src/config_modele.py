"""
Configuration d'un modele : validation, description et construction d'une fabrique de modeles neufs.

Pourquoi ce module ? Un modele est decrit par un petit dictionnaire (une "configuration") : une famille
(XGBoost, LightGBM, ...), une perte d'apprentissage (native, quadratique asymetrique, forme NASA), les
parametres de cette perte et les hyperparametres. Les scripts (scripts/generer_predictions_arbres.py),
l'inference en direct (src/inference.py) et les notebooks construisent leurs modeles a partir de ces
configurations, au lieu de recopier les reglages a la main. La configuration de reference est
CONFIG_PHASE3 : le XGBoost de la Phase 3 (notebook 03).

Schema d'une configuration (un dictionnaire Python) :

- "famille" : "xgboost", "lightgbm", "random_forest" ou "reference_naive" ;
- "perte" : "native", "quadratique_asymetrique" ou "nasa" ;
- "parametres_perte" : {} (native) | {"alpha": float} (quadratique_asymetrique) |
  {"h_sous": float, "h_sur": float, "d_max": float optionnel, 50 par defaut} (nasa) ;
- "hyperparametres" : dictionnaire passe tel quel (**params) a la fabrique de src/modeles.py ;
- champs descriptifs (non utilises pour construire le modele) : "nom", "regle_de_choix",
  "metriques_cv", "metriques_test", "source", "date".

Combinaisons valides : xgboost et lightgbm avec les 3 pertes ; random_forest et reference_naive
avec la perte "native" seulement (les pertes sur mesure n'existent que pour le boosting).
"""

import copy
import numbers
import os

import numpy as np

from src.modeles import (
    D_MAX_DEFAUT,
    fabrique_lightgbm,
    fabrique_lightgbm_asymetrique,
    fabrique_lightgbm_nasa,
    fabrique_random_forest,
    fabrique_reference_naive,
    fabrique_xgboost,
    fabrique_xgboost_asymetrique,
    fabrique_xgboost_nasa,
)

# Chemin ABSOLU de la racine du depot, calcule depuis l'emplacement de ce fichier (meme principe que
# DATA_DIR dans src/preprocessing.py) : il est le meme quel que soit le dossier depuis lequel Python est
# lance (racine du depot, notebooks/, tests/, scripts/). Utilise par les autres modules pour leurs chemins.
DOSSIER_SRC = os.path.dirname(os.path.abspath(__file__))  # dossier qui contient ce fichier : src/
RACINE_DEPOT = os.path.normpath(os.path.join(DOSSIER_SRC, ".."))  # src/.. simplifie en <racine du depot>

# Familles de modeles connues, et pertes acceptees pour chacune d'elles
FAMILLES_VALIDES = ["xgboost", "lightgbm", "random_forest", "reference_naive"]
PERTES_VALIDES = ["native", "quadratique_asymetrique", "nasa"]
PERTES_PAR_FAMILLE = {
    "xgboost": ["native", "quadratique_asymetrique", "nasa"],  # boosting : perte sur mesure possible
    "lightgbm": ["native", "quadratique_asymetrique", "nasa"],
    "random_forest": ["native"],  # pas de perte sur mesure pour une foret aleatoire
    "reference_naive": ["native"],  # la reference naive predit la moyenne : aucune perte a choisir
}

# Parametres de perte attendus pour chaque perte : obligatoires d'un cote, optionnels de l'autre
PARAMETRES_PERTE_OBLIGATOIRES = {
    "native": [],
    "quadratique_asymetrique": ["alpha"],
    "nasa": ["h_sous", "h_sur"],
}
PARAMETRES_PERTE_OPTIONNELS = {
    "native": [],
    "quadratique_asymetrique": [],
    "nasa": ["d_max"],  # seuil du prolongement quadratique, D_MAX_DEFAUT (50) s'il est absent
}

# Configuration de reference : le XGBoost de la Phase 3 (notebook 03), modele a arbres des scripts,
# de l'inference en direct et de l'application
CONFIG_PHASE3 = {
    "famille": "xgboost",
    "perte": "native",
    "parametres_perte": {},
    "hyperparametres": {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.05,
        "random_state": 42,
    },
    "nom": "XGBoost Phase 3",
    "regle_de_choix": "baseline Phase 3",
    "metriques_cv": {},
    "metriques_test": {},
    "source": "notebooks/03_modelisation.ipynb",
    "date": "2026-09-30",
}

# Les trois autres modeles compares a "reglages de base" dans le notebook 05 (section B), recopies a
# l'identique : ils sont aussi proposes par l'application de demo (scripts/generer_predictions_arbres.py)
CONFIG_REFERENCE_NAIVE = {
    "famille": "reference_naive", "perte": "native", "parametres_perte": {}, "hyperparametres": {},
    "nom": "Reference naive (moyenne)",
}
CONFIG_RANDOM_FOREST_BASE = {
    "famille": "random_forest", "perte": "native", "parametres_perte": {},
    "hyperparametres": {"n_estimators": 200, "random_state": 42, "n_jobs": -1},  # 200 arbres, graine 42, tous les coeurs
    "nom": "Random Forest (base)",
}
CONFIG_LIGHTGBM_BASE = {
    "famille": "lightgbm", "perte": "native", "parametres_perte": {},
    "hyperparametres": {"n_jobs": 4},  # 4 coeurs ; le reste : valeurs de fabrique_lightgbm (200 arbres, pas 0.05, ...)
    "nom": "LightGBM (base)",
}

# Reglages FINAUX des modeles a arbres, pour les 31 features (ecart a l'etat initial) : recherche Optuna
# de Matisse sur FD001 (XGBoost : 60 essais, notebook 04f ; LightGBM et Random Forest : 40 essais, notebook
# 04g ; TPE graine 42, memes plis par moteur). Valeurs recopiees a l'identique du notebook 04g.
# Les reglages "base" ci-dessus servent seulement a CHOISIR les features (regle du script des arbres) ;
# ceux-ci servent au modele final, celui de la demo. Ordre identique aux notebooks 04e puis 04f :
# d'abord les features (reglages fixes), ensuite les hyperparametres (sur les features retenues).
CONFIG_XGBOOST_FINAL = {
    "famille": "xgboost", "perte": "native", "parametres_perte": {},
    "hyperparametres": {
        "max_depth": 4,                # arbres moins profonds que la Phase 3 (5)
        "learning_rate": 0.0633,       # pas d'apprentissage
        "n_estimators": 422,           # nombre d'arbres
        "subsample": 0.5789,           # part des lignes tiree pour chaque arbre
        "colsample_bytree": 0.8676,    # part des features tiree pour chaque arbre
        "min_child_weight": 1.3287,    # poids minimal d'une feuille
        "reg_lambda": 29.1205,         # regularisation L2 des feuilles (forte)
        "random_state": 42,
    },
    "nom": "XGBoost (reglages finaux 04f)",
    "regle_de_choix": "Optuna 60 essais + confirmation sur 10 decoupages neufs (notebook 04f)",
    "metriques_cv": {},
    # FD001 sur 1 coeur, comme le modele final du script des arbres (12,70 / 220,0 au 04g, sur 16 coeurs)
    "metriques_test": {"rmse": 12.73, "score_nasa": 223.5, "nb_surestimes": 2},
    "source": "notebooks/04f_modele_final.ipynb",
    "date": "2026-11-05",
}
CONFIG_LIGHTGBM_FINAL = {
    "famille": "lightgbm", "perte": "native", "parametres_perte": {},
    "hyperparametres": {
        "n_estimators": 549,           # nombre d'arbres
        "learning_rate": 0.02744,      # pas d'apprentissage
        "num_leaves": 8,               # feuilles par arbre (31 par defaut)
        "min_child_samples": 72,       # lignes minimales par feuille
        "subsample": 0.6534,           # part des lignes tiree...
        "subsample_freq": 1,           # ... a chaque arbre (sinon subsample est ignore par LightGBM)
        "colsample_bytree": 0.9839,    # part des features tiree pour chaque arbre
        "reg_lambda": 2.5794,          # regularisation L2
        "n_jobs": 4,                   # 4 coeurs (verifie : memes predictions qu'au notebook 04g)
    },
    "nom": "LightGBM (reglages finaux 04g)",
    "regle_de_choix": "Optuna 40 essais (notebook 04g)",
    "metriques_cv": {},
    "metriques_test": {"rmse": 12.76, "score_nasa": 234.6, "nb_surestimes": 3},  # FD001, notebook 04g
    "source": "notebooks/04g_comparaison_modeles.ipynb",
    "date": "2026-11-05",
}
CONFIG_RANDOM_FOREST_FINAL = {
    "famille": "random_forest", "perte": "native", "parametres_perte": {},
    "hyperparametres": {
        "n_estimators": 391,           # nombre d'arbres
        "max_depth": 18,               # profondeur maximale (non bornee par defaut)
        "min_samples_leaf": 3,         # lignes minimales par feuille
        "max_features": 0.5432,        # part des features testee a chaque coupure
        "random_state": 42,
        "n_jobs": -1,                  # tous les coeurs (ne change pas le resultat)
    },
    "nom": "Random Forest (reglages finaux 04g)",
    "regle_de_choix": "Optuna 40 essais (notebook 04g)",
    "metriques_cv": {},
    "metriques_test": {"rmse": 13.14, "score_nasa": 243.3, "nb_surestimes": 3},  # FD001, notebook 04g
    "source": "notebooks/04g_comparaison_modeles.ipynb",
    "date": "2026-11-05",
}


def est_un_nombre(valeur):
    """
    Indique si une valeur est un nombre reel utilisable comme parametre (int, float ou type numpy).

    Les booleens sont refuses : en Python, True est aussi un entier (True == 1), ce qui laisserait
    passer {"alpha": true} sans erreur.

    Parameters
    ----------
    valeur : objet quelconque
        La valeur a tester.

    Returns
    -------
    bool
        Vrai si `valeur` est un nombre reel fini (pas un booleen, pas NaN, pas l'infini).
    """
    if isinstance(valeur, (bool, np.bool_)):  # True / False ne sont pas des parametres numeriques
        return False
    if not isinstance(valeur, numbers.Real):  # numbers.Real couvre int, float, numpy.int64, numpy.float64...
        return False
    return bool(np.isfinite(float(valeur)))  # on refuse NaN et l'infini


def valider_config(config):
    """
    Verifie qu'une configuration de modele respecte le schema, et leve une erreur claire sinon.

    Verifications, dans l'ordre :
    1. la configuration est un dictionnaire ;
    2. "famille" est presente et connue ; "perte" est presente et connue ;
    3. la combinaison famille + perte est autorisee ;
    4. "parametres_perte" (optionnel, {} par defaut) est un dictionnaire qui contient exactement les
       parametres attendus pour cette perte (aucun manquant, aucun inconnu), tous numeriques et > 0 ;
    5. "hyperparametres" (optionnel, {} par defaut) est un dictionnaire a cles texte, vide pour la
       reference naive (sa fabrique n'accepte aucun reglage).
    Les champs descriptifs (nom, regle_de_choix, metriques_cv, ...) ne sont pas controles.

    Parameters
    ----------
    config : dict
        La configuration a verifier (n'est pas modifiee).

    Returns
    -------
    None
        Ne renvoie rien : si la fonction se termine sans erreur, la configuration est valide.

    Raises
    ------
    ValueError
        Avec un message en francais qui dit precisement ce qui ne va pas.
    """
    # 1. la configuration doit etre un dictionnaire (un objet JSON une fois relu)
    if not isinstance(config, dict):
        raise ValueError(f"La configuration doit etre un dictionnaire, recu : {type(config).__name__}.")

    # 2a. la famille de modele : presente et connue
    if "famille" not in config:
        raise ValueError(f"Champ 'famille' manquant dans la configuration (valeurs possibles : {FAMILLES_VALIDES}).")
    famille = config["famille"]
    if famille not in FAMILLES_VALIDES:
        raise ValueError(f"Famille de modele inconnue : {famille!r} (valeurs possibles : {FAMILLES_VALIDES}).")

    # 2b. la perte d'apprentissage : presente et connue
    if "perte" not in config:
        raise ValueError(f"Champ 'perte' manquant dans la configuration (valeurs possibles : {PERTES_VALIDES}).")
    perte = config["perte"]
    if perte not in PERTES_VALIDES:
        raise ValueError(f"Perte inconnue : {perte!r} (valeurs possibles : {PERTES_VALIDES}).")

    # 3. la combinaison famille + perte doit exister
    pertes_autorisees = PERTES_PAR_FAMILLE[famille]
    if perte not in pertes_autorisees:
        raise ValueError(
            f"Combinaison invalide : la famille {famille!r} n'accepte que la (les) perte(s) {pertes_autorisees}, "
            f"pas {perte!r}."
        )

    # 4. les parametres de la perte : un dictionnaire ({} si le champ est absent)
    parametres_perte = config.get("parametres_perte", {})
    if not isinstance(parametres_perte, dict):
        raise ValueError(
            f"'parametres_perte' doit etre un dictionnaire, recu : {type(parametres_perte).__name__}."
        )

    obligatoires = PARAMETRES_PERTE_OBLIGATOIRES[perte]
    optionnels = PARAMETRES_PERTE_OPTIONNELS[perte]

    # 4a. aucun parametre obligatoire ne doit manquer
    for nom_parametre in obligatoires:
        if nom_parametre not in parametres_perte:
            raise ValueError(
                f"Parametre de perte manquant : la perte {perte!r} exige {obligatoires} "
                f"(manque {nom_parametre!r})."
            )

    # 4b. aucun parametre inconnu (une faute de frappe serait sinon ignoree en silence)
    for nom_parametre in parametres_perte:
        if nom_parametre not in obligatoires and nom_parametre not in optionnels:
            if len(obligatoires) == 0 and len(optionnels) == 0:  # cas de la perte native
                raise ValueError(
                    f"La perte {perte!r} n'accepte aucun parametre de perte, recu : {nom_parametre!r}."
                )
            raise ValueError(
                f"Parametre de perte inconnu pour la perte {perte!r} : {nom_parametre!r} "
                f"(attendus : {obligatoires}, optionnels : {optionnels})."
            )

    # 4c. chaque parametre de perte est un nombre strictement positif (alpha, h_sous, h_sur, d_max)
    for nom_parametre in parametres_perte:
        valeur = parametres_perte[nom_parametre]
        if not est_un_nombre(valeur):
            raise ValueError(f"Le parametre de perte {nom_parametre!r} doit etre un nombre, recu : {valeur!r}.")
        if float(valeur) <= 0:
            raise ValueError(f"Le parametre de perte {nom_parametre!r} doit etre > 0, recu : {valeur!r}.")

    # 5. les hyperparametres : un dictionnaire a cles texte ({} si le champ est absent)
    hyperparametres = config.get("hyperparametres", {})
    if not isinstance(hyperparametres, dict):
        raise ValueError(f"'hyperparametres' doit etre un dictionnaire, recu : {type(hyperparametres).__name__}.")
    for nom_hyperparametre in hyperparametres:
        if not isinstance(nom_hyperparametre, str):
            raise ValueError(f"Nom d'hyperparametre invalide (texte attendu) : {nom_hyperparametre!r}.")
    if famille == "reference_naive" and len(hyperparametres) > 0:
        raise ValueError(
            "La reference naive n'accepte aucun hyperparametre (elle predit toujours la moyenne), "
            f"recu : {list(hyperparametres.keys())}."
        )


def convertir_en_types_python(valeur):
    """
    Convertit recursivement une valeur en types Python simples, ecrivables en JSON.

    Le module json ne sait pas ecrire les types numpy (numpy.int64, numpy.bool_, numpy.ndarray),
    qui sortent pourtant souvent des calculs de metriques. On les remplace par leurs
    equivalents Python (int, float, bool, list). Les dictionnaires et listes sont parcourus en
    profondeur ; les tuples deviennent des listes (JSON n'a pas de tuple).

    Parameters
    ----------
    valeur : objet
        La valeur a convertir (dict, list, tuple, nombre numpy, tableau numpy, texte, ...).

    Returns
    -------
    objet
        Une NOUVELLE valeur, composee uniquement de dict, list, str, int, float, bool et None.
    """
    # dictionnaire : on convertit chaque valeur (les cles deviennent du texte, comme en JSON)
    if isinstance(valeur, dict):
        resultat = {}
        for cle in valeur:
            resultat[str(cle)] = convertir_en_types_python(valeur[cle])
        return resultat

    # liste ou tuple : on convertit chaque element, dans l'ordre
    if isinstance(valeur, (list, tuple)):
        resultat = []
        for element in valeur:
            resultat.append(convertir_en_types_python(element))
        return resultat

    # tableau numpy : on le transforme en liste Python, puis on convertit ses elements
    if isinstance(valeur, np.ndarray):
        return convertir_en_types_python(valeur.tolist())

    # booleen numpy AVANT les nombres (un booleen n'est pas un nombre dans notre schema)
    if isinstance(valeur, np.bool_):
        return bool(valeur)

    # entier numpy (numpy.int64, numpy.int32, ...) : entier Python
    if isinstance(valeur, np.integer):
        return int(valeur)

    # reel numpy (numpy.float64, numpy.float32, ...) : reel Python
    if isinstance(valeur, np.floating):
        return float(valeur)

    # tout le reste (str, int, float, bool, None) est deja ecrivable en JSON
    return valeur


def construire_fabrique_depuis_config(config):
    """
    Construit, a partir d'une configuration, une fabrique de modeles : une fonction SANS argument
    qui renvoie a chaque appel un modele NEUF, non entraine, correspondant a la configuration.

    C'est exactement ce qu'attend `evaluer_en_validation_croisee` (src/evaluation.py), qui appelle la
    fabrique une fois par pli ; les scripts et l'inference en direct l'utilisent aussi pour le modele final.
    La configuration est validee puis recopiee au moment de l'appel : la modifier ensuite ne change
    pas les modeles produits par la fabrique.

    Parameters
    ----------
    config : dict
        Une configuration valide (voir `valider_config`).

    Returns
    -------
    callable
        fabrique() -> modele neuf (XGBRegressor, LGBMRegressor, RegresseurCentre, RandomForestRegressor
        ou DummyRegressor selon la famille et la perte).

    Raises
    ------
    ValueError
        Si la configuration est invalide.
    """
    valider_config(config)  # erreur claire tout de suite, plutot qu'au premier pli de validation croisee

    # copies independantes de ce dont la fabrique a besoin (la config d'origine peut changer ensuite)
    famille = config["famille"]
    perte = config["perte"]
    parametres_perte = copy.deepcopy(config.get("parametres_perte", {}))
    hyperparametres = copy.deepcopy(config.get("hyperparametres", {}))

    def fabrique():
        """
        Renvoie un modele neuf, non entraine, correspondant a la configuration capturee.

        Returns
        -------
        estimateur scikit-learn
            Le modele, avec .fit et .predict.
        """
        # --- XGBoost : 3 pertes possibles ---
        if famille == "xgboost":
            if perte == "native":
                return fabrique_xgboost(**hyperparametres)
            if perte == "quadratique_asymetrique":
                return fabrique_xgboost_asymetrique(float(parametres_perte["alpha"]), **hyperparametres)
            # perte == "nasa" (seule possibilite restante, garantie par valider_config)
            d_max = float(parametres_perte.get("d_max", D_MAX_DEFAUT))  # 50 cycles si absent
            return fabrique_xgboost_nasa(
                float(parametres_perte["h_sous"]), float(parametres_perte["h_sur"]), d_max=d_max, **hyperparametres
            )

        # --- LightGBM : 3 pertes possibles ---
        if famille == "lightgbm":
            if perte == "native":
                return fabrique_lightgbm(**hyperparametres)
            if perte == "quadratique_asymetrique":
                return fabrique_lightgbm_asymetrique(float(parametres_perte["alpha"]), **hyperparametres)
            d_max = float(parametres_perte.get("d_max", D_MAX_DEFAUT))
            return fabrique_lightgbm_nasa(
                float(parametres_perte["h_sous"]), float(parametres_perte["h_sur"]), d_max=d_max, **hyperparametres
            )

        # --- Random Forest : perte native seulement ---
        if famille == "random_forest":
            return fabrique_random_forest(**hyperparametres)

        # --- Reference naive (famille == "reference_naive") : predit toujours la moyenne de y ---
        return fabrique_reference_naive()

    return fabrique


def decrire_config(config):
    """
    Resume une configuration en une ligne lisible, pour l'affichage (scripts, notebooks, metadonnees).

    Exemples : "XGBoost Phase 3 (xgboost, perte native)" ou
    "LightGBM NASA (lightgbm, perte nasa, h_sous=13, h_sur=8)".

    Parameters
    ----------
    config : dict
        Une configuration (on suppose qu'elle a au moins "famille" et "perte").

    Returns
    -------
    str
        Le nom (ou la famille si le nom est absent), suivi de la famille, de la perte et de ses parametres.
    """
    nom = config.get("nom", config["famille"])  # le nom lisible s'il existe, sinon la famille
    morceaux = [config["famille"], "perte " + config["perte"]]  # elements affiches entre parentheses

    parametres_perte = config.get("parametres_perte", {})
    for nom_parametre in parametres_perte:  # on ajoute chaque parametre de perte : "alpha=2.0", ...
        morceaux.append(f"{nom_parametre}={parametres_perte[nom_parametre]}")

    return f"{nom} ({', '.join(morceaux)})"
