"""
Configuration du modele retenu (Phase 4) : lecture, validation, ecriture et construction du modele.

Pourquoi ce module ? L'optimisation des hyperparametres (Phase 4) choisit UN modele : une famille
(XGBoost, LightGBM, ...), une perte d'apprentissage (native, quadratique asymetrique, forme NASA),
les parametres de cette perte et les hyperparametres. Ce choix est ecrit dans un petit fichier JSON
(docs/resultats/04_config_retenue.json), lisible par un humain et versionne avec le code. La demo,
le notebook de la Phase 4 et les scripts relisent ce fichier : un seul endroit decrit "le modele
retenu", et personne n'a a recopier les reglages a la main.

Schema d'une configuration (un dictionnaire Python, ou un objet JSON) :

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
import json
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

# Chemin ABSOLU du fichier de configuration retenue, calcule depuis l'emplacement de ce fichier
# (meme principe que DATA_DIR dans src/preprocessing.py) : il est le meme quel que soit le dossier
# depuis lequel Python est lance (racine du depot, notebooks/, tests/, scripts/).
DOSSIER_SRC = os.path.dirname(os.path.abspath(__file__))  # dossier qui contient ce fichier : src/
RACINE_DEPOT = os.path.normpath(os.path.join(DOSSIER_SRC, ".."))  # src/.. simplifie en <racine du depot>
CHEMIN_CONFIG_RETENUE = os.path.join(RACINE_DEPOT, "docs", "resultats", "04_config_retenue.json")

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

# Configuration de reference : le XGBoost de la Phase 3 (notebook 03), utilisee si aucun fichier
# de configuration retenue n'existe encore
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
    qui sortent pourtant souvent d'Optuna ou des calculs de metriques. On les remplace par leurs
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


def charger_config(chemin=None):
    """
    Lit une configuration de modele depuis un fichier JSON et la valide.

    Si le fichier n'existe pas (par exemple avant que l'optimisation de la Phase 4 n'ait ecrit son
    resultat), on l'explique a l'ecran et on renvoie une copie de CONFIG_PHASE3 : la demo reste
    utilisable avec le modele de reference de la Phase 3.

    Parameters
    ----------
    chemin : str, optionnel
        Chemin du fichier JSON. Par defaut : CHEMIN_CONFIG_RETENUE (docs/resultats/04_config_retenue.json).

    Returns
    -------
    dict
        La configuration lue et validee, ou une copie independante de CONFIG_PHASE3.

    Raises
    ------
    ValueError
        Si le fichier existe mais n'est pas un JSON valide, ou si la configuration est invalide.
    """
    if chemin is None:  # aucun chemin donne : on lit la configuration retenue par defaut
        chemin = CHEMIN_CONFIG_RETENUE

    # fichier absent : message explicite puis repli sur la configuration de la Phase 3
    if not os.path.exists(chemin):
        print(f"[config] Fichier de configuration introuvable : {chemin}")
        print(f"[config] -> on utilise la configuration de reference : {CONFIG_PHASE3['nom']}.")
        return copy.deepcopy(CONFIG_PHASE3)  # copie profonde : modifier le resultat ne touche pas CONFIG_PHASE3

    # lecture du fichier JSON (UTF-8, comme a l'ecriture)
    with open(chemin, "r", encoding="utf-8") as fichier:
        try:
            config = json.load(fichier)
        except json.JSONDecodeError as erreur:
            raise ValueError(f"Le fichier {chemin} n'est pas un JSON valide : {erreur}") from erreur

    valider_config(config)  # leve une ValueError claire si la configuration est incoherente
    return config


def sauvegarder_config(config, chemin=None):
    """
    Valide une configuration puis l'ecrit en JSON indente (UTF-8), en creant le dossier si besoin.

    Les types numpy eventuels (numpy.float64, numpy.int64, tableaux, ...) sont convertis en types
    Python avant l'ecriture (voir `convertir_en_types_python`). Si la configuration est invalide,
    rien n'est ecrit.

    Parameters
    ----------
    config : dict
        La configuration a ecrire (n'est pas modifiee).
    chemin : str, optionnel
        Chemin du fichier JSON. Par defaut : CHEMIN_CONFIG_RETENUE.

    Returns
    -------
    str
        Le chemin du fichier ecrit.

    Raises
    ------
    ValueError
        Si la configuration est invalide (voir `valider_config`).
    """
    if chemin is None:  # aucun chemin donne : on ecrit la configuration retenue par defaut
        chemin = CHEMIN_CONFIG_RETENUE

    config_python = convertir_en_types_python(config)  # copie en types Python simples
    valider_config(config_python)  # on valide AVANT d'ecrire : un fichier invalide n'est jamais cree

    # creation du dossier parent s'il n'existe pas encore (ex. docs/resultats/)
    dossier_parent = os.path.dirname(os.path.abspath(chemin))
    os.makedirs(dossier_parent, exist_ok=True)

    # ecriture : indentation de 2 espaces (lisible dans une revue de code), accents gardes tels quels
    with open(chemin, "w", encoding="utf-8") as fichier:
        json.dump(config_python, fichier, indent=2, ensure_ascii=False)
        fichier.write("\n")  # fin de ligne finale, comme tout fichier texte du depot
    return chemin


def construire_fabrique_depuis_config(config):
    """
    Construit, a partir d'une configuration, une fabrique de modeles : une fonction SANS argument
    qui renvoie a chaque appel un modele NEUF, non entraine, correspondant a la configuration.

    C'est exactement ce qu'attend `evaluer_en_validation_croisee` (src/evaluation.py), qui appelle la
    fabrique une fois par pli, et c'est aussi ce qu'utilise la demo pour l'entrainement final.
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
    Resume une configuration en une ligne lisible, pour l'affichage (demo, notebook, logs).

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
