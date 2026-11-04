"""
Fabriques de modeles (Phase 4) et pertes d'apprentissage sur mesure pour la prediction de RUL.

Une "fabrique" est une fonction qui renvoie un modele NEUF, non entraine, avec des reglages connus.
La validation croisee (src/evaluation.py) en appelle une a chaque pli : chaque pli part donc d'un
modele vierge, sans aucune memoire du pli precedent.

Le module contient aussi deux pertes d'apprentissage qui penalisent davantage les SURESTIMATIONS
(le cas dangereux : le modele croit qu'il reste plus de cycles qu'en realite) :

1. une perte quadratique asymetrique : L = w * (p - y)^2, avec w = alpha si p > y, 1 sinon ;
2. une perte de forme NASA : L = exp(-d / h_sous) - 1 si d < 0, exp(d / h_sur) - 1 sinon (d = p - y).

Ces pertes servent pendant l'APPRENTISSAGE (elles changent les predictions). La NOTE d'evaluation,
elle, reste toujours le score NASA officiel 13/10 (src/evaluation.py).
"""

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBRegressor


# Reglages XGBoost de la Phase 3 (notebook 03), reproduits a l'identique
PARAMETRES_XGBOOST_PHASE3 = {
    "n_estimators": 200,   # nombre d'arbres
    "max_depth": 5,        # profondeur maximale de chaque arbre
    "learning_rate": 0.05, # pas d'apprentissage (part de chaque nouvel arbre dans la prediction)
    "random_state": 42,    # graine, pour la reproductibilite
}

# Reglages LightGBM "comparables" : meme nombre d'arbres et meme pas que XGBoost Phase 3,
# le reste aux valeurs par defaut de LightGBM (31 feuilles par arbre, profondeur non bornee)
PARAMETRES_LIGHTGBM_DEFAUT = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "random_state": 42,
    "verbose": -1,           # pas de messages d'information a chaque entrainement
    "deterministic": True,   # demande a LightGBM des resultats identiques d'une execution a l'autre
    "force_row_wise": True,  # recommande par la doc LightGBM avec deterministic=True (evite une
                             # instabilite numerique liee au choix automatique row-wise / col-wise)
}

# Reglages Random Forest : 200 arbres, tout le reste aux valeurs par defaut de scikit-learn
PARAMETRES_RANDOM_FOREST_DEFAUT = {
    "n_estimators": 200,
    "random_state": 42,
    "n_jobs": -1,  # utilise tous les coeurs du processeur (n'influence pas le resultat, seulement la vitesse)
}

# Seuil (en cycles) au-dela duquel la perte NASA d'apprentissage est prolongee de facon quadratique
D_MAX_DEFAUT = 50.0


def fusionner_parametres(parametres_defaut, parametres_utilisateur):
    """
    Construit un dictionnaire de reglages : les valeurs par defaut, ecrasees par celles de l'utilisateur.

    Parameters
    ----------
    parametres_defaut : dict
        Les reglages par defaut.
    parametres_utilisateur : dict
        Les reglages passes par l'appelant (prioritaires).

    Returns
    -------
    dict
        Un nouveau dictionnaire (les deux dictionnaires d'origine ne sont pas modifies).
    """
    parametres = {}  # nouveau dictionnaire, rempli en deux temps
    for cle in parametres_defaut:  # 1) on recopie les valeurs par defaut
        parametres[cle] = parametres_defaut[cle]
    for cle in parametres_utilisateur:  # 2) l'utilisateur ecrase ce qu'il veut changer
        parametres[cle] = parametres_utilisateur[cle]
    return parametres


# ---------------------------------------------------------------------------------------------------
# Fabriques des modeles "classiques"
# ---------------------------------------------------------------------------------------------------

def fabrique_xgboost(**params):
    """
    Renvoie un XGBRegressor neuf, par defaut avec exactement les reglages de la Phase 3.

    Parameters
    ----------
    **params
        Reglages XGBoost a changer, par exemple max_depth=6 (les autres gardent la valeur Phase 3 :
        n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42).

    Returns
    -------
    xgboost.XGBRegressor
        Le modele, non entraine.
    """
    parametres = fusionner_parametres(PARAMETRES_XGBOOST_PHASE3, params)
    return XGBRegressor(**parametres)


def fabrique_lightgbm(**params):
    """
    Renvoie un LGBMRegressor neuf, avec des reglages comparables a XGBoost Phase 3.

    Par defaut : n_estimators=200, learning_rate=0.05, random_state=42, verbose=-1, et deux reglages
    de reproductibilite : deterministic=True (resultats identiques d'une execution a l'autre avec les
    memes donnees et les memes reglages, sur la meme machine) et force_row_wise=True (recommande par la
    documentation LightGBM avec deterministic=True). Coute un peu de vitesse, gagne la reproductibilite.

    Parameters
    ----------
    **params
        Reglages LightGBM a changer (prioritaires sur les valeurs par defaut ci-dessus).

    Returns
    -------
    lightgbm.LGBMRegressor
        Le modele, non entraine.
    """
    parametres = fusionner_parametres(PARAMETRES_LIGHTGBM_DEFAUT, params)
    return LGBMRegressor(**parametres)


def fabrique_random_forest(**params):
    """
    Renvoie un RandomForestRegressor neuf : 200 arbres, random_state=42, n_jobs=-1.

    Tous les autres reglages sont ceux par defaut de scikit-learn (arbres de profondeur non bornee).

    Parameters
    ----------
    **params
        Reglages scikit-learn a changer (prioritaires sur les valeurs par defaut ci-dessus).

    Returns
    -------
    sklearn.ensemble.RandomForestRegressor
        Le modele, non entraine.
    """
    parametres = fusionner_parametres(PARAMETRES_RANDOM_FOREST_DEFAUT, params)
    return RandomForestRegressor(**parametres)


def fabrique_reference_naive():
    """
    Renvoie un modele "naif" qui predit toujours la moyenne de la cible d'entrainement.

    C'est la borne basse : un modele qui ne bat pas cette reference n'a rien appris des capteurs.

    Returns
    -------
    sklearn.dummy.DummyRegressor
        Le modele, non entraine (strategy="mean").
    """
    return DummyRegressor(strategy="mean")


# ---------------------------------------------------------------------------------------------------
# Enveloppe : partir de la moyenne de y avec une perte sur mesure
# ---------------------------------------------------------------------------------------------------

class RegresseurCentre(BaseEstimator, RegressorMixin):
    """
    Enveloppe un modele de boosting pour qu'il parte de la moyenne de y, meme avec une perte sur mesure.

    Le piege : avec la perte native (reg:squarederror), XGBoost et LightGBM commencent par predire la
    moyenne de y, puis chaque arbre corrige l'erreur restante. Avec une perte sur mesure (callable),
    ils ne savent plus calculer ce point de depart : XGBoost part de 0,5 (base_score par defaut) et
    LightGBM part de 0 (et `predict` n'ajoute aucun point de depart). Les premiers arbres gaspillent
    alors leur capacite a "monter" de 0 a ~86 cycles.

    La solution : au fit, on retire la moyenne de y (la cible centree vaut 0 en moyenne) et on
    entraine le modele sur cette cible centree, en partant de 0 ; au predict, on rajoute la moyenne.
    C'est exactement le point de depart de la perte native.

    Parameters
    ----------
    modele : estimateur scikit-learn (XGBRegressor, LGBMRegressor, ...)
        Le modele a envelopper. Il est recopie (clone) a chaque fit : l'objet passe n'est jamais modifie.

    Attributes
    ----------
    moyenne_y_ : float
        La moyenne de y apprise au fit, rajoutee a chaque prediction.
    modele_ : estimateur
        La copie du modele, entrainee sur y - moyenne_y_.
    """

    def __init__(self, modele=None):
        """
        Enregistre le modele a envelopper (convention scikit-learn : aucun calcul dans __init__).

        Parameters
        ----------
        modele : estimateur scikit-learn
            Le modele a envelopper.
        """
        self.modele = modele

    def fit(self, X, y):
        """
        Entraine une copie du modele sur la cible centree y - moyenne(y).

        Parameters
        ----------
        X : pandas.DataFrame ou numpy.ndarray
            Les features.
        y : array-like
            La cible (RUL plafonnee).

        Returns
        -------
        RegresseurCentre
            L'objet lui-meme, entraine (convention scikit-learn).
        """
        y_reel = np.asarray(y, dtype=float)  # la cible en tableau de reels
        self.moyenne_y_ = float(np.mean(y_reel))  # point de depart : la moyenne de y
        y_centre = y_reel - self.moyenne_y_  # cible centree : vaut 0 en moyenne

        self.modele_ = clone(self.modele)  # copie neuve du modele, l'original reste intact
        self.modele_.fit(X, y_centre)  # le modele apprend les ecarts a la moyenne
        return self

    def predict(self, X):
        """
        Predit la RUL : prediction du modele sur la cible centree + moyenne de y.

        Parameters
        ----------
        X : pandas.DataFrame ou numpy.ndarray
            Les features.

        Returns
        -------
        numpy.ndarray
            Les predictions, dans l'unite d'origine (cycles).
        """
        check_is_fitted(self)  # erreur claire si on predit avant d'avoir entraine
        predictions_centrees = np.asarray(self.modele_.predict(X), dtype=float)
        return predictions_centrees + self.moyenne_y_  # on rajoute le point de depart


# ---------------------------------------------------------------------------------------------------
# Perte 1 : quadratique asymetrique  L = w * (p - y)^2,  w = alpha si p > y, 1 sinon
# ---------------------------------------------------------------------------------------------------

def perte_quadratique_asymetrique(y_true, y_pred, alpha):
    """
    Calcule la perte quadratique asymetrique de chaque exemple : L = w * (p - y)^2.

    w = alpha si p > y (surestimation, le cas dangereux), w = 1 sinon. alpha = 1 redonne la perte
    quadratique classique ; alpha = 3 rend une surestimation 3 fois plus couteuse qu'une
    sous-estimation de meme taille.

    Parameters
    ----------
    y_true : array-like
        Les vraies valeurs y.
    y_pred : array-like
        Les predictions p.
    alpha : float
        Poids des surestimations (alpha >= 1 pour penaliser davantage les surestimations).

    Returns
    -------
    numpy.ndarray
        La perte de chaque exemple.
    """
    vrais = np.asarray(y_true, dtype=float)
    predits = np.asarray(y_pred, dtype=float)
    residus = predits - vrais  # positif = surestimation

    poids = np.ones_like(residus)  # poids 1 par defaut (sous-estimation ou prediction exacte)
    surestimation = residus > 0  # Vrai la ou le modele surestime
    poids[surestimation] = alpha  # poids alpha sur les surestimations

    return poids * residus ** 2


def gradient_hessienne_quadratique_asymetrique(y_true, y_pred, alpha):
    """
    Calcule le gradient et la hessienne de la perte quadratique asymetrique, par rapport a p.

    L = w * (p - y)^2  =>  dL/dp = 2 * w * (p - y)  et  d2L/dp2 = 2 * w.
    Le boosting a besoin de ces deux quantites pour construire chaque arbre (methode de Newton).

    Parameters
    ----------
    y_true : array-like
        Les vraies valeurs y.
    y_pred : array-like
        Les predictions courantes p.
    alpha : float
        Poids des surestimations.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        Le gradient et la hessienne de chaque exemple.
    """
    vrais = np.asarray(y_true, dtype=float)
    predits = np.asarray(y_pred, dtype=float)
    residus = predits - vrais  # positif = surestimation

    poids = np.ones_like(residus)  # w = 1 par defaut
    surestimation = residus > 0
    poids[surestimation] = alpha  # w = alpha la ou p > y

    gradient = 2.0 * poids * residus  # derivee premiere
    hessienne = 2.0 * poids  # derivee seconde (constante par morceaux)
    return gradient, hessienne


def construire_objectif_asymetrique(alpha):
    """
    Construit l'objectif sur mesure (perte quadratique asymetrique) pour XGBRegressor / LGBMRegressor.

    Les deux bibliotheques acceptent comme `objective` une fonction f(y_true, y_pred) qui renvoie
    (gradient, hessienne). On fabrique ici cette fonction avec alpha "fixe a l'interieur".

    Note : avec alpha = 1, le gradient vaut 2 * (p - y) et la hessienne 2, soit le double de la perte
    native de XGBoost (gradient p - y, hessienne 1). Les arbres sont quasiment les memes ; seule la
    regularisation L2 de XGBoost (reg_lambda, 1 par defaut) pese relativement deux fois moins.
    LightGBM (lambda_l2 = 0 par defaut) n'est pas concerne.

    Parameters
    ----------
    alpha : float
        Poids des surestimations (1 = perte quadratique classique).

    Returns
    -------
    callable
        objectif(y_true, y_pred) -> (gradient, hessienne).
    """
    def objectif(y_true, y_pred):
        """
        Objectif sur mesure : gradient et hessienne de la perte quadratique asymetrique.

        Parameters
        ----------
        y_true : numpy.ndarray
            Les vraies valeurs (fournies par XGBoost / LightGBM).
        y_pred : numpy.ndarray
            Les predictions courantes (fournies par XGBoost / LightGBM).

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
            Le gradient et la hessienne de chaque exemple.
        """
        return gradient_hessienne_quadratique_asymetrique(y_true, y_pred, alpha)

    return objectif


# ---------------------------------------------------------------------------------------------------
# Perte 2 : forme NASA  L = exp(-d/h_sous) - 1 si d < 0, exp(d/h_sur) - 1 sinon  (d = p - y)
# ---------------------------------------------------------------------------------------------------

def derivees_nasa_au_point(d, h_sous, h_sur):
    """
    Calcule la perte NASA et ses deux derivees en des ecarts d SANS prolongement (formule exacte).

    Avec d = p - y :
    - d < 0  : L = exp(-d/h_sous) - 1,  L' = -exp(-d/h_sous) / h_sous,  L'' = exp(-d/h_sous) / h_sous^2 ;
    - d >= 0 : L = exp(d/h_sur) - 1,    L' = exp(d/h_sur) / h_sur,      L'' = exp(d/h_sur) / h_sur^2.
    Les derivees sont prises par rapport a p (comme d = p - y, dL/dp = dL/dd).
    A n'utiliser que sur des ecarts bornes (voir `perte_et_derivees_nasa`), sinon exp() explose.

    Parameters
    ----------
    d : numpy.ndarray
        Les ecarts p - y (supposes bornes).
    h_sous : float
        Denominateur des sous-estimations (13 dans le score officiel).
    h_sur : float
        Denominateur des surestimations (10 dans le score officiel).

    Returns
    -------
    (numpy.ndarray, numpy.ndarray, numpy.ndarray)
        La perte L, la derivee premiere L' et la derivee seconde L'' en chaque ecart.
    """
    perte = np.zeros_like(d)
    derivee_1 = np.zeros_like(d)
    derivee_2 = np.zeros_like(d)

    # branche "sous-estimation" (d < 0) : penalite plus douce
    sous = d < 0
    exponentielle_sous = np.exp(-d[sous] / h_sous)
    perte[sous] = exponentielle_sous - 1.0
    derivee_1[sous] = -exponentielle_sous / h_sous
    derivee_2[sous] = exponentielle_sous / h_sous ** 2

    # branche "surestimation" (d >= 0) : penalite plus severe si h_sur < h_sous
    sur = d >= 0
    exponentielle_sur = np.exp(d[sur] / h_sur)
    perte[sur] = exponentielle_sur - 1.0
    derivee_1[sur] = exponentielle_sur / h_sur
    derivee_2[sur] = exponentielle_sur / h_sur ** 2

    return perte, derivee_1, derivee_2


def perte_et_derivees_nasa(y_true, y_pred, h_sous, h_sur, d_max=D_MAX_DEFAUT):
    """
    Calcule la perte de forme NASA, son gradient et sa hessienne, avec un prolongement stable.

    Probleme de stabilite : en debut d'apprentissage, les ecarts peuvent atteindre +-100 cycles, et
    l'exponentielle grandit tres vite (exp(100/5) ~ 5e8). Choix retenu : la perte est EXACTEMENT la
    forme NASA tant que |d| <= d_max ; au-dela, on la prolonge par sa parabole tangente au point
    d_max (developpement de Taylor d'ordre 2) :

        L(d) = L(b) + L'(b) * (d - b) + 0.5 * L''(b) * (d - b)^2,   avec b = +d_max ou -d_max.

    Ainsi la perte, son gradient et sa hessienne restent continus au raccord, le gradient ne croit
    plus que lineairement et la hessienne reste fixee a L''(b) : plus aucune exponentielle d'un
    grand nombre n'est calculee. La perte reste convexe et minimale en d = 0. Le choix d_max = 50
    cycles (hypothese de travail) laisse la forme NASA intacte sur les ecarts courants une fois le
    modele entraine (RMSE ~ 17 cycles) et ne change que les ecarts extremes du debut d'apprentissage.

    Parameters
    ----------
    y_true : array-like
        Les vraies valeurs y.
    y_pred : array-like
        Les predictions courantes p.
    h_sous : float
        Denominateur des sous-estimations (plus il est petit, plus la sous-estimation coute cher).
    h_sur : float
        Denominateur des surestimations (plus il est petit, plus la surestimation coute cher).
    d_max : float, optionnel
        Seuil (en cycles) au-dela duquel la perte devient quadratique (50 par defaut).

    Returns
    -------
    (numpy.ndarray, numpy.ndarray, numpy.ndarray)
        La perte, le gradient (dL/dp) et la hessienne (d2L/dp2) de chaque exemple.
    """
    vrais = np.asarray(y_true, dtype=float)
    predits = np.asarray(y_pred, dtype=float)
    d = predits - vrais  # positif = surestimation, negatif = sous-estimation

    # point de raccord b : d lui-meme s'il est dans [-d_max, d_max], sinon la borne la plus proche
    d_borne = np.clip(d, -d_max, d_max)
    exces = d - d_borne  # 0 dans la zone exacte, sinon de combien on depasse le seuil

    # perte et derivees exactes au point de raccord (jamais d'exponentielle d'un ecart > d_max)
    perte_b, derivee_1_b, derivee_2_b = derivees_nasa_au_point(d_borne, h_sous, h_sur)

    # prolongement de Taylor d'ordre 2 (quand exces = 0, on retrouve exactement la forme NASA)
    perte = perte_b + derivee_1_b * exces + 0.5 * derivee_2_b * exces ** 2
    gradient = derivee_1_b + derivee_2_b * exces  # derivee de la parabole
    hessienne = derivee_2_b  # derivee seconde de la parabole : constante = L''(b)
    return perte, gradient, hessienne


def construire_objectif_nasa(h_sous, h_sur, d_max=D_MAX_DEFAUT):
    """
    Construit l'objectif sur mesure "forme NASA" pour XGBRegressor / LGBMRegressor.

    C'est la bonne place pour l'idee "faire varier 13 et 10" : ces denominateurs changent la perte
    d'APPRENTISSAGE (donc les predictions), alors que la note d'evaluation reste le score NASA 13/10.

    Note sur la vitesse d'apprentissage : pour une feuille d'arbre dont tous les exemples sont du
    meme cote (par exemple tous surestimes), le pas de Newton -somme(L') / somme(L'') vaut exactement
    -h_sur (ou +h_sous), quelle que soit la taille des erreurs. Chaque arbre ne deplace donc les
    predictions que d'environ learning_rate * h cycles, alors qu'avec la perte quadratique le pas est
    proportionnel a l'erreur. Avec un petit h (ex. h_sur = 5) et les reglages Phase 3 (200 arbres,
    learning_rate 0.05), l'apprentissage peut ne pas avoir fini de converger : n_estimators et
    learning_rate doivent etre regles EN MEME TEMPS que h_sous / h_sur. Pour XGBoost, les hessiennes
    sont petites (1/h^2 pres de d = 0) : min_child_weight (1 par defaut, compare a une somme de
    hessiennes) et reg_lambda pesent donc plus lourd qu'avec la perte quadratique.

    Parameters
    ----------
    h_sous : float
        Denominateur des sous-estimations.
    h_sur : float
        Denominateur des surestimations.
    d_max : float, optionnel
        Seuil du prolongement quadratique (voir `perte_et_derivees_nasa`).

    Returns
    -------
    callable
        objectif(y_true, y_pred) -> (gradient, hessienne).
    """
    def objectif(y_true, y_pred):
        """
        Objectif sur mesure : gradient et hessienne de la perte de forme NASA (prolongee).

        Parameters
        ----------
        y_true : numpy.ndarray
            Les vraies valeurs (fournies par XGBoost / LightGBM).
        y_pred : numpy.ndarray
            Les predictions courantes (fournies par XGBoost / LightGBM).

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
            Le gradient et la hessienne de chaque exemple.
        """
        perte, gradient, hessienne = perte_et_derivees_nasa(y_true, y_pred, h_sous, h_sur, d_max=d_max)
        return gradient, hessienne  # le boosting n'a besoin que des derivees, pas de la perte elle-meme

    return objectif


# ---------------------------------------------------------------------------------------------------
# Fabriques des modeles a perte sur mesure
# ---------------------------------------------------------------------------------------------------

def construire_xgboost_sur_mesure(objectif, params):
    """
    Construit un XGBoost (reglages Phase 3 par defaut) avec un objectif sur mesure, enveloppe pour
    partir de la moyenne de y.

    base_score (point de depart d'XGBoost) est mis a 0 : la cible vue par XGBoost est centree
    (moyenne 0) par `RegresseurCentre`, donc 0 correspond bien a la moyenne de y. Sans cela, XGBoost
    partirait de 0,5 avec un objectif sur mesure.

    Parameters
    ----------
    objectif : callable
        objectif(y_true, y_pred) -> (gradient, hessienne).
    params : dict
        Reglages XGBoost a changer (prioritaires ; "objective" est toujours remplace par `objectif`).

    Returns
    -------
    RegresseurCentre
        Le modele enveloppe, non entraine.
    """
    parametres = fusionner_parametres(PARAMETRES_XGBOOST_PHASE3, params)
    if "base_score" not in params:  # sauf si l'utilisateur impose son propre point de depart
        parametres["base_score"] = 0.0
    parametres["objective"] = objectif
    return RegresseurCentre(XGBRegressor(**parametres))


def construire_lightgbm_sur_mesure(objectif, params):
    """
    Construit un LightGBM (reglages par defaut de `fabrique_lightgbm`) avec un objectif sur mesure,
    enveloppe pour partir de la moyenne de y (avec un objectif sur mesure, LightGBM part de 0).

    Parameters
    ----------
    objectif : callable
        objectif(y_true, y_pred) -> (gradient, hessienne).
    params : dict
        Reglages LightGBM a changer (prioritaires ; "objective" est toujours remplace par `objectif`).

    Returns
    -------
    RegresseurCentre
        Le modele enveloppe, non entraine.
    """
    parametres = fusionner_parametres(PARAMETRES_LIGHTGBM_DEFAUT, params)
    parametres["objective"] = objectif
    return RegresseurCentre(LGBMRegressor(**parametres))


def fabrique_xgboost_asymetrique(alpha, **params):
    """
    Renvoie un XGBoost neuf entraine avec la perte quadratique asymetrique (poids alpha sur les surestimations).

    Parameters
    ----------
    alpha : float
        Poids des surestimations (1 = perte quadratique classique, 2 ou 3 = surestimation 2 ou 3 fois
        plus couteuse).
    **params
        Reglages XGBoost a changer (Phase 3 par defaut).

    Returns
    -------
    RegresseurCentre
        Le modele enveloppe (qui part de la moyenne de y), non entraine.
    """
    objectif = construire_objectif_asymetrique(alpha)
    return construire_xgboost_sur_mesure(objectif, params)


def fabrique_lightgbm_asymetrique(alpha, **params):
    """
    Renvoie un LightGBM neuf entraine avec la perte quadratique asymetrique (poids alpha sur les surestimations).

    Parameters
    ----------
    alpha : float
        Poids des surestimations (1 = perte quadratique classique).
    **params
        Reglages LightGBM a changer (valeurs de `fabrique_lightgbm` par defaut).

    Returns
    -------
    RegresseurCentre
        Le modele enveloppe (qui part de la moyenne de y), non entraine.
    """
    objectif = construire_objectif_asymetrique(alpha)
    return construire_lightgbm_sur_mesure(objectif, params)


def fabrique_xgboost_nasa(h_sous, h_sur, d_max=D_MAX_DEFAUT, **params):
    """
    Renvoie un XGBoost neuf entraine avec la perte de forme NASA (denominateurs h_sous, h_sur).

    Attention : avec un petit h, regler aussi n_estimators / learning_rate (voir la note de
    `construire_objectif_nasa` : chaque arbre deplace les predictions d'environ learning_rate * h).

    Parameters
    ----------
    h_sous : float
        Denominateur des sous-estimations (13 = forme du score officiel).
    h_sur : float
        Denominateur des surestimations (10 = forme du score officiel ; plus petit = plus prudent).
    d_max : float, optionnel
        Seuil du prolongement quadratique (50 cycles par defaut).
    **params
        Reglages XGBoost a changer (Phase 3 par defaut).

    Returns
    -------
    RegresseurCentre
        Le modele enveloppe (qui part de la moyenne de y), non entraine.
    """
    objectif = construire_objectif_nasa(h_sous, h_sur, d_max=d_max)
    return construire_xgboost_sur_mesure(objectif, params)


def fabrique_lightgbm_nasa(h_sous, h_sur, d_max=D_MAX_DEFAUT, **params):
    """
    Renvoie un LightGBM neuf entraine avec la perte de forme NASA (denominateurs h_sous, h_sur).

    Attention : avec un petit h, regler aussi n_estimators / learning_rate (voir la note de
    `construire_objectif_nasa` : chaque arbre deplace les predictions d'environ learning_rate * h).

    Parameters
    ----------
    h_sous : float
        Denominateur des sous-estimations (13 = forme du score officiel).
    h_sur : float
        Denominateur des surestimations (10 = forme du score officiel ; plus petit = plus prudent).
    d_max : float, optionnel
        Seuil du prolongement quadratique (50 cycles par defaut).
    **params
        Reglages LightGBM a changer (valeurs de `fabrique_lightgbm` par defaut).

    Returns
    -------
    RegresseurCentre
        Le modele enveloppe (qui part de la moyenne de y), non entraine.
    """
    objectif = construire_objectif_nasa(h_sous, h_sur, d_max=d_max)
    return construire_lightgbm_sur_mesure(objectif, params)
