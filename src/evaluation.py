"""
Metriques et protocoles d'evaluation (Phase 4) pour la prediction de RUL sur FD001.

On y trouve :
- les metriques : RMSE, MAE, score NASA, nombre de surestimations, alertes manquees / fausses alertes ;
- le jeu complet de metriques du protocole (docs/protocole_evaluation.md, sections 3, 4 et 8) : metriques de
  regression par pli, erreur par tranche de RUL, lecture en classification "panne dans les N prochains
  cycles" (seuil choisi sur la validation) et comparaisons appariees de plusieurs modeles (Bonferroni) ;
- la validation croisee par moteur (GroupKFold), avec verification explicite des plis ;
- l'evaluation sur le jeu de test officiel (dernier cycle de chaque moteur test) ;
- l'analyse par moteur et la comparaison statistique appariee de deux modeles ;
- une demonstration (pour la soutenance) que les constantes 13/10 du SCORE NASA ne sont pas des
  reglages a optimiser.

Les calculs reprennent exactement ceux du notebook 03 (RMSE "a la main", score NASA en boucle explicite,
GroupKFold(n_splits=5, shuffle=True, random_state=42)), pour que les chiffres de la Phase 3 soient reproduits.
"""

import itertools
import time

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold


# ---------------------------------------------------------------------------------------------------
# Metriques
# ---------------------------------------------------------------------------------------------------

def rmse(y_true, y_pred):
    """
    Calcule le RMSE (racine de l'erreur quadratique moyenne), meme calcul a la main que le notebook 03.

    Parameters
    ----------
    y_true : array-like
        Les vraies valeurs de RUL.
    y_pred : array-like
        Les valeurs predites par le modele.

    Returns
    -------
    float
        sqrt(moyenne((y_true - y_pred)^2)), dans la meme unite que la RUL (des cycles).
    """
    vrais = np.asarray(y_true, dtype=float)  # conversion en tableau numpy de reels (accepte listes et Series)
    predits = np.asarray(y_pred, dtype=float)
    ecarts = vrais - predits  # ecart de chaque exemple
    return float(np.sqrt(np.mean(ecarts ** 2)))  # racine de la moyenne des ecarts au carre


def mae(y_true, y_pred):
    """
    Calcule la MAE (erreur absolue moyenne) : en moyenne, de combien de cycles se trompe-t-on ?

    Moins sensible que le RMSE aux grosses erreurs isolees (pas d'elevation au carre).

    Parameters
    ----------
    y_true : array-like
        Les vraies valeurs de RUL.
    y_pred : array-like
        Les valeurs predites par le modele.

    Returns
    -------
    float
        moyenne(|y_true - y_pred|), en cycles.
    """
    vrais = np.asarray(y_true, dtype=float)
    predits = np.asarray(y_pred, dtype=float)
    ecarts_absolus = np.abs(vrais - predits)  # valeur absolue de chaque ecart
    return float(np.mean(ecarts_absolus))


def score_nasa(y_true, y_pred, a_sous=13, a_sur=10):
    """
    Calcule le score NASA (Saxena et al., 2008), asymetrique : penalise plus fortement
    une surestimation de la RUL (dangereux) qu'une sous-estimation (prudent).

    Avec d = RUL_predite - RUL_vraie :
    - d < 0 (sous-estimation) : penalite exp(-d / a_sous) - 1 ;
    - d >= 0 (surestimation)  : penalite exp(d / a_sur) - 1.
    Les constantes 13 et 10 sont celles de la litterature : elles definissent la NOTE, ce ne sont
    pas des reglages a optimiser (voir `demontrer_constantes_score_degenerees`).

    Parameters
    ----------
    y_true : array-like
        Les vraies valeurs de RUL.
    y_pred : array-like
        Les valeurs predites par le modele.
    a_sous : float, optionnel
        Denominateur pour les sous-estimations (13 dans la litterature).
    a_sur : float, optionnel
        Denominateur pour les surestimations (10 dans la litterature).

    Returns
    -------
    float
        La somme des penalites individuelles sur tous les exemples (0 = parfait).
    """
    vrais = np.asarray(y_true, dtype=float)
    predits = np.asarray(y_pred, dtype=float)

    total = 0.0  # on cumule la penalite de chaque exemple

    for vrai, predit in zip(vrais, predits):  # boucle explicite, un exemple a la fois (comme le notebook 03)
        d = predit - vrai  # positif si on surestime, negatif si on sous-estime

        if d < 0:  # sous-estimation : penalite plus douce
            total += np.exp(-d / a_sous) - 1
        else:  # surestimation : penalite plus severe
            total += np.exp(d / a_sur) - 1

    return float(total)


def nb_surestimes(y_true, y_pred, seuil=20):
    """
    Compte les exemples surestimes de plus de `seuil` cycles (le cas dangereux).

    Le seuil de 20 cycles est un choix d'interpretation de la Phase 3, pas une valeur du cahier des charges.

    Parameters
    ----------
    y_true : array-like
        Les vraies valeurs de RUL.
    y_pred : array-like
        Les valeurs predites par le modele.
    seuil : float, optionnel
        Ecart (prediction - verite) au-dela duquel on compte une surestimation (20 par defaut).

    Returns
    -------
    int
        Le nombre d'exemples pour lesquels prediction - verite > seuil.
    """
    vrais = np.asarray(y_true, dtype=float)
    predits = np.asarray(y_pred, dtype=float)
    ecarts = predits - vrais  # positif = surestimation (dangereux)
    return int(np.sum(ecarts > seuil))  # Vrai compte 1, Faux compte 0


def alertes_manquees(y_true, y_pred, seuil_alerte=30):
    """
    Compte les alertes manquees : la vraie RUL est <= seuil_alerte, mais la prediction est > seuil_alerte.

    Lecture metier : on suppose qu'un moteur est envoye en maintenance des que sa RUL predite passe
    sous `seuil_alerte` cycles. Une alerte manquee est un moteur qui aurait du etre arrete (il lui
    reste peu de cycles) mais que le modele laisse voler : c'est le cas DANGEREUX.
    Le seuil de 30 cycles est une HYPOTHESE DE TRAVAIL de l'equipe (aucune valeur officielle dans
    le cahier des charges ni dans Saxena et al. 2008) : a faire varier si on discute du seuil.

    Parameters
    ----------
    y_true : array-like
        Les vraies valeurs de RUL.
    y_pred : array-like
        Les valeurs predites par le modele.
    seuil_alerte : float, optionnel
        Seuil d'intervention en cycles (30 par defaut, hypothese de travail).

    Returns
    -------
    int
        Le nombre de cas "vraie RUL <= seuil ET prediction > seuil".
    """
    vrais = np.asarray(y_true, dtype=float)
    predits = np.asarray(y_pred, dtype=float)
    aurait_du_alerter = vrais <= seuil_alerte  # le moteur est vraiment proche de la panne
    n_alerte_pas = predits > seuil_alerte  # ... mais le modele ne declenche pas l'alerte
    cas_manques = aurait_du_alerter & n_alerte_pas  # les deux a la fois
    return int(np.sum(cas_manques))


def fausses_alertes(y_true, y_pred, seuil_alerte=30):
    """
    Compte les fausses alertes : la vraie RUL est > seuil_alerte, mais la prediction est <= seuil_alerte.

    Lecture metier : le modele envoie en maintenance un moteur qui avait encore de la marge.
    Cout reel (immobilisation, maintenance inutile) mais pas de risque de securite.
    Meme hypothese de travail que `alertes_manquees` pour le seuil de 30 cycles.

    Parameters
    ----------
    y_true : array-like
        Les vraies valeurs de RUL.
    y_pred : array-like
        Les valeurs predites par le modele.
    seuil_alerte : float, optionnel
        Seuil d'intervention en cycles (30 par defaut, hypothese de travail).

    Returns
    -------
    int
        Le nombre de cas "vraie RUL > seuil ET prediction <= seuil".
    """
    vrais = np.asarray(y_true, dtype=float)
    predits = np.asarray(y_pred, dtype=float)
    avait_de_la_marge = vrais > seuil_alerte  # le moteur n'etait pas encore proche de la panne
    alerte_declenchee = predits <= seuil_alerte  # ... mais le modele declenche l'alerte
    cas_inutiles = avait_de_la_marge & alerte_declenchee
    return int(np.sum(cas_inutiles))


# ---------------------------------------------------------------------------------------------------
# Validation croisee par moteur
# ---------------------------------------------------------------------------------------------------

def selectionner_lignes(donnees, indices):
    """
    Selectionne des lignes par position, que `donnees` soit un objet pandas ou un tableau numpy.

    Parameters
    ----------
    donnees : pandas.DataFrame, pandas.Series ou numpy.ndarray
        Le tableau dont on veut extraire des lignes.
    indices : numpy.ndarray d'entiers
        Les positions des lignes a garder (0 = premiere ligne).

    Returns
    -------
    meme type que `donnees`
        Les lignes selectionnees.
    """
    if hasattr(donnees, "iloc"):  # objet pandas : on selectionne par position avec .iloc
        return donnees.iloc[indices]
    return donnees[indices]  # tableau numpy : l'indexation directe selectionne par position


def calculer_plis(X, y, groups, n_splits=5, random_state=42):
    """
    Calcule les plis de validation croisee par moteur (GroupKFold melange), et les verifie.

    Memes reglages que le notebook 03 : GroupKFold(n_splits=5, shuffle=True, random_state=42).
    On ne fait pas confiance aveuglement au mecanisme interne de scikit-learn : on verifie
    nous-memes (1) qu'aucun moteur n'est a la fois en train et en validation dans un pli,
    (2) que chaque moteur est en validation exactement une fois sur l'ensemble des plis.

    Parameters
    ----------
    X : pandas.DataFrame ou numpy.ndarray
        Les features (seul le nombre de lignes compte pour GroupKFold).
    y : pandas.Series ou numpy.ndarray
        La cible.
    groups : pandas.Series ou numpy.ndarray
        Le numero de moteur de chaque ligne.
    n_splits : int, optionnel
        Nombre de plis (5 par defaut).
    random_state : int, optionnel
        Graine du melange des moteurs entre plis (42 par defaut).

    Returns
    -------
    list de (numpy.ndarray, numpy.ndarray)
        Pour chaque pli, le couple (idx_train, idx_val) de positions de lignes.
    """
    groupes = np.asarray(groups)  # numeros de moteur, ligne par ligne, en tableau numpy

    gkf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    plis = []  # on part d'une liste vide, remplie pli par pli
    nb_passages_en_validation = {}  # pour chaque moteur : combien de fois il est en validation

    for idx_train, idx_val in gkf.split(X, y, groups=groupes):  # un pli a la fois
        moteurs_train = set(groupes[idx_train])  # moteurs utilises pour apprendre dans ce pli
        moteurs_val = set(groupes[idx_val])  # moteurs mis de cote pour la validation dans ce pli

        # verification 1 : aucun moteur des deux cotes
        moteurs_communs = moteurs_train & moteurs_val
        assert len(moteurs_communs) == 0, f"moteur(s) present(s) des deux cotes d'un pli : {sorted(moteurs_communs)}"

        # on compte les passages en validation de chaque moteur
        for moteur_id in moteurs_val:
            if moteur_id not in nb_passages_en_validation:
                nb_passages_en_validation[moteur_id] = 0
            nb_passages_en_validation[moteur_id] = nb_passages_en_validation[moteur_id] + 1

        plis.append((idx_train, idx_val))

    # verification 2 : chaque moteur est en validation exactement une fois
    tous_les_moteurs = np.unique(groupes)
    for moteur_id in tous_les_moteurs:
        nb_passages = nb_passages_en_validation.get(moteur_id, 0)  # 0 si le moteur n'a jamais ete en validation
        assert nb_passages == 1, f"le moteur {moteur_id} est en validation {nb_passages} fois au lieu d'une"

    return plis


def evaluer_en_validation_croisee(fabrique_modele, X, y, groups, plis, seuil_surestimation=20, seuil_alerte=30):
    """
    Evalue un type de modele en validation croisee par moteur, avec un modele NEUF a chaque pli.

    Pour chaque pli : on cree un modele neuf avec `fabrique_modele()`, on l'entraine sur les moteurs
    d'entrainement du pli, on predit les moteurs de validation (jamais vus par ce modele). Chaque
    ligne de X est predite exactement une fois, par le modele du pli ou elle etait en validation :
    ce sont les predictions "hors-pli" (OOF, out-of-fold), utiles pour analyser les erreurs.
    Comme dans le notebook 03, les predictions de validation croisee ne sont PAS clippees a 0.

    Parameters
    ----------
    fabrique_modele : callable
        Fonction SANS argument qui renvoie un modele neuf non entraine (avec .fit et .predict),
        par exemple `fabrique_xgboost` de src/modeles.py. Pour changer un reglage, on ecrit une
        petite fonction `def ma_fabrique(): return fabrique_xgboost(max_depth=6)`.
    X : pandas.DataFrame ou numpy.ndarray
        Les features (une ligne par cycle).
    y : pandas.Series ou numpy.ndarray
        La cible (RUL plafonnee).
    groups : pandas.Series ou numpy.ndarray
        Le numero de moteur de chaque ligne : sert a reverifier, pli par pli, qu'aucun moteur n'est
        des deux cotes (au cas ou les plis passes ne viendraient pas de `calculer_plis`).
    plis : list de (numpy.ndarray, numpy.ndarray)
        Les plis renvoyes par `calculer_plis` (les memes pour tous les modeles compares).
    seuil_surestimation : float, optionnel
        Seuil (en cycles) pour compter les surestimations dangereuses (20 par defaut).
    seuil_alerte : float, optionnel
        Seuil d'intervention pour les alertes manquees / fausses alertes (30 par defaut, hypothese).

    Returns
    -------
    dict
        - "rmse_par_pli", "mae_par_pli", "nasa_par_pli", "surestimes_par_pli" : listes, une valeur par pli ;
        - "rmse_moyen", "rmse_ecart_type", "mae_moyen", "mae_ecart_type", "nasa_moyen", "nasa_ecart_type",
          "surestimes_moyen", "surestimes_ecart_type" : moyenne et ecart-type entre plis (np.std, ddof=0,
          comme le notebook 03) ;
        - "predictions_oof" : numpy.ndarray, une prediction hors-pli par ligne de X, dans le meme ordre ;
        - "rmse_oof", "mae_oof", "nasa_oof", "surestimes_oof", "alertes_manquees_oof", "fausses_alertes_oof",
          "biais_oof" : metriques calculees sur TOUTES les predictions hors-pli d'un coup (biais = moyenne
          de prediction - verite : negatif = le modele sous-estime en moyenne) ;
        - "temps_entrainement_par_pli", "temps_prediction_par_pli" : listes, en secondes ;
        - "temps_entrainement_total", "temps_prediction_total" : en secondes.
    """
    vrais = np.asarray(y, dtype=float)  # la cible en tableau numpy, pour les metriques globales
    groupes = np.asarray(groups)  # numeros de moteur, ligne par ligne

    # tableau des predictions hors-pli, rempli pli par pli (NaN = pas encore predit)
    predictions_oof = np.full(len(vrais), np.nan)

    rmse_par_pli = []
    mae_par_pli = []
    nasa_par_pli = []
    surestimes_par_pli = []
    temps_entrainement_par_pli = []
    temps_prediction_par_pli = []

    for idx_train, idx_val in plis:  # les memes plis pour tous les modeles compares
        # garde-fou : aucun moteur ne doit etre a la fois en entrainement et en validation
        moteurs_communs = set(groupes[idx_train]) & set(groupes[idx_val])
        assert len(moteurs_communs) == 0, f"moteur(s) des deux cotes d'un pli : {sorted(moteurs_communs)}"

        # donnees de ce pli : moteurs d'entrainement d'un cote, moteurs de validation de l'autre
        X_tr = selectionner_lignes(X, idx_train)
        y_tr = selectionner_lignes(y, idx_train)
        X_val = selectionner_lignes(X, idx_val)
        y_val = vrais[idx_val]

        # un modele TOUT NEUF a chaque pli : aucune memoire du pli precedent
        modele = fabrique_modele()

        debut = time.perf_counter()  # chronometre de l'entrainement
        modele.fit(X_tr, y_tr)  # apprend uniquement sur les moteurs d'entrainement de ce pli
        temps_entrainement = time.perf_counter() - debut

        debut = time.perf_counter()  # chronometre de la prediction
        y_pred = np.asarray(modele.predict(X_val))  # predit des moteurs jamais vus par ce modele
        temps_prediction = time.perf_counter() - debut

        predictions_oof[idx_val] = y_pred  # on range ces predictions a la place de leurs lignes

        # metriques de ce pli
        rmse_par_pli.append(rmse(y_val, y_pred))
        mae_par_pli.append(mae(y_val, y_pred))
        nasa_par_pli.append(score_nasa(y_val, y_pred))
        surestimes_par_pli.append(nb_surestimes(y_val, y_pred, seuil=seuil_surestimation))
        temps_entrainement_par_pli.append(temps_entrainement)
        temps_prediction_par_pli.append(temps_prediction)

    # chaque ligne doit avoir ete predite exactement une fois (plis disjoints qui couvrent tout)
    nb_non_predites = int(np.sum(np.isnan(predictions_oof)))
    assert nb_non_predites == 0, f"{nb_non_predites} lignes sans prediction hors-pli : les plis ne couvrent pas tout X"

    # on regroupe les resultats dans un dictionnaire, cle par cle
    resultats = {}
    resultats["rmse_par_pli"] = rmse_par_pli
    resultats["mae_par_pli"] = mae_par_pli
    resultats["nasa_par_pli"] = nasa_par_pli
    resultats["surestimes_par_pli"] = surestimes_par_pli

    # moyennes et ecarts-types entre plis (np.std avec ddof=0, comme dans le notebook 03)
    resultats["rmse_moyen"] = float(np.mean(rmse_par_pli))
    resultats["rmse_ecart_type"] = float(np.std(rmse_par_pli))
    resultats["mae_moyen"] = float(np.mean(mae_par_pli))
    resultats["mae_ecart_type"] = float(np.std(mae_par_pli))
    resultats["nasa_moyen"] = float(np.mean(nasa_par_pli))
    resultats["nasa_ecart_type"] = float(np.std(nasa_par_pli))
    resultats["surestimes_moyen"] = float(np.mean(surestimes_par_pli))
    resultats["surestimes_ecart_type"] = float(np.std(surestimes_par_pli))

    # predictions hors-pli et metriques calculees sur toutes ces predictions d'un coup
    resultats["predictions_oof"] = predictions_oof
    resultats["rmse_oof"] = rmse(vrais, predictions_oof)
    resultats["mae_oof"] = mae(vrais, predictions_oof)
    resultats["nasa_oof"] = score_nasa(vrais, predictions_oof)
    resultats["surestimes_oof"] = nb_surestimes(vrais, predictions_oof, seuil=seuil_surestimation)
    resultats["alertes_manquees_oof"] = alertes_manquees(vrais, predictions_oof, seuil_alerte=seuil_alerte)
    resultats["fausses_alertes_oof"] = fausses_alertes(vrais, predictions_oof, seuil_alerte=seuil_alerte)
    resultats["biais_oof"] = float(np.mean(predictions_oof - vrais))  # negatif = sous-estimation moyenne

    # temps de calcul
    resultats["temps_entrainement_par_pli"] = temps_entrainement_par_pli
    resultats["temps_prediction_par_pli"] = temps_prediction_par_pli
    resultats["temps_entrainement_total"] = float(np.sum(temps_entrainement_par_pli))
    resultats["temps_prediction_total"] = float(np.sum(temps_prediction_par_pli))
    return resultats


# ---------------------------------------------------------------------------------------------------
# Analyse par moteur et comparaison statistique
# ---------------------------------------------------------------------------------------------------

def erreurs_par_moteur(y_true, y_pred, groups):
    """
    Resume les erreurs moteur par moteur : une ligne par moteur.

    Utile pour (1) reperer les moteurs mal predits, (2) obtenir des valeurs APPARIEES par moteur
    (ex. RMSE par moteur) pour comparer statistiquement deux modeles avec `comparer_apparie`.

    Parameters
    ----------
    y_true : array-like
        Les vraies valeurs de RUL (une par ligne).
    y_pred : array-like
        Les predictions (une par ligne, dans le meme ordre), par exemple les predictions hors-pli.
    groups : array-like
        Le numero de moteur de chaque ligne.

    Returns
    -------
    pandas.DataFrame
        Colonnes : "unit_number", "rmse" (RMSE du moteur), "biais" (moyenne de prediction - verite :
        negatif = sous-estimation), "surestimation_max" (plus grande surestimation du moteur, 0 s'il
        n'est jamais surestime), "nb_lignes". Triee par numero de moteur.
    """
    vrais = np.asarray(y_true, dtype=float)
    predits = np.asarray(y_pred, dtype=float)
    groupes = np.asarray(groups)

    lignes = []  # une entree (dictionnaire) par moteur, transformee en tableau a la fin

    for moteur_id in np.unique(groupes):  # np.unique renvoie les moteurs tries
        masque = groupes == moteur_id  # Vrai sur les lignes de ce moteur
        vrais_moteur = vrais[masque]
        predits_moteur = predits[masque]
        ecarts_moteur = predits_moteur - vrais_moteur  # positif = surestimation

        plus_grand_ecart = float(np.max(ecarts_moteur))  # le pire ecart signe du moteur
        surestimation_max = max(plus_grand_ecart, 0.0)  # 0 si le moteur n'est jamais surestime

        ligne = {}
        ligne["unit_number"] = moteur_id
        ligne["rmse"] = rmse(vrais_moteur, predits_moteur)
        ligne["biais"] = float(np.mean(ecarts_moteur))
        ligne["surestimation_max"] = surestimation_max
        ligne["nb_lignes"] = int(np.sum(masque))
        lignes.append(ligne)

    return pd.DataFrame(lignes)


def comparer_apparie(valeurs_a, valeurs_b):
    """
    Compare deux modeles sur des valeurs APPARIEES avec un test t apparie et un test de Wilcoxon signe.

    Pourquoi apparier ? Les deux modeles sont evalues sur les MEMES unites (par exemple les memes
    100 moteurs) : la valeur a[i] et la valeur b[i] concernent le meme moteur i. Certains moteurs sont
    difficiles pour tous les modeles, d'autres faciles : cette difficulte commune "s'annule" quand on
    regarde la difference a[i] - b[i] moteur par moteur. Comparer les differences est donc bien plus
    sensible que comparer deux moyennes independantes, ou la variabilite entre moteurs noierait l'effet.

    - Test t apparie (scipy.stats.ttest_rel) : teste si la MOYENNE des differences a[i] - b[i] est nulle.
      Il suppose que ces differences suivent a peu pres une loi normale (raisonnable avec ~100 moteurs,
      sauf differences tres asymetriques ou quelques valeurs extremes).
    - Test de Wilcoxon signe (scipy.stats.wilcoxon) : version "par rangs", sans hypothese de normalite.
      Il classe les |differences| et teste si les differences positives et negatives s'equilibrent
      (en gros : la MEDIANE des differences est-elle nulle ?). Plus robuste aux valeurs extremes.
    Dans les deux cas, une petite p-valeur (ex. < 0,05) signifie qu'un ecart aussi grand serait peu
    probable si les deux modeles etaient en realite equivalents. Si les deux tests sont d'accord,
    la conclusion est plus solide. Attention : Wilcoxon echoue si toutes les differences sont nulles.

    Parameters
    ----------
    valeurs_a : array-like
        Les valeurs du modele A (ex. RMSE par moteur), une par unite.
    valeurs_b : array-like
        Les valeurs du modele B, sur les memes unites et dans le meme ordre.

    Returns
    -------
    dict
        - "n" : nombre de paires ;
        - "difference_moyenne" : moyenne de a - b (negatif = A a des valeurs plus petites que B) ;
        - "statistique_t", "p_valeur_t" : resultat du test t apparie ;
        - "statistique_wilcoxon", "p_valeur_wilcoxon" : resultat du test de Wilcoxon signe.
    """
    a = np.asarray(valeurs_a, dtype=float)
    b = np.asarray(valeurs_b, dtype=float)
    assert a.shape == b.shape, f"les deux series doivent avoir la meme taille ({a.shape} vs {b.shape})"

    differences = a - b  # difference unite par unite (moteur par moteur)

    resultat_t = stats.ttest_rel(a, b)  # test t apparie
    resultat_wilcoxon = stats.wilcoxon(a, b)  # test de Wilcoxon signe (par rangs)

    resultats = {}
    resultats["n"] = int(len(a))
    resultats["difference_moyenne"] = float(np.mean(differences))
    resultats["statistique_t"] = float(resultat_t.statistic)
    resultats["p_valeur_t"] = float(resultat_t.pvalue)
    resultats["statistique_wilcoxon"] = float(resultat_wilcoxon.statistic)
    resultats["p_valeur_wilcoxon"] = float(resultat_wilcoxon.pvalue)
    return resultats


# ---------------------------------------------------------------------------------------------------
# Jeu de test officiel
# ---------------------------------------------------------------------------------------------------

def evaluer_sur_test(modele, X_test_dernier, y_test_vraie, plafond=125, seuil_surestimation=20, seuil_alerte=30):
    """
    Evalue un modele DEJA entraine sur le jeu de test officiel (dernier cycle de chaque moteur test).

    Comme en Phase 3 : on predit, on clippe a 0 (une RUL negative n'a pas de sens physique), puis on
    compare a la RUL BRUTE de RUL_FD001.txt (non plafonnee). On ajoute un RMSE calcule contre la
    RUL vraie plafonnee (min(RUL, plafond)), cle distincte, pour comparer avec les articles qui
    font ce choix : le modele ne peut de toute facon pas predire au-dela du plafond appris.

    Parameters
    ----------
    modele : objet avec .predict
        Le modele entraine (typiquement sur les 100 moteurs d'entrainement).
    X_test_dernier : pandas.DataFrame ou numpy.ndarray
        Les features du dernier cycle de chaque moteur test (100 lignes).
    y_test_vraie : array-like
        La RUL vraie brute de RUL_FD001.txt, dans le meme ordre.
    plafond : float, optionnel
        Plafond utilise pour "rmse_contre_rul_plafonnee" (125 par defaut, comme la cible d'entrainement).
    seuil_surestimation : float, optionnel
        Seuil pour compter les surestimations dangereuses (20 par defaut).
    seuil_alerte : float, optionnel
        Seuil d'intervention pour les alertes (30 par defaut, hypothese de travail).

    Returns
    -------
    dict
        - "rmse", "mae", "score_nasa", "nb_surestimes", "alertes_manquees", "fausses_alertes" : contre la
          RUL brute de RUL_FD001.txt (protocole Phase 3) ;
        - "rmse_contre_rul_plafonnee" : RMSE contre min(RUL brute, plafond) ;
        - "predictions" : numpy.ndarray des 100 predictions clippees a 0.
    """
    vrais = np.asarray(y_test_vraie, dtype=float)

    y_pred = np.asarray(modele.predict(X_test_dernier))  # une prediction par moteur test
    y_pred = np.clip(y_pred, a_min=0, a_max=None)  # une RUL negative n'a pas de sens physique

    # RUL vraie plafonnee : la meme regle que la cible d'entrainement (Phase 2)
    vrais_plafonnes = np.minimum(vrais, plafond)

    resultats = {}
    resultats["rmse"] = rmse(vrais, y_pred)
    resultats["mae"] = mae(vrais, y_pred)
    resultats["score_nasa"] = score_nasa(vrais, y_pred)
    resultats["nb_surestimes"] = nb_surestimes(vrais, y_pred, seuil=seuil_surestimation)
    resultats["alertes_manquees"] = alertes_manquees(vrais, y_pred, seuil_alerte=seuil_alerte)
    resultats["fausses_alertes"] = fausses_alertes(vrais, y_pred, seuil_alerte=seuil_alerte)
    resultats["rmse_contre_rul_plafonnee"] = rmse(vrais_plafonnes, y_pred)
    resultats["predictions"] = y_pred
    return resultats


# ---------------------------------------------------------------------------------------------------
# Demonstration pour la soutenance : les constantes 13/10 du score ne sont pas des reglages
# ---------------------------------------------------------------------------------------------------

def demontrer_constantes_score_degenerees(y_true, y_pred, facteurs=(1, 2, 5, 10, 100, 1000)):
    """
    Montre que "optimiser" les constantes 13/10 du SCORE NASA serait degenere.

    Idee : le score est calcule APRES coup, a partir de predictions deja faites. Changer ses
    constantes ne change donc AUCUNE prediction (le modele n'est meme pas consulte) : ca change
    seulement la note. Et quand on agrandit les denominateurs (13·k, 10·k), chaque penalite
    exp(|d| / (10·k)) - 1 se rapproche de |d| / (10·k), qui tend vers 0 quand k grandit : le score
    tend vers 0 pour N'IMPORTE QUEL modele, meme mauvais. Chercher les constantes qui minimisent le
    score reviendrait donc a pousser k vers l'infini : la note ne mesurerait plus rien.
    La bonne facon d'utiliser l'idee du coach : mettre ces denominateurs dans la perte
    d'APPRENTISSAGE (voir `construire_objectif_nasa` dans src/modeles.py), la note restant 13/10.

    Parameters
    ----------
    y_true : array-like
        Les vraies valeurs de RUL.
    y_pred : array-like
        Les predictions d'un modele (deja calculees).
    facteurs : tuple de float, optionnel
        Les facteurs k appliques aux deux denominateurs (k = 1 redonne le score officiel 13/10).

    Returns
    -------
    pandas.DataFrame
        Une ligne par facteur : "facteur", "a_sous", "a_sur", "score_nasa", "predictions_inchangees"
        (Vrai si les predictions sont restees identiques bit a bit apres le calcul du score).
    """
    predits = np.asarray(y_pred, dtype=float)
    copie_avant = predits.copy()  # photo des predictions AVANT tout calcul de score

    lignes = []  # une entree par facteur
    for facteur in facteurs:  # un jeu de constantes a la fois
        a_sous = 13 * facteur  # denominateur des sous-estimations, multiplie par k
        a_sur = 10 * facteur  # denominateur des surestimations, multiplie par k
        score = score_nasa(y_true, predits, a_sous=a_sous, a_sur=a_sur)

        ligne = {}
        ligne["facteur"] = facteur
        ligne["a_sous"] = a_sous
        ligne["a_sur"] = a_sur
        ligne["score_nasa"] = score
        # les predictions sont-elles toujours exactement les memes ? (elles doivent l'etre)
        ligne["predictions_inchangees"] = bool(np.array_equal(predits, copie_avant))
        lignes.append(ligne)

    return pd.DataFrame(lignes)


# ---------------------------------------------------------------------------------------------------
# Protocole d'evaluation complet (docs/protocole_evaluation.md, sections 3, 4 et 8)
# ---------------------------------------------------------------------------------------------------

# tranches de RUL vraie de la section 3 du protocole : (etiquette, borne basse EXCLUE, borne haute INCLUSE).
# La derniere tranche (> 125) est toujours vide en validation (cible plafonnee a 125) ; elle ne sert que sur le
# test officiel, ou la RUL vraie n'est pas plafonnee.
TRANCHES_RUL_DEFAUT = (
    ("<= 30", -np.inf, 30.0),
    ("31-60", 30.0, 60.0),
    ("61-125", 60.0, 125.0),
    ("> 125", 125.0, np.inf),
)


def metriques_regression(y_true, y_pred, seuil_surestimation=20):
    """
    Calcule toutes les metriques de regression de la section 3 du protocole sur un jeu de predictions.

    Les metriques viennent de `sklearn.metrics` quand elles y existent (noms du protocole), sinon des
    fonctions de ce module (score NASA, surestimations). Le score NASA est donne en SOMME (definition de
    Saxena et al. 2008) ET en moyenne par prediction, seule forme comparable entre validation (environ
    4 000 lignes par pli) et test (une ligne par moteur). La MAPE est calculee pour etre documentee puis
    ecartee : elle divise par la RUL vraie, qui vaut 0 au dernier cycle de chaque moteur d'entrainement.

    Parameters
    ----------
    y_true : array-like
        Les vraies valeurs de RUL.
    y_pred : array-like
        Les predictions, dans le meme ordre.
    seuil_surestimation : float, optionnel
        Ecart (prediction - verite) au-dela duquel une surestimation est comptee comme dangereuse (20 par defaut).

    Returns
    -------
    dict
        "n", "mae", "mse", "rmse", "r2", "erreur_absolue_mediane", "erreur_max", "biais" (moyenne de
        prediction - verite : positif = surestimation moyenne), "nasa_somme", "nasa_moyen",
        "surestimes" (nombre d'ecarts > seuil), "surestimes_pct" (en % des predictions), "mape" (fraction,
        1.0 = 100 %, calcul de scikit-learn qui remplace une RUL vraie nulle par un epsilon minuscule),
        "nb_rul_vraie_nulle" (nombre de divisions par zero evitees par cet epsilon).
    """
    vrais = np.asarray(y_true, dtype=float)  # verite en tableau de reels
    predits = np.asarray(y_pred, dtype=float)  # predictions en tableau de reels
    assert vrais.shape == predits.shape, f"tailles differentes : {vrais.shape} vs {predits.shape}"

    n = int(len(vrais))  # nombre de predictions evaluees
    nasa = score_nasa(vrais, predits)  # somme des penalites asymetriques
    surestimes = nb_surestimes(vrais, predits, seuil=seuil_surestimation)  # cas dangereux

    resultats = {}
    resultats["n"] = n
    resultats["mae"] = float(mean_absolute_error(vrais, predits))  # erreur moyenne en cycles
    resultats["mse"] = float(mean_squared_error(vrais, predits))  # erreur quadratique moyenne
    resultats["rmse"] = float(np.sqrt(resultats["mse"]))  # meme unite que la RUL
    resultats["r2"] = float(r2_score(vrais, predits))  # 0 = aussi bon que predire la moyenne
    resultats["erreur_absolue_mediane"] = float(median_absolute_error(vrais, predits))  # erreur typique
    resultats["erreur_max"] = float(max_error(vrais, predits))  # pire cas (en valeur absolue)
    resultats["biais"] = float(np.mean(predits - vrais))  # positif = surestimation moyenne (dangereux)
    resultats["nasa_somme"] = nasa
    resultats["nasa_moyen"] = nasa / n  # score par prediction : comparable entre validation et test
    resultats["surestimes"] = surestimes
    resultats["surestimes_pct"] = 100.0 * surestimes / n
    resultats["mape"] = float(mean_absolute_percentage_error(vrais, predits))  # a documenter puis ecarter
    resultats["nb_rul_vraie_nulle"] = int(np.sum(vrais == 0))  # lignes ou la MAPE divise par (presque) zero
    return resultats


def metriques_regression_par_pli(y_true, y_pred, plis, seuil_surestimation=20):
    """
    Calcule `metriques_regression` pli par pli, a partir de predictions hors-pli deja produites.

    Pour chaque pli, on ne garde que les lignes de validation de ce pli : chacune a ete predite par le
    modele du pli, qui ne l'avait jamais vue. Aucun reentrainement n'est fait ici.

    Parameters
    ----------
    y_true : array-like
        La cible de toutes les lignes d'entrainement (RUL plafonnee), dans l'ordre utilise pour les plis.
    y_pred : array-like
        Les predictions hors-pli, une par ligne, dans le meme ordre.
    plis : list de (numpy.ndarray, numpy.ndarray)
        Les plis de `calculer_plis` ; seule la partie validation (idx_val) est utilisee.
    seuil_surestimation : float, optionnel
        Seuil de surestimation dangereuse (20 par defaut).

    Returns
    -------
    pandas.DataFrame
        Une ligne par pli : colonne "pli" (numero a partir de 1), puis les cles de `metriques_regression`.
    """
    vrais = np.asarray(y_true, dtype=float)
    predits = np.asarray(y_pred, dtype=float)

    lignes = []  # une entree par pli
    numero_pli = 0
    for idx_train, idx_val in plis:  # idx_train n'est pas utilise : seules les lignes de validation comptent
        numero_pli = numero_pli + 1
        ligne = {"pli": numero_pli}
        metriques = metriques_regression(vrais[idx_val], predits[idx_val], seuil_surestimation=seuil_surestimation)
        for cle in metriques:  # recopie des metriques du pli, cle par cle
            ligne[cle] = metriques[cle]
        lignes.append(ligne)
    return pd.DataFrame(lignes)


def resumer_plis(tableau_par_pli, colonne_pli="pli"):
    """
    Resume un tableau "une ligne par pli" en moyenne et ecart-type entre plis de chaque colonne numerique.

    L'ecart-type est calcule avec ddof=0 (np.std), comme dans le notebook 03 et `evaluer_en_validation_croisee`.

    Parameters
    ----------
    tableau_par_pli : pandas.DataFrame
        Par exemple la sortie de `metriques_regression_par_pli`.
    colonne_pli : str, optionnel
        Nom de la colonne du numero de pli, exclue du resume ("pli" par defaut).

    Returns
    -------
    dict
        Pour chaque colonne c : "<c>_moyenne" et "<c>_ecart_type".
    """
    resume = {}
    for colonne in tableau_par_pli.columns:  # une colonne (une metrique) a la fois
        if colonne == colonne_pli:  # le numero de pli n'est pas une metrique
            continue
        valeurs = np.asarray(tableau_par_pli[colonne], dtype=float)
        resume[f"{colonne}_moyenne"] = float(np.mean(valeurs))
        resume[f"{colonne}_ecart_type"] = float(np.std(valeurs))  # ddof=0
    return resume


def erreurs_par_tranche_rul(y_true, y_pred, tranches=TRANCHES_RUL_DEFAUT):
    """
    Calcule MAE, RMSE et biais separement par tranche de RUL VRAIE (section 3 du protocole).

    Le RMSE global melange le debut de vie (le plafond 125 rend la tache facile) et la fin de vie, la seule
    zone ou une decision de maintenance se prend : cette decomposition montre ou le modele se trompe.

    Parameters
    ----------
    y_true : array-like
        Les vraies valeurs de RUL (definissent la tranche de chaque ligne).
    y_pred : array-like
        Les predictions, dans le meme ordre.
    tranches : tuple de (str, float, float), optionnel
        (etiquette, borne basse exclue, borne haute incluse) ; par defaut TRANCHES_RUL_DEFAUT
        (<= 30, 31-60, 61-125, > 125).

    Returns
    -------
    pandas.DataFrame
        Une ligne par tranche : "tranche", "n", "mae", "rmse", "biais" (NaN si la tranche est vide).
    """
    vrais = np.asarray(y_true, dtype=float)
    predits = np.asarray(y_pred, dtype=float)

    lignes = []  # une entree par tranche
    for etiquette, borne_basse, borne_haute in tranches:
        masque = (vrais > borne_basse) & (vrais <= borne_haute)  # lignes dont la RUL vraie tombe dans la tranche
        ligne = {"tranche": etiquette, "n": int(np.sum(masque))}
        if ligne["n"] == 0:  # tranche vide (ex. > 125 en validation) : rien a mesurer
            ligne["mae"] = np.nan
            ligne["rmse"] = np.nan
            ligne["biais"] = np.nan
        else:
            ligne["mae"] = mae(vrais[masque], predits[masque])
            ligne["rmse"] = rmse(vrais[masque], predits[masque])
            ligne["biais"] = float(np.mean(predits[masque] - vrais[masque]))  # positif = surestimation
        lignes.append(ligne)
    return pd.DataFrame(lignes)


def metriques_classification_alerte(y_true, y_pred, horizon, seuil_alerte=None):
    """
    Lit une regression de RUL comme une classification "panne dans les `horizon` prochains cycles ?".

    - classe positive (1, "a retirer") : RUL vraie <= horizon ; negative (0) sinon ;
    - decision : alerte si RUL predite <= seuil_alerte (par defaut seuil_alerte = horizon) ;
    - score de classement pour la ROC et la precision moyenne : -RUL predite (plus la RUL predite est basse,
      plus le moteur est juge urgent). Ces deux metriques ne dependent pas du seuil.
    Matrice de confusion avec lignes = vrai, colonnes = predit : vn, fp (retire trop tot : cout de
    maintenance), fn (laisse voler alors qu'il allait tomber en panne : le cas grave), vp.

    Parameters
    ----------
    y_true : array-like
        Les vraies valeurs de RUL.
    y_pred : array-like
        Les RUL predites, dans le meme ordre.
    horizon : float
        N, en cycles : definit la classe positive.
    seuil_alerte : float ou None, optionnel
        Seuil de decision sur la RUL predite ; None (defaut) : egal a `horizon`.

    Returns
    -------
    dict
        "horizon", "seuil_alerte", "n", "nb_positifs", "proportion_positifs", "vn", "fp", "fn", "vp",
        "nb_alertes", "accuracy", "precision", "rappel", "f1" (0 quand non defini, zero_division=0),
        "auc_roc", "precision_moyenne" (NaN si une seule classe est presente).
    """
    vrais = np.asarray(y_true, dtype=float)
    predits = np.asarray(y_pred, dtype=float)
    if seuil_alerte is None:  # par defaut, on alerte des que la RUL predite passe sous l'horizon
        seuil_alerte = horizon

    etiquettes = (vrais <= horizon).astype(int)  # 1 = le moteur tombe en panne dans les N cycles
    alertes = (predits <= seuil_alerte).astype(int)  # 1 = le modele demande le retrait
    scores = -predits  # plus la RUL predite est basse, plus le score est haut

    # matrice de confusion 2 x 2, meme si une classe est absente (labels=[0, 1] force les deux)
    matrice = confusion_matrix(etiquettes, alertes, labels=[0, 1])
    vn = int(matrice[0, 0])
    fp = int(matrice[0, 1])
    fn = int(matrice[1, 0])
    vp = int(matrice[1, 1])

    resultats = {}
    resultats["horizon"] = float(horizon)
    resultats["seuil_alerte"] = float(seuil_alerte)
    resultats["n"] = int(len(vrais))
    resultats["nb_positifs"] = int(np.sum(etiquettes))
    resultats["proportion_positifs"] = float(np.mean(etiquettes))  # a afficher a cote de l'accuracy
    resultats["vn"] = vn
    resultats["fp"] = fp
    resultats["fn"] = fn
    resultats["vp"] = vp
    resultats["nb_alertes"] = int(np.sum(alertes))
    resultats["accuracy"] = float((vn + vp) / len(vrais))
    resultats["precision"] = float(precision_score(etiquettes, alertes, zero_division=0))
    resultats["rappel"] = float(recall_score(etiquettes, alertes, zero_division=0))
    resultats["f1"] = float(f1_score(etiquettes, alertes, zero_division=0))

    # AUC et precision moyenne : definies seulement si les deux classes sont presentes
    nb_classes = len(np.unique(etiquettes))
    if nb_classes == 2:
        resultats["auc_roc"] = float(roc_auc_score(etiquettes, scores))
        resultats["precision_moyenne"] = float(average_precision_score(etiquettes, scores))
    else:
        resultats["auc_roc"] = np.nan
        resultats["precision_moyenne"] = np.nan
    return resultats


def choisir_seuil_alerte(y_true, y_pred, horizon, rappel_min=0.95):
    """
    Choisit le seuil d'alerte sur la RUL predite, sur des donnees de VALIDATION (jamais sur le test).

    Regle de la section 4 du protocole : parmi les seuils qui donnent un rappel >= `rappel_min`, on prend
    celui qui maximise la precision ; a precision egale, celui qui donne le plus grand rappel. Les seuils
    candidats sont toutes les valeurs de RUL predite (courbe precision-rappel de scikit-learn sur le
    score -RUL predite) : alerte si RUL predite <= seuil.

    Parameters
    ----------
    y_true : array-like
        Les vraies valeurs de RUL de validation (par exemple la cible des predictions hors-pli).
    y_pred : array-like
        Les RUL predites correspondantes (predictions hors-pli).
    horizon : float
        N, en cycles : classe positive = RUL vraie <= N.
    rappel_min : float, optionnel
        Rappel minimal exige (0.95 par defaut).

    Returns
    -------
    dict
        "horizon", "rappel_min", "seuil_alerte" (en cycles de RUL predite ; NaN si aucun seuil n'atteint le
        rappel minimal), "precision_validation", "rappel_validation" au seuil retenu.
    """
    vrais = np.asarray(y_true, dtype=float)
    predits = np.asarray(y_pred, dtype=float)
    etiquettes = (vrais <= horizon).astype(int)
    scores = -predits  # score de classement : alerte si score >= seuil_score, c'est-a-dire RUL predite <= -seuil_score

    # precisions[i] et rappels[i] correspondent a seuils_score[i] (le dernier point, sans seuil, est ignore)
    precisions, rappels, seuils_score = precision_recall_curve(etiquettes, scores)

    meilleur_indice = None  # aucun seuil retenu pour l'instant
    for i in range(len(seuils_score)):  # un seuil candidat a la fois
        if rappels[i] < rappel_min:  # contrainte de securite non respectee : seuil rejete
            continue
        if meilleur_indice is None:
            meilleur_indice = i
            continue
        meilleure_precision = precisions[meilleur_indice]
        if precisions[i] > meilleure_precision:  # meilleure precision : nouveau seuil retenu
            meilleur_indice = i
        elif precisions[i] == meilleure_precision and rappels[i] > rappels[meilleur_indice]:  # egalite : plus de rappel
            meilleur_indice = i

    resultats = {"horizon": float(horizon), "rappel_min": float(rappel_min)}
    if meilleur_indice is None:  # aucun seuil n'atteint le rappel minimal (ne devrait pas arriver : seuil infini)
        resultats["seuil_alerte"] = np.nan
        resultats["precision_validation"] = np.nan
        resultats["rappel_validation"] = np.nan
    else:
        resultats["seuil_alerte"] = float(-seuils_score[meilleur_indice])  # retour en cycles de RUL predite
        resultats["precision_validation"] = float(precisions[meilleur_indice])
        resultats["rappel_validation"] = float(rappels[meilleur_indice])
    return resultats


def comparer_modeles_par_paires(valeurs_par_modele, seuil_p=0.05):
    """
    Compare toutes les paires de modeles sur des valeurs appariees (test t et Wilcoxon), avec correction de Bonferroni.

    Chaque paire passe par `comparer_apparie`. Avec m paires testees, la probabilite d'au moins un faux
    "significatif" grimpe avec m : la correction de Bonferroni multiplie chaque p-valeur par m (plafonnee a 1).
    Conclusion (fixee avant les resultats) : "A meilleur" si la p-valeur de Wilcoxon corrigee est < seuil_p et
    la difference moyenne A - B est negative (valeurs plus petites = erreurs plus petites), "B meilleur" dans
    le cas symetrique, sinon "pas de difference significative". Si toutes les differences sont nulles
    (deux modeles identiques), les tests ne sont pas definis : p-valeurs fixees a 1.

    Parameters
    ----------
    valeurs_par_modele : dict
        Nom du modele -> array-like de valeurs appariees (ex. RMSE par moteur), memes unites et meme ordre
        pour tous les modeles. L'ordre des cles fixe l'ordre des paires.
    seuil_p : float, optionnel
        Seuil de significativite applique aux p-valeurs CORRIGEES (0.05 par defaut).

    Returns
    -------
    pandas.DataFrame
        Une ligne par paire : "modele_a", "modele_b", "n", "difference_moyenne" (A - B),
        "nb_unites_a_meilleur" (unites ou A < B), "p_t", "p_t_bonferroni", "p_wilcoxon",
        "p_wilcoxon_bonferroni", "nb_comparaisons", "conclusion".
    """
    noms = list(valeurs_par_modele.keys())
    paires = list(itertools.combinations(noms, 2))  # toutes les paires (A, B) avec A avant B
    nb_comparaisons = len(paires)

    lignes = []  # une entree par paire
    for nom_a, nom_b in paires:
        a = np.asarray(valeurs_par_modele[nom_a], dtype=float)
        b = np.asarray(valeurs_par_modele[nom_b], dtype=float)
        differences = a - b

        ligne = {"modele_a": nom_a, "modele_b": nom_b, "n": int(len(a))}
        ligne["difference_moyenne"] = float(np.mean(differences))
        ligne["nb_unites_a_meilleur"] = int(np.sum(differences < 0))

        if np.all(differences == 0):  # deux modeles identiques : aucun test possible
            ligne["p_t"] = 1.0
            ligne["p_wilcoxon"] = 1.0
        else:
            resultat = comparer_apparie(a, b)
            ligne["p_t"] = resultat["p_valeur_t"]
            ligne["p_wilcoxon"] = resultat["p_valeur_wilcoxon"]

        # correction de Bonferroni : p x m, plafonnee a 1
        ligne["p_t_bonferroni"] = min(1.0, ligne["p_t"] * nb_comparaisons)
        ligne["p_wilcoxon_bonferroni"] = min(1.0, ligne["p_wilcoxon"] * nb_comparaisons)
        ligne["nb_comparaisons"] = nb_comparaisons

        # conclusion selon la regle ecrite dans la docstring
        if ligne["p_wilcoxon_bonferroni"] < seuil_p and ligne["difference_moyenne"] < 0:
            ligne["conclusion"] = f"{nom_a} meilleur"
        elif ligne["p_wilcoxon_bonferroni"] < seuil_p and ligne["difference_moyenne"] > 0:
            ligne["conclusion"] = f"{nom_b} meilleur"
        else:
            ligne["conclusion"] = "pas de difference significative"
        lignes.append(ligne)

    # ordre des colonnes fixe, pour un tableau lisible
    colonnes = ["modele_a", "modele_b", "n", "difference_moyenne", "nb_unites_a_meilleur", "p_t", "p_t_bonferroni",
                "p_wilcoxon", "p_wilcoxon_bonferroni", "nb_comparaisons", "conclusion"]
    return pd.DataFrame(lignes, columns=colonnes)
