"""
Metriques et protocoles d'evaluation (Phase 4) pour la prediction de RUL sur FD001.

On y trouve :
- les metriques : RMSE, MAE, score NASA, nombre de surestimations, alertes manquees / fausses alertes ;
- la validation croisee par moteur (GroupKFold), avec verification explicite des plis ;
- l'evaluation sur le jeu de test officiel (dernier cycle de chaque moteur test) ;
- l'analyse par moteur et la comparaison statistique appariee de deux modeles ;
- une demonstration (pour la soutenance) que les constantes 13/10 du SCORE NASA ne sont pas des
  reglages a optimiser.

Les calculs reprennent exactement ceux du notebook 03 (RMSE "a la main", score NASA en boucle explicite,
GroupKFold(n_splits=5, shuffle=True, random_state=42)), pour que les chiffres de la Phase 3 soient reproduits.
"""

import time

import numpy as np
import pandas as pd
from scipy import stats
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
