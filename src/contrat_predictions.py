"""
Contrat d'echange des fichiers de predictions entre les scripts d'entrainement et le tableau de bord.

Pourquoi ce module existe : plusieurs scripts produisent des predictions de RUL (modeles a arbres
sur FD001 a FD004, LSTM, CNN 1D), et l'application de demo (app/) doit pouvoir les afficher toutes
de la meme facon, sans connaitre le code qui les a produites. On fixe donc UN format de fichier,
ecrit et relu uniquement par les fonctions de ce module :

    sorties/predictions/<SOUS_JEU>__<modele>.csv.gz   -> les predictions, une ligne par (moteur, cycle)
    sorties/predictions/<SOUS_JEU>__<modele>.json     -> les metadonnees (metriques, protocole, date...)

Exemple : sorties/predictions/FD001__xgboost_phase3.csv.gz et sorties/predictions/FD001__xgboost_phase3.json

Colonnes obligatoires du CSV (une ligne par moteur test ET par cycle observe, TOUS les cycles) :
- unit_number      : numero du moteur test ;
- time_cycles      : numero du cycle observe ;
- rul_predite      : RUL predite par le modele a ce cycle (>= 0, deja clippee a 0) ;
- rul_vraie        : RUL vraie = RUL_FD00X[moteur] + (dernier cycle observe - cycle) ;
- rul_vraie_cappee : min(rul_vraie, 125), ce que le modele a appris a predire ;
- regime           : numero de regime operatoire (entier ; 0 pour les sous-jeux a un seul regime).
Des colonnes supplementaires sont autorisees (elles sont conservees telles quelles).

Les metriques de test (cle "metriques_test" des metadonnees) sont calculees au DERNIER cycle observe
de chaque moteur test, exactement comme `src.evaluation.evaluer_sur_test` : la fonction
`metriques_dernier_cycle` de ce module fait ce calcul a partir du fichier de predictions, et sert a la
fois aux producteurs (pour remplir les metadonnees) et au tableau de bord (controle de coherence).

Le dossier par defaut peut etre remplace par la variable d'environnement
AERO_PREDICT_DOSSIER_PREDICTIONS (utile pour les tests automatiques, qui ecrivent dans un dossier
temporaire sans toucher aux vraies predictions).
"""

import datetime
import json
import math
import os

import numpy as np
import pandas as pd

from src.config_modele import RACINE_DEPOT, convertir_en_types_python
from src.evaluation import alertes_manquees, fausses_alertes, mae, nb_surestimes, rmse, score_nasa

# ---------------------------------------------------------------------------------------------------
# Constantes du contrat
# ---------------------------------------------------------------------------------------------------

# dossier par defaut des fichiers de predictions (chemin absolu ; sorties/ est ignore par git)
DOSSIER_PREDICTIONS = os.path.join(RACINE_DEPOT, "sorties", "predictions")

# variable d'environnement qui, si elle est definie, remplace le dossier par defaut
VARIABLE_ENV_DOSSIER = "AERO_PREDICT_DOSSIER_PREDICTIONS"

# version du format : a incrementer si le contrat change (ecrite dans chaque fichier de metadonnees)
VERSION_CONTRAT = 1

# les quatre sous-jeux C-MAPSS, dans l'ordre d'affichage
SOUS_JEUX = ["FD001", "FD002", "FD003", "FD004"]

# les modeles connus : cle (utilisee dans les noms de fichiers) -> nom affiche ; l'ordre sert a l'affichage
MODELES_CONNUS = {
    "xgboost_phase3": "XGBoost Phase 3",
    "lstm": "LSTM",
    "cnn1d": "CNN 1D",
}

# les deux familles de modeles possibles
FAMILLES_VALIDES = ["arbres", "sequentiel"]

# colonnes obligatoires du fichier de predictions, dans l'ordre d'ecriture
COLONNES_OBLIGATOIRES = ["unit_number", "time_cycles", "rul_predite", "rul_vraie", "rul_vraie_cappee", "regime"]

# cles obligatoires du fichier de metadonnees
CLES_METADONNEES_OBLIGATOIRES = [
    "sous_jeu",
    "modele",
    "nom_affiche",
    "famille",
    "metriques_test",
    "metriques_validation",
    "protocole_validation",
    "duree_entrainement_s",
    "date",
    "source",
]

# metriques de test obligatoires (calculees au dernier cycle de chaque moteur test)
METRIQUES_TEST_OBLIGATOIRES = [
    "rmse",
    "rmse_contre_rul_plafonnee",
    "mae",
    "score_nasa",
    "nb_surestimes",
    "alertes_manquees",
    "fausses_alertes",
]

# plafond de la RUL (Phase 2) : rul_vraie_cappee = min(rul_vraie, PLAFOND_RUL)
PLAFOND_RUL = 125

# tolerance numerique pour verifier rul_vraie_cappee et comparer des metriques recalculees
TOLERANCE = 1e-6

# suffixes des deux fichiers et separateur entre sous-jeu et modele dans le nom
SUFFIXE_PREDICTIONS = ".csv.gz"
SUFFIXE_METADONNEES = ".json"
SEPARATEUR_NOM = "__"


# ---------------------------------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------------------------------

def dossier_par_defaut():
    """
    Renvoie le dossier des predictions a utiliser quand aucun dossier n'est donne explicitement.

    Si la variable d'environnement AERO_PREDICT_DOSSIER_PREDICTIONS est definie (et non vide), elle
    l'emporte : c'est ce qui permet aux tests et a l'application de travailler sur un autre dossier.
    Sinon, c'est DOSSIER_PREDICTIONS (<racine du depot>/sorties/predictions).

    Returns
    -------
    str
        Le chemin absolu du dossier des predictions.
    """
    valeur_env = os.environ.get(VARIABLE_ENV_DOSSIER, "")  # chaine vide si la variable n'existe pas
    if valeur_env.strip() != "":  # variable definie et non vide : elle l'emporte
        return os.path.abspath(valeur_env)
    return DOSSIER_PREDICTIONS


def resoudre_dossier(dossier=None):
    """
    Renvoie le dossier a utiliser : celui donne en argument, ou le dossier par defaut.

    Parameters
    ----------
    dossier : str ou None, optionnel
        Dossier explicite ; None pour utiliser `dossier_par_defaut()`.

    Returns
    -------
    str
        Le chemin absolu du dossier.
    """
    if dossier is None:
        return dossier_par_defaut()
    return os.path.abspath(str(dossier))  # accepte aussi un pathlib.Path (ex. tmp_path de pytest)


def verifier_sous_jeu(sous_jeu):
    """
    Verifie qu'un nom de sous-jeu fait partie de SOUS_JEUX.

    Parameters
    ----------
    sous_jeu : str
        Le nom a verifier (ex. "FD001").

    Raises
    ------
    ValueError
        Si le sous-jeu est inconnu.
    """
    if sous_jeu not in SOUS_JEUX:
        raise ValueError(f"Sous-jeu inconnu : {sous_jeu!r}. Sous-jeux possibles : {', '.join(SOUS_JEUX)}.")


def verifier_modele(modele):
    """
    Verifie qu'une cle de modele fait partie de MODELES_CONNUS.

    Parameters
    ----------
    modele : str
        La cle a verifier (ex. "xgboost_phase3").

    Raises
    ------
    ValueError
        Si le modele est inconnu.
    """
    if modele not in MODELES_CONNUS:
        cles_possibles = ", ".join(list(MODELES_CONNUS.keys()))
        raise ValueError(f"Modele inconnu : {modele!r}. Modeles possibles : {cles_possibles}.")


def chemin_predictions(sous_jeu, modele, dossier=None):
    """
    Construit le chemin du fichier de predictions d'un (sous-jeu, modele).

    Parameters
    ----------
    sous_jeu : str
        Le sous-jeu (ex. "FD001").
    modele : str
        La cle du modele (ex. "xgboost_phase3").
    dossier : str ou None, optionnel
        Dossier des predictions ; None pour le dossier par defaut.

    Returns
    -------
    str
        <dossier>/<SOUS_JEU>__<modele>.csv.gz

    Raises
    ------
    ValueError
        Si le sous-jeu ou le modele est inconnu.
    """
    verifier_sous_jeu(sous_jeu)
    verifier_modele(modele)
    nom_fichier = f"{sous_jeu}{SEPARATEUR_NOM}{modele}{SUFFIXE_PREDICTIONS}"
    return os.path.join(resoudre_dossier(dossier), nom_fichier)


def chemin_metadonnees(sous_jeu, modele, dossier=None):
    """
    Construit le chemin du fichier de metadonnees (JSON) d'un (sous-jeu, modele).

    Parameters
    ----------
    sous_jeu : str
        Le sous-jeu (ex. "FD001").
    modele : str
        La cle du modele (ex. "xgboost_phase3").
    dossier : str ou None, optionnel
        Dossier des predictions ; None pour le dossier par defaut.

    Returns
    -------
    str
        <dossier>/<SOUS_JEU>__<modele>.json

    Raises
    ------
    ValueError
        Si le sous-jeu ou le modele est inconnu.
    """
    verifier_sous_jeu(sous_jeu)
    verifier_modele(modele)
    nom_fichier = f"{sous_jeu}{SEPARATEUR_NOM}{modele}{SUFFIXE_METADONNEES}"
    return os.path.join(resoudre_dossier(dossier), nom_fichier)


# ---------------------------------------------------------------------------------------------------
# Aides pour les producteurs
# ---------------------------------------------------------------------------------------------------

def calculer_rul_vraie(df, rul_finale):
    """
    Ajoute les colonnes "rul_vraie" et "rul_vraie_cappee" a un DataFrame de cycles de moteurs test.

    Les fichiers RUL_FD00X.txt ne donnent la RUL qu'au dernier cycle observe. Comme la RUL baisse
    d'exactement 1 a chaque cycle :

        rul_vraie(t) = RUL_FD00X[moteur] + (dernier_cycle_observe_du_moteur - t)

    Parameters
    ----------
    df : pandas.DataFrame
        Au moins "unit_number" et "time_cycles" (n'est pas modifie, une copie est renvoyee).
    rul_finale : dict, pandas.Series ou array-like
        RUL vraie au dernier cycle observe de chaque moteur :
        - dict ou Series : numero de moteur -> RUL ;
        - liste ou tableau numpy : dans l'ordre croissant des numeros de moteur (ligne i de
          RUL_FD00X.txt = i-eme plus petit numero de moteur, comme dans data/readme.txt).

    Returns
    -------
    pandas.DataFrame
        Copie de df avec "rul_vraie" et "rul_vraie_cappee" (= min(rul_vraie, 125)).

    Raises
    ------
    ValueError
        Si le nombre de RUL ne correspond pas au nombre de moteurs, ou si un moteur n'a pas de RUL.
    """
    resultat = df.copy()  # on ne modifie jamais le DataFrame de l'appelant
    moteurs_tries = np.sort(resultat["unit_number"].unique())  # numeros de moteur, du plus petit au plus grand

    # on construit un dictionnaire moteur -> RUL au dernier cycle, quel que soit le format recu
    rul_par_moteur = {}
    if isinstance(rul_finale, (dict, pd.Series)):
        for moteur_id in moteurs_tries:
            if moteur_id not in rul_finale:
                raise ValueError(f"Aucune RUL finale fournie pour le moteur {moteur_id}.")
            rul_par_moteur[moteur_id] = float(rul_finale[moteur_id])
    else:
        valeurs = np.asarray(rul_finale, dtype=float)  # liste ou tableau : dans l'ordre des moteurs
        if len(valeurs) != len(moteurs_tries):
            raise ValueError(
                f"{len(moteurs_tries)} moteurs dans les donnees mais {len(valeurs)} valeurs de RUL finale."
            )
        for position in range(len(moteurs_tries)):
            rul_par_moteur[moteurs_tries[position]] = float(valeurs[position])

    # dernier cycle observe de chaque moteur
    dernier_cycle_par_moteur = resultat.groupby("unit_number")["time_cycles"].max()

    # calcul ligne par ligne, avec des tableaux numpy (boucle explicite sur les moteurs)
    rul_vraie = np.zeros(len(resultat), dtype=float)
    numeros = resultat["unit_number"].to_numpy()
    cycles = resultat["time_cycles"].to_numpy()
    for moteur_id in moteurs_tries:
        masque = numeros == moteur_id  # Vrai sur les lignes de ce moteur
        dernier_cycle = dernier_cycle_par_moteur[moteur_id]
        rul_vraie[masque] = rul_par_moteur[moteur_id] + (dernier_cycle - cycles[masque])

    resultat["rul_vraie"] = rul_vraie
    resultat["rul_vraie_cappee"] = np.minimum(rul_vraie, PLAFOND_RUL)  # meme plafond que la cible (Phase 2)
    return resultat


def extraire_dernier_cycle(df):
    """
    Garde, pour chaque moteur, la ligne de son dernier cycle observe.

    C'est a ce cycle que le jeu de test officiel est evalue (la RUL de RUL_FD00X.txt y est connue).

    Parameters
    ----------
    df : pandas.DataFrame
        Un fichier de predictions (au moins "unit_number" et "time_cycles").

    Returns
    -------
    pandas.DataFrame
        Une ligne par moteur, triee par numero de moteur, index remis a 0..n-1.
    """
    tri = df.sort_values(["unit_number", "time_cycles"])  # cycles dans l'ordre pour chaque moteur
    derniers = tri.groupby("unit_number").tail(1)  # derniere ligne de chaque moteur = dernier cycle
    return derniers.reset_index(drop=True)


# ---------------------------------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------------------------------

def valider_predictions(df):
    """
    Verifie qu'un DataFrame respecte le contrat des fichiers de predictions.

    Controles : colonnes obligatoires presentes, aucune valeur manquante dans ces colonnes, valeurs
    numeriques, rul_predite >= 0, rul_vraie >= 0, rul_vraie_cappee = min(rul_vraie, 125), regime
    entier, et pas de doublon (unit_number, time_cycles).

    Parameters
    ----------
    df : pandas.DataFrame
        Les predictions a verifier.

    Raises
    ------
    ValueError
        Avec un message en francais qui dit precisement ce qui ne va pas.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Les predictions doivent etre un pandas.DataFrame.")

    if len(df) == 0:
        raise ValueError("Le fichier de predictions est vide (aucune ligne).")

    # 1. colonnes obligatoires
    colonnes_manquantes = []
    for colonne in COLONNES_OBLIGATOIRES:
        if colonne not in df.columns:
            colonnes_manquantes.append(colonne)
    if len(colonnes_manquantes) > 0:
        raise ValueError(f"Colonnes obligatoires manquantes : {', '.join(colonnes_manquantes)}.")

    # 2. valeurs numeriques et aucune valeur manquante
    for colonne in COLONNES_OBLIGATOIRES:
        if not pd.api.types.is_numeric_dtype(df[colonne]):
            raise ValueError(f"La colonne {colonne!r} doit etre numerique (type trouve : {df[colonne].dtype}).")
        nb_manquants = int(df[colonne].isna().sum())
        if nb_manquants > 0:
            raise ValueError(f"La colonne {colonne!r} contient {nb_manquants} valeur(s) manquante(s) (NaN).")

    # 3. RUL predite positive ou nulle (une RUL negative n'a pas de sens physique : clipper a 0 avant)
    nb_negatives = int((df["rul_predite"] < 0).sum())
    if nb_negatives > 0:
        raise ValueError(
            f"{nb_negatives} valeur(s) de rul_predite sont negatives : clipper les predictions a 0 avant d'ecrire."
        )

    # 4. RUL vraie positive ou nulle
    nb_vraies_negatives = int((df["rul_vraie"] < 0).sum())
    if nb_vraies_negatives > 0:
        raise ValueError(f"{nb_vraies_negatives} valeur(s) de rul_vraie sont negatives.")

    # 5. rul_vraie_cappee = min(rul_vraie, 125)
    attendu = np.minimum(df["rul_vraie"].to_numpy(dtype=float), PLAFOND_RUL)
    ecart_max = float(np.max(np.abs(df["rul_vraie_cappee"].to_numpy(dtype=float) - attendu)))
    if ecart_max > TOLERANCE:
        raise ValueError(
            f"rul_vraie_cappee doit valoir min(rul_vraie, {PLAFOND_RUL}) (ecart maximal trouve : {ecart_max:.3f})."
        )

    # 6. regime entier (0 pour les sous-jeux a un seul regime)
    regimes = df["regime"].to_numpy(dtype=float)
    if not np.all(regimes == np.round(regimes)):
        raise ValueError("La colonne 'regime' doit contenir des entiers (numero de regime operatoire).")

    # 7. pas de doublon (moteur, cycle) : une seule prediction par moteur et par cycle
    doublons = df.duplicated(subset=["unit_number", "time_cycles"])
    nb_doublons = int(doublons.sum())
    if nb_doublons > 0:
        raise ValueError(f"{nb_doublons} doublon(s) (unit_number, time_cycles) : une seule ligne par moteur et par cycle.")


def valider_metadonnees(meta):
    """
    Verifie qu'un dictionnaire de metadonnees respecte le contrat.

    Controles : cles obligatoires presentes, sous-jeu et modele connus, famille valide, metriques de
    test completes et numeriques, metriques de validation sous forme de dictionnaire, duree numerique
    positive, date au format AAAA-MM-JJ. Les cles supplementaires sont autorisees.

    Parameters
    ----------
    meta : dict
        Les metadonnees a verifier.

    Raises
    ------
    ValueError
        Avec un message en francais qui dit precisement ce qui ne va pas.
    """
    if not isinstance(meta, dict):
        raise ValueError("Les metadonnees doivent etre un dictionnaire.")

    # 1. cles obligatoires
    cles_manquantes = []
    for cle in CLES_METADONNEES_OBLIGATOIRES:
        if cle not in meta:
            cles_manquantes.append(cle)
    if len(cles_manquantes) > 0:
        raise ValueError(f"Cles de metadonnees manquantes : {', '.join(cles_manquantes)}.")

    # 2. sous-jeu, modele, famille
    verifier_sous_jeu(meta["sous_jeu"])
    verifier_modele(meta["modele"])
    if meta["famille"] not in FAMILLES_VALIDES:
        raise ValueError(f"Famille inconnue : {meta['famille']!r}. Familles possibles : {', '.join(FAMILLES_VALIDES)}.")

    # 3. textes non vides
    for cle in ["nom_affiche", "protocole_validation", "source"]:
        if not isinstance(meta[cle], str) or meta[cle].strip() == "":
            raise ValueError(f"La cle {cle!r} doit etre un texte non vide.")

    # 4. metriques de test : dictionnaire complet, valeurs numeriques
    metriques_test = meta["metriques_test"]
    if not isinstance(metriques_test, dict):
        raise ValueError("'metriques_test' doit etre un dictionnaire.")
    metriques_manquantes = []
    for cle in METRIQUES_TEST_OBLIGATOIRES:
        if cle not in metriques_test:
            metriques_manquantes.append(cle)
    if len(metriques_manquantes) > 0:
        raise ValueError(f"Metriques de test manquantes : {', '.join(metriques_manquantes)}.")
    for cle in METRIQUES_TEST_OBLIGATOIRES:
        valeur = metriques_test[cle]
        if isinstance(valeur, bool) or not isinstance(valeur, (int, float, np.integer, np.floating)):
            raise ValueError(f"La metrique de test {cle!r} doit etre un nombre (trouve : {valeur!r}).")
        if math.isnan(float(valeur)):
            raise ValueError(f"La metrique de test {cle!r} vaut NaN.")

    # 5. metriques de validation : dictionnaire (eventuellement vide)
    if not isinstance(meta["metriques_validation"], dict):
        raise ValueError("'metriques_validation' doit etre un dictionnaire (eventuellement vide).")

    # 6. duree d'entrainement : nombre positif ou nul
    duree = meta["duree_entrainement_s"]
    if isinstance(duree, bool) or not isinstance(duree, (int, float, np.integer, np.floating)):
        raise ValueError(f"'duree_entrainement_s' doit etre un nombre de secondes (trouve : {duree!r}).")
    if math.isnan(float(duree)) or float(duree) < 0:
        raise ValueError(f"'duree_entrainement_s' doit etre positive ou nulle (trouve : {duree!r}).")

    # 7. date au format AAAA-MM-JJ
    date_texte = meta["date"]
    if not isinstance(date_texte, str):
        raise ValueError(f"'date' doit etre un texte au format AAAA-MM-JJ (trouve : {date_texte!r}).")
    try:
        datetime.datetime.strptime(date_texte, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"'date' doit etre au format AAAA-MM-JJ (trouve : {date_texte!r}).") from None


# ---------------------------------------------------------------------------------------------------
# Ecriture et lecture
# ---------------------------------------------------------------------------------------------------

def nettoyer_pour_json(valeur):
    """
    Convertit une valeur en types Python ecrivables en JSON strict (types numpy convertis, NaN -> None).

    Parameters
    ----------
    valeur : objet
        dict, list, nombre numpy, tableau numpy, texte...

    Returns
    -------
    objet
        Une nouvelle valeur composee de dict, list, str, int, float, bool et None.
    """
    convertie = convertir_en_types_python(valeur)  # types numpy -> types Python (src/config_modele.py)

    # JSON strict n'accepte ni NaN ni l'infini : on les remplace par None (null en JSON)
    if isinstance(convertie, dict):
        resultat = {}
        for cle in convertie:
            resultat[cle] = nettoyer_pour_json(convertie[cle])
        return resultat
    if isinstance(convertie, list):
        resultat = []
        for element in convertie:
            resultat.append(nettoyer_pour_json(element))
        return resultat
    if isinstance(convertie, float) and (math.isnan(convertie) or math.isinf(convertie)):
        return None
    return convertie


def ecrire_predictions(df, metadonnees, dossier=None):
    """
    Valide puis ecrit un fichier de predictions (CSV gzip) et ses metadonnees (JSON).

    Le DataFrame est trie par (unit_number, time_cycles), les colonnes obligatoires sont placees en
    premier (les colonnes supplementaires sont conservees apres), et le dossier est cree si besoin.
    Les metadonnees recoivent automatiquement "version_contrat", "nb_lignes" et "nb_moteurs".

    Parameters
    ----------
    df : pandas.DataFrame
        Les predictions (voir COLONNES_OBLIGATOIRES).
    metadonnees : dict
        Les metadonnees (voir CLES_METADONNEES_OBLIGATOIRES).
    dossier : str ou None, optionnel
        Dossier de sortie ; None pour le dossier par defaut.

    Returns
    -------
    tuple of str
        (chemin du CSV gzip, chemin du JSON).

    Raises
    ------
    ValueError
        Si les predictions ou les metadonnees ne respectent pas le contrat.
    """
    valider_predictions(df)
    meta = nettoyer_pour_json(dict(metadonnees))  # copie convertie : on ne modifie pas le dict de l'appelant
    valider_metadonnees(meta)

    # ordre des colonnes : les obligatoires d'abord, puis les supplementaires dans leur ordre d'origine
    colonnes_ordonnees = list(COLONNES_OBLIGATOIRES)
    for colonne in df.columns:
        if colonne not in colonnes_ordonnees:
            colonnes_ordonnees.append(colonne)

    # tri par moteur puis par cycle, index remis a 0..n-1
    a_ecrire = df[colonnes_ordonnees].sort_values(["unit_number", "time_cycles"]).reset_index(drop=True)

    # informations ajoutees automatiquement (utiles pour verifier un fichier a l'oeil)
    meta["version_contrat"] = VERSION_CONTRAT
    meta["nb_lignes"] = int(len(a_ecrire))
    meta["nb_moteurs"] = int(a_ecrire["unit_number"].nunique())

    chemin_csv = chemin_predictions(meta["sous_jeu"], meta["modele"], dossier)
    chemin_json = chemin_metadonnees(meta["sous_jeu"], meta["modele"], dossier)
    os.makedirs(os.path.dirname(chemin_csv), exist_ok=True)  # cree le dossier s'il n'existe pas

    a_ecrire.to_csv(chemin_csv, index=False, compression="gzip")
    with open(chemin_json, "w", encoding="utf-8") as fichier:
        json.dump(meta, fichier, ensure_ascii=False, indent=2, allow_nan=False)
        fichier.write("\n")

    return chemin_csv, chemin_json


def lire_metadonnees(sous_jeu, modele, dossier=None):
    """
    Lit et valide le fichier de metadonnees d'un (sous-jeu, modele).

    Parameters
    ----------
    sous_jeu : str
        Le sous-jeu (ex. "FD001").
    modele : str
        La cle du modele (ex. "xgboost_phase3").
    dossier : str ou None, optionnel
        Dossier des predictions ; None pour le dossier par defaut.

    Returns
    -------
    dict
        Les metadonnees validees.

    Raises
    ------
    FileNotFoundError
        Si le fichier n'existe pas.
    ValueError
        Si le contenu ne respecte pas le contrat.
    """
    chemin_json = chemin_metadonnees(sous_jeu, modele, dossier)
    with open(chemin_json, "r", encoding="utf-8") as fichier:
        meta = json.load(fichier)
    valider_metadonnees(meta)
    return meta


def lire_predictions(sous_jeu, modele, dossier=None):
    """
    Lit et valide un fichier de predictions et ses metadonnees.

    Parameters
    ----------
    sous_jeu : str
        Le sous-jeu (ex. "FD001").
    modele : str
        La cle du modele (ex. "xgboost_phase3").
    dossier : str ou None, optionnel
        Dossier des predictions ; None pour le dossier par defaut.

    Returns
    -------
    tuple
        (df, metadonnees) : le DataFrame trie par (unit_number, time_cycles), avec unit_number,
        time_cycles et regime en entiers, et le dictionnaire de metadonnees.

    Raises
    ------
    FileNotFoundError
        Si l'un des deux fichiers n'existe pas.
    ValueError
        Si le contenu ne respecte pas le contrat.
    """
    chemin_csv = chemin_predictions(sous_jeu, modele, dossier)
    df = pd.read_csv(chemin_csv, compression="gzip")
    valider_predictions(df)

    # types entiers pour les identifiants (le CSV peut les relire en reels selon les valeurs)
    for colonne in ["unit_number", "time_cycles", "regime"]:
        df[colonne] = df[colonne].astype(int)
    df = df.sort_values(["unit_number", "time_cycles"]).reset_index(drop=True)

    meta = lire_metadonnees(sous_jeu, modele, dossier)
    return df, meta


def lister_predictions(dossier=None):
    """
    Liste les fichiers de predictions disponibles (CSV gzip ET JSON presents, JSON lisible et valide).

    Les fichiers dont le nom ne suit pas le contrat, ou dont les metadonnees sont absentes ou
    invalides, sont ignores (l'application n'affiche que ce qu'elle peut lire).

    Parameters
    ----------
    dossier : str ou None, optionnel
        Dossier des predictions ; None pour le dossier par defaut.

    Returns
    -------
    pandas.DataFrame
        Colonnes : sous_jeu, modele, nom_affiche, famille, chemin_predictions, chemin_metadonnees ;
        triees dans l'ordre de SOUS_JEUX puis de MODELES_CONNUS. Vide (avec ces colonnes) si rien.
    """
    colonnes = ["sous_jeu", "modele", "nom_affiche", "famille", "chemin_predictions", "chemin_metadonnees"]
    dossier_absolu = resoudre_dossier(dossier)
    lignes = []

    if os.path.isdir(dossier_absolu):
        # on parcourt les combinaisons connues dans l'ordre d'affichage : le tri est donc deja le bon
        for sous_jeu in SOUS_JEUX:
            for modele in MODELES_CONNUS:
                chemin_csv = chemin_predictions(sous_jeu, modele, dossier_absolu)
                chemin_json = chemin_metadonnees(sous_jeu, modele, dossier_absolu)
                if not (os.path.isfile(chemin_csv) and os.path.isfile(chemin_json)):
                    continue  # couple incomplet : rien a afficher
                try:
                    meta = lire_metadonnees(sous_jeu, modele, dossier_absolu)
                except (ValueError, OSError):
                    continue  # metadonnees illisibles ou hors contrat : on ignore ce couple
                if meta["sous_jeu"] != sous_jeu or meta["modele"] != modele:
                    continue  # le contenu ne correspond pas au nom du fichier : on ignore
                ligne = {}
                ligne["sous_jeu"] = sous_jeu
                ligne["modele"] = modele
                ligne["nom_affiche"] = meta["nom_affiche"]
                ligne["famille"] = meta["famille"]
                ligne["chemin_predictions"] = chemin_csv
                ligne["chemin_metadonnees"] = chemin_json
                lignes.append(ligne)

    return pd.DataFrame(lignes, columns=colonnes)


# ---------------------------------------------------------------------------------------------------
# Metriques au dernier cycle (producteurs et controle de coherence)
# ---------------------------------------------------------------------------------------------------

def metriques_dernier_cycle(df, plafond=PLAFOND_RUL, seuil_surestimation=20, seuil_alerte=30):
    """
    Recalcule les metriques de test au dernier cycle observe de chaque moteur, comme `evaluer_sur_test`.

    Protocole (identique a src/evaluation.py) : prediction clippee a 0, comparee a la RUL vraie BRUTE
    (non plafonnee, celle de RUL_FD00X.txt au dernier cycle) ; on ajoute un RMSE contre la RUL vraie
    plafonnee min(RUL, plafond).

    Parameters
    ----------
    df : pandas.DataFrame
        Un fichier de predictions respectant le contrat (tous les cycles de tous les moteurs).
    plafond : float, optionnel
        Plafond pour "rmse_contre_rul_plafonnee" (125 par defaut).
    seuil_surestimation : float, optionnel
        Seuil des surestimations dangereuses (20 cycles par defaut).
    seuil_alerte : float, optionnel
        Seuil d'intervention pour les alertes (30 cycles par defaut, hypothese de travail).

    Returns
    -------
    dict
        rmse, rmse_contre_rul_plafonnee, mae, score_nasa, nb_surestimes, alertes_manquees,
        fausses_alertes, nb_moteurs (types Python simples).
    """
    derniers = extraire_dernier_cycle(df)  # une ligne par moteur : son dernier cycle observe
    vrais = derniers["rul_vraie"].to_numpy(dtype=float)
    predits = np.clip(derniers["rul_predite"].to_numpy(dtype=float), a_min=0, a_max=None)
    vrais_plafonnes = np.minimum(vrais, plafond)

    resultats = {}
    resultats["rmse"] = rmse(vrais, predits)
    resultats["rmse_contre_rul_plafonnee"] = rmse(vrais_plafonnes, predits)
    resultats["mae"] = mae(vrais, predits)
    resultats["score_nasa"] = score_nasa(vrais, predits)
    resultats["nb_surestimes"] = nb_surestimes(vrais, predits, seuil=seuil_surestimation)
    resultats["alertes_manquees"] = alertes_manquees(vrais, predits, seuil_alerte=seuil_alerte)
    resultats["fausses_alertes"] = fausses_alertes(vrais, predits, seuil_alerte=seuil_alerte)
    resultats["nb_moteurs"] = int(len(derniers))
    return resultats


def verifier_coherence(df, metadonnees, tolerance_relative=1e-4):
    """
    Compare les metriques de test des metadonnees a celles recalculees depuis le fichier de predictions.

    Sert de preuve dans le tableau de bord : les chiffres affiches ne sont pas recopies a la main,
    ils se recalculent a l'identique a partir des predictions.

    Parameters
    ----------
    df : pandas.DataFrame
        Le fichier de predictions.
    metadonnees : dict
        Ses metadonnees (cle "metriques_test").
    tolerance_relative : float, optionnel
        Ecart relatif tolere (1e-4 par defaut : arrondis d'ecriture du CSV).

    Returns
    -------
    pandas.DataFrame
        Une ligne par metrique obligatoire : metrique, metadonnees, recalculee, ecart, coherent (bool).
    """
    recalculees = metriques_dernier_cycle(df)
    lignes = []
    for cle in METRIQUES_TEST_OBLIGATOIRES:
        valeur_meta = float(metadonnees["metriques_test"][cle])
        valeur_recalculee = float(recalculees[cle])
        ecart = valeur_recalculee - valeur_meta
        # tolerance relative (avec un plancher absolu pour les valeurs proches de 0)
        limite = tolerance_relative * max(1.0, abs(valeur_meta))
        ligne = {}
        ligne["metrique"] = cle
        ligne["metadonnees"] = valeur_meta
        ligne["recalculee"] = valeur_recalculee
        ligne["ecart"] = ecart
        ligne["coherent"] = bool(abs(ecart) <= limite)
        lignes.append(ligne)
    return pd.DataFrame(lignes)
