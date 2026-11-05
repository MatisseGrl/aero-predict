"""
Preparation des quatre sous-jeux C-MAPSS (FD001 a FD004) avec une seule et meme interface.

Pourquoi ce module ? Le pipeline valide des Phases 1 a 3 (src/preprocessing.py, src/features.py) ne
traite que FD001 : un seul regime operatoire (les trois reglages operatoires sont quasi constants).
FD002 et FD004 melangent six regimes : d'un cycle a l'autre, le meme moteur vole dans des conditions
differentes, et les capteurs sautent d'un niveau a un autre sans que le moteur se soit degrade. Ce module
generalise la preparation sans toucher au code valide :

1. regime operatoire de chaque ligne (sous-jeux a plusieurs regimes) : K-Means sur les 3 reglages,
   appris sur le TRAIN seulement, puis `kmeans.predict` sur toute nouvelle donnee ;
2. capteurs retires : ceux qui ne prennent qu'une seule valeur (nunique() == 1, critere de la Phase 1),
   sur tout le train si la normalisation est globale, A L'INTERIEUR DE CHAQUE regime sinon ;
3. normalisation min-max : un seul scaler (globale, FD001 et FD003) ou un scaler par regime
   ("par_regime", FD002 et FD004), toujours appris sur le train et applique en `transform` seul ;
4. PUIS lissage (moyenne mobile causale) et features glissantes, moteur par moteur. L'ordre compte :
   lisser des valeurs brutes de FD002 ferait la moyenne de cycles voles dans des regimes differents,
   donc de niveaux de capteur qui n'ont rien a voir avec l'usure ; une fois normalises par regime, les
   cycles successifs sont comparables et la moyenne mobile ne lisse plus que le bruit ;
5. RUL brute puis plafonnee (train), RUL vraie a chaque cycle (test), comme src/features.py ;
6. option `ecart_initial` (notebook 04e) : ecart a l'etat initial de chaque capteur (src/features.py,
   `calculer_ecart_etat_initial`), calcule APRES la normalisation (par regime pour FD002 / FD004) et le
   lissage, moteur par moteur ; X devient alors capteurs lisses + time_cycles + ecarts (+ one-hot du regime).
   Sans l'option (defaut), rien ne change.

Pour FD001, le resultat est identique a `preparer_train` / `preparer_test` de src/features.py
(meme X, meme y, memes groupes, memes colonnes dans le meme ordre) : verifie par le notebook 01b.
Le lissage et les features glissantes sont recodes ici bloc par bloc (un moteur a la fois) pour aller
plus vite sur FD002 et FD004 (2,5 a 3 fois plus de lignes et de moteurs), avec exactement les memes
calculs pandas (rolling causal, min_periods=1) : l'egalite FD001 le prouve.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

from src.features import (
    ajouter_ecart_etat_initial,
    ajouter_rul_vraie_test,
    construire_liste_features,
    construire_liste_features_ecart_initial,
)
from src.preprocessing import COLUMNS, DATA_DIR, calculer_rul, find_constant_sensors, get_sensor_columns

# ---------------------------------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------------------------------

# les quatre sous-jeux, dans l'ordre d'affichage (meme liste que src/contrat_predictions.py)
SOUS_JEUX = ["FD001", "FD002", "FD003", "FD004"]

# les deux fichiers possibles de chaque sous-jeu
SPLITS_VALIDES = ["train", "test"]

# les trois reglages operatoires (colonnes 3 a 5 du fichier brut, data/readme.txt)
COLONNES_REGLAGES = ["op_setting_1", "op_setting_2", "op_setting_3"]

# les deux normalisations possibles
NORMALISATIONS_VALIDES = ["globale", "par_regime"]

# reglages du K-Means (fixes pour la reproductibilite : memes regimes a chaque execution)
GRAINE_KMEANS = 42
N_INIT_KMEANS = 10  # nombre d'initialisations differentes ; K-Means garde la meilleure (plus petite inertie)

# Description de chaque sous-jeu.
# - nb_moteurs_* et nb_lignes_* : COMPTES sur les fichiers data/train_FD00X.txt et data/test_FD00X.txt
#   (verifies par le notebook 01b) ;
# - nb_regimes : data/readme.txt ("Conditions: ONE / SIX"), confirme sur les donnees par K-Means
#   (silhouette maximale a k = 6 pour FD002 et FD004, voir `choisir_nombre_regimes` et le notebook 01b) ;
# - modes_de_panne : data/readme.txt ("Fault Modes"), NON verifiable sur les fichiers (aucune colonne ne
#   dit quel mode de panne touche quel moteur).
# Attention : data/readme.txt annonce pour FD004 248 moteurs train et 249 moteurs test ; les fichiers en
# contiennent l'inverse (249 train, 248 test, et RUL_FD004.txt a bien 248 lignes). On garde les comptes
# des fichiers, ce sont eux qui sont utilises.
DESCRIPTION_SOUS_JEUX = {
    "FD001": {
        "nb_regimes": 1,
        "modes_de_panne": ["HPC Degradation"],
        "nb_moteurs_train": 100,
        "nb_moteurs_test": 100,
        "nb_lignes_train": 20631,
        "nb_lignes_test": 13096,
    },
    "FD002": {
        "nb_regimes": 6,
        "modes_de_panne": ["HPC Degradation"],
        "nb_moteurs_train": 260,
        "nb_moteurs_test": 259,
        "nb_lignes_train": 53759,
        "nb_lignes_test": 33991,
    },
    "FD003": {
        "nb_regimes": 1,
        "modes_de_panne": ["HPC Degradation", "Fan Degradation"],
        "nb_moteurs_train": 100,
        "nb_moteurs_test": 100,
        "nb_lignes_train": 24720,
        "nb_lignes_test": 16596,
    },
    "FD004": {
        "nb_regimes": 6,
        "modes_de_panne": ["HPC Degradation", "Fan Degradation"],
        "nb_moteurs_train": 249,
        "nb_moteurs_test": 248,
        "nb_lignes_train": 61249,
        "nb_lignes_test": 41214,
    },
}

# Faut-il ajouter le regime courant comme feature (one-hot) ? Decide par la validation croisee
# (scripts/generer_predictions_arbres.py --ablations, sorties/resultats/01b_ablations_preparation.csv),
# jamais par le test. Regle fixee avant les ablations : seulement si le RMSE CV de XGBoost Phase 3 baisse
# ET le test de Wilcoxon apparie par moteur donne p < 0,05. Valeur par defaut par sous-jeu.
REGIME_EN_FEATURE_PAR_DEFAUT = {
    "FD001": False,
    "FD002": False,
    "FD003": False,
    "FD004": False,
}


# ---------------------------------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------------------------------

def verifier_sous_jeu(sous_jeu):
    """
    Verifie qu'un nom de sous-jeu fait partie de SOUS_JEUX.

    Parameters
    ----------
    sous_jeu : str
        Le nom a verifier (ex. "FD002").

    Raises
    ------
    ValueError
        Si le sous-jeu est inconnu.
    """
    if sous_jeu not in SOUS_JEUX:
        raise ValueError(f"Sous-jeu inconnu : {sous_jeu!r}. Sous-jeux possibles : {', '.join(SOUS_JEUX)}.")


def charger_sous_jeu(sous_jeu, split):
    """
    Charge le fichier brut train ou test d'un sous-jeu, avec les 26 noms de colonnes de src/preprocessing.py.

    Meme lecture que `load_fd001` (src/preprocessing.py), pour n'importe quel sous-jeu.

    Parameters
    ----------
    sous_jeu : str
        "FD001", "FD002", "FD003" ou "FD004".
    split : str
        "train" ou "test".

    Returns
    -------
    pandas.DataFrame
        Une ligne par (moteur, cycle), 26 colonnes : unit_number, time_cycles, op_setting_1 a 3,
        sensor_1 a sensor_21.

    Raises
    ------
    ValueError
        Si le sous-jeu ou le split est inconnu.
    """
    verifier_sous_jeu(sous_jeu)
    if split not in SPLITS_VALIDES:
        raise ValueError(f"Split inconnu : {split!r}. Valeurs possibles : {', '.join(SPLITS_VALIDES)}.")

    chemin = f"{DATA_DIR}/{split}_{sous_jeu}.txt"  # ex. <racine>/data/train_FD002.txt
    df = pd.read_csv(
        chemin,
        sep=r"\s+",     # separateur : espaces de largeur variable (comme FD001)
        header=None,    # pas de ligne d'en-tete dans le fichier brut
        names=COLUMNS,  # les 26 noms de colonnes deja definis pour FD001
    )
    return df


def charger_rul_sous_jeu(sous_jeu):
    """
    Charge RUL_FD00X.txt : la vraie RUL au dernier cycle observe de chaque moteur test.

    Ligne 1 = moteur 1, ligne 2 = moteur 2, etc. (data/readme.txt), comme `charger_rul_test` pour FD001.

    Parameters
    ----------
    sous_jeu : str
        "FD001", "FD002", "FD003" ou "FD004".

    Returns
    -------
    numpy.ndarray
        Les RUL vraies (entiers, non plafonnees), dans l'ordre croissant des numeros de moteur test.
    """
    verifier_sous_jeu(sous_jeu)
    chemin = f"{DATA_DIR}/RUL_{sous_jeu}.txt"
    rul = pd.read_csv(chemin, sep=r"\s+", header=None, names=["RUL_vraie"])
    return rul["RUL_vraie"].values  # tableau numpy, comme pour FD001


# ---------------------------------------------------------------------------------------------------
# Regimes operatoires
# ---------------------------------------------------------------------------------------------------

def creer_kmeans(nb_regimes):
    """
    Cree un K-Means non entraine avec les reglages du projet (graine 42, 10 initialisations).

    Reproductibilite : avec la meme graine, deux entrainements donnent les memes groupes et les memes
    numeros de regime, mais les centres peuvent differer d'environ 1e-15 (moyennes calculees en
    parallele, ordre des additions flottantes variable). Seuls les numeros de regime servent ensuite.

    Parameters
    ----------
    nb_regimes : int
        Le nombre de groupes (clusters) a chercher.

    Returns
    -------
    sklearn.cluster.KMeans
        Le modele, non entraine.
    """
    return KMeans(n_clusters=nb_regimes, random_state=GRAINE_KMEANS, n_init=N_INIT_KMEANS)


def choisir_nombre_regimes(reglages, valeurs_k=None, taille_echantillon=5000):
    """
    Mesure, pour plusieurs nombres de groupes k, la qualite d'un K-Means sur les reglages operatoires.

    Deux indicateurs, pour justifier k par les donnees plutot que de le recopier :
    - l'inertie : somme des distances au carre de chaque point au centre de son groupe (plus petit =
      groupes plus serres). Elle baisse toujours quand k augmente ; ce qui compte, c'est un "coude" net ;
    - le coefficient de silhouette (entre -1 et 1) : pour chaque point, compare la distance moyenne aux
      points de son groupe et la distance moyenne au groupe voisin le plus proche. Proche de 1 = groupes
      bien separes ; il est MAXIMAL pour le bon nombre de groupes. Calcule sur un echantillon tire avec
      une graine fixe (le calcul exact sur 50 000 points compare toutes les paires : trop long).

    Parameters
    ----------
    reglages : pandas.DataFrame
        Les colonnes de reglages operatoires du TRAIN (op_setting_1 a 3).
    valeurs_k : list de int, optionnel
        Les nombres de groupes a tester (2 a 10 par defaut).
    taille_echantillon : int, optionnel
        Nombre de points utilises pour la silhouette (5 000 par defaut).

    Returns
    -------
    pandas.DataFrame
        Une ligne par k : "k", "inertie", "silhouette".
    """
    if valeurs_k is None:
        valeurs_k = list(range(2, 11))  # 2, 3, ..., 10

    taille = min(taille_echantillon, len(reglages))  # pas plus de points qu'il n'y en a
    lignes = []
    for k in valeurs_k:  # un nombre de groupes a la fois
        kmeans = creer_kmeans(k)
        kmeans.fit(reglages)  # apprend les k centres sur les reglages
        silhouette = silhouette_score(reglages, kmeans.labels_, sample_size=taille, random_state=GRAINE_KMEANS)
        ligne = {}
        ligne["k"] = k
        ligne["inertie"] = float(kmeans.inertia_)
        ligne["silhouette"] = float(silhouette)
        lignes.append(ligne)
    return pd.DataFrame(lignes)


def apprendre_regimes(train_brut, nb_regimes):
    """
    Apprend les regimes operatoires sur le TRAIN : K-Means sur les trois reglages operatoires.

    Pour un sous-jeu a un seul regime (FD001, FD003), il n'y a rien a apprendre : on renvoie None et
    toutes les lignes recevront le regime 0.

    Parameters
    ----------
    train_brut : pandas.DataFrame
        Le train brut du sous-jeu (au moins les colonnes de reglages).
    nb_regimes : int
        Le nombre de regimes (1, ou 6 pour FD002 et FD004).

    Returns
    -------
    sklearn.cluster.KMeans ou None
        Le K-Means entraine sur le train, ou None s'il n'y a qu'un regime.
    """
    if nb_regimes == 1:
        return None  # un seul regime : pas de clustering

    kmeans = creer_kmeans(nb_regimes)
    kmeans.fit(train_brut[COLONNES_REGLAGES])  # appris sur le train SEULEMENT
    return kmeans


def attribuer_regimes(df, kmeans):
    """
    Donne a chaque ligne son numero de regime, avec un K-Means deja appris (predict seul, jamais fit).

    Parameters
    ----------
    df : pandas.DataFrame
        Des lignes brutes (au moins les colonnes de reglages operatoires).
    kmeans : sklearn.cluster.KMeans ou None
        Le K-Means appris sur le train, ou None pour un sous-jeu a un seul regime.

    Returns
    -------
    numpy.ndarray d'entiers
        Le regime de chaque ligne (0 partout si kmeans est None), dans l'ordre des lignes de df.
    """
    if kmeans is None:
        return np.zeros(len(df), dtype=int)  # un seul regime : le regime 0
    regimes = kmeans.predict(df[COLONNES_REGLAGES])  # le centre le plus proche de chaque ligne
    return np.asarray(regimes, dtype=int)


def tableau_valeurs_distinctes_par_regime(df, capteurs, colonne_regime="regime"):
    """
    Compte le nombre de valeurs distinctes de chaque capteur a l'interieur de chaque regime.

    Sert a documenter le critere de retrait des capteurs (notebook 01b) : un capteur qui ne prend qu'une
    seule valeur dans chaque regime ne porte aucune information une fois le regime connu.

    Parameters
    ----------
    df : pandas.DataFrame
        Les lignes du train, avec la colonne de regime.
    capteurs : list de str
        Les colonnes capteurs a examiner.
    colonne_regime : str, optionnel
        Le nom de la colonne de regime ("regime" par defaut).

    Returns
    -------
    pandas.DataFrame
        Une ligne par capteur, une colonne par regime (nombre de valeurs distinctes).
    """
    regimes = np.sort(df[colonne_regime].unique())  # numeros de regime, du plus petit au plus grand
    tableau = pd.DataFrame(index=capteurs)
    for regime in regimes:  # un regime a la fois
        lignes_regime = df[df[colonne_regime] == regime]
        valeurs_distinctes = lignes_regime[capteurs].nunique()  # nombre de valeurs distinctes par capteur
        tableau[f"regime_{regime}"] = valeurs_distinctes
    return tableau


def trouver_capteurs_constants_par_regime(df, capteurs, colonne_regime="regime"):
    """
    Identifie les capteurs constants DANS CHAQUE regime (une seule valeur distincte par regime).

    Generalisation directe du critere de la Phase 1 (`find_constant_sensors`, nunique() == 1) : dans
    FD002 et FD004, aucun capteur n'est constant sur tout le train (son niveau change avec le regime),
    mais certains ne bougent jamais a regime fixe. Un capteur est retire s'il est constant dans TOUS les
    regimes : il ne dit alors rien de plus que le regime lui-meme. Un capteur constant dans certains
    regimes seulement est garde (il varie, donc peut porter de l'information, dans les autres).

    Parameters
    ----------
    df : pandas.DataFrame
        Les lignes du train, avec la colonne de regime.
    capteurs : list de str
        Les colonnes capteurs a tester.
    colonne_regime : str, optionnel
        Le nom de la colonne de regime ("regime" par defaut).

    Returns
    -------
    list de str
        Les capteurs constants dans tous les regimes, dans l'ordre de `capteurs`.
    """
    valeurs_distinctes = tableau_valeurs_distinctes_par_regime(df, capteurs, colonne_regime)

    capteurs_constants = []
    for capteur in capteurs:  # un capteur a la fois
        maximum_sur_les_regimes = valeurs_distinctes.loc[capteur].max()  # le regime ou il varie le plus
        if maximum_sur_les_regimes == 1:  # une seule valeur dans chaque regime
            capteurs_constants.append(capteur)
    return capteurs_constants


def decrire_regimes(contexte):
    """
    Resume les regimes appris : centre de chaque groupe (valeurs des reglages) et nombre de lignes.

    Parameters
    ----------
    contexte : dict
        Le contexte renvoye par `preparer_sous_jeu`.

    Returns
    -------
    pandas.DataFrame
        Une ligne par regime : les 3 reglages au centre du groupe (moyenne sur le train pour un sous-jeu
        a un regime), le nombre de lignes train et test de ce regime.
    """
    train = contexte["train"]
    test = contexte["test"]
    lignes = []
    for regime in range(contexte["nb_regimes"]):  # un regime a la fois
        ligne = {}
        ligne["regime"] = regime
        for position in range(len(COLONNES_REGLAGES)):
            colonne = COLONNES_REGLAGES[position]
            if contexte["kmeans"] is None:
                ligne[colonne] = float(train[colonne].mean())  # un seul regime : moyenne des reglages
            else:
                ligne[colonne] = float(contexte["kmeans"].cluster_centers_[regime, position])
        ligne["nb_lignes_train"] = int((train["regime"] == regime).sum())
        ligne["nb_lignes_test"] = int((test["regime"] == regime).sum())
        lignes.append(ligne)
    return pd.DataFrame(lignes)


# ---------------------------------------------------------------------------------------------------
# Normalisation, lissage et features glissantes
# ---------------------------------------------------------------------------------------------------

def apprendre_scalers(train, capteurs, normalisation, nb_regimes):
    """
    Apprend la normalisation min-max sur le TRAIN : un seul scaler (globale) ou un scaler par regime.

    Parameters
    ----------
    train : pandas.DataFrame
        Le train, avec la colonne "regime" et les colonnes capteurs (capteurs constants deja retires).
    capteurs : list de str
        Les colonnes capteurs a normaliser.
    normalisation : str
        "globale" : un seul MinMaxScaler appris sur toutes les lignes (methode FD001) ;
        "par_regime" : un MinMaxScaler par regime, appris sur les seules lignes de ce regime.
    nb_regimes : int
        Le nombre de regimes.

    Returns
    -------
    dict
        regime (int) -> MinMaxScaler entraine. En normalisation globale, le MEME scaler est range sous
        chaque numero de regime (une seule regle pour toutes les lignes).
    """
    scalers = {}

    if normalisation == "globale":
        scaler = MinMaxScaler()
        scaler.fit(train[capteurs])  # minimums et maximums de chaque capteur sur tout le train
        for regime in range(nb_regimes):
            scalers[regime] = scaler  # meme regle quel que soit le regime
        return scalers

    # normalisation par regime : minimums et maximums appris separement dans chaque regime
    for regime in range(nb_regimes):
        lignes_regime = train[train["regime"] == regime]
        assert len(lignes_regime) > 0, f"aucune ligne d'entrainement dans le regime {regime}"
        scaler = MinMaxScaler()
        scaler.fit(lignes_regime[capteurs])
        scalers[regime] = scaler
    return scalers


def normaliser_par_regime(df, capteurs, scalers):
    """
    Applique a chaque ligne le scaler de son regime (transform seul : rien n'est appris ici).

    Parameters
    ----------
    df : pandas.DataFrame
        Les lignes a normaliser, avec la colonne "regime" (n'est pas modifie, une copie est renvoyee).
    capteurs : list de str
        Les colonnes capteurs a normaliser.
    scalers : dict
        regime -> MinMaxScaler appris sur le train (voir `apprendre_scalers`).

    Returns
    -------
    pandas.DataFrame
        Copie de df avec les capteurs normalises.

    Raises
    ------
    ValueError
        Si une ligne a un regime sans scaler.
    """
    df_normalise = df.copy()
    valeurs = df_normalise[capteurs].to_numpy(dtype=float, copy=True)  # tableau (lignes x capteurs)
    regimes = df_normalise["regime"].to_numpy()

    for regime in np.unique(regimes):  # seulement les regimes presents dans ces lignes
        if regime not in scalers:
            raise ValueError(f"Aucun scaler appris pour le regime {regime}.")
        masque = regimes == regime  # Vrai sur les lignes de ce regime
        lignes_regime = df_normalise.loc[masque, capteurs]  # DataFrame : garde les noms de colonnes du fit
        valeurs[masque, :] = scalers[regime].transform(lignes_regime)  # minimums / maximums du TRAIN

    df_normalise[capteurs] = valeurs
    return df_normalise


def positions_par_moteur(df):
    """
    Renvoie, pour chaque moteur, les positions (0, 1, 2, ...) de ses lignes dans df, dans leur ordre.

    Parameters
    ----------
    df : pandas.DataFrame
        Au moins la colonne "unit_number".

    Returns
    -------
    dict
        numero de moteur -> numpy.ndarray des positions de ses lignes (ordre croissant = ordre du fichier).
    """
    return df.groupby("unit_number", sort=False).indices


def lisser_par_moteur(df, capteurs, fenetre):
    """
    Moyenne mobile causale des capteurs, moteur par moteur : memes calculs que `lisser_capteurs`.

    Memes regles que src/preprocessing.py : causale (le cycle t n'utilise que les cycles <= t, rolling()
    de pandas) et jamais a travers deux moteurs. Seule difference : on traite un moteur a la fois avec
    tous ses capteurs d'un coup (un seul rolling par moteur), au lieu d'un masque par capteur et par
    moteur : meme resultat, beaucoup plus rapide sur FD002 et FD004.

    Parameters
    ----------
    df : pandas.DataFrame
        Les lignes a lisser (n'est pas modifie, une copie est renvoyee).
    capteurs : list de str
        Les colonnes capteurs a lisser.
    fenetre : int
        Nombre de cycles de la moyenne mobile.

    Returns
    -------
    pandas.DataFrame
        Copie de df avec les capteurs lisses.
    """
    df_lisse = df.copy()
    tableau_capteurs = df_lisse[capteurs]  # les colonnes capteurs seules
    valeurs_lissees = tableau_capteurs.to_numpy(dtype=float, copy=True)

    positions = positions_par_moteur(df_lisse)
    for moteur_id in positions:  # un moteur a la fois
        lignes_moteur = positions[moteur_id]  # positions des lignes de ce moteur
        bloc = tableau_capteurs.iloc[lignes_moteur]  # ses cycles, dans l'ordre du fichier
        moyenne = bloc.rolling(fenetre, min_periods=1).mean()  # causale, sur ce moteur seulement
        valeurs_lissees[lignes_moteur, :] = moyenne.to_numpy()

    df_lisse[capteurs] = valeurs_lissees
    return df_lisse


def ajouter_features_glissantes_par_moteur(df, capteurs, fenetre):
    """
    Ajoute la moyenne et l'ecart-type glissants de chaque capteur, moteur par moteur.

    Memes calculs, memes noms et meme ordre de colonnes que `construire_features_glissantes`
    (src/features.py) : "<capteur>_glissante_moy" puis "<capteur>_glissante_std", capteur par capteur ;
    l'ecart-type du 1er cycle de chaque moteur (une seule valeur : NaN) est remplace par 0, apres avoir
    verifie qu'il y a exactement un NaN par moteur et par capteur.

    Parameters
    ----------
    df : pandas.DataFrame
        Les lignes deja lissees (n'est pas modifie, une copie est renvoyee).
    capteurs : list de str
        Les colonnes capteurs.
    fenetre : int
        Nombre de cycles de la fenetre glissante.

    Returns
    -------
    (pandas.DataFrame, list de str, list de str)
        Le tableau enrichi, la liste des colonnes "moyenne glissante" et celle des colonnes
        "ecart-type glissant" (dans l'ordre des capteurs).
    """
    df_features = df.copy()
    tableau_capteurs = df_features[capteurs]
    nb_lignes = len(df_features)
    moyennes = np.zeros((nb_lignes, len(capteurs)), dtype=float)  # remplies moteur par moteur
    ecarts_types = np.zeros((nb_lignes, len(capteurs)), dtype=float)

    positions = positions_par_moteur(df_features)
    for moteur_id in positions:  # un moteur a la fois
        lignes_moteur = positions[moteur_id]
        bloc = tableau_capteurs.iloc[lignes_moteur]
        fenetres = bloc.rolling(fenetre, min_periods=1)  # fenetre causale sur ce moteur seulement
        moyennes[lignes_moteur, :] = fenetres.mean().to_numpy()
        ecarts_types[lignes_moteur, :] = fenetres.std().to_numpy()  # NaN au 1er cycle (une seule valeur)

    # on verifie le nombre de NaN avant de les remplacer : 1 par moteur et par capteur, pas un de plus
    nb_nan = int(np.isnan(ecarts_types).sum())
    nb_nan_attendu = len(positions) * len(capteurs)
    assert nb_nan == nb_nan_attendu, f"attendu {nb_nan_attendu} NaN (1er cycle de chaque moteur), trouve {nb_nan}"
    ecarts_types[np.isnan(ecarts_types)] = 0.0  # aucune dispersion observee au 1er cycle : 0

    # ajout des colonnes dans le meme ordre que src/features.py : moyenne puis ecart-type, capteur par capteur
    colonnes_moyenne = []
    colonnes_ecart_type = []
    for position in range(len(capteurs)):
        capteur = capteurs[position]
        colonne_moyenne = capteur + "_glissante_moy"
        colonne_ecart_type = capteur + "_glissante_std"
        df_features[colonne_moyenne] = moyennes[:, position]
        df_features[colonne_ecart_type] = ecarts_types[:, position]
        colonnes_moyenne.append(colonne_moyenne)
        colonnes_ecart_type.append(colonne_ecart_type)

    return df_features, colonnes_moyenne, colonnes_ecart_type


def colonnes_one_hot_regime(nb_regimes):
    """
    Renvoie les noms des colonnes one-hot du regime : "regime_0", "regime_1", ...

    Parameters
    ----------
    nb_regimes : int
        Le nombre de regimes.

    Returns
    -------
    list de str
        Un nom de colonne par regime.
    """
    colonnes = []
    for regime in range(nb_regimes):
        colonnes.append(f"regime_{regime}")
    return colonnes


def ajouter_one_hot_regime(df, nb_regimes):
    """
    Ajoute une colonne 0/1 par regime ("one-hot") : regime_r vaut 1 si la ligne est dans le regime r.

    Exemple avec 3 regimes : une ligne du regime 2 recoit regime_0 = 0, regime_1 = 0, regime_2 = 1.
    Utilise seulement si la validation croisee le justifie (voir REGIME_EN_FEATURE_PAR_DEFAUT).

    Parameters
    ----------
    df : pandas.DataFrame
        Les lignes, avec la colonne "regime" (n'est pas modifie, une copie est renvoyee).
    nb_regimes : int
        Le nombre de regimes (une colonne par regime, meme si un regime est absent de df).

    Returns
    -------
    (pandas.DataFrame, list de str)
        La copie enrichie et la liste des colonnes ajoutees.
    """
    df_one_hot = df.copy()
    colonnes = colonnes_one_hot_regime(nb_regimes)
    regimes = df_one_hot["regime"].to_numpy()
    for regime in range(nb_regimes):
        df_one_hot[colonnes[regime]] = (regimes == regime).astype(float)  # 1.0 si ce regime, 0.0 sinon
    return df_one_hot, colonnes


# ---------------------------------------------------------------------------------------------------
# Preparation complete
# ---------------------------------------------------------------------------------------------------

def appliquer_preparation(contexte, df_brut):
    """
    Applique a de nouvelles donnees BRUTES exactement les transformations apprises sur le train.

    Rien n'est reappris : regime par `kmeans.predict`, capteurs retires = ceux trouves sur le train,
    scaler du regime en `transform` seul, puis lissage et features glissantes moteur par moteur (puis
    ecart a l'etat initial et one-hot du regime si le contexte les utilise). Chaque etape travaille moteur par moteur : preparer un
    seul moteur ou toute la flotte donne exactement les memes valeurs pour ce moteur.

    Parameters
    ----------
    contexte : dict
        Le contexte renvoye par `preparer_sous_jeu` (on y lit kmeans, capteurs_retires, capteurs,
        scalers, fenetre, nb_regimes, regime_en_feature et, s'il existe, ecart_initial).
    df_brut : pandas.DataFrame
        Des lignes au format brut (colonnes de src/preprocessing.COLUMNS), un ou plusieurs moteurs,
        cycles dans l'ordre pour chaque moteur. Non modifie. L'index doit etre sans doublon.

    Returns
    -------
    pandas.DataFrame
        Les donnees preparees (meme index que df_brut) : colonne "regime", capteurs normalises puis
        lisses, features glissantes (et colonnes "<capteur>_ecartinit_<k>" et one-hot du regime le cas
        echeant).

    Raises
    ------
    ValueError
        Si une colonne necessaire manque, si l'index a des doublons, ou si une valeur manque.
    """
    capteurs = contexte["capteurs"]
    fenetre = contexte["fenetre"]

    # 1. controles d'entree (message clair plutot qu'une erreur obscure plus loin)
    colonnes_necessaires = ["unit_number", "time_cycles"]
    for colonne in COLONNES_REGLAGES:
        colonnes_necessaires.append(colonne)
    for colonne in capteurs:
        colonnes_necessaires.append(colonne)
    colonnes_manquantes = []
    for colonne in colonnes_necessaires:
        if colonne not in df_brut.columns:
            colonnes_manquantes.append(colonne)
    if len(colonnes_manquantes) > 0:
        raise ValueError(f"Colonnes manquantes dans les donnees brutes : {', '.join(colonnes_manquantes)}.")
    if not df_brut.index.is_unique:
        raise ValueError("L'index des donnees brutes contient des doublons : utiliser reset_index(drop=True).")
    nb_manquants = int(df_brut[colonnes_necessaires].isna().sum().sum())
    if nb_manquants > 0:
        raise ValueError(f"{nb_manquants} valeur(s) manquante(s) dans les colonnes necessaires.")

    # 2. capteurs retires : exactement ceux trouves sur le train (jamais recalcules sur ces donnees)
    colonnes_a_retirer = []
    for capteur in contexte["capteurs_retires"]:
        if capteur in df_brut.columns:
            colonnes_a_retirer.append(capteur)
    donnees = df_brut.drop(columns=colonnes_a_retirer)

    # 3. regime de chaque ligne : K-Means appris sur le train, predict seul (0 partout si un seul regime)
    position_regime = list(donnees.columns).index("op_setting_3") + 1  # juste apres les reglages
    donnees.insert(position_regime, "regime", attribuer_regimes(donnees, contexte["kmeans"]))

    # 4. normalisation : scaler du regime de chaque ligne, transform seul
    donnees = normaliser_par_regime(donnees, capteurs, contexte["scalers"])

    # 5. lissage puis features glissantes, moteur par moteur (APRES la normalisation, voir l'en-tete)
    donnees = lisser_par_moteur(donnees, capteurs, fenetre)
    donnees, colonnes_moyenne, colonnes_ecart_type = ajouter_features_glissantes_par_moteur(donnees, capteurs, fenetre)

    # 6. option : ecart a l'etat initial, sur les capteurs normalises (par regime si besoin) PUIS lisses ;
    #    .get : un contexte sans cette cle (construit avant l'option) equivaut a None, comportement inchange
    ecart_initial = contexte.get("ecart_initial")
    if ecart_initial is not None:
        donnees, _ = ajouter_ecart_etat_initial(donnees, capteurs, ecart_initial)  # noms deja connus du contexte

    # 7. regime en one-hot, seulement si la validation croisee l'a justifie
    if contexte["regime_en_feature"]:
        donnees, colonnes_regime = ajouter_one_hot_regime(donnees, contexte["nb_regimes"])

    return donnees


def preparer_sous_jeu(sous_jeu, fenetre=5, plafond=125, normalisation=None, regime_en_feature=None,
                      ecart_initial=None):
    """
    Prepare le train et le test d'un sous-jeu et renvoie tout ce qu'il faut pour entrainer et evaluer.

    Etapes : chargement -> regimes (K-Means appris sur le train si plusieurs regimes) -> capteurs
    constants retires (appris sur le train) -> scalers appris sur le train -> pour le train ET le test,
    `appliquer_preparation` (normalisation, lissage, features) -> RUL brute puis plafonnee (train) ->
    RUL vraie a chaque cycle (test) -> X, y, groupes, dernier cycle de chaque moteur test.

    Parameters
    ----------
    sous_jeu : str
        "FD001", "FD002", "FD003" ou "FD004".
    fenetre : int, optionnel
        Taille de la fenetre (lissage ET features glissantes), 5 par defaut comme en Phase 3.
    plafond : int, optionnel
        Plafond de la RUL (Phase 2), 125 par defaut.
    normalisation : str ou None, optionnel
        "globale" ou "par_regime" ; None (defaut) : "globale" pour un sous-jeu a un regime (FD001, FD003),
        "par_regime" sinon (FD002, FD004). Forcer "globale" sur FD002 / FD004 sert a mesurer ce que
        change la normalisation conditionnelle (ablation).
    regime_en_feature : bool ou None, optionnel
        Ajouter le regime courant en one-hot aux features ; None (defaut) : REGIME_EN_FEATURE_PAR_DEFAUT.
    ecart_initial : int ou None, optionnel
        None (defaut) : features de la Phase 3, comportement inchange. Un entier k (20 dans le notebook
        04e) : ajoute "<capteur>_ecartinit_<k>" (calcule apres normalisation et lissage) et les features
        deviennent capteurs lisses + time_cycles + ecarts a l'etat initial (+ one-hot du regime si utilise).

    Returns
    -------
    dict
        Le "contexte", avec les cles :

        - "sous_jeu" : str ;
        - "train" : pandas.DataFrame, train enrichi (regime, capteurs normalises puis lisses, features
          glissantes, RUL, RUL_cappee) ;
        - "test" : pandas.DataFrame, test enrichi, TOUS les cycles (regime, capteurs, features,
          RUL_vraie et RUL_vraie_cappee a chaque cycle) ;
        - "X", "y", "groups" : features du train, cible RUL_cappee, numero de moteur de chaque ligne ;
        - "feature_cols" : list de str, ordre des features (capteurs lisses, moyennes glissantes,
          ecarts-types glissants, time_cycles, puis regime_0... si one-hot ; avec `ecart_initial` :
          capteurs lisses, time_cycles, ecarts a l'etat initial, puis regime_0... si one-hot) ;
        - "capteurs" : list de str, capteurs normalises puis lisses (entree des reseaux sequentiels) ;
        - "colonnes_moyenne", "colonnes_ecart_type", "colonnes_regime" : list de str ;
        - "X_test_dernier" : features du dernier cycle de chaque moteur test (meme ordre que X) ;
        - "y_test_vraie" : numpy.ndarray, RUL vraie brute de RUL_FD00X.txt ;
        - "dernier_cycle_par_moteur" : pandas.DataFrame, la ligne du dernier cycle de chaque moteur test ;
        - "moteurs_test" : numpy.ndarray, numeros de moteur dans le meme ordre que y_test_vraie ;
        - "normalisation" : "globale" ou "par_regime" ; "nb_regimes" : int ;
          "regime_en_feature" : bool ;
        - "kmeans" : sklearn.cluster.KMeans appris sur le train, ou None (un seul regime) ;
        - "scalers" : dict regime -> MinMaxScaler appris sur le train ;
        - "capteurs_retires" : list de str ; "fenetre" : int ; "plafond" : int ;
        - "ecart_initial" : int ou None ; "colonnes_ecart_initial" : list de str (vide sans l'option).

    Raises
    ------
    ValueError
        Si un parametre est invalide.
    """
    verifier_sous_jeu(sous_jeu)
    nb_regimes = DESCRIPTION_SOUS_JEUX[sous_jeu]["nb_regimes"]

    # reglages par defaut, puis verification
    if normalisation is None:
        if nb_regimes == 1:
            normalisation = "globale"
        else:
            normalisation = "par_regime"
    if normalisation not in NORMALISATIONS_VALIDES:
        raise ValueError(f"Normalisation inconnue : {normalisation!r} (valeurs possibles : {NORMALISATIONS_VALIDES}).")
    if regime_en_feature is None:
        regime_en_feature = REGIME_EN_FEATURE_PAR_DEFAUT[sous_jeu]
    if regime_en_feature and nb_regimes == 1:
        raise ValueError(f"{sous_jeu} n'a qu'un regime : le regime en feature serait une colonne constante.")

    # --- 1. ce qui est APPRIS sur le train : regimes, capteurs retires, scalers ---
    train_brut = charger_sous_jeu(sous_jeu, "train")
    kmeans = apprendre_regimes(train_brut, nb_regimes)  # None si un seul regime

    train_avec_regime = train_brut.copy()
    train_avec_regime["regime"] = attribuer_regimes(train_avec_regime, kmeans)

    capteurs_bruts = get_sensor_columns(train_brut.columns)  # les 21 capteurs
    if normalisation == "globale":
        # methode de la Phase 1 telle quelle : constant sur tout le train
        capteurs_retires = find_constant_sensors(train_avec_regime, capteurs_bruts)
    else:
        # constant a l'interieur de chaque regime
        capteurs_retires = trouver_capteurs_constants_par_regime(train_avec_regime, capteurs_bruts)

    capteurs = []  # capteurs gardes, dans l'ordre d'origine
    for capteur in capteurs_bruts:
        if capteur not in capteurs_retires:
            capteurs.append(capteur)

    scalers = apprendre_scalers(train_avec_regime, capteurs, normalisation, nb_regimes)

    # le contexte, rempli au fur et a mesure (appliquer_preparation y lit ce qui a ete appris)
    contexte = {}
    contexte["sous_jeu"] = sous_jeu
    contexte["nb_regimes"] = nb_regimes
    contexte["normalisation"] = normalisation
    contexte["regime_en_feature"] = bool(regime_en_feature)
    contexte["kmeans"] = kmeans
    contexte["scalers"] = scalers
    contexte["capteurs_retires"] = capteurs_retires
    contexte["capteurs"] = capteurs
    contexte["fenetre"] = fenetre
    contexte["plafond"] = plafond
    contexte["ecart_initial"] = ecart_initial

    # --- 2. train : memes transformations que pour toute nouvelle donnee, puis la cible ---
    train = appliquer_preparation(contexte, train_brut)
    train = calculer_rul(train)  # RUL brute par cycle (Phase 2)
    rul_brute = train["RUL"].to_numpy()
    train["RUL_cappee"] = np.minimum(rul_brute, plafond)  # meme regle que plafonner_rul (Phase 2)

    # noms des colonnes de features (meme ordre que src/features.py, one-hot a la fin)
    colonnes_moyenne = []
    colonnes_ecart_type = []
    for capteur in capteurs:
        colonnes_moyenne.append(capteur + "_glissante_moy")
        colonnes_ecart_type.append(capteur + "_glissante_std")
    feature_cols = construire_liste_features(capteurs, colonnes_moyenne, colonnes_ecart_type)
    colonnes_ecart_initial = []  # reste vide sans l'option : comportement par defaut inchange
    if ecart_initial is not None:
        # memes noms que `ajouter_ecart_etat_initial` (src/features.py), dans l'ordre des capteurs
        for capteur in capteurs:
            colonnes_ecart_initial.append(capteur + "_ecartinit_" + str(ecart_initial))
        # jeu du notebook 04e : capteurs lisses + time_cycles + ecarts a l'etat initial
        feature_cols = construire_liste_features_ecart_initial(capteurs, colonnes_ecart_initial)
    colonnes_regime = []
    if regime_en_feature:
        colonnes_regime = colonnes_one_hot_regime(nb_regimes)
        for colonne in colonnes_regime:
            feature_cols.append(colonne)

    X = train[feature_cols]
    y = train["RUL_cappee"]
    groups = train["unit_number"]

    # --- 3. test : transformations apprises, puis RUL vraie a chaque cycle ---
    test = appliquer_preparation(contexte, charger_sous_jeu(sous_jeu, "test"))
    y_test_vraie = charger_rul_sous_jeu(sous_jeu)
    test = ajouter_rul_vraie_test(test, y_test_vraie, plafond)

    # dernier cycle observe de chaque moteur, trie par numero de moteur (aligne avec RUL_FD00X.txt)
    indices_derniers_cycles = test.groupby("unit_number")["time_cycles"].idxmax()
    dernier_cycle_par_moteur = test.loc[indices_derniers_cycles].sort_values("unit_number").reset_index(drop=True)
    assert dernier_cycle_par_moteur.shape[0] == len(y_test_vraie), "un dernier cycle par moteur test attendu"
    assert np.array_equal(dernier_cycle_par_moteur["RUL_vraie"].values, y_test_vraie), (
        "RUL_vraie au dernier cycle differente de RUL_FD00X.txt : probleme d'alignement des moteurs"
    )
    X_test_dernier = dernier_cycle_par_moteur[feature_cols]

    # garde-fou : aucune valeur manquante dans les features ni dans la cible
    assert int(X.isna().sum().sum()) == 0, "NaN dans les features du train"
    assert int(y.isna().sum()) == 0, "NaN dans la cible du train"
    assert int(test[feature_cols].isna().sum().sum()) == 0, "NaN dans les features du test"

    contexte["train"] = train
    contexte["test"] = test
    contexte["X"] = X
    contexte["y"] = y
    contexte["groups"] = groups
    contexte["feature_cols"] = feature_cols
    contexte["colonnes_moyenne"] = colonnes_moyenne
    contexte["colonnes_ecart_type"] = colonnes_ecart_type
    contexte["colonnes_regime"] = colonnes_regime
    contexte["colonnes_ecart_initial"] = colonnes_ecart_initial
    contexte["X_test_dernier"] = X_test_dernier
    contexte["y_test_vraie"] = y_test_vraie
    contexte["dernier_cycle_par_moteur"] = dernier_cycle_par_moteur
    contexte["moteurs_test"] = dernier_cycle_par_moteur["unit_number"].values
    return contexte
