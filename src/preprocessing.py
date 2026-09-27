"""
Fonctions de pretraitement partagees pour le dataset NASA C-MAPSS (FD001).

Ce module regroupe les etapes deja ecrites et validees dans les notebooks
01_eda_preparation.ipynb (chargement, capteurs constants, normalisation,
lissage) et 02_target_engineering.ipynb (calcul de la RUL, plafonnement).
Il existe pour que le notebook de la Phase 3 puisse reutiliser ces etapes
sans les recopier une 3e fois - la logique elle-meme n'a pas change.
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

DATA_DIR = "../data"  # chemin relatif depuis notebooks/ vers le dossier data

# Noms des 26 colonnes du fichier brut NASA (voir data/readme.txt) - les fichiers texte
# n'ont pas de ligne d'en-tete, donc on doit fournir les noms de colonnes nous-memes
COLUMNS = [
    "unit_number",   # identifiant du moteur : 1 a 100 pour FD001 train
    "time_cycles",   # numero du cycle pour ce moteur, commence a 1
    "op_setting_1",  # reglage operationnel 1
    "op_setting_2",  # reglage operationnel 2
    "op_setting_3",  # reglage operationnel 3
]
for i in range(1, 22):  # on ajoute les 21 capteurs bruts, nommes sensor_1 a sensor_21
    COLUMNS.append(f"sensor_{i}")  # ici, on genere le nom de chaque capteur un par un


def load_fd001(split):
    """
    Charge le fichier train ou test de FD001 dans un DataFrame pandas.

    Parameters
    ----------
    split : str
        "train" ou "test" - selectionne quel fichier brut charger.

    Returns
    -------
    pandas.DataFrame
        Une ligne par (moteur, cycle), avec les 26 colonnes nommees.
    """
    path = f"{DATA_DIR}/{split}_FD001.txt"  # construit le chemin complet du fichier
    df = pd.read_csv(
        path,
        sep=r"\s+",     # le fichier brut est separe par des espaces (largeur variable), pas des virgules
        header=None,    # le fichier brut n'a pas de ligne d'en-tete
        names=COLUMNS,  # donc on fournit les noms de colonnes nous-memes
    )
    return df  # renvoie le tableau charge


def get_sensor_columns(columns):
    """
    Construit la liste des noms de colonnes capteurs (sensor_1, sensor_2, ...)
    presentes dans une liste de colonnes donnee.

    Parameters
    ----------
    columns : liste de str
        Les noms de colonnes a filtrer (typiquement df.columns).

    Returns
    -------
    list de str
        Les noms de colonnes qui commencent par "sensor_", dans l'ordre.
    """
    sensor_cols = []  # on part d'une liste vide
    for column_name in columns:  # on parcourt chaque nom de colonne, dans l'ordre
        if column_name.startswith("sensor_"):  # on ne garde que les colonnes capteurs
            sensor_cols.append(column_name)  # on l'ajoute a notre liste
    return sensor_cols


def find_constant_sensors(df, sensor_cols):
    """
    Identifie les capteurs constants (une seule valeur distincte sur tout le tableau).

    Un capteur constant sur l'ensemble des moteurs ne peut pas etre correle a la
    degradation : il ne varie jamais. On utilise nunique() == 1 plutot que
    variance == 0, a cause d'un probleme de precision flottante (observe sur
    sensor_5 et sensor_16 : valeur unique, mais variance calculee != 0 exact).

    Parameters
    ----------
    df : pandas.DataFrame
        Le tableau contenant les colonnes capteurs a tester.
    sensor_cols : list de str
        Les noms de colonnes capteurs a tester.

    Returns
    -------
    list de str
        Les noms des capteurs constants (nunique() == 1).
    """
    unique_values_per_sensor = df[sensor_cols].nunique()  # nombre de valeurs distinctes par capteur

    constant_sensors = []  # on part d'une liste vide
    for sensor_name in sensor_cols:  # on parcourt chaque capteur
        n_unique = unique_values_per_sensor[sensor_name]  # son nombre de valeurs distinctes
        if n_unique == 1:  # s'il ne prend jamais plus d'une valeur
            constant_sensors.append(sensor_name)  # il est constant : on l'ajoute a la liste
    return constant_sensors


def normalize_minmax(df, sensor_cols):
    """
    Applique une normalisation min-max (entre 0 et 1) sur les colonnes capteurs.

    Choix retenu en Phase 1 (voir docs/01_eda_preparation.md) : cout nul pour les
    modeles a arbres, entrees bornees adaptees a un futur LSTM, hypothese
    gaussienne de la standardisation de toute facon mise a mal par nos donnees.

    Parameters
    ----------
    df : pandas.DataFrame
        Le tableau a normaliser (n'est pas modifie sur place, une copie est renvoyee).
    sensor_cols : list de str
        Les noms de colonnes capteurs a normaliser.

    Returns
    -------
    (pandas.DataFrame, sklearn.preprocessing.MinMaxScaler)
        Le tableau normalise (copie), et le scaler entraine (utile pour appliquer
        exactement la meme regle a un autre jeu de donnees, ex. le jeu de test).
    """
    df_normalise = df.copy()  # copie independante, pour ne pas modifier df sur place

    minmax_scaler = MinMaxScaler()  # cree l'outil de normalisation min-max
    # fit_transform apprend les minimums/maximums sur df, puis ramene les capteurs entre 0 et 1
    df_normalise[sensor_cols] = minmax_scaler.fit_transform(df[sensor_cols])

    return df_normalise, minmax_scaler


def lisser_capteurs(df, sensor_cols, fenetre=5):
    """
    Applique une moyenne mobile causale sur les colonnes capteurs, moteur par moteur.

    Deux regles de correction respectees (voir docs/01_eda_preparation.md) :
    1. Causalite : le lissage du cycle t n'utilise que des cycles <= t (rolling()
       de pandas est causal par defaut).
    2. Par moteur, jamais a travers deux moteurs differents : le lissage est
       calcule separement sur chaque moteur, jamais sur tout le tableau empile.

    Parameters
    ----------
    df : pandas.DataFrame
        Le tableau a lisser (n'est pas modifie sur place, une copie est renvoyee).
    sensor_cols : list de str
        Les noms de colonnes capteurs a lisser.
    fenetre : int, optionnel
        Nombre de cycles utilises pour la moyenne mobile (5 par defaut, cf. README).

    Returns
    -------
    pandas.DataFrame
        Le tableau avec les colonnes capteurs lissees (copie).
    """
    df_lisse = df.copy()  # copie independante, pour ne pas modifier df sur place

    for sensor_name in sensor_cols:  # on lisse chaque capteur, un par un
        for moteur_id in df_lisse["unit_number"].unique():  # on parcourt chaque moteur separement
            masque = df_lisse["unit_number"] == moteur_id  # Vrai sur les lignes de ce moteur, Faux sinon
            # rolling() ne regarde que les lignes de cette selection (un seul moteur) : jamais de fuite
            # entre deux moteurs, et jamais de cycle futur (rolling est causal par defaut)
            df_lisse.loc[masque, sensor_name] = (
                df_lisse.loc[masque, sensor_name].rolling(fenetre, min_periods=1).mean()
            )

    return df_lisse


def calculer_rul(df):
    """
    Calcule la vraie RUL (Remaining Useful Life) par cycle, moteur par moteur.

    Dans train_FD001.txt, chaque moteur est suivi jusqu'a sa panne : le dernier
    cycle enregistre pour un moteur correspond au moment de la panne. La RUL a
    un cycle donne se deduit donc directement :

        RUL = dernier_cycle_du_moteur - cycle_actuel

    Parameters
    ----------
    df : pandas.DataFrame
        Le tableau contenant au moins "unit_number" et "time_cycles"
        (n'est pas modifie sur place, une copie est renvoyee).

    Returns
    -------
    pandas.DataFrame
        Le tableau avec une nouvelle colonne "RUL" (copie).
    """
    df_rul = df.copy()  # copie independante, pour ne pas modifier df sur place
    df_rul["RUL"] = 0  # valeur provisoire, remplacee ligne par ligne juste apres

    for moteur_id in df_rul["unit_number"].unique():  # unique() donne la liste des identifiants de moteurs
        masque = df_rul["unit_number"] == moteur_id  # Vrai sur les lignes de ce moteur, Faux sinon

        # Le dernier cycle enregistre pour ce moteur correspond a sa panne
        dernier_cycle = df_rul.loc[masque, "time_cycles"].max()  # valeur maximale de time_cycles pour ce moteur

        # La RUL de chaque ligne de ce moteur est la distance a ce dernier cycle
        df_rul.loc[masque, "RUL"] = dernier_cycle - df_rul.loc[masque, "time_cycles"]

    return df_rul


def plafonner_rul(df, plafond=125):
    """
    Plafonne la colonne RUL a une valeur maximale (Piecewise Linear RUL).

    En tout debut de vie, le signal capteur varie a peine d'un cycle a l'autre :
    demander a un modele de distinguer une RUL de 300 d'une RUL de 280 a partir
    d'un signal quasi immobile est une tache impossible. Tant que la RUL brute
    depasse le plafond, elle est remplacee par le plafond ; en dessous, elle
    reste inchangee (voir docs/02_target_engineering.md pour la preuve empirique).

    Parameters
    ----------
    df : pandas.DataFrame
        Le tableau contenant au moins la colonne "RUL"
        (n'est pas modifie sur place, une copie est renvoyee).
    plafond : int, optionnel
        La valeur de plafonnement (125 par defaut, cf. cahier des charges section 3).

    Returns
    -------
    pandas.DataFrame
        Le tableau avec une nouvelle colonne "RUL_cappee" (copie).
    """
    df_cappee = df.copy()  # copie independante, pour ne pas modifier df sur place
    df_cappee["RUL_cappee"] = 0  # valeur provisoire

    for index_ligne in df_cappee.index:  # on parcourt chaque ligne du tableau une par une
        rul_brute = df_cappee.loc[index_ligne, "RUL"]  # RUL brute de cette ligne

        if rul_brute > plafond:  # si la RUL brute depasse le plafond
            df_cappee.loc[index_ligne, "RUL_cappee"] = plafond  # on la remplace par le plafond
        else:  # sinon (RUL brute deja sous le plafond)
            df_cappee.loc[index_ligne, "RUL_cappee"] = rul_brute  # on la garde telle quelle

    return df_cappee
