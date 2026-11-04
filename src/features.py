"""
Construction des features (Phase 3) et preparation complete des jeux train / test FD001.

Ce module reprend, sans en changer la logique, le pipeline de notebooks/03_modelisation.ipynb :

1. capteurs constants retires (identifies sur le train uniquement) ;
2. normalisation min-max apprise sur le train (le test ne fait que `transform`) ;
3. lissage par moyenne mobile de 5 cycles, moteur par moteur ;
4. RUL brute puis plafonnee a 125 (train uniquement : le test n'a pas de panne observee) ;
5. features glissantes (moyenne et ecart-type sur 5 cycles), NaN du 1er cycle remplaces par 0 ;
6. 46 features = 15 capteurs lisses + 15 moyennes glissantes + 15 ecarts-types glissants + time_cycles.

Il existe pour que le notebook de la Phase 4 et la demo n'aient pas a recopier ces etapes une
nouvelle fois : un seul endroit a corriger si une etape change, et un seul endroit a tester.

Option (Phase 4, notebook 04e) : `preparer_train(..., ecart_initial=20)` ajoute la variable
"ecart a l'etat initial" (famille `ecartinit_k` de notebooks/04e_feature_engineering.ipynb, voir
`calculer_ecart_etat_initial`) et remplace les 46 features par le jeu retenu en 04e : 31 features =
15 capteurs lisses + time_cycles + 15 ecarts a l'etat initial. Sans cette option (defaut), rien ne change.
"""

import numpy as np
import pandas as pd

from src.preprocessing import (
    DATA_DIR,
    calculer_rul,
    find_constant_sensors,
    get_sensor_columns,
    lisser_capteurs,
    load_fd001,
    normalize_minmax,
    plafonner_rul,
)


def ajouter_features_glissantes(df, sensor_cols, fenetre):
    """
    Ajoute, pour chaque capteur, deux colonnes calculees sur une fenetre glissante de cycles :
    la moyenne et l'ecart-type des `fenetre` derniers cycles, moteur par moteur.

    Reprise a l'identique de la fonction du notebook 03 (meme logique, memes noms de colonnes).
    Au tout premier cycle d'un moteur, l'ecart-type d'une seule valeur n'existe pas : pandas
    renvoie NaN a cet endroit (remplace par 0 ensuite, voir `remplir_nan_ecarts_types`).

    Parameters
    ----------
    df : pandas.DataFrame
        Le tableau source (n'est pas modifie sur place, une copie est renvoyee).
    sensor_cols : list de str
        Les noms de colonnes capteurs pour lesquelles calculer les features.
    fenetre : int
        Nombre de cycles utilises pour la fenetre glissante.

    Returns
    -------
    pandas.DataFrame
        Le tableau avec 2 nouvelles colonnes par capteur : "<capteur>_glissante_moy" et
        "<capteur>_glissante_std".
    """
    df = df.copy()  # copie independante, pour ne pas modifier df sur place

    for sensor_name in sensor_cols:  # on traite chaque capteur, un par un
        colonne_moyenne = sensor_name + "_glissante_moy"  # nom de la nouvelle colonne "moyenne"
        colonne_ecart_type = sensor_name + "_glissante_std"  # nom de la nouvelle colonne "ecart-type"
        df[colonne_moyenne] = 0.0  # valeurs provisoires, remplacees moteur par moteur juste apres
        df[colonne_ecart_type] = 0.0

        for moteur_id in df["unit_number"].unique():  # on traite chaque moteur separement
            masque = df["unit_number"] == moteur_id  # Vrai sur les lignes de ce moteur, Faux sinon
            # rolling() est causal par defaut (ne regarde que le passe) et ne porte que sur ce moteur
            df.loc[masque, colonne_moyenne] = df.loc[masque, sensor_name].rolling(fenetre, min_periods=1).mean()
            df.loc[masque, colonne_ecart_type] = df.loc[masque, sensor_name].rolling(fenetre, min_periods=1).std()

    return df


def lister_colonnes_suffixe(colonnes, suffixe):
    """
    Renvoie, dans l'ordre, les noms de colonnes qui se terminent par un suffixe donne.

    Meme logique que les deux petites boucles du notebook 03 (cellules 8 et 10) qui retrouvent les
    colonnes "_glissante_std" et "_glissante_moy" : l'ordre obtenu est l'ordre des colonnes du
    tableau, donc l'ordre des capteurs (les features glissantes sont ajoutees capteur par capteur).

    Parameters
    ----------
    colonnes : liste de str
        Les noms de colonnes a filtrer (typiquement df.columns).
    suffixe : str
        Le suffixe recherche, par exemple "_glissante_std".

    Returns
    -------
    list de str
        Les noms de colonnes qui se terminent par `suffixe`, dans leur ordre d'apparition.
    """
    colonnes_trouvees = []  # on part d'une liste vide
    for colonne in colonnes:  # on parcourt chaque nom de colonne, dans l'ordre
        if colonne.endswith(suffixe):  # on ne garde que celles qui finissent par le suffixe
            colonnes_trouvees.append(colonne)
    return colonnes_trouvees


def remplir_nan_ecarts_types(df, colonnes_ecart_type, nb_nan_attendu):
    """
    Remplace par 0 les NaN des colonnes "ecart-type glissant" (1er cycle de chaque moteur).

    Au 1er cycle d'un moteur, la fenetre ne contient qu'une valeur : l'ecart-type n'existe pas et
    pandas renvoie NaN. Une dispersion de 0 est la valeur la plus honnete (aucune variation encore
    observee), meme choix que le notebook 03 (cellule 8). On verifie au passage qu'il y a exactement
    le nombre de NaN attendu (1 par moteur et par capteur) : un NaN ailleurs signalerait un probleme.

    Parameters
    ----------
    df : pandas.DataFrame
        Le tableau contenant les colonnes d'ecart-type (n'est pas modifie sur place, copie renvoyee).
    colonnes_ecart_type : list de str
        Les noms des colonnes "<capteur>_glissante_std".
    nb_nan_attendu : int
        Le nombre de NaN attendu avant remplissage (nombre de moteurs x nombre de capteurs).

    Returns
    -------
    pandas.DataFrame
        Le tableau avec les NaN des ecarts-types remplaces par 0 (copie).
    """
    df_rempli = df.copy()  # copie independante, pour ne pas modifier df sur place

    # total de NaN sur toutes les colonnes d'ecart-type, avant remplissage
    nb_nan_avant = df_rempli[colonnes_ecart_type].isna().sum().sum()
    assert nb_nan_avant == nb_nan_attendu, (
        f"attendu {nb_nan_attendu} NaN (1 par moteur et par capteur, au 1er cycle), trouve {nb_nan_avant}"
    )

    # remplace ces NaN par 0, exactement comme dans le notebook 03
    df_rempli[colonnes_ecart_type] = df_rempli[colonnes_ecart_type].fillna(0.0)

    # verification : plus aucun NaN apres le remplissage
    nb_nan_apres = df_rempli[colonnes_ecart_type].isna().sum().sum()
    assert nb_nan_apres == 0, f"il reste {nb_nan_apres} NaN apres le fillna"

    return df_rempli


def construire_features_glissantes(df, sensor_cols, fenetre):
    """
    Ajoute les features glissantes puis remplit les NaN du 1er cycle : etapes 5 du pipeline.

    Regroupe `ajouter_features_glissantes` + `remplir_nan_ecarts_types`, utilises de la meme facon
    sur le train et sur le test.

    Parameters
    ----------
    df : pandas.DataFrame
        Le tableau deja lisse (n'est pas modifie sur place, une copie est renvoyee).
    sensor_cols : list de str
        Les noms de colonnes capteurs (les 15 capteurs restants).
    fenetre : int
        Nombre de cycles de la fenetre glissante.

    Returns
    -------
    (pandas.DataFrame, list de str, list de str)
        Le tableau enrichi, la liste des colonnes "moyenne glissante" et la liste des colonnes
        "ecart-type glissant" (dans l'ordre des capteurs).
    """
    # moyenne et ecart-type glissants, capteur par capteur et moteur par moteur
    df_features = ajouter_features_glissantes(df, sensor_cols, fenetre)

    # noms des nouvelles colonnes, retrouves par leur suffixe (meme ordre que les capteurs)
    colonnes_moyenne = lister_colonnes_suffixe(df_features.columns, "_glissante_moy")
    colonnes_ecart_type = lister_colonnes_suffixe(df_features.columns, "_glissante_std")

    # 1 NaN par moteur et par capteur (le 1er cycle de chaque moteur)
    nb_moteurs = df_features["unit_number"].nunique()
    nb_nan_attendu = nb_moteurs * len(sensor_cols)
    df_features = remplir_nan_ecarts_types(df_features, colonnes_ecart_type, nb_nan_attendu)

    return df_features, colonnes_moyenne, colonnes_ecart_type


def construire_liste_features(sensor_cols_restants, colonnes_moyenne, colonnes_ecart_type):
    """
    Construit la liste ordonnee des 46 features, dans le meme ordre que le notebook 03 (cellule 10).

    Ordre : 15 capteurs lisses, puis 15 moyennes glissantes, puis 15 ecarts-types glissants,
    puis time_cycles. unit_number reste exclu : c'est une etiquette de moteur, pas une mesure.

    Parameters
    ----------
    sensor_cols_restants : list de str
        Les 15 capteurs informatifs (apres retrait des capteurs constants).
    colonnes_moyenne : list de str
        Les 15 colonnes "<capteur>_glissante_moy".
    colonnes_ecart_type : list de str
        Les 15 colonnes "<capteur>_glissante_std".

    Returns
    -------
    list de str
        Les noms des features, dans l'ordre attendu par les modeles.
    """
    feature_cols = []  # on part d'une liste vide, remplie bloc par bloc
    for colonne in sensor_cols_restants:  # 1er bloc : les capteurs lisses
        feature_cols.append(colonne)
    for colonne in colonnes_moyenne:  # 2e bloc : les moyennes glissantes
        feature_cols.append(colonne)
    for colonne in colonnes_ecart_type:  # 3e bloc : les ecarts-types glissants
        feature_cols.append(colonne)
    feature_cols.append("time_cycles")  # derniere colonne : le numero du cycle en cours
    return feature_cols


def calculer_ecart_etat_initial(valeurs, k):
    """
    Ecart a l'etat initial d'UN capteur lisse sur UN moteur : valeur actuelle - moyenne de ses debuts.

    Definition reprise a l'identique de notebooks/04e_feature_engineering.ipynb (cellule 6, fonction
    `statistique_fenetre`, branche "ecartinit"), memes operations numpy dans le meme ordre :

        ecartinit_k(t) = x(t) - moyenne(x(1), ..., x(min(t, k)))

    avec x le capteur deja lisse, t le rang du cycle dans le moteur (1 = premier cycle observe).
    - Pour t >= k, la reference est FIGEE : la moyenne des k premiers cycles du moteur.
    - Pour t < k, la reference est la moyenne des t cycles deja observes (pas des k premiers, qui
      incluraient des cycles futurs) : la variable est donc causale. Au tout premier cycle, elle vaut 0.

    Exemple, k = 3 et x = [10, 12, 14, 20] : references [10, 11, 12, 12], ecarts [0, 1, 2, 8].

    Parameters
    ----------
    valeurs : numpy.ndarray
        Le signal lisse d'un capteur pour un seul moteur, dans l'ordre des cycles (1, 2, 3, ...).
    k : int
        Nombre de cycles de reference (20 dans le jeu retenu par le notebook 04e).

    Returns
    -------
    numpy.ndarray
        Une valeur par cycle, meme longueur que `valeurs`.
    """
    valeurs = np.asarray(valeurs, dtype=float)  # tableau numpy de reels (une copie si besoin)
    somme_cumulee = np.cumsum(valeurs)  # somme des cycles 1..t, pour chaque t
    nb_cycles = np.arange(1, len(valeurs) + 1)  # 1, 2, 3, ... : nombre de cycles connus au cycle t
    moyenne_debut = somme_cumulee / nb_cycles  # cas t < k : moyenne des t premiers cycles
    if len(valeurs) >= k:
        # cas t >= k : moyenne FIGEE des k premiers cycles (meme calcul que le notebook 04e)
        moyenne_debut[k:] = somme_cumulee[k - 1] / k
    return valeurs - moyenne_debut  # ecart entre la valeur actuelle et la reference du debut de vie


def ajouter_ecart_etat_initial(df, sensor_cols, k):
    """
    Ajoute, pour chaque capteur, la colonne "<capteur>_ecartinit_<k>", calculee moteur par moteur.

    Chaque moteur est traite seul (ses lignes, dans l'ordre du tableau) : un moteur ne voit jamais les
    cycles d'un autre. Le tableau doit etre trie par moteur puis par cycle, et chaque moteur doit
    commencer a son premier cycle observe (c'est le cas des fichiers C-MAPSS, train et test).

    Parameters
    ----------
    df : pandas.DataFrame
        Le tableau avec les capteurs DEJA lisses (n'est pas modifie sur place, une copie est renvoyee).
    sensor_cols : list de str
        Les capteurs pour lesquels calculer l'ecart a l'etat initial.
    k : int
        Nombre de cycles de reference.

    Returns
    -------
    (pandas.DataFrame, list de str)
        Le tableau enrichi et la liste des nouvelles colonnes (dans l'ordre des capteurs).
    """
    df_ecart = df.copy()  # copie independante, pour ne pas modifier df sur place
    nb_lignes = len(df_ecart)

    # positions (0, 1, 2, ...) des lignes de chaque moteur, dans l'ordre du tableau
    positions_par_moteur = df_ecart.groupby("unit_number", sort=False).indices

    colonnes_ecart = []  # noms des nouvelles colonnes, remplis capteur par capteur
    for sensor_name in sensor_cols:  # un capteur a la fois
        valeurs_capteur = df_ecart[sensor_name].to_numpy(dtype=float)  # signal lisse, toutes lignes
        ecarts = np.zeros(nb_lignes)  # espace reserve, rempli moteur par moteur
        for moteur_id in positions_par_moteur:  # un moteur a la fois : jamais de melange
            lignes_moteur = positions_par_moteur[moteur_id]  # positions des lignes de ce moteur
            ecarts[lignes_moteur] = calculer_ecart_etat_initial(valeurs_capteur[lignes_moteur], k)
        nom_colonne = sensor_name + "_ecartinit_" + str(k)  # ex. "sensor_11_ecartinit_20"
        df_ecart[nom_colonne] = ecarts
        colonnes_ecart.append(nom_colonne)

    return df_ecart, colonnes_ecart


def construire_liste_features_ecart_initial(sensor_cols_restants, colonnes_ecart):
    """
    Construit la liste ordonnee du jeu retenu par le notebook 04e (31 features pour FD001).

    Meme ordre que le notebook 04e (cellules 13 et 29 : `base16 + colonnes_de("ecartinit", 20)`) :
    capteurs lisses, puis time_cycles, puis les ecarts a l'etat initial. L'ordre est garde a l'identique
    pour pouvoir comparer les chiffres au notebook 04e.

    Parameters
    ----------
    sensor_cols_restants : list de str
        Les capteurs informatifs (lisses).
    colonnes_ecart : list de str
        Les colonnes "<capteur>_ecartinit_<k>".

    Returns
    -------
    list de str
        Les noms des features, dans l'ordre attendu par les modeles.
    """
    feature_cols = []  # on part d'une liste vide, remplie bloc par bloc
    for colonne in sensor_cols_restants:  # 1er bloc : les capteurs lisses
        feature_cols.append(colonne)
    feature_cols.append("time_cycles")  # puis le numero du cycle en cours
    for colonne in colonnes_ecart:  # dernier bloc : les ecarts a l'etat initial
        feature_cols.append(colonne)
    return feature_cols


def preparer_train(fenetre=5, plafond=125, ecart_initial=None):
    """
    Enchaine toutes les etapes du notebook 03 sur train_FD001.txt et renvoie tout ce qu'il faut
    pour entrainer un modele et preparer le test sans fuite d'information.

    Ordre exact du notebook 03 : chargement -> retrait des capteurs constants -> min-max (appris ici)
    -> lissage -> RUL brute -> RUL plafonnee -> features glissantes -> NaN du 1er cycle a 0 -> X, y, groups.

    Parameters
    ----------
    fenetre : int, optionnel
        Taille de la fenetre (lissage ET features glissantes), 5 par defaut comme en Phase 3.
    plafond : int, optionnel
        Plafond de la RUL (Phase 2), 125 par defaut.
    ecart_initial : int ou None, optionnel
        None (defaut) : les 46 features de la Phase 3, comportement inchange. Un entier k (20 dans le
        notebook 04e) : ajoute les colonnes "<capteur>_ecartinit_<k>" (`ajouter_ecart_etat_initial`) et
        X devient le jeu du notebook 04e : capteurs lisses + time_cycles + ecarts (31 features pour FD001).

    Returns
    -------
    dict
        Un dictionnaire (le "contexte train") avec les cles :

        - "train" : pandas.DataFrame, le train enrichi (20 631 lignes pour FD001) ;
        - "X" : pandas.DataFrame, les 46 features, meme ordre de colonnes que le notebook 03
          (ou les 31 features du notebook 04e si `ecart_initial` est donne) ;
        - "y" : pandas.Series, la cible "RUL_cappee" ;
        - "groups" : pandas.Series, le numero de moteur de chaque ligne (pour GroupKFold) ;
        - "feature_cols" : list de str, les noms des 46 features ;
        - "sensor_cols_restants" : list de str, les 15 capteurs informatifs ;
        - "constant_sensors" : list de str, les capteurs constants retires (appris sur le train) ;
        - "colonnes_moyenne", "colonnes_ecart_type" : list de str, les colonnes glissantes ;
        - "scaler" : sklearn.preprocessing.MinMaxScaler, appris sur le train, a reutiliser tel quel ;
        - "fenetre" : int, "plafond" : int, les reglages utilises ;
        - "ecart_initial" : int ou None, le reglage utilise ; "colonnes_ecart_initial" : list de str
          (vide si `ecart_initial` vaut None).
    """
    # --- Phases 1 et 2 : memes appels, dans le meme ordre que la cellule 4 du notebook 03 ---
    train = load_fd001("train")  # chargement du fichier d'entrainement FD001

    sensor_cols = get_sensor_columns(train.columns)  # les 21 capteurs bruts
    constant_sensors = find_constant_sensors(train, sensor_cols)  # capteurs constants (Phase 1)
    train = train.drop(columns=constant_sensors)  # on les retire
    sensor_cols_restants = get_sensor_columns(train.columns)  # les 15 capteurs informatifs restants

    # normalisation min-max (0 a 1) : le scaler est APPRIS ici, sur le train seulement
    train, scaler = normalize_minmax(train, sensor_cols_restants)
    train = lisser_capteurs(train, sensor_cols_restants, fenetre=fenetre)  # moyenne mobile, moteur par moteur
    train = calculer_rul(train)  # RUL brute par cycle
    train = plafonner_rul(train, plafond=plafond)  # RUL plafonnee (125 par defaut)

    # --- Phase 3 : features glissantes + NaN du 1er cycle remplaces par 0 (cellules 6 et 8) ---
    train, colonnes_moyenne, colonnes_ecart_type = construire_features_glissantes(
        train, sensor_cols_restants, fenetre
    )

    # --- Features, cible et groupes (cellule 10) ---
    feature_cols = construire_liste_features(sensor_cols_restants, colonnes_moyenne, colonnes_ecart_type)

    # --- Option Phase 4 (notebook 04e) : ecart a l'etat initial, sur les capteurs deja lisses ---
    colonnes_ecart_initial = []  # reste vide sans l'option : comportement par defaut inchange
    if ecart_initial is not None:
        train, colonnes_ecart_initial = ajouter_ecart_etat_initial(train, sensor_cols_restants, ecart_initial)
        # jeu retenu par le notebook 04e : capteurs lisses + time_cycles + ecarts a l'etat initial
        feature_cols = construire_liste_features_ecart_initial(sensor_cols_restants, colonnes_ecart_initial)

    X = train[feature_cols]  # les colonnes de features (entrees du modele) : 46 par defaut
    y = train["RUL_cappee"]  # la cible a predire (Phase 2)
    groups = train["unit_number"]  # quel moteur pour chaque ligne : sert uniquement a GroupKFold

    # on regroupe tout dans un dictionnaire, cle par cle, pour que l'appelant retrouve chaque objet par son nom
    contexte_train = {}
    contexte_train["train"] = train
    contexte_train["X"] = X
    contexte_train["y"] = y
    contexte_train["groups"] = groups
    contexte_train["feature_cols"] = feature_cols
    contexte_train["sensor_cols_restants"] = sensor_cols_restants
    contexte_train["constant_sensors"] = constant_sensors
    contexte_train["colonnes_moyenne"] = colonnes_moyenne
    contexte_train["colonnes_ecart_type"] = colonnes_ecart_type
    contexte_train["scaler"] = scaler
    contexte_train["fenetre"] = fenetre
    contexte_train["plafond"] = plafond
    contexte_train["ecart_initial"] = ecart_initial
    contexte_train["colonnes_ecart_initial"] = colonnes_ecart_initial
    return contexte_train


def charger_rul_test():
    """
    Charge RUL_FD001.txt : la vraie RUL au dernier cycle observe de chaque moteur test.

    La ligne 1 du fichier correspond au moteur 1, la ligne 2 au moteur 2, etc. (data/readme.txt).

    Returns
    -------
    numpy.ndarray
        Les 100 RUL vraies (entiers, non plafonnees), dans l'ordre des moteurs 1 a 100.
    """
    chemin = f"{DATA_DIR}/RUL_FD001.txt"  # meme construction de chemin que le notebook 03
    rul_test_vraie = pd.read_csv(chemin, sep=r"\s+", header=None, names=["RUL_vraie"])
    return rul_test_vraie["RUL_vraie"].values  # on renvoie un tableau numpy, comme y_test_vraie du notebook


def ajouter_rul_vraie_test(test, y_test_vraie, plafond):
    """
    Ajoute au test la vraie RUL a CHAQUE cycle (et sa version plafonnee), moteur par moteur.

    RUL_FD001.txt ne donne la RUL qu'au dernier cycle observe. Comme la RUL baisse d'exactement 1 a
    chaque cycle, on en deduit la RUL vraie a n'importe quel cycle t du meme moteur :

        RUL_vraie(t) = RUL_FD001[moteur] + (dernier_cycle_du_moteur - t)

    Utile pour la demo (courbe predite vs vraie sur toute la trajectoire d'un moteur test).
    La version plafonnee (min(RUL_vraie, plafond)) est ce que le modele a appris a predire.

    Parameters
    ----------
    test : pandas.DataFrame
        Le test, avec au moins "unit_number" et "time_cycles" (n'est pas modifie, copie renvoyee).
    y_test_vraie : numpy.ndarray
        Les RUL vraies au dernier cycle, dans l'ordre croissant des numeros de moteur.
    plafond : int
        Plafond utilise pour la colonne "RUL_vraie_cappee".

    Returns
    -------
    pandas.DataFrame
        Le test avec deux nouvelles colonnes : "RUL_vraie" et "RUL_vraie_cappee" (copie).
    """
    test_rul = test.copy()  # copie independante, pour ne pas modifier test sur place
    test_rul["RUL_vraie"] = 0  # valeurs provisoires, remplacees moteur par moteur juste apres

    moteurs_tries = np.sort(test_rul["unit_number"].unique())  # numeros de moteur, du plus petit au plus grand
    assert len(moteurs_tries) == len(y_test_vraie), (
        f"{len(moteurs_tries)} moteurs dans le test mais {len(y_test_vraie)} valeurs dans RUL_FD001.txt"
    )

    for position in range(len(moteurs_tries)):  # un moteur a la fois
        moteur_id = moteurs_tries[position]  # numero du moteur
        rul_au_dernier_cycle = y_test_vraie[position]  # sa RUL vraie au dernier cycle observe
        masque = test_rul["unit_number"] == moteur_id  # Vrai sur les lignes de ce moteur, Faux sinon
        dernier_cycle = test_rul.loc[masque, "time_cycles"].max()  # dernier cycle observe de ce moteur
        # RUL vraie a chaque cycle t : RUL au dernier cycle + nombre de cycles restant jusqu'au dernier
        test_rul.loc[masque, "RUL_vraie"] = rul_au_dernier_cycle + (dernier_cycle - test_rul.loc[masque, "time_cycles"])

    # version plafonnee : comme la cible d'entrainement, la RUL ne depasse jamais le plafond
    test_rul["RUL_vraie_cappee"] = test_rul["RUL_vraie"].clip(upper=plafond)
    return test_rul


def preparer_test(contexte_train):
    """
    Applique au jeu de test officiel (test_FD001.txt) exactement les etapes du train, sans fuite.

    Deux regles strictes (notebook 03, cellule 18) :
    1. memes capteurs constants retires (ceux trouves sur le train, jamais recalcules sur le test) ;
    2. meme scaler, en `transform` seul (jamais `fit` sur le test).
    Puis meme lissage, memes features glissantes, meme remplissage des NaN du 1er cycle, et meme ecart a
    l'etat initial si le contexte train l'utilise (option `ecart_initial` de `preparer_train`).

    Parameters
    ----------
    contexte_train : dict
        Le dictionnaire renvoye par `preparer_train` (on y lit constant_sensors,
        sensor_cols_restants, scaler, fenetre, plafond, feature_cols et ecart_initial).

    Returns
    -------
    dict
        Un dictionnaire (le "contexte test") avec les cles :

        - "test" : pandas.DataFrame, le test enrichi, TOUS les cycles (13 096 lignes pour FD001),
          avec en plus "RUL_vraie" et "RUL_vraie_cappee" a chaque cycle (voir `ajouter_rul_vraie_test`) ;
        - "dernier_cycle_par_moteur" : pandas.DataFrame, la ligne du dernier cycle de chaque moteur,
          triee par unit_number (100 lignes) ;
        - "X_test_dernier" : pandas.DataFrame, les features de ces 100 lignes (memes colonnes, meme ordre que X) ;
        - "y_test_vraie" : numpy.ndarray, la RUL vraie brute de RUL_FD001.txt (non plafonnee) ;
        - "moteurs_test" : numpy.ndarray, les numeros de moteur dans le meme ordre que y_test_vraie.
    """
    # on relit dans le contexte train tout ce qui a ete APPRIS sur le train
    constant_sensors = contexte_train["constant_sensors"]
    sensor_cols_restants = contexte_train["sensor_cols_restants"]
    scaler = contexte_train["scaler"]
    fenetre = contexte_train["fenetre"]
    plafond = contexte_train["plafond"]
    feature_cols = contexte_train["feature_cols"]
    # .get : un contexte construit avant l'ajout de l'option n'a pas cette cle (equivaut a None)
    ecart_initial = contexte_train.get("ecart_initial")

    test = load_fd001("test")  # chargement du fichier de test FD001
    test = test.drop(columns=constant_sensors)  # memes capteurs retires que sur le train

    # transform seul (pas fit_transform) : on applique le scaler deja appris sur le train, sans le reajuster
    test[sensor_cols_restants] = scaler.transform(test[sensor_cols_restants])

    test = lisser_capteurs(test, sensor_cols_restants, fenetre=fenetre)  # meme lissage, moteur par moteur
    test, colonnes_moyenne, colonnes_ecart_type = construire_features_glissantes(
        test, sensor_cols_restants, fenetre
    )  # memes features glissantes + meme traitement du 1er cycle

    # meme ecart a l'etat initial que le train (chaque moteur test part de son propre 1er cycle observe)
    if ecart_initial is not None:
        test, _ = ajouter_ecart_etat_initial(test, sensor_cols_restants, ecart_initial)

    # RUL vraie fournie par la NASA au dernier cycle de chaque moteur, puis a chaque cycle
    y_test_vraie = charger_rul_test()
    test = ajouter_rul_vraie_test(test, y_test_vraie, plafond)

    # pour chaque moteur, on garde la ligne dont time_cycles est maximal (dernier cycle enregistre),
    # triee par numero de moteur pour etre alignee avec RUL_FD001.txt (cellule 20 du notebook 03)
    indices_derniers_cycles = test.groupby("unit_number")["time_cycles"].idxmax()
    dernier_cycle_par_moteur = test.loc[indices_derniers_cycles].sort_values("unit_number").reset_index(drop=True)
    assert dernier_cycle_par_moteur.shape[0] == len(y_test_vraie), "un dernier cycle par moteur attendu"

    # verification d'alignement : au dernier cycle, la RUL vraie par cycle doit redonner RUL_FD001.txt
    assert np.array_equal(dernier_cycle_par_moteur["RUL_vraie"].values, y_test_vraie), (
        "RUL_vraie au dernier cycle differente de RUL_FD001.txt : probleme d'alignement des moteurs"
    )

    X_test_dernier = dernier_cycle_par_moteur[feature_cols]  # memes features, meme ordre que le train

    # on regroupe tout dans un dictionnaire, cle par cle
    contexte_test = {}
    contexte_test["test"] = test
    contexte_test["dernier_cycle_par_moteur"] = dernier_cycle_par_moteur
    contexte_test["X_test_dernier"] = X_test_dernier
    contexte_test["y_test_vraie"] = y_test_vraie
    contexte_test["moteurs_test"] = dernier_cycle_par_moteur["unit_number"].values
    return contexte_test
