"""
Deep learning sequentiel (Phase 3, suite) : LSTM et CNN 1D pour predire la RUL a partir de fenetres de cycles.

Idee generale. Les modeles a arbres (Phase 3 et 4) voient UNE ligne par cycle : les capteurs lisses du cycle,
quelques statistiques glissantes et time_cycles. Un reseau sequentiel voit une FENETRE : les `longueur`
derniers cycles du meme moteur (30 par defaut), sous forme d'un tableau (longueur x nombre d'entrees). Il
apprend lui-meme quels motifs dans cette petite serie temporelle annoncent la panne.

Ce module contient, dans l'ordre :

1. la preparation des entrees : `construire_fenetres` (une fenetre par ligne du tableau, dans le meme ordre
   que les lignes, jamais de donnees d'un autre moteur ni du futur), sa version economique en memoire
   `construire_fenetres_paresseuses` (memes fenetres, construites lot par lot) et la mise a l'echelle de
   time_cycles ;
2. les deux architectures, volontairement simples et standard : `ModeleLSTM` et `ModeleCNN1D` ;
3. l'apprentissage : `entrainer_modele` (Adam, lots, perte quadratique eventuellement asymetrique,
   arret precoce sur des moteurs de validation, graines fixees) et `predire` ; `executer_entrainement` et
   `executer_taches` lancent des apprentissages independants, l'un apres l'autre ou en parallele (un
   processus et 1 coeur par apprentissage, memes resultats dans les deux cas) ;
4. la validation croisee par moteur avec les MEMES plis que les arbres :
   `evaluer_en_validation_croisee_sequentielle` (memes cles de resultats que
   `src.evaluation.evaluer_en_validation_croisee`, pour comparer les tableaux ligne a ligne), decoupee en
   `preparer_taches_validation_croisee` et `assembler_validation_croisee` pour le script ;
5. la sauvegarde et le chargement des poids avec leurs hyperparametres (`sauvegarder_modele`,
   `charger_modele`).

La cible est mise a l'echelle pendant l'apprentissage (RUL / 125, donc entre 0 et 1) pour que les poids
du reseau restent dans des ordres de grandeur stables ; toutes les fonctions publiques prennent et
renvoient pourtant des RUL en CYCLES : la mise a l'echelle reste cachee a l'interieur du module.
"""

import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

# ORDRE DES IMPORTS VOULU (ne pas le "ranger") : src.config_modele charge xgboost et lightgbm, donc leur
# bibliotheque OpenMP (calcul parallele), AVANT celle de PyTorch. Constate le 2026-11-04 sur cette machine : si
# PyTorch est charge en premier, un XGBoost multi-coeurs lance ensuite dans le meme processus plante (erreur de
# segmentation, ou "OMP: Error #179") ; dans l'ordre inverse il fonctionne (voir aussi NB_THREADS_DEFAUT). Trois
# copies d'OpenMP coexistent : celle de Homebrew (xgboost, lightgbm), celle de scikit-learn et celle de PyTorch.
# Importer ce module en premier est donc sans danger ; importer `torch` directement avant xgboost ne l'est pas.
from src.config_modele import RACINE_DEPOT, convertir_en_types_python
from src.evaluation import alertes_manquees, fausses_alertes, mae, nb_surestimes, rmse, score_nasa

import torch  # noqa: E402  (apres src.config_modele, voir ci-dessus)
from torch import nn  # noqa: E402

# ---------------------------------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------------------------------

# Longueur des fenetres par defaut : 30 cycles. Justification par les donnees (pas par la litterature) :
# le plus court moteur test de FD001 a 31 cycles observes, donc au dernier cycle (celui qui est note) la
# fenetre de 30 cycles est toujours remplie de vraies mesures. Pour FD002 et FD004, 6 et 11 moteurs test
# ont moins de 30 cycles : leur fenetre est completee par remplissage (voir `construire_fenetres`).
LONGUEUR_FENETRE_DEFAUT = 30

# La cible (RUL plafonnee a 125, Phase 2) est divisee par 125 pendant l'apprentissage : elle vaut alors
# entre 0 et 1, comme les capteurs normalises. Les predictions sont remultipliees par 125 a la sortie.
ECHELLE_CIBLE = 125.0

# time_cycles (de 1 a 543 selon les sous-jeux) est divise par une constante FIXE, choisie a la main et non
# apprise sur les donnees (donc aucune fuite possible) : 100 cycles. Les valeurs deviennent de l'ordre de
# 0 a 5, du meme ordre de grandeur que les capteurs normalises (0 a 1).
ECHELLE_TIME_CYCLES = 100.0
NOM_COLONNE_TIME_CYCLES_ECHELLE = "time_cycles_echelle"

# Nombre de coeurs de calcul utilises par PyTorch : 1 par defaut, applique des l'import de ce module.
# Trois raisons :
# 1. BLOCAGE CONSTATE le 2026-11-04 : quand xgboost ou lightgbm ont deja calcule dans le MEME processus Python
#    (suite pytest complete, notebook 03b qui compare aux arbres), deux bibliotheques OpenMP (calcul parallele)
#    coexistent : celle de xgboost / lightgbm et celle de PyTorch. Un calcul PyTorch sur plusieurs coeurs peut
#    alors attendre indefiniment ses coeurs auxiliaires (blocage dans __kmp_join_barrier, 0 % CPU). Reproduit
#    dans la suite pytest. Avec 1 coeur, PyTorch n'ouvre aucune
#    region parallele OpenMP : plus de blocage (meme effet que la variable d'environnement OMP_NUM_THREADS=1) ;
# 2. reproductibilite : resultats identiques au bit pres d'une execution a l'autre (verifie : 1 ou 6 coeurs
#    donnent d'ailleurs les memes predictions, voir 03b_cpu_vs_mps.csv et le notebook 03b) ;
# 3. vitesse : nos petits reseaux n'occupent pas bien plusieurs coeurs (mesure : 1 coeur aussi rapide ou plus
#    rapide que 6) ; le script gagne du temps en lancant plusieurs apprentissages de 1 coeur en parallele.
NB_THREADS_DEFAUT = 1
torch.set_num_threads(NB_THREADS_DEFAUT)

# Les deux types de reseaux connus (memes cles que src/contrat_predictions.py : "lstm" et "cnn1d")
TYPES_MODELES = ["lstm", "cnn1d"]

# Hyperparametres par defaut des deux reseaux : reglages "raisonnables", non optimises (voir les docstrings
# des classes pour la justification de chaque valeur)
HYPERPARAMETRES_LSTM_DEFAUT = {
    "taille_cachee": 64,  # 64 nombres resument la sequence a chaque pas de temps
    "nb_couches": 2,      # deux LSTM empilees
    "dropout": 0.2,       # 20 % des neurones eteints au hasard pendant l'apprentissage
}
HYPERPARAMETRES_CNN_DEFAUT = {
    "canaux": [32, 64, 64],  # nombre de filtres de chacun des 3 blocs de convolution
    "noyaux": [5, 5, 3],     # largeur (en cycles) des filtres de chaque bloc
    "dropout": 0.2,
}

# Reglages d'apprentissage par defaut (les memes pour les deux reseaux et pour tous les sous-jeux)
REGLAGES_ENTRAINEMENT_DEFAUT = {
    "taux_apprentissage": 1e-3,  # pas d'Adam : valeur par defaut de PyTorch
    "taille_lot": 256,           # nombre de fenetres par mise a jour des poids
    "max_epoques": 80,           # nombre maximal de passages sur les donnees d'apprentissage
    "patience": 10,              # arret si la perte de validation ne s'ameliore plus pendant 10 epoques
    "alpha": 1.0,                # 1 = perte quadratique classique (MSE) ; > 1 = surestimations plus couteuses
}

# Dossier par defaut des poids des modeles (les fichiers *.pt sont ignores par git)
DOSSIER_MODELES = os.path.join(RACINE_DEPOT, "models")


# ---------------------------------------------------------------------------------------------------
# Reglages de PyTorch et graines
# ---------------------------------------------------------------------------------------------------

def configurer_torch(nb_threads=NB_THREADS_DEFAUT):
    """
    Fixe le nombre de coeurs de calcul utilises par PyTorch sur le processeur (CPU).

    Le module regle deja 1 coeur a l'import (voir NB_THREADS_DEFAUT et les raisons ecrites a cote).
    ATTENTION : ne demander plus d'un coeur que dans un processus ou xgboost / lightgbm n'ont pas encore
    calcule, sinon un calcul PyTorch peut se bloquer indefiniment (conflit entre deux bibliotheques OpenMP).

    Parameters
    ----------
    nb_threads : int, optionnel
        Nombre de coeurs (1 par defaut).

    Returns
    -------
    int
        Le nombre de coeurs effectivement utilise par PyTorch apres le reglage.
    """
    torch.set_num_threads(nb_threads)  # nombre de coeurs pour les calculs "a l'interieur" d'une operation
    return torch.get_num_threads()


def fixer_graines(graine):
    """
    Fixe toutes les sources de hasard utilisees pendant l'apprentissage, pour des resultats reproductibles.

    Un reseau de neurones utilise le hasard a trois endroits : les valeurs initiales de ses poids, l'ordre
    dans lequel les fenetres sont presentees (melange a chaque epoque) et le dropout (neurones eteints au
    hasard). Avec la meme graine, les memes donnees et la meme machine, on obtient donc le meme modele.
    On demande aussi a PyTorch de n'utiliser que des algorithmes deterministes (avertissement seulement
    si une operation n'a pas de version deterministe, au lieu d'une erreur).

    Parameters
    ----------
    graine : int
        La graine (ex. 42).
    """
    random.seed(graine)  # hasard de la bibliotheque standard de Python
    np.random.seed(graine)  # hasard de numpy
    torch.manual_seed(graine)  # hasard de PyTorch (processeur et, le cas echeant, carte graphique / MPS)
    torch.use_deterministic_algorithms(True, warn_only=True)  # algorithmes deterministes uniquement


def choisir_appareil(appareil="cpu"):
    """
    Renvoie l'appareil de calcul PyTorch demande ("cpu" ou "mps", la puce graphique des Mac).

    Parameters
    ----------
    appareil : str, optionnel
        "cpu" (defaut) ou "mps".

    Returns
    -------
    torch.device
        L'appareil.

    Raises
    ------
    ValueError
        Si "mps" est demande alors qu'il n'est pas disponible, ou si le nom est inconnu.
    """
    if appareil == "cpu":
        return torch.device("cpu")
    if appareil == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("L'appareil 'mps' n'est pas disponible sur cette machine.")
        return torch.device("mps")
    raise ValueError(f"Appareil inconnu : {appareil!r} (choix possibles : 'cpu', 'mps').")


# ---------------------------------------------------------------------------------------------------
# Preparation des entrees
# ---------------------------------------------------------------------------------------------------

def ajouter_time_cycles_echelle(df, echelle=ECHELLE_TIME_CYCLES):
    """
    Ajoute la colonne "time_cycles_echelle" = time_cycles / echelle (constante fixe, rien n'est appris).

    Parameters
    ----------
    df : pandas.DataFrame
        Le tableau, avec une colonne "time_cycles" (n'est pas modifie, une copie est renvoyee).
    echelle : float, optionnel
        La constante de division (100 par defaut).

    Returns
    -------
    pandas.DataFrame
        Copie de df avec la nouvelle colonne.
    """
    df_echelle = df.copy()  # copie independante, pour ne pas modifier df sur place
    df_echelle[NOM_COLONNE_TIME_CYCLES_ECHELLE] = df_echelle["time_cycles"].astype(float) / echelle
    return df_echelle


def lister_colonnes_entree(colonnes_capteurs, avec_time_cycles, colonnes_supplementaires=None):
    """
    Construit la liste ordonnee des colonnes d'entree du reseau : capteurs, colonnes en plus, puis time_cycles.

    Parameters
    ----------
    colonnes_capteurs : list de str
        Les capteurs normalises et lisses (15 pour FD001).
    avec_time_cycles : bool
        Vrai pour ajouter la colonne "time_cycles_echelle" en derniere position.
    colonnes_supplementaires : list de str ou None, optionnel
        Colonnes ajoutees apres les capteurs et avant time_cycles (ex. les ecarts a l'etat initial
        normalises, voir `apprendre_min_max`). None (defaut) : aucune, comportement d'origine.

    Returns
    -------
    list de str
        Les noms de colonnes, dans l'ordre ou le reseau les recoit.
    """
    colonnes = []  # on part d'une liste vide, remplie dans l'ordre
    for colonne in colonnes_capteurs:  # d'abord les capteurs
        colonnes.append(colonne)
    if colonnes_supplementaires is not None:  # puis les colonnes en plus, s'il y en a
        for colonne in colonnes_supplementaires:
            colonnes.append(colonne)
    if avec_time_cycles:  # enfin, si demande, l'age du moteur mis a l'echelle (toujours en dernier)
        colonnes.append(NOM_COLONNE_TIME_CYCLES_ECHELLE)
    return colonnes


def apprendre_min_max(df, colonnes):
    """
    Apprend le minimum et le maximum de chaque colonne, sur le tableau donne (le TRAIN uniquement).

    Sert a mettre a l'echelle des entrees supplementaires des reseaux (ex. l'ecart a l'etat initial,
    src/features.py), comme les capteurs le sont deja par le min-max de la preparation : on apprend les
    bornes sur le train, puis on applique les MEMES bornes au test (`appliquer_min_max`), sans jamais
    regarder le test. Exemple : une colonne qui va de -0,2 a 0,6 sur le train devient (x + 0,2) / 0,8.

    Parameters
    ----------
    df : pandas.DataFrame
        Le tableau d'apprentissage (train prepare).
    colonnes : list de str
        Les colonnes a mettre a l'echelle.

    Returns
    -------
    dict
        colonne -> {"min": float, "max": float}, les bornes vues sur df.
    """
    parametres = {}  # un couple (min, max) par colonne
    for colonne in colonnes:
        valeurs = df[colonne].to_numpy(dtype=float)  # toutes les lignes du train pour cette colonne
        bornes = {}
        bornes["min"] = float(np.min(valeurs))
        bornes["max"] = float(np.max(valeurs))
        parametres[colonne] = bornes
    return parametres


def appliquer_min_max(df, colonnes, parametres, suffixe="_mm"):
    """
    Applique des bornes min-max deja apprises (`apprendre_min_max`) et ajoute les colonnes "<colonne><suffixe>".

    Les valeurs du train tombent entre 0 et 1 ; celles d'un autre tableau (test) peuvent en sortir un peu,
    c'est normal : les bornes ne sont jamais reapprises sur lui. Une colonne constante sur le train (max = min)
    est seulement decalee (division par 1) pour eviter une division par zero.

    Parameters
    ----------
    df : pandas.DataFrame
        Le tableau a transformer (n'est pas modifie sur place, une copie est renvoyee).
    colonnes : list de str
        Les colonnes a mettre a l'echelle (dans cet ordre).
    parametres : dict
        Le resultat de `apprendre_min_max` (appris sur le train).
    suffixe : str, optionnel
        Ajoute au nom de chaque colonne mise a l'echelle ("_mm" par defaut, pour "min-max").

    Returns
    -------
    (pandas.DataFrame, list de str)
        La copie enrichie et les noms des nouvelles colonnes (dans l'ordre de `colonnes`).
    """
    df_echelle = df.copy()  # copie independante, pour ne pas modifier df sur place
    nouvelles_colonnes = []
    for colonne in colonnes:
        minimum = parametres[colonne]["min"]
        etendue = parametres[colonne]["max"] - minimum
        if etendue <= 0:  # colonne constante sur le train : pas de division par zero
            etendue = 1.0
        nom = colonne + suffixe  # ex. "sensor_11_ecartinit_20_mm"
        df_echelle[nom] = (df_echelle[colonne].to_numpy(dtype=float) - minimum) / etendue
        nouvelles_colonnes.append(nom)
    return df_echelle, nouvelles_colonnes


def construire_fenetres(df, colonnes_entree, longueur=LONGUEUR_FENETRE_DEFAUT, colonne_cible=None):
    """
    Construit UNE fenetre glissante par ligne (cycle) du tableau, dans le meme ordre que les lignes.

    La fenetre de la ligne "moteur m, cycle t" contient les `longueur` derniers cycles OBSERVES du MEME
    moteur, du plus ancien au plus recent, le dernier etant le cycle t lui-meme : cycles t-longueur+1 a t.
    Jamais de donnees d'un autre moteur (calcul moteur par moteur), jamais du futur (aucun cycle > t).

    Remplissage au debut de vie : si le moteur n'a encore que k < longueur cycles au cycle t (debut de
    trajectoire, ou moteur test tres court), les longueur - k premieres positions de la fenetre repetent
    la PREMIERE ligne observee du moteur. Exemple avec longueur = 5, au 3e cycle d'un moteur :
    fenetre = [cycle 1, cycle 1, cycle 1, cycle 2, cycle 3]. Pourquoi repeter le premier cycle plutot que
    mettre des zeros ? Un zero est une vraie valeur pour un capteur normalise (son minimum) : il ferait
    croire au reseau a un moteur dans un etat extreme. Repeter le premier cycle dit seulement "rien n'a
    change avant", ce qui est le plus neutre (le moteur est neuf en debut de trajectoire). Le moteur
    lui-meme reste la seule source de donnees : pas d'emprunt a un autre moteur.

    Comme il y a une fenetre par ligne, dans l'ordre des lignes, la prediction de la fenetre i correspond
    exactement a la ligne i du tableau : les predictions hors-pli d'un reseau s'alignent donc ligne a
    ligne sur celles des arbres (comparaison appariee possible), et on peut predire a chaque cycle d'un
    moteur test (courbes de la demo).

    Parameters
    ----------
    df : pandas.DataFrame
        Le tableau, avec "unit_number", "time_cycles" et les colonnes d'entree (aucun NaN). Les lignes d'un
        meme moteur peuvent etre dans n'importe quel ordre : on les remet dans l'ordre des cycles pour
        construire les fenetres, mais le resultat suit l'ordre des lignes de df.
    colonnes_entree : list de str
        Les colonnes qui forment les entrees du reseau (une "couche" de la fenetre par colonne).
    longueur : int, optionnel
        Nombre de cycles par fenetre (30 par defaut).
    colonne_cible : str ou None, optionnel
        Nom de la colonne cible (ex. "RUL_cappee") ; None si on n'en a pas besoin (ex. prediction).

    Returns
    -------
    tuple
        (fenetres, cible, unit_number, time_cycles) :

        - fenetres : numpy.ndarray float32 de forme (nb_lignes, longueur, nb_entrees), le tenseur 3D ;
        - cible : numpy.ndarray float32 (nb_lignes,), ou None si colonne_cible est None ;
        - unit_number : numpy.ndarray (nb_lignes,), le moteur de chaque fenetre ;
        - time_cycles : numpy.ndarray (nb_lignes,), le cycle "present" (dernier) de chaque fenetre.

    Raises
    ------
    ValueError
        Si une colonne manque, si les entrees contiennent des NaN, si longueur < 1, ou si un moteur a deux
        lignes pour le meme cycle.
    """
    if longueur < 1:
        raise ValueError(f"La longueur de fenetre doit etre >= 1 (recu : {longueur}).")

    # verification : toutes les colonnes demandees existent
    colonnes_necessaires = ["unit_number", "time_cycles"]
    for colonne in colonnes_entree:
        colonnes_necessaires.append(colonne)
    for colonne in colonnes_necessaires:
        if colonne not in df.columns:
            raise ValueError(f"Colonne manquante pour construire les fenetres : {colonne!r}.")

    valeurs = df[colonnes_entree].to_numpy(dtype=np.float32)  # (nb_lignes, nb_entrees)
    if np.isnan(valeurs).any():
        raise ValueError("Les colonnes d'entree contiennent des NaN : impossible de construire les fenetres.")

    numeros = df["unit_number"].to_numpy()  # moteur de chaque ligne
    cycles = df["time_cycles"].to_numpy()  # cycle de chaque ligne
    nb_lignes = len(df)
    nb_entrees = len(colonnes_entree)

    # le tenseur 3D a remplir : une fenetre (longueur x nb_entrees) par ligne
    fenetres = np.zeros((nb_lignes, longueur, nb_entrees), dtype=np.float32)

    # decalages des positions de la fenetre par rapport au cycle present : -(longueur-1), ..., -1, 0
    decalages = np.arange(longueur) - (longueur - 1)

    for moteur_id in pd.unique(numeros):  # un moteur a la fois : jamais de melange entre moteurs
        positions = np.flatnonzero(numeros == moteur_id)  # numeros des lignes de ce moteur dans df
        ordre_chronologique = np.argsort(cycles[positions], kind="stable")  # du plus ancien au plus recent
        positions_triees = positions[ordre_chronologique]

        # garde-fou : un seul enregistrement par cycle et par moteur
        cycles_tries = cycles[positions_triees]
        if len(cycles_tries) > 1 and not np.all(np.diff(cycles_tries) > 0):
            raise ValueError(f"Le moteur {moteur_id} a plusieurs lignes pour un meme cycle.")

        valeurs_moteur = valeurs[positions_triees]  # trajectoire du moteur, ordre chronologique

        # rang de chaque cycle dans la trajectoire (0 = premier cycle observe)
        rangs = np.arange(len(positions_triees))
        # indices[k, j] = rang du cycle a la position j de la fenetre du k-ieme cycle : k-(longueur-1)+j
        indices = rangs[:, np.newaxis] + decalages[np.newaxis, :]
        # un rang negatif "avant le debut du moteur" est remplace par 0, c'est-a-dire par la premiere
        # ligne du moteur : c'est le remplissage par repetition. Aucun rang ne depasse k : jamais de futur.
        indices = np.maximum(indices, 0)

        # valeurs_moteur[indices] a la forme (nb_cycles_du_moteur, longueur, nb_entrees) ; on range chaque
        # fenetre a la place de SA ligne dans df, pour garder l'ordre des lignes
        fenetres[positions_triees] = valeurs_moteur[indices]

    cible = None
    if colonne_cible is not None:
        if colonne_cible not in df.columns:
            raise ValueError(f"Colonne cible manquante : {colonne_cible!r}.")
        cible = df[colonne_cible].to_numpy(dtype=np.float32)

    return fenetres, cible, numeros.copy(), cycles.copy()


class FenetresParesseuses:
    """
    Les memes fenetres que `construire_fenetres`, mais construites a la demande, lot par lot.

    Pourquoi ? Le tenseur 3D de `construire_fenetres` recopie chaque ligne `longueur` fois : pour FD002
    avec 50 cycles, 53 759 x 50 x 31 valeurs float32, soit environ 330 Mo par processus, plus une copie
    par pli de validation croisee. Avec 15 apprentissages en parallele, la memoire de la machine ne suit
    plus (constate le 2026-11-06). Ici, on ne garde que :

    - `valeurs` : le tableau 2D des entrees, une ligne par cycle (environ 7 Mo pour FD002) ;
    - `lignes` : pour chaque fenetre, les numeros des `longueur` lignes de `valeurs` qui la composent
      (entiers sur 32 bits, environ 10 Mo pour FD002 avec 50 cycles).

    La fenetre i vaut alors `valeurs[lignes[i]]` : EXACTEMENT les memes nombres, dans le meme ordre, que
    `construire_fenetres(...)[0][i]` (meme regle : jamais un autre moteur, jamais le futur, premier cycle
    repete en debut de vie). Les apprentissages donnent donc les memes poids au bit pres ; un test le
    verifie (tests/test_fenetres_paresseuses.py).

    Utilisation, comme un tableau numpy (n, longueur, n_entrees) :

    - `fenetres[indices]` ou `fenetres[debut:fin]` : CONSTRUIT ces fenetres et renvoie un numpy.ndarray
      (c'est ce que font `entrainer_modele` et `predire_echelle`, un lot a la fois) ;
    - `fenetres.selection(indices)` : un sous-ensemble, LUI AUSSI paresseux (rien n'est construit), pour
      separer les moteurs d'un pli sans copier de fenetres ;
    - `len(fenetres)`, `fenetres.shape`.
    """

    def __init__(self, valeurs, lignes):
        """
        Parameters
        ----------
        valeurs : numpy.ndarray float32
            Forme (nb_lignes, n_entrees) : les entrees de chaque cycle.
        lignes : numpy.ndarray int32
            Forme (nb_fenetres, longueur) : les lignes de `valeurs` qui forment chaque fenetre.
        """
        self.valeurs = valeurs
        self.lignes = lignes

    @property
    def shape(self):
        """Forme du tenseur 3D equivalent : (nb_fenetres, longueur, n_entrees)."""
        return (self.lignes.shape[0], self.lignes.shape[1], self.valeurs.shape[1])

    def __len__(self):
        """Nombre de fenetres."""
        return self.lignes.shape[0]

    def __getitem__(self, cle):
        """
        Construit les fenetres demandees (indices, masque, tranche ou tenseur torch d'indices).

        Returns
        -------
        numpy.ndarray float32
            Forme (nb_demandees, longueur, n_entrees), ou (longueur, n_entrees) pour un seul entier.
        """
        if isinstance(cle, torch.Tensor):
            cle = cle.numpy()
        return self.valeurs[self.lignes[cle]]  # une seule operation numpy : (n, longueur) -> (n, longueur, n_entrees)

    def selection(self, cle):
        """
        Sous-ensemble paresseux des fenetres (indices, masque ou tranche) : aucune fenetre n'est construite.

        Returns
        -------
        FenetresParesseuses
            Le sous-ensemble ; il partage le tableau `valeurs` (pas de copie).
        """
        if isinstance(cle, torch.Tensor):
            cle = cle.numpy()
        return FenetresParesseuses(self.valeurs, self.lignes[cle])


def construire_fenetres_paresseuses(df, colonnes_entree, longueur=LONGUEUR_FENETRE_DEFAUT, colonne_cible=None):
    """
    Meme travail et memes verifications que `construire_fenetres`, mais renvoie des FenetresParesseuses.

    Le calcul des lignes de chaque fenetre est celui de `construire_fenetres` (moteur par moteur, ordre des
    cycles, rang negatif remplace par 0) ; seule la derniere etape change : on garde les NUMEROS des lignes
    au lieu de recopier leurs valeurs.

    Parameters
    ----------
    df, colonnes_entree, longueur, colonne_cible :
        Comme `construire_fenetres`.

    Returns
    -------
    tuple
        (fenetres, cible, unit_number, time_cycles), comme `construire_fenetres`, avec `fenetres` de type
        FenetresParesseuses.
    """
    if longueur < 1:
        raise ValueError(f"La longueur de fenetre doit etre >= 1 (recu : {longueur}).")

    # verification : toutes les colonnes demandees existent
    colonnes_necessaires = ["unit_number", "time_cycles"]
    for colonne in colonnes_entree:
        colonnes_necessaires.append(colonne)
    for colonne in colonnes_necessaires:
        if colonne not in df.columns:
            raise ValueError(f"Colonne manquante pour construire les fenetres : {colonne!r}.")

    valeurs = df[colonnes_entree].to_numpy(dtype=np.float32)  # (nb_lignes, nb_entrees), ordre des lignes de df
    if np.isnan(valeurs).any():
        raise ValueError("Les colonnes d'entree contiennent des NaN : impossible de construire les fenetres.")

    numeros = df["unit_number"].to_numpy()
    cycles = df["time_cycles"].to_numpy()
    lignes = np.zeros((len(df), longueur), dtype=np.int32)  # lignes[i, j] : ligne de df a la position j de la fenetre i
    decalages = np.arange(longueur) - (longueur - 1)  # -(longueur-1), ..., -1, 0

    for moteur_id in pd.unique(numeros):  # un moteur a la fois : jamais de melange entre moteurs
        positions = np.flatnonzero(numeros == moteur_id)
        ordre_chronologique = np.argsort(cycles[positions], kind="stable")
        positions_triees = positions[ordre_chronologique]

        # garde-fou : un seul enregistrement par cycle et par moteur
        cycles_tries = cycles[positions_triees]
        if len(cycles_tries) > 1 and not np.all(np.diff(cycles_tries) > 0):
            raise ValueError(f"Le moteur {moteur_id} a plusieurs lignes pour un meme cycle.")

        rangs = np.arange(len(positions_triees))
        indices = np.maximum(rangs[:, np.newaxis] + decalages[np.newaxis, :], 0)  # meme regle que construire_fenetres
        # rang dans la trajectoire -> numero de ligne dans df ; chaque fenetre rangee a la place de SA ligne
        lignes[positions_triees] = positions_triees[indices]

    cible = None
    if colonne_cible is not None:
        if colonne_cible not in df.columns:
            raise ValueError(f"Colonne cible manquante : {colonne_cible!r}.")
        cible = df[colonne_cible].to_numpy(dtype=np.float32)

    return FenetresParesseuses(valeurs, lignes), cible, numeros.copy(), cycles.copy()


# ---------------------------------------------------------------------------------------------------
# Les deux architectures
# ---------------------------------------------------------------------------------------------------

class ModeleLSTM(nn.Module):
    """
    Reseau LSTM : lit la fenetre cycle par cycle, du plus ancien au plus recent, puis predit la RUL.

    Architecture (standard, volontairement simple) :

    1. deux couches LSTM empilees de 64 unites. A chaque cycle de la fenetre, une LSTM met a jour une
       "memoire" de 64 nombres a partir du cycle lu et de sa memoire precedente ; des "portes" apprises
       decident ce qu'il faut garder ou oublier. La 2e couche lit la suite des memoires de la 1re ;
    2. on garde la sortie du DERNIER cycle de la fenetre (le cycle present), qui a "vu" toute la fenetre ;
    3. dropout (20 %) puis une couche lineaire 64 -> 1 : la RUL mise a l'echelle (RUL / 125).

    Pourquoi ces valeurs ? 64 unites et 2 couches sont des tailles moyennes pour 15 a 16 entrees et
    30 cycles (environ 54 000 poids, a comparer aux 20 631 fenetres de FD001 : assez pour apprendre des
    motifs, pas assez pour tout memoriser, le dropout et l'arret precoce limitant le reste). Ces reglages
    ne sont pas optimises (pas de recherche d'hyperparametres pour les reseaux, faute de temps de calcul).

    Parameters
    ----------
    n_entrees : int
        Nombre de colonnes d'entree par cycle (15 capteurs, ou 16 avec time_cycles).
    taille_cachee : int, optionnel
        Nombre d'unites de chaque couche LSTM (64 par defaut).
    nb_couches : int, optionnel
        Nombre de couches LSTM empilees (2 par defaut).
    dropout : float, optionnel
        Proportion de neurones eteints au hasard pendant l'apprentissage, entre les couches LSTM et avant
        la couche de sortie (0,2 par defaut). Aucun effet en prediction.
    """

    def __init__(self, n_entrees, taille_cachee=64, nb_couches=2, dropout=0.2):
        """
        Cree les couches du reseau (les poids sont tires au hasard : fixer la graine avant).

        Parameters
        ----------
        n_entrees : int
            Nombre de colonnes d'entree par cycle.
        taille_cachee : int, optionnel
            Nombre d'unites par couche LSTM.
        nb_couches : int, optionnel
            Nombre de couches LSTM.
        dropout : float, optionnel
            Proportion de dropout.
        """
        super().__init__()
        # PyTorch n'applique le dropout entre couches LSTM que s'il y a au moins 2 couches
        dropout_entre_couches = 0.0
        if nb_couches > 1:
            dropout_entre_couches = dropout
        # batch_first=True : les entrees ont la forme (lot, cycles, entrees), comme nos fenetres
        self.lstm = nn.LSTM(
            input_size=n_entrees,
            hidden_size=taille_cachee,
            num_layers=nb_couches,
            batch_first=True,
            dropout=dropout_entre_couches,
        )
        self.dropout = nn.Dropout(dropout)  # avant la couche de sortie
        self.sortie = nn.Linear(taille_cachee, 1)  # 64 nombres -> 1 RUL (mise a l'echelle)

    def forward(self, x):
        """
        Calcule la RUL mise a l'echelle de chaque fenetre du lot.

        Parameters
        ----------
        x : torch.Tensor
            Les fenetres, forme (lot, longueur, n_entrees).

        Returns
        -------
        torch.Tensor
            Forme (lot,) : la RUL predite divisee par 125.
        """
        sorties, _ = self.lstm(x)  # sorties de la 2e couche a chaque cycle : (lot, longueur, taille_cachee)
        derniere_sortie = sorties[:, -1, :]  # celle du dernier cycle (le present) : (lot, taille_cachee)
        derniere_sortie = self.dropout(derniere_sortie)
        prediction = self.sortie(derniere_sortie)  # (lot, 1)
        return prediction.squeeze(-1)  # (lot,)


class ModeleCNN1D(nn.Module):
    """
    Reseau convolutif 1D : fait glisser de petits filtres le long des cycles de la fenetre, puis predit la RUL.

    Architecture (standard, inspiree des reseaux "tout convolutifs" pour series temporelles : blocs
    Conv1d + normalisation + ReLU, puis moyenne globale, puis couche lineaire) :

    1. trois blocs "convolution 1D -> BatchNorm -> ReLU", avec 32, 64 puis 64 filtres de largeur 5, 5 et
       3 cycles. Un filtre de largeur 5 regarde 5 cycles consecutifs de toutes les entrees a la fois et
       produit un nombre (par exemple "le capteur 11 monte et le capteur 12 baisse sur ces 5 cycles") ; il
       est applique a toutes les positions de la fenetre. Le remplissage ("padding") garde 30 positions.
       BatchNorm remet les sorties de chaque filtre a une echelle stable, ce qui accelere l'apprentissage ;
       ReLU (max(0, x)) apporte la non-linearite. Apres les 3 blocs, chaque position "voit" 11 cycles ;
    2. moyenne globale sur les 30 positions ("global average pooling") : un nombre par filtre (64), qui
       dit a quel point chaque motif est present dans la fenetre ;
    3. dropout (20 %) puis couche lineaire 64 -> 1 : la RUL mise a l'echelle.

    Environ 26 000 poids. Reglages non optimises, comme pour le LSTM.

    Parameters
    ----------
    n_entrees : int
        Nombre de colonnes d'entree par cycle (les "canaux" d'entree de la 1re convolution).
    canaux : list de int, optionnel
        Nombre de filtres de chaque bloc ([32, 64, 64] par defaut).
    noyaux : list de int, optionnel
        Largeur (impaire, en cycles) des filtres de chaque bloc ([5, 5, 3] par defaut).
    dropout : float, optionnel
        Proportion de dropout avant la couche de sortie (0,2 par defaut).
    """

    def __init__(self, n_entrees, canaux=(32, 64, 64), noyaux=(5, 5, 3), dropout=0.2):
        """
        Cree les couches du reseau (les poids sont tires au hasard : fixer la graine avant).

        Parameters
        ----------
        n_entrees : int
            Nombre de colonnes d'entree par cycle.
        canaux : list de int, optionnel
            Nombre de filtres par bloc.
        noyaux : list de int, optionnel
            Largeur impaire des filtres par bloc.
        dropout : float, optionnel
            Proportion de dropout.

        Raises
        ------
        ValueError
            Si canaux et noyaux n'ont pas la meme longueur, ou si une largeur de filtre est paire.
        """
        super().__init__()
        if len(canaux) != len(noyaux):
            raise ValueError("canaux et noyaux doivent avoir le meme nombre d'elements (un par bloc).")

        couches = []  # liste des couches, dans l'ordre ou les donnees les traversent
        canaux_entree = n_entrees  # le 1er bloc recoit les colonnes d'entree comme "canaux"
        for position in range(len(canaux)):  # un bloc de convolution a la fois
            largeur = int(noyaux[position])
            if largeur % 2 == 0:
                raise ValueError(f"La largeur de filtre doit etre impaire (recu : {largeur}).")
            nb_filtres = int(canaux[position])
            # padding = largeur // 2 : autant de positions ajoutees de chaque cote, la sortie garde 30 positions
            couches.append(nn.Conv1d(canaux_entree, nb_filtres, kernel_size=largeur, padding=largeur // 2))
            couches.append(nn.BatchNorm1d(nb_filtres))  # normalisation de chaque filtre sur le lot
            couches.append(nn.ReLU())  # non-linearite : max(0, x)
            canaux_entree = nb_filtres  # le bloc suivant recoit les sorties de ce bloc
        self.convolutions = nn.Sequential(*couches)  # enchaine les couches dans l'ordre de la liste
        self.dropout = nn.Dropout(dropout)
        self.sortie = nn.Linear(canaux_entree, 1)  # 64 nombres -> 1 RUL (mise a l'echelle)

    def forward(self, x):
        """
        Calcule la RUL mise a l'echelle de chaque fenetre du lot.

        Parameters
        ----------
        x : torch.Tensor
            Les fenetres, forme (lot, longueur, n_entrees).

        Returns
        -------
        torch.Tensor
            Forme (lot,) : la RUL predite divisee par 125.
        """
        x = x.transpose(1, 2)  # Conv1d attend (lot, canaux, longueur) : on echange les deux derniers axes
        x = self.convolutions(x)  # (lot, 64, longueur)
        x = x.mean(dim=2)  # moyenne globale sur les positions de la fenetre : (lot, 64)
        x = self.dropout(x)
        prediction = self.sortie(x)  # (lot, 1)
        return prediction.squeeze(-1)  # (lot,)


def hyperparametres_par_defaut(type_modele):
    """
    Renvoie une copie des hyperparametres par defaut d'un type de reseau.

    Parameters
    ----------
    type_modele : str
        "lstm" ou "cnn1d".

    Returns
    -------
    dict
        Copie independante des hyperparametres par defaut.

    Raises
    ------
    ValueError
        Si le type est inconnu.
    """
    if type_modele == "lstm":
        source = HYPERPARAMETRES_LSTM_DEFAUT
    elif type_modele == "cnn1d":
        source = HYPERPARAMETRES_CNN_DEFAUT
    else:
        raise ValueError(f"Type de modele inconnu : {type_modele!r} (choix possibles : {TYPES_MODELES}).")
    copie = {}  # copie cle par cle (les listes sont recopiees pour ne jamais modifier la constante)
    for cle in source:
        valeur = source[cle]
        if isinstance(valeur, list):
            copie[cle] = list(valeur)
        else:
            copie[cle] = valeur
    return copie


def creer_modele(type_modele, n_entrees, graine=42, hyperparametres=None):
    """
    Cree un reseau NEUF (poids initiaux tires au hasard avec une graine fixee).

    Parameters
    ----------
    type_modele : str
        "lstm" ou "cnn1d".
    n_entrees : int
        Nombre de colonnes d'entree par cycle.
    graine : int, optionnel
        Graine des poids initiaux (42 par defaut) : meme graine = memes poids de depart.
    hyperparametres : dict ou None, optionnel
        Hyperparametres a changer ; les autres gardent leur valeur par defaut.

    Returns
    -------
    torch.nn.Module
        Le reseau, non entraine, sur le processeur.
    """
    parametres = hyperparametres_par_defaut(type_modele)  # valeurs par defaut...
    if hyperparametres is not None:
        for cle in hyperparametres:  # ... ecrasees par celles de l'appelant
            parametres[cle] = hyperparametres[cle]

    fixer_graines(graine)  # les poids initiaux dependent de cette graine
    if type_modele == "lstm":
        return ModeleLSTM(
            n_entrees,
            taille_cachee=parametres["taille_cachee"],
            nb_couches=parametres["nb_couches"],
            dropout=parametres["dropout"],
        )
    return ModeleCNN1D(
        n_entrees,
        canaux=parametres["canaux"],
        noyaux=parametres["noyaux"],
        dropout=parametres["dropout"],
    )


def compter_parametres(modele):
    """
    Compte les poids appris d'un reseau.

    Parameters
    ----------
    modele : torch.nn.Module
        Le reseau.

    Returns
    -------
    int
        Le nombre total de nombres ajustes pendant l'apprentissage.
    """
    total = 0
    for parametre in modele.parameters():  # chaque tableau de poids du reseau
        if parametre.requires_grad:
            total = total + parametre.numel()  # numel = nombre d'elements du tableau
    return int(total)


# ---------------------------------------------------------------------------------------------------
# Perte, prediction et apprentissage
# ---------------------------------------------------------------------------------------------------

def perte_quadratique_asymetrique_torch(predictions, cibles, alpha=1.0):
    """
    Perte quadratique asymetrique moyenne (version PyTorch de src/modeles.py) : moyenne de w * (p - y)^2.

    w = alpha si p > y (surestimation, le cas dangereux), 1 sinon. alpha = 1 redonne l'erreur quadratique
    moyenne classique (MSE). Meme perte que celle des arbres (src/modeles.py), mais ecrite avec des operations
    PyTorch pour que le gradient soit calcule automatiquement.

    Parameters
    ----------
    predictions : torch.Tensor
        Les predictions p (forme (lot,)).
    cibles : torch.Tensor
        Les vraies valeurs y (meme forme).
    alpha : float, optionnel
        Poids des surestimations (1 par defaut).

    Returns
    -------
    torch.Tensor
        Un nombre (tenseur de dimension 0) : la perte moyenne du lot.
    """
    residus = predictions - cibles  # positif = surestimation
    poids_surestimation = torch.full_like(residus, float(alpha))  # alpha partout...
    poids_un = torch.ones_like(residus)  # ... ou 1 partout
    poids = torch.where(residus > 0, poids_surestimation, poids_un)  # alpha la ou p > y, 1 ailleurs
    return torch.mean(poids * residus ** 2)


def appareil_du_modele(modele):
    """
    Renvoie l'appareil (cpu ou mps) sur lequel se trouvent les poids d'un reseau.

    Parameters
    ----------
    modele : torch.nn.Module
        Le reseau.

    Returns
    -------
    torch.device
        L'appareil de son premier tableau de poids.
    """
    for parametre in modele.parameters():
        return parametre.device
    return torch.device("cpu")  # reseau sans poids (cas theorique)


def predire_echelle(modele, fenetres, taille_lot=2048):
    """
    Calcule les sorties BRUTES du reseau (RUL / 125, sans clip), en mode prediction (dropout eteint).

    Parameters
    ----------
    modele : torch.nn.Module
        Le reseau.
    fenetres : numpy.ndarray, torch.Tensor ou FenetresParesseuses
        Les fenetres, forme (n, longueur, n_entrees).
    taille_lot : int, optionnel
        Nombre de fenetres traitees a la fois (sans effet sur le resultat, seulement sur la memoire).

    Returns
    -------
    numpy.ndarray
        Forme (n,) : les sorties du reseau (float64).
    """
    appareil = appareil_du_modele(modele)
    entrees = fenetres
    if not isinstance(fenetres, FenetresParesseuses):  # paresseuses : chaque lot est construit a la demande
        entrees = torch.as_tensor(fenetres, dtype=torch.float32)  # pas de copie si c'est deja du float32
    modele.eval()  # mode prediction : dropout eteint, BatchNorm utilise ses statistiques apprises

    morceaux = []  # sorties de chaque lot, recollees a la fin
    with torch.no_grad():  # pas de calcul de gradient : plus rapide, moins de memoire
        for debut in range(0, len(entrees), taille_lot):  # un lot de fenetres a la fois
            lot = torch.as_tensor(entrees[debut:debut + taille_lot], dtype=torch.float32).to(appareil)
            sortie = modele(lot)
            morceaux.append(sortie.detach().cpu().numpy())

    if len(morceaux) == 0:  # aucune fenetre : tableau vide
        return np.zeros(0, dtype=float)
    return np.concatenate(morceaux).astype(float)


def predire(modele, fenetres, taille_lot=2048, echelle_cible=ECHELLE_CIBLE):
    """
    Predit la RUL (en cycles) de chaque fenetre : sortie du reseau x 125, puis clip a 0.

    Une RUL negative n'a pas de sens physique : on la ramene a 0, comme pour les modeles a arbres sur le
    test. (Attention : la validation croisee des arbres, elle, ne clippe pas ; voir le notebook 03b.)

    Parameters
    ----------
    modele : torch.nn.Module
        Le reseau entraine.
    fenetres : numpy.ndarray, torch.Tensor ou FenetresParesseuses
        Les fenetres, forme (n, longueur, n_entrees).
    taille_lot : int, optionnel
        Nombre de fenetres traitees a la fois.
    echelle_cible : float, optionnel
        L'echelle de la cible utilisee a l'apprentissage (125 par defaut).

    Returns
    -------
    numpy.ndarray
        Forme (n,) : les RUL predites, en cycles, >= 0.
    """
    sorties = predire_echelle(modele, fenetres, taille_lot=taille_lot)
    predictions = sorties * echelle_cible  # retour en cycles
    return np.clip(predictions, a_min=0.0, a_max=None)  # une RUL negative n'a pas de sens physique


def copier_etat(modele):
    """
    Fait une copie independante des poids d'un reseau (pour garder ceux de la meilleure epoque).

    Parameters
    ----------
    modele : torch.nn.Module
        Le reseau.

    Returns
    -------
    dict
        Nom de chaque tableau de poids -> copie de ce tableau.
    """
    etat = modele.state_dict()  # dictionnaire nom -> tableau de poids (les vrais tableaux du reseau)
    copie = {}
    for nom in etat:  # on recopie chaque tableau, sinon la copie changerait avec le reseau
        copie[nom] = etat[nom].detach().clone()
    return copie


def entrainer_modele(
    modele,
    fenetres_entrainement,
    y_entrainement,
    fenetres_validation=None,
    y_validation=None,
    alpha=1.0,
    taux_apprentissage=1e-3,
    taille_lot=256,
    max_epoques=80,
    patience=10,
    nb_epoques=None,
    graine=42,
    appareil="cpu",
    echelle_cible=ECHELLE_CIBLE,
    verbeux=False,
):
    """
    Entraine un reseau avec Adam, par lots, avec arret precoce sur des moteurs de validation.

    Deroulement d'une EPOQUE (un passage complet sur les fenetres d'apprentissage) :
    1. on melange les fenetres (ordre tire avec la graine) et on les decoupe en LOTS de `taille_lot` ;
    2. pour chaque lot : prediction, calcul de la perte (quadratique, asymetrique si alpha > 1, sur la
       cible divisee par 125), calcul du gradient (dans quel sens bouger chaque poids pour diminuer la
       perte), puis un pas de l'optimiseur Adam ;
    3. si des fenetres de validation sont fournies : on calcule la perte de validation (dropout eteint).

    ARRET PRECOCE (si validation fournie) : on garde les poids de l'epoque ou la perte de validation a ete
    la plus basse ; si elle ne s'ameliore plus pendant `patience` epoques, on arrete et on remet ces poids.
    Les moteurs de validation ne servent qu'a decider QUAND s'arreter, jamais a ajuster les poids.

    SANS validation (modele final) : `nb_epoques` doit etre donne (fixe a l'avance par la validation) ;
    on entraine exactement ce nombre d'epoques et on garde les poids de la derniere.

    Parameters
    ----------
    modele : torch.nn.Module
        Le reseau neuf (voir `creer_modele`), modifie sur place.
    fenetres_entrainement : numpy.ndarray ou FenetresParesseuses
        Fenetres d'apprentissage, forme (n, longueur, n_entrees).
    y_entrainement : array-like
        RUL cible de chaque fenetre, EN CYCLES (plafonnee a 125).
    fenetres_validation : numpy.ndarray, FenetresParesseuses ou None, optionnel
        Fenetres des moteurs de validation (arret precoce) ; None pour un nombre d'epoques fixe.
    y_validation : array-like ou None, optionnel
        RUL cible des fenetres de validation, en cycles.
    alpha : float, optionnel
        Poids des surestimations dans la perte (1 = MSE classique).
    taux_apprentissage : float, optionnel
        Pas d'Adam (1e-3 par defaut).
    taille_lot : int, optionnel
        Nombre de fenetres par lot (256 par defaut).
    max_epoques : int, optionnel
        Nombre maximal d'epoques avec arret precoce (80 par defaut).
    patience : int, optionnel
        Nombre d'epoques sans amelioration avant l'arret (10 par defaut).
    nb_epoques : int ou None, optionnel
        Nombre exact d'epoques quand il n'y a pas de validation.
    graine : int, optionnel
        Graine du melange des lots et du dropout (42 par defaut).
    appareil : str, optionnel
        "cpu" (defaut) ou "mps".
    echelle_cible : float, optionnel
        Division appliquee a la cible pendant l'apprentissage (125 par defaut).
    verbeux : bool, optionnel
        Vrai pour afficher la perte a chaque epoque.

    Returns
    -------
    dict
        L'historique de l'apprentissage :

        - "perte_entrainement" : liste, perte moyenne des lots de chaque epoque (cible / 125, dropout actif) ;
        - "perte_validation" : liste, perte de validation de chaque epoque (vide sans validation) ;
        - "rmse_validation" : liste, RMSE de validation en cycles de chaque epoque (vide sans validation) ;
        - "meilleure_epoque" : int, epoque (numerotee a partir de 1) des poids gardes ;
        - "nb_epoques_effectuees" : int ; "arret_precoce" : bool (arret avant max_epoques) ;
        - "duree_s" : float, duree totale ; "duree_par_epoque_s" : float ;
        - "nb_fenetres_entrainement", "nb_fenetres_validation" : int.

    Raises
    ------
    ValueError
        Si ni validation ni nb_epoques ne sont fournis.
    """
    avec_validation = fenetres_validation is not None
    if not avec_validation and nb_epoques is None:
        raise ValueError("Sans fenetres de validation, il faut donner nb_epoques (fixe par la validation).")

    appareil_torch = choisir_appareil(appareil)
    fixer_graines(graine)  # melange des lots et dropout reproductibles
    generateur = torch.Generator()  # generateur dedie au melange des fenetres (sur le processeur)
    generateur.manual_seed(graine)

    modele.to(appareil_torch)  # on place les poids sur l'appareil de calcul
    optimiseur = torch.optim.Adam(modele.parameters(), lr=taux_apprentissage)

    # conversion en tenseurs PyTorch, cible divisee par 125 (entre 0 et 1)
    # (FenetresParesseuses : gardees telles quelles, chaque lot est construit a la demande ; memes nombres)
    x_entrainement = fenetres_entrainement
    if not isinstance(fenetres_entrainement, FenetresParesseuses):
        x_entrainement = torch.as_tensor(fenetres_entrainement, dtype=torch.float32)
    cible_entrainement = torch.as_tensor(np.asarray(y_entrainement, dtype=np.float32) / echelle_cible)
    nb_fenetres = len(x_entrainement)

    if avec_validation:
        x_validation = fenetres_validation
        if not isinstance(fenetres_validation, FenetresParesseuses):
            x_validation = torch.as_tensor(fenetres_validation, dtype=torch.float32)
        y_validation_cycles = np.asarray(y_validation, dtype=float)
        cible_validation = torch.as_tensor((y_validation_cycles / echelle_cible).astype(np.float32))
        nb_epoques_max = max_epoques
    else:
        nb_epoques_max = nb_epoques

    historique = {}
    historique["perte_entrainement"] = []
    historique["perte_validation"] = []
    historique["rmse_validation"] = []

    meilleure_perte = float("inf")  # meilleure perte de validation vue jusqu'ici
    meilleure_epoque = 0
    meilleur_etat = None
    epoques_sans_amelioration = 0
    arret_precoce = False

    debut = time.perf_counter()  # chronometre de l'apprentissage
    for epoque in range(1, nb_epoques_max + 1):  # une epoque = un passage sur toutes les fenetres
        modele.train()  # mode apprentissage : dropout actif, BatchNorm calcule ses statistiques sur le lot
        ordre = torch.randperm(nb_fenetres, generator=generateur)  # nouvel ordre aleatoire des fenetres
        somme_pertes = 0.0

        for debut_lot in range(0, nb_fenetres, taille_lot):  # un lot a la fois
            indices_lot = ordre[debut_lot:debut_lot + taille_lot]
            x_lot = torch.as_tensor(x_entrainement[indices_lot], dtype=torch.float32).to(appareil_torch)
            y_lot = cible_entrainement[indices_lot].to(appareil_torch)

            optimiseur.zero_grad()  # on efface les gradients du lot precedent
            sortie = modele(x_lot)  # prediction (mise a l'echelle) de chaque fenetre du lot
            perte = perte_quadratique_asymetrique_torch(sortie, y_lot, alpha=alpha)
            perte.backward()  # calcul du gradient de la perte par rapport a chaque poids
            optimiseur.step()  # un pas d'Adam : chaque poids bouge un peu dans le sens qui reduit la perte

            somme_pertes = somme_pertes + perte.item() * len(indices_lot)  # perte totale du lot

        perte_epoque = somme_pertes / nb_fenetres  # perte moyenne par fenetre sur l'epoque
        historique["perte_entrainement"].append(perte_epoque)

        if not avec_validation:  # modele final : pas de validation, on continue jusqu'a nb_epoques
            if verbeux:
                print(f"  epoque {epoque:3d} : perte entrainement {perte_epoque:.5f}", flush=True)
            continue

        # perte de validation, dropout eteint (meme perte que l'apprentissage, sur la cible / 125)
        sorties_validation = predire_echelle(modele, x_validation)
        predictions_validation = torch.as_tensor(sorties_validation, dtype=torch.float32)
        perte_validation = float(perte_quadratique_asymetrique_torch(predictions_validation, cible_validation, alpha=alpha))
        # RMSE de validation en cycles (predictions remises en cycles et clippees a 0, comme `predire`)
        predictions_cycles = np.clip(sorties_validation * echelle_cible, a_min=0.0, a_max=None)
        rmse_validation = rmse(y_validation_cycles, predictions_cycles)
        historique["perte_validation"].append(perte_validation)
        historique["rmse_validation"].append(rmse_validation)

        if verbeux:
            print(f"  epoque {epoque:3d} : perte entrainement {perte_epoque:.5f}, perte validation "
                  f"{perte_validation:.5f}, RMSE validation {rmse_validation:.2f}", flush=True)

        if perte_validation < meilleure_perte:  # amelioration : on garde ces poids
            meilleure_perte = perte_validation
            meilleure_epoque = epoque
            meilleur_etat = copier_etat(modele)
            epoques_sans_amelioration = 0
        else:  # pas d'amelioration : on se rapproche de l'arret
            epoques_sans_amelioration = epoques_sans_amelioration + 1
            if epoques_sans_amelioration >= patience:
                arret_precoce = True
                break

    duree = time.perf_counter() - debut
    nb_epoques_effectuees = len(historique["perte_entrainement"])

    if avec_validation:
        modele.load_state_dict(meilleur_etat)  # on remet les poids de la meilleure epoque
    else:
        meilleure_epoque = nb_epoques_effectuees  # sans validation : on garde la derniere epoque

    modele.eval()  # le reseau rendu est en mode prediction
    historique["meilleure_epoque"] = int(meilleure_epoque)
    historique["nb_epoques_effectuees"] = int(nb_epoques_effectuees)
    historique["arret_precoce"] = bool(arret_precoce)
    historique["duree_s"] = float(duree)
    historique["duree_par_epoque_s"] = float(duree / max(nb_epoques_effectuees, 1))
    historique["nb_fenetres_entrainement"] = int(nb_fenetres)
    nb_fenetres_validation = 0
    if avec_validation:
        nb_fenetres_validation = int(len(x_validation))
    historique["nb_fenetres_validation"] = nb_fenetres_validation
    return historique


# ---------------------------------------------------------------------------------------------------
# Metriques, separation par moteurs et validation croisee
# ---------------------------------------------------------------------------------------------------

def calculer_metriques(y_true, y_pred, seuil_surestimation=20, seuil_alerte=30):
    """
    Calcule les metriques usuelles du projet sur un ensemble de predictions (ex. moteurs de validation).

    Parameters
    ----------
    y_true : array-like
        Les vraies RUL.
    y_pred : array-like
        Les RUL predites.
    seuil_surestimation : float, optionnel
        Seuil des surestimations dangereuses (20 cycles par defaut).
    seuil_alerte : float, optionnel
        Seuil d'intervention (30 cycles par defaut, hypothese de travail).

    Returns
    -------
    dict
        rmse, mae, score_nasa, nb_surestimes, alertes_manquees, fausses_alertes, biais, nb_lignes.
    """
    vrais = np.asarray(y_true, dtype=float)
    predits = np.asarray(y_pred, dtype=float)
    resultats = {}
    resultats["rmse"] = rmse(vrais, predits)
    resultats["mae"] = mae(vrais, predits)
    resultats["score_nasa"] = score_nasa(vrais, predits)
    resultats["nb_surestimes"] = nb_surestimes(vrais, predits, seuil=seuil_surestimation)
    resultats["alertes_manquees"] = alertes_manquees(vrais, predits, seuil_alerte=seuil_alerte)
    resultats["fausses_alertes"] = fausses_alertes(vrais, predits, seuil_alerte=seuil_alerte)
    resultats["biais"] = float(np.mean(predits - vrais))  # negatif = sous-estimation moyenne
    resultats["nb_lignes"] = int(len(vrais))
    return resultats


def separer_moteurs(moteurs, fraction_validation=0.2, graine=42):
    """
    Separe une liste de moteurs en deux groupes disjoints (apprentissage / validation), au hasard, graine fixe.

    Parameters
    ----------
    moteurs : array-like
        Les numeros de moteur (chacun une seule fois, ou avec repetitions : seuls les uniques comptent).
    fraction_validation : float, optionnel
        Part des moteurs mis en validation (0,2 par defaut), arrondie a l'entier le plus proche, au moins 1.
    graine : int, optionnel
        Graine du tirage (42 par defaut).

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        Les moteurs d'apprentissage et les moteurs de validation, chacun trie par numero.
    """
    moteurs_uniques = np.unique(np.asarray(moteurs))  # chaque moteur une seule fois, trie
    nb_validation = int(round(fraction_validation * len(moteurs_uniques)))
    nb_validation = max(nb_validation, 1)  # au moins un moteur de validation
    nb_validation = min(nb_validation, len(moteurs_uniques) - 1)  # au moins un moteur d'apprentissage

    generateur = np.random.default_rng(graine)  # tirage reproductible
    ordre = generateur.permutation(len(moteurs_uniques))  # ordre aleatoire des moteurs
    moteurs_validation = np.sort(moteurs_uniques[ordre[:nb_validation]])
    moteurs_apprentissage = np.sort(moteurs_uniques[ordre[nb_validation:]])
    return moteurs_apprentissage, moteurs_validation


def completer_reglages(reglages_entrainement=None):
    """
    Construit les reglages d'apprentissage : REGLAGES_ENTRAINEMENT_DEFAUT, ecrases par ceux de l'appelant.

    Parameters
    ----------
    reglages_entrainement : dict ou None, optionnel
        Reglages a changer (ex. {"alpha": 2.26}).

    Returns
    -------
    dict
        Un nouveau dictionnaire complet (alpha, taux_apprentissage, taille_lot, max_epoques, patience).
    """
    reglages = {}
    for cle in REGLAGES_ENTRAINEMENT_DEFAUT:  # 1) valeurs par defaut
        reglages[cle] = REGLAGES_ENTRAINEMENT_DEFAUT[cle]
    if reglages_entrainement is not None:
        for cle in reglages_entrainement:  # 2) l'appelant ecrase ce qu'il veut changer
            reglages[cle] = reglages_entrainement[cle]
    return reglages


def executer_entrainement(tache):
    """
    Execute UNE tache d'apprentissage complete : creer un reseau neuf, l'entrainer, predire, chronometrer.

    Pourquoi une "tache" (un dictionnaire) ? Pour pouvoir lancer plusieurs apprentissages independants en
    meme temps, chacun dans son propre processus (voir `executer_taches`) : un processus ne recoit que des
    donnees, et cette fonction, definie dans un module importable, sait tout refaire a partir d'elles.
    Le resultat ne depend pas du fait que la tache tourne seule ou en parallele (memes donnees, meme
    graine, meme nombre de coeurs par apprentissage).

    Parameters
    ----------
    tache : dict
        Cles obligatoires :

        - "type_modele" : "lstm" ou "cnn1d" ;
        - "fenetres_entrainement", "y_entrainement" : donnees d'apprentissage (y en cycles) ;
        - "graine" : int.

        Cles optionnelles :

        - "fenetres_validation", "y_validation" : moteurs d'arret precoce (sinon "nb_epoques" obligatoire) ;
        - "nb_epoques" : nombre d'epoques fixe (modele final) ;
        - "hyperparametres" : dict ; "reglages" : dict (voir `completer_reglages`) ;
        - "a_predire" : dict nom -> fenetres a predire apres l'apprentissage (ex. {"validation": ...}) ;
        - "nb_threads" : coeurs utilises par cet apprentissage (1 par defaut) ;
        - "appareil" : "cpu" (defaut) ou "mps" ;
        - "renvoyer_poids" : bool, vrai pour renvoyer les poids appris (pour les sauvegarder ensuite) ;
        - "etiquette" : texte affiche dans la progression.

    Returns
    -------
    dict
        - "historique" : l'historique de `entrainer_modele` ;
        - "predictions" : dict nom -> RUL predites (cycles, >= 0) pour chaque entree de "a_predire" ;
        - "temps_prediction_s" : dict nom -> duree de la prediction en secondes ;
        - "nb_parametres" : int ;
        - "etat_poids" : poids appris (si "renvoyer_poids"), sinon None ;
        - "etiquette" : str.
    """
    nb_threads = tache.get("nb_threads", 1)  # 1 coeur par apprentissage par defaut
    nb_threads_avant = torch.get_num_threads()
    torch.set_num_threads(nb_threads)

    etiquette = tache.get("etiquette", "")
    reglages = completer_reglages(tache.get("reglages"))
    n_entrees = tache["fenetres_entrainement"].shape[2]

    # un reseau NEUF, poids initiaux tires avec la graine de la tache
    modele = creer_modele(tache["type_modele"], n_entrees, graine=tache["graine"],
                          hyperparametres=tache.get("hyperparametres"))

    historique = entrainer_modele(
        modele,
        tache["fenetres_entrainement"],
        tache["y_entrainement"],
        fenetres_validation=tache.get("fenetres_validation"),
        y_validation=tache.get("y_validation"),
        alpha=reglages["alpha"],
        taux_apprentissage=reglages["taux_apprentissage"],
        taille_lot=reglages["taille_lot"],
        max_epoques=reglages["max_epoques"],
        patience=reglages["patience"],
        nb_epoques=tache.get("nb_epoques"),
        graine=tache["graine"],
        appareil=tache.get("appareil", "cpu"),
    )

    # predictions demandees, chacune chronometree
    predictions = {}
    temps_prediction = {}
    a_predire = tache.get("a_predire", {})
    for nom in a_predire:
        debut = time.perf_counter()
        predictions[nom] = predire(modele, a_predire[nom])
        temps_prediction[nom] = time.perf_counter() - debut

    resultat = {}
    resultat["historique"] = historique
    resultat["predictions"] = predictions
    resultat["temps_prediction_s"] = temps_prediction
    resultat["nb_parametres"] = compter_parametres(modele)
    resultat["etat_poids"] = None
    if tache.get("renvoyer_poids", False):
        resultat["etat_poids"] = copier_etat(modele.to("cpu"))
    resultat["etiquette"] = etiquette

    torch.set_num_threads(nb_threads_avant)  # on remet le reglage du processus tel qu'il etait
    return resultat


def executer_taches(taches, nb_processus=1, verbeux=True):
    """
    Execute une liste de taches d'apprentissage, l'une apres l'autre ou plusieurs en meme temps.

    Avec nb_processus > 1, chaque tache tourne dans un processus Python separe (module
    concurrent.futures) : sur une machine a plusieurs coeurs, 5 apprentissages de 1 coeur chacun en
    parallele vont environ 5 fois plus vite qu'un seul apprentissage qui utiliserait 5 coeurs (un petit
    reseau n'occupe pas bien plusieurs coeurs a la fois). Les resultats sont renvoyes dans l'ordre des
    taches et sont les memes qu'en sequentiel (chaque tache est independante et a sa propre graine).

    Parameters
    ----------
    taches : list de dict
        Les taches (voir `executer_entrainement`).
    nb_processus : int, optionnel
        Nombre de processus en parallele (1 = sequentiel, dans le processus courant).
    verbeux : bool, optionnel
        Vrai pour afficher une ligne a la fin de chaque tache.

    Returns
    -------
    list de dict
        Le resultat de chaque tache, dans l'ordre de `taches`.
    """
    resultats = []
    if nb_processus <= 1:  # sequentiel : une tache apres l'autre, dans ce processus
        for tache in taches:
            resultat = executer_entrainement(tache)
            if verbeux:
                afficher_fin_tache(resultat)
            resultats.append(resultat)
        return resultats

    # parallele : un processus Python par tache en cours (au plus nb_processus a la fois)
    resultats_par_position = {}
    with ProcessPoolExecutor(max_workers=nb_processus) as executeur:
        futurs = {}  # futur (calcul en cours) -> position de la tache dans la liste
        for position in range(len(taches)):
            futur = executeur.submit(executer_entrainement, taches[position])
            futurs[futur] = position
        for futur in as_completed(futurs):  # au fur et a mesure que les taches se terminent
            resultat = futur.result()
            if verbeux:
                afficher_fin_tache(resultat)
            resultats_par_position[futurs[futur]] = resultat

    for position in range(len(taches)):  # on remet les resultats dans l'ordre des taches
        resultats.append(resultats_par_position[position])
    return resultats


def afficher_fin_tache(resultat):
    """
    Affiche une ligne de progression a la fin d'une tache d'apprentissage.

    Parameters
    ----------
    resultat : dict
        Le resultat de `executer_entrainement`.
    """
    historique = resultat["historique"]
    texte = f"  termine : {resultat['etiquette']} | epoque gardee {historique['meilleure_epoque']}"
    texte = texte + f"/{historique['nb_epoques_effectuees']} | {historique['duree_s']:.0f} s"
    if len(historique["rmse_validation"]) > 0:
        position_meilleure = historique["meilleure_epoque"] - 1  # les listes commencent a 0
        texte = texte + f" | RMSE arret precoce {historique['rmse_validation'][position_meilleure]:.2f}"
    print(texte, flush=True)


def preparer_taches_validation_croisee(
    fenetres,
    y,
    groups,
    plis,
    type_modele,
    hyperparametres=None,
    reglages_entrainement=None,
    fraction_arret_precoce=0.2,
    graine=42,
    appareil="cpu",
    nb_threads=1,
):
    """
    Prepare les taches d'apprentissage d'une validation croisee (une par pli), sans les executer.

    Etape 1 de `evaluer_en_validation_croisee_sequentielle` (voir sa documentation pour le protocole). Separee
    pour que le script puisse regrouper les taches de plusieurs reseaux ou sous-jeux dans un seul lot de
    processus paralleles, puis assembler chaque validation croisee avec `assembler_validation_croisee`.

    Parameters
    ----------
    fenetres, y, groups, plis, type_modele : voir `evaluer_en_validation_croisee_sequentielle`
        Donnees, plis et type de reseau.
    hyperparametres, reglages_entrainement, fraction_arret_precoce, graine, appareil, nb_threads : optionnels
        Memes significations et memes valeurs par defaut que dans `evaluer_en_validation_croisee_sequentielle`.

    Returns
    -------
    (list de dict, list de list de int)
        Les taches (une par pli, dans l'ordre des plis) et les moteurs d'arret precoce de chaque pli.
    """
    reglages = completer_reglages(reglages_entrainement)
    vrais = np.asarray(y, dtype=float)
    groupes = np.asarray(groups)

    # une tache d'apprentissage par pli (executee ensuite, en sequentiel ou en parallele)
    taches = []
    moteurs_arret_precoce_par_pli = []
    numero_pli = 0
    for idx_train, idx_val in plis:  # les memes plis que les arbres
        numero_pli = numero_pli + 1

        # garde-fou 1 : aucun moteur des deux cotes du pli
        moteurs_communs = set(groupes[idx_train]) & set(groupes[idx_val])
        assert len(moteurs_communs) == 0, f"moteur(s) des deux cotes d'un pli : {sorted(moteurs_communs)}"

        # separation des moteurs d'entrainement du pli : apprentissage / arret precoce
        moteurs_entrainement_pli = np.unique(groupes[idx_train])
        moteurs_apprentissage, moteurs_arret = separer_moteurs(
            moteurs_entrainement_pli, fraction_validation=fraction_arret_precoce, graine=graine
        )
        masque_apprentissage = np.isin(groupes[idx_train], moteurs_apprentissage)
        idx_apprentissage = idx_train[masque_apprentissage]  # lignes des moteurs d'apprentissage
        idx_arret = idx_train[~masque_apprentissage]  # lignes des moteurs d'arret precoce

        # garde-fou 2 : les moteurs d'arret precoce ne sont jamais des moteurs de validation du pli
        assert len(set(moteurs_arret) & set(groupes[idx_val])) == 0, "moteur d'arret precoce en validation"

        liste_moteurs_arret = []
        for moteur_id in moteurs_arret:  # en types Python simples (ecriture JSON possible)
            liste_moteurs_arret.append(int(moteur_id))
        moteurs_arret_precoce_par_pli.append(liste_moteurs_arret)

        tache = {}
        tache["type_modele"] = type_modele
        tache["hyperparametres"] = hyperparametres
        tache["reglages"] = reglages
        tache["graine"] = graine
        tache["fenetres_entrainement"] = fenetres[idx_apprentissage]
        tache["y_entrainement"] = vrais[idx_apprentissage]
        tache["fenetres_validation"] = fenetres[idx_arret]
        tache["y_validation"] = vrais[idx_arret]
        tache["a_predire"] = {"validation_pli": fenetres[idx_val]}
        tache["nb_threads"] = nb_threads
        tache["appareil"] = appareil
        tache["etiquette"] = (f"{type_modele} pli {numero_pli}/{len(plis)} ({len(moteurs_apprentissage)} moteurs "
                              f"d'apprentissage, {len(moteurs_arret)} d'arret precoce, "
                              f"{len(np.unique(groupes[idx_val]))} de validation)")
        taches.append(tache)

    return taches, moteurs_arret_precoce_par_pli


def assembler_validation_croisee(y, plis, resultats_taches, moteurs_arret_precoce_par_pli, seuil_surestimation=20,
                                 seuil_alerte=30, etiquette="", verbeux=True):
    """
    Assemble les resultats des taches d'une validation croisee : predictions hors-pli et metriques.

    Etape 3 de `evaluer_en_validation_croisee_sequentielle` (voir sa documentation pour les cles renvoyees).

    Parameters
    ----------
    y : array-like
        La cible de chaque ligne, en cycles.
    plis : list de (numpy.ndarray, numpy.ndarray)
        Les plis (les memes que ceux qui ont servi a preparer les taches).
    resultats_taches : list de dict
        Le resultat de `executer_entrainement` pour chaque pli, dans l'ordre des plis.
    moteurs_arret_precoce_par_pli : list de list de int
        Les moteurs d'arret precoce de chaque pli.
    seuil_surestimation : float, optionnel
        Seuil des surestimations dangereuses (20 par defaut).
    seuil_alerte : float, optionnel
        Seuil d'intervention (30 par defaut, hypothese de travail).
    etiquette : str, optionnel
        Texte affiche dans la progression.
    verbeux : bool, optionnel
        Vrai pour afficher le RMSE de chaque pli.

    Returns
    -------
    dict
        Les resultats de la validation croisee (memes cles que `evaluer_en_validation_croisee_sequentielle`).
    """
    vrais = np.asarray(y, dtype=float)

    # predictions hors-pli et metriques pli par pli
    predictions_oof = np.full(len(vrais), np.nan)  # NaN = ligne pas encore predite
    rmse_par_pli = []
    mae_par_pli = []
    nasa_par_pli = []
    surestimes_par_pli = []
    temps_entrainement_par_pli = []
    temps_prediction_par_pli = []
    historiques = []
    meilleures_epoques = []
    nb_parametres = 0

    for position in range(len(plis)):
        idx_val = plis[position][1]
        resultat = resultats_taches[position]
        y_pred = resultat["predictions"]["validation_pli"]
        predictions_oof[idx_val] = y_pred  # chaque prediction a la place de sa ligne
        y_val = vrais[idx_val]

        rmse_par_pli.append(rmse(y_val, y_pred))
        mae_par_pli.append(mae(y_val, y_pred))
        nasa_par_pli.append(score_nasa(y_val, y_pred))
        surestimes_par_pli.append(nb_surestimes(y_val, y_pred, seuil=seuil_surestimation))
        temps_entrainement_par_pli.append(resultat["historique"]["duree_s"])
        temps_prediction_par_pli.append(resultat["temps_prediction_s"]["validation_pli"])
        historiques.append(resultat["historique"])
        meilleures_epoques.append(resultat["historique"]["meilleure_epoque"])
        nb_parametres = resultat["nb_parametres"]

        if verbeux:
            print(f"[{etiquette}] pli {position + 1} : RMSE validation du pli = {rmse_par_pli[-1]:.2f}", flush=True)

    nb_non_predites = int(np.sum(np.isnan(predictions_oof)))
    assert nb_non_predites == 0, f"{nb_non_predites} lignes sans prediction hors-pli : les plis ne couvrent pas tout"

    resultats = {}
    resultats["rmse_par_pli"] = rmse_par_pli
    resultats["mae_par_pli"] = mae_par_pli
    resultats["nasa_par_pli"] = nasa_par_pli
    resultats["surestimes_par_pli"] = surestimes_par_pli

    # moyennes et ecarts-types entre plis (np.std avec ddof=0, comme src/evaluation.py)
    resultats["rmse_moyen"] = float(np.mean(rmse_par_pli))
    resultats["rmse_ecart_type"] = float(np.std(rmse_par_pli))
    resultats["mae_moyen"] = float(np.mean(mae_par_pli))
    resultats["mae_ecart_type"] = float(np.std(mae_par_pli))
    resultats["nasa_moyen"] = float(np.mean(nasa_par_pli))
    resultats["nasa_ecart_type"] = float(np.std(nasa_par_pli))
    resultats["surestimes_moyen"] = float(np.mean(surestimes_par_pli))
    resultats["surestimes_ecart_type"] = float(np.std(surestimes_par_pli))

    resultats["predictions_oof"] = predictions_oof
    resultats["rmse_oof"] = rmse(vrais, predictions_oof)
    resultats["mae_oof"] = mae(vrais, predictions_oof)
    resultats["nasa_oof"] = score_nasa(vrais, predictions_oof)
    resultats["surestimes_oof"] = nb_surestimes(vrais, predictions_oof, seuil=seuil_surestimation)
    resultats["alertes_manquees_oof"] = alertes_manquees(vrais, predictions_oof, seuil_alerte=seuil_alerte)
    resultats["fausses_alertes_oof"] = fausses_alertes(vrais, predictions_oof, seuil_alerte=seuil_alerte)
    resultats["biais_oof"] = float(np.mean(predictions_oof - vrais))

    resultats["temps_entrainement_par_pli"] = temps_entrainement_par_pli
    resultats["temps_prediction_par_pli"] = temps_prediction_par_pli
    resultats["temps_entrainement_total"] = float(np.sum(temps_entrainement_par_pli))
    resultats["temps_prediction_total"] = float(np.sum(temps_prediction_par_pli))

    resultats["historiques"] = historiques
    resultats["meilleures_epoques"] = meilleures_epoques
    resultats["moteurs_arret_precoce_par_pli"] = moteurs_arret_precoce_par_pli
    resultats["nb_parametres"] = nb_parametres
    return resultats


def evaluer_en_validation_croisee_sequentielle(
    fenetres,
    y,
    groups,
    plis,
    type_modele,
    hyperparametres=None,
    reglages_entrainement=None,
    fraction_arret_precoce=0.2,
    graine=42,
    appareil="cpu",
    seuil_surestimation=20,
    seuil_alerte=30,
    nb_processus=1,
    nb_threads=1,
    verbeux=True,
):
    """
    Evalue un type de reseau en validation croisee par moteur, avec les MEMES plis que les arbres.

    Pour chaque pli :
    1. les moteurs d'entrainement du pli sont separes (graine fixe) en moteurs d'APPRENTISSAGE (80 %) et
       moteurs d'ARRET PRECOCE (20 %) ; les moteurs de validation du pli ne servent JAMAIS a l'arret
       precoce (sinon ils influenceraient le modele qui les predit, et le score serait optimiste) ;
    2. un reseau NEUF est entraine sur les moteurs d'apprentissage, l'arret precoce surveille les moteurs
       d'arret precoce ;
    3. il predit les moteurs de validation du pli : ce sont les predictions hors-pli (OOF), rangees a la
       place de leurs lignes (une fenetre par ligne : alignement exact avec les predictions des arbres).

    Les cles du resultat sont celles de `src.evaluation.evaluer_en_validation_croisee` (memes metriques,
    memes noms), plus des informations propres aux reseaux (historiques d'apprentissage, meilleures
    epoques). Difference a connaitre : les predictions des reseaux sont clippees a 0 (`predire`), celles
    des arbres en validation croisee ne le sont pas.

    Parameters
    ----------
    fenetres : numpy.ndarray
        Une fenetre par ligne du train, forme (nb_lignes, longueur, n_entrees) (voir `construire_fenetres`).
    y : array-like
        La cible (RUL plafonnee) de chaque ligne, en cycles.
    groups : array-like
        Le numero de moteur de chaque ligne.
    plis : list de (numpy.ndarray, numpy.ndarray)
        Les plis de `src.evaluation.calculer_plis` (les memes que pour les arbres).
    type_modele : str
        "lstm" ou "cnn1d".
    hyperparametres : dict ou None, optionnel
        Hyperparametres du reseau (valeurs par defaut sinon).
    reglages_entrainement : dict ou None, optionnel
        Reglages de `entrainer_modele` (alpha, taux_apprentissage, taille_lot, max_epoques, patience) ;
        REGLAGES_ENTRAINEMENT_DEFAUT sinon.
    fraction_arret_precoce : float, optionnel
        Part des moteurs d'entrainement du pli reservee a l'arret precoce (0,2 par defaut).
    graine : int, optionnel
        Graine des poids initiaux, du melange des lots et de la separation des moteurs (42 par defaut).
    appareil : str, optionnel
        "cpu" (defaut) ou "mps".
    seuil_surestimation : float, optionnel
        Seuil des surestimations dangereuses (20 par defaut).
    seuil_alerte : float, optionnel
        Seuil d'intervention (30 par defaut, hypothese de travail).
    nb_processus : int, optionnel
        Nombre de plis entraines en meme temps (1 = l'un apres l'autre ; resultats identiques).
    nb_threads : int, optionnel
        Coeurs utilises par chaque apprentissage (1 par defaut).
    verbeux : bool, optionnel
        Vrai pour afficher la progression pli par pli.

    Returns
    -------
    dict
        Memes cles que `evaluer_en_validation_croisee` ("rmse_par_pli", "rmse_moyen", "rmse_ecart_type",
        "mae_moyen", "nasa_moyen", "surestimes_oof", "alertes_manquees_oof", "fausses_alertes_oof",
        "biais_oof", "predictions_oof", "temps_entrainement_total", "temps_prediction_total", ...), plus :

        - "historiques" : liste, l'historique de `entrainer_modele` de chaque pli ;
        - "meilleures_epoques" : liste, la meilleure epoque de chaque pli ;
        - "moteurs_arret_precoce_par_pli" : liste de listes, les moteurs d'arret precoce de chaque pli ;
        - "nb_parametres" : int, le nombre de poids du reseau.
    """
    taches, moteurs_arret_precoce_par_pli = preparer_taches_validation_croisee(
        fenetres, y, groups, plis, type_modele, hyperparametres=hyperparametres,
        reglages_entrainement=reglages_entrainement, fraction_arret_precoce=fraction_arret_precoce, graine=graine,
        appareil=appareil, nb_threads=nb_threads,
    )
    resultats_taches = executer_taches(taches, nb_processus=nb_processus, verbeux=verbeux)
    return assembler_validation_croisee(y, plis, resultats_taches, moteurs_arret_precoce_par_pli,
                                        seuil_surestimation=seuil_surestimation, seuil_alerte=seuil_alerte,
                                        etiquette=type_modele, verbeux=verbeux)


# ---------------------------------------------------------------------------------------------------
# Sauvegarde et chargement
# ---------------------------------------------------------------------------------------------------

def chemin_modele(sous_jeu, type_modele, dossier=None):
    """
    Construit le chemin du fichier de poids d'un reseau : models/<SOUS_JEU>__<type_modele>.pt.

    Parameters
    ----------
    sous_jeu : str
        Le sous-jeu (ex. "FD001").
    type_modele : str
        "lstm" ou "cnn1d".
    dossier : str ou None, optionnel
        Dossier des poids ; None pour DOSSIER_MODELES (<racine du depot>/models).

    Returns
    -------
    str
        Le chemin du fichier .pt.
    """
    if type_modele not in TYPES_MODELES:
        raise ValueError(f"Type de modele inconnu : {type_modele!r} (choix possibles : {TYPES_MODELES}).")
    if dossier is None:
        dossier = DOSSIER_MODELES
    return os.path.join(str(dossier), f"{sous_jeu}__{type_modele}.pt")


def sauvegarder_modele(modele, chemin, type_modele, n_entrees, hyperparametres=None, informations=None):
    """
    Sauvegarde les poids d'un reseau ET tout ce qu'il faut pour le reconstruire (type, entrees, reglages).

    Des poids seuls ne suffisent pas : pour les recharger, il faut recreer exactement le meme reseau (meme
    type, meme nombre d'entrees, memes tailles de couches). On les ecrit donc dans le meme fichier, avec
    des informations libres (colonnes d'entree, longueur de fenetre, nombre d'epoques, metriques...).

    Parameters
    ----------
    modele : torch.nn.Module
        Le reseau entraine.
    chemin : str
        Le fichier a ecrire (ex. models/FD001__lstm.pt) ; le dossier est cree si besoin.
    type_modele : str
        "lstm" ou "cnn1d".
    n_entrees : int
        Nombre de colonnes d'entree par cycle.
    hyperparametres : dict ou None, optionnel
        Hyperparametres du reseau (valeurs par defaut si None).
    informations : dict ou None, optionnel
        Informations libres (types simples : textes, nombres, listes, dictionnaires).

    Returns
    -------
    str
        Le chemin ecrit.
    """
    parametres = hyperparametres_par_defaut(type_modele)
    if hyperparametres is not None:
        for cle in hyperparametres:
            parametres[cle] = hyperparametres[cle]
    infos = {}
    if informations is not None:
        infos = informations

    contenu = {}
    contenu["etat_poids"] = copier_etat(modele.to("cpu"))  # poids ramenes sur le processeur
    contenu["type_modele"] = type_modele
    contenu["n_entrees"] = int(n_entrees)
    # types Python simples uniquement : le chargement "weights_only" de PyTorch refuse les objets numpy
    contenu["hyperparametres"] = convertir_en_types_python(parametres)
    contenu["informations"] = convertir_en_types_python(infos)

    dossier = os.path.dirname(os.path.abspath(str(chemin)))
    os.makedirs(dossier, exist_ok=True)  # cree models/ si besoin
    torch.save(contenu, str(chemin))
    return str(chemin)


def charger_modele(chemin, appareil="cpu"):
    """
    Recharge un reseau sauvegarde par `sauvegarder_modele`, pret a predire.

    Parameters
    ----------
    chemin : str
        Le fichier .pt.
    appareil : str, optionnel
        "cpu" (defaut) ou "mps".

    Returns
    -------
    (torch.nn.Module, dict)
        Le reseau (en mode prediction) et le contenu du fichier (type_modele, n_entrees, hyperparametres,
        informations).
    """
    appareil_torch = choisir_appareil(appareil)
    # weights_only=True : on ne relit que des tableaux de poids et des types simples (pas de code arbitraire)
    contenu = torch.load(str(chemin), map_location=appareil_torch, weights_only=True)
    modele = creer_modele(contenu["type_modele"], contenu["n_entrees"], graine=0,
                          hyperparametres=contenu["hyperparametres"])
    modele.load_state_dict(contenu["etat_poids"])  # on remplace les poids tires au hasard par les poids appris
    modele.to(appareil_torch)
    modele.eval()
    return modele, contenu
