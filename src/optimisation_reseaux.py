"""
Optimisation des reseaux sequentiels (Phase 4) : architectures supplementaires, entrees candidates et
validation croisee d'UNE configuration, pli par pli, utilisable par Optuna. Tout est parametre par
sous-jeu (FD001 par defaut ; FD002 a FD004 prepares par src/sous_jeux.py, comme au notebook 03b).

Pourquoi ce module ? Le notebook 03b a entraine le LSTM et le CNN 1D avec des reglages fixes a l'avance
(2 x 64 unites, pas 1e-3, ...), jamais optimises. Ce module permet de chercher de meilleurs reglages avec
la MEME methode que les arbres (notebooks 04d, 04e, 04f) :

- les memes plis par moteur (GroupKFold, shuffle, graine 42) et la meme separation des moteurs d'arret
  precoce que `src.deep_learning.preparer_taches_validation_croisee` : la configuration de reference
  (LSTM 2 x 64, fenetre 30, ecartinit_20) redonne donc exactement le RMSE de validation du notebook 03b ;
- une configuration est un petit dictionnaire (type de reseau, hyperparametres, reglages d'apprentissage,
  entrees) : Optuna en tire une a chaque essai, et la meme fonction l'evalue ;
- l'evaluation se fait pli par pli et appelle une fonction apres chaque pli : Optuna peut ainsi abandonner
  ("elaguer") un essai clairement mauvais des les premiers plis, sans attendre les 5.

Le jeu de test n'est utilise que par `evaluer_sur_test_dernier_cycle`, appele une seule fois a la fin, par
le script, pour la configuration retenue.

Contenu, dans l'ordre :
1. preparation des tableaux (une fois par processus et par valeur de k) et des fenetres (une fois par jeu
   d'entrees et longueur), avec la pente glissante comme famille d'entrees candidate (notebook 04e, F2) ;
2. deux architectures tirees de la litterature : CNN puis LSTM, et LSTM avec attention ;
3. la validation croisee d'une configuration, pli par pli, et l'apprentissage d'un modele final ;
4. les espaces de recherche Optuna de chaque type de reseau.
"""

import copy
import os
import time

import numpy as np

# Ordre des imports voulu (voir src/deep_learning.py) : les modules du projet, qui chargent xgboost et
# lightgbm, AVANT torch, sinon conflit entre bibliotheques OpenMP.
from src.config_modele import convertir_en_types_python
from src.deep_learning import (
    ajouter_time_cycles_echelle,
    appliquer_min_max,
    apprendre_min_max,
    calculer_metriques,
    construire_fenetres_paresseuses,
    creer_modele,
    entrainer_modele,
    fixer_graines,
    lister_colonnes_entree,
    predire,
    separer_moteurs,
)
from src.evaluation import calculer_plis, rmse, score_nasa

import torch  # noqa: E402  (apres les modules du projet)
from torch import nn  # noqa: E402

# ---------------------------------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------------------------------

PLAFOND_RUL = 125  # plafond de la RUL (Phase 2)
N_PLIS = 5  # nombre de plis de la validation croisee
GRAINE_PLIS = 42  # plis des arbres (Phase 3, notebooks 04 et 05)
GRAINE = 42  # graine des poids initiaux, de l'ordre des lots et de la separation d'arret precoce
FRACTION_ARRET_PRECOCE = 0.2  # 20 % des moteurs d'entrainement du pli servent a l'arret precoce

# Les types de reseaux connus de ce module : les deux du notebook 03b, plus deux de la litterature
TYPES_RESEAUX = ["lstm", "cnn1d", "cnn_lstm", "lstm_attention"]

# Configuration de reference : le LSTM retenu au notebook 03b (configuration "A"), recopie a l'identique.
# Sert de point de comparaison a toute configuration trouvee par Optuna.
CONFIG_LSTM_REFERENCE = {
    "type": "lstm",
    "hyperparametres": {"taille_cachee": 64, "nb_couches": 2, "dropout": 0.2},
    "reglages": {"taux_apprentissage": 1e-3, "taille_lot": 256, "max_epoques": 80, "patience": 10, "alpha": 1.0},
    "entrees": {"longueur": 30, "ecart_initial": 20, "capteurs": True, "pente": None},
}

# Meme chose pour le CNN 1D retenu au notebook 03b
CONFIG_CNN_REFERENCE = {
    "type": "cnn1d",
    "hyperparametres": {"canaux": [32, 64, 64], "noyaux": [5, 5, 3], "dropout": 0.2},
    "reglages": {"taux_apprentissage": 1e-3, "taille_lot": 256, "max_epoques": 80, "patience": 10, "alpha": 1.0},
    "entrees": {"longueur": 30, "ecart_initial": 20, "capteurs": True, "pente": None},
}

# Mode "essai rapide" (variable d'environnement AERO_ESSAI_RAPIDE=1) : 2 epoques au plus par reseau, pour
# verifier en quelques minutes qu'une etape du script va jusqu'au bout. Ses chiffres n'ont AUCUN sens ;
# scripts/optimiser_reseaux.py ecrit alors ses resultats dans un dossier a part, jamais dans sorties/resultats.
ESSAI_RAPIDE = os.environ.get("AERO_ESSAI_RAPIDE") == "1"

# Sous-jeux connus. FD001 : pipeline de src/features.py (Phases 1 a 3) ; les autres : src/sous_jeux.py
# (normalisation par regime pour FD002 et FD004), exactement comme preparer_donnees du notebook 03b.
SOUS_JEUX = ["FD001", "FD002", "FD003", "FD004"]
SOUS_JEU_DEFAUT = "FD001"

# Valeurs candidates de k (ecart a l'etat initial). Seules celles STRICTEMENT plus petites que le plus court
# moteur test du sous-jeu sont tirees par Optuna (voir `valeurs_ecart_initial`) : FD001 (31 cycles) garde
# 20, 25 et 30 ; FD002 (21 cycles) seulement 20.
VALEURS_ECART_INITIAL = [20, 25, 30]

# Caches du processus : la preparation des donnees (~15 s) n'est faite qu'une fois par processus pour un
# meme (sous-jeu, k, pente). Les fenetres sont paresseuses (src.deep_learning.FenetresParesseuses :
# construites lot par lot, environ 20 Mo pour FD002 avec 50 cycles, au lieu d'environ 0,5 Go pour le
# tenseur 3D complet) ; on ne garde quand meme que le DERNIER jeu d'entrees demande. Constate le
# 2026-11-06 avec les tenseurs complets : 15 processus sur FD002 saturaient les 31 Go de la machine.
# Chaque processus a ses propres caches (rien n'est partage entre processus).
_CACHE_TABLEAUX = {}
_CACHE_FENETRES = {}


# ---------------------------------------------------------------------------------------------------
# 1. Preparation des donnees
# ---------------------------------------------------------------------------------------------------

def calculer_pente_glissante(valeurs, k):
    """
    Pente de la droite des moindres carres sur les min(t, k) derniers cycles d'UN capteur d'UN moteur.

    Meme definition que la famille F2 ("pente_k") du notebook 04e : au cycle t, on ajuste une droite
    sur les points (rang du cycle, valeur) des cycles t-k+1 a t, et on garde sa pente. Causale : seuls
    les cycles deja observes interviennent. Au tout premier cycle (un seul point), la pente vaut 0.

    Formule fermee (pas de boucle sur les fenetres) : avec n points (x_i, v_i),
        pente = (n * somme(x v) - somme(x) * somme(v)) / (n * somme(x^2) - somme(x)^2).
    Les sommes glissantes sont obtenues par sommes cumulees : somme sur la fenetre = cumul(t) - cumul(t-k).

    Exemple, k = 3 et v = [1, 2, 4, 7] : pentes [0, 1, 1.5, 2.5].

    Parameters
    ----------
    valeurs : numpy.ndarray
        Le signal lisse d'un capteur pour un seul moteur, dans l'ordre des cycles.
    k : int
        Nombre maximal de cycles dans la fenetre (k >= 2).

    Returns
    -------
    numpy.ndarray
        Une pente par cycle, meme longueur que `valeurs`.
    """
    v = np.asarray(valeurs, dtype=float)
    nb = len(v)
    x = np.arange(nb, dtype=float)  # rang du cycle (0, 1, 2, ...) : l'abscisse de la droite

    # sommes cumulees, precedees d'un 0 pour pouvoir soustraire "cumul(t - k)" meme au debut
    cumul_x = np.concatenate([[0.0], np.cumsum(x)])
    cumul_v = np.concatenate([[0.0], np.cumsum(v)])
    cumul_xv = np.concatenate([[0.0], np.cumsum(x * v)])
    cumul_xx = np.concatenate([[0.0], np.cumsum(x * x)])

    fin = np.arange(1, nb + 1)  # la fenetre du cycle t se termine apres la position t (exclue dans le cumul)
    debut = np.maximum(fin - k, 0)  # elle commence k positions avant, ou au premier cycle
    n = (fin - debut).astype(float)  # nombre de points dans chaque fenetre : min(t, k)

    somme_x = cumul_x[fin] - cumul_x[debut]
    somme_v = cumul_v[fin] - cumul_v[debut]
    somme_xv = cumul_xv[fin] - cumul_xv[debut]
    somme_xx = cumul_xx[fin] - cumul_xx[debut]

    numerateur = n * somme_xv - somme_x * somme_v
    denominateur = n * somme_xx - somme_x * somme_x  # nul seulement quand n = 1
    pentes = np.zeros(nb)
    masque = denominateur > 0  # au premier cycle (un seul point), la pente reste a 0
    pentes[masque] = numerateur[masque] / denominateur[masque]
    return pentes


def ajouter_pente_glissante(df, capteurs, k):
    """
    Ajoute, pour chaque capteur, la colonne "<capteur>_pente_<k>", calculee moteur par moteur.

    Le tableau doit etre trie par moteur puis par cycle (c'est le cas apres src/features.py).

    Parameters
    ----------
    df : pandas.DataFrame
        Le tableau avec les capteurs deja lisses (n'est pas modifie, une copie est renvoyee).
    capteurs : list de str
        Les capteurs concernes.
    k : int
        Nombre de cycles de la fenetre de la pente.

    Returns
    -------
    (pandas.DataFrame, list de str)
        Le tableau enrichi et les noms des nouvelles colonnes.
    """
    df_pente = df.copy()  # copie independante
    positions_par_moteur = df_pente.groupby("unit_number", sort=False).indices  # lignes de chaque moteur
    nouvelles_colonnes = []
    for capteur in capteurs:  # un capteur a la fois
        valeurs = df_pente[capteur].to_numpy(dtype=float)
        pentes = np.zeros(len(df_pente))
        for moteur_id in positions_par_moteur:  # un moteur a la fois : jamais de melange entre moteurs
            lignes = positions_par_moteur[moteur_id]
            pentes[lignes] = calculer_pente_glissante(valeurs[lignes], k)
        nom = capteur + "_pente_" + str(k)
        df_pente[nom] = pentes
        nouvelles_colonnes.append(nom)
    return df_pente, nouvelles_colonnes


def verifier_sous_jeu(sous_jeu):
    """Leve une erreur claire si le sous-jeu n'est pas un des SOUS_JEUX."""
    if sous_jeu not in SOUS_JEUX:
        raise ValueError(f"Sous-jeu inconnu : {sous_jeu!r} (choix possibles : {SOUS_JEUX}).")


def preparer_tableaux(ecart_initial, pente=None, sous_jeu=SOUS_JEU_DEFAUT):
    """
    Prepare (une fois par processus) le train et le test d'un sous-jeu pour une valeur de k donnee.

    Reprend a l'identique `preparer_donnees` de scripts/entrainer_deep_learning.py : src/features.py pour
    FD001, src/sous_jeux.py pour FD002 a FD004 (normalisation, par regime si besoin, lissage, RUL plafonnee,
    ecart a l'etat initial), puis time_cycles / 100, puis min-max de l'ecart initial appris sur le TRAIN
    seulement. Si `pente` est donne, ajoute aussi la pente glissante de chaque capteur, mise a l'echelle de
    la meme facon (bornes du train).

    Parameters
    ----------
    ecart_initial : int ou None
        k de l'ecart a l'etat initial (20 au notebook 03b), None pour ne pas le calculer.
    pente : int ou None, optionnel
        Fenetre de la pente glissante, None (defaut) pour ne pas la calculer.
    sous_jeu : str, optionnel
        "FD001" (defaut) a "FD004".

    Returns
    -------
    dict
        "train", "test" (DataFrames), "capteurs", "colonnes_ecart" et "colonnes_pente" (mises a l'echelle),
        "y" (RUL plafonnee du train), "groups", "X" (features des arbres, pour calculer_plis),
        "y_test_vraie", "moteurs_test".
    """
    verifier_sous_jeu(sous_jeu)
    cle = (sous_jeu, ecart_initial, pente)
    if cle in _CACHE_TABLEAUX:  # deja prepare dans ce processus
        return _CACHE_TABLEAUX[cle]

    # imports tardifs : ne chargent les donnees que si besoin
    if sous_jeu == "FD001":
        from src.features import preparer_test, preparer_train
        contexte_train = preparer_train(fenetre=5, plafond=PLAFOND_RUL, ecart_initial=ecart_initial)
        contexte_test = preparer_test(contexte_train)  # memes calculs, moteur test par moteur test
        train_brut = contexte_train["train"]
        test_brut = contexte_test["test"]
        capteurs = contexte_train["sensor_cols_restants"]
        colonnes_ecart_brutes = contexte_train["colonnes_ecart_initial"]
        X = contexte_train["X"]
        y_test_vraie = contexte_test["y_test_vraie"]
        moteurs_test = contexte_test["moteurs_test"]
    else:
        from src.sous_jeux import preparer_sous_jeu
        contexte = preparer_sous_jeu(sous_jeu, fenetre=5, plafond=PLAFOND_RUL, ecart_initial=ecart_initial)
        train_brut = contexte["train"]
        test_brut = contexte["test"]
        capteurs = contexte["capteurs"]
        colonnes_ecart_brutes = contexte["colonnes_ecart_initial"]
        X = contexte["X"]
        y_test_vraie = contexte["y_test_vraie"]
        moteurs_test = contexte["moteurs_test"]
    train = ajouter_time_cycles_echelle(train_brut)
    test = ajouter_time_cycles_echelle(test_brut)

    # ecart a l'etat initial : bornes min-max apprises sur le train, appliquees telles quelles au test
    colonnes_ecart = []
    if ecart_initial is not None:
        bornes = apprendre_min_max(train, colonnes_ecart_brutes)
        train, colonnes_ecart = appliquer_min_max(train, colonnes_ecart_brutes, bornes)
        test, _ = appliquer_min_max(test, colonnes_ecart_brutes, bornes)

    # pente glissante (famille F2 du notebook 04e) : meme traitement
    colonnes_pente = []
    if pente is not None:
        train, colonnes_pente_brutes = ajouter_pente_glissante(train, capteurs, pente)
        test, _ = ajouter_pente_glissante(test, capteurs, pente)
        bornes_pente = apprendre_min_max(train, colonnes_pente_brutes)
        train, colonnes_pente = appliquer_min_max(train, colonnes_pente_brutes, bornes_pente)
        test, _ = appliquer_min_max(test, colonnes_pente_brutes, bornes_pente)

    tableaux = {}
    tableaux["train"] = train
    tableaux["test"] = test
    tableaux["capteurs"] = capteurs
    tableaux["colonnes_ecart"] = colonnes_ecart
    tableaux["colonnes_pente"] = colonnes_pente
    tableaux["y"] = train["RUL_cappee"].to_numpy(dtype=float)
    tableaux["groups"] = train["unit_number"].to_numpy()
    tableaux["X"] = X
    tableaux["y_test_vraie"] = np.asarray(y_test_vraie, dtype=float)
    tableaux["moteurs_test"] = np.asarray(moteurs_test)
    _CACHE_TABLEAUX[cle] = tableaux
    return tableaux


def colonnes_des_entrees(tableaux, entrees):
    """
    Liste les colonnes d'entree du reseau pour une description d'entrees donnee.

    Ordre : capteurs lisses (si "capteurs"), ecarts a l'etat initial, pentes, puis time_cycles / 100
    (toujours, decision du notebook 03b). Avec les reglages de reference, c'est exactement l'ordre du 03b.

    Parameters
    ----------
    tableaux : dict
        Le resultat de `preparer_tableaux`.
    entrees : dict
        "capteurs" (bool), "ecart_initial" (int ou None), "pente" (int ou None), "longueur" (int).

    Returns
    -------
    list de str
        Les colonnes, dans l'ordre ou le reseau les recoit.
    """
    colonnes_capteurs = []
    if entrees.get("capteurs", True):
        for colonne in tableaux["capteurs"]:
            colonnes_capteurs.append(colonne)
    supplementaires = []
    for colonne in tableaux["colonnes_ecart"]:
        supplementaires.append(colonne)
    for colonne in tableaux["colonnes_pente"]:
        supplementaires.append(colonne)
    return lister_colonnes_entree(colonnes_capteurs, True, colonnes_supplementaires=supplementaires)


def preparer_fenetres(entrees, sous_jeu=SOUS_JEU_DEFAUT):
    """
    Construit les fenetres du train et du test pour une description d'entrees (seul le dernier jeu
    demande reste en cache, voir _CACHE_FENETRES).

    Parameters
    ----------
    entrees : dict
        "longueur", "ecart_initial", "capteurs", "pente" (voir `colonnes_des_entrees`).
    sous_jeu : str, optionnel
        "FD001" (defaut) a "FD004".

    Returns
    -------
    dict
        "fenetres", "y", "groups", "X", "fenetres_test", "test" (DataFrame), "y_test_vraie",
        "moteurs_test", "colonnes_entree".
    """
    cle = (sous_jeu, entrees["longueur"], entrees.get("ecart_initial"), entrees.get("capteurs", True), entrees.get("pente"))
    if cle in _CACHE_FENETRES:
        return _CACHE_FENETRES[cle]
    _CACHE_FENETRES.clear()  # un seul jeu de fenetres en memoire par processus

    tableaux = preparer_tableaux(entrees.get("ecart_initial"), entrees.get("pente"), sous_jeu=sous_jeu)
    colonnes = colonnes_des_entrees(tableaux, entrees)
    # memes fenetres que construire_fenetres (03b), construites a la demande (voir _CACHE_FENETRES)
    fenetres, _, _, _ = construire_fenetres_paresseuses(tableaux["train"], colonnes, longueur=entrees["longueur"])
    fenetres_test, _, _, _ = construire_fenetres_paresseuses(tableaux["test"], colonnes, longueur=entrees["longueur"])

    donnees = {}
    donnees["fenetres"] = fenetres
    donnees["y"] = tableaux["y"]
    donnees["groups"] = tableaux["groups"]
    donnees["X"] = tableaux["X"]
    donnees["fenetres_test"] = fenetres_test
    donnees["test"] = tableaux["test"]
    donnees["y_test_vraie"] = tableaux["y_test_vraie"]
    donnees["moteurs_test"] = tableaux["moteurs_test"]
    donnees["colonnes_entree"] = colonnes
    _CACHE_FENETRES[cle] = donnees
    return donnees


# ---------------------------------------------------------------------------------------------------
# 2. Architectures supplementaires (litterature)
# ---------------------------------------------------------------------------------------------------

class ModeleCNNLSTM(nn.Module):
    """
    Reseau hybride : des convolutions 1D extraient des motifs locaux, puis un LSTM lit leur suite.

    Idee reprise de nombreux articles C-MAPSS (blocs "CNN + LSTM") : la convolution resume chaque petit
    groupe de cycles consecutifs (ex. "le capteur 11 monte sur ces 5 cycles"), le LSTM suit l'evolution
    de ces motifs le long de la fenetre. Comme pour le LSTM simple, on garde la sortie du dernier cycle.

    Parameters
    ----------
    n_entrees : int
        Nombre de colonnes d'entree par cycle.
    nb_filtres : int, optionnel
        Nombre de filtres de chaque convolution (32 par defaut).
    noyau : int, optionnel
        Largeur impaire des filtres, en cycles (5 par defaut).
    nb_convolutions : int, optionnel
        Nombre de blocs "convolution -> BatchNorm -> ReLU" (2 par defaut).
    taille_cachee : int, optionnel
        Nombre d'unites du LSTM (64 par defaut).
    dropout : float, optionnel
        Dropout avant la couche de sortie (0,2 par defaut).
    """

    def __init__(self, n_entrees, nb_filtres=32, noyau=5, nb_convolutions=2, taille_cachee=64, dropout=0.2):
        """Cree les couches (fixer la graine avant : poids tires au hasard)."""
        super().__init__()
        couches = []
        canaux_entree = n_entrees
        for _ in range(nb_convolutions):  # un bloc de convolution a la fois
            couches.append(nn.Conv1d(canaux_entree, nb_filtres, kernel_size=noyau, padding=noyau // 2))
            couches.append(nn.BatchNorm1d(nb_filtres))
            couches.append(nn.ReLU())
            canaux_entree = nb_filtres
        self.convolutions = nn.Sequential(*couches)
        self.lstm = nn.LSTM(input_size=nb_filtres, hidden_size=taille_cachee, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.sortie = nn.Linear(taille_cachee, 1)

    def forward(self, x):
        """
        Calcule la RUL mise a l'echelle de chaque fenetre du lot.

        Parameters
        ----------
        x : torch.Tensor
            Forme (lot, longueur, n_entrees).

        Returns
        -------
        torch.Tensor
            Forme (lot,).
        """
        z = self.convolutions(x.transpose(1, 2))  # (lot, filtres, longueur) : convolution le long des cycles
        sorties, _ = self.lstm(z.transpose(1, 2))  # le LSTM relit les motifs dans l'ordre des cycles
        derniere = self.dropout(sorties[:, -1, :])  # sortie du cycle present
        return self.sortie(derniere).squeeze(-1)


class ModeleLSTMAttention(nn.Module):
    """
    LSTM avec attention temporelle : au lieu de ne garder que la sortie du dernier cycle, le reseau
    apprend un poids par cycle de la fenetre et fait une moyenne ponderee de toutes les sorties.

    Idee de plusieurs articles C-MAPSS (LSTM + attention) : certains cycles de la fenetre (un saut d'un
    capteur, par exemple) peuvent etre plus informatifs que le dernier. Les poids d'attention sont
    calcules par une petite couche lineaire suivie d'un softmax (poids positifs, de somme 1). La sortie
    du dernier cycle est concatenee au resume pondere, pour ne rien perdre par rapport au LSTM simple.

    Parameters
    ----------
    n_entrees : int
        Nombre de colonnes d'entree par cycle.
    taille_cachee : int, optionnel
        Unites de chaque couche LSTM (64 par defaut).
    nb_couches : int, optionnel
        Couches LSTM empilees (2 par defaut).
    dropout : float, optionnel
        Dropout entre couches et avant la sortie (0,2 par defaut).
    """

    def __init__(self, n_entrees, taille_cachee=64, nb_couches=2, dropout=0.2):
        """Cree les couches (fixer la graine avant : poids tires au hasard)."""
        super().__init__()
        dropout_entre_couches = 0.0
        if nb_couches > 1:
            dropout_entre_couches = dropout
        self.lstm = nn.LSTM(input_size=n_entrees, hidden_size=taille_cachee, num_layers=nb_couches,
                            batch_first=True, dropout=dropout_entre_couches)
        self.score = nn.Linear(taille_cachee, 1)  # un score d'importance par cycle
        self.dropout = nn.Dropout(dropout)
        self.sortie = nn.Linear(2 * taille_cachee, 1)  # [resume pondere, dernier cycle] -> RUL

    def forward(self, x):
        """
        Calcule la RUL mise a l'echelle de chaque fenetre du lot.

        Parameters
        ----------
        x : torch.Tensor
            Forme (lot, longueur, n_entrees).

        Returns
        -------
        torch.Tensor
            Forme (lot,).
        """
        sorties, _ = self.lstm(x)  # (lot, longueur, taille_cachee)
        poids = torch.softmax(self.score(sorties).squeeze(-1), dim=1)  # (lot, longueur), somme 1 par fenetre
        resume = torch.sum(sorties * poids.unsqueeze(-1), dim=1)  # moyenne ponderee des sorties
        combine = torch.cat([resume, sorties[:, -1, :]], dim=1)  # + sortie du dernier cycle
        return self.sortie(self.dropout(combine)).squeeze(-1)


def construire_reseau(config, n_entrees, graine=GRAINE):
    """
    Cree un reseau NEUF a partir d'une configuration (poids initiaux tires avec la graine).

    Pour "lstm" et "cnn1d", passe par `src.deep_learning.creer_modele` (memes poids initiaux que le
    notebook 03b a reglages egaux) ; les deux autres types sont definis dans ce module.

    Parameters
    ----------
    config : dict
        La configuration ("type", "hyperparametres").
    n_entrees : int
        Nombre de colonnes d'entree par cycle.
    graine : int, optionnel
        Graine des poids initiaux.

    Returns
    -------
    torch.nn.Module
        Le reseau non entraine.
    """
    type_reseau = config["type"]
    hyperparametres = config["hyperparametres"]
    if type_reseau in ("lstm", "cnn1d"):
        return creer_modele(type_reseau, n_entrees, graine=graine, hyperparametres=hyperparametres)
    fixer_graines(graine)  # meme principe que creer_modele : graine fixee juste avant de tirer les poids
    if type_reseau == "cnn_lstm":
        return ModeleCNNLSTM(n_entrees, **hyperparametres)
    if type_reseau == "lstm_attention":
        return ModeleLSTMAttention(n_entrees, **hyperparametres)
    raise ValueError(f"Type de reseau inconnu : {type_reseau!r} (choix possibles : {TYPES_RESEAUX}).")


# ---------------------------------------------------------------------------------------------------
# 3. Validation croisee d'une configuration, modele final, test
# ---------------------------------------------------------------------------------------------------

def plis_sous_jeu(graine_plis=GRAINE_PLIS, sous_jeu=SOUS_JEU_DEFAUT):
    """
    Renvoie les plis par moteur d'un sous-jeu (calculer_plis, 5 plis) pour une graine de decoupage.

    La graine 42 donne les plis des arbres et du notebook 03b ; d'autres graines (1000, 1001, ...) donnent
    des decoupages NEUFS, utilises pour confirmer un choix fait sur les plis de la graine 42.

    Parameters
    ----------
    graine_plis : int, optionnel
        Graine du melange des moteurs entre plis.
    sous_jeu : str, optionnel
        "FD001" (defaut) a "FD004".

    Returns
    -------
    list de (numpy.ndarray, numpy.ndarray)
        Les plis (idx_train, idx_val).
    """
    tableaux = preparer_tableaux(20, sous_jeu=sous_jeu)  # les plis ne dependent que des lignes et des moteurs
    return calculer_plis(tableaux["X"], tableaux["y"], tableaux["groups"], n_splits=N_PLIS, random_state=graine_plis)


def entrainer_un_pli(config, donnees, idx_train, idx_val, graine=GRAINE):
    """
    Entraine un reseau neuf sur un pli et predit ses moteurs de validation.

    Meme protocole que `src.deep_learning.preparer_taches_validation_croisee` : 20 % des moteurs
    d'entrainement du pli (tires avec la graine) servent a l'arret precoce, jamais les moteurs de
    validation du pli.

    Parameters
    ----------
    config : dict
        La configuration du reseau.
    donnees : dict
        Le resultat de `preparer_fenetres`.
    idx_train, idx_val : numpy.ndarray
        Les lignes d'entrainement et de validation du pli.
    graine : int, optionnel
        Graine des poids, des lots et de la separation d'arret precoce.

    Returns
    -------
    dict
        "predictions" (RUL predites des lignes idx_val, en cycles), "meilleure_epoque", "duree_s".
    """
    groupes = donnees["groups"]
    moteurs_communs = set(groupes[idx_train]) & set(groupes[idx_val])
    assert len(moteurs_communs) == 0, "un moteur est des deux cotes du pli"

    # separation apprentissage / arret precoce parmi les moteurs d'entrainement du pli
    moteurs_apprentissage, _ = separer_moteurs(np.unique(groupes[idx_train]),
                                               fraction_validation=FRACTION_ARRET_PRECOCE, graine=graine)
    masque = np.isin(groupes[idx_train], moteurs_apprentissage)
    idx_apprentissage = idx_train[masque]
    idx_arret = idx_train[~masque]

    reglages = config["reglages"]
    max_epoques = reglages["max_epoques"]
    patience = reglages["patience"]
    if ESSAI_RAPIDE:  # verification du script seulement : 2 epoques
        max_epoques = 2
        patience = 1
    modele = construire_reseau(config, donnees["fenetres"].shape[2], graine=graine)
    # selection : sous-ensembles paresseux, aucune fenetre recopiee (seulement les numeros de lignes)
    historique = entrainer_modele(
        modele,
        donnees["fenetres"].selection(idx_apprentissage),
        donnees["y"][idx_apprentissage],
        fenetres_validation=donnees["fenetres"].selection(idx_arret),
        y_validation=donnees["y"][idx_arret],
        alpha=reglages.get("alpha", 1.0),
        taux_apprentissage=reglages["taux_apprentissage"],
        taille_lot=reglages["taille_lot"],
        max_epoques=max_epoques,
        patience=patience,
        graine=graine,
    )
    resultat = {}
    resultat["predictions"] = predire(modele, donnees["fenetres"].selection(idx_val))
    resultat["meilleure_epoque"] = historique["meilleure_epoque"]
    resultat["duree_s"] = historique["duree_s"]
    return resultat


def evaluer_config_cv(config, graine_plis=GRAINE_PLIS, graine=GRAINE, apres_chaque_pli=None, sous_jeu=SOUS_JEU_DEFAUT):
    """
    Validation croisee 5 plis d'une configuration, un pli apres l'autre (dans ce processus).

    Apres chaque pli, `apres_chaque_pli(numero_pli, rmse_moyen_jusque_la)` est appele s'il est fourni :
    c'est la que l'objectif Optuna signale le resultat intermediaire et peut abandonner l'essai (il leve
    alors une exception qui interrompt la boucle).

    Parameters
    ----------
    config : dict
        La configuration du reseau.
    graine_plis : int, optionnel
        Graine du decoupage en plis (42 = plis des arbres et du 03b).
    graine : int, optionnel
        Graine des reseaux.
    apres_chaque_pli : callable ou None, optionnel
        Fonction appelee apres chaque pli.
    sous_jeu : str, optionnel
        "FD001" (defaut) a "FD004".

    Returns
    -------
    dict
        "rmse_cv_moyen", "rmse_par_pli", "meilleures_epoques", "oof" (predictions hors-pli, une par ligne
        du train), "duree_s".
    """
    donnees = preparer_fenetres(config["entrees"], sous_jeu=sous_jeu)
    plis = plis_sous_jeu(graine_plis, sous_jeu=sous_jeu)
    oof = np.full(len(donnees["y"]), np.nan)  # une prediction hors-pli par ligne du train
    rmse_par_pli = []
    meilleures_epoques = []
    debut = time.perf_counter()
    numero = 0
    for idx_train, idx_val in plis:  # un pli a la fois
        resultat = entrainer_un_pli(config, donnees, idx_train, idx_val, graine=graine)
        oof[idx_val] = resultat["predictions"]
        rmse_par_pli.append(rmse(donnees["y"][idx_val], resultat["predictions"]))
        meilleures_epoques.append(resultat["meilleure_epoque"])
        if apres_chaque_pli is not None:
            apres_chaque_pli(numero, float(np.mean(rmse_par_pli)))
        numero = numero + 1

    resultat_cv = {}
    resultat_cv["rmse_cv_moyen"] = float(np.mean(rmse_par_pli))
    resultat_cv["rmse_par_pli"] = rmse_par_pli
    resultat_cv["meilleures_epoques"] = meilleures_epoques
    resultat_cv["oof"] = oof
    resultat_cv["duree_s"] = time.perf_counter() - debut
    return resultat_cv


def executer_tache_pli(tache):
    """
    Tache d'un processus de calcul : un pli d'une configuration (pour paralleliser plusieurs CV).

    Parameters
    ----------
    tache : dict
        "config", "graine_plis", "numero_pli", "graine", "sous_jeu" (FD001 s'il manque).

    Returns
    -------
    dict
        La tache, plus "idx_val", "predictions", "meilleure_epoque", "duree_s".
    """
    torch.set_num_threads(1)  # 1 coeur par apprentissage (voir src/deep_learning.py)
    sous_jeu = tache.get("sous_jeu", SOUS_JEU_DEFAUT)
    donnees = preparer_fenetres(tache["config"]["entrees"], sous_jeu=sous_jeu)
    idx_train, idx_val = plis_sous_jeu(tache["graine_plis"], sous_jeu=sous_jeu)[tache["numero_pli"]]
    resultat = entrainer_un_pli(tache["config"], donnees, idx_train, idx_val, graine=tache["graine"])
    sortie = dict(tache)
    sortie["idx_val"] = idx_val
    sortie["predictions"] = resultat["predictions"]
    sortie["meilleure_epoque"] = resultat["meilleure_epoque"]
    sortie["duree_s"] = resultat["duree_s"]
    return sortie


def entrainer_modele_final(config, nb_epoques, graine=GRAINE, sous_jeu=SOUS_JEU_DEFAUT):
    """
    Entraine le modele final sur tous les moteurs du train, nombre d'epoques fixe (sans arret precoce).

    Le nombre d'epoques vient de la validation croisee (mediane des meilleures epoques des plis), jamais
    du test, comme au notebook 03b.

    Parameters
    ----------
    config : dict
        La configuration du reseau.
    nb_epoques : int
        Nombre d'epoques.
    graine : int, optionnel
        Graine des poids et des lots.
    sous_jeu : str, optionnel
        "FD001" (defaut) a "FD004".

    Returns
    -------
    torch.nn.Module
        Le reseau entraine, en mode prediction.
    """
    torch.set_num_threads(1)
    if ESSAI_RAPIDE:  # verification du script seulement : 1 epoque
        nb_epoques = 1
    donnees = preparer_fenetres(config["entrees"], sous_jeu=sous_jeu)
    reglages = config["reglages"]
    modele = construire_reseau(config, donnees["fenetres"].shape[2], graine=graine)
    entrainer_modele(
        modele,
        donnees["fenetres"],
        donnees["y"],
        alpha=reglages.get("alpha", 1.0),
        taux_apprentissage=reglages["taux_apprentissage"],
        taille_lot=reglages["taille_lot"],
        nb_epoques=int(nb_epoques),
        graine=graine,
    )
    return modele


def predictions_test_dernier_cycle(modele, config, sous_jeu=SOUS_JEU_DEFAUT):
    """
    Predit la RUL de chaque moteur test a son DERNIER cycle observe (le cycle note par le benchmark).

    Parameters
    ----------
    modele : torch.nn.Module
        Le reseau entraine.
    config : dict
        Sa configuration (pour retrouver les memes entrees).
    sous_jeu : str, optionnel
        "FD001" (defaut) a "FD004".

    Returns
    -------
    numpy.ndarray
        Une RUL predite par moteur test, dans l'ordre de "moteurs_test".
    """
    donnees = preparer_fenetres(config["entrees"], sous_jeu=sous_jeu)
    test = donnees["test"]
    moteurs = test["unit_number"].to_numpy()
    cycles = test["time_cycles"].to_numpy()
    positions = []  # position de la ligne du dernier cycle de chaque moteur test
    for moteur_id in donnees["moteurs_test"]:
        lignes = np.flatnonzero(moteurs == moteur_id)
        positions.append(lignes[np.argmax(cycles[lignes])])
    return predire(modele, donnees["fenetres_test"][np.asarray(positions)])


def evaluer_sur_test_dernier_cycle(predictions, y_test_vraie):
    """
    Metriques du test officiel (dernier cycle de chaque moteur), contre la RUL brute et la RUL plafonnee.

    Parameters
    ----------
    predictions : array-like
        Une RUL predite par moteur test.
    y_test_vraie : array-like
        La vraie RUL (RUL_FD00X.txt), non plafonnee.

    Returns
    -------
    dict
        rmse, rmse_contre_rul_plafonnee, mae, score_nasa, nb_surestimes, alertes_manquees, fausses_alertes,
        biais, nb_lignes.
    """
    vrais = np.asarray(y_test_vraie, dtype=float)
    predits = np.asarray(predictions, dtype=float)
    metriques = calculer_metriques(vrais, predits)
    metriques["rmse_contre_rul_plafonnee"] = rmse(np.minimum(vrais, PLAFOND_RUL), predits)
    metriques["score_nasa"] = score_nasa(vrais, predits)
    return metriques


# ---------------------------------------------------------------------------------------------------
# 4. Espaces de recherche Optuna
# ---------------------------------------------------------------------------------------------------

def reglages_apprentissage_essai(essai):
    """
    Tire les reglages d'apprentissage communs a tous les reseaux (pas d'Adam, taille des lots).

    Bornes : le pas va de 3e-4 a 5e-3 (echelle logarithmique, autour de la valeur par defaut 1e-3) ;
    les lots de 128 a 512 fenetres (en dessous, une epoque devient trop lente sur un coeur).
    max_epoques (80), patience (10) et alpha (1, perte quadratique) restent ceux du notebook 03b :
    l'objectif est le RMSE, qu'une perte asymetrique degraderait par construction.

    Parameters
    ----------
    essai : optuna.trial.Trial
        L'essai Optuna en cours.

    Returns
    -------
    dict
        Les reglages d'apprentissage.
    """
    reglages = {}
    reglages["taux_apprentissage"] = essai.suggest_float("taux_apprentissage", 3e-4, 5e-3, log=True)
    reglages["taille_lot"] = essai.suggest_categorical("taille_lot", [128, 256, 512])
    reglages["max_epoques"] = 80
    reglages["patience"] = 10
    reglages["alpha"] = 1.0
    return reglages


def valeurs_ecart_initial(sous_jeu=SOUS_JEU_DEFAUT):
    """
    Valeurs de k que Optuna peut tirer pour un sous-jeu : celles de VALEURS_ECART_INITIAL strictement plus
    petites que le nombre de cycles du plus court moteur test.

    Meme regle que pour FD001 (docs/04_pistes_non_retenues.md : k ne depasse pas 30, le plus court moteur
    test a 31 cycles). FD002 : 21 cycles, donc k = 20 seulement. Si aucune valeur ne passe (FD004, 19
    cycles), on garde 20, la valeur deja appliquee a ce sous-jeu par le notebook 03b.

    Parameters
    ----------
    sous_jeu : str, optionnel
        "FD001" (defaut) a "FD004".

    Returns
    -------
    list de int
        Les valeurs possibles de k.
    """
    test = preparer_tableaux(20, sous_jeu=sous_jeu)["test"]
    plus_court = int(test.groupby("unit_number")["time_cycles"].max().min())  # cycles du plus court moteur test
    valeurs = []
    for k in VALEURS_ECART_INITIAL:
        if k < plus_court:
            valeurs.append(k)
    if len(valeurs) == 0:
        valeurs.append(20)
    return valeurs


def entrees_essai(essai, sous_jeu=SOUS_JEU_DEFAUT):
    """
    Tire la longueur de fenetre et k de l'ecart a l'etat initial.

    Longueur : 20 a 50 cycles par pas de 5 (le 03b a teste 30 et 50). k : parmi `valeurs_ecart_initial`
    (20, 25 ou 30 sur FD001 ; 20 seulement sur FD002).

    Parameters
    ----------
    essai : optuna.trial.Trial
        L'essai Optuna en cours.
    sous_jeu : str, optionnel
        "FD001" (defaut) a "FD004".

    Returns
    -------
    dict
        La description des entrees.
    """
    entrees = {}
    entrees["longueur"] = essai.suggest_int("longueur", 20, 50, step=5)
    entrees["ecart_initial"] = essai.suggest_categorical("ecart_initial", valeurs_ecart_initial(sous_jeu))
    entrees["capteurs"] = True
    entrees["pente"] = None
    return entrees


def config_depuis_essai(essai, type_reseau, sous_jeu=SOUS_JEU_DEFAUT):
    """
    Tire une configuration complete pour un type de reseau (espace de recherche Optuna).

    Parameters
    ----------
    essai : optuna.trial.Trial
        L'essai Optuna en cours.
    type_reseau : str
        Un des TYPES_RESEAUX.
    sous_jeu : str, optionnel
        "FD001" (defaut) a "FD004" (seul l'espace de k en depend).

    Returns
    -------
    dict
        La configuration ("type", "hyperparametres", "reglages", "entrees").
    """
    hyperparametres = {}
    if type_reseau == "lstm":
        hyperparametres["taille_cachee"] = essai.suggest_categorical("taille_cachee", [32, 64, 96, 128, 192])
        hyperparametres["nb_couches"] = essai.suggest_int("nb_couches", 1, 3)
        hyperparametres["dropout"] = essai.suggest_float("dropout", 0.0, 0.5)
    elif type_reseau == "cnn1d":
        nb_blocs = essai.suggest_int("nb_blocs", 2, 4)
        nb_filtres = essai.suggest_categorical("nb_filtres", [16, 32, 64, 128])
        noyau = essai.suggest_categorical("noyau", [3, 5, 7, 9])
        canaux = []  # meme nombre de filtres et meme largeur pour tous les blocs (espace plus petit)
        noyaux = []
        for _ in range(nb_blocs):
            canaux.append(nb_filtres)
            noyaux.append(noyau)
        hyperparametres["canaux"] = canaux
        hyperparametres["noyaux"] = noyaux
        hyperparametres["dropout"] = essai.suggest_float("dropout", 0.0, 0.5)
    elif type_reseau == "cnn_lstm":
        hyperparametres["nb_filtres"] = essai.suggest_categorical("nb_filtres", [16, 32, 64])
        hyperparametres["noyau"] = essai.suggest_categorical("noyau", [3, 5, 7])
        hyperparametres["nb_convolutions"] = essai.suggest_int("nb_convolutions", 1, 2)
        hyperparametres["taille_cachee"] = essai.suggest_categorical("taille_cachee", [32, 64, 128])
        hyperparametres["dropout"] = essai.suggest_float("dropout", 0.0, 0.5)
    elif type_reseau == "lstm_attention":
        hyperparametres["taille_cachee"] = essai.suggest_categorical("taille_cachee", [32, 64, 128])
        hyperparametres["nb_couches"] = essai.suggest_int("nb_couches", 1, 2)
        hyperparametres["dropout"] = essai.suggest_float("dropout", 0.0, 0.5)
    else:
        raise ValueError(f"Type de reseau inconnu : {type_reseau!r}.")

    config = {}
    config["type"] = type_reseau
    config["hyperparametres"] = hyperparametres
    config["reglages"] = reglages_apprentissage_essai(essai)
    config["entrees"] = entrees_essai(essai, sous_jeu)
    return config


def parametres_optuna_reference(type_reseau):
    """
    Parametres Optuna equivalents a la configuration de reference du 03b (pour l'essai n° 0).

    Faire evaluer la reference comme premier essai sert de controle : sur FD001, on doit retrouver le RMSE
    de validation du notebook 03b (8,94 pour le LSTM), sinon le protocole differe. Sur les autres
    sous-jeux, c'est le point de depart : les reglages de FD001 appliques tels quels (k = 20 est toujours
    dans l'espace).

    Parameters
    ----------
    type_reseau : str
        "lstm" ou "cnn1d" (pas de reference pour les deux autres types).

    Returns
    -------
    dict ou None
        Les parametres a passer a `study.enqueue_trial`, ou None.
    """
    parametres = {"taux_apprentissage": 1e-3, "taille_lot": 256, "longueur": 30, "ecart_initial": 20, "dropout": 0.2}
    if type_reseau == "lstm":
        parametres["taille_cachee"] = 64
        parametres["nb_couches"] = 2
        return parametres
    return None  # le CNN de reference ([32, 64, 64]) n'est pas dans l'espace (filtres egaux) : pas d'essai 0


def copie_config(config):
    """
    Copie independante d'une configuration, convertie en types Python simples (ecrivable en JSON).

    Parameters
    ----------
    config : dict
        La configuration.

    Returns
    -------
    dict
        La copie.
    """
    return convertir_en_types_python(copy.deepcopy(config))
