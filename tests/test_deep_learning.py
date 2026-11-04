"""
Tests du deep learning sequentiel (src/deep_learning.py) : fenetres, reseaux, apprentissage, sauvegarde.

Tous les apprentissages se font sur de tout petits jeux synthetiques et de tout petits reseaux (quelques
centaines de poids, 1 a 3 epoques) : la suite reste rapide (quelques secondes), et chaque test verifie
UNE propriete precise (pas de fuite entre moteurs, pas de futur, reproductibilite, etc.).
"""

import numpy as np
import pandas as pd
import pytest

# src.deep_learning est importe AVANT torch : il charge xgboost / lightgbm en premier (ordre qui evite un plantage
# OpenMP quand d'autres tests font ensuite calculer XGBoost dans le meme processus ; voir src/deep_learning.py)
from src.deep_learning import (
    NOM_COLONNE_TIME_CYCLES_ECHELLE,
    ModeleCNN1D,
    ModeleLSTM,
    ajouter_time_cycles_echelle,
    charger_modele,
    chemin_modele,
    compter_parametres,
    construire_fenetres,
    creer_modele,
    entrainer_modele,
    evaluer_en_validation_croisee_sequentielle,
    executer_taches,
    lister_colonnes_entree,
    perte_quadratique_asymetrique_torch,
    predire,
    predire_tous_cycles,
    sauvegarder_modele,
    separer_moteurs,
)
from src.evaluation import calculer_plis, rmse
from src.modeles import perte_quadratique_asymetrique

import torch  # noqa: E402  (apres les modules du projet, voir ci-dessus)

# Correctif d'un blocage constate le 2026-11-04 : la suite complete (`python -m pytest -q`) restait bloquee
# indefiniment (0 % CPU, attente dans __kmp_join_barrier pendant un calcul LSTM) quand ces tests tournaient
# dans le meme processus que des tests qui avaient deja fait calculer xgboost / lightgbm (ex. test_config_modele).
# Cause : deux bibliotheques OpenMP (calcul parallele) dans le meme processus, celle de xgboost / lightgbm et celle
# de PyTorch ; un calcul PyTorch sur plusieurs coeurs attend alors des coeurs auxiliaires qui ne repondent pas.
# Remede : PyTorch sur 1 coeur (aucune region parallele OpenMP), comme avec OMP_NUM_THREADS=1. src/deep_learning.py
# le regle deja a l'import ; on le redit ici explicitement, pour que ces tests ne dependent pas d'un autre reglage.
torch.set_num_threads(1)

# petits reseaux pour des tests rapides (memes classes que les vrais modeles, tailles reduites)
PETIT_LSTM = {"taille_cachee": 8, "nb_couches": 2, "dropout": 0.1}
PETIT_CNN = {"canaux": [4, 8], "noyaux": [3, 3], "dropout": 0.1}


def fabriquer_tableau_deux_moteurs():
    """
    Fabrique un petit tableau a 2 moteurs, ou chaque valeur dit d'ou elle vient.

    Moteur 1 : 3 cycles ; moteur 2 : 5 cycles. Colonne "a" = 100 * moteur + cycle (ex. 203 = moteur 2,
    cycle 3) ; colonne "b" = -a. On peut donc lire dans une fenetre quel moteur et quel cycle elle contient.

    Returns
    -------
    pandas.DataFrame
        8 lignes, colonnes unit_number, time_cycles, a, b, RUL_cappee.
    """
    moteurs = [1, 1, 1, 2, 2, 2, 2, 2]
    cycles = [1, 2, 3, 1, 2, 3, 4, 5]
    df = pd.DataFrame()
    df["unit_number"] = moteurs
    df["time_cycles"] = cycles
    valeurs_a = []
    for position in range(len(moteurs)):  # a = 100 * moteur + cycle
        valeurs_a.append(100.0 * moteurs[position] + cycles[position])
    df["a"] = valeurs_a
    df["b"] = -df["a"]
    df["RUL_cappee"] = [2.0, 1.0, 0.0, 4.0, 3.0, 2.0, 1.0, 0.0]
    return df


def fabriquer_jeu_synthetique(nb_moteurs=12, nb_cycles=40, graine=0):
    """
    Fabrique un jeu synthetique facile a apprendre : la RUL se lit dans la tendance d'un capteur.

    Chaque moteur a `nb_cycles` cycles ; RUL = nb_cycles - cycle (plafonnee a 125, sans effet ici) ;
    capteur "c1" = 1 - RUL / nb_cycles + petit bruit (il monte quand la panne approche) ; "c2" = bruit pur.

    Parameters
    ----------
    nb_moteurs : int, optionnel
        Nombre de moteurs.
    nb_cycles : int, optionnel
        Nombre de cycles par moteur.
    graine : int, optionnel
        Graine du bruit.

    Returns
    -------
    pandas.DataFrame
        Colonnes unit_number, time_cycles, c1, c2, RUL_cappee.
    """
    generateur = np.random.default_rng(graine)
    lignes = []
    for moteur_id in range(1, nb_moteurs + 1):
        for cycle in range(1, nb_cycles + 1):
            rul = float(nb_cycles - cycle)
            ligne = {}
            ligne["unit_number"] = moteur_id
            ligne["time_cycles"] = cycle
            ligne["c1"] = 1.0 - rul / nb_cycles + 0.02 * generateur.normal()
            ligne["c2"] = generateur.normal()
            ligne["RUL_cappee"] = rul
            lignes.append(ligne)
    return pd.DataFrame(lignes)


# ---------------------------------------------------------------------------------------------------
# Fenetres
# ---------------------------------------------------------------------------------------------------

def test_fenetres_forme_et_remplissage_par_repetition():
    """Forme (n, longueur, entrees) ; au 2e cycle du moteur 1, les positions manquantes repetent le cycle 1."""
    df = fabriquer_tableau_deux_moteurs()
    fenetres, cible, numeros, cycles = construire_fenetres(df, ["a", "b"], longueur=4, colonne_cible="RUL_cappee")

    assert fenetres.shape == (8, 4, 2)  # une fenetre par ligne, 4 cycles, 2 entrees
    assert fenetres.dtype == np.float32
    # ligne 1 = moteur 1, cycle 2 : fenetre [cycle 1, cycle 1, cycle 1, cycle 2] (remplissage par le cycle 1)
    assert list(fenetres[1, :, 0]) == [101.0, 101.0, 101.0, 102.0]
    # ligne 7 = moteur 2, cycle 5 : fenetre complete, cycles 2 a 5, sans remplissage
    assert list(fenetres[7, :, 0]) == [202.0, 203.0, 204.0, 205.0]
    # la 2e entree suit la meme logique (b = -a)
    assert list(fenetres[7, :, 1]) == [-202.0, -203.0, -204.0, -205.0]
    # cible, moteur et cycle de chaque fenetre : ceux de sa ligne
    assert list(cible) == list(df["RUL_cappee"].astype(np.float32))
    assert list(numeros) == list(df["unit_number"])
    assert list(cycles) == list(df["time_cycles"])


def test_fenetres_jamais_un_autre_moteur_ni_le_futur():
    """Chaque fenetre ne contient que des cycles <= au cycle present, du meme moteur, dernier = cycle present."""
    df = fabriquer_tableau_deux_moteurs()
    fenetres, _, numeros, cycles = construire_fenetres(df, ["a"], longueur=4)

    for position in range(len(df)):  # une fenetre a la fois
        valeurs = fenetres[position, :, 0]
        moteurs_lus = np.floor(valeurs / 100.0)  # a = 100 * moteur + cycle
        cycles_lus = valeurs - 100.0 * moteurs_lus
        assert np.all(moteurs_lus == numeros[position])  # jamais un autre moteur
        assert np.all(cycles_lus <= cycles[position])  # jamais le futur
        assert cycles_lus[-1] == cycles[position]  # la derniere position est le cycle present
        assert np.all(np.diff(cycles_lus) >= 0)  # ordre chronologique dans la fenetre


def test_fenetres_alignees_sur_l_ordre_des_lignes_meme_melangees():
    """Si on melange les lignes du tableau, la fenetre de chaque ligne reste la meme (alignement ligne a ligne)."""
    df = fabriquer_tableau_deux_moteurs()
    fenetres_ordre, _, _, _ = construire_fenetres(df, ["a", "b"], longueur=3)

    ordre_melange = [5, 0, 7, 2, 4, 1, 6, 3]  # un ordre quelconque des 8 lignes
    df_melange = df.iloc[ordre_melange].reset_index(drop=True)
    fenetres_melange, _, numeros, cycles = construire_fenetres(df_melange, ["a", "b"], longueur=3)

    for position in range(len(ordre_melange)):
        ligne_origine = ordre_melange[position]  # cette ligne melangee etait la ligne ligne_origine
        assert np.array_equal(fenetres_melange[position], fenetres_ordre[ligne_origine])
        assert numeros[position] == df.loc[ligne_origine, "unit_number"]
        assert cycles[position] == df.loc[ligne_origine, "time_cycles"]


def test_fenetres_longueur_un_redonne_les_lignes():
    """Avec une fenetre de 1 cycle, chaque fenetre est exactement la ligne elle-meme."""
    df = fabriquer_tableau_deux_moteurs()
    fenetres, _, _, _ = construire_fenetres(df, ["a", "b"], longueur=1)
    assert np.array_equal(fenetres[:, 0, :], df[["a", "b"]].to_numpy(dtype=np.float32))


def test_fenetres_erreurs_explicites():
    """NaN dans les entrees, cycle en double, colonne manquante ou longueur nulle : erreur claire."""
    df = fabriquer_tableau_deux_moteurs()

    df_nan = df.copy()
    df_nan.loc[2, "a"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        construire_fenetres(df_nan, ["a"], longueur=3)

    df_double = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # moteur 1, cycle 1 deux fois
    with pytest.raises(ValueError, match="meme cycle"):
        construire_fenetres(df_double, ["a"], longueur=3)

    with pytest.raises(ValueError, match="manquante"):
        construire_fenetres(df, ["capteur_absent"], longueur=3)

    with pytest.raises(ValueError, match="longueur"):
        construire_fenetres(df, ["a"], longueur=0)


def test_fenetres_sur_le_vrai_train_fd001(contexte_train):
    """Sur FD001 : 20 631 fenetres ; dernier pas = la ligne ; 1re fenetre du moteur 2 = son 1er cycle repete."""
    train = contexte_train["train"]
    capteurs = contexte_train["sensor_cols_restants"]
    fenetres, cible, numeros, _ = construire_fenetres(train, capteurs, longueur=30, colonne_cible="RUL_cappee")

    assert fenetres.shape == (20631, 30, 15)
    valeurs = train[capteurs].to_numpy(dtype=np.float32)
    assert np.array_equal(fenetres[:, -1, :], valeurs)  # le dernier pas de chaque fenetre est sa propre ligne
    assert np.array_equal(cible, train["RUL_cappee"].to_numpy(dtype=np.float32))

    # premiere ligne du moteur 2 : sa fenetre ne contient QUE son propre 1er cycle (aucun cycle du moteur 1)
    position_moteur_2 = int(np.flatnonzero(numeros == 2)[0])
    for pas in range(30):
        assert np.array_equal(fenetres[position_moteur_2, pas, :], valeurs[position_moteur_2])


def test_time_cycles_echelle_et_liste_des_entrees():
    """time_cycles est divise par la constante fixe ; la colonne est ajoutee en derniere position."""
    df = fabriquer_tableau_deux_moteurs()
    df_echelle = ajouter_time_cycles_echelle(df, echelle=100.0)
    assert list(df_echelle[NOM_COLONNE_TIME_CYCLES_ECHELLE]) == [0.01, 0.02, 0.03, 0.01, 0.02, 0.03, 0.04, 0.05]
    assert NOM_COLONNE_TIME_CYCLES_ECHELLE not in df.columns  # le tableau d'origine n'est pas modifie
    assert lister_colonnes_entree(["a", "b"], True) == ["a", "b", NOM_COLONNE_TIME_CYCLES_ECHELLE]
    assert lister_colonnes_entree(["a", "b"], False) == ["a", "b"]


# ---------------------------------------------------------------------------------------------------
# Reseaux et perte
# ---------------------------------------------------------------------------------------------------

def test_formes_de_sortie_des_deux_reseaux():
    """Une RUL (mise a l'echelle) par fenetre, quelle que soit la longueur de fenetre."""
    torch.manual_seed(0)
    for longueur in [30, 12]:
        x = torch.rand(7, longueur, 16)  # 7 fenetres de `longueur` cycles et 16 entrees
        lstm = ModeleLSTM(16)
        cnn = ModeleCNN1D(16)
        lstm.eval()
        cnn.eval()
        assert lstm(x).shape == (7,)
        assert cnn(x).shape == (7,)
    # tailles par defaut : environ 54 000 et 26 000 poids (valeurs citees dans les docstrings)
    assert compter_parametres(ModeleLSTM(16)) == 54337
    assert compter_parametres(ModeleCNN1D(16)) == 25633


def test_cnn_refuse_un_filtre_de_largeur_paire():
    """Une largeur paire ne garde pas la longueur de la fenetre avec padding = largeur // 2 : refusee."""
    with pytest.raises(ValueError, match="impaire"):
        ModeleCNN1D(4, canaux=[8], noyaux=[4])


def test_perte_torch_identique_a_la_perte_numpy_de_src_modeles():
    """La perte PyTorch est la moyenne de la perte quadratique asymetrique de src/modeles.py ; alpha = 1 -> MSE."""
    vrais = np.array([0.5, 0.2, 0.9, 0.4])
    predits = np.array([0.6, 0.1, 0.9, 0.7])  # surestimation, sous-estimation, exact, surestimation
    for alpha in [1.0, 2.26]:
        attendu = float(np.mean(perte_quadratique_asymetrique(vrais, predits, alpha)))
        obtenu = float(perte_quadratique_asymetrique_torch(torch.tensor(predits), torch.tensor(vrais), alpha=alpha))
        assert obtenu == pytest.approx(attendu)
    mse = float(np.mean((predits - vrais) ** 2))
    assert float(perte_quadratique_asymetrique_torch(torch.tensor(predits), torch.tensor(vrais), 1.0)) == pytest.approx(mse)


def test_predire_renvoie_des_cycles_clippes_a_zero():
    """La sortie est remultipliee par 125 et ramenee a 0 si elle est negative."""
    modele = creer_modele("cnn1d", 2, graine=0, hyperparametres=PETIT_CNN)
    with torch.no_grad():  # on force une sortie tres negative : poids de sortie nuls, biais -1
        modele.sortie.weight.zero_()
        modele.sortie.bias.fill_(-1.0)
    predictions = predire(modele, np.zeros((5, 10, 2), dtype=np.float32))
    assert np.all(predictions == 0.0)
    with torch.no_grad():  # biais +0.5 -> 0.5 x 125 = 62.5 cycles
        modele.sortie.bias.fill_(0.5)
    assert np.allclose(predire(modele, np.zeros((5, 10, 2), dtype=np.float32)), 62.5)


# ---------------------------------------------------------------------------------------------------
# Apprentissage
# ---------------------------------------------------------------------------------------------------

def preparer_donnees_synthetiques(longueur=10):
    """
    Fenetres et cibles du jeu synthetique, separees en moteurs d'apprentissage (1-9) et de validation (10-12).

    Parameters
    ----------
    longueur : int, optionnel
        Longueur des fenetres.

    Returns
    -------
    tuple
        (fenetres_app, y_app, fenetres_val, y_val, df, fenetres, y).
    """
    df = fabriquer_jeu_synthetique()
    fenetres, y, numeros, _ = construire_fenetres(df, ["c1", "c2"], longueur=longueur, colonne_cible="RUL_cappee")
    masque_app = numeros <= 9
    return fenetres[masque_app], y[masque_app], fenetres[~masque_app], y[~masque_app], df, fenetres, y


@pytest.mark.parametrize("type_modele", ["lstm", "cnn1d"])
def test_mini_apprentissage_fait_baisser_l_erreur(type_modele):
    """Quelques epoques sur un probleme facile : le RMSE de validation baisse nettement par rapport au depart."""
    f_app, y_app, f_val, y_val, _, _, _ = preparer_donnees_synthetiques()
    hyper = PETIT_LSTM
    if type_modele == "cnn1d":
        hyper = PETIT_CNN
    modele = creer_modele(type_modele, 2, graine=1, hyperparametres=hyper)
    rmse_avant = rmse(y_val, predire(modele, f_val))  # reseau neuf : predictions au hasard

    historique = entrainer_modele(modele, f_app, y_app, f_val, y_val, taux_apprentissage=1e-2, taille_lot=32,
                                  max_epoques=3, patience=5, graine=1)
    rmse_apres = rmse(y_val, predire(modele, f_val))

    assert rmse_apres < 0.8 * rmse_avant  # l'apprentissage a reduit l'erreur d'au moins 20 %
    assert historique["perte_entrainement"][-1] < historique["perte_entrainement"][0]
    assert historique["nb_epoques_effectuees"] == 3


def test_arret_precoce_garde_les_poids_de_la_meilleure_epoque():
    """Les poids rendus sont ceux de la meilleure epoque de validation, et leur RMSE est celui de l'historique."""
    f_app, y_app, f_val, y_val, _, _, _ = preparer_donnees_synthetiques()
    modele = creer_modele("lstm", 2, graine=3, hyperparametres=PETIT_LSTM)
    historique = entrainer_modele(modele, f_app, y_app, f_val, y_val, taux_apprentissage=5e-2, taille_lot=32,
                                  max_epoques=6, patience=2, graine=3)

    pertes = historique["perte_validation"]
    meilleure_position = int(np.argmin(pertes))  # epoque (comptee a partir de 0) de la plus petite perte
    assert historique["meilleure_epoque"] == meilleure_position + 1
    # les poids remis en place sont bien ceux de cette epoque : on retrouve son RMSE de validation
    assert rmse(y_val, predire(modele, f_val)) == pytest.approx(historique["rmse_validation"][meilleure_position])
    # l'arret ne peut survenir qu'apres `patience` epoques sans amelioration
    if historique["arret_precoce"]:
        assert historique["nb_epoques_effectuees"] == historique["meilleure_epoque"] + 2


def test_sans_validation_il_faut_un_nombre_d_epoques_fixe():
    """Modele final : sans moteurs de validation, nb_epoques est obligatoire et respecte exactement."""
    f_app, y_app, _, _, _, _, _ = preparer_donnees_synthetiques()
    modele = creer_modele("cnn1d", 2, graine=0, hyperparametres=PETIT_CNN)
    with pytest.raises(ValueError, match="nb_epoques"):
        entrainer_modele(modele, f_app, y_app)
    historique = entrainer_modele(modele, f_app, y_app, nb_epoques=2, taille_lot=64)
    assert historique["nb_epoques_effectuees"] == 2
    assert historique["meilleure_epoque"] == 2
    assert historique["perte_validation"] == []


@pytest.mark.parametrize("type_modele", ["lstm", "cnn1d"])
def test_reproductibilite_a_graine_egale(type_modele):
    """Meme graine -> exactement les memes predictions ; autre graine -> predictions differentes."""
    f_app, y_app, f_val, y_val, _, _, _ = preparer_donnees_synthetiques()
    hyper = PETIT_LSTM
    if type_modele == "cnn1d":
        hyper = PETIT_CNN

    predictions = []
    for graine in [7, 7, 8]:  # deux fois la graine 7, puis la graine 8
        modele = creer_modele(type_modele, 2, graine=graine, hyperparametres=hyper)
        entrainer_modele(modele, f_app, y_app, f_val, y_val, taille_lot=32, max_epoques=2, patience=5, graine=graine)
        predictions.append(predire(modele, f_val))

    assert np.array_equal(predictions[0], predictions[1])  # bit pour bit
    assert not np.array_equal(predictions[0], predictions[2])


def test_taches_en_parallele_identiques_au_sequentiel():
    """Lancer deux apprentissages dans deux processus donne exactement les memes resultats qu'a la suite."""
    f_app, y_app, f_val, _, _, _, _ = preparer_donnees_synthetiques()
    taches = []
    for graine in [1, 2]:
        tache = {}
        tache["type_modele"] = "cnn1d"
        tache["hyperparametres"] = PETIT_CNN
        tache["fenetres_entrainement"] = f_app
        tache["y_entrainement"] = y_app
        tache["nb_epoques"] = 1
        tache["graine"] = graine
        tache["reglages"] = {"taille_lot": 64}
        tache["a_predire"] = {"validation": f_val}
        taches.append(tache)
    sequentiel = executer_taches(taches, nb_processus=1, verbeux=False)
    parallele = executer_taches(taches, nb_processus=2, verbeux=False)
    for position in range(2):
        assert np.array_equal(sequentiel[position]["predictions"]["validation"],
                              parallele[position]["predictions"]["validation"])


# ---------------------------------------------------------------------------------------------------
# Separation par moteurs et validation croisee
# ---------------------------------------------------------------------------------------------------

def test_separer_moteurs_disjoints_complets_reproductibles():
    """20 % des moteurs en validation, groupes disjoints, union complete, meme tirage a graine egale."""
    moteurs = np.arange(1, 101)
    apprentissage, validation = separer_moteurs(moteurs, fraction_validation=0.2, graine=42)
    assert len(validation) == 20 and len(apprentissage) == 80
    assert len(set(apprentissage) & set(validation)) == 0
    assert sorted(set(apprentissage) | set(validation)) == list(moteurs)
    apprentissage_2, validation_2 = separer_moteurs(moteurs, fraction_validation=0.2, graine=42)
    assert np.array_equal(validation, validation_2)
    _, validation_3 = separer_moteurs(moteurs, fraction_validation=0.2, graine=1)
    assert not np.array_equal(validation, validation_3)


def test_validation_croisee_sequentielle_sur_petit_jeu():
    """OOF completes et alignees ; moteurs d'arret precoce jamais en validation ; memes cles que les arbres."""
    _, _, _, _, df, fenetres, y = preparer_donnees_synthetiques()
    groups = df["unit_number"].to_numpy()
    plis = calculer_plis(fenetres[:, -1, :], y, groups, n_splits=3, random_state=42)

    resultats = evaluer_en_validation_croisee_sequentielle(
        fenetres, y, groups, plis, "cnn1d", hyperparametres=PETIT_CNN,
        reglages_entrainement={"max_epoques": 2, "patience": 5, "taille_lot": 64}, verbeux=False,
    )

    assert resultats["predictions_oof"].shape == (len(df),)
    assert not np.any(np.isnan(resultats["predictions_oof"]))
    assert np.all(resultats["predictions_oof"] >= 0)
    # chaque prediction OOF vient du modele du pli ou sa ligne etait en validation : on le verifie sur le RMSE
    for position in range(len(plis)):
        idx_val = plis[position][1]
        assert rmse(y[idx_val], resultats["predictions_oof"][idx_val]) == pytest.approx(resultats["rmse_par_pli"][position])
        moteurs_val = set(groups[idx_val])
        assert len(set(resultats["moteurs_arret_precoce_par_pli"][position]) & moteurs_val) == 0

    cles_arbres = ["rmse_par_pli", "mae_par_pli", "nasa_par_pli", "surestimes_par_pli", "rmse_moyen", "rmse_ecart_type",
                   "mae_moyen", "nasa_moyen", "surestimes_oof", "alertes_manquees_oof", "fausses_alertes_oof",
                   "biais_oof", "predictions_oof", "temps_entrainement_total", "temps_prediction_total"]
    for cle in cles_arbres:
        assert cle in resultats
    assert len(resultats["meilleures_epoques"]) == 3


# ---------------------------------------------------------------------------------------------------
# Prediction a chaque cycle, sauvegarde et chargement
# ---------------------------------------------------------------------------------------------------

def test_predire_tous_cycles_une_ligne_par_cycle():
    """Une prediction par ligne du tableau, dans le meme ordre, toutes >= 0."""
    df = fabriquer_jeu_synthetique(nb_moteurs=3, nb_cycles=15)
    modele = creer_modele("lstm", 2, graine=0, hyperparametres=PETIT_LSTM)
    resultat = predire_tous_cycles(modele, df, ["c1", "c2"], longueur=10)
    assert len(resultat) == len(df)
    assert list(resultat["unit_number"]) == list(df["unit_number"])
    assert list(resultat["time_cycles"]) == list(df["time_cycles"])
    assert np.all(resultat["rul_predite"] >= 0)


@pytest.mark.parametrize("type_modele", ["lstm", "cnn1d"])
def test_sauvegarde_et_chargement_redonnent_les_memes_predictions(tmp_path, type_modele):
    """Poids + hyperparametres + informations relus a l'identique : memes predictions au bit pres."""
    f_app, y_app, f_val, _, _, _, _ = preparer_donnees_synthetiques()
    hyper = PETIT_LSTM
    if type_modele == "cnn1d":
        hyper = PETIT_CNN
    modele = creer_modele(type_modele, 2, graine=5, hyperparametres=hyper)
    entrainer_modele(modele, f_app, y_app, nb_epoques=1, taille_lot=64, graine=5)
    predictions_avant = predire(modele, f_val)

    chemin = chemin_modele("FD001", type_modele, dossier=tmp_path)
    assert chemin.endswith(f"FD001__{type_modele}.pt")
    informations = {"colonnes_entree": ["c1", "c2"], "longueur_fenetre": 10, "nb_epoques": 1}
    sauvegarder_modele(modele, chemin, type_modele, 2, hyperparametres=hyper, informations=informations)

    modele_relu, contenu = charger_modele(chemin)
    assert np.array_equal(predire(modele_relu, f_val), predictions_avant)
    assert contenu["type_modele"] == type_modele
    assert contenu["n_entrees"] == 2
    assert contenu["informations"]["colonnes_entree"] == ["c1", "c2"]
    for cle in hyper:
        assert contenu["hyperparametres"][cle] == hyper[cle]
