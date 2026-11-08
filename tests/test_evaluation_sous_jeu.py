"""
Tests du socle des notebooks de resultats : src/evaluation_sous_jeu.py et src/figures_resultats.py.

Plan de reorganisation des notebooks (docs/plan_refonte_notebooks.md, etape B) :

(a) `test_verifier_moteur_egal_fichiers_demo` : pour les 4 sous-jeux, plusieurs moteurs test et tous les modeles
    disponibles, la prediction recalculee cycle par cycle depuis data/test_FD00X.txt (`verifier_moteur`) est egale,
    a 1e-4 cycle pres, a celle des fichiers de la demo (sorties/predictions/). Pendant le recalcul, le dossier des
    predictions est remplace par un dossier VIDE : si `verifier_moteur` lisait ces fichiers, il echouerait
    (principe 3 du plan : la preuve n'est pas circulaire) ;
(b) `test_evaluer_fd001_chiffres_de_reference` : `evaluer_sous_jeu("FD001")` redonne les chiffres de reference
    (plan, section 9.2, tels que confirmes par la regeneration du 2026-11-07 sur cette machine) ;
(c) `test_evaluer_tous_les_sous_jeux` : la fonction marche sur les 4 sous-jeux (tous ont des reseaux depuis le
    calcul de nuit FD003 / FD004 du 2026-11-07 au 08), et ses tableaux sont coherents entre eux ;
(d) `test_figures` : chaque fonction de figure produit une figure, sur un sous-jeu avec reseaux (FD001) et sur un
    sous-jeu SANS reseau. Aucun sous-jeu reel n'en est depourvu depuis le 2026-11-08 : le cas est simule sur FD003
    (`evaluation_sans_reseau` : dossier de predictions sans les fichiers des reseaux, et FD003 retire de
    ENSEMBLES_RETENUS le temps du test) ;
(e) `test_complements_arbres` : erreur d'entrainement des arbres (pli par pli) et validation des arbres sur les
    3 decoupages neufs du 04h, relues et controlees (scripts/valider_arbres_finaux.py --complements) ;
(f) `test_generateur_ecrit_le_sous_jeu_dans_les_parametres` : le script de generation ecrit le sous-jeu dans la
    cellule de parametres de chaque notebook genere (un notebook relance a la main recalcule son sous-jeu) ;
(g) `test_modele_sans_reseau_s_execute` : les cellules de code du modele notebooks/resultats/_modele.ipynb
    s'executent sur le FD003 simule sans reseau (meme simulation qu'en (d), et poids des reseaux retires de
    l'inference en direct), et passent par leurs branches « sans reseau ». Ajoute le 2026-11-08 : depuis que les
    4 sous-jeux ont des reseaux, aucun notebook genere ne passe plus par ces branches.

Les tests lisent sorties/ et models/, non versionnes : s'ils manquent, le test est saute avec la commande a lancer.
Lancement, depuis la racine du depot : python -m pytest -q tests/test_evaluation_sous_jeu.py (environ 2 minutes).
"""

import os
import sys

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")  # aucune fenetre : les figures sont seulement construites

# racine du depot (dossier parent de tests/) importable, pour "import src..." quel que soit le dossier de lancement
RACINE_DEPOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RACINE_DEPOT not in sys.path:
    sys.path.insert(0, RACINE_DEPOT)

# src.evaluation_sous_jeu importe src.inference, qui charge xgboost AVANT torch (ordre voulu, src/deep_learning.py)
from src import evaluation_sous_jeu as es  # noqa: E402
from src import figures_resultats as fr  # noqa: E402
from src import inference as inf  # noqa: E402
from src.contrat_predictions import SOUS_JEUX, VARIABLE_ENV_DOSSIER  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

TOLERANCE_EGALITE = 1e-4  # cycles : meme tolerance que tests/test_anti_triche.py
COMMANDE_PREPARATION = (
    "python scripts/preparer_demo.py (arbres, réseaux du 04h, inférence), puis "
    "LOKY_MAX_CPU_COUNT=4 python scripts/valider_arbres_finaux.py, puis la même commande avec --complements"
)

# l'objet d'inference du DERNIER sous-jeu charge : un seul a la fois en memoire (une Random Forest pese lourd)
_INFERENCE_COURANTE = {}


def sauter_si_resultats_absents(sous_jeu):
    """
    Saute le test si les fichiers non versionnes dont il depend manquent (predictions de la demo, validation des arbres).

    Parameters
    ----------
    sous_jeu : str
        "FD001" a "FD004".
    """
    modeles, absents = es.modeles_publies(sous_jeu)
    if len(modeles) == 0:
        pytest.skip(f"{sous_jeu} : aucune prediction de la demo (lancer {COMMANDE_PREPARATION})")
    if not os.path.exists(es.FICHIER_CV_OOF):
        pytest.skip(f"validation croisee des arbres absente (lancer {COMMANDE_PREPARATION})")


def inference_reelle(sous_jeu):
    """
    Charge les vrais modeles de models/ pour un sous-jeu (garde en memoire tant qu'on reste sur ce sous-jeu).

    Rien n'est ecrit sur le disque : un modele a arbres absent est entraine en memoire, un reseau absent est
    indisponible.

    Parameters
    ----------
    sous_jeu : str
        "FD001" a "FD004".

    Returns
    -------
    dict
        Le resultat de `src.inference.charger_modeles`.
    """
    if _INFERENCE_COURANTE.get("sous_jeu") != sous_jeu:
        _INFERENCE_COURANTE.clear()  # libere les modeles du sous-jeu precedent
        _INFERENCE_COURANTE["sous_jeu"] = sous_jeu
        _INFERENCE_COURANTE["inference"] = inf.charger_modeles(sous_jeu, entrainer_si_absent=True, sauvegarder=False)
    return _INFERENCE_COURANTE["inference"]


def evaluation_sans_reseau(sous_jeu, monkeypatch, dossier):
    """
    Evalue un sous-jeu comme s'il n'avait pas de reseau (cas d'un sous-jeu sans ensemble retenu au notebook 04h).

    Deux changements, annules par pytest a la fin du test (monkeypatch) : (1) le dossier des predictions est
    remplace par `dossier`, ou l'on recopie seulement les fichiers des modeles qui ne sont pas des reseaux ;
    (2) le sous-jeu est retire de ENSEMBLES_RETENUS, tel que src.evaluation_sous_jeu le voit. Les vrais fichiers
    de sorties/predictions/ ne sont pas touches.

    Parameters
    ----------
    sous_jeu : str
        "FD001" a "FD004".
    monkeypatch : pytest.MonkeyPatch
        L'outil de pytest qui remet tout en place a la fin du test.
    dossier : pathlib.Path
        Un dossier vide (tmp_path de pytest), qui devient le dossier des predictions.

    Returns
    -------
    dict
        Le resultat de `es.evaluer_sous_jeu(sous_jeu)` sans reseau.
    """
    import shutil

    # (1) copie des fichiers de predictions (et de leurs metadonnees) des modeles qui ne sont pas des reseaux
    for cle in es.MODELES_CONNUS:
        if cle in inf.MODELES_RESEAUX:
            continue  # les reseaux restent absents du dossier
        for source in [es.chemin_predictions(sous_jeu, cle), es.chemin_metadonnees(sous_jeu, cle)]:
            if os.path.exists(source):
                shutil.copy(source, dossier / os.path.basename(source))
    monkeypatch.setenv(VARIABLE_ENV_DOSSIER, str(dossier))

    # (2) ensembles retenus sans ce sous-jeu (copie : le dictionnaire du module n'est pas modifie)
    ensembles = {}
    for nom in es.ENSEMBLES_RETENUS:
        if nom != sous_jeu:
            ensembles[nom] = es.ENSEMBLES_RETENUS[nom]
    monkeypatch.setattr(es, "ENSEMBLES_RETENUS", ensembles)
    return es.evaluer_sous_jeu(sous_jeu)


@pytest.mark.parametrize("sous_jeu", SOUS_JEUX)
def test_verifier_moteur_egal_fichiers_demo(sous_jeu, monkeypatch, tmp_path):
    """(a) Recalcul depuis les donnees brutes, sans lire sorties/predictions/, puis egalite avec les fichiers de la demo."""
    sauter_si_resultats_absents(sous_jeu)
    modeles_demo, _ = es.modeles_publies(sous_jeu)  # les modeles que la demo affiche pour ce sous-jeu
    inference = inference_reelle(sous_jeu)
    for cle in modeles_demo:
        if cle not in inference["modeles"]:
            pytest.skip(f"{sous_jeu}/{cle} : {inference['indisponibles'].get(cle)}")

    # trois moteurs : le premier, le 42 (celui du plan, section 5.3) et le dernier ; tous leurs cycles
    moteurs = inf.lister_moteurs_test(sous_jeu)
    moteurs_testes = [moteurs[0], 42, moteurs[-1]]
    for moteur in moteurs_testes:
        # 1) recalcul avec un dossier de predictions VIDE : verifier_moteur ne peut pas lire les fichiers de la demo
        dossier_vide = tmp_path / f"predictions_vides_{moteur}"
        dossier_vide.mkdir()
        monkeypatch.setenv(VARIABLE_ENV_DOSSIER, str(dossier_vide))
        verification = es.verifier_moteur(sous_jeu, moteur, inference=inference, modeles=modeles_demo)
        monkeypatch.delenv(VARIABLE_ENV_DOSSIER)

        nb_cycles = inf.mesures_moteur_test(sous_jeu, moteur)["nb_cycles"]
        assert len(verification["trajectoire"]) == nb_cycles  # une ligne par cycle observe
        assert list(verification["trajectoire"]["time_cycles"]) == list(range(1, nb_cycles + 1))
        assert verification["modeles"] == modeles_demo
        assert len(verification["dernier_cycle"]) == len(modeles_demo)

        # 2) seulement ensuite, comparaison aux fichiers de la demo (dossier du contrat)
        detail, resume = es.comparer_aux_fichiers_demo(verification, tolerance=TOLERANCE_EGALITE)
        assert len(detail) == nb_cycles * len(modeles_demo)  # chaque cycle de chaque modele compare
        assert list(resume["cle"]) == modeles_demo
        assert resume["identique"].all(), f"{sous_jeu}, moteur {moteur} :\n{resume.to_string()}"
        assert float(resume["ecart_max"].max()) <= TOLERANCE_EGALITE


def test_evaluer_fd001_chiffres_de_reference():
    """(b) FD001 : chiffres de reference du test officiel et de la validation croisee (plan 9.2, regeneration du 2026-11-07)."""
    sauter_si_resultats_absents("FD001")
    evaluation = es.evaluer_sous_jeu("FD001")
    assert evaluation["modeles"] == ["reference_naive", "random_forest", "xgboost_phase3", "lightgbm", "lstm", "cnn1d"]

    # test officiel, contre la RUL brute : (RMSE, score NASA, surestimes > 20). Plan 9.2 : XGBoost 12,73 sur la
    # machine de Matisse (1 coeur) ; 12,78 sur cette machine (regeneration du 2026-11-07, ecart signale).
    # Reseaux : ceux du notebook 04h (journal du plan, 2026-11-07), pas les chiffres 03b de la section 9.2.
    attendus_test = {
        "reference_naive": (43.07, 33629.2, 41),
        "random_forest": (13.14, 243.3, 3),
        "xgboost_phase3": (12.78, 232.5, 2),
        "lightgbm": (12.76, 234.6, 3),
        "lstm": (11.16, 227.2, 4),
        "cnn1d": (11.68, 189.3, 4),
    }
    brute = evaluation["regression"]["test_brute"].set_index("cle")
    for cle in attendus_test:
        rmse_attendu, nasa_attendu, surestimes_attendus = attendus_test[cle]
        assert round(float(brute.loc[cle, "rmse"]), 2) == rmse_attendu, cle
        assert round(float(brute.loc[cle, "nasa_somme"]), 1) == nasa_attendu, cle
        assert int(brute.loc[cle, "surestimes"]) == surestimes_attendus, cle

    # RMSE contre la RUL plafonnee a 125
    attendus_plafonnee = {"xgboost_phase3": 11.46, "lightgbm": 11.50, "random_forest": 11.84, "lstm": 10.09, "cnn1d": 10.33}
    plafonnee = evaluation["regression"]["test_plafonnee"].set_index("cle")
    for cle in attendus_plafonnee:
        assert round(float(plafonnee.loc[cle, "rmse"]), 2) == attendus_plafonnee[cle], cle

    # validation croisee : arbres (reglages finaux, GroupKFold 5 plis graine 42), reseaux (04h, 3 decoupages neufs)
    attendus_cv = {"reference_naive": 41.69, "random_forest": 9.95, "xgboost_phase3": 9.12, "lightgbm": 9.06,
                   "lstm": 7.38, "cnn1d": 8.30, "Ridge": 16.95}
    validation = evaluation["validation"]["tableau"].set_index("cle")
    for cle in attendus_cv:
        assert round(float(validation.loc[cle, "rmse_cv_moyen"]), 2) == attendus_cv[cle], cle
    assert validation.loc["lstm", "ecart_type_entre"] == "3 decoupages"
    assert validation.loc["xgboost_phase3", "ecart_type_entre"] == "5 plis"

    # baselines (notebook 05, G.2 et G.7, memes 31 variables et memes plis) : Ridge R2 0,834, RMSE train 15,70
    complet = evaluation["validation"]["complet"].set_index("cle")
    assert round(float(complet.loc["Ridge", "r2_moyenne"]), 3) == 0.834
    train = evaluation["validation"]["train_baselines"].set_index("cle")
    assert round(float(train.loc["Ridge", "rmse_moyenne"]), 2) == 15.70

    # section 1 : 11 moteurs test de vraie RUL > 125, plancher 3,74 (notebook 05, H.1)
    assert evaluation["donnees"]["nb_rul_vraie_sup_plafond"] == 11
    assert round(evaluation["donnees"]["plancher_rmse_rul_brute"], 2) == 3.74
    assert evaluation["donnees"]["nb_features_arbres"] == 31


@pytest.mark.parametrize("sous_jeu", SOUS_JEUX)
def test_evaluer_tous_les_sous_jeux(sous_jeu):
    """(c) Les 4 sous-jeux : modeles attendus (6, reseaux du 04h compris), tableaux complets et coherents."""
    sauter_si_resultats_absents(sous_jeu)
    evaluation = es.evaluer_sous_jeu(sous_jeu)
    nb_moteurs_test = evaluation["donnees"]["nb_moteurs_test"]

    # modeles : 6 sur chaque sous-jeu, avec les reseaux du 04h (FD003 et FD004 depuis le calcul du 2026-11-07 au 08)
    assert sous_jeu in es.ENSEMBLES_RETENUS
    assert evaluation["modeles"] == ["reference_naive", "random_forest", "xgboost_phase3", "lightgbm", "lstm", "cnn1d"]
    assert len(evaluation["validation"]["reseaux"]) == 2
    assert len(evaluation["absents"]) == 0
    nb_lignes = len(evaluation["modeles"]) + len(es.BASELINES)
    assert evaluation["ordre"][-5:] == es.BASELINES

    # tableaux de regression : une ligne par modele et baseline ; les metriques publiees se recalculent
    assert len(evaluation["regression"]["test_brute"]) == nb_lignes
    assert len(evaluation["regression"]["test_plafonnee"]) == nb_lignes
    assert evaluation["test"]["controle_contrat"]["coherent"].all()
    for cle in evaluation["ordre"]:
        assert len(evaluation["test"]["predictions"][cle]) == nb_moteurs_test

    # chaque modele appris bat Ridge et la reference naive en RMSE de test (contre la RUL plafonnee)
    plafonnee = evaluation["regression"]["test_plafonnee"].set_index("cle")
    for cle in evaluation["modeles"]:
        if cle != "reference_naive":
            assert plafonnee.loc[cle, "rmse"] < plafonnee.loc["Ridge", "rmse"] < plafonnee.loc["reference_naive", "rmse"]

    # classification : 3 horizons ; arbres et baselines ont deux regles de seuil, les reseaux une seule
    classification = evaluation["classification"]
    for cle in evaluation["ordre"]:
        nb_regles = len(classification[classification["cle"] == cle])
        if cle in inf.MODELES_RESEAUX:
            assert nb_regles == len(es.HORIZONS)
        else:
            assert nb_regles == 2 * len(es.HORIZONS)
    assert set(evaluation["seuils_matrices"].keys()) == set(evaluation["ordre"])

    # comparaisons appariees : toutes les paires des modeles disponibles et de Ridge
    nb_compares = len(evaluation["modeles"]) + 1
    nb_paires = nb_compares * (nb_compares - 1) // 2
    assert len(evaluation["comparaisons"]["test"]) == nb_paires
    assert len(evaluation["erreurs_par_moteur"]["test"]) == nb_moteurs_test

    # tranches de RUL : la tranche > 125 n'existe que sur le test (cible d'entrainement plafonnee)
    tranches = evaluation["tranches"]
    assert len(tranches[(tranches["ensemble"] == "validation") & (tranches["tranche"] == "> 125")]) == 0
    assert len(evaluation["non_fait"]) >= 3


def test_figures(monkeypatch, tmp_path):
    """(d) Chaque figure de la section 7.1 destinee a resultats/ se construit, avec et sans reseau."""
    for sous_jeu in ["FD001", "FD003"]:
        sauter_si_resultats_absents(sous_jeu)
        if sous_jeu == "FD001":
            evaluation = es.evaluer_sous_jeu(sous_jeu)  # avec ses deux reseaux
        else:
            # sans reseau (cas simule, voir `evaluation_sans_reseau`) : 4 modeles, reseaux absents par decision
            evaluation = evaluation_sans_reseau(sous_jeu, monkeypatch, tmp_path)
            assert evaluation["modeles"] == ["reference_naive", "random_forest", "xgboost_phase3", "lightgbm"]
            assert evaluation["absents"]["lstm"] == es.RAISON_SANS_RESEAU
            assert evaluation["absents"]["cnn1d"] == es.RAISON_SANS_RESEAU
            assert len(evaluation["validation"]["reseaux"]) == 0
            assert evaluation["validation"]["decoupages_neufs"] is None
        figures = []
        figures.append(fr.figure_predit_vs_vrai(evaluation))
        figures.append(fr.figure_predit_vs_vrai(evaluation, ensemble="validation"))
        figures.append(fr.figure_residus_selon_rul(evaluation))
        figures.append(fr.figure_barres_baselines(evaluation))
        figures.append(fr.figure_roc_precision_rappel(evaluation, horizon=30))
        figures.append(fr.figure_matrices_confusion(evaluation, horizon=30))
        figures.append(fr.figure_erreur_par_moteur(evaluation))
        for figure in figures:
            assert isinstance(figure, matplotlib.figure.Figure)
            assert "?" in figure.get_suptitle()  # le titre est une question (plan, section 2)
            plt.close(figure)

    # trajectoire d'un moteur : quelques cycles suffisent ici (le recalcul complet est teste en (a))
    verification = es.verifier_moteur("FD003", 42, inference=inference_reelle("FD003"), cycles=[1, 10, 20])
    # la simulation « sans reseau » ci-dessus ne touche ni l'inference reelle ni ses reseaux : la trajectoire
    # est tracee avec les 6 modeles, reseaux compris (le cas sans reseau de la section 7 est teste en (g))
    for cle in inf.MODELES_RESEAUX:
        assert cle in verification["modeles"], cle
    figure = fr.figure_trajectoire_moteur(verification)
    assert isinstance(figure, matplotlib.figure.Figure)
    assert "?" in figure.get_suptitle()
    plt.close(figure)


def test_complements_arbres():
    """(e) Erreur d'entrainement des arbres et decoupages neufs du 04h (scripts/valider_arbres_finaux.py --complements)."""
    sauter_si_resultats_absents("FD001")
    if not os.path.exists(es.FICHIER_CV_TRAIN_PAR_PLI) or not os.path.exists(es.FICHIER_CV_DECOUPAGES_NEUFS):
        pytest.skip(f"complements de la validation des arbres absents (lancer {es.COMMANDE_COMPLEMENTS})")
    for sous_jeu in ["FD001", "FD003"]:
        evaluation = es.evaluer_sous_jeu(sous_jeu)

        # erreur d'entrainement : chaque modele a arbres, 5 plis ; le RMSE de validation relu est celui recalcule
        # depuis les predictions hors-pli (controle fait par lire_complements_arbres), et l'arbre apprend mieux
        # ses moteurs d'entrainement que des moteurs jamais vus
        train = evaluation["validation"]["train_arbres"].set_index("cle")
        validation = evaluation["validation"]["tableau"].set_index("cle")
        for cle in ["random_forest", "xgboost_phase3", "lightgbm"]:
            assert train.loc[cle, "rmse_train_moyen"] < train.loc[cle, "rmse_val_moyen"], cle
            assert abs(train.loc[cle, "rmse_val_moyen"] - validation.loc[cle, "rmse_cv_moyen"]) < 1e-9, cle
        assert len(evaluation["validation"]["train_arbres_par_pli"]) == 4 * 5  # reference naive et 3 arbres

        # decoupages neufs : sur les sous-jeux qui ont des reseaux (les quatre depuis le 2026-11-08)
        assert sous_jeu in es.ENSEMBLES_RETENUS
        for cle in ["random_forest", "xgboost_phase3", "lightgbm", "lstm", "cnn1d"]:
            assert not pd.isna(validation.loc[cle, "rmse_decoupages_neufs_moyen"]), cle
        # pour les reseaux, c'est leur propre validation (3 decoupages neufs du 04h)
        assert validation.loc["lstm", "rmse_decoupages_neufs_moyen"] == validation.loc["lstm", "rmse_cv_moyen"]
        assert len(evaluation["validation"]["complements_manquants"]) == 0


def test_generateur_ecrit_le_sous_jeu_dans_les_parametres():
    """(f) Le script de generation recrit la cellule « parameters » : FD002.ipynb relance a la main reste FD002."""
    import importlib.util

    import nbformat

    chemin_script = os.path.join(RACINE_DEPOT, "scripts", "generer_notebooks_resultats.py")
    specification = importlib.util.spec_from_file_location("generer_notebooks_resultats", chemin_script)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)  # imports et constantes seulement (pas de main)

    # on lit le modele et on injecte FD002 ; la cellule de parametres doit le contenir, et elle seule change
    notebook = nbformat.read(module.CHEMIN_MODELE, as_version=4)
    sources_avant = []
    for cellule in notebook.cells:
        sources_avant.append(cellule.source)
    position = module.injecter_parametres(notebook, "FD002", 7, True)
    source = notebook.cells[position].source
    assert 'SOUS_JEU = "FD002"' in source
    assert "MOTEUR = 7 " in source
    assert "ENTRAINER = True " in source
    assert 'SOUS_JEU = "FD001"' not in source
    for numero in range(len(notebook.cells)):
        if numero != position:
            assert notebook.cells[numero].source == sources_avant[numero]
    # une figure par appel plt.show() du modele : le controle du nombre de figures en depend
    assert module.nb_figures_attendues(notebook) >= 9


# plt.show() du notebook avec le moteur Agg (sans fenetre) : avertissement attendu, sans objet ici
@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")
def test_modele_sans_reseau_s_execute(monkeypatch, tmp_path, capsys):
    """(g) Le modele des notebooks de resultats s'execute sur un sous-jeu sans reseau (FD003 simule)."""
    import IPython.display
    import nbformat

    sauter_si_resultats_absents("FD003")

    # (1) meme simulation qu'en (d) : predictions des reseaux absentes, FD003 retire de ENSEMBLES_RETENUS
    evaluation_sans_reseau("FD003", monkeypatch, tmp_path)

    # (2) l'inference en direct ne charge plus les reseaux : c'est ce qui arrive a un sous-jeu sans poids dans
    # models/04h/ (obtenir_reseau leve ValueError, le reseau passe dans « indisponibles »)
    charger_modeles_original = inf.charger_modeles

    def charger_modeles_sans_reseau(sous_jeu, *arguments, **options):
        """
        Appelle le vrai `charger_modeles`, puis retire les reseaux de l'objet d'inference.

        Parameters
        ----------
        sous_jeu : str
            "FD001" a "FD004".
        *arguments, **options
            Transmis tels quels a `src.inference.charger_modeles`.

        Returns
        -------
        dict
            L'objet d'inference, sans reseau dans « modeles » ; chaque reseau retire est dans « indisponibles ».
        """
        inference = charger_modeles_original(sous_jeu, *arguments, **options)
        for cle in inf.MODELES_RESEAUX:
            if cle in inference["modeles"]:
                del inference["modeles"][cle]  # le reseau n'est plus disponible
                inference["indisponibles"][cle] = "poids absents (cas simulé par le test)"
        return inference

    monkeypatch.setattr(inf, "charger_modeles", charger_modeles_sans_reseau)

    # (3) on garde tout ce que le notebook affiche avec display(...) (tableaux et textes Markdown)
    affichages = []

    def garder_affichage(*objets, **options):
        """
        Remplace IPython.display.display : garde les objets affiches au lieu de les montrer.

        Parameters
        ----------
        *objets
            Les objets passes a display.
        **options
            Ignorees.
        """
        for objet in objets:
            affichages.append(objet)

    monkeypatch.setattr(IPython.display, "display", garder_affichage)

    # (4) parametres du notebook, et dossier de travail attendu par le notebook (il cherche src/ au-dessus)
    monkeypatch.setenv("SOUS_JEU", "FD003")
    monkeypatch.setenv("ENTRAINER", "0")
    monkeypatch.chdir(os.path.join(RACINE_DEPOT, "notebooks", "resultats"))

    # (5) execution des cellules de code, dans l'ordre, dans un meme espace de noms (comme un noyau Jupyter)
    notebook = nbformat.read(os.path.join(RACINE_DEPOT, "notebooks", "resultats", "_modele.ipynb"), as_version=4)
    espace = {"__name__": "__main__"}
    for numero in range(len(notebook.cells)):
        cellule = notebook.cells[numero]
        if cellule.cell_type != "code":
            continue  # les cellules Markdown ne s'executent pas
        lignes = []
        for ligne in cellule.source.splitlines():
            if ligne.strip().startswith("%"):
                lignes.append("pass")  # commande magique de Jupyter (%matplotlib inline) : sans objet ici
            else:
                lignes.append(ligne)
        code = compile("\n".join(lignes), f"_modele.ipynb, cellule {numero}", "exec")
        exec(code, espace)  # une erreur dans une cellule fait echouer le test, avec le numero de la cellule
        plt.close("all")  # les figures sont seulement construites

    # (6) les branches « sans reseau » ont bien ete parcourues
    sortie = capsys.readouterr().out
    assert f"FD003 : {es.RAISON_SANS_RESEAU}." in sortie  # cellule des reseaux (section 2)
    textes_markdown = []
    for objet in affichages:
        if isinstance(objet, IPython.display.Markdown):
            textes_markdown.append(objet.data)
    tout_le_markdown = "\n".join(textes_markdown)
    assert "Pas de réseau sur FD003" in tout_le_markdown  # phrase de la validation (section 3)
    assert "Les réseaux de FD003" not in tout_le_markdown  # pas de verdict des regles du 04h sans reseau
    assert "aucun réseau ne fait mieux" not in tout_le_markdown  # pas de phrase reseaux contre arbres (section 4)
    assert "le meilleur réseau" not in tout_le_markdown
    # la section 7 recalcule et compare les 4 modeles restants, sans les reseaux
    assert "lstm" not in espace["verification"]["modeles"]
    assert len(espace["verification"]["modeles"]) == 4
