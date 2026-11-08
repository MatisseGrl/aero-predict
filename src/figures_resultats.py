"""
Figures des notebooks de resultats (notebooks/resultats/FD00X.ipynb), une fonction matplotlib par figure.

Pourquoi ce module ? (plan de reorganisation, docs/plan_refonte_notebooks.md, sections 2, 3 et 7.1)
Les 4 notebooks de resultats sont generes depuis un seul modele : chaque figure doit donc etre tracee par le
meme code pour les 4 sous-jeux. Chaque fonction reprend le code de la cellule d'origine (notebooks 04a, 04b, 05,
01b, 03b, 04g), applique aux objets renvoyes par src/evaluation_sous_jeu.py, et chaque figure a pour titre la
question a laquelle elle repond.

| Fonction | Figure (plan, section 7.1) | Cellule d'origine | Section du notebook |
|---|---|---|---|
| `figure_predit_vs_vrai` | RUL predite vs vraie, une case par modele | 04a (7), 05 D.3, 03b F.6 | 4 |
| `figure_residus_selon_rul` | Residus selon la RUL vraie | 04a (7) | 4 |
| `figure_barres_baselines` | Barres RMSE / score NASA contre baselines | 01b F.3, 05 B.3 | 4 |
| `figure_roc_precision_rappel` | ROC et precision-rappel, deux panneaux | 04b (7) | 5 |
| `figure_matrices_confusion` | Matrices de confusion, une par modele | 04b (7) | 5 |
| `figure_erreur_par_moteur` | Erreur par moteur, comparaisons appariees | 05 B.2 / G.6 (tableaux) | 6 |
| `figure_trajectoire_moteur` | RUL predite cycle par cycle | 05 J.3 / J.4, 03b F.7, 04g | 7 |

Chaque fonction renvoie la figure (matplotlib.figure.Figure) sans l'afficher : le notebook l'affiche. Si un
`chemin` est donne, la figure y est aussi enregistree ; jamais par defaut dans docs/figures/ (plan, section 6.4 :
les PNG des docs de phase ne doivent pas etre ecrases).
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from src.evaluation import rmse, score_nasa

# ---------------------------------------------------------------------------------------------------
# Couleurs et reperes communs
# ---------------------------------------------------------------------------------------------------

# une couleur fixe par modele : la meme que la demo (recopiee de app/figures.py, COULEURS_MODELES, pour que les
# notebooks ne dependent pas de l'application ; a changer aux deux endroits si besoin)
COULEURS_MODELES = {
    "reference_naive": "#A6761D",  # brun
    "random_forest": "#0072B2",  # bleu fonce
    "xgboost_phase3": "#56B4E9",  # bleu ciel
    "lightgbm": "#E69F00",  # orange
    "lstm": "#8E44AD",  # violet
    "cnn1d": "#16A085",  # bleu-vert
}
COULEUR_BASELINE = "#B0B0B0"  # gris clair : les baselines du protocole
COULEUR_ALEATOIRE = "#7F7F7F"  # gris : la baseline aleatoire sur les courbes ROC et precision-rappel
COULEUR_VERITE = "#111111"  # RUL vraie : noir (comme la demo)
COULEUR_VERITE_PLAFONNEE = "#7F7F7F"  # RUL vraie plafonnee : gris, en pointilles
COULEUR_DANGER = "#D62728"  # rouge : reserve au danger (surestimation, zone d'alerte)
COULEUR_TEXTE = "#52514e"  # gris fonce (reperes, annotations), comme le notebook 05

PLAFOND_RUL = 125
SEUIL_SURESTIMATION = 20


def couleur(cle):
    """
    Couleur d'un modele ou d'une baseline.

    Parameters
    ----------
    cle : str
        Cle d'un modele (ex. "lstm") ou nom d'une baseline (ex. "Ridge").

    Returns
    -------
    str
        Couleur hexadecimale (gris clair pour une baseline).
    """
    return COULEURS_MODELES.get(cle, COULEUR_BASELINE)


def grille_panneaux(nb_panneaux, largeur_panneau=5.5, hauteur_panneau=5.0, nb_colonnes_max=3, partager=True):
    """
    Cree une grille de panneaux (au plus `nb_colonnes_max` par ligne) et renvoie la liste des axes utiles.

    Les panneaux en trop (grille incomplete) sont masques.

    Parameters
    ----------
    nb_panneaux : int
        Nombre de panneaux voulus (ex. 6 modeles sur FD001, 4 sur FD003).
    largeur_panneau, hauteur_panneau : float, optionnel
        Taille d'un panneau, en pouces.
    nb_colonnes_max : int, optionnel
        Nombre maximal de panneaux par ligne (3 par defaut).
    partager : bool, optionnel
        Vrai : memes axes pour tous les panneaux (comparaison directe).

    Returns
    -------
    (matplotlib.figure.Figure, list)
        La figure et les `nb_panneaux` axes, dans l'ordre de lecture.
    """
    nb_colonnes = min(nb_panneaux, nb_colonnes_max)
    nb_lignes = int(np.ceil(nb_panneaux / nb_colonnes))
    fig, axes = plt.subplots(nb_lignes, nb_colonnes, figsize=(largeur_panneau * nb_colonnes, hauteur_panneau * nb_lignes),
                             sharex=partager, sharey=partager, squeeze=False)
    liste_axes = []
    for ax in axes.flat:  # ravel : les panneaux a la suite, ligne par ligne
        liste_axes.append(ax)
    for ax in liste_axes[nb_panneaux:]:  # panneaux en trop : masques
        ax.set_visible(False)
    for ax in liste_axes[:nb_panneaux]:
        # axes partages : matplotlib n'ecrit les graduations que sur la derniere ligne ; si un panneau de cette
        # ligne est masque, le panneau du dessus perdrait les siennes : on les remet partout
        ax.tick_params(labelbottom=True, labelleft=True)
    return fig, liste_axes[:nb_panneaux]


def enregistrer(fig, chemin):
    """
    Enregistre la figure si un chemin est donne (rien sinon).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        La figure.
    chemin : str ou None
        Le fichier PNG voulu ; None : rien n'est ecrit.
    """
    if chemin is not None:
        fig.savefig(chemin, dpi=120, bbox_inches="tight")


def donnees_ensemble(evaluation, cle, ensemble):
    """
    Verite et predictions d'un modele, en validation (hors-pli) ou sur le test officiel.

    Parameters
    ----------
    evaluation : dict
        Le resultat de `evaluer_sous_jeu`.
    cle : str
        Le modele.
    ensemble : str
        "test" (une prediction par moteur test, RUL brute) ou "validation" (predictions hors-pli, RUL plafonnee).

    Returns
    -------
    (numpy.ndarray, numpy.ndarray) ou (None, None)
        (vrais, predits), ou (None, None) si le modele n'a pas de predictions hors-pli (reseaux).
    """
    if ensemble == "test":
        return evaluation["test"]["rul_vraie"], evaluation["test"]["predictions"][cle]
    if ensemble == "validation":
        if cle not in evaluation["validation"]["oof"]:
            return None, None
        return evaluation["validation"]["y"], evaluation["validation"]["oof"][cle]
    raise ValueError(f"ensemble inconnu : {ensemble!r} ('test' ou 'validation')")


# ---------------------------------------------------------------------------------------------------
# Section 4 : test officiel
# ---------------------------------------------------------------------------------------------------

def figure_predit_vs_vrai(evaluation, ensemble="test", modeles=None, chemin=None):
    """
    RUL predite contre RUL vraie, une case par modele (notebook 04a section 7, notebook 05 D.3).

    Diagonale = prediction parfaite ; au-dessus = surestimation (dangereux). Les moteurs surestimes de plus de
    20 cycles sont en triangles rouges ; la ligne horizontale rappelle le plafond de 125 (aucun modele n'a appris a
    predire au-dela, d'ou l'erreur sur les moteurs test de vraie RUL > 125).

    Parameters
    ----------
    evaluation : dict
        Le resultat de `evaluer_sous_jeu`.
    ensemble : str, optionnel
        "test" (defaut : un point par moteur test) ou "validation" (predictions hors-pli des modeles qui en ont).
    modeles : list de str ou None, optionnel
        Les modeles tracés (None : les modeles disponibles du sous-jeu).
    chemin : str ou None, optionnel
        Fichier PNG ou enregistrer la figure.

    Returns
    -------
    matplotlib.figure.Figure
        La figure.
    """
    if modeles is None:
        modeles = evaluation["modeles"]
    # on ne garde que les modeles qui ont des predictions pour cet ensemble (pas de hors-pli pour les reseaux)
    modeles_traces = []
    for cle in modeles:
        vrais, predits = donnees_ensemble(evaluation, cle, ensemble)
        if vrais is not None:
            modeles_traces.append(cle)

    fig, axes = grille_panneaux(len(modeles_traces))
    # bord des axes : la plus grande RUL (vraie ou predite) des modeles traces, plus 10 cycles de marge
    limite = 0.0
    for cle in modeles_traces:
        vrais, predits = donnees_ensemble(evaluation, cle, ensemble)
        limite = max(limite, float(np.max(vrais)), float(np.max(predits)))
    limite = limite + 10

    for ax, cle in zip(axes, modeles_traces):  # un panneau par modele
        vrais, predits = donnees_ensemble(evaluation, cle, ensemble)
        surestime = (predits - vrais) > SEUIL_SURESTIMATION  # Vrai pour les surestimations dangereuses
        if ensemble == "test":
            taille_point = 28
            transparence = 0.6
        else:
            taille_point = 4  # environ 20 000 points en validation : petits et transparents
            transparence = 0.15
        ax.scatter(vrais[~surestime], predits[~surestime], color=couleur(cle), alpha=transparence, s=taille_point,
                   label="prédictions")  # ~ : "non"
        ax.scatter(vrais[surestime], predits[surestime], color=COULEUR_DANGER, marker="^", s=taille_point * 2,
                   alpha=min(1.0, transparence + 0.3), label=f"surestimés > {SEUIL_SURESTIMATION} ({int(surestime.sum())})")
        ax.plot([0, limite], [0, limite], linestyle="--", color="black", linewidth=1, label="prédiction = vérité")
        ax.plot([0, limite], [SEUIL_SURESTIMATION, limite + SEUIL_SURESTIMATION], linestyle=":", color=COULEUR_TEXTE,
                label=f"vérité + {SEUIL_SURESTIMATION}")
        ax.axhline(PLAFOND_RUL, color=COULEUR_TEXTE, linestyle="-.", linewidth=0.8)
        if ensemble == "test":
            # test : une prediction par moteur ; le score NASA est la somme des penalites (definition officielle)
            texte_metriques = f"RMSE = {rmse(vrais, predits):.2f}, score NASA = {score_nasa(vrais, predits):.1f}"
        else:
            # validation : toutes les lignes hors-pli regroupees (ce RMSE n'est pas la moyenne des 5 plis du tableau
            # de la section 3) ; score NASA ramene a une prediction, la somme sur ~20 000 lignes n'etant pas lisible
            texte_metriques = (f"RMSE hors-pli regroupé = {rmse(vrais, predits):.2f}, NASA moyen = "
                               f"{score_nasa(vrais, predits) / len(vrais):.2f}")
        ax.set_title(f"{evaluation['noms'][cle]}\n{texte_metriques}, surestimés > 20 : {int(surestime.sum())}/{len(vrais)}",
                     fontsize=10)
        ax.set_xlim(0, limite)
        ax.set_ylim(0, limite)
        ax.set_xlabel("RUL vraie (cycles)")
        ax.set_ylabel("RUL prédite (cycles)")
        ax.legend(fontsize=8, loc="upper left")

    if ensemble == "test":
        texte_ensemble = f"test officiel, {len(evaluation['test']['rul_vraie'])} moteurs, dernier cycle observé"
    else:
        texte_ensemble = "validation croisée, prédictions hors-pli des 5 plis"
    fig.suptitle(f"{evaluation['sous_jeu']} — Le modèle prédit-il la bonne RUL ? ({texte_ensemble})")
    fig.tight_layout()
    enregistrer(fig, chemin)
    return fig


def moyenne_par_paquet(x, valeurs, largeur=10):
    """
    Moyenne de `valeurs` par paquet de `largeur` cycles de x (pour tracer la tendance des residus, notebook 04a).

    Parameters
    ----------
    x : numpy.ndarray
        La RUL vraie de chaque point.
    valeurs : numpy.ndarray
        Le residu de chaque point.
    largeur : int, optionnel
        Largeur d'un paquet, en cycles (10 par defaut).

    Returns
    -------
    (list, list)
        Les centres des paquets et la moyenne des residus de chaque paquet (paquets de moins de 5 points ignores).
    """
    centres = []
    moyennes = []
    for debut in range(0, int(np.max(x)) + 1, largeur):  # 0-9, 10-19, ...
        masque = (x >= debut) & (x < debut + largeur)
        if masque.sum() >= 5:  # on ignore les paquets presque vides
            centres.append(debut + largeur / 2)
            moyennes.append(np.mean(valeurs[masque]))
    return centres, moyennes


def figure_residus_selon_rul(evaluation, ensemble="test", modeles=None, chemin=None):
    """
    Residus (predit - vrai) selon la RUL vraie, une case par modele (notebook 04a, section 7, figure 2).

    Montre si les modeles se trompent surtout pres de la panne (zone de decision) ou loin d'elle (effet du
    plafond). Ligne rouge : moyenne des residus par paquet de 10 cycles ; > 0 = surestimation.

    Parameters
    ----------
    evaluation : dict
        Le resultat de `evaluer_sous_jeu`.
    ensemble : str, optionnel
        "test" (defaut) ou "validation".
    modeles : list de str ou None, optionnel
        Les modeles traces (None : les modeles disponibles sans la reference naive, dont les residus vont de -50 a
        +90 cycles et ecraseraient l'echelle commune des autres panneaux ; la passer explicitement pour la voir).
    chemin : str ou None, optionnel
        Fichier PNG ou enregistrer la figure.

    Returns
    -------
    matplotlib.figure.Figure
        La figure.
    """
    if modeles is None:
        modeles = []
        for cle in evaluation["modeles"]:
            if cle != "reference_naive":  # hors echelle : elle predit la meme valeur pour tous les moteurs
                modeles.append(cle)
    modeles_traces = []
    for cle in modeles:
        vrais, predits = donnees_ensemble(evaluation, cle, ensemble)
        if vrais is not None:
            modeles_traces.append(cle)

    fig, axes = grille_panneaux(len(modeles_traces), hauteur_panneau=4.5)
    for ax, cle in zip(axes, modeles_traces):
        vrais, predits = donnees_ensemble(evaluation, cle, ensemble)
        residus = predits - vrais  # positif = surestimation
        if ensemble == "test":
            ax.scatter(vrais, residus, color=couleur(cle), alpha=0.7, s=24)
        else:
            ax.scatter(vrais, residus, color=couleur(cle), alpha=0.15, s=4)
        centres, moyennes = moyenne_par_paquet(vrais, residus)
        ax.plot(centres, moyennes, color=COULEUR_DANGER, marker="o", markersize=3, label="moyenne par paquet de 10 cycles")
        ax.axhline(0, color="black", linestyle="--")
        ax.axhline(SEUIL_SURESTIMATION, color="grey", linestyle=":", label="surestimation de 20 cycles")
        ax.axvline(PLAFOND_RUL, color=COULEUR_TEXTE, linestyle="-.", linewidth=0.8)
        ax.set_title(evaluation["noms"][cle], fontsize=10)
        ax.set_xlabel("RUL vraie (cycles)")
        ax.set_ylabel("Résidu = prédit − vrai (cycles)")
        ax.legend(fontsize=8, loc="lower left")
    if ensemble == "test":
        texte_ensemble = "test officiel"
    else:
        texte_ensemble = "validation croisée"
    fig.suptitle(f"{evaluation['sous_jeu']} — Où les modèles se trompent-ils, selon la distance à la panne ? ({texte_ensemble})")
    fig.tight_layout()
    enregistrer(fig, chemin)
    return fig


def figure_barres_baselines(evaluation, chemin=None):
    """
    RMSE et score NASA du test, modeles contre les 5 baselines (notebooks 01b F.3 et 05 B.3).

    Un panneau par critere (jamais deux echelles sur le meme axe). Panneau RMSE : barre = contre la RUL brute,
    losange = contre la RUL plafonnee a 125, tiret = plancher du a la seule RUL plafonnee (un modele parfait sous
    125 ne descend pas plus bas contre la RUL brute). Panneau score NASA : echelle logarithmique, la reference
    naive et les baselines constantes depassant de plusieurs ordres de grandeur les modeles appris.

    Parameters
    ----------
    evaluation : dict
        Le resultat de `evaluer_sous_jeu`.
    chemin : str ou None, optionnel
        Fichier PNG ou enregistrer la figure.

    Returns
    -------
    matplotlib.figure.Figure
        La figure.
    """
    ordre = evaluation["ordre"]  # modeles disponibles puis baselines
    brute = evaluation["regression"]["test_brute"].set_index("cle")
    plafonnee = evaluation["regression"]["test_plafonnee"].set_index("cle")
    positions = np.arange(len(ordre))
    etiquettes = []
    for cle in ordre:
        etiquettes.append(evaluation["noms"][cle])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # (1) RMSE test : barre = RUL brute ; losange = RUL plafonnee ; tiret = plancher du plafond
    ax = axes[0]
    for position, cle in zip(positions, ordre):
        valeur = float(brute.loc[cle, "rmse"])
        ax.bar(position, valeur, width=0.6, color=couleur(cle), alpha=0.9)
        ax.scatter(position, float(plafonnee.loc[cle, "rmse"]), marker="D", s=40, color="white", edgecolors="black",
                   zorder=3)
        ax.text(position, valeur + 0.8, f"{valeur:.1f}", ha="center", fontsize=8)
    plancher = evaluation["donnees"]["plancher_rmse_rul_brute"]
    ax.axhline(plancher, color="black", linestyle="--", linewidth=1.2,
               label=f"plancher dû au plafond de 125 ({plancher:.2f})")
    ax.scatter([], [], marker="D", s=40, color="white", edgecolors="black", label="RMSE contre la RUL plafonnée à 125")
    ax.set_ylabel("RMSE test (cycles), contre la RUL brute")
    ax.set_title("RMSE du test officiel (plus bas = mieux)", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")

    # (2) score NASA du test, echelle logarithmique
    ax = axes[1]
    for position, cle in zip(positions, ordre):
        valeur = float(brute.loc[cle, "nasa_somme"])
        ax.bar(position, valeur, width=0.6, color=couleur(cle), alpha=0.9)
        ax.text(position, valeur * 1.15, f"{valeur:,.0f}".replace(",", " "), ha="center", fontsize=8)  # milliers separes
    ax.set_yscale("log")
    ax.set_ylabel("score NASA (13/10), somme sur les moteurs test — échelle log")
    ax.set_title("Score NASA du test officiel (plus bas = mieux)", fontsize=10)

    for ax in axes:
        ax.set_xticks(positions)
        ax.set_xticklabels(etiquettes, rotation=35, ha="right", fontsize=9)
    fig.suptitle(f"{evaluation['sous_jeu']} — Nos modèles font-ils mieux que des modèles volontairement simples ? "
                 "(couleur : modèles ; gris : baselines)")
    fig.tight_layout()
    enregistrer(fig, chemin)
    return fig


# ---------------------------------------------------------------------------------------------------
# Section 5 : classification
# ---------------------------------------------------------------------------------------------------

def figure_roc_precision_rappel(evaluation, horizon=30, modeles=None, chemin=None):
    """
    Courbes ROC et precision-rappel sur le test, modeles superposes, baseline aleatoire comprise (notebook 04b).

    Score de classement : -RUL predite (plus la RUL predite est basse, plus le moteur est juge urgent). Au test,
    peu de moteurs sont proches de la panne : la ROC est en escalier et son AUC parait souvent tres bonne ; la
    courbe precision-rappel (ligne du hasard = proportion de positifs) montre le cout en precision de chaque
    niveau de rappel (plan, note de la section 5.3). D'ou les deux panneaux cote a cote.

    Parameters
    ----------
    evaluation : dict
        Le resultat de `evaluer_sous_jeu`.
    horizon : int, optionnel
        N : classe positive = RUL vraie <= N (30 par defaut).
    modeles : list de str ou None, optionnel
        Les modeles traces (None : les modeles disponibles, sans la reference naive, constante : une seule
        valeur de score, donc une ROC diagonale ; la baseline aleatoire est toujours ajoutee).
    chemin : str ou None, optionnel
        Fichier PNG ou enregistrer la figure.

    Returns
    -------
    matplotlib.figure.Figure
        La figure.
    """
    if modeles is None:
        modeles = []
        for cle in evaluation["modeles"]:
            if cle != "reference_naive":
                modeles.append(cle)
    vrais = evaluation["test"]["rul_vraie"]
    etiquettes = (vrais <= horizon).astype(int)  # 1 = le moteur tombe en panne dans les N cycles
    proportion = float(np.mean(etiquettes))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    traces = []
    for cle in modeles:
        traces.append(cle)
    traces.append("Aleatoire uniforme")  # la baseline aleatoire, toujours sur le meme graphique (protocole §9)
    for cle in traces:
        scores = -evaluation["test"]["predictions"][cle]  # plus la RUL predite est basse, plus le score est haut
        if cle == "Aleatoire uniforme":
            couleur_courbe = COULEUR_ALEATOIRE
            style = ":"
        else:
            couleur_courbe = couleur(cle)
            style = "-"
        nom = evaluation["noms"][cle]
        taux_fp, taux_vp, _ = roc_curve(etiquettes, scores)
        auc = roc_auc_score(etiquettes, scores)
        axes[0].plot(taux_fp, taux_vp, color=couleur_courbe, linestyle=style, label=f"{nom} (AUC={auc:.3f})")
        precisions, rappels, _ = precision_recall_curve(etiquettes, scores)
        ap = average_precision_score(etiquettes, scores)
        axes[1].plot(rappels, precisions, color=couleur_courbe, linestyle=style, label=f"{nom} (AP={ap:.3f})")

    axes[0].plot([0, 1], [0, 1], linestyle=":", color="black", linewidth=0.8)  # diagonale : hasard theorique
    axes[0].set_xlabel("taux de faux positifs")
    axes[0].set_ylabel("taux de vrais positifs (rappel)")
    axes[0].set_title(f"Courbe ROC — N = {horizon}")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[1].axhline(proportion, linestyle=":", color="black", linewidth=0.8, label=f"hasard = % de positifs ({100 * proportion:.0f} %)")
    axes[1].set_xlabel("rappel")
    axes[1].set_ylabel("précision")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title(f"Courbe précision-rappel — N = {horizon}")
    axes[1].legend(loc="lower left", fontsize=8)
    fig.suptitle(f"{evaluation['sous_jeu']} — Les modèles repèrent-ils les moteurs qui tomberont en panne dans les "
                 f"{horizon} prochains cycles ? (test, {int(etiquettes.sum())} positifs sur {len(etiquettes)})")
    fig.tight_layout()
    enregistrer(fig, chemin)
    return fig


def figure_matrices_confusion(evaluation, horizon=30, modeles=None, chemin=None):
    """
    Matrices de confusion du test, une par modele, a un seul N (notebook 04b, section 7).

    Seuil : celui choisi sur la validation (rappel >= 0,95, meilleure precision) quand le modele a des predictions
    hors-pli, sinon le seuil par defaut N (reseaux). Lignes = vrai, colonnes = predit.

    Parameters
    ----------
    evaluation : dict
        Le resultat de `evaluer_sous_jeu`.
    horizon : int, optionnel
        N (30 par defaut, valeur des seuils de `evaluation["seuils_matrices"]`).
    modeles : list de str ou None, optionnel
        Les modeles (None : les modeles disponibles).
    chemin : str ou None, optionnel
        Fichier PNG ou enregistrer la figure.

    Returns
    -------
    matplotlib.figure.Figure
        La figure.
    """
    if modeles is None:
        modeles = evaluation["modeles"]
    classification = evaluation["classification"]
    fig, axes = grille_panneaux(len(modeles), largeur_panneau=5.2, hauteur_panneau=4.6, partager=False)
    noms_cases = [["Vrai négatif", "Faux positif\n(retiré trop tôt)"], ["Faux négatif\n(laissé voler : grave)", "Vrai positif"]]
    for ax, cle in zip(axes, modeles):
        # seuil retenu pour ce modele a cet horizon : choisi sur la validation s'il existe, sinon N
        lignes = classification[(classification["cle"] == cle) & (classification["N"] == horizon)]
        choisies = lignes[lignes["regle_seuil"] == "choisi sur validation"]
        if len(choisies) > 0:
            ligne = choisies.iloc[0]
        else:
            ligne = lignes[lignes["regle_seuil"] == "defaut (= N)"].iloc[0]
        matrice = np.array([[ligne["test_vn"], ligne["test_fp"]], [ligne["test_fn"], ligne["test_vp"]]], dtype=int)
        ax.imshow(matrice, cmap="Blues")
        for i in range(2):
            for j in range(2):
                couleur_texte = "black"
                if matrice[i, j] > matrice.max() / 2:
                    couleur_texte = "white"
                ax.text(j, i, f"{matrice[i, j]}\n{noms_cases[i][j]}", ha="center", va="center", color=couleur_texte, fontsize=9)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["prédit : pas d'alerte", "prédit : alerte"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["vrai : RUL > N", "vrai : RUL ≤ N"])
        ax.set_title(f"{evaluation['noms'][cle]} — alerte si RUL prédite ≤ {ligne['seuil']:.1f}\n({ligne['regle_seuil']} ; "
                     f"rappel {ligne['test_rappel']:.2f}, précision {ligne['test_precision']:.2f})", fontsize=9)
    fig.suptitle(f"{evaluation['sous_jeu']} — Combien d'alertes manquées et de fausses alertes ? (test, N = {horizon})")
    fig.tight_layout()
    enregistrer(fig, chemin)
    return fig


# ---------------------------------------------------------------------------------------------------
# Section 6 : comparaison des modeles, moteur par moteur
# ---------------------------------------------------------------------------------------------------

def figure_erreur_par_moteur(evaluation, modeles=None, chemin=None):
    """
    Erreur absolue de chaque moteur test, par modele, avec les p-valeurs des comparaisons appariees.

    Donnees des tableaux des notebooks 05 (B.2, G.6) : erreur absolue au dernier cycle de chaque moteur test, et
    test de Wilcoxon apparie (Bonferroni) entre chaque modele et le modele de plus petite erreur moyenne. Les
    points d'un meme moteur sont relies d'un modele a l'autre (gris tres clair) : c'est sur ces differences,
    moteur par moteur, que porte le test, pas sur les deux moyennes.

    Parameters
    ----------
    evaluation : dict
        Le resultat de `evaluer_sous_jeu`.
    modeles : list de str ou None, optionnel
        Les modeles (None : les modeles disponibles sans la reference naive, plus Ridge, comme la section 6).
    chemin : str ou None, optionnel
        Fichier PNG ou enregistrer la figure.

    Returns
    -------
    matplotlib.figure.Figure
        La figure.
    """
    erreurs = evaluation["erreurs_par_moteur"]["test"]  # une ligne par moteur test, une colonne par modele
    if modeles is None:
        modeles = []
        for cle in evaluation["modeles"]:
            if cle != "reference_naive":  # hors echelle (erreur de 40 cycles environ) : citee dans le titre
                modeles.append(cle)
        modeles.append("Ridge")

    # modele de reference des p-valeurs : la plus petite erreur absolue moyenne
    reference = modeles[0]
    for cle in modeles:
        if erreurs[cle].mean() < erreurs[reference].mean():
            reference = cle
    comparaisons = evaluation["comparaisons"]["test"]
    nom_reference = evaluation["noms"][reference]

    fig, ax = plt.subplots(figsize=(max(9, 1.8 * len(modeles)), 6))
    positions = np.arange(len(modeles))
    erreur_max = float(np.max(erreurs[modeles].to_numpy()))  # plus grande erreur tracee
    hauteur_textes = erreur_max * 1.03  # les textes (moyenne, p-valeur) juste au-dessus du plus haut point
    # liens moteur par moteur entre modeles voisins (gris tres clair)
    for indice in erreurs.index:
        valeurs = []
        for cle in modeles:
            valeurs.append(erreurs.loc[indice, cle])
        ax.plot(positions, valeurs, color="#E0E0E0", linewidth=0.5, zorder=1)
    generateur = np.random.default_rng(0)  # leger decalage horizontal des points, fixe, pour les voir tous
    for position, cle in zip(positions, modeles):
        valeurs = erreurs[cle].to_numpy(dtype=float)
        decalages = generateur.uniform(-0.12, 0.12, size=len(valeurs))
        ax.scatter(position + decalages, valeurs, color=couleur(cle), s=14, alpha=0.7, zorder=2)
        ax.boxplot(valeurs, positions=[position], widths=0.45, showfliers=False, zorder=3,
                   medianprops={"color": "black"}, boxprops={"color": COULEUR_TEXTE}, whiskerprops={"color": COULEUR_TEXTE},
                   capprops={"color": COULEUR_TEXTE})
        # p-valeur corrigee contre la reference (ligne de la paire dans le tableau des comparaisons)
        nom = evaluation["noms"][cle]
        if cle == reference:
            texte = f"moyenne {np.mean(valeurs):.2f}\n(référence)"
        else:
            paire = comparaisons[((comparaisons["modele_a"] == nom) & (comparaisons["modele_b"] == nom_reference))
                                 | ((comparaisons["modele_a"] == nom_reference) & (comparaisons["modele_b"] == nom))]
            p_corrigee = float(paire["p_wilcoxon_bonferroni"].iloc[0])
            texte = f"moyenne {np.mean(valeurs):.2f}\np corrigée = {p_corrigee:.2g}"
        ax.text(position, hauteur_textes, texte, ha="center", va="bottom", fontsize=8)  # au-dessus de tous les points
    etiquettes = []
    for cle in modeles:
        etiquettes.append(evaluation["noms"][cle])
    ax.set_xticks(positions)
    ax.set_xticklabels(etiquettes, rotation=20, ha="right")
    ax.set_ylabel("erreur absolue au dernier cycle (cycles)")
    ax.set_ylim(0, erreur_max * 1.18)
    texte_naive = ""
    if "reference_naive" in erreurs.columns and "reference_naive" not in modeles:
        texte_naive = f" ; référence naïve hors échelle : moyenne {erreurs['reference_naive'].mean():.1f}"
    ax.set_title(f"Wilcoxon apparié contre {nom_reference}, p-valeurs × {int(comparaisons['nb_comparaisons'].iloc[0])} "
                 f"(Bonferroni){texte_naive}", fontsize=9)
    fig.suptitle(f"{evaluation['sous_jeu']} — Les écarts entre modèles sont-ils réels, moteur par moteur ? "
                 f"(test, {len(erreurs)} moteurs)")
    fig.tight_layout()
    enregistrer(fig, chemin)
    return fig


# ---------------------------------------------------------------------------------------------------
# Section 7 : verifier un moteur
# ---------------------------------------------------------------------------------------------------

def figure_trajectoire_moteur(verification, montrer_verite=True, chemin=None):
    """
    RUL predite a chaque cycle par chaque modele, recalculee depuis les donnees brutes (notebook 05 J.3 / J.4).

    La meme courbe que la demo : une courbe par modele ; sous la ligne pointillee rouge (seuil d'alerte), le
    modele declencherait une alerte maintenance. La RUL vraie (et sa version plafonnee, la cible apprise) est
    ajoutee si `montrer_verite` est vrai (revelation, J.4).

    Parameters
    ----------
    verification : dict
        Le resultat de `verifier_moteur`.
    montrer_verite : bool, optionnel
        Vrai (defaut) : la RUL vraie est tracee.
    chemin : str ou None, optionnel
        Fichier PNG ou enregistrer la figure.

    Returns
    -------
    matplotlib.figure.Figure
        La figure.
    """
    from src.evaluation_sous_jeu import nom_affiche  # import local : evite de charger PyTorch pour les autres figures

    trajectoire = verification["trajectoire"]
    seuil = verification["seuil_alerte"]
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.axhspan(0, seuil, color=COULEUR_DANGER, alpha=0.08, linewidth=0)  # zone d'alerte (comme la demo et le 04g)
    ax.axhline(seuil, color=COULEUR_DANGER, linestyle="--", linewidth=1.2, label=f"seuil d'alerte ({seuil:.0f} cycles)")
    for cle in verification["modeles"]:  # une courbe et une couleur par modele
        ax.plot(trajectoire["time_cycles"], trajectoire[cle], color=couleur(cle), linewidth=2,
                label=f"RUL prédite — {nom_affiche(cle)}")
    if montrer_verite:
        ax.plot(trajectoire["time_cycles"], trajectoire["rul_vraie"], color=COULEUR_VERITE, linewidth=2.5,
                label=f"RUL vraie (RUL_{verification['sous_jeu']}.txt)")
        ax.plot(trajectoire["time_cycles"], trajectoire["rul_vraie_cappee"], color=COULEUR_VERITE_PLAFONNEE,
                linestyle=":", linewidth=2, label="RUL vraie plafonnée à 125 (la cible apprise)")
    ax.set_ylim(bottom=0)  # l'axe vertical part de 0 cycle restant
    ax.set_xlabel("cycle (vol) observé : à chaque cycle t, les modèles ne voient que les mesures 1 à t")
    ax.set_ylabel("RUL (cycles restants)")
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1.0))  # legende a droite, hors des courbes
    ax.grid(color="#E5E5E5")
    fig.suptitle(f"{verification['sous_jeu']}, moteur test n° {verification['moteur']} — Que prédit chaque modèle, "
                 f"cycle après cycle ? (recalculé depuis test_{verification['sous_jeu']}.txt)")
    fig.tight_layout()
    enregistrer(fig, chemin)
    return fig
