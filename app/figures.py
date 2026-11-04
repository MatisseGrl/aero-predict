"""
Figures Plotly du tableau de bord de demo (aucune dependance a Streamlit : testables seules).

Regles de lisibilite, pensees pour un videoprojecteur et un jury :
- une couleur FIXE par modele, la meme sur toutes les pages (COULEURS_MODELES) ;
- le ROUGE est reserve au danger : zone d'alerte, seuil, surestimations dangereuses, alertes manquees ;
- chaque figure a un titre explicite, des axes avec unites et une legende ;
- les nombres des info-bulles sont ecrits a la francaise (virgule decimale, espace des milliers).
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.contrat_predictions import MODELES_CONNUS
from src.demo import STATUT_ALERTE_CORRECTE, STATUT_ALERTE_MANQUEE, STATUT_FAUSSE_ALERTE, STATUT_RIEN_A_SIGNALER
from src.donnees_demo import LIBELLES_STATUTS, ORDRE_STATUTS

# ---------------------------------------------------------------------------------------------------
# Couleurs (palette lisible par les daltoniens, inspiree d'Okabe et Ito)
# ---------------------------------------------------------------------------------------------------

# une couleur par modele, identique sur toutes les pages
COULEURS_MODELES = {
    "xgboost_phase3": "#56B4E9",  # bleu ciel
    "xgboost_retenu": "#0072B2",  # bleu fonce
    "lstm": "#8E44AD",  # violet
    "cnn1d": "#16A085",  # bleu-vert
}
COULEUR_MODELE_INCONNU = "#555555"  # gris fonce, si un nouveau modele apparait un jour

COULEUR_VERITE = "#111111"  # RUL vraie : noir
COULEUR_VERITE_PLAFONNEE = "#7F7F7F"  # RUL vraie plafonnee a 125 : gris, en pointilles
COULEUR_DANGER = "#D62728"  # rouge : reserve au danger et aux alertes
COULEUR_PERTURBATION = "#E69F00"  # orange : signal ou prediction avec derive injectee (page "Et si...")
COULEUR_NEUTRE = "#4C72B0"  # bleu-gris : barres neutres (histogrammes)

# couleur de chaque statut d'alerte (memes statuts que src/demo.py)
COULEURS_STATUTS = {
    STATUT_ALERTE_MANQUEE: COULEUR_DANGER,  # le cas dangereux
    STATUT_ALERTE_CORRECTE: "#2E7D32",  # vert : bonne decision
    STATUT_FAUSSE_ALERTE: "#E69F00",  # orange : cout, pas de risque
    STATUT_RIEN_A_SIGNALER: "#9E9E9E",  # gris : rien a faire
}

# couleurs et symboles des trois etudes Optuna (graphique du front de Pareto)
COULEURS_ETUDES = ["#009E73", "#CC79A7", "#F0B400", "#999999"]
SYMBOLES_ETUDES = ["circle", "square", "diamond", "cross"]

TAILLE_POLICE = 16  # police de base des figures (lisible au fond d'une salle)

# configuration de la barre d'outils Plotly passee a st.plotly_chart (zoom et survol gardes)
CONFIG_PLOTLY = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


def couleur_modele(modele):
    """
    Renvoie la couleur fixe d'un modele.

    Parameters
    ----------
    modele : str
        La cle du modele (ex. "xgboost_retenu").

    Returns
    -------
    str
        Une couleur hexadecimale.
    """
    return COULEURS_MODELES.get(modele, COULEUR_MODELE_INCONNU)


def appliquer_style(fig, titre, titre_x, titre_y, hauteur=480):
    """
    Applique le style commun a toutes les figures : fond blanc, police lisible, legende sous le graphique.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        La figure a modifier (modifiee sur place).
    titre : str
        Le titre affiche en haut.
    titre_x, titre_y : str
        Les titres des axes (avec les unites).
    hauteur : int, optionnel
        La hauteur en pixels (480 par defaut).

    Returns
    -------
    plotly.graph_objects.Figure
        La meme figure (pour enchainer les appels).
    """
    fig.update_layout(
        template="plotly_white",
        title={"text": titre, "x": 0.0, "xanchor": "left", "font": {"size": TAILLE_POLICE + 3}},
        font={"size": TAILLE_POLICE},
        height=hauteur,
        margin={"l": 70, "r": 30, "t": 70, "b": 60},
        legend={"orientation": "h", "yanchor": "top", "y": -0.18, "xanchor": "left", "x": 0.0},
        separators=", ",  # virgule decimale et espace des milliers, a la francaise
        hoverlabel={"font": {"size": TAILLE_POLICE - 1}},
    )
    fig.update_xaxes(title_text=titre_x, showgrid=True, gridcolor="#E5E5E5", zeroline=False)
    fig.update_yaxes(title_text=titre_y, showgrid=True, gridcolor="#E5E5E5", zeroline=False)
    return fig


def ajouter_zone_alerte(fig, seuil_alerte, ligne=None, colonne=None, avec_texte=True):
    """
    Colore en rouge pale la zone RUL <= seuil et trace la ligne du seuil (tirets rouges).

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        La figure (modifiee sur place).
    seuil_alerte : float
        Le seuil d'alerte en cycles.
    ligne, colonne : int ou None, optionnel
        La case du sous-graphique (None pour une figure simple).
    avec_texte : bool, optionnel
        Vrai pour ecrire "seuil d'alerte : N cycles (hypothese de travail)" sur la ligne.

    Returns
    -------
    None
    """
    texte = ""
    if avec_texte:
        texte = f"seuil d'alerte : {seuil_alerte:g} cycles (hypothèse de travail)"
    fig.add_hrect(y0=0, y1=seuil_alerte, fillcolor=COULEUR_DANGER, opacity=0.08, line_width=0, row=ligne, col=colonne)
    fig.add_hline(
        y=seuil_alerte, line_dash="dash", line_color=COULEUR_DANGER, line_width=1.5,
        annotation_text=texte, annotation_position="top left",
        annotation_font={"color": COULEUR_DANGER, "size": TAILLE_POLICE - 2},
        row=ligne, col=colonne,
    )


# ---------------------------------------------------------------------------------------------------
# Page "Moteur en direct"
# ---------------------------------------------------------------------------------------------------

def figure_moteur_en_direct(trajectoires, cycle_courant, seuil_alerte, reveler_verite, titre,
                            cycle_max_axe, rul_max_axe, premieres_alertes):
    """
    Trace la RUL predite par chaque modele jusqu'au cycle courant, avec la verite et la zone d'alerte.

    Les axes sont fixes (toute la duree du moteur) : pendant le rejeu, les courbes avancent dans un
    cadre qui ne bouge pas.

    Parameters
    ----------
    trajectoires : list de (str, pandas.DataFrame)
        (cle du modele, lignes du moteur triees par cycle : time_cycles, rul_predite, rul_vraie,
        rul_vraie_cappee). La verite est lue dans la premiere trajectoire.
    cycle_courant : int
        Le "present" du rejeu : rien n'est trace apres ce cycle.
    seuil_alerte : float
        Le seuil d'alerte en cycles.
    reveler_verite : bool
        Vrai pour tracer la RUL vraie et la RUL vraie plafonnee.
    titre : str
        Le titre de la figure.
    cycle_max_axe : int
        Le dernier cycle observe (fixe l'axe horizontal).
    rul_max_axe : float
        La plus grande RUL a afficher (fixe l'axe vertical).
    premieres_alertes : dict
        cle du modele -> (cycle, RUL vraie) de sa premiere alerte jusqu'au cycle courant, ou None.

    Returns
    -------
    plotly.graph_objects.Figure
        La figure.
    """
    fig = go.Figure()
    ajouter_zone_alerte(fig, seuil_alerte)

    # verite (lue dans la premiere trajectoire : elle est la meme pour tous les modeles d'un moteur)
    if reveler_verite and len(trajectoires) > 0:
        verite = trajectoires[0][1]
        verite_passee = verite[verite["time_cycles"] <= cycle_courant]
        fig.add_trace(go.Scatter(
            x=verite_passee["time_cycles"], y=verite_passee["rul_vraie"], mode="lines",
            name="RUL vraie", line={"color": COULEUR_VERITE, "width": 3},
            hovertemplate="RUL vraie : %{y:.0f} cycles<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=verite_passee["time_cycles"], y=verite_passee["rul_vraie_cappee"], mode="lines",
            name="RUL vraie plafonnée à 125 (ce que le modèle apprend à prédire)",
            line={"color": COULEUR_VERITE_PLAFONNEE, "width": 2.5, "dash": "dot"},
            hovertemplate="RUL vraie plafonnée : %{y:.0f} cycles<extra></extra>",
        ))

    # une courbe par modele, jusqu'au cycle courant, et un point sur le cycle courant
    for modele, trajectoire in trajectoires:
        passe = trajectoire[trajectoire["time_cycles"] <= cycle_courant]
        couleur = couleur_modele(modele)
        nom = MODELES_CONNUS.get(modele, modele)
        fig.add_trace(go.Scatter(
            x=passe["time_cycles"], y=passe["rul_predite"], mode="lines", name=f"RUL prédite — {nom}",
            line={"color": couleur, "width": 3}, legendgroup=modele,
            hovertemplate=f"{nom} : " + "%{y:.1f} cycles<extra></extra>",
        ))
        if len(passe) > 0:
            fig.add_trace(go.Scatter(
                x=[passe["time_cycles"].iloc[-1]], y=[passe["rul_predite"].iloc[-1]], mode="markers",
                marker={"color": couleur, "size": 13, "line": {"color": "white", "width": 2}},
                showlegend=False, legendgroup=modele, hoverinfo="skip",
            ))
        # premiere alerte de ce modele (triangle pointe vers le bas, dans la couleur du modele)
        premiere = premieres_alertes.get(modele)
        if premiere is not None:
            cycle_alerte, rul_vraie_alerte = premiere
            lignes_alerte = trajectoire[trajectoire["time_cycles"] == cycle_alerte]
            rul_predite_alerte = float(lignes_alerte["rul_predite"].iloc[0])
            fig.add_trace(go.Scatter(
                x=[cycle_alerte], y=[rul_predite_alerte], mode="markers",
                name=f"1re alerte — {nom} (cycle {cycle_alerte})",
                marker={"color": couleur, "size": 18, "symbol": "triangle-down", "line": {"color": COULEUR_DANGER, "width": 2}},
                legendgroup=modele,
                hovertemplate=(
                    f"1re alerte {nom} : cycle {cycle_alerte}<br>"
                    f"RUL vraie à ce cycle : {rul_vraie_alerte:.0f} cycles<extra></extra>"
                ),
            ))

    # trait vertical du cycle courant
    fig.add_vline(
        x=cycle_courant, line_dash="dot", line_color="#555555", line_width=1.5,
        annotation_text=f"cycle {cycle_courant}", annotation_position="top left",
    )

    appliquer_style(fig, titre, "Cycle de fonctionnement observé", "RUL (cycles restants)", hauteur=480)
    fig.update_xaxes(range=[0, cycle_max_axe + max(3, 0.02 * cycle_max_axe)])
    fig.update_yaxes(range=[0, rul_max_axe * 1.08])
    fig.update_layout(hovermode="x unified")
    return fig


def figure_capteurs(donnees_moteur, capteurs, cycle_courant, cycle_max_axe, regimes=None):
    """
    Trace les capteurs bruts choisis d'un moteur, jusqu'au cycle courant (un sous-graphique par capteur).

    Parameters
    ----------
    donnees_moteur : pandas.DataFrame
        Les donnees brutes du moteur (time_cycles + colonnes sensor_N), triees par cycle.
    capteurs : list de str
        Les capteurs a tracer (ex. ["sensor_11", "sensor_4"]).
    cycle_courant : int
        Le "present" du rejeu.
    cycle_max_axe : int
        Le dernier cycle observe (fixe l'axe horizontal).
    regimes : pandas.Series ou None, optionnel
        Le regime operatoire de chaque ligne (meme ordre que donnees_moteur) ; si plusieurs regimes,
        les points sont colores par regime (les valeurs brutes sautent d'un regime a l'autre).

    Returns
    -------
    plotly.graph_objects.Figure
        La figure.
    """
    nb_capteurs = len(capteurs)
    fig = make_subplots(rows=nb_capteurs, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=capteurs)
    masque_passe = donnees_moteur["time_cycles"].to_numpy() <= cycle_courant
    passe = donnees_moteur[masque_passe]

    plusieurs_regimes = regimes is not None and len(np.unique(np.asarray(regimes))) > 1
    for numero in range(nb_capteurs):
        capteur = capteurs[numero]
        if plusieurs_regimes:
            regimes_passe = np.asarray(regimes)[masque_passe]
            fig.add_trace(go.Scatter(
                x=passe["time_cycles"], y=passe[capteur], mode="markers", name=capteur, showlegend=False,
                marker={"size": 6, "color": regimes_passe, "colorscale": "Viridis", "showscale": False},
                customdata=regimes_passe,
                hovertemplate=f"{capteur} : " + "%{y:.2f}<br>régime %{customdata}<extra></extra>",
            ), row=numero + 1, col=1)
        else:
            fig.add_trace(go.Scatter(
                x=passe["time_cycles"], y=passe[capteur], mode="lines", name=capteur, showlegend=False,
                line={"color": "#333333", "width": 1.8},
                hovertemplate=f"{capteur} : " + "%{y:.2f}<extra></extra>",
            ), row=numero + 1, col=1)
        fig.add_vline(x=cycle_courant, line_dash="dot", line_color="#555555", row=numero + 1, col=1)
        fig.update_yaxes(title_text="valeur brute", row=numero + 1, col=1)

    titre = "Capteurs bruts du moteur, jusqu'au cycle courant (valeurs du fichier test, non normalisées)"
    appliquer_style(fig, titre, "", "", hauteur=150 + 170 * nb_capteurs)
    fig.update_xaxes(range=[0, cycle_max_axe + max(3, 0.02 * cycle_max_axe)])
    fig.update_xaxes(title_text="Cycle de fonctionnement observé", row=nb_capteurs, col=1)
    fig.update_layout(hovermode="x unified", showlegend=False)
    return fig


# ---------------------------------------------------------------------------------------------------
# Page "Flotte"
# ---------------------------------------------------------------------------------------------------

def figure_nuage_flotte(tableau, seuil_alerte, seuil_surestimation, titre):
    """
    Nuage RUL predite contre RUL vraie au dernier cycle, avec la diagonale et la zone de surestimation.

    Parameters
    ----------
    tableau : pandas.DataFrame
        Le resultat de `tableau_flotte` (moteur, rul_vraie, rul_predite, ecart, statut).
    seuil_alerte : float
        Le seuil d'alerte (lignes verticale et horizontale).
    seuil_surestimation : float
        Au-dela de cet ecart positif, une surestimation est comptee comme dangereuse (20 cycles).
    titre : str
        Le titre de la figure.

    Returns
    -------
    plotly.graph_objects.Figure
        La figure.
    """
    valeur_max = max(float(tableau["rul_vraie"].max()), float(tableau["rul_predite"].max()), seuil_alerte) * 1.08
    fig = go.Figure()

    # zone de surestimation dangereuse : RUL predite > RUL vraie + seuil (polygone rouge pale)
    fig.add_trace(go.Scatter(
        x=[0, valeur_max, valeur_max, 0], y=[seuil_surestimation, valeur_max + seuil_surestimation, valeur_max * 2, valeur_max * 2],
        fill="toself", fillcolor="rgba(214, 39, 40, 0.10)", line={"width": 0}, mode="lines",
        name=f"zone de surestimation > {seuil_surestimation:g} cycles (dangereuse)", hoverinfo="skip",
    ))
    # diagonale : prediction parfaite
    fig.add_trace(go.Scatter(
        x=[0, valeur_max], y=[0, valeur_max], mode="lines", name="prédiction parfaite (prédite = vraie)",
        line={"color": "#333333", "dash": "dash", "width": 1.5}, hoverinfo="skip",
    ))

    # un groupe de points par statut (legende = les 4 statuts)
    for statut in ORDRE_STATUTS:
        lignes = tableau[tableau["statut"] == statut]
        if len(lignes) == 0:
            continue
        fig.add_trace(go.Scatter(
            x=lignes["rul_vraie"], y=lignes["rul_predite"], mode="markers",
            name=f"{LIBELLES_STATUTS[statut]} ({len(lignes)})",
            marker={"color": COULEURS_STATUTS[statut], "size": 11, "line": {"color": "white", "width": 1}},
            customdata=np.stack([lignes["moteur"].to_numpy(), lignes["ecart"].to_numpy()], axis=1),
            hovertemplate=(
                "Moteur %{customdata[0]}<br>RUL vraie : %{x:.0f} cycles<br>RUL prédite : %{y:.1f} cycles"
                "<br>Écart : %{customdata[1]:+.1f} cycles<extra>" + LIBELLES_STATUTS[statut] + "</extra>"
            ),
        ))

    # seuil d'alerte sur les deux axes : les quadrants correspondent aux 4 statuts
    fig.add_hline(y=seuil_alerte, line_dash="dot", line_color=COULEUR_DANGER, line_width=1.2)
    fig.add_vline(x=seuil_alerte, line_dash="dot", line_color=COULEUR_DANGER, line_width=1.2,
                  annotation_text=f"seuil {seuil_alerte:g}", annotation_position="top right",
                  annotation_font={"color": COULEUR_DANGER})

    appliquer_style(fig, titre, "RUL vraie au dernier cycle observé (cycles)", "RUL prédite (cycles)", hauteur=560)
    fig.update_xaxes(range=[0, valeur_max])
    fig.update_yaxes(range=[0, valeur_max])
    return fig


def figure_histogramme_ecarts(tableau, seuil_surestimation, titre):
    """
    Histogramme des ecarts RUL predite - RUL vraie au dernier cycle, surestimations dangereuses en rouge.

    Parameters
    ----------
    tableau : pandas.DataFrame
        Le resultat de `tableau_flotte` (colonne "ecart").
    seuil_surestimation : float
        Seuil des surestimations dangereuses (20 cycles).
    titre : str
        Le titre de la figure.

    Returns
    -------
    plotly.graph_objects.Figure
        La figure.
    """
    ecarts = tableau["ecart"].to_numpy(dtype=float)
    dangereux = ecarts[ecarts > seuil_surestimation]
    autres = ecarts[ecarts <= seuil_surestimation]

    # classes de 5 cycles, alignees sur 0, couvrant tous les ecarts
    borne_basse = 5.0 * np.floor(min(float(np.min(ecarts)), 0.0) / 5.0)
    borne_haute = 5.0 * np.ceil(max(float(np.max(ecarts)), seuil_surestimation) / 5.0) + 5.0
    classes = {"start": borne_basse, "end": borne_haute, "size": 5}

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=autres, xbins=classes, name=f"écart ≤ +{seuil_surestimation:g} cycles ({len(autres)} moteurs)",
        marker_color=COULEUR_NEUTRE, hovertemplate="écart %{x} : %{y} moteurs<extra></extra>",
    ))
    fig.add_trace(go.Histogram(
        x=dangereux, xbins=classes, name=f"surestimation > {seuil_surestimation:g} cycles ({len(dangereux)} moteurs)",
        marker_color=COULEUR_DANGER, hovertemplate="écart %{x} : %{y} moteurs<extra></extra>",
    ))
    fig.add_vline(x=0, line_color="#333333", line_width=1.5)
    fig.add_vline(x=seuil_surestimation, line_dash="dash", line_color=COULEUR_DANGER,
                  annotation_text=f"+{seuil_surestimation:g}", annotation_font={"color": COULEUR_DANGER})
    appliquer_style(
        fig, titre, "Écart RUL prédite − RUL vraie (cycles) : > 0 = surestimation", "Nombre de moteurs", hauteur=420
    )
    fig.update_layout(barmode="stack", bargap=0.05)
    return fig


# ---------------------------------------------------------------------------------------------------
# Page "Comparaison des modeles"
# ---------------------------------------------------------------------------------------------------

def figure_barres_comparaison(tableau, normaliser, seuil_surestimation, seuil_alerte):
    """
    Barres groupees par sous-jeu (une couleur par modele) : RMSE, score NASA, surestimes, alertes manquees.

    Parameters
    ----------
    tableau : pandas.DataFrame
        Le resultat de `tableau_comparaison`.
    normaliser : bool
        Vrai pour diviser le score NASA et les comptes par le nombre de moteurs test (sous-jeux de
        tailles differentes : 100 a 259 moteurs).
    seuil_surestimation : float
        Seuil des surestimations dangereuses (pour le titre).
    seuil_alerte : float
        Seuil d'alerte (pour le titre).

    Returns
    -------
    plotly.graph_objects.Figure
        La figure (2 x 2 sous-graphiques).
    """
    if normaliser:
        titres = [
            "RMSE au dernier cycle (cycles)",
            "Score NASA moyen par moteur",
            f"Moteurs surestimés de plus de {seuil_surestimation:g} cycles (% des moteurs)",
            f"Alertes manquées, seuil {seuil_alerte:g} cycles (% des moteurs)",
        ]
    else:
        titres = [
            "RMSE au dernier cycle (cycles)",
            "Score NASA (somme sur les moteurs)",
            f"Moteurs surestimés de plus de {seuil_surestimation:g} cycles",
            f"Alertes manquées, seuil {seuil_alerte:g} cycles",
        ]
    fig = make_subplots(rows=2, cols=2, subplot_titles=titres, vertical_spacing=0.16, horizontal_spacing=0.08)

    # ordre des modeles : celui du contrat ; ordre des sous-jeux : celui du tableau
    modeles_presents = []
    for modele in MODELES_CONNUS:
        if modele in tableau["modele"].tolist():
            modeles_presents.append(modele)

    for modele in modeles_presents:
        lignes = tableau[tableau["modele"] == modele]
        nb_moteurs = lignes["nb_moteurs"].to_numpy(dtype=float)
        valeurs_par_case = []
        valeurs_par_case.append(lignes["rmse"].to_numpy(dtype=float))
        if normaliser:
            valeurs_par_case.append(lignes["score_nasa"].to_numpy(dtype=float) / nb_moteurs)
            valeurs_par_case.append(100.0 * lignes["nb_surestimes"].to_numpy(dtype=float) / nb_moteurs)
            valeurs_par_case.append(100.0 * lignes["alertes_manquees"].to_numpy(dtype=float) / nb_moteurs)
        else:
            valeurs_par_case.append(lignes["score_nasa"].to_numpy(dtype=float))
            valeurs_par_case.append(lignes["nb_surestimes"].to_numpy(dtype=float))
            valeurs_par_case.append(lignes["alertes_manquees"].to_numpy(dtype=float))
        formats = ["%{y:.2f}", "%{y:,.1f}", "%{y:.0f}", "%{y:.0f}"]
        if normaliser:
            formats = ["%{y:.2f}", "%{y:.2f}", "%{y:.1f} %", "%{y:.1f} %"]

        for numero_case in range(4):
            ligne_case = numero_case // 2 + 1  # cases (1,1) (1,2) (2,1) (2,2)
            colonne_case = numero_case % 2 + 1
            fig.add_trace(go.Bar(
                x=lignes["sous_jeu"], y=valeurs_par_case[numero_case], name=MODELES_CONNUS.get(modele, modele),
                marker_color=couleur_modele(modele), legendgroup=modele, showlegend=(numero_case == 0),
                texttemplate=formats[numero_case], textposition="outside", cliponaxis=False,
                textfont={"size": TAILLE_POLICE - 4},
                customdata=nb_moteurs,
                hovertemplate=(
                    MODELES_CONNUS.get(modele, modele) + " — %{x}<br>" + formats[numero_case]
                    + "<br>%{customdata:.0f} moteurs test<extra></extra>"
                ),
            ), row=ligne_case, col=colonne_case)

    appliquer_style(fig, "Métriques de test au dernier cycle observé, par sous-jeu et par modèle", "", "", hauteur=760)
    # une etiquette trop large pour sa barre est masquee (la valeur reste lisible au survol)
    fig.update_layout(barmode="group", legend={"y": -0.08}, uniformtext={"minsize": TAILLE_POLICE - 6, "mode": "hide"})
    fig.update_yaxes(rangemode="tozero")
    return fig


def figure_pareto(analyse):
    """
    Front de Pareto des essais Optuna (RMSE CV moyen, score NASA CV moyen), essai retenu mis en evidence.

    Parameters
    ----------
    analyse : dict
        Le resultat de `analyser_optuna` (essais, seuil_rmse, retenu, meilleur_rmse...).

    Returns
    -------
    plotly.graph_objects.Figure
        La figure.
    """
    essais = analyse["essais"]
    fig = go.Figure()

    # un nuage par etude (couleur et symbole propres), info-bulle = parametres de l'essai
    etudes = []
    for etude in essais["etude"].tolist():
        if etude not in etudes:
            etudes.append(etude)
    for position in range(len(etudes)):
        etude = etudes[position]
        lignes = essais[essais["etude"] == etude]
        couleur = COULEURS_ETUDES[position % len(COULEURS_ETUDES)]
        symbole = SYMBOLES_ETUDES[position % len(SYMBOLES_ETUDES)]
        fig.add_trace(go.Scatter(
            x=lignes["rmse_cv_moyen"], y=lignes["nasa_cv_moyen"], mode="markers",
            name=f"étude {etude} ({len(lignes)} essais)",
            marker={"color": couleur, "symbol": symbole, "size": 9, "opacity": 0.75},
            customdata=np.stack([lignes["numero"].to_numpy(), lignes["surestimes_oof"].to_numpy(),
                                 lignes["parametres_texte"].to_numpy()], axis=1),
            hovertemplate=(
                f"<b>{etude} n°" + "%{customdata[0]}</b><br>RMSE CV moyen : %{x:.2f}<br>"
                "Score NASA CV moyen : %{y:,.0f}<br>Surestimations hors-pli > 20 : %{customdata[1]}"
                "<br>%{customdata[2]}<extra></extra>"
            ),
        ))

    # front de Pareto : essais non domines, relies par RMSE croissant
    front = essais[essais["sur_le_front"]].sort_values("rmse_cv_moyen")
    fig.add_trace(go.Scatter(
        x=front["rmse_cv_moyen"], y=front["nasa_cv_moyen"], mode="lines+markers",
        name=f"front de Pareto ({len(front)} essais non dominés)",
        line={"color": "#222222", "width": 2}, marker={"color": "#222222", "size": 7}, hoverinfo="skip",
    ))

    # seuil de la regle de choix : RMSE <= 1,05 x le meilleur RMSE du front
    fig.add_vline(
        x=analyse["seuil_rmse"], line_dash="dash", line_color="#555555",
        annotation_text=f"seuil de la règle : {analyse['seuil_rmse']:.2f}".replace(".", ","),
        annotation_position="top right",
    )

    # essai retenu (etoile, couleur du modele retenu) et meilleur RMSE (losange noir)
    retenu = analyse["retenu"]
    meilleur = analyse["meilleur_rmse"]
    fig.add_trace(go.Scatter(
        x=[retenu["rmse_cv_moyen"]], y=[retenu["nasa_cv_moyen"]], mode="markers",
        name=f"retenu par la règle : {retenu['etude']} n°{int(retenu['numero'])}",
        marker={"color": couleur_modele("xgboost_retenu"), "symbol": "star", "size": 24, "line": {"color": "white", "width": 1.5}},
        hovertemplate=(
            f"<b>Retenu : {retenu['etude']} n°{int(retenu['numero'])}</b><br>RMSE CV moyen : %{{x:.2f}}<br>"
            f"Score NASA CV moyen : %{{y:,.0f}}<br>{retenu['parametres_texte']}<extra></extra>"
        ),
    ))
    if not (retenu["etude"] == meilleur["etude"] and retenu["numero"] == meilleur["numero"]):
        fig.add_trace(go.Scatter(
            x=[meilleur["rmse_cv_moyen"]], y=[meilleur["nasa_cv_moyen"]], mode="markers",
            name=f"meilleur RMSE du front : {meilleur['etude']} n°{int(meilleur['numero'])}",
            marker={"color": "#111111", "symbol": "diamond-open", "size": 20, "line": {"width": 3}},
            hovertemplate=(
                f"<b>Meilleur RMSE : {meilleur['etude']} n°{int(meilleur['numero'])}</b><br>RMSE CV moyen : %{{x:.2f}}<br>"
                f"Score NASA CV moyen : %{{y:,.0f}}<br>{meilleur['parametres_texte']}<extra></extra>"
            ),
        ))

    appliquer_style(
        fig, f"Optuna : {len(essais)} essais en validation croisée (5 plis), front de Pareto et essai retenu",
        "RMSE CV moyen (cycles) — plus petit = plus précis", "Score NASA CV moyen — plus petit = moins de retards",
        hauteur=600,
    )
    # vue initiale resserree autour du front (double-clic dans le graphique : tous les essais)
    cadrage = cadrage_pareto(analyse)
    fig.update_xaxes(range=cadrage["x"])
    fig.update_yaxes(range=cadrage["y"])
    return fig


def cadrage_pareto(analyse):
    """
    Calcule la vue initiale du graphique de Pareto : resserree autour du front et du seuil de la regle.

    Quelques essais tres mauvais (RMSE ou score NASA eleves) ecrasent sinon la zone interessante dans un
    coin. La vue va du plus petit RMSE au plus grand de (seuil + 1, 90e centile des RMSE), et du plus
    petit score NASA au 90e centile des scores ; le nombre d'essais visibles est renvoye pour l'afficher.

    Parameters
    ----------
    analyse : dict
        Le resultat de `analyser_optuna`.

    Returns
    -------
    dict
        "x" et "y" : [minimum, maximum] des axes ; "nb_visibles" : essais dans cette vue.
    """
    rmse = analyse["essais"]["rmse_cv_moyen"].to_numpy(dtype=float)
    nasa = analyse["essais"]["nasa_cv_moyen"].to_numpy(dtype=float)
    x_min = float(np.min(rmse)) - 0.25
    x_max = max(analyse["seuil_rmse"] + 1.0, float(np.quantile(rmse, 0.9)))
    y_min = float(np.min(nasa)) * 0.95
    y_max = float(np.quantile(nasa, 0.9)) * 1.05
    nb_visibles = 0
    for position in range(len(rmse)):
        if x_min <= rmse[position] <= x_max and y_min <= nasa[position] <= y_max:
            nb_visibles = nb_visibles + 1
    cadrage = {}
    cadrage["x"] = [x_min, x_max]
    cadrage["y"] = [y_min, y_max]
    cadrage["nb_visibles"] = nb_visibles
    return cadrage


# ---------------------------------------------------------------------------------------------------
# Page "Et si..."
# ---------------------------------------------------------------------------------------------------

def figure_simulation(resultat, seuil_alerte, capteur, cycle_debut, nom_modele, titre):
    """
    Deux graphiques superposes : RUL predite avec / sans derive, puis le capteur avec / sans derive.

    Parameters
    ----------
    resultat : dict
        Le resultat de `simuler_derive`.
    seuil_alerte : float
        Le seuil d'alerte en cycles.
    capteur : str
        Le capteur perturbe.
    cycle_debut : int
        Le premier cycle perturbe.
    nom_modele : str
        Le nom lisible du modele.
    titre : str
        Le titre de la figure.

    Returns
    -------
    plotly.graph_objects.Figure
        La figure.
    """
    tableau = resultat["tableau"]
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10, row_heights=[0.62, 0.38],
        subplot_titles=["RUL prédite par le modèle (inférence en direct)", f"Capteur {capteur} (valeur brute)"],
    )

    fig.add_trace(go.Scatter(
        x=tableau["time_cycles"], y=tableau["rul_vraie"], mode="lines", name="RUL vraie (inchangée)",
        line={"color": COULEUR_VERITE, "width": 2.5}, hovertemplate="RUL vraie : %{y:.0f}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=tableau["time_cycles"], y=tableau["rul_vraie_cappee"], mode="lines", name="RUL vraie plafonnée à 125",
        line={"color": COULEUR_VERITE_PLAFONNEE, "width": 2, "dash": "dot"}, hoverinfo="skip",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=tableau["time_cycles"], y=tableau["rul_predite_sans"], mode="lines", name=f"{nom_modele} — données d'origine",
        line={"color": couleur_modele("xgboost_retenu"), "width": 3},
        hovertemplate="sans dérive : %{y:.1f}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=tableau["time_cycles"], y=tableau["rul_predite_avec"], mode="lines", name=f"{nom_modele} — avec dérive injectée",
        line={"color": COULEUR_PERTURBATION, "width": 3, "dash": "dash"},
        hovertemplate="avec dérive : %{y:.1f}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=tableau["time_cycles"], y=tableau["capteur_sans"], mode="lines", name=f"{capteur} d'origine",
        line={"color": "#333333", "width": 1.8}, hovertemplate="origine : %{y:.3f}<extra></extra>",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=tableau["time_cycles"], y=tableau["capteur_avec"], mode="lines", name=f"{capteur} avec dérive",
        line={"color": COULEUR_PERTURBATION, "width": 2.2, "dash": "dash"}, hovertemplate="avec dérive : %{y:.3f}<extra></extra>",
    ), row=2, col=1)

    # zone d'alerte APRES les courbes : Plotly n'ajoute pas de forme a un sous-graphique encore vide
    ajouter_zone_alerte(fig, seuil_alerte, ligne=1, colonne=1)
    for numero_ligne in [1, 2]:
        fig.add_vline(x=cycle_debut, line_dash="dot", line_color=COULEUR_PERTURBATION, line_width=2, row=numero_ligne, col=1)
    fig.add_annotation(
        x=cycle_debut, y=1.0, yref="y domain", xref="x", text=f"début de la dérive : cycle {cycle_debut}",
        showarrow=False, xanchor="left", font={"color": COULEUR_PERTURBATION}, row=1, col=1,
    )

    appliquer_style(fig, titre, "", "", hauteur=760)
    fig.update_yaxes(title_text="RUL (cycles)", rangemode="tozero", row=1, col=1)
    fig.update_yaxes(title_text="valeur brute", row=2, col=1)
    fig.update_xaxes(title_text="Cycle de fonctionnement observé", row=2, col=1)
    fig.update_layout(hovermode="x unified", legend={"y": -0.10})
    return fig


# ---------------------------------------------------------------------------------------------------
# Page "Fiabilite du prototype"
# ---------------------------------------------------------------------------------------------------

def figure_temps_reponse(resultats):
    """
    Histogramme des temps de preparation + prediction d'un moteur, avec p50, p95 et la cible.

    Parameters
    ----------
    resultats : dict
        Le resultat de `mesurer_temps_reponse` (temps_moteur_ms, p50_ms, p95_ms, cible_ms, nb_mesures).

    Returns
    -------
    plotly.graph_objects.Figure
        La figure.
    """
    temps = np.asarray(resultats["temps_moteur_ms"], dtype=float)
    cible = float(resultats["cible_ms"])
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=temps, nbinsx=40, marker_color=COULEUR_NEUTRE, name=f"{len(temps)} mesures (un moteur chacune)",
        hovertemplate="%{x} ms : %{y} mesures<extra></extra>",
    ))
    fig.add_vline(x=resultats["p50_ms"], line_color="#2E7D32", line_width=2,
                  annotation_text=f"p50 = {resultats['p50_ms']:.1f} ms".replace(".", ","), annotation_position="top right")
    fig.add_vline(x=resultats["p95_ms"], line_color="#222222", line_width=2, line_dash="dash",
                  annotation_text=f"p95 = {resultats['p95_ms']:.1f} ms".replace(".", ","), annotation_position="bottom right")
    fig.add_vline(x=cible, line_color=COULEUR_DANGER, line_width=2, line_dash="dot",
                  annotation_text=f"cible = {cible:g} ms", annotation_position="top left",
                  annotation_font={"color": COULEUR_DANGER})
    appliquer_style(
        fig, "Temps de préparation + prédiction d'un moteur (tous ses cycles), mesuré sur cette machine",
        "Temps (ms)", "Nombre de mesures", hauteur=420,
    )
    fig.update_xaxes(range=[0, max(cible, float(np.max(temps))) * 1.1])
    return fig
