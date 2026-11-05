"""
Figures Plotly de l'application de demo (aucune dependance a Streamlit : testables seules).

Regles de lisibilite, pensees pour un videoprojecteur et un jury :
- une couleur FIXE par modele, la meme sur les deux pages (COULEURS_MODELES) ;
- le ROUGE est reserve au danger : zone d'alerte, seuil, alertes ;
- chaque figure a un titre explicite, des axes avec unites et une legende ;
- les nombres des info-bulles sont ecrits a la francaise (virgule decimale, espace des milliers).
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.contrat_predictions import MODELES_CONNUS

# ---------------------------------------------------------------------------------------------------
# Couleurs (palette lisible par les daltoniens, inspiree d'Okabe et Ito)
# ---------------------------------------------------------------------------------------------------

# une couleur par modele, identique sur toutes les pages
COULEURS_MODELES = {
    "reference_naive": "#A6761D",  # brun
    "random_forest": "#0072B2",  # bleu fonce
    "xgboost_phase3": "#56B4E9",  # bleu ciel
    "lightgbm": "#E69F00",  # orange
    "lstm": "#8E44AD",  # violet
    "cnn1d": "#16A085",  # bleu-vert
}
COULEUR_MODELE_INCONNU = "#555555"  # gris fonce, si un nouveau modele apparait un jour

COULEUR_VERITE = "#111111"  # RUL vraie : noir
COULEUR_VERITE_PLAFONNEE = "#7F7F7F"  # RUL vraie plafonnee a 125 : gris, en pointilles
COULEUR_DANGER = "#D62728"  # rouge : reserve au danger et aux alertes

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
        La cle du modele (ex. "lstm").

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


def ajouter_zone_alerte(fig, seuil_alerte):
    """
    Colore en rouge pale la zone RUL <= seuil et trace la ligne du seuil (tirets rouges).

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        La figure (modifiee sur place).
    seuil_alerte : float
        Le seuil d'alerte en cycles.

    Returns
    -------
    None
    """
    fig.add_hrect(y0=0, y1=seuil_alerte, fillcolor=COULEUR_DANGER, opacity=0.08, line_width=0)
    fig.add_hline(
        y=seuil_alerte, line_dash="dash", line_color=COULEUR_DANGER, line_width=1.5,
        annotation_text=f"seuil d'alerte : {seuil_alerte:g} cycles", annotation_position="top left",
        annotation_font={"color": COULEUR_DANGER, "size": TAILLE_POLICE - 2},
    )


# ---------------------------------------------------------------------------------------------------
# Page "Tester un moteur"
# ---------------------------------------------------------------------------------------------------

def figure_prediction_en_direct(predictions, seuil_alerte, titre, cycle_max_axe, rul_max_axe, verite=None):
    """
    Trace la RUL predite EN DIRECT par chaque modele, du cycle 1 au cycle courant, et la verite si revelee.

    Les axes sont fixes (toute la duree du moteur) : pendant la lecture, les courbes avancent dans un cadre
    qui ne bouge pas. Tant que `verite` vaut None, AUCUNE information sur la RUL vraie n'est dans la figure
    (ni courbe, ni info-bulle) : rien ne peut fuiter vers le navigateur.

    Parameters
    ----------
    predictions : dict
        cle du modele -> liste des RUL predites aux cycles 1, 2, ..., t (meme longueur pour tous).
    seuil_alerte : float
        Le seuil d'alerte en cycles.
    titre : str
        Le titre de la figure.
    cycle_max_axe : int
        Le dernier cycle du moteur (fixe l'axe horizontal).
    rul_max_axe : float
        La plus grande RUL a afficher (fixe l'axe vertical).
    verite : pandas.DataFrame ou None, optionnel
        Lignes des cycles 1..t : "time_cycles", "rul_vraie", "rul_vraie_cappee" ; None : verite cachee.

    Returns
    -------
    plotly.graph_objects.Figure
        La figure.
    """
    fig = go.Figure()
    ajouter_zone_alerte(fig, seuil_alerte)

    # verite (seulement si elle a ete revelee)
    if verite is not None and len(verite) > 0:
        fig.add_trace(go.Scatter(
            x=verite["time_cycles"], y=verite["rul_vraie"], mode="lines", name="RUL vraie",
            line={"color": COULEUR_VERITE, "width": 3}, hovertemplate="RUL vraie : %{y:.0f} cycles<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=verite["time_cycles"], y=verite["rul_vraie_cappee"], mode="lines",
            name="RUL vraie plafonnée à 125 (cible apprise)",
            line={"color": COULEUR_VERITE_PLAFONNEE, "width": 2.5, "dash": "dot"},
            hovertemplate="RUL vraie plafonnée : %{y:.0f} cycles<extra></extra>",
        ))

    # une courbe par modele (cycles 1..t), un point sur le cycle courant, un triangle a la premiere alerte
    cycle_courant = 0
    for modele in predictions:
        valeurs = predictions[modele]
        if len(valeurs) == 0:
            continue
        cycles = list(range(1, len(valeurs) + 1))
        cycle_courant = max(cycle_courant, len(valeurs))
        couleur = couleur_modele(modele)
        nom = MODELES_CONNUS.get(modele, modele)
        fig.add_trace(go.Scatter(
            x=cycles, y=valeurs, mode="lines", name=nom, line={"color": couleur, "width": 3}, legendgroup=modele,
            hovertemplate=f"{nom} : " + "%{y:.1f} cycles<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[cycles[-1]], y=[valeurs[-1]], mode="markers", showlegend=False, legendgroup=modele, hoverinfo="skip",
            marker={"color": couleur, "size": 13, "line": {"color": "white", "width": 2}},
        ))
        for position in range(len(valeurs)):  # premiere alerte : RUL predite <= seuil (regle de src/alertes.py)
            if valeurs[position] <= seuil_alerte:
                fig.add_trace(go.Scatter(
                    x=[cycles[position]], y=[valeurs[position]], mode="markers", showlegend=False,
                    legendgroup=modele, hovertemplate=f"1re alerte {nom} : cycle {cycles[position]}<extra></extra>",
                    marker={"color": couleur, "size": 17, "symbol": "triangle-down",
                            "line": {"color": COULEUR_DANGER, "width": 2}},
                ))
                break

    # trait vertical du cycle courant
    if cycle_courant > 0:
        fig.add_vline(x=cycle_courant, line_dash="dot", line_color="#555555", line_width=1.5)

    appliquer_style(fig, titre, "Cycle (vol) reçu", "RUL (cycles restants)", hauteur=460)
    fig.update_xaxes(range=[0, cycle_max_axe + max(3, 0.02 * cycle_max_axe)])
    fig.update_yaxes(range=[0, rul_max_axe * 1.08])
    fig.update_layout(hovermode="x unified")
    return fig


# ---------------------------------------------------------------------------------------------------
# Page "Comparer les modeles"
# ---------------------------------------------------------------------------------------------------

def figure_comparaison(tableau, titres):
    """
    Une rangee de quatre petits diagrammes en barres (une barre par modele, sa couleur fixe), un par metrique.

    Parameters
    ----------
    tableau : pandas.DataFrame
        Le resultat de `tableau_comparaison_sous_jeu` (colonnes modele, nom_affiche et les metriques).
    titres : dict
        colonne de metrique -> titre du diagramme, dans l'ordre d'affichage.

    Returns
    -------
    plotly.graph_objects.Figure
        La figure.
    """
    colonnes = list(titres.keys())
    noms_titres = []
    for colonne in colonnes:
        noms_titres.append(titres[colonne])
    fig = make_subplots(rows=1, cols=len(colonnes), subplot_titles=noms_titres, horizontal_spacing=0.07)
    formats = {"rmse": "%{y:.2f}", "score_nasa": "%{y:,.0f}"}

    for position in range(len(tableau)):
        modele = tableau["modele"].iloc[position]
        nom = tableau["nom_affiche"].iloc[position]
        for numero in range(len(colonnes)):
            colonne = colonnes[numero]
            format_valeur = formats.get(colonne, "%{y:.0f}")
            fig.add_trace(go.Bar(
                x=[nom], y=[float(tableau[colonne].iloc[position])], name=nom, marker_color=couleur_modele(modele),
                legendgroup=modele, showlegend=False, texttemplate=format_valeur, textposition="outside",
                cliponaxis=False, hovertemplate=f"{nom}<br>" + format_valeur + "<extra></extra>",
            ), row=1, col=numero + 1)

    appliquer_style(fig, "", "", "", hauteur=360)
    fig.update_layout(margin={"l": 40, "r": 20, "t": 50, "b": 40}, bargap=0.25)
    fig.update_xaxes(showticklabels=True, tickfont={"size": TAILLE_POLICE - 4})
    fig.update_yaxes(rangemode="tozero")
    return fig
