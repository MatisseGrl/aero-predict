"""
Page "Comparer les modeles" : pour un sous-jeu, un tableau et un graphique des metriques de test.

Metriques lues dans les metadonnees des fichiers sorties/predictions/<SOUS_JEU>__<modele>.json (calculees au
dernier cycle observe de chaque moteur test, voir src/contrat_predictions.py). Seuls les modeles de
`MODELES_AFFICHES` (src/donnees_demo.py) sont montres.
"""

import streamlit as st

from app import commun, figures
from src import donnees_demo as dd

st.title("Comparer les modèles")

liste = commun.inventaire()
sous_jeux = dd.sous_jeux_disponibles(liste)
if len(sous_jeux) == 0:
    st.warning(f"Aucun fichier de prédictions. Lancer : `{commun.COMMANDE_PREPARATION}`", icon=":material/folder_off:")
    st.stop()

# choix du sous-jeu (garde d'une visite a l'autre)
commun.restaurer_choix("comparaison_sous_jeu", sous_jeux)
sous_jeu = st.selectbox("Sous-jeu", sous_jeux, key="comparaison_sous_jeu", format_func=dd.description_sous_jeu)
commun.memoriser_choix("comparaison_sous_jeu")

tableau = commun.comparaison(sous_jeu)
seuils = dd.seuils_metriques()

# titres des colonnes (seuils lus dans le code du contrat, jamais ecrits en dur)
titres = {}
titres["rmse"] = "RMSE test (cycles)"
titres["score_nasa"] = "Score NASA test"
titres["nb_surestimes"] = f"Surestimés de plus de {seuils['seuil_surestimation']:g} cycles"
titres["alertes_manquees"] = f"Alertes manquées (seuil {seuils['seuil_alerte']:g})"

# tableau : une ligne par modele
affiche = tableau[["nom_affiche", "rmse", "score_nasa", "nb_surestimes", "alertes_manquees"]].copy()
affiche["rmse"] = affiche["rmse"].round(2)
affiche["score_nasa"] = affiche["score_nasa"].round(0).astype(int)
affiche["nb_surestimes"] = affiche["nb_surestimes"].astype(int)
affiche["alertes_manquees"] = affiche["alertes_manquees"].astype(int)
noms_colonnes = {"nom_affiche": "Modèle"}
for colonne in titres:
    noms_colonnes[colonne] = titres[colonne]
format_rmse = {titres["rmse"]: st.column_config.NumberColumn(format="%.2f")}  # deux decimales, comme le graphique
st.dataframe(affiche.rename(columns=noms_colonnes), hide_index=True, width="stretch", column_config=format_rmse)

# graphique : une barre par modele, un petit diagramme par metrique
st.plotly_chart(figures.figure_comparaison(tableau, titres), key="comparaison_barres", config=figures.CONFIG_PLOTLY)

# une ligne de legende : ce que mesurent ces chiffres et d'ou ils viennent
nb_moteurs = int(tableau["nb_moteurs"].max())
st.caption(
    f"Dernier cycle observé des {nb_moteurs} moteurs test de {sous_jeu} ; plus petit = mieux. "
    f"Source : `sorties/predictions/{sous_jeu}__<modèle>.json`."
)
