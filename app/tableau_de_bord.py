"""
Tableau de bord de demo du prototype Aero Predict (application web locale Streamlit).

Lancement, depuis la racine du depot :

    streamlit run app/tableau_de_bord.py

Ce fichier ne fait que trois choses : rendre `import src...` et `import app...` possibles, regler la
page (titre, largeur), et declarer les six pages (navigation multipage `st.navigation`). Le contenu de
chaque page est dans app/vues/, la lecture des donnees dans src/donnees_demo.py, les figures dans
app/figures.py et les outils communs (caches, messages) dans app/commun.py.

Les fichiers de predictions sont produits par `python scripts/preparer_demo.py` ; l'application
fonctionne avec n'importe quel sous-ensemble de fichiers presents.
"""

import os
import sys

# racine du depot = dossier parent de app/ ; on l'ajoute au chemin d'import (pour "import src...")
DOSSIER_APP = os.path.dirname(os.path.abspath(__file__))
RACINE_DEPOT = os.path.dirname(DOSSIER_APP)
if RACINE_DEPOT not in sys.path:  # evite d'ajouter deux fois le meme dossier
    sys.path.insert(0, RACINE_DEPOT)

import streamlit as st  # noqa: E402

from app import commun  # noqa: E402
from src import donnees_demo as dd  # noqa: E402

# reglages de la page : titre de l'onglet, mise en page large (graphiques plus lisibles au projecteur)
st.set_page_config(
    page_title="Aero Predict — démo du prototype",
    page_icon=":material/flight:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# les six pages, dans l'ordre de la demo (chemins relatifs a ce fichier)
PAGES = [
    st.Page("vues/accueil.py", title="Accueil — le problème", icon=":material/home:", default=True),  # adresse "/"
    st.Page("vues/moteur_en_direct.py", title="Moteur en direct", icon=":material/monitor_heart:", url_path="moteur"),
    st.Page("vues/flotte.py", title="Flotte", icon=":material/view_list:", url_path="flotte"),
    st.Page("vues/comparaison.py", title="Comparaison des modèles", icon=":material/compare_arrows:", url_path="comparaison"),
    st.Page("vues/simulation.py", title="Et si… (simulation)", icon=":material/science:", url_path="simulation"),
    st.Page("vues/fiabilite.py", title="Fiabilité du prototype", icon=":material/verified:", url_path="fiabilite"),
]


def afficher_barre_laterale():
    """
    Affiche dans la barre laterale l'etat des fichiers de predictions et un bouton pour les relire.

    Returns
    -------
    None
    """
    with st.sidebar:
        st.subheader("Fichiers de prédictions")
        dossier = dd.dossier_predictions()
        liste = commun.inventaire()
        st.caption(f"Dossier lu : `{commun.chemin_lisible(dossier)}`")
        if len(liste) == 0:
            st.write("Aucun fichier.")
            st.code(commun.COMMANDE_PREPARATION, language="bash")
        else:
            for sous_jeu in dd.sous_jeux_disponibles(liste):
                noms = []
                for modele in dd.modeles_disponibles(liste, sous_jeu):
                    noms.append(dd.nom_modele(modele))
                st.markdown(f"**{sous_jeu}** : {', '.join(noms)}")
        # vider les caches : utile si des fichiers ont ete supprimes pendant que l'application tourne
        if st.button("Relire tous les fichiers", icon=":material/refresh:", key="relire_fichiers"):
            st.cache_data.clear()
            st.rerun()
        st.divider()
        st.caption("Lancement : `streamlit run app/tableau_de_bord.py` (depuis la racine du dépôt)")


afficher_barre_laterale()
page = st.navigation(PAGES, position="top")  # menu horizontal en haut : plus de place pour les graphiques
page.run()
