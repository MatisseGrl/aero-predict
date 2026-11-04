"""
Application de demo du prototype Aero Predict (application web locale Streamlit), deux pages.

Lancement, depuis la racine du depot :

    streamlit run app/tableau_de_bord.py

Ce fichier ne fait que quatre choses : regler le calcul parallele sur un seul coeur (voir ci-dessous), rendre
`import src...` et `import app...` possibles, regler la page (titre, largeur) et declarer les deux pages
(navigation multipage `st.navigation`) :

1. "Tester un moteur" (app/vues/prediction_en_direct.py, page par defaut) : prediction de la RUL en direct ;
2. "Comparer les modeles" (app/vues/comparaison.py) : metriques de test des fichiers sorties/predictions/.
"""

import os
import sys

# ---------------------------------------------------------------------------------------------------
# Calcul parallele : un seul coeur, AVANT tout import de numpy / xgboost / torch
# ---------------------------------------------------------------------------------------------------
# La page "Tester un moteur" fait calculer XGBoost, LightGBM et PyTorch dans le meme processus. Deux bibliotheques
# OpenMP (calcul parallele) y coexistent alors, et un calcul multi-coeurs de l'une peut attendre indefiniment les fils
# de l'autre (blocage constate le 2026-11-04 dans la suite de tests, voir src/deep_learning.py). OMP_NUM_THREADS=1,
# lu par chaque bibliotheque OpenMP a son demarrage, fixe a 1 le nombre de coeurs par defaut de toutes (PyTorch,
# scikit-learn, XGBoost sans n_jobs explicite). Il doit etre ecrit avant leur premier chargement : d'ou sa place,
# tout en haut du point d'entree. src/inference.py regle en plus PyTorch et les modeles a arbres sur 1 coeur.
# Exception : sous pytest, on ne modifie pas l'environnement de toute la suite de tests ; src/inference.py y suffit.
if "pytest" not in sys.modules:
    os.environ["OMP_NUM_THREADS"] = "1"

# racine du depot = dossier parent de app/ ; on l'ajoute au chemin d'import (pour "import src...")
DOSSIER_APP = os.path.dirname(os.path.abspath(__file__))
RACINE_DEPOT = os.path.dirname(DOSSIER_APP)
if RACINE_DEPOT not in sys.path:  # evite d'ajouter deux fois le meme dossier
    sys.path.insert(0, RACINE_DEPOT)

import streamlit as st  # noqa: E402

# reglages de la page : titre de l'onglet, mise en page large (graphiques plus lisibles au projecteur)
st.set_page_config(
    page_title="Aero Predict — démo du prototype",
    page_icon=":material/flight:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# les deux pages, dans l'ordre de la demo (chemins relatifs a ce fichier)
PAGES = [
    st.Page("vues/prediction_en_direct.py", title="Tester un moteur", icon=":material/flight_takeoff:", default=True),
    st.Page("vues/comparaison.py", title="Comparer les modèles", icon=":material/compare_arrows:", url_path="comparaison"),
]

page = st.navigation(PAGES, position="top")  # menu horizontal en haut, pas de barre laterale
page.run()
