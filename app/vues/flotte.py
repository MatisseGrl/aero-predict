"""
Page 3 du tableau de bord : tous les moteurs test d'un sous-jeu au dernier cycle observe, tries par urgence.

On y trouve : un controle de coherence (metriques recalculees depuis le fichier = metadonnees), les
indicateurs de la flotte (memes formules que la commande `flotte` de src/demo.py), le tableau des
moteurs filtrable par statut, le nuage RUL predite / RUL vraie et l'histogramme des ecarts. Un moteur
selectionne dans le tableau peut etre ouvert dans la page "Moteur en direct".
"""

import streamlit as st

from app import commun, figures
from src import donnees_demo as dd
from src.contrat_predictions import PLAFOND_RUL
from src.demo import SEUIL_ALERTE_DEFAUT, SEUIL_SURESTIMATION

# colonnes affichees dans le tableau de la flotte : colonne interne -> titre affiche
COLONNES_AFFICHEES = {
    "moteur": "Moteur",
    "statut_affiche": "Statut",
    "rul_predite": "RUL prédite",
    "rul_vraie": "RUL vraie",
    "ecart": "Écart (prédite − vraie)",
    "nb_cycles": "Cycles observés",
    "cycle_premiere_alerte": "1re alerte (cycle)",
    "rul_vraie_premiere_alerte": "RUL vraie à la 1re alerte",
}

# couleur de fond de la colonne "Statut" (rouge reserve a l'alerte manquee)
STYLES_STATUTS = {}
for statut_interne in figures.COULEURS_STATUTS:
    STYLES_STATUTS[dd.LIBELLES_STATUTS[statut_interne]] = (
        f"background-color: {figures.COULEURS_STATUTS[statut_interne]}; color: white; font-weight: 600"
    )


def style_statut(valeur):
    """
    Donne le style CSS d'une cellule "Statut" (fond colore selon le statut).

    Parameters
    ----------
    valeur : str
        Le libelle du statut affiche.

    Returns
    -------
    str
        Le style CSS ("" si le statut est inconnu).
    """
    return STYLES_STATUTS.get(valeur, "")


def libelle_filtre(statut, comptes):
    """
    Libelle d'une pastille de filtre : le statut et son nombre de moteurs ("ALERTE MANQUÉE (7)").

    Parameters
    ----------
    statut : str
        Le statut (src/demo.py).
    comptes : dict
        statut -> nombre de moteurs.

    Returns
    -------
    str
        Le libelle.
    """
    return f"{dd.LIBELLES_STATUTS[statut]} ({comptes[statut]})"


def ouvrir_en_direct(sous_jeu, moteur):
    """
    Prepare la page "Moteur en direct" sur un moteur donne, puis y bascule.

    Parameters
    ----------
    sous_jeu : str
        Le sous-jeu.
    moteur : int
        Le numero du moteur test.

    Returns
    -------
    None
    """
    # les cles des widgets de l'autre page sont oubliees au changement de page : on passe par les memos
    for cle in ["direct_sous_jeu", "direct_moteur"]:
        if cle in st.session_state:
            del st.session_state[cle]
    st.session_state["memo_direct_sous_jeu"] = sous_jeu
    st.session_state["memo_direct_moteur"] = int(moteur)
    st.switch_page("vues/moteur_en_direct.py")


st.title("Flotte : tous les moteurs test au dernier cycle observé")
st.caption(
    "Une ligne par moteur du jeu de test, à son dernier cycle observé (là où la RUL vraie est fournie "
    "par la NASA), triée par urgence : la plus petite RUL prédite en haut."
)

liste = commun.inventaire()
if len(liste) == 0:
    commun.message_aucun_fichier()
    st.stop()

# ---------------------------------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------------------------------

sous_jeux = dd.sous_jeux_disponibles(liste)
colonne_sj, colonne_modele, colonne_seuil = st.columns([1, 1.6, 1.6])
with colonne_sj:
    commun.restaurer_choix("flotte_sous_jeu", sous_jeux)
    sous_jeu = st.selectbox("Sous-jeu", sous_jeux, key="flotte_sous_jeu")
    commun.memoriser_choix("flotte_sous_jeu")
modeles = dd.modeles_disponibles(liste, sous_jeu)
with colonne_modele:
    commun.restaurer_choix("flotte_modele", modeles)
    if "flotte_modele" not in st.session_state and "xgboost_retenu" in modeles:
        st.session_state["flotte_modele"] = "xgboost_retenu"  # par defaut : le modele retenu en Phase 4
    modele = st.selectbox("Modèle", modeles, key="flotte_modele", format_func=dd.nom_modele)
    commun.memoriser_choix("flotte_modele")
with colonne_seuil:
    commun.restaurer_choix("seuil_alerte", list(range(5, 101, 5)))
    if "seuil_alerte" not in st.session_state:
        st.session_state["seuil_alerte"] = int(SEUIL_ALERTE_DEFAUT)
    seuil_alerte = st.slider(
        "Seuil d'alerte (cycles) — hypothèse de travail", min_value=5, max_value=100, step=5, key="seuil_alerte",
        help="ALERTE si la RUL prédite est inférieure ou égale à ce seuil (règle de src/demo.py).",
    )
    commun.memoriser_choix("seuil_alerte")

df, meta = commun.predictions(sous_jeu, modele)
tableau, resume = commun.flotte(sous_jeu, modele, seuil_alerte)

# ---------------------------------------------------------------------------------------------------
# Controle de coherence : les chiffres ne sont pas recopies, ils se recalculent depuis le fichier
# ---------------------------------------------------------------------------------------------------

controle = dd.controler_coherence(df, meta)
if controle["coherent"]:
    st.success(
        f"Contrôle de cohérence : les {controle['nb_metriques']} métriques de test recalculées à partir du "
        f"fichier ({commun.fr(len(df), 0)} lignes, {df['unit_number'].nunique()} moteurs) sont égales à celles "
        f"de ses métadonnées (écart absolu maximal : {controle['ecart_max']:.1e}).",
        icon=":material/task_alt:",
    )
else:
    st.error(
        "Contrôle de cohérence : des métriques recalculées depuis le fichier diffèrent des métadonnées "
        "(détail ci-dessous).", icon=":material/error:",
    )
with st.expander("Détail du contrôle de cohérence (fonction metriques_dernier_cycle du contrat)"):
    st.dataframe(controle["tableau"], hide_index=True, width="stretch")
    st.caption(
        f"Métadonnées calculées avec les seuils par défaut du contrat ({SEUIL_ALERTE_DEFAUT:g} et "
        f"{SEUIL_SURESTIMATION} cycles), indépendamment du curseur ci-dessus."
    )

# ---------------------------------------------------------------------------------------------------
# Indicateurs de la flotte (seuil choisi)
# ---------------------------------------------------------------------------------------------------

indicateurs = [
    ("Moteurs test", str(resume["nb_moteurs"]), "Moteurs du jeu de test de ce sous-jeu."),
    ("RMSE (cycles)", commun.fr(resume["rmse"], 2), "Erreur quadratique moyenne au dernier cycle, contre la RUL vraie brute."),
    ("Score NASA", commun.fr(resume["score_nasa"], 1), "Somme sur les moteurs ; pénalise plus les surestimations."),
    (f"Surestimés > {SEUIL_SURESTIMATION}", str(resume["nb_surestimes"]), "Moteurs dont la RUL prédite dépasse la vraie de plus de 20 cycles."),
    ("Alertes émises", str(resume["alertes_emises"]), f"RUL prédite ≤ {seuil_alerte} cycles."),
    ("Alertes correctes", str(resume["alertes_correctes"]), "RUL vraie et RUL prédite toutes deux sous le seuil."),
    ("Alertes manquées", str(resume["alertes_manquees"]), "RUL vraie ≤ seuil mais RUL prédite > seuil : le cas dangereux."),
    ("Fausses alertes", str(resume["fausses_alertes"]), "Alerte alors que la RUL vraie dépasse le seuil."),
]
# deux lignes de quatre cartes : precision du modele, puis decisions d'alerte
for debut_ligne in [0, 4]:
    colonnes = st.columns(4)
    for position in range(4):
        libelle, valeur, aide = indicateurs[debut_ligne + position]
        with colonnes[position]:
            st.metric(libelle, valeur, border=True, help=aide)
st.caption(
    f"Moteurs qu'il aurait fallu arrêter (RUL vraie ≤ {seuil_alerte}) : {resume['nb_sous_seuil']}. "
    "RMSE, score NASA et surestimations ne dépendent pas du seuil d'alerte."
)

# ---------------------------------------------------------------------------------------------------
# Tableau des moteurs, filtrable par statut
# ---------------------------------------------------------------------------------------------------

comptes = dd.compter_statuts(tableau)


def libelle_avec_compte(statut):
    """
    Libelle d'une pastille de filtre pour le tableau courant (voir `libelle_filtre`).

    Parameters
    ----------
    statut : str
        Le statut.

    Returns
    -------
    str
        "ALERTE MANQUÉE (7)", par exemple.
    """
    return libelle_filtre(statut, comptes)


statuts_presents = []
for statut in dd.ORDRE_STATUTS:
    if comptes[statut] > 0:
        statuts_presents.append(statut)
commun.restaurer_choix("flotte_filtre", statuts_presents)
if "flotte_filtre" not in st.session_state:
    st.session_state["flotte_filtre"] = list(statuts_presents)
filtre = st.pills(
    "Filtrer par statut", statuts_presents, selection_mode="multi", key="flotte_filtre",
    format_func=libelle_avec_compte,
)
commun.memoriser_choix("flotte_filtre")
if filtre is None:
    filtre = []

# lignes gardees par le filtre (ordre d'urgence conserve)
garder = []
for statut in tableau["statut"].tolist():
    garder.append(statut in filtre)
tableau_filtre = tableau[garder].copy()
tableau_filtre["statut_affiche"] = tableau_filtre["statut"].map(dd.LIBELLES_STATUTS)
tableau_affiche = tableau_filtre[list(COLONNES_AFFICHEES.keys())].rename(columns=COLONNES_AFFICHEES)

# mise en forme : statut colore, nombres arrondis
formats = {
    "RUL prédite": "{:.1f}",
    "RUL vraie": "{:.0f}",
    "Écart (prédite − vraie)": "{:+.1f}",
    "1re alerte (cycle)": "{:.0f}",
    "RUL vraie à la 1re alerte": "{:.0f}",
}
tableau_style = tableau_affiche.style.map(style_statut, subset=["Statut"]).format(formats, na_rep="—")

st.markdown(
    f"**{len(tableau_filtre)} moteurs affichés sur {len(tableau)}** — cliquer sur une ligne pour pouvoir "
    "ouvrir ce moteur dans la page « Moteur en direct »."
)
selection = st.dataframe(
    tableau_style, hide_index=True, width="stretch", height=380, on_select="rerun",
    selection_mode="single-row", key="flotte_tableau",
)
lignes_choisies = []
if selection is not None and "selection" in selection:
    lignes_choisies = selection["selection"].get("rows", [])
if len(lignes_choisies) > 0:
    moteur_choisi = int(tableau_affiche["Moteur"].iloc[lignes_choisies[0]])
    if st.button(f"Ouvrir le moteur {moteur_choisi} dans « Moteur en direct »", icon=":material/open_in_new:",
                 type="primary", key="flotte_ouvrir"):
        ouvrir_en_direct(sous_jeu, moteur_choisi)
commun.sources(f"sorties/predictions/{sous_jeu}__{modele}.csv.gz", f"sorties/predictions/{sous_jeu}__{modele}.json")

# ---------------------------------------------------------------------------------------------------
# Graphiques : nuage predite / vraie et histogramme des ecarts
# ---------------------------------------------------------------------------------------------------

colonne_nuage, colonne_histogramme = st.columns([1.15, 1])
with colonne_nuage:
    titre = f"{sous_jeu} — {dd.nom_modele(modele)} : RUL prédite contre RUL vraie ({resume['nb_moteurs']} moteurs)"
    st.plotly_chart(
        figures.figure_nuage_flotte(tableau, seuil_alerte, SEUIL_SURESTIMATION, titre),
        key="flotte_nuage", config=figures.CONFIG_PLOTLY,
    )
with colonne_histogramme:
    titre = f"Écarts au dernier cycle — {resume['nb_surestimes']} surestimations de plus de {SEUIL_SURESTIMATION} cycles"
    st.plotly_chart(
        figures.figure_histogramme_ecarts(tableau, SEUIL_SURESTIMATION, titre),
        key="flotte_histogramme", config=figures.CONFIG_PLOTLY,
    )
    nb_au_dela = int((tableau["rul_vraie"] > PLAFOND_RUL).sum())
    st.caption(
        f"Les modèles apprennent une RUL plafonnée à {PLAFOND_RUL} : les {nb_au_dela} moteurs dont la RUL "
        f"vraie dépasse {PLAFOND_RUL} au dernier cycle sont mécaniquement sous-estimés (écarts négatifs)."
    )
