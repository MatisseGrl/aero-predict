"""
Page 2 du tableau de bord (coeur de la demo) : rejeu cycle par cycle d'un moteur du jeu de test.

A chaque cycle "present" : la RUL predite par chaque modele choisi jusqu'a ce cycle, la RUL vraie (si
on la revele), la zone d'alerte, la premiere alerte de chaque modele, des cartes indicateurs (RUL
predite, ecart, statut d'alerte avec la regle `classer_alerte` de src/demo.py) et les capteurs bruts.

Le bouton Lecture fait avancer le temps tout seul : la zone qui depend du cycle courant est un
"fragment" Streamlit (`st.fragment(run_every=...)`), relance a intervalle regulier SANS relancer toute
la page (les selecteurs du haut ne bougent pas, l'application reste reactive).
"""

import streamlit as st

from app import commun, figures
from src import donnees_demo as dd
from src.demo import SEUIL_ALERTE_DEFAUT

# intervalle entre deux images du rejeu (secondes) ; la vitesse se regle par le nombre de cycles par image
INTERVALLE_LECTURE_S = 0.5
CYCLES_PAR_IMAGE = [1, 3, 6, 12]  # vitesses proposees : 2, 6, 12 et 24 cycles par seconde
CYCLES_PAR_IMAGE_DEFAUT = 6

# cles de session (etat de la page)
CLE_CYCLE = "direct_cycle"  # cycle courant (cle du curseur)
CLE_LECTURE = "direct_lecture"  # Vrai pendant la lecture automatique
CLE_IDENTITE = "direct_identite"  # (sous-jeu, moteur) affiche : sert a detecter un changement de moteur


def libelle_vitesse(cycles_par_image):
    """
    Ecrit la vitesse de rejeu correspondant a un nombre de cycles par image ("12 cycles/s").

    Parameters
    ----------
    cycles_par_image : int
        Nombre de cycles avances a chaque image.

    Returns
    -------
    str
        Le libelle affiche.
    """
    cycles_par_seconde = cycles_par_image / INTERVALLE_LECTURE_S
    return f"{cycles_par_seconde:g} cycles/s"


def basculer_lecture(premier_cycle, dernier_cycle):
    """
    Bouton Lecture / Pause : demarre ou arrete le rejeu (si on est a la fin, on repart du debut).

    Parameters
    ----------
    premier_cycle, dernier_cycle : int
        Les bornes du rejeu.

    Returns
    -------
    None
    """
    if st.session_state.get(CLE_LECTURE, False):
        st.session_state[CLE_LECTURE] = False
        return
    if st.session_state.get(CLE_CYCLE, dernier_cycle) >= dernier_cycle:
        st.session_state[CLE_CYCLE] = premier_cycle  # rejeu termine : on recommence au premier cycle
    st.session_state[CLE_LECTURE] = True


def aller_au_cycle(cycle):
    """
    Place le curseur sur un cycle donne et arrete la lecture (boutons "debut" et "fin").

    Parameters
    ----------
    cycle : int
        Le cycle vise.

    Returns
    -------
    None
    """
    st.session_state[CLE_CYCLE] = cycle
    st.session_state[CLE_LECTURE] = False


def afficher_carte_modele(modele, etat, reveler_verite):
    """
    Affiche la carte indicateurs d'un modele au cycle courant : RUL predite, ecart, statut, 1re alerte.

    Parameters
    ----------
    modele : str
        La cle du modele.
    etat : dict
        Le resultat de `etat_au_cycle`.
    reveler_verite : bool
        Faux : on n'affiche ni l'ecart ni le statut (ils utilisent la RUL vraie), seulement la decision.

    Returns
    -------
    None
    """
    couleur = figures.couleur_modele(modele)
    with st.container(border=True):
        st.markdown(
            f"<span style='color:{couleur};font-weight:700;font-size:1.05rem'>■ {dd.nom_modele(modele)}</span>",
            unsafe_allow_html=True,
        )
        delta = None
        sens = ""
        if reveler_verite:
            if etat["ecart"] > 0:  # le modele croit qu'il reste plus de cycles qu'en realite
                sens = "surestimation"
            elif etat["ecart"] < 0:  # le modele est plus pessimiste que la realite
                sens = "sous-estimation"
            else:
                sens = "exacte"
            delta = f"{commun.fr_signe(etat['ecart'])} vs vraie ({sens})"
        couleur_delta = "inverse"  # surestimation (ecart positif) en rouge : le sens dangereux
        if sens == "exacte":
            couleur_delta = "off"
        st.metric("RUL prédite", f"{commun.fr(etat['rul_predite'])} cycles", delta=delta, delta_color=couleur_delta)
        if reveler_verite:
            commun.badge_statut(etat["statut"])  # l'explication du statut est dans l'info-bulle du badge
        elif etat["alerte"]:
            st.badge("ALERTE émise", color="red")
        else:
            st.badge("pas d'alerte", color="gray")
        if etat["premiere_alerte"] is None:
            st.caption("1re alerte : aucune jusqu'à ce cycle.")
        else:
            cycle_alerte, rul_vraie_alerte = etat["premiere_alerte"]
            if reveler_verite:
                st.caption(f"1re alerte : cycle {cycle_alerte} (il restait alors {commun.fr(rul_vraie_alerte, 0)} cycles).")
            else:
                st.caption(f"1re alerte : cycle {cycle_alerte}.")


# ---------------------------------------------------------------------------------------------------
# Selection : sous-jeu, moteur, modeles, seuil
# ---------------------------------------------------------------------------------------------------

st.title("Moteur en direct")
st.caption(
    "Rejeu d'un moteur du jeu de test, cycle par cycle : à chaque instant, ce que prédit chaque modèle "
    "avec les seules mesures reçues jusque-là, et — si on la révèle — ce qui s'est vraiment passé."
)

liste = commun.inventaire()
if len(liste) == 0:
    commun.message_aucun_fichier()
    st.stop()

sous_jeux = dd.sous_jeux_disponibles(liste)
colonne_sj, colonne_moteur, colonne_modeles, colonne_seuil = st.columns([1, 1, 2.4, 1.6])

with colonne_sj:
    commun.restaurer_choix("direct_sous_jeu", sous_jeux)
    sous_jeu = st.selectbox("Sous-jeu", sous_jeux, key="direct_sous_jeu")
    commun.memoriser_choix("direct_sous_jeu")

modeles = dd.modeles_disponibles(liste, sous_jeu)
df_reference, meta_reference = commun.predictions(sous_jeu, modeles[0])  # sert a lister les moteurs
moteurs = dd.lister_moteurs(df_reference)

with colonne_moteur:
    commun.restaurer_choix("direct_moteur", moteurs)
    if "direct_moteur" not in st.session_state:  # par defaut : le moteur le plus proche de la panne
        st.session_state["direct_moteur"] = dd.moteur_le_plus_proche_de_la_panne(df_reference)
    moteur = st.selectbox(
        "Moteur test", moteurs, key="direct_moteur",
        help="Par défaut : le moteur test le plus proche de la panne au dernier cycle observé.",
    )
    commun.memoriser_choix("direct_moteur")

with colonne_modeles:
    commun.restaurer_choix("direct_modeles", modeles)
    if "direct_modeles" not in st.session_state:
        st.session_state["direct_modeles"] = list(modeles)  # par defaut : tous les modeles disponibles
    modeles_choisis = st.multiselect(
        "Modèles superposés", modeles, key="direct_modeles", format_func=dd.nom_modele,
        placeholder="Choisir au moins un modèle",
    )
    commun.memoriser_choix("direct_modeles")

with colonne_seuil:
    commun.restaurer_choix("seuil_alerte", list(range(5, 101, 5)))
    if "seuil_alerte" not in st.session_state:
        st.session_state["seuil_alerte"] = int(SEUIL_ALERTE_DEFAUT)
    seuil_alerte = st.slider(
        "Seuil d'alerte (cycles) — hypothèse de travail", min_value=5, max_value=100, step=5, key="seuil_alerte",
        help="ALERTE si la RUL prédite est inférieure ou égale à ce seuil (règle de src/demo.py).",
    )
    commun.memoriser_choix("seuil_alerte")

if len(modeles_choisis) == 0:
    st.info("Choisir au moins un modèle à afficher.")
    st.stop()

# trajectoire de ce moteur pour chaque modele choisi (meme moteur, memes cycles)
trajectoires = []
for modele in modeles_choisis:
    df_modele, meta_modele = commun.predictions(sous_jeu, modele)
    try:
        trajectoires.append((modele, dd.extraire_trajectoire(df_modele, moteur)))
    except ValueError as erreur:
        st.warning(f"{dd.nom_modele(modele)} : {erreur}")
if len(trajectoires) == 0:
    st.stop()

reference = trajectoires[0][1]
premier_cycle = int(reference["time_cycles"].min())
dernier_cycle = int(reference["time_cycles"].max())

# echelle verticale fixe pour tout le rejeu : la plus grande valeur de RUL vraie ou predite
rul_max = max(float(reference["rul_vraie"].max()), float(reference["rul_vraie_cappee"].max()), float(seuil_alerte))
for modele, trajectoire in trajectoires:
    rul_max = max(rul_max, float(trajectoire["rul_predite"].max()))

# nouveau moteur choisi : on affiche toute la trajectoire, lecture arretee
if st.session_state.get(CLE_IDENTITE) != (sous_jeu, moteur):
    st.session_state[CLE_IDENTITE] = (sous_jeu, moteur)
    st.session_state[CLE_CYCLE] = dernier_cycle
    st.session_state[CLE_LECTURE] = False
if CLE_CYCLE not in st.session_state:  # retour sur la page : la cle du curseur a ete oubliee
    st.session_state[CLE_CYCLE] = dernier_cycle
if CLE_LECTURE not in st.session_state:
    st.session_state[CLE_LECTURE] = False
# cycle hors bornes (ex. moteur plus court) : on le ramene dans les bornes
if st.session_state[CLE_CYCLE] > dernier_cycle:
    st.session_state[CLE_CYCLE] = dernier_cycle
if st.session_state[CLE_CYCLE] < premier_cycle:
    st.session_state[CLE_CYCLE] = premier_cycle

# ---------------------------------------------------------------------------------------------------
# Commandes du rejeu : lecture / pause, vitesse, verite
# ---------------------------------------------------------------------------------------------------

colonne_lecture, colonne_debut, colonne_fin, colonne_vitesse, colonne_verite = st.columns(
    [1.3, 1.2, 1.2, 2.6, 1.8], vertical_alignment="bottom"
)
with colonne_lecture:
    if st.session_state[CLE_LECTURE]:
        st.button("Pause", icon=":material/pause:", type="primary", key="bouton_lecture",
                  on_click=basculer_lecture, args=(premier_cycle, dernier_cycle), width="stretch")
    else:
        st.button("Lecture", icon=":material/play_arrow:", type="primary", key="bouton_lecture",
                  on_click=basculer_lecture, args=(premier_cycle, dernier_cycle), width="stretch")
with colonne_debut:
    st.button("Début", icon=":material/skip_previous:", key="bouton_debut",
              on_click=aller_au_cycle, args=(premier_cycle,), width="stretch")
with colonne_fin:
    st.button("Fin", icon=":material/skip_next:", key="bouton_fin",
              on_click=aller_au_cycle, args=(dernier_cycle,), width="stretch")
with colonne_vitesse:
    commun.restaurer_choix("direct_vitesse", CYCLES_PAR_IMAGE)
    if "direct_vitesse" not in st.session_state:
        st.session_state["direct_vitesse"] = CYCLES_PAR_IMAGE_DEFAUT
    cycles_par_image = st.segmented_control(
        "Vitesse du rejeu", CYCLES_PAR_IMAGE, key="direct_vitesse", format_func=libelle_vitesse, required=True,
    )
    commun.memoriser_choix("direct_vitesse")
with colonne_verite:
    commun.restaurer_choix("direct_verite", [True, False])
    if "direct_verite" not in st.session_state:
        st.session_state["direct_verite"] = True
    reveler_verite = st.toggle(
        "Révéler la vérité", key="direct_verite",
        help="RUL vraie = RUL_FD00X.txt au dernier cycle + cycles restants : connue après coup seulement.",
    )
    commun.memoriser_choix("direct_verite")

if cycles_par_image is None:  # securite (required=True l'evite normalement)
    cycles_par_image = CYCLES_PAR_IMAGE_DEFAUT

# donnees capteurs brutes de ce moteur (et regime de chaque cycle, lu dans le fichier de predictions)
donnees_brutes = commun.donnees_brutes_test(sous_jeu)
donnees_moteur = donnees_brutes[donnees_brutes["unit_number"] == moteur].sort_values("time_cycles")
regime_par_cycle = {}
for position in range(len(reference)):
    regime_par_cycle[int(reference["time_cycles"].iloc[position])] = int(reference["regime"].iloc[position])
regimes_moteur = []
for cycle in donnees_moteur["time_cycles"].tolist():
    regimes_moteur.append(regime_par_cycle.get(int(cycle), 0))
capteurs_possibles = dd.capteurs_variables(donnees_brutes)

# ---------------------------------------------------------------------------------------------------
# Zone animee : tout ce qui depend du cycle courant (fragment relance pendant la lecture)
# ---------------------------------------------------------------------------------------------------

if st.session_state[CLE_LECTURE]:
    intervalle = INTERVALLE_LECTURE_S
else:
    intervalle = None  # pas de relance automatique hors lecture


@st.fragment(run_every=intervalle)
def zone_rejeu():
    """
    Affiche le curseur du cycle courant, les cartes indicateurs, la courbe de RUL et les capteurs.

    Pendant la lecture, chaque relance avance le cycle courant de `cycles_par_image` cycles ; au
    dernier cycle, la lecture s'arrete et la page entiere est relancee (pour arreter le minuteur).

    Returns
    -------
    None
    """
    if st.session_state.get(CLE_LECTURE, False):
        nouveau_cycle = min(int(st.session_state[CLE_CYCLE]) + int(cycles_par_image), dernier_cycle)
        st.session_state[CLE_CYCLE] = nouveau_cycle
        if nouveau_cycle >= dernier_cycle:  # fin du rejeu
            st.session_state[CLE_LECTURE] = False
            st.rerun()  # relance complete : arrete le minuteur et remet le bouton sur "Lecture"

    cycle_courant = st.slider(
        "Cycle courant (le « présent » du rejeu)", min_value=premier_cycle, max_value=dernier_cycle, key=CLE_CYCLE,
    )

    # etat de chaque modele au cycle courant
    etats = {}
    premieres_alertes = {}
    for modele, trajectoire in trajectoires:
        etat = dd.etat_au_cycle(trajectoire, cycle_courant, seuil_alerte)
        etats[modele] = etat
        premieres_alertes[modele] = etat["premiere_alerte"]

    # courbe principale (a gauche) et cartes indicateurs (a droite)
    colonne_graphique, colonne_cartes = st.columns([3.1, 1.25])
    with colonne_graphique:
        titre = f"Moteur test {moteur} ({sous_jeu}) : RUL prédite, cycle {cycle_courant} sur {dernier_cycle}"
        figure = figures.figure_moteur_en_direct(
            trajectoires, cycle_courant, seuil_alerte, reveler_verite, titre, dernier_cycle, rul_max, premieres_alertes
        )
        st.plotly_chart(figure, key="graphique_rul", config=figures.CONFIG_PLOTLY)
    with colonne_cartes:
        colonne_cycle, colonne_verite = st.columns(2)
        with colonne_cycle:
            st.metric(f"Cycle (sur {dernier_cycle})", str(cycle_courant), border=True,
                      help=f"{dernier_cycle} = dernier cycle observé de ce moteur dans le jeu de test.")
        with colonne_verite:
            etat_reference = etats[trajectoires[0][0]]
            if reveler_verite:
                st.metric("RUL vraie", f"{commun.fr(etat_reference['rul_vraie'], 0)}", border=True,
                          help="Cycles restants avant la panne, connus seulement après coup.")
            else:
                st.metric("RUL vraie", "masquée", border=True)
        for modele, trajectoire in trajectoires:  # une carte par modele, empilees
            afficher_carte_modele(modele, etats[modele], reveler_verite)

    # capteurs bruts
    commun.restaurer_choix("direct_capteurs", capteurs_possibles)
    if "direct_capteurs" not in st.session_state:
        st.session_state["direct_capteurs"] = dd.choisir_capteurs_par_defaut(capteurs_possibles)
    capteurs = st.multiselect(
        "Capteurs bruts affichés", capteurs_possibles, key="direct_capteurs", max_selections=4,
        help="Capteurs non constants de ce sous-jeu. Noms : ceux des fichiers NASA (voir l'encadré ci-dessous).",
    )
    commun.memoriser_choix("direct_capteurs")
    if len(capteurs) > 0:
        figure_capteurs = figures.figure_capteurs(donnees_moteur, capteurs, cycle_courant, dernier_cycle, regimes_moteur)
        st.plotly_chart(figure_capteurs, key="graphique_capteurs", config=figures.CONFIG_PLOTLY)
        if len(set(regimes_moteur)) > 1:
            st.caption(
                "Ce sous-jeu a plusieurs régimes de vol : les valeurs brutes sautent d'un régime à l'autre "
                "(points colorés par régime, numéro de régime lu dans le fichier de prédictions)."
            )


zone_rejeu()

# ---------------------------------------------------------------------------------------------------
# Aide a la lecture
# ---------------------------------------------------------------------------------------------------

with st.expander("Comment lire cette page", icon=":material/help:"):
    st.markdown(
        f"- **Zone rouge** : RUL ≤ {seuil_alerte} cycles. Quand la courbe d'un modèle y entre, il déclenche "
        "une alerte (triangle : première alerte).\n"
        "- **RUL vraie** (noir) : RUL fournie par la NASA au dernier cycle observé, augmentée d'un cycle "
        "par cycle en remontant le temps. **Plafonnée** (pointillés) : ce que les modèles ont appris à "
        "prédire (plafond de la Phase 2).\n"
        "- **Statut** : *alerte correcte* (RUL vraie et prédite sous le seuil), **ALERTE MANQUÉE** (le moteur "
        "est sous le seuil mais le modèle n'alerte pas : le cas dangereux), *fausse alerte* (maintenance trop "
        "tôt, sans risque), *rien à signaler*. Même règle que `classer_alerte` de `src/demo.py`.\n"
        "- Ces prédictions sont lues dans des fichiers calculés à l'avance pour tous les cycles "
        "(`sorties/predictions/`) ; la page « Et si… » montre une inférence calculée en direct."
    )
    commun.sources(f"sorties/predictions/{sous_jeu}__<modèle>.csv.gz", f"data/test_{sous_jeu}.txt")

with st.expander("Que mesurent ces capteurs ?", icon=":material/sensors:"):
    st.markdown(
        "Les fichiers NASA nomment les colonnes *sensor measurement 1 à 21* (`data/readme.txt`), sans nom "
        "physique. Le tableau 2 de Saxena et al. (2008, `data/Damage Propagation Modeling.pdf`) liste 21 "
        "grandeurs fournies comme données capteurs, mais **ni l'article ni le readme ne disent que la "
        "colonne sensor_N correspond à la N-ième ligne de ce tableau** : l'application garde donc le nom "
        "`sensor_N`. Le tableau ci-dessous est donné à titre d'information."
    )
    st.dataframe(dd.tableau_grandeurs_saxena(), hide_index=True, width="stretch")
