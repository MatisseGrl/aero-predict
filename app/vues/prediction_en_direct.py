"""
Page "Tester un moteur" (page par defaut) : la RUL d'un moteur jamais vu, predite EN DIRECT cycle par cycle.

Deroule : on choisit un sous-jeu et un moteur du jeu de TEST (tire au hasard, choisi par son numero, ou importe
depuis un fichier) ; Lecture fait arriver les cycles un par un ; a chaque nouveau cycle t, chaque modele predit
la RUL a partir des seules mesures 1..t (`src.inference.predire_au_cycle`) : rien n'est lu dans un fichier de
predictions. La RUL vraie reste cachee jusqu'au bouton "Reveler la vraie RUL".

La lecture automatique est un "fragment" Streamlit (`st.fragment(run_every=...)`), relance a intervalle regulier
sans relancer toute la page (meme mecanique que l'ancienne page de rejeu).
"""

import streamlit as st

from app import commun, figures
from src import donnees_demo as dd
from src import inference as inf
from src.contrat_predictions import PLAFOND_RUL, SOUS_JEUX
from src.demo import SEUIL_ALERTE_DEFAUT, classer_alerte

# rythme de la lecture : une image toutes les 0,5 s, 6 nouveaux cycles calcules par image (12 cycles par seconde)
INTERVALLE_LECTURE_S = 0.5
CYCLES_PAR_IMAGE = 6

# cles de session (etat de la page)
CLE_MOTEUR = "live_moteur"  # dict du moteur courant (mesures, origine, numero, RUL finale), ou absent
CLE_PREDICTIONS = "live_predictions"  # dict modele -> liste des RUL predites aux cycles 1..t
CLE_TEMPS = "live_temps"  # resultat de predire_au_cycle du dernier pas calcule (temps en ms)
CLE_CYCLE = "live_cycle"  # dernier cycle recu (0 : aucun)
CLE_LECTURE = "live_lecture"  # Vrai pendant la lecture automatique
CLE_REVELE = "live_revele"  # Vrai une fois la verite revelee
CLE_IDENTITE = "live_identite"  # (sous-jeu, jeton du moteur) des predictions en memoire
CLE_JETON = "live_jeton"  # compteur : un nouveau numero a chaque nouveau moteur
CLE_GRAINE = "live_graine_tirage"  # graine du tirage, seulement si un test la fixe (sinon tirage libre)
CLE_NUMERO = "live_numero"  # liste deroulante du numero de moteur
CLE_ERREUR_IMPORT = "live_erreur_import"  # message d'erreur du dernier import


# ---------------------------------------------------------------------------------------------------
# Actions des boutons (fonctions de rappel : executees avant le reaffichage de la page)
# ---------------------------------------------------------------------------------------------------

def remettre_a_zero():
    """
    Oublie les predictions calculees : on repart du cycle 0 (aucun cycle recu), lecture arretee.

    Returns
    -------
    None
    """
    st.session_state[CLE_PREDICTIONS] = {}
    st.session_state[CLE_TEMPS] = None
    st.session_state[CLE_CYCLE] = 0
    st.session_state[CLE_LECTURE] = False


def installer_moteur(moteur, origine, nom_fichier=""):
    """
    Installe un nouveau moteur a tester : predictions oubliees, verite cachee, lecture arretee.

    Parameters
    ----------
    moteur : dict
        "sous_jeu", "moteur" (numero ou None), "mesures", "rul_finale" (ou None si inconnue).
    origine : str
        "tirage", "choix" ou "import".
    nom_fichier : str, optionnel
        Le nom du fichier importe.

    Returns
    -------
    None
    """
    installe = {}
    for cle in ["sous_jeu", "moteur", "mesures", "rul_finale"]:
        installe[cle] = moteur[cle]
    installe["origine"] = origine
    installe["nom_fichier"] = nom_fichier
    installe["nb_cycles"] = int(len(moteur["mesures"]))
    st.session_state[CLE_JETON] = st.session_state.get(CLE_JETON, 0) + 1
    installe["jeton"] = st.session_state[CLE_JETON]
    st.session_state[CLE_MOTEUR] = installe
    st.session_state[CLE_REVELE] = False
    remettre_a_zero()


def tirer_au_hasard(sous_jeu):
    """
    Bouton "Tirer un moteur au hasard" : un moteur du jeu de test, identite masquee ("moteur mystere").

    Parameters
    ----------
    sous_jeu : str
        Le sous-jeu choisi.

    Returns
    -------
    None
    """
    moteur = inf.tirer_moteur_test(sous_jeu, graine=st.session_state.get(CLE_GRAINE))
    installer_moteur(moteur, "tirage")
    st.session_state[CLE_NUMERO] = None  # la liste des numeros ne montre pas le moteur tire


def choisir_numero(sous_jeu):
    """
    Liste deroulante du numero : charge le moteur test choisi.

    Parameters
    ----------
    sous_jeu : str
        Le sous-jeu choisi.

    Returns
    -------
    None
    """
    numero = st.session_state.get(CLE_NUMERO)
    if numero is not None:
        installer_moteur(inf.mesures_moteur_test(sous_jeu, int(numero)), "choix")


def importer_fichier(sous_jeu):
    """
    Bouton "Tester ce fichier" : lit et valide le fichier importe (message clair s'il est invalide).

    Parameters
    ----------
    sous_jeu : str
        Le sous-jeu choisi (modeles utilises pour ce fichier).

    Returns
    -------
    None
    """
    st.session_state[CLE_ERREUR_IMPORT] = ""
    fichier = st.session_state.get("live_fichier")
    if fichier is None:
        st.session_state[CLE_ERREUR_IMPORT] = "Choisir d'abord un fichier."
        return
    try:
        mesures = inf.lire_fichier_mesures(fichier.getvalue())
    except inf.ErreurFichierMesures as erreur:
        st.session_state[CLE_ERREUR_IMPORT] = str(erreur)
        return
    moteur = {}
    moteur["sous_jeu"] = sous_jeu
    moteur["moteur"] = int(mesures["unit_number"].iloc[0])
    moteur["mesures"] = mesures
    rul = st.session_state.get("live_rul_import")
    moteur["rul_finale"] = None
    if rul is not None:
        moteur["rul_finale"] = float(rul)
    installer_moteur(moteur, "import", nom_fichier=fichier.name)
    st.session_state[CLE_NUMERO] = None


def basculer_lecture(nb_cycles):
    """
    Bouton Lecture / Pause (a la fin du moteur, Lecture recommence au debut et recalcule tout).

    Parameters
    ----------
    nb_cycles : int
        Le nombre de cycles du moteur.

    Returns
    -------
    None
    """
    if st.session_state.get(CLE_LECTURE, False):
        st.session_state[CLE_LECTURE] = False
        return
    if st.session_state.get(CLE_CYCLE, 0) >= nb_cycles:
        remettre_a_zero()
    st.session_state[CLE_LECTURE] = True


def basculer_verite():
    """
    Bouton "Reveler la vraie RUL" / "Masquer la vraie RUL".

    Returns
    -------
    None
    """
    st.session_state[CLE_REVELE] = not st.session_state.get(CLE_REVELE, False)


# ---------------------------------------------------------------------------------------------------
# Calcul et affichage
# ---------------------------------------------------------------------------------------------------

def calculer_jusqu_a(inference, mesures, cycle_cible, modeles):
    """
    Calcule EN DIRECT les predictions manquantes de chaque modele, cycle par cycle, jusqu'a cycle_cible.

    A chaque cycle c, `predire_au_cycle` ne recoit que les mesures 1..c (il tronque lui-meme les mesures).

    Parameters
    ----------
    inference : dict
        Le resultat de `src.inference.charger_modeles`.
    mesures : pandas.DataFrame
        Les mesures brutes du moteur.
    cycle_cible : int
        Le dernier cycle a calculer.
    modeles : list de str
        Les modeles affiches.

    Returns
    -------
    None
    """
    predictions = st.session_state[CLE_PREDICTIONS]
    for modele in modeles:
        if modele not in predictions:
            predictions[modele] = []
    for cycle in range(1, cycle_cible + 1):
        manquants = []
        for modele in modeles:
            if len(predictions[modele]) < cycle:
                manquants.append(modele)
        if len(manquants) == 0:
            continue
        resultat = inf.predire_au_cycle(inference, mesures, cycle, manquants)
        for modele in manquants:
            predictions[modele].append(resultat["predictions"][modele])
        st.session_state[CLE_TEMPS] = resultat  # temps du dernier pas calcule


def nom_du_moteur(moteur, revele):
    """
    Le nom affiche du moteur, et sa provenance : "Moteur mystere" tant qu'un moteur tire au hasard n'est pas revele.

    Parameters
    ----------
    moteur : dict
        Le moteur installe.
    revele : bool
        Vrai si la verite a ete revelee.

    Returns
    -------
    (str, str)
        Le nom court (titre) et la provenance (une ligne).
    """
    source = f"data/test_{moteur['sous_jeu']}.txt"
    if moteur["origine"] == "import":
        return f"Fichier importé : {moteur['nom_fichier']}", f"modèles de {moteur['sous_jeu']}"
    if moteur["origine"] == "tirage" and not revele:
        return "Moteur mystère", f"tiré au hasard dans {source}"
    if moteur["origine"] == "tirage":
        return f"Moteur test n° {moteur['moteur']}", f"tiré au hasard dans {source}"
    return f"Moteur test n° {moteur['moteur']}", f"choisi dans {source}"


def afficher_carte(modele, predictions, cycle, verite, seuil_alerte, temps):
    """
    Carte compacte d'un modele : RUL predite au cycle courant, temps de calcul, alerte (et statut si revele).

    Parameters
    ----------
    modele : str
        La cle du modele.
    predictions : list de float
        Ses predictions aux cycles 1..t.
    cycle : int
        Le cycle courant t.
    verite : pandas.DataFrame ou None
        La verite des cycles 1..t, ou None si elle est cachee.
    seuil_alerte : float
        Le seuil d'alerte.
    temps : dict ou None
        Le resultat de `predire_au_cycle` du dernier pas (temps en ms).

    Returns
    -------
    None
    """
    couleur = figures.couleur_modele(modele)
    with st.container(border=True):
        st.markdown(f"<span style='color:{couleur};font-weight:700'>■ {dd.nom_modele(modele)}</span>",
                    unsafe_allow_html=True)
        if cycle == 0 or len(predictions) < cycle:
            st.metric("RUL prédite (cycles)", "—")
            return
        rul_predite = float(predictions[cycle - 1])
        delta = None
        if verite is not None:
            rul_vraie = float(verite["rul_vraie"].iloc[cycle - 1])
            delta = f"{commun.fr_signe(rul_predite - rul_vraie)} vs vraie"
        st.metric("RUL prédite (cycles)", commun.fr(rul_predite), delta=delta, delta_color="inverse")
        if verite is not None:
            alerte, statut = classer_alerte(rul_predite, rul_vraie, seuil_alerte)  # regle de src/demo.py
            commun.badge_statut(statut)
        elif rul_predite <= seuil_alerte:
            st.badge("ALERTE", color="red")
        else:
            st.badge("pas d'alerte", color="gray")
        if temps is not None and modele in temps["temps_ms"]:
            st.caption(f"{commun.fr(temps['temps_ms'][modele])} ms (+ {commun.fr(temps['temps_preparation_ms'])} ms de préparation)")


def afficher_preuves(inference, sous_jeu):
    """
    Encadre ferme "Pourquoi ce n'est pas truque ?" : quatre faits verifiables et les tests qui les prouvent.

    Parameters
    ----------
    inference : dict
        Le resultat de `charger_modeles`.
    sous_jeu : str
        Le sous-jeu.

    Returns
    -------
    None
    """
    contexte = inference["contexte"]
    with st.expander("Pourquoi ce n'est pas truqué ?", expanded=False, icon=":material/verified:"):
        st.markdown(
            f"- Modèles entraînés sur `data/train_{sous_jeu}.txt` seulement ({contexte['nb_moteurs_train']} moteurs) ; "
            f"les moteurs testés viennent de `data/test_{sous_jeu}.txt` (d'autres moteurs, `data/readme.txt`).\n"
            "- Au cycle t, les modèles ne reçoivent que les cycles 1 à t, préparés avec les paramètres appris sur "
            "le train (jamais réajustés).\n"
            "- Chaque prédiction est calculée à l'instant (temps affiché sous chaque modèle) : aucun fichier de "
            "prédictions n'est lu.\n"
            f"- Tests automatiques : `{inf.TEST_EGALITE_PRECALCULE}` (direct = courbes précalculées) et "
            f"`{inf.TEST_CYCLES_FUTURS}` (le futur ne change rien)."
        )


# ---------------------------------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------------------------------

st.title("Tester un moteur")
st.caption("Les modèles ne voient que les vols reçus jusqu'ici ; la vraie RUL reste cachée jusqu'à ce que vous la révéliez.")

# --- selecteurs : sous-jeu, modeles, seuil ---
colonne_sous_jeu, colonne_modeles, colonne_seuil = st.columns([1.7, 2, 1.2])
with colonne_sous_jeu:
    commun.restaurer_choix("live_sous_jeu", list(SOUS_JEUX))
    sous_jeu = st.selectbox("Sous-jeu", list(SOUS_JEUX), key="live_sous_jeu", format_func=dd.description_sous_jeu)
    commun.memoriser_choix("live_sous_jeu")

inference = commun.modeles_inference(sous_jeu)  # en cache : preparation apprise sur le train + modeles
modeles_possibles = []
for modele in inf.MODELES_INFERENCE:
    if modele in inference["modeles"]:
        modeles_possibles.append(modele)

with colonne_modeles:
    commun.restaurer_choix("live_modeles", modeles_possibles)
    if "live_modeles" not in st.session_state:
        st.session_state["live_modeles"] = list(modeles_possibles)  # par defaut : tous les modeles disponibles
    modeles_choisis = st.multiselect("Modèles", modeles_possibles, key="live_modeles", format_func=dd.nom_modele)
    commun.memoriser_choix("live_modeles")
with colonne_seuil:
    commun.restaurer_choix("seuil_alerte", list(range(5, 101, 5)))
    if "seuil_alerte" not in st.session_state:
        st.session_state["seuil_alerte"] = int(SEUIL_ALERTE_DEFAUT)
    seuil_alerte = st.slider("Seuil d'alerte (cycles)", min_value=5, max_value=100, step=5, key="seuil_alerte",
                             help="ALERTE si la RUL prédite est inférieure ou égale au seuil (hypothèse de travail).")
    commun.memoriser_choix("seuil_alerte")

for modele in inference["indisponibles"]:  # modele absent : dit clairement, jamais cache
    st.warning(f"{dd.nom_modele(modele)} indisponible : {inference['indisponibles'][modele]}", icon=":material/warning:")
for message in inference["messages"]:  # ex. XGBoost entraine au premier chargement
    st.caption(message)

# --- source du moteur : tirage, numero, ou fichier importe ---
colonne_tirage, colonne_numero, colonne_vide = st.columns([1.6, 1.2, 2.1], vertical_alignment="bottom")
with colonne_tirage:
    st.button("Tirer un moteur au hasard", icon=":material/casino:", type="primary", key="live_tirer",
              on_click=tirer_au_hasard, args=(sous_jeu,), width="stretch")
with colonne_numero:
    st.selectbox("ou choisir un moteur test", commun.moteurs_test(sous_jeu), index=None, placeholder="numéro",
                 key=CLE_NUMERO, on_change=choisir_numero, args=(sous_jeu,))
with st.expander("Importer un fichier de mesures", expanded=False, icon=":material/upload_file:"):
    st.file_uploader("Mesures d'un moteur (format C-MAPSS ou CSV)", type=["txt", "csv"], key="live_fichier")
    st.number_input("RUL vraie au dernier cycle (facultatif, pour révéler)", min_value=0, value=None, step=1,
                    key="live_rul_import")
    st.button("Tester ce fichier", key="live_importer", on_click=importer_fichier, args=(sous_jeu,))
    if st.session_state.get(CLE_ERREUR_IMPORT, "") != "":
        st.error(st.session_state[CLE_ERREUR_IMPORT], icon=":material/error:")
    st.caption("Exemples : `app/exemples/` (un moteur test par sous-jeu ; RUL vraie dans `LISEZMOI.md`).")

moteur = st.session_state.get(CLE_MOTEUR)
# un moteur du jeu de test appartient a son sous-jeu : changer de sous-jeu l'oublie
if moteur is not None and moteur["origine"] != "import" and moteur["sous_jeu"] != sous_jeu:
    del st.session_state[CLE_MOTEUR]
    moteur = None
if moteur is None:
    st.info("Tirer un moteur au hasard ou choisir un numéro.", icon=":material/flight:")
    afficher_preuves(inference, sous_jeu)
    st.stop()
if len(modeles_choisis) == 0:
    st.info("Choisir au moins un modèle.")
    st.stop()

# predictions en memoire : valables pour ce moteur ET les modeles de ce sous-jeu seulement
if st.session_state.get(CLE_IDENTITE) != (sous_jeu, moteur["jeton"]):
    st.session_state[CLE_IDENTITE] = (sous_jeu, moteur["jeton"])
    remettre_a_zero()

mesures = moteur["mesures"]
nb_cycles = moteur["nb_cycles"]
revele = bool(st.session_state.get(CLE_REVELE, False))
verite_connue = moteur["rul_finale"] is not None

# --- commandes : lecture / pause, debut, verite ---
colonne_nom, colonne_lecture, colonne_debut, colonne_verite = st.columns([2.6, 1, 1, 1.5], vertical_alignment="bottom")
nom_court, provenance = nom_du_moteur(moteur, revele)
with colonne_nom:
    st.subheader(nom_court)
    st.caption(provenance)
with colonne_lecture:
    if st.session_state.get(CLE_LECTURE, False):
        st.button("Pause", icon=":material/pause:", key="live_bouton_lecture", on_click=basculer_lecture,
                  args=(nb_cycles,), width="stretch")
    else:
        st.button("Lecture", icon=":material/play_arrow:", type="primary", key="live_bouton_lecture",
                  on_click=basculer_lecture, args=(nb_cycles,), width="stretch")
with colonne_debut:
    st.button("Début", icon=":material/skip_previous:", key="live_bouton_debut", on_click=remettre_a_zero,
              width="stretch")
with colonne_verite:
    libelle = "Révéler la vraie RUL"
    if revele:
        libelle = "Masquer la vraie RUL"
    aide = None
    if not verite_connue:
        aide = "RUL vraie inconnue pour ce fichier : la saisir dans « Importer un fichier de mesures »."
    st.button(libelle, icon=":material/visibility:", key="live_bouton_verite", on_click=basculer_verite,
              disabled=not verite_connue, help=aide, width="stretch")

intervalle = None  # pas de relance automatique hors lecture
if st.session_state.get(CLE_LECTURE, False):
    intervalle = INTERVALLE_LECTURE_S


@st.fragment(run_every=intervalle)
def zone_lecture():
    """
    Calcule les nouveaux cycles (pendant la lecture) puis affiche les cartes des modeles et la courbe.

    Pendant la lecture, chaque relance fait arriver CYCLES_PAR_IMAGE nouveaux cycles et les fait predire ;
    au dernier cycle, la lecture s'arrete et la page entiere est relancee (pour arreter le minuteur).

    Returns
    -------
    None
    """
    if st.session_state.get(CLE_LECTURE, False):
        cible = min(int(st.session_state[CLE_CYCLE]) + CYCLES_PAR_IMAGE, nb_cycles)
        calculer_jusqu_a(inference, mesures, cible, modeles_choisis)
        st.session_state[CLE_CYCLE] = cible
        if cible >= nb_cycles:  # fin du moteur
            st.session_state[CLE_LECTURE] = False
            st.rerun()
    cycle = int(st.session_state[CLE_CYCLE])
    calculer_jusqu_a(inference, mesures, cycle, modeles_choisis)  # modele ajoute en cours de route : rattrapage

    # verite des cycles 1..t, seulement si elle est revelee (sinon rien ne part vers le navigateur)
    verite = None
    if revele and verite_connue:
        verite_complete = inf.verite_moteur(mesures, moteur["rul_finale"])
        verite = verite_complete[verite_complete["time_cycles"] <= cycle]
        rul_vraie = "—"
        if cycle > 0:
            rul_vraie = commun.fr(verite["rul_vraie"].iloc[-1], 0)
        st.markdown(
            f"**RUL vraie au cycle {cycle} : {rul_vraie} cycles** — au dernier cycle ({nb_cycles}) : "
            f"{commun.fr(moteur['rul_finale'], 0)} (`RUL_{sous_jeu}.txt`)."
        )

    # cartes : cycle recu puis une carte par modele
    colonnes = st.columns(len(modeles_choisis) + 1)
    with colonnes[0]:
        st.metric("Cycle reçu", f"{cycle} / {nb_cycles}", border=True)
    predictions_affichees = {}
    for position in range(len(modeles_choisis)):
        modele = modeles_choisis[position]
        valeurs = st.session_state[CLE_PREDICTIONS].get(modele, [])
        predictions_affichees[modele] = valeurs[:cycle]
        with colonnes[position + 1]:
            afficher_carte(modele, valeurs, cycle, verite, seuil_alerte, st.session_state.get(CLE_TEMPS))

    # courbe : axe vertical fixe sans indice sur la verite tant qu'elle est cachee
    rul_max = max(float(PLAFOND_RUL), float(seuil_alerte))
    for modele in predictions_affichees:
        for valeur in predictions_affichees[modele]:
            rul_max = max(rul_max, float(valeur))
    if verite is not None and len(verite) > 0:
        rul_max = max(rul_max, float(verite["rul_vraie"].max()))
    titre = f"{nom_court} — RUL prédite en direct, cycle {cycle} sur {nb_cycles}"
    figure = figures.figure_prediction_en_direct(predictions_affichees, seuil_alerte, titre, nb_cycles, rul_max, verite)
    st.plotly_chart(figure, key="live_graphique", config=figures.CONFIG_PLOTLY)
    if cycle == 0:
        st.caption("Appuyer sur Lecture : les cycles arrivent un par un et chaque modèle prédit à chaque cycle.")


zone_lecture()
afficher_preuves(inference, sous_jeu)
