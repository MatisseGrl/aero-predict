"""
Page 5 du tableau de bord : "Et si un capteur derivait ?" (sensibilite du modele, FD001, inference en direct).

Ici, rien n'est lu dans un fichier de predictions : a chaque reglage, l'application prepare les donnees
brutes du moteur avec l'artefact du modele retenu (`preparer_donnees_brutes` de src/demo.py) et predit
la RUL a chaque cycle (`predire_rul`), une fois sur les mesures d'origine, une fois avec une derive
injectee sur un capteur choisi. L'artefact est charge (ou entraine s'il manque) par `charger_ou_entrainer`.
"""

import numpy as np
import streamlit as st

from app import commun, figures
from src import donnees_demo as dd
from src.demo import SEUIL_ALERTE_DEFAUT

# reglages des curseurs (amplitudes proposees)
POURCENTAGE_MAX = 3.0  # derive multiplicative de -3 % a +3 %
POURCENTAGE_DEFAUT = 1.0
FRACTION_ETENDUE_MAX = 1.0  # decalage additif de -1 a +1 fois l'etendue du capteur sur le train
FRACTION_ETENDUE_DEFAUT = 0.25

st.title("Et si… un capteur dérivait ?")
st.caption(
    "Sensibilité du modèle retenu à une dérive injectée sur un capteur, moteur test FD001, inférence "
    "calculée en direct à chaque réglage (aucun fichier précalculé)."
)

artefact = commun.artefact()  # charge models/modele_retenu.joblib, ou l'entraine s'il manque
test_brut, rul_test = commun.donnees_test_fd001()
nom_modele = dd.nom_modele("xgboost_retenu")

# moteurs du jeu de test, et RUL vraie au dernier cycle de chacun (ligne i de RUL_FD001.txt = i-eme moteur)
moteurs = []
for moteur_id in np.sort(test_brut["unit_number"].unique()):
    moteurs.append(int(moteur_id))
nb_cycles_par_moteur = test_brut.groupby("unit_number").size()
moteur_le_plus_long = int(nb_cycles_par_moteur.idxmax())

# ---------------------------------------------------------------------------------------------------
# Reglages
# ---------------------------------------------------------------------------------------------------

colonne_moteur, colonne_capteur, colonne_type, colonne_seuil = st.columns([1, 1, 1.6, 1.4])
with colonne_moteur:
    commun.restaurer_choix("sim_moteur", moteurs)
    if "sim_moteur" not in st.session_state:
        st.session_state["sim_moteur"] = moteur_le_plus_long
    moteur = st.selectbox(
        "Moteur test FD001", moteurs, key="sim_moteur",
        help="Par défaut : le moteur test qui a le plus de cycles observés (la dérive a le temps d'agir).",
    )
    commun.memoriser_choix("sim_moteur")
capteurs = list(artefact["sensor_cols_restants"])  # seuls les capteurs utilises par le modele
with colonne_capteur:
    commun.restaurer_choix("sim_capteur", capteurs)
    if "sim_capteur" not in st.session_state:
        st.session_state["sim_capteur"] = dd.choisir_capteurs_par_defaut(capteurs, nombre=1)[0]
    capteur = st.selectbox(
        "Capteur perturbé", capteurs, key="sim_capteur",
        help="Capteurs utilisés par le modèle (les capteurs constants ont été retirés en Phase 1).",
    )
    commun.memoriser_choix("sim_capteur")
with colonne_type:
    commun.restaurer_choix("sim_type", list(dd.TYPES_DERIVE.keys()))
    type_derive = st.radio(
        "Type de dérive", list(dd.TYPES_DERIVE.keys()), key="sim_type", format_func=dd.TYPES_DERIVE.get,
        horizontal=True,
    )
    commun.memoriser_choix("sim_type")
with colonne_seuil:
    commun.restaurer_choix("seuil_alerte", list(range(5, 101, 5)))
    if "seuil_alerte" not in st.session_state:
        st.session_state["seuil_alerte"] = int(SEUIL_ALERTE_DEFAUT)
    seuil_alerte = st.slider(
        "Seuil d'alerte (cycles) — hypothèse de travail", min_value=5, max_value=100, step=5, key="seuil_alerte",
    )
    commun.memoriser_choix("seuil_alerte")

donnees_moteur = test_brut[test_brut["unit_number"] == moteur].sort_values("time_cycles")
premier_cycle = int(donnees_moteur["time_cycles"].min())
dernier_cycle = int(donnees_moteur["time_cycles"].max())
rul_finale = float(rul_test[moteurs.index(moteur)])
etendue = dd.etendue_capteur_train(artefact, capteur)

colonne_amplitude, colonne_debut, colonne_mode = st.columns([1.6, 1.6, 1])
with colonne_amplitude:
    if type_derive == "pourcentage":
        amplitude = st.slider(
            "Amplitude de la dérive (%)", min_value=-POURCENTAGE_MAX, max_value=POURCENTAGE_MAX,
            value=POURCENTAGE_DEFAUT, step=0.1, format="%.1f %%", key="sim_amplitude_pourcentage",
            help="Valeur brute multipliée par (1 + amplitude / 100) à partir du cycle de début.",
        )
        moyenne = float(donnees_moteur[capteur].mean())
        equivalent = abs(moyenne * amplitude / 100.0)
    else:
        pas = etendue["etendue"] / 100.0
        amplitude = st.slider(
            f"Décalage ajouté à {capteur} (unités brutes)", min_value=-FRACTION_ETENDUE_MAX * etendue["etendue"],
            max_value=FRACTION_ETENDUE_MAX * etendue["etendue"], value=FRACTION_ETENDUE_DEFAUT * etendue["etendue"],
            step=pas, format="%.4f", key=f"sim_amplitude_decalage_{capteur}",
            help="Valeur ajoutée à la mesure brute à partir du cycle de début.",
        )
        equivalent = abs(amplitude)
    part_etendue = 100.0 * equivalent / etendue["etendue"]
    st.caption(
        f"Soit {commun.fr(equivalent, 4)} en valeur brute, {commun.fr(part_etendue, 0)} % de l'étendue de "
        f"{capteur} sur le train ({commun.fr(etendue['minimum'], 4)} à {commun.fr(etendue['maximum'], 4)})."
    )
with colonne_debut:
    cle_debut = f"sim_debut_{moteur}"  # une cle par moteur : bornes differentes d'un moteur a l'autre
    cycle_debut = st.slider(
        "Début de la dérive (cycle)", min_value=premier_cycle, max_value=dernier_cycle,
        value=premier_cycle + (dernier_cycle - premier_cycle) // 2, key=cle_debut,
    )
with colonne_mode:
    progressive = st.toggle(
        "Dérive progressive", value=False, key="sim_progressive",
        help="Activé : la dérive monte linéairement de 0 (cycle de début) à l'amplitude complète (dernier cycle). "
             "Désactivé : elle est appliquée d'un coup (un défaut soudain).",
    )

# ---------------------------------------------------------------------------------------------------
# Inference en direct : avec et sans derive
# ---------------------------------------------------------------------------------------------------

resultat = dd.simuler_derive(
    artefact, donnees_moteur, rul_finale, capteur, type_derive, amplitude, cycle_debut, progressive, seuil_alerte
)
dernier = resultat["dernier"]

colonnes = st.columns(5)
with colonnes[0]:
    st.metric("RUL vraie au dernier cycle", f"{commun.fr(dernier['rul_vraie'], 0)} cycles", border=True,
              help="RUL_FD001.txt : elle ne change pas, la dérive est artificielle.")
with colonnes[1]:
    st.metric("RUL prédite — mesures d'origine", f"{commun.fr(dernier['sans'])} cycles", border=True)
    commun.badge_statut(dernier["statut_sans"])
with colonnes[2]:
    st.metric("RUL prédite — avec dérive", f"{commun.fr(dernier['avec'])} cycles",
              delta=f"{commun.fr_signe(dernier['difference'])} cycles", delta_color="off", border=True)
    commun.badge_statut(dernier["statut_avec"])
with colonnes[3]:
    textes = []
    for premiere in [resultat["premiere_alerte_sans"], resultat["premiere_alerte_avec"]]:
        if premiere is None:
            textes.append("—")  # aucune alerte sur les cycles observes
        else:
            textes.append(str(premiere[0]))
    st.metric("1re alerte : cycle (origine / dérive)", " / ".join(textes), border=True,
              help="Premier cycle où la RUL prédite passe sous le seuil d'alerte ; « — » : aucune alerte.")
with colonnes[4]:
    temps_total = resultat["temps_ms_sans"] + resultat["temps_ms_avec"]
    st.metric("Inférence en direct", f"{commun.fr(temps_total, 0)} ms", border=True,
              help=f"Préparation des données brutes + prédiction de {len(resultat['tableau'])} cycles, deux fois "
                   "(avec et sans dérive), sur cette machine.")

# description courte de la derive, pour le titre du graphique
if type_derive == "pourcentage":
    description_derive = f"{commun.fr_signe(amplitude, 1)} %"
else:
    description_derive = f"{commun.fr_signe(amplitude, 4)} en valeur brute"
if progressive:
    description_derive = description_derive + ", progressive"
titre = f"Moteur test {moteur} (FD001) : dérive de {capteur} ({description_derive}) à partir du cycle {cycle_debut}"
st.plotly_chart(
    figures.figure_simulation(resultat, seuil_alerte, capteur, cycle_debut, nom_modele, titre),
    key="simulation_graphique", config=figures.CONFIG_PLOTLY,
)
st.caption(
    f"Écart maximal entre les deux prédictions : {commun.fr(resultat['difference_max'])} cycles. "
    f"Modèle : {artefact['description_modele']}, entraîné le {artefact['date_entrainement']}."
)
commun.sources(commun.chemin_lisible(dd.chemin_artefact_demo()), "data/test_FD001.txt", "data/RUL_FD001.txt")

# ---------------------------------------------------------------------------------------------------
# Lecture du resultat (texte pedagogique, factuel)
# ---------------------------------------------------------------------------------------------------

with st.container(border=True):
    st.markdown(
        "**Comment lire ce résultat.** La courbe bleue est la prédiction sur les mesures réelles du moteur ; "
        "la courbe orange, celle du **même modèle** sur les mêmes mesures dont **un seul capteur** a été "
        f"modifié à partir du cycle {cycle_debut}. L'écart entre les deux mesure la **sensibilité du modèle** "
        "à ce capteur : si les courbes restent superposées, le modèle s'appuie peu sur ce capteur dans cette "
        "plage de valeurs ; si la courbe orange descend, la dérive rapproche l'alerte.\n\n"
        "**Ce que cette simulation ne montre pas.** La RUL vraie ne change pas : la dérive est ajoutée à la "
        "main sur un seul capteur, alors qu'une vraie dégradation modifie ensemble plusieurs grandeurs "
        "physiques liées. On teste la réaction du modèle à une entrée modifiée, **pas la physique du moteur**.\n\n"
        "**Particularité d'un modèle à arbres.** XGBoost découpe chaque variable en intervalles appris sur le "
        "train : il ne prolonge pas une tendance. Au-delà des valeurs rencontrées à l'entraînement (pour "
        f"{capteur} : {commun.fr(etendue['minimum'], 4)} à {commun.fr(etendue['maximum'], 4)} en valeur brute), "
        "sa prédiction cesse de varier avec ce capteur, même si la dérive continue de grandir."
    )
