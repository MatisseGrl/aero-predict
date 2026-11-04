"""
Page 6 du tableau de bord : fiabilite du prototype (temps de reponse, reproductibilite, limites connues).

- Temps de reponse : bouton qui lance `mesurer_benchmark` de src/demo.py (peu de repetitions) et
  compare le 95e centile a la cible (CIBLE_MS_DEFAUT de src/demo.py).
- Reproductibilite : graines fixees (lues dans le code et les fichiers), nombre de tests automatises
  (compte par scripts/preparer_demo.py, ou recompte ici), coherence de tous les fichiers de predictions.
- Limites connues : phrases reprises telles quelles de docs/STATUS.md (rien n'est invente ici).
"""

import os

import pandas as pd
import streamlit as st

from app import commun, figures
from src import donnees_demo as dd
from src.demo import CIBLE_MS_DEFAUT

# choix proposes pour la mesure des temps de reponse (courte, pour rester compatible avec une demo)
REPETITIONS_POSSIBLES = [1, 2, 3]
NOMBRES_MOTEURS_POSSIBLES = [10, 50, 100]


def afficher_modele_servi():
    """
    Section A : quel modele la demo sert-elle, entraine quand, avec quelle configuration et quelles versions.

    Returns
    -------
    None
    """
    st.header("A. Le modèle servi par la démo")
    chemin = dd.chemin_artefact_demo()
    artefact = commun.artefact_si_present()
    if artefact is None:
        st.info(
            f"Aucun artefact dans `{commun.chemin_lisible(chemin)}` : il sera entraîné automatiquement au premier "
            "passage sur la page « Et si… » ou à la première mesure de temps (commande équivalente : "
            "`python -m src.demo entrainer`)."
        )
        return
    identique = dd.comparer_artefact_et_config(artefact)
    colonnes = st.columns(4)
    with colonnes[0]:
        date_et_heure = artefact["date_entrainement"].split("T")  # "2026-11-04T16:34:58" -> date, heure
        st.metric("Entraîné le", date_et_heure[0], border=True, help=f"À {date_et_heure[-1]}.")
    with colonnes[1]:
        taille_ko = os.path.getsize(chemin) / 1024.0
        st.metric("Taille de l'artefact", f"{commun.fr(taille_ko, 0)} Ko", border=True)
    with colonnes[2]:
        st.metric("Données d'entraînement", f"{artefact['nb_moteurs_train']} moteurs", border=True,
                  help=f"{commun.fr(artefact['nb_lignes_train'], 0)} cycles, {len(artefact['feature_cols'])} features.")
    with colonnes[3]:
        if identique is True:
            texte = "identique"
        elif identique is False:
            texte = "DIFFÉRENTE"
        else:
            texte = "fichier absent"
        st.metric("Config. vs 04_config_retenue.json", texte, border=True,
                  help="Compare la configuration enregistrée dans l'artefact à celle du fichier versionné.")
    if identique is False:
        st.error("L'artefact a été entraîné avec une autre configuration : relancer `python -m src.demo entrainer`.")
    st.caption(f"Modèle : {artefact['description_modele']}.")
    versions = artefact.get("versions", {})
    lignes = []
    for bibliotheque in versions:
        lignes.append((bibliotheque, versions[bibliotheque]))
    with st.expander("Versions des bibliothèques enregistrées dans l'artefact"):
        st.dataframe(pd.DataFrame(lignes, columns=["bibliothèque", "version"]), hide_index=True, width="stretch")
    commun.sources(commun.chemin_lisible(chemin), "docs/resultats/04_config_retenue.json")


def lancer_mesure(repetitions, nb_moteurs):
    """
    Lance la mesure des temps de reponse et garde le resultat dans la session (affiche ensuite).

    Parameters
    ----------
    repetitions : int
        Nombre de passages sur les moteurs.
    nb_moteurs : int
        Nombre de moteurs mesures.

    Returns
    -------
    None
    """
    artefact = commun.artefact()
    resultats = dd.mesurer_temps_reponse(artefact, dd.chemin_artefact_demo(), repetitions, nb_moteurs, CIBLE_MS_DEFAUT)
    st.session_state["fiabilite_mesure"] = resultats


def afficher_temps_reponse():
    """
    Section B : mesure des temps de reponse a la demande, comparee a la cible.

    Returns
    -------
    None
    """
    st.header("B. Temps de réponse (mesurés maintenant, sur cette machine)")
    st.caption(
        "Protocole de `mesurer_benchmark` (src/demo.py) : un passage d'échauffement non compté, puis, pour chaque "
        "moteur, préparation de ses données brutes + prédiction de tous ses cycles, chronométrées ; temps de la "
        f"flotte entière ; temps de chargement de l'artefact. Cible : p95 ≤ {CIBLE_MS_DEFAUT:g} ms pour un moteur."
    )
    colonne_rep, colonne_nb, colonne_bouton = st.columns([1, 1, 1.4], vertical_alignment="bottom")
    with colonne_rep:
        repetitions = st.selectbox("Répétitions", REPETITIONS_POSSIBLES, index=0, key="fiabilite_repetitions")
    with colonne_nb:
        nb_moteurs = st.selectbox("Moteurs mesurés (FD001)", NOMBRES_MOTEURS_POSSIBLES, index=2, key="fiabilite_nb_moteurs")
    with colonne_bouton:
        if st.button("Mesurer maintenant", icon=":material/timer:", type="primary", key="fiabilite_mesurer"):
            with st.spinner("Mesure en cours…"):
                lancer_mesure(repetitions, nb_moteurs)

    resultats = st.session_state.get("fiabilite_mesure")
    if resultats is None:
        st.info("Cliquer sur « Mesurer maintenant » (quelques secondes).")
        return
    colonnes = st.columns(5)
    with colonnes[0]:
        st.metric("p50 (un moteur)", f"{commun.fr(resultats['p50_ms'])} ms", border=True)
    with colonnes[1]:
        st.metric("p95 (un moteur)", f"{commun.fr(resultats['p95_ms'])} ms", border=True,
                  help=f"Comparé à la cible de {CIBLE_MS_DEFAUT:g} ms.")
    with colonnes[2]:
        st.metric("Maximum", f"{commun.fr(resultats['max_ms'])} ms", border=True)
    with colonnes[3]:
        st.metric("Flotte entière (médiane)", f"{commun.fr(resultats['temps_flotte_s'], 2)} s", border=True,
                  help=f"{resultats['nb_mesures_flotte']} mesure(s) : préparation + prédiction des moteurs test au dernier cycle.")
    with colonnes[4]:
        st.metric("Chargement de l'artefact", f"{commun.fr(resultats['temps_chargement_ms'], 0)} ms", border=True)
    if resultats["cible_atteinte"]:
        st.success(
            f"p95 = {commun.fr(resultats['p95_ms'])} ms ≤ cible {resultats['cible_ms']:g} ms, sur "
            f"{resultats['nb_mesures']} mesures ({resultats['nb_moteurs']} moteurs × {resultats['repetitions']} répétition(s)), "
            f"mesuré le {resultats['date'].replace('T', ' à ')}.", icon=":material/task_alt:",
        )
    else:
        st.error(
            f"p95 = {commun.fr(resultats['p95_ms'])} ms > cible {resultats['cible_ms']:g} ms "
            f"({resultats['nb_mesures']} mesures, le {resultats['date'].replace('T', ' à ')})."
        )
    st.plotly_chart(figures.figure_temps_reponse(resultats), key="fiabilite_histogramme", config=figures.CONFIG_PLOTLY)
    st.caption("Ces temps dépendent du processeur et de sa charge au moment de la mesure : ils ne valent que pour cette machine.")


def recompter_tests():
    """
    Recompte les tests automatises (pytest --collect-only) et garde le resultat dans la session.

    Returns
    -------
    None
    """
    st.session_state["fiabilite_tests"] = dd.compter_tests_pytest()


def afficher_reproductibilite():
    """
    Section C : graines fixees, nombre de tests automatises, coherence de tous les fichiers de predictions.

    Returns
    -------
    None
    """
    st.header("C. Reproductibilité")
    liste = commun.inventaire()
    colonne_graines, colonne_tests = st.columns([1.3, 1])

    with colonne_graines:
        st.subheader("Graines aléatoires fixées")
        if len(liste) > 0:
            graines = dd.lister_graines(liste, dd.dossier_predictions())
        else:
            graines = dd.lister_graines(None)
        st.dataframe(graines, hide_index=True, width="stretch", height=300)
        st.caption("Valeurs lues dans le code (introspection) et dans les fichiers, pas recopiées.")

    with colonne_tests:
        st.subheader("Tests automatisés (pytest)")
        infos = dd.lire_infos_reproductibilite()
        if infos is not None and infos.get("nb_tests") is not None:
            st.metric("Tests collectés", str(infos["nb_tests"]), border=True,
                      help=f"Compté par scripts/preparer_demo.py le {infos.get('date', '?')}.")
            details = []
            if infos.get("commit"):
                details.append(f"commit `{infos['commit']}`")
            if infos.get("branche"):
                details.append(f"branche `{infos['branche']}`")
            if infos.get("modifications_locales") is True:
                details.append("avec des modifications locales non commitées")
            st.caption(f"Compté le {infos.get('date', '?')} ; " + ", ".join(details) + ".")
        else:
            st.caption("Pas encore compté par `scripts/preparer_demo.py`.")
        st.button("Recompter maintenant (pytest --collect-only)", icon=":material/fact_check:",
                  key="fiabilite_recompter", on_click=recompter_tests)
        recompte = st.session_state.get("fiabilite_tests")
        if recompte is not None:
            if recompte["nb_tests"] is None:
                st.error(f"Collecte impossible (code de retour {recompte['code_retour']}).")
            else:
                st.success(
                    f"{recompte['nb_tests']} tests collectés le {recompte['date'].replace('T', ' à ')} "
                    f"(en {commun.fr(recompte['duree_s'])} s).", icon=":material/task_alt:",
                )

    st.subheader("Cohérence de tous les fichiers de prédictions")
    if len(liste) == 0:
        commun.message_aucun_fichier()
        return
    comparaison, validation, controles = commun.comparaison()
    nb_coherents = int(controles["coherent"].sum())
    if nb_coherents == len(controles):
        st.success(
            f"{nb_coherents} fichiers sur {len(controles)} : métriques recalculées depuis les prédictions = métadonnées.",
            icon=":material/task_alt:",
        )
    else:
        st.error(f"{len(controles) - nb_coherents} fichier(s) incohérent(s) sur {len(controles)}.")
    st.dataframe(
        controles.rename(columns={
            "sous_jeu": "Sous-jeu", "modele": "Modèle", "nb_lignes": "Lignes", "nb_moteurs": "Moteurs",
            "coherent": "Cohérent", "ecart_max": "Écart maximal",
        }),
        hide_index=True, width="stretch",
        column_config={"Écart maximal": st.column_config.NumberColumn(format="%.1e")},
    )


def afficher_limites():
    """
    Section D : limites connues, reprises telles quelles de docs/STATUS.md.

    Returns
    -------
    None
    """
    st.header("D. Limites connues (reprises de docs/STATUS.md)")
    limites = dd.extraire_limites_status()
    if not limites["present"]:
        st.info("`docs/STATUS.md` absent.")
        return
    if len(limites["phrases"]) == 0:
        st.info("Aucune limite signalée dans les sections « Phase actuelle » et « Dernière modification ».")
    for phrase in limites["phrases"]:
        st.markdown(f"> {phrase}")
    date_entree = limites["date"]
    if date_entree == "":
        date_entree = "date non trouvée"
    st.caption(
        "Phrases citées sans modification (mise en forme retirée), sections « Phase actuelle » et « Dernière "
        f"modification » de docs/STATUS.md (entrée du {date_entree}), retenues parce qu'elles contiennent un "
        "de ces mots : " + ", ".join(dd.MOTS_CLES_LIMITES) + "."
    )
    with st.expander("Texte complet de ces deux sections"):
        for titre in limites["sections"]:
            st.markdown(f"**{titre}**")
            st.markdown(limites["sections"][titre])


st.title("Fiabilité du prototype")
afficher_modele_servi()
st.divider()
afficher_temps_reponse()
st.divider()
afficher_reproductibilite()
st.divider()
afficher_limites()
