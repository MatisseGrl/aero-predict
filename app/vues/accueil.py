"""
Page 1 du tableau de bord : le probleme, les donnees, le pipeline et quelques indicateurs cles.

Tous les nombres affiches sont lus dans les fichiers du depot (docs/resultats/, data/readme.txt,
src/sous_jeux.py, artefact du modele) ou recalcules : aucun n'est ecrit dans ce fichier.
"""

import streamlit as st

from app import commun
from src import donnees_demo as dd
from src.contrat_predictions import PLAFOND_RUL
from src.demo import SEUIL_ALERTE_DEFAUT, SEUIL_SURESTIMATION

# noms de colonnes lisibles pour le tableau des sous-jeux (les colonnes inconnues gardent leur nom)
NOMS_COLONNES_SOUS_JEUX = {
    "sous_jeu": "Sous-jeu",
    "nb_regimes": "Régimes de vol",
    "modes_de_panne": "Modes de panne",
    "nb_moteurs_train": "Moteurs train",
    "nb_moteurs_test": "Moteurs test",
    "nb_lignes_train": "Lignes train",
    "nb_lignes_test": "Lignes test",
    "moteurs_train": "Moteurs train",
    "moteurs_test": "Moteurs test",
    "conditions": "Conditions",
}


def afficher_probleme():
    """
    Affiche le probleme en trois phrases.

    Returns
    -------
    None
    """
    nb_capteurs = dd.nombre_capteurs_bruts()
    st.title("Aero Predict — estimer la durée de vie restante de turboréacteurs")
    st.markdown(
        "**Le problème.** Un turboréacteur s'use un peu à chaque cycle de vol : la **RUL** (*Remaining "
        "Useful Life*) est le nombre de cycles qu'il peut encore effectuer avant la panne.  \n"
        f"Le prototype estime cette RUL à partir des {nb_capteurs} capteurs mesurés à chaque cycle, pour "
        "**décider quand retirer un moteur** et planifier sa maintenance.  \n"
        "Les deux erreurs n'ont pas le même coût : **surestimer** la RUL laisse voler un moteur trop "
        "longtemps (**risque de panne**), la **sous-estimer** l'envoie en maintenance trop tôt (coût, sans "
        "risque) ; le score NASA pénalise donc davantage les surestimations."
    )


def afficher_donnees():
    """
    Affiche les quatre sous-jeux C-MAPSS (regimes, modes de panne, nombre de moteurs) et leur source.

    Returns
    -------
    None
    """
    st.subheader("Les données : NASA C-MAPSS (simulation)")
    st.markdown(
        "Des moteurs **simulés** par le logiciel C-MAPSS de la NASA (pas des moteurs réels) : dans le jeu "
        "d'entraînement, chaque moteur est suivi jusqu'à la panne ; dans le jeu de test, la série s'arrête "
        "avant, et la RUL vraie au dernier cycle est fournie à part (fichiers `RUL_FD00X.txt`)."
    )
    tableau, source = dd.decrire_sous_jeux()
    affiche = tableau.rename(columns=NOMS_COLONNES_SOUS_JEUX)
    st.dataframe(affiche, hide_index=True, width="stretch")
    commun.sources(source, "data/readme.txt")

    # ecarts entre le readme officiel et les fichiers (affiches plutot que caches)
    if source != "data/readme.txt":
        ecarts = dd.ecarts_readme(tableau, dd.lire_readme_donnees())
        if len(ecarts) > 0:
            st.info("Écart constaté entre le readme officiel et les fichiers :  \n" + "  \n".join(ecarts),
                    icon=":material/info:")


def texte_bloc(titre, lignes):
    """
    Affiche un bloc du schema du pipeline : un titre et quelques lignes courtes, dans un cadre.

    Parameters
    ----------
    titre : str
        Le titre du bloc.
    lignes : list de str
        Les lignes de texte (Markdown).

    Returns
    -------
    None
    """
    with st.container(border=True, height="stretch"):
        st.markdown(f"#### {titre}")
        for ligne in lignes:
            st.markdown(ligne)


def afficher_pipeline():
    """
    Affiche le pipeline en cinq blocs : donnees -> preparation -> modeles -> evaluation -> decision.

    Les chiffres de la preparation sont lus dans l'artefact du modele retenu s'il existe deja.

    Returns
    -------
    None
    """
    st.subheader("Le pipeline, de la donnée capteur à la décision")
    artefact = commun.artefact_si_present()
    liste = commun.inventaire()

    # contenu de chaque bloc
    blocs = []
    blocs.append(("1. Données", [
        "C-MAPSS : 4 sous-jeux",
        f"{dd.nombre_capteurs_bruts()} capteurs + 3 réglages de vol par cycle",
        "train : jusqu'à la panne ; test : coupé avant",
    ]))
    if artefact is not None:
        resume = dd.resume_pipeline(artefact)
        blocs.append(("2. Préparation", [
            f"{resume['nb_capteurs_retires']} capteurs constants retirés",
            "normalisation min-max apprise sur le train seul",
            f"lissage + statistiques glissantes sur {resume['fenetre']} cycles → {resume['nb_features']} features",
            f"cible : RUL plafonnée à {resume['plafond']}",
        ]))
    else:
        blocs.append(("2. Préparation", [
            "capteurs constants retirés",
            "normalisation min-max apprise sur le train seul",
            "lissage + statistiques glissantes",
            f"cible : RUL plafonnée à {PLAFOND_RUL}",
        ]))
    noms_modeles = []
    for modele in liste["modele"].tolist():
        if dd.nom_modele(modele) not in noms_modeles:
            noms_modeles.append(dd.nom_modele(modele))
    if len(noms_modeles) == 0:
        lignes_modeles = ["XGBoost (modèle de la démo)", "autres modèles : fichiers de prédictions à produire"]
    else:
        lignes_modeles = []
        for nom in noms_modeles:
            lignes_modeles.append(f"- {nom}")
    blocs.append(("3. Modèles", lignes_modeles))
    blocs.append(("4. Évaluation", [
        "au dernier cycle de chaque moteur test",
        "RMSE + score NASA (asymétrique)",
        f"surestimations > {SEUIL_SURESTIMATION} cycles",
        "validation croisée groupée par moteur",
    ]))
    blocs.append(("5. Décision d'alerte", [
        f"ALERTE si RUL prédite ≤ {SEUIL_ALERTE_DEFAUT:g} cycles (hypothèse de travail)",
        "4 statuts : correcte, manquée, fausse, rien à signaler",
    ]))

    # 5 blocs separes par des fleches
    largeurs = []
    for position in range(len(blocs)):
        largeurs.append(1.0)
        if position < len(blocs) - 1:
            largeurs.append(0.12)
    colonnes = st.columns(largeurs, vertical_alignment="center")
    for position in range(len(blocs)):
        titre, lignes = blocs[position]
        with colonnes[2 * position]:
            texte_bloc(titre, lignes)
        if position < len(blocs) - 1:
            with colonnes[2 * position + 1]:
                st.markdown("<div style='text-align:center;font-size:2.2rem;color:#555'>→</div>", unsafe_allow_html=True)
    if artefact is not None:
        commun.sources("models/modele_retenu.joblib (préparation apprise sur le train FD001)")


def afficher_indicateurs():
    """
    Affiche les indicateurs cles du modele retenu sur FD001, compares a la Phase 3, lus dans docs/resultats/.

    Returns
    -------
    None
    """
    st.subheader("Indicateurs clés — FD001, jeu de test officiel (évalué une fois)")
    config = commun.config_retenue()
    comparaison_test = commun.csv_resultat("04_comparaison_test.csv")
    if config is None or "metriques_test" not in config:
        st.info("`docs/resultats/04_config_retenue.json` absent : indicateurs de la Phase 4 indisponibles.")
        return
    metriques = config["metriques_test"]

    # ligne de reference : XGBoost Phase 3, lue dans 04_comparaison_test.csv si elle existe
    reference = None
    if comparaison_test is not None:
        lignes = comparaison_test[comparaison_test["modele"] == "XGBoost Phase 3"]
        if len(lignes) > 0:
            reference = lignes.iloc[0]

    colonnes = st.columns(4)
    definitions = [
        ("RMSE test (cycles)", "rmse", "rmse_test", 2),
        ("Score NASA test", "score_nasa", "nasa_test", 1),
        (f"Surestimés > {SEUIL_SURESTIMATION} cycles", "nb_surestimes", "surestimes_test", 0),
        (f"Alertes manquées (seuil {SEUIL_ALERTE_DEFAUT:g})", "alertes_manquees", "alertes_manquees_test", 0),
    ]
    for position in range(len(definitions)):
        libelle, cle_config, cle_reference, decimales = definitions[position]
        valeur = float(metriques[cle_config])
        arguments_delta = {}
        if reference is not None:
            arguments_delta = commun.delta_comparaison(valeur, reference[cle_reference], decimales, "XGBoost Phase 3")
        with colonnes[position]:
            st.metric(
                libelle, commun.fr(valeur, decimales), border=True,
                help="Plus petit = mieux. Flèche rouge : moins bon que la Phase 3 ; verte : meilleur.",
                **arguments_delta,
            )
    nom_retenu = config.get("nom", config.get("famille", ""))
    st.caption(
        f"Modèle retenu : {nom_retenu}. Des écarts de cette taille, sur le seul jeu de test FD001, ne "
        "suffisent pas à conclure qu'un modèle est meilleur : voir les limites reprises de docs/STATUS.md "
        "(page « Fiabilité »)."
    )
    commun.sources("docs/resultats/04_config_retenue.json", "docs/resultats/04_comparaison_test.csv")

    # etat de la demo : fichiers de predictions et tests automatises
    liste = commun.inventaire()
    infos = dd.lire_infos_reproductibilite()
    colonnes = st.columns(3)
    with colonnes[0]:
        st.metric("Fichiers de prédictions disponibles", str(len(liste)), border=True,
                  help="Couples (sous-jeu, modèle) lisibles dans le dossier des prédictions.")
    with colonnes[1]:
        st.metric("Sous-jeux couverts", str(len(dd.sous_jeux_disponibles(liste))), border=True)
    with colonnes[2]:
        if infos is not None and infos.get("nb_tests") is not None:
            st.metric("Tests automatisés (pytest)", str(infos["nb_tests"]), border=True,
                      help=f"Comptés le {infos.get('date', '?')} par scripts/preparer_demo.py.")
        else:
            st.metric("Tests automatisés (pytest)", "non comptés", border=True,
                      help="Lancer scripts/preparer_demo.py, ou le bouton de la page « Fiabilité ».")


afficher_probleme()
afficher_donnees()
st.divider()
afficher_pipeline()
st.divider()
afficher_indicateurs()
