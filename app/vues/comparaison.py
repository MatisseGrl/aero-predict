"""
Page 4 du tableau de bord : comparaison de tous les modeles disponibles, puis le choix du modele en Phase 4.

Section 1 : tous les (sous-jeu x modele) dont les predictions sont disponibles (metriques de test au
dernier cycle, metriques et protocole de validation), en tableau et en barres groupees.
Section 2 : FD001, Phase 4 : tableaux de validation croisee et de test, front de Pareto Optuna
reconstruit depuis les CSV, regle de choix rejouee et configuration retenue.
Section 3 : tableaux produits par les autres scripts (deep learning, arbres sur les 4 sous-jeux).
"""

import pandas as pd
import streamlit as st

from app import commun, figures
from src import donnees_demo as dd
from src.contrat_predictions import PLAFOND_RUL
from src.demo import SEUIL_ALERTE_DEFAUT, SEUIL_SURESTIMATION

# titres affiches du tableau de comparaison (colonne interne -> titre)
COLONNES_COMPARAISON = {
    "sous_jeu": "Sous-jeu",
    "nom_affiche": "Modèle",
    "nb_moteurs": "Moteurs test",
    "rmse": "RMSE test",
    "rmse_contre_rul_plafonnee": "RMSE test (RUL plafonnée)",
    "mae": "MAE test",
    "score_nasa": "Score NASA test",
    "score_nasa_par_moteur": "Score NASA / moteur",
    "nb_surestimes": f"Surestimés > {SEUIL_SURESTIMATION}",
    "alertes_manquees": "Alertes manquées",
    "fausses_alertes": "Fausses alertes",
    "nb_au_dela_plafond": f"RUL vraie > {PLAFOND_RUL}",
    "part_au_dela_plafond": f"RUL vraie > {PLAFOND_RUL} (%)",
    "rmse_validation": "RMSE validation (CV moyen)",
    "nasa_validation": "Score NASA validation (CV moyen)",
}

# titres affiches du tableau de validation croisee de la Phase 4 (04_comparaison_cv.csv)
COLONNES_CV = {
    "section": "Section",
    "modele": "Modèle",
    "rmse_cv_moyen": "RMSE CV moyen",
    "rmse_cv_ecart_type": "RMSE CV écart-type",
    "mae_cv_moyen": "MAE CV moyen",
    "nasa_cv_moyen": "Score NASA CV moyen",
    "surestimes_oof": f"Surestimés > {SEUIL_SURESTIMATION} (hors-pli)",
    "alertes_manquees_oof": "Alertes manquées (hors-pli)",
    "fausses_alertes_oof": "Fausses alertes (hors-pli)",
    "temps_entrainement_5_plis_s": "Entraînement 5 plis (s)",
}

# titres affiches du tableau de test de la Phase 4 (04_comparaison_test.csv)
COLONNES_TEST = {
    "modele": "Modèle",
    "rmse_test": "RMSE test",
    "rmse_test_rul_plafonnee": "RMSE test (RUL plafonnée)",
    "mae_test": "MAE test",
    "nasa_test": "Score NASA test",
    "surestimes_test": f"Surestimés > {SEUIL_SURESTIMATION}",
    "alertes_manquees_test": "Alertes manquées",
    "fausses_alertes_test": "Fausses alertes",
    "rmse_test_moyenne_5_graines": "RMSE test, moyenne 5 graines",
    "rmse_test_ecart_type_5_graines": "RMSE test, écart-type 5 graines",
}


def garder_et_renommer(tableau, colonnes):
    """
    Garde les colonnes connues d'un tableau (dans l'ordre du dictionnaire) et leur donne un titre lisible.

    Parameters
    ----------
    tableau : pandas.DataFrame
        Le tableau d'origine.
    colonnes : dict
        colonne interne -> titre affiche (les colonnes absentes du tableau sont ignorees).

    Returns
    -------
    pandas.DataFrame
        Le tableau reduit et renomme.
    """
    presentes = []
    for colonne in colonnes:
        if colonne in tableau.columns:
            presentes.append(colonne)
    return tableau[presentes].rename(columns=colonnes)


def afficher_protocoles(comparaison):
    """
    Affiche les protocoles de validation, regroupes : un texte par protocole, avec les modeles concernes.

    Parameters
    ----------
    comparaison : pandas.DataFrame
        Le resultat de `tableau_comparaison` (colonnes protocole_validation, sous_jeu, nom_affiche).

    Returns
    -------
    None
    """
    protocoles = {}  # texte du protocole -> liste des "FD00X modele" qui l'utilisent
    for position in range(len(comparaison)):
        texte = comparaison["protocole_validation"].iloc[position]
        etiquette = f"{comparaison['sous_jeu'].iloc[position]} {comparaison['nom_affiche'].iloc[position]}"
        if texte not in protocoles:
            protocoles[texte] = []
        protocoles[texte].append(etiquette)
    with st.expander(f"Protocoles de validation ({len(protocoles)} protocole(s) distinct(s))", expanded=False):
        for texte in protocoles:
            st.markdown(f"**Utilisé par :** {', '.join(protocoles[texte])}")
            st.markdown(f"> {texte}")


# titres affiches du tableau des tests apparies
COLONNES_TESTS = {
    "sous_jeu": "Sous-jeu",
    "nb_moteurs": "Moteurs test",
    "rmse_a": "RMSE A",
    "rmse_b": "RMSE B",
    "mae_a": "Erreur abs. moyenne A",
    "mae_b": "Erreur abs. moyenne B",
    "difference_moyenne": "Écart moyen A − B (cycles)",
    "p_valeur_t": "p (test t apparié)",
    "p_valeur_wilcoxon": "p (Wilcoxon)",
    "conclusion": "Conclusion",
}


def afficher_tests_apparies(liste):
    """
    Compare deux modeles choisis, moteur par moteur, sur chaque sous-jeu ou les deux sont disponibles.

    Parameters
    ----------
    liste : pandas.DataFrame
        L'inventaire des fichiers de predictions.

    Returns
    -------
    None
    """
    st.subheader("Tests appariés moteur par moteur (jeu de test, dernier cycle)")
    modeles = []
    for modele in dd.MODELES_CONNUS:
        if modele in liste["modele"].tolist():
            modeles.append(modele)
    if len(modeles) < 2:
        st.caption("Un seul modèle disponible : pas de comparaison possible.")
        return

    # par defaut : le modele retenu (A) contre la Phase 3 (B), s'ils existent
    colonne_a, colonne_b = st.columns(2)
    with colonne_a:
        commun.restaurer_choix("comparaison_modele_a", modeles)
        if "comparaison_modele_a" not in st.session_state and "xgboost_retenu" in modeles:
            st.session_state["comparaison_modele_a"] = "xgboost_retenu"
        modele_a = st.selectbox("Modèle A", modeles, key="comparaison_modele_a", format_func=dd.nom_modele)
        commun.memoriser_choix("comparaison_modele_a")
    autres = []
    for modele in modeles:
        if modele != modele_a:
            autres.append(modele)
    with colonne_b:
        commun.restaurer_choix("comparaison_modele_b", autres)
        if "comparaison_modele_b" not in st.session_state and "xgboost_phase3" in autres:
            st.session_state["comparaison_modele_b"] = "xgboost_phase3"
        modele_b = st.selectbox("Modèle B (référence)", autres, key="comparaison_modele_b", format_func=dd.nom_modele)
        commun.memoriser_choix("comparaison_modele_b")

    tests = commun.tests_apparies(modele_a, modele_b)
    if len(tests) == 0:
        st.caption("Aucun sous-jeu n'a de prédictions pour ces deux modèles à la fois.")
        return
    st.dataframe(
        tests.rename(columns=COLONNES_TESTS), hide_index=True, width="stretch",
        column_config={
            "RMSE A": st.column_config.NumberColumn(format="%.2f"),
            "RMSE B": st.column_config.NumberColumn(format="%.2f"),
            "Erreur abs. moyenne A": st.column_config.NumberColumn(format="%.2f"),
            "Erreur abs. moyenne B": st.column_config.NumberColumn(format="%.2f"),
            "Écart moyen A − B (cycles)": st.column_config.NumberColumn(format="%+.2f"),
            "p (test t apparié)": st.column_config.NumberColumn(format="%.4f"),
            "p (Wilcoxon)": st.column_config.NumberColumn(format="%.4f"),
            "Conclusion": st.column_config.TextColumn(width="large"),
        },
    )
    # phrases explicites pour les resultats significatifs (lus dans le tableau ci-dessus)
    for position in range(len(tests)):
        ligne = tests.iloc[position]
        if "significativement" in ligne["conclusion"]:
            st.markdown(
                f"- **{ligne['sous_jeu']}** : {ligne['conclusion']} (erreur absolue au dernier cycle "
                f"{commun.fr_signe(ligne['difference_moyenne'], 2)} cycle par moteur en moyenne, "
                f"p = {ligne['p_valeur_t']:.1e} et {ligne['p_valeur_wilcoxon']:.1e}, {ligne['nb_moteurs']} moteurs)."
            )
    st.caption(
        "Pour chaque moteur test : erreur absolue |RUL prédite − RUL vraie| au dernier cycle observé, pour A et "
        "pour B ; les deux séries sont comparées par `comparer_apparie` (src/evaluation.py) : test t apparié et "
        f"test de Wilcoxon signé ; « significatif » si les deux p-valeurs sont < {dd.SEUIL_SIGNIFICATIVITE:g}. "
        "Le modèle « retenu » a été choisi sur FD001 seulement (règle de la Phase 4, section 2) : sur les autres "
        "sous-jeux, ses résultats ne sont pas garantis meilleurs."
    )


def afficher_section_disponibles():
    """
    Section 1 : tableau et barres de tous les (sous-jeu x modele) disponibles.

    Returns
    -------
    None
    """
    st.header("1. Tous les modèles disponibles, sur tous les sous-jeux")
    liste = commun.inventaire()
    if len(liste) == 0:
        commun.message_aucun_fichier()
        return
    comparaison, validation, controles = commun.comparaison()

    constantes = dd.constantes_score_nasa()
    st.caption(
        "Métriques de **test** calculées au dernier cycle observé de chaque moteur test, contre la RUL vraie "
        f"brute (contrat de prédictions). Score NASA : somme sur les moteurs de exp(−écart / {constantes['a_sous']}) − 1 "
        f"pour une sous-estimation, exp(écart / {constantes['a_sur']}) − 1 pour une surestimation. Métriques de "
        "**validation** : celles déclarées par chaque script, avec son protocole."
    )
    affiche = garder_et_renommer(comparaison, COLONNES_COMPARAISON)
    st.dataframe(
        affiche, hide_index=True, width="stretch",
        column_config={
            "RMSE test": st.column_config.NumberColumn(format="%.2f"),
            "RMSE test (RUL plafonnée)": st.column_config.NumberColumn(format="%.2f"),
            "MAE test": st.column_config.NumberColumn(format="%.2f"),
            "Score NASA test": st.column_config.NumberColumn(format="%.1f"),
            "Score NASA / moteur": st.column_config.NumberColumn(format="%.2f"),
            "RMSE validation (CV moyen)": st.column_config.NumberColumn(format="%.2f"),
            "Score NASA validation (CV moyen)": st.column_config.NumberColumn(format="%.0f"),
            f"RUL vraie > {PLAFOND_RUL} (%)": st.column_config.NumberColumn(format="%.0f %%"),
        },
        placeholder="—",
    )
    afficher_protocoles(comparaison)
    nb_coherents = int(controles["coherent"].sum())
    if nb_coherents == len(controles):
        st.success(
            f"Contrôle de cohérence : pour les {len(controles)} fichiers, les métriques de test recalculées depuis "
            "les prédictions sont égales aux métadonnées.", icon=":material/task_alt:",
        )
    else:
        st.error(f"Contrôle de cohérence : {len(controles) - nb_coherents} fichier(s) incohérent(s) (voir la page Fiabilité).")
    commun.sources("sorties/predictions/<SOUS_JEU>__<modèle>.json (métadonnées du contrat)")

    # part des moteurs test au-dela du plafond, par sous-jeu (lue dans les fichiers de predictions)
    morceaux = []
    deja_vus = []
    for position in range(len(comparaison)):
        sous_jeu = comparaison["sous_jeu"].iloc[position]
        if sous_jeu in deja_vus:
            continue  # meme moteurs test pour tous les modeles d'un sous-jeu
        deja_vus.append(sous_jeu)
        morceaux.append(
            f"{sous_jeu} : {commun.fr(comparaison['part_au_dela_plafond'].iloc[position], 0)} % "
            f"({comparaison['nb_au_dela_plafond'].iloc[position]} sur {comparaison['nb_moteurs'].iloc[position]})"
        )
    st.caption(
        f"Moteurs test dont la RUL vraie au dernier cycle dépasse le plafond appris ({PLAFOND_RUL}) : "
        + " ; ".join(morceaux) + ". Un modèle entraîné sur une cible plafonnée ne peut pas les prédire "
        "exactement : ils pèsent lourd dans le RMSE contre la RUL brute, beaucoup moins dans le RMSE contre "
        "la RUL plafonnée (colonne dédiée du tableau)."
    )

    normaliser = st.toggle(
        "Rapporter au nombre de moteurs test (sous-jeux de tailles différentes)", value=True, key="comparaison_normaliser",
        help="Score NASA moyen par moteur, et comptes en % des moteurs : comparables entre sous-jeux.",
    )
    st.plotly_chart(
        figures.figure_barres_comparaison(comparaison, normaliser, SEUIL_SURESTIMATION, SEUIL_ALERTE_DEFAUT),
        key="comparaison_barres", config=figures.CONFIG_PLOTLY,
    )

    afficher_tests_apparies(liste)

    with st.expander("Toutes les métriques de validation déclarées par les scripts"):
        st.dataframe(validation, hide_index=True, width="stretch", placeholder="—")
        commun.sources("clé « metriques_validation » des métadonnées")


def afficher_section_phase4():
    """
    Section 2 : FD001, Phase 4 (validation croisee, test, front de Pareto, regle de choix).

    Returns
    -------
    None
    """
    st.header("2. FD001 — Phase 4 : comparaison en validation croisée et choix du modèle")

    tableau_cv = commun.csv_resultat("04_comparaison_cv.csv")
    if tableau_cv is not None:
        st.subheader("Validation croisée : mêmes 5 plis GroupKFold par moteur pour tous les modèles")
        st.dataframe(
            garder_et_renommer(tableau_cv, COLONNES_CV), hide_index=True, width="stretch",
            column_config={
                "RMSE CV moyen": st.column_config.NumberColumn(format="%.2f"),
                "RMSE CV écart-type": st.column_config.NumberColumn(format="%.2f"),
                "MAE CV moyen": st.column_config.NumberColumn(format="%.2f"),
                "Score NASA CV moyen": st.column_config.NumberColumn(format="%.0f"),
                "Entraînement 5 plis (s)": st.column_config.NumberColumn(format="%.1f"),
            },
        )
        commun.sources("docs/resultats/04_comparaison_cv.csv")
    else:
        st.info("`docs/resultats/04_comparaison_cv.csv` absent.")

    tableau_test = commun.csv_resultat("04_comparaison_test.csv")
    if tableau_test is not None:
        st.subheader("Jeu de test officiel FD001 (évalué une fois, après le choix du modèle)")
        st.dataframe(
            garder_et_renommer(tableau_test, COLONNES_TEST), hide_index=True, width="stretch",
            column_config={
                "RMSE test": st.column_config.NumberColumn(format="%.2f"),
                "RMSE test (RUL plafonnée)": st.column_config.NumberColumn(format="%.2f"),
                "MAE test": st.column_config.NumberColumn(format="%.2f"),
                "Score NASA test": st.column_config.NumberColumn(format="%.1f"),
                "RMSE test, moyenne 5 graines": st.column_config.NumberColumn(format="%.2f"),
                "RMSE test, écart-type 5 graines": st.column_config.NumberColumn(format="%.3f"),
            },
            placeholder="—",
        )
        commun.sources("docs/resultats/04_comparaison_test.csv")

    # front de Pareto Optuna, reconstruit depuis les CSV
    st.subheader("Optimisation Optuna : front de Pareto (RMSE CV, score NASA CV)")
    analyse = commun.optuna()
    if analyse is None:
        st.info("CSV Optuna absents de `docs/resultats/` (`04_optuna_*.csv`).")
        return
    st.plotly_chart(figures.figure_pareto(analyse), key="comparaison_pareto", config=figures.CONFIG_PLOTLY)
    st.caption(
        f"{analyse['nb_essais']} essais, {analyse['nb_front']} sur le front de Pareto (aucun autre essai n'est "
        f"meilleur sur les deux critères à la fois), dont {analyse['nb_admissibles']} sous le seuil de la règle "
        f"(RMSE CV ≤ {commun.fr(analyse['seuil_rmse'], 2)}). Survoler un point pour voir ses paramètres. "
        f"Vue initiale resserrée autour du front ({figures.cadrage_pareto(analyse)['nb_visibles']} essais visibles "
        f"sur {analyse['nb_essais']}) : double-cliquer dans le graphique pour voir tous les essais."
    )
    commun.sources("docs/resultats/04_optuna_xgb_quadratique.csv", "04_optuna_xgb_nasa.csv", "04_optuna_lgbm_quadratique.csv")

    # regle de choix et configuration retenue
    config = commun.config_retenue()
    colonne_regle, colonne_config = st.columns([1.2, 1])
    with colonne_regle:
        st.markdown("**Règle de choix (fixée avant tout résultat de test)**")
        if config is not None and "regle_de_choix" in config:
            st.markdown(f"> {config['regle_de_choix']}")
        retenu = analyse["retenu"]
        texte_retenu = f"{retenu['etude']} n°{int(retenu['numero'])}"
        if analyse["config_coherente"] is True:
            st.success(
                f"La règle, rejouée par l'application sur les CSV Optuna (`appliquer_regle_de_choix`), désigne "
                f"l'essai {texte_retenu} ; sa configuration est identique à celle de `04_config_retenue.json`.",
                icon=":material/task_alt:",
            )
        elif analyse["config_coherente"] is False:
            st.error(
                f"La règle rejouée désigne l'essai {texte_retenu}, dont la configuration diffère de "
                "`04_config_retenue.json`."
            )
    with colonne_config:
        st.markdown("**Configuration retenue**")
        if config is None:
            st.info("`docs/resultats/04_config_retenue.json` absent.")
        else:
            lignes = []
            lignes.append(("modèle", config.get("nom", "")))
            lignes.append(("famille", config.get("famille", "")))
            lignes.append(("perte d'apprentissage", config.get("perte", "")))
            parametres_perte = config.get("parametres_perte", {})
            for cle in parametres_perte:
                lignes.append((f"perte : {cle}", dd.formater_nombre(parametres_perte[cle])))
            hyperparametres = config.get("hyperparametres", {})
            for cle in hyperparametres:
                lignes.append((cle, dd.formater_nombre(hyperparametres[cle])))
            tableau_config = pd.DataFrame(lignes, columns=["paramètre", "valeur"])
            st.dataframe(tableau_config, hide_index=True, width="stretch", height=300)
            commun.sources("docs/resultats/04_config_retenue.json")


def afficher_section_autres_tableaux():
    """
    Section 3 : tableaux produits par les scripts deep learning et arbres (s'ils existent).

    Returns
    -------
    None
    """
    st.header("3. Autres tableaux de résultats")
    for nom_fichier, titre in [
        (dd.FICHIER_DEEP_LEARNING, "Deep learning (LSTM, CNN 1D)"),
        (dd.FICHIER_SOUS_JEUX_ARBRES, "Modèles à arbres sur les quatre sous-jeux"),
    ]:
        st.subheader(titre)
        tableau = commun.csv_resultat(nom_fichier)
        if tableau is None:
            st.caption(f"`docs/resultats/{nom_fichier}` pas encore produit.")
            continue
        st.dataframe(tableau, hide_index=True, width="stretch", placeholder="—")
        commun.sources(f"docs/resultats/{nom_fichier}")


st.title("Comparaison des modèles")
afficher_section_disponibles()
st.divider()
afficher_section_phase4()
st.divider()
afficher_section_autres_tableaux()
