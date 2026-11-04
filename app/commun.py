"""
Outils Streamlit partages par toutes les pages du tableau de bord : caches, messages, mise en forme.

Les caches (`st.cache_data`, `st.cache_resource`) rendent l'application instantanee apres le premier
affichage. Leur cle contient la "signature" des fichiers lus (date de modification + taille) : si un
script reecrit un fichier de predictions pendant que l'application tourne, la nouvelle version est
relue automatiquement, sans redemarrer.
"""

import os

import streamlit as st

from src import donnees_demo as dd
from src.config_modele import CHEMIN_CONFIG_RETENUE, RACINE_DEPOT
from src.contrat_predictions import chemin_metadonnees, chemin_predictions
from src.demo import STATUT_ALERTE_CORRECTE, STATUT_ALERTE_MANQUEE, STATUT_FAUSSE_ALERTE, STATUT_RIEN_A_SIGNALER

# couleur du badge Streamlit de chaque statut (rouge reserve au cas dangereux)
COULEURS_BADGES = {
    STATUT_ALERTE_MANQUEE: "red",
    STATUT_ALERTE_CORRECTE: "green",
    STATUT_FAUSSE_ALERTE: "orange",
    STATUT_RIEN_A_SIGNALER: "gray",
}

# commande a lancer pour produire les fichiers de predictions (affichee quand il n'y en a aucun)
COMMANDE_PREPARATION = "python scripts/preparer_demo.py"
COMMANDE_ARBRES_FD001 = "python scripts/generer_predictions_arbres.py --sous-jeux FD001"


# ---------------------------------------------------------------------------------------------------
# Mise en forme des nombres (a la francaise)
# ---------------------------------------------------------------------------------------------------

def fr(valeur, decimales=1):
    """
    Ecrit un nombre a la francaise : virgule decimale, espace pour les milliers (1053.9 -> "1 053,9").

    Parameters
    ----------
    valeur : float
        Le nombre.
    decimales : int, optionnel
        Nombre de decimales (1 par defaut).

    Returns
    -------
    str
        Le nombre formate.
    """
    texte_anglais = f"{float(valeur):,.{decimales}f}"  # "1,053.9"
    texte_temporaire = texte_anglais.replace(",", "_")  # "1_053.9"
    texte_virgule = texte_temporaire.replace(".", ",")  # "1_053,9"
    return texte_virgule.replace("_", " ")  # espace fine insecable pour les milliers


def fr_signe(valeur, decimales=1):
    """
    Ecrit un nombre a la francaise avec son signe ("+3,2", "-12,4").

    Parameters
    ----------
    valeur : float
        Le nombre.
    decimales : int, optionnel
        Nombre de decimales (1 par defaut).

    Returns
    -------
    str
        Le nombre signe.
    """
    if float(valeur) >= 0:
        return "+" + fr(valeur, decimales)
    # tiret ASCII : st.metric reconnait un ecart negatif (fleche vers le bas) a ce seul caractere
    return "-" + fr(abs(float(valeur)), decimales)


def delta_comparaison(valeur, reference, decimales, libelle_reference):
    """
    Prepare l'ecart affiche sous un st.metric quand on compare une valeur a une reference.

    Ecart nul (a la precision affichee) : texte "identique a ...", gris, sans fleche. Sinon : ecart signe,
    couleurs inversees (pour une metrique ou "plus petit = mieux" : une hausse est en rouge).

    Parameters
    ----------
    valeur : float
        La valeur affichee.
    reference : float
        La valeur de reference.
    decimales : int
        Precision d'affichage.
    libelle_reference : str
        Le nom de la reference (ex. "XGBoost Phase 3").

    Returns
    -------
    dict
        Arguments a passer a st.metric : delta, delta_color, delta_arrow.
    """
    ecart = round(float(valeur) - float(reference), decimales)
    arguments = {}
    if ecart == 0:
        arguments["delta"] = f"identique à {libelle_reference}"
        arguments["delta_color"] = "gray"
        arguments["delta_arrow"] = "off"
    else:
        arguments["delta"] = f"{fr_signe(ecart, decimales)} vs {libelle_reference}"
        arguments["delta_color"] = "inverse"
        arguments["delta_arrow"] = "auto"
    return arguments


def chemin_lisible(chemin):
    """
    Raccourcit un chemin pour l'affichage : relatif a la racine du depot s'il est dedans.

    Parameters
    ----------
    chemin : str
        Le chemin absolu.

    Returns
    -------
    str
        Par exemple "sorties/predictions".
    """
    chemin_absolu = os.path.abspath(chemin)
    if chemin_absolu.startswith(RACINE_DEPOT + os.sep):
        return os.path.relpath(chemin_absolu, RACINE_DEPOT)
    return chemin_absolu


# ---------------------------------------------------------------------------------------------------
# Caches : predictions (via le contrat)
# ---------------------------------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _inventaire_en_cache(dossier, signature):
    """
    Version en cache de `inventaire_predictions` (la signature du dossier fait partie de la cle).

    Parameters
    ----------
    dossier : str
        Le dossier des predictions.
    signature : tuple
        La signature du dossier (`signature_dossier`) : change si un fichier change.

    Returns
    -------
    pandas.DataFrame
        L'inventaire.
    """
    return dd.inventaire_predictions(dossier)


def inventaire():
    """
    Renvoie l'inventaire des fichiers de predictions disponibles (relu si un fichier a change).

    Returns
    -------
    pandas.DataFrame
        Le resultat de `inventaire_predictions` pour le dossier courant.
    """
    dossier = dd.dossier_predictions()
    return _inventaire_en_cache(dossier, dd.signature_dossier(dossier))


@st.cache_data(show_spinner="Lecture du fichier de prédictions…")
def _predictions_en_cache(sous_jeu, modele, dossier, signature):
    """
    Version en cache de `charger_predictions`.

    Parameters
    ----------
    sous_jeu, modele : str
        Le couple a lire.
    dossier : str
        Le dossier des predictions.
    signature : tuple
        Signatures du CSV et du JSON (cle de cache).

    Returns
    -------
    tuple
        (DataFrame, metadonnees).
    """
    return dd.charger_predictions(sous_jeu, modele, dossier)


def predictions(sous_jeu, modele):
    """
    Lit les predictions d'un (sous-jeu, modele), depuis le cache si le fichier n'a pas change.

    Parameters
    ----------
    sous_jeu, modele : str
        Le couple a lire.

    Returns
    -------
    tuple
        (DataFrame des predictions, metadonnees).
    """
    dossier = dd.dossier_predictions()
    signature = (
        dd.signature_fichier(chemin_predictions(sous_jeu, modele, dossier)),
        dd.signature_fichier(chemin_metadonnees(sous_jeu, modele, dossier)),
    )
    return _predictions_en_cache(sous_jeu, modele, dossier, signature)


@st.cache_data(show_spinner="Calcul du tableau de la flotte…")
def _flotte_en_cache(sous_jeu, modele, dossier, signature, seuil_alerte):
    """
    Version en cache du tableau de flotte et de son resume (dependent du seuil d'alerte).

    Parameters
    ----------
    sous_jeu, modele, dossier : str
        Le fichier concerne.
    signature : tuple
        Signature des fichiers (cle de cache).
    seuil_alerte : float
        Le seuil d'alerte.

    Returns
    -------
    tuple
        (tableau de flotte, resume des indicateurs).
    """
    df, meta = dd.charger_predictions(sous_jeu, modele, dossier)
    tableau = dd.tableau_flotte(df, seuil_alerte)
    resume = dd.resumer_tableau_flotte(tableau, seuil_alerte)
    return tableau, resume


def flotte(sous_jeu, modele, seuil_alerte):
    """
    Renvoie le tableau de flotte et son resume, depuis le cache si possible.

    Parameters
    ----------
    sous_jeu, modele : str
        Le couple a afficher.
    seuil_alerte : float
        Le seuil d'alerte.

    Returns
    -------
    tuple
        (tableau de flotte, resume).
    """
    dossier = dd.dossier_predictions()
    signature = (
        dd.signature_fichier(chemin_predictions(sous_jeu, modele, dossier)),
        dd.signature_fichier(chemin_metadonnees(sous_jeu, modele, dossier)),
    )
    return _flotte_en_cache(sous_jeu, modele, dossier, signature, float(seuil_alerte))


@st.cache_data(show_spinner="Lecture des métadonnées…")
def _comparaison_en_cache(dossier, signature):
    """
    Version en cache du tableau de comparaison, du detail de validation et des controles de coherence.

    Parameters
    ----------
    dossier : str
        Le dossier des predictions.
    signature : tuple
        Signature du dossier (cle de cache).

    Returns
    -------
    tuple
        (tableau_comparaison, tableau_validation_detaille, controler_tous_les_fichiers).
    """
    liste = dd.inventaire_predictions(dossier)
    comparaison = dd.tableau_comparaison(liste, dossier)
    validation = dd.tableau_validation_detaille(liste, dossier)
    controles = dd.controler_tous_les_fichiers(liste, dossier)
    return comparaison, validation, controles


def comparaison():
    """
    Renvoie (tableau de comparaison, detail de validation, controles de coherence) pour tous les fichiers.

    Returns
    -------
    tuple
        Voir `_comparaison_en_cache`.
    """
    dossier = dd.dossier_predictions()
    return _comparaison_en_cache(dossier, dd.signature_dossier(dossier))


@st.cache_data(show_spinner="Tests appariés…")
def _tests_apparies_en_cache(modele_a, modele_b, dossier, signature):
    """
    Version en cache de `comparer_deux_modeles`.

    Parameters
    ----------
    modele_a, modele_b : str
        Les deux modeles compares.
    dossier : str
        Le dossier des predictions.
    signature : tuple
        Signature du dossier (cle de cache).

    Returns
    -------
    pandas.DataFrame
        Le resultat de `comparer_deux_modeles`.
    """
    liste = dd.inventaire_predictions(dossier)
    return dd.comparer_deux_modeles(liste, modele_a, modele_b, dossier)


def tests_apparies(modele_a, modele_b):
    """
    Compare deux modeles moteur par moteur sur chaque sous-jeu commun (depuis le cache si possible).

    Parameters
    ----------
    modele_a, modele_b : str
        Les deux modeles compares.

    Returns
    -------
    pandas.DataFrame
        Le resultat de `comparer_deux_modeles`.
    """
    dossier = dd.dossier_predictions()
    return _tests_apparies_en_cache(modele_a, modele_b, dossier, dd.signature_dossier(dossier))


# ---------------------------------------------------------------------------------------------------
# Caches : donnees brutes, resultats de la Phase 4, artefact
# ---------------------------------------------------------------------------------------------------

@st.cache_data(show_spinner="Lecture des données capteurs brutes…")
def donnees_brutes_test(sous_jeu):
    """
    Donnees capteurs brutes du jeu de test d'un sous-jeu (en cache : le fichier ne change pas).

    Parameters
    ----------
    sous_jeu : str
        Le sous-jeu.

    Returns
    -------
    pandas.DataFrame
        Le resultat de `charger_donnees_brutes_test`.
    """
    return dd.charger_donnees_brutes_test(sous_jeu)


@st.cache_data(show_spinner=False)
def _csv_resultat_en_cache(nom_fichier, signature):
    """
    Version en cache de `lire_csv_resultat`.

    Parameters
    ----------
    nom_fichier : str
        Le nom du CSV dans docs/resultats/.
    signature : tuple
        Signature du fichier (cle de cache).

    Returns
    -------
    pandas.DataFrame ou None
        Le tableau, ou None s'il n'existe pas.
    """
    return dd.lire_csv_resultat(nom_fichier)


def csv_resultat(nom_fichier):
    """
    Lit un CSV de docs/resultats/ (None s'il n'existe pas), depuis le cache si possible.

    Parameters
    ----------
    nom_fichier : str
        Le nom du fichier.

    Returns
    -------
    pandas.DataFrame ou None
        Le tableau.
    """
    chemin = os.path.join(dd.DOSSIER_RESULTATS, nom_fichier)
    return _csv_resultat_en_cache(nom_fichier, dd.signature_fichier(chemin))


@st.cache_data(show_spinner=False)
def _config_retenue_en_cache(signature):
    """
    Version en cache de `lire_config_retenue`.

    Parameters
    ----------
    signature : tuple
        Signature du fichier (cle de cache).

    Returns
    -------
    dict ou None
        La configuration retenue.
    """
    return dd.lire_config_retenue()


def config_retenue():
    """
    Lit docs/resultats/04_config_retenue.json (None s'il n'existe pas).

    Returns
    -------
    dict ou None
        La configuration retenue.
    """
    return _config_retenue_en_cache(dd.signature_fichier(CHEMIN_CONFIG_RETENUE))


@st.cache_data(show_spinner="Reconstruction du front de Pareto…")
def _optuna_en_cache(signature):
    """
    Version en cache de `analyser_optuna`.

    Parameters
    ----------
    signature : tuple
        Signatures des CSV Optuna et de la configuration (cle de cache).

    Returns
    -------
    dict ou None
        Le resultat de `analyser_optuna`.
    """
    return dd.analyser_optuna()


def optuna():
    """
    Analyse des essais Optuna (front de Pareto, regle de choix rejouee), depuis le cache si possible.

    Returns
    -------
    dict ou None
        Le resultat de `analyser_optuna`.
    """
    signatures = []
    for nom in sorted(os.listdir(dd.DOSSIER_RESULTATS)):
        if nom.startswith("04_optuna_"):
            signatures.append(dd.signature_fichier(os.path.join(dd.DOSSIER_RESULTATS, nom)))
    signatures.append(dd.signature_fichier(CHEMIN_CONFIG_RETENUE))
    return _optuna_en_cache(tuple(signatures))


@st.cache_resource(show_spinner="Chargement du modèle retenu (entraînement automatique s'il manque)…")
def _artefact_en_cache(chemin, signature):
    """
    Charge l'artefact (et l'entraine s'il manque) une seule fois par version du fichier.

    Parameters
    ----------
    chemin : str
        Le fichier de l'artefact.
    signature : tuple
        Signature du fichier (cle de cache).

    Returns
    -------
    dict
        L'artefact.
    """
    return dd.charger_artefact_demo(chemin)


def artefact():
    """
    Renvoie l'artefact du modele retenu (modele + preparation apprise sur le train).

    Returns
    -------
    dict
        L'artefact pret a predire.
    """
    chemin = dd.chemin_artefact_demo()
    return _artefact_en_cache(chemin, dd.signature_fichier(chemin))


@st.cache_resource(show_spinner=False)
def _artefact_si_present_en_cache(chemin, signature):
    """
    Charge l'artefact s'il existe, sans jamais l'entrainer (page d'accueil).

    Parameters
    ----------
    chemin : str
        Le fichier de l'artefact.
    signature : tuple
        Signature du fichier (cle de cache).

    Returns
    -------
    dict ou None
        L'artefact, ou None s'il n'existe pas.
    """
    return dd.charger_artefact_si_present(chemin)


def artefact_si_present():
    """
    Renvoie l'artefact s'il existe deja sur le disque (None sinon), sans entrainement.

    Returns
    -------
    dict ou None
        L'artefact.
    """
    chemin = dd.chemin_artefact_demo()
    return _artefact_si_present_en_cache(chemin, dd.signature_fichier(chemin))


@st.cache_data(show_spinner=False)
def donnees_test_fd001():
    """
    Donnees test brutes FD001 et RUL vraie au dernier cycle (en cache).

    Returns
    -------
    tuple
        (DataFrame brut, tableau numpy des 100 RUL).
    """
    return dd.donnees_test_fd001()


# ---------------------------------------------------------------------------------------------------
# Elements d'interface reutilises
# ---------------------------------------------------------------------------------------------------

def message_aucun_fichier():
    """
    Affiche le message "aucun fichier de predictions", avec le dossier lu et la commande a lancer.

    Returns
    -------
    None
    """
    dossier = dd.dossier_predictions()
    st.warning(
        f"Aucun fichier de prédictions trouvé dans `{chemin_lisible(dossier)}`.\n\n"
        f"Pour les produire, lancer depuis la racine du dépôt : `{COMMANDE_PREPARATION}` "
        f"(ou, pour les seuls modèles à arbres sur FD001 : `{COMMANDE_ARBRES_FD001}`).\n\n"
        "Les pages « Et si… » et « Fiabilité » fonctionnent sans ces fichiers.",
        icon=":material/folder_off:",
    )


def badge_statut(statut):
    """
    Affiche le statut d'alerte d'un moteur sous forme de badge colore.

    Parameters
    ----------
    statut : str
        Un statut de src/demo.py.

    Returns
    -------
    None
    """
    st.badge(
        dd.LIBELLES_STATUTS.get(statut, statut), color=COULEURS_BADGES.get(statut, "gray"),
        help=dd.EXPLICATIONS_STATUTS.get(statut),
    )


def restaurer_choix(cle, options):
    """
    Remet dans un widget la valeur choisie lors d'une visite precedente de la page (si elle est encore valide).

    Streamlit oublie la valeur d'un widget quand on change de page ; on la garde donc sous une
    seconde cle ("memo_<cle>"), recopiee par `memoriser_choix` apres chaque affichage.

    Parameters
    ----------
    cle : str
        La cle du widget.
    options : list
        Les options possibles du widget (une valeur memorisee absente des options est ignoree).

    Returns
    -------
    None
    """
    cle_memo = "memo_" + cle
    if cle in st.session_state:
        valeur = st.session_state[cle]
        if isinstance(valeur, list):  # multiselect : on retire les valeurs devenues invalides
            valides = []
            for element in valeur:
                if element in options:
                    valides.append(element)
            st.session_state[cle] = valides
        elif valeur not in options:
            del st.session_state[cle]  # valeur devenue invalide (ex. autre sous-jeu) : on l'oublie
        return
    if cle_memo not in st.session_state:
        return
    valeur = st.session_state[cle_memo]
    if isinstance(valeur, list):
        valides = []
        for element in valeur:
            if element in options:
                valides.append(element)
        st.session_state[cle] = valides
    elif valeur in options:
        st.session_state[cle] = valeur


def memoriser_choix(cle):
    """
    Recopie la valeur actuelle d'un widget sous "memo_<cle>" (voir `restaurer_choix`).

    Parameters
    ----------
    cle : str
        La cle du widget.

    Returns
    -------
    None
    """
    if cle in st.session_state:
        st.session_state["memo_" + cle] = st.session_state[cle]


def sources(*fichiers):
    """
    Affiche, en petit, la liste des fichiers d'ou viennent les chiffres de la section.

    Parameters
    ----------
    *fichiers : str
        Les chemins (relatifs a la racine du depot) des fichiers lus.

    Returns
    -------
    None
    """
    morceaux = []
    for fichier in fichiers:
        morceaux.append(f"`{fichier}`")
    st.caption("Source : " + ", ".join(morceaux))
