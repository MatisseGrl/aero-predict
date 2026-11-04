"""
Outils Streamlit partages par les deux pages de l'application : caches, mise en forme, petits elements.

Les caches (`st.cache_data`, `st.cache_resource`) rendent l'application instantanee apres le premier
affichage. Leur cle contient la "signature" des fichiers lus (date de modification + taille) : si un script
reecrit un fichier de predictions ou de modele pendant que l'application tourne, il est relu automatiquement.
"""

import os

import streamlit as st

from src import donnees_demo as dd
from src.alertes import STATUT_ALERTE_CORRECTE, STATUT_ALERTE_MANQUEE, STATUT_FAUSSE_ALERTE, STATUT_RIEN_A_SIGNALER

# couleur du badge Streamlit de chaque statut (rouge reserve au cas dangereux)
COULEURS_BADGES = {
    STATUT_ALERTE_MANQUEE: "red",
    STATUT_ALERTE_CORRECTE: "green",
    STATUT_FAUSSE_ALERTE: "orange",
    STATUT_RIEN_A_SIGNALER: "gray",
}

# racine du depot = dossier parent de app/
RACINE_DEPOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# commande a lancer pour produire les fichiers de predictions et les modeles (affichee s'il en manque)
COMMANDE_PREPARATION = "python scripts/preparer_demo.py"


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
    return texte_virgule.replace("_", " ")  # espace fine insecable pour les milliers


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
# Caches : fichiers de predictions (page "Comparer les modeles")
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


@st.cache_data(show_spinner="Lecture des fichiers de prédictions…")
def _comparaison_en_cache(sous_jeu, dossier, signature):
    """
    Version en cache de `tableau_comparaison_sous_jeu`.

    Parameters
    ----------
    sous_jeu : str
        Le sous-jeu.
    dossier : str
        Le dossier des predictions.
    signature : tuple
        Signature du dossier (cle de cache).

    Returns
    -------
    pandas.DataFrame
        Le tableau de comparaison du sous-jeu.
    """
    liste = dd.inventaire_predictions(dossier)
    return dd.tableau_comparaison_sous_jeu(liste, sous_jeu, dossier)


def comparaison(sous_jeu):
    """
    Tableau de comparaison des modeles affiches d'un sous-jeu, depuis le cache si possible.

    Parameters
    ----------
    sous_jeu : str
        Le sous-jeu.

    Returns
    -------
    pandas.DataFrame
        Le resultat de `tableau_comparaison_sous_jeu`.
    """
    dossier = dd.dossier_predictions()
    return _comparaison_en_cache(sous_jeu, dossier, dd.signature_dossier(dossier))


# ---------------------------------------------------------------------------------------------------
# Caches : inference en direct (page "Tester un moteur")
# ---------------------------------------------------------------------------------------------------

@st.cache_resource(show_spinner="Chargement des modèles entraînés…")
def _modeles_inference_en_cache(sous_jeu, dossier, signatures):
    """
    Charge une seule fois par sous-jeu (et par version des fichiers de modeles) l'objet d'inference.

    Parameters
    ----------
    sous_jeu : str
        Le sous-jeu.
    dossier : str
        Le dossier des modeles.
    signatures : tuple
        Signatures des fichiers de modeles du sous-jeu (cle de cache : un modele regenere est relu).

    Returns
    -------
    dict
        Le resultat de `src.inference.charger_modeles`.
    """
    from src import inference as inf  # import ici : seule cette page charge PyTorch

    return inf.charger_modeles(sous_jeu, dossier=dossier)


def modeles_inference(sous_jeu):
    """
    Renvoie l'objet d'inference d'un sous-jeu (preparation apprise sur le train + modeles), en cache.

    Parameters
    ----------
    sous_jeu : str
        Le sous-jeu.

    Returns
    -------
    dict
        Le resultat de `src.inference.charger_modeles`.
    """
    from src import inference as inf

    dossier = inf.dossier_modeles()
    signatures = []
    for modele in inf.MODELES_ARBRES:
        signatures.append(dd.signature_fichier(inf.chemin_artefact_arbres(sous_jeu, modele, dossier)))
    for modele in inf.MODELES_RESEAUX:
        signatures.append(dd.signature_fichier(inf.chemin_poids_reseau(sous_jeu, modele, dossier)))
    return _modeles_inference_en_cache(sous_jeu, dossier, tuple(signatures))


@st.cache_data(show_spinner=False)
def moteurs_test(sous_jeu):
    """
    Numeros des moteurs du jeu de test d'un sous-jeu (en cache : le fichier ne change pas).

    Parameters
    ----------
    sous_jeu : str
        Le sous-jeu.

    Returns
    -------
    list de int
        Les numeros de moteurs.
    """
    from src import inference as inf

    return inf.lister_moteurs_test(sous_jeu)


# ---------------------------------------------------------------------------------------------------
# Elements d'interface reutilises
# ---------------------------------------------------------------------------------------------------

def badge_statut(statut):
    """
    Affiche le statut d'alerte d'un moteur sous forme de badge colore.

    Parameters
    ----------
    statut : str
        Un statut de src/alertes.py.

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
