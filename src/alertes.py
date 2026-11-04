"""
Decision d'alerte de maintenance : un moteur doit-il etre envoye en maintenance, et la decision etait-elle juste ?

Pourquoi ce module ? L'application (app/, page "Tester un moteur") et la couche de donnees de l'application
(src/donnees_demo.py) affichent, pour une RUL predite et la RUL vraie, l'un des quatre statuts ci-dessous. La
regle est la meme que celle de `alertes_manquees` et `fausses_alertes` (src/evaluation.py) : elle est ecrite
une seule fois, ici, et testee par tests/test_alertes.py.

Regle : le modele declenche une alerte des que la RUL predite est INFERIEURE OU EGALE au seuil ; le moteur
"aurait du" etre arrete si sa vraie RUL est inferieure ou egale au seuil. Exemple avec un seuil de 30 cycles :
RUL predite 45, RUL vraie 12 -> pas d'alerte alors qu'il en fallait une -> "ALERTE MANQUEE" (le cas dangereux).
"""

# seuil d'intervention par defaut, en cycles (hypothese de travail de l'equipe, voir src/evaluation.py)
SEUIL_ALERTE_DEFAUT = 30.0

# les 4 statuts possibles d'un moteur au regard de l'alerte (meme definition que src/evaluation.py)
STATUT_ALERTE_CORRECTE = "alerte correcte"  # vraie RUL <= seuil ET RUL predite <= seuil
STATUT_ALERTE_MANQUEE = "ALERTE MANQUEE"  # vraie RUL <= seuil MAIS RUL predite > seuil (dangereux)
STATUT_FAUSSE_ALERTE = "fausse alerte"  # vraie RUL > seuil MAIS RUL predite <= seuil (cout, pas de risque)
STATUT_RIEN_A_SIGNALER = "rien a signaler"  # vraie RUL > seuil ET RUL predite > seuil


def classer_alerte(rul_predite, rul_vraie, seuil_alerte=SEUIL_ALERTE_DEFAUT):
    """
    Dit si le modele declenche une alerte, et si cette decision etait la bonne.

    Regle (identique a `alertes_manquees` / `fausses_alertes` de src/evaluation.py) : le modele
    declenche une alerte des que la RUL predite est INFERIEURE OU EGALE au seuil ; le moteur
    "aurait du" etre arrete si sa vraie RUL est inferieure ou egale au seuil. Un cas limite
    egal au seuil (30 cycles predits, ou 30 cycles vrais) compte donc comme "en alerte".

    Parameters
    ----------
    rul_predite : float
        La RUL predite (en cycles).
    rul_vraie : float
        La vraie RUL (en cycles).
    seuil_alerte : float, optionnel
        Le seuil d'intervention (30 cycles par defaut, hypothese de travail de l'equipe).

    Returns
    -------
    (bool, str)
        - Vrai si une alerte est declenchee (RUL predite <= seuil) ;
        - le statut : STATUT_ALERTE_CORRECTE, STATUT_ALERTE_MANQUEE, STATUT_FAUSSE_ALERTE ou
          STATUT_RIEN_A_SIGNALER.
    """
    alerte_declenchee = float(rul_predite) <= seuil_alerte  # decision du modele
    aurait_du_alerter = float(rul_vraie) <= seuil_alerte  # ce qu'il aurait fallu faire

    if alerte_declenchee and aurait_du_alerter:  # bonne alerte
        statut = STATUT_ALERTE_CORRECTE
    elif aurait_du_alerter:  # il fallait alerter, le modele ne l'a pas fait : le cas dangereux
        statut = STATUT_ALERTE_MANQUEE
    elif alerte_declenchee:  # alerte alors que le moteur avait de la marge
        statut = STATUT_FAUSSE_ALERTE
    else:  # ni alerte, ni besoin d'alerte
        statut = STATUT_RIEN_A_SIGNALER

    return bool(alerte_declenchee), statut
