"""
Configuration partagee par tous les tests pytest du depot.

pytest charge automatiquement ce fichier avant les tests. Il sert a deux choses :
1. rendre `import src...` possible, quel que soit le dossier depuis lequel on lance pytest ;
2. fournir des "fixtures" : des objets prepares une seule fois (ici le pipeline train/test complet,
   qui prend quelques secondes) puis partages par tous les tests qui en ont besoin.
"""

import os
import sys

import pytest

# on ajoute la racine du depot (dossier parent de tests/) au chemin d'import de Python
DOSSIER_TESTS = os.path.dirname(os.path.abspath(__file__))
RACINE_DEPOT = os.path.dirname(DOSSIER_TESTS)
if RACINE_DEPOT not in sys.path:  # evite d'ajouter deux fois le meme dossier
    sys.path.insert(0, RACINE_DEPOT)

from src.features import preparer_test, preparer_train  # noqa: E402


@pytest.fixture(scope="session")
def contexte_train():
    """
    Prepare le train FD001 complet (pipeline du notebook 03), une seule fois pour toute la session.

    Returns
    -------
    dict
        Le dictionnaire renvoye par `preparer_train()`.
    """
    return preparer_train()


@pytest.fixture(scope="session")
def contexte_test(contexte_train):
    """
    Prepare le test FD001 complet a partir du contexte train, une seule fois pour toute la session.

    Parameters
    ----------
    contexte_train : dict
        La fixture ci-dessus (pytest la fournit automatiquement).

    Returns
    -------
    dict
        Le dictionnaire renvoye par `preparer_test(contexte_train)`.
    """
    return preparer_test(contexte_train)
