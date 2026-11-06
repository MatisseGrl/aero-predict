#!/usr/bin/env bash
# Enchaine toutes les etapes de l'optimisation des reseaux (notebook 04h) pendant la nuit, sur FD001.
#
# Lancement, depuis la racine du depot, dans Git Bash :
#     bash scripts/nuit_04h.sh
# Suivi : sorties/resultats/04h_nuit.log (une ligne par etape, plus les lignes de chaque etape).
#
# Ordre (chaque etape relit les resultats de la precedente, voir scripts/optimiser_reseaux.py) :
#   1. nettoyer   : essais Optuna interrompus par un arret precedent -> FAIL (les essais finis sont gardes) ;
#   2. optuna     : LSTM (100 essais utiles, 11 processus) et CNN 1D (80 essais utiles, 4 processus) en
#                   parallele ; une etude deja avancee reprend la ou elle s'etait arretee ;
#   3. confirmer  : meilleur essai de chaque reseau contre la reference du 03b, 3 decoupages neufs ;
#   4. ablation   : familles d'entrees (pente, sans capteurs) sur le LSTM retenu ;
#   5. ensemble   : moyennes de 5 graines LSTM, puis 5 LSTM + 5 CNN ;
#   6. final      : modeles finaux et test officiel FD001, lu une seule fois.
# Duree estimee : 7 a 9 h sur la machine de Matisse (16 coeurs logiques), tout le processeur est occupe.
#
# Mise en veille : un petit processus Python demande a Windows de ne pas mettre la machine en veille tant
# que le script tourne (SetThreadExecutionState, aucun reglage de Windows n'est modifie ; la demande
# disparait a la fin du script). Fermer le capot met quand meme en veille.

set -euo pipefail  # arret a la premiere erreur

cd "$(dirname "$0")/.."  # racine du depot
PYTHON=./.venv/Scripts/python.exe
JOURNAL=sorties/resultats/04h_nuit.log
SCRIPT=scripts/optimiser_reseaux.py

# empeche la mise en veille automatique pendant tout le script
"$PYTHON" -c "import ctypes, time; ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001); time.sleep(10**7)" &
PID_EVEIL=$!
trap 'kill $PID_EVEIL 2>/dev/null || true' EXIT  # la demande disparait avec ce processus, meme en cas d'erreur

etape() {
    # affiche l'etape dans le journal, puis la lance en ajoutant sa sortie au journal
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') : $* ===" | tee -a "$JOURNAL"
    # "|| true" : grep renvoie une erreur quand il n'a aucune ligne a garder, ce qui ne doit pas tout arreter
    "$PYTHON" "$SCRIPT" "$@" 2>&1 | { grep -v -i "warning" || true; } | tee -a "$JOURNAL"
}

etape nettoyer

# les deux etudes Optuna en meme temps (11 + 4 = 15 processus d'un coeur)
echo "=== $(date '+%Y-%m-%d %H:%M:%S') : optuna lstm + cnn1d ===" | tee -a "$JOURNAL"
"$PYTHON" "$SCRIPT" optuna --reseau lstm --essais 100 --processus 11 >> "$JOURNAL" 2>&1 &
PID_LSTM=$!
"$PYTHON" "$SCRIPT" optuna --reseau cnn1d --essais 80 --processus 4 >> "$JOURNAL" 2>&1 &
PID_CNN=$!
wait $PID_LSTM
wait $PID_CNN

etape confirmer --reseau lstm --processus 15
etape confirmer --reseau cnn1d --processus 15
etape ablation --reseau lstm --processus 15
etape ensemble --processus 15
etape final --processus 10

echo "=== $(date '+%Y-%m-%d %H:%M:%S') : TOUT EST TERMINE ===" | tee -a "$JOURNAL"
