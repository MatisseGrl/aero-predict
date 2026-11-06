#!/usr/bin/env bash
# Enchaine toutes les etapes de l'optimisation des reseaux (notebook 04h) pendant la nuit : FD001, puis FD002.
#
# Lancement, depuis la racine du depot, dans Git Bash :
#     bash scripts/nuit_04h.sh
#     AERO_PROCESSUS=8 bash scripts/nuit_04h.sh     # moins de processus : moins de chaleur, plus lent
# Suivi : sorties/resultats/04h_nuit.log (une ligne par etape, plus les lignes de chaque etape).
# Arret propre le matin : bash scripts/arreter_04h.sh (les essais finis restent dans la base).
#
# Reprise : relancer la meme commande la nuit suivante. Une etude Optuna deja complete ne relance aucun
# essai, et une etape dont le fichier de resultats existe deja est sautee : le script reprend la ou il
# s'etait arrete (a l'essai Optuna pres, l'essai interrompu est refait).
#
# Ordre, pour chaque sous-jeu (chaque etape relit les resultats de la precedente, voir scripts/optimiser_reseaux.py) :
#   1. optuna     : LSTM (100 essais utiles, 3/4 des processus) et CNN 1D (80 essais utiles, le reste) en
#                   parallele ; quand le CNN a fini, autant de travailleurs LSTM de plus prennent ses coeurs ;
#   2. confirmer  : meilleur essai de chaque reseau contre la reference du 03b, 3 decoupages neufs ;
#   3. ablation   : familles d'entrees (pente, sans capteurs) sur le LSTM retenu ;
#   4. ensemble   : moyennes de 5 graines LSTM, puis 5 LSTM + 5 CNN ;
#   5. final      : modeles finaux et test officiel du sous-jeu, lu une seule fois.
# Nombre de processus (1 coeur chacun) : 12 par defaut (AERO_PROCESSUS). Mesure du 2026-11-07 (FD002, LSTM de
# reference, 8 coeurs physiques et 16 logiques) : 8 processus 14 600 fenetres/s au total, 12 processus 18 900,
# 15 processus 19 400 ; 12 donnent donc 97 % du debit de 15, avec moins de chaleur et de memoire (environ
# 0,6 Go par processus avec les fenetres paresseuses).
# Durees estimees (tout le processeur est occupe) : fin de FD001 environ 4 h ; FD002 environ 11 h en tout,
# donc sur deux nuits.
#
# Mise en veille : un petit processus Python demande a Windows de ne pas mettre la machine en veille tant
# que le script tourne (SetThreadExecutionState, aucun reglage de Windows n'est modifie ; la demande
# disparait a la fin du script). Fermer le capot met quand meme en veille.

set -euo pipefail  # arret a la premiere erreur

cd "$(dirname "$0")/.."  # racine du depot
PYTHON=./.venv/Scripts/python.exe
RESULTATS=sorties/resultats
JOURNAL=$RESULTATS/04h_nuit.log
SCRIPT=scripts/optimiser_reseaux.py
ESSAIS_LSTM=100
ESSAIS_CNN=80
PROCESSUS=${AERO_PROCESSUS:-12}
PROCESSUS_CNN=$(( PROCESSUS / 4 ))  # 12 -> 3 pour le CNN, 9 pour le LSTM
if [ "$PROCESSUS_CNN" -lt 1 ]; then
    PROCESSUS_CNN=1
fi
PROCESSUS_LSTM=$(( PROCESSUS - PROCESSUS_CNN ))
PROCESSUS_FINAL=$(( PROCESSUS < 10 ? PROCESSUS : 10 ))  # 10 modeles finaux au plus
if [ "${AERO_ESSAI_RAPIDE:-0}" = "1" ]; then
    # essai rapide (verification du script, quelques minutes) : 2 epoques par reseau, tres peu d'essais,
    # tout est ecrit dans sorties/essai_rapide (base Optuna comprise), jamais dans les vrais resultats
    RESULTATS=sorties/essai_rapide
    JOURNAL=$RESULTATS/04h_nuit.log
    ESSAIS_LSTM=3
    ESSAIS_CNN=2
fi
mkdir -p "$RESULTATS"

# empeche la mise en veille automatique pendant tout le script
"$PYTHON" -c "import ctypes, time; ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001); time.sleep(10**7)" &
PID_EVEIL=$!
trap 'kill $PID_EVEIL 2>/dev/null || true' EXIT  # la demande disparait avec ce processus, meme en cas d'erreur

horodatage() {
    date '+%Y-%m-%d %H:%M:%S'
}

etape() {
    # affiche l'etape dans le journal, puis la lance en ajoutant sa sortie au journal
    echo "=== $(horodatage) : $* ===" | tee -a "$JOURNAL"
    # "|| true" : grep renvoie une erreur quand il n'a aucune ligne a garder, ce qui ne doit pas tout arreter
    "$PYTHON" "$SCRIPT" "$@" 2>&1 | { grep -v -i "warning" || true; } | tee -a "$JOURNAL"
}

etape_si_absente() {
    # $1 : fichier de resultats de l'etape ; la suite : l'etape. Sautee si le fichier existe deja.
    local fichier=$RESULTATS/$1
    shift
    if [ -f "$fichier" ]; then
        echo "=== $(horodatage) : deja faite, sautee : $* ($fichier existe) ===" | tee -a "$JOURNAL"
    else
        etape "$@"
    fi
}

optuna_sous_jeu() {
    # $1 : sous-jeu ; $2 : debut des noms de fichiers ("04h_" pour FD001, "04h_fd002_" pour FD002)
    local sous_jeu=$1
    local prefixe=$2
    echo "=== $(horodatage) : optuna lstm + cnn1d $sous_jeu ===" | tee -a "$JOURNAL"
    "$PYTHON" "$SCRIPT" optuna --sous-jeu "$sous_jeu" --reseau lstm --essais $ESSAIS_LSTM --processus $PROCESSUS_LSTM >> "$JOURNAL" 2>&1 &
    local pid_lstm=$!
    "$PYTHON" "$SCRIPT" optuna --sous-jeu "$sous_jeu" --reseau cnn1d --essais $ESSAIS_CNN --processus $PROCESSUS_CNN >> "$JOURNAL" 2>&1 &
    local pid_cnn=$!
    wait $pid_cnn

    # le CNN a fini : ses coeurs passent au LSTM (un travailleur ne fait rien si l'etude est deja finie)
    local pids_renforts=""
    local numero
    for numero in $(seq $PROCESSUS_LSTM $(( PROCESSUS - 1 ))); do
        "$PYTHON" "$SCRIPT" travailleur --sous-jeu "$sous_jeu" --reseau lstm --essais $ESSAIS_LSTM --numero $numero \
            > "$RESULTATS/${prefixe}journal_lstm_$numero.log" 2>&1 &
        pids_renforts="$pids_renforts $!"
    done
    wait $pid_lstm
    local pid
    for pid in $pids_renforts; do
        wait $pid
    done
    # resume refait une fois TOUS les travailleurs finis (les renforts peuvent finir apres le resume de l'etape)
    etape resumer --sous-jeu "$sous_jeu" --reseau lstm
}

etapes_sous_jeu() {
    # toutes les etapes d'un sous-jeu ; $1 : sous-jeu ; $2 : debut des noms de fichiers
    local sous_jeu=$1
    local prefixe=$2
    optuna_sous_jeu "$sous_jeu" "$prefixe"
    etape_si_absente "${prefixe}confirmation_lstm.json" confirmer --sous-jeu "$sous_jeu" --reseau lstm --processus $PROCESSUS
    etape_si_absente "${prefixe}confirmation_cnn1d.json" confirmer --sous-jeu "$sous_jeu" --reseau cnn1d --processus $PROCESSUS
    etape_si_absente "${prefixe}ablation_lstm.json" ablation --sous-jeu "$sous_jeu" --reseau lstm --processus $PROCESSUS
    etape_si_absente "${prefixe}ensemble.json" ensemble --sous-jeu "$sous_jeu" --processus $PROCESSUS
    etape_si_absente "${prefixe}test.json" final --sous-jeu "$sous_jeu" --processus $PROCESSUS_FINAL
}

etape nettoyer  # essais Optuna interrompus par un arret precedent -> FAIL (les essais finis sont gardes)

etapes_sous_jeu FD001 04h_
etapes_sous_jeu FD002 04h_fd002_

echo "=== $(horodatage) : TOUT EST TERMINE (FD001 et FD002) ===" | tee -a "$JOURNAL"
