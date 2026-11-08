#!/usr/bin/env bash
# Arrete proprement, sur macOS, le calcul de nuit des reseaux (scripts/nuit_04h_fd003_fd004_mac.sh, ou
# scripts/nuit_04h.sh lance sur un Mac) : le lanceur, tous les processus Python de scripts/optimiser_reseaux.py,
# leurs processus enfants de calcul parallele, et le caffeinate (anti-veille) du lanceur. Equivalent Mac de
# scripts/arreter_04h.sh (Windows, PowerShell).
#
# Lancement, depuis la racine du depot, dans le Terminal :
#     bash scripts/arreter_04h_mac.sh
#
# Les autres programmes Python (noyaux Jupyter, nbconvert, pytest, Streamlit...) ne sont PAS touches : on ne
# vise que les processus dont la ligne de commande cite le lanceur ou scripts/optimiser_reseaux.py, puis leurs
# descendants (relation parent -> enfant, `pgrep -P`).
#
# Les essais Optuna termines restent dans la base. Les essais en cours restent marques "en cours" (RUNNING) :
# l'etape "nettoyer", au debut du prochain lancement du script de nuit, les marque FAIL, et le calcul reprend.

set -uo pipefail  # pas de "-e" : un processus deja disparu entre deux commandes ne doit pas tout arreter

cd "$(dirname "$0")/.."  # racine du depot
RACINE=$(pwd -P)  # chemin absolu, liens symboliques resolus (sert a reconnaitre les orphelins, plus bas)

# --- Motifs (expressions regulieres de pgrep -f, appliquees a la ligne de commande complete) ---
# lanceur : un shell (bash ou sh) qui execute un des scripts de nuit ; on exige le shell devant, pour ne pas
# viser un editeur ou un "tail" ouvert sur le meme fichier
MOTIF_LANCEUR='(^|/)(ba)?sh .*scripts/nuit_04h[a-z0-9_]*\.sh'
# calcul : tout processus qui execute scripts/optimiser_reseaux.py (etapes, travailleurs Optuna, renforts)
MOTIF_CALCUL='scripts/optimiser_reseaux\.py'

# --- Processus a ne jamais viser : ce script et ses ancetres ---
# (si la commande qui lance ce script contenait un des motifs, pgrep la trouverait sinon)
PROTEGES=" "
pid_courant=$$
while [ -n "$pid_courant" ] && [ "$pid_courant" -gt 1 ]; do
    PROTEGES="$PROTEGES$pid_courant "
    pid_courant=$(ps -o ppid= -p "$pid_courant" 2>/dev/null | tr -d ' ')  # remonte au parent
done

est_protege() {
    # vrai si le numero $1 est ce script ou un de ses ancetres
    case "$PROTEGES" in
        *" $1 "*) return 0 ;;
        *) return 1 ;;
    esac
}

descendants() {
    # affiche tous les descendants du processus $1 (enfants, petits-enfants...), les plus profonds d'abord
    local enfant
    for enfant in $(pgrep -P "$1" 2>/dev/null); do
        descendants "$enfant"  # d'abord les descendants de l'enfant
        echo "$enfant"         # puis l'enfant lui-meme
    done
}

lister_cibles() {
    # affiche, un par ligne, tous les processus a arreter : enfants d'abord, puis leurs parents
    local racines=""
    local pid
    # 1. les lanceurs et les processus de calcul, trouves par leur ligne de commande
    for pid in $(pgrep -f "$MOTIF_LANCEUR" 2>/dev/null) $(pgrep -f "$MOTIF_CALCUL" 2>/dev/null); do
        if ! est_protege "$pid"; then
            racines="$racines $pid"
        fi
    done
    # 2. leurs descendants : processus de calcul parallele (ProcessPoolExecutor en mode "spawn" : leur ligne
    #    de commande est "python -c from multiprocessing.spawn import spawn_main ...", elle ne cite pas le
    #    script), gestionnaire de ressources de multiprocessing, caffeinate du lanceur, tubes grep / tee
    for pid in $racines; do
        descendants "$pid"
    done
    # 3. les racines elles-memes, apres leurs enfants
    for pid in $racines; do
        echo "$pid"
    done
    # 4. orphelins d'un arret precedent mal fini : processus de multiprocessing dont le parent a disparu (ils
    #    sont alors rattaches au processus n° 1 de macOS) ET dont le dossier courant est ce depot. Constate sous
    #    Windows le 2026-11-06 : 15 orphelins gardaient ~13 Go de memoire. Un orphelin de Jupyter ou de pytest
    #    lance depuis ce meme dossier serait aussi vise, mais un orphelin n'a de toute facon plus de parent a
    #    qui rendre son resultat.
    for pid in $(pgrep -f 'multiprocessing\.(spawn|resource_tracker)' 2>/dev/null); do
        if [ "$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')" = "1" ]; then
            # dossier courant du processus (lsof : -a = ET, -d cwd = dossier courant, -Fn = nom seulement)
            local dossier
            dossier=$(lsof -a -p "$pid" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p')
            if [ "$dossier" = "$RACINE" ]; then
                echo "$pid"
            fi
        fi
    done
}

# --- Arret, en plusieurs passes ---
# Passe 1 : SIGTERM (arret demande) ; le lanceur ne lance alors plus d'etape, son "trap" arrete caffeinate.
# Passes suivantes : on relit la liste (un parent a pu lancer un processus entre-temps) et on recommence.
# Derniere passe : SIGKILL (arret force) pour ce qui resterait.
total=0
for passe in 1 2 3 4 5; do
    cibles=$(lister_cibles | awk '!deja_vu[$0]++')  # sans doublons, ordre garde
    if [ -z "$cibles" ]; then
        break  # plus rien a arreter
    fi
    signal=TERM
    if [ "$passe" -eq 5 ]; then
        signal=KILL  # derniere chance
    fi
    for pid in $cibles; do
        # une ligne par processus vise : numero et debut de la ligne de commande
        echo "passe $passe, SIG$signal : $pid $(ps -o command= -p "$pid" 2>/dev/null | cut -c 1-110)"
        kill -s "$signal" "$pid" 2>/dev/null && total=$(( total + 1 ))
    done
    sleep 2  # laisse aux processus le temps de se terminer
done

# --- Bilan ---
restants=$(lister_cibles | awk '!deja_vu[$0]++')
echo "signaux envoyes : $total"
if [ -n "$restants" ]; then
    echo "ATTENTION, processus encore presents : $restants"
    exit 1
fi
echo "plus aucun processus du calcul de nuit"
