#!/usr/bin/env bash
# Enchaine toutes les etapes de l'optimisation des reseaux (notebook 04h) pendant la nuit, sur macOS :
# FD003, puis FD004. Meme logique, memes etapes et memes nombres d'essais que scripts/nuit_04h.sh (Matisse,
# ecrit pour Windows / Git Bash), adaptes a un Mac.
#
# Lancement, depuis la racine du depot, dans le Terminal :
#     bash scripts/nuit_04h_fd003_fd004_mac.sh
#     AERO_PROCESSUS=8 bash scripts/nuit_04h_fd003_fd004_mac.sh         # moins de processus : moins de chaleur, plus lent
#     AERO_SOUS_JEUX=FD004 bash scripts/nuit_04h_fd003_fd004_mac.sh     # un seul sous-jeu
#     AERO_ESSAI_RAPIDE=1 AERO_PROCESSUS=4 bash scripts/nuit_04h_fd003_fd004_mac.sh   # verification en quelques minutes
# Suivi : sorties/resultats/04h_nuit_fd003_fd004.log (une ligne par etape, plus les lignes de chaque etape).
#     tail -f sorties/resultats/04h_nuit_fd003_fd004.log
# Arret propre : bash scripts/arreter_04h_mac.sh (les essais Optuna finis restent dans la base).
#
# Base Optuna : sorties/resultats/04h_optuna.db, la meme que celle de Matisse (etudes FD001 et FD002, copie de
# docs/resultats/04h/04h_optuna.db). Les etudes 04h_fd003_lstm, 04h_fd003_cnn1d, 04h_fd004_lstm et
# 04h_fd004_cnn1d s'y ajoutent ; les etudes deja presentes ne sont pas modifiees (sauf l'etape "nettoyer",
# qui marque FAIL les essais restes "en cours" dans TOUTES les etudes de la base, voir plus bas).
#
# Reprise : relancer la meme commande. Une etude Optuna deja complete ne relance aucun essai, et une etape
# dont le fichier de resultats existe deja est sautee : le script reprend la ou il s'etait arrete (a l'essai
# Optuna pres, l'essai interrompu est refait).
#
# Ordre, pour chaque sous-jeu (chaque etape relit les resultats de la precedente, voir scripts/optimiser_reseaux.py) :
#   1. optuna     : LSTM (100 essais utiles, 3/4 des processus) et CNN 1D (80 essais utiles, le reste) en
#                   parallele ; quand le CNN a fini, autant de travailleurs LSTM de plus prennent ses coeurs ;
#   2. confirmer  : meilleur essai de chaque reseau contre la reference du 03b, 3 decoupages neufs ;
#   3. ablation   : familles d'entrees (pente, sans capteurs) sur le LSTM retenu ;
#   4. ensemble   : moyennes de 5 graines LSTM, puis 5 LSTM + 5 CNN ;
#   5. final      : modeles finaux et test officiel du sous-jeu, lu une seule fois.
#
# Mise en veille : `caffeinate -i -w <numero de ce script>` (outil livre avec macOS) empeche la mise en veille
# "pour inactivite" tant que ce script tourne. Aucun reglage de macOS n'est modifie : la demande disparait a la
# fin du script. Fermer le capot d'un portable met quand meme en veille (sauf ecran externe et secteur).

set -euo pipefail  # arret a la premiere erreur, variable inconnue = erreur, erreur dans un tube = erreur

# --- Emplacements ---
cd "$(dirname "$0")/.."  # racine du depot (guillemets : le chemin contient des espaces)
PYTHON=./.venv/bin/python  # Python du venv (macOS : bin/python, et non Scripts/python.exe comme sous Windows)
RESULTATS=sorties/resultats  # dossier des resultats (non versionne)
JOURNAL=$RESULTATS/04h_nuit_fd003_fd004.log  # journal de la nuit, distinct de celui de Matisse (04h_nuit.log)
SCRIPT=scripts/optimiser_reseaux.py  # le script qui fait le calcul, etape par etape

# --- Nombres d'essais : les memes que Matisse pour FD001 et FD002 ---
ESSAIS_LSTM=100  # essais utiles (termines ou elagues) de l'etude LSTM
ESSAIS_CNN=80    # essais utiles de l'etude CNN 1D

# --- Sous-jeux a traiter : FD003 puis FD004 par defaut ---
SOUS_JEUX=${AERO_SOUS_JEUX:-"FD003 FD004"}

# --- Repartition des processus (1 coeur chacun) ---
PROCESSUS=${AERO_PROCESSUS:-12}  # 12 par defaut, comme Matisse
PROCESSUS_CNN=$(( PROCESSUS / 4 ))  # un quart pour le CNN : 12 -> 3
if [ "$PROCESSUS_CNN" -lt 1 ]; then
    PROCESSUS_CNN=1  # au moins un travailleur CNN
fi
PROCESSUS_LSTM=$(( PROCESSUS - PROCESSUS_CNN ))  # le reste pour le LSTM : 12 -> 9
PROCESSUS_FINAL=$(( PROCESSUS < 10 ? PROCESSUS : 10 ))  # 10 modeles finaux au plus (5 LSTM + 5 CNN)

# --- Mode essai rapide (verification du script) ---
if [ "${AERO_ESSAI_RAPIDE:-0}" = "1" ]; then
    # 2 epoques par reseau, tres peu d'essais ; tout est ecrit dans sorties/essai_rapide (base Optuna
    # comprise), jamais dans les vrais resultats (scripts/optimiser_reseaux.py fait le meme choix de dossier)
    RESULTATS=sorties/essai_rapide
    JOURNAL=$RESULTATS/04h_nuit_fd003_fd004.log
    ESSAIS_LSTM=3
    ESSAIS_CNN=2
fi
mkdir -p "$RESULTATS"  # cree le dossier s'il n'existe pas encore

# --- 1 fil de calcul par processus ---
# scripts/optimiser_reseaux.py fixe deja OMP_NUM_THREADS, MKL_NUM_THREADS et OPENBLAS_NUM_THREADS a 1 avant
# d'importer numpy et torch ; on exporte ici les memes valeurs (aucune contradiction possible), pour que tout
# processus lance par ce script parte deja bride. En plus, specifique au Mac : numpy y est construit sur la
# bibliotheque Accelerate d'Apple (numpy.show_config() : "blas: accelerate"), dont le nombre de fils se regle
# par VECLIB_MAXIMUM_THREADS, que le script Python ne fixe pas.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# --- Verification des sous-jeux demandes ---
for sous_jeu in $SOUS_JEUX; do
    case "$sous_jeu" in
        FD001|FD002|FD003|FD004) ;;  # sous-jeu connu : on continue
        *)
            echo "Sous-jeu inconnu dans AERO_SOUS_JEUX : $sous_jeu (choix : FD001 FD002 FD003 FD004)" >&2
            exit 1
            ;;
    esac
done

# --- Anti-veille : caffeinate surveille ce script ($$) et s'arrete tout seul quand il se termine ---
caffeinate -i -w $$ &
PID_EVEIL=$!
# ceinture et bretelles : on l'arrete aussi explicitement a la sortie, meme en cas d'erreur
trap 'kill $PID_EVEIL 2>/dev/null || true' EXIT

horodatage() {
    # date et heure, pour le journal
    date '+%Y-%m-%d %H:%M:%S'
}

prefixe_fichiers() {
    # debut des noms de fichiers et d'etudes d'un sous-jeu, meme regle que prefixe() de
    # scripts/optimiser_reseaux.py : "04h_" pour FD001 (noms d'origine), "04h_fd003_" pour FD003, etc.
    local sous_jeu=$1
    if [ "$sous_jeu" = "FD001" ]; then
        echo "04h_"
    else
        # tr : passage en minuscules (FD003 -> fd003)
        echo "04h_$(echo "$sous_jeu" | tr '[:upper:]' '[:lower:]')_"
    fi
}

etape() {
    # affiche l'etape dans le journal, puis la lance en ajoutant sa sortie au journal
    echo "=== $(horodatage) : $* ===" | tee -a "$JOURNAL"
    # "|| true" : grep renvoie une erreur quand il n'a aucune ligne a garder, ce qui ne doit pas tout arreter ;
    # une erreur de Python, elle, arrete bien tout le script (set -o pipefail)
    "$PYTHON" "$SCRIPT" "$@" 2>&1 | { grep -v -i "warning" || true; } | tee -a "$JOURNAL"
}

etape_si_absente() {
    # $1 : fichier de resultats de l'etape ; la suite : l'etape. Sautee si le fichier existe deja.
    local fichier=$RESULTATS/$1
    shift  # retire $1 : il reste l'etape et ses options
    if [ -f "$fichier" ]; then
        echo "=== $(horodatage) : deja faite, sautee : $* ($fichier existe) ===" | tee -a "$JOURNAL"
    else
        etape "$@"
    fi
}

optuna_sous_jeu() {
    # $1 : sous-jeu ; $2 : debut des noms de fichiers ("04h_fd003_" pour FD003, "04h_fd004_" pour FD004)
    local sous_jeu=$1
    local prefixe=$2
    echo "=== $(horodatage) : optuna lstm + cnn1d $sous_jeu ===" | tee -a "$JOURNAL"

    # les deux etudes en parallele, en arriere-plan ; chacune lance ses propres travailleurs
    "$PYTHON" "$SCRIPT" optuna --sous-jeu "$sous_jeu" --reseau lstm --essais $ESSAIS_LSTM --processus $PROCESSUS_LSTM >> "$JOURNAL" 2>&1 &
    local pid_lstm=$!
    "$PYTHON" "$SCRIPT" optuna --sous-jeu "$sous_jeu" --reseau cnn1d --essais $ESSAIS_CNN --processus $PROCESSUS_CNN >> "$JOURNAL" 2>&1 &
    local pid_cnn=$!
    wait $pid_cnn  # on attend la fin du CNN (le plus court)

    # le CNN a fini : ses coeurs passent au LSTM (un travailleur ne fait rien si l'etude est deja finie)
    local pids_renforts=""
    local numero
    for numero in $(seq $PROCESSUS_LSTM $(( PROCESSUS - 1 ))); do
        # numeros a la suite de ceux des travailleurs LSTM deja lances (graine d'echantillonneur differente)
        "$PYTHON" "$SCRIPT" travailleur --sous-jeu "$sous_jeu" --reseau lstm --essais $ESSAIS_LSTM --numero $numero \
            > "$RESULTATS/${prefixe}journal_lstm_$numero.log" 2>&1 &
        pids_renforts="$pids_renforts $!"
    done
    wait $pid_lstm  # fin de l'etude LSTM principale
    local pid
    for pid in $pids_renforts; do
        wait $pid  # fin de chaque renfort
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

# --- Deroulement ---
DEBUT=$(date +%s)  # pour la duree totale, en fin de journal
echo "=== $(horodatage) : debut ($SOUS_JEUX, $PROCESSUS processus : $PROCESSUS_LSTM LSTM + $PROCESSUS_CNN CNN, resultats dans $RESULTATS) ===" | tee -a "$JOURNAL"

etape nettoyer  # essais Optuna interrompus par un arret precedent -> FAIL (les essais finis sont gardes)

for sous_jeu in $SOUS_JEUX; do
    etapes_sous_jeu "$sous_jeu" "$(prefixe_fichiers "$sous_jeu")"
done

DUREE=$(( $(date +%s) - DEBUT ))  # secondes ecoulees
echo "=== $(horodatage) : TOUT EST TERMINE ($SOUS_JEUX) en $(( DUREE / 3600 )) h $(( DUREE % 3600 / 60 )) min $(( DUREE % 60 )) s ===" | tee -a "$JOURNAL"
