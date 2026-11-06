#!/usr/bin/env bash
# Arrete proprement le calcul de nuit (scripts/nuit_04h.sh) : tous les processus Python de l'optimisation
# des reseaux, et seulement eux (les noyaux Jupyter et les autres programmes Python ne sont pas touches).
#
# Lancement, depuis la racine du depot, dans Git Bash :
#     bash scripts/arreter_04h.sh
#
# Les essais Optuna termines restent dans la base. Les essais en cours restent marques "en cours" : l'etape
# "nettoyer", au debut du prochain `bash scripts/nuit_04h.sh`, les marque FAIL et la nuit suivante reprend.

powershell -NoProfile -Command '
$cibles = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -like "*optimiser_reseaux.py*" -or $_.CommandLine -like "*SetThreadExecutionState*" -or $_.CommandLine -like "*nuit_04h.sh*"
}
# processus de calcul paralleles (ProcessPoolExecutor) : leur ligne de commande ne cite pas le script, seulement
# le numero de leur parent. Constate le 2026-11-06 : sans ca, 15 processus orphelins gardaient ~13 Go de memoire.
$parents = @($cibles | ForEach-Object { "parent_pid=" + $_.ProcessId })
$enfants = Get-CimInstance Win32_Process | Where-Object {
    $ligne = $_.CommandLine
    $_.CommandLine -like "*spawn_main*" -and ($parents | Where-Object { $ligne -like "*$_[,)]*" })
}
$cibles = @($enfants) + @($cibles)  # les enfants d abord
foreach ($p in $cibles) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }
Write-Output ("processus arretes : " + @($cibles).Count)
'
