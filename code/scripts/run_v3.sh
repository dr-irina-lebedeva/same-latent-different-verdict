#!/usr/bin/env bash
# v3 run on a cloud GPU. Stops at the first error (set -euo pipefail) and keeps a full log.
#   bash run_v3.sh smoke     # ~10 min: n=4, checks that every model loads; look at runs/smoke/cons_grid.png
#   bash run_v3.sh full      # ~3 h: n=300 (photos 100-399), then packs results + images
set -euo pipefail
MODE=${1:?usage: bash run_v3.sh smoke|full}
if [ "$MODE" = smoke ]; then R=runs/smoke; N=4; OFF=0; C=20; else R=runs/n300; N=300; OFF=100; C=1000; fi
LOG=v3_$MODE.log
echo "== $MODE start $(date -u)" | tee "$LOG"
python dod_mac_pilot.py --n $N --photo-offset $OFF --out $R --device cuda 2>&1 | tee -a "$LOG"
for s in calib aeroblade score report; do
  python dod_extras.py --run $R --calib $C --stage $s --device cuda 2>&1 | tee -a "$LOG"
done
python dod_v3.py --run $R --device cuda 2>&1 | tee -a "$LOG"
# checks: the files that must exist
for f in report.md report_extras.md report_v3.md cons_check.json cons_grid.png v3_new_scores.csv v3_seed_scores.csv; do
  [ -s "$R/$f" ] || { echo "MISSING $R/$f" | tee -a "$LOG"; exit 1; }
done
grep -q "commfor: loaded" "$LOG" && echo "OK commfor loaded" | tee -a "$LOG" || echo "WARNING: commfor did not load" | tee -a "$LOG"
grep "sanity PSNR cons" "$LOG" | tail -1 || true
pip freeze > $R/pip_freeze.txt
# exact model revisions (HF commit hashes) and checksums of the local detector checkpoints
{ for d in "${HF_HOME:-$HOME/.cache/huggingface}"/hub/models--*; do
    [ -f "$d/refs/main" ] && echo "$(basename "$d" | sed 's/^models--//; s/--/\//') $(cat "$d/refs/main")"; done
  find cache -type f \( -name "*.pth" -o -name "*.pt" -o -name "*.bin" -o -name "*.safetensors" \) -exec sha256sum {} \; 2>/dev/null
  echo "command: $0 $MODE  N=$N offset=$OFF calib=$C"; } > $R/model_revisions.txt || true
tar czf results_v3_$MODE.tgz "$LOG" $R/*.md $R/*.csv $R/*.json $R/*.png $R/pip_freeze.txt $R/model_revisions.txt *.py run_v3.sh
tar czf images_v3_$MODE.tgz $R/png $R/data $R/latents
ls -lh results_v3_$MODE.tgz images_v3_$MODE.tgz
echo "== $MODE DONE $(date -u)" | tee -a "$LOG"
