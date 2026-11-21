#!/usr/bin/env bash
# Add-on, run AFTER run_e2.sh full on the same pod (packages and images are already there). ~15 min.
#   bash run_e4.sh
# Output: results_e4.tgz (~25 MB): features for E4/E10, round-trip control scores, report_e4.md
set -euo pipefail
R=runs/n300; LOG=e4.log
[ -d $R/png/gen/cons ] || { echo "STOP: $R not found. Run run_e2.sh full first (it unpacks images_v4_full.tgz)."; exit 1; }
echo "== E4 add-on start $(date -u)" | tee "$LOG"
python dod_e4.py --run $R --device cuda 2>&1 | tee -a "$LOG"
for f in feats_clip.npy feats_dino.npy feats_index.csv e4_feats_check.json rtctrl_scores.csv report_e4.md; do
  [ -s "$R/$f" ] || { echo "MISSING $R/$f" | tee -a "$LOG"; exit 1; }
done
tar czf results_e4.tgz "$LOG" $R/feats_* $R/e4_*.json $R/rtctrl_scores.csv $R/rtctrl_feats_*.npy $R/report_e4.md $R/univfd_fc_weights.pth dod_e4.py run_e4.sh
ls -lh results_e4.tgz | tee -a "$LOG"
echo "== E4 add-on DONE $(date -u). Download results_e4.tgz (and results_e2_full.tgz) BEFORE stopping the pod." | tee -a "$LOG"
