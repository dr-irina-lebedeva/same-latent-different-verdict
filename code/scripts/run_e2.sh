#!/usr/bin/env bash
# E2: AEROBLADE with the Consistency Decoder's own autoencoder, and E5: B-Free under JPEG, on the paper's images.
#   bash run_e2.sh smoke   # ~5 min with images_v4_full.tgz uploaded (~15 min without): 4 latents, 20 calibration photos
#   bash run_e2.sh full    # ~45-60 min with images_v4_full.tgz uploaded; ~2.5 h without (regenerates the images first)
# Output: results_e2_<mode>.tgz (small; download this before stopping the pod)
set -euo pipefail
MODE=${1:?usage: bash run_e2.sh smoke|full}
if [ "$MODE" = smoke ]; then N=4; C=20; else N=300; C=1000; fi
OFF=100; LOG=e2_$MODE.log
echo "== E2 $MODE start $(date -u)" | tee "$LOG"
python -c "import torch,sys;sys.exit(0 if torch.__version__.startswith('2.8.0') else 1)" 2>/dev/null || \
  pip install -q "torch==2.8.0" "torchvision==0.23.0" --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -2 | tee -a "$LOG"
pip install -q "diffusers==0.42.1" "timm==1.0.31" "transformers==5.22.2" "accelerate==1.15.0" \
  "huggingface_hub==2.4.0" "lpips==0.1.4" "scikit-learn==1.9.1" scikit-image pyyaml tqdm pandas 2>&1 | tail -2 | tee -a "$LOG"
python -c "import torch,diffusers;print('torch',torch.__version__,'diffusers',diffusers.__version__,'gpu',torch.cuda.get_device_name(0))" | tee -a "$LOG"
TGZ=$(ls images_v4_full.tgz ../images_v4_full.tgz 2>/dev/null | head -1 || true)
if [ -n "$TGZ" ]; then
  R=runs/n300
  if [ ! -f $R/.unpacked_ok ]; then
    echo "checking $TGZ (expect 1540393575 bytes)" | tee -a "$LOG"
    gzip -t "$TGZ" || { echo "STOP: $TGZ is incomplete or corrupt. Delete it and runs/n300, then re-upload (or delete it to regenerate)." | tee -a "$LOG"; exit 1; }
    rm -rf $R; echo "unpacking $TGZ" | tee -a "$LOG"; tar xzf "$TGZ" && touch $R/.unpacked_ok
  fi
else
  # no saved images: regenerate them exactly as in the paper (same scripts, arguments and seeds as run_v4.sh)
  if [ "$MODE" = smoke ]; then R=runs/smoke; else R=runs/n300; fi
  python dod_mac_pilot.py --n $N --photo-offset $OFF --out $R --device cuda 2>&1 | tee -a "$LOG"
  python dod_extras.py --run $R --calib $C --stage calib --device cuda 2>&1 | tee -a "$LOG"
  python dod_v3.py --run $R --stage cons --device cuda 2>&1 | tee -a "$LOG"
fi
python dod_e2.py --run $R --n $N --calib $C --device cuda 2>&1 | tee -a "$LOG"
for f in e2_check.json e2_scores.csv e5_status.json e5_bfree_jpeg.csv report_e2.md e2_summary.json; do
  [ -s "$R/$f" ] || { echo "MISSING $R/$f" | tee -a "$LOG"; exit 1; }
done
pip freeze > $R/pip_freeze_e2.txt
tar czf results_e2_$MODE.tgz "$LOG" $R/e2_* $R/e5_status.json $R/e5_bfree_jpeg.csv $R/e5_demo_out.csv $R/report_e2.md $R/pip_freeze_e2.txt dod_e2.py run_e2.sh ../protocols/PLAN_E2_E5_roundtrip.md
ls -lh results_e2_$MODE.tgz | tee -a "$LOG"
grep -E "check:|bfjpeg gate|Planned reading" "$LOG" | tail -4
echo "== E2 $MODE DONE $(date -u). Download results_e2_$MODE.tgz BEFORE stopping the pod." | tee -a "$LOG"
