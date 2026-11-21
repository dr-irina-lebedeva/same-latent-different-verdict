#!/usr/bin/env bash
# E3: LatentLock on SDXL. Run on the same pod AFTER run_e2.sh full and run_e4.sh (probes need their features).
#   bash run_sdxl.sh pilot   # 30 sources + 200 fresh calibration photos, ~25-35 min incl. the SDXL download
#   bash run_sdxl.sh full    # only if the pilot gate passed: 300 sources + 1,000 fresh calibration, ~2-2.5 h
# Output: results_sdxl_<mode>.tgz (scores, checks, report, 12 example images; download before stopping the pod)
set -euo pipefail
MODE=${1:?usage: bash run_sdxl.sh pilot|full}
if [ "$MODE" = pilot ]; then N=30; C=200; else N=300; C=1000; fi
O=runs/sdxl_$MODE; LOG=sdxl_$MODE.log
echo "== SDXL $MODE start $(date -u)" | tee "$LOG"
python -c "import torch,diffusers;print('torch',torch.__version__,'diffusers',diffusers.__version__,'gpu',torch.cuda.get_device_name(0))" | tee -a "$LOG"
[ -s runs/n300/feats_clip.npy ] || echo "NOTE: runs/n300/feats_clip.npy missing (run_e4.sh output not found): the CLIP/DINO probes will be skipped" | tee -a "$LOG"
if [ "$MODE" = full ]; then
  python -c "import json,sys;g=json.load(open('runs/sdxl_pilot/gate.json'));sys.exit(0 if g['passed'] else 1)" 2>/dev/null || \
    { echo "STOP: run the pilot first; its gate must pass" | tee -a "$LOG"; exit 1; }
fi
python dod_sdxl.py --out $O --sd15 runs/n300 --n $N --calib $C --device cuda 2>&1 | tee -a "$LOG"
for f in sdxl_scores.csv decode_check.json gate.json report_sdxl.md sdxl_summary.json data/meta.json; do
  [ -s "$O/$f" ] || { echo "MISSING $O/$f" | tee -a "$LOG"; exit 1; }
done
pip freeze > $O/pip_freeze_sdxl.txt
EX=$(for d in orig fp16fix taesdxl; do for k in 0000 0001 0002 0003; do echo $O/png1024/gen/$d/$k.png; done; done)
tar czf results_sdxl_$MODE.tgz "$LOG" $O/sdxl_scores.csv $O/*.json $O/data/meta.json $O/report_sdxl.md $O/pip_freeze_sdxl.txt $EX dod_sdxl.py run_sdxl.sh ../protocols/PLAN_SDXL.md
ls -lh results_sdxl_$MODE.tgz | tee -a "$LOG"
grep -E "gate:|Planned reading" "$LOG" | tail -3
echo "== SDXL $MODE DONE $(date -u). Download results_sdxl_$MODE.tgz BEFORE stopping the pod." | tee -a "$LOG"
