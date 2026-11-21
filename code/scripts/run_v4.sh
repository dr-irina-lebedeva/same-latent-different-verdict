#!/usr/bin/env bash
# v4 add-on run: regenerates the frozen v3 images with identical settings, then the new stages.
#   bash run_v4.sh smoke   # ~10-15 min: the first 4 main-run photos (offset 100), checks every model + the gate
#   bash run_v4.sh full    # ~2.5-3 h: all 300 photos + 1,000 calibration photos
# Output: results_v4_<mode>.tgz (small, download this) and images_v4_<mode>.tgz (lossless PNGs, ~2-3 GB, optional)
set -euo pipefail
MODE=${1:?usage: bash run_v4.sh smoke|full}
if [ "$MODE" = smoke ]; then R=runs/smoke; N=4; C=20; else R=runs/n300; N=300; C=1000; fi
OFF=100                       # identical to the frozen main run (photos 100-399 of the fixed order)
LOG=v4_$MODE.log
echo "== $MODE start $(date -u)" | tee "$LOG"
# same torch as v3 (2.8.0+cu128); installed only if the template has a different one
python -c "import torch,sys;sys.exit(0 if torch.__version__.startswith('2.8.0') else 1)" 2>/dev/null || \
  pip install -q "torch==2.8.0" "torchvision==0.23.0" --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -2 | tee -a "$LOG"
pip install -q "diffusers==0.42.1" "timm==1.0.31" "transformers==5.22.2" "accelerate==1.15.0" \
  "huggingface_hub==2.4.0" "lpips==0.1.4" "scikit-learn==1.9.1" scikit-image pyyaml tqdm pandas 2>&1 | tail -2 | tee -a "$LOG"
python -c "import torch,diffusers,timm;print('torch',torch.__version__,'diffusers',diffusers.__version__,'timm',timm.__version__,'gpu',torch.cuda.get_device_name(0))" | tee -a "$LOG"
# 1) frozen pipeline, unchanged scripts and arguments (data, latents, decoders, UnivFD/NPR/probes)
python dod_mac_pilot.py --n $N --photo-offset $OFF --out $R --device cuda 2>&1 | tee -a "$LOG"
# 2) independent calibration photos (same selection code and seed as v3)
python dod_extras.py --run $R --calib $C --stage calib --device cuda 2>&1 | tee -a "$LOG"
# 3) Consistency Decoder on gen + real_enc latents, seed 3000+i (same code as v3)
python dod_v3.py --run $R --stage cons --device cuda 2>&1 | tee -a "$LOG"
# exact model revisions now on disk (compared with frozen/model_revisions.txt by the gate)
{ for d in "${HF_HOME:-$HOME/.cache/huggingface}"/hub/models--*; do
    [ -f "$d/refs/main" ] && echo "$(basename "$d" | sed 's/^models--//; s/--/\//') $(cat "$d/refs/main")"; done
  find cache -type f \( -name "*.pth" -o -name "*.pt" \) -exec sha256sum {} \; 2>/dev/null; } > $R/model_revisions_v4.txt || true
# 4) new stages: gate, quality, Community-Forensics x JPEG, B-Free (optional, self-skipping), report
python dod_v4.py --run $R --device cuda 2>&1 | tee -a "$LOG"
for f in gate.json quality_v4.csv cf_scores_v4.csv report_v4.md; do
  [ -s "$R/$f" ] || { echo "MISSING $R/$f" | tee -a "$LOG"; exit 1; }
done
pip freeze > $R/pip_freeze_v4.txt
tar czf results_v4_$MODE.tgz "$LOG" $R/*.md $R/*.csv $R/*.json $R/*.png $R/*.txt *.py run_v4.sh
tar czf images_v4_$MODE.tgz $R/png $R/data
ls -lh results_v4_$MODE.tgz images_v4_$MODE.tgz | tee -a "$LOG"
grep -A12 '"verdict"' $R/gate.json | tee -a "$LOG" || true
echo "== $MODE DONE $(date -u). Download results_v4_$MODE.tgz (and images_v4_$MODE.tgz if you want the images) BEFORE stopping the pod." | tee -a "$LOG"
