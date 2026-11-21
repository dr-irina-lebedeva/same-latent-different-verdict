"""M3 mitigation test: does training a detector on several decoders remove the decoder swing?

Linear probes on CLIP ViT-L/14 and DINOv2 features (same features as the main run).
Train split: sources k < N/2. Test split: k >= N/2. Calibration: 1,000 independent real photos
(features are extracted once and cached; ~5 min on an M4).

Training sets (real = untouched photos, fake = generated latents decoded by the listed decoders):
  orig only          the setting of the main paper (one decoder at training time)
  orig + ftmse       leave TAESD out   -> test on TAESD = an unseen decoder
  orig + taesd       leave ft-MSE out  -> test on ft-MSE = an unseen decoder
  all three          upper bound (every test decoder seen in training)

    python dod_mitigation.py --run runs/n100 2>&1 | tee mitigation.log
Output: <run>/report_mitigation.md
"""
import argparse, json, warnings
from pathlib import Path

import numpy as np
from PIL import Image

import dod_mac_pilot as M

warnings.filterwarnings("ignore", message=".*encountered in matmul")
log = M.log
DECS = ["orig", "ftmse", "taesd"]
TRAIN_SETS = {"orig only": ["orig"], "orig + ftmse (TAESD unseen)": ["orig", "ftmse"],
              "orig + taesd (ft-MSE unseen)": ["orig", "taesd"], "all three": DECS}


def calib_feats(a, name, cls):
    fp = a.run / f"feat_calib_{name}.npy"
    cm = a.run / "data" / "calib_meta.json"
    if not cm.exists():
        return None
    nc = len(json.loads(cm.read_text()))
    if fp.exists() and len(np.load(fp)) == nc:
        return np.load(fp).astype(np.float64)
    fx = cls(a.device, a.fake_models)
    F = np.stack([fx(Image.open(a.run / "data" / "calib" / f"{k:04d}.png")) for k in range(nc)])
    del fx; M.free(a.device)
    np.save(fp, F); log(f"mitigation: calib features {name} done")
    return F.astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--fake-models", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    a.device = M.pick_device(a.device)
    a.out = a.run
    meta = json.loads((a.run / "data" / "meta.json").read_text())
    a.n = a.n_eff = len(meta)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    main_imgs = M.all_images(a)
    half = a.n_eff // 2
    L = []; P = lambda *s: L.append(" ".join(str(x) for x in s))
    P("# Mitigation test (M3): train on several decoders\n")
    P(f"Linear probes; train sources k < {half}, test sources k >= {half} (n = {a.n_eff - half}). "
      "AUROC: untouched real vs generated (test split). TPR at a 5% FPR threshold set on independent "
      "real photos when available, else on test reals. 'swing' = max - min TPR across decoders.\n")
    for name, cls in [("clip", M.ClipFeat), ("dino", M.DinoFeat)]:
        F = np.load(a.run / f"feat_{name}.npy").astype(np.float64)
        assert len(F) == len(main_imgs)
        idx = {(g, d, i): j for j, (g, d, i, _) in enumerate(main_imgs)}
        Fc = calib_feats(a, name, cls)
        P(f"## {name.upper()} probe\n")
        P("| trained on | " + " | ".join(f"AUROC {d}" for d in DECS) + " | "
          + " | ".join(f"TPR {d}" for d in DECS) + " | swing | FPR real (test) |")
        P("|---|" + "---|" * (2 * len(DECS) + 2))
        for tname, tdecs in TRAIN_SETS.items():
            tr = [idx[("real", "none", i)] for i in range(half)]
            # balance: same number of fakes per class as reals, spread over the training decoders
            for i in range(half):
                tr.append(idx[("gen", tdecs[i % len(tdecs)], i)])
            y = [0] * half + [1] * half
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=5000)).fit(F[tr], y)
            sc = lambda g, d: clf.decision_function(F[[idx[(g, d, i)] for i in range(half, a.n_eff)]])
            real = sc("real", "none")
            thr = np.quantile(clf.decision_function(Fc), 0.95) if Fc is not None else np.quantile(real, 0.95)
            au = [M.auroc(real, sc("gen", d)) for d in DECS]
            tp = [float(np.mean(sc("gen", d) > thr)) for d in DECS]
            P(f"| {tname} | " + " | ".join(f"{x:.3f}" for x in au) + " | "
              + " | ".join(f"{x:.2f}" for x in tp) + f" | {max(tp) - min(tp):.2f} | {np.mean(real > thr):.2f} |")
        P("")
    P("How to read: if 'swing' shrinks from 'orig only' to the leave-one-out rows, training on more than "
      "one decoder makes the detector robust even to a decoder it never saw. Probes are small (linear, "
      f"{half} sources per class); treat this as a proof of concept, not a new detector.\n")
    (a.run / "report_mitigation.md").write_text("\n".join(L))
    log(f"report: wrote {a.run / 'report_mitigation.md'}")


if __name__ == "__main__":
    main()
