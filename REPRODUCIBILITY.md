# Reproducibility guide

Detailed map from every result in the paper to the script and stored data that produce it, the GPU pipelines, model versions and the format of the score files. For a quick check of the main result, see `reproduce_table1.py` in the [README](README.md). Section numbers S1 to S19 refer to the supplementary material at the end of the arXiv paper.

## Where each result in the paper comes from

| paper item | produced by | stored scores / report |
|---|---|---|
| Fig. 1 (decoder flips, example latent) | `figures/fig_teaser_option.py` | `scores/sd15_main_n300`, `scores/sd15_main_n300_v4` |
| Table 1 (decoder swap at 1% and 5% FPR) | `scripts/dod_v3.py`; B-Free from `scripts/dod_v4.py`; intervals in supplement S2, S3 and Guide | `scores/sd15_main_n300`, `scores/sd15_main_n300_v4` |
| Table 2 (AUROC and paired score change) | `scripts/dod_mac_pilot.py`, `scripts/dod_extras.py`, `scripts/dod_v3.py` | `scores/sd15_main_n300` |
| Fig. 2 (a: TPR per decoder, b: ReconScreen) | `figures/fig_story.py` | `scores/sd15_main_n300`, `scores/sd15_main_n300_v4` |
| Fig. 3 (AEROBLADE decoder x autoencoder) | `figures/fig_aeromatch.py` | `scores/sd15_main_n300`, `scores/sd15_main_n300_e2e5/e2_scores.csv` |
| Table 3 (decoder x generative history) | `scripts/dod_history.py` | `scores/sd15_main_n300/history_scores.csv` |
| Table 4 (SDXL) | `scripts/run_sdxl.sh`, `scripts/dod_sdxl.py` | `scores/sdxl_n300` |
| FLUX.1 check | `scripts/dod_mac_pilot.py`, `scripts/dod_extras.py` | `scores/flux1_n100` |
| Table 5 (JPEG) | `scripts/dod_jpeg.py`; Community-Forensics: `scripts/dod_v4.py`; B-Free: `scripts/dod_e2.py` (E5) | `scores/sd15_main_n300/jpeg_scores.csv`, `scores/sd15_main_n300_v4`, `scores/sd15_main_n300_e2e5` |
| Round-trip flag rates | `analysis/groupA.py`, `analysis/v33_roundtrip_ci.py` | `scores/groupA_roundtrip_flip.csv` |
| Table 6 (ReconScreen) | `analysis/diag_reanalysis.py`, `analysis/bfree_screen.py` | `scores/sd15_main_n300`, `scores/sd15_main_n300_v4/bfree_screen.csv` |

Supplement-only results:

| supplement item | produced by |
|---|---|
| S5 Consistency Decoder gates and seeds; S7 screen, all cells | `scripts/dod_v3.py`, `analysis/diag_reanalysis.py` |
| S10 score change vs. reconstruction quality | `scripts/dod_mac_pilot.py` (`per_image_quality.csv`) |
| S11 frequency and LPIPS-matched interventions (pilot) | `scripts/dod_bandswap.py`, `scripts/dod_bandswap2.py` |
| S12 probes trained on several decoders (no claim) | `scripts/dod_mitigation.py` (retrains probes on detector features; not stored as scores) |
| S13 pilot replication | `scripts/dod_mac_pilot.py`, `scripts/dod_extras.py`, `scripts/dod_history.py` on the pilot photographs |
| S14 add-on run (quality on all 300 photographs, Community-Forensics under JPEG, B-Free) | `scripts/run_v4.sh`, `scripts/dod_v4.py`; tables by `analysis/make_supp_v4.py` |
| S15 screen test on B-Free, Figs. S1 and S2 | `analysis/bfree_screen.py`, `figures/fig_gallery.py` |
| S16 flip rate, all-decoder TPR, autoencoder ensembles | `analysis/groupA.py` |
| S17 E2 and E5 | `scripts/run_e2.sh`, `scripts/dod_e2.py` |
| S17 E4/E10, round-trip controls, seed agreement | `scripts/run_e4.sh`, `scripts/dod_e4.py`, `analysis/e4_e10.py`, `analysis/e10.py`, `analysis/seed_agreement.py` |
| S17 SDXL, Fig. S3 | `scripts/dod_sdxl.py`, `figures/fig_sdxl.py` (`fig_sdxl_values.csv`) |
| Guide tables (5% FPR intervals, round-trip intervals) | `analysis/v33_roundtrip_ci.py` |
| Tables S1 to S13 | `analysis/build_supp.py`, `analysis/md2tex.py`. *Requires the original run folders* written by the GPU pipelines (their run reports); does not run on a fresh clone alone. |

## Entry points

GPU pipelines (single 48 GB GPU), run from `code/scripts/`. They regenerate all images from the stored seeds, which takes several GPU hours per run. Reference scores used by the reproduction gates are in `code/scripts/frozen/`.

1. `bash run_v3.sh smoke`, then `bash run_v3.sh full`: main SD1.5 run (`dod_mac_pilot.py`: data, latents, all decoders, UnivFD, NPR, probes; `dod_extras.py`: calibration set, AEROBLADE, FLUX.1 reconstructions; `dod_v3.py`: Consistency Decoder, Community-Forensics, matched-autoencoder AEROBLADE, gates). Needs only this repository plus internet access to download COCO and the models; it writes the run folder `runs/n300/`.
2. `python dod_history.py --run <run>` (Table 3), `python dod_jpeg.py --run <run>` (Table 5), `python dod_bandswap.py --run <run>`, `python dod_bandswap2.py --run <run>` (S11), `python dod_mitigation.py --run <run>` (S12). *Requires the run folder written by step 1* (decoded images and latents); not runnable on a fresh clone alone.
3. `bash run_v4.sh full`: add-on run on the same images (reproduction gate against `frozen/`, full reconstruction quality, Community-Forensics under JPEG, B-Free). Regenerates the images itself; writes `images_v4_full.tgz`.
4. `bash run_e2.sh full`: E2 (AEROBLADE with the Consistency Decoder's own encoder-decoder path) and E5 (B-Free under JPEG). Uses `images_v4_full.tgz` from step 3 if present; without it, it regenerates the images first (about 2.5 h).
5. `bash run_e4.sh full`: CLIP/DINOv2 features for E4/E10 and the round-trip controls. *Requires the run folder unpacked by step 4* (`runs/n300/png`) on the same machine.
6. `bash run_sdxl.sh pilot`, then `bash run_sdxl.sh full`: LatentLock on SDXL. Uses `frozen/`; the CLIP/DINO probes are scored only if the features from step 5 are present.

Setup: `pip install -r code/requirements_freeze.txt` (CUDA build of PyTorch; `requirements_freeze_sdxl.txt` for the SDXL run). Accept the FLUX.1-schnell licence on Hugging Face to run FLUX.1. NPR and UnivFD weights come from their official releases, as described in `dod_mac_pilot.py`.

CPU analyses on the stored scores (seconds to a minute), from the repository root:

```
python code/analysis/diag_reanalysis.py scores/sd15_main_n300 diag.md                     # Table 6, S7
python code/analysis/bfree_screen.py scores/sd15_main_n300 scores/sd15_main_n300_v4       # S15
python code/analysis/groupA.py scores/sd15_main_n300 scores/sd15_main_n300_v4 out/        # round trip, S16
python code/analysis/v33_roundtrip_ci.py scores/sd15_main_n300 scores/sd15_main_n300_v4   # Guide
python code/analysis/seed_agreement.py scores/sd15_main_n300 out.csv                      # S17
python code/analysis/e4_e10.py scores/sd15_main_n300_e4                                   # E4 (needs torch and the UnivFD head, see below)
python code/analysis/e10.py scores/sd15_main_n300_e4                                      # E10 (same)
python code/figures/fig_story.py scores/sd15_main_n300 out/ scores/sd15_main_n300_v4      # Fig. 2
python code/figures/fig_aeromatch.py scores/sd15_main_n300 out/ scores/sd15_main_n300_e2e5/e2_scores.csv   # Fig. 3
python code/figures/fig_sdxl.py scores/sdxl_n300 out/                                     # Fig. S3
```

The two E4/E10 scripts read the official UnivFD linear head (`univfd_fc_weights.pth`, not redistributed here); `dod_mac_pilot.py` downloads it from the UniversalFakeDetect release into its cache, and you can copy it into `scores/sd15_main_n300_e4/`.

`fig_teaser_option.py` (Fig. 1) and `fig_gallery.py` (Figs. S1, S2) also need the decoded PNGs, which `run_v4.sh` regenerates:
`python code/figures/fig_teaser_option.py scores/sd15_main_n300 scores/sd15_main_n300_v4 <png dir> out/ 6 80 16` (source 6, zoom box at 80,16).

## Environment and model versions

`model_revisions.txt`: Hugging Face revisions and SHA-256 of the downloaded weights for the SD1.5 run. `v3_versions.json`: package and model versions of the main run. `detector_status.json`: preprocessing per detector. B-Free's code commit and weight checksum: `scores/sd15_main_n300_v4/bfree_status.json`. The SDXL run (`stabilityai/stable-diffusion-xl-base-1.0`, `madebyollin/sdxl-vae-fp16-fix`, `madebyollin/taesdxl`) and the FLUX.1 run (`black-forest-labs/FLUX.1-schnell`, `madebyollin/taef1`) loaded the current Hub revision at run time without pinning it; the model ids are in the scripts.

AEROBLADE here is a single-autoencoder implementation (SD1.5 autoencoder, LPIPS-VGG layer 2) in `dod_extras.py`; the matched-autoencoder variants are in `dod_v3.py`. As stated in the paper, the official AEROBLADE default (minimum over the SD1, SD2 and Kandinsky 2.1 autoencoders) was not run.

## Score files

Long format: `group, decoder, idx, detector, score` (higher = more likely generated).

- `group`: `real` (untouched evaluation photograph), `real_enc` (VAE-encoded photograph decoded by `decoder`), `gen` (text-to-image latent), `i2i` (img2img latent), `calib` (independent calibration photograph), `jpgQ_*` (JPEG at quality Q), `h_s30/h_s50/h_s70` (img2img at strength 0.3/0.5/0.7), `flux_real` (photograph encoded with the FLUX.1 VAE, decoder `fluxvae` or `taef1`), `paper_calib` (SDXL run: the SD1.5 calibration set, secondary threshold).
- `decoder`: `orig`, `ftmse`, `taesd`, `cons` (Consistency Decoder, seed 3000+idx), `cons_s1/cons_s2` (seeds 4000+idx / 5000+idx), `taef1`, `fluxvae`; SDXL: `orig`, `fp16fix`, `taesdxl`.
- `detector`: `univfd`, `npr`, `probe_clip`, `probe_dino` (probe scores valid for idx >= N/2 only), `aeroblade` / `aero_orig`, `aero_ftmse`, `aero_taesd`, `aero_cons` (AEROBLADE with each reconstruction autoencoder), `aero_sdxl`, `commfor`, `bfree`.

Folders:

- `sd15_main_n300/`: main SD1.5 run (300 sources, 1,000 calibration photographs). `sources_manifest.json` (COCO id, caption, original size and crop box per source), `calibration_manifest.json` (COCO id and crop box per calibration photograph; no overlap with the sources), `per_image_quality.csv` (PSNR / LPIPS against the photograph and against the original decoder, high-frequency energy), `cons_grid.png` (generated latents 0-3 decoded with orig, ft-MSE, TAESD and the Consistency Decoder), `seed_agreement_posthoc.csv`.
- `sd15_main_n300_v4/`: add-on run on the same images: `quality_v4.csv`, `cf_scores_v4.csv`, `bfree_scores_v4.csv`, `gate.json` (reproduction gate), `bfree_status.json`, `bfree_screen.csv`.
- `sd15_main_n300_e2e5/`: E2 (`aero_cons`) and E5 (B-Free on JPEG copies).
- `sd15_main_n300_e4/`: feature index, CLIP/DINOv2 feature arrays (`feats_clip.npy`, `feats_dino.npy`), round-trip controls (`rtctrl_scores.csv`), E4/E10 results.
- `sdxl_n300/`: SDXL run, gates, COCO-id manifest of sources and calibration photographs.
- `flux1_n100/`: FLUX.1-schnell (100 sources). `flux_manifest.csv`: COCO ids of every photograph set, used for the overlap check in S14.
- `sd15_pilot_n100/`: disjoint pilot run (100 sources), including the frequency interventions (`bandswap*`) and `calib_npr_scores.txt` (NPR scores of the pilot's calibration photographs, used for the disjoint FLUX.1 NPR thresholds).
- `groupA_*.csv`: round-trip flag rates and autoencoder ensembles (S16).

The `report_*.md` files are the reports as written by the runs. Some of them call the reading rules that were fixed in the plan before the run "pre-registered"; these rules were not registered publicly (save times in supplement S17).
