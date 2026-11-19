# Analysis of the original runs

Two scripts check the numbers in the original report (dated December 4, 2025), poster, and
slides, none of which are in this repository, against the result files in
[`../results/original/`](../results/original/), and test what those results mean. Both run on a
CPU.

## `recompute_original.py`

Needs only the result files (pandas and NumPy, a few seconds):

```bash
python analysis/recompute_original.py [--report PROJECT_REPORT.txt]
```

- `out/report_table_check.csv`: every accuracy cell of report Tables 1 to 5 beside the file it
  should come from (the third column of Tables 3 and 4 isn't checked). Tables 1 and 2 match in 86 of 98 and 76 of 91 cells; the rows at 15%, 25%, 70%,
  and 90% hold most of the differences. Table 5's N = 50 and N = 500 columns differ at 20%,
  30%, and 50%, and its N = 1,000 column can't be checked: it has no result file. The script rewrites this file only
  when it can read the report's text; without it, it writes the summary alone.
- `out/summary_original.csv`: per run and model, the area under the accuracy curve, the
  occlusion at which accuracy falls below 50%, and the pretrained models' mean accuracy from
  10% to 100% occlusion against chance (1/N).

## `checks.py`

Needs the original runs' saved CelebA-HQ samples (not in the repository; CelebA-HQ may not be
redistributed). `--check custom` also needs the project's saved model as `novel_model/model.pth` (attached to the
repository's [release](https://github.com/ryanjosephkamp/deep-learning-studies/releases)) and the
background-removed run's fine-tuned model as
`experiments/exper__NO_BG__size_100__20251203_012434/novel_model/model_nobg_finetuned.pth`, both
under `--weights` (the fine-tuned model is not distributed), and
`--check detector` needs SFace's ONNX file from the OpenCV Zoo:

```bash
python analysis/checks.py --experiments EXPERIMENTS_DIR --check pixels
python analysis/checks.py --experiments EXPERIMENTS_DIR --check custom --weights PROJECT_DIR
python analysis/checks.py --experiments EXPERIMENTS_DIR --check detector --sface face_recognition_sface_2021dec.onnx
```

- `out/check_pixel_baseline.csv`: nearest neighbor on raw pixels, no learning, under the
  original protocol. It matches 100% of images at 10% occlusion for N = 50, 100, and 200 (99.2%
  at N = 500), so the task the runs measured is easy.
- `out/check_custom_rerun.csv`: the saved custom model rerun on the saved samples of the N = 50,
  100, and 200 runs, and the fine-tuned model on the background-removed N = 100 run (N = 500 was
  not rerun). It reproduces each result file within three images at every level (at most 4
  points, at N = 50). The originals ran on an A100, where PyTorch allows TF32 for convolutions by
  default, which can flip near-ties; whether it did here wasn't tested. The `NOBG100_master` rows run the model
  before any fine-tuning on the background-removed images.
- `out/check_detector_sface.csv` and `out/check_detector_face_coverage.csv`: an approximate
  re-creation, for SFace only, of why the pretrained models collapse on the 100-image sample (the
  original runs saved no detections). DeepFace's default detector finds a face in 95 of the 100 clean
  images, in 3 once the square covers 5% of the frame, and in none at 10%; DeepFace then embeds the
  whole frame against a gallery of (mostly) face crops. The check re-creates that path for SFace
  (without eye alignment) and compares it with the same face crop for probe and gallery and with
  whole frames on both sides. The coverage file gives the share of the face box detected on the
  clean image that each square covers. The other five models weren't rerun.
