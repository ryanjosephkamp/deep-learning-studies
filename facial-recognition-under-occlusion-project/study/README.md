# Face identification under occlusion

`run_study.py` asks the identification question the original project did not: given a different
photo of the same person, with part of the face covered, does a model still pick that person out
of 1,000 people held out from the newly trained networks? It also tests the project's two ideas, the spatial attention
layer and training with occluded examples, when every model sees the same aligned face crop.

- Split: CelebA-HQ identities with at least two images; 1,000 test identities drawn with a fixed
  seed (each keeps its lowest-numbered image that survives alignment as the gallery image, the
  rest are probes; an identity left with fewer than two aligned images leaves the test set); every
  other identity with at least two images trains. No identity is in both.
- Alignment: YuNet finds the face on the clean 256 × 256 image and five landmarks map it onto the
  112 × 112 ArcFace template. Occluders are drawn on the aligned crop, so they never change where
  the face is found.
- Conditions: centered black squares covering 0 to 60% of the crop, an eye band, a lower-face
  band, an off-center square, and gray and noise fills; plus the original project's protocol
  (each gallery image occluded and matched back to itself) for comparison.
- Models: SFace and ArcFace ResNet-50 (pretrained), raw pixels, the original project's saved
  model on full frames, and the project's architecture trained on the training identities with
  attention on or off and occlusion augmentation on or off (CosFace loss), plus a batch-hard
  triplet loss; three seeds each.
- `analyze.py` resamples test identities for every interval, and the training seeds too for the
  trained configurations, and tests
  three primary contrasts, with a Holm correction.

The aggregate results of the November 11, 2026 run are in
[`../results/study/`](../results/study/).

## Data

CelebA and CelebA-HQ are for non-commercial research only and may not be redistributed, so no
images or identity labels are in this repository. Kaggle labels the copy below GPL 2; that label
doesn't change CelebA's terms. CelebA-Dialog's images and identity labels are also for
non-commercial research only. You need:

| File | Where | Check |
|---|---|---|
| CelebA-HQ at 256 × 256, `00000.jpg` to `29999.jpg` | Kaggle, `badasstechie/celebahq-resized-256x256` | 30,000 files, 296,389,711 bytes; image set fingerprint `fac6e0001b10b927dc88f91b0b0d4de0158d6536cc97f10216d921977a926418` (printed by `match_images.py`) |
| `identity_hq.txt`, one `<index>.jpg <identity>` line per CelebA-HQ image | CelebA-Dialog's "identity (HQ)" file, saved under this name ([project page](https://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebA_Dialog.html), [repository](https://github.com/ziqihuangg/CelebA-Dialog); Google Drive file `1yd0bfYkcQ9_BqIxhjyjZ6UHgNVg7n63V`) | SHA-256 `919c06655ace6cfd03cbec04ff717125c478b88b48f4a46cb293ddf0bc7422f1` (printed by `check_labels.py`); 30,000 lines, 6,217 identities |
| CelebA-Dialog's "Image (HQ)": the same 30,000 pictures at 1024 × 1024, named `0.jpg` to `29999.jpg` by CelebA-HQ index | Google Drive file `1A2dNWabg6_um-V3lhw1tyead5hCpjaW8`, linked from the CelebA-Dialog repository | a 2,734,306,468-byte zip; needed only to build the image map, and can be deleted afterward |

Weights, in one folder passed as `--weights`:

| File | Where | SHA-256 |
|---|---|---|
| `face_detection_yunet_2023mar.onnx` (YuNet, 232,589 bytes) | [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet); MIT License | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |
| `face_recognition_sface_2021dec.onnx` (SFace) | [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface); Apache License 2.0 | `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` |
| `w600k_r50.onnx` (ArcFace ResNet-50, WebFace600K), from `buffalo_l.zip` | [InsightFace v0.7 release](https://github.com/deepinsight/insightface/releases/tag/v0.7); non-commercial research only | `4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43` |
| `model.pth` (the original project's saved model, 6.5 MB) | attached to this repository's [release](https://github.com/ryanjosephkamp/deep-learning-studies/releases); trained on CelebA-HQ images, so non-commercial research only; without it, leave `original` out of `--stages` | `74042b212800d0ae3d7326533f3a6516a9a71a5d94ec92d6b1c6e2f2ae7784ed` |

`run_study.py` checks the SFace and original-model hashes and records every hash in
`run_log.json`.

## Rebuilding the identity mapping

The Kaggle copy numbers its files in an order unrelated to the CelebA-HQ index that
`identity_hq.txt` uses, so its file numbers can't be used as the index. `match_images.py` finds
each Kaggle image's index by matching small thumbnails against the "Image (HQ)" copy, whose file
names are the index:

```bash
pip install -r study/requirements.txt
python study/match_images.py --images DATA/celeba_hq_256 --ref DATA/hq_ref --out DATA/image_map.csv
```

Expect every image matched, with a match correlation of at least 0.999, about 30 images taking
a candidate other than their best to keep the map one-to-one (CelebA-HQ holds a few near-duplicate
pictures), and the map fingerprint
`c9a8b69ab607f9d47810926693f0e682af6c7939769b1d8285f697973438fc21`. If a near-duplicate pair
lands the other way round, the fingerprint differs while the map still works; the next step
decides.

Align the images, then test the labels on the aligned crops before anything trains:

```bash
python study/run_study.py --images DATA/celeba_hq_256 --labels DATA/identity_hq.txt \
  --image-map DATA/image_map.csv --weights WEIGHTS --out OUT --stages prepare align
python study/check_labels.py --results OUT --labels DATA/identity_hq.txt \
  --image-map DATA/image_map.csv --weights WEIGHTS
```

`check_labels.py` scores two readings of the labels with ArcFace on 5,000 aligned crops. Reading
them by file number should give a rank-1 near 0.001 (DOES NOT FIT); reading them through the map
should give a rank-1 near 0.97 and genuine pairs far above impostors (fits). It exits with code
0 only if the map fits.

## Running

```bash
python study/run_study.py --images DATA/celeba_hq_256 --labels DATA/identity_hq.txt \
  --image-map DATA/image_map.csv --weights WEIGHTS --out OUT --device cuda
python study/analyze.py --results OUT
```

The same `--out` resumes: finished units are skipped and training restarts from its last
checkpoint. After the pretrained stage, SFace and ArcFace must each reach a clean-probe rank-1 of
at least 0.5 (`label_check_min_rank1` in `config.json`); otherwise the run writes `HALTED.json`
and stops before any later stage. The `summary` stage rebuilds the summary from every finished unit each time it runs. `--smoke` runs a small synthetic version on a CPU with no data or
weights.

On an NVIDIA A10 with 30 CPU threads, the November 11, 2026 run took 32 minutes for alignment and
the pretrained models (CPU) and 86 minutes for the rest, 75 of them training the 15 runs.
`analyze.py` took under a minute on two CPU threads.

`OUT/eval/` holds per-probe results, and `OUT/prepare/` the split and the aligned crops; both
carry CelebA file names and identity numbers, so only the aggregates in `../results/study/` are
in this repository.

`python study/check_near_copies.py --results OUT`, run from the project folder, reads the clean
per-probe results and writes `results/study/near_copies.csv`, a sensitivity check: the probes most
similar to their own gallery image (ArcFace R50 cosine of at least 0.9) and the clean rank-1 of five
models with and without them (the paper's Limitations).

The optional `deepface` stage (DeepFace 0.0.96 with TensorFlow 2.19.0 and tf-keras 2.19.0) puts
DeepFace's six models on the same crops with the detector skipped, for the first probe of each
identity and the centered squares only; it was not part of the run in
`../results/study/`.
