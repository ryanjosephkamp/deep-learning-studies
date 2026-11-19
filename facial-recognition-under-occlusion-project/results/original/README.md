# Original results

The accuracy files written by the project's Colab runs (November 30 to December 3, 2025, on an
A100 runtime), copied unchanged from the project's Drive folder. Each file has one row per
occlusion level and one column per model; a value is the share of probes whose nearest gallery
embedding (L2) was their own clean image. The `exper__*` file names are the run folders' names; the timestamp
is the run's start.

The sampled CelebA-HQ images, reference embeddings, and per-run weights stayed on Drive; CelebA-HQ
may not be redistributed.

## Runs the report, poster, and slides draw on

| File | N | Levels | Custom model | Notes |
|---|---|---|---|---|
| `exper__size_50__20251201_192436.csv` | 50 | 21 | `novel_model/model.pth` | Report Table 5, N = 50 column |
| `exper__size_100__20251130_235656.csv` | 100 | 21 | `novel_model/model.pth` | Report Table 1 and the slides' N = 100 table |
| `exper__size_200__20251201_203249.csv` | 200 | 21 | `novel_model/model.pth` | Report Table 2 |
| `exper__size_500__20251201_004627.csv` | 500 | 21 | `novel_model/model.pth` | Report Table 5, N = 500 column |
| `exper__NO_BG__size_50__20251203_001017.csv` | 50 | 21 | fine-tuned on these 50 images | Same images as the N = 50 run, backgrounds removed |
| `exper__NO_BG__size_100__20251203_012434.csv` | 100 | 21 | fine-tuned on these 100 images | Report Tables 3 and 4; same images as the N = 100 run |
| `exper__NO_BG__size_200__20251203_022706.csv` | 200 | 21 | fine-tuned on these 200 images | Same images as the N = 200 run |

`novel_model/model.pth` was trained once, for 5 epochs with batch size 16, on 1,000 images drawn
at random (no seed) from the 30,000; that list was not saved. The standard runs loaded it
unchanged. The evidence differs by run: the N = 50 and N = 200 run folders hold a copy with the
same SHA-256; the N = 100 and N = 500 runs (November 30 and December 1) kept no copy, but their
code loads that file, which was written before they started, and a CPU rerun with it reproduces
the N = 100 file within two images per level. The file is attached to the repository's [release](https://github.com/ryanjosephkamp/deep-learning-studies/releases)
(SHA-256 `74042b212800d0ae3d7326533f3a6516a9a71a5d94ec92d6b1c6e2f2ae7784ed`). Each background-removed run fine-tuned the model for
5 more epochs on that run's own background-removed images, the images it was then tested on.

The reported runs are the ones that finished. Nine other run folders hold a sample and no result
file (six at N = 50 and one each at N = 100, 500, and 1,000), and one background-removed N = 50
folder is empty.

No run with N = 1,000 finished: `exper__size_1000__20251201_045416` stopped before writing results.

## Other runs

| File | N | Levels | Notes |
|---|---|---|---|
| `frfp1_results_20251130.csv` | 50 | 101 | The first full run (morning of November 30); its sample wasn't saved |
| `exper__size_100__20251130_194129.csv` | 100 | 101 | A 1%-step run on a different sample |
| `exper__NO_BG__size_50__20251201_234404.csv`, `..._20251202_001740.csv`, `..._20251202_182734.csv` | 50 | 21 | Earlier background-removed runs, each with a model trained on that run's images (`model_nobg.pth`) |
| `exper__NO_BG__size_50__20251202_224921.csv` | 50 | 21 | Same images, master weights, and fine-tuning code as `..._20251203_001017`; with no seed, the two runs score 24% and 70% at 20% occlusion |
| `exper__size_5__*.csv` (five files) | 5 | 21 | Pipeline test runs |
