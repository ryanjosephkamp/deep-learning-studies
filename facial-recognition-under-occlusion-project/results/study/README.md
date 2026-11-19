# Study results

Aggregate outputs of [`study/run_study.py`](../../study/run_study.py), `study/analyze.py`, and
`study/check_near_copies.py` from the run of
November 11, 2026 (NVIDIA A10, 30 CPU threads; environment, settings, weight hashes, and counts in
`run_log.json`). Per-probe results, the split, and the aligned crops carry CelebA file names and
identity numbers and are not included.

| Path | What |
|---|---|
| `run_log.json` | settings (`config`), environment, SHA-256 of the weights, label file, and image map, image counts after alignment (`counts`), and the label check's clean rank-1 (`label_check`) |
| `DONE.json` | written when every requested unit had finished: unit count and the environment |
| `run.log` | the console output: image matching, the label check on 5,000 aligned crops, the split, and one line per unit with its time |
| `summary.csv` | one row per model (trained runs per seed), protocol (`identity` or `same_image`), and condition: probes, identities, rank-1, mean rank, and TAR at FAR 1% and 0.1% with the threshold set per condition (`tar_at_far_*`) or fixed on clean probes (`tar_fixed_far_*`, with the false accept rate it gives, `far_fixed_far_*`) |
| `analysis/intervals.csv` | estimate and 95% interval for rank-1 and fixed-threshold TAR per model, protocol, and condition; intervals resample test identities and, for trained configurations, their three seeds |
| `analysis/seed_spread.csv` | per-seed values with their minimum, maximum, range, and standard deviation |
| `analysis/contrasts_primary.csv` | the three primary contrasts (P1, P2, P3): estimate, 95% and 98.3% intervals, bootstrap and Holm p-values (with the text to quote), the range over one-seed-per-configuration combinations, and whether the effect is claimed |
| `analysis/main_effects_by_condition.csv` | the augmentation and attention main effects at each condition, the terms the primary contrasts average over (descriptive) |
| `analysis/contrasts_by_condition.csv` | the five simple contrasts at each condition (descriptive, unadjusted) |
| `analysis/seed_pairs.csv` | each simple contrast's nine seed-pair differences per condition |
| `analysis/protocol_gap.csv` | same-image minus identity rank-1 per model and square, with intervals |
| `near_copies.csv` | a sensitivity check: for five models, clean identity rank-1 with and without the probes most similar to their own gallery image (ArcFace R50 cosine of at least 0.9), from `study/check_near_copies.py` |
| `eval/<model>/fixed_thresholds.json` | each model's verification thresholds at FAR 1% and 0.1%, set on clean probes |
| `eval/<run>/attention_check.csv` | mean attention weight inside and outside the occluder on the first 256 probes, for runs with the attention layer |
| `train/<run>/train_log.csv` | per-epoch loss, learning rate, and seconds for each of the 15 training runs |

Rank-1 chance is 0.001 (1,000 gallery identities). Conditions named `square_*` are centered black
squares covering that share of the 112 × 112 crop; `frame_square_*` (raw frame pixels and the
original model) cover that share of the 256 × 256 frame, a different axis.
