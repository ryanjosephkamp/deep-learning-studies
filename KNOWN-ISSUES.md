# Known issues

This file lists known and potential problems in this project: a claim that may be stronger than its evidence, a
result that depends on a choice the text doesn't state, or something that can't yet be reproduced from this
repository. They were found in reviews after release. Each entry says what was observed, where, and its status. An
entry marked "potential" has not been rerun.

I'm working through them as time allows. A fixed entry stays here with the commit that fixed it.

If you find something else, please [open an issue](https://github.com/ryanjosephkamp/deep-learning-studies/issues).

Limits that the paper already states (oracle alignment, synthetic occluders, one dataset and split, three seeds per
configuration, no validation split, and what can't be regenerated without the withheld per-probe files) are not
repeated here; see the paper's Limitations and "Reproducing the results" sections.

## Facial recognition under occlusion

The study in [`facial-recognition-under-occlusion-project/`](facial-recognition-under-occlusion-project/). Paths
below are relative to that folder.

### Claims and wording

#### KI-01. The verification thresholds are set on the test identities and held fixed in the intervals

- **Kind:** wording. **Status:** open, November 25, 2026.
- **Where:** `study/evaluate.py` (lines 13 to 14 and 132 to 143) and `study/analyze.py` (lines 99 to 101 and 150 to
  172); the paper's study design section and its results for the pretrained models (the two places that mention a
  deployed system), and the "Seeds" item in Limitations.
- **Observed:** each model's fixed thresholds are set on the clean probes of the same 1,000 test identities whose
  occluded results are then reported. So in the clean condition, the actual false accept rate at the fixed threshold
  equals its target (1% and 0.1%) for every model except the three triplet runs, which come out slightly below
  (`results/study/summary.csv`). The paper says the thresholds are set "on clean identity probes," and calls this
  reading what "a deployed system would" use and "the reading a deployed system would see"; a deployed system sets
  its threshold on other data. The bootstrap intervals for the fixed-threshold true accept rate resample identities
  and seeds but keep each stored threshold, so they leave out the threshold's own uncertainty; Limitations mention
  the fixed gallery but not the fixed threshold. How much wider the intervals would be has not been measured.
- **Affects:** the two "deployed system" phrases (and the same phrase in `study/evaluate.py`), and the stated width
  of the fixed-threshold true accept rate intervals in the verification table and text. No point estimate changes.
- **To check:** `study/evaluate.py` `fixed()`; the clean rows (`square_0`, identity protocol) of
  `results/study/summary.csv`, columns `far_fixed_far_0.01` and `far_fixed_far_0.001`.

#### KI-02. The closest prior work isn't cited

- **Kind:** prior work. **Status:** open, November 25, 2026.
- **Where:** the paper's related work section and references.
- **Observed:** X. Xu, N. Sarafianos, and I. A. Kakadiaris, "On Improving the Generalization of Face Recognition in
  the Presence of Occlusions" (CVPR Workshops 2020, arXiv:2006.06787), combines an attention mechanism with a
  training strategy that balances occluded and clean faces: the project's two ideas, together. The paper cites a
  survey and a masked-face benchmark for occlusion and says it doesn't compare with methods built for occluded faces,
  but doesn't cite this work. The paper claims no novelty for the combination.
- **Affects:** related work.
- **To check:** `paper/sections/related.tex`, `paper/sections/references.tex`.

#### KI-03. No statement on demographic scope or fitness for real use

- **Kind:** wording. **Status:** open, November 25, 2026.
- **Where:** the paper's Limitations, the chapter, and the READMEs.
- **Observed:** the text says the data are mostly frontal, high-quality photos of celebrities, but nowhere says that
  no demographic or subgroup analysis was done, or that the results say nothing about whether these models are fit
  for real-world use.
- **Affects:** Limitations, and the chapter and READMEs that summarize them.
- **To check:** `paper/sections/limitations.tex`, `chapter/chapter.md`, `README.md`.

### Methods and results

#### KI-04. Augmentation's cost on clean faces is measured at one 40-epoch training budget

- **Kind:** result. **Status:** potential, November 25, 2026 (not rerun).
- **Where:** `study/config.json` (`train`), `study/train.py` (the learning-rate schedule), and
  `results/study/train/cosface_*/train_log.csv`; the paper's ablation results (the trade-off), Discussion,
  Conclusion, and abstract, and the chapter and site that repeat them.
- **Observed:** every trained arm runs 40 epochs on a cosine learning-rate schedule that reaches zero at epoch 40,
  with no validation split (a limit the paper states). In the arms trained with occlusion augmentation, each training
  input is occluded with probability 0.5. Their CosFace loss at epoch 40 is 7.53 to 7.77, which the matching arms
  without augmentation had already reached by epoch 19 to 21; those end at 3.75 to 4.08. The two losses aren't
  directly comparable, since the augmented arms' loss includes occluded inputs, so this shows a possible confound,
  not a cause: at this budget, the augmented arms may simply be less trained on clean faces. The paper reads the
  13.4-point [12.0, 14.9] clean cost as a trade inherent to augmentation.
- **Affects:** the clean cost of augmentation and the reading of augmentation as a trade. A longer training run of
  each arm would test it; none has been run.
- **To check:** the `loss` and `lr` columns of `results/study/train/cosface_*_seed*/train_log.csv`, and
  `occlusion_probability` and `epochs` in `study/config.json`.

### Reproducibility

#### KI-05. Exact reruns weren't tested

- **Kind:** reproducibility. **Status:** open, November 25, 2026.
- **Where:** `results/study/run_log.json` (`environment`); the paper's "Reproducing the results."
- **Observed:** the study's runs are seeded, but the run log records cuDNN's deterministic mode off
  (`cudnn_deterministic: false`, with `cudnn_benchmark: false` and `tf32_cudnn: true`), and no seeded rerun was
  compared with the committed results. Whether a rerun reproduces the scores exactly is untested, and the paper
  doesn't say so. (That the original 2025 runs' PyTorch version wasn't logged is already stated in
  `requirements.txt`.)
- **Affects:** what a reader can expect from rerunning `study/run_study.py`.
- **To check:** `results/study/run_log.json`.

#### KI-06. No tests of the scoring or the identity split

- **Kind:** reproducibility. **Status:** open, November 25, 2026.
- **Where:** `study/`.
- **Observed:** the repository has no test files. `study/run_study.py --smoke` runs the pipeline on synthetic data
  and stops if rank-1 falls below a sanity floor (`label_check_min_rank1`), but it checks no known answer: not a
  rank, not an accept rate at a threshold, and not that the training and test identities don't overlap. The one
  assertion in `study/` is in `check_near_copies.py`. The paper's numbers agree with the committed results because
  `paper/make_numbers.py` generates them from those files, which doesn't test the scoring itself.
- **Affects:** how a reader can check the rank and verification code without the data.
- **To check:** `git ls-files`, `study/run_study.py`.
