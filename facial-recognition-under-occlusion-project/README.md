# Face matching under synthetic occlusion

The project trains a small CNN with a spatial attention layer and triplet loss to match face images partly covered by a centered black square, and compares it with six pretrained models in DeepFace on CelebA-HQ. On 100 images, the custom model matched 90% of images with 20% of the image covered, where the pretrained models matched 0 to 6%. Two later checks change how to read that. Matching on raw pixels, with no model at all, finds 100% of the same images at 10% occlusion, so the task is easy. And for SFace at 5 to 10% occlusion, an approximate re-creation traces most of the failure to the pipeline rather than the model: DeepFace's face detector stops finding the face once the square appears, and the whole frame is then embedded against a gallery of face crops. The same is likely for the other pretrained models, which weren't rerun.

Coursework for CS 6073 Deep Learning (final project), University of Cincinnati, fall 2025.

An identification study ([`study/`](study/), results in [`results/study/`](results/study/)) asks the question the original protocol could not: whether a model picks out a different photo of the same person, among 1,000 people held out from the newly trained networks, when part of the face is covered. The write-up is a paper ([`paper/main.tex`](paper/main.tex), [PDF](https://ryanjosephkamp.github.io/deep-learning-studies/facial-recognition-under-occlusion-project/paper.pdf)) and a [chapter of the course site](https://ryanjosephkamp.github.io/deep-learning-studies/facial-recognition-under-occlusion-project/) ([`chapter/chapter.md`](chapter/chapter.md)).

Known and potential problems found after release, with the status of each, are listed in [`KNOWN-ISSUES.md`](../KNOWN-ISSUES.md) at the repository's root.

## Question

The project measures how matching accuracy falls as more of each face image is hidden, whether a CNN with spatial attention trained on occluded examples holds up better than pretrained face recognition models, and whether removing the background helps.

## Setup

- Data: N random CelebA-HQ images (256 × 256). The gallery holds the clean images; each probe is one of them with a centered black square covering 0 to 100% of its area, in 5% steps. A probe is correct when its nearest gallery embedding is its own clean copy, so the task is matching an image to itself, not to another photo of the same person.
- Custom model: four convolution blocks with a 7 × 7 spatial attention layer after the first and a 128-dimensional embedding, trained for 5 epochs with triplet loss (margin 1.0, Adam, learning rate 0.0005). The positive is the anchor with a random 10 to 50% centered square.
- Baselines: VGG-Face, FaceNet, FaceNet512, OpenFace, ArcFace, and SFace through DeepFace with its default face detector, used as released, matched by Euclidean distance on unnormalized embeddings.
- Scale: one run each at N = 50, 100, 200, and 500, plus background-removed copies (`rembg`) of the N = 50, 100, and 200 samples. No random seed was fixed. The result files are in [`results/original/`](results/original/).
- The comparison is not matched. The custom model was trained once, on 1,000 CelebA-HQ images drawn without a seed (the list wasn't saved, so a few may overlap a test sample), with the same kind of square it is tested on; the pretrained models were used as released, without project-specific training. Each background-removed run fine-tuned it on that run's own test images. No version without the attention layer was tested.

## Results

- With no occlusion, all seven models matched every image in the reported runs. At N = 100, the custom model matched 99% at 10% occlusion, 90% at 20%, 73% at 30%, 50% at 40%, and 31% at 50%. The pretrained models fell to 1 to 13% at 10%; VGG-Face held 72% at 5%.
- At N = 200, the custom model scored 95.5% at 10%, 78% at 20%, 59.5% at 30%, and 19.5% at 50%; the pretrained models stayed between 0 and 10.5% from 10% on.
- Removing the background lowered the custom model's accuracy from 90% to 56% at 20% occlusion and from 73% to 33% at 30%. VGG-Face dropped from 72% to 42% at 5%.
- Raw pixels (nearest neighbor on the 256 × 256 images, no learning) match 100% at 10% occlusion, 76% at 20%, and 54% at 30% on the N = 100 sample, and 90% at 20% once the background is removed. So the pretrained models' 1 to 13% at 10% isn't a lack of information.
- The original runs saved no detection results, so this was re-created on CPU for SFace, with OpenCV's Haar detector at DeepFace's settings (without DeepFace's eye alignment) and the original's Euclidean matching. The detector finds a face in 95 of the 100 clean images, 3 at 5% occlusion, and none at 10%, and missed probes are embedded from the whole frame. That path gives 30% at 5% and 11% at 10% (the saved SFace results are 16% and 2%, so the re-creation is approximate); the same face crop for probe and gallery gives 100% and 84%. From 15% on, the square covers over a third of the face box detected on the clean image (median), and SFace falls with the same face crop on both sides and with whole frames on both sides: 26 to 28% at 15%, 4 to 17% at 20%. The other five models weren't rerun, and VGG-Face's 72% at 5% shows the effect differs by model.
- The custom model, rerun on CPU from the saved weights, reproduces the N = 50, 100, and 200 curves within three images at every level. On the background-removed images, the model before fine-tuning scores 68% at 20% occlusion; the fine-tuned model in the report scores 56%.

The original report (dated December 4, 2025; not in this repository) also gives statistics (a t-test, an effect size, and confidence intervals) that no code in the repository computes, and a column for N = 1,000, a run that stopped before writing results; they are left out here. [`analysis/`](analysis/) recomputes the report's numbers from the result files and holds the checks above.

## Study results

The identification study's main results (the paper and chapter give the intervals, tests, and limits):

- With each face aligned before it is covered, the pretrained ArcFace R50 identifies 96.8% of probes clean, 89.5% with 20% of the crop covered, and 41.3% at 50%. SFace starts at 92.3% and falls to 33.6% at 20%. These occlusion levels are shares of the 112 × 112 crop, not of the original's frame, so they can't be set against the numbers above, and aligning on the clean image makes them likely an upper bound for a real pipeline.
- In the project's network trained from scratch, training with occluded faces adds 12.1 points of rank-1 on the centered black squares from 10 to 50% it trained with (95% interval +10.9 to +13.4) and 5.2 on average on occluders it never saw (+4.5 to +6.0), and costs 13.4 points on clean faces.
- The attention layer's main comparison, averaged over the 14 occluders of the first two, is +1.0 point (95% bootstrap interval +0.5 to +1.5); it comes from the noise fills and the lower band, and with three seeds per configuration, whether it is significant depends on the test. It costs 2.2 points on clean faces.
- The four CosFace configurations are weak identifiers (15.6 to 31.3% clean rank-1), so the study is about the differences between them. A fifth, trained with a batch-hard triplet loss, failed to train (1.3% clean rank-1).

## Run it

```bash
pip install -r requirements.txt
# put the CelebA-HQ 256 × 256 images in celeba_dataset/celeba_hq_256/
python frfpv1_main.py
```

The script saves the demo figures, then runs N = 100, 500, and 1000, writing `experiments/exper__size_N__<timestamp>/results/accuracy_vs_occlusion.csv` for each. A new model trains on a seeded draw of 1,000 images, saves their names next to its weights, and every test sample excludes them. It reuses a saved model in `novel_model/` if one exists; delete it to retrain. The original model from the release has no saved list of its training images, so with it, a test sample may include some of them (the script warns). On Colab, paths default to `/content/drive/MyDrive/frfp1`. The original project code keeps the fall 2025 phrases “novel model” and “key innovation”; the paper and chapter give the current reading.

```bash
python analysis/recompute_original.py   # the report's numbers, from results/original/
```

[`study/README.md`](study/README.md) explains how to obtain the data, rebuild the identity mapping, and run the identification study.

## License

The code is under the MIT License. CelebA-HQ is built from [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), which is for non-commercial research only and may not be redistributed, so no images or identity labels are included; download CelebA-HQ to run the code. CelebA-HQ's creators released the files that rebuild it under CC BY-NC 4.0, and CelebA-Dialog's identity labels carry CelebA's terms.

The original model's weights (`model.pth`), attached to this repository's [release](https://github.com/ryanjosephkamp/deep-learning-studies/releases), were trained on CelebA-HQ images and are for non-commercial research only. The study's pretrained models have their own terms: InsightFace's ArcFace R50 is for non-commercial research only, SFace is under the Apache License 2.0, and YuNet is under the MIT License ([`study/README.md`](study/README.md) gives the sources and hashes).

The paper's references are in [`paper/sections/references.tex`](paper/sections/references.tex); [`literature/README.md`](literature/README.md) lists the original project's reading.
