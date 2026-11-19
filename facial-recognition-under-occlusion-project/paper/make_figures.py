# make_figures.py
# Draws the paper's figures from the committed result files (no image data is used or needed):
#   results/original/*.csv, analysis/out/*.csv       the original runs and their CPU checks
#   results/study/analysis/*.csv                      the identification study
# Writes paper/figures/*.pdf (for the paper) and chapter/figures/*.png (for the site chapter).
#
# Usage (from the project folder or from paper/):
#     python paper/make_figures.py

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
OUT_PDF = PAPER / "figures"
OUT_PNG = ROOT / "chapter/figures"
STUDY = ROOT / "results/study/analysis"

plt.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "legend.fontsize": 6.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "lines.linewidth": 1.4,
    "pdf.fonttype": 42, "font.family": "DejaVu Sans",
})

# One color per model, in a fixed order, with a marker as a second cue.
STYLE = {
    "arcface_r50": ("ArcFace R50", "#2a78d6", "o"),
    "sface": ("SFace", "#eb6834", "s"),
    "cosface_att_aug": ("CNN, both", "#1baf7a", "D"),
    "cosface_noatt_aug": ("CNN, augmentation", "#eda100", "v"),
    "cosface_att_noaug": ("CNN, attention", "#e87ba4", "^"),
    "cosface_noatt_noaug": ("CNN, neither", "#008300", "P"),
    "triplet_att_aug": ("CNN, both, triplet", "#6250d6", "X"),
    "pixels_crop": ("Crop pixels", "#7a7974", "."),
    "original_model": ("Original model", "#e34948", "o"),
    "pixels_frame": ("Frame pixels", "#7a7974", "."),
}
INK, MUTED = "#0b0b0b", "#52514e"


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def save(fig, name):
    OUT_PDF.mkdir(exist_ok=True)
    OUT_PNG.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None})  # reproducible bytes
    fig.savefig(OUT_PNG / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def fig_original():
    """The original runs on their frame-coverage axis: the custom model against raw pixels, and the SFace detector check."""
    runs = {"50": "exper__size_50__20251201_192436", "100": "exper__size_100__20251130_235656",
            "200": "exper__size_200__20251201_203249", "500": "exper__size_500__20251201_004627"}
    pix = read(ROOT / "analysis/out/check_pixel_baseline.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.35))
    ax = axes[0]
    shades = {"50": "#9dc3ef", "100": "#5d9be3", "200": "#2a78d6", "500": "#174f94"}
    for n, f in runs.items():
        rows = read(ROOT / f"results/original/{f}.csv")
        x = [100 * float(r["Occlusion"]) for r in rows]
        ax.plot(x, [100 * float(r["Custom"]) for r in rows], color=shades[n], marker="o", ms=2.5,
                label=f"Custom model, N = {n}")
        ax.plot([100 * float(r["Occlusion"]) for r in pix], [100 * float(r[f]) for r in pix], color=shades[n],
                ls="--", lw=1.1)
    ax.plot([], [], color=MUTED, ls="--", lw=1.1, label="Raw pixels (same sample)")
    ax.set_xlim(0, 70)
    ax.set_ylim(0, 102)
    ax.set_xlabel("Black square, % of the 256-px frame")
    ax.set_ylabel("Same-image matching accuracy (%)")
    ax.set_title("(a) Custom model and raw pixels", loc="left")
    ax.legend(frameon=False, loc="upper right")

    ax = axes[1]
    det = read(ROOT / "analysis/out/check_detector_sface.csv")
    x = [100 * float(r["Occlusion"]) for r in det]
    series = [("saved_sface", "SFace, saved original results", "#eb6834", "-", "s"),
              ("deepface_path", "SFace, DeepFace path re-created", "#eb6834", ":", "s"),
              ("same_crop", "SFace, same crop for probe and gallery", "#6250d6", "-", "D"),
              ("face_detected", "Face found by the detector", MUTED, "--", None)]
    for col, lab, colr, ls, mk in series:
        ax.plot(x, [100 * float(r[col]) for r in det], color=colr, ls=ls, marker=mk, ms=2.5, label=lab)
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 102)
    ax.set_xlabel("Black square, % of the 256-px frame")
    ax.set_ylabel("Share of the N = 100 sample (%)")
    ax.set_title("(b) SFace detector check, N = 100", loc="left")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig_original")


def fig_occluders():
    """The study's occluders on the 112 x 112 aligned crop, with the five template landmarks (no face)."""
    # Typed: the destination landmarks of OpenCV's FaceRecognizerSF.alignCrop (the 112 x 112 ArcFace
    # template), from modules/objdetect/src/face_recognize.cpp in OpenCV 4.x.
    lm = [(38.2946, 51.6963), (73.5318, 51.5014), (56.0252, 71.7366), (41.5493, 92.3655), (70.7299, 92.2041)]
    import json
    cfg = json.load(open(ROOT / "study/config.json"))
    size = cfg["crop_size"]
    (e0, e1), (l0, l1) = cfg["conditions"]["eyes_band_rows"], cfg["conditions"]["lower_band_rows"]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.3))
    for ax in axes:
        ax.add_patch(Rectangle((0, 0), size, size, fill=False, ec=INK, lw=0.8))
        ax.set_xlim(-2, size + 2)
        ax.set_ylim(size + 2, -2)
        ax.set_aspect("equal")
        ax.axis("off")
    ax = axes[0]
    for cov, shade in ((0.6, "#e6e5e0"), (0.4, "#c3c2b7"), (0.2, "#8a8984"), (0.1, "#52514e")):
        side = int((cov * size * size) ** 0.5)
        t = (size - side) // 2
        ax.add_patch(Rectangle((t, t), side, side, fc=shade, ec="white", lw=0.8))
        ax.text(t + 1.5, t + 1.5, f"{int(cov * 100)}%", va="top", ha="left", fontsize=6.5,
                color="white" if cov <= 0.2 else INK)
    ax.set_title("(a) Centered squares, % of the crop", loc="left")
    ax = axes[1]
    ax.add_patch(Rectangle((0, int(e0 * size)), size, int(e1 * size) - int(e0 * size), fc="#52514e", ec="none"))
    ax.text(size / 2, (int(e0 * size) + int(e1 * size)) / 2, f"eye band, rows {e0:.0%} to {e1:.0%}", color="white",
            ha="center", va="center", fontsize=6.5)
    ax.set_title(f"(b) Eye band ({e1 - e0:.0%} of the crop)", loc="left")
    ax = axes[2]
    ax.add_patch(Rectangle((0, int(l0 * size)), size, int(l1 * size) - int(l0 * size), fc="#52514e", ec="none"))
    ax.text(size / 2, (int(l0 * size) + int(l1 * size)) / 2, f"lower band, rows {l0:.0%} to {l1:.0%}", color="white",
            ha="center", va="center", fontsize=6.5)
    ax.set_title(f"(c) Lower band ({l1 - l0:.0%} of the crop)", loc="left")
    for ax in axes:
        ax.scatter([p[0] for p in lm], [p[1] for p in lm], s=14, c="#eb6834", edgecolors="white", linewidths=0.6,
                   zorder=5)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_occluders")


def fig_squares():
    """Identity rank-1 against centered-square coverage of the crop, with 95% intervals."""
    iv = read(STUDY / "intervals.csv")
    get = {(r["model"], r["condition"]): r for r in iv if r["protocol"] == "identity" and r["metric"] == "rank1"}
    covs = ["0.00", "0.05", "0.10", "0.15", "0.20", "0.25", "0.30", "0.40", "0.50", "0.60"]
    x = [100 * float(c) for c in covs]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.5))
    panels = [(["arcface_r50", "sface", "cosface_noatt_noaug", "cosface_att_aug", "pixels_crop"], 100, "(a) Pretrained models, two CNNs, and crop pixels"),
              (["cosface_att_aug", "cosface_noatt_aug", "cosface_att_noaug", "cosface_noatt_noaug", "triplet_att_aug",
                "pixels_crop"], 35, "(b) Trained CNNs and crop pixels (y axis to 35%)")]
    for ax, (models, ymax, title) in zip(axes, panels):
        for m in models:
            lab, col, mk = STYLE[m]
            y = [100 * float(get[(m, f"square_{c}")]["estimate"]) for c in covs]
            lo = [100 * float(get[(m, f"square_{c}")]["ci_low"]) for c in covs]
            hi = [100 * float(get[(m, f"square_{c}")]["ci_high"]) for c in covs]
            ax.fill_between(x, lo, hi, color=col, alpha=0.18, lw=0)
            ax.plot(x, y, color=col, marker=mk, ms=3, label=lab)
        ax.set_ylim(0, ymax)
        ax.set_xlim(0, 61)
        ax.set_xlabel("Centered black square, % of the 112-px crop")
        ax.set_ylabel("Identity rank-1 (%)")
        ax.set_title(title, loc="left")
        ax.legend(frameon=False, loc="upper right", ncol=1)
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig_squares")


def fig_other():
    """Other occluders beside the centered black square of matched area, identity rank-1 with 95% intervals."""
    iv = read(STUDY / "intervals.csv")
    get = {(r["model"], r["condition"]): r for r in iv if r["protocol"] == "identity" and r["metric"] == "rank1"}
    groups = [("Eye band\n26%", [("square_0.25", "square 25%"), ("eyes_band", "eye band")]),
              ("Lower band\n40%", [("square_0.40", "square 40%"), ("lower_band", "lower band")]),
              ("Off-center\n20%", [("square_0.20", "centered 20%"), ("random_square_0.20", "off-center 20%")]),
              ("Fills\n20%", [("square_0.20", "black 20%"), ("gray_square_0.20", "gray 20%"), ("noise_square_0.20", "noise 20%")]),
              ("Fills\n40%", [("square_0.40", "black 40%"), ("gray_square_0.40", "gray 40%"), ("noise_square_0.40", "noise 40%")])]
    models = ["arcface_r50", "sface", "cosface_att_aug", "cosface_noatt_aug", "cosface_att_noaug", "cosface_noatt_noaug"]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), gridspec_kw={"width_ratios": [1, 1]})
    for ax, ms, ymax, title in ((axes[0], models[:2], 100, "(a) Pretrained models"),
                                 (axes[1], models[2:], 25, "(b) CosFace CNNs (y axis to 25%)")):
        pos, ticks, labels = 0, [], []
        offs = {m: (i - (len(ms) - 1) / 2) * 0.22 for i, m in enumerate(ms)}
        for gname, conds in groups:
            for cond, clab in conds:
                for m in ms:
                    r = get[(m, cond)]
                    y = 100 * float(r["estimate"])
                    lo, hi = 100 * float(r["ci_low"]), 100 * float(r["ci_high"])
                    lab, col, mk = STYLE[m]
                    filled = cond.startswith("square_")
                    ax.errorbar(pos + offs[m], y, yerr=[[y - lo], [hi - y]], fmt=mk, ms=3.2, color=col,
                                mfc=col if not filled else "white", mec=col, elinewidth=0.9, capsize=0)
                ticks.append(pos)
                labels.append(clab)
                pos += 1
            pos += 0.6
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=6.2)
        ax.set_ylim(0, ymax)
        ax.set_ylabel("Identity rank-1 (%)")
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
        for m in ms:
            lab, col, mk = STYLE[m]
            ax.plot([], [], mk, color=col, ms=3.2, label=lab)
    axes[1].plot([], [], "o", color=MUTED, mfc="white", ms=3.2, label="open marker: centered black square")
    h = [x for ax in axes for x in ax.get_legend_handles_labels()[0]]
    lab = [x for ax in axes for x in ax.get_legend_handles_labels()[1]]
    fig.tight_layout(w_pad=2.0, rect=(0, 0.1, 1, 1))
    fig.legend(h, lab, frameon=False, loc="lower center", ncol=4, columnspacing=1.2)
    save(fig, "fig_other")


def fig_contrasts():
    """(a) The three primary contrasts; (b) the per-condition main effects they average over."""
    prim = read(STUDY / "contrasts_primary.csv")
    me = read(STUDY / "main_effects_by_condition.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9), gridspec_kw={"width_ratios": [0.8, 2.2]})
    ax = axes[0]
    for i, r in enumerate(prim):
        y = -i
        est = 100 * float(r["estimate"])
        ax.plot([100 * float(r["ci983_low"]), 100 * float(r["ci983_high"])], [y, y], color=MUTED, lw=0.9)
        ax.plot([100 * float(r["ci95_low"]), 100 * float(r["ci95_high"])], [y, y], color=INK, lw=2.6,
                solid_capstyle="butt")
        ax.plot([100 * float(r["seed_combo_min"]), 100 * float(r["seed_combo_max"])], [y - 0.22, y - 0.22],
                color="#eb6834", lw=1.2)
        ax.plot(est, y, "o", color="white", mec=INK, ms=4.5, zorder=5)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks([0, -1, -2])
    ax.set_yticklabels(["P1 augmentation,\ntrained-on squares", "P2 augmentation,\nunseen occluders",
                        "P3 attention,\nP1 and P2 occluders"], fontsize=6.5)
    ax.set_ylim(-3.7, 0.5)
    ax.set_xlabel("Main effect on rank-1 (points)")
    ax.set_title("(a) Primary contrasts", loc="left")
    ax.plot([], [], color=INK, lw=2.6, label="95% interval")
    ax.plot([], [], color=MUTED, lw=0.9, label="98.3% interval")
    ax.plot([], [], color="#eb6834", lw=1.2, label="one seed per config.")
    ax.legend(frameon=False, loc="lower right", fontsize=6)
    ax.grid(axis="y", visible=False)

    ax = axes[1]
    order = ["square_0.00", "square_0.05", "square_0.10", "square_0.15", "square_0.20", "square_0.25", "square_0.30",
             "square_0.40", "square_0.50", "square_0.60", "eyes_band", "lower_band", "random_square_0.20",
             "gray_square_0.20", "noise_square_0.20", "gray_square_0.40", "noise_square_0.40"]
    names = ["clean", "5%", "10%", "15%", "20%", "25%", "30%", "40%", "50%", "60%", "eye band", "lower band",
             "off-center 20%", "gray 20%", "noise 20%", "gray 40%", "noise 40%"]
    get = {(r["main_effect"], r["condition"]): r for r in me}
    for k, (eff, col, mk, dx) in enumerate((("augmentation", "#eda100", "v", -0.17), ("attention", "#e87ba4", "^", 0.17))):
        for i, c in enumerate(order):
            r = get[(eff, c)]
            y = 100 * float(r["estimate"])
            lo, hi = 100 * float(r["ci_low"]), 100 * float(r["ci_high"])
            ax.errorbar(i + dx, y, yerr=[[y - lo], [hi - y]], fmt=mk, ms=3.4, color=col, elinewidth=0.9,
                        label=f"{eff.capitalize()} main effect" if i == 0 else None)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvspan(1.5, 8.5, color="#f1f0ec", zorder=0, lw=0)
    ax.text(5, 15.5, "P1 squares (in P3 too)", ha="center", fontsize=6, color=MUTED)
    ax.axvspan(9.5, 16.5, color="#f6f5f2", zorder=0, lw=0)
    ax.text(13, 15.5, "P2 unseen occluders (in P3 too)", ha="center", fontsize=6, color=MUTED)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(names, rotation=55, ha="right", fontsize=6.3)
    ax.set_ylim(-16, 17)
    ax.set_ylabel("Points of rank-1")
    ax.set_title("(b) Main effects at each condition (descriptive, unadjusted)", loc="left")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", visible=False)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_contrasts")


def fig_gap():
    """Same-image against identity rank-1: crop models on crop coverage, frame models on frame coverage."""
    gap = read(STUDY / "protocol_gap.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.95))
    crop = ["arcface_r50", "sface", "cosface_att_aug", "cosface_noatt_noaug", "triplet_att_aug", "pixels_crop"]
    frame = ["original_model", "pixels_frame"]
    for ax, models, xlab, title, xmax in (
            (axes[0], crop, "Centered black square, % of the 112-px crop", "(a) Crop models", 52),
            (axes[1], frame, "Black square, % of the 256-px frame", "(b) Frame models", 52)):
        for m in models:
            rows = [r for r in gap if r["model"] == m]
            rows.sort(key=lambda r: float(r["condition"].split("_")[-1]))
            rows = [r for r in rows if float(r["condition"].split("_")[-1]) <= 0.5]
            x = [100 * float(r["condition"].split("_")[-1]) for r in rows]
            lab, col, mk = STYLE[m]
            ax.plot(x, [100 * float(r["same_image_rank1"]) for r in rows], color=col, marker=mk, ms=3, label=lab)
            ax.plot(x, [100 * float(r["identity_rank1"]) for r in rows], color=col, marker=mk, ms=3, ls=":",
                    mfc="white")
        ax.plot([], [], color=MUTED, label="solid: same image")
        ax.plot([], [], color=MUTED, ls=":", label="dotted: identity")
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, 102)
        ax.set_xlabel(xlab)
        ax.set_ylabel("Rank-1 (%)")
        ax.set_title(title, loc="left")
        # below the axes, so no legend sits on a curve
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=4 if ax is axes[0] else 2,
                  fontsize=6, columnspacing=1.2, handlelength=1.8)
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig_gap")


if __name__ == "__main__":
    fig_original()
    fig_occluders()
    fig_squares()
    fig_other()
    fig_contrasts()
    fig_gap()
