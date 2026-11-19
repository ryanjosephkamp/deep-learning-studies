# make_site_figures.py
# Draws the five figures of the chapter's web page in the course book's own style: Old Standard type,
# one ink and a red second ink, in a day (light) and a night (dark) printing. The data, axis limits,
# ticks, marks, and text are those of paper/make_figures.py: this script reads the same result files
# with the same functions and repeats its plotting steps, changing only colors, type, sizes, where the
# keys sit, and, in the occluder drawing, where six labels sit so that squares and landmarks no longer
# touch them, and a filled crop (listed in ACCEPTED below).
#
# Writes, for each figure, chapter/figures/site/<name>.light.svg and <name>.dark.svg (panels side by
# side, for the page's plate at about 600 to 1,100 CSS px wide) and <name>.phone.light.svg and
# <name>.phone.dark.svg (panels stacked, for about 360 CSS px wide), all with text drawn as paths and a
# transparent background, made to sit on the plate mount; and chapter/figures/site/manifest.json with
# the files, the colors, and the result of the check below.
#
# The check builds each figure in memory with paper/make_figures.py and with this script, in every render
# (both printings, side by side and stacked), and compares them panel by panel, exactly: every line's x
# and y data, every collection's offsets (where their transform places them), paths, and path codes, every
# patch, every text and where it sits, the axis scales and limits, and the visible ticks and their labels;
# whether each of these is drawn at all (shown, not transparent, with a stroke, fill, or markers), how it
# is clipped, and which coordinates it is placed in; each line's style, class of dash sequence, marker,
# and marker fill; the order of stroke widths within each panel (which lines are thicker, thinner, or the
# same), including the key samples. Colors and sizes may change, but which series share a color may not,
# and every labeled series and key entry must have its documented color (COLORS). Every deliberate
# difference is listed, value for value, in ACCEPTED. The check also records the project files each figure
# read, and draws every render again to compare it with the file on disk, so a stale or edited file fails.
#
# Old Standard comes from the site's own fonts in docs/assets/fonts (SIL Open Font License); pass --fonts
# to use another copy.
#
# Usage (from the project folder):
#     python chapter/make_site_figures.py            # draw every render, then check
#     python chapter/make_site_figures.py --check    # check only

import argparse
import builtins
import contextlib
import io
import json
import re
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager, transforms
from matplotlib.collections import Collection, LineCollection, PathCollection
from matplotlib.colors import to_hex, to_rgb, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.text import Text
from matplotlib.ticker import MultipleLocator

CHAPTER = Path(__file__).resolve().parent
ROOT = CHAPTER.parent
OUT = CHAPTER / "figures/site"
FONTS = ROOT.parent / "docs/assets/fonts"
sys.path.insert(0, str(ROOT / "paper"))
sys.dont_write_bytecode = True  # nothing is written into paper/
import make_figures as paper  # noqa: E402  (its rcParams stay global; the site figures set their own)

FIGURES = ["fig_original", "fig_occluders", "fig_squares", "fig_gap", "fig_contrasts"]
SVG_LIMIT = 300_000  # bytes; a heavier figure is written as WebP at twice its widest display size instead
WIDEST_CSS_PX, PHONE_CSS_PX = 1100, 420

# The book's two printings (assets/book.css), and the plate mount each figure sits on.
PRINTINGS = {
    "light": {"mount": "#fbf8f1", "ink": "#1b1714", "ink2": "#594f43", "rubric": "#b8321a", "crop": "#fbf8f1",
              "band_alpha": 0.16, "grid_alpha": 0.11, "span_alpha": (0.055, 0.035)},
    "dark": {"mount": "#1c1f24", "ink": "#ece5d6", "ink2": "#b3ab9b", "rubric": "#ff7a52", "crop": "#ece5d6",
             "band_alpha": 0.22, "grid_alpha": 0.10, "span_alpha": (0.075, 0.045)},
}

# One color per model or series, kept across all five figures: (day printing, night printing).
# ArcFace R50 takes the rubric and the plain CNN the ink; raw pixels are a warm gray. Every color
# has at least 3:1 contrast against its printing's mount (checked when the figures are drawn).
COLORS = {
    "arcface_r50": ("#b8321a", "#ff7a52"),
    "sface": ("#285994", "#8cc3fc"),
    "sface_same_crop": ("#5e8ab8", "#6291ba"),
    "cosface_att_aug": ("#00503d", "#4cae86"),
    "cosface_noatt_aug": ("#b58127", "#e6b55d"),
    "cosface_att_noaug": ("#9b4270", "#e88db6"),
    "cosface_noatt_noaug": ("#1b1714", "#ece5d6"),
    "triplet_att_aug": ("#8273bc", "#8e7fc9"),
    "pixels_crop": ("#8d8881", "#76716a"),
    "pixels_frame": ("#8d8881", "#76716a"),
    "original_model": ("#743e19", "#e1ad80"),
}
# The original model's runs at each sample size, in steps of its own color (N = 200 is its color).
SAMPLE_STEPS = {"50": ("#b9805e", "#946a45"), "100": ("#985c36", "#bd8b5f"),
                "200": ("#743e19", "#e1ad80"), "500": ("#4d260a", "#fad4b4")}
# The two main effects wear the colors of the CNN that has only that idea.
EFFECT_COLOR = {"augmentation": "cosface_noatt_aug", "attention": "cosface_att_noaug"}

# The occluder drawing shows the crop as a pale image with dark occluders in both printings: the crop is
# filled with the printing's "crop" color and the occluders are this ink laid over it.
OCCLUDER_INK = "#1b1714"

# Type sizes in points (and the turn of fig_contrasts' rotated tick labels, in degrees). Side by side, the
# figures are drawn 7.1 in wide for a plate of about 714 CSS px; stacked, about 3.6 to 3.75 in wide for
# about 360 CSS px, where 8 pt comes out near 11 px.
TYPES = {
    "wide": {"base": 9.0, "title": 10.5, "label": 9.5, "tick": 9.0, "key": 9.0, "note": 8.5, "span": 8.5,
             "corner": 7.5, "turn": 55},
    "phone": {"base": 9.0, "title": 10.0, "label": 9.5, "tick": 9.0, "key": 9.0, "note": 8.5, "span": 8.0,
              "corner": 8.0, "turn": 65},
}
LAYOUTS = ("wide", "phone")

# Each size label in the occluder figure's panel (a) stands on a baseline this many rows below its square's
# top edge, so it fits inside the ring before the next square (the paper hangs it from the corner, where it
# touches the next square's edge at this type size).
CORNER_BASELINE = 5.6

# Rows (on the 112-px crop) for the occluder figure's band labels: above the eyes in the eye band
# (rows 33 to 62, eyes near row 52), and below the mouth corners in the lower band (rows 67 to 112,
# mouth corners near row 92).
BAND_LABEL_ROWS = {"eye": 40.5, "lower": 103.5}

# In the stacked fig_contrasts, the P2 note's height (points of rank-1): at the foot of its span, where the
# effects leave room once the key hangs below the panel.
P2_NOTE_ROW_PHONE = -15.0

# Differences from the paper's figures that are deliberate. Each names the property, the paper's value, and
# the one value the site figure may have instead ("layout" limits an entry to one layout); a "move" entry
# names artists that leave the color group of "old" and join that of "new". The check lists them and still
# passes, and fails if an entry no longer applies.
_RING = "the size label stands inside its ring instead of hanging from the corner, so it clears the next square"
_CROP = "the crop is filled as a pale image, so the occluders read dark on it in both printings"
_HOLLOW = ("the estimate is a ring with nothing inside, so the 95% interval shows through it even where it is "
           "shorter than the marker (P3)")
ACCEPTED = [
    *[{"where": f"fig_occluders.panels[0].texts[{i}].position", "old": [t + 1.5, t + 1.5],
       "new": [t + 1.5, t + 5.6], "reason": _RING} for i, t in enumerate((13, 21, 31, 38))],
    *[{"where": f"fig_occluders.panels[0].texts[{i}].va", "old": "top", "new": "baseline", "reason": _RING}
      for i in range(4)],
    {"where": "fig_occluders.panels[1].texts[0].position", "old": [56.0, 47.5], "new": [56.0, 40.5],
     "reason": "the eye-band label moved up within the band so the eye landmarks no longer sit on it"},
    {"where": "fig_occluders.panels[2].texts[0].position", "old": [56.0, 89.5], "new": [56.0, 103.5],
     "reason": "the lower-band label moved down within the band so the mouth-corner landmarks no longer sit on it"},
    *[{"where": f"fig_occluders.panels[{i}].patches[0].filled", "old": False, "new": True, "reason": _CROP}
      for i in range(3)],
    {"where": "fig_contrasts.panels[1].texts[1].position", "layout": "phone", "old": [13.0, 15.5],
     "new": [13.0, -15.0],
     "reason": "stacked, the panel is too narrow for the two span notes to share a row, so the P2 note sits at "
               "the foot of its span"},
    *[{"where": f"fig_contrasts.panels[0].lines[{j}].fill", "old": "paper", "new": "none", "reason": _HOLLOW}
      for j in (3, 7, 11)],
    {"where": "fig_contrasts.color_groups", "move": ["panels[0].lines[3]", "panels[0].lines[7]",
                                                     "panels[0].lines[11]"],
     "old": "panels[0].lines[1]", "new": "panels[0].lines[2]",
     "reason": "the estimates take the rubric, the color of their one-seed ranges, instead of the ink of the 95% "
               "intervals"},
]


class Look:
    """One render: a printing's inks (page tokens as attributes, series colors in .color) and a layout
    ("wide", panels side by side, or "phone", panels stacked) with its type sizes in .type."""

    def __init__(self, name, layout="wide"):
        self.name, self.layout = name, layout
        self.tag = name if layout == "wide" else f"{layout}.{name}"
        self.__dict__.update(PRINTINGS[name])
        i = 0 if name == "light" else 1
        self.color = {k: v[i] for k, v in COLORS.items()}
        self.step = {k: v[i] for k, v in SAMPLE_STEPS.items()}
        self.type = TYPES[layout]

    def shade(self, strength):
        """The occluder ink laid over the crop at the given strength, as one opaque color."""
        ink, crop = np.array(to_rgb(OCCLUDER_INK)), np.array(to_rgb(self.crop))
        return to_hex(crop + strength * (ink - crop))


def contrast(a, b):
    """WCAG contrast ratio of two colors."""
    def lum(c):
        rgb = np.array(to_rgb(c))
        rgb = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def style(pr):
    """rcParams for one printing: Old Standard, quiet axes with no box, a faint grid, keys without frames."""
    axis = to_rgba(pr.ink2, 0.75)
    return {
        "font.family": "Old Standard TT", "font.size": pr.type["base"],
        "axes.titlesize": pr.type["title"], "axes.titlelocation": "left", "axes.titlepad": 8,
        "axes.titlecolor": pr.ink, "axes.labelsize": pr.type["label"], "axes.labelcolor": pr.ink,
        "axes.labelpad": 5, "text.color": pr.ink,
        "axes.facecolor": "none", "figure.facecolor": "none", "savefig.facecolor": "none",
        "savefig.transparent": True,
        "axes.edgecolor": axis, "axes.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False,
        "axes.axisbelow": True, "axes.grid": True, "axes.grid.axis": "y",
        "grid.color": pr.ink, "grid.alpha": pr.grid_alpha, "grid.linewidth": 0.6,
        "xtick.color": axis, "ytick.color": axis, "xtick.labelcolor": pr.ink2, "ytick.labelcolor": pr.ink2,
        "xtick.labelsize": pr.type["tick"], "ytick.labelsize": pr.type["tick"],
        "xtick.major.size": 3, "ytick.major.size": 3, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.major.pad": 3.5, "ytick.major.pad": 3.5,
        "lines.linewidth": 1.5, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "lines.dash_capstyle": "round", "lines.dash_joinstyle": "round", "lines.scale_dashes": True,
        "lines.dashed_pattern": (2.6, 3.0), "lines.dotted_pattern": (0.05, 2.1),
        "patch.linewidth": 0.8,
        "legend.frameon": False, "legend.fontsize": pr.type["key"], "legend.labelcolor": pr.ink,
        "legend.handlelength": 2.6, "legend.handletextpad": 0.6, "legend.labelspacing": 0.4,
        "legend.columnspacing": 1.4, "legend.borderaxespad": 0.3,
        "svg.fonttype": "path", "svg.hashsalt": "make_site_figures", "axes.unicode_minus": True,
    }


def load_fonts(folder, scratch):
    """Registers Old Standard with matplotlib, converting the site's WOFF2 files to TrueType first."""
    from fontTools.ttLib import TTFont  # fontTools with brotli reads WOFF2
    for name in ("old-standard-400", "old-standard-400i", "old-standard-700"):
        src = Path(folder) / f"{name}.woff2"
        if not src.exists():
            sys.exit(f"Old Standard not found at {src}; pass --fonts with a folder that holds its WOFF2 files.")
        font = TTFont(src)
        font.flavor = None
        dst = Path(scratch) / f"{name}.ttf"
        font.save(dst)
        font_manager.fontManager.addfont(str(dst))


def keys_below(fig, axes, gap=6):
    """Hangs each listed panel's key under its x-axis label, `gap` points below it."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in axes:
        depth = (ax.bbox.y0 - ax.xaxis.get_tightbbox(renderer).y0) * 72 / fig.dpi
        shift = transforms.ScaledTranslation(0, -(depth + gap) / 72, fig.dpi_scale_trans)
        legend = ax.get_legend()
        legend.set_loc("upper center")
        legend.set_bbox_to_anchor((0.5, 0), transform=ax.transAxes + shift)


def left_edge(fig, axes):
    """The leftmost point, in pixels, of the panels' y-axis labels and tick labels."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    return min(ax.yaxis.get_tightbbox(renderer).x0 for ax in axes if ax.axison)


def titles_to_edge(fig, axes):
    """Starts each panel title at the figure's left edge, under which the y-axis labels sit, rather than at
    the axes, so long titles fit the narrow, stacked layout."""
    if not any(ax.axison for ax in axes):
        return
    left = left_edge(fig, axes)
    for ax in axes:
        if ax.axison:
            shift = transforms.ScaledTranslation((left - ax.bbox.x0) / fig.dpi, 0, fig.dpi_scale_trans)
            ax._left_title.set_transform(ax.transAxes + shift + ax.titleOffsetTrans)


def widen_to_edge(fig, axes):
    """Extends each stacked panel to the left until its own y-axis labels reach the figure's left edge, so a
    panel with short tick labels is not held to the margin of one with long ones."""
    left = left_edge(fig, axes)
    renderer = fig.canvas.get_renderer()
    width = fig.bbox.width
    for ax in axes:
        gain = (ax.yaxis.get_tightbbox(renderer).x0 - left) / width
        if gain > 0:
            box = ax.get_position()
            ax.set_position([box.x0 - gain, box.y0, box.width + gain, box.height])


def lay(fig, axes, pr, keys, wide, phone):
    """Arranges a figure that was plotted at the paper's size and panel widths: side by side at the wide
    size, or stacked one panel per row at the phone size; then hangs the listed panels' keys under their
    x-axis labels and fits everything with tight_layout. Plotting at the paper's geometry first keeps
    autoscaled limits the same as the paper's to the last bit (matplotlib takes lines and spans drawn in
    axes coordinates into them through the figure's geometry)."""
    fig.full_width = pr.layout == "phone" and phone.get("full_width", False)
    if pr.layout == "phone":
        stack = fig.add_gridspec(len(axes), 1, height_ratios=phone.get("heights"))
        for i, ax in enumerate(axes):
            ax.set_subplotspec(stack[i, 0])
        fig.set_size_inches(*phone["size"])
    else:
        fig.set_size_inches(*wide["size"])
        if "widths" in wide:
            axes[0].get_gridspec().set_width_ratios(wide["widths"])
    keys = phone.get("keys", keys) if pr.layout == "phone" else keys
    if len(keys):
        keys_below(fig, keys)
    if pr.layout == "phone":
        titles_to_edge(fig, axes)
        fig.tight_layout(h_pad=phone.get("h_pad", 2.2))
        if phone.get("own_left"):
            widen_to_edge(fig, axes)
            titles_to_edge(fig, axes)
    else:
        fig.tight_layout(w_pad=wide["w_pad"])


def marker_ring(pr, marker="o"):
    """A thin ring of the mount around filled markers, so crossing marks stay apart (none on the small point
    marker, which a ring would make look open)."""
    return {"mew": 0} if marker == "." else {"mec": pr.mount, "mew": 0.6}


# ----------------------------------------------------------------------------------------------------
# The five figures. Each repeats the loops of the function of the same name in paper/make_figures.py,
# at the paper's figure size; only colors, sizes, the arrangement of panels and keys, and, in the occluder
# drawing, six labels and the filled crop differ.

def fig_original(pr):
    """The original runs on their frame-coverage axis: the custom model against raw pixels, and the SFace detector check."""
    runs = {"50": "exper__size_50__20251201_192436", "100": "exper__size_100__20251130_235656",
            "200": "exper__size_200__20251201_203249", "500": "exper__size_500__20251201_004627"}
    pix = paper.read(ROOT / "analysis/out/check_pixel_baseline.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.35))
    ax = axes[0]
    for n, f in runs.items():
        rows = paper.read(ROOT / f"results/original/{f}.csv")
        x = [100 * float(r["Occlusion"]) for r in rows]
        ax.plot(x, [100 * float(r["Custom"]) for r in rows], color=pr.step[n], marker="o", ms=3.6,
                **marker_ring(pr), label=f"Custom model, N = {n}")
        ax.plot([100 * float(r["Occlusion"]) for r in pix], [100 * float(r[f]) for r in pix], color=pr.step[n],
                ls="--", lw=1.15)
    ax.plot([], [], color=pr.ink2, ls="--", lw=1.15, label="Raw pixels (same sample)")
    ax.set_xlim(0, 70)
    ax.set_ylim(0, 102)
    ax.set_xticks(range(0, 71, 10))
    ax.set_yticks(range(0, 101, 20))
    ax.set_xlabel("Black square, % of the 256-px frame")
    ax.set_ylabel("Same-image matching accuracy (%)")
    ax.set_title("(a) Custom model and raw pixels", loc="left")
    ax.legend(frameon=False, loc="upper center")

    ax = axes[1]
    det = paper.read(ROOT / "analysis/out/check_detector_sface.csv")
    x = [100 * float(r["Occlusion"]) for r in det]
    series = [("saved_sface", "SFace, saved original results", pr.color["sface"], "-", "s"),
              ("deepface_path", "SFace, DeepFace path re-created", pr.color["sface"], ":", "s"),
              ("same_crop", "SFace, same crop for probe and gallery", pr.color["sface_same_crop"], "-", "D"),
              ("face_detected", "Face found by the detector", pr.ink2, "--", None)]
    for col, lab, colr, ls, mk in series:
        ax.plot(x, [100 * float(r[col]) for r in det], color=colr, ls=ls, marker=mk, ms=3.8, **marker_ring(pr),
                label=lab)
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 102)
    ax.set_xticks(range(0, 31, 5))
    ax.set_yticks(range(0, 101, 20))
    ax.set_xlabel("Black square, % of the 256-px frame")
    ax.set_ylabel("Share of the N = 100 sample (%)")
    ax.set_title("(b) SFace detector check, N = 100", loc="left")
    ax.legend(frameon=False, loc="upper center")
    lay(fig, axes, pr, keys=axes, wide={"size": (7.1, 4.2), "w_pad": 2.4}, phone={"size": (3.75, 8.4)})
    return fig


def fig_occluders(pr):
    """The study's occluders on the 112 x 112 aligned crop, with the five template landmarks (no face)."""
    lm = [(38.2946, 51.6963), (73.5318, 51.5014), (56.0252, 71.7366), (41.5493, 92.3655), (70.7299, 92.2041)]
    cfg = json.load(open(ROOT / "study/config.json"))
    size = cfg["crop_size"]
    (e0, e1), (l0, l1) = cfg["conditions"]["eyes_band_rows"], cfg["conditions"]["lower_band_rows"]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.3))
    for ax in axes:
        ax.add_patch(Rectangle((0, 0), size, size, fc=pr.crop, ec=pr.ink2, lw=0.8))  # the crop, as a pale image
        ax.set_xlim(-2, size + 2)
        ax.set_ylim(size + 2, -2)
        ax.set_aspect("equal")
        ax.axis("off")
    ax = axes[0]
    for cov, shade in ((0.6, pr.shade(0.10)), (0.4, pr.shade(0.26)), (0.2, pr.shade(0.56)), (0.1, pr.shade(0.78))):
        side = int((cov * size * size) ** 0.5)
        t = (size - side) // 2
        ax.add_patch(Rectangle((t, t), side, side, fc=shade, ec=pr.crop, lw=1.0))
        ax.text(t + 1.5, t + CORNER_BASELINE, f"{int(cov * 100)}%", va="baseline", ha="left",
                fontsize=pr.type["corner"], color=pr.crop if cov <= 0.2 else OCCLUDER_INK)
    ax.set_title("(a) Centered squares, % of the crop", loc="left")
    # The band labels sit where the landmarks leave room (the paper centers them on the band, under the dots).
    ax = axes[1]
    ax.add_patch(Rectangle((0, int(e0 * size)), size, int(e1 * size) - int(e0 * size), fc=pr.shade(0.78), ec="none"))
    ax.text(size / 2, BAND_LABEL_ROWS["eye"], f"eye band, rows {e0:.0%} to {e1:.0%}", color=pr.crop,
            ha="center", va="center", fontsize=pr.type["note"])
    ax.set_title(f"(b) Eye band ({e1 - e0:.0%} of the crop)", loc="left")
    ax = axes[2]
    ax.add_patch(Rectangle((0, int(l0 * size)), size, int(l1 * size) - int(l0 * size), fc=pr.shade(0.78), ec="none"))
    ax.text(size / 2, BAND_LABEL_ROWS["lower"], f"lower band, rows {l0:.0%} to {l1:.0%}", color=pr.crop,
            ha="center", va="center", fontsize=pr.type["note"])
    ax.set_title(f"(c) Lower band ({l1 - l0:.0%} of the crop)", loc="left")
    for ax in axes:
        ax.scatter([p[0] for p in lm], [p[1] for p in lm], s=26, c=pr.rubric, edgecolors=pr.mount, linewidths=0.9,
                   zorder=5)
    lay(fig, axes, pr, keys=[], wide={"size": (7.1, 2.75), "w_pad": 3.0},
        phone={"size": (3.6, 6.9), "h_pad": 1.0, "full_width": True})
    return fig


def fig_squares(pr):
    """Identity rank-1 against centered-square coverage of the crop, with 95% intervals."""
    iv = paper.read(paper.STUDY / "intervals.csv")
    get = {(r["model"], r["condition"]): r for r in iv if r["protocol"] == "identity" and r["metric"] == "rank1"}
    covs = ["0.00", "0.05", "0.10", "0.15", "0.20", "0.25", "0.30", "0.40", "0.50", "0.60"]
    x = [100 * float(c) for c in covs]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.5))
    panels = [(["arcface_r50", "sface", "cosface_noatt_noaug", "cosface_att_aug", "pixels_crop"], 100, "(a) Pretrained models, two CNNs, and crop pixels"),
              (["cosface_att_aug", "cosface_noatt_aug", "cosface_att_noaug", "cosface_noatt_noaug", "triplet_att_aug",
                "pixels_crop"], 35, "(b) Trained CNNs and crop pixels (y axis to 35%)")]
    for ax, (models, ymax, title) in zip(axes, panels):
        for m in models:
            lab, _, mk = paper.STYLE[m]
            col = pr.color[m]
            y = [100 * float(get[(m, f"square_{c}")]["estimate"]) for c in covs]
            lo = [100 * float(get[(m, f"square_{c}")]["ci_low"]) for c in covs]
            hi = [100 * float(get[(m, f"square_{c}")]["ci_high"]) for c in covs]
            ax.fill_between(x, lo, hi, color=col, alpha=pr.band_alpha, lw=0)
            ax.plot(x, y, color=col, marker=mk, ms=4.2, **marker_ring(pr, mk), label=lab)
        ax.set_ylim(0, ymax)
        ax.set_xlim(0, 61)
        ax.set_xticks(range(0, 61, 10))
        ax.set_yticks(range(0, ymax + 1, 20 if ymax == 100 else 5))
        ax.set_xlabel("Centered black square, % of the 112-px crop")
        ax.set_ylabel("Identity rank-1 (%)")
        ax.set_title(title, loc="left")
        ax.legend(frameon=False, loc="upper center", ncol=2)
    lay(fig, axes, pr, keys=axes, wide={"size": (7.1, 3.75), "w_pad": 3.2}, phone={"size": (3.75, 7.6)})
    return fig


def fig_contrasts(pr):
    """(a) The three primary contrasts; (b) the per-condition main effects they average over."""
    prim = paper.read(paper.STUDY / "contrasts_primary.csv")
    me = paper.read(paper.STUDY / "main_effects_by_condition.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9), gridspec_kw={"width_ratios": [0.8, 2.2]})
    ax = axes[0]
    for i, r in enumerate(prim):
        y = -i
        est = 100 * float(r["estimate"])
        ax.plot([100 * float(r["ci983_low"]), 100 * float(r["ci983_high"])], [y, y], color=pr.ink2, lw=1.0)
        ax.plot([100 * float(r["ci95_low"]), 100 * float(r["ci95_high"])], [y, y], color=pr.ink, lw=3.2,
                solid_capstyle="butt")
        ax.plot([100 * float(r["seed_combo_min"]), 100 * float(r["seed_combo_max"])], [y - 0.22, y - 0.22],
                color=pr.rubric, lw=1.4)
        # A ring with nothing inside, under the interval lines, so the 95% interval shows through it even
        # where the interval is shorter than the marker (P3).
        ax.plot(est, y, "o", color=pr.rubric, mfc="none", mew=1.3, ms=6.0, zorder=1.9)
    ax.axvline(0, color=pr.ink2, lw=0.8)
    ax.set_yticks([0, -1, -2])
    ax.set_yticklabels(["P1 augmentation,\ntrained-on squares", "P2 augmentation,\nunseen occluders",
                        "P3 attention,\nP1 and P2 occluders"], fontsize=pr.type["tick"], color=pr.ink, linespacing=1.25)
    ax.set_ylim(-3.7, 0.5)
    ax.xaxis.set_major_locator(MultipleLocator(5))  # the paper's ticks, without touching the autoscaled limits
    ax.set_xlabel("Main effect on rank-1 (points)")
    ax.set_title("(a) Primary contrasts", loc="left")
    ax.plot([], [], color=pr.ink, lw=3.2, solid_capstyle="butt", label="95% interval")
    ax.plot([], [], color=pr.ink2, lw=1.0, label="98.3% interval")
    ax.plot([], [], color=pr.rubric, lw=1.4, label="one seed per config.")
    ax.legend(frameon=False, loc="upper center", fontsize=pr.type["key"], handlelength=2.0)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.tick_params(axis="y", length=0)

    ax = axes[1]
    order = ["square_0.00", "square_0.05", "square_0.10", "square_0.15", "square_0.20", "square_0.25", "square_0.30",
             "square_0.40", "square_0.50", "square_0.60", "eyes_band", "lower_band", "random_square_0.20",
             "gray_square_0.20", "noise_square_0.20", "gray_square_0.40", "noise_square_0.40"]
    names = ["clean", "5%", "10%", "15%", "20%", "25%", "30%", "40%", "50%", "60%", "eye band", "lower band",
             "off-center 20%", "gray 20%", "noise 20%", "gray 40%", "noise 40%"]
    get = {(r["main_effect"], r["condition"]): r for r in me}
    for k, (eff, col, mk, dx) in enumerate((("augmentation", pr.color[EFFECT_COLOR["augmentation"]], "v", -0.17),
                                             ("attention", pr.color[EFFECT_COLOR["attention"]], "^", 0.17))):
        for i, c in enumerate(order):
            r = get[(eff, c)]
            y = 100 * float(r["estimate"])
            lo, hi = 100 * float(r["ci_low"]), 100 * float(r["ci_high"])
            ax.errorbar(i + dx, y, yerr=[[y - lo], [hi - y]], fmt=mk, ms=5.6, color=col, elinewidth=1.2,
                        **marker_ring(pr), label=f"{eff.capitalize()} main effect" if i == 0 else None)
    ax.axhline(0, color=pr.ink2, lw=0.8)
    ax.axvspan(1.5, 8.5, color=pr.ink, alpha=pr.span_alpha[0], zorder=0, lw=0)
    ax.text(5, 15.5, "P1 squares (in P3 too)", ha="center", fontsize=pr.type["span"], color=pr.ink2, style="italic")
    ax.axvspan(9.5, 16.5, color=pr.ink, alpha=pr.span_alpha[1], zorder=0, lw=0)
    ax.text(13, 15.5 if pr.layout == "wide" else P2_NOTE_ROW_PHONE, "P2 unseen occluders (in P3 too)", ha="center",
            fontsize=pr.type["span"], color=pr.ink2, style="italic")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(names, rotation=pr.type["turn"], ha="right", fontsize=pr.type["tick"], rotation_mode="anchor")
    ax.set_ylim(-16, 17)
    ax.set_yticks(range(-15, 16, 5))
    ax.set_ylabel("Points of rank-1")
    ax.set_title("(b) Main effects at each condition (descriptive, unadjusted)", loc="left")
    ax.legend(frameon=False, loc="lower right", handlelength=1.4)
    ax.grid(axis="x", visible=False)
    lay(fig, axes, pr, keys=axes[:1], wide={"size": (7.1, 3.7), "widths": [0.85, 2.15], "w_pad": 1.8},
        phone={"size": (3.75, 7.6), "heights": [0.8, 1.2], "own_left": True, "keys": axes})
    return fig


def fig_gap(pr):
    """Same-image against identity rank-1: crop models on crop coverage, frame models on frame coverage."""
    gap = paper.read(paper.STUDY / "protocol_gap.csv")
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
            lab, _, mk = paper.STYLE[m]
            col = pr.color[m]
            ax.plot(x, [100 * float(r["same_image_rank1"]) for r in rows], color=col, marker=mk, ms=4.2,
                    **marker_ring(pr, mk), label=lab)
            ax.plot(x, [100 * float(r["identity_rank1"]) for r in rows], color=col, marker=mk, ms=4.2, ls=":",
                    mfc=pr.mount, mec=col, mew=1.0)
        ax.plot([], [], color=pr.ink2, label="solid: same image")
        ax.plot([], [], color=pr.ink2, ls=":", label="dotted: identity")
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, 102)
        ax.set_xticks(range(0, 51, 10))
        ax.set_yticks(range(0, 101, 20))
        ax.set_xlabel(xlab)
        ax.set_ylabel("Rank-1 (%)")
        ax.set_title(title, loc="left")
        ax.legend(frameon=False, loc="upper center", ncol=2, fontsize=pr.type["key"])
    lay(fig, axes, pr, keys=axes, wide={"size": (7.1, 4.1), "w_pad": 2.4}, phone={"size": (3.75, 8.0)})
    return fig


SITE = {"fig_original": fig_original, "fig_occluders": fig_occluders, "fig_squares": fig_squares,
        "fig_gap": fig_gap, "fig_contrasts": fig_contrasts}


# ----------------------------------------------------------------------------------------------------
# Drawing

def render(name, pr):
    """Draws one figure in one printing and layout, in memory; returns the file's bytes, its suffix, and its
    drawn width and height in points (for the page's width and height attributes)."""
    with plt.rc_context(style(pr)):
        fig = SITE[name](pr)
        bounds = "tight"
        if getattr(fig, "full_width", False):
            # Trimmed top and bottom only, so a drawing narrower than its sheet keeps the sheet's width.
            tight = fig.get_tightbbox(fig.canvas.get_renderer())
            bounds = transforms.Bbox([[0, tight.y0 - 0.04], [fig.get_size_inches()[0], tight.y1 + 0.04]])
        out = io.BytesIO()
        fig.savefig(out, format="svg", bbox_inches=bounds, pad_inches=0.04, metadata={"Date": None})
        data, suffix = out.getvalue(), ".svg"
        head = data[:1000].decode()
        box = [round(float(head.split(f'{k}="')[1].split("pt")[0]), 1) for k in ("width", "height")]
        if len(data) > SVG_LIMIT:
            # Too heavy as a vector drawing: a raster at twice the widest display size instead.
            from PIL import Image
            png, webp = io.BytesIO(), io.BytesIO()
            widest = WIDEST_CSS_PX if pr.layout == "wide" else PHONE_CSS_PX
            fig.savefig(png, format="png", dpi=2 * widest / fig.get_size_inches()[0], bbox_inches="tight",
                        pad_inches=0.04)
            Image.open(png).save(webp, "WEBP", quality=92, method=6)
            data, suffix = webp.getvalue(), ".webp"
        plt.close(fig)
    return data, suffix, box


def draw(name, pr):
    """Draws one figure in one printing and layout and writes it; returns the file name, its size in bytes,
    and its drawn width and height in points."""
    data, suffix, box = render(name, pr)
    path = OUT / f"{name}.{pr.tag}{suffix}"
    other = path.with_suffix(".webp" if suffix == ".svg" else ".svg")
    if other.exists():  # an earlier drawing in the other format
        other.unlink()
    path.write_bytes(data)
    return path.name, len(data), box


# ----------------------------------------------------------------------------------------------------
# The check

@contextlib.contextmanager
def files_read(log):
    """Notes every project file opened while a figure is built."""
    real_open = builtins.open

    def noting_open(file, *args, **kwargs):
        try:
            p = Path(file).resolve()
            if ROOT in p.parents:
                log.add(p.relative_to(ROOT).as_posix())
        except TypeError:
            pass
        return real_open(file, *args, **kwargs)

    builtins.open = noting_open
    try:
        yield
    finally:
        builtins.open = real_open


def build_paper(name):
    """The paper's figure, built in memory by make_figures' own function, with its save step held back."""
    held, real_save = [], paper.save
    paper.save = lambda fig, _name: held.append(fig)
    try:
        getattr(paper, name)()
    finally:
        paper.save = real_save
    return held[0]


PAPER_FACES = {"#ffffff", PRINTINGS["light"]["mount"], PRINTINGS["dark"]["mount"]}


def arr(values):
    return np.asarray(values, dtype=float)


def marker_of(artist):
    m = artist.get_marker()
    return None if m in (None, "None", "none", "", " ") else str(m)


def fill_of(line):
    """How a line's markers are filled: "filled" in their own color, "paper" (the page's color, so they read
    as open), "none" (so what lies under them shows), or a partial fill style; None without markers."""
    if marker_of(line) is None:
        return None
    if line.get_fillstyle() != "full":
        return line.get_fillstyle()
    face = to_rgba(line.get_markerfacecolor())
    if face[3] == 0:
        return "none"
    return "paper" if to_hex(face) in PAPER_FACES else "filled"


def dash_class(seq):
    """Solid, dotted, dashed, or dash-dot, from an on/off sequence in multiples of the line width."""
    if not seq:
        return "solid"
    ons = list(seq)[0::2]
    if max(ons) <= 1:
        return "dotted"
    return "dash-dot" if min(ons) <= 1 else "dashed"


def stroke_of(artist):
    """A line's stroke: its width, its on/off dash sequence in multiples of that width (None when solid), and
    the sequence's class; None when the artist draws no stroke. (Matplotlib keeps the unscaled sequences in
    _unscaled_dash_pattern and _us_linestyles.)"""
    if isinstance(artist, Line2D):
        if artist.get_linestyle() in ("None", "none", "", " "):
            return None
        width, (_, seq) = artist.get_linewidth(), artist._unscaled_dash_pattern
    elif isinstance(artist, LineCollection):
        width, (_, seq) = max(artist.get_linewidths()), artist._us_linestyles[0]
    else:
        return None
    if width <= 0:
        return None
    seq = None if not seq else tuple(round(float(v), 6) for v in seq)
    return {"width": float(width), "pattern": seq, "class": dash_class(seq)}


def drawn(a):
    """Whether an artist puts anything on the page: shown, not fully transparent, and with a stroke, a fill,
    markers, or text that can be seen."""
    if not a.get_visible() or (a.get_alpha() is not None and a.get_alpha() <= 0):
        return False
    if isinstance(a, Line2D):
        stroke = stroke_of(a) is not None and to_rgba(a.get_color())[3] > 0
        marks = marker_of(a) is not None and a.get_markersize() > 0 and (
            to_rgba(a.get_markerfacecolor())[3] > 0
            or (a.get_markeredgewidth() > 0 and to_rgba(a.get_markeredgecolor())[3] > 0))
        return bool(stroke or marks)
    if isinstance(a, Collection):
        faces = any(c[3] > 0 for c in a.get_facecolors())
        edges = any(w > 0 for w in a.get_linewidths()) and any(c[3] > 0 for c in a.get_edgecolors())
        sized = not isinstance(a, PathCollection) or any(s > 0 for s in a.get_sizes())
        return bool((faces or edges) and sized)
    if isinstance(a, Patch):
        return bool((a.get_fill() and a.get_facecolor()[3] > 0)
                    or (a.get_linewidth() > 0 and a.get_edgecolor()[3] > 0))
    if isinstance(a, Text):
        return bool(a.get_text()) and to_rgba(a.get_color())[3] > 0 and a.get_fontsize() > 0
    return True


def placed_by(transform, ax):
    """The coordinates a transform places an artist in."""
    for name, t in (("data", ax.transData), ("x data, y axes", ax.get_xaxis_transform()),
                    ("x axes, y data", ax.get_yaxis_transform()), ("axes", ax.transAxes)):
        if transform is t or transform == t:
            return name
    return "points" if isinstance(transform, transforms.IdentityTransform) else "other"


def clipped_by(a, ax):
    """How an artist is clipped: to the axes, to something else, or not at all."""
    if not a.get_clip_on():
        return "off"
    box, path = a.get_clip_box(), a.get_clip_path()
    box = "none" if box is None else ("axes" if np.allclose(box.bounds, ax.bbox.bounds) else "other")
    path = "none" if path is None else ("axes" if getattr(path, "_patch", None) is ax.patch else "other")
    return f"box {box}, path {path}"


def shown(a, ax, transform):
    return {"drawn": drawn(a), "placed": placed_by(transform, ax), "clip": clipped_by(a, ax)}


def line_record(line, ax):
    stroke = stroke_of(line)
    return {"x": arr(line.get_xdata()), "y": arr(line.get_ydata()), **shown(line, ax, line.get_transform()),
            "linestyle": line.get_linestyle(), "dashes": None if stroke is None else stroke["class"],
            "marker": marker_of(line), "fill": fill_of(line), "label": line.get_label()}


def collection_record(c, ax):
    offsets, placed = arr(c.get_offsets()).reshape(-1, 2), placed_by(c.get_offset_transform(), ax)
    if len(offsets) and placed != "points":  # where the offsets land, in data coordinates, after their transform
        offsets = arr((c.get_offset_transform() - ax.transData).transform(offsets))
    rec = {"type": type(c).__name__, **shown(c, ax, c.get_transform()), "offsets": offsets, "offsets_placed": placed}
    if isinstance(c, LineCollection):
        stroke = stroke_of(c)
        rec.update(segments=[arr(s) for s in c.get_segments()], dashes=None if stroke is None else stroke["class"])
    else:
        rec.update(paths=[arr(p.vertices) for p in c.get_paths()],
                   codes=[None if p.codes is None else arr(p.codes) for p in c.get_paths()])
    return rec


def patch_record(p, ax):
    path = p.get_path()
    rec = {"type": type(p).__name__, **shown(p, ax, p.get_data_transform()),
           "codes": None if path.codes is None else arr(path.codes)}
    if isinstance(p, Rectangle):
        rec.update(xy=arr(p.get_xy()), width=float(p.get_width()), height=float(p.get_height()),
                   filled=bool(p.get_fill()))
    else:
        rec["path"] = arr(p.get_patch_transform().transform(path.vertices))
    return rec


def text_record(t, ax):
    return {"text": t.get_text(), "position": arr(t.get_position()), "ha": t.get_ha(), "va": t.get_va(),
            **shown(t, ax, t.get_transform())}


def visible_ticks(axis, lo, hi):
    lo, hi = min(lo, hi), max(lo, hi)
    pairs = zip(axis.get_majorticklocs(), axis.get_majorticklabels())
    return [(float(v), lab.get_text(), drawn(lab)) for v, lab in pairs if lo - 1e-9 <= v <= hi + 1e-9]


def handle_record(h):
    rec = {"type": type(h).__name__, "drawn": drawn(h)}
    if isinstance(h, Line2D):
        stroke = stroke_of(h)
        rec.update(linestyle=h.get_linestyle(), dashes=None if stroke is None else stroke["class"],
                   marker=marker_of(h), fill=fill_of(h))
    return rec


def width_ranks(ax):
    """The order of the panel's stroke widths (lines, error bars, and key samples), as ranks: which strokes
    are thicker or thinner than which, and which are the same."""
    widths = {}
    leg = ax.get_legend()
    for kind, artists in (("lines", ax.get_lines()), ("collections", ax.collections),
                          ("legend", leg.legend_handles if leg else [])):
        for j, a in enumerate(artists):
            stroke = stroke_of(a)
            if stroke is not None and drawn(a):
                widths[f"{kind}[{j}]"] = round(stroke["width"], 6)
    levels = sorted(set(widths.values()))
    return {k: levels.index(w) for k, w in widths.items()}


def capture(fig):
    """Everything a figure plots, panel by panel, without colors or sizes, which the site style changes."""
    panels, patterns = [], {}
    for ax in fig.axes:
        leg = ax.get_legend()
        titles = {"left": ax._left_title, "center": ax.title, "right": ax._right_title}
        rec = {
            "lines": [line_record(ln, ax) for ln in ax.get_lines()],
            "collections": [collection_record(c, ax) for c in ax.collections],
            "patches": [patch_record(p, ax) for p in ax.patches],
            "texts": [text_record(t, ax) for t in ax.texts],
            "titles": [(t.get_text(), drawn(t)) for t in titles.values()],
            "xlabel": (ax.get_xlabel(), drawn(ax.xaxis.label)), "ylabel": (ax.get_ylabel(), drawn(ax.yaxis.label)),
            "xscale": ax.get_xscale(), "yscale": ax.get_yscale(),
            "xlim": arr(ax.get_xlim()), "ylim": arr(ax.get_ylim()),
            "aspect": str(ax.get_aspect()), "axis_shown": bool(ax.axison),
            "width_ranks": width_ranks(ax),
        }
        if ax.axison:
            rec["xticks"] = visible_ticks(ax.xaxis, *ax.get_xlim())
            rec["yticks"] = visible_ticks(ax.yaxis, *ax.get_ylim())
        rec["legend"] = None if leg is None else {
            "drawn": bool(leg.get_visible() and leg.handlelength > 0),
            "texts": [(t.get_text(), drawn(t)) for t in leg.get_texts()],
            "handles": [handle_record(h) for h in leg.legend_handles]}
        for a in [*ax.get_lines(), *ax.collections, *(leg.legend_handles if leg else [])]:
            stroke = stroke_of(a)
            if stroke is not None:
                patterns.setdefault(stroke["class"], set()).add(stroke["pattern"])
        panels.append(rec)
    return {"panels": panels, "figure_legends": [[t.get_text() for t in lg.get_texts()] for lg in fig.legends],
            "figure_texts": [t.get_text() for t in fig.texts],
            # each dash style (dotted, dashed) uses one on/off sequence throughout the figure
            "one_pattern_per_dash_style": all(len(v) == 1 for v in patterns.values())}


def identity_color(a, background):
    """The color that tells a series apart: a line's stroke, or its markers' fill (their edge when they are
    open); a collection's or patch's fill (its edge when unfilled); laid over the background where
    translucent."""
    if isinstance(a, Line2D):
        if stroke_of(a) is not None:
            c = to_rgba(a.get_color(), a.get_alpha())
        elif fill_of(a) == "filled":
            c = to_rgba(a.get_markerfacecolor(), a.get_alpha())
        else:
            c = to_rgba(a.get_markeredgecolor(), a.get_alpha())
    elif isinstance(a, LineCollection):
        c = tuple(a.get_edgecolors()[0])
    elif isinstance(a, Collection):
        faces = a.get_facecolors()
        c = tuple(faces[0]) if len(faces) and faces[0][3] > 0 else tuple(a.get_edgecolors()[0])
    else:
        c = a.get_facecolor() if a.get_fill() and a.get_facecolor()[3] > 0 else a.get_edgecolor()
    return to_hex(c[3] * np.array(c[:3]) + (1 - c[3]) * np.array(to_rgb(background)))


def color_groups(fig, background):
    """Which drawn series (lines, collections, patches, and key samples) share a color: a list of groups."""
    colors = {}
    for i, ax in enumerate(fig.axes):
        leg = ax.get_legend()
        for kind, artists in (("lines", ax.get_lines()), ("collections", ax.collections),
                              ("patches", ax.patches), ("legend", leg.legend_handles if leg else [])):
            for j, a in enumerate(artists):
                if drawn(a):
                    colors[f"panels[{i}].{kind}[{j}]"] = identity_color(a, background)
    groups = {}
    for where, color in colors.items():
        groups.setdefault(color, []).append(where)
    return sorted(sorted(g) for g in groups.values())


def documented_colors(pr):
    """The color each series and key entry should have in a render, by its label, from COLORS and the page
    inks."""
    table = {label: pr.color[key] for key, (label, _, _) in paper.STYLE.items()}
    table.update({"SFace, saved original results": pr.color["sface"],
                  "SFace, DeepFace path re-created": pr.color["sface"],
                  "SFace, same crop for probe and gallery": pr.color["sface_same_crop"],
                  "95% interval": pr.ink, "one seed per config.": pr.rubric})
    table.update({f"Custom model, N = {n}": pr.step[n] for n in SAMPLE_STEPS})
    table.update({f"{eff.capitalize()} main effect": pr.color[key] for eff, key in EFFECT_COLOR.items()})
    table.update({label: pr.ink2 for label in ("Raw pixels (same sample)", "Face found by the detector",
                                               "solid: same image", "dotted: identity", "98.3% interval")})
    return table


def series_colors(fig, pr):
    """Every labeled series and key entry against its documented color; returns the mismatches."""
    table, wrong = documented_colors(pr), []
    for i, ax in enumerate(fig.axes):
        named = [(f"panels[{i}].lines[{j}]", ln.get_label(), ln) for j, ln in enumerate(ax.get_lines())
                 if not ln.get_label().startswith("_")]
        leg = ax.get_legend()
        if leg:
            named += [(f"panels[{i}].legend[{j}]", t.get_text(), h)
                      for j, (t, h) in enumerate(zip(leg.get_texts(), leg.legend_handles))]
        for where, label, a in named:
            got = identity_color(a, pr.mount)
            if label not in table:
                wrong.append(f"{where}: no documented color for {label!r}")
            elif got != to_hex(table[label]):
                wrong.append(f"{where} ({label}): {got}, documented {to_hex(table[label])}")
    return wrong


def differences(a, b, where=""):
    """Every place two captured records differ, as (where, paper's value, site's value); arrays must match
    exactly, value for value."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a, float), np.asarray(b, float)
        same = a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
        return [] if same else [(where, a, b)]
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return [(where, sorted(a), sorted(b))]
        return [d for k in a for d in differences(a[k], b[k], f"{where}.{k}")]
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [(where, f"{len(a)} items", f"{len(b)} items")]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in differences(x, y, f"{where}[{i}]")]
    return [] if a == b else [(where, a, b)]


def leaves(a):
    """How many values a captured record holds (an array counts once)."""
    if isinstance(a, dict):
        return sum(leaves(v) for v in a.values())
    if isinstance(a, (list, tuple)):
        return sum(leaves(v) for v in a)
    return 1


def show(v):
    return repr(v.tolist() if isinstance(v, np.ndarray) else v)


def same_value(a, b):
    if isinstance(a, (np.ndarray, list, tuple)) or isinstance(b, (np.ndarray, list, tuple)):
        a, b = np.asarray(a, float), np.asarray(b, float)
        return a.shape == b.shape and np.array_equal(a, b)
    return a == b


def regroup(groups, entry):
    """The paper's color groups with an accepted move applied: the listed artists leave the group of
    entry["old"] and join the group of entry["new"]."""
    movers = set(entry["move"])
    out = []
    for g in groups:
        kept = [m for m in g if m not in movers]
        if entry["new"] in kept:
            kept += sorted(movers)
        if kept:
            out.append(sorted(kept))
    return sorted(out)


def check(name, looks):
    """Compares the paper's figure with the site figure in each printing and layout; returns the manifest
    record."""
    read_paper, read_site = set(), set()
    with files_read(read_paper):
        fig = build_paper(name)
    ref, ref_groups = capture(fig), color_groups(fig, "#ffffff")
    plt.close(fig)
    entries = [e for e in ACCEPTED if e["where"].startswith(f"{name}.")]
    found, used = [], {}
    for pr in looks:
        with plt.rc_context(style(pr)), files_read(read_site):
            fig = SITE[name](pr)
            got, groups, wrong = capture(fig), color_groups(fig, pr.mount), series_colors(fig, pr)
            plt.close(fig)
        mine = [e for e in entries if e.get("layout", pr.layout) == pr.layout]
        for where, old, new in differences(ref, got, name):
            entry = next((e for e in mine if e["where"] == where and "move" not in e
                          and same_value(e["old"], old) and same_value(e["new"], new)), None)
            if entry:
                used.setdefault(id(entry), (entry, []))[1].append(pr.tag)
            else:
                found.append(f"{pr.tag}: {where}: {show(old)} vs {show(new)}")
        expected = ref_groups
        for entry in [e for e in mine if "move" in e]:
            if any(set(entry["move"]) | {entry["old"]} <= set(g) for g in expected):
                expected = regroup(expected, entry)
                used.setdefault(id(entry), (entry, []))[1].append(pr.tag)
        if groups != expected:
            lost = [g for g in expected if g not in groups]
            found.append(f"{pr.tag}: {name}.color_groups: the paper's {lost[:2]} are not kept")
        found += [f"{pr.tag}: {name}.{w}" for w in wrong]
    for entry in entries:
        layouts = {entry["layout"]} if "layout" in entry else set(LAYOUTS)
        expected_tags = {pr.tag for pr in looks if pr.layout in layouts}
        seen = set(used.get(id(entry), (entry, []))[1])
        if seen != expected_tags:
            found.append(f"accepted difference not found in {sorted(expected_tags - seen)}: {entry['where']}")
    if read_paper != read_site:
        found.append(f"files read differ: {sorted(read_paper ^ read_site)}")
    if not ref["one_pattern_per_dash_style"]:
        found.append("the paper's figure mixes dash sequences within a style")
    panels = ref["panels"]
    return {
        "source_script": "paper/make_figures.py",
        "source_function": name,
        "data_files": sorted(read_paper),
        "panels": len(panels),
        "series_compared": sum(len(p["lines"]) + len(p["collections"]) + len(p["patches"]) for p in panels),
        "lines": sum(len(p["lines"]) for p in panels),
        "collections": sum(len(p["collections"]) for p in panels),
        "patches": sum(len(p["patches"]) for p in panels),
        "text_strings": sum(len(p["texts"]) + len([t for t in p["titles"] if t[0]]) + bool(p["xlabel"][0])
                            + bool(p["ylabel"][0]) + len(p.get("xticks", [])) + len(p.get("yticks", []))
                            + (len(p["legend"]["texts"]) if p["legend"] else 0) for p in panels),
        "values_compared": leaves(ref),
        "color_groups": len(ref_groups),
        "renders_compared": [pr.tag for pr in looks],
        "result": "pass" if not found else "fail",
        "differences": found,
        "accepted_differences": [
            {"where": e["where"], **({"layout": e["layout"]} if "layout" in e else {}),
             **({"move": e["move"], "from_group_of": e["old"], "to_group_of": e["new"]} if "move" in e
                else {"paper": e["old"], "site": e["new"]}),
             "reason": e["reason"], "renders": tags} for e, tags in used.values()],
    }


def normalized(svg):
    """An SVG's text with its generated element IDs (clip paths, glyphs, and the like) renamed in order of
    appearance, so two drawings of the same figure compare equal whatever IDs they were given."""
    ids = {}
    for m in re.finditer(r'\bid="([^"]+)"', svg):
        ids.setdefault(m.group(1), f"id{len(ids)}")
    return re.sub(r'(\bid="|url\(#|href="#)([^")]+)', lambda m: m.group(1) + ids.get(m.group(2), m.group(2)), svg)


def shipped(name, looks):
    """Draws every render of a figure again in memory and compares it with the file on disk (SVGs after
    renaming their generated IDs); returns, per render, "matches" or what is wrong."""
    result = {}
    for pr in looks:
        data, suffix, _ = render(name, pr)
        path = OUT / f"{name}.{pr.tag}{suffix}"
        if not path.exists():
            result[pr.tag] = f"{path.name} is missing"
            continue
        disk = path.read_bytes()
        same = normalized(disk.decode()) == normalized(data.decode()) if suffix == ".svg" else disk == data
        result[pr.tag] = "matches" if same else f"{path.name} differs from a fresh drawing"
    return result


# ----------------------------------------------------------------------------------------------------

def color_table(printings):
    """The color mapping for the manifest, with each color's contrast against its mount."""
    labels = {k: v[0] for k, v in paper.STYLE.items()}
    labels["sface_same_crop"] = "SFace, same crop for probe and gallery"
    table = {}
    for key in COLORS:
        row = {"label": labels[key], "marker": paper.STYLE.get(key, (None, None, "D"))[2]}
        for pr in printings:
            row[pr.name] = pr.color[key]
            row[f"contrast_{pr.name}"] = round(contrast(pr.color[key], pr.mount), 2)
        table[key] = row
    steps = {}
    for n in SAMPLE_STEPS:
        steps[f"N = {n}"] = {pr.name: pr.step[n] for pr in printings}
        steps[f"N = {n}"].update({f"contrast_{pr.name}": round(contrast(pr.step[n], pr.mount), 2) for pr in printings})
    roles = {}
    for pr in printings:
        for role, value in (("ink", pr.ink), ("ink2", pr.ink2), ("rubric", pr.rubric)):
            roles.setdefault(role, {})[pr.name] = value
            roles[role][f"contrast_{pr.name}"] = round(contrast(value, pr.mount), 2)
    roles["ink"]["used_for"] = "titles, axis labels, key text, the 95% intervals in fig_contrasts, CNN neither"
    roles["ink2"]["used_for"] = ("tick labels, axes, the keys for line styles, raw pixels (same sample), face found "
                                 "by the detector, the 98.3% intervals, zero lines, notes")
    roles["rubric"]["used_for"] = ("ArcFace R50, the primary-contrast estimates and their one-seed ranges, the "
                                   "template landmarks")
    return table, steps, roles


def main():
    ap = argparse.ArgumentParser(
        description="Draw the chapter's web figures in the course book's style, and check them against the paper's.")
    ap.add_argument("--check", action="store_true",
                    help="only compare the figures with the paper's and update the manifest")
    ap.add_argument("--fonts", default=str(FONTS), help="folder with Old Standard's WOFF2 files (default: docs/assets/fonts)")
    args = ap.parse_args()
    printings = [Look("light"), Look("dark")]
    looks = [Look(name, layout) for layout in LAYOUTS for name in ("light", "dark")]
    for pr in printings:
        for key, value in list(pr.color.items()) + list(pr.step.items()):
            assert contrast(value, pr.mount) >= 3.0, f"{key} in the {pr.name} printing is under 3:1 on its mount"

    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    with tempfile.TemporaryDirectory() as scratch:
        load_fonts(args.fonts, scratch)
        figures = manifest.get("figures", {})
        for name in FIGURES:
            entry = figures.get(name, {})
            if not args.check:
                entry["files"] = {}
                for pr in looks:
                    file, size, (width, height) = draw(name, pr)
                    entry["files"][pr.tag] = {"file": file, "bytes": size, "width_pt": width, "height_pt": height}
            entry["check"] = check(name, looks)
            entry["check"]["shipped_files"] = shipped(name, looks)
            for tag, state in entry["check"]["shipped_files"].items():
                if state != "matches":
                    entry["check"]["differences"].append(f"{tag}: {state}")
                    entry["check"]["result"] = "fail"
            figures[name] = entry
    expected = {"manifest.json"} | {f"{n}.{pr.tag}{s}" for n in FIGURES for pr in looks for s in (".svg", ".webp")}
    unexpected = sorted(p.name for p in OUT.iterdir() if p.name not in expected)

    table, steps, roles = color_table(printings)
    manifest = {
        "about": ("The chapter's five figures in the course book's style, one file per printing and layout. "
                  "Show <name>.light.svg in the day printing and <name>.dark.svg in the night printing, on the "
                  "plate mount; on narrow screens (about 360 CSS px), show <name>.phone.light.svg and "
                  "<name>.phone.dark.svg, which stack the same panels one per row. Each has a transparent "
                  "background and its text drawn as paths."),
        "generator": "chapter/make_site_figures.py",
        "source_script": "paper/make_figures.py",
        "font": "Old Standard TT (SIL Open Font License), from the site's docs/assets/fonts",
        "printings": {pr.name: {"mount": pr.mount, "ink": pr.ink, "ink2": pr.ink2, "rubric": pr.rubric,
                                "occluder_crop": pr.crop, "occluder_ink": OCCLUDER_INK} for pr in printings},
        "layouts": {"wide": "panels side by side, drawn 7.1 in wide for about 600 to 1,100 CSS px",
                    "phone": "panels stacked one per row, drawn 3.75 in wide for about 360 CSS px"},
        "colors": table,
        "original_model_sample_steps": steps,
        "page_inks": roles,
        "effect_colors": {k: f"same as {v}" for k, v in EFFECT_COLOR.items()},
        "figures": figures,
        "unexpected_files": unexpected,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"{'figure':15s} {'result':6s} {'panels':>6s} {'series':>6s} {'texts':>6s} {'values':>6s} {'colors':>6s}"
          f"  {'accepted':>8s}  files on disk")
    for name in FIGURES:
        c = figures[name]["check"]
        files = ", ".join(f"{k} {v['bytes'] / 1000:.0f} KB {c['shipped_files'].get(k, '?')}"
                          for k, v in figures[name].get("files", {}).items())
        print(f"{name:15s} {c['result']:6s} {c['panels']:6d} {c['series_compared']:6d} {c['text_strings']:6d}"
              f" {c['values_compared']:6d} {c['color_groups']:6d}  {len(c['accepted_differences']):8d}  {files}")
        for d in c["differences"]:
            print("    differs:", d)
    if unexpected:
        print("unexpected files in the output folder:", ", ".join(unexpected))
    if unexpected or any(figures[n]["check"]["result"] != "pass" for n in FIGURES):
        sys.exit(1)


if __name__ == "__main__":
    main()
