# make_numbers.py
# Writes the numbers and tables used by main.tex from the committed result files, so that the
# numbers in the paper are not typed by hand. The few typed values are in TYPED below, and the
# paper's "Reproducing the results" section lists them.
#
# Inputs (relative to the project folder):
#   results/original/*.csv              the original Colab runs' accuracy files
#   analysis/out/*.csv                  the CPU checks of those runs
#   results/study/                      the identification study: summary.csv, analysis/*.csv,
#                                       eval/*/attention_check.csv, train/*/train_log.csv,
#                                       run_log.json, run.log
#   study/config.json                   the study's settings
#   frfpv1/models.py                    the network (parameter counts)
#   frfpv1/config.py                    the original model's training settings
#
# Outputs (in paper/): numbers.tex and the table_*.tex files.
#
# Usage (from the project folder or from paper/):
#     python paper/make_numbers.py

import csv
import importlib.util
import json
import re
import statistics
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP, Decimal
from pathlib import Path

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
ORIG = ROOT / "results/original"
CHECKS = ROOT / "analysis/out"
STUDY = ROOT / "results/study"

# Typed: one estimate. (The original model's training settings are read from frfpv1/config.py,
# which keeps the values of the original Colab training run.)
TYPED = {
    "orig_overlap_share": "30",     # about 1 in 30: 1,000 training images of 30,000
}


def frfpv1_settings():
    """The original pipeline's training settings, read from frfpv1/config.py as text (importing it
    would need the Colab environment's packages)."""
    text = (ROOT / "frfpv1/config.py").read_text()

    def value(name):
        return re.search(rf"^{name}\s*=\s*([^#\n]+)", text, flags=re.M).group(1).strip()
    lo, hi = (float(v) for v in value("OCCLUSION_TRAINING_RANGE").strip("()").split(","))
    return {"train_images": int(value("TRAIN_SUBSET_SIZE")), "epochs": int(value("NUM_EPOCHS")),
            "batch": int(value("TRAIN_BATCH_SIZE")), "lr": float(value("LEARNING_RATE")),
            "margin": float(value("TRIPLET_MARGIN")), "occ_lo": lo, "occ_hi": hi}

# Short names used in the keys.
MODELS = {
    "arc": "arcface_r50", "sf": "sface",
    "nn": "cosface_noatt_noaug", "at": "cosface_att_noaug", "au": "cosface_noatt_aug", "both": "cosface_att_aug",
    "tri": "triplet_att_aug", "px": "pixels_crop", "orig": "original_model", "pxf": "pixels_frame",
}
SQUARES = {"s00": "square_0.00", "s05": "square_0.05", "s10": "square_0.10", "s15": "square_0.15",
           "s20": "square_0.20", "s25": "square_0.25", "s30": "square_0.30", "s40": "square_0.40",
           "s50": "square_0.50", "s60": "square_0.60"}
OTHER = {"eye": "eyes_band", "low": "lower_band", "off20": "random_square_0.20", "gray20": "gray_square_0.20",
         "noise20": "noise_square_0.20", "gray40": "gray_square_0.40", "noise40": "noise_square_0.40"}
CONDS = {**SQUARES, **OTHER}
FRAME_LEVELS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
FRAMES = {f"f{l:02d}" if l < 100 else "f100": f"frame_square_{l / 100:.2f}" for l in FRAME_LEVELS}
LABEL = {"arc": "ArcFace R50", "sf": "SFace", "nn": "CNN, neither", "at": "CNN, attention",
         "au": "CNN, augmentation", "both": "CNN, both", "tri": "CNN, both, triplet", "px": "Crop pixels",
         "orig": "Original model", "pxf": "Frame pixels"}
SIMPLE = {"attA": "attention, with augmentation", "attN": "attention, without augmentation",
          "augA": "augmentation, with attention", "augN": "augmentation, without attention",
          "triC": "CosFace minus batch-hard triplet"}
ORIG_RUNS = {"50": "exper__size_50__20251201_192436", "100": "exper__size_100__20251130_235656",
             "200": "exper__size_200__20251201_203249", "500": "exper__size_500__20251201_004627",
             "nobg100": "exper__NO_BG__size_100__20251203_012434",
             "nobg50a": "exper__NO_BG__size_50__20251202_224921", "nobg50b": "exper__NO_BG__size_50__20251203_001017"}
PRETRAINED_ORIG = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "ArcFace", "SFace"]


def dec(x, places):
    q = Decimal(1).scaleb(-places)
    return str(Decimal(repr(round(float(x), 10))).quantize(q, rounding=ROUND_HALF_UP))


def pct(x, places=1):
    return dec(100 * float(x), places)


def signed(s):
    """'+1.0' or '$-$1.0' from a formatted number; zero keeps a sign only if the value had one."""
    if s.startswith("-"):
        return "$-$" + s[1:]
    return "+" + s


def spct(x, places=1):
    """Signed percentage points. A value that rounds to zero keeps its sign (-0.0 or +0.0); an exact zero has none."""
    s = dec(100 * float(x), places)
    if float(x) == 0:
        return s
    if float(x) < 0 and not s.startswith("-"):
        s = "-" + s
    return signed(s)


def fmt_orig(run, x):
    """Accuracy in % on an original run: whole numbers for N up to 200 (N = 200 has halves), one decimal for N = 500."""
    s = pct(x, 1)
    return s if run == "500" else s.removesuffix(".0")


def num(n):
    return f"{int(n):,}"


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def level_key(x):
    return f"{int(round(float(x) * 100)):02d}" if round(float(x) * 100) < 100 else "100"


def t_two_sided(t, df):
    """Two-sided p of Student's t with df degrees of freedom (df need not be whole), by the regularized
    incomplete beta function: p = I_x(df/2, 1/2) with x = df / (df + t^2)."""
    import math
    a, b, x = df / 2, 0.5, df / (df + t * t)
    if x >= 1:
        return 1.0
    if x <= 0:
        return 0.0

    def cf(a, b, x):  # continued fraction for the incomplete beta (Lentz's method)
        tiny, c, d = 1e-300, 1.0, 1 - (a + b) * x / (a + 1)
        d = 1 / (d if abs(d) > tiny else tiny)
        h = d
        for m in range(1, 300):
            for num_ in (m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)), -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))):
                d = 1 + num_ * d
                d = 1 / (d if abs(d) > tiny else tiny)
                c = 1 + num_ / c
                c = c if abs(c) > tiny else tiny
                h *= d * c
            if abs(d * c - 1) < 1e-15:
                break
        return h

    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1 - x) - lbeta) / a
    return front * cf(a, b, x) if x < (a + 1) / (a + b + 2) else 1 - math.exp(b * math.log(1 - x) + a * math.log(x) - lbeta) / b * cf(b, a, 1 - x)


def primary_definitions():
    """P1 to P3 as study/analyze.py defines them (condition lists and configuration pairs), read without
    importing it."""
    import ast
    tree = ast.parse((ROOT / "study/analyze.py").read_text())
    env = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ("SQUARES_TRAINED", "UNSEEN", "AUG_MAIN", "ATT_MAIN", "PRIMARY"):
                env[name] = eval(compile(ast.Expression(node.value), "analyze.py", "eval"), {}, dict(env))
    return env["PRIMARY"]


def compute():
    N = dict(TYPED)
    cfg = json.load(open(ROOT / "study/config.json"))
    log = json.load(open(STUDY / "run_log.json"))
    tc = cfg["train"]

    # ---- settings (config.json and run_log.json) -------------------------------------------
    N["seed"] = str(cfg["seed"])
    N["crop"] = str(cfg["crop_size"])
    N["frame"] = str(cfg["frame_size"])
    N["frame_small"] = str(cfg["frame_size"] // 2)
    N["yunet_thr"] = str(cfg["align"]["score_threshold"])
    N["epochs"] = str(tc["epochs"])
    N["batch"] = str(tc["batch_size"])
    N["lr"] = f"{tc['lr']:g}"
    N["wd"] = dec(tc["weight_decay"], 4)
    N["cos_s"] = dec(tc["cosface_scale"], 0)
    N["cos_m"] = dec(tc["cosface_margin"], 2)
    N["tri_m"] = dec(tc["triplet_margin"], 1)
    N["tri_p"] = str(tc["triplet_identities_per_batch"])
    N["tri_k"] = str(tc["triplet_images_per_identity"])
    N["occ_lo"] = pct(tc["occlusion_range"][0], 0)
    N["occ_hi"] = pct(tc["occlusion_range"][1], 0)
    N["occ_prob"] = pct(tc["occlusion_probability"], 0)
    fo = frfpv1_settings()
    N["orig_train_images"] = num(fo["train_images"])
    N["orig_epochs"] = str(fo["epochs"])
    N["orig_batch"] = str(fo["batch"])
    N["orig_lr"] = f"{fo['lr']:.4f}"
    N["orig_margin"] = dec(fo["margin"], 1)
    N["orig_occ_lo"] = pct(fo["occ_lo"], 0)
    N["orig_occ_hi"] = pct(fo["occ_hi"], 0)
    # the saved model's SHA-256, as the study config and the GPU run both record it; printed in four
    # groups of 16 so it can break across lines
    sha = cfg["original_model"]["sha256"]
    assert sha == log["weights_sha256"]["model.pth"], "model.pth hash differs between config and run log"
    N["orig_sha256"] = "\\allowbreak ".join(sha[i:i + 16] for i in range(0, len(sha), 16))
    N["emb"] = str(tc["embedding_size"])
    N["n_seeds"] = str(len(tc["seeds"]))
    N["att_probes"] = str(cfg["attention_probes"])
    N["label_floor"] = dec(cfg["label_check_min_rank1"], 1)
    c = cfg["conditions"]
    N["eye_r0"], N["eye_r1"] = pct(c["eyes_band_rows"][0], 0), pct(c["eyes_band_rows"][1], 0)
    N["low_r0"], N["low_r1"] = pct(c["lower_band_rows"][0], 0), pct(c["lower_band_rows"][1], 0)
    N["eye_area"] = pct(c["eyes_band_rows"][1] - c["eyes_band_rows"][0], 0)
    N["low_area"] = pct(c["lower_band_rows"][1] - c["lower_band_rows"][0], 0)
    N["n_square_levels"] = str(len(c["square_coverages"]))
    N["n_conditions"] = str(len(c["square_coverages"]) + 3 + len(c["fill_variants"]))
    N["n_occluded"] = str(int(N["n_conditions"]) - 1)  # every condition but the clean one
    N["n_occluded"] = str(len(c["square_coverages"]) - 1 + 3 + len(c["fill_variants"]))
    N["n_trained_squares"] = str(sum(0.1 <= v <= 0.5 for v in c["square_coverages"]))
    N["n_unseen"] = str(3 + len(c["fill_variants"]))
    N["n_p3"] = str(int(N["n_trained_squares"]) + int(N["n_unseen"]))
    N["far1"], N["far01"] = pct(cfg["far_levels"][0], 0), pct(cfg["far_levels"][1], 1)
    # square side in pixels on the crop, as study/data.py square_box draws it
    N["side_s20"] = str(int((0.2 * cfg["crop_size"] ** 2) ** 0.5))
    env = log["environment"]
    N["py"], N["torch_v"] = env["python"], env["torch"].split("+")[0]
    N["cuda"], N["cv2_v"], N["pd_v"], N["np_v"] = env["cuda"], env["cv2"], env["pandas"], env["numpy"]
    N["pil_v"] = env["PIL"]
    N["threads"] = str(env["cpu_count"])
    cnt = log["counts"]
    for k in ("aligned", "not_aligned", "test_identities", "gallery", "probes", "train_identities", "train_images"):
        N[f"n_{k}"] = num(cnt[k])
    N["probes_per_id"] = dec(cnt["probes"] / cnt["test_identities"], 1)
    N["train_per_id"] = dec(cnt["train_images"] / cnt["train_identities"], 1)
    N["chance"] = pct(1 / cnt["gallery"], 1)
    N["n_images_total"] = num(cnt["aligned"] + cnt["not_aligned"])
    lc = log["label_check"]["clean_rank1"]
    N["lc_sf"], N["lc_arc"] = dec(lc["sface"], 3), dec(lc["arcface_r50"], 3)
    fin = log["finishes"]
    N["wall_s1"], N["wall_s2"] = num(round(fin[0]["seconds"])), num(round(fin[1]["seconds"]))
    tot = fin[0]["seconds"] + fin[1]["seconds"]
    N["wall_s"] = num(round(tot))
    N["wall_h"], N["wall_m"] = str(int(tot // 3600)), str(int(round((tot % 3600) / 60)))
    N["units"] = str(fin[1]["planned"])
    N["units1"], N["units2"] = str(fin[0]["ran"]), str(fin[1]["ran"])
    N["n_runs"] = str(len(cfg["runs"]) * len(tc["seeds"]))

    # run.log: the image map and the label-fit check
    rl = (STUDY / "run.log").read_text().splitlines()
    m = re.search(r"min ([\d.]+), 1st percentile ([\d.]+), median ([\d.]+)", "\n".join(rl))
    N["map_min"], N["map_p1"], N["map_med"] = m.groups()
    N["map_near"] = re.search(r"(\d+) images below 0.02", "\n".join(rl)).group(1)
    N["map_second"] = re.search(r"(\d+) images took a candidate other than their best", "\n".join(rl)).group(1)
    for line in rl:
        f = line.split()
        if line.startswith("file numbers"):
            N["fit_file_gen"], N["fit_file_imp"], N["fit_file_r1"] = f[2], f[3], f[4]
        if line.startswith("image_map.csv"):
            N["fit_map_gen"], N["fit_map_imp"], N["fit_map_r1"], N["fit_map_top"] = f[1], f[2], f[3], f[4]
            N["fit_map_pairs"] = num(f[5])
        if "embedding" in line and "aligned crops" in line:
            N["fit_n"] = num(re.search(r"embedding (\d+) aligned", line).group(1))
    m = re.search(r"labels: (\d+) images, (\d+) identities, (\d+) lines skipped; roles: \{'train': (\d+), 'probe': (\d+), 'unused': (\d+), 'gallery': (\d+)\}", "\n".join(rl))
    N["n_label_images"], N["n_identities"] = num(m.group(1)), num(m.group(2))
    N["n_skipped"] = m.group(3)
    N["role_train"], N["role_probe"], N["role_unused"] = num(m.group(4)), num(m.group(5)), num(m.group(6))

    # versions of the original Colab runs, from the top-level requirements.txt
    req = (ROOT / "requirements.txt").read_text()
    for pkg, key in (("deepface", "deepface_v"), ("tensorflow", "tf_v"), ("rembg", "rembg_v"), ("onnxruntime", "ort_v")):
        N[key] = re.search(rf"^{pkg}==([\d.]+)", req, re.M).group(1)

    # ---- the network ------------------------------------------------------------------------
    try:
        spec = importlib.util.spec_from_file_location("frfpv1_models", ROOT / "frfpv1/models.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        net = mod.OcclusionAwareFaceModel(128, attention=True)
        N["params"] = num(sum(p.numel() for p in net.parameters()))
        N["params_att"] = num(sum(p.numel() for p in net.attention.parameters()))
        N["params_noatt"] = num(sum(p.numel() for p in mod.OcclusionAwareFaceModel(128, attention=False).parameters()))
        N["params_m"] = dec(sum(p.numel() for p in net.parameters()) / 1e6, 1)
    except ImportError:  # without PyTorch: the counts printed by this script with torch 2.9.1
        N.update({"params": "1,621,696", "params_att": "3,136", "params_noatt": "1,618,560", "params_m": "1.6"})

    # ---- original runs ----------------------------------------------------------------------
    orig = {k: read(ORIG / f"{v}.csv") for k, v in ORIG_RUNS.items()}
    ov = {}
    for k, rows in orig.items():
        for r in rows:
            lev = level_key(r["Occlusion"])
            for col, val in r.items():
                if col == "Occlusion":
                    continue
                ov[(k, col, lev)] = float(val)
    short = {"Custom": "cu", "VGG-Face": "vgg", "Facenet": "fn", "Facenet512": "fn512", "OpenFace": "of",
             "ArcFace": "af", "SFace": "sfo"}
    for (k, col, lev), v in ov.items():
        N[f"o_{k}_{short[col]}_{lev}"] = fmt_orig(k, v)
    # pretrained models at 5% and 10%, N = 100: their range
    for lev in ("05", "10"):
        vals = [ov[("100", m_, lev)] for m_ in PRETRAINED_ORIG]
        N[f"o_100_pre_{lev}_lo"], N[f"o_100_pre_{lev}_hi"] = pct(min(vals), 0), pct(max(vals), 0)
    rest = [ov[("100", m_, "10")] for m_ in PRETRAINED_ORIG if m_ != "VGG-Face"]
    N["o_100_pre_s20_hi"] = pct(max(ov[("100", m_, "20")] for m_ in PRETRAINED_ORIG), 0)
    N["o_100_pre_s20_lo"] = pct(min(ov[("100", m_, "20")] for m_ in PRETRAINED_ORIG), 0)

    so = read(CHECKS / "summary_original.csv")
    for r in so:
        if r["run"] in ("N50", "N100", "N200", "N500") and r["model"] == "Custom":
            n = r["run"][1:]
            N[f"auc_cu_{n}"] = dec(r["auc"], 3)
            N[f"cross_cu_{n}"] = pct(r["crossing_50pct"], 1)
        if r["run"] in ("N50", "N100", "N200", "N500") and r["model"] == "VGG-Face":
            n = r["run"][1:]
            N[f"auc_vgg_{n}"] = dec(r["auc"], 3)
            N[f"cross_vgg_{n}"] = pct(r["crossing_50pct"], 1)
    for n in ("50", "100", "200", "500"):
        pre = [r for r in so if r["run"] == f"N{n}" and r["model"] in PRETRAINED_ORIG]
        N[f"pre_mean_{n}"] = pct(statistics.mean(float(r["mean_acc_10_to_100pct"]) for r in pre), 2)
        N[f"chance_{n}"] = pct(float(pre[0]["chance"]), 2)

    # pixel baseline and the custom model's lead over it
    pb = read(CHECKS / "check_pixel_baseline.csv")
    colmap = {"50": ORIG_RUNS["50"], "100": ORIG_RUNS["100"], "200": ORIG_RUNS["200"], "500": ORIG_RUNS["500"],
              "nobg100": ORIG_RUNS["nobg100"]}
    pix = {}
    for r in pb:
        lev = level_key(r["Occlusion"])
        for k, col in colmap.items():
            pix[(k, lev)] = float(r[col])
            N[f"px_{k}_{lev}"] = fmt_orig(k, r[col])
    lead = {}
    for k in ("50", "100", "200", "500"):
        for lev in ("05", "10", "15", "20", "25", "30", "35", "40", "50"):
            d = ov[(k, "Custom", lev)] - pix[(k, lev)]
            lead[(k, lev)] = d
            s = fmt_orig(k, d)
            N[f"lead_{k}_{lev}"] = signed(s) if s not in ("0", "0.0") else "0"
            N[f"lead_{k}_{lev}_abs"] = s.lstrip("-")
    N["lead_500_max"] = pct(max(lead[("500", l)] for l in ("05", "10", "15", "20", "25", "30", "35", "40", "50")), 1)

    # the custom model rerun on CPU
    cr = read(CHECKS / "check_custom_rerun.csv")
    for run in ("N50", "N100", "N200", "NOBG100_finetuned"):
        rows = [r for r in cr if r["run"] == run]
        N[f"rerun_{run.lower()}_max"] = dec(max(abs(float(r["diff_points"])) for r in rows), 1)
        N[f"rerun_{run.lower()}_same"] = str(sum(float(r["diff_points"]) == 0 for r in rows))
        N["rerun_levels"] = str(len(rows))
    for r in cr:
        if r["run"] == "NOBG100_master":
            N[f"master_nobg100_{level_key(r['Occlusion'])}"] = pct(r["rerun"], 0)

    # detector check (N = 100)
    for r in read(CHECKS / "check_detector_sface.csv"):
        lev = level_key(r["Occlusion"])
        N[f"det_found_{lev}"] = pct(r["face_detected"], 0)
        N[f"det_path_{lev}"] = pct(r["deepface_path"], 0)
        N[f"det_one_{lev}"] = pct(r["deepface_path_largest_share_to_one_gallery_image"], 0)
        N[f"det_same_{lev}"] = pct(r["same_crop"], 0)
        N[f"det_whole_{lev}"] = pct(r["whole_image"], 0)
        N[f"det_saved_{lev}"] = pct(r["saved_sface"], 0)
    for r in read(CHECKS / "check_detector_face_coverage.csv"):
        lev = level_key(r["Occlusion"])
        N[f"side_{lev}"] = r["square_side_px"]
        N[f"cov_{lev}"] = pct(r["median_share_of_face_box_covered"], 0)
        N["face_box"] = dec(r["median_face_box_side_px"], 0)

    # ---- the identification study -----------------------------------------------------------
    iv = {(r["model"], r["protocol"], r["condition"], r["metric"]): r for r in read(STUDY / "analysis/intervals.csv")}
    for s, model in MODELS.items():
        for cs, cond in {**CONDS, **FRAMES}.items():
            r = iv.get((model, "identity", cond, "rank1"))
            if r is None:
                continue
            N[f"r1_{s}_{cs}"] = pct(r["estimate"])
            N[f"r1_{s}_{cs}_lo"], N[f"r1_{s}_{cs}_hi"] = pct(r["ci_low"]), pct(r["ci_high"])
            for met, tag in (("tar_fixed_0.01", "tarf1"), ("tar_fixed_0.001", "tarf01")):
                t = iv.get((model, "identity", cond, met))
                if t is not None:
                    N[f"{tag}_{s}_{cs}"] = pct(t["estimate"])
                    N[f"{tag}_{s}_{cs}_lo"], N[f"{tag}_{s}_{cs}_hi"] = pct(t["ci_low"]), pct(t["ci_high"])

    summ = read(STUDY / "summary.csv")
    by = {}
    for r in summ:
        base = re.sub(r"_seed\d+$", "", r["model"])
        by.setdefault((base, r["protocol"], r["condition"]), []).append(r)
    inv = {v: k for k, v in MODELS.items()}
    allc = {**CONDS, **FRAMES}
    invc = {v: k for k, v in allc.items()}
    far_max = {}
    for (base, prot, cond), rows in by.items():
        if base not in inv or cond not in invc or prot != "identity":
            continue
        s, cs = inv[base], invc[cond]
        mean = lambda col: statistics.mean(float(r[col]) for r in rows)
        N[f"far1_{s}_{cs}"] = pct(mean("far_fixed_far_0.01"), 2)
        N[f"far01_{s}_{cs}"] = pct(mean("far_fixed_far_0.001"), 3)
        N[f"tarpc1_{s}_{cs}"] = pct(mean("tar_at_far_0.01"), 1)
        N[f"tarpc01_{s}_{cs}"] = pct(mean("tar_at_far_0.001"), 1)
        for r in rows:
            far_max.setdefault(s, []).append((float(r["far_fixed_far_0.01"]), cond, float(r["far_fixed_far_0.001"])))
    for s, vals in far_max.items():
        top = max(vals)
        N[f"farmax1_{s}"] = pct(top[0], 2) if s not in ("au",) else pct(top[0], 3)
        N[f"farmax1_{s}_cond"] = top[1].replace("frame_square_", "").replace("square_", "")
        N[f"farmax01_{s}"] = pct(max(v[2] for v in vals), 3)
    # the highest actual FAR at the 1% threshold among the four CosFace configurations and the pretrained models
    N["farmax1_cos_pre"] = pct(max(max(v[0] for v in far_max[s]) for s in ("arc", "sf", "nn", "at", "au", "both")), 3)
    N["farmax01_cos"] = pct(max(max(v[2] for v in far_max[s]) for s in ("nn", "at", "au", "both")), 3)
    top01 = max((v[2], s, v[1]) for s in ("nn", "at", "au", "both") for v in far_max[s])
    N["farmax01_cos_model"], N["farmax01_cos_cond"] = LABEL[top01[1]], top01[2].replace("square_", "")
    # the original model's frame square where its fixed-threshold FAR peaks
    N["farmax1_orig_cond_pct"] = pct(float(N["farmax1_orig_cond"]), 0)

    # protocol gap
    for r in read(STUDY / "analysis/protocol_gap.csv"):
        if r["model"] not in inv or r["condition"] not in invc:
            continue
        s, cs = inv[r["model"]], invc[r["condition"]]
        N[f"si_{s}_{cs}"] = pct(r["same_image_rank1"])
        N[f"gap_{s}_{cs}"] = pct(r["gap"])
        N[f"gap_{s}_{cs}_lo"], N[f"gap_{s}_{cs}_hi"] = pct(r["ci_low"]), pct(r["ci_high"])
    for lev in ("05", "10", "15", "20", "25", "30", "35", "40", "50"):
        d = float(N[f"si_orig_f{lev}"]) - float(N[f"si_pxf_f{lev}"])
        N[f"lead_study_{lev}"] = signed(dec(d, 1))
        N[f"lead_study_{lev}_abs"] = dec(abs(d), 1)

    # primary contrasts
    for r in read(STUDY / "analysis/contrasts_primary.csv"):
        p = r["contrast"]
        N[p] = spct(r["estimate"])
        N[f"{p}_abs"] = pct(abs(float(r["estimate"])))
        N[f"{p}_lo"], N[f"{p}_hi"] = spct(r["ci95_low"]), spct(r["ci95_high"])
        N[f"{p}_lo983"], N[f"{p}_hi983"] = spct(r["ci983_low"]), spct(r["ci983_high"])
        N[f"{p}_smin"], N[f"{p}_smax"] = spct(r["seed_combo_min"]), spct(r["seed_combo_max"])
        N[f"{p}_p"] = r["p_boot_text"].replace("< ", "")
        N[f"{p}_holm"] = r["p_holm_text"].replace("< ", "")
        N[f"{p}_past0"] = r["resamples_past_zero"]
        N["combos"] = r["seed_combos"]
        N["resamples"] = num(r["resamples"])
        N[f"{p}_claimed"] = r["claimed"]
    B = int(N["resamples"].replace(",", ""))
    # seven places, so that the floor visibly sits below the 0.001 it is quoted against
    N["p_floor"] = dec(2 / (B + 1), 7)
    N["holm_floor"] = dec(3 * 2 / (B + 1), 7)
    N["resamples1"] = num(B + 1)

    # main effects by condition
    me = {}
    for r in read(STUDY / "analysis/main_effects_by_condition.csv"):
        cs = invc[r["condition"]]
        k = "aug" if r["main_effect"] == "augmentation" else "att"
        me[(k, cs)] = r
        N[f"{k}_{cs}"] = spct(r["estimate"])
        N[f"{k}_{cs}_abs"] = pct(abs(float(r["estimate"])))
        N[f"{k}_{cs}_lo"], N[f"{k}_{cs}_hi"] = spct(r["ci_low"]), spct(r["ci_high"])
        N[f"{k}_{cs}_lo_abs"], N[f"{k}_{cs}_hi_abs"] = pct(abs(float(r["ci_low"]))), pct(abs(float(r["ci_high"])))
    p3 = [cs for cs in CONDS if cs not in ("s00", "s05", "s60")]
    trained = ["s10", "s15", "s20", "s25", "s30", "s40", "s50"]
    excl = [cs for cs in p3 if float(me[("att", cs)]["ci_low"]) > 0 or float(me[("att", cs)]["ci_high"]) < 0]
    N["att_excl_n"] = str(len(excl))
    N["att_excl_pos_n"] = str(sum(float(me[("att", cs)]["estimate"]) > 0 for cs in excl))
    N["att_trained_mean"] = spct(statistics.mean(float(me[("att", cs)]["estimate"]) for cs in trained))
    N["att_other_mean"] = spct(statistics.mean(float(me[("att", cs)]["estimate"]) for cs in p3 if cs not in excl))
    N["att_other_n"] = str(len(p3) - len(excl))
    tv = [float(me[("att", cs)]["estimate"]) for cs in trained]
    N["att_trained_min"], N["att_trained_max"] = spct(min(tv)), spct(max(tv))
    N["att_trained_excl_n"] = str(sum(cs in excl for cs in trained))
    au = {cs: float(me[("aug", cs)]["estimate"]) for cs in OTHER}
    N["aug_unseen_max_cond"] = max(au, key=au.get)
    N["aug_unseen_min_cond"] = min(au, key=au.get)

    # simple contrasts
    sc = {}
    for r in read(STUDY / "analysis/contrasts_by_condition.csv"):
        code = {v: k for k, v in SIMPLE.items()}[r["contrast"]]
        if r["protocol"] != "identity":
            continue
        cs = invc[r["condition"]]
        sc[(code, cs)] = r
        N[f"sc_{code}_{cs}"] = spct(r["difference"])
        N[f"sc_{code}_{cs}_abs"] = pct(abs(float(r["difference"])))
        N[f"sc_{code}_{cs}_lo"], N[f"sc_{code}_{cs}_hi"] = spct(r["ci_low"]), spct(r["ci_high"])
    ident = [r for r in read(STUDY / "analysis/contrasts_by_condition.csv") if r["protocol"] == "identity"]
    cos = [r for r in ident if "triplet" not in r["contrast"]]
    N["sc_cos_n"] = str(len(cos))
    N["sc_cos_excl_n"] = str(sum(float(r["ci_low"]) > 0 or float(r["ci_high"]) < 0 for r in cos))
    N["sc_tri_n"] = str(len(ident) - len(cos))

    # the primary contrasts with training seeds as the unit: a Welch-type t on the per-seed means over each
    # contrast's conditions (with three seeds per configuration, the bootstrap's seed half can only redraw
    # the same three values)
    seed_r1 = {}
    for r in summ:
        m_ = re.match(r"(.+)_seed(\d+)$", r["model"])
        if m_ and r["protocol"] == "identity":
            seed_r1[(m_.group(1), m_.group(2), r["condition"])] = 100 * float(r["rank1"])
    for code, _, pairs, conds in primary_definitions():
        w = {}
        for a_, b_ in pairs:
            w[a_] = w.get(a_, 0) + 0.5
            w[b_] = w.get(b_, 0) - 0.5
        means = {c: [statistics.mean(seed_r1[(c, sd, x)] for x in conds) for sd in sorted({k[1] for k in seed_r1 if k[0] == c})] for c in w}
        est = sum(w[c] * statistics.mean(v) for c, v in means.items())
        parts = [w[c] ** 2 * statistics.variance(v) / len(v) for c, v in means.items()]
        var = sum(parts)
        if var == 0:
            raise SystemExit(f"{code}: no seed-to-seed variation in any configuration; report it without a t test")
        df = var ** 2 / sum(p_ ** 2 / (len(means[c]) - 1) for p_, c in zip(parts, means))
        tval = est / var ** 0.5
        pv = t_two_sided(tval, df)
        N[f"{code}_seed_t"], N[f"{code}_seed_df"] = dec(tval, 1), dec(df, 1)
        N[f"{code}_seed_p"] = dec(pv, 3) if pv >= 0.001 else "0.001"
        N[f"{code}_seed_p_below"] = "yes" if pv < 0.001 else "no"
        # the same contrast with seeds matched across configurations: every configuration's seed s starts from
        # the same seed (study/train.py), so a t on the per-seed contrast values, with n - 1 degrees of freedom
        seeds = sorted({k[1] for k in seed_r1 if k[0] in w})
        d = [sum(w[c] * statistics.mean(seed_r1[(c, sd, x)] for x in conds) for c in w) for sd in seeds]
        sd_ = statistics.stdev(d)
        if sd_ == 0:  # every seed gives the same contrast, so t is undefined: stop rather than print a value
            raise SystemExit(f"{code}: the per-seed contrast is identical across seeds; report it without a t test")
        tp = statistics.mean(d) / (sd_ / len(d) ** 0.5)
        pp = t_two_sided(tp, len(d) - 1)
        N[f"{code}_pair_t"], N[f"{code}_pair_df"] = dec(tp, 1), str(len(d) - 1)
        N[f"{code}_pair_p"] = dec(pp, 3) if pp >= 0.001 else "0.001"
        N[f"{code}_pair_min"], N[f"{code}_pair_max"] = spct(min(d) / 100), spct(max(d) / 100)

    # mean rank of the right identity under noise fills, augmentation without attention (per seed)
    for lev in ("20", "40"):
        ranks = [float(r["mean_rank"]) for r in summ if re.match(r"cosface_noatt_aug_seed\d+$", r["model"])
                 and r["protocol"] == "identity" and r["condition"] == f"noise_square_0.{lev}"]
        N[f"mrank_au_noise{lev}_min"], N[f"mrank_au_noise{lev}_max"] = dec(min(ranks), 0), dec(max(ranks), 0)

    # probes that are near-copies of their gallery image (study/check_near_copies.py, from the per-probe files)
    ncr = {r["model"]: r for r in read(STUDY / "near_copies.csv")}
    N["nc_n"] = ncr["arcface_r50"]["near_copies"]
    N["nc_cos"] = dec(ncr["arcface_r50"]["threshold"], 1)
    N["nc_orig_hits"], N["nc_orig_hits_all"] = ncr["original_model"]["hits_on_near_copies"], ncr["original_model"]["hits"]
    for s, m_ in (("orig", "original_model"), ("pxc", "pixels_crop"), ("pxf", "pixels_frame"), ("arc", "arcface_r50"), ("sf", "sface")):
        N[f"nc_r1_{s}_without"] = pct(ncr[m_]["rank1_without_near_copies"])
    N["sc_n"] = str(len(ident))
    N["sc_excl_n"] = str(sum(float(r["ci_low"]) > 0 or float(r["ci_high"]) < 0 for r in ident))
    N["sc_chance_n"] = str(round(0.05 * len(ident)))

    # seed spread (identity rank-1, range over seeds)
    ss = [r for r in read(STUDY / "analysis/seed_spread.csv") if r["protocol"] == "identity" and r["metric"] == "rank1"]
    for s in ("nn", "at", "au", "both", "tri"):
        rows = [r for r in ss if r["model"] == MODELS[s]]
        rng = [float(r["range"]) for r in rows]
        N[f"spread_med_{s}"] = pct(statistics.median(rng))
        top = max(rows, key=lambda r: float(r["range"]))
        N[f"spread_max_{s}"] = pct(top["range"])
        N[f"spread_max_{s}_cond"] = top["condition"]
        N[f"spread_n_{s}"] = str(len(rows))
    for s in ("au",):
        for cs in ("noise20",):
            r = next(r for r in ss if r["model"] == MODELS[s] and r["condition"] == CONDS[cs])
            N[f"seed_{s}_{cs}_min"], N[f"seed_{s}_{cs}_max"] = pct(r["min"], 2), pct(r["max"], 2)

    # attention check, mean of three seeds
    for s in ("at", "both", "tri"):
        acc = {}
        for sd in tc["seeds"]:
            for r in read(STUDY / f"eval/{MODELS[s]}_seed{sd}/attention_check.csv"):
                acc.setdefault(r["condition"], []).append(r)
        for cond, rows in acc.items():
            cs = invc[cond]
            if rows[0]["attention_inside"] != "":
                N[f"ain_{s}_{cs}"] = dec(statistics.mean(float(r["attention_inside"]) for r in rows), 3)
                N[f"ain4_{s}_{cs}"] = dec(statistics.mean(float(r["attention_inside"]) for r in rows), 4)
            N[f"aout_{s}_{cs}"] = dec(statistics.mean(float(r["attention_outside"]) for r in rows), 3)
            N[f"aout4_{s}_{cs}"] = dec(statistics.mean(float(r["attention_outside"]) for r in rows), 4)

    # training logs
    first, last, secs = {}, {}, {}
    for s in ("nn", "at", "au", "both", "tri"):
        for sd in tc["seeds"]:
            rows = read(STUDY / f"train/{MODELS[s]}_seed{sd}/train_log.csv")
            first.setdefault(s, []).append(float(rows[0]["loss"]))
            last.setdefault(s, []).append(float(rows[-1]["loss"]))
            secs.setdefault(s, []).append(sum(float(r["seconds"]) for r in rows))
            N["train_epochs_logged"] = str(len(rows))
    cos1 = first["nn"] + first["at"] + first["au"] + first["both"]
    N["cos_ep1_lo"], N["cos_ep1_hi"] = dec(min(cos1), 1), dec(max(cos1), 1)
    noaug = last["nn"] + last["at"]
    aug = last["au"] + last["both"]
    N["cos_last_noaug_lo"], N["cos_last_noaug_hi"] = dec(min(noaug), 2), dec(max(noaug), 2)
    N["cos_last_aug_lo"], N["cos_last_aug_hi"] = dec(min(aug), 2), dec(max(aug), 2)
    N["tri_ep1_lo"], N["tri_ep1_hi"] = dec(min(first["tri"]), 3), dec(max(first["tri"]), 3)
    N["tri_last_lo"], N["tri_last_hi"] = dec(min(last["tri"]), 3), dec(max(last["tri"]), 3)
    # training time per run: the unit times in run.log (each includes loading and saving)
    unit = {}
    for line in rl:
        m = re.search(r"\] train (\w+) seed (\d+) ([\d.]+)s$", line)
        if m:
            unit.setdefault(m.group(1), []).append(float(m.group(3)))
    allsec = [v for k in unit for v in unit[k]]
    assert len(allsec) == int(N["n_runs"])
    N["train_s"] = num(round(sum(allsec)))
    N["train_min"] = str(round(sum(allsec) / 60))
    N["train_s_lo"] = num(round(min(statistics.mean(v) for v in unit.values())))
    N["train_s_hi"] = num(round(max(statistics.mean(v) for v in unit.values())))

    # bounds over many values: rounded outward, so "below" and "more than" stay true for the raw value
    raw_farmax = max(max(v[0] for v in far_max[s]) for s in ("arc", "sf", "nn", "at", "au", "both"))
    N["farmax1_cos_pre_ub"] = str(Decimal(repr(100 * raw_farmax)).quantize(Decimal("0.001"), rounding=ROUND_CEILING))
    # the analysis environment, from the comment in study/requirements.txt
    m = re.search(r"analyze\.py's outputs were produced on Python (\d+(?:\.\d+)*) with numpy (\d+(?:\.\d+)*) and pandas (\d+(?:\.\d+)*)",
                  (ROOT / "study/requirements.txt").read_text())
    N["an_py"], N["an_np"], N["an_pd"] = m.groups()

    # comparisons used in the text
    r1v = {(s, cs): float(iv[(MODELS[s], "identity", c, "rank1")]["estimate"]) for s in ("arc", "sf") for cs, c in CONDS.items()}
    N["arc_gt_sf_n"] = str(sum(r1v[("arc", cs)] > r1v[("sf", cs)] for cs in CONDS))
    fills = []
    for s in ("arc", "sf"):
        for grp in (("s20", "gray20", "noise20"), ("s40", "gray40", "noise40")):
            v = [r1v[(s, cs)] for cs in grp]
            fills.append(max(v) - min(v))
    N["fill_max_pre"] = pct(max(fills))
    N["fill_max_pre_ub"] = str(Decimal(repr(100 * max(fills))).quantize(Decimal("0.1"), rounding=ROUND_CEILING))
    gaps = [float(r["gap"]) for r in read(STUDY / "analysis/protocol_gap.csv")
            if float(r["condition"].split("_")[-1]) <= 0.5]
    N["gap_min_all"] = pct(min(gaps))
    N["gap_min_all_lb"] = str(Decimal(repr(100 * min(gaps))).quantize(Decimal("0.1"), rounding=ROUND_FLOOR))
    N["gap_n_all"] = str(len(gaps))
    lower = [cs for cs in CONDS if cs != "s00" and float(N[f"ain_at_{cs}"]) < float(N[f"aout_at_{cs}"])]
    N["att_at_lower_n"] = str(len(lower))
    N["att_at_cond_n"] = str(len(CONDS) - 1)
    N["att_at_higher"] = ", ".join(cs for cs in CONDS if cs != "s00" and cs not in lower)
    N["drop_train"] = num(int(N["role_train"].replace(",", "")) - cnt["train_images"])
    N["drop_probe"] = num(int(N["role_probe"].replace(",", "")) - cnt["probes"])
    N["px_10_lo"] = min((N[f"px_{k}_10"] for k in ("50", "100", "200", "500")), key=float)
    N["cos_clean_lo"] = pct(min(float(iv[(MODELS[s], "identity", "square_0.00", "rank1")]["estimate"]) for s in ("nn", "at", "au", "both")))
    N["cos_clean_hi"] = pct(max(float(iv[(MODELS[s], "identity", "square_0.00", "rank1")]["estimate"]) for s in ("nn", "at", "au", "both")))

    # differences used in the text
    N["clean_cost_both_vs_nn"] = pct(float(iv[(MODELS["nn"], "identity", "square_0.00", "rank1")]["estimate"]) -
                                     float(iv[(MODELS["both"], "identity", "square_0.00", "rank1")]["estimate"]))
    return (N,)


# ---- tables -----------------------------------------------------------------------------------

def table_original(N):
    """N = 100 original run, with the raw-pixel baseline, frame coverage."""
    levels = ["00", "05", "10", "15", "20", "25", "30", "40", "50", "60", "100"]
    cols = [("cu", "Cust."), ("vgg", "VGG"), ("fn", "FN"), ("fn512", "FN512"), ("of", "OF"), ("af", "AF"), ("sfo", "SF")]
    out = ["\\begin{tabular}{@{}r" + "r" * len(cols) + "r@{}}", "\\toprule",
           "Occ. & " + " & ".join(c for _, c in cols) + " & Pixels \\\\", "\\midrule"]
    for lev in levels:
        lab = f"{int(lev)}\\%"
        px = N.get(f"px_100_{lev}", "--") if lev != "00" else N["px_100_00"]
        out.append(lab + " & " + " & ".join(N[f"o_100_{k}_{lev}"] for k, _ in cols) + f" & {px} \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


def table_lead(N):
    levels = ["05", "10", "15", "20", "25", "30", "35", "40", "50"]
    out = ["\\begin{tabular}{@{}rrrrrr@{}}", "\\toprule",
           "Occ. & $N=50$ & $N=100$ & $N=200$ & $N=500$ & Study \\\\", "\\midrule"]
    for lev in levels:
        cells = [N[f"lead_{n}_{lev}"] for n in ("50", "100", "200", "500")] + [N[f"lead_study_{lev}"]]
        out.append(f"{int(lev)}\\% & " + " & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


def table_arms(N):
    rows = [
        ("ArcFace R50", "pretrained", "crop", "cosine"),
        ("SFace", "pretrained", "crop", "cosine"),
        ("Crop pixels", "none", "crop", "$L_2$"),
        ("Frame pixels", "none", f"frame, {N['frame_small']} px", "$L_2$"),
        ("Original model", "saved", f"frame, {N['frame']} px", "$L_2$"),
        ("CNN, neither", "CosFace", "crop", "cosine"),
        ("CNN, attention", "CosFace, att.", "crop", "cosine"),
        ("CNN, augmentation", "CosFace, aug.", "crop", "cosine"),
        ("CNN, both", "CosFace, att., aug.", "crop", "cosine"),
        ("CNN, both, triplet", "triplet, att., aug.", "crop", "cosine"),
    ]
    out = ["\\begin{tabular}{@{}llll@{}}", "\\toprule", "Model & Training & Input & Match \\\\", "\\midrule"]
    out += [" & ".join(r) + " \\\\" for r in rows]
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


def table_verif(N):
    rows = [("arc", "s00"), ("arc", "s20"), ("arc", "s50"), ("arc", "eye"), ("arc", "low"),
            ("sf", "s00"), ("sf", "s20"), ("sf", "s50"), ("sf", "eye"), ("sf", "low"),
            ("both", "s00"), ("both", "s20"), ("both", "s50"),
            ("nn", "s00"), ("nn", "s20"), ("nn", "s50")]
    cname = {"s00": "clean", "s20": "square 20\\%", "s50": "square 50\\%", "eye": "eye band", "low": "lower band"}
    out = ["\\begin{tabular}{@{}llrrrr@{}}", "\\toprule",
           "Model & Condition & TAR & FAR & TAR, re-set & TAR$_{0.1}$ \\\\", "\\midrule"]
    prev = None
    for s, cs in rows:
        name = LABEL[s] if s != prev else ""
        if s != prev and prev is not None:
            out.append("\\addlinespace")
        prev = s
        out.append(f"{name} & {cname[cs]} & {N[f'tarf1_{s}_{cs}']} & {N[f'far1_{s}_{cs}']} & "
                   f"{N[f'tarpc1_{s}_{cs}']} & {N[f'tarf01_{s}_{cs}']} \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


def table_attention(N):
    conds = [("s10", "Square 10\\%"), ("s30", "Square 30\\%"), ("s50", "Square 50\\%"), ("eye", "Eye band"),
             ("low", "Lower band"), ("gray20", "Gray 20\\%"), ("noise20", "Noise 20\\%")]
    out = ["\\begin{tabular}{@{}lrrr@{}}", "\\toprule",
           "Condition & CNN, both & CNN, attention & Triplet \\\\", "\\midrule"]
    for cs, lab in conds:
        cells = [f"{N[f'ain_{s}_{cs}']} / {N[f'aout_{s}_{cs}']}" for s in ("both", "at", "tri")]
        out.append(lab + " & " + " & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


def table_primary(N):
    rows = [("P1", ""), ("P2", ""), ("P3", "")]
    out = ["\\begin{tabular}{@{}lrrrr@{}}", "\\toprule",
           "& Est. & 95\\% interval & 98.3\\% interval & Seed combos \\\\", "\\midrule"]
    for p, lab in rows:
        out.append(f"{p}{lab} & {N[p]} & [{N[p + '_lo']}, {N[p + '_hi']}] & [{N[p + '_lo983']}, {N[p + '_hi983']}] & "
                   f"{N[p + '_smin']} to {N[p + '_smax']} \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


TABLES = {"table_original.tex": table_original, "table_lead.tex": table_lead, "table_arms.tex": table_arms,
          "table_verif.tex": table_verif, "table_attention.tex": table_attention, "table_primary.tex": table_primary}


def main():
    (N,) = compute()
    lines = ["% Generated by make_numbers.py. Do not edit."]
    for k in sorted(N):
        lines.append(f"\\expandafter\\def\\csname n@{k}\\endcsname{{{N[k]}}}")
    lines.append("\\newcommand{\\N}[1]{\\csname n@#1\\endcsname}")
    (PAPER / "numbers.tex").write_text("\n".join(lines) + "\n")
    for fn, f in TABLES.items():
        (PAPER / fn).write_text("% Generated by make_numbers.py. Do not edit.\n" + f(N))


if __name__ == "__main__":
    main()
