// Check the built course site (docs/) against the repository's committed results.
//
//   node tools/check_site.mjs [<repo>] [<chapter-folder> ...]
//
// For each chapter, it checks that:
//  1. every value the face plate shows (data/faces.json), and the interval ends it draws, equal the
//     matching row of results/study/analysis/intervals.csv (shown values rounded half up to one decimal
//     as the paper rounds), and its conditions and their geometry are the study's (study/config.json,
//     drawn as study/data.py draws them);
//  2. every value it shares with the paper equals the paper's generated value in paper/numbers.tex;
//  3. every number in every caption and note that plate can show (each condition, each test) is one of
//     that condition's values, the chance level, or a setting of the study;
//  4. every figure on the page is byte-identical to the chapter's committed figure;
//  5. the hero's sentence is the chapter's own, its key numbers are the paper's values, their labels
//     name the matching condition, and its miniature's values are the face plate's;
//  6. the SFace plate's cutoffs, detector settings, model files, hashes, and addresses, and occluders
//     are the study's (results/study/eval/sface/fixed_thresholds.json, study/config.json,
//     study/data.py, study/README.md); its verdicts decide as the study does (strictly above the
//     cutoff); and every number its sentences can show is the score, a cutoff, or a false accept rate;
//  7. the seed-draw plate's values, draws, and sentences match the study's results (check_seeds.mjs).
//  8. the portraits the face plates offer are the chapter's (chapter/portraits/manifest.json), byte for
//     byte, each credited as a synthetic portrait.
// Every live plate on the page must have its data and its checks. It doesn't check the figures'
// drawing (chapter/make_site_figures.py --check does) or run the face models.
// It writes data/checks.json beside each chapter's data, which the plates show, and exits 1 on any failure.
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { join, resolve, dirname } from "node:path";
import { pathToFileURL, fileURLToPath } from "node:url";
import { checkSeeds } from "./check_seeds.mjs";

const repo = resolve(process.argv[2] || join(dirname(fileURLToPath(import.meta.url)), ".."));
const chapters = process.argv.slice(3).length ? process.argv.slice(3) : ["facial-recognition-under-occlusion-project"];
const fails = [];
let passes = 0;
const ok = (cond, msg) => { if (cond) passes++; else fails.push(msg); };

function csv(path) {
  const [head, ...rows] = readFileSync(path, "utf8").trim().split("\n");
  const cols = head.split(",");
  return rows.map((r) => Object.fromEntries(r.split(",").map((v, i) => [cols[i], v])));
}
function halfUp1(share) {
  // 100 * share rounded to ten places, then half up to one decimal (paper/make_numbers.py's dec()).
  const n = Math.round(100 * Number(share) * 1e10) / 1e10;
  return (Math.floor(Number((n * 10).toFixed(6)) + 0.5) / 10).toFixed(1);
}
const MODEL = { arc: "arcface_r50", sf: "sface", both: "cosface_att_aug", au: "cosface_noatt_aug", at: "cosface_att_noaug",
  nn: "cosface_noatt_noaug", tri: "triplet_att_aug", px: "pixels_crop" };
const COND = { square_0: "s00", "square_0.05": "s05", "square_0.10": "s10", "square_0.15": "s15", "square_0.20": "s20",
  "square_0.25": "s25", "square_0.30": "s30", "square_0.40": "s40", "square_0.50": "s50", "square_0.60": "s60",
  eyes_band: "eye", lower_band: "low", "random_square_0.20": "off20", "gray_square_0.20": "gray20",
  "noise_square_0.20": "noise20", "gray_square_0.40": "gray40", "noise_square_0.40": "noise40" };
const condKey = (k) => COND[k === "square_0.00" ? "square_0" : k];
const decode = (s) => s.replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"').replace(/&#x27;|&#39;/g, "'");

for (const ch of chapters) {
  const proj = join(repo, ch);
  const site = join(repo, "docs", ch);
  const html = readFileSync(join(site, "index.html"), "utf8");
  const macros = new Map([...readFileSync(join(proj, "paper/numbers.tex"), "utf8").matchAll(/csname n@([^\\]+)\\endcsname\{([^}]*)\}/g)].map((m) => [m[1], m[2]]));
  const counts = {};
  const shown = new Set([...html.matchAll(/data-demo="([^"]+)"/g)].map((m) => m[1]));
  for (const kind of shown) ok(existsSync(join(site, `data/${kind}.json`)), `the page shows a ${kind} plate without its data`);
  ok(shown.has("faces"), "the page has no face plate");
  const section = (name, fn) => { const before = passes + fails.length; fn(); counts[name] = passes + fails.length - before; };
  ok(!html.includes("{{"), "a {{key}} was left unfilled on the page");

  // 1 to 3: the face plate
  const data = JSON.parse(readFileSync(join(site, "data/faces.json"), "utf8"));
  const model = await import(pathToFileURL(join(repo, "docs/assets/demos/occluders-model.js")).href);
  section("faces", () => {
    const iv = new Map(csv(join(proj, "results/study/analysis/intervals.csv")).filter((r) => r.metric === "rank1")
      .map((r) => [`${r.model}|${r.protocol}|${r.condition}`, r]));
    for (const c of data.conditions) {
      for (const [field, proto] of [["identity", "identity"], ["same", "same_image"]]) {
        for (const [short, name] of Object.entries(MODEL)) {
          const r = iv.get(`${name}|${proto}|${c.id}`);
          const v = c[field] && c[field][short];
          ok(Boolean(r) === Boolean(v), `${c.id} ${field} ${short}: present in one source only`);
          if (!r || !v) continue;
          ok(v.s === halfUp1(r.estimate), `${c.id} ${field} ${short}: ${v.s} vs ${halfUp1(r.estimate)}`);
          ok(v.lo === halfUp1(r.ci_low) && v.hi === halfUp1(r.ci_high), `${c.id} ${field} ${short}: interval`);
          ok(Math.abs(v.v - 100 * r.estimate) < 1e-3, `${c.id} ${field} ${short}: plotted value`);
          ok(Math.abs(v.l - 100 * r.ci_low) < 1e-3 && Math.abs(v.h - 100 * r.ci_high) < 1e-3, `${c.id} ${field} ${short}: plotted interval`);
        }
      }
    }
    // the conditions: every crop condition the study ran, drawn as study/data.py draws it
    const cfg = JSON.parse(readFileSync(join(proj, "study/config.json"), "utf8"));
    const crop = cfg.crop_size, cc = cfg.conditions;
    ok(data.crop === crop, `crop size: page ${data.crop}, study ${crop}`);
    const side = (cov) => Math.floor(Math.sqrt(cov * crop * crop));
    const centered = (cov) => { const s = side(cov), o = Math.max(0, Math.floor((crop - s) / 2)); return s ? [o, o, s, s] : null; };
    const want = new Map();
    for (const cov of cc.square_coverages) want.set(`square_${cov.toFixed(2)}`, { rect: centered(cov), fill: "black" });
    for (const [id, rows] of [["eyes_band", cc.eyes_band_rows], ["lower_band", cc.lower_band_rows]]) {
      const r0 = Math.floor(rows[0] * crop), r1 = Math.floor(rows[1] * crop);
      want.set(id, { rect: [0, r0, crop, r1 - r0], fill: "black" });
    }
    want.set(`random_square_${cc.random_square_coverage.toFixed(2)}`, { side: side(cc.random_square_coverage), fill: "black" });
    for (const [fill, cov] of cc.fill_variants) want.set(`${fill}_square_${cov.toFixed(2)}`, { rect: centered(cov), fill });
    ok(data.conditions.length === want.size && data.conditions.every((c) => want.has(c.id)), `conditions: page ${data.conditions.length}, study ${want.size}`);
    for (const c of data.conditions) {
      const w = want.get(c.id);
      if (!w) continue;
      if (w.side !== undefined) ok(c.rect && c.rect[2] === w.side && c.rect[3] === w.side && c.rect[0] + w.side <= crop && c.rect[1] + w.side <= crop, `${c.id}: example square`);
      else ok(JSON.stringify(c.rect) === JSON.stringify(w.rect), `${c.id}: geometry ${JSON.stringify(c.rect)} vs ${JSON.stringify(w.rect)}`);
      if (c.rect) ok(c.fill === w.fill, `${c.id}: fill`);
    }
    let shared = 0;
    for (const c of data.conditions) {
      for (const short of Object.keys(MODEL)) {
        const key = `r1_${short}_${condKey(c.id)}`;
        if (macros.has(key) && c.identity && c.identity[short]) {
          shared++;
          ok(c.identity[short].s === macros.get(key), `${key}: page ${c.identity[short].s}, paper ${macros.get(key)}`);
        }
      }
    }
    ok(shared > 100, `only ${shared} values shared with the paper`);
    ok(data.chance === macros.get("chance"), `chance: page ${data.chance}, paper ${macros.get("chance")}`);
    ok(String(data.train_range[0]) === macros.get("occ_lo") && String(data.train_range[1]) === macros.get("occ_hi"), `training range: page ${data.train_range}`);
    ok(String(data.seeds) === macros.get("n_seeds"), `seeds: page ${data.seeds}, paper ${macros.get("n_seeds")}`);
    for (const c of data.conditions) {
      const note = model.fitNote(data, c);
      for (const n of note.match(/\d+/g) || []) ok([String(data.train_range[0]), String(data.train_range[1]), "128"].includes(n), `fit note for ${c.id} has ${n}`);
    }
    for (const c of data.conditions) {
      for (const test of ["identity", "same"]) {
        let text = model.caption(data, c, test);
        for (const m of data.models) text = text.split(m.name).join("MODEL");  // "ArcFace R50" is a name, not a number
        const allowed = new Set([data.chance, ...[c.identity, c.same].filter(Boolean).flatMap((s) => Object.values(s).flatMap((v) => [v.s, v.lo, v.hi]))]);
        for (const n of text.match(/\d+(?:\.\d+)?/g) || []) {
          if (/^\d+$/.test(n) && text.includes(`${n}% of the crop`) && String(c.size) === n) { passes++; continue; }
          if (n === "95") { passes++; continue; }  // "95% interval"
          ok(allowed.has(n), `caption for ${c.id} (${test}) has ${n}, not a value of that condition`);
        }
      }
    }
  });

  // 4. figures
  section("figures", () => {
    let figs = 0;
    for (const m of html.matchAll(/(?:src|srcset)="figures\/([^"?]+)"/g)) {
      const committed = join(proj, "chapter/figures", m[1]);
      ok(existsSync(committed) && readFileSync(join(site, "figures", m[1])).equals(readFileSync(committed)), `figure ${m[1]} differs from the committed one`);
      figs++;
    }
    ok(figs >= 5, `only ${figs} figures on the page`);
  });

  // 5. the hero
  section("hero", () => {
    const md = readFileSync(join(proj, "chapter/chapter.md"), "utf8").replace(/\s+/g, " ");
    const line = html.match(/<p class="hero-line">([\s\S]*?)<\/p>/);
    ok(Boolean(line), "the hero has no sentence");
    if (line) {
      const plain = decode(line[1].replace(/<em>|<\/em>/g, "*").replace(/<[^>]+>/g, "")).replace(/’/g, "'").trim();
      ok(md.includes(plain.replace(/\.$/, "")), `the hero's sentence is not the chapter's: ${plain}`);
    }
    const dts = [...html.matchAll(/<dt data-n="([^"]*)">([^<]*)<\/dt>/g)];
    ok(dts.length >= 2, `the hero shows ${dts.length} key numbers`);
    const dds = [...html.matchAll(/<dt data-n="([^"]*)">[^<]*<\/dt><dd>([^<]*)<\/dd>/g)];
    ok(dds.length === dts.length, "a hero number has no label");
    for (const [, keys, label] of dds) {
      const nums = (decode(label).replace(/rank-1/g, "rank-one").match(/\d+(?:\.\d+)?/g) || []);
      const frame = keys.match(/px_(\d+)_/), cropKey = keys.match(/_s(\d\d)\b/);
      if (frame) ok(nums.length === 1 && decode(label).includes(`${Number(frame[1])}% of the frame`), `hero label for ${keys} doesn't name ${Number(frame[1])}% of the frame`);
      else if (cropKey) ok(nums.length === 1 && decode(label).includes(`${Number(cropKey[1])}% of the aligned crop`), `hero label for ${keys} doesn't name ${Number(cropKey[1])}% of the aligned crop`);
      else ok(nums.length === 0, `hero label for ${keys} has a number`);
    }
    for (const [, keys, text] of dts) {
      const shown = decode(text);
      const left = keys.split(" ").filter(Boolean).reduce((s, k) => { ok(macros.has(k) && s.includes(macros.get(k)), `hero number ${shown} lacks ${k}`); return s.replace(macros.get(k), ""); }, shown);
      for (const n of left.match(/\d+(?:\.\d+)?/g) || []) ok(n === "100", `hero number ${shown} has ${n}, not a paper value`);
    }
    const hd = JSON.parse(html.match(/<script type="application\/json" class="hero-data">([\s\S]*?)<\/script>/)[1]);
    const squares = data.conditions.filter((c) => c.kind === "square").sort((a, b) => a.size - b.size);
    ok(hd.steps.length === squares.length, "the hero's steps differ from the face plate's squares");
    hd.steps.forEach((s, i) => {
      const c = squares[i];
      ok(s.size === c.size && JSON.stringify(s.rect) === JSON.stringify(c.rect), `hero step ${s.size}: square`);
      ok(s.arc === c.identity.arc.s && s.sf === c.identity.sf.s && s.arcv === c.identity.arc.v && s.sfv === c.identity.sf.v, `hero step ${s.size}: values`);
    });
    const note = html.match(/<p class="hero-note">([^<]*)<\/p>/);
    ok(note && note[1].includes(macros.get("n_test_identities")), "the hero's note lacks the number of test identities");
  });

  // 6. the SFace plate
  if (shown.has("sface") && existsSync(join(site, "data/sface.json"))) {
    const sd = JSON.parse(readFileSync(join(site, "data/sface.json"), "utf8"));
    const words = await import(pathToFileURL(join(repo, "docs/assets/demos/still-model.js")).href);
    section("sface", () => {
      const ft = JSON.parse(readFileSync(join(proj, "results/study/eval/sface/fixed_thresholds.json"), "utf8"));
      ok(ft.reference_condition === "square_0.00" && ft.metric === "cosine", "fixed thresholds: not the clean cosine ones");
      ok(sd.thresholds.length === Object.keys(ft.thresholds).length, "threshold count");
      for (const t of sd.thresholds) ok(ft.thresholds[String(t.far)] === t.cos, `threshold for ${t.far}: ${t.cos}`);
      ok(sd.thresholds.every((t, i) => i === 0 || t.far < sd.thresholds[i - 1].far), "thresholds are listed from the loosest");
      const far = (x) => macros.get(x === 0.01 ? "far1" : "far01");
      for (const t of sd.thresholds) ok(words.farText(t.far) === `${far(t.far)}%`, `false accept rate ${t.far} reads ${words.farText(t.far)}`);
      const cfg = JSON.parse(readFileSync(join(proj, "study/config.json"), "utf8"));
      const readme = readFileSync(join(proj, "study/README.md"), "utf8");
      const sha = (f) => (readme.match(new RegExp("`" + f.replace(/\./g, "\\.") + "`[^\\n]*`([0-9a-f]{64})`")) || [])[1];
      ok(sd.detector.file === cfg.align.detector_file && sd.detector.score_threshold === cfg.align.score_threshold, "detector file or score threshold");
      ok(sd.detector.sha256 === sha(sd.detector.file), "detector SHA-256 differs from study/README.md");
      const sf = cfg.pretrained.find((m) => m.key === "sface");
      ok(sd.recognizer.file === sf.file && sd.recognizer.sha256 === sf.sha256 && sf.sha256 === sha(sf.file), "SFace file or SHA-256");
      const zoo = (dir, f) => new RegExp(`^https://media\\.githubusercontent\\.com/media/opencv/opencv_zoo/[0-9a-f]{40}/models/${dir}/${f.replace(/\./g, "\\.")}$`);
      ok(zoo("face_recognition_sface", sf.file).test(sd.recognizer.url), `SFace address: ${sd.recognizer.url}`);
      ok(zoo("face_detection_yunet", sd.detector.file).test(sd.detector.url), `YuNet address: ${sd.detector.url}`);
      ok(sd.recognizer.url.split("/")[7] === sd.detector.url.split("/")[7], "the two models come from different OpenCV Zoo commits");
      const create = readFileSync(join(proj, "study/data.py"), "utf8").match(/FaceDetectorYN\.create\([^,]+,\s*''\s*,\s*\([^)]*\)\s*,\s*score_threshold\s*,\s*([\d.]+)\s*,\s*(\d+)\)/);
      ok(Boolean(create) && Number(create[1]) === sd.detector.nms, `detector NMS: page ${sd.detector.nms}, study ${create && create[1]}`);
      ok(new RegExp(`\\(YuNet, ${sd.detector.bytes.toLocaleString("en-US")} bytes\\)`).test(readme), "detector size differs from study/README.md");
      for (const o of sd.occluders) {
        const c = data.conditions.find((x) => x.id === o.id);
        ok(c && JSON.stringify(c.rect) === JSON.stringify(o.rect) && c.fill === o.fill, `occluder ${o.id} differs from the face plate's`);
      }
      // sentences: every number is the score, a cutoff, or a false accept rate, and verdicts decide as the study does
      const rates = sd.thresholds.map((t) => words.farText(t.far).replace("%", ""));
      const scores = [];
      for (let k = -200; k <= 200; k++) scores.push(k / 200);
      for (const t of sd.thresholds) scores.push(t.cos, t.cos + 1e-9, t.cos - 1e-9, Number(t.cos.toFixed(3)), Number(t.cos.toFixed(3)) + 4e-4, Number(t.cos.toFixed(3)) - 4e-4);
      const occs = [null, ...["square", "eye", "low"].map((kind) => sd.occluders.find((o) => o.kind === kind))];
      ok(occs.every((o, i) => i === 0 || o), "the SFace plate lacks a square or band");
      const allowedNums = (s) => new Set([s.toFixed(3), ...sd.thresholds.map((t) => t.cos.toFixed(3)), ...rates]);
      for (const s of scores) {
        const allowed = allowedNums(s);
        for (const n of (words.scoreLine(s, sd.thresholds) + " " + words.cutoffNote(sd.thresholds)).match(/-?\d+(?:\.\d+)?/g) || []) ok(allowed.has(n), `SFace score line for ${s} has ${n}`);
        const above = s > sd.thresholds[0].cos;
        ok(words.passes(s, sd.thresholds[0]) === above, `cutoff decision at ${s}`);
        for (const source of ["camera", "photo", "portrait", "drawn"]) {
          for (const occ of occs) {
            const v = words.verdict(s, sd.thresholds, source, occ);
            ok(v.startsWith(above ? "Accepted" : "Refused"), `verdict at ${s}`);
            for (const n of (v + " " + words.sourceNote(source)).match(/-?\d+(?:\.\d+)?/g) || []) ok(allowed.has(n), `SFace verdict for ${s} has ${n}`);
          }
        }
      }
      for (const n of words.NO_FACE.match(/\d+/g) || []) ok(false, `the no-face line has ${n}`);
    });
  }

  // 7. the seed-draw plate
  if (shown.has("seeds") && existsSync(join(site, "data/seeds.json"))) {
    section("seeds", () => {});
    const before = passes + fails.length;
    await checkSeeds({ repo, chapter: ch, ok });
    counts.seeds = passes + fails.length - before;
  }

  // 8. the portraits
  section("portraits", () => {
    const man = join(proj, "chapter/portraits/manifest.json");
    if (!existsSync(man)) { ok(data.faces.length === 1 && data.faces[0].id === "drawn", "faces without a portrait manifest"); return; }
    const m = JSON.parse(readFileSync(man, "utf8"));
    ok(/synthetic portrait of a person who doesn't exist/.test(m.credit), "the portraits' credit doesn't say they are synthetic");
    ok(data.faces.length === m.portraits.length, `faces: page ${data.faces.length}, manifest ${m.portraits.length}`);
    m.portraits.forEach((q, i) => {
      const f = data.faces[i];
      ok(f && f.src === `portraits/${q.file}` && f.name === q.name && f.credit === m.credit, `portrait ${q.file}: page entry`);
      ok(existsSync(join(site, "portraits", q.file)) && readFileSync(join(site, "portraits", q.file)).equals(readFileSync(join(proj, "chapter/portraits", q.file))), `portrait ${q.file} differs from the chapter's`);
    });
    if (existsSync(join(site, "data/sface.json"))) {
      const sd = JSON.parse(readFileSync(join(site, "data/sface.json"), "utf8"));
      ok(JSON.stringify(sd.faces) === JSON.stringify(data.faces), "the SFace plate offers other faces than the face plate");
    }
  });

  const stamp = { checked: new Date().toLocaleDateString("en-CA"), passed: passes, failed: fails.length, counts,
    checker: "tools/check_site.mjs",
    against: ["results/study/analysis", "results/study/eval/sface/fixed_thresholds.json", "study/config.json", "study/README.md", "paper/numbers.tex", "chapter/chapter.md", "chapter/figures"] };
  writeFileSync(join(site, "data/checks.json"), JSON.stringify(stamp, null, 1) + "\n");
  console.log(ch, counts);
}
for (const f of fails) console.log("FAIL:", f);
console.log(`${passes} checks passed, ${fails.length} failed`);
process.exit(fails.length ? 1 : 0);
