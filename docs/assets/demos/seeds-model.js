// The seed-draw plate's model: pure functions over the study's per-seed results, no DOM. The page
// and the site's checks both import this file, so what the page says is what the checks hold to the
// data in results/study/analysis.
//
// A draw takes one training seed for each of the four configurations. With three seeds each, there
// are 3 × 3 × 3 × 3 = 81 draws, numbered in the order study/analyze.py enumerates them (the
// configurations sorted by name, the last one's seed changing fastest).

// ---- draws -----------------------------------------------------------------------------------------

/** How many draws there are: the number of seeds to the power of the number of configurations. */
export function drawCount(data) {
  return data.seeds.length ** data.configs.length;
}

/** Draw number i (0 to drawCount - 1) as one seed id per configuration, in data.configs order. */
export function drawAt(data, i) {
  const n = data.seeds.length;
  const out = new Array(data.configs.length);
  for (let k = data.configs.length - 1; k >= 0; k--) {
    out[k] = data.seeds[i % n];
    i = Math.floor(i / n);
  }
  return out;
}

/** The number of a draw given as one seed id per configuration. */
export function drawIndex(data, draw) {
  let i = 0;
  for (const s of draw) i = i * data.seeds.length + data.seeds.indexOf(s);
  return i;
}

// ---- the contrasts -----------------------------------------------------------------------------

const mean = (xs) => { let s = 0; for (const x of xs) s += x; return s / xs.length; };

/** One primary contrast's record from the data (P1, P2, or P3). */
export function contrastOf(data, key) {
  return data.contrasts.find((c) => c.key === key);
}

/** A contrast's weight on each configuration: +1/2 for the first of each pair, -1/2 for the second. */
export function weights(data, key) {
  const c = contrastOf(data, key);
  const w = {};
  for (const [a, b] of c.pairs) {
    w[a] = (w[a] || 0) + 1 / c.pairs.length;
    w[b] = (w[b] || 0) - 1 / c.pairs.length;
  }
  return w;
}

/** One draw's value of a contrast, as a share of probes. For each configuration pair (with the idea,
 *  without it), the mean over the contrast's conditions of the first one's rank-1 minus the second
 *  one's, each at its drawn seed; then the mean over the pairs (study/analyze.py, seed_combo). */
export function contrast(data, key, draw) {
  const c = contrastOf(data, key);
  const seedOf = {};
  data.configs.forEach((cf, k) => { seedOf[cf.id] = draw[k]; });
  const at = c.conditions.map((cond) => data.conditions.indexOf(cond));
  return mean(c.pairs.map(([a, b]) => mean(at.map((j) => data.rank1[a][seedOf[a]][j] - data.rank1[b][seedOf[b]][j]))));
}

/** The contrast in all draws, in draw order. */
export function all81(data, key) {
  return Array.from({ length: drawCount(data) }, (_, i) => contrast(data, key, drawAt(data, i)));
}

/** The study's estimate (study/analyze.py): each configuration's rank-1 averaged over its seeds, then
 *  over the contrast's conditions, then the pairs' differences averaged. It equals the mean of all
 *  draws. */
export function estimate(data, key) {
  const c = contrastOf(data, key);
  const at = c.conditions.map((cond) => data.conditions.indexOf(cond));
  const group = (cf) => mean(at.map((j) => mean(data.seeds.map((s) => data.rank1[cf][s][j]))));
  return mean(c.pairs.map(([a, b]) => group(a) - group(b)));
}

/** The lowest and highest value over all draws. */
export function range81(data, key) {
  const v = all81(data, key);
  return { min: Math.min(...v), max: Math.max(...v) };
}

// ---- numbers as the paper writes them ----------------------------------------------------------

/** 100x with one decimal, unsigned, rounded as paper/make_numbers.py's dec(): to ten places first,
 *  then half up. toFixed rounds the exact binary value, and the half-up step works on its digits. */
export function pct1(x) {
  const [whole, frac] = Math.abs(100 * x).toFixed(10).split(".");
  let tenths = Number(whole) * 10 + Number(frac[0]);
  if (frac[1] >= "5") tenths += 1;
  return `${Math.floor(tenths / 10)}.${tenths % 10}`;
}

/** A share as signed percentage points (make_numbers.py's spct()): "+12.1", "−1.3", or "0.0" for an
 *  exact zero; a value that rounds to zero keeps its sign. */
export function points(x) {
  const s = pct1(x);
  if (x === 0) return s;
  return (x < 0 ? "−" : "+") + s;
}

// ---- the number line -------------------------------------------------------------------------

/** The shared axis, in points: zero, every draw, and every interval end, with a little room. */
export function axis(data) {
  const vals = [0];
  for (const c of data.contrasts) {
    const r = range81(data, c.key);
    vals.push(100 * r.min, 100 * r.max, 100 * c.study.ci95_low, 100 * c.study.ci95_high);
  }
  let lo = Math.min(...vals), hi = Math.max(...vals);
  const pad = 0.04 * (hi - lo);
  lo -= pad; hi += pad;
  const step = [1, 2, 5, 10, 20, 50].find((s) => (hi - lo) / s <= 8) ?? 100;
  const ticks = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi; t += step) ticks.push({ at: t, major: Math.round(t / step) % 2 === 0 });
  return { lo, hi, step, ticks };
}

/** Where a value in points sits on the axis, as a percentage of its width. */
export function place(ax, v) {
  return (100 * (v - ax.lo)) / (ax.hi - ax.lo);
}

/** Each draw's height in its strip, from 0 to 1: draws in order of value take golden-ratio steps,
 *  so draws with close values sit apart. */
export function lanes(values) {
  const order = values.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const y = new Array(values.length);
  order.forEach(([, i], rank) => { y[i] = (rank * 0.6180339887498949 + 0.5) % 1; });
  return y;
}

// ---- words -------------------------------------------------------------------------------------
//
// Each sentence is built as a list of words and slots. A slot holds one number the sentence shows and
// names it (for example "P1.draw", this draw's value of P1), and about() marks a phrase that speaks of
// one contrast; text() joins the list. The checks hold every slot to the value its name says, every
// contrast's value to a phrase about that contrast, and find no digit outside a slot, so the words can
// be rewritten without touching a value, and a value cannot move to another phrase unnoticed.

const slot = (name, value) => ({ name, value: String(value) });
const about = (key, parts) => ({ about: key, parts });

/** Words and slots from a template: interpolated slots, phrases, and lists stay as they are, strings
 *  are words. */
function say(strings, ...vals) {
  const out = [];
  strings.forEach((w, i) => { out.push(w); if (i < vals.length) out.push(vals[i]); });
  return out.flat(Infinity).filter((p) => p !== "");
}

/** Visit a sentence's words and slots in order, with the contrast each phrase is about (or null). */
export function walk(parts, fn, key = null) {
  for (const p of [parts].flat(Infinity)) {
    if (typeof p === "object" && p.about) walk(p.parts, fn, p.about);
    else fn(p, key);
  }
}

/** A sentence's text. */
export function text(parts) {
  let out = "";
  walk(parts, (p) => { out += typeof p === "string" ? p : p.value; });
  return out;
}

const IDEA = { augmentation: "Occlusion training", attention: "Attention" };
export const AXES = {
  augmentation: ["Without occlusion training", "With occlusion training"],
  attention: ["Without attention", "With attention"],
};

/** The smallest and largest of P1's centered squares, in percent of the crop, read from their names. */
export function squareSizes(data) {
  const sizes = contrastOf(data, "P1").conditions.map((c) => Math.round(100 * Number(c.replace(/^square_/, ""))));
  return [Math.min(...sizes), Math.max(...sizes)];
}

/** A signed value in a slot, then its unit: "+1.0 point", "+12.1 points". */
function inPoints(name, s) {
  return [slot(name, s), s.replace(/^[+−]/, "") === "1.0" ? " point" : " points"];
}

// The condition sets: the trained-on squares (P1's), the held-out occluders (P2's), and both (P3's).
const SET = { squares: "P1", heldout: "P2", both: "P3" };
const count = (data, set) => slot(`set.${set}`, contrastOf(data, SET[set]).conditions.length);

/** Which conditions a contrast averages over. */
function scope(data, key) {
  if (key === "P1") {
    const [from, to] = squareSizes(data);
    return say`on the ${count(data, "squares")} trained-on squares (centered, ${slot("squares.from", from)} to ${slot("squares.to", to)}% of the crop)`;
  }
  if (key === "P2") return say`on the ${count(data, "heldout")} held-out occluders (none seen in training)`;
  return say`on the ${count(data, "squares")} squares and the ${count(data, "heldout")} held-out occluders together (${count(data, "both")} conditions)`;
}

/** A contrast's name on its number line. */
export function titleParts(data, key) {
  return say`${IDEA[contrastOf(data, key).factor]}, ${scope(data, key)}`;
}
export const title = (data, key) => text(titleParts(data, key));

/** Whether the draws stay on one side of zero. */
export function side(min, max) {
  if (min > 0) return "above zero in every one";
  if (max < 0) return "below zero in every one";
  return "on both sides of zero";
}

const drawn = (data, key, draw) => slot(`${key}.draw`, points(contrast(data, key, draw)));
const lowest = (data, key) => slot(`${key}.min`, points(range81(data, key).min));
const highest = (data, key) => slot(`${key}.max`, points(range81(data, key).max));
const study = (data, key, f) => slot(`${key}.${f}`, contrastOf(data, key).s[f]);
const combos = (data) => slot("combos", data.combos);
const level = slot("level", 95);

/** The three readings under a number line: this draw, the range over all draws, and the study. */
export function readoutParts(data, key, draw) {
  return [
    { label: say`this draw`, value: [about(key, say`${drawn(data, key, draw)}`)] },
    { label: say`all ${combos(data)}`, value: [about(key, say`${lowest(data, key)} to ${highest(data, key)}`)] },
    { label: say`study`, value: [about(key, say`${study(data, key, "estimate")} (${level}%: ${study(data, key, "lo")} to ${study(data, key, "hi")})`)] },
  ];
}
export const readout = (data, key, draw) => readoutParts(data, key, draw).map((p) => ({ label: text(p.label), value: text(p.value) }));

/** The number line's label for a screen reader. */
export function rowLabelParts(data, key, draw) {
  const d = drawn(data, key, draw);
  return [about(key, say`${titleParts(data, key)}: ${inPoints(d.name, d.value)} in this draw; ${lowest(data, key)} to ${highest(data, key)} over all ${combos(data)} draws; the study's estimate ${study(data, key, "estimate")}, ${level}% interval ${study(data, key, "lo")} to ${study(data, key, "hi")}.`)];
}
export const rowLabel = (data, key, draw) => text(rowLabelParts(data, key, draw));

/** How a contrast is built, and the exact value beside the sampled ones. */
export function detailParts(data, key) {
  const c = contrastOf(data, key);
  const other = c.factor === "attention" ? "occlusion training" : "attention";
  return [about(key, say`${titleParts(data, key)}: the configurations marked + minus those marked −, once with ${other} and once without, averaged. The mean of all ${combos(data)} draws is the study's estimate, ${inPoints(`${key}.estimate`, c.s.estimate)}; its ${level}% bootstrap interval, which also resamples the test identities, is ${study(data, key, "lo")} to ${study(data, key, "hi")}.`)];
}
export const detail = (data, key) => text(detailParts(data, key));

/** The sentence under the plate, for the current draw. */
export function captionParts(data, draw) {
  const p3 = range81(data, "P3");
  const p1d = drawn(data, "P1", draw);
  return say`In this draw, ${about("P1", say`occlusion training changes rank-1 by ${inPoints(p1d.name, p1d.value)} on the ${count(data, "squares")} trained-on squares`)} ${about("P2", say`and by ${drawn(data, "P2", draw)} on the ${count(data, "heldout")} held-out occluders`)}, ${about("P3", say`and attention changes it by ${drawn(data, "P3", draw)} on those ${count(data, "both")} conditions together`)}; over all ${combos(data)} draws, ${about("P1", say`occlusion training's effect on the squares stays between ${lowest(data, "P1")} and ${highest(data, "P1")}`)}, ${about("P3", say`and attention's between ${lowest(data, "P3")} and ${highest(data, "P3")}, ${side(p3.min, p3.max)}`)}.`;
}
export const caption = (data, draw) => text(captionParts(data, draw));

/** The note under the plate's controls: what the dots are. */
export function legendParts(data) {
  const k = slot("configs", data.configs.length), s = slot("seeds", data.seeds.length);
  const product = data.configs.flatMap((_, i) => (i ? [" × ", s] : [s]));
  return say`Each of the ${k} configurations was trained with ${s} seeds, so there are ${product} = ${combos(data)} ways to take one seed per configuration. Pale dots: all ${combos(data)} possible draws. Dark dots: the draws visited so far. Red: this draw. Ring and bar: the study's estimate, which averages all ${s} seeds of each configuration, and its ${level}% bootstrap interval, which resamples the test identities and the seeds. Values are percentage points of identity rank-1.`;
}
export const legend = (data) => text(legendParts(data));

/** How many of the possible draws have been visited. */
export function tallyParts(visited, n) {
  const all = slot("possible", n);
  return visited >= n ? say`All ${all} possible draws visited` : say`${slot("visited", visited)} of ${all} possible draws visited`;
}
export const tally = (visited, n) => text(tallyParts(visited, n));

/** A draw spelled out, for a screen reader. */
export function drawNameParts(data, draw) {
  return data.configs.flatMap((cf, k) => [k ? "; " : "", `${cf.name}, seed `, slot(`seed.${k}`, draw[k])]).filter((p) => p !== "");
}
export const drawName = (data, draw) => text(drawNameParts(data, draw));
