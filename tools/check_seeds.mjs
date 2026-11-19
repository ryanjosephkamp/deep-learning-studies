// Check the seed-draw plate against the study's committed results.
//
//   import { checkSeeds } from "./check_seeds.mjs";
//   const stats = await checkSeeds({ repo, chapter, ok });
//
// repo is the course repository, chapter the study's folder in it, and ok(condition, message) records
// one check. The plate's data is read from <repo>/docs/<chapter>/data/seeds.json (or dataPath), and its
// model from <repo>/docs/assets/demos/seeds-model.js (or modelPath). It checks that:
//  1. every per-seed value in the data equals results/study/analysis/seed_spread.csv, and each
//     configuration's mean over its seeds equals its estimate in intervals.csv (to 1e-12);
//  2. each contrast's conditions equal the in_P1, in_P2, or in_P3 flags of main_effects_by_condition.csv,
//     its configuration pairs equal AUG_MAIN or ATT_MAIN in study/analyze.py, and each pair differs only in
//     the contrast's factor (study/config.json);
//  3. the study's values in the data equal contrasts_primary.csv;
//  4. the model reproduces each contrast's estimate, seed_combo_min, seed_combo_max, and seed_combos (to
//     1e-12), the mean of all draws equals the estimate, every draw equals the value built from
//     seed_pairs.csv's seed-pair differences (to 1e-12), and every draw has the estimate's sign;
//  5. the rounded strings equal the paper's macros in paper/numbers.tex, and the model's rounding equals
//     an exact reimplementation of paper/make_numbers.py's for every value the plate can show;
//  6. the chapter's section on the project's two ideas quotes the same values;
//  7. every number in every caption, readout, label, and note the plate can show, over all draws, sits in
//     a named slot and equals the value that name stands for (this draw's P1 in the P1 slot, P3's lowest
//     draw in P3's, the paper's macro for an estimate or interval end, and so on); each contrast's value
//     sits in a phrase about that contrast; no digit is in the words around the slots; the P1 squares lie
//     within the training range the text gives, and P2's conditions are not squares; and the axis holds
//     zero and every value.
// It returns { captions, draws }: how many texts were checked and how many draws there are.
import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const TOL = 1e-12;  // shares of probes; the study's values are written at full double precision

/** Rows of a CSV file with a header, with quoted fields (some labels hold commas). */
function csv(path) {
  const text = readFileSync(path, "utf8").replace(/\r/g, "");
  const rows = [];
  let row = [], field = "", quoted = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (ch === '"') quoted = false;
      else field += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ",") { row.push(field); field = ""; }
    else if (ch === "\n") { row.push(field); rows.push(row); row = []; field = ""; }
    else field += ch;
  }
  if (field || row.length) { row.push(field); rows.push(row); }
  const [head, ...body] = rows;
  return body.filter((r) => r.length === head.length).map((r) => Object.fromEntries(r.map((v, i) => [head[i], v])));
}

// ---- make_numbers.py's rounding, exactly ------------------------------------------------------

/** A double's exact value as n / 10^k (n a BigInt), with its sign. */
function exactDecimal(v) {
  const dv = new DataView(new ArrayBuffer(8));
  dv.setFloat64(0, v);
  const hi = dv.getUint32(0), lo = dv.getUint32(4);
  const neg = hi >>> 31 === 1;
  const bits = (hi >>> 20) & 0x7ff;
  let m = (BigInt(hi & 0xfffff) << 32n) | BigInt(lo);
  let e = -1074;
  if (bits) { m |= 1n << 52n; e = bits - 1075; }
  if (e >= 0) return { neg, n: m << BigInt(e), k: 0 };
  return { neg, n: m * 5n ** BigInt(-e), k: -e };
}

/** spct(x) of make_numbers.py: Decimal(repr(round(100x, 10))) half up to one place, signed, with a
 *  true minus sign. round() is half to even on the exact value, and repr() of a ten-place value gives
 *  that value back, so this works on exact decimals throughout. */
function refPoints(x) {
  const { neg, n, k } = exactDecimal(100 * x);
  let m10 = n;  // round(., 10), half to even
  if (k > 10) {
    const d = 10n ** BigInt(k - 10), q = n / d, r = n % d;
    m10 = 2n * r > d || (2n * r === d && q % 2n === 1n) ? q + 1n : q;
  } else m10 = n * 10n ** BigInt(10 - k);
  const d = 10n ** 9n, q = m10 / d, r = m10 % d;
  const t = 2n * r >= d ? q + 1n : q;  // half up to one place
  const s = `${t / 10n}.${t % 10n}`;
  if (x === 0) return s;
  return (neg ? "−" : "+") + s;
}

const tex = (s) => s.replace("−", "$-$");

export async function checkSeeds({ repo, chapter, ok, dataPath, modelPath }) {
  const proj = join(repo, chapter);
  const A = join(proj, "results/study/analysis");
  dataPath ??= join(repo, "docs", chapter, "data/seeds.json");
  if (!modelPath) {
    modelPath = join(repo, "docs/assets/demos/seeds-model.js");
  }
  const data = JSON.parse(readFileSync(dataPath, "utf8"));
  const M = await import(pathToFileURL(modelPath).href);
  const keys = data.contrasts.map((c) => c.key);

  // 1. per-seed values
  const spread = csv(join(A, "seed_spread.csv")).filter((r) => r.protocol === "identity" && r.metric === "rank1");
  const sp = new Map(spread.map((r) => [`${r.model}|${r.condition}`, r]));
  const seedCols = Object.keys(spread[0]).filter((k) => k.startsWith("seed_")).map((k) => k.slice(5));
  ok(JSON.stringify(seedCols) === JSON.stringify(data.seeds), `seeds: data ${data.seeds}, seed_spread.csv ${seedCols}`);
  const iv = new Map(csv(join(A, "intervals.csv")).filter((r) => r.protocol === "identity" && r.metric === "rank1")
    .map((r) => [`${r.model}|${r.condition}`, r]));
  for (const cf of data.configs) {
    data.conditions.forEach((cond, j) => {
      const r = sp.get(`${cf.id}|${cond}`);
      ok(Boolean(r), `${cf.id} ${cond}: not in seed_spread.csv`);
      if (!r) return;
      for (const s of data.seeds) ok(data.rank1[cf.id][s][j] === Number(r[`seed_${s}`]), `${cf.id} seed ${s} ${cond}: data ${data.rank1[cf.id][s][j]}, csv ${r[`seed_${s}`]}`);
      const m = data.seeds.reduce((a, s) => a + data.rank1[cf.id][s][j], 0) / data.seeds.length;
      const e = iv.get(`${cf.id}|${cond}`);
      ok(e && Math.abs(m - Number(e.estimate)) < TOL, `${cf.id} ${cond}: seed mean ${m}, intervals.csv ${e && e.estimate}`);
    });
  }

  // 2. what each contrast averages over
  const mains = csv(join(A, "main_effects_by_condition.csv"));
  const analyze = readFileSync(join(proj, "study/analyze.py"), "utf8");
  const tuples = (name) => [...analyze.match(new RegExp(`^${name} = \\[([^\\]]*)\\]`, "m"))[1].matchAll(/\('(\w+)', '(\w+)'\)/g)].map((m) => [m[1], m[2]]);
  const runs = new Map(JSON.parse(readFileSync(join(proj, "study/config.json"), "utf8")).runs.map((r) => [r.name, r]));
  const FLAG = { attention: "attention", augmentation: "augment" };
  for (const c of data.contrasts) {
    const want = mains.filter((r) => r[`in_${c.key}`] === "True").map((r) => r.condition).sort();
    ok(JSON.stringify([...c.conditions].sort()) === JSON.stringify(want), `${c.key}: conditions ${c.conditions} vs main_effects_by_condition.csv ${want}`);
    ok(c.conditions.every((x) => data.conditions.includes(x)), `${c.key}: a condition missing from the data`);
    const pm = analyze.match(new RegExp(`\\('${c.key}', '[^']*', (\\w+),`));
    ok(pm && JSON.stringify(tuples(pm[1])) === JSON.stringify(c.pairs), `${c.key}: pairs ${JSON.stringify(c.pairs)} vs analyze.py`);
    const other = c.factor === "attention" ? "augmentation" : "attention";
    for (const [a, b] of c.pairs) {
      const ra = runs.get(a), rb = runs.get(b);
      ok(ra[FLAG[c.factor]] === true && rb[FLAG[c.factor]] === false && ra[FLAG[other]] === rb[FLAG[other]] && ra.loss === rb.loss,
        `${c.key}: ${a} and ${b} do not differ only in ${c.factor}`);
    }
  }
  for (const cf of data.configs) {
    const r = runs.get(cf.id);
    ok(r && r.attention === cf.attention && r.augment === cf.augmentation, `${cf.id}: flags differ from config.json`);
  }

  // 3. the study's values
  const prim = new Map(csv(join(A, "contrasts_primary.csv")).map((r) => [r.contrast, r]));
  for (const c of data.contrasts) {
    const r = prim.get(c.key);
    for (const f of ["estimate", "ci95_low", "ci95_high", "seed_combo_min", "seed_combo_max"]) {
      ok(c.study[f] === Number(r[f]), `${c.key} ${f}: data ${c.study[f]}, csv ${r[f]}`);
    }
    ok(c.study.seed_combos === Number(r.seed_combos) && data.combos === Number(r.seed_combos), `${c.key}: seed_combos`);
  }

  // 4. the model against the study
  const simple = new Map([...analyze.match(/^SIMPLE = \[([\s\S]*?)^\]/m)[1].matchAll(/\('([^']+)', '(\w+)', '(\w+)'\)/g)].map((m) => [`${m[2]}|${m[3]}`, m[1]]));
  const pairDiff = new Map(csv(join(A, "seed_pairs.csv")).filter((r) => r.protocol === "identity")
    .map((r) => [`${r.contrast}|${r.condition}|${r.seed_a}|${r.seed_b}`, Number(r.difference)]));
  const n = M.drawCount(data);
  ok(n === data.combos, `draws: model ${n}, data ${data.combos}`);
  ok(data.seeds.length ** data.configs.length === data.combos, "seeds to the power of configurations is not the number of draws");
  const draws = {}, rebuilt = {};  // the model's draws, and the same draws rebuilt here from seed_pairs.csv
  for (const c of data.contrasts) {
    const r = prim.get(c.key);
    const est = M.estimate(data, c.key);
    ok(Math.abs(est - Number(r.estimate)) < TOL, `${c.key} estimate: model ${est}, csv ${r.estimate}`);
    const v = M.all81(data, c.key);
    draws[c.key] = v;
    rebuilt[c.key] = [];
    ok(v.length === Number(r.seed_combos), `${c.key}: ${v.length} draws, csv ${r.seed_combos}`);
    ok(Math.abs(Math.min(...v) - Number(r.seed_combo_min)) < TOL, `${c.key} seed_combo_min: model ${Math.min(...v)}, csv ${r.seed_combo_min}`);
    ok(Math.abs(Math.max(...v) - Number(r.seed_combo_max)) < TOL, `${c.key} seed_combo_max: model ${Math.max(...v)}, csv ${r.seed_combo_max}`);
    const mean = v.reduce((a, x) => a + x, 0) / v.length;
    ok(Math.abs(mean - est) < TOL, `${c.key}: mean of all draws ${mean}, estimate ${est}`);
    const rng = M.range81(data, c.key);
    ok(rng.min === Math.min(...v) && rng.max === Math.max(...v), `${c.key}: range81 differs from all81`);
    ok(v.every((x) => Math.sign(x) === Math.sign(Number(r.estimate))), `${c.key}: a draw has the other sign from the estimate`);
    for (let i = 0; i < n; i++) {
      const draw = M.drawAt(data, i);
      ok(M.drawIndex(data, draw) === i, `draw ${i}: index round trip`);
      const seedOf = Object.fromEntries(data.configs.map((cf, k) => [cf.id, draw[k]]));
      const parts = c.pairs.map(([a, b]) => {
        const label = simple.get(`${a}|${b}`);
        const ds = c.conditions.map((cond) => pairDiff.get(`${label}|${cond}|${seedOf[a]}|${seedOf[b]}`));
        return ds.reduce((s, x) => s + x, 0) / ds.length;
      });
      const ref = parts.reduce((s, x) => s + x, 0) / parts.length;
      rebuilt[c.key].push(ref);
      ok(Math.abs(v[i] - ref) < TOL, `${c.key} draw ${i}: model ${v[i]}, seed_pairs.csv ${ref}`);
    }
  }
  // the draw order is study/analyze.py's: configurations sorted by name, the last one's seed fastest
  const sorted = data.configs.map((c) => c.id);
  ok(JSON.stringify(sorted) === JSON.stringify([...sorted].sort()), "configurations are not in analyze.py's sorted order");

  // 5. rounding
  const macros = new Map([...readFileSync(join(proj, "paper/numbers.tex"), "utf8").matchAll(/csname n@([^\\]+)\\endcsname\{([^}]*)\}/g)].map((m) => [m[1], m[2]]));
  for (const c of data.contrasts) {
    const r = prim.get(c.key);
    for (const [f, suffix, col] of [["estimate", "", "estimate"], ["lo", "_lo", "ci95_low"], ["hi", "_hi", "ci95_high"],
      ["smin", "_smin", "seed_combo_min"], ["smax", "_smax", "seed_combo_max"]]) {
      ok(tex(c.s[f]) === macros.get(c.key + suffix), `${c.key}${suffix}: data ${c.s[f]}, paper ${macros.get(c.key + suffix)}`);
      ok(M.points(Number(r[col])) === c.s[f], `${c.key}${suffix}: model rounds ${r[col]} to ${M.points(Number(r[col]))}, data ${c.s[f]}`);
      ok(refPoints(Number(r[col])) === c.s[f], `${c.key}${suffix}: exact rounding ${refPoints(Number(r[col]))}, data ${c.s[f]}`);
    }
    const rng = M.range81(data, c.key);
    ok(M.points(rng.min) === c.s.smin && M.points(rng.max) === c.s.smax, `${c.key}: the plate's range ${M.points(rng.min)} to ${M.points(rng.max)}, paper ${c.s.smin} to ${c.s.smax}`);
    for (const x of draws[c.key]) ok(M.points(x) === refPoints(x), `${c.key}: model rounds ${x} to ${M.points(x)}, exact ${refPoints(x)}`);
  }
  ok(String(data.combos) === macros.get("combos"), `combos: data ${data.combos}, paper ${macros.get("combos")}`);
  ok(String(data.seeds.length) === macros.get("n_seeds"), `seeds: data ${data.seeds.length}, paper ${macros.get("n_seeds")}`);
  const count = Object.fromEntries(data.contrasts.map((c) => [c.key, String(c.conditions.length)]));
  ok(count.P1 === macros.get("n_trained_squares"), `P1 conditions: ${count.P1}, paper ${macros.get("n_trained_squares")}`);
  ok(count.P2 === macros.get("n_unseen"), `P2 conditions: ${count.P2}, paper ${macros.get("n_unseen")}`);
  ok(count.P3 === macros.get("n_p3"), `P3 conditions: ${count.P3}, paper ${macros.get("n_p3")}`);

  // 6. the chapter
  const md = readFileSync(join(proj, "chapter/chapter.md"), "utf8");
  const sec = (md.match(/^## What the project's two ideas do\n([\s\S]*?)(?=^## )/m) || [, ""])[1];
  ok(sec.length > 0, "chapter: no section \"What the project's two ideas do\"");
  const C = Object.fromEntries(data.contrasts.map((c) => [c.key, c.s]));
  for (const s of [`${C.P1.estimate.slice(1)} points`, `${C.P1.lo} to ${C.P1.hi}`, `${C.P2.estimate.slice(1)} points`, `${C.P2.lo} to ${C.P2.hi}`,
    `${C.P3.estimate} point`, `${C.P3.lo} to ${C.P3.hi}`, `all ${data.combos} ways to take one seed per configuration`]) {
    ok(sec.includes(s), `chapter: the section does not quote "${s}"`);
  }

  // 7. every number the plate can show, slot by slot
  let captions = 0;
  const NCOUNT = { squares: "n_trained_squares", heldout: "n_unseen", both: "n_p3" };
  const MACRO = { estimate: "", lo: "_lo", hi: "_hi", min: "_smin", max: "_smax" };
  const configs = String(new Set(data.contrasts.flatMap((c) => c.pairs.flat())).size);
  const digitsOf = (draw) => { const out = []; for (let k = data.configs.length - 1, j = draw; k >= 0; k--) { out[k] = data.seeds[j % data.seeds.length]; j = Math.floor(j / data.seeds.length); } return out; };
  // The value a slot's name stands for, from the study's files and the paper, not from the model.
  function expected(name, ctx) {
    const [a, b] = name.split(".");
    if (name === "combos" || name === "possible") return macros.get("combos");
    if (name === "seeds") return macros.get("n_seeds");
    if (name === "configs") return configs;
    if (name === "level") return "95";  // the interval columns are ci95_*
    if (name === "squares.from") return macros.get("occ_lo");
    if (name === "squares.to") return macros.get("occ_hi");
    if (name === "visited") return String(ctx.visited);
    if (a === "seed") return digitsOf(ctx.draw)[Number(b)];
    if (a === "set") return macros.get(NCOUNT[b]);
    if (keys.includes(a) && b === "draw") return refPoints(rebuilt[a][ctx.draw]);
    if (keys.includes(a) && b in MACRO) return macros.get(a + MACRO[b]);
    return undefined;
  }
  // A sentence's words and slots in order, each with the contrast its phrase is about (or null).
  const walk = (parts, key = null, out = []) => {
    for (const p of [parts].flat(Infinity)) {
      if (p && typeof p === "object" && p.about) walk(p.parts, p.about, out);
      else out.push([p, key]);
    }
    return out;
  };
  // Every slot equals what its name stands for; a contrast's value (P1.draw, P3.min, ...) sits in a
  // phrase about that contrast; and the words hold no digit (but "rank-1").
  const holds = (parts, ctx, what, asText) => {
    captions++;
    const seq = walk(parts);
    ok(seq.map(([p]) => (typeof p === "string" ? p : p.value)).join("") === asText && M.text(parts) === asText, `${what}: the text differs from its parts`);
    for (const [p, key] of seq) {
      if (typeof p === "string") { ok(!/\d/.test(p.replace(/rank-1/g, "")), `${what}: a digit outside a slot in "${p}"`); continue; }
      const want = expected(p.name, ctx);
      ok(want !== undefined && tex(p.value) === want, `${what}: slot ${p.name} shows ${p.value}, expected ${want}`);
      const owner = p.name.split(".")[0];
      if (keys.includes(owner)) ok(key === owner, `${what}: ${p.name} sits in a phrase about ${key ?? "no contrast"}`);
    }
  };
  const p3 = M.range81(data, "P3");
  for (let i = 0; i < n; i++) {
    const draw = M.drawAt(data, i);
    ok(JSON.stringify(draw) === JSON.stringify(digitsOf(i)), `draw ${i}: drawAt differs from analyze.py's order`);
    const ctx = { draw: i };
    const cap = M.caption(data, draw);
    holds(M.captionParts(data, draw), ctx, `caption, draw ${i}`, cap);
    ok(cap.includes(M.side(p3.min, p3.max)) && (p3.min > 0) === cap.includes("above zero"), `caption, draw ${i}: side of zero`);
    for (const c of data.contrasts) {
      const rp = M.readoutParts(data, c.key, draw), rs = M.readout(data, c.key, draw);
      rp.forEach((p, j) => {
        holds(p.label, ctx, `readout ${c.key} label ${j}, draw ${i}`, rs[j].label);
        holds(p.value, ctx, `readout ${c.key} value ${j}, draw ${i}`, rs[j].value);
      });
      const first = walk(rp[0].value).filter(([p]) => typeof p !== "string");
      ok(first.length === 1 && first[0][0].name === `${c.key}.draw`, `readout ${c.key}, draw ${i}: "this draw" is not ${c.key}'s draw`);
      holds(M.rowLabelParts(data, c.key, draw), ctx, `row label ${c.key}, draw ${i}`, M.rowLabel(data, c.key, draw));
    }
    holds(M.drawNameParts(data, draw), ctx, `draw name ${i}`, M.drawName(data, draw));
  }
  for (const c of data.contrasts) {
    holds(M.titleParts(data, c.key), {}, `title ${c.key}`, M.title(data, c.key));
    holds(M.detailParts(data, c.key), {}, `detail ${c.key}`, M.detail(data, c.key));
  }
  holds(M.legendParts(data), {}, "legend", M.legend(data));
  for (let k = 0; k <= n; k++) holds(M.tallyParts(k, n), { visited: k }, `tally ${k}`, M.tally(k, n));
  // the P1 squares are centered squares within the training range the text gives, from end to end
  const p1 = M.contrastOf(data, "P1").conditions;
  const sizes = p1.map((c) => Math.round(100 * Number(c.replace(/^square_/, ""))));
  ok(p1.every((c) => /^square_\d\.\d+$/.test(c)), "P1: a condition that is not a centered square");
  ok(Math.min(...sizes) === Number(macros.get("occ_lo")) && Math.max(...sizes) === Number(macros.get("occ_hi")), `P1 squares run ${Math.min(...sizes)} to ${Math.max(...sizes)}%, training ${macros.get("occ_lo")} to ${macros.get("occ_hi")}%`);
  ok(JSON.stringify(M.squareSizes(data)) === JSON.stringify([Math.min(...sizes), Math.max(...sizes)]), "squareSizes differs");
  // P2's conditions are held out: none is a centered square in the training range
  ok(M.contrastOf(data, "P2").conditions.every((c) => !p1.includes(c) && !/^square_/.test(c)), "P2: a condition that training saw");
  const ax = M.axis(data);
  ok(ax.lo < 0 && ax.hi > 0, "axis: zero is not inside");
  for (const k of keys) {
    ok(M.range81(data, k).min * 100 >= ax.lo && M.range81(data, k).max * 100 <= ax.hi, `axis: ${k}'s draws fall outside`);
    const c = M.contrastOf(data, k);
    ok(100 * c.study.ci95_low >= ax.lo && 100 * c.study.ci95_high <= ax.hi, `axis: ${k}'s interval falls outside`);
  }
  for (const t of ax.ticks) ok(Number.isInteger(t.at) && t.at >= ax.lo && t.at <= ax.hi, `axis: tick ${t.at}`);
  return { captions, draws: n };
}
