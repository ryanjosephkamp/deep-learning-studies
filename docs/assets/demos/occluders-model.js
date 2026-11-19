// The occluder explorer's model: pure functions over the study's measured results, no DOM.
// The page and the site's build checks both import this file, so what the page says is what the
// checks hold to the data.

export const KINDS = [
  { key: "square", label: "Centered square" },
  { key: "eye", label: "Eye band" },
  { key: "low", label: "Lower band" },
  { key: "off", label: "Off-center square" },
  { key: "gray", label: "Gray square" },
  { key: "noise", label: "Noise square" },
];

// What each synthetic occluder is like, and how well that likeness holds.
export const FIT = {
  square: "Synthetic: a flat black square drawn on the aligned crop.",
  eye: "Like sunglasses? Loose: a flat black band across the full width of the crop, drawn after the face was found, so no detector ever had to cope with it. Real sunglasses weren't tested.",
  low: "Like a face mask? Loose: a flat black band across the full width of the crop, drawn after the face was found, so no detector ever had to cope with it. Real masks weren't tested.",
  off: "Synthetic: a black square at a random place in each photo. The drawing shows one example position.",
  gray: "Synthetic: a flat gray square (value 128), never seen in training.",
  noise: "Synthetic: a square of random pixels, never seen in training. The drawing is an illustration; the study's noise is random in every pixel and color channel.",
};

/** The note under the crop: what the occluder is and how well it stands for a real one. */
export function fitNote(data, cond) {
  if (!cond.rect) return "No occluder: the clean condition, on the aligned crop.";
  if (cond.kind === "square") {
    const [lo, hi] = data.train_range;
    return `${FIT.square} The networks with occlusion training saw this kind of square in training, at ${lo} to ${hi}% of the crop.`;
  }
  return FIT[cond.kind];
}

/** The conditions of one kind, in order of size. */
export function conditionsOf(data, kind) {
  return data.conditions.filter((c) => c.kind === kind).sort((a, b) => a.size - b.size);
}

/** The condition for a kind and a size, or the nearest one the study ran. */
export function pick(data, kind, size) {
  const list = conditionsOf(data, kind);
  let best = list[0];
  for (const c of list) if (Math.abs(c.size - size) < Math.abs(best.size - size)) best = c;
  return best;
}

/** The trained network with the highest rank-1 under a condition and test. */
export function bestTrained(data, cond, test) {
  const scores = test === "same" ? cond.same : cond.identity;
  let best = null;
  for (const m of data.models) {
    if (m.group !== "trained" || !scores[m.id]) continue;
    if (!best || scores[m.id].v > scores[best.id].v) best = m;
  }
  return best;
}

/** One sentence about the current condition, built only from the data's own values. */
export function caption(data, cond, test) {
  const name = (k) => data.models.find((m) => m.id === k).name;
  const id = cond.identity;
  const chance = `Picking a gallery photo at random would be right ${data.chance}% of the time.`;
  if (test === "same") {
    if (!cond.same) return `${cond.label}: the same-photo control wasn't run with this setting. It covers the clean crop and five centered black squares, up to half the crop.`;
    const s = cond.same;
    const how = cond.rect ? "each covered gallery photo matched back to its own clean copy" : "each gallery photo matched back to itself";
    return `${cond.label}, scored the original way, rerun in this study on the aligned crops (${how}): ${name("arc")} ${s.arc.s}% and ${name("sf")} ${s.sf.s}%, against ${id.arc.s}% and ` +
      `${id.sf.s}% when the probe is a different photo of the same person. ${chance}`;
  }
  const b = bestTrained(data, cond, "identity");
  return `${cond.label}: ${name("arc")} identifies ${id.arc.s}% of new photos (95% interval ${id.arc.lo} to ` +
    `${id.arc.hi}%), and ${name("sf")} ${id.sf.s}%. Among the trained networks, “${b.name}” leads at ${id[b.id].s}%. ` +
    chance;
}

/** The rectangle an occluder covers on the crop, as [x, y, w, h] in crop pixels, or null. */
export function rectOf(data, cond) {
  return cond.rect;
}
