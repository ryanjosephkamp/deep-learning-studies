// The words of "Does it still know you?", as pure functions of the numbers, so the page and the site's
// checks use the same sentences. No DOM.
//
// A score passes a cutoff when it is strictly above it, as in the study's study/evaluate.py. Scores
// and cutoffs are shown to three decimals; when the two print the same, the sentence says which side
// the unrounded score is on.

/** A byte count in megabytes, as the page shows sizes. */
export const MB = (n) => `${(n / 1e6).toFixed(n < 1e6 ? 1 : 0)} MB`;

/** "1%" or "0.1%" for a false accept rate given as a share. */
export const farText = (far) => `${Number((far * 100).toPrecision(2))}%`;

/** Whether a score passes a cutoff, as the study decides it. */
export const passes = (s, t) => s > t.cos;

const three = (v) => v.toFixed(3);

/** The score, and where it sits against the study's cutoffs (listed from the loosest). */
export function scoreLine(s, thr) {
  const passed = thr.filter((t) => passes(s, t));
  const where = passed.length === thr.length ? "above both of the study's cutoffs"
    : passed.length ? `above the study's ${farText(passed[0].far)} cutoff, not above its ${farText(thr[thr.length - 1].far)} cutoff`
    : "not above either of the study's cutoffs";
  const tie = thr.find((t) => three(s) === three(t.cos));
  const close = tie ? ` At three decimals it equals the ${farText(tie.far)} cutoff; unrounded, it is ${passes(s, tie) ? "just above" : "at or just below"} it.` : "";
  return `Cosine score ${three(s)}: ${where} (${thr.map((t) => `${three(t.cos)} for ${farText(t.far)}`).join(", ")}).${close}`;
}

/** What the cutoffs are. */
export function cutoffNote(thr) {
  return `Each cutoff is the score the study set on clean test faces so that ${thr.map((t) => farText(t.far)).join(" or ")} of impostor pairs (two different people) scored above it.`;
}

/** What the score means for a verifier set at the study's loosest cutoff. */
export function verdict(s, thr, source, occ) {
  const loose = thr[0];
  const ok = passes(s, loose);
  const same = source !== "camera";
  const head = ok ? `Accepted at the study's ${farText(loose.far)} cutoff.` : `Refused at the study's ${farText(loose.far)} cutoff.`;
  const tail = !ok && occ
    ? " In the study, with the cutoff fixed on clean faces, covering made both pretrained verifiers refuse the right person more often, while their false accept rates fell."
    : ok && same && occ ? " This compares a covered copy with its own clear picture, the same image on both sides, as in the original test; a new photo of the same person is a harder test." : "";
  return head + tail;
}

/** The note under the demo: what is being compared, and that it is SFace on this device, not the study. */
export function sourceNote(source) {
  const what = source === "camera"
    ? "Each camera frame is a new picture of you, compared one to one with the clear reference you kept, as the study's verification test compared a new photo with a clean gallery photo."
    : source === "photo"
      ? "The covered picture is a copy of your photo, compared with its own clear version: the same image on both sides, as in the original test."
      : source === "portrait"
        ? "The portrait, a synthetic face of a person who doesn't exist, is compared with a covered copy of itself: the same image on both sides, as in the original test."
        : source === "drawn"
          ? "The drawn face is compared with a covered copy of itself; SFace was trained on photos, so a drawing tells little."
          : "";
  return `${what} The score is SFace on this device, not a result from the study's test set.`.trim();
}

/** The line shown when no face is found in a camera frame. */
export const NO_FACE = "No face found in this frame. In an approximate re-creation of the original pipeline, its face detector also lost faces once a square covered them, which explains most of SFace's collapse at small squares there; the study aligned each face before covering it.";
